"""Groq LLM provider (architecture 14.1; the hosted provider for the deployed app).

Replaces the local Ollama daemon as the default, for one reason: the demo has to
run for an instructor who has no Ollama installed and no 4.7 GB model on disk.
The retrieval half of the system is unaffected -- `providers/minilm.py` still runs
`sentence-transformers/all-MiniLM-L6-v2` locally, so the embedding stage and the
`RAG_MIN_SCORE` calibration are identical in both deployments. Only generation
moves off-box.

GROQ IS AN OPENAI-COMPATIBLE ENDPOINT
--------------------------------------
`api.groq.com/openai/v1/chat/completions` speaks the same wire format as the
`OpenAIProvider` in this package, so the request body and the response parsing are
the same shape. This is still a separate class rather than a `base_url` on
`OpenAIProvider`, for the one reason that actually differs and is not cosmetic:

**gpt-oss is a reasoning model, and reasoning is billed against `max_tokens`.**
Measured on `openai/gpt-oss-120b` with the exact triage prompt the pipeline sends,
returning a single label `D_performance`:

    max_tokens   finish_reason   completion_tokens   content
        16       length                     16        ""
        32       length                     32        ""
        64       length                     64        ""
       128       length                    128        ""
       256       stop                      132    "D_performance"

The tokens go into the model's private `reasoning` field, not into `content`. Two
consequences, both of which are silent failures rather than errors:

- `OllamaProvider._CLASSIFY_MAX_TOKENS = 16` is correct for a non-reasoning local
  model and would be a **no-op** here: content is always empty, the LLM triage
  layer never fires, and the system looks like it is working while running on
  rules alone. So this provider sets its own classify budget.
- `reasoning_effort: "low"` is honoured by Groq and cuts the same call from 132
  completion tokens to 36, so a 64-token budget suffices. Sent on every call.

`generate` additionally retries once when `finish_reason` is `length` with empty
content, which is the same failure at a larger scale (a long question can push
reasoning past the pipeline's 256-token generation cap). The retry is bounded and
non-recursive; see `_chat`.

The key comes from `GROQ_API_KEY` and the model from `GROQ_MODEL`, neither of which
reaches `Config`. Same reasoning as `OpenAIProvider`, and the same line in
`.env.example`: a provider owns its own credentials so `Config` stays a settings
object rather than a secret bag. `RAG_LLM_MODEL` is the fallback when `GROQ_MODEL`
is unset, so the model knob has one obvious name per provider.

Errors propagate, for the reason the module docstring in `ollama.py` gives: the
pipeline turns them into a class ERROR answer carrying the retrieved chunks,
whereas swallowing them here would present an outage as the assistant having
nothing to say.
"""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

# Hosted models answer in well under a second. Generous, but bounded, so a hung
# connection surfaces as a class ERROR answer instead of a frozen demo.
_TIMEOUT_S = 60.0

# A triage reply is one label. NOT 16, which is what the non-reasoning providers
# use: see the module docstring's measurement table -- gpt-oss spends ~36 tokens
# reasoning at `reasoning_effort: low` before it emits the label, and anything
# below that returns empty content and silently disables the LLM triage layer.
_CLASSIFY_MAX_TOKENS = 64

_CLASSIFY_TEMPERATURE = 0.0

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_ENV_KEY = "GROQ_API_KEY"
_ENV_MODEL = "GROQ_MODEL"
_ENV_BASE_URL = "GROQ_BASE_URL"

# Fallback only if GROQ_MODEL and RAG_LLM_MODEL are both unset. A deployed
# instance with neither is misconfigured, and this keeps the app answerable
# rather than raising a second error on top of the missing key.
_FALLBACK_MODEL = "openai/gpt-oss-120b"

# Ceiling for the length-retry. Reasoning-heavy models vary a lot in how much
# thinking a given prompt buys, so the retry has to have somewhere to stop.
_RETRY_CEILING = 1024


class GroqProvider:
    """Talks to Groq's OpenAI-compatible chat completions endpoint.

    Implements the `LLMProvider` protocol structurally, like every other provider
    in this package, so `isinstance(provider, LLMProvider)` is the runtime check.
    """

    name = "groq"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        # GROQ_MODEL wins over the passed model, so a deployed instance can be
        # repointed at a different model without touching RAG_* settings.
        self.model = (
            os.environ.get(_ENV_MODEL) or model or _FALLBACK_MODEL
        )
        self.name = f"groq:{self.model}"
        self.api_key = api_key or os.environ.get(_ENV_KEY, "")
        self.base_url = (
            base_url or os.environ.get(_ENV_BASE_URL) or _DEFAULT_BASE_URL
        ).rstrip("/")
        if not self.api_key:
            # Raised at construction, not at first call: a missing key is a
            # deployment mistake, and failing here names it instead of surfacing
            # later as a 401 inside a generation call.
            raise RuntimeError(
                f"{_ENV_KEY} is not set. It is required only for "
                "RAG_PROVIDER=groq; the local ollama provider needs no key."
            )

    # -- request ---------------------------------------------------------
    def _post(self, system: str, user: str, *, temperature: float,
              max_tokens: int) -> tuple[str, str | None]:
        """(content, finish_reason) for one chat completion.

        `reasoning_effort: "low"` is sent on every call. It is not a
        quality/latency trade-off to revisit: on a reasoning model it is the
        difference between the classify call returning a label and returning
        nothing at all, and the measured saving is 132 -> 36 tokens.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user or ""},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return "", None
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        return (content if isinstance(content, str) else "",
                choice.get("finish_reason"))

    def _chat(self, system: str, user: str, *, temperature: float,
              max_tokens: int) -> str:
        content, finish = self._post(system, user, temperature=temperature,
                                     max_tokens=max_tokens)
        if content.strip() or finish != "length" or max_tokens >= _RETRY_CEILING:
            return content

        # Truncated with nothing to show: every token went into reasoning. Retry
        # once with more headroom. Bounded and non-recursive on purpose -- an
        # unbounded retry against a reasoning model is a way to turn a token
        # budget into a hung demo.
        logger.info(
            "groq returned no content at max_tokens=%d (%s); retrying at %d",
            max_tokens, finish, min(max_tokens * 2, _RETRY_CEILING),
        )
        content, _ = self._post(system, user, temperature=temperature,
                                max_tokens=min(max_tokens * 2, _RETRY_CEILING))
        return content

    # -- LLMProvider -----------------------------------------------------
    def generate(self, system: str, user: str, *, temperature: float,
                 max_tokens: int) -> str:
        """Stripped model text, or "" if the provider returned nothing usable.

        Never returns None and never raises on an empty completion, so callers
        can treat "" as "the model said nothing" instead of guarding every call.
        """
        return self._chat(system, user, temperature=temperature,
                          max_tokens=max_tokens).strip()

    def classify(self, system: str, user: str, *, labels: list[str]) -> str:
        """One label, nothing else, at temperature 0.

        Same contract as `OllamaProvider.classify`: the allowed set is restated in
        the user turn, and an out-of-set label is treated as no answer by
        `triage.classify_llm`.
        """
        allowed = ", ".join(labels) if labels else ""
        prompt = f"{user}\n\nReply with exactly one of: {allowed}" if allowed else user
        raw = self._chat(system, prompt, temperature=_CLASSIFY_TEMPERATURE,
                         max_tokens=_CLASSIFY_MAX_TOKENS)
        return raw.strip().splitlines()[0].strip() if raw.strip() else ""
