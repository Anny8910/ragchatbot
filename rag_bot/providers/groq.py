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
import re
import time

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

# Rate-limit backoff. The free tier allows a small number of requests per
# minute, and an eval row costs two calls, so 51 rows is a burst that trips 429.
# Geometric from 2s, capped at 30s, four attempts: about 2 + 4 + 8 = 14s of
# waiting per call in the worst case, which is short enough that a slow run
# stays watchable and long enough to clear a one-minute window in practice.
_RATE_LIMIT_BACKOFF_S = (2.0, 4.0, 8.0, 16.0, 30.0)
_MAX_RETRIES = 4

# Retried: 429 (rate limited) and 5xx (upstream trouble). Not retried: 401/403
# (the key is wrong, and retrying only delays that message) or 4xx like 400
# (the payload is wrong, and a different payload is a bug, not a retry).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Groq names the window in the 429 body but not in the headers: "tokens per day
# (TPD)", "requests per day (RPD)", and the corresponding per-minute forms. Only
# the daily ones are terminal -- a per-minute window clears in seconds, and
# retrying is right.
_DAILY_QUOTA = re.compile(r"on (tokens|requests) per day", re.IGNORECASE)
_QUOTA_USED = re.compile(r"Used (\d+)", re.IGNORECASE)
_QUOTA_LIMIT = re.compile(r"Limit (\d+)", re.IGNORECASE)
_QUOTA_RETRY = re.compile(r"try again in ([0-9.]+)(ms|s|m|h)", re.IGNORECASE)


def _daily_quota_error(body: str) -> str | None:
    """A message for a 429 caused by a daily cap, or None for a normal one.

    The message carries the numbers because they are the diagnosis: "429" on its
    own reads as a bug in the harness, and "tokens per day: 199945 of 200000,
    resets in 17h" reads as a quota that will be there tomorrow.
    """
    match = _DAILY_QUOTA.search(body or "")
    if match is None:
        return None
    window = match.group(1).lower()
    used = _QUOTA_USED.search(body)
    limit = _QUOTA_LIMIT.search(body)
    retry = _QUOTA_RETRY.search(body)
    parts = [f"groq daily quota exhausted ({window} per day)"]
    if used and limit:
        parts.append(f"{used.group(1)} of {limit.group(1)} used")
    if retry:
        amount, unit = retry.group(1), retry.group(2).lower()
        seconds = float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
        # Labelled "shortest window", because that is what it is. The same 429
        # body carries "try again in 2 min" while the daily budget it is really
        # about does not reset for ~18 hours -- the value refers to the fastest
        # window that must clear, and reading it as the answer to "when can I
        # run again" is how an 18-hour wait becomes a 2-minute one.
        parts.append(
            f"shortest window clears in {seconds / 60:.0f} min"
            if seconds >= 60 else f"shortest window clears in {seconds:.0f}s"
        )
    return "; ".join(parts) + " -- not retried, because retrying cannot help"


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
        # Populated after every successful call. The eval runner reads these to
        # pace itself: the free tier enforces several separate limits (per
        # minute, per day, per model) and the binding one changes during a run,
        # so a fixed sleep either wastes time under a limit that is not
        # binding or gets 429'd by one that is.
        self.last_headers: dict[str, str] = {}
        self.last_total_tokens: int | None = None

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
        response = self._post_with_backoff(payload)
        response.raise_for_status()
        data = response.json()
        # Exposed so a caller that runs many calls in a loop (the eval runner)
        # can pace itself from Groq's own accounting rather than from a guess.
        self.last_headers = {
            k: v for k, v in response.headers.items()
            if k.lower().startswith("x-ratelimit-")
        }
        total = (data.get("usage") or {}).get("total_tokens")
        if isinstance(total, int):
            self.last_total_tokens = total
        choices = data.get("choices") or []
        if not choices:
            return "", None
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        return (content if isinstance(content, str) else "",
                choice.get("finish_reason"))

    def _post_with_backoff(self, payload: dict) -> requests.Response:
        """POST, retrying only the status codes that mean "ask again later".

        The free Groq tier rate-limits per minute, and a single eval row costs
        two calls (triage classify, then generation). Fifty-one rows therefore
        arrives as a burst that trips 429 -- and because a 429 is an
        `HTTPError`, the pipeline correctly turns it into an error answer, so
        without this the eval reported a 22-row failure rate that was really
        "the harness was too fast".

        Only 429 and 5xx are retried. A 401 or 400 will fail identically on
        every attempt, and retrying those just delays the message that actually
        explains the problem. `Retry-After` is honoured when present, since
        Groq sends it and it is more accurate than a guess.

        A 429 that is really a DAILY quota is not retried at all. Groq reports
        that only in the response body -- "Rate limit reached ... on tokens per
        day (TPD): Limit 200000, Used 199945 ... Please try again in 7.776s"
        -- and never in the headers, which keep reporting the per-minute window
        as healthy. Retrying it obeys the `Retry-After` it asks for, so a 429
        that clears in 8 seconds gets retried and fails again, and one that
        clears in 17 hours gets retried 4 times across 7 minutes of sleeping
        before surfacing. Both are wrong: the first wastes the retry, the second
        is a hung process. So a body-quota 429 raises immediately, and its
        message says what actually happened.
        """
        delay = _RATE_LIMIT_BACKOFF_S[0]
        last_error: requests.HTTPError | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=_TIMEOUT_S,
                )
            except requests.Timeout:
                # A timeout is a network symptom, so it is retried on the same
                # terms as a 5xx. The request may still have been processed,
                # but these are short generations and a duplicate is cheaper
                # than a failed eval row.
                last_error = requests.Timeout(
                    f"groq did not respond within {_TIMEOUT_S}s")
            else:
                if response.status_code not in _RETRYABLE_STATUS:
                    return response
                last_error = requests.HTTPError(
                    f"{response.status_code} from groq", response=response)
                if response.status_code == 429:
                    if (quota := _daily_quota_error(response.text)) is not None:
                        raise requests.HTTPError(quota, response=response)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = int(retry_after)

            if attempt == _MAX_RETRIES:
                break
            logger.warning(
                "groq %s (attempt %d/%d); sleeping %.1fs",
                last_error, attempt + 1, _MAX_RETRIES + 1, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, _RATE_LIMIT_BACKOFF_S[-1])

        assert last_error is not None
        raise last_error

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
