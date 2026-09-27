"""OpenAI-compatible hosted LLM provider (architecture 14.1, phase P7).

Not the default. D3 makes the local Ollama daemon the default so the demo needs
no credentials at all, and this module exists so `RAG_PROVIDER=openai` is a
one-line change rather than a code change.

Raw `requests` rather than the `openai` SDK, because `openai` is not in the
pinned `requirements.txt` and the phase ground rules forbid adding a dependency.
`requests` is already pinned and is what `sources/fetch.py` uses, so this adds
nothing new to the dependency set.

The API key comes from `OPENAI_API_KEY` in the environment, not from
`Config`. `.env.example` settles this: "Only change RAG_PROVIDER if you point at
a hosted LLM, in which case that provider's own key variable belongs here too."
So the provider reads its own key, and `Config` gains no new field.

Errors propagate, for the same reason as `OllamaProvider`: the pipeline turns them
into a class ERROR answer rather than an empty one.
"""
from __future__ import annotations

import os

import requests

# Hosted models answer in under a second. This is generous, and exists only so a
# hung connection surfaces as a class ERROR answer instead of a frozen demo.
_TIMEOUT_S = 120.0

_CLASSIFY_MAX_TOKENS = 16
_CLASSIFY_TEMPERATURE = 0.0

_DEFAULT_BASE_URL = "https://api.openai.com"
_ENV_KEY = "OPENAI_API_KEY"
_ENV_BASE_URL = "OPENAI_BASE_URL"


class OpenAIProvider:
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint.

    Implements the `LLMProvider` protocol structurally, like every other provider
    in this package, so `isinstance(provider, LLMProvider)` is the runtime check.
    """

    name = "openai"

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model
        self.name = f"openai:{model}"
        self.api_key = api_key or os.environ.get(_ENV_KEY, "")
        self.base_url = (base_url or os.environ.get(_ENV_BASE_URL)
                         or _DEFAULT_BASE_URL).rstrip("/")
        if not self.api_key:
            # Raised at construction, not at first call: a missing key is a
            # deployment mistake, and failing here names it instead of surfacing
            # later as a 401 inside a generation call.
            raise RuntimeError(
                f"{_ENV_KEY} is not set. It is required only for "
                f"RAG_PROVIDER=openai; the default ollama provider needs no key."
            )

    def _chat(self, system: str, user: str, *, temperature: float,
              max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user or ""},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
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
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""

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
