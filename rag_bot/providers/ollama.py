"""Ollama LLM provider (architecture 14.1, phase P7).

The local daemon is the default provider (D3): the whole demo runs credential-free,
which is why this is a first-class module rather than a `requests` call inlined
in the pipeline.

Errors are allowed to propagate. `triage.classify_llm` catches them and degrades
to the deterministic rule layer, and `pipeline.answer_question` catches them and
returns a class ERROR answer that still carries its chunks. Swallowing them here
would turn "the daemon is down" into "the assistant has nothing to say", which is
the failure the whole error-handling path exists to avoid.
"""
from __future__ import annotations

# A generate call on a cold 8B model legitimately takes a minute; 300s is the
# point past which the demo is broken rather than slow. Not a config key: the
# phase doc forbids inventing one, and a request with no timeout can hang forever.
_TIMEOUT_S = 300.0

# A triage reply is one label. 16 tokens covers "D_performance" with room to
# spare, and capping num_predict keeps the cheap layer cheap.
_CLASSIFY_MAX_TOKENS = 16

# Classification is a routing decision, not prose, so it is run at temperature 0
# for reproducibility. `generate` uses the configured temperature, which is
# deliberately not 0 (the spec asks for a repeated-run diff at 0.1).
_CLASSIFY_TEMPERATURE = 0.0


class OllamaProvider:
    """Talks to a local Ollama daemon over its HTTP API.

    Implements the `LLMProvider` protocol structurally, like every other provider
    in this package, so `isinstance(provider, LLMProvider)` is the runtime check.
    """

    def __init__(self, host: str, model: str):
        from ollama import Client

        self.host = host
        self.model = model
        self.name = f"ollama:{model}"
        self._client = Client(host=host, timeout=_TIMEOUT_S)

    def _chat(self, system: str, user: str, *, temperature: float,
              max_tokens: int) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user or ""},
            ],
            stream=False,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
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
        """One label, nothing else.

        The allowed labels are restated in the user turn because the protocol
        passes them separately and an 8B model will drift outside the set without
        the reminder. `triage.classify_llm` treats an out-of-set label as no
        answer at all, so drift costs a rule-layer retry rather than a wrong
        class -- but stating the set is cheaper than the retry.
        """
        allowed = ", ".join(labels) if labels else ""
        prompt = f"{user}\n\nReply with exactly one of: {allowed}" if allowed else user
        raw = self._chat(system, prompt, temperature=_CLASSIFY_TEMPERATURE,
                         max_tokens=_CLASSIFY_MAX_TOKENS)
        # First line only: a chatty model that appends reasoning after the label
        # still gets parsed, and triage strips the quotes and whitespace anyway.
        return raw.strip().splitlines()[0].strip() if raw.strip() else ""
