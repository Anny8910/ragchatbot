"""GroqProvider tests. No network: `requests.post` is stubbed.

The behaviour worth locking down here is the reasoning-model handling in
`rag_bot/providers/groq.py`, because a regression in it is SILENT: the provider
returns "" instead of raising, the pipeline treats that as "the model said
nothing", and the LLM triage layer quietly stops running while every test that
checks "did an answer come back" still passes.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import requests

from rag_bot.providers.groq import (
    _CLASSIFY_MAX_TOKENS,
    _ENV_BASE_URL,
    _ENV_KEY,
    _ENV_MODEL,
    GroqProvider,
)
from rag_bot.providers.base import LLMProvider


class _Response:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text}")

    def json(self) -> dict:
        return self._payload


def _completion(content: str, *, finish_reason: str = "stop",
                reasoning: str = "") -> dict:
    message: dict[str, str] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    return {"choices": [{"index": 0, "message": message,
                         "finish_reason": finish_reason}]}


@pytest.fixture(autouse=True)
def _clean_groq_env(monkeypatch):
    """No ambient GROQ_* variables: these tests must not depend on the .env."""
    for key in (_ENV_KEY, _ENV_MODEL, _ENV_BASE_URL):
        monkeypatch.delenv(key, raising=False)


# -- construction ---------------------------------------------------------
def test_satisfies_the_llm_provider_protocol():
    """The runtime check `app.py` and `pipeline.py` rely on."""
    assert isinstance(GroqProvider(api_key="k"), LLMProvider)


def test_missing_key_raises_a_named_error_at_construction(monkeypatch):
    """A missing key is a deployment mistake, and must be named as one.

    Construction, not first call: an opaque 401 raised from inside a generation
    call is much harder to diagnose than a message that names the missing
    variable.
    """
    with pytest.raises(RuntimeError, match=_ENV_KEY):
        GroqProvider()


def test_model_comes_from_the_environment_then_the_argument(monkeypatch):
    """GROQ_MODEL wins, so a deployed instance can be repointed without touching
    RAG_* settings. The passed model is the fallback that keeps
    `app.build_llm(cfg)` authoritative for every other provider."""
    monkeypatch.setenv(_ENV_MODEL, "openai/gpt-oss-120b")
    assert GroqProvider(model="ignored", api_key="k").model == "openai/gpt-oss-120b"
    monkeypatch.delenv(_ENV_MODEL)
    assert GroqProvider(model="from-cfg", api_key="k").model == "from-cfg"


def test_provider_name_carries_the_model():
    """`name` is what the UI's pipeline trace prints, so it must identify the
    model actually in use."""
    assert GroqProvider(model="m1", api_key="k").name == "groq:m1"


# -- request shape --------------------------------------------------------
def test_classify_budget_exceeds_a_non_reasoning_models(monkeypatch):
    """gpt-oss spends ~36 completion tokens reasoning before emitting a label, so
    a 16-token budget (correct for a non-reasoning local model) returns empty
    content. This is the constant that would have to change for that to regress.
    """
    assert _CLASSIFY_MAX_TOKENS >= 64


def test_requests_send_reasoning_effort_low(monkeypatch):
    """`reasoning_effort: "low"` is what makes a small token budget viable at all:
    measured 132 -> 36 completion tokens on the same triage prompt. It is sent
    unconditionally, not only on the classify path."""
    seen: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(json)
        return _Response(_completion("D_performance"))

    monkeypatch.setattr(requests, "post", fake_post)
    GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.1,
                                                  max_tokens=256)
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["max_tokens"] == 256
    assert seen[0]["temperature"] == 0.1
    assert [m["role"] for m in seen[0]["messages"]] == ["system", "user"]


def test_classify_restates_the_allowed_label_set(monkeypatch):
    """An 8B-class model drifts outside the set without the reminder, and
    `triage.classify_llm` treats an out-of-set label as no answer at all."""
    seen: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(json)
        return _Response(_completion("C_advice"))

    monkeypatch.setattr(requests, "post", fake_post)
    out = GroqProvider(model="m", api_key="k").classify(
        "sys", "Which fund is best?", labels=["A_or_B", "C_advice", "D_performance"])
    assert out == "C_advice"
    assert "Reply with exactly one of: A_or_B, C_advice, D_performance" in \
        seen[0]["messages"][1]["content"]


def test_classify_keeps_only_the_first_line(monkeypatch):
    """A model that appends reasoning after the label still parses."""
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _Response(_completion("A_or_B\nbecause it is a fact")),
    )
    assert GroqProvider(model="m", api_key="k").classify(
        "s", "u", labels=["A_or_B"]) == "A_or_B"


def test_key_is_sent_as_a_bearer_token(monkeypatch):
    seen: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(headers or {})
        return _Response(_completion("ok"))

    monkeypatch.setattr(requests, "post", fake_post)
    GroqProvider(model="m", api_key="secret-key").generate("s", "u",
                                                           temperature=0.0,
                                                           max_tokens=8)
    assert seen[0]["Authorization"] == "Bearer secret-key"


# -- response handling ----------------------------------------------------
def test_empty_choices_return_empty_string(monkeypatch):
    """Never None, never raises: callers treat "" as 'the model said nothing'."""
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Response({"choices": []}))
    assert GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                         max_tokens=8) == ""


def test_truncated_with_no_content_is_retried_once(monkeypatch):
    """Every token went into `reasoning`, so `content` is empty and
    `finish_reason` is "length". Without the retry the pipeline reports a class
    ERROR for a model that would have answered at a larger budget."""
    budgets: list[int] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        budgets.append(json["max_tokens"])
        if len(budgets) == 1:
            return _Response(_completion("", finish_reason="length",
                                         reasoning="thinking..."))
        return _Response(_completion("The expense ratio is 1.03%."))

    monkeypatch.setattr(requests, "post", fake_post)
    out = GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                        max_tokens=256)
    assert out == "The expense ratio is 1.03%."
    assert budgets == [256, 512]


def test_retry_is_bounded_and_does_not_recurse(monkeypatch):
    """An unbounded retry against a reasoning model is a way to turn a token
    budget into a hung demo."""
    calls: list[int] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["max_tokens"])
        return _Response(_completion("", finish_reason="length"))

    monkeypatch.setattr(requests, "post", fake_post)
    assert GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                         max_tokens=256) == ""
    assert calls == [256, 512]


def test_truncation_with_real_content_is_not_retried(monkeypatch):
    """Only an EMPTY result is a reasoning-budget failure. A truncated answer
    still has text, and the sentence-limit validator in `answer/validate.py`
    owns that case -- retrying here would duplicate its job and hide it."""
    calls: list[int] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["max_tokens"])
        return _Response(_completion("A truncated but real answer",
                                     finish_reason="length"))

    monkeypatch.setattr(requests, "post", fake_post)
    out = GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                        max_tokens=256)
    assert out == "A truncated but real answer"
    assert calls == [256]


def test_http_error_propagates(monkeypatch):
    """The pipeline turns an exception into a class ERROR answer that still shows
    the retrieved chunks. Swallowing it here would present an outage as the
    assistant having nothing to say."""
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Response({"error": "rate limited"},
                                                  status=429))

    with pytest.raises(requests.HTTPError):
        GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                      max_tokens=8)
