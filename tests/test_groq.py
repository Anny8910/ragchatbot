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
import time

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

    def __init__(self, payload: dict, status: int = 200,
                 headers: dict | None = None):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)
        self.headers = headers or {}

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


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_error_propagates_immediately(monkeypatch, status):
    """The pipeline turns an exception into a class ERROR answer that still shows
    the retrieved chunks. Swallowing it here would present an outage as the
    assistant having nothing to say.

    4xx is not retried: a wrong key or a malformed payload fails identically
    every time, so retrying only delays the message that explains the problem.
    The single-attempt assertion is the point -- a retry loop around a 401 turns
    a one-line fix into a 14-second wait times four."""
    attempts = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: attempts.append(k)
                        or _Response({"error": "nope"}, status=status))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(requests.HTTPError):
        GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                      max_tokens=8)
    assert len(attempts) == 1, f"{status} was retried {len(attempts)} times"


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    """A 429 must be retried, not reported as a failure.

    This was a real failure mode, not a hypothetical: the free Groq tier
    rate-limits per minute and an eval row costs two calls, so the 51-row eval
    tripped 429 partway through. Because a 429 is an HTTPError the pipeline
    correctly turned it into an error answer, and the eval report showed a
    22-row failure rate that was really "the harness ran too fast".
    """
    responses = [_Response({"error": "rate limited"}, status=429,
                           headers={"Retry-After": "0"}),
                 _Response(_completion("Exit load: 1%."))]

    def fake_post(*a, **k):
        return responses.pop(0)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    out = GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                         max_tokens=8)
    assert out == "Exit load: 1%."
    assert not responses, "the retry did not happen"


def test_retry_after_header_is_honoured(monkeypatch):
    """Groq sends `Retry-After`, and it is more accurate than a guessed backoff."""
    slept: list[float] = []
    responses = [_Response({"error": "rate limited"}, status=429,
                           headers={"Retry-After": "7"}),
                 _Response(_completion("ok"))]

    monkeypatch.setattr(requests, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(time, "sleep", slept.append)

    GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                  max_tokens=8)
    assert 7.0 in slept, slept


def test_rate_limit_gives_up_after_the_bounded_number_of_attempts(monkeypatch):
    """Bounded, so a rate limit cannot turn into a hung demo: the exception still
    reaches the pipeline, which reports it as a class ERROR answer."""
    attempts = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: attempts.append(1)
                        or _Response({"error": "rate limited"}, status=429))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(requests.HTTPError):
        GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                      max_tokens=8)
    assert len(attempts) == 5, len(attempts)   # 1 + 4 retries


def test_server_error_is_retried(monkeypatch):
    """5xx is upstream trouble rather than a bad request, so it is worth
    retrying on the same terms as a 429."""
    responses = [_Response({"error": "bad gateway"}, status=502),
                 _Response(_completion("ok"))]
    monkeypatch.setattr(requests, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                          max_tokens=8) == "ok"


def test_timeout_is_retried(monkeypatch):
    """A timeout is a network symptom, so retrying is right. These are short
    generations, so a duplicate is cheaper than a failed row."""
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("timed out")
        return _Response(_completion("ok"))

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                          max_tokens=8) == "ok"


def test_rate_limit_headers_are_exposed_after_a_call(monkeypatch):
    """The eval runner paces itself from these, so the provider has to keep
    them. They are the only way to know which of Groq's several limits is the
    binding one during a long run."""
    response = _Response(
        _completion("ok"),
        headers={"x-ratelimit-remaining-tokens": "7900",
                 "x-ratelimit-reset-tokens": "547ms",
                 "content-type": "application/json"},
    )
    monkeypatch.setattr(requests, "post", lambda *a, **k: response)

    provider = GroqProvider(model="m", api_key="k")
    provider.generate("s", "u", temperature=0.0, max_tokens=8)

    assert provider.last_headers["x-ratelimit-remaining-tokens"] == "7900"
    # Non-rate-limit headers are not carried, so a caller cannot come to depend
    # on anything but the accounting.
    assert "content-type" not in provider.last_headers


def test_last_headers_exist_before_any_call():
    """The eval runner reads this attribute between rows, so a provider that has
    not made a call yet must not raise on it."""
    provider = GroqProvider(model="m", api_key="k")
    assert provider.last_headers == {}
    assert provider.last_total_tokens is None


_DAILY_BODY = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_1` "
    "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used "
    "199945, Requested 73. Please try again in 17.8s."
)


def test_daily_quota_429_is_not_retried(monkeypatch):
    """A daily cap is not a rate to back off from.

    Groq reports it only in the body, and the headers keep advertising the
    per-minute window as healthy. So the generic 429 path honours the 8-second
    `Retry-After`, retries, fails again, and spends four attempts and seven
    minutes of sleeping before surfacing a message that still says "429". This
    is the case that turned an eval run into a hang.
    """
    attempts = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: attempts.append(1)
                        or _Response({"error": _DAILY_BODY}, status=429))
    monkeypatch.setattr(time, "sleep", lambda _s: pytest.fail("slept on a daily cap"))

    with pytest.raises(requests.HTTPError) as excinfo:
        GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                      max_tokens=8)

    assert len(attempts) == 1
    message = str(excinfo.value)
    assert "daily quota" in message
    assert "199945 of 200000" in message


def test_daily_quota_message_reports_the_reset_delay():
    from rag_bot.providers.groq import _daily_quota_error

    hours = _daily_quota_error(
        "on tokens per day (TPD): Limit 200000, Used 199945. "
        "Please try again in 17h57m7.2s."
    )
    assert "1020 min" in hours
    minutes = _daily_quota_error(
        "on requests per day (RPD): Limit 1000, Used 999. Please try again in 2.5m."
    )
    assert "2 min" in minutes


def test_per_minute_429_is_still_retried(monkeypatch):
    """Only the DAILY windows are terminal. A per-minute cap clears in seconds,
    and retrying is exactly right -- treating it as terminal would turn a
    recoverable blip into a hard failure."""
    responses = [_Response({"error": "Rate limit reached ... on tokens per "
                                     "minute: Limit 8000"}, status=429),
                 _Response(_completion("ok"))]
    attempts = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: attempts.append(1) or responses.pop(0))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert GroqProvider(model="m", api_key="k").generate("s", "u", temperature=0.0,
                                                          max_tokens=8) == "ok"
    assert len(attempts) == 2
