"""Tests for the eval runner's rate-limit pacing.

The pacing is not a convenience. The Groq free tier enforces a per-minute
budget, a per-day budget and a per-model budget at the same time, and a 52-row
eval runs into the daily one: a 200k-token budget against a few thousand tokens
per row. Reading the wrong header, or parsing a reset value as seconds when it
is "17h57m7.2s", turns a quota pause into a wall of 429s, so the parsing is
tested directly rather than trusted.
"""

from __future__ import annotations

import pytest

from rag_bot.eval.run_eval import (
    MAX_PACE_WAIT_S,
    DailyCapExhausted,
    _pace_sleep,
)


class _LLM:
    def __init__(self, **headers: str):
        self.last_headers = headers


def _headers(**overrides: str) -> dict[str, str]:
    base = {
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7500",
        "x-ratelimit-reset-tokens": "30s",
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "900",
        "x-ratelimit-reset-requests": "40s",
    }
    base.update(overrides)
    return base


def test_no_wait_when_the_budget_covers_the_next_row():
    """Pacing is per-row, not per-run.

    The first version compared the window against the cost of every remaining
    row, so 40 rows looked like 160k tokens against an 8k per-minute window and
    always read as exhausted -- it raised `DailyCapExhausted` on a healthy
    quota. The budget that matters is the next row's; the rest refills behind
    it.
    """
    assert _pace_sleep(_LLM(**_headers()), remaining_rows=40) == 0.0


def test_no_headers_means_no_wait():
    """A fake or local provider has no rate-limit accounting, and the runner
    must not stall waiting for headers that will never arrive."""
    assert _pace_sleep(_LLM(), remaining_rows=50) == 0.0


def test_short_reset_waits_for_the_full_window(capsys):
    wait = _pace_sleep(
        _LLM(**_headers(**{"x-ratelimit-remaining-tokens": "10",
                           "x-ratelimit-reset-tokens": "90s"})),
        remaining_rows=10,
    )
    assert wait == 90.0
    assert "pacing" in capsys.readouterr().err


def test_millisecond_reset_is_parsed_as_seconds():
    wait = _pace_sleep(
        _LLM(**_headers(**{"x-ratelimit-remaining-tokens": "10",
                           "x-ratelimit-reset-tokens": "547ms"})),
        remaining_rows=10,
    )
    assert 0.5 < wait < 1.0


def test_hour_formatted_reset_is_not_read_as_seconds():
    """`17h57m7.2s` ends in "s", so a naive `float(text[:-1])` order parses it
    as 17 seconds -- off by a factor of ~3800, and silently.

    The real value is what makes this a daily-cap signal rather than a
    per-minute one, so getting it wrong is the difference between pacing and
    hammering a closed window.
    """
    with pytest.raises(DailyCapExhausted):
        _pace_sleep(
            _LLM(**_headers(**{"x-ratelimit-remaining-tokens": "110",
                               "x-ratelimit-reset-tokens": "17h57m7.2s"})),
            remaining_rows=40,
        )


def test_hour_formatted_reset_below_the_ceiling_waits():
    """Multi-unit parsing, on the branch that decides a long-but-tolerable wait.
    59 minutes is under the 10-minute ceiling only in the sense that the test
    overrides nothing, so this uses the smallest hour-form Groq emits that
    exceeds the ceiling and checks the raise, plus one just under it."""
    wait = _pace_sleep(
        _LLM(**_headers(**{"x-ratelimit-remaining-tokens": "10",
                           "x-ratelimit-reset-tokens": "0h0m30.0s"})),
        remaining_rows=10,
    )
    assert wait == pytest.approx(30.0, rel=0.01)
    with pytest.raises(DailyCapExhausted):
        _pace_sleep(
            _LLM(**_headers(**{"x-ratelimit-remaining-tokens": "10",
                               "x-ratelimit-reset-tokens": "1h0m0.0s"})),
            remaining_rows=10,
        )


def test_exhausted_daily_budget_raises_rather_than_sleeping():
    """The point of the ceiling: hours of waiting is not a plan, it is a hung
    process. The caller turns this into exit code 3."""
    with pytest.raises(DailyCapExhausted) as excinfo:
        _pace_sleep(
            _LLM(**_headers(**{"x-ratelimit-remaining-tokens": "0",
                               "x-ratelimit-reset-tokens": "18h0m0.0s"})),
            remaining_rows=52,
        )
    message = str(excinfo.value)
    assert "52 rows remain" in message
    assert "min" in message


def test_ceiling_is_configurable_and_documented_by_the_test():
    """The test asserts against the real default so that raising it does not
    quietly turn every long window into a multi-hour sleep."""
    assert MAX_PACE_WAIT_S == 600.0


def test_request_window_is_also_considered():
    """Two calls per row, so a request window with fewer than two left is the
    binding constraint even when the token budget looks fine."""
    with pytest.raises(DailyCapExhausted):
        _pace_sleep(
            _LLM(**_headers(**{"x-ratelimit-remaining-requests": "1",
                               "x-ratelimit-reset-requests": "2h0m0.0s"})),
            remaining_rows=40,
        )


def test_malformed_header_is_ignored_rather_than_crashing():
    """A provider that returns nonsense in a header should not take down a
    52-row run; the next successful call will supply a good one."""
    assert _pace_sleep(
        _LLM(**_headers(**{"x-ratelimit-reset-tokens": "soon",
                           "x-ratelimit-limit-tokens": "nope"})),
        remaining_rows=10,
    ) == 0.0
