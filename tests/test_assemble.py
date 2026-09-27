"""Answer assembly and the footer (architecture 14.3).

The footer is the one part of an answer that is guaranteed to be truthful, because
no model wrote it. These tests are therefore mostly about the failure that cannot
be caught downstream: a plausible, wrong date.

A model asked to write "Last updated from sources: 27 Sep 2026" produces that
line for any input, including a snapshot taken in 2024. The date only means
something because it is read from the cited chunk's `fetched_at`, so the tests
check the mapping from timestamp to string, and check that every path through
`format_footer` produces a date or the word "unknown" -- never nothing, and never
a guess.
"""
from __future__ import annotations

import pytest

from rag_bot.answer.assemble import (
    assemble,
    build_footer,
    cited_chunk,
    format_date,
    format_footer,
    parse_fetched_at,
)
from rag_bot.types import Chunk, Outcome, ScoredChunk, Validation


def _scored(
    *,
    source_id: str = "groww-S1",
    scheme_id: str | None = "S1",
    scheme_name: str = "HDFC Large Cap Fund",
    publisher: str = "Groww",
    url: str = "https://groww.in/mutual-funds/hdfc-large-cap-fund",
    fetched_at: str | None = "2026-09-27T10:00:00",
    score: float = 0.7,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"{source_id}::0", text="Expense ratio: 1.03", source_id=source_id,
            url=url, publisher=publisher, source_tier="brief", scheme_id=scheme_id,
            scheme_name=scheme_name, heading="Expense ratio", topic="expense_ratio",
            chunk_index=0, n_tokens=8, fetched_at=fetched_at,
            factsheet_url="https://www.hdfcfund.com/mutual-funds/factsheets",
            content_hash="abc123",
        ),
        score=score,
    )


# ---------------------------------------------------------------------------
# format_footer: the date is real or it is "unknown"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("iso,expected", [
    ("2026-09-27T10:00:00", "27 Sep 2026"),
    ("2026-09-27", "27 Sep 2026"),
    ("2026-09-27T10:00:00+05:30", "27 Sep 2026"),
    ("2026-09-27T10:00:00Z", "27 Sep 2026"),
    ("2026-09-27T00:00:00z", "27 Sep 2026"),
    ("2026-01-05T09:00:00", "05 Jan 2026"),
    ("2026-12-31T23:59:59", "31 Dec 2026"),
])
def test_iso_timestamps_render_as_dd_mon_yyyy(iso, expected):
    assert format_date(iso) == expected
    assert format_footer(iso) == f"Last updated from sources: {expected}"


@pytest.mark.parametrize("bad", [
    None, "", "   ", "not a date", "2026-13-45", "27/09/2026", "yesterday",
    "2026", 12345, [],
])
def test_unparseable_dates_render_unknown_and_never_raise(bad):
    assert format_date(bad) == "unknown"
    assert format_footer(bad) == "Last updated from sources: unknown"


def test_footer_never_omits_the_date():
    """An omitted date reads as "we know and are not telling you". "unknown"
    reads as what it is, which is the only honest thing to say about an
    untimestamped snapshot."""
    for iso in (None, "", "garbage", "2026-09-27T10:00:00"):
        assert "Last updated from sources:" in format_footer(iso)
        assert format_footer(iso).split(": ", 1)[1] != ""


def test_month_names_are_locale_independent():
    assert "Sep" in format_date("2026-09-27")
    assert "Jan" in format_date("2026-01-01")
    assert "Dec" in format_date("2026-12-01")
    assert parse_fetched_at("2026-09-27") is not None


def test_z_suffix_parses_on_every_supported_python():
    """fromisoformat only accepted "Z" from 3.11; the explicit handling means
    the footer cannot depend on the interpreter's minor version."""
    assert parse_fetched_at("2026-09-27T10:00:00Z") is not None
    assert parse_fetched_at("2026-09-27T10:00:00z") is not None


# ---------------------------------------------------------------------------
# the footer is assembled in code
# ---------------------------------------------------------------------------

def test_footer_names_publisher_and_scheme():
    footer = build_footer(_scored())
    assert footer == ("Source: Groww - HDFC Large Cap Fund\n"
                      "Last updated from sources: 27 Sep 2026")


def test_footer_with_no_chunk_still_states_the_date_as_unknown():
    assert build_footer(None) == "Last updated from sources: unknown"


# ---------------------------------------------------------------------------
# cited_chunk
# ---------------------------------------------------------------------------

def test_cited_chunk_is_found_by_the_validated_url():
    """Located via validation.cited_url, which came from chunk metadata -- never
    from model text."""
    a, b = _scored(), _scored(source_id="groww-S2", url="https://groww.in/b")
    validation = Validation(cited_url="https://groww.in/b")
    assert cited_chunk([a, b], validation).chunk.source_id == "groww-S2"


def test_cited_chunk_falls_back_to_rank_one():
    """Matches check_single_url's documented fallback."""
    a, b = _scored(), _scored(source_id="groww-S2", url="https://groww.in/b")
    assert cited_chunk([a, b], Validation()).chunk.source_id == "groww-S1"


def test_cited_chunk_with_no_chunks_is_none():
    assert cited_chunk([], Validation()) is None


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def _assemble(outcome, body, chunks, **kwargs):
    return assemble(
        outcome, body, chunks,
        k=kwargs.pop("k", 4),
        top_score=kwargs.pop("top_score", None),
        validation=kwargs.pop("validation", Validation()),
        latency=kwargs.pop("latency", {"generate": 0.5}),
        triage_layer=kwargs.pop("triage_layer", "rules"),
        reason=kwargs.pop("reason", None),
    )


def test_class_a_answer_gets_body_plus_footer():
    chunks = [_scored()]
    answer = _assemble(
        Outcome.A_ANSWERED, "The expense ratio is 1.03%.", chunks,
        validation=Validation(cited_url=chunks[0].chunk.url))
    assert answer.text == ("The expense ratio is 1.03%.\n"
                           "Source: Groww - HDFC Large Cap Fund\n"
                           "Last updated from sources: 27 Sep 2026")
    assert answer.source_url == "https://groww.in/mutual-funds/hdfc-large-cap-fund"
    assert answer.publisher == "Groww"
    assert answer.scheme_name == "HDFC Large Cap Fund"
    assert answer.last_updated == "27 Sep 2026"


def test_class_a_never_renders_a_bare_footer():
    answer = _assemble(Outcome.A_ANSWERED, "", [_scored()])
    assert answer.text == ""


@pytest.mark.parametrize("outcome", [
    Outcome.B_NOT_IN_CORPUS, Outcome.C_ADVICE_REFUSED,
    Outcome.D_PERFORMANCE_REFUSED, Outcome.ERROR,
])
def test_non_a_outcomes_have_no_source_no_footer(outcome):
    """F7 gives class A exactly one URL and the other classes none, so a refusal
    can never be mistaken for a sourced answer."""
    body = "This is a refusal template."
    answer = _assemble(outcome, body, [_scored()])
    assert answer.text == body
    assert answer.source_url is None
    assert answer.publisher is None
    assert answer.last_updated is None
    assert "Source:" not in answer.text
    assert "http" not in answer.text


def test_every_outcome_carries_retrieved_chunks():
    """A class D refusal that shows what retrieval found is auditable; one that
    hides it looks identical to a refusal that never looked."""
    chunks = [_scored(score=0.81), _scored(source_id="groww-S2", score=0.6)]
    for outcome in (Outcome.A_ANSWERED, Outcome.B_NOT_IN_CORPUS,
                    Outcome.C_ADVICE_REFUSED, Outcome.D_PERFORMANCE_REFUSED,
                    Outcome.ERROR):
        answer = _assemble(outcome, "body", chunks)
        assert answer.retrieved_chunks == chunks, outcome
        assert len(answer.retrieved_chunks) == 2, outcome


def test_error_answer_still_carries_chunks():
    """The demo keeps working and no citation is fabricated, because the citation
    comes from chunk metadata rather than model text."""
    chunks = [_scored()]
    answer = _assemble(Outcome.ERROR, "", chunks)
    assert answer.retrieved_chunks == chunks
    assert answer.source_url is None


def test_top_score_is_derived_when_not_supplied():
    answer = _assemble(Outcome.A_ANSWERED, "x.", [_scored(score=0.81)])
    assert answer.top_score == 0.81


def test_supplied_top_score_wins():
    answer = _assemble(Outcome.A_ANSWERED, "x.", [_scored(score=0.81)],
                       top_score=0.42)
    assert answer.top_score == 0.42


def test_latency_and_triage_metadata_pass_through():
    answer = _assemble(
        Outcome.A_ANSWERED, "x.", [_scored()],
        latency={"scrub": 0.001, "triage": 0.002, "retrieve": 0.05,
                 "gate": 0.01, "generate": 0.9, "assemble": 0.001},
        triage_layer="llm", reason="no_rule_matched")
    assert answer.latency["generate"] == 0.9
    assert answer.triage_layer == "llm"
    assert answer.reason == "no_rule_matched"


def test_validation_is_carried_onto_the_answer():
    validation = Validation(sentences_ok=False, no_figures=False,
                            unverified=True, cited_url="https://groww.in/x",
                            notes=["removed return figures: ['12.4%']"])
    answer = _assemble(Outcome.A_ANSWERED, "The expense ratio is 1.03%.", [_scored()],
                       validation=validation)
    assert answer.validation is validation
    assert answer.validation.notes == ["removed return figures: ['12.4%']"]


def test_missing_fetched_at_renders_unknown_in_the_footer():
    """The cited chunk has no timestamp, so the date is unknown. A fabricated date
    is the single worst outcome of this module, because it looks correct."""
    chunks = [_scored(fetched_at=None)]
    answer = _assemble(
        Outcome.A_ANSWERED, "The exit load is 1%.", chunks,
        validation=Validation(cited_url=chunks[0].chunk.url))
    assert "Last updated from sources: unknown" in answer.text
    assert answer.last_updated == "unknown"


def test_garbage_fetched_at_renders_unknown_in_the_footer():
    chunks = [_scored(fetched_at="sometime last year")]
    answer = _assemble(
        Outcome.A_ANSWERED, "The exit load is 1%.", chunks,
        validation=Validation(cited_url=chunks[0].chunk.url))
    assert "Last updated from sources: unknown" in answer.text
    assert "last year" not in answer.text


def test_the_footer_date_comes_from_the_cited_chunk_not_the_top_chunk():
    """Two chunks, two different fetch dates. The answer is about the cited one,
    so the footer must be about the cited one."""
    a = _scored(source_id="groww-S1", url="https://groww.in/a",
                fetched_at="2026-01-01T00:00:00", scheme_name="HDFC Large Cap Fund")
    b = _scored(source_id="groww-S2", url="https://groww.in/b",
                fetched_at="2026-09-27T00:00:00", scheme_name="HDFC Flexi Cap")
    answer = _assemble(
        Outcome.A_ANSWERED, "The exit load is 1%.", [a, b],
        validation=Validation(cited_url="https://groww.in/b"))
    assert "27 Sep 2026" in answer.text
    assert "01 Jan 2026" not in answer.text
    assert answer.scheme_name == "HDFC Flexi Cap"
    assert answer.last_updated == "27 Sep 2026"


def test_assemble_never_raises_on_hostile_input():
    for body in ("", "   ", None, "\x00", "Source: [1]", "x" * 10000):
        for chunks in ([], [_scored(fetched_at=None)], [_scored(fetched_at="junk")]):
            for validation in (Validation(), Validation(cited_url="https://nowhere")):
                _assemble(Outcome.A_ANSWERED, body, chunks, validation=validation)
                _assemble(Outcome.ERROR, body, chunks, validation=validation)
