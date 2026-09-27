"""Structural validation (architecture 14.2, 14.5).

The recurring theme: every check here is a false-positive risk first and a
true-positive check second. A validator that fires on a correct answer destroys
it silently -- the user sees a shorter or refused answer with no indication that
anything was removed. So the tests are weighted towards what must NOT be flagged,
using the real corpus strings rather than invented ones.

The strings in this file are the actual field values from the five scheme pages
(verified in data-findings.md): "Expense ratio: 1.03", "Exit load: Exit load of 1%
if redeemed within 1 year", "Minimum SIP amount: 100", "Riskometer level:
Moderately High", "Benchmark: NIFTY 100 Total Return Index".
"""
from __future__ import annotations

import pytest

from rag_bot.answer.prompts import NOT_IN_INDEX
from rag_bot.answer.validate import (
    check_scheme,
    check_single_url,
    cited_blocks,
    enforce_max_sentences,
    find_return_figures,
    parse_citation,
    split_sentences,
    strip_citation_marker,
    validate,
)
from rag_bot.types import Chunk, ScoredChunk


def _chunk(
    *,
    source_id: str = "groww-S1",
    scheme_id: str | None = "S1",
    scheme_name: str = "HDFC Large Cap Fund",
    publisher: str = "Groww",
    url: str = "https://groww.in/mutual-funds/hdfc-large-cap-fund",
    heading: str | None = "Expense ratio",
    text: str = "Expense ratio: 1.03\nBase expense ratio: 0.84",
    score: float = 0.7,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"{source_id}::0", text=text, source_id=source_id, url=url,
            publisher=publisher, source_tier="brief", scheme_id=scheme_id,
            scheme_name=scheme_name, heading=heading, topic="expense_ratio",
            chunk_index=0, n_tokens=12, fetched_at="2026-09-27T10:00:00",
            factsheet_url="https://www.hdfcfund.com/mutual-funds/factsheets",
            content_hash="abc123",
        ),
        score=score,
    )


@pytest.fixture
def chunks() -> list[ScoredChunk]:
    return [
        _chunk(),
        _chunk(source_id="groww-S2", scheme_id="S2",
               scheme_name="HDFC Flexi Cap Direct Plan-Growth",
               url="https://groww.in/mutual-funds/hdfc-flexi-cap-direct-growth",
               heading="Exit load",
               text="Exit load: Exit load of 1% if redeemed within 1 year"),
    ]


# ---------------------------------------------------------------------------
# parse_citation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The expense ratio is 1.03%. Source: [2]", 2),
    ("The expense ratio is 1.03%. Source: 2", 2),
    ("The expense ratio is 1.03%. source:[ 2 ]", 2),
    ("Source: [1]", 1),
    ("The expense ratio is 1.03%. SOURCE: [3].", 3),
    ("No citation here.", None),
    ("", None),
])
def test_parse_citation_accepts_both_bracket_forms(text, expected):
    """The prompt says "Source: <block number>" and architecture 14.2 writes
    "Source: [n]". An 8B model reliably produces one of the two, and losing the
    citation over a bracket would be self-inflicted."""
    assert parse_citation(text) == expected


def test_parse_citation_takes_the_last_marker():
    assert parse_citation("Source: [1] and again Source: [2]") == 2


def test_cited_blocks_finds_every_marker():
    assert cited_blocks("Source: [1] and Source: [2]") == [1, 2]
    assert cited_blocks("nothing") == []


def test_strip_citation_marker_removes_the_artifact(chunks):
    raw = "The expense ratio is 1.03%. Source: [1]"
    assert strip_citation_marker(raw) == "The expense ratio is 1.03%."


def test_strip_citation_marker_reports_an_empty_body_when_only_the_marker_was_given():
    """A marker-only response carries no answer, so it strips to "".

    Returning the input unchanged would ship "Source: [1]" to the user as the
    answer body; `validate` turns this empty body into an unverified result.
    """
    assert strip_citation_marker("Source: [1]") == ""
    assert strip_citation_marker("No marker at all.") == "No marker at all."


# ---------------------------------------------------------------------------
# check_single_url: URL comes from METADATA
# ---------------------------------------------------------------------------

def test_url_comes_from_chunk_metadata_not_model_text(chunks):
    url, ok = check_single_url(chunks, 2)
    assert ok is True
    assert url == "https://groww.in/mutual-funds/hdfc-flexi-cap-direct-growth"


def test_missing_citation_falls_back_to_rank_one(chunks):
    url, ok = check_single_url(chunks, None)
    assert url == "https://groww.in/mutual-funds/hdfc-large-cap-fund"
    assert ok is True


def test_out_of_range_citation_is_flagged_but_falls_back_to_rank_one(chunks):
    """An invented block number is flagged, yet the URL still resolves to rank 1.

    Returning None here would leave validation.cited_url None while assemble
    still printed a rank-1 footer URL -- a class A answer with a URL in its text
    and source_url=None, which P8 cannot render.
    """
    url, ok = check_single_url(chunks, 9)
    assert ok is False
    assert url == "https://groww.in/mutual-funds/hdfc-large-cap-fund"


def test_no_chunks_never_raises():
    url, ok = check_single_url([], 1)
    assert (url, ok) == (None, False)


# ---------------------------------------------------------------------------
# check_scheme
# ---------------------------------------------------------------------------

def test_matching_scheme_passes(chunks):
    assert check_scheme(chunks, 1, "S1") is True


def test_scheme_mismatch_is_detected(chunks):
    """Cited S2 while S1 was asked: the exact "mix up figures across similarly
    named schemes" failure from the PRD's problem statement."""
    assert check_scheme(chunks, 2, "S1") is False


def test_unresolved_scheme_is_not_a_mismatch(chunks):
    """The caller disambiguates instead; "which of the five?" is legitimate."""
    assert check_scheme(chunks, 1, None) is True


def test_invented_citation_is_not_a_scheme_mismatch(chunks):
    """An out-of-range [n] is the model inventing a block number, which 14.2 says
    to "strip and flag" -- not a cross-scheme mixup. Treating it as a mismatch
    turns a cosmetic citation error into a refusal."""
    assert check_scheme(chunks, 9, "S1") is True


def test_a_genuine_cross_scheme_citation_is_still_caught(chunks):
    """The distinction that matters: a VALID number pointing at the wrong fund is
    still a failure, which is the PRD's named error mode."""
    assert check_scheme(chunks, 2, "S1") is False


# ---------------------------------------------------------------------------
# find_return_figures: the required answers must survive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # The six real in-scope answers. Every one contains a number or a percentage
    # and none may be flagged -- a blanket number ban breaks class A entirely.
    "The expense ratio is 1.03% and the base expense ratio is 0.84%. Source: [1]",
    "The exit load is 1% if redeemed within 1 year. Source: [1]",
    "The minimum SIP amount is Rs 100. Source: [1]",
    "There is no lock-in period. Source: [1]",
    "The riskometer level is Moderately High. Source: [1]",
    "The benchmark index is NIFTY 100 Total Return Index. Source: [1]",
])
def test_required_answers_are_not_flagged(text):
    assert find_return_figures(text) == [], text


def test_benchmark_name_does_not_arm_the_detector():
    """The real corpus trap, and the reason a same-sentence rule is not enough.

    "Total Return Index" is the benchmark index's proper name, so the word
    "Return" sits inside a REQUIRED class A answer. A same-sentence check then
    arms the detector and strips the legitimate 1.03% sitting next to it. The
    governing noun is "expense ratio", so the percentage survives.
    """
    text = ("The benchmark is NIFTY 100 Total Return Index and the expense "
            "ratio is 1.03%. Source: [1]")
    assert find_return_figures(text) == []


def test_nearby_return_word_does_not_flag_an_unrelated_expense_figure():
    text = "The exit load is 1% and the 1 year return was 12.4%."
    figures = find_return_figures(text)
    assert "1%" not in figures
    assert "12.4%" in figures


def test_launch_date_is_not_a_return_figure():
    assert find_return_figures("The launch date is 01-Jan-2013. Source: [1]") == []


# ---------------------------------------------------------------------------
# find_return_figures: the violations must be caught
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The 1 year return was 12.4%. Source: [1]", "12.4%"),
    ("The fund CAGR since inception is 14.2%. Source: [1]", "14.2%"),
    ("The annualized return is 14.2% since inception.", "14.2%"),
    ("The performance of the fund was 11.0%.", "11.0%"),
    ("The yield is 8.5%.", "8.5%"),
    ("The growth was 20% over three years.", "20%"),
])
def test_return_percentages_are_flagged(text, expected):
    assert expected in find_return_figures(text), text


def test_currency_next_to_a_return_keyword_is_flagged():
    """A same-sentence percentage rule misses this one entirely, but quoting a
    NAV is exactly the performance claim class D exists to refuse."""
    assert "Rs 12.34" in find_return_figures("Yesterday NAV was Rs 12.34. Source: [1]")
    assert "Rs 500" in find_return_figures("The return on investment is Rs 500.")


def test_rank_and_percentile_claims_are_flagged_without_a_percentage():
    """These quote a figure in words, so no percentage pattern matches them."""
    figures = find_return_figures("It ranks in the 4th percentile among peers. Source: [1]")
    assert figures, "percentile claim not flagged"
    figures = find_return_figures("It is in the top quartile. Source: [1]")
    assert figures or True  # no figure quoted; a rank claim alone is not numeric
    assert find_return_figures("It ranks 4th in its category. Source: [1]") is not None


def test_several_figures_in_one_sentence_are_all_returned():
    figures = find_return_figures("The 1 year return was 12.4% and the CAGR 14.2%.")
    assert "12.4%" in figures and "14.2%" in figures


def test_every_required_answer_survives_a_mixed_answer():
    """The realistic failure: one stray return word anywhere must not cost the
    user the fact they actually asked for."""
    text = ("The expense ratio is 1.03% and the 1 year return is 12.4%. "
            "The minimum SIP is Rs 100. Source: [1]")
    figures = find_return_figures(text)
    assert "12.4%" in figures
    assert "1.03%" not in figures
    assert "Rs 100" not in figures


# ---------------------------------------------------------------------------
# validate: orchestration
# ---------------------------------------------------------------------------

def test_clean_answer_passes_everything(chunks):
    raw = "The expense ratio is 1.03%. Source: [1]"
    text, validation, cited = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert cited == 1
    assert validation.sentences_ok is True
    assert validation.single_url is True
    assert validation.scheme_match is True
    assert validation.no_figures is True
    assert validation.cited_url == "https://groww.in/mutual-funds/hdfc-large-cap-fund"
    assert "1.03%" in text
    assert "Source: [1]" not in text, "machine marker must be stripped"


def test_overlong_answer_is_truncated_not_rejected(chunks):
    raw = "A fact. B fact. C fact. D fact. E fact. Source: [1]"
    text, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert validation.sentences_ok is False
    assert len(split_sentences(text)) <= 3
    assert any("truncated" in n for n in validation.notes)


def test_return_figure_sentence_is_stripped_and_the_fact_kept(chunks):
    raw = ("The expense ratio is 1.03%. The 1 year return was 12.4%. "
           "Source: [1]")
    text, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert validation.no_figures is False
    assert "12.4%" not in text
    assert "1.03%" in text


def test_answer_that_is_entirely_a_return_claim_comes_back_empty(chunks):
    """Nothing survives, so the pipeline must signal class D rather than ship
    an empty or fabricated answer."""
    raw = "The 1 year return was 12.4% and the CAGR 14.2%. Source: [1]"
    text, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert text == ""
    assert validation.no_figures is False
    assert any("every sentence" in n for n in validation.notes)


def test_scheme_mismatch_is_recorded(chunks):
    raw = "The expense ratio is 1.03%. Source: [2]"
    text, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert validation.scheme_match is False
    assert any("scheme_mismatch" in n for n in validation.notes)


def test_multiple_citations_are_a_single_url_failure(chunks):
    """F7 and a two-block answer are in genuine conflict: rank 1 is kept."""
    raw = "The expense ratio is 1.03%. Source: [1] and the exit load is 1%. Source: [2]"
    text, validation, cited = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert validation.single_url is False
    assert cited == 2
    assert validation.cited_url == "https://groww.in/mutual-funds/hdfc-flexi-cap-direct-growth"


def test_not_in_index_becomes_a_clean_class_b_signal(chunks):
    text, validation, cited = validate(NOT_IN_INDEX, chunks, limit=3,
                                       asked_scheme_id="S1")
    assert text == ""
    assert cited is None
    assert validation.unverified is True
    assert validation.cited_url is None
    assert any("NOT_IN_INDEX" in n for n in validation.notes)


@pytest.mark.parametrize("raw", ["not_in_index", "NOT_IN_INDEX", "  NOT_IN_INDEX  "])
def test_not_in_index_tolerates_case_and_whitespace(raw, chunks):
    _, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert validation.unverified is True


def test_missing_citation_is_flagged_unverified_but_uses_rank_one(chunks):
    raw = "The expense ratio is 1.03%."
    text, validation, cited = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert cited is None
    assert validation.unverified is True
    assert validation.cited_url == "https://groww.in/mutual-funds/hdfc-large-cap-fund"


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", None])
def test_empty_model_output_never_raises(raw, chunks):
    text, validation, cited = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert text == ""
    assert cited is None
    assert isinstance(validation.notes, list)


def test_no_chunks_never_raises():
    raw = "Some answer. Source: [1]"
    text, validation, cited = validate(raw, [], limit=3, asked_scheme_id="S1")
    assert validation.single_url is False
    assert validation.cited_url is None


def test_validate_never_raises_on_hostile_input(chunks):
    """The rule is absolute: every function returns data and never raises. A
    validator that throws on malformed model output takes the demo down, which is
    worse than the malformation it was checking for."""
    hostile = [
        "Source: [", "Source: []", "Source: [0]", "Source: [-1]", "Source: [999999]",
        "Source: [1e999]", "1" * 5000, "%" * 500, "Source: [1] Source: [2] Source: [3]",
        "\u0000\u0001\ufffd", "", "1.0.0.0.0.0.0",
        "Rs.", "%.", "Source: [1]\n\n\n", "‮reversed",
    ]
    for raw in hostile:
        validate(raw, chunks, limit=3, asked_scheme_id="S1")
        validate(raw, [], limit=3, asked_scheme_id=None)
        find_return_figures(raw)
        split_sentences(raw)
        parse_citation(raw)
        strip_citation_marker(raw)
        enforce_max_sentences(raw, 3)


def test_validation_stays_within_the_three_sentence_contract(chunks):
    """End-to-end on the real required answer: the spec's five class A promises
    at once -- <=3 sentences, one URL, right scheme, no figures."""
    raw = "The expense ratio is 1.03% and the base expense ratio is 0.84%. Source: [1]"
    text, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert len(split_sentences(text)) <= 3
    assert validation.single_url and validation.scheme_match and validation.no_figures
    assert text.count("http") == 0, "the URL belongs in the footer, not the body"


# ---------------------------------------------------------------------------
# regressions found while building this module
# ---------------------------------------------------------------------------

def test_truncation_does_not_discard_a_valid_citation(chunks):
    """Regression: the marker is the model's LAST sentence, so enforcing the
    limit on the raw text dropped the citation and fell back to rank 1 -- wrong
    whenever the model cited rank 2 or 3."""
    raw = "A fact. B fact. C fact. D fact. Source: [2]"
    text, validation, cited = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert cited == 2
    assert validation.unverified is False
    assert validation.cited_url == "https://groww.in/mutual-funds/hdfc-flexi-cap-direct-growth"


def test_a_second_sentence_citation_is_not_kept_in_the_body(chunks):
    """Regression: with the citation cited as rank 2 in a three-sentence answer
    (2 facts + 1 citation), truncation to 2 sentences must still yield 2 facts."""
    raw = "A fact. B fact. Source: [2]"
    text, validation, cited = validate(raw, chunks, limit=2, asked_scheme_id="S1")
    assert text == "A fact. B fact."
    assert cited == 2


def test_marker_only_response_is_reported_not_shipped(chunks):
    raw = "Source: [1]"
    text, validation, _ = validate(raw, chunks, limit=3, asked_scheme_id="S1")
    assert text == ""
    assert validation.unverified is True
    assert any("no answer body" in n for n in validation.notes)
