"""Exact scheme resolution (architecture 14.4).

The failure this module exists to prevent: the PRD's own problem statement is that
LLMs "mix up figures across similarly named schemes". A wrong scheme_id is worse
than no scheme_id, because a wrong one scopes retrieval *confidently* to the wrong
fund. So the tests below are mostly about what must NOT resolve.
"""
from __future__ import annotations

import pytest

from rag_bot.answer.schemes import _normalise, resolve_scheme
from rag_bot.config import load
from rag_bot.sources.fetch import load_sources
from rag_bot.types import Source


@pytest.fixture(scope="module")
def sources() -> list[Source]:
    return load_sources(load().sources_file)


def ids(sources: list[Source]) -> dict[str, str]:
    return {s.scheme_id: s.scheme_name for s in sources}


# ---------------------------------------------------------------------------
# exact match works
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("What is the expense ratio of HDFC Large Cap Fund?", "S1"),
    ("What is the expense ratio of the large cap fund?", "S1"),
    ("expense ratio of large cap fund", "S1"),
    ("Ongoing charges on the HDFC Flexi Cap Fund?", "S2"),
    ("What is the lock-in on the ELSS?", "S3"),
    ("Minimum SIP for the small cap fund?", "S4"),
    ("Tell me the benchmark for the balanced advantage fund.", "S5"),
    ("the ELSS tax saver fund's exit load", "S3"),
    ("What about the flexi cap fund?", "S2"),
])
def test_exact_alias_match_resolves(sources, question, expected):
    scheme_id, ambiguous = resolve_scheme(question, sources)
    assert scheme_id == expected, question
    assert ambiguous is False


def test_scheme_id_itself_is_an_alias(sources):
    assert resolve_scheme("What is the expense ratio of S1?", sources) == ("S1", False)


def test_matching_is_case_and_whitespace_insensitive(sources):
    assert resolve_scheme("WHAT IS THE EXPENSE RATIO OF   HDFC   LARGE CAP FUND?",
                          sources) == ("S1", False)
    assert resolve_scheme("  large cap fund  ", sources) == ("S1", False)


def test_several_aliases_of_one_scheme_is_not_ambiguous(sources):
    """"hdfc large cap fund" and "large cap fund" both hit, but that is ONE fund."""
    scheme_id, ambiguous = resolve_scheme(
        "Compare the expense ratio of HDFC Large Cap Fund against the large cap fund",
        sources)
    assert scheme_id == "S1"
    assert ambiguous is False


# ---------------------------------------------------------------------------
# what must NOT resolve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "equity",
    "the equity",
    "Equity Large Cap",
    # "equity small cap" is S4's full category label. It contains "small cap
    # fund"-adjacent words but not the alias, so it must stay unresolved.
    "equity small cap",
])
def test_bare_category_words_do_not_resolve(sources, question):
    """Architecture 14.4, and the spec's named example.

    "equity" is in "HDFC Equity Fund" (S2) and in the category labels of S1
    ("Equity Large Cap") and S4 ("Equity Small Cap"). A substring or fuzzy match
    here silently attributes one fund's figure to another.
    """
    scheme_id, ambiguous = resolve_scheme(question, sources)
    assert scheme_id != "S2", question
    assert scheme_id is None, question


def test_bare_category_word_does_not_resolve_to_its_own_scheme_either(sources):
    """'large cap' is S1's own category label, which is why it is not an alias."""
    for question in ("large cap", "small cap", "flexi cap", "balanced advantage"):
        scheme_id, _ = resolve_scheme(question, sources)
        assert scheme_id is None, question


def test_no_match_returns_none_not_ambiguous(sources):
    assert resolve_scheme("What is the expense ratio?", sources) == (None, False)
    assert resolve_scheme("Who won the 1994 World Cup?", sources) == (None, False)


def test_two_schemes_is_ambiguous_not_a_pick(sources):
    """Architecture 14.4 item 6: the assistant does not guess across schemes."""
    scheme_id, ambiguous = resolve_scheme(
        "What's the minimum SIP for the flexi cap fund versus the small cap fund?",
        sources)
    assert scheme_id is None
    assert ambiguous is True


def test_three_schemes_is_also_ambiguous(sources):
    scheme_id, ambiguous = resolve_scheme(
        "Compare the large cap fund, the ELSS and the small cap fund", sources)
    assert scheme_id is None and ambiguous is True


def test_alias_does_not_match_inside_a_longer_token(sources):
    """"s1" must not match inside "s1234"."""
    assert resolve_scheme("tell me about s1234 the fund", sources) == (None, False)


def test_substring_of_an_alias_does_not_resolve(sources):
    """"large cap fund" must not match inside "large cap fundsector"."""
    assert resolve_scheme("the large cap fundsector exposure", sources) == (None, False)


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------

def test_empty_inputs(sources):
    assert resolve_scheme("", sources) == (None, False)
    assert resolve_scheme("anything", []) == (None, False)
    assert resolve_scheme("anything", None) == (None, False)


def test_sources_without_a_scheme_id_are_skipped():
    """A cross-scheme page cannot scope retrieval to one fund."""
    cross = Source(
        source_id="x", url="https://x.example", publisher="x.example",
        source_tier="brief", scheme_id=None, scheme_name="All schemes",
        factsheet_url=None, aliases=("all schemes",),
    )
    assert resolve_scheme("tell me about all schemes", [cross]) == (None, False)


def test_aliases_with_punctuation_match():
    src = Source(
        source_id="s", url="https://x.example", publisher="x",
        source_tier="brief", scheme_id="S9", scheme_name="Test",
        factsheet_url=None, aliases=("a/c wise fund", "hdfc test (direct)"),
    )
    assert resolve_scheme("the A/C WISE fund please", [src]) == ("S9", False)
    assert resolve_scheme("HDFC Test (Direct) fees", [src]) == ("S9", False)


def test_empty_alias_is_ignored():
    src = Source(
        source_id="s", url="https://x.example", publisher="x",
        source_tier="brief", scheme_id="S9", scheme_name="Test",
        factsheet_url=None, aliases=("", "   "),
    )
    assert resolve_scheme("anything at all", [src]) == (None, False)


def test_normalise_collapses_whitespace():
    assert _normalise("  HDFC   Large\n\tCap Fund  ") == "hdfc large cap fund"


# ---------------------------------------------------------------------------
# the real registry must have no bare category words in its alias tables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bare", [
    "large cap", "small cap", "flexi cap", "balanced advantage", "equity",
])
def test_registry_aliases_exclude_bare_category_words(sources, bare):
    """The table is the mechanism; this asserts the mechanism was maintained."""
    for source in sources:
        assert bare not in {a.lower() for a in source.aliases}, (
            f"{source.scheme_id} has the bare category word {bare!r} as an alias"
        )


def test_every_scheme_is_reachable_by_at_least_one_alias(sources):
    """An unreachable scheme could never be resolved, and would silently fall
    through to an unscoped answer."""
    reachable = set()
    for question in [a for s in sources for a in s.aliases]:
        scheme_id, _ = resolve_scheme(f"tell me about {question}", sources)
        if scheme_id:
            reachable.add(scheme_id)
    expected = {s.scheme_id for s in sources if s.scheme_id}
    assert reachable == expected, f"unreachable: {expected - reachable}"


def test_five_schemes_are_registered(sources):
    assert len(sources) == 5
    assert {s.scheme_id for s in sources} == {"S1", "S2", "S3", "S4", "S5"}


# ---------------------------------------------------------------------------
# the safety property: unresolved must fail SAFE, never misattribute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    # misspellings and concatenations a real user will type
    ("fees on the largecap fund", None),
    ("smallcap fund sip", None),
    ("flexicap fund", None),
    ("balancedadvantage fund", None),
    ("tax saver", None),
    # partial phrases: a bare category word is not a scheme
    ("flexi cap exit load", None),
    ("tax saver lock in", None),
    ("the large fund", None),
    ("cap fund", None),
    ("hdfc fund", None),
    ("growth fund", None),
    ("value fund", None),
])
def test_unrecognised_phrasing_resolves_to_nothing(sources, question, expected):
    """Failing to resolve is the SAFE direction: the caller then asks which of the
    five schemes (§14.4 item 5). Fuzzy matching would turn these into a confident
    answer about the wrong fund, which is the failure the PRD is named for."""
    assert resolve_scheme(question, sources) == (expected, False), question


@pytest.mark.parametrize("question,expected", [
    ("hdfc equity fund", "S2"),
    ("equity fund", "S2"),
    ("flexi cap fund", "S2"),
    ("large cap fund", "S1"),
    ("small cap fund", "S4"),
    ("balanced advantage fund", "S5"),
    ("elss tax saver fund", "S3"),
])
def test_recognisable_phrasing_resolves_to_the_right_scheme(sources, question, expected):
    assert resolve_scheme(question, sources)[0] == expected


def test_no_confusable_resolves_to_the_wrong_scheme(sources):
    """Sweep: every string either resolves correctly or not at all.

    This is the invariant the whole module exists for. A wrong scheme_id scopes
    retrieval confidently to the wrong fund, so 'never wrong' matters far more than
    'usually right'.
    """
    truth = {
        "large cap fund": "S1", "hdfc large cap": "S1",
        "equity fund": "S2", "flexi cap fund": "S2",
        "elss": "S3", "tax saver fund": "S3",
        "small cap fund": "S4",
        "balanced advantage fund": "S5",
    }
    for text, expected in truth.items():
        got = resolve_scheme(f"what is the fee on the {text}", sources)[0]
        assert got in (expected, None), (text, got)
