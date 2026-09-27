"""Sentence splitting (architecture 14.2).

The clause is duplicated from the phase doc on purpose:

    "The ratio is 1.05%." is ONE sentence, not two. "Rs. 500" is one sentence.

It is duplicated because it is the spec's own example, and it is a
false-positive test in disguise. A splitter that breaks "1.05%" in half reports a
perfectly correct one-sentence answer as two sentences, `enforce_max_sentences`
truncates it, and the class A user sees a mangled answer with a broken citation.
The corpus is full of percentages, so a naive splitter breaks the product's core
capability on the most common answer there is.
"""
from __future__ import annotations

import pytest

from rag_bot.answer.validate import enforce_max_sentences, split_sentences


# ---------------------------------------------------------------------------
# decimals: the clause from the spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The expense ratio is 1.05%.", 1),
    ("The ratio is 1.05% p.a.", 1),
    ("The NAV was 1447.3830 on 01-Jan-2013.", 1),
    ("Expense ratios run from 0.84% to 1.05%.", 1),
    ("The exit load is 1% if redeemed within 1 year.", 1),
    ("The minimum SIP amount is Rs 100.", 1),
    ("Base expense ratio: 0.84\nTotal expense ratio: 1.03", 1),
    ("Fund name: HDFC Large Cap Fund\nScheme name: HDFC Large Cap Fund Direct", 1),
])
def test_period_between_digits_never_splits(text, expected):
    assert len(split_sentences(text)) == expected, text


@pytest.mark.parametrize("text", [
    "The minimum SIP is Rs. 500 and the exit load is 1%.",
    "The minimum SIP is Rs.500 and the exit load is 1%.",
    "The ratio is 1.05% p.a. Source: [1]",
    "Approx. 1.03% is the expense ratio.",
    "e.g. the exit load is 1%.",
    "See No. 4 in the note.",
    "Initials like J. Smith wrote this.",
])
def test_abbreviations_do_not_split(text):
    assert len(split_sentences(text)) == 1, text


# ---------------------------------------------------------------------------
# genuine boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("One. Two. Three. Four.", 4),
    ("Is it 1.05%? Yes it is.", 2),
    ("Stop!", 1),
    ("Really? Yes.", 2),
    ("", 0),
    ("   ", 0),
    ("\n\n", 0),
])
def test_real_boundaries_split(text, expected):
    assert len(split_sentences(text)) == expected, repr(text)


def test_urls_are_not_split():
    """A period inside a URL is not a sentence end.

    The model is told to cite by block number, so a URL should not appear in
    answer text -- but if it does, splitting "groww.in" into "groww" + "in"
    would both corrupt the answer and produce a fake citation token.
    """
    for url in ("groww.in", "www.amfiindia.com", "hdfcfund.com/mutual-funds/factsheets"):
        text = f"See {url} for the factsheet."
        assert len(split_sentences(text)) == 1, url


def test_newlines_are_not_sentence_boundaries():
    """A model that wraps one sentence across two lines has still written one.

    Counting the wrap as a sentence would truncate a correct three-sentence
    answer to two whenever the 8B model happens to line-wrap.
    """
    text = "The expense ratio is 1.03%\nand the base expense ratio is 0.84%"
    assert len(split_sentences(text)) == 1
    assert "1.03%" in split_sentences(text)[0]


def test_index_name_is_not_split():
    """The benchmark's proper name contains "Index", and "1.05%." ends in a digit.

    Together these are the two highest-risk shapes in the real corpus.
    """
    text = "The benchmark is NIFTY 100 Total Return Index. The expense ratio is 1.03%."
    assert len(split_sentences(text)) == 2
    assert split_sentences(text)[0] == "The benchmark is NIFTY 100 Total Return Index."


# ---------------------------------------------------------------------------
# enforce_max_sentences
# ---------------------------------------------------------------------------

def test_under_limit_is_returned_unchanged():
    text = "The expense ratio is 1.03%. Source: [1]"
    out, ok = enforce_max_sentences(text, 3)
    assert ok is True
    assert out == text


def test_over_limit_truncates_on_a_sentence_boundary():
    text = "One fact. Two facts. Three facts. Four facts."
    out, ok = enforce_max_sentences(text, 3)
    assert ok is False
    assert out == "One fact. Two facts. Three facts."


def test_truncation_drops_whole_sentences_not_mid_sentence():
    """A half-sentence plus a dropped citation is worse than a shorter answer."""
    text = "One. Two. Three. This fourth sentence is the one that gets cut off"
    out, ok = enforce_max_sentences(text, 3)
    assert ok is False
    assert out == "One. Two. Three."
    assert not out.endswith("cut")


def test_limit_of_one_keeps_the_first_sentence():
    out, ok = enforce_max_sentences("First. Second. Third.", 1)
    assert ok is False
    assert out == "First."


def test_zero_limit_never_raises():
    out, ok = enforce_max_sentences("Anything at all.", 0)
    assert ok is False
    assert out == ""


def test_decimal_answer_survives_the_limit():
    """The real regression: a one-sentence percentage answer must stay one sentence.

    A naive splitter calls this two sentences, so the limit trips and the answer
    is truncated to "The expense ratio is 1.05" -- a wrong answer shipped as a
    correct one.
    """
    text = "The expense ratio is 1.05% and the base expense ratio is 0.84%. Source: [1]"
    out, ok = enforce_max_sentences(text, 3)
    assert ok is True
    assert out == text
    assert "1.05%" in out and "0.84%" in out
