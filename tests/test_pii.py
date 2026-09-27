"""PII scrubber tests.

The two fixture tables are the real deliverable (implementation.md P5): a scrubber
that eats "1.05%" has broken this product more thoroughly than one that leaks an
email, because the breakage is silent. So the must-not-redact direction is
asserted with the same weight as the must-redact direction, and one test runs the
scrubber over the actual corpus text.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag_bot.safety.pii import (
    ScrubResult,
    scrub,
    scrub_for_log,
)

FIXTURES = Path(__file__).parent / "fixtures"
MUST_REDACT = FIXTURES / "pii_must_redact.txt"
MUST_NOT_REDACT = FIXTURES / "pii_must_not_redact.txt"


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def must_redact_lines() -> list[str]:
    return _lines(MUST_REDACT)


@pytest.fixture(scope="module")
def must_not_redact_lines() -> list[str]:
    return _lines(MUST_NOT_REDACT)


# ---------------------------------------------------------------------------
# the fixture tables
# ---------------------------------------------------------------------------

def test_fixture_tables_are_not_empty(must_redact_lines, must_not_redact_lines):
    assert len(must_redact_lines) >= 6
    assert len(must_not_redact_lines) >= 6


def test_every_must_redact_line_is_redacted(must_redact_lines):
    """Every line of the must-redact table must fire at least one rule."""
    missed = [t for t in must_redact_lines if scrub(t).clean]
    assert not missed, f"PII leaked: {missed}"


def test_every_must_not_redact_line_survives(must_not_redact_lines):
    """Every line of the must-not-redact table must come back byte-identical."""
    damaged = [(t, scrub(t)) for t in must_not_redact_lines if not scrub(t).clean]
    assert not damaged, f"scrubber ate the product's answers: {damaged}"


def test_must_not_redact_is_checked_in_both_directions(must_not_redact_lines):
    """Guard the guard: the table is only meaningful if it would catch a sweep.

    A blanket digit regex -- the failure mode this phase exists to prevent -- has
    to fail this table, or the table is not testing anything.
    """
    import re

    blanket = re.compile(r"[0-9]{4,}")
    survivors = [t for t in must_not_redact_lines if not blanket.search(t)]
    assert survivors, "must-not-redact table has no 4+ digit runs to protect"


# ---------------------------------------------------------------------------
# each rule in isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, rule",
    [
        ("My PAN is ABCDE1234F, what is the expense ratio?", "pan"),
        ("Aadhaar number 4829 1396 2510 attached", "aadhaar"),
        ("Aadhaar 482913625110 please", "aadhaar"),
        ("Adhaar: XXXX XXXX 1234", "aadhaar"),
        ("my email is ramesh.iyer@example.com", "email"),
        ("call me on 9876543210 about the exit load", "phone"),
        ("call +91-9876543210 today", "phone"),
        ("a/c no 1234567890", "account"),
        ("my pin code is 4321", "otp"),
    ],
)
def test_single_rule_fires_in_isolation(text, rule):
    result = scrub(text)
    assert result.rules_fired == [rule], result
    assert not result.clean
    assert "redacted]" in result.text


def test_pan_keeps_the_question_intact():
    result = scrub("My PAN is ABCDE1234F, what is the expense ratio?")
    assert result.text == "My PAN is [PAN redacted], what is the expense ratio?"


def test_email_redaction_keeps_surrounding_text():
    result = scrub("my email is ramesh.iyer@example.com")
    assert result.text == "my email is [email redacted]"


def test_two_rules_fire_on_one_line():
    result = scrub("the OTP is 654321 and the a/c no is 4455667788")
    assert set(result.rules_fired) == {"otp", "account"}
    assert "[OTP redacted]" in result.text
    assert "[account redacted]" in result.text


# ---------------------------------------------------------------------------
# keyword proximity: the difference between PII and an answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["minimum SIP 482913", "I invested 482913 last month",
                                  "the 482913 SIP date is fine"])
def test_otp_digits_without_a_keyword_survive(text):
    """Same digits, no keyword nearby: not PII. This is the OTP false positive."""
    assert scrub(text).clean, text


@pytest.mark.parametrize("text", ["otp 482913", "the OTP is 482913", "verification code 482913",
                                  "482913 is the otp", "my pin: 482913"])
def test_otp_keyword_before_or_after_redacts(text):
    assert "otp" in scrub(text).rules_fired, text


@pytest.mark.parametrize("text", ["minimum SIP 12345678", "NAV moved to 12345678",
                                  "expense ratio 1.05 for 12345678 units"])
def test_long_digits_without_a_keyword_survive(text):
    assert scrub(text).clean, text


@pytest.mark.parametrize("text", ["a/c no 12345678", "account number 12345678",
                                  "12345678 is my folio number", "acct 12345678"])
def test_account_keyword_before_or_after_redacts(text):
    assert "account" in scrub(text).rules_fired, text


def test_keyword_beyond_the_window_does_not_count():
    """40 chars is the window; a keyword further away is not a governing keyword."""
    far = "otp" + " " * 45 + "482913"
    assert scrub(far).clean, far
    near = "otp" + " " * 30 + "482913"
    assert "otp" in scrub(near).rules_fired, near


def test_account_keyword_does_not_trigger_the_otp_rule():
    """'a/c' governs the account rule, not the OTP rule: a 6-digit amount is not an OTP."""
    assert scrub("a/c transfer limit of 100000 rupees").clean
    assert "otp" not in scrub("a/c no 1234").rules_fired


def test_eight_to_sixteen_digit_run_is_an_account_not_an_otp():
    result = scrub("folio number is 5566778899")
    assert result.rules_fired == ["account"]


def test_seven_digit_run_matches_neither_rule():
    assert scrub("a/c no 1234567").clean


# ---------------------------------------------------------------------------
# the hard rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The expense ratio is 1.05%.",
    "Exit load is 1% for less than 7 days.",
    "total expense ratio 1.41%, minimum investment Rs 100",
    "Minimum SIP is Rs 500.",
    "Minimum SIP is ₹500.",
])
def test_percentages_and_currency_are_never_redacted(text):
    assert scrub(text).text == text


def test_pan_is_caught_whatever_the_case():
    """PANs are upper-case by format, but users retype them in lower case."""
    assert "pan" in scrub("pan abcde1234f").rules_fired


@pytest.mark.parametrize("text", ["aadhaar 4829-1396-2510", "aadhaar: 4829/1396/2510",
                                  "Aadhaar number 4829  1396  2510"])
def test_aadhaar_separators(text):
    """'-', '/' and repeated spaces are how people actually write an Aadhaar."""
    assert "aadhaar" in scrub(text).rules_fired, text


def test_amount_survives_a_pii_keyword_in_the_window():
    """The reason protected spans exist, not a hypothetical.

    'my pin code' puts an OTP keyword inside 40 chars of 'Rs 5000'. Without the
    protection the 5000 -- a real product answer -- would be redacted.
    """
    result = scrub("my pin code, minimum SIP is Rs 5000")
    assert "Rs 5000" in result.text


def test_account_number_still_redacted_next_to_a_protected_amount():
    """Partial redaction: the folio goes, the SIP amount stays."""
    result = scrub("a/c no 1234567890, minimum SIP is Rs 500")
    assert "1234567890" not in result.text
    assert "Rs 500" in result.text
    assert "account" in result.rules_fired


def test_no_bare_digit_sweep():
    """Any 4+ digit run with no PII keyword must survive."""
    for text in ["1234", "12345678", "987654321", "555 444 333"]:
        assert scrub(text).clean, text


def test_bare_ten_digit_run_is_redacted_by_length_not_keyword():
    """A [6-9]-leading 10-digit run is phone-shaped on its own."""
    result = scrub("9876543210")
    assert result.rules_fired == ["phone"]


def test_ten_digit_run_not_starting_6_to_9_is_not_a_phone():
    """Tightening: '1234567890' is far more often a folio number than a phone."""
    assert scrub("1234567890").clean


def test_real_corpus_text_is_untouched():
    """The strongest false-positive test available: run over the real corpus.

    The corpus is the set of strings the assistant must be able to echo, so if
    scrubbing it is a no-op the product is intact.
    """
    from rag_bot.config import load
    from rag_bot.ingest.chunker import chunk_all
    from rag_bot.ingest.loaders import load_from_manifest

    cfg = load()
    docs = load_from_manifest(f"{cfg.corpus_dir}/manifest.csv")
    chunks = chunk_all(
        docs, target_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap, tokenizer=lambda t: len(t.split()),
    )
    assert chunks, "corpus is empty; this test would be vacuous"

    dirty = [c.text for c in chunks if not scrub(c.text).clean]
    assert not dirty, f"scrubber alters the corpus: {dirty[:3]}"

    for line in scrub("\n".join(c.text for c in chunks)).text.splitlines():
        assert line in "\n".join(c.text for c in chunks)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------

def test_clean_result_on_ordinary_text():
    result = scrub("What is the exit load for HDFC Large Cap Fund?")
    assert result.clean and result.rules_fired == []
    assert result.text == "What is the exit load for HDFC Large Cap Fund?"


def test_result_is_frozen():
    result = scrub("hello")
    with pytest.raises(Exception):
        result.text = "nope"  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, 42, b"bytes", [], {}, object()])
def test_never_raises_on_any_input(value):
    result = scrub(value)  # type: ignore[arg-type]
    assert isinstance(result, ScrubResult)
    assert result.clean


def test_empty_and_whitespace():
    assert scrub("").clean
    assert scrub("   ").clean
    assert scrub("   ").text == "   "


def test_redacted_text_carries_no_digits_back():
    result = scrub("My PAN is ABCDE1234F, aadhaar 4829 1396 2510, a/c no 1234567890, otp 482913")
    for digits in ("ABCDE1234F", "4829", "1396", "2510", "1234567890", "482913"):
        assert digits not in result.text, digits
    assert set(result.rules_fired) == {"pan", "aadhaar", "account", "otp"}


def test_scrub_for_log_returns_plain_string():
    out = scrub_for_log("My PAN is ABCDE1234F, expense ratio 1.05%")
    assert isinstance(out, str)
    assert out == "My PAN is [PAN redacted], expense ratio 1.05%"


def test_scrub_for_log_agrees_with_scrub():
    text = "a/c no 1234567890 and otp 482913"
    assert scrub_for_log(text) == scrub(text).text


def test_repeated_scrub_is_stable():
    """Idempotent: re-scrubbing a redacted string finds nothing new."""
    text = "My PAN is ABCDE1234F, a/c no 1234567890, otp 482913"
    once = scrub(text)
    twice = scrub(once.text)
    assert twice.text == once.text
    assert not twice.clean or twice.rules_fired == []


def test_country_coded_phone_is_labelled_phone_not_aadhaar():
    """+91 9876543210 is 12 digits once the country code is counted.

    Letting the aadhaar rule match first still removed the digits, so nothing
    leaked -- but rules_fired is what the P8 trace log records, and an "[Aadhaar
    redacted]" tag on a phone number is a wrong answer, not a harmless label.
    """
    result = scrub("send the OTP to +919876543210")
    assert result.rules_fired == ["phone"]
    assert "9876543210" not in result.text


def test_twelve_digit_aadhaar_is_still_labelled_aadhaar():
    """Reordering must not cost the 12-digit Aadhaar its own label."""
    result = scrub("Aadhaar 482913625110 please")
    assert result.rules_fired == ["aadhaar"]


def test_four_four_four_grouped_aadhaar_is_still_aadhaar():
    assert scrub("Aadhaar number 4829 1396 2510 attached").rules_fired == ["aadhaar"]
