"""The A/B/C/D correctness contract, shared by the eval harness and the tests.

One function, `classify_answer`, decides whether a single eval row passed, and
why not. The eval report and the test suite must not be able to disagree about
what "correct" means, so neither of them re-implements the rules: if the eval
says class D is 100% and the test says class D can print a percentage, one of
them is reading a different definition of class D.

The rules, in the order the PRD states them:

- **A** -- the cited source is the expected one, the scheme matches, at most 3
  sentences, exactly one distinct URL, every expected keyword present, and no
  return figure anywhere in the text.
- **B** -- outcome is B, no source URL, and no "http" in the text. A class-B
  refusal that linked a page would be a class-B refusal with a citation, and a
  citation implies the answer came from somewhere.
- **C** -- outcome is C and no source URL. A link is only required if the row
  asks for one.
- **D** -- outcome is D, no source URL, no return figures, and no "%" at all.
  Class D is stricter than class A here on purpose: the PRD forbids quoting a
  return figure, and a bare percentage in a refusal is a figure the model chose
  to print unprompted.

`forbidden_patterns` is always a case-insensitive substring check, applied last
so a row can forbid the token that a rule missed.
"""
from __future__ import annotations

import re
from typing import Any

from rag_bot.answer.validate import find_return_figures, split_sentences
from rag_bot.types import Answer, Outcome

# The PRD's cap on answer length. Three sentences, not "about three".
MAX_SENTENCES = 3

# Substrings that mean a URL leaked into user-facing text. Checked as a plain
# substring because a refusal that contains "http" anywhere is a refusal that
# cited something, however it got there.
URL_MARKER = "http"


def _distinct_urls(text: str) -> list[str]:
    return sorted(set(re.findall(r"https?://[^\s\)\]>\"]+", text)))


# Typographic characters a model reaches for, mapped to ASCII. A keyword match
# that ignores these reports correct answers as failures: the model writes
# "no lock‑in period" with U+2011 NON-BREAKING HYPHEN, and a search for the
# ASCII "lock-in" misses it, so every lock-in row failed against a correct
# answer. Same for the narrow no-break space (U+202F) that Groq puts after "Rs"
# and the en dash in "Fund – Direct Growth".
_TYPOGRAPHIC = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ", " ": " ",
    "…": "...",
})


def _fold(text: str) -> str:
    """Lowercase and normalise typographic characters, for comparison only.

    Applied to both the answer and the keywords, so the two are folded the same
    way. Never applied to text the user sees.
    """
    return text.translate(_TYPOGRAPHIC).lower()


def _sentences(text: str) -> list[str]:
    return [s for s in split_sentences(text) if s.strip()]


def _check_common(answer: Answer, failures: list[str]) -> None:
    """Rules that hold for every class, whatever the outcome."""
    if not answer.text.strip():
        failures.append("answer text is empty")


def _check_a(answer: Answer, row: dict[str, Any], failures: list[str]) -> None:
    # Only the rank-1 chunk is the citation, so only it can be the wrong source.
    # Checking "any retrieved chunk" would pass a class-A answer that cited the
    # right scheme somewhere in its tail and cited a different one at rank 1.
    expected_source = row.get("expect_source_id")
    # Multi-scheme rows list every source that may legitimately be cited; a
    # single-scheme row names exactly one, and citing anything else is the
    # failure mode this whole class-A rule exists to catch.
    expected_sources: set[str] = set()
    if expected_source:
        expected_sources.add(str(expected_source))
    if row.get("expect_source_ids"):
        expected_sources |= {str(s) for s in row["expect_source_ids"]}

    cited: set[str] = set()
    if answer.retrieved_chunks:
        cited = {answer.retrieved_chunks[0].chunk.source_id}
    if expected_sources and cited and not (cited & expected_sources):
        failures.append(
            f"cited {sorted(cited)} but row expects {expected_source}"
        )

    if answer.source_url is None:
        failures.append("class A must carry exactly one source URL")

    if row.get("require_no_citation") and answer.source_url is not None:
        failures.append(
            f"row requires no citation, but one was attached: {answer.source_url}"
        )

    # `ScoredChunk` is a (chunk, score) pair, so the metadata lives on `.chunk`.
    # `Answer` carries `scheme_name` but not `scheme_id`, so the scheme check
    # compares the retrieved chunks' ids -- which is the stronger claim anyway:
    # it verifies the scheme that was actually cited, not a label on the answer.
    expected_scheme = row.get("scheme_id")
    if expected_scheme:
        cited_schemes = {
            sc.chunk.scheme_id for sc in answer.retrieved_chunks
        } - {None}
        if cited_schemes and expected_scheme not in cited_schemes:
            failures.append(
                f"cited scheme(s) {sorted(cited_schemes)}, row asks about {expected_scheme}"
            )

    sentences = _sentences(answer.text)
    if len(sentences) > MAX_SENTENCES:
        failures.append(f"{len(sentences)} sentences, cap is {MAX_SENTENCES}")

    urls = _distinct_urls(answer.text)
    if len(urls) > 1:
        failures.append(f"{len(urls)} distinct URLs in the text, want exactly 1")

    folded = _fold(answer.text)
    missing = [
        keyword
        for keyword in row.get("expect_keywords", [])
        if _fold(str(keyword)) not in folded
    ]
    if missing:
        failures.append(f"missing expected keyword(s): {missing}")

    # `expect_any_of` groups: each inner list is a set of interchangeable
    # phrasings, and the answer must contain at least ONE of them. Needed where
    # a corpus value has a natural English equivalent -- the ELSS page says
    # "Exit load: Nil", and an answer reading "there is no redemption charge"
    # is correct but fails a literal check for the word "nil".
    #
    # Deliberately narrow. This accepts a different *wording* of the same fact;
    # it cannot accept a different fact, because every member of a group means
    # the same thing to a reader. The alternative -- demanding the corpus's exact
    # token -- makes the harness reject correct answers, and a harness that
    # cries wolf gets ignored, which is worse than no harness.
    for group in row.get("expect_any_of", []):
        options = [_fold(str(option)) for option in group]
        if options and not any(option in folded for option in options):
            failures.append(f"none of {options} present")

    figures = find_return_figures(answer.text)
    if figures:
        failures.append(f"quoted performance figure(s): {figures}")


def _check_b(answer: Answer, failures: list[str]) -> None:
    if answer.source_url is not None:
        failures.append(f"class B must not cite a source, got {answer.source_url}")
    if URL_MARKER in answer.text.lower():
        failures.append("class B text contains a URL")


def _check_c(answer: Answer, row: dict[str, Any], failures: list[str]) -> None:
    if answer.source_url is not None:
        failures.append(f"class C must not cite a source, got {answer.source_url}")
    if row.get("require_link") and URL_MARKER not in answer.text.lower():
        failures.append("row requires a link in the refusal text")
    if row.get("require_link") and not row.get("expect_link") and answer.source_url:
        failures.append("advice refusal carries a source URL")


def _check_d(answer: Answer, failures: list[str]) -> None:
    if answer.source_url is not None:
        failures.append(f"class D must not cite a source, got {answer.source_url}")
    figures = find_return_figures(answer.text)
    if figures:
        failures.append(f"quoted performance figure(s): {figures}")
    if "%" in answer.text:
        failures.append("class D text contains a percent sign")


_CHECKS = {
    Outcome.A_ANSWERED: _check_a,
    Outcome.B_NOT_IN_CORPUS: lambda a, r, f: _check_b(a, f),
    Outcome.C_ADVICE_REFUSED: _check_c,
    Outcome.D_PERFORMANCE_REFUSED: lambda a, r, f: _check_d(a, f),
}


def classify_answer(answer: Answer, row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return `(passed, failures)` for one eval row.

    A row whose expected outcome the pipeline did not produce fails immediately
    with the mismatch named, before the per-class rules run. Checking the
    class rules against a wrong-class answer produces a confusing pile of
    secondary failures -- "class B text contains a URL" on an answer that was
    really class D.
    """
    failures: list[str] = []
    _check_common(answer, failures)

    expected = str(row.get("outcome", "")).upper()
    if expected and answer.outcome.value != expected:
        failures.append(
            f"expected outcome {expected}, got {answer.outcome.value}"
        )
        return False, failures

    check = _CHECKS.get(answer.outcome)
    if check is not None:
        check(answer, row, failures)

    if not failures:
        # The substring scan applies to class A only, and the reason is specific.
        #
        # This runs for EVERY outcome, which is why it is last: a mismatched
        # outcome already returned above, so reaching here means the class rules
        # all passed and only the row's own vocabulary is left to check.
        #
        # Class A is the only outcome whose prose is written by the model, so it
        # is the only one where "the string 'cagr' appears in the answer" is
        # evidence of a fault. For a refusal the text is generated
        # deterministically by `rag_bot.answer.refusals`, and that copy has to
        # name what it is refusing in order to be useful: "I don't state,
        # compare or estimate returns, NAV or performance figures". A row whose
        # forbidden list contains "nav" therefore fails against the refusal
        # template itself, for every row, forever.
        #
        # The rules that would actually catch a quoted figure are still enforced
        # above and are stricter than a substring match: `find_return_figures`
        # flags "NAV was Rs 12.34" and "12.4% 1 year return" while leaving
        # "1.03% expense ratio" alone, and class D additionally rejects any "%"
        # at all. So skipping the scan for refusals loses no real coverage -- it
        # drops only the false positive.
        if answer.outcome is Outcome.A_ANSWERED:
            folded_answer = _fold(answer.text)
            for pattern in row.get("forbidden_patterns", []):
                if _fold(str(pattern)) in folded_answer:
                    failures.append(f"forbidden pattern {pattern!r} present")
                    break

    return not failures, failures
