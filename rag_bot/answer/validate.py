"""Structural validation of a generated answer (architecture 14.2).

Generation is not trusted until these pass. Every function here returns data and
never raises: a validator that throws on malformed model output takes the demo
down, which is worse than the malformation it was checking for.

The two checks that decide whether this module is useful or catastrophic are both
false-positive checks, because a check that fires wrongly destroys a correct
answer silently:

- `split_sentences` must not split "1.05%" into two sentences. A naive splitter
  reports a one-sentence answer as two and truncates it into nonsense.
- `find_return_figures` must not flag "1.05%". The corpus's own answers are
  percentages, so a blanket number ban breaks the product's core capability.
"""
from __future__ import annotations

import re

from rag_bot.answer.prompts import NOT_IN_INDEX
from rag_bot.types import ScoredChunk, Validation

# ---------------------------------------------------------------------------
# sentence splitting
# ---------------------------------------------------------------------------

# Tokens whose trailing period is not a sentence end. The rendered corpus contains
# none of these (checked: no "Rs.", "p.a.", "No.", "approx." in any chunk), so this
# list exists purely for model output, which writes "Rs. 500" and "1.05% p.a."
# unprompted.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "st", "no", "vs", "etc", "approx", "appt",
    "inc", "ltd", "co", "corp", "dept", "est", "fig", "al", "cf", "eg", "ie",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "a/c", "p.a", "p.m", "a.m", "i.e", "e.g", "w.r.t", "asst", "capt", "sgt",
    # Currency. "Rs. 500" is a single sentence, and splitting it makes the exit
    # load answer read as two sentences, which then trips the 3-sentence limit.
    "rs", "inr", "usd", "eur", "gbp", "₹",
})

_CLOSERS = ".!?"
_SENTENCE_END = re.compile(r"[.!?]+")


def _is_boundary(text: str, index: int) -> bool:
    """Is the punctuation at `index` a real sentence boundary?"""
    # 1. Never split on a period between digits: "1.05", "1447.3830", "01-Jan-2013".
    prev_ch = text[index - 1] if index > 0 else ""
    next_ch = text[index + 1] if index + 1 < len(text) else ""
    if prev_ch.isdigit() and next_ch.isdigit():
        return False

    # 2. Not an abbreviation. Checked on the token immediately before the period,
    #    and "Rs.500" is handled because the next char being a digit does not
    #    rescue an abbreviation.
    token = re.search(r"([A-Za-z./]+)$", text[:index])
    if token:
        word = token.group(1).lower().rstrip(".")
        if word in _ABBREVIATIONS:
            return False
        # A single capital letter before a period is an initial ("J. Smith").
        if len(word) == 1 and word.isalpha():
            return False

    # 3. A boundary must be followed by the start of a sentence. A lowercase letter
    #    means we are inside a token -- which is what keeps "groww.in" and
    #    "NIFTY 100 Total Return Index" from being torn apart.
    if next_ch and not next_ch.isspace() and not next_ch.isupper() \
            and not next_ch.isdigit() and next_ch not in "\"'([“‘":
        return False
    if next_ch.isdigit() and prev_ch.isalpha() and not prev_ch.isupper():
        return False
    return True


def split_sentences(text: str) -> list[str]:
    """Decimal- and abbreviation-aware.

    "The ratio is 1.05%." is ONE sentence. "Rs. 500" is one sentence. Never splits
    on a period between digits. Newlines are NOT boundaries: a model that wraps an
    answer across lines has still written one sentence, and counting the wrap as a
    sentence would truncate a correct answer.
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        index = match.start()
        if not _is_boundary(text, index):
            continue
        chunk = text[start:match.end()].strip()
        if chunk:
            sentences.append(chunk)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def enforce_max_sentences(text: str, limit: int) -> tuple[str, bool]:
    """(truncated_text, ok). Truncate on a sentence boundary.

    Truncation drops whole sentences rather than cutting mid-sentence, because a
    half-sentence plus a dropped citation is worse than a shorter answer.
    """
    if limit <= 0:
        return "", False
    sentences = split_sentences(text)
    if not sentences:
        return text.strip(), True
    if len(sentences) <= limit:
        return text.strip(), True
    return " ".join(sentences[:limit]).strip(), False


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------

_CITATION = re.compile(r"source\s*:\s*\[?\s*(\d+)\s*\]?", re.IGNORECASE)


def parse_citation(text: str) -> int | None:
    """Extract the block number from a trailing "Source: [3]". None if absent.

    Both bracketed and bare forms are accepted. The system prompt says
    "Source: <block number>" and architecture 14.2 writes "Source: [n]"; an 8B
    model reliably produces one of the two, and losing the citation over a bracket
    would be self-inflicted.
    """
    if not text:
        return None
    matches = _CITATION.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (TypeError, ValueError):
        return None


def cited_blocks(text: str) -> list[int]:
    """Every block number cited in the text, in order of appearance.

    The model is asked for ONE citation. More than one is a validation failure,
    because the one-source rule (F7) and a two-block answer are in genuine
    conflict: rank 1 is kept and the rest belong in the chunk expander.
    """
    if not text:
        return []
    out: list[int] = []
    for raw in _CITATION.findall(text):
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def strip_citation_marker(text: str) -> str:
    """Remove the model's "Source: [n]" marker.

    The marker is a machine artifact for the validator's benefit: the user-facing
    citation is assembled in code from the cited chunk's METADATA (assemble.py),
    which is what makes it impossible for the model to invent a URL. Leaving the
    raw marker in would print "Source: [1]" immediately above "Source: groww.in -
    HDFC Large Cap Fund", i.e. two citations, one of them meaningless to a reader.
    """
    if not text:
        return ""
    if not _CITATION.search(text):
        return text.strip()
    # There WAS a marker, so an empty result is real information: the model
    # returned a citation and no answer. Returning the input here would ship
    # "Source: [1]" to the user as the answer body.
    return re.sub(r"[ \t]{2,}", " ", _CITATION.sub("", text)).strip()


def check_single_url(
    chunks: list[ScoredChunk], cited_n: int | None
) -> tuple[str | None, bool]:
    """(url, ok). The URL comes from the CITED CHUNK'S METADATA, never from model text.

    If cited_n is None, rank 1 is used. ok is False when more than one distinct
    source_id was cited -- keep rank 1, the rest belong in the chunk expander.

    An out-of-range citation is a model-invented block number, so rank 1 is used
    and ok is False. Returning (None, False) here instead would desynchronise the
    three fields that must agree: `validation.cited_url` would be None while
    `assemble` still printed a rank-1 footer URL, producing a class A answer with
    a URL in its text and `source_url=None`, which is exactly the state the P8
    renderer cannot display.
    """
    if not chunks:
        return (None, False)

    index = 0 if cited_n is None else cited_n - 1
    if index < 0 or index >= len(chunks):
        return (chunks[0].chunk.url, False)
    return (chunks[index].chunk.url, True)


def check_scheme(
    chunks: list[ScoredChunk], cited_n: int | None, asked_scheme_id: str | None
) -> bool:
    """False when the cited chunk's scheme_id differs from the asked scheme.

    True when no scheme was asked for, or none could be resolved: the caller
    disambiguates instead of rejecting, because "which of the five?" is a
    legitimate state, not a mismatch.

    An out-of-range citation is NOT a mismatch. It is the model inventing a block
    number, which 14.2 says to "strip and flag", and conflating the two would
    turn a cosmetic citation error into a refusal. An invalid number falls back to
    rank 1 exactly as `check_single_url` does, so a genuine cross-scheme mixup --
    a valid number pointing at the wrong fund -- is still caught.
    """
    if asked_scheme_id is None or not chunks:
        return True
    index = 0 if cited_n is None else cited_n - 1
    if index < 0 or index >= len(chunks):
        index = 0
    return chunks[index].chunk.scheme_id == asked_scheme_id


# ---------------------------------------------------------------------------
# the figure check (architecture 14.5)
# ---------------------------------------------------------------------------

# The product's own answers. A percentage governed by one of these nouns IS the
# answer, so it must survive no matter what else is in the sentence.
_FEE_NOUNS = (
    "total expense ratio", "base expense ratio", "expense ratio", "ter",
    "exit load", "load", "charges", "charge", "minimum sip", "sip amount",
    "minimum investment", "minimum withdrawal", "minimum lump-sum",
    "minimum lump sum", "riskometer", "lock-in", "lock in", "holding period",
)

# Words that mean a figure here is a return/NAV/performance number.
_RETURN_NOUNS = (
    "returns", "return", "returned", "returning", "cagr", "nav", "performance",
    "yield", "yields", "growth", "gain", "gains", "profit", "percentile",
    "ranking", "rank", "since inception", "absolute return", "annualised",
    "annualized", "money", "earned", "made a profit",
)

# A rank or percentile claim is a performance claim even with no percentage:
# "it ranks in the 4th percentile" quotes a figure in words.
_RANK_CLAIM = re.compile(
    r"\b\d+(?:st|nd|rd|th)\s+(?:percentile|quartile|rank)\b"
    r"|\brank(?:ed|ing)?\s+(?:in|among|amongst|between)\s+the\s+\d+"
    r"|\b\d+\s*(?:st|nd|rd|th)?\s*(?:percentile|quartile)\b",
    re.IGNORECASE,
)

_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?(?:%|percent\b)", re.IGNORECASE)
_CURRENCY = re.compile(r"(?:rs\.?|inr|₹)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)

# How far back to look for the noun that governs a figure. Wide enough for
# "The 1 year return was 12.4%", narrow enough that a governing noun beats a
# keyword 40 characters away.
_GOVERNING_WINDOW = 48

# Every scheme in the corpus is named "... - Direct Growth", and "growth" is a
# return noun. So in a sentence that names a fund, the nearest return noun is
# almost always the one in the fund's own name, at a closer offset than the
# governing noun that actually decides the figure:
#
#     "The minimum SIP amount for the HDFC Large Cap Fund - Direct Growth is
#      Rs 100 per month."
#
# Nearest-noun-wins then reads "Direct Growth" as governing the Rs 100, strips
# the sentence, and -- because nothing is left -- routes the question to class D.
# That is a false refusal of a correct, required answer, and it only showed up
# once the eval set asked about funds whose names end in "Growth" (all of them).
#
# Masked unconditionally, and the trade-off is worth stating plainly. Every plan
# in this corpus is a "Direct Growth" plan, so the phrase appears in nearly every
# answer that names a fund; treating it as a return noun made the check
# unusable. Restricting the mask to obvious name positions was tried and is
# wrong: the model also writes "HDFC Flexi Cap Direct Plan-Growth is Rs 100",
# where the name sits mid-sentence and no name-position pattern can reach it
# without also swallowing the "Direct growth of 12%" it is meant to spare.
#
# The residual gap is narrow and recorded rather than hidden: a return claim
# phrased specifically as "direct growth of X%" would be missed. Nothing in the
# label set is phrased that way, and the other return nouns -- return, cagr,
# nav, performance, yield, rank, percentile, since inception -- are untouched,
# so the class-D rows are unaffected.
#
# The separator between "plan" and "growth" is whatever the model typed, so any
# dash variant is accepted. A version matching only literal spaces missed
# "Direct Plan‑Growth" (U+2011) and the false refusal came straight back.
_SCHEME_NAME_SUFFIX = re.compile(
    r"\bdirect\s*(?:plan\s*[\-\u2010-\u2015\u2212]?\s*)?growth\b"
    r"|\bdirect\s+growth\s+option\b",
    re.IGNORECASE,
)


def _governing_class(before: str, after: str) -> str:
    """'fee', 'return' or 'unknown' for the figure governed by the nearest noun.

    NEAREST NOUN WINS, not "any keyword in the sentence". That distinction is not
    a refinement, it is a correctness requirement forced by the real corpus:

        "The benchmark is NIFTY 100 Total Return Index and the expense ratio is
         1.03%."

    The word "Return" is inside the benchmark index's proper name. A
    sentence-scoped check strips the 1.03% -- a REQUIRED class-A answer -- because
    a performance word appears in the same sentence. Requiring the nearest
    preceding noun to be a return noun keeps the 1.03% and still catches "The 1
    year return was 12.4%", where the nearest noun is "return".
    """
    fee_at = _nearest(before, _FEE_NOUNS)
    return_at = _nearest(before, _RETURN_NOUNS)
    if fee_at is None and return_at is None:
        # Nothing before it: look just after, for "12.4% annual return".
        if _nearest(after, _RETURN_NOUNS) is not None:
            return "return"
        return "unknown"
    if return_at is None:
        return "fee"
    if fee_at is None:
        return "return"
    # The NEARER noun governs: _nearest returns the closest match, so the larger
    # index wins. This is what keeps "exit load is 1% and the 1 year return is
    # 12.4%" honest in both directions -- "exit load" governs the 1%, and "return"
    # governs the 12.4%, even though both nouns sit in the one sentence.
    return "fee" if fee_at >= return_at else "return"


def _nearest(before: str, nouns: tuple[str, ...]) -> int | None:
    """Index of the closest occurrence of any noun in `before`, or None."""
    best: int | None = None
    for noun in nouns:
        match = None
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(noun)}(?![a-z0-9])",
                                 before, re.IGNORECASE):
            pass  # keep the last match: the closest one to the figure
        if match is not None:
            if best is None or match.start() > best:
                best = match.start()
    return best


def _scan_sentence(sentence: str) -> list[str]:
    """Every return/NAV figure in ONE sentence. The single definition of the rule.

    `find_return_figures` and `validate`'s sentence filter both call this, rather
    than each re-implementing "is this figure a return". Two copies of that
    decision is the stale-copy bug P6 hit with the class-B message: they drift,
    and the drift is invisible because both copies look right in isolation.
    """
    found: list[str] = []
    # The fund's own name cannot make a figure a return figure -- see
    # _SCHEME_NAME_SUFFIX for what this cost before it was masked.
    sentence = _SCHEME_NAME_SUFFIX.sub("direct plan", sentence)
    claim = _RANK_CLAIM.search(sentence)
    if claim:
        found.append(claim.group())

    for pattern in (_PERCENT, _CURRENCY):
        for match in pattern.finditer(sentence):
            before = sentence[max(0, match.start() - _GOVERNING_WINDOW):match.start()]
            after = sentence[match.end():match.end() + _GOVERNING_WINDOW]
            if _governing_class(before, after) == "return":
                found.append(match.group())
    return found


def find_return_figures(text: str) -> list[str]:
    """Return every performance/NAV figure found in the text.

    A percentage is flagged ONLY when a return/NAV noun governs it.
    "1.05% expense ratio" survives; "12.4% 1 year return" does not. Currency
    amounts next to return nouns are flagged too ("NAV was Rs 12.34"), as are
    rank and percentile claims, which quote a figure without a percentage.

    Sentence-scoped rather than document-scoped, so one stray "return" elsewhere
    cannot strip an unrelated required answer.
    """
    if not text:
        return []
    found: list[str] = []
    for sentence in split_sentences(text):
        found.extend(_scan_sentence(sentence))
    return found


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def validate(
    raw: str,
    chunks: list[ScoredChunk],
    *,
    limit: int,
    asked_scheme_id: str | None,
) -> tuple[str, Validation, int | None]:
    """Run every check. Returns (final_text, Validation, cited_block_number).

    The documented order is sentence limit -> citation parse -> single URL ->
    scheme match -> figure scan -> NOT_IN_INDEX. Two deliberate deviations, both
    forced by the citation marker's position:

    1. NOT_IN_INDEX is tested first. Every check below is meaningless for a
       sentinel: there is no citation to parse and no sentence to truncate.

    2. The citation is parsed and stripped from the RAW text BEFORE the sentence
       limit is enforced. "Source: [n]" is the model's LAST sentence, so
       truncating first discards a perfectly valid citation and silently falls
       back to rank 1 -- wrong whenever the model cited rank 2 or 3. The
       observable order of every check is unchanged; only the marker handling
       moves, so the marker also never counts toward the 3-sentence budget and
       never survives as a stray sentence when the figure scan empties the body.
    """
    validation = Validation()
    text = (raw or "").strip()

    if text.strip().upper().startswith(NOT_IN_INDEX):
        validation.notes.append("model reported NOT_IN_INDEX")
        validation.unverified = True
        return ("", validation, None)

    # 1. citation, parsed from the raw text and removed from the body
    cited_n = parse_citation(text)
    all_cited = cited_blocks(text)

    if len(set(all_cited)) > 1:
        # 14.2: "Keep rank-1; move the rest to the chunk expander, not the
        # answer." The model lays its claims out as "<claim 1> Source: [1]
        # <claim 2> Source: [2]", so keeping rank 1 means cutting at the FIRST
        # marker. Stripping the markers alone would leave "<claim 1> and
        # <claim 2>" in the body: a two-source answer that looks like one.
        markers = list(_CITATION.finditer(text))
        if len(markers) > 1:
            text = text[:markers[0].start()]
            validation.notes.append(
                f"model cited {len(set(all_cited))} blocks; kept rank 1 and dropped "
                "the rest from the answer"
            )

    text = strip_citation_marker(text)

    # 2. sentence limit, applied to the body without the marker
    if not text.strip():
        # The model emitted the marker and nothing else. There is no answer to
        # validate, so report it rather than shipping a body of "Source: [1]".
        validation.notes.append("model returned no answer body")
        validation.unverified = True
        return ("", validation, cited_n)

    text, sentences_ok = enforce_max_sentences(text, limit)
    validation.sentences_ok = sentences_ok
    if not sentences_ok:
        validation.notes.append(f"truncated to {limit} sentences")

    if cited_n is None:
        validation.unverified = True
        validation.notes.append("no citation found; using rank 1")

    # 3. single URL -- from METADATA
    url, single_url = check_single_url(chunks, cited_n)
    if len(set(all_cited)) > 1:
        single_url = False  # the extra blocks went to the chunk expander, not here
    if cited_n is not None and not (1 <= cited_n <= len(chunks)):
        single_url = False
        validation.notes.append(f"cited block {cited_n} is not in the context")
    validation.single_url = single_url
    validation.cited_url = url

    # 4. scheme match
    validation.scheme_match = check_scheme(chunks, cited_n, asked_scheme_id)
    if not validation.scheme_match:
        validation.notes.append("scheme_mismatch: cited chunk is a different scheme")

    # 5. figure scan -- strip offending sentences, keep the rest
    figures = find_return_figures(text)
    if figures:
        kept = [s for s in split_sentences(text) if not _scan_sentence(s)]
        validation.no_figures = False
        validation.notes.append(f"removed return figures: {figures}")
        if kept:
            text = " ".join(kept)
        else:
            # Nothing survived: the whole answer was a return claim. Signal class D
            # rather than shipping an empty answer.
            validation.notes.append("every sentence carried a return figure")
            text = ""

    return (strip_citation_marker(text), validation, cited_n)
