"""Allowlist extraction from a snapshot's embedded scheme data.

This module is the structural guarantee behind the PRD's "no performance
figures" constraint. Rather than scrubbing performance figures *out* of scraped
text, it selects a fixed, small set of scheme facts and emits nothing else, so
returns, NAV, and rankings cannot reach the corpus in the first place.

Why not a denylist: a denylist regex was run over all 97 data fields of the
HDFC ELSS page to test the original design. It failed in both directions.

  false positive -- it would have deleted `benchmark_name =
  "NIFTY 500 Total Return Index"` because of the word "Return", and
  `stamp_duty = "0.005% (from July 1st, 2020)"` because of the "%". Both are
  facts, and benchmark is an in-scope topic.

  false negative -- it would have passed `nav = 1447.383` (a bare float with no
  keyword and no %), and `return_stats = {"return1d": 0.38, "return1w": -1.22,
  ...}` (a `\\breturn\\b` cannot match `return1d`, because there is no word
  boundary before a digit, and the values carry no %).

Under allowlist extraction both failure modes are impossible, and
`ALLOWED_FIELDS & FORBIDDEN_FIELDS == set()` is an assertable invariant rather
than an aspiration. See docs/data-findings.md section 4.

FOLDED PHASE. The original implementation guide had a separate P2 "sanitize
snapshots" phase, a denylist scrubber. It was dissolved into this phase and into
the builder in the next phase: `FORBIDDEN_FIELDS` and `FORBIDDEN_PATTERNS` are
defined here, and `guard()` here is enforced by the builder. Definition precedes
enforcement, so the forbidden set cannot be quietly widened by whoever writes the
extractor.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# --------------------------------------------------------------------------
# The allowlist
# --------------------------------------------------------------------------
# A fixed TUPLE, not a set or dict: the rendered order is the document order, so
# the extracted text is byte-stable across runs and the content_hash is
# reproducible.
#
# Every entry was verified present and sane across all five snapshots.
ALLOWED_FIELDS: tuple[str, ...] = (
    # --- identity ---
    "fund_name",
    "scheme_name",
    "category",
    "sub_category",
    "amc",
    "plan_type",
    "scheme_type",
    "isin",
    "launch_date",
    "registrar_agent",
    # --- expense_ratio ---
    "expense_ratio",
    "base_expense_ratio",
    # --- exit_load ---
    "exit_load",
    # --- min_sip ---
    "min_sip_investment",
    "min_investment_amount",
    "min_withdrawal",
    # --- lock_in ---
    "lock_in",
    # --- riskometer ---
    "nfo_risk",
    # --- benchmark ---
    "benchmark_name",
    # --- official publisher pointer ---
    # NOT a factsheet: no factsheet PDF exists on any of the five pages
    # (brochure_link is null, zero PDF links). sid_url is the scheme's official
    # AMC domain, included so a class-D refusal can point at an official
    # publisher rather than a broker page. It is labelled as the AMC website so
    # it is never mistaken for a factsheet.
    "sid_url",
    # --- mandate ---
    "description",
)

# --------------------------------------------------------------------------
# The forbidden set
# --------------------------------------------------------------------------
# Never selected, so never rendered, so never chunked, so never indexed. This is
# the retired denylist's one genuine contribution, kept as a positive assertion
# rather than a set of regexes to hunt for.
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        # performance
        "nav",
        "nav_date",
        "return_stats",
        "simple_return",
        "sip_return",
        "stats",
        "peerComparison",
        "historic_fund_expense",
        "historic_exit_loads",
        # investment advice
        "analysis",
        # rankings
        "category_info",
        "groww_rating",
        "crisil_rating",
        "portfolio_turnover",
        # portfolio and size
        "holdings",
        "aum",
        # SEO / meta
        "meta_desc",
        "meta_title",
        "meta_robots",
        # contact-bearing blocks (the only corpus-side PII carriers)
        "amc_info",
        "rta_details",
        "stp_details",
        "swp_details",
        # misc out-of-scope
        "actions",
        "fund_news",
        "video_url",
        "logo_url",
    }
)

# --------------------------------------------------------------------------
# Forbidden patterns
# --------------------------------------------------------------------------
# Text-level assertions run over the RENDERED text by guard(). Deliberately NOT
# included, because each of these produced a false positive on a real in-scope
# fact during design (docs/data-findings.md section 4):
#
#   bare `return`  -> "NIFTY 500 Total Return Index" is the benchmark's name
#   bare `%`        -> "0.005% (from July 1st, 2020)" is a fee
#   `growth`        -> "HDFC Large Cap Fund Direct Growth" is the plan name
#
# A % is legitimate here: expense ratios and exit loads are percentages, and
# they are the product. Only a percentage adjacent to return language is a leak.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    # camelCase return keys, e.g. return1d / return3y. Would appear only if raw
    # JSON leaked into the text. The word-boundary trap: \breturn\b does NOT
    # match these, which is exactly how the denylist leaked them.
    (r"return\d+[dwmqy]", "raw return-stat key"),
    (r"\bcagr\b", "CAGR"),
    (r"since\s+inception", "since-inception return"),
    (r"\bstat_\d+[ymd]\b", "raw stat_1y style return key"),
    (r"FUND_RETURN", "FUND_RETURN type tag"),
    (r"\bnav\b", "NAV"),
    (r"\brank(?:ed|ing)?\b", "ranking"),
    (r"percentile", "percentile"),
    (r"growth\s+of\s+\d", "growth-of figure"),
    (r"\bp\.\s?a\.\b", "per-annum performance qualifier"),
    # PII
    (r"[\w.+-]+@[\w-]+\.\w{2,}", "email address"),
    (r"(?:\+\d[\d\s().-]{8,}\d)", "phone number"),
    (r"\b\d{5}\s?\d{5}\b", "phone number (10-digit)"),
    (r"enter\s+your\s+pan", "PAN capture prompt"),
)

_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in FORBIDDEN_PATTERNS
)

# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
# Fixed section order. Headings are `## ` so the chunker can cut on them, and
# rows are `Label: value` so a table flattened this way stays answerable.
# (architecture §9.2: a table shredded into fragments is unanswerable.)
_LABEL: dict[str, str] = {
    "fund_name": "Fund name",
    "scheme_name": "Scheme name",
    "category": "Category",
    "sub_category": "Sub-category",
    "amc": "Asset management company",
    "plan_type": "Plan type",
    "scheme_type": "Plan option",
    "isin": "ISIN",
    "launch_date": "Launch date",
    "registrar_agent": "Registrar and transfer agent",
    "expense_ratio": "Expense ratio",
    "base_expense_ratio": "Base expense ratio",
    "exit_load": "Exit load",
    "min_sip_investment": "Minimum SIP amount",
    "min_investment_amount": "Minimum lump-sum investment",
    "min_withdrawal": "Minimum withdrawal amount",
    "lock_in": "Lock-in period",
    "nfo_risk": "Riskometer level",
    "benchmark_name": "Benchmark",
    "sid_url": "Official AMC website",
    "description": "Objective",
}

# Which `## ` section each field belongs under. Anything unlisted goes to
# "Other details" rather than being dropped.
#
# Fees are split ACROSS topics, not grouped into one "Fees and charges" section.
# Topic is a first-class retrieval and filter dimension (the Chroma store
# where-filters on it, and the eval harness reports a per-topic hit-rate), so
# each in-scope topic gets its own section. Grouping expense ratio, exit load,
# and minimum SIP under one heading put all three rows in one chunk, which
# first-match-wins topic assignment then labelled "expense_ratio" -- leaving
# exit_load and min_sip with no labelled chunk at all.
_SECTION: dict[str, str] = {
    **{k: "Scheme identity" for k in (
        "fund_name", "scheme_name", "category", "sub_category", "amc",
        "plan_type", "scheme_type", "isin", "launch_date", "registrar_agent",
    )},
    "expense_ratio": "Expense ratio",
    "base_expense_ratio": "Expense ratio",
    "exit_load": "Exit load",
    **{k: "Minimum investments" for k in (
        "min_sip_investment", "min_investment_amount", "min_withdrawal",
    )},
    "lock_in": "Lock-in period",
    "nfo_risk": "Riskometer",
    "benchmark_name": "Benchmark",
    "sid_url": "Official sources",
    "description": "Objective",
}

_SECTION_ORDER: tuple[str, ...] = (
    "Scheme identity",
    "Expense ratio",
    "Exit load",
    "Minimum investments",
    "Lock-in period",
    "Riskometer",
    "Benchmark",
    "Official sources",
    "Objective",
    "Other details",
)

# Fields whose value is prose rather than a single fact; emitted as a paragraph
# under their label instead of a `Label: value` row.
_PROSE_FIELDS = {"description"}

# --------------------------------------------------------------------------
# Units for bare numeric fields
# --------------------------------------------------------------------------
# Groww's embedded JSON payload stores some values as unitless numbers whose unit
# is carried by the FIELD NAME rather than the value:
#
#     "expense_ratio": "1.03"        (rendered by the page as "1.03%")
#     "min_sip_investment": 500      (rendered by the page as "Rs 500")
#
# Extracted verbatim, the corpus said "Expense ratio: 1.03" and "Minimum SIP
# amount: 500", and the LLM faithfully repeated those unitless numbers. That is
# the product's headline answer delivered without its unit -- "the expense ratio
# is 1.03" is not a complete fact, and a reader cannot tell a percentage from a
# rupee amount.
#
# The unit is therefore attached here, in the extractor, rather than asked of the
# model in the prompt. Two reasons, in order of importance:
#
# 1. It is corroborated by the SAME snapshot, not imported from outside. The page
#    carries a second rendering of the same fact in its analysis blob --
#    `analysis_desc: "Lower expense ratio: 1.03%"` -- so the % is the page's own
#    rendering of the field being extracted, not an assumption. Groww displays
#    minimums in rupees throughout, and the brief's own PII fixtures write
#    "Minimum SIP is Rs 500".
# 2. A unit in the prompt is a unit the model can silently decline to apply, or
#    invent. Here it is in the indexed text, so it is embedded, retrieved, and
#    cited along with the figure.
#
# SAFETY RULES, both load-bearing:
#
# - Applied ONLY when the value is a bare number -- an int, a float, or a string
#   that is nothing but digits and separators. A value that already carries text
#   ("1.03%", "Rs 500", "Nil", the free-text exit load) is passed through
#   untouched, so a field is never double-suffixed. The distinction matters
#   because the five pages disagree on shape: S1 stores `expense_ratio` as the
#   string "1.03" while `min_sip_investment` is a real int.
# - "%" only, never "p.a.". `FORBIDDEN_PATTERNS` blocks `p. a.` as a per-annum
#   performance qualifier (see its entry), and the corpus must not grow a phrase
#   the guard is designed to reject.
#
# These are the ONLY annotated fields. Any new numeric field is rendered
# unitless until someone adds it here with the same evidence.
_UNIT_SUFFIX: dict[str, str] = {
    "expense_ratio": "%",
    "base_expense_ratio": "%",
}
_CURRENCY_FIELDS: frozenset[str] = frozenset(
    {"min_sip_investment", "min_investment_amount", "min_withdrawal"}
)

# A value that is nothing but a number: optional sign, digits, and thousands or
# decimal separators. Anchored, so "1.03% per annum" and "1 year" both fail.
_BARE_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)*$")

_MISSING = object()


def _annotate_unit(field_name: str, text: str) -> str:
    """Attach the field's unit to a value already known to be a bare number."""
    if field_name in _UNIT_SUFFIX:
        return f"{text}{_UNIT_SUFFIX[field_name]}"
    if field_name in _CURRENCY_FIELDS:
        return f"Rs {text}"
    return text


def normalize_value(field_name: str, value: Any) -> Any:
    """Normalise the three value shapes that are not already display-ready.

    - `lock_in` is a dict of {years, months, days}, all-null on four of the five
      schemes. Rendered as an explicit "None" so the corpus can answer "is there
      a lock-in?" negatively instead of being silent on it.
    - `nfo_risk` is inconsistently formatted across pages: S1/S4/S5 say
      "Moderately High Riskometer", S2/S3 say "Moderately High". The trailing
      "Riskometer" is stripped so all five schemes are phrased identically.
    - `exit_load` arrives with stray newlines on some pages and has three
      distinct shapes across the five schemes (time-based, "Nil", and a partial
      load keyed to a 15% threshold on S5). Whitespace is collapsed; the text
      itself is never reinterpreted.
    - Bare numeric values for the fields in `_UNIT_SUFFIX` / `_CURRENCY_FIELDS`
      are annotated with their unit, because the payload carries the unit in the
      field name and not in the value. See the table's own comment for the
      evidence and the two safety rules.
    """
    if value is _MISSING or value is None:
        return None

    if field_name == "lock_in":
        if not isinstance(value, dict):
            return str(value)
        parts = []
        for key, unit in (("years", "year"), ("months", "month"), ("days", "day")):
            n = value.get(key)
            if isinstance(n, int) and n > 0:
                parts.append(f"{n} {unit}" + ("s" if n != 1 else ""))
        return ", ".join(parts) if parts else "None"

    if field_name == "nfo_risk" and isinstance(value, str):
        cleaned = re.sub(
            r"\s*riskometer\s*$", "", value.strip(), flags=re.IGNORECASE
        )
        return cleaned or value.strip()

    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
        # A bare number in a string is still a bare number: S1 stores
        # `expense_ratio` as the string "1.03" and S2 as a real float, and the
        # unit belongs to the field either way.
        if _BARE_NUMBER.match(text):
            return _annotate_unit(field_name, text)
        return text

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return _annotate_unit(field_name, str(value))

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def field_value(data: dict[str, Any], field_name: str) -> Any:
    """Read one field and normalise it. Missing keys return None."""
    return normalize_value(field_name, data.get(field_name, _MISSING))


def render_document_text(data: dict[str, Any]) -> str:
    """Render the allowlisted fields of a scheme payload as loader text.

    Only keys in ALLOWED_FIELDS are ever read. FORBIDDEN_FIELDS keys are never
    touched, so a new performance field added to the page cannot leak into the
    corpus without also being added to this tuple.
    """
    grouped: dict[str, list[str]] = {s: [] for s in _SECTION_ORDER}
    prose: list[str] = []

    for name in ALLOWED_FIELDS:
        value = field_value(data, name)
        if value is None or value == "":
            continue
        label = _LABEL.get(name, name.replace("_", " ").capitalize())
        if name in _PROSE_FIELDS:
            # description is prose, not a single fact: it goes under its own
            # heading as a paragraph rather than as a `Label: value` row.
            prose.append(f"{label}: {value}")
            continue
        grouped[_SECTION.get(name, "Other details")].append(f"{label}: {value}")

    blocks: list[str] = [
        f"## {section}\n" + "\n".join(rows)
        for section, rows in grouped.items()
        if rows
    ]
    if prose:
        blocks.append("## Objective\n" + "\n".join(prose))

    return "\n\n".join(blocks).strip() + "\n"


def selected_fields(data: dict[str, Any]) -> tuple[str, ...]:
    """The allowlisted keys actually present in this payload."""
    return tuple(k for k in ALLOWED_FIELDS if k in data)


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------
@dataclass
class GuardResult:
    """Outcome of a corpus guard check."""

    source_id: str
    ok: bool
    forbidden_fields_used: tuple[str, ...] = ()
    pattern_hits: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return f"{self.source_id}: clean"
        parts = []
        if self.forbidden_fields_used:
            parts.append(
                f"forbidden fields rendered: {sorted(self.forbidden_fields_used)}"
            )
        if self.pattern_hits:
            parts.append(f"forbidden patterns matched: {sorted(self.pattern_hits)}")
        return f"{self.source_id}: " + "; ".join(parts)


def scan_forbidden_patterns(text: str) -> tuple[str, ...]:
    """Return the labels of every FORBIDDEN_PATTERNS entry that matches."""
    return tuple(
        label for pattern, label in _COMPILED if pattern.search(text)
    )


def guard(
    data: dict[str, Any] | None,
    text: str,
    *,
    source_id: str,
) -> GuardResult:
    """Check that nothing forbidden reached the text. Never raises.

    Two independent checks:

    1. Structural -- did the renderer read any forbidden field? This is the
       load-bearing one, because under allowlist extraction the answer should
       always be no.
    2. Textual -- does the rendered text match any FORBIDDEN_PATTERNS entry?
       This is defence in depth: it catches a leak that arrived by some route
       other than the renderer, e.g. a field's own prose.

    P3's builder calls this and HARD-FAILS the build on a non-ok result, so a
    leak can never reach the vector index.
    """
    used: tuple[str, ...] = ()
    notes: list[str] = []

    if data is not None:
        leaked = FORBIDDEN_FIELDS & set(data.keys())
        rendered = _rendered_labels(text)
        used = tuple(sorted(leaked & rendered))
        if leaked:
            notes.append(
                f"payload carries {len(leaked)} forbidden fields, none rendered"
            )

    hits = scan_forbidden_patterns(text)
    return GuardResult(
        source_id=source_id,
        ok=not used and not hits,
        forbidden_fields_used=used,
        pattern_hits=hits,
        notes=notes,
    )


def _rendered_labels(text: str) -> set[str]:
    """Labels appearing as `Label:` in the rendered text."""
    found: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^([A-Z][^:]{2,60}):", line.strip())
        if m:
            found.add(m.group(1).strip())
    return found


def _label_for(field_name: str) -> str:
    return _LABEL.get(field_name, field_name.replace("_", " ").capitalize())


def assert_allowlist_disjoint() -> None:
    """The load-bearing invariant. Raises if it is ever violated.

    If a field were ever added to both sets, the allowlist would win (it is what
    the renderer iterates) and the forbidden set would be a lie. Make that
    impossible to ship silently.
    """
    overlap = set(ALLOWED_FIELDS) & FORBIDDEN_FIELDS
    if overlap:
        raise ValueError(
            f"ALLOWED_FIELDS and FORBIDDEN_FIELDS overlap: {sorted(overlap)}"
        )


def allowed_field_names() -> Iterable[str]:
    return ALLOWED_FIELDS
