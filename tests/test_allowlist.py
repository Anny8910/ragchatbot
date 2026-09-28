"""Allowlist tests -- the structural guarantee behind the no-figures constraint."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from rag_bot.config import load
from rag_bot.ingest.allowlist import (
    ALLOWED_FIELDS,
    FORBIDDEN_FIELDS,
    FORBIDDEN_PATTERNS,
    assert_allowlist_disjoint,
    field_value,
    guard,
    normalize_value,
    render_document_text,
    scan_forbidden_patterns,
)
from rag_bot.sources.fetch import load_sources

SNAPSHOTS = Path(load().corpus_dir) / "snapshots"


def _payloads() -> dict[str, dict]:
    """Every committed snapshot's scheme payload, keyed by source_id."""
    out = {}
    for p in sorted(SNAPSHOTS.glob("*.html")):
        if not p.exists():
            continue
        m = re.search(
            r'(?s)<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            p.read_text(encoding="utf-8"),
        )
        if m:
            out[p.stem] = json.loads(
                __import__("html").unescape(m.group(1))
            )["props"]["pageProps"]["mfServerSideData"]
    return out


# --------------------------------------------------------------------------
# The load-bearing invariant
# --------------------------------------------------------------------------
def test_allowlist_and_forbidden_are_disjoint():
    """If a field were in both sets, the forbidden set would be a lie."""
    assert_allowlist_disjoint()  # raises on overlap
    assert not (set(ALLOWED_FIELDS) & FORBIDDEN_FIELDS)


def test_assert_allowlist_disjoint_actually_raises():
    """Prove the invariant is enforced, not merely documented."""
    import rag_bot.ingest.allowlist as mod

    original = mod.FORBIDDEN_FIELDS
    try:
        mod.FORBIDDEN_FIELDS = frozenset(set(original) | {"expense_ratio"})
        with pytest.raises(ValueError, match="overlap"):
            mod.assert_allowlist_disjoint()
    finally:
        mod.FORBIDDEN_FIELDS = original


def test_every_forbidden_field_exists_in_the_real_payloads():
    """A forbidden name that does not exist would be a typo, giving false
    confidence that the field is blocked."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    all_keys = set().union(*(set(p) for p in payloads.values()))
    unknown = FORBIDDEN_FIELDS - all_keys
    assert not unknown, f"FORBIDDEN_FIELDS names keys absent from every payload: {sorted(unknown)}"


def test_every_allowed_field_exists_in_the_real_payloads():
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    for p in payloads.values():
        missing = [k for k in ALLOWED_FIELDS if k not in p]
        assert not missing, f"ALLOWED_FIELDS names keys absent from the payload: {missing}"


# --------------------------------------------------------------------------
# The two denylist bugs, as regression tests
# --------------------------------------------------------------------------
def test_benchmark_name_survives_extraction():
    """REGRESSION 1. A denylist regex matching the word "Return" deleted
    "NIFTY 500 Total Return Index" -- an in-scope topic -- from the corpus."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    for source_id, p in payloads.items():
        text = render_document_text(p)
        assert p["benchmark_name"] in text, (
            f"{source_id}: benchmark {p['benchmark_name']!r} was dropped"
        )


def _walk(node):
    """Yield (key, value) for every mapping anywhere inside node.

    The performance payloads are inconsistently shaped: `return_stats` and
    `stats` are LISTS of dicts, while `simple_return` and `sip_return` are
    single DICTS. A walker that assumed one shape would silently check nothing.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_no_forbidden_value_reaches_the_text():
    """REGRESSION 2. A denylist regex passed NAV and three return payloads
    through: `nav` is a bare float with no keyword, and `\\breturn\\b` cannot
    match the camelCase key `return1d`."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    checked_returns = 0
    for source_id, p in payloads.items():
        text = render_document_text(p)

        # NAV: a bare decimal that is not one of our allowlisted numbers
        nav = str(p["nav"])
        assert nav not in text, f"{source_id}: NAV {nav} leaked into the text"

        # every return figure in every performance payload
        for stats_key in ("return_stats", "simple_return", "sip_return", "stats"):
            for k, v in _walk(p.get(stats_key)):
                if not isinstance(v, (int, float)):
                    continue
                assert f"{k}: {v}" not in text, (
                    f"{source_id}: {stats_key} value {k}={v} leaked"
                )
                assert re.search(rf"\b{k}\b", text) is None, (
                    f"{source_id}: key {k!r} leaked"
                )
                checked_returns += 1

    # the test is only meaningful if it actually inspected return figures
    assert checked_returns > 50, f"only checked {checked_returns} return values"


def test_guard_is_clean_on_every_real_snapshot():
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    for source_id, p in payloads.items():
        text = render_document_text(p)
        result = guard(p, text, source_id=source_id)
        assert result.ok, result.describe()
        assert result.forbidden_fields_used == ()
        assert result.pattern_hits == ()


def test_guard_catches_a_leaked_return_key():
    """The guard must actually fail on a leak, not merely report clean."""
    payload = {"nav": 1447.383, "expense_ratio": "1.03"}
    text = "Expense ratio: 1.03\nNAV: 1447.383"
    result = guard(payload, text, source_id="probe")
    assert not result.ok
    assert "NAV" in result.pattern_hits


def test_guard_catches_a_camelcase_return_key():
    text = "Return performance: return1d: 0.38"
    assert "raw return-stat key" in scan_forbidden_patterns(text)


# --------------------------------------------------------------------------
# Deliberate non-matches: the false positives the denylist would have caused
# --------------------------------------------------------------------------
def test_bare_return_word_is_not_forbidden():
    """`return` alone must NOT be forbidden: benchmark names contain it."""
    for text in (
        "Benchmark: NIFTY 500 Total Return Index",
        "Benchmark: BSE 250 SmallCap Total Return Index",
        "Benchmark: NIFTY 50 Hybrid Composite Debt 50:50 Index",
    ):
        assert scan_forbidden_patterns(text) == (), f"false positive on {text!r}"


def test_percentage_is_not_forbidden():
    """A `%` must NOT be forbidden: fees are percentages and are the product."""
    for text in (
        "Expense ratio: 1.03",
        "Exit load: Exit load of 1% if redeemed within 1 year",
        "Exit load: Exit Load for units in excess of 15% of the investment, "
        "1% will be charged for redemption within 1 year",
    ):
        assert scan_forbidden_patterns(text) == (), f"false positive on {text!r}"


def test_growth_is_not_forbidden():
    """`Growth` appears in every scheme name (Direct Growth plan)."""
    for text in (
        "Scheme name: HDFC Large Cap Fund Direct Growth",
        "Plan option: Growth",
    ):
        assert scan_forbidden_patterns(text) == (), f"false positive on {text!r}"


def test_riskometer_level_is_not_forbidden():
    """We keep the riskometer LEVEL. Only its numeric score would be excluded."""
    assert scan_forbidden_patterns("Riskometer level: Moderately High") == ()


def test_expense_ratio_line_survives_guarding():
    """The P2 spec's must-preserve test, restated against the new design."""
    text = "## Fees and charges\nExpense ratio: 1.05\nExit load: Nil"
    assert scan_forbidden_patterns(text) == ()


def test_pii_patterns_fire():
    assert scan_forbidden_patterns("Contact us at support@example.com") == (
        "email address",
    )
    assert scan_forbidden_patterns("Call +91 98765 43210 for help")


# --------------------------------------------------------------------------
# Value normalisation
# --------------------------------------------------------------------------
def test_lock_in_renders_years():
    assert normalize_value("lock_in", {"years": 3, "months": 0, "days": 0}) == "3 years"


def test_lock_in_renders_none_when_absent():
    """Four of five schemes have no lock-in. It must render explicitly so the
    corpus can answer 'is there a lock-in?' negatively."""
    assert normalize_value("lock_in", {"years": None, "months": None, "days": None}) == "Nil"
    assert normalize_value("lock_in", None) is None


def test_riskometer_trailing_word_is_stripped():
    """S1/S4/S5 say 'Moderately High Riskometer'; S2/S3 say 'Moderately High'."""
    assert normalize_value("nfo_risk", "Moderately High Riskometer") == "Moderately High"
    assert normalize_value("nfo_risk", "Moderately High") == "Moderately High"


def test_all_five_schemes_render_riskometer_identically():
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    levels = {
        sid: field_value(p, "nfo_risk") for sid, p in payloads.items()
    }
    assert set(levels.values()) == {"Moderately High"}, levels


def test_exit_load_whitespace_is_collapsed():
    raw = "Exit load of 1% if redeemed within 1 year\r\n"
    assert normalize_value("exit_load", raw) == (
        "Exit load of 1% if redeemed within 1 year"
    )


def test_exit_load_has_three_distinct_shapes():
    """documented in data-findings: the renderer must not assume one format."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    shapes = {sid: (p["exit_load"] or "")[:40] for sid, p in payloads.items()}
    assert len(set(shapes.values())) >= 3, shapes


# --------------------------------------------------------------------------
# Rendered output
# --------------------------------------------------------------------------
def test_render_emits_headings_and_label_value_lines():
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    text = render_document_text(next(iter(payloads.values())))
    assert "## Expense ratio" in text
    assert "## Exit load" in text
    assert "## Minimum investments" in text
    assert re.search(r"^Expense ratio: \S+", text, re.M)
    assert "## Lock-in period" in text
    assert "## Riskometer" in text
    assert "## Benchmark" in text


def test_render_is_deterministic():
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    p = next(iter(payloads.values()))
    assert render_document_text(p) == render_document_text(p)


def test_render_omits_absent_fields():
    text = render_document_text({"expense_ratio": "1.03"})
    assert "Expense ratio: 1.03" in text
    assert "Benchmark" not in text
    assert "Riskometer" not in text


def test_lock_in_none_is_present_in_text():
    """S1 has no lock-in; the chunk must still carry the answer."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    s1 = payloads.get("hdfc_large_cap_growth")
    if s1:
        assert "Lock-in period: Nil" in render_document_text(s1)


def test_factsheet_url_is_absent_but_amc_site_is_present():
    """No factsheet PDF exists on any page. The loader must not imply one."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    for source_id, p in payloads.items():
        assert p.get("brochure_link") is None
        text = render_document_text(p)
        assert "Official AMC website" in text
        assert p["sid_url"] in text


def test_forbidden_patterns_list_is_nonempty_and_labelled():
    assert len(FORBIDDEN_PATTERNS) >= 8
    for _pattern, label in FORBIDDEN_PATTERNS:
        assert isinstance(label, str) and label


# --- unit annotation on bare numeric fields ---------------------------------
# Groww's payload carries the unit in the field NAME, not in the value
# (`"expense_ratio": "1.03"`). Rendered verbatim, the corpus said "Expense
# ratio: 1.03" and the model repeated the unitless number, which is the product's
# headline answer delivered incomplete. See the _UNIT_SUFFIX table in
# rag_bot/ingest/allowlist.py for the evidence that % is the page's own rendering.


def test_expense_ratio_gets_a_percent_sign():
    assert normalize_value("expense_ratio", "1.03") == "1.03%"
    assert normalize_value("base_expense_ratio", 0.84) == "0.84%"


def test_minimum_investments_get_a_currency_prefix():
    for field in ("min_sip_investment", "min_investment_amount", "min_withdrawal"):
        assert normalize_value(field, 500) == "Rs 500"


def test_units_are_never_doubled_on_a_value_that_already_has_one():
    """Only int/float is annotated. A string value has already been through the
    string branch, so a field arriving as "1.03%" or "Nil" must survive intact."""
    assert normalize_value("expense_ratio", "1.03%") == "1.03%"
    assert normalize_value("expense_ratio", "Nil") == "Nil"
    assert normalize_value("min_sip_investment", "Rs 500") == "Rs 500"


def test_no_per_annum_qualifier_is_added():
    """`FORBIDDEN_PATTERNS` blocks `p. a.` as a per-annum performance qualifier,
    so the extractor must not grow a phrase its own guard would reject."""
    out = normalize_value("expense_ratio", "1.03")
    assert "p.a" not in out.lower() and "p. a." not in out.lower()
    assert not scan_forbidden_patterns(out), scan_forbidden_patterns(out)


def test_annotation_does_not_touch_unlisted_numeric_fields():
    """Adding a numeric field must not silently inherit a unit. Unlisted fields
    render bare until someone adds the evidence for them."""
    assert normalize_value("portfolio_turnover", 18) == "18"


def test_real_snapshots_render_units_everywhere_they_apply():
    """End-to-end on the actual corpus, not a hand-built payload: a unit added to
    the table but absent from a real render would be a silent regression."""
    payloads = _payloads()
    if not payloads:
        pytest.skip("no snapshots; run the fetcher")
    for source_id, p in payloads.items():
        text = render_document_text(p)
        assert re.search(r"Expense ratio: \S+%", text), source_id
        assert re.search(r"Minimum SIP amount: Rs [\d,]+", text), source_id
