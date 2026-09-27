"""Loader tests. Use realistic fixtures built from the real field shapes."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag_bot.config import load
from rag_bot.ingest.loaders import (
    _fallback_text,
    extract_payload,
    load_all,
    load_from_manifest,
    load_source,
)
from rag_bot.sources.fetch import load_sources
from rag_bot.types import Source

SNAPSHOTS = Path(load().corpus_dir) / "snapshots"
MANIFEST = Path(load().corpus_dir) / "manifest.csv"


@pytest.fixture(scope="module")
def sources():
    return load_sources(load().sources_file)


def _need_snapshots():
    if not list(SNAPSHOTS.glob("*.html")):
        pytest.skip("no snapshots; run the fetcher")


# --------------------------------------------------------------------------
# Payload extraction
# --------------------------------------------------------------------------
def test_extract_payload_from_a_synthetic_page():
    payload = {"fund_name": "Test Fund", "expense_ratio": "1.03", "nav": 12.34}
    page = (
        "<html><body><script id=\"__NEXT_DATA__\" type=\"application/json\">"
        + json.dumps({"props": {"pageProps": {"mfServerSideData": payload}}})
        + "</script></body></html>"
    )
    assert extract_payload(page) == payload


def test_extract_payload_returns_none_when_absent():
    assert extract_payload("<html><body>no blob</body></html>") is None


def test_extract_payload_returns_none_on_bad_json():
    page = '<script id="__NEXT_DATA__">{not json}</script>'
    assert extract_payload(page) is None


def test_extract_payload_returns_none_on_unexpected_shape():
    page = (
        '<script id="__NEXT_DATA__">'
        + json.dumps({"props": {"pageProps": {"somethingElse": 1}}})
        + "</script>"
    )
    assert extract_payload(page) is None


def test_extract_payload_never_raises_on_fuzzed_input():
    for junk in ("", "<script id=__NEXT_DATA__></script>", "\x00\xff not json"):
        assert extract_payload(junk) is None


# --------------------------------------------------------------------------
# The text format contract
# --------------------------------------------------------------------------
def test_loader_emits_headings_and_label_value_lines(sources):
    _need_snapshots()
    src = next(s for s in sources if s.scheme_id == "S1")
    src = Source(**{**src.__dict__, "snapshot_path": str(SNAPSHOTS / f"{src.source_id}.html")})
    doc = load_source(src)
    assert doc.text
    assert re.search(r"^## .+$", doc.text, re.M)
    assert re.search(r"^[A-Z][^:\n]{2,40}: \S+", doc.text, re.M)


def test_loader_output_is_the_demo_chunk_facts(sources):
    """The expense ratio and its label must be in the same text, adjacent."""
    _need_snapshots()
    src = next(s for s in sources if s.scheme_id == "S1")
    src = Source(**{**src.__dict__, "snapshot_path": str(SNAPSHOTS / f"{src.source_id}.html")})
    text = load_source(src).text
    assert "Expense ratio: 1.03" in text
    assert "## Expense ratio" in text
    assert "## Exit load" in text
    assert "Benchmark: NIFTY 100 Total Return Index" in text
    assert "Riskometer level: Moderately High" in text


def test_load_all_five_produce_distinct_nonempty_documents():
    _need_snapshots()
    docs = load_from_manifest(str(MANIFEST))
    assert len(docs) == 5
    assert all(d.text.strip() for d in docs)
    texts = {d.text for d in docs}
    assert len(texts) == 5, "documents should differ per scheme"


def test_load_all_never_raises_on_a_missing_snapshot(sources):
    src = sources[0]
    broken = Source(**{**src.__dict__, "snapshot_path": "/nonexistent/nope.html"})
    doc = load_source(broken)
    assert doc.text == ""


def test_load_source_never_raises_on_a_directory(sources):
    src = sources[0]
    broken = Source(**{**src.__dict__, "snapshot_path": str(SNAPSHOTS)})
    assert load_source(broken).text == ""


def test_load_source_handles_no_snapshot_path(sources):
    src = Source(**{**sources[0].__dict__, "snapshot_path": None})
    assert load_source(src).text == ""


def test_load_all_on_empty_input():
    assert load_all([]) == []


# --------------------------------------------------------------------------
# Factsheet: none exists, and the loader must not pretend otherwise
# --------------------------------------------------------------------------
def test_loader_never_sets_a_factsheet_url(sources):
    """No factsheet PDF exists on any of the five pages. The loader must leave
    factsheet_url null rather than inventing a link."""
    _need_snapshots()
    for doc in load_from_manifest(str(MANIFEST)):
        assert doc.source.factsheet_url is None


def test_loader_offers_the_official_amc_site_instead(sources):
    _need_snapshots()
    doc = next(d for d in load_from_manifest(str(MANIFEST)) if d.source.scheme_id == "S1")
    assert "Official AMC website: https://www.hdfcfund.com" in doc.text


# --------------------------------------------------------------------------
# Fallback path
# --------------------------------------------------------------------------
def test_fallback_strips_scripts_and_nav():
    page = (
        "<html><head><style>.x{}</style></head><body>"
        "<nav>Home About</nav><script>evil()</script>"
        "<h2>Fees</h2><p>Expense ratio for the scheme is stated by the AMC.</p>"
        "<footer>copyright</footer></body></html>"
    )
    text = _fallback_text(page)
    assert "## Fees" in text
    assert "Expense ratio for the scheme" in text
    for junk in ("evil()", "Home About", "copyright", ".x{}"):
        assert junk not in text


def test_fallback_does_not_scrape_tables():
    """3 of the 4 rendered tables on these pages are performance data. The
    fallback path has no allowlist discipline, so it must not import tables."""
    page = (
        "<html><body><table><tr><th>1 year</th><th>Returns</th></tr>"
        "<tr><td>2025</td><td>14.2%</td></tr></table></body></html>"
    )
    text = _fallback_text(page)
    assert "14.2" not in text
    assert "1 year" not in text


def test_fallback_on_a_real_snapshot_with_payload_is_not_used(sources):
    _need_snapshots()
    src = sources[0]
    src = Source(**{**src.__dict__, "snapshot_path": str(SNAPSHOTS / f"{src.source_id}.html")})
    text = load_source(src).text
    # the allowlist path, not the fallback: no long scraped paragraphs
    assert "## Objective" in text
