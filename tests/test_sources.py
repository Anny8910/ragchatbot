"""Source registry and manifest tests. No network."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from rag_bot.config import load
from rag_bot.sources.fetch import load_sources
from rag_bot.sources.manifest import HEADER, NOTES, read_manifest, write_manifest
from rag_bot.types import Source

SOURCES_YAML = load().sources_file

# Category labels / bare category words that must never be usable as a scheme
# alias, because they either name no unique scheme or collide with another
# scheme's category label (architecture §14.4).
FORBIDDEN_BARE_ALIASES = {
    "large cap",
    "small cap",
    "mid cap",
    "flexi cap",
    "balanced advantage",
    "hybrid",
    "equity",          # S2 is "HDFC Equity Fund" but this is S1/S4's category word
    "equity large cap",  # S1's Groww category label
    "equity small cap",  # S4's Groww category label
}


@pytest.fixture(scope="module")
def sources() -> list[Source]:
    return load_sources(SOURCES_YAML)


def test_five_sources_present(sources):
    assert len(sources) == 5
    assert [s.scheme_id for s in sources] == ["S1", "S2", "S3", "S4", "S5"]


def test_aliases_are_mutually_exclusive(sources):
    """The spec's explicit requirement: no alias may resolve to two schemes."""
    owner: dict[str, str] = {}
    for src in sources:
        for alias in src.aliases:
            key = " ".join(alias.lower().split())
            assert key, f"{src.source_id} has an empty alias"
            assert key not in owner, (
                f"alias {key!r} is claimed by both {owner[key]} and "
                f"{src.source_id}"
            )
            owner[key] = src.source_id


def test_no_alias_is_a_bare_category_word(sources):
    for src in sources:
        for alias in src.aliases:
            key = " ".join(alias.lower().split())
            assert key not in FORBIDDEN_BARE_ALIASES, (
                f"{src.source_id} uses bare category word {key!r} as an alias"
            )


def test_aliases_include_scheme_id(sources):
    for src in sources:
        assert src.scheme_id.lower() in {a.lower() for a in src.aliases}


def test_aliases_are_lowercase_and_trimmed(sources):
    for src in sources:
        for alias in src.aliases:
            assert alias == alias.lower(), f"{src.source_id}: {alias!r} not lowercase"
            assert alias == alias.strip(), f"{src.source_id}: {alias!r} not trimmed"


def test_scheme_id_alone_resolves_unambiguously(sources):
    """'s1'..'s5' must each be owned by exactly one scheme."""
    for src in sources:
        others = [
            s.scheme_id
            for s in sources
            if src.scheme_id.lower() in {a.lower() for a in s.aliases}
        ]
        assert others == [src.scheme_id]


def test_s5_url_is_the_corrected_slug(sources):
    """The brief's S5 URL 404s. Guard against it creeping back in."""
    s5 = next(s for s in sources if s.scheme_id == "S5")
    assert s5.url.endswith("hdfc-balanced-advantage-fund-direct-growth")
    assert "-direct-plan-" not in s5.url
    assert s5.source_id in NOTES


def test_manifest_round_trip(tmp_path, sources):
    out = tmp_path / "manifest.csv"
    write_manifest(sources, str(out))

    with out.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == HEADER
    assert [r[1] for r in rows[1:]] == ["S1", "S2", "S3", "S4", "S5"]

    back = read_manifest(str(out))
    assert [s.source_id for s in back] == [s.source_id for s in sources]
    assert [s.status for s in back] == [s.status for s in sources]


def test_manifest_notes_only_on_deviating_sources(tmp_path, sources):
    out = tmp_path / "manifest.csv"
    write_manifest(sources, str(out))
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    noted = {r["source_id"] for r in rows if r["notes"]}
    # S5: substituted URL. S2: rebranded by HDFC since the brief was written.
    assert noted == {"hdfc_balanced_advantage_growth", "hdfc_equity_growth"}


def test_manifest_header_has_notes_column():
    assert HEADER[-1] == "notes"
    assert len(HEADER) == 11  # 10 from the spec, plus the notes deviation


def _norm(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def test_live_fund_name_resolves_for_every_snapshot(sources):
    """Every snapshot's own fund_name must be reachable through the alias table.

    This is the test that caught a real defect: the brief calls S2 'HDFC Equity
    Fund', but the live page reports 'HDFC Flexi Cap Direct Plan-Growth'. With
    only the brief's naming in the aliases, a user asking about the fund by the
    name HDFC actually uses would not resolve to any scheme.

    Skips when snapshots are absent, so the suite still runs on a clean checkout.
    """
    import html as htmllib
    import json

    snapshots = Path(load().corpus_dir) / "snapshots"
    checked = 0
    for src in sources:
        snap = snapshots / f"{src.source_id}.html"
        if not snap.exists():
            pytest.skip(f"snapshot missing for {src.source_id}; run the fetcher")
        raw = snap.read_text(encoding="utf-8")
        m = re.search(r'(?s)<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', raw)
        assert m, f"{src.source_id}: no __NEXT_DATA__ in snapshot"
        data = json.loads(htmllib.unescape(m.group(1)))
        fund_name = data["props"]["pageProps"]["mfServerSideData"]["fund_name"]

        normalized = _norm(fund_name)
        aliases = {_norm(a) for a in src.aliases}
        assert any(a in normalized for a in aliases), (
            f"{src.source_id}: live fund_name {fund_name!r} matches none of its "
            f"aliases {sorted(aliases)}"
        )
        checked += 1
    assert checked == 5


def test_no_source_is_disclosed_as_official(sources):
    """PRD Q1 was resolved 2026-09-27: groww.in IS acceptable, but only on the
    condition that no row is presented as an official AMC/SEBI/AMFI document.

    Groww is a broker, so all five rows must stay source_tier "brief" and must
    name their real publisher. A row claiming "official_ref" or an
    hdfcfund.com/sebi/amfi publisher would misstate provenance on every citation
    the assistant emits.
    """
    for src in sources:
        assert src.source_tier == "brief", (
            f"{src.source_id}: tier is {src.source_tier!r}; the groww.in corpus is "
            f"not official and must not be labelled as such"
        )
        assert src.publisher == "groww.in", (
            f"{src.source_id}: publisher {src.publisher!r} must be disclosed "
            f"accurately"
        )


def test_manifest_discloses_publisher_and_tier(tmp_path, sources):
    """Every manifest row must carry a non-empty publisher and tier, so D2 can be
    audited row by row."""
    out = tmp_path / "manifest.csv"
    write_manifest(sources, str(out))
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5
    for r in rows:
        assert r["publisher"] == "groww.in"
        assert r["source_tier"] == "brief"


# -- class-D factsheet link (resolved 2026-09-27) --------------------------


def test_every_source_carries_a_factsheet_link(sources):
    """Class D must have somewhere official to send the user.

    The five groww pages contain no factsheet (brochure_link is null, zero PDF
    links), so the link is the AMC's own factsheet page. A source without one
    would make the class-D refusal a dead end.
    """
    for src in sources:
        assert src.factsheet_url, f"{src.source_id} has no factsheet_url"
        assert src.factsheet_url.startswith("https://"), src.factsheet_url


def test_the_factsheet_link_is_the_official_amc_not_the_broker(sources):
    for src in sources:
        assert "hdfcfund.com" in src.factsheet_url, src.factsheet_url
        assert "groww.in" not in src.factsheet_url, (
            "the class-D link must be the official publisher, not the broker"
        )


def test_the_factsheet_link_is_not_month_stamped(sources):
    """A "Fund Facts - <Scheme>_July 26.pdf" URL rots within a month."""
    for src in sources:
        low = src.factsheet_url.lower()
        assert not low.endswith(".pdf"), "a month-stamped PDF URL would rot"
        for stamp in ("2025", "2026", "jan", "july", "monthly"):
            assert stamp not in low.split("/")[-1], src.factsheet_url


def test_the_registry_join_survives_the_manifest_round_trip(tmp_path, sources):
    """read_manifest alone would drop factsheet_url and aliases.

    The manifest has no column for either, because it records what was FETCHED
    and the factsheet page is deliberately not fetched. Without the registry join
    the class-D link silently becomes None.
    """
    out = tmp_path / "manifest.csv"
    write_manifest(sources, str(out))
    with_registry = read_manifest(str(out), registry=sources)
    without = read_manifest(str(out))
    for src in with_registry:
        assert src.factsheet_url == sources[0].factsheet_url
        assert src.aliases
    assert all(s.factsheet_url is None for s in without)
    assert all(s.aliases == () for s in without)


def test_the_factsheet_link_reaches_chunk_metadata():
    """The link has to survive all the way into the index, not stop at the Source."""
    from rag_bot.config import load
    from rag_bot.index.store import Store

    cfg = load()
    store = Store(cfg.index_dir, cfg.embed_model, "v1")
    if store.count() == 0:
        pytest.skip("no index built")
    got = store._collection.get(include=["metadatas"])
    urls = {m["factsheet_url"] for m in got["metadatas"]}
    assert urls == {"https://www.hdfcfund.com/mutual-funds/factsheets"}
