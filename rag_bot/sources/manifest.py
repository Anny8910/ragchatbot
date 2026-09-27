"""Manifest read/write. data/corpus/manifest.csv IS deliverable D2."""
from __future__ import annotations

import csv
from pathlib import Path

from rag_bot.types import Source

# DEVIATION from the P1 spec's 10-column header: a trailing `notes` column was
# added. PRD open question Q1 requires honest disclosure of provenance, and S5's
# URL is a documented substitution from the brief. That belongs in the
# deliverable itself, not only in docs/data-findings.md. See
# docs/data-findings.md sections 1 and 6.
HEADER = [
    "source_id",
    "scheme_id",
    "scheme_name",
    "url",
    "publisher",
    "source_tier",
    "fetched_at",
    "snapshot_path",
    "content_hash",
    "status",
    "notes",
]

# Per-source provenance notes. Only deviations are recorded; a blank cell means
# the URL is verbatim from the brief.
NOTES: dict[str, str] = {
    "hdfc_balanced_advantage_growth": (
        "URL substituted: brief's "
        "hdfc-balanced-advantage-fund-direct-plan-growth returns HTTP 404; "
        "corrected slug verified 200. See docs/data-findings.md section 1."
    ),
    "hdfc_equity_growth": (
        "Name drift: brief/PRD call this 'HDFC Equity Fund', but the live page "
        "reports fund_name 'HDFC Flexi Cap Direct Plan-Growth' as of "
        "2026-09-27. HDFC has rebranded it. Aliases cover both names."
    ),
}


def _row(src: Source) -> dict[str, str]:
    return {
        "source_id": src.source_id,
        "scheme_id": src.scheme_id or "",
        "scheme_name": src.scheme_name,
        "url": src.url,
        "publisher": src.publisher,
        "source_tier": src.source_tier,
        "fetched_at": src.fetched_at or "",
        "snapshot_path": src.snapshot_path or "",
        "content_hash": src.content_hash or "",
        "status": src.status,
        "notes": NOTES.get(src.source_id, ""),
    }


def write_manifest(sources: list[Source], out_path: str) -> None:
    """Write CSV with HEADER, sorted by scheme_id."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(sources, key=lambda s: (s.scheme_id or ""))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for src in ordered:
            writer.writerow(_row(src))


def read_manifest(path: str) -> list[Source]:
    """Inverse of write_manifest."""
    out: list[Source] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                Source(
                    source_id=row["source_id"],
                    url=row["url"],
                    publisher=row["publisher"],
                    source_tier=row["source_tier"],
                    scheme_id=row["scheme_id"] or None,
                    scheme_name=row["scheme_name"],
                    factsheet_url=None,
                    aliases=(),
                    fetched_at=row["fetched_at"] or None,
                    snapshot_path=row["snapshot_path"] or None,
                    content_hash=row["content_hash"] or None,
                    status=row["status"],
                )
            )
    return out
