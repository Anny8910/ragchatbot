"""Fetch and snapshot the registered sources.

Deliverable D2 support: writes data/corpus/snapshots/{source_id}.html and, via
manifest.py, data/corpus/manifest.csv.

No crawling, no link following, no retries beyond one, never raises.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import requests
import yaml

from rag_bot.config import load
from rag_bot.sources.manifest import write_manifest
from rag_bot.types import Source

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Groww pages are 450-500KB. A 30s timeout was observed aborting at 112KB of a
# 453KB page, so the default is deliberately generous. See data-findings.md §6.
DEFAULT_TIMEOUT = 120

# Sequential fetching with a pause, so we are a polite client and do not trip
# any per-IP rate limiting across five requests.
FETCH_SLEEP_SECONDS = 3.0

MIN_VISIBLE_TEXT = 200

_TAG_RE = re.compile(r"(?s)<(script|style|noscript)\b.*?</\1>")
_ANY_TAG_RE = re.compile(r"(?s)<[^>]+>")


def load_sources(path: str) -> list[Source]:
    """Read sources.yaml into Source objects. Raises on a malformed file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("sources") or []
    out: list[Source] = []
    for e in entries:
        out.append(
            Source(
                source_id=e["source_id"],
                url=e["url"],
                publisher=e["publisher"],
                source_tier=e["source_tier"],
                scheme_id=e.get("scheme_id"),
                scheme_name=e["scheme_name"],
                factsheet_url=e.get("factsheet_url"),
                aliases=tuple(e.get("aliases") or ()),
            )
        )
    return out


def snapshot_path_for(source_id: str, corpus_dir: str) -> Path:
    return Path(corpus_dir) / "snapshots" / f"{source_id}.html"


def _visible_text(html: str) -> str:
    """Rough visible-text length, used only for the JS-shell liveness check."""
    stripped = _TAG_RE.sub(" ", html)
    stripped = _ANY_TAG_RE.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _fetch_record(
    src: Source, *, refresh: bool, timeout: int, corpus_dir: str
) -> tuple[str | None, str, str | None, str | None]:
    """Do the work. Returns (html_or_None, status, fetched_at, content_hash).

    fetched_at and content_hash are preserved verbatim when an existing snapshot
    is reused, so re-running the fetcher does not churn the manifest.
    """
    dest = snapshot_path_for(src.source_id, corpus_dir)

    if dest.exists() and not refresh:
        existing = dest.read_text(encoding="utf-8")
        prior = _sidecar(src.source_id, corpus_dir)
        return existing, "ok", (prior or {}).get("fetched_at"), (
            hashlib.sha256(existing.encode("utf-8")).hexdigest()
        )

    try:
        response = requests.get(
            src.url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        print(f"  {src.source_id}: request failed: {exc}", file=sys.stderr)
        return None, "fetch_failed", None, None

    # An explicit status check is mandatory. The P1 spec's "JS shell under 200
    # visible chars" heuristic does NOT catch a dead URL: Groww's 404 page is
    # 38,950 bytes of perfectly readable text. Only the status code catches it.
    if response.status_code != 200:
        print(
            f"  {src.source_id}: HTTP {response.status_code} for {src.url}",
            file=sys.stderr,
        )
        return None, "blocked", None, None

    html = response.text

    if len(_visible_text(html)) < MIN_VISIBLE_TEXT:
        return None, "blocked", None, None

    # Liveness probe, not parsing: a genuine scheme page always embeds its
    # scheme data. A soft-404, login wall, or consent interstitial does not.
    # P2 does the actual JSON extraction; this only rejects non-scheme pages.
    if "__NEXT_DATA__" not in html:
        print(
            f"  {src.source_id}: no __NEXT_DATA__ payload; treating as blocked",
            file=sys.stderr,
        )
        return None, "blocked", None, None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")

    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    _write_sidecar(src.source_id, corpus_dir, fetched_at, digest)
    return html, "ok", fetched_at, digest


def _sidecar_path(source_id: str, corpus_dir: str) -> Path:
    return Path(corpus_dir) / "snapshots" / f"{source_id}.meta.json"


def _sidecar(source_id: str, corpus_dir: str) -> dict | None:
    p = _sidecar_path(source_id, corpus_dir)
    if not p.exists():
        return None
    try:
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_sidecar(
    source_id: str, corpus_dir: str, fetched_at: str, content_hash: str
) -> None:
    """Record fetch provenance beside the snapshot, so an idempotent re-run does
    not reset fetched_at. This is a build artifact, not a corpus document."""
    import json

    p = _sidecar_path(source_id, corpus_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"fetched_at": fetched_at, "content_hash": content_hash}, indent=2
        ),
        encoding="utf-8",
    )


def fetch_source(
    src: Source, *, refresh: bool = False, timeout: int = DEFAULT_TIMEOUT
) -> tuple[str | None, str]:
    """Return (html_text_or_None, status). status in {ok, fetch_failed, blocked}."""
    corpus_dir = load().corpus_dir
    html, status, _fetched_at, _digest = _fetch_record(
        src, refresh=refresh, timeout=timeout, corpus_dir=corpus_dir
    )
    return html, status


def fetch_all(path: str, *, refresh: bool = False) -> list[Source]:
    """Fetch every source; never raise. Returns updated Source objects."""
    cfg = load()
    try:
        sources = load_sources(path)
    except Exception as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return []

    updated: list[Source] = []
    for i, src in enumerate(sources):
        if i:
            time.sleep(FETCH_SLEEP_SECONDS)
        try:
            _html, status, fetched_at, digest = _fetch_record(
                src,
                refresh=refresh,
                timeout=DEFAULT_TIMEOUT,
                corpus_dir=cfg.corpus_dir,
            )
        except Exception as exc:  # defensive: one bad source must not stop the run
            print(f"  {src.source_id}: unexpected error: {exc}", file=sys.stderr)
            status, fetched_at, digest = "fetch_failed", None, None

        updated.append(
            replace(
                src,
                status=status,
                fetched_at=fetched_at,
                snapshot_path=(
                    str(snapshot_path_for(src.source_id, cfg.corpus_dir))
                    if status == "ok"
                    else None
                ),
                content_hash=digest,
            )
        )
        print(f"{src.scheme_id} {src.source_id}: {status} {src.url}")

    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and snapshot sources.")
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch even if a snapshot exists"
    )
    parser.add_argument(
        "--sources", default=None, help="path to sources.yaml (default from config)"
    )
    args = parser.parse_args(argv)

    cfg = load()
    path = args.sources or cfg.sources_file

    sources = fetch_all(path, refresh=args.refresh)
    if not sources:
        print("no sources fetched", file=sys.stderr)
        return 1

    out = Path(cfg.corpus_dir) / "manifest.csv"
    write_manifest(sources, str(out))

    blocked = [s for s in sources if s.status != "ok"]
    print(f"\nwrote {out}")
    print(f"ok={len(sources) - len(blocked)} not_ok={len(blocked)}")
    if blocked:
        for s in blocked:
            print(f"  BLOCKED {s.scheme_id} {s.source_id}: {s.url}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
