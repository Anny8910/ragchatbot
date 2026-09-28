"""Sidebar: corpus status, a rebuild/reset pair, and the retrieval knobs.

The warning banner is the important part. A source whose manifest status is
`fetch_failed` or `blocked` is silently absent from the index, and the failure
mode is not an error message -- it is a confident answer about the wrong fund,
or a confident "not in sources" for something that IS covered. Naming the
missing scheme is the difference between a demo that is quietly wrong and one
that is visibly incomplete.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import streamlit as st

from rag_bot.config import Config

PROBLEM_STATUSES = {"fetch_failed", "blocked", "missing", "error"}


def read_manifest_statuses(manifest_path: str) -> dict[str, dict[str, str]]:
    """{source_id: {status, scheme_name, url}} from the build manifest."""
    target = Path(manifest_path)
    if not target.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with target.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("source_id") or "").strip()
            if sid:
                out[sid] = {
                    "status": (row.get("status") or "").strip(),
                    "scheme_name": (row.get("scheme_name") or "").strip(),
                    "url": (row.get("url") or "").strip(),
                    "fetched_at": (row.get("fetched_at") or "").strip(),
                }
    return out


def latest_fetch_date(statuses: dict[str, dict[str, str]]) -> str:
    """Newest snapshot date, or 'unknown'. Answers are as-of this date."""
    dates = sorted(
        info["fetched_at"][:10] for info in statuses.values() if info.get("fetched_at")
    )
    return dates[-1] if dates else "unknown"


def missing_sources(cfg: Config, statuses: dict[str, dict[str, str]]) -> list[str]:
    """Scheme names whose snapshot did not make it into the corpus."""
    return [
        f"{info['scheme_name'] or sid} ({info['status']})"
        for sid, info in sorted(statuses.items())
        if info["status"].lower() in PROBLEM_STATUSES
    ]


def _run_builder(*args: str) -> tuple[bool, str]:
    """Run the index builder as a subprocess so its stdout is not swallowed."""
    cmd = [sys.executable, "-m", "rag_bot.ingest.builder", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]


def render_sidebar(cfg: Config, chunk_count: int) -> tuple[int, float, str]:
    """Draw the sidebar. Returns (k, min_score, corpus_status_line)."""
    with st.sidebar:
        st.header("Corpus")

        statuses = read_manifest_statuses(str(Path(cfg.corpus_dir) / "manifest.csv"))
        files = sum(1 for _ in Path(cfg.corpus_dir).joinpath("snapshots").glob("*.html"))
        fetched = latest_fetch_date(statuses)
        st.caption(
            f"{len(statuses)} sources · {files} snapshots · {chunk_count} chunks "
            f"indexed · pages fetched {fetched}"
        )

        missing = missing_sources(cfg, statuses)
        if missing:
            st.error(
                "**Incomplete corpus.** These sources were not indexed, so answers "
                "about them will be wrong or will claim 'not in sources':\n\n"
                + "\n".join(f"- {name}" for name in missing)
            )
        else:
            st.success("All sources indexed.")

        st.header("Retrieval")
        k = st.slider(
            "Chunks (k)", min_value=1, max_value=8, value=int(cfg.top_k),
            help="How many retrieved chunks the model is shown.",
        )
        min_score = st.slider(
            "Minimum score", min_value=0.0, max_value=1.0,
            value=float(cfg.min_score), step=0.01,
            help="Below this cosine similarity the answer is refused as class B.",
        )

        st.header("Maintenance")
        if st.button("Rebuild index", use_container_width=True):
            with st.spinner("Rebuilding index from snapshots..."):
                ok, output = _run_builder("--rebuild")
            (st.success if ok else st.error)(
                "Index rebuilt." if ok else "Rebuild failed."
            )
            st.text_area("builder output", output, height=180, disabled=True)
            st.cache_resource.clear()

        if st.button("Reset conversation", use_container_width=True):
            for key in ("messages", "pending_question"):
                st.session_state.pop(key, None)
            st.rerun()

    status_line = (
        f"{len(statuses)} sources · {files} snapshots · {chunk_count} chunks · "
        f"fetched {fetched}"
    )
    return k, min_score, status_line
