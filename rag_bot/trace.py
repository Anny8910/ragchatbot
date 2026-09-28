"""One JSONL line per query, so a run can be inspected after the fact.

The single rule that matters here: **the question is redacted before it is
written.** `data/logs/` is gitignored today, but a log file is the easiest way
to put a PAN into a public repo -- someone flips the ignore rule to debug a
demo, or copies a line into an issue. So redaction happens on the way in, not
as a later cleanup pass.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from rag_bot.answer.validate import find_return_figures, split_sentences
from rag_bot.safety.pii import scrub
from rag_bot.types import Answer

log = logging.getLogger(__name__)

LOG_PATH = "data/logs/run.jsonl"

_URL = re.compile(r"https?://[^\s)\]>\"']+")


def _url_count(text: str) -> int:
    return len(set(_URL.findall(text)))


def build_record(answer: Answer, question: str) -> dict:
    """The dict that becomes one JSONL line. Pure -- no I/O, so it is testable.

    `question` is the REDACTED string, never the raw one. It is the most useful
    field for debugging and also the most dangerous, so it is written only after
    `scrub`, and the `pii` field records which rules fired so a reader can tell
    a clean question from a mangled one at a glance.

    Uses `scrub` rather than `scrub_for_log` because the `pii` field has to
    report WHICH rules fired, and `scrub_for_log` throws that away. Same regexes,
    same output string; `scrub` just also returns `rules_fired`.
    """
    scrubbed = scrub(question or "")
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": scrubbed.text,
        "outcome": answer.outcome.value,
        "scheme": answer.scheme_name,
        "top_score": round(answer.top_score, 4) if answer.top_score is not None else None,
        "k": answer.k,
        "cited_url": answer.source_url,
        "url_count": _url_count(answer.text or ""),
        "sentences": len(split_sentences(answer.text or "")),
        "figures": find_return_figures(answer.text or ""),
        "pii": "clean" if scrubbed.clean else f"redacted:{','.join(scrubbed.rules_fired)}",
        "triage_layer": answer.triage_layer,
        "latency": {stage: round(ms, 1) for stage, ms in (answer.latency or {}).items()},
    }


def log_query(answer: Answer, question: str, *, path: str = LOG_PATH) -> None:
    """Append one JSONL line to data/logs/run.jsonl. Never raises.

    A trace logger that can take down the app is worse than no trace logger: the
    answer was already correct, and losing it because the disk is full helps
    nobody.
    """
    try:
        record = build_record(answer, question)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        log.warning("trace log write failed (%s); the answer is unaffected", exc)


def read_records(path: str = LOG_PATH) -> list[dict]:
    """Parse the log back. Used by the UI's trace panel and by tests."""
    target = Path(path)
    if not target.exists():
        return []
    out: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
