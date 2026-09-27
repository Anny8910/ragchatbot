"""Load a snapshot into uniform, structured text.

The output text format is a CONTRACT with the chunker:

  - headings on their own line as `## Heading`
  - facts as `Label: value` lines, consecutive, so a table flattened this way
    stays answerable (architecture §9.2)
  - paragraphs separated by a blank line

Extraction is ALLOWLIST-based: `load_source` reads only ALLOWED_FIELDS from the
page's embedded scheme payload. It does not scrape and scrub. See
rag_bot/ingest/allowlist.py for the evidence that motivated this, and
docs/data-findings.md section 4.

Never raises. A source that cannot be loaded yields a Document with empty text
and a logged reason, so one bad snapshot cannot stop a build.
"""
from __future__ import annotations

import html as htmllib
import json
import logging
import re
import sys
from typing import Any

from rag_bot.ingest.allowlist import (
    guard,
    render_document_text,
    selected_fields,
)
from rag_bot.types import Document, Source

log = logging.getLogger(__name__)

_NEXT_DATA_RE = re.compile(
    r'(?s)<script id="__NEXT_DATA__"[^>]*>(.*?)</script>'
)
_MF_PAYLOAD_PATH = ("props", "pageProps", "mfServerSideData")

# Fallback path only. Strip these before taking text.
_DROP_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "svg", "iframe", "template",
)
_DROP_RE = re.compile(
    r"(?is)<(%s)\b.*?</\1>" % "|".join(_DROP_TAGS)
)
_HEADING_RE = re.compile(r"(?is)<h([1-4])\b[^>]*>(.*?)</h\1>")
_PARA_RE = re.compile(r"(?is)<p\b[^>]*>(.*?)</p>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _dig(obj: Any, path: tuple[str, ...]) -> Any | None:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def extract_payload(html: str) -> dict[str, Any] | None:
    """Pull the scheme payload out of a snapshot's __NEXT_DATA__ blob.

    Returns None when the blob is absent, unparseable, or has an unexpected
    shape. Never raises.
    """
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        data = json.loads(htmllib.unescape(m.group(1)))
    except (ValueError, TypeError) as exc:
        log.warning("__NEXT_DATA__ is not valid JSON: %s", exc)
        return None
    payload = _dig(data, _MF_PAYLOAD_PATH)
    if not isinstance(payload, dict):
        return None
    return payload


def _fallback_text(html: str) -> str:
    """Best-effort text when the embedded payload is unusable.

    Tables are DELIBERATELY not extracted. Of the four server-rendered tables on
    these pages, three are performance data (SIP returns, category rank, peer
    returns) and none contains fees, lock-in, benchmark, or riskometer. Scraping
    tables here would import exactly what the corpus must not contain, in the
    one code path that has no allowlist discipline applied to it.
    """
    body = _DROP_RE.sub(" ", html)
    out: list[str] = []
    for match in _HEADING_RE.finditer(body):
        text = _WS_RE.sub(" ", _TAG_RE.sub(" ", match.group(2))).strip()
        if text:
            out.append(f"## {text}")
    for match in _PARA_RE.finditer(body):
        text = _WS_RE.sub(" ", _TAG_RE.sub(" ", match.group(1))).strip()
        if len(text) > 40:
            out.append(text)
    return "\n\n".join(out) + "\n" if out else ""


def load_source(src: Source) -> Document:
    """Read src.snapshot_path and return a Document of structured text.

    - Extracts the allowlisted fields from the embedded scheme payload.
    - Falls back to a headings-and-paragraphs scrape if the payload is missing
      or malformed, and logs why. Tables are not scraped; see _fallback_text.
    - Returns a Document with empty text (and a logged reason) on any failure.
      Never raises.
    """
    if not src.snapshot_path:
        return _empty(src, "no snapshot_path recorded in the manifest")

    try:
        with open(src.snapshot_path, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        return _empty(src, f"cannot read snapshot: {exc}")

    payload = extract_payload(raw)
    if payload is None:
        log.warning(
            "%s: no usable mfServerSideData payload; using fallback text",
            src.source_id,
        )
        text = _fallback_text(raw)
        if not text.strip():
            return _empty(src, "no embedded payload and fallback text was empty")
        return Document(text=text, source=src, heading=None)

    try:
        text = render_document_text(payload)
    except Exception as exc:  # defensive: a bad field must not kill the build
        log.warning("%s: allowlist render failed: %s", src.source_id, exc)
        return _empty(src, f"allowlist render failed: {exc}")

    result = guard(payload, text, source_id=src.source_id)
    if not result.ok:
        # Not fatal here. P3's builder hard-fails on a non-ok guard, which is
        # the right place to stop; this log makes the problem visible early.
        log.error("corpus guard: %s", result.describe())

    fields = selected_fields(payload)
    log.info(
        "%s: extracted %d allowlisted fields -> %d chars",
        src.source_id,
        len(fields),
        len(text),
    )
    return Document(text=text, source=src, heading=None)


def _empty(src: Source, reason: str) -> Document:
    print(f"  {src.source_id}: {reason}", file=sys.stderr)
    log.warning("%s: %s", src.source_id, reason)
    return Document(text="", source=src, heading=None)


def load_all(sources: list[Source]) -> list[Document]:
    """Load every source. Never raises. Order follows the input."""
    return [load_source(src) for src in sources]


def load_from_manifest(manifest_path: str) -> list[Document]:
    """Convenience: read the manifest and load every source it lists."""
    from rag_bot.sources.manifest import read_manifest

    return load_all(read_manifest(manifest_path))
