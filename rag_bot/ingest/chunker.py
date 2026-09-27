"""Section-aware, table-aware chunking (architecture §9.2).

The tokenizer is INJECTED, never imported here, so this module is testable
without downloading an embedding model. Tests pass a whitespace stub; P3 supplies
the real word-piece tokenizer.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

from rag_bot.types import Chunk, Document, Topic

# Hard cap from the PRD: MiniLM truncates past 256 word pieces. The builder
# asserts every chunk fits, but the chunker also splits defensively.
MAX_WORD_PIECES = 256

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$")
_LABEL_RE = re.compile(r"^[A-Z][^:]{2,60}:\s*\S")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Keyword rules for per-chunk topic assignment. First match wins, so order is
# significant. `statement` is absent: that topic was dropped from scope, since no
# source in the corpus mentions statements. See docs/data-findings.md section 5.
TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("expense_ratio", ("expense ratio", "base expense ratio")),
    ("exit_load", ("exit load", "exit-load", "redemption charge")),
    ("min_sip", ("minimum sip", "sip amount", "lump-sum", "minimum withdrawal")),
    ("lock_in", ("lock-in", "lock in", "lockin")),
    ("riskometer", ("riskometer", "risk level")),
    ("benchmark", ("benchmark", "nifty", "bse", "tri", "total return index")),
)

# Keywords that must not decide the topic on their own: they are far too common
# inside identity text ("HDFC Large Cap Fund Direct Growth" contains "Growth").
_TOPIC_BLOCKLIST = ("direct growth", "growth plan", "capital appreciation")


def _classify_topic(heading: str | None, text: str) -> Topic:
    haystack = f"{heading or ''}\n{text}".lower()
    for label in _TOPIC_BLOCKLIST:
        # strip identity noise before matching
        haystack = haystack.replace(label, " ")
    for topic, keywords in TOPIC_RULES:
        if any(k in haystack for k in keywords):
            return topic  # type: ignore[return-value]
    return "other"


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split at `## `/`### ` lines. A chunk never crosses a boundary."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    out: list[tuple[str | None, str]] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip("\n")
        if heading is None and not body.strip():
            continue  # preamble before the first heading
        if not body.strip():
            continue
        out.append((heading, body))
    return out


def _build_units(body: str) -> list[str]:
    """Split a section into units, in descending priority.

    a) runs of consecutive `Label: value` lines (a flattened table row-group)
    b) blank-line-separated paragraphs
    c) sentences
    d) words
    """
    units: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            units.append("\n".join(run))
            run.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if _LABEL_RE.match(stripped):
            run.append(stripped)
        else:
            flush()
            if stripped:
                units.append(stripped)
    flush()

    if not units:
        return []

    out: list[str] = []
    for unit in units:
        if "\n" in unit or _LABEL_RE.match(unit):
            out.append(unit)
        elif len(unit.split()) > 40:
            out.extend(s for s in _SENTENCE_RE.split(unit) if s.strip())
        else:
            out.append(unit)
    return out


def _hard_split(unit: str, count: Callable[[str], int], max_tokens: int) -> list[str]:
    """Split an oversized unit on word boundaries. Always makes progress."""
    if count(unit) <= max_tokens:
        return [unit]
    words = unit.split()
    if not words:
        return [unit]
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count(" ".join(current)) >= max_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces or [unit]


def chunk_document(
    doc: Document,
    *,
    target_tokens: int,
    overlap_tokens: int,
    tokenizer: Callable[[str], int],
) -> list[Chunk]:
    """Section-aware, table-aware chunking.

    1. Split at `## `/`### `; never chunk across a section boundary.
    2. Within a section build units (table row-groups, then paragraphs, then
       sentences, then words).
    3. Accumulate until the next unit would exceed target_tokens; emit; start the
       next chunk with the previous chunk's last overlap_tokens worth of units.
    4. A table row-group keeps its header line with its rows.
    5. Forward-progress guarantee: an oversized unit is hard-split, and the loop
       ALWAYS advances.
    6. Anything over 256 word-pieces is hard-split.
    7. topic is assigned from heading + keyword rules, first match wins.
    8. id = f"{source_id}::{ordinal}", ordinal from 0 per document.
    9. content_hash = sha256 of the chunk text.
    """
    if not doc.text or not doc.text.strip():
        return []

    max_tokens = min(target_tokens, MAX_WORD_PIECES)
    overlap = max(0, min(overlap_tokens, max_tokens - 1))
    chunks: list[Chunk] = []
    ordinal = 0
    src = doc.source

    for heading, body in _split_sections(doc.text):
        units = _build_units(body)
        if not units:
            continue

        expanded: list[str] = []
        for unit in units:
            expanded.extend(_hard_split(unit, tokenizer, max_tokens))

        buffer: list[str] = []
        for unit in expanded:
            candidate = "\n".join(buffer + [unit])
            if buffer and tokenizer(candidate) > max_tokens:
                chunks.append(
                    _make_chunk(
                        doc, heading, buffer, ordinal, tokenizer
                    )
                )
                ordinal += 1
                # carry overlap from the tail of what we just emitted
                carry: list[str] = []
                budget = overlap
                for prev in reversed(buffer):
                    size = tokenizer(prev)
                    if budget - size < 0:
                        break
                    carry.insert(0, prev)
                    budget -= size
                buffer = carry
            buffer.append(unit)

        if buffer:
            chunks.append(
                _make_chunk(doc, heading, buffer, ordinal, tokenizer)
            )
            ordinal += 1

    return chunks


def _make_chunk(
    doc: Document,
    heading: str | None,
    buffer: list[str],
    ordinal: int,
    tokenizer: Callable[[str], int],
) -> Chunk:
    text = "\n".join(buffer)
    topic = _classify_topic(heading, text)
    n_tokens = tokenizer(text)
    return Chunk(
        id=f"{doc.source.source_id}::{ordinal}",
        text=text,
        source_id=doc.source.source_id,
        url=doc.source.url,
        publisher=doc.source.publisher,
        source_tier=doc.source.source_tier,
        scheme_id=doc.source.scheme_id,
        scheme_name=doc.source.scheme_name,
        heading=heading,
        topic=topic,
        chunk_index=ordinal,
        n_tokens=n_tokens,
        fetched_at=doc.source.fetched_at,
        factsheet_url=doc.source.factsheet_url,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def chunk_all(docs: list[Document], **kw) -> list[Chunk]:
    """Chunk every document. Chunk ids stay unique because source_ids differ."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kw))
    return out
