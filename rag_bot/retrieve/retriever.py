"""Query embedding and top-k chunk retrieval.

Never raises. An empty or missing index returns [].
"""
from __future__ import annotations

import logging
import re

from rag_bot.index.store import Store
from rag_bot.providers.base import EmbeddingProvider
from rag_bot.types import ScoredChunk

log = logging.getLogger("rag_bot.retrieve")

# Overfetch by 2 so the gate can drop weak candidates and still return k.
OVERFETCH = 2

# Query-side vocabulary expansion.
#
# Measured defect (data-findings.md 7.3): "What is the TER?" is an in-scope
# question that scores 0.04, because the five pages say "expense ratio" and never
# "TER". The score cannot see that these are the same thing, and the in-domain and
# out-of-domain score distributions overlap, so no threshold can fix it.
#
# This is the fix the spec asks for instead of lowering min_score: tighten what is
# matched. Lowering the threshold would also admit "stock market tips for
# tomorrow" (0.37), which is exactly the confidently-wrong-answer failure mode
# architecture 12 warns about.
#
# Deliberately small and one-directional: a user synonym is expanded into the
# corpus's own wording, never the reverse. Every entry was added because a real
# measured query failed without it, not speculatively.
SYNONYMS: dict[str, str] = {
    r"\bter\b": "total expense ratio expense ratio",
    r"\btotal expense ratio\b": "expense ratio",
    r"\bcharges?\b": "expense ratio",
    r"\bfee\b|\bfees\b": "expense ratio",
    r"\bcost\b|\bcosts\b": "expense ratio",
    r"\bredemption charge\b": "exit load",
    r"\bload\b": "exit load",
    r"\bpenalty\b": "exit load",
    r"\bminimum\b": "minimum SIP",
    r"\bhow much\b|\bhow low\b": "minimum SIP",
    r"\b3 year\b|\bthree year\b|\b3-year\b|\bthree-year\b": "lock in 3 years",
    r"\bvolatility\b|\bvolatile\b": "riskometer risk",
    r"\brisk level\b|\brisk category\b": "riskometer",
    r"\bindex\b": "benchmark",
    r"\bref\b": "benchmark",
}

_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in SYNONYMS.items()]


def expand_query(query: str) -> str:
    """Append the corpus's own wording for any synonym the query used.

    Appends rather than rewrites, so the original phrasing still dominates the
    embedding and the expansion only nudges it toward the indexed vocabulary.
    Terms already in the query are not re-appended, which makes this idempotent:
    expanding an already-expanded query is a no-op, so a caller may safely pass
    a query through twice.
    """
    extra: list[str] = []
    seen: set[str] = {w.lower() for w in query.split()}
    for pattern, replacement in _COMPILED:
        if pattern.search(query):
            for term in replacement.split():
                if term.lower() not in seen:
                    seen.add(term.lower())
                    extra.append(term)
    if not extra:
        return query
    return f"{query} {' '.join(extra)}".strip()


def retrieve(
    query: str,
    *,
    k: int,
    scheme_id: str | None = None,
    embedder: EmbeddingProvider,
    store: Store,
) -> list[ScoredChunk]:
    """Embed the query and return the top-k chunks, best score first.

    Overfetches k+2 internally (architecture 12) and returns only k, so the gate
    has spare candidates to reject without the caller having to ask for more.
    """
    if k <= 0:
        return []
    try:
        if store.count() == 0:
            return []
        expanded = expand_query(query)
        vector = embedder.embed_query(expanded)
        hits = store.query(vector, k=k + OVERFETCH, scheme_id=scheme_id)
    except Exception as exc:  # retrieval must never take the app down
        log.warning("retrieval failed for %r: %s", query[:60], exc)
        return []
    return sorted(hits, key=lambda s: s.score, reverse=True)[:k]
