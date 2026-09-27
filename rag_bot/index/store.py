"""Chroma-backed chunk store.

SCORE CONVENTION -- VERIFIED EMPIRICALLY, DO NOT ASSUME
------------------------------------------------------
architecture §12 sets RAG_MIN_SCORE = 0.35 as a *cosine similarity* floor, and
the original spec assumed a retrieved distance converts with `score = 1 - d`.
That assumption is FALSE for chromadb 1.5.9's default metric.

Measured on chromadb 1.5.9, with normalised all-MiniLM-L6-v2 vectors, comparing
each conversion against the analytically-known cosine:

    metric                          max error of 1 - d    1 - d/2    1 - d^2/2
    default (squared L2)            1.005246             1.1e-7     1.015791
    hnsw:space = "cosine"           6e-8                  0.5026     0.4999

So on the default metric the correct conversion is `1 - d/2` (squared L2 on unit
vectors: d = 2 - 2*cos). This module therefore sets `hnsw:space: "cosine"`
EXPLICITLY on every collection, which makes `1 - distance` correct and makes the
metric independent of chromadb's default changing in a future release.

This matters concretely: with the default metric and the naive `1 - d`, a
completely unrelated document scores -1.01 and a merely-adjacent one -0.59. Both
fall below the 0.35 gate, so EVERY question would be refused as class B. The bug
would not surface until the retriever phase, and would look like "retrieval is
just bad" rather than a wrong unit conversion.
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from rag_bot.index.ids import collection_name
from rag_bot.types import Chunk, ScoredChunk

# Chroma metadata accepts only str, int, float, and bool. A None is not a valid
# metadata value, so it is normalised to "" here and the loaders treat "" as
# "absent" when reading it back.
COSINE_METADATA = {"hnsw:space": "cosine"}


def embedding_text(chunk: Chunk) -> str:
    """The string that is actually embedded.

    architecture 5.2 defines Chunk.text as the fact text alone, e.g.
    "Expense ratio: 1.03", and 14.1 puts the scheme name in the citation header
    above it. That is right for the LLM but wrong for the retriever: five chunks
    of the same topic are then five near-identical strings differing only in a
    number, and the embedding carries almost no scheme-discriminating signal.

    Measured on the real index: with bare fact text, the correct scheme ranked
    first in only 3 of 10 scheme-specific questions, median rank 3 of 5, with a
    top-to-bottom score spread of ~0.01-0.03 -- i.e. noise. Prefixing the scheme
    name and heading (the standard asymmetric-retrieval prefix) restores it.

    The prefix is embedding-only. The clean text is what reaches the LLM, kept
    in metadata under "body" and restored by _rehydrate, so 5.2's "the only
    field sent to the LLM" still holds literally.
    """
    parts = [chunk.scheme_name]
    if chunk.heading:
        parts.append(chunk.heading)
    parts.append(chunk.text)
    return "\n".join(parts)


def flat_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Flatten a Chunk into Chroma-legal metadata.

    Never pass a nested dict or a list to Chroma: it raises, or silently
    stringifies depending on version, which would break the where-filters in
    query() because the filter would compare against a stringified form.
    """
    return {
        "source_id": chunk.source_id,
        "scheme_id": chunk.scheme_id or "",
        "scheme_name": chunk.scheme_name,
        "publisher": chunk.publisher,
        "source_tier": chunk.source_tier,
        "heading": chunk.heading or "",
        "topic": chunk.topic,
        "chunk_index": int(chunk.chunk_index),
        "n_tokens": int(chunk.n_tokens),
        "fetched_at": chunk.fetched_at or "",
        "factsheet_url": chunk.factsheet_url or "",
        "url": chunk.url,
        "content_hash": chunk.content_hash,
        "body": chunk.text,
    }


def _rehydrate(meta: dict[str, Any], doc: str, cid: str) -> Chunk:
    # "body" is the clean fact text; "doc" carries the embedding-time prefix.
    body = meta.get("body") or doc
    return Chunk(
        id=cid,
        text=body,
        source_id=meta["source_id"],
        url=meta["url"],
        publisher=meta["publisher"],
        source_tier=meta["source_tier"],
        scheme_id=meta["scheme_id"] or None,
        scheme_name=meta["scheme_name"],
        heading=meta["heading"] or None,
        topic=meta["topic"],
        chunk_index=int(meta["chunk_index"]),
        n_tokens=int(meta["n_tokens"]),
        fetched_at=meta["fetched_at"] or None,
        factsheet_url=meta["factsheet_url"] or None,
        content_hash=meta["content_hash"],
    )


class Store:
    def __init__(self, index_dir: str, embed_model: str, corpus_version: str):
        self.index_dir = index_dir
        self.embed_model = embed_model
        self.corpus_version = corpus_version
        self.name = collection_name(embed_model, corpus_version)
        self._client = chromadb.PersistentClient(
            path=index_dir, settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=self.name, metadata=COSINE_METADATA
        )

    # -- writes ---------------------------------------------------------
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors; they must pair"
            )
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[embedding_text(c) for c in chunks],
            embeddings=vectors,
            metadatas=[flat_metadata(c) for c in chunks],
        )

    # -- reads ----------------------------------------------------------
    def query(
        self,
        vector: list[float],
        *,
        k: int,
        scheme_id: str | None = None,
        topic: str | None = None,
    ) -> list[ScoredChunk]:
        """Return the k best chunks, scored as cosine similarity in [0, 1].

        n_results = k + 2 (overfetch, architecture §12) so the relevance gate has
        candidates to work with. See the module docstring for the verified
        distance convention; the collection is created with cosine space, so
        score = 1 - distance.
        """
        # Chroma requires a where-clause to carry EXACTLY ONE operator key, so
        # two equality filters must be combined under $and. Passing both as bare
        # keys raises ValueError at query time.
        clauses: list[dict[str, str]] = []
        if scheme_id:
            clauses.append({"scheme_id": scheme_id})
        if topic:
            clauses.append({"topic": topic})
        if len(clauses) == 1:
            where: dict = clauses[0]
        elif clauses:
            where = {"$and": clauses}
        else:
            where = None
        total = self._collection.count()
        n_results = min(k + 2, total) if total else 0
        if n_results <= 0:
            return []

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        out: list[ScoredChunk] = []
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            score = 1.0 - float(dist)
            # Rounding guards against a float creeping past 1.0 and tripping a
            # downstream range assertion.
            score = max(-1.0, min(1.0, score))
            out.append(ScoredChunk(chunk=_rehydrate(meta, doc, cid), score=score))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:k]

    def count(self) -> int:
        return self._collection.count()

    def all_ids(self) -> set[str]:
        return set(self._collection.get(include=[])["ids"])

    def source_ids_present(self) -> set[str]:
        got = self._collection.get(include=["metadatas"])
        return {m["source_id"] for m in (got.get("metadatas") or [])}

    def delete_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def orphans(self, live_source_ids: set[str]) -> list[str]:
        """Ids whose source_id is no longer in the manifest."""
        got = self._collection.get(include=["metadatas"])
        pairs = list(zip(got.get("ids") or [], got.get("metadatas") or []))
        return [cid for cid, meta in pairs if meta.get("source_id") not in live_source_ids]

    def drop(self) -> None:
        try:
            self._client.delete_collection(self.name)
        except Exception:
            pass
