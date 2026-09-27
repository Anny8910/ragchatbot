"""Index builder: manifest -> documents -> chunks -> embeddings -> Chroma.

This is where the corpus guard stops being advice and starts being a gate. The
loader logs a guard failure; this module refuses to write the index. A leak that
reaches the vector store is unrecoverable after the fact, because retrieval has
no way to tell a return figure from a fee, so the check is a hard failure by
design rather than a warning.

Pipeline (architecture §8):

    manifest -> load_all -> HARD GUARD -> chunk_all -> assert_fits EVERY chunk
             -> embed (disk-cached by content_hash) -> upsert
             -> delete orphans -> summary

Usage:
    python -m rag_bot.ingest.builder                  # incremental
    python -m rag_bot.ingest.builder --rebuild        # drop, then rebuild
    python -m rag_bot.ingest.builder --dry-run        # no writes, no embeddings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from rag_bot.config import Config, load, validate
from rag_bot.index.store import Store, embedding_text
from rag_bot.ingest.allowlist import (
    ALLOWED_FIELDS,
    FORBIDDEN_FIELDS,
    assert_allowlist_disjoint,
    guard,
)
from rag_bot.ingest.chunker import chunk_all
from rag_bot.ingest.loaders import extract_payload, load_from_manifest
from rag_bot.providers.fake import FakeEmbeddingProvider
from rag_bot.providers.minilm import MAX_WORD_PIECES, MiniLMProvider
from rag_bot.types import Chunk

log = logging.getLogger("rag_bot.builder")

# Bumping this invalidates the cache directory and the collection name.
CORPUS_VERSION = "v1"
EMBED_BATCH = 32


class GuardFailure(RuntimeError):
    """Raised when a document would put forbidden content into the index."""


# ---------------------------------------------------------------- cache


class EmbeddingCache:
    """One .npy per content_hash under <index_dir>/.embeddings/<model-slug>/.

    Keyed by the chunk's content_hash, so an unchanged corpus re-runs with zero
    embedding work. A changed model name gets a separate directory, which is
    what keeps a re-embed from mixing vectors from two models.
    """

    def __init__(self, root: str, embed_model: str):
        from rag_bot.index.ids import slugify

        self.dir = Path(root) / ".embeddings" / slugify(embed_model)
        self.hits = 0
        self.misses = 0

    def get(self, content_hash: str) -> list[float] | None:
        path = self._path(content_hash)
        if not path.exists():
            return None
        try:
            import numpy as np

            vec = np.load(path)
        except Exception:
            return None
        self.hits += 1
        return vec.tolist()

    def put(self, content_hash: str, vector: list[float]) -> None:
        import numpy as np

        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self._path(content_hash), np.asarray(vector, dtype="float32"))

    def _path(self, content_hash: str) -> Path:
        return self.dir / f"{content_hash}.npy"

    def prune(self, live_hashes: set[str]) -> int:
        if not self.dir.is_dir():
            return 0
        removed = 0
        for p in self.dir.glob("*.npy"):
            if p.stem not in live_hashes:
                p.unlink()
                removed += 1
        return removed


def _embed_key(chunk: Chunk) -> str:
    """Cache key for one chunk's vector, derived from the embedded text."""
    return hashlib.blake2b(
        embedding_text(chunk).encode("utf-8"), digest_size=16
    ).hexdigest()


def _get_provider(cfg: Config, *, offline: bool):
    if offline:
        log.warning("using the FAKE embedding provider; this index is NOT usable")
        return FakeEmbeddingProvider()
    return MiniLMProvider(cfg.embed_model)


# ---------------------------------------------------------------- guard


def _leaf_values(value, depth: int = 0):
    """Yield every scalar nested inside a payload value."""
    if depth > 4:
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaf_values(v, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _leaf_values(v, depth + 1)
    elif value is not None:
        yield value


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# A forbidden value only counts as evidence of a leak if it is distinctive
# enough that a match is unlikely to be coincidence.
#
# Thresholds are empirical, and both were forced by real payloads:
#   - Strings shorter than 12 chars are skipped. peerComparison and holdings are
#     lists of OTHER funds' records whose categorical fields contain the values
#     "High", "Medium" and "Open-ended". "High" occurs legitimately in
#     "Riskometer level: Moderately High", so a substring match flagged the
#     clean S5 snapshot as a leak. Short categorical words carry no evidence.
#   - Numerics shorter than 4 chars are skipped for the same reason: a rating of
#     "3" or a tenure of "1" appears in ordinary text by chance.
# Both classes are matched on word boundaries, so "1.2" cannot match inside
# "1.21" and "High" cannot match inside "Moderately High".
_MIN_STRING = 12
_MIN_NUMBER = 4


def _appears_in_text(value: str, text: str, allowed_values: set[str]) -> bool:
    if value in allowed_values:
        return False
    if _is_number(value):
        if len(value) < _MIN_NUMBER:
            return False
        return re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", text) is not None
    if len(value) < _MIN_STRING:
        return False
    return re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text) is not None


def hard_guard(doc, source_id: str) -> None:
    """Refuse the build if this document would put forbidden content in.

    Two layers, and the second one is the one that actually catches things:

    1. allowlist.guard() -> the forbidden PATTERNS never appear in the text.
    2. Value overlap. guard()'s own structural check compares forbidden field
       NAMES against rendered LABELS, which cannot intersect by construction
       ("nav" vs "NAV"), so it never fires. This layer instead looks for the
       forbidden fields' VALUES inside the rendered text.

    Values are excluded from the comparison when the same string is also carried
       by an allowlisted field, which is what keeps it false-positive-free:
       `category_info` repeats "Large Cap", and `category` legitimately renders
       "Large Cap". Blanked as one, and every scheme would fail its own build.
       Short numerics (< 4 chars) are skipped for the same reason -- a rating of
       "3" or a day count of "1" appears in text by coincidence.
    """
    text = doc.text
    result = guard(None, text, source_id=source_id)
    if not result.ok:
        raise GuardFailure(result.describe())

    payload = None
    if doc.source.snapshot_path:
        try:
            with open(doc.source.snapshot_path, encoding="utf-8") as fh:
                payload = extract_payload(fh.read())
        except OSError:
            payload = None
    if payload is None:
        return

    allowed_values: set[str] = set()
    for name in ALLOWED_FIELDS:
        if name in payload:
            allowed_values.update(str(v).strip() for v in _leaf_values(payload[name]))

    hits: list[str] = []
    for name in sorted(FORBIDDEN_FIELDS & set(payload)):
        for value in _leaf_values(payload[name]):
            s = str(value).strip()
            if s and _appears_in_text(s, text, allowed_values):
                hits.append(f"{name}={s[:60]}")
                break

    if hits:
        raise GuardFailure(
            f"{source_id}: forbidden values reached the text: {hits[:8]}"
        )


# ---------------------------------------------------------------- build


def _summarise(chunks: list[Chunk], docs, max_pieces: int, name: str) -> str:
    topics = Counter(c.topic for c in chunks)
    schemes = Counter(d.source.source_id for d in docs)
    lines = [
        f"collection     {name}",
        f"sources        {len(docs)} ({', '.join(f'{k}={v}' for k, v in sorted(schemes.items()))})",
        f"chunks         {len(chunks)}",
        f"max word-piece {max_pieces} / {MAX_WORD_PIECES} cap",
        "per-topic      "
        + "  ".join(f"{t}={topics.get(t, 0)}" for t in sorted(topics)),
    ]
    return "\n".join(lines)


def build(cfg: Config, *, rebuild: bool = False, dry_run: bool = False,
          offline: bool = False) -> int:
    assert_allowlist_disjoint()

    problems = validate(cfg)
    if problems:
        raise GuardFailure("invalid configuration: " + "; ".join(problems))

    docs = load_from_manifest(f"{cfg.corpus_dir}/manifest.csv")
    if not docs:
        log.error("manifest lists no sources; nothing to build")
        return 1

    for doc in docs:
        hard_guard(doc, doc.source.source_id)

    provider = _get_provider(cfg, offline=offline)
    if hasattr(provider, "assert_fits"):
        tokenizer = provider.count_tokens
    else:
        tokenizer = lambda t: len(t.split())

    chunks = chunk_all(
        docs,
        target_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap,
        tokenizer=tokenizer,
    )
    if not chunks:
        log.error("no chunks produced")
        return 1

    max_pieces = 0
    for c in chunks:
        # The cap applies to the EMBEDDED text, which is the prefixed string, not
        # to Chunk.text. Checking the clean text would undercount and let the
        # prefix push a chunk over the limit.
        n = provider.count_tokens(embedding_text(c))
        if n > max_pieces:
            max_pieces = n
        if n > MAX_WORD_PIECES:
            raise GuardFailure(
                f"{c.id}: {n} word pieces exceeds the {MAX_WORD_PIECES} cap; "
                f"refusing to index a chunk MiniLM would silently truncate"
            )
    # Belt and braces: the provider's own check, so the rule is not duplicated.
    if hasattr(provider, "assert_fits"):
        for c in chunks:
            provider.assert_fits(embedding_text(c))

    if dry_run:
        print(_summarise(chunks, docs, max_pieces, "(dry run, nothing written)"))
        print("guard          passed for all sources")
        return 0

    store = Store(cfg.index_dir, cfg.embed_model, CORPUS_VERSION)
    if rebuild:
        store.drop()
        store = Store(cfg.index_dir, cfg.embed_model, CORPUS_VERSION)

    cache = EmbeddingCache(cfg.index_dir, cfg.embed_model)
    vectors: list[list[float]] = []
    recomputed = 0
    # Keyed on the EMBEDDED text, not Chunk.content_hash. The content hash covers
    # the fact text only, so keying on it would serve the pre-prefix vectors for
    # ever after the prefix was introduced.
    keys = [_embed_key(c) for c in chunks]
    for key in keys:
        vectors.append(cache.get(key))
    todo = [i for i, v in enumerate(vectors) if v is None]
    recomputed = len(todo)

    if todo:
        fresh = provider.embed_documents(
            [embedding_text(chunks[i]) for i in todo], batch_size=EMBED_BATCH
        )
        for i, vec in zip(todo, fresh):
            vectors[i] = vec
            cache.put(keys[i], vec)

    store.upsert(chunks, vectors)

    live_sources = {d.source.source_id for d in docs}
    orphans = store.orphans(live_sources)
    deleted = store.delete_ids(orphans)
    cache.prune(set(keys))

    print(_summarise(chunks, docs, max_pieces, store.name))
    print(
        f"embeddings     {cache.hits} reused, {recomputed} recomputed "
        f"(cache: {cache.dir})"
    )
    print(
        f"orphans        {deleted} deleted, {store.count()} chunks in index, "
        f"collection version {CORPUS_VERSION}"
    )
    print("guard          passed for all sources")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Chroma index from the corpus.")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop the collection first, then rebuild from scratch")
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk and guard only; write nothing and embed nothing")
    ap.add_argument("--offline", action="store_true",
                    help="use the fake embedding provider (tests only)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    # HuggingFace reachability probes and Chroma telemetry are not news; without
    # this a --dry-run buries its own summary under 40 lines of httpx noise.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "chromadb", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load()
    try:
        return build(cfg, rebuild=args.rebuild, dry_run=args.dry_run,
                     offline=args.offline)
    except GuardFailure as exc:
        # Deliberately not log.error-and-continue: the point of the gate is that
        # a leaking corpus produces no index at all.
        log.error("build aborted: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
