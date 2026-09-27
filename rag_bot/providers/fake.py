"""Deterministic fake embedding provider.

*** THIS IS NOT SEMANTIC. *** Tokens are hashed with a fixed seed and projected
onto 384 dimensions, so two texts sharing vocabulary land near each other and
unrelated texts do not. It is not a sentence embedding and must never be used to
answer a real question.

It exists so that the test suite, the `--dry-run` path, and CI all run with no
model download and no network. Any retrieval quality measured against this
provider is meaningless; only plumbing is under test.
"""
from __future__ import annotations

import hashlib
import math

DIM = 384
_SEED = b"rag_bot_fake_embedder_v1"


def _token_vector(token: str) -> list[float]:
    digest = hashlib.blake2b(
        _SEED + token.encode("utf-8"), digest_size=48
    ).digest()
    return [(b / 255.0) - 0.5 for b in digest[:DIM // 3]]


class FakeEmbeddingProvider:
    name = "fake-hashed-384"

    def __init__(self, dim: int = DIM):
        self._dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = text.lower().split() or [text.lower()]
        for token in tokens:
            tv = _token_vector(token)
            for i, v in enumerate(tv):
                vec[i % self._dim] += v
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def count_tokens(self, text: str) -> int:
        """Whitespace count, so tests need no tokenizer. Not word pieces."""
        return len(text.split())

    def assert_fits(self, text: str) -> None:
        from rag_bot.providers.minilm import MAX_WORD_PIECES

        n = self.count_tokens(text)
        if n > MAX_WORD_PIECES:
            raise ValueError(f"chunk is {n} units, over the {MAX_WORD_PIECES} cap")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
