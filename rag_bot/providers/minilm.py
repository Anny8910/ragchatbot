"""all-MiniLM-L6-v2 embedding provider.

The model is loaded LAZILY. Importing this module must not trigger a download,
so the offline test suite and the `--dry-run` path never pay for it.
"""
from __future__ import annotations

import os

# The 256 cap from the PRD. MiniLM silently truncates past this, which would
# make a chunk appear to "fit" while the tail was dropped from the embedding.
MAX_WORD_PIECES = 256

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMProvider:
    name = DEFAULT_MODEL

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self.name = model_name

    # -- lazy loading ---------------------------------------------------
    def _ensure(self):
        if self._model is not None:
            return self._model
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_dir,
            device="cpu",
        )
        self._tokenizer = self._model.tokenizer
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._ensure()
        return self._tokenizer

    # -- token counting -------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """True word-piece length, WITHOUT truncation.

        Using the truncating tokenizer here would defeat the entire point: an
        over-long chunk would report exactly MAX_WORD_PIECES and pass assert_fits.
        """
        tok = self.tokenizer
        return len(tok.encode(text, add_special_tokens=False, truncation=False))

    def assert_fits(self, text: str) -> None:
        n = self.count_tokens(text)
        if n > MAX_WORD_PIECES:
            raise ValueError(
                f"chunk is {n} word pieces, over the {MAX_WORD_PIECES} cap; "
                f"MiniLM would silently truncate it"
            )

    # -- embedding ------------------------------------------------------
    def embed_documents(self, texts: list[str], *, batch_size: int = 32):
        if not texts:
            return []
        model = self._ensure()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
