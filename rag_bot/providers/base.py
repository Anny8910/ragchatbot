"""Provider protocols. Members only, no implementations."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int:
        """Word pieces, for the 256 cap. Must NOT truncate."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def generate(
        self, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> str: ...

    def classify(self, system: str, user: str, *, labels: list[str]) -> str: ...
