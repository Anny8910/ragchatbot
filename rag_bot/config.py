"""Configuration: the single source of truth for env vars (architecture §15)."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULTS: dict[str, str] = {
    "RAG_PROVIDER": "ollama",
    "RAG_LLM_MODEL": "llama3.5:8b",
    "RAG_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RAG_OLLAMA_HOST": "http://localhost:11434",
    "RAG_TEMPERATURE": "0.1",
    "RAG_MAX_SENTENCES": "3",
    "RAG_TOP_K": "3",
    "RAG_MIN_SCORE": "0.35",
    "RAG_CHUNK_TOKENS": "180",
    "RAG_CHUNK_OVERLAP": "40",
    "RAG_CORPUS_DIR": "data/corpus",
    "RAG_INDEX_DIR": "data/index",
    "RAG_SOURCES_FILE": "rag_bot/sources/sources.yaml",
    "RAG_TRIAGE_LLM": "true",
    "RAG_PII_REDACT": "true",
    "RAG_SANITIZE_PERF": "true",
    "RAG_HISTORY_TURNS": "0",
    "RAG_LOG_TRACE": "true",
}

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Config:
    provider: str
    llm_model: str
    embed_model: str
    ollama_host: str
    temperature: float
    max_sentences: int
    top_k: int
    min_score: float
    chunk_tokens: int
    chunk_overlap: int
    corpus_dir: str
    index_dir: str
    sources_file: str
    triage_llm: bool
    pii_redact: bool
    sanitize_perf: bool
    history_turns: int
    log_trace: bool


def _as_bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return True


def load() -> Config:
    """Read DEFAULTS overlaid with os.environ, coerce types, no validation.

    A .env file is read via python-dotenv when present, but is never required:
    the default build must run with no .env at all (architecture D3). Real
    environment variables win over .env values, so a one-off override on the
    command line still behaves as expected.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        # python-dotenv missing or the .env unparseable: fall back to DEFAULTS
        # plus whatever is already in the environment. Never raise.
        pass

    def raw(key: str) -> str:
        return os.environ.get(key, DEFAULTS[key])

    return Config(
        provider=raw("RAG_PROVIDER"),
        llm_model=raw("RAG_LLM_MODEL"),
        embed_model=raw("RAG_EMBED_MODEL"),
        ollama_host=raw("RAG_OLLAMA_HOST"),
        temperature=float(raw("RAG_TEMPERATURE")),
        max_sentences=int(raw("RAG_MAX_SENTENCES")),
        top_k=int(raw("RAG_TOP_K")),
        min_score=float(raw("RAG_MIN_SCORE")),
        chunk_tokens=int(raw("RAG_CHUNK_TOKENS")),
        chunk_overlap=int(raw("RAG_CHUNK_OVERLAP")),
        corpus_dir=raw("RAG_CORPUS_DIR"),
        index_dir=raw("RAG_INDEX_DIR"),
        sources_file=raw("RAG_SOURCES_FILE"),
        triage_llm=_as_bool(raw("RAG_TRIAGE_LLM")),
        pii_redact=_as_bool(raw("RAG_PII_REDACT")),
        sanitize_perf=_as_bool(raw("RAG_SANITIZE_PERF")),
        history_turns=int(raw("RAG_HISTORY_TURNS")),
        log_trace=_as_bool(raw("RAG_LOG_TRACE")),
    )


def validate(cfg: Config) -> list[str]:
    """Return a list of human-readable problems. Empty list == usable.

    Check: provider is known; ollama_host is a URL; min_score in [0,1];
    chunk_tokens <= 224 (headroom under the 256 word-piece cap, architecture §9.2);
    chunk_overlap < chunk_tokens. Never raises -- callers decide what to do.
    """
    problems: list[str] = []

    known_providers = {"ollama", "fake", "openai", "none"}
    if cfg.provider not in known_providers:
        problems.append(
            f"unknown RAG_PROVIDER {cfg.provider!r}; expected one of "
            f"{sorted(known_providers)}"
        )

    if not cfg.ollama_host.startswith(("http://", "https://")):
        problems.append(
            f"RAG_OLLAMA_HOST {cfg.ollama_host!r} is not an http(s) URL"
        )

    if not 0.0 <= cfg.min_score <= 1.0:
        problems.append(
            f"RAG_MIN_SCORE {cfg.min_score} is outside [0, 1]; it is a cosine "
            f"similarity, not a distance"
        )

    if cfg.chunk_tokens > 224:
        problems.append(
            f"RAG_CHUNK_TOKENS {cfg.chunk_tokens} exceeds 224; leave headroom "
            f"under the 256 word-piece cap (architecture §9.2)"
        )

    if cfg.chunk_overlap >= cfg.chunk_tokens:
        problems.append(
            f"RAG_CHUNK_OVERLAP {cfg.chunk_overlap} must be less than "
            f"RAG_CHUNK_TOKENS {cfg.chunk_tokens}"
        )

    return problems
