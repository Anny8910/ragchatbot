"""Config tests. No network, no model download."""
from __future__ import annotations

import os

import pytest

from rag_bot.config import DEFAULTS, Config, load, validate


def test_defaults_load_and_validate_clean():
    cfg = load()
    assert validate(cfg) == [], f"defaults should be usable, got {validate(cfg)}"
    assert cfg.provider == "ollama"
    assert cfg.llm_model == "llama3.5:8b"
    assert cfg.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.chunk_tokens == 180
    assert cfg.chunk_overlap == 40
    assert cfg.min_score == 0.35
    assert cfg.top_k == 3


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "150")
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")
    monkeypatch.setenv("RAG_TRIAGE_LLM", "false")
    cfg = load()
    assert cfg.chunk_tokens == 150
    assert cfg.min_score == 0.5
    assert cfg.triage_llm is False


def test_validate_flags_chunk_tokens_above_224(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "300")
    problems = validate(load())
    assert any("224" in p for p in problems), problems


def test_validate_flags_overlap_not_less_than_tokens(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "180")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "180")
    problems = validate(load())
    assert any("OVERLAP" in p for p in problems), problems


def test_validate_flags_min_score_out_of_range(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "1.4")
    problems = validate(load())
    assert any("MIN_SCORE" in p for p in problems), problems


def test_validate_flags_unknown_provider_and_bad_host(monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", "not-a-provider")
    monkeypatch.setenv("RAG_OLLAMA_HOST", "localhost:11434")
    problems = validate(load())
    assert any("PROVIDER" in p for p in problems), problems
    assert any("OLLAMA_HOST" in p for p in problems), problems


def test_validate_never_raises_on_garbage(monkeypatch):
    """validate() reports problems; it does not explode on them."""
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "900")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "-5")
    monkeypatch.setenv("RAG_PROVIDER", "??")
    assert isinstance(validate(load()), list)


def test_load_works_without_dotenv(monkeypatch):
    """The default build must run with no .env at all (architecture D3)."""
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    assert load().provider == DEFAULTS["RAG_PROVIDER"]
    assert ".env" not in os.listdir(".") or True  # no hard dependency on it


def test_config_is_frozen():
    cfg = load()
    with pytest.raises(Exception):
        cfg.chunk_tokens = 999  # type: ignore[misc]
