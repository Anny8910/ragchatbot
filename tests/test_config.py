"""Config tests. No network, no model download."""
from __future__ import annotations

import dataclasses
import os

import dotenv
import pytest

from rag_bot.config import DEFAULTS, load, validate


@pytest.fixture(autouse=True)
def _isolate_from_ambient_config(monkeypatch):
    """Make these tests depend on DEFAULTS and their own monkeypatches only.

    `config.load()` calls `load_dotenv(override=False)` and then reads
    `os.environ`, so without this a developer's `.env` becomes the subject under
    test: `test_defaults_load_and_validate_clean` asserts on DEFAULTS while
    actually reading whatever the local machine happens to have configured. That
    made the suite pass on a clean checkout and fail in the developer's own
    directory, which is the worst possible time for it to fail.

    The release gate is "pytest green with no network and no model download". A
    test that only passes when you have no `.env` is not that.
    """
    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


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
    """The default build must run with no .env at all (architecture D3).

    Exercises the real fallback rather than asserting a tautology: `load_dotenv`
    is made to raise, exactly as it would if python-dotenv were not installed,
    and `load()` must still return usable DEFAULTS instead of propagating.
    """

    def _boom(*_a, **_k):
        raise ImportError("no dotenv here")

    monkeypatch.setattr(dotenv, "load_dotenv", _boom)
    cfg = load()
    assert cfg.provider == DEFAULTS["RAG_PROVIDER"]
    assert validate(cfg) == []


def test_validate_accepts_every_hosted_provider():
    """`groq` joined `openai` as an OpenAI-compatible hosted provider.

    Regression guard for the `known_providers` set: a provider that
    `app.build_llm` handles but `validate` rejects makes the app refuse to start,
    and one that `validate` accepts but `build_llm` does not makes it crash on
    first use. Validating the set here is the cheap half of that contract.
    """
    for provider in ("ollama", "openai", "groq", "none", "fake"):
        cfg = dataclasses.replace(load(), provider=provider)
        assert not any("PROVIDER" in p for p in validate(cfg)), provider


def test_validate_rejects_an_unknown_provider():
    cfg = dataclasses.replace(load(), provider="mistral-local")
    assert any("PROVIDER" in p for p in validate(cfg))


def test_config_is_frozen():
    cfg = load()
    with pytest.raises(Exception):
        cfg.chunk_tokens = 999  # type: ignore[misc]
