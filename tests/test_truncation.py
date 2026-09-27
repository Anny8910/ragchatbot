"""Truncation must fail the build, never silently drop text.

MiniLM truncates at 256 word pieces without complaint. A chunk over the cap
therefore still "embeds successfully" while the tail is discarded, which shows
up much later as a wrong answer that no test can explain. The cap has to be
asserted on the TRUE word-piece count, computed with truncation disabled.
"""
from __future__ import annotations

import pytest

from rag_bot.ingest.chunker import chunk_all
from rag_bot.providers.fake import FakeEmbeddingProvider
from rag_bot.providers.minilm import MAX_WORD_PIECES, MiniLMProvider
from rag_bot.types import Chunk, Document, Source


def make_document(text: str) -> Document:
    return Document(
        text=text,
        source=Source(
            source_id="S1",
            url="https://groww.in/mutual-funds/x",
            publisher="groww.in",
            source_tier="brief",
            scheme_id="S1",
            scheme_name="Test Scheme",
            factsheet_url=None,
            aliases=("Test Scheme",),
            fetched_at="2026-09-25T10:00:00+05:30",
            snapshot_path=None,
            content_hash="abc",
            status="ok",
        ),
    )


def make_chunk(i: int = 0, text: str = "Expense ratio: 1.21") -> Chunk:
    return Chunk(
        id=f"S1::{i}",
        text=text,
        source_id="S1",
        url="https://groww.in/mutual-funds/x",
        publisher="groww.in",
        source_tier="brief",
        scheme_id="S1",
        scheme_name="Test Scheme",
        heading="Expense ratio",
        topic="expense_ratio",
        chunk_index=i,
        n_tokens=6,
        fetched_at="2026-09-25T10:00:00+05:30",
        factsheet_url=None,
        content_hash=f"hash{i:04d}",
    )


# -- the cap itself --------------------------------------------------------


def test_the_cap_is_256():
    assert MAX_WORD_PIECES == 256


def test_a_300_word_piece_chunk_raises():
    provider = FakeEmbeddingProvider()
    over = " ".join(["expense"] * 300)
    with pytest.raises(ValueError, match="cap"):
        provider.assert_fits(over)


def test_the_error_names_the_offending_length():
    provider = FakeEmbeddingProvider()
    with pytest.raises(ValueError) as exc:
        provider.assert_fits(" ".join(["expense"] * 300))
    assert "300" in str(exc.value)


def test_a_chunk_at_the_cap_does_not_raise():
    FakeEmbeddingProvider().assert_fits(" ".join(["expense"] * MAX_WORD_PIECES))


def test_a_chunk_one_over_the_cap_raises():
    provider = FakeEmbeddingProvider()
    provider.assert_fits(" ".join(["expense"] * MAX_WORD_PIECES))
    with pytest.raises(ValueError):
        provider.assert_fits(" ".join(["expense"] * (MAX_WORD_PIECES + 1)))


# -- the count must be the TRUE length, not the truncated one ---------------


def test_count_tokens_is_not_capped_at_the_limit():
    """A counting function that saturates at the cap makes assert_fits useless."""
    provider = FakeEmbeddingProvider()
    n = provider.count_tokens(" ".join(["expense"] * 1000))
    assert n == 1000, "count_tokens saturated; it must report the real length"


def test_count_tokens_grows_past_256():
    provider = FakeEmbeddingProvider()
    assert provider.count_tokens(" ".join(["w"] * 400)) > MAX_WORD_PIECES


# -- the real tokenizer: these tests need MiniLM in the local HF cache -----


@pytest.fixture(scope="module")
def minilm():
    try:
        return MiniLMProvider()
    except Exception as exc:  # pragma: no cover - offline environment
        pytest.skip(f"MiniLM unavailable: {exc}")


def test_minilm_import_does_not_load_the_model():
    """Importing the module must not download; the load has to be lazy."""
    import importlib

    import rag_bot.providers.minilm as m

    importlib.reload(m)
    provider = m.MiniLMProvider()
    assert provider._model is None
    assert provider._tokenizer is None


def test_minilm_counts_real_word_pieces(minilm):
    text = "Expense ratio: 1.21 per annum, charged on the daily NAV."
    n = minilm.count_tokens(text)
    assert n > 0
    assert n == len(minilm.tokenizer.encode(text, add_special_tokens=False,
                                            truncation=False))


def test_minilm_word_pieces_exceed_whitespace_words(minilm):
    """Sub-word splitting means whitespace counting understates the real cost."""
    text = " ".join(["expense-ratio,"] * 40)
    assert minilm.count_tokens(text) > len(text.split())


def test_minilm_rejects_a_300_word_piece_chunk(minilm):
    """The literal 300-word case from the spec, measured in real word pieces."""
    text = " ".join(["expense"] * 300)
    assert minilm.count_tokens(text) > MAX_WORD_PIECES
    with pytest.raises(ValueError, match="cap"):
        minilm.assert_fits(text)


def test_minilm_does_not_truncate_short_text(minilm):
    text = "Exit load: Nil"
    assert minilm.count_tokens(text) == len(
        minilm.tokenizer.encode(text, add_special_tokens=False, truncation=False)
    )


# -- embeddings are normalised ---------------------------------------------


def test_minilm_embeddings_are_l2_normalised(minilm):
    import math

    vec = minilm.embed_query("Expense ratio: 1.21")
    assert len(vec) == 384
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_minilm_embedding_of_empty_text_is_finite(minilm):
    import math

    vec = minilm.embed_documents([""])[0]
    assert all(math.isfinite(v) for v in vec)


# -- the chunker must not produce over-cap chunks on its own ---------------


def test_the_chunker_keeps_its_own_chunks_under_the_cap():
    """A target of 180 with a 256 cap leaves headroom; verify the gap holds."""
    doc = make_document("\n".join(f"Expense ratio: {i}.00 per annum" for i in range(60)))
    chunks = chunk_all(
        [doc],
        target_tokens=180,
        overlap_tokens=40,
        tokenizer=lambda t: len(t.split()),
    )
    assert chunks
    for c in chunks:
        assert len(c.text.split()) <= MAX_WORD_PIECES


def test_a_single_unbreakable_block_is_still_split():
    doc = make_document(" ".join(["expense"] * 900))
    chunks = chunk_all(
        [doc],
        target_tokens=180,
        overlap_tokens=40,
        tokenizer=lambda t: len(t.split()),
    )
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= MAX_WORD_PIECES for c in chunks)


# -- the builder must refuse an over-cap chunk -----------------------------


def _cfg(tmp_path, **overrides):
    """A Config pointed at tmp_path, so a test can never write the real index."""
    import dataclasses

    from rag_bot.config import load

    base = load()
    return dataclasses.replace(
        base, index_dir=str(tmp_path / "index"),
        corpus_dir=str(tmp_path / "corpus"), **overrides
    )


def test_the_builder_refuses_to_index_an_over_cap_chunk(tmp_path, monkeypatch):
    """The end-to-end guarantee: no index is written if a chunk is too long.

    The chunker hard-splits, so an over-cap chunk cannot arise from a long
    document in normal operation. The guard therefore has to be tested directly:
    chunk_all is stubbed to hand back one oversized chunk, which is the shape a
    future chunker regression would produce.
    """
    from rag_bot.ingest import builder

    doc = make_document("Expense ratio: 1.21")
    over = make_chunk(0, text=" ".join(["expense"] * 400))
    assert FakeEmbeddingProvider().count_tokens(over.text) > MAX_WORD_PIECES

    monkeypatch.setattr(builder, "load_from_manifest", lambda p: [doc])
    monkeypatch.setattr(builder, "chunk_all", lambda docs, **kw: [over])
    monkeypatch.setattr(
        builder, "_get_provider", lambda cfg, offline: FakeEmbeddingProvider()
    )

    cfg = _cfg(tmp_path)
    with pytest.raises(builder.GuardFailure, match="cap"):
        builder.build(cfg, offline=True)

    assert not (tmp_path / "index").exists(), "a refusing build must write nothing"


def test_the_builder_refuses_an_invalid_chunk_size(tmp_path, monkeypatch):
    """RAG_CHUNK_TOKENS above the 224 headroom limit must stop the build.

    Without this the only thing between a 400-token target and MiniLM's silent
    truncation at 256 is the chunker's good behaviour, not an actual gate.
    """
    from rag_bot.ingest import builder

    monkeypatch.setattr(
        builder, "load_from_manifest", lambda p: [make_document("Expense ratio: 1.21")]
    )
    cfg = _cfg(tmp_path, chunk_tokens=400)
    with pytest.raises(builder.GuardFailure, match="invalid configuration"):
        builder.build(cfg, offline=True)
    assert not (tmp_path / "index").exists()


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from rag_bot.ingest import builder

    doc = make_document("## Expense ratio\nExpense ratio: 1.21")
    monkeypatch.setattr(builder, "load_from_manifest", lambda p: [doc])
    monkeypatch.setattr(
        builder, "_get_provider", lambda cfg, offline: FakeEmbeddingProvider()
    )
    cfg = _cfg(tmp_path)
    assert builder.build(cfg, dry_run=True, offline=True) == 0
    assert "dry run" in capsys.readouterr().out
    assert not (tmp_path / "index").exists()

