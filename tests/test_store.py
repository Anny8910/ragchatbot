"""Store tests: naming, flat metadata, round trip, filters, orphans.

Uses the fake provider so the suite runs with no model download and no network.
"""
from __future__ import annotations

import pytest

from rag_bot.index.ids import collection_name, slugify
from rag_bot.index.store import Store, embedding_text, flat_metadata
from rag_bot.providers.fake import FakeEmbeddingProvider
from rag_bot.types import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_chunk(i: int = 0, topic: str = "expense_ratio", scheme_id: str = "S1",
               text: str | None = None) -> Chunk:
    return Chunk(
        id=f"{scheme_id}::{i}",
        text=text if text is not None else f"Expense ratio: {1.0 + i / 100:.2f}",
        source_id=scheme_id,
        url="https://groww.in/mutual-funds/x",
        publisher="groww.in",
        source_tier="brief",
        scheme_id=scheme_id,
        scheme_name=f"Scheme {scheme_id}",
        heading="Expense ratio",
        topic=topic,
        chunk_index=i,
        n_tokens=6,
        fetched_at="2026-09-25T10:00:00+05:30",
        factsheet_url=None,
        content_hash=f"hash{i:04d}",
    )


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(str(tmp_path), EMBED_MODEL, "v1")


# -- collection naming (architecture D5) -----------------------------------


def test_collection_name_has_the_three_part_format():
    name = collection_name(EMBED_MODEL, "v1")
    assert name.startswith("mf_faq__")
    assert name.endswith("__v1")
    parts = name.split("__")
    assert len(parts) == 3
    assert parts[1] == "sentence-transformers-all-minilm-l6-v2"
    assert parts[2] == "v1"


def test_collection_name_chromadb_legal_charset():
    name = collection_name("sentence-transformers/all-MiniLM-L6-v2", "v1")
    assert all(ch.isalnum() or ch in "._-" for ch in name)


def test_collection_name_encodes_model_and_version():
    a = collection_name(EMBED_MODEL, "v1")
    b = collection_name(EMBED_MODEL, "v2")
    c = collection_name("BAAI/bge-small-en", "v1")
    assert a != b, "a corpus version bump must produce a different collection"
    assert a != c, "a model change must produce a different collection"


def test_collection_name_survives_a_very_long_model_path():
    name = collection_name("org/" + "x" * 300 + "/model", "v1")
    assert len(name) < 512
    assert not name.endswith("_")


@pytest.mark.parametrize("raw,expect", [
    ("all-MiniLM-L6-v2", "all-minilm-l6-v2"),
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers-all-minilm-l6-v2"),
    ("a b/c:d", "a-b-c-d"),
    ("***", "x"),
])
def test_slugify(raw, expect):
    assert slugify(raw) == expect


# -- flat metadata --------------------------------------------------------


def test_metadata_values_are_all_flat_chroma_scalars():
    meta = flat_metadata(make_chunk())
    allowed = (str, int, float, bool)
    for key, value in meta.items():
        assert isinstance(value, allowed), f"{key}={value!r} is {type(value)}"
        assert not isinstance(value, (list, dict, tuple, set))


def test_metadata_contains_no_none():
    # Chroma rejects None. factsheet_url is None on every Source today, so this
    # is the field that would actually break a build.
    meta = flat_metadata(make_chunk())
    assert meta["factsheet_url"] == ""
    assert not any(v is None for v in meta.values())


def test_metadata_carries_the_retrieval_filters():
    meta = flat_metadata(make_chunk(scheme_id="S3", topic="exit_load"))
    assert meta["scheme_id"] == "S3"
    assert meta["topic"] == "exit_load"


def test_metadata_keeps_the_provenance_fields():
    meta = flat_metadata(make_chunk())
    for key in ("source_id", "url", "publisher", "source_tier", "fetched_at",
                "content_hash", "scheme_name"):
        assert key in meta, key


# -- the embedded text vs the LLM-facing text ------------------------------


def test_embedding_text_names_the_scheme():
    """Five same-topic chunks must not embed to near-identical vectors."""
    c = make_chunk(0, scheme_id="S1", topic="expense_ratio")
    et = embedding_text(c)
    assert c.scheme_name in et
    assert c.text in et


def test_embedding_text_includes_the_heading():
    c = make_chunk(0)
    assert c.heading in embedding_text(c)


def test_embedding_text_differs_across_schemes_for_the_same_fact():
    a = make_chunk(0, scheme_id="S1", text="Expense ratio: Nil")
    b = make_chunk(0, scheme_id="S2", text="Expense ratio: Nil")
    assert a.text == b.text, "same fact text, which is the case that was ambiguous"
    assert embedding_text(a) != embedding_text(b)


def test_the_llm_facing_text_stays_clean_after_a_round_trip(store):
    """architecture 5.2: Chunk.text is the only field sent to the LLM.

    The embedding prefix must not leak back out of the store, or the generator
    would see "HDFC Equity Fund\nBenchmark: ..." instead of the fact.
    """
    p = FakeEmbeddingProvider()
    c = make_chunk(0, scheme_id="S2", text="Benchmark: NIFTY 500 Total Return Index")
    store.upsert([c], p.embed_documents([embedding_text(c)]))
    got = store.query(p.embed_query(embedding_text(c)), k=1)[0].chunk
    assert got.text == "Benchmark: NIFTY 500 Total Return Index"
    assert got.text == c.text
    assert not got.text.startswith(c.scheme_name)


def test_the_body_is_available_as_metadata(store):
    assert flat_metadata(make_chunk())["body"] == make_chunk().text


def test_metadata_stays_flat_with_the_body_added():
    meta = flat_metadata(make_chunk())
    for value in meta.values():
        assert isinstance(value, (str, int, float, bool))
        assert not isinstance(value, (list, dict, tuple, set))


# -- round trip ------------------------------------------------------------


def test_upsert_then_query_returns_the_chunk(store):
    chunks = [make_chunk(0)]
    p = FakeEmbeddingProvider()
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    assert store.count() == 1
    hits = store.query(p.embed_query("Expense ratio: 1.00"), k=3)
    assert len(hits) == 1
    assert hits[0].chunk.id == "S1::0"
    assert hits[0].chunk.text == "Expense ratio: 1.00"


def test_query_scores_are_cosine_similarity_in_range(store):
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    p = FakeEmbeddingProvider()
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    for hit in store.query(p.embed_query("Expense ratio: 1.00"), k=3):
        assert -1.0 <= hit.score <= 1.0
    # A self-match on a normalised vector must be ~1.0, which is the
    # observable signature of the 1 - distance convention holding.
    exact = store.query(p.embed_query(chunks[0].text), k=1)[0]
    assert exact.score == pytest.approx(1.0, abs=1e-4)
    assert exact.chunk.id == "S1::0"


def test_query_is_sorted_by_descending_score(store):
    p = FakeEmbeddingProvider()
    chunks = [make_chunk(i) for i in range(6)]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    scores = [h.score for h in store.query(p.embed_query("Expense ratio: 1.03"), k=6)]
    assert scores == sorted(scores, reverse=True)


def test_query_respects_k(store):
    p = FakeEmbeddingProvider()
    chunks = [make_chunk(i) for i in range(6)]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    assert len(store.query(p.embed_query("Expense ratio"), k=2)) == 2


def test_query_overfetches_by_two_but_returns_k(store):
    """architecture 12 asks for k+2 so the gate has spare candidates."""
    p = FakeEmbeddingProvider()
    chunks = [make_chunk(i) for i in range(10)]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    assert len(store.query(p.embed_query("Expense ratio"), k=3)) == 3


def test_query_on_an_empty_collection_returns_nothing(store):
    assert store.query([0.0] * 384, k=3) == []


def test_upsert_rejects_a_length_mismatch(store):
    with pytest.raises(ValueError):
        store.upsert([make_chunk(0), make_chunk(1)], [[0.0] * 384])


def test_upsert_of_nothing_is_a_no_op(store):
    store.upsert([], [])
    assert store.count() == 0


# -- re-upsert is idempotent ----------------------------------------------


def test_reupserting_the_same_ids_does_not_duplicate(store):
    p = FakeEmbeddingProvider()
    c = make_chunk(0)
    store.upsert([c], p.embed_documents([c.text]))
    store.upsert([c], p.embed_documents([c.text]))
    assert store.count() == 1


# -- where filters ---------------------------------------------------------


def test_query_filters_by_topic(store):
    p = FakeEmbeddingProvider()
    chunks = [
        make_chunk(0, topic="expense_ratio", text="Expense ratio: 1.03"),
        make_chunk(1, topic="exit_load", text="Exit load: Nil"),
        make_chunk(2, topic="benchmark", text="Benchmark: NIFTY 500 TRI"),
    ]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    hits = store.query(p.embed_query("Expense ratio: 1.03"), k=3, topic="exit_load")
    assert [h.chunk.topic for h in hits] == ["exit_load"]


def test_query_filters_by_scheme_id(store):
    p = FakeEmbeddingProvider()
    chunks = [
        make_chunk(0, scheme_id="S1"),
        make_chunk(0, scheme_id="S2"),
        make_chunk(0, scheme_id="S3"),
    ]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    hits = store.query(p.embed_query("Expense ratio: 1.03"), k=5, scheme_id="S2")
    assert [h.chunk.scheme_id for h in hits] == ["S2"]


def test_query_filters_by_scheme_and_topic_together(store):
    p = FakeEmbeddingProvider()
    chunks = [
        make_chunk(0, scheme_id="S1", topic="expense_ratio"),
        make_chunk(1, scheme_id="S1", topic="exit_load"),
        make_chunk(2, scheme_id="S2", topic="exit_load"),
    ]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    hits = store.query(
        p.embed_query("Exit load: 1%"), k=5, scheme_id="S1", topic="exit_load"
    )
    assert len(hits) == 1
    assert (hits[0].chunk.scheme_id, hits[0].chunk.topic) == ("S1", "exit_load")


def test_a_filter_matching_nothing_returns_nothing(store):
    p = FakeEmbeddingProvider()
    store.upsert([make_chunk(0)], p.embed_documents(["Expense ratio: 1.03"]))
    assert store.query(p.embed_query("x"), k=3, scheme_id="S99") == []


# -- orphans ---------------------------------------------------------------


def test_orphans_finds_chunks_whose_source_left_the_manifest(store):
    p = FakeEmbeddingProvider()
    chunks = [make_chunk(0, scheme_id="S1"), make_chunk(0, scheme_id="S2")]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    assert sorted(store.orphans({"S1", "S2"})) == []
    assert store.orphans({"S1"}) == ["S2::0"]


def test_delete_orphans_shrinks_the_index(store):
    p = FakeEmbeddingProvider()
    chunks = [make_chunk(0, scheme_id="S1"), make_chunk(0, scheme_id="S2")]
    store.upsert(chunks, p.embed_documents([c.text for c in chunks]))
    store.delete_ids(store.orphans({"S1"}))
    assert store.count() == 1
    assert store.source_ids_present() == {"S1"}


def test_delete_ids_of_nothing_is_a_no_op(store):
    assert store.delete_ids([]) == 0


# -- persistence -----------------------------------------------------------


def test_the_index_survives_a_restart(tmp_path):
    p = FakeEmbeddingProvider()
    first = Store(str(tmp_path), EMBED_MODEL, "v1")
    first.upsert([make_chunk(0)], p.embed_documents(["Expense ratio: 1.03"]))
    second = Store(str(tmp_path), EMBED_MODEL, "v1")
    assert second.count() == 1
    assert second.query(p.embed_query("Expense ratio: 1.03"), k=1)[0].chunk.id == "S1::0"


def test_drop_empties_the_index(store):
    p = FakeEmbeddingProvider()
    store.upsert([make_chunk(0)], p.embed_documents(["Expense ratio: 1.03"]))
    store.drop()
    assert Store(store.index_dir, EMBED_MODEL, "v1").count() == 0


def test_a_different_version_uses_a_different_collection(tmp_path):
    p = FakeEmbeddingProvider()
    v1 = Store(str(tmp_path), EMBED_MODEL, "v1")
    v1.upsert([make_chunk(0)], p.embed_documents(["Expense ratio: 1.03"]))
    v2 = Store(str(tmp_path), EMBED_MODEL, "v2")
    assert v2.name != v1.name
    assert v2.count() == 0
