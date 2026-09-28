"""Retriever and relevance-gate tests. No model download: fake provider."""
from __future__ import annotations

import pytest

from rag_bot.providers.fake import FakeEmbeddingProvider
from rag_bot.retrieve.gate import (
    DEFAULT_COVERED_TOPICS,
    TOPIC_LABELS,
    WEAK_MARGIN,
    GateResult,
    evaluate,
)
from rag_bot.retrieve.retriever import expand_query, retrieve
from rag_bot.types import Chunk, ScoredChunk

MIN_SCORE = 0.35


def scored(score: float, scheme_id: str = "S1", topic: str = "expense_ratio") -> ScoredChunk:
    chunk = Chunk(
        id=f"{scheme_id}::0", text=f"Expense ratio: {score:.2f}", source_id=scheme_id,
        url="https://groww.in/mutual-funds/x", publisher="groww.in", source_tier="brief",
        scheme_id=scheme_id, scheme_name=f"Scheme {scheme_id}", heading="Expense ratio",
        topic=topic, chunk_index=0, n_tokens=6, fetched_at=None, factsheet_url=None,
        content_hash=f"h{score}",
    )
    return ScoredChunk(chunk=chunk, score=score)


@pytest.fixture
def store(tmp_path):
    from rag_bot.index.store import Store

    return Store(str(tmp_path), "fake-model", "v1")


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


# ===================================================================
# the six cases the spec names
# ===================================================================


def test_empty_index_rejects_with_empty_index():
    r = evaluate([], min_score=MIN_SCORE, index_empty=True, covered_topics=[])
    assert not r.accepted
    assert r.reason == "empty_index"
    assert r.top_score is None


def test_no_candidates_rejects_with_no_candidates():
    r = evaluate([], min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert not r.accepted
    assert r.reason == "no_candidates"
    assert r.top_score is None


def test_a_point_two_score_against_min_point_three_five_rejects_low_score():
    r = evaluate([scored(0.20)], min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert not r.accepted
    assert r.reason == "low_score"
    assert r.top_score == pytest.approx(0.20)


def test_a_point_three_six_score_accepts_with_weak_evidence():
    r = evaluate([scored(0.36)], min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert r.accepted
    assert r.weak_evidence is True
    assert r.reason is None


def test_an_eighty_score_accepts_without_weak_evidence():
    r = evaluate([scored(0.80)], min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert r.accepted
    assert r.weak_evidence is False


def test_the_class_b_message_contains_no_url():
    for r in (
        evaluate([], min_score=MIN_SCORE, index_empty=False, covered_topics=[]),
        evaluate([scored(0.10)], min_score=MIN_SCORE, index_empty=False,
                 covered_topics=[]),
    ):
        assert "http" not in (r.message or "")
        assert "groww" not in (r.message or "").lower()


# ===================================================================
# gate behaviour
# ===================================================================


def test_an_accepted_result_has_no_message():
    r = evaluate([scored(0.90)], min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert r.accepted and r.message is None


def test_the_gate_uses_only_the_top_chunk():
    """A strong chunk behind a weak one still fails; the gate is not an average."""
    r = evaluate([scored(0.10), scored(0.90)], min_score=MIN_SCORE,
                 index_empty=False, covered_topics=[])
    assert not r.accepted and r.reason == "low_score"
    assert r.top_score == pytest.approx(0.10)


def test_chunks_must_be_pre_sorted_for_the_gate_to_be_correct():
    """Documents the contract: the retriever sorts, the gate reads chunks[0]."""
    ordered = evaluate([scored(0.90), scored(0.10)], min_score=MIN_SCORE,
                       index_empty=False, covered_topics=[])
    assert ordered.accepted
    # Passing them unsorted is a caller bug; the gate trusts index 0 by design,
    # because a gate that re-sorts would be a second ranking policy in disguise.
    assert evaluate([scored(0.10), scored(0.90)], min_score=MIN_SCORE,
                    index_empty=False, covered_topics=[]).top_score == pytest.approx(0.10)


def test_weak_evidence_boundary_is_exact():
    at = evaluate([scored(MIN_SCORE + WEAK_MARGIN)], min_score=MIN_SCORE,
                  index_empty=False, covered_topics=[])
    assert at.accepted and not at.weak_evidence, "the margin is exclusive"
    just_under = evaluate([scored(MIN_SCORE + WEAK_MARGIN - 0.001)], min_score=MIN_SCORE,
                          index_empty=False, covered_topics=[])
    assert just_under.accepted and just_under.weak_evidence


def test_a_score_exactly_at_the_threshold_is_accepted():
    r = evaluate([scored(MIN_SCORE)], min_score=MIN_SCORE, index_empty=False,
                 covered_topics=[])
    # Accepted, and still flagged weak: it sits inside the 0.05 margin.
    assert r.accepted and r.weak_evidence


def test_a_score_just_below_the_threshold_is_rejected():
    r = evaluate([scored(MIN_SCORE - 0.001)], min_score=MIN_SCORE, index_empty=False,
                 covered_topics=[])
    assert not r.accepted and r.reason == "low_score"


def test_the_gate_never_raises_on_junk_input():
    for chunks in ([], [scored(-1.0)], [scored(0.0)], [scored(1.0)] * 5):
        assert isinstance(evaluate(chunks, min_score=MIN_SCORE, index_empty=False,
                                   covered_topics=["expense_ratio"]), GateResult)


# ===================================================================
# the class-B copy
# ===================================================================


def test_the_class_b_message_names_the_covered_topics():
    r = evaluate([], min_score=MIN_SCORE, index_empty=False,
                 covered_topics=list(DEFAULT_COVERED_TOPICS))
    for topic in DEFAULT_COVERED_TOPICS:
        assert TOPIC_LABELS[topic] in r.message


def test_the_class_b_message_never_promises_statement_download():
    """`statement` was dropped from scope; a refusal must not claim it."""
    r = evaluate([], min_score=MIN_SCORE, index_empty=False,
                 covered_topics=list(DEFAULT_COVERED_TOPICS))
    assert "statement" not in r.message.lower()


def test_the_class_b_message_says_returns_are_not_covered():
    r = evaluate([], min_score=MIN_SCORE, index_empty=False,
                 covered_topics=list(DEFAULT_COVERED_TOPICS))
    assert "returns" in r.message.lower()


def test_the_empty_index_message_is_its_own_copy():
    r = evaluate([], min_score=MIN_SCORE, index_empty=True, covered_topics=[])
    assert "indexed yet" in r.message


# ===================================================================
# no relative thresholds -- the failure mode the spec calls out
# ===================================================================


def test_an_off_topic_question_with_five_equal_bad_matches_is_refused():
    """All five scores are terrible but identical. A relative rule would pass."""
    chunks = [scored(0.10, scheme_id=f"S{i}") for i in range(1, 6)]
    r = evaluate(chunks, min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert not r.accepted and r.reason == "low_score"


def test_one_good_chunk_among_awful_ones_is_accepted():
    chunks = [scored(0.80), *[scored(0.02, scheme_id=f"S{i}") for i in range(2, 6)]]
    r = evaluate(chunks, min_score=MIN_SCORE, index_empty=False, covered_topics=[])
    assert r.accepted and r.top_score == pytest.approx(0.80)


# ===================================================================
# retriever
# ===================================================================


def test_retrieve_on_an_empty_index_returns_nothing(store, embedder):
    assert retrieve("What is the expense ratio?", k=3, embedder=embedder, store=store) == []


def test_retrieve_returns_sorted_descending(store, embedder):
    from rag_bot.index.store import embedding_text

    p = embedder
    chunks = [
        scored(0.1, "S1", "expense_ratio").chunk,
        scored(0.9, "S2", "exit_load").chunk,
        scored(0.5, "S3", "benchmark").chunk,
    ]
    chunks = [c.__class__(**{**c.__dict__, "content_hash": f"h{i}"})
              for i, c in enumerate(chunks)]
    store.upsert(chunks, p.embed_documents([embedding_text(c) for c in chunks]))
    hits = retrieve("expense ratio", k=3, embedder=p, store=store)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_retrieve_never_exceeds_k(store, embedder):
    from rag_bot.index.store import embedding_text

    chunks = []
    for i in range(8):
        c = scored(0.5, f"S{i % 5 + 1}").chunk
        chunks.append(c.__class__(**{**c.__dict__, "id": f"S{i % 5 + 1}::{i}",
                                     "content_hash": f"h{i}"}))
    store.upsert(chunks, embedder.embed_documents([embedding_text(c) for c in chunks]))
    assert len(retrieve("expense ratio", k=3, embedder=embedder, store=store)) == 3


def test_retrieve_with_k_zero_returns_nothing(store, embedder):
    assert retrieve("expense ratio", k=0, embedder=embedder, store=store) == []


def test_retrieve_filters_by_scheme(store, embedder):
    from rag_bot.index.store import embedding_text

    chunks = []
    for sid in ("S1", "S2", "S3"):
        c = scored(0.5, sid).chunk
        chunks.append(c.__class__(**{**c.__dict__, "content_hash": f"h{sid}"}))
    store.upsert(chunks, embedder.embed_documents([embedding_text(c) for c in chunks]))
    hits = retrieve("expense ratio", k=5, scheme_id="S2", embedder=embedder, store=store)
    assert {h.chunk.scheme_id for h in hits} == {"S2"}


def test_retrieve_never_raises_on_a_broken_store(store, embedder):
    class Broken:
        def count(self):
            return 5

        def query(self, *a, **k):
            raise RuntimeError("chroma exploded")

    assert retrieve("expense ratio", k=3, embedder=embedder, store=Broken()) == []


# ===================================================================
# query-side synonym expansion
# ===================================================================


@pytest.mark.parametrize("query,expected", [
    ("What is the TER?", "expense ratio"),
    ("what are the charges?", "expense ratio"),
    ("any redemption charge?", "exit load"),
    ("is there a lock in period", "lock in"),
    ("how volatile is it", "riskometer"),
    ("which index does it track", "benchmark"),
    ("how much is the minimum", "minimum SIP"),
    ("holding period on the flexi cap scheme?", "lock in"),
])
def test_synonyms_reach_the_corpus_vocabulary(query, expected):
    assert expected.lower() in expand_query(query).lower()


def test_expansion_keeps_the_original_query():
    q = "What is the TER of HDFC Large Cap?"
    assert expand_query(q).startswith(q)


def test_a_query_with_no_synonyms_is_unchanged():
    q = "What is the expense ratio of HDFC Large Cap Fund?"
    assert expand_query(q) == q


def test_expansion_is_idempotent():
    once = expand_query("What is the TER?")
    assert expand_query(once) == once


def test_expansion_does_not_fire_on_substrings():
    """TER must not match inside "later" or "terrible"."""
    for q in ("Is that a later thing?", "what a terrible fund"):
        assert expand_query(q) == q
