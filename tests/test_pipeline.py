"""Pipeline tests: the orchestration contract, with no network and no model.

Three of these are regression tests for defects that produced *plausible* output
rather than an error, which is why they were not caught by anything else:

- `_stage` recorded seconds while both consumers labelled the value `ms`, so the
  UI's pipeline trace -- the panel whose whole job is showing what the pipeline
  did -- reported a 600ms generation call as "1ms".
- The no-scheme path returned class E before the gate ran, so an out-of-corpus
  question that named no scheme was answered with "which scheme do you mean?".
- A provider returning "" looks identical to a model choosing to say nothing.

Uses the fake embedding provider (not semantic -- plumbing only) and a stub LLM.
"""
from __future__ import annotations

import time

import pytest

from dataclasses import replace

from rag_bot.config import load
from rag_bot.index.store import Store
from rag_bot.pipeline import _stage, answer_question
from rag_bot.providers.base import LLMProvider
from rag_bot.providers.fake import FakeEmbeddingProvider
from rag_bot.types import Chunk, Outcome, Source

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class StubLLM:
    """Returns a canned generation and a canned classification. Never raises."""

    name = "stub"

    def __init__(self, reply: str = "The expense ratio is 1.03%.\nSource: 1",
                 label: str = "A_or_B"):
        self._reply = reply
        self._label = label
        self.calls: list[tuple[str, str]] = []

    def generate(self, system, user, *, temperature, max_tokens):
        self.calls.append(("generate", user))
        return self._reply

    def classify(self, system, user, *, labels):
        self.calls.append(("classify", user))
        return self._label


class BrokenLLM:
    """Every call raises, the way a dead endpoint or a revoked key does."""

    name = "broken"

    def generate(self, system, user, *, temperature, max_tokens):
        raise ConnectionError("endpoint unreachable")

    def classify(self, system, user, *, labels):
        raise ConnectionError("endpoint unreachable")


def _sources() -> list[Source]:
    return [
        Source(
            source_id="hdfc_large_cap_growth",
            url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            publisher="groww.in", source_tier="brief", scheme_id="S1",
            scheme_name="HDFC Large Cap Fund - Direct Growth",
            factsheet_url="https://www.hdfcfund.com/mutual-funds/factsheets",
            aliases=("hdfc large cap fund", "large cap fund", "s1"),
            fetched_at="2026-09-27T12:20:52+05:30",
        ),
    ]


def _chunk(i: int = 0) -> Chunk:
    return Chunk(
        id=f"S1::{i}",
        text=f"Expense ratio: 1.0{i}%\nBase expense ratio: 0.8{i}%",
        source_id="hdfc_large_cap_growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        publisher="groww.in", source_tier="brief", scheme_id="S1",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        heading="Expense ratio", topic="expense_ratio", chunk_index=i,
        n_tokens=12, fetched_at="2026-09-27T12:20:52+05:30",
        factsheet_url="https://www.hdfcfund.com/mutual-funds/factsheets",
        content_hash=f"hash{i:04d}",
    )


@pytest.fixture
def store(tmp_path) -> Store:
    """One indexed chunk, embedded with the deterministic fake provider."""
    s = Store(str(tmp_path), EMBED_MODEL, "v1")
    embedder = FakeEmbeddingProvider()
    chunks = [_chunk(0)]
    s.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return s


@pytest.fixture
def cfg():
    return load()


@pytest.fixture
def open_gate(cfg):
    """The same config with the relevance threshold removed.

    Needed because the fake embedder is hashed, not semantic: it cannot tell
    "expense ratio" from "pasta", so real similarity scores are noise and the
    gate correctly refuses everything. That is the gate working, not a bug -- but
    it means these tests cannot exercise any path past the gate unless they lower
    the bar themselves. Only the gate's own decision should be under test in
    `test_off_topic_question_with_no_scheme_is_class_b_not_class_e`; the rest are
    about what happens *after* a chunk has been accepted.
    """
    return replace(cfg, min_score=0.0)


def _ask(question: str, store: Store, cfg, llm=None, sources=None):
    return answer_question(
        question, cfg=cfg, store=store, embedder=FakeEmbeddingProvider(),
        llm=llm or StubLLM(), sources=sources if sources is not None else _sources(),
    )


# -- the stage timer ------------------------------------------------------
def test_stage_records_milliseconds():
    """Both consumers label the value `ms`:
    `ui.components.render_timing` formats `f"{stage} {ms:.0f}ms"` and
    `trace.build_record` rounds to one decimal. A seconds value renders as
    "1ms" for a 600ms call, which is worse than no timing at all -- it is timing
    that is confidently wrong on the screen meant to prove the pipeline ran."""
    latency: dict[str, float] = {}
    with _stage(latency, "generate"):
        time.sleep(0.12)
    assert 100 <= latency["generate"] <= 400, latency
    assert latency["generate"] > 1.0, "a 120ms sleep must not read as ~0.12 ms"


def test_stage_records_a_stage_that_raised():
    """A failing stage is the one worth seeing on the timeline."""
    latency: dict[str, float] = {}
    with pytest.raises(ValueError):
        with _stage(latency, "retrieve"):
            raise ValueError("boom")
    assert "retrieve" in latency


def test_latency_sums_to_something_plausible_against_the_wall_clock(store, cfg):
    """End-to-end version of the same contract: the recorded stages must account
    for the real elapsed time, not be 1000x smaller than it."""
    started = time.perf_counter()
    answer = _ask("What is the expense ratio of the HDFC Large Cap Fund?", store, cfg)
    wall_ms = (time.perf_counter() - started) * 1000
    assert answer.latency
    assert sum(answer.latency.values()) <= wall_ms * 1.5


# -- the no-scheme path ---------------------------------------------------
def test_off_topic_question_with_no_scheme_is_class_b_not_class_e(store, cfg):
    """PRD section 5 class B: a fact absent from the five pages gets "not in the
    indexed sources" plus the covered topics.

    It used to be class E, because the scheme check ran before the gate: naming
    no scheme means scheme_id is None, and the disambiguation branch returned
    before anything could look at the retrieval scores. The answer then claimed
    the question was answerable-but-underspecified, and listed five schemes that
    have nothing to do with the subject.
    """
    answer = _ask("What is the SEBI circular number for portfolio disclosure?",
                  store, cfg)
    assert answer.outcome is Outcome.B_NOT_IN_CORPUS
    assert answer.source_url is None
    assert "http" not in answer.text.lower()
    assert "expense ratio" in answer.text.lower(), "must name what it can answer"


def test_in_scope_question_with_no_scheme_is_class_e(store, open_gate):
    """The complementary case, and the reason the gate is not simply unconditional:
    "what is the expense ratio?" IS answerable, it just does not say which of the
    five funds. Refusing it as class B would be a false negative that hides a
    covered fact."""
    answer = _ask("What is the expense ratio?", store, open_gate)
    assert answer.outcome is Outcome.E_NEEDS_SCHEME
    assert answer.source_url is None
    assert "HDFC Large Cap Fund" in answer.text


def test_class_b_never_invents_a_citation(store, cfg):
    """The single most important refusal property. Checked on the low-score path
    and on the empty-result path, because they are different code."""
    for question in ("Who won the 1994 FIFA World Cup?", "How do I cook pasta?"):
        answer = _ask(question, store, cfg)
        assert answer.outcome is Outcome.B_NOT_IN_CORPUS
        assert answer.source_url is None
        assert "groww.in" not in answer.text


# -- invariants that hold on every path -----------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of the HDFC Large Cap Fund?",   # A
        "What is the expense ratio?",                             # E
        "Who won the 1994 FIFA World Cup?",                       # B
        "Should I buy the HDFC Large Cap Fund?",                  # C
        "Which HDFC fund has the best 1-year return?",             # D
    ],
)
def test_every_answer_carries_its_chunks_and_never_raises(store, cfg, question):
    """`pipeline` documents two invariants that apply to refusals too: the answer
    always carries `retrieved_chunks` (so a refusal is auditable rather than
    indistinguishable from one that never looked), and the function never
    raises, because a demo that crashes is a worse outcome than the answer it was
    trying to prevent."""
    answer = _ask(question, store, cfg)
    assert answer.outcome.value in {"A", "B", "C", "D", "E", "error"}
    assert isinstance(answer.retrieved_chunks, list)


def test_a_broken_provider_becomes_an_error_answer_not_an_exception(store, open_gate):
    """Errors are allowed to propagate out of a provider so the pipeline can
    decide what they mean. A dead endpoint must not take the app down, and must
    not produce a confident answer with no source either."""
    answer = _ask("What is the expense ratio of the HDFC Large Cap Fund?", store,
                  open_gate, llm=BrokenLLM())
    assert answer.outcome is Outcome.ERROR
    assert answer.source_url is None
    assert "error" in (answer.reason or "")
    assert answer.retrieved_chunks, "an error answer must still show what was found"


def test_class_a_citation_comes_from_chunk_metadata_not_model_text(store, open_gate):
    """The URL is read from the cited chunk's metadata, so a model that invents a
    plausible link cannot get its URL onto the screen."""
    llm = StubLLM(
        reply="The expense ratio is 1.03%.\n"
              "Source: https://groww.in/mutual-funds/some-fund-i-invented"
    )
    answer = _ask("What is the expense ratio of the HDFC Large Cap Fund?", store,
                  open_gate, llm=llm)
    assert answer.source_url == "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    assert "invented" not in (answer.source_url or "")


def test_class_b_never_generates(store, cfg):
    """The gate refuses before generation, so an out-of-corpus question cannot be
    talked into an answer no matter what the model felt like saying. Refusing
    *after* generating and hoping the text looks like a refusal is a refusal that
    can fail.

    `triage_llm` is on, so one classify call is expected and correct -- the point
    is that no prompt ever asks for an answer."""
    llm = StubLLM()
    answer = _ask("Who won the 1994 FIFA World Cup?", store, cfg, llm=llm)
    assert answer.outcome is Outcome.B_NOT_IN_CORPUS
    assert "generate" not in [name for name, _ in llm.calls], llm.calls


def test_advice_and_performance_short_circuit_before_retrieval(store, cfg):
    """Classes C and D cost no embedding call and cannot be influenced by what
    happens to be indexed."""
    for question, expected in (
        ("Should I buy the HDFC Large Cap Fund?", Outcome.C_ADVICE_REFUSED),
        ("Which HDFC fund has the best 1-year return?", Outcome.D_PERFORMANCE_REFUSED),
    ):
        llm = StubLLM()
        answer = _ask(question, store, cfg, llm=llm)
        assert answer.outcome is expected
        assert answer.source_url is None
        assert llm.calls == [], f"{expected.value} reached the LLM"
        assert answer.triage_layer == "rules", "must be the deterministic layer"


def test_pii_is_scrubbed_before_the_model_sees_the_question(store, cfg):
    """Input boundary, not a later cleanup pass: the scrub runs before triage, so
    the PAN cannot reach the embedder, the LLM, or the trace log."""
    llm = StubLLM()
    answer = _ask("My PAN is ABCDE1234F, what is the expense ratio of the HDFC "
                  "Large Cap Fund?", store, cfg, llm=llm)
    for _name, prompt in llm.calls:
        assert "ABCDE1234F" not in prompt
    assert "ABCDE1234F" not in answer.text
