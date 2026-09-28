"""End-to-end orchestration (architecture 14.1, phase P7).

One function per stage, in one fixed order:

    1. scrub PII (P5)          -> redacted question
    2. triage (P6)             -> C or D short-circuits here, no retrieval
    3. resolve scheme (P6)     -> scheme_id or ambiguity
    4. retrieve (P4), scheme-scoped
    5. gate (P4)               -> class B short-circuits, the LLM is never called
    6. generate (P7)
    7. validate + assemble (P7)

Two invariants hold on every path, including every error path:

- **Every `Answer` carries `retrieved_chunks`**, refusals included. A class D
  refusal that shows what retrieval found is auditable; one that hides it is
  indistinguishable from a refusal that never looked.
- **The function never raises.** It returns an `Answer` with `outcome=ERROR`. A
  pipeline exception would take the demo down, which is a worse outcome than any
  answer it was trying to prevent.

The scheme resolution in step 3 is folded into step 2: P6's `classify()` resolves
the scheme on *every* path, including class A, and re-resolving would only risk
two sources of truth disagreeing.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from rag_bot.answer.assemble import assemble
from rag_bot.answer.generator import generate_answer, retry_with_stricter_instruction
from rag_bot.answer.prompts import build_disambiguation
from rag_bot.answer.refusals import (
    DEFAULT_COVERED_TOPICS,
    refusal_c,
    refusal_d,
)
from rag_bot.answer.triage import classify
from rag_bot.answer.validate import validate
from rag_bot.config import Config
from rag_bot.index.store import Store
from rag_bot.providers.base import EmbeddingProvider, LLMProvider
from rag_bot.retrieve.gate import evaluate
from rag_bot.retrieve.retriever import retrieve
from rag_bot.safety.pii import scrub
from rag_bot.types import Answer, Outcome, ScoredChunk, Source, TriageLayer, Validation

logger = logging.getLogger(__name__)

# Generation cap for a three-sentence answer. Not a config key: the phase ground
# rules forbid inventing one, and an 8B model needs a ceiling or a confused
# generation runs to its context limit and the demo waits.
_MAX_TOKENS = 256

_ERROR_TEXT = (
    "Something went wrong while answering that. The retrieved sources are listed "
    "below, but I can't give you a verified answer from them right now."
)


@contextmanager
def _stage(latency: dict[str, float], name: str) -> Iterator[None]:
    """Record a stage's wall time, including when it raises."""
    start = time.perf_counter()
    try:
        yield
    finally:
        latency[name] = latency.get(name, 0.0) + (time.perf_counter() - start)


def _source_for(sources: list[Source], scheme_id: str | None) -> Source | None:
    for source in sources or []:
        if source.scheme_id == scheme_id:
            return source
    return None


def _error_answer(
    chunks: list[ScoredChunk],
    *,
    cfg: Config,
    latency: dict[str, float],
    reason: str,
    triage_layer: TriageLayer | None,
    exception: Exception | None = None,
) -> Answer:
    """The never-raise landing spot. Carries `chunks` so the demo can still show
    what retrieval found, and cites nothing, because a citation is derived from
    chunk metadata and a failed run has no verified one."""
    if exception is not None:
        logger.warning("answer_question failed: %s: %s",
                       type(exception).__name__, exception)
    return assemble(
        Outcome.ERROR, _ERROR_TEXT, chunks,
        k=cfg.top_k,
        top_score=chunks[0].score if chunks else None,
        validation=Validation(unverified=True, notes=[reason]),
        latency=latency,
        triage_layer=triage_layer,
        reason=reason,
    )


def answer_question(
    question: str,
    *,
    cfg: Config,
    store: Store,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    sources: list[Source],
) -> Answer:
    """Answer one question. Never raises; returns an Answer for every input."""
    latency: dict[str, float] = {}
    chunks: list[ScoredChunk] = []
    triage_layer: TriageLayer | None = None

    try:
        # -- 1. scrub PII (P5). Input boundary, before anything reads the text.
        with _stage(latency, "scrub"):
            redacted = scrub(question).text if cfg.pii_redact else (question or "")

        # -- 2. triage (P6). C and D short-circuit before retrieval, so a refusal
        #    costs no embedding call and cannot be influenced by what was indexed.
        with _stage(latency, "triage"):
            triage = classify(redacted, sources=sources, provider=llm,
                              use_llm=cfg.triage_llm)
        triage_layer = triage.layer

        if triage.outcome is Outcome.C_ADVICE_REFUSED:
            with _stage(latency, "assemble"):
                text, _url = refusal_c(redacted)
            return assemble(Outcome.C_ADVICE_REFUSED, text, [],
                            k=cfg.top_k, top_score=None, validation=Validation(),
                            latency=latency, triage_layer=triage_layer,
                            reason=triage.reason)

        if triage.outcome is Outcome.D_PERFORMANCE_REFUSED:
            # The factsheet link goes in the refusal TEXT, never in source_url:
            # F7 gives class D no source, and a source_url on a refusal is what
            # makes a refusal mistakable for a sourced answer.
            source = _source_for(sources, triage.scheme_id)
            with _stage(latency, "assemble"):
                text, _url = refusal_d(
                    source.scheme_name if source else "",
                    source.factsheet_url if source else None,
                )
            return assemble(Outcome.D_PERFORMANCE_REFUSED, text, [],
                            k=cfg.top_k, top_score=None, validation=Validation(),
                            latency=latency, triage_layer=triage_layer,
                            reason=triage.reason)

        # -- 3. scheme, resolved by classify() above. Unresolvable or ambiguous
        #    means scheme_id is None, which becomes class E below.
        scheme_id = triage.scheme_id
        if scheme_id is None:
            # 14.4.5: an unresolvable scheme gets a question back, not a figure
            # from whichever of the five funds happened to rank first. Retrieval
            # still runs so the demo can show what WAS found, and the Answer
            # carries those chunks for the same reason every refusal does.
            with _stage(latency, "retrieve"):
                chunks = retrieve(redacted, k=cfg.top_k, scheme_id=None,
                                  embedder=embedder, store=store)

            # The gate runs here too, not only on the answered path. Asking
            # "which scheme?" about something the corpus does not cover at all
            # ("what is the SEBI circular number for portfolio disclosure?")
            # names no scheme, so the E branch below would claim the question
            # was answerable-but-underspecified. PRD section 5 class B is the
            # correct outcome: the fact is not in the five pages, and a
            # disambiguation prompt would be a refusal that pretends the answer
            # is one question away. Retrieval scored below the floor is the
            # evidence for that, and it is the same evidence the answered path
            # refuses on -- one rule, two call sites.
            with _stage(latency, "gate"):
                gate = evaluate(chunks, min_score=cfg.min_score,
                                index_empty=store.count() == 0,
                                covered_topics=list(DEFAULT_COVERED_TOPICS))
            if gate.is_class_b:
                return assemble(Outcome.B_NOT_IN_CORPUS, gate.message or "", chunks,
                                k=cfg.top_k, top_score=gate.top_score,
                                validation=Validation(), latency=latency,
                                triage_layer=triage_layer, reason=gate.reason)

            names = [s.scheme_name for s in sources if s.scheme_id]
            with _stage(latency, "assemble"):
                return assemble(Outcome.E_NEEDS_SCHEME,
                                build_disambiguation(names), chunks,
                                k=cfg.top_k,
                                top_score=chunks[0].score if chunks else None,
                                validation=Validation(), latency=latency,
                                triage_layer=triage_layer, reason="no_scheme")

        # -- 4. retrieve, scheme-scoped so wrong-scheme chunks cannot enter.
        with _stage(latency, "retrieve"):
            chunks = retrieve(redacted, k=cfg.top_k, scheme_id=scheme_id,
                              embedder=embedder, store=store)

        # -- 5. gate. Class B short-circuits here: the LLM is never called, so a
        #    low-scoring question cannot be talked into an answer.
        with _stage(latency, "gate"):
            gate = evaluate(chunks, min_score=cfg.min_score,
                            index_empty=store.count() == 0,
                            covered_topics=list(DEFAULT_COVERED_TOPICS))
        if gate.is_class_b:
            return assemble(Outcome.B_NOT_IN_CORPUS, gate.message or "", chunks,
                            k=cfg.top_k, top_score=gate.top_score,
                            validation=Validation(), latency=latency,
                            triage_layer=triage_layer, reason=gate.reason)

        # -- 6. generate
        with _stage(latency, "generate"):
            raw = generate_answer(llm, redacted, chunks,
                                  temperature=cfg.temperature,
                                  max_tokens=_MAX_TOKENS)

        # -- 7. validate + assemble
        with _stage(latency, "validate"):
            text, validation, cited = validate(
                raw, chunks, limit=cfg.max_sentences, asked_scheme_id=scheme_id)

            # One retry when truncation cost the answer its citation (14.2).
            if (not validation.sentences_ok and validation.unverified
                    and cited is None and raw.strip()):
                with _stage(latency, "generate_retry"):
                    raw = retry_with_stricter_instruction(
                        llm, redacted, chunks, temperature=cfg.temperature,
                        max_tokens=_MAX_TOKENS)
                if raw.strip():
                    text, validation, cited = validate(
                        raw, chunks, limit=cfg.max_sentences,
                        asked_scheme_id=scheme_id)

        outcome, reason = _classify_validation(text, validation)
        with _stage(latency, "assemble"):
            return assemble(outcome, text, chunks,
                            k=cfg.top_k,
                            top_score=chunks[0].score if chunks else None,
                            validation=validation, latency=latency,
                            triage_layer=triage_layer, reason=reason)

    except Exception as exc:  # noqa: BLE001 - the invariant is "never raises"
        return _error_answer(chunks, cfg=cfg, latency=latency,
                             reason=f"error:{type(exc).__name__}",
                             triage_layer=triage_layer, exception=exc)


def _classify_validation(text: str, validation: Validation) -> tuple[Outcome, str]:
    """Decide the final class from the validation result.

    Only two validation outcomes downgrade an answer, and both are the safe
    direction. Everything else ships as class A with the flag attached, because
    `Validation` exists to surface a caveat to the UI, not to silently discard a
    correct answer:

    - **No text left at all.** Either the model said NOT_IN_INDEX, or every
      sentence was stripped for a return figure. Those are different problems, so
      they are checked in that order: a NOT_IN_INDEX means class B (the fact is
      genuinely not in the sources), an emptied figure scan means class D (the
      model wanted to discuss performance).
    - **Scheme mismatch.** Downgraded to class B rather than shipped. The PRD's
      named failure is an LLM "mixing up figures across similarly named schemes",
      and the safest response to a figure that may belong to the wrong fund is to
      not show it. Five HDFC schemes make this the most likely factual error in
      the product, and scheme-scoped retrieval already makes it rare.

    A missing citation is NOT a downgrade. 14.2 says to re-derive the URL from the
    top-scoring chunk and mark the answer `unverified` if it is still absent, so
    that is exactly what `validate` did and the answer ships with the flag.
    """
    if not text:
        if validation.no_figures:
            return Outcome.B_NOT_IN_CORPUS, "not_in_index"
        return Outcome.D_PERFORMANCE_REFUSED, "all_sentences_removed_for_figures"
    if not validation.scheme_match:
        return Outcome.B_NOT_IN_CORPUS, "scheme_mismatch"
    return Outcome.A_ANSWERED, None
