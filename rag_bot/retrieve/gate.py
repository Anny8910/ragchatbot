"""The class-B relevance gate: the only place the LLM is skipped.

Checks in the order architecture 12.1 specifies. Absolute threshold only.

There is deliberately NO relative or margin-based rule here. With a five-scheme
corpus an off-topic question returns five equally bad matches, and a rule of the
form "the top result is much better than the rest" cheerfully picks the best of
them. A confidence ratio measures how confidently the retriever picked the wrong
thing.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag_bot.types import ScoredChunk

# A candidate this close to the threshold is accepted but flagged, so the UI can
# mark the answer as weakly evidenced instead of presenting it as solid.
WEAK_MARGIN = 0.05

# Topic label and the covered-topic list for the class-B copy live in
# rag_bot/answer/refusals.py, which is the single source of truth. They used to be
# defined here as well, which meant two copies of the same user-facing string and
# a stale-copy bug when the topics changed. The re-export keeps the existing
# `from rag_bot.retrieve.gate import DEFAULT_COVERED_TOPICS` call sites working.
from rag_bot.answer.refusals import (  # noqa: F401
    DEFAULT_COVERED_TOPICS,
    TOPIC_LABELS,
    refusal_b,
)


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str | None        # None | "empty_index" | "no_candidates" | "low_score"
    message: str | None       # user-facing class-B copy; None when accepted
    weak_evidence: bool
    top_score: float | None

    @property
    def is_class_b(self) -> bool:
        return not self.accepted


def _class_b_message(covered_topics: list[str]) -> str:
    """Name the covered topics. Must never contain a URL.

    A class-B answer that cites a source would be worse than no answer: it would
    point the user at a page that does not contain what they asked for.
    """
    return refusal_b(covered_topics)[0]


def evaluate(
    chunks: list[ScoredChunk],
    *,
    min_score: float,
    index_empty: bool,
    covered_topics: list[str],
) -> GateResult:
    """Decide whether to answer or refuse as class B. Never raises."""
    message = _class_b_message(covered_topics)

    if index_empty:
        return GateResult(
            accepted=False, reason="empty_index",
            message="No sources are indexed yet, so I can't answer anything yet.",
            weak_evidence=False, top_score=None,
        )

    if not chunks:
        return GateResult(
            accepted=False, reason="no_candidates", message=message,
            weak_evidence=False, top_score=None,
        )

    top = chunks[0].score

    if top < min_score:
        return GateResult(
            accepted=False, reason="low_score", message=message,
            weak_evidence=False, top_score=top,
        )

    return GateResult(
        accepted=True, reason=None, message=None,
        weak_evidence=top < min_score + WEAK_MARGIN, top_score=top,
    )
