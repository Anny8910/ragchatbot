"""Shared data contracts for the whole pipeline.

Implements the data model in architecture §5. Nothing else in the codebase
defines a dataclass that crosses a module boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# The six in-scope fact topics. "statement" was removed: "how to download a
# statement" has no source in the brief's five scheme pages (it is an
# investor-portal help-centre topic, not a fund attribute), and it was dropped
# from scope rather than sourced. See docs/data-findings.md.
#
# Note: riskometer and benchmark are two distinct topics from two distinct
# fields (nfo_risk, benchmark_name). docs/PRD.md §4 previously listed them as a
# single "riskometer and benchmark" item, which made "the six topics" an
# unverifiable count. They are enumerated separately here on purpose.
Topic = Literal[
    "expense_ratio", "exit_load", "min_sip", "lock_in",
    "riskometer", "benchmark", "other",
]
SourceTier = Literal["brief", "official_ref"]
SourceStatus = Literal["ok", "fetch_failed", "blocked", "pending"]
TriageLayer = Literal["rules", "llm"]


class Outcome(str, Enum):
    """The four answer classes from PRD §5, plus error."""
    A_ANSWERED = "A"
    B_NOT_IN_CORPUS = "B"
    C_ADVICE_REFUSED = "C"
    D_PERFORMANCE_REFUSED = "D"
    ERROR = "error"


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    publisher: str
    source_tier: SourceTier
    scheme_id: str | None            # "S1".."S5", or None for cross-scheme pages
    scheme_name: str
    factsheet_url: str | None
    aliases: tuple[str, ...]         # exact-match strings for scheme resolution (§14.4)
    fetched_at: str | None = None    # ISO-8601; drives the "Last updated" footer
    snapshot_path: str | None = None
    content_hash: str | None = None
    status: SourceStatus = "pending"


@dataclass(frozen=True)
class Document:
    """A snapshot loaded into a uniform text form (architecture §7)."""
    text: str
    source: Source
    heading: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str                          # "{source_id}::{ordinal}"
    text: str                        # the only text sent to the LLM
    source_id: str
    url: str
    publisher: str
    source_tier: SourceTier
    scheme_id: str | None
    scheme_name: str
    heading: str | None
    topic: Topic
    chunk_index: int
    n_tokens: int                    # word pieces, <= 256 asserted at build time
    fetched_at: str | None
    factsheet_url: str | None
    content_hash: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float                     # cosine similarity in [0, 1]


@dataclass
class Validation:
    sentences_ok: bool = True
    single_url: bool = True
    scheme_match: bool = True
    no_figures: bool = True
    cited_url: str | None = None
    unverified: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class Answer:
    outcome: Outcome
    text: str
    source_url: str | None           # exactly one for A; None for B/C/D/error
    publisher: str | None
    scheme_name: str | None
    retrieved_chunks: list[ScoredChunk]   # populated for every outcome incl. refusals
    top_score: float | None
    k: int
    last_updated: str | None
    validation: Validation
    latency: dict[str, float]
    triage_layer: TriageLayer | None = None
    reason: str | None = None        # machine-readable refusal reason


@dataclass
class TriageResult:
    outcome: Outcome                 # A, B, C, or D
    reason: str                      # matched rule name, or "no_rule_matched"
    layer: TriageLayer | None
    scheme_id: str | None = None     # resolved scheme, if any
    scheme_ambiguous: bool = False   # >1 scheme named in the question
