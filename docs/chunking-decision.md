# Chunking decision and the `RAG_MIN_SCORE` calibration

Deliverable F3. Written during P4 against the real index: 5 sources, 45 chunks,
real `all-MiniLM-L6-v2` vectors, 48 labeled eval rows.

## 1. The chunking strategy

**Section-aware, table-row-group units, target 180 word pieces, 40 overlap, hard
split at 256.** Implemented in `rag_bot/ingest/chunker.py`.

Why this and not fixed-size or sentence-based:

| Criterion | Fixed 180 | Sentence-split | **Chosen** |
|---|---|---|---|
| Label stays with its value | often split apart | yes | yes — a `Label: value` row is one indivisible unit |
| Table row + header kept together | no | fragile | yes, row-groups are the unit |
| Retrieval precision per chunk | diluted by neighbours | good | good |
| `heading` available for topic rules | no | yes | yes |

Two decisions were forced by measurement during P2, not chosen up front:

- **Fees are split per topic.** All three fee rows (expense ratio, exit load,
  minimum SIP) originally shared a `## Fees and charges` heading and therefore one
  chunk. First-match-wins topic assignment then labelled the whole chunk
  `expense_ratio`, and `exit_load` and `min_sip` ended up with **zero** labelled
  chunks. Since `topic` is a retrieval filter dimension, that silently removed two
  of the six topics from filtering. One heading per topic instead.
- **The eval set is written before the chunker is finalised.** §9.3's experiment is
  only meaningful if the labels exist first.

Result: 45 chunks, 9 per source, five for each of the six topics and 15 `other`.
Max **88 word pieces** against the 256 cap.

**Token counts are only meaningful with the real tokenizer.** Measured with a
whitespace stand-in the max chunk was 42 "tokens"; with MiniLM's word-piece
tokenizer it is 88 — sub-word splitting costs ~1.8×. The 180 target therefore has
far more headroom than a whitespace count suggests, which is the right way round.

## 2. Score distribution on the eval set

`retrieve()` → top-1 score, all 48 rows, `RAG_MIN_SCORE` not yet applied:

| outcome | n | min | median | max |
|---|---|---|---|---|
| **A** (answerable) | 33 | **+0.397** | +0.675 | +0.848 |
| **B** (not in corpus) | 5 | +0.364 | +0.621 | **+0.784** |
| C (advice) | 4 | +0.593 | +0.635 | +0.752 |
| D (performance) | 4 | +0.385 | +0.649 | +0.697 |
| E (edge) | 2 | +0.509 | +0.551 | +0.675 |

**A and B overlap completely.** Lowest A = 0.397, highest B = 0.784. There is no
threshold that keeps every answerable question and rejects any not-in-corpus
question, and the two constraints are 0.39 apart in opposite directions.

Worse, the gate as specified has **zero effect on the labeled B rows**: 0 of 5 are
below 0.35, and 0 of 33 A rows are. It neither rescues a B nor costs an A.

## 3. Why the score cannot do this job

`architecture.md` §12 assumed on-topic pairs at 0.45–0.75 and off-topic at
0.05–0.20. That holds only for questions with *no* topical overlap:

| question | score | gate at 0.35 |
|---|---|---|
| How do I cook pasta? | 0.037 | refused ✓ |
| Who won the 1994 FIFA World Cup? | 0.092 | refused ✓ |
| What was the NAV yesterday? | 0.069 | refused ✓ |
| How much money is invested in the small cap fund overall? (AUM) | **0.784** | accepted ✗ |
| Which HDFC fund has the best 1-year return? | **0.690** | accepted ✗ |
| What was yesterday's NAV for the Large Cap Fund? | **0.688** | accepted ✗ |

The pattern is clear: the labeled B/C/D questions are **topically adjacent
financial questions**, not off-topic ones. `AUM` retrieves the minimum-SIP chunk
at 0.78; "best 1-year return" retrieves the benchmark chunk at 0.69. The score
measures semantic similarity, and these questions *are* semantically similar to
the corpus — they are about the same subject matter, asking for a fact the corpus
deliberately does not contain.

**So class B/C/D must be decided from the question, not from the retrieved
chunks.** That is the triage router's job (P6, architecture §13): rule patterns for
performance/NAV/AUM/ranking/advice language, applied *before* the score is
trusted. The score is a floor for questions with no topical overlap at all, and
nothing more.

This is a correction to §12's premise, not to its intent — §13 already puts rules
first. What was wrong was the assumption that the score alone could carry class B.

## 4. The calibration

Constraint from the data:

- never reject an answerable question → `min_score ≤ 0.397`
- reject a not-in-corpus question → `min_score > 0.784` — **impossible**

`RAG_MIN_SCORE` stays at **0.35**. It sits below every A row with 0.047 of margin
and above the non-topical band (0.037–0.092), which is the only work it can do.

**0.40 was considered and rejected.** It would catch exactly one labeled B row
(`b-statement`, 0.364) while leaving only 0.003 of margin below `a-lock-S2`
(0.397) — a valid "Holding period on the flexi cap scheme?" question. Trading a
certain false refusal of a real question for one adjacent-B catch is a bad deal,
and 0.003 of margin would break on any corpus or model drift. Revisit only once
the rule layer handles class B by pattern, at which point an aggressive floor
costs nothing.

The build prints the floor next to the model that produced it, because a cosine
threshold is meaningless without naming the embedding model.

## 5. What fixed the vocabulary gap

Query-side synonym expansion in `retriever.py`, added in P4. The corpus says
"expense ratio" and never "TER", so a correct question scored 0.04 and no
threshold could rescue it.

| query | before | after |
|---|---|---|
| What is the TER? | +0.040 | **+0.550** |
| What are the charges? | +0.093 | **+0.494** |
| any redemption charge? | +0.173 | **+0.464** |
| which index does it track | +0.323 | **+0.458** |

This is the fix the spec asks for instead of moving the threshold: tighten *what is
matched*, not how strict the cutoff is. Expansion is append-only, one-directional
(user wording → corpus wording), deduplicated against the query, and idempotent.

## 6. Retrieval accuracy

Gold-scheme top-1 accuracy on the 31 A rows that name a source: **31/31** with the
scheme filter supplied, and **8/10** unfiltered after the P3 embedding prefix (up
from 3/10 before it). Both misses are S2, whose live name is *HDFC Flexi Cap
Direct Plan-Growth* while the brief calls it *HDFC Equity Fund* — an alias
resolution problem for §14.4, not an embedding problem.

## 7. Open

- The rule layer for class B/C/D is unbuilt (P6). Until it exists, the labeled B
  rows would be answered from adjacent chunks. That is the known worst-case
  failure and the reason P6 is not optional.
- Per-chunking-strategy hit-rate (§9.3's full experiment) is deferred to the eval
  harness (P9); §1 records why the chosen strategy was picked.
