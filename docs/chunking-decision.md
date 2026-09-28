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

`RAG_MIN_SCORE` stays at **0.35**.

Re-measured in P9 over the full 51-row set with the built rule router in place
(`python3 -m rag_bot.eval.run_eval --calibrate`, `all-MiniLM-L6-v2`):

| threshold | in-corpus accepted | out-of-corpus accepted | gap |
|---|---|---|---|
| 0.20–0.36 | 33/33 | 5/18 | 28 |
| **0.38** | **33/33** | **4/18** | **29** |
| 0.40 | 32/33 | 4/18 | 28 |
| 0.52 | 31/33 | 3/18 | 28 |
| 0.60 | 23/33 | 2/18 | 21 |

The widest gap is at 0.38, and it is a gap of exactly one row: it rejects one
out-of-corpus question that 0.35 accepts. The 0.20→0.36 plateau is flat, meaning
the floor does nothing at all across that whole range — exactly as §3 predicted,
now that the rule router handles the adjacent-B cases by pattern.

**0.38 was still not adopted.** The gain is a single row, and it costs the entire
margin below `a-lock-S2` (top-1 0.397) down to 0.017 — inside the noise of any
corpus change. A threshold that buys one refusal by making a correct question
one refactor away from a false refusal is a bad trade, so the number stays where
§4's analysis put it. Recorded because the table above is the measurement, and
"we measured it and declined" is a decision worth being able to audit.

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

## 7. Head-to-head against the alternatives

§1 argues the strategy on mechanism. This is the measurement, over the 33
in-corpus rows, all with the real embedder. "topic hit" = the rank-1 chunk
carries the topic the row asks about; that is the quantity the strategy is
supposed to protect, since `topic` is a retrieval filter dimension.

| strategy | chunks | topic hit | note |
|---|---|---|---|
| **Section-aware, row-group units (chosen)** | **45** | **33/33** | label travels with its value |
| Fixed 180 word-pieces | 61 | 26/33 | splits `Label: value` rows; 7 misses |
| Whole-page per source | 5 | 12/33 | a page holds all six topics; the filter cannot discriminate |

Whole-page is the instructive failure. Five beautiful, complete chunks, and a
topic hit rate below a coin flip, because one chunk per scheme means
`topic=exit_load` retrieves the same object as `topic=expense_ratio`. The cost
of the chosen strategy is more chunks and a larger index; the benefit is that
every topic stays independently addressable, which is what makes the whole
scheme-scoped, topic-filtered design work at all.

Overhead measured directly: the chosen strategy produces 9 chunks per source
against 45, and the index rebuild takes 4.1s versus 2.3s for whole-page.

## 8. Open

- Per-chunking-strategy *end-to-end* class-A accuracy is not measured. §7 compares
  topic hit-rate, which is the mechanism; running all three strategies through
  the full pipeline with a real LLM would cost ~150 API calls per strategy and is
  not yet justified by a 33-row set. The claim "chosen strategy is better" rests
  on §7 and §1, not on an end-to-end number.
- §7's table is measured against a 5-page corpus. Per-candidate topic hit-rate
  will not transfer to a large corpus: at 5 pages a whole-page chunk is already
  20% of the corpus, and the same strategy on 5,000 pages would be 5,000 chunks
  of the same size with the same hit rate. The conclusion "keep the label with its
  value" should transfer; the specific numbers should not be quoted elsewhere.
- `a-bench-S5` ("This hybrid fund's benchmark index?") and the two multi-scheme
  rows are labelled A in the eval set but route to class E, because they name no
  scheme. That is the design working as specified — one scheme per question — and
  the labels are being corrected to E rather than the router being loosened.


## 8. Running the eval: one run per day on the free tier

The Groq free tier enforces several limits at once: 8k tokens per minute, 1k
requests per minute, 200k tokens per day, and a per-model cap. The binding one
for a full eval is the **daily token budget**.

A full 52-row run costs roughly 150k-200k tokens -- two calls per row (triage
classify, then generation) at ~1.5k each, and reasoning tokens count towards the
budget even at `reasoning_effort: "low"`. That is the whole daily allowance, so
**a complete eval can be run about once per day**, and a partial one is not
free: a retried 429 still spends tokens.

Three things about this are worth writing down, because each one cost a run:

1. **The daily cap is not in the response headers.** `x-ratelimit-remaining-tokens`
   reads ~7900 of 8000 and resets in 547ms throughout the run, even while the
   daily budget is spent. Pacing on those headers alone looks correct and gets
   429'd anyway. The only place the daily window appears is the 429 body:
   `on tokens per day (TPD): Limit 200000, Used 199945 ... Please try again in 17.8s`.
2. **A daily 429 must not be retried.** The generic backoff honours the
   `Retry-After` in that same message, so an 8-second reset gets retried and
   fails again, and a 17-hour reset gets retried four times across seven
   minutes of sleeping. `GroqProvider._post_with_backoff` now detects the daily
   window in the body and raises immediately; a per-minute 429 is still retried,
   because that one really does clear in seconds.
3. **Reset durations are compound.** `17h57m7.2s` ends in "s", so a naive
   `float(text[:-1])` reads it as 17 *seconds* -- 3800x too small, silently,
   since the multi-unit branch is never reached. And "547ms" contains an "m",
   so a substring test for the compound form reads it as 547 *minutes*. The
   parser in `rag_bot/eval/run_eval.py` matches each form with its own anchored
   regex; `tests/test_eval_runner.py` pins all four.

### Using the runner

```
# a targeted re-check, which costs a few thousand tokens
python -m rag_bot.eval.run_eval --only a-lock-S2,a-lock-S5

# the full run: do this once, and only when the daily budget can cover it
python -m rag_bot.eval.run_eval
```

The runner paces itself between rows from Groq's own rate-limit headers, tracks
the tokens it has spent against `EVAL_DAILY_TOKENS` (default 200k), and exits 3
with a partial-results note rather than printing a pass table over the rows that
happened to run. The rows it skips are the ones most likely to fail, so a
partial run reported as a pass would be worse than no run.

To pay for more than one run a day, the options are a paid Groq tier, a second
key, or shortening the run (`--only`).
