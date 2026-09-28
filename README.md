# HDFC mutual fund FAQ assistant

A Streamlit chatbot that answers questions about five HDFC mutual fund schemes
from a fixed, auditable corpus. It is facts-only: it states what the retrieved
sources say, cites exactly one of them, and refuses anything it cannot ground.

Five schemes, from the brief (S2) rather than the PRD's fund names. HDFC has
rebranded S2 to "Flexi Cap"; the corpus carries the current name and the alias
table covers both.

| id | corpus name | alias also accepted |
|----|-------------|---------------------|
| S1 | HDFC Large Cap Fund – Direct Growth | large cap |
| S2 | HDFC Flexi Cap Direct Plan-Growth | equity, flexi cap |
| S3 | HDFC ELSS Tax Saver Fund – Direct Growth | elss, tax saver |
| S4 | HDFC Small Cap Fund – Direct Growth | small cap |
| S5 | HDFC Balanced Advantage Fund – Direct Growth | balanced advantage |

## Answers have to be one of five things

This is the core constraint, and most of the code exists to enforce it. Every
question lands in exactly one class, and the class determines the shape of the
answer.

| class | when | what the user gets |
|-------|------|--------------------|
| A | a fact in the corpus | ≤3 sentences, one source URL, `Last updated from sources:` |
| B | a fact that is not in the corpus | what is covered, and an explicit "I don't have that" |
| C | advice ("should I", "which is better") | refusal, no figures, link to the AMC |
| D | returns, NAV, AUM, rankings ("best performing") | refusal, link to the factsheet |
| E | a real question, no resolvable scheme | a clarifying question, never a guess |
| error | the pipeline raised | an apology; sources still shown |

B, C, D and E are correctness gates: one failure each fails the eval. A is a
quality bar at 90%.

Two things that look like bugs and are not:

- **A B question that names a scheme is expected to reach the answered path.**
  Whether a fact is in the corpus is a property of the retrieved chunks, not of
  the question, so it is decided by the relevance gate or by the model returning
  `NOT_IN_INDEX`. The gate is what makes B a decision rather than a guess.
- **Refusals name what they are refusing.** A class D answer says "I don't state
  returns, NAV or performance figures", so a check for the word "nav" in a
  refusal is a false positive, not a leak.

## Running it locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Build the index from the committed snapshots (~1 min, downloads the embedder).
python -m rag_bot.ingest.builder

# With a hosted provider:
export RAG_PROVIDER=groq
export GROQ_API_KEY=...          # or put these in .env, which is gitignored
export GROQ_MODEL=openai/gpt-oss-120b

streamlit run app.py
```

With no `.env` at all it talks to a local Ollama daemon instead, which must be
running (`ollama serve`). Every `RAG_*` setting has a working default in
`rag_bot/config.py`; see `.env.example` for the full list.

## Tests and the eval

```bash
pytest -q                              # 580 tests
python -m rag_bot.eval.run_eval         # 52 rows against the real pipeline
python -m rag_bot.eval.run_eval --only a-lock-S2,a-lock-S5   # targeted re-check
```

`run_eval` writes `deliverables/sample_qa.md`. It refuses to write that file
unless the run produced at least one B, C and D row, because a Q&A file showing
only answered questions misrepresents the system.

**The Groq free tier allows 200k tokens per day and a full eval run costs
roughly that much, so a complete eval runs about once per day.** The runner paces
itself from Groq's own rate-limit headers, tracks what it has spent, and exits 3
with a note rather than printing a pass table over the rows that happened to run.
`docs/chunking-decision.md` §8 covers why the daily cap is not visible in the
response headers and what that cost to discover.

## Deploying to Render

The service is `render.yaml` in this repo. In the dashboard: **New → Blueprint**,
point it at the repository, and set `GROQ_API_KEY` under the service's
Environment tab. The key is deliberately absent from `render.yaml` because
blueprint values are committed.

`plan: starter` is not optional. The free tier is 512MB; this image loads torch,
sentence-transformers and a Chroma client before answering its first question, so
on free it starts, loads the embedder, and is OOM-killed during the first
question — which looks deployed and then fails under use. If it is still killed
at first question, go to `standard` (2GB).

The container builds its own index from the committed snapshots during the build,
so a cold start never reaches the Hugging Face Hub. The build fails if the index
comes out with fewer than 40 chunks, rather than leaving a running service with
nothing to retrieve from.

To build the image locally:

```bash
docker build -t hdfc-faq .
docker run --rm -p 10000:10000 -e RAG_PROVIDER=groq -e GROQ_API_KEY=... hdfc-faq
```

## Layout

```
app.py                      Streamlit entrypoint
rag_bot/config.py           every RAG_* setting, with defaults
rag_bot/ingest/             snapshot -> allowlisted fields -> chunks
rag_bot/index/              Chroma store
rag_bot/retrieve/           query expansion, retrieval, the class-B gate
rag_bot/answer/             triage, prompts, generation, validation
rag_bot/safety/             PII scrubbing, performance-figure detection
rag_bot/pipeline.py         the five outcomes, end to end
rag_bot/eval/               the 52-row eval set, runner and report
docs/                       PRD, architecture, chunking decision, findings
```

## Reading further

`docs/architecture.md` is the design and the reasoning behind it, including the
trade-offs that were measured rather than guessed. `docs/chunking-decision.md`
records the chunking experiments and the `RAG_MIN_SCORE` calibration.
