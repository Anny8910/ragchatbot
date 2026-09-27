# Architecture — Mutual Fund FAQ Assistant (facts-only RAG chatbot)

**Product:** RAG Chatbot — mutual fund FAQ assistant
**Source:** `docs/PRD.md`
**Status:** Draft
**Audience:** Presenter, classmates, instructor

> Read this alongside the PRD. The PRD says *what*; this says *how*, stage by stage, in the
> order the brief mandates: **Loading → Chunking → Embedding → Vector store → Retrieve → Generate**.
>
> Design bias: **the class should be able to watch the constraint be enforced, not be asked to
> trust a prompt.** Every hard rule in the brief — no advice, no performance claims, no PII,
> one source per answer — is implemented as a structural check in code. The prompt asks; the
> pipeline disposes.

---

## 1. What this system is not doing

The brief looks like a RAG demo and is actually a **compliance-flavored retrieval system**. The
interesting engineering is not "call an LLM with some chunks." It is:

1. Keeping performance figures **out of the index entirely** (§8)
2. Deciding *not to answer* three different ways (§13, §6)
3. Stopping figures from leaking into answers and logs (§16)
4. Not confusing five similarly named HDFC schemes with each other (§5.3, §14.4)

A RAG system that only does steps 1–2 of §6 of this document will get class A right and fail
the graded requirements.

---

## 2. Decisions at a glance

Fixed by the brief are marked **[B]**. The rest resolve PRD §13 open questions.

| # | Decision | Choice | Why | Reversible? |
|---|----------|--------|-----|-------------|
| D1 | Embedding model **[B]** | `sentence-transformers/all-MiniLM-L6-v2`, local, CPU | Brief-fixed. No API key, no cost, deterministic | Yes — but changes vector space (see D5) |
| D2 | Vector store **[B]** | ChromaDB `PersistentClient` on disk | Brief-fixed | Partly — `index/store.py` is the only importer |
| D3 | Generation LLM | **Local Ollama** (`llama3.5:8b` / `qwen2.5:7b`) by default; hosted behind the same interface | Resolves Q2. The demo must not fail on network, and the corpus is 5 pages — a 7B model is sufficient | Yes — `RAG_PROVIDER` |
| D4 | Corpus handling | **Snapshot once, index locally.** Pages fetched at build time into `data/corpus/snapshots/`, committed to git | Decouples the demo from live network; makes "Last updated from sources:" a real date; makes the corpus reproducible (Q4) | No — this is the right shape |
| D5 | Collection naming | `mf_faq__{embed_model}__{corpus_version}` | Vectors from two models, or two corpus versions, never mix | Yes |
| D6 | Advice/performance triage | **Pre-retrieval router**, deterministic patterns first, optional LLM classifier as fallback | Refusals must not depend on retrieval or generation working — that's the whole point of a refusal | Yes |
| D7 | Performance data | **Stripped from the corpus at load time**, plus a post-generation figure check | "No performance claims" enforced structurally: the numbers are not in the index to be quoted (§8) | Yes, but the check should stay regardless |
| D8 | PII handling | Redact at the **input boundary**, before embedding, prompting, or logging | F11. The first thing that touches a user query is the scrubber, not the embedder | No |
| D9 | Chunking | **Section-aware, 180-word-piece target, 40 overlap, table-aware, 256 hard cap** — provisional, to be confirmed by the §10 experiment | PRD §6.2 defers this to data inspection; the constraints are known now | Yes — env-configurable |
| D10 | Source list deliverable | The build **generates** `manifest.csv` (D2) from the snapshot step | The deliverable cannot drift from what was actually indexed | No |
| D11 | Chat history | **Off** by default (Q6) | History degrades retrieval and adds failure modes; F16 is P2 | Yes |
| D12 | Citation | Exactly one URL per answer, taken from the **cited chunk's metadata**, not from model text | A URL is looked up, not parsed out of prose | Partly |

**Q1 (groww.in vs AMC/SEBI/AMFI) is deliberately not decided here.** It is a sourcing-policy
question for the instructor, not an engineering one. The architecture stores `publisher` per
source so the answer is correct under either ruling — see §7.2.

---

## 3. System overview

```
BUILD TIME (once, or when sources change) ─────────────────────────────────────────────────┐
│                                                                                          │
│  sources.yaml ──▶ fetch ──▶ snapshot ──▶ sanitize ──▶ chunk ──▶ embed ──▶ Chroma      │
│  (5 URLs +          (once)   (HTML→text,   (strip PII    (section-  (MiniLM   (on disk,  │
│   official refs)              PDFs, dated)  + returns)    aware)     local)    versioned) │
│       │                          │                           │                        │
│       └──────────────────────────┴───────────────────────────┴──▶ manifest.csv = D2      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                                            │ reads (no network)
┌───────────────────────────────────────────────────────────▼──────────────────────────────┐
│ QUERY TIME (per question)                                                                 │
│                                                                                          │
│  question ─▶① PII scrub ─▶② triage router ──▶③ embed ─▶④ top-k ─▶⑤ gate ─▶⑥ generate   │
│                  │              │                        │          │          │          │
│                  │              ├─ class C → refusal + educational link                 │
│                  │              ├─ class D → refusal + factsheet link, no figures      │
│                  │              └─ class A/B? continue                                 │
│                  ▼                                                                           │
│           redacted query                                            ⑦ validate + assemble │
│                                                        ≤3 sentences · 1 URL · footer    │
│                                                                                          │
│  bypassed: class B stops at ⑤ — LLM is never called                                     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Four stages, not six, run on every question — and two of them exist only to refuse.**
The brief's six stages are all present; PII scrubbing and triage are prepended because the
brief's constraints (no advice, no performance claims, no PII) are checkable *before* retrieval
and must be, or they degrade into prompt suggestions.

---

## 4. Module layout

```
app.py                          # Streamlit entry; only file importing streamlit
rag_bot/
├── config.py                   # env parsing + validation; single source of truth
├── pipeline.py                 # stage orchestration, top to bottom, one function per stage
│
├── sources/
│   ├── sources.yaml            # the 5 brief URLs + any official refs, with publisher
│   ├── fetch.py                # stage 1: fetch → snapshot, stamp fetch date
│   ├── sanitize.py             # strip PII patterns + all performance/NAV content
│   └── manifest.py             # build manifest.csv (deliverable D2)
│
├── ingest/
│   ├── loaders.py              # snapshot file → Document(text, url, scheme_id, heading)
│   ├── chunker.py              # stage 3: Document[] → Chunk[] (section + table aware)
│   └── builder.py              # CLI: `python -m rag_bot.ingest.builder --rebuild`
│
├── index/
│   ├── store.py                # Chroma wrapper: upsert, query, count, drop
│   └── ids.py                  # content-addressed ids
│
├── providers/
│   ├── base.py                 # EmbeddingProvider / LLMProvider protocols
│   ├── minilm.py               # local sentence-transformers (brief-fixed)
│   ├── ollama.py               # local generation (D3)
│   ├── openai.py               # hosted generation, optional
│   └── fake.py                 # deterministic stubs for tests, no model download
│
├── retrieve/
│   ├── retriever.py            # stage 4: embed query → top-k with scores
│   └── gate.py                 # stage 5: relevance threshold → class B decision
│
├── answer/
│   ├── triage.py               # stage 2: class C / D detection (D6)
│   ├── prompts.py              # system prompt + numbered context blocks (citation contract)
│   ├── generator.py            # stage 6: LLM call
│   ├── validate.py             # stage 7: sentence count, single URL, scheme match, figures
│   └── assemble.py             # builds the final Answer object incl. "Last updated" footer
│
├── safety/
│   └── pii.py                  # stage ①: PII detection + redaction (D8)
│
├── ui/
│   ├── components.py           # sources, chunk expander, four answer states
│   └── sidebar.py              # index status, k, rebuild/reset
│
└── eval/
    ├── eval_set.yaml           # labeled questions across classes A–D
    ├── run_eval.py             # per-class metrics
    └── classes.py              # the A/B/C/D classifier used by eval AND by tests
```

**Dependency rule:** arrows point one way —
`ui → pipeline → {answer, retrieve, ingest}`, `answer → {safety, providers}`,
`retrieve → {providers, index}`, `ingest → {index, sources}`. Nothing under `ingest/`,
`retrieve/`, or `safety/` imports `ui/` or `app.py`. The eval classifier and the runtime
triage share `eval/classes.py`'s contract, so a question misclassified in the demo is the
same bug the eval suite catches.

---

## 5. Data model

### 5.1 Source

One row per URL in `sources.yaml`. This is the unit the deliverable D2 reports.

| Field | Example | Notes |
|-------|---------|-------|
| `source_id` | `hdfc_large_cap_growth` | Stable slug, used in ids and logs |
| `url` | `https://groww.in/...` | Canonical URL — the thing we cite |
| `publisher` | `groww.in` \| `hdfc.com` \| `sebi.gov.in` \| `amfiindia.com` | Makes Q1 auditable per row |
| `source_tier` | `brief` \| `official_ref` | The 5 brief URLs vs added official pages |
| `scheme_id` | `S1` … `S5` \| `null` | `null` for cross-scheme reference pages |
| `scheme_name` | `HDFC Large Cap Fund – Direct Growth` | For the answer's opening sentence |
| `factsheet_url` | `https://...` | For class D's "link the official factsheet" |
| `fetched_at` | `2026-09-27T10:14:02+05:30` | Drives the "Last updated from sources:" footer |
| `snapshot_path` | `snapshots/hdfc_large_cap_growth.html` | Local copy; the only thing ever parsed |
| `content_hash` | `sha256:…` | Change detection |
| `status` | `ok` \| `fetch_failed` \| `blocked` | A `blocked` row is loud, not silent |

### 5.2 Chunk

| Field | Notes |
|-------|-------|
| `id` | `{source_id}::{ordinal}` — stable across rebuilds |
| `text` | Sanitized chunk text. The only field sent to the LLM |
| `url`, `source_id`, `publisher` | Citation material |
| `scheme_id` | Enables the cross-scheme guard (§14.4) |
| `heading` | Nearest preceding section heading; the citation label |
| `topic` | `expense_ratio` \| `exit_load` \| `min_sip` \| `lock_in` \| `riskometer` \| `benchmark` \| `statement` \| `other` |
| `chunk_index`, `n_tokens` | Position and size |
| `fetched_at` | Carried through to the answer footer |
| `factsheet_url` | Carried through for class D |

`topic` is assigned at chunk time by heading/keyword rules. It earns its place twice: it
routes class-D questions to the right factsheet, and it lets the eval suite report accuracy
per topic instead of one blended number.

**Note (added in P3, from measurement).** `text` is the fact text alone, e.g.
`Expense ratio: 1.03`, and §14.1 puts the scheme name in the citation header above it. That
is correct for the generator but leaves the *retriever* with no scheme signal: the five chunks
of one topic become near-identical strings, and the correct scheme ranked first in only 3 of
10 questions with a score spread of 0.01–0.03. `Store.embedding_text()` therefore embeds
`scheme_name` + `heading` + `text` (8/10, spread 0.10–0.23). The prefix is embedding-only —
`body` carries the clean text back out — so "the only field sent to the LLM" still holds
literally. Consequences: the 256 word-piece cap is checked on the *prefixed* string, and the
embedding cache is keyed on the embedded text rather than `content_hash`.

### 5.3 Answer

```python
Answer(
  outcome      = "A" | "B" | "C" | "D" | "error",
  text,                       # ≤3 sentences for A; refusal copy otherwise
  source_url,                 # exactly one, or None for B/C/D
  publisher, scheme_name,
  retrieved_chunks,           # always populated (teaching + eval)
  top_score, k, query_vector_id,
  last_updated,               # fetched_at of the cited source
  validation,                 # {sentences_ok, single_url, scheme_match, no_figures}
  latency, triage_layer,      # "rules" | "llm" | None
)
```

`retrieved_chunks` is populated **even for classes C and D**, and even for class B where the
gate rejected everything. When the assistant refuses, the class still sees what retrieval
found — which is the difference between "it refused" and "it doesn't work."

---

## 6. Stage 0 — PII scrub (pre-flight)

Runs before anything else touches the query. `safety/pii.py`.

| Pattern | Example | Action |
|---------|---------|--------|
| PAN | `ABCDE1234F` | Replace with `[PAN redacted]` |
| Aadhaar | `XXXX XXXX 1234`, 12 digits | Replace |
| Account number | 8–16 consecutive digits, contextual (`a/c`, `account`) | Replace |
| OTP | 4–6 digits, contextual (`otp`, `code`) | Replace |
| Email | standard pattern | Replace |
| Phone | 10 digits, optional `+91` | Replace |

**The false-positive problem, stated plainly.** A blunt digit regex destroys this product's
core capability: "expense ratio 1.5%", "exit load 1%", "min SIP ₹500" are the answers we
exist to give. So the rules are:

- Digit runs are redacted **only with a PII keyword nearby** or at a PII-specific length
  (10-char PAN, 12-digit Aadhaar), never as a bare `[0-9]{4,}` sweep.
- **Percentages and currency amounts are on an explicit allowlist** and are never redacted.
- Each rule is a separate named function, so tuning one does not disturb the others.
- `pii.py` ships with a fixture table of *must-not-redact* strings (real fee figures from the
  corpus) and a *must-redact* table. Both are asserted in tests. A scrubber that redacts
  "1.5%" has broken the product more thoroughly than one that leaks an email, because it is
  silent.

The redacted string — not the raw query — is what gets embedded, sent to the LLM, written to
logs, and echoed back. If a PII pattern survives, it never leaves `pii.py`.

---

## 7. Stage 1 — Loading and snapshotting

`sources/fetch.py`, run by `python -m rag_bot.sources.fetch`.

```
for each entry in sources.yaml:
    if snapshot exists and content_hash matches and not --refresh:
        skip (idempotent)
    else:
        html = fetch(url)                     # 1 network call per source, ever
        text = html → readable text           # strip nav, footer, cookie banners, ads
        save snapshot with fetched_at stamp
        write manifest row
```

**Why snapshot at all.** Three reasons, in order of importance:

1. The demo cannot fail because groww.in is down, rate-limiting, or redesigning its DOM.
2. "Last updated from sources: 27 Sep 2026" has to mean something. A date captured at fetch
   time is meaningful; a date rendered at query time is decoration.
3. Fee figures change. A committed snapshot makes "our answer was correct as of the fetch
   date" a defensible statement, and the README can say so explicitly.

**Fetch discipline.** Polite, sequential, with a real User-Agent, cached, and `--refresh` to
re-fetch deliberately. No crawling, no link-following beyond an explicitly listed source
(PRD §9 "scope discipline"). A source that fails is recorded as `status: fetch_failed` and the
build continues — but the UI shows a warning banner, because a silently missing scheme is
exactly how a demo ends up citing the wrong fund.

**HTML → text is where quality is won or lost.** A naive `get_text()` produces a wall of
navigation labels and a shredded fee table. Requirements for `loaders.py`:

- Drop `nav`, `header`, `footer`, `aside`, cookie/consent dialogs, and ad containers.
- **Render tables row-wise as `Label: value` lines.** A fee table shredded into fragments is
  unanswerable; a table flattened to "Expense ratio (Direct Growth): 1.05%" is answerable.
  This single rule is worth more to answer quality than any chunking parameter.
- Emit heading structure so `heading` metadata is real rather than inferred.
- Record the `factsheet_url` found on the page for class D.

### 7.2 Publisher and tier, per row

Every chunk and every citation carries `publisher` and `source_tier`. Under either resolution
of Q1 the system is correct: if the instructor rules groww.in unacceptable, the rows are
filtered at index time and only official sources remain, with no code change. The source list
shows a mixed-publisher column rather than a flat list of five URLs, so nothing is passed off
as official when it isn't.

---

## 8. Stage 2 — Sanitization at load (the load-bearing decision)

`sources/sanitize.py`, applied to every snapshot **before** chunking, so the sanitized text is
what enters the index.

### 8.1 Remove performance content

Stripped from the corpus, by rule:

- Return figures and their labels — 1Y/3Y/5Y/_since-inception returns, CAGR, absolute return,
  "returns", "performance", "gain", "% p.a."
- NAV and price series, "current NAV", "as of <date>" price tables
- Fund ranking/percentile/score tables ("ranked 12 of 45", "peer group rank")
- Comparative performance tables and any "best performing"/"top performer" language
- SIP calculator inputs, projections, and portfolio illustrations

Retained and marked: a link to the factsheet, and the sentence "Performance figures are
available in the official factsheet" where the page had a performance section.

**Why strip rather than instruct.** A groww.in scheme page is *mostly* returns — they are the
page's headline content. Two consequences follow, and both are load-bearing:

1. **Class D becomes structurally safe.** With return data absent from the index, an answered
   response *cannot* quote a return figure, because the figure is not retrievable. The
   constraint is enforced by data absence, not by a prompt instruction the model may ignore.
2. **Class B gets cleaner.** "What was the NAV yesterday?" and "How did the fund perform?"
   stop being high-scoring retrievals. Without stripping, those questions match the returns
   section at ~0.8 similarity, sail past the relevance gate, and produce a class-A-looking
   answer full of numbers. Stripping the section makes them fall below the gate, where they
   belong.

Cost: the assistant can no longer answer a question we have deliberately forbidden it to
answer. That is the intended trade.

### 8.2 Remove PII and contact details

Registration prompts, "enter your PAN", agent phone numbers, email addresses, and any
placeholder account field in the scraped page are removed. This is about the *corpus*, not
the user — different concern from §6, same redaction utilities.

### 8.3 Provenance

Sanitization writes a per-source report: rules fired, blocks removed, before/after token
counts. Two reasons: it is a deliverable-adjacent artifact showing the constraint is real, and
over-aggressive stripping shows up immediately as a corpus diff rather than as a mysteriously
unanswerable question.

---

## 9. Stage 3 — Chunking

PRD §6.2 defers the strategy to data inspection. What is *not* deferred are the constraints,
and they are enough to eliminate most of the design space.

### 9.1 Hard constraints

| Constraint | Source |
|------------|--------|
| ≤256 word-piece tokens, or content is silently truncated away | MiniLM `max_seq_length` |
| Never split a fee-table row or a lock-in sentence | PRD §6.2 |
| Every chunk keeps `url` + `heading` | PRD §6.2 (class A and D depend on it) |
| Every chunk keeps `scheme_id` | Guards cross-scheme confusion (§14.4) |

### 9.2 Recommended default (D9), to be confirmed by experiment

- **Section-aware**: cut on headings first, never across them.
- **Target 180 word-pieces, 40 overlap (22%)** — sits under the 256 cap with headroom for
  headings and table flattening, and overlap is generous because a lock-in sentence that
  straddles a boundary is a demo failure.
- **Table-aware**: table-derived text is chunked by row-group, keeping the column header with
  its rows so "Expense ratio" is never separated from its value.
- **Per-scheme isolation**: chunks never span two schemes, even if the page interleaves them.
- **Forward-progress guarantee**: a pathological unit hard-splits and logs, rather than
  looping. A hung build is an unrecoverable demo.

### 9.3 The experiment that settles it

The brief asks for a data-driven decision, so run one and write it up. For each candidate
strategy:

| Candidate | Target | Overlap |
|-----------|--------|---------|
| A. Flat, heading-blind | 180 | 40 |
| B. Section-aware | 180 | 40 |
| C. Section-aware, table-aware | 180 | 40 |
| D. Section-aware, table-aware, smaller | 120 | 30 |
| E. One chunk per section (no size cap) | — | — |

Metric: **topic hit-rate** — for each of the 6 in-scope topics × 5 schemes (30 labeled
questions), does the correct `source_id` appear in the top-k? Plus: max observed chunk length
must be ≤256 for the candidate to be admissible at all.

`eval/eval_set.yaml` exists before chunking is finalized, which is what makes the experiment
possible. Writing the labels first is the whole trick — otherwise the 30 questions get
invented *after* seeing the scores, and the number means nothing.

**E is worth testing precisely because it probably fails the 256 cap** — and that failure is
the evidence that justifies the whole strategy discussion in the write-up.

---

## 10. Stage 4 — Embedding

**[B] `sentence-transformers/all-MiniLM-L6-v2`, local, CPU, 384 dimensions.**

| Property | Value | Consequence |
|----------|-------|-------------|
| Max input | 256 word-pieces | Hard cap on chunk size (§9.1) |
| Similarity | Sentence-level, cosine | Tuned for short sentences; degrades on long passages. Another reason chunks stay near 180 |
| Model download | ~90 MB, once | Must be pre-downloaded and committed to the offline cache, or the demo day needs network |
| Cost / key | None | Retrieval works with zero credentials — the single most useful property for a classroom |

**Truncation must be detected, not assumed.** `minilm.py` asserts `n_tokens(chunk) <= 256` and
raises at build time if a chunk exceeds it. A silently truncated chunk produces an answer that
is confidently missing half its content, which is the worst failure mode in this product.

Embeddings are cached on disk keyed by `content_hash`, so a one-scheme edit re-embeds one
scheme. Re-running the build on an unchanged corpus does zero embedding work.

---

## 11. Stage 5 — Vector store

**[B] ChromaDB `PersistentClient(path=data/index)`.**

- Collection: `mf_faq__all-MiniLM-L6-v2__v1` (D5). Bumping `corpus_version` after any
  sanitization-rule change forces a clean rebuild, which prevents stale pre-sanitization
  vectors from surviving a rule update.
- Metadata per chunk: the §5.2 fields. Chroma metadata values must be scalars, so `topic`,
  `scheme_id`, `tier` are stored as strings and filtered on in queries.
- Query: `collection.query(query_embeddings=[v], n_results=k+2, where=...)`. The
  `where` filter supports the scheme-scoped query (§14.4) and lets a UI dropdown restrict
  retrieval to one scheme.
- `hnsw:space` **set explicitly to `"cosine"`**, not "left at Chroma's default". chromadb
  1.5.9's default is squared L2, for which the correct conversion is `1 - d/2`, not `1 - d`
  (§7.1 of `data-findings.md` has the measurement). The naive `1 - d` on the default metric
  scores an unrelated document at **-1.01**, which would drive every question to class B.
  Being explicit also makes the conversion survive a future default change.
- A `where` clause may carry only **one** operator, so a combined `scheme_id` + `topic`
  filter must be written as `{"$and": [{...}, {...}]}`. Passing both as bare keys raises.
- The stored `document` is the **embedding** text (§5.2 note), which carries the scheme name
  and heading as a prefix; the clean fact text is kept in metadata as `body`.
- The index lives in `data/index/` and is **gitignored** — it is derived state, and a committed
  vector index is an unreviewable binary diff.

---

## 12. Stage 6 — Retrieval and the relevance gate

```
vector     = minilm.embed(redacted_query)
candidates = store.query(vector, n = k + 2)          # overfetch, see below
scores     = 1 - distance                            # cosine
```

`k` default **3**, exposed in the sidebar. Overfetching by 2 lets the gate drop weak candidates
and still return a full `k`.

### 12.1 Gate → class B

| Check | Condition | Outcome |
|-------|-----------|---------|
| Empty index | `count() == 0` | B, reason: "no documents indexed" |
| No candidates | empty result | B, reason: "nothing matched" |
| Absolute score | `top_score < RAG_MIN_SCORE` (default **0.35**, see below) | B, reason: "not covered by the indexed sources" |
| Borderline | `top_score < RAG_MIN_SCORE + 0.05` | Proceed, flag `weak_evidence` in UI |

**Why 0.35 and not the 0.25 used for generic RAG.** MiniLM cosine scores on short, on-topic
question-to-chunk pairs typically land 0.45–0.75, while off-topic pairs sit 0.05–0.20. A 0.25
threshold sits in the empty middle and would wave through weak matches. The exact value is
**calibrated against the eval set at build time and printed at startup with the model name**,
because a similarity threshold is meaningless without the model that produced it — and a
threshold that is wrong in the permissive direction produces confidently wrong financial
figures, which is the worst outcome this system can have.

Relative thresholds ("top result much better than the rest") are explicitly **not** used: with
a 5-scheme corpus, an off-topic question returns five equally bad matches, and a relative rule
happily picks the best of them.

**Correction from P3 measurement (`data-findings.md` §7.3).** The "0.45–0.75 on-topic vs
0.05–0.20 off-topic" figures above do not hold on this corpus. Measured: scheme-naming
questions score 0.60–0.82, but *out-of-domain* questions reach 0.30–0.37 (`stock market tips for
tomorrow` = 0.37) and *in-domain* questions using vocabulary absent from the corpus fall to
0.04 (`What is the TER?` — the pages say "expense ratio", never "TER"). The two distributions
**overlap**, so 0.35 cannot separate them: no threshold will.

Two consequences for Stage 6. First, the score is a weak floor, not the gate's main input —
topic and scheme have to come from the §13/§14 routing, which is the argument for keeping
two-layer triage. Second, vocabulary gaps need an explicit synonym/alias map
(`TER` → `expense ratio`); lowering the threshold instead would also admit `stock market tips`.
The 0.35 default is kept as a starting value, and the eval-set calibration this section
promises is still owed before the number is trusted.

### 12.2 Class B output

Polite, specific, and honest about *why*: name the six covered topics, do not invent a source,
and show the near-miss chunks dimmed in the expander. "I don't have that in the indexed
sources" beats a generic failure, because a class B that looks broken costs more than one that
looks limited.

---

## 13. Stage 7 — Triage router (classes C and D)

`answer/triage.py`, running **before** retrieval (D6).

### 13.1 Why pre-retrieval

A performance question ("which of these has the best 1-year return?") retrieves *extremely
well* — return data is the most prominent content on a fund page. Routed after retrieval, the
generator receives exactly the numbers it must not quote, and the constraint degrades to a
prompt request. Routed before retrieval, the numbers are never in the prompt.

Refusals also become independent of retrieval and generation health: if Chroma or the LLM is
broken on demo day, class C and D still work. For a graded brief where "does it refuse
correctly?" is worth as much as "does it answer correctly?", that is worth a lot.

### 13.2 Two layers, deterministic first

| Layer | Mechanism | Latency | Catches |
|-------|-----------|---------|---------|
| 1. Rules | Named pattern sets with a shared match reason | ~0 ms | The brief's own examples, and most of what a class will ask live |
| 2. LLM classifier | One cheap call, 3-label output, only if layer 1 returns `unsure` | ~200–400 ms | Paraphrases the rules miss |

Layer 1 fires first and its hit is logged as `triage_layer: "rules"`. The four prepared demo
questions are all layer-1 hits, so the refusal demo costs **zero LLM calls** and cannot fail on
a network hiccup. Layer 2 exists so that layer 1's blind spots are a tuning problem rather than
a product boundary, and its usage rate is reported after each demo run so the rule list can be
improved from real traffic.

Rule sets:

- **Class C (advice):** "should I", "which is better for me", "is it safe to", "recommend",
  "good time to", "worth buying", "suitable for me", "portfolio allocation", "how much should"
- **Class D (performance):** "best performing", "1 year return", "returns of", "CAGR",
  "how much did … return", "NAV of", "which performed", "top performer", "ranking"

**A mandatory false-positive test:** the router must not fire on any of the 6 in-scope topic
questions ("What is the ELSS lock-in period?", "What is the exit load?"). Bare keywords like
`risk`, `better`, or `return` are banned from the rule set for exactly this reason — "riskometer"
is an in-scope *fact*, and "return" appears in "returns period" as an exit-load synonym. This
test is in `tests/test_triage.py` and in the eval suite, and it is a release gate, not a
nice-to-have.

### 13.3 Refusal copy

- **Class C:** acknowledge the question type, state the facts-only limit in one sentence, give
  one genuinely relevant educational link (SEBI/AMFI investor education, or the scheme's own
  KIM), and do **not** restate the premise in advisory language. "The ELSS has a 3-year lock-in
  period" is a fact and is fine; "the ELSS is a good choice for tax saving" is advice and is not.
- **Class D:** state that returns are not provided or compared, quote **no figure**, and link
  the official factsheet from the `factsheet_url` metadata.

Both are pre-written templates with the link injected. Not model-generated: a refusal is the
one place where a fixed, reviewable string is strictly better than a plausible one.

---

## 14. Stage 8 — Generation, validation, assembly

### 14.1 Citation contract

Context blocks are numbered and each carries its citation material:

```
[1] HDFC Large Cap Fund – Direct Growth · source: groww.in · section: "Fees and charges"
    <chunk text>

[2] HDFC Equity Fund – Direct Growth · source: groww.in · section: "Exit load"
    <chunk text>
```

System prompt, in substance:

```
You are a facts-only assistant for HDFC mutual fund scheme pages.
1. Answer only from the numbered context. Never use outside knowledge.
2. Maximum 3 sentences. No preamble, no closing pleasantries.
3. Do not give investment advice. If asked whether to buy, hold, or sell, say you only provide facts.
4. Do not state, compare, or estimate returns, NAV, or performance. None are present in the context.
5. End with: "Source: <the single best block number>."
6. If the context does not contain the answer, reply exactly: NOT_IN_INDEX
```

Rule 4 is a belt to §8's braces: the returns are not in the context, and if a future corpus
change puts them there, the prompt still refuses.

### 14.2 Post-generation validation

`answer/validate.py`. Generation is not trusted until these pass:

| Check | Failure action | Requirement |
|-------|----------------|-------------|
| Sentences ≤3 | Truncate at the sentence boundary; if the answer lost its citation, retry once with a stricter instruction | PRD §9 |
| `Source: [n]` present | Re-derive from the top-scoring chunk; if still absent, mark `unverified` and surface the retrieval score | F7 |
| `[n]` is in the sent set | Strip and flag — a citation the model invented | F7 |
| **Exactly one distinct URL** across cited blocks | Keep rank-1; move the rest to the chunk expander, not the answer | F7, D12 |
| **Cited chunk's `scheme_id` == scheme asked about** (§14.4) | Flag `scheme_mismatch`; refuse with a disambiguation prompt | Correctness |
| **No return/NAV figure** in the output (§14.5) | Strip the sentence; if nothing survives, convert to class D refusal | F10 |
| `NOT_IN_INDEX` emitted | Convert to a clean class B, no source | F8 |

Sentence counting must not split on decimal points — "1.05%" and "3.5" are one token, not two
sentences. A naive splitter reports a one-sentence answer as two and truncates it into
nonsense. Use an abbreviation- and decimal-aware splitter, covered by a fixture test.

### 14.3 The "Last updated from sources:" footer

Assembled by `answer/assemble.py`, **not** by the model:

```
The exit load for periods under 7 days is 1%, and NIL after 365 days. Minimum SIP is ₹500.
Source: HDFC Large Cap Fund – Direct Growth
Last updated from sources: 27 Sep 2026
```

The date is the `fetched_at` of the **cited** source. Since there is exactly one source per
answer, this is unambiguous. The footer is appended after validation, so the model cannot
omit, misdate, or contradict it.

### 14.4 The cross-scheme guard

The PRD's own problem statement is that LLMs "mix up figures across similarly named schemes."
Five HDFC schemes make this the most likely factual error in the product, so it gets a
dedicated mechanism:

1. Chunk metadata carries `scheme_id` (S1–S5).
2. An alias table resolves question → `scheme_id`: "HDFC Large Cap", "large cap fund",
   "S1". Matching is **exact against the alias table**, never fuzzy — "equity" appears in both
   "HDFC Equity Fund" (S2) and the category labels of S1 and S4, and a fuzzy match here
   silently attributes a figure to the wrong fund.
3. When a scheme is resolved, retrieval is `where={"scheme_id": "S2"}`-scoped, so wrong-scheme
   chunks cannot enter the context at all.
4. Post-generation, the cited chunk's `scheme_id` is compared to the asked scheme. A mismatch
   is a validation failure, not a cosmetic warning.
5. **Unresolvable scheme** → the assistant asks which of the five schemes, listing them. It
   does not guess across schemes.
6. **Two schemes in one question** ("expense ratio of large cap and ELSS?") → answers the
   primary scheme and says it handles one scheme per question, because the one-source rule
   (F7) and a two-scheme question are in genuine conflict. The conflict is stated rather than
   papered over.

### 14.5 The figure check

A post-generation scan for return/NAV patterns, used as the safety net behind §8:

- Flag a percentage **only when a return/NAV keyword appears in the same sentence** — "1.05%
  expense ratio" is a required answer, "12.4% 1-year return" is not. A blanket number ban
  would break class A, which is most of the product.
- Also flag absolute currency amounts adjacent to return keywords, NAV phrasing, and
  percentile/rank claims.

---

## 15. Configuration

`.env` (gitignored) or real environment variables. Defaults are chosen so a fresh clone runs
with **no API key at all** — D3 makes the local LLM the default, so the whole demo is
credential-free.

| Variable | Default | Effect | PRD |
|----------|---------|--------|-----|
| `RAG_PROVIDER` | `ollama` | `ollama` \| `openai` | D3, Q2 |
| `RAG_LLM_MODEL` | `llama3.5:8b` | Generation model | §6.6 |
| `RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Brief-fixed; changes collection name | **[B]** |
| `RAG_OLLAMA_HOST` | `http://localhost:11434` | Local model endpoint | D3 |
| `RAG_TEMPERATURE` | `0.1` | Near-deterministic, small nudge off greedy | §6.6 |
| `RAG_MAX_SENTENCES` | `3` | Enforced by validation | §9 |
| `RAG_TOP_K` | `3` | Chunks given to the LLM | F15 |
| `RAG_MIN_SCORE` | `0.35` | Class B threshold; **calibrate and print at startup** | F8 |
| `RAG_CHUNK_TOKENS` | `180` | Chunk target, under the 256 cap | F15 |
| `RAG_CHUNK_OVERLAP` | `40` | Overlap word-pieces | F15 |
| `RAG_CORPUS_DIR` | `data/corpus` | Snapshots + manifest | F1 |
| `RAG_INDEX_DIR` | `data/index` | Chroma persist dir | F5 |
| `RAG_SOURCES_FILE` | `rag_bot/sources/sources.yaml` | Source registry | F1 |
| `RAG_TRIAGE_LLM` | `true` | Enable layer-2 classifier | D6 |
| `RAG_PII_REDACT` | `true` | Cannot be disabled in a graded build | F11 |
| `RAG_SANITIZE_PERF` | `true` | Strip returns/NAV at load | F10, D7 |
| `RAG_HISTORY_TURNS` | `0` | Off by default (D11, Q6) | F16 |
| `RAG_LOG_TRACE` | `true` | JSONL traces to `data/logs/` | §9 |

`RAG_PII_REDACT` and `RAG_SANITIZE_PERF` exist for debugging and eval comparison. Neither
should be `false` in a submitted build, and the README says so.

---

## 16. Failure modes

| Condition | Detection | Response | Class |
|-----------|-----------|----------|-------|
| MiniLM model not downloaded | Preflight check | Fail fast with the download command — do not discover this on stage | — |
| Ollama not running | Connection refused at startup | Preflight warning; retrieval still works, generation degrades to showing chunks | — |
| `RAG_PROVIDER=openai` with no key | `config.validate()` | Fail fast, offer the local default | — |
| Source fetch failed / blocked | Status in manifest | Build continues, UI shows a warning banner naming the missing scheme | — |
| Index empty | `count() == 0` | B, "no documents indexed — use Rebuild index" | B |
| No chunk above threshold | Gate | B, with near-miss chunks shown dimmed | B |
| All topics stripped by sanitization | Per-source sanitize report | Loud warning: "sanitization removed 40% of this source" | — |
| Chunk exceeds 256 word-pieces | Assert in `minilm.py` | **Fail the build** — never index a truncated chunk | — |
| Router false positive on an in-scope question | Test gate | Blocks release; logged with `triage_layer` for tuning | — |
| Model returns 5 sentences | Sentence check | Truncate at boundary; retry once | — |
| Model cites a block it wasn't given | Citation check | Strip, mark `unverified` | — |
| Model returns a return figure | Figure check | Strip sentence; convert to class D refusal | D |
| Model answers the wrong scheme | Scheme check | Refuse, ask which scheme | — |
| PII pattern survives scrubbing | Fixture tests + log scan | Blocking test failure | — |
| Two schemes in one question | Alias resolution | Answer primary, state the one-scheme limit | — |
| Generation fails, retrieval fine | Exception | Show chunks + scores, "could not generate an answer" | — |
| Demo needs the video fallback (Q7) | — | Scripted, repeatable run; snapshots make it deterministic | — |

**The asymmetry, deliberately preserved:** retrieval failure means *no answer*; generation
failure means *chunks but no prose*. Handing the class the retrieved chunks when the LLM dies
turns a broken demo into a demonstration of the architecture, which is the point of the
exercise. It also never shows a fabricated citation, because citations come from retrieval
metadata, not from the model.

---

## 17. UI

Tiny, per PRD §7. Three regions:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Facts-only. No investment advice.                                   │
│ Sources: HDFC AMC · 5 schemes · indexed 27 Sep 2026 · 214 chunks     │
│ Try:  "Exit load of HDFC Large Cap?"                                │
│       "ELSS lock-in period?"                                        │
│       "Should I buy the small cap fund?"                            │
├──────────────────────────────────────────────────────────────────────┤
│ You: Minimum SIP for HDFC ELSS Tax Saver Fund?                       │
│                                                                      │
│ ▣ A — ANSWERED                                                       │
│   The minimum SIP is ₹500 per month. There is no lock-in on           │
│   regular purchases; the 3-year lock-in applies to the tax-saving      │
│   section under Section 80C.                                         │
│   🔗 groww.in/mutual-funds/hdfc-elss-tax-saver-…  · HDFC AMC         │
│   Last updated from sources: 27 Sep 2026                             │
│   ▸ Retrieved chunks (3)                                              │
│      ▸ [1] ELSS · "SIP" · 0.71 · 168 wp            ← top match       │
│      ▸ [2] ELSS · "Minimum investment" · 0.58 · 142 wp                │
│      ▸ [3] ELSS · "Section 80C" · 0.44 · 201 wp                      │
│   ⏱ scrub 0.1ms · triage rules · retrieve 27ms · generate 2.1s         │
├──────────────────────────────────────────────────────────────────────┤
│ ▷ C — ADVICE REFUSED   (muted, no source, distinct border)           │
│ ▷ B — NOT IN SOURCES  (muted, lists covered topics)                   │
│ ▷ D — PERFORMANCE REFUSED (no figures, factsheet link only)           │
│                                                                      │
│ status: retrieving → generating → done        [Rebuild] [Reset]       │
└──────────────────────────────────────────────────────────────────────┘
```

The four states are visually distinct by construction: class A carries a source link and a
score, B–D carry a reason chip and no source. A refusal can never be mistaken for an answer,
which matters because a demo where refusal *looks* like an answer has failed the requirement
even if the code is right.

The chunk expander is the teaching surface. Showing the raw sanitized chunk next to the answer
is what makes "retrieve-then-generate" visible — and after §8 it also visibly demonstrates
that the returns were removed before indexing.

---

## 18. Evaluation and testing

### 18.1 Per-class metrics

The PRD's requirements are per-class, so the eval is too. One blended score would hide a
perfect class A and a failing class D.

| Class | Metric | Pass bar |
|-------|--------|----------|
| A | Figure/sentence correctness vs. the sanitized snapshot; cited URL correct; scheme correct; ≤3 sentences; exactly one URL | ≥90% of labeled A questions |
| B | No URL rendered; covered topics listed | 100% — a single fabricated citation here is a graded failure |
| C | Refusal issued; educational link present; no advisory phrasing | 100% |
| D | Refusal issued; **zero** return/NAV figures; factsheet link present | 100% |
| Retrieval | Topic hit-rate: correct `source_id` in top-k | ≥85% per candidate chunking strategy |
| Guard | Router false-positive rate on the 6 in-scope topics | **0** — release gate |
| Guard | PII must-redact / must-not-redact fixtures | 100% — release gate |
| Guard | No chunk >256 word-pieces | 100% — build-time assert |

The three "100%" rows are gates, not targets. A fabricated citation, an advice answer, or a
quoted return figure is a correctness failure, not a metric to trade off.

### 18.2 Test layers

| Layer | Scope | Needs model? |
|-------|-------|--------------|
| Unit | `pii.py` fixtures, sentence splitter on decimals, chunker boundaries, gate thresholds, triage rules, citation parser, scheme alias table | No — `providers/fake.py` |
| Contract | Every provider satisfies both protocols; `triage.classify()` agrees with `eval/classes.py` | No |
| Integration | Snapshot → sanitize → chunk → index → query, asserting the expected `source_id` in top-k | No (fake embedder) |
| Corpus | Assert no sanitized snapshot contains a return/NAV pattern | No |
| Eval | Full `eval_set.yaml` per §18.1, per chunking strategy | Local model |
| Manual | PRD §12 acceptance checklist | Yes |

`providers/fake.py` returns deterministic hashed embeddings and scripted responses, so the
suite runs offline with no model download and no key. A classroom machine must be able to
`pytest` green with no network.

### 18.3 Eval set shape

30 A-class questions (6 topics × 5 schemes) plus 3–4 each of B, C, D, plus 2 multi-scheme and
2 scheme-ambiguous questions. The B/C/D rows carry `forbidden_patterns` so the checks in
§18.1 are mechanical rather than judgement calls.

---

## 19. Observability

One structured event per query, human-readable for the demo, JSONL to `data/logs/`:

```
rag.query  class=A scheme=S3 top_score=0.71 k=3 scrub=clean
           triage=rules(advice) cited=hdfc_elss…::12 url_count=1
           sentences=2 figures=none pii=clean
           latency={embed:31ms, search:4ms, generate:2140ms}
```

Deliberately minimal — no metrics backend, no dashboards (PRD §9). What matters for a graded
demo is being able to say "top score 0.71, chunk S3 paragraph 12, cited URL count 1, figures
none," and having the log agree with the screen.

**Log hygiene:** query logs are redacted at write time by the same `pii.py` used at the input
boundary, and `pii=clean|redacted:<rules>` is recorded. A committed log file is the easiest
way to leak a PAN into a public repo.

**No back-end screenshots** (PRD §10): the six pipeline stages are surfaced as in-app text
panels — chunk lists, scores, and the validation report — so the demo shows the system's own
UI rather than screenshots of internal tooling.

---

## 20. Repository layout and deliverables

```
RAG Chatbot/
├── README.md                  # D3: setup, scope, known limits, as-of date, no-advice notice
├── app.py
├── requirements.txt           # pinned; MiniLM + chromadb + streamlit + ollama client
├── .env.example               # all-optional: the default build needs no key
├── .gitignore                 # .env, data/index/, data/logs/
├── rag_bot/                   # the pipeline
├── data/
│   ├── corpus/
│   │   ├── sources.yaml       # source registry (5 brief + official refs)
│   │   ├── snapshots/         # committed page copies + factsheet PDFs
│   │   └── manifest.csv       # **generated** → D2
│   ├── eval/eval_set.yaml     # the 6 topics × 5 schemes, plus B/C/D rows
│   ├── index/                 # gitignored
│   └── logs/                  # gitignored, redacted
├── deliverables/
│   ├── sample_qa.md           # **generated** → D4, all four classes
│   ├── demo_script.md         # 3-minute walkthrough, video path (Q7)
│   └── disclaimer.txt         # D5, copied verbatim into the UI
├── tests/
└── docs/
    ├── PRD.md
    ├── problemstatement.txt
    ├── architecture.md        # this file
    └── chunking-decision.md   # §9.3 experiment write-up (PRD §6.2 deliverable)
```

**Two deliverables are build outputs, not hand-written documents.** `manifest.csv` (D2) and
`sample_qa.md` (D4) are generated from the same run that built the index, so they cannot
describe a corpus the code did not actually index. D4 is generated with a guarantee that B, C,
and D each appear at least once, because the failure mode for this submission is a Q&A file
containing fifteen variations of "here is the expense ratio."

---

## 21. Traceability

| PRD ref | Requirement | Implemented by | Verified by |
|---------|-------------|----------------|-------------|
| F1 | Scope and snapshot sources | `sources/fetch.py`, `sources.yaml` | Manifest rows; fetch-status check |
| F2 | Load HTML + PDF into one document form | `ingest/loaders.py` (table flattening §7.1) | Integration test; manual table read |
| F3 | Data-justified chunking, keeps URL + heading | `ingest/chunker.py` + `docs/chunking-decision.md` | §9.3 experiment; hit-rate per strategy |
| F4 | Embed locally, `all-MiniLM-L6-v2` **[B]** | `providers/minilm.py` | Collection name; no API key needed |
| F5 | Persist in on-disk Chroma **[B]**; rebuildable | `index/store.py`, `builder.py --rebuild` | One-command rebuild |
| F6 | Context-only answers, ≤3 sentences | `prompts.py` + `validate.py` sentence check | Fixture tests; §18.1 |
| F7 | Exactly one source link | `validate.py` URL-uniqueness + `assemble.py` | §18.1 class A |
| F8 | Refuse out-of-corpus, no fabricated source | `retrieve/gate.py` | §18.1 class B (100% bar) |
| F9 | Refuse advice + educational link | `answer/triage.py` class C | §18.1 class C (100% bar) |
| F10 | Refuse performance, no figures, factsheet link | `sanitize.py` §8 + triage class D + figure check | Corpus test + §18.1 class D |
| F11 | PII scrub before embed/prompt/log | `safety/pii.py` | Must-redact / must-not-redact fixtures |
| F12 | "Last updated from sources:" footer | `assemble.py` §14.3, from `fetched_at` | Snapshot date visible in UI |
| F13 | Tiny UI: welcome, 3 examples, disclaimer | `ui/components.py`, `deliverables/disclaimer.txt` | PRD §12 |
| F14 | Chunk expander; visible `k` and score | `ui/components.py` | Screenshot-free in-app panel |
| F15 | Configurable `k`, chunk size | `config.py` + sidebar | PRD §12 |
| F16 | Multi-turn chat (P2) | Off by default (D11) | Deferred until time allows |
| F17 | Labeled eval set, per-class accuracy | `eval/`, §18.1 | Eval report |
| F18 | Prototype link or ≤3-min video | `deliverables/demo_script.md` | Q7 decision |
| NFR latency | A few seconds end-to-end | Local embed + local LLM; stage timings | Measured p95 |
| NFR no PII | Nothing stored or forwarded | §6 + log redaction §19 | Log scan in CI |
| NFR no performance claims | No figure computed or quoted | §8 removal + §14.5 check | Corpus test + class D bar |
| NFR explainability | Six stages observable | Chunk panels + trace log | 3-minute walkthrough |
| NFR reproducibility | Pinned deps, committed snapshots | `requirements.txt`, `data/corpus/snapshots/` | Clean-clone run |
| NFR no secrets | Keys in `.env`, never committed | `.gitignore`, `.env.example` | `git log` check |
| D2 | Source list of 5 URLs | Generated `manifest.csv` | Row count; publisher column |
| D4 | Sample Q&A, all four classes | Generated `sample_qa.md` | Class coverage assertion |
| D5 | Disclaimer in the UI | `deliverables/disclaimer.txt` | Visible in running app |

---

## 22. Build order

Each phase ends in something demonstrable; no phase can leave a non-running app.

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| 0 | `sources.yaml`, `fetch.py`, snapshots, `manifest.csv` | 5 rows with real `fetched_at`; **D2 done** | 
| 1 | `sanitize.py` + report | Returns/NAV gone from every snapshot; before/after counts logged |
| 2 | `loaders.py` + `chunker.py` + `eval_set.yaml` labels | Fee tables readable as `Label: value`; no chunk >256 wp |
| 3 | `minilm.py`, `index/store.py`, `builder.py` | Index builds; re-run is a no-op; truncation assert holds |
| 4 | `retriever.py` + `gate.py` | Six topics hit correct source; off-topic question scores below threshold |
| 5 | `pii.py` | Both fixture tables pass |
| 6 | `triage.py` | Classes C and D fire; zero false positives on in-scope topics |
| 7 | `prompts.py`, `generator.py`, `validate.py`, `assemble.py` | Class A answers with one URL and ≤3 sentences |
| 8 | `app.py` + UI | Four states distinct; expander works; disclaimer visible; **D5 done** |
| 9 | `eval/run_eval.py`, chunking-decision write-up | Per-class report; **§9.3 decision documented**; **D4 done** |
| 10 | README, demo script, video path | **D1/D3 done**; PRD §12 checklist ticked |

**Phases 0–1 are the ones that can invalidate the project**, and they come first on purpose. If
the pages turn out to be unfetchable (robots, JS-rendered, login-walled), or if sanitization
strips so much that the six topics become unanswerable, that is a scoping conversation with the
instructor — best had in the first hour, not the last. Phases 9–10 are what get cut under
time pressure, and they are presentation rather than correctness.

---

## 23. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **groww.in blocks or JS-renders** (§Q1/Q4) | **Critical — no corpus** | Phase 0 proves it first. Fallback: official AMC/SEBI/AMFI pages, which are the brief's stated preference anyway |
| Sanitization strips too much, leaving topics unanswerable | High | Per-source report with token counts; §9.3 hit-rate catches it; re-add narrowly rather than disabling the rule |
| MiniLM sentence-level similarity misranks across five similar schemes | High | Scheme-scoped retrieval + §14.4 guard; measured by the scheme-correct metric |
| `RAG_MIN_SCORE` mistuned for MiniLM | High — either over-refuses or quotes wrong figures | Calibrate on the eval set, print at startup with model name |
| A demo-day model download or Ollama failure | Medium-high | Pre-download the model, commit the offline cache, preflight checks, hosted fallback |
| Sanitizer misses a returns phrasing | Medium | Corpus test scans every snapshot for return patterns; two-layer defense |
| Triage rules over-trigger on an in-scope question | Medium | False-positive rate is a release gate; layer-2 classifier for the tail |
| Scope creep past the 5 schemes | Medium | `sources.yaml` is the registry; anything not listed requires a deliberate edit |
| Refusal classes shipped as one generic message | Medium | Four distinct outcomes in data model, triage, UI, and eval |
| Instructor rejects groww.in sources (Q1) | Medium | `publisher` + `source_tier` filtering (§7.2) — config, not rewrite |

---

## 24. Deferred / explicitly not building

Decisions, not oversights:

- Hybrid retrieval (BM25 + vector) or a cross-encoder reranker — a real quality win, but a
  second retrieval story to explain, and the scheme guard is the higher-value fix here.
- LLM-as-judge faithfulness scoring — second model call, more latency, its own failure modes.
  The structural validators cover the graded requirements.
- Query rewriting for multi-hop questions — explicitly handled by asking for one scheme at a
  time instead (§14.4.6), which is cheaper and more honest.
- Live re-fetch at query time — directly contradicts D4 and would endanger the demo.
- Streaming tokens — the status indicator covers the wait; streaming complicates validation
  of sentence count and single-URL rules, which matter more.
- Multi-scheme answers — conflicts with the one-source rule; deferred until the instructor's
  ruling on D2 is known.

---

## 25. Open questions carried from the PRD

| # | Question | Architecture impact | Status |
|---|----------|---------------------|--------|
| Q1 | groww.in vs AMC/SEBI/AMFI sourcing | `publisher`/`source_tier` per source; filterable at index time | **Blocking for Phase 0** — needs an instructor answer |
| Q2 | Which LLM | Local Ollama default (D3); hosted behind one interface | Deferred, low risk |
| Q3 | Exactly 5 URLs, or 5 + official refs? | `source_tier` supports both; manifest states the count | Needs an answer before D2 is final |
| Q4 | Fetchable? Current at fetch time? | Phase 0 proves it; snapshots make the as-of date defensible | **Blocking for Phase 0** |
| Q5 | Factsheet PDFs required? | `loaders.py` handles PDF; class D links the factsheet either way | P1, low risk |
| Q6 | Multi-turn chat worth it? | Off by default (D11) | Deferred |
| Q7 | Hosting or video fallback? | Affects phase 10 effort split | Decide early |

**Q1 and Q4 are the two that can stop the project.** Both are answered by Phase 0 in under an
hour, which is why Phase 0 is first and why it is a gate rather than a task.

---

## 26. Document history

| Date | Change |
|------|--------|
| 2026-09-27 | Initial architecture for a generic class-demo RAG chatbot (hosted embeddings, generic corpus). Superseded. |
| 2026-09-27 | Rewritten for the mutual fund FAQ assistant per the rewritten PRD. Fixed embeddings to `all-MiniLM-L6-v2` and store to ChromaDB per the brief; added the 256-word-piece cap as a chunking constraint; added the pre-retrieval triage router for the advice and performance classes; added performance-stripping at load time as the structural basis for the no-claims rule; added the PII scrubber at the input boundary, the cross-scheme guard, the single-source citation contract, the "last updated" provenance chain, per-class eval gates, and deliverable-generating build steps. |
