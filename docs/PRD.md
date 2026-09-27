# Product Requirements Document

**Product:** Mutual Fund FAQ Assistant (facts-only RAG chatbot) — RAG Chatbot
**Source:** `docs/problemstatement.txt` (mutual fund FAQs milestone brief, 2026-09-27)
**Status:** Draft
**Audience:** Presenter, classmates, instructor

> Supersedes the earlier intent-based draft, which was written while
> `problemstatement.txt` was empty. This version is derived from the actual milestone
> brief. Where the brief leaves a decision open, it is listed in §13 rather than guessed.

---

## 1. Problem

Retail investors asking factual questions about mutual fund schemes — expense ratio, exit
load, minimum SIP, ELSS lock-in, riskometer, benchmark, how to download a statement — get
answers from search results, broker apps, and forum posts that vary in accuracy and are
frequently outdated. Generic LLM chatbots make it worse: they answer confidently from
training data, mix up figures across similarly named schemes, and blur the line between a
**fact** and **investment advice**.

The gap this project fills is narrow and deliberate: a **facts-only** assistant that answers
strictly from a small, scoped set of official public pages, cites exactly one source per
answer, and declines to advise.

---

## 2. Goals

### Demo success

1. Scope one AMC and 3–5 schemes; build a working prototype over those pages.
2. Answer the six named fact types correctly, each with **one** source link.
3. Refuse an opinionated question politely, with a relevant educational link.
4. Refuse a returns/performance question **without** quoting any figure, and point to the
   official factsheet instead.
5. Show the RAG stages in order: **Loading → Chunking → Embedding → Vector store → Retrieve → Generate**.
6. Deliver a working link, or a ≤3-minute demo video if hosting is not possible.

### Non-goals (this build)

- Investment advice, buy/sell recommendations, portfolio allocation, goal planning
- Computing, comparing, or ranking returns or fund performance
- Storing or processing any PII (PAN, Aadhaar, account numbers, OTP, email, phone)
- Web search or live crawling at query time
- Multi-user accounts, payments, production SLAs, or model fine-tuning

---

## 3. Users

| User | Job to be done |
|------|----------------|
| Retail investor comparing schemes | Get a reliable, sourced figure for one scheme's terms, fast |
| Support / content team | Answer repetitive scheme FAQs consistently, without inventing figures |
| Instructor | See a scoped RAG system handle in-domain, out-of-domain, and *unsafe-to-answer* questions correctly |

**Primary flow:** ask a factual question → retrieve top-k chunks → generate an answer of ≤3
sentences using only those chunks → show one source link → append "Last updated from sources: <date>".

**Secondary flows:** inspect retrieved chunks; rebuild the index; reset it.

---

## 4. Scope: AMC and schemes

One AMC: **HDFC Asset Management**. Five schemes, one per URL in the brief.

| # | Scheme | Category | Plan | Source URL |
|---|--------|----------|------|-----------|
| S1 | HDFC Large Cap Fund | Large cap | Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| S2 | HDFC Equity Fund | Flexi cap | Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| S3 | HDFC ELSS Tax Saver Fund | ELSS (tax) | Direct Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| S4 | HDFC Small Cap Fund | Small cap | Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| S5 | HDFC Balanced Advantage Fund | Balanced Advantage (hybrid) | Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-plan-growth` |

**In-scope topics — six, enumerated so the count is verifiable:**

1. expense ratio · 2. exit load · 3. minimum SIP · 4. ELSS lock-in period ·
5. riskometer · 6. benchmark

> **Two corrections to this section (2026-09-27), both evidence-based — see
> `docs/data-findings.md`.**
>
> *The brief named seven facts, not six.* The original wording here read "the six
> named in the brief" and then listed "riskometer **and** benchmark" as one item, so
> the list held seven. Riskometer and benchmark are two distinct facts from two
> distinct fields (`nfo_risk`, `benchmark_name`) and are enumerated separately, which
> is why the `6 topics × 5 schemes = 30` class-A eval count in `docs/architecture.md`
> and `docs/implementation.md` holds.
>
> *"How to download a statement" has been dropped from scope.* It is absent from all
> five pages: zero of the 97 data fields on any page mentions a statement, download,
> or capital gains. It is an investor-portal help-centre topic, not a fund attribute.
> Dropped rather than sourced from an invented page, and recorded as a known limit.

**Corpus sources to collect:** the pages above plus supporting official material — factsheets,
KIM/SID, scheme FAQ pages, fee-and-charges pages, riskometer/benchmark notes, and
statement/tax-document guides, from AMC, SEBI, or AMFI.

> **Sourcing — RESOLVED 2026-09-27 (§13-Q1).** The brief says to collect pages "from
> AMC/SEBI/AMFI" and forbids third-party blogs, but the five URLs it supplies are on
> **groww.in**, a broker/aggregator rather than the AMC or a regulator. **Answer: the
> groww.in pages are acceptable sources.** They are the primary corpus, they remain
> `source_tier: brief` rather than `official_ref`, and every row in the manifest
> discloses `publisher` so no citation is presented as official when it is not.
> Groww is neither a blog nor a forum: it is a regulated broker publishing
> standardised scheme data, which is materially different from the third-party
> commentary the brief's ban is aimed at. Any supporting page that *is* an official
> AMC/SEBI/AMFI document uses `source_tier: official_ref`.

---

## 5. Answer outcomes

The single most important product surface. A question resolves to exactly one of four
classes, and the UI must make them visually distinguishable.

| Class | Trigger example | Required behaviour |
|-------|-----------------|--------------------|
| **A — Answered** | "Exit load of HDFC Large Cap?" | ≤3 sentences, grounded only in retrieved chunks, **exactly one** source link, "Last updated from sources: <date>" |
| **B — Not in corpus** | "What is the NAV of yesterday?" / a fact absent from the 5 pages | Polite "not in the indexed sources", no source link invented, suggest the topics that *are* covered |
| **C — Advice refused** | "Should I buy the ELSS?" / "Is small cap better for me?" | Polite facts-only refusal naming the limit, plus one relevant **educational** link (e.g. a SEBI investor-education page). Must not restate the user's premise as a recommendation |
| **D — Performance refused** | "Which of these has the best 1-year return?" | State that the assistant does not provide or compare returns, quote **no figures**, link to the official factsheet |

Class D is deliberately separate from class C. "Should I buy" is advice; "which performed
best" is a performance claim. Collapsing them into one refusal would let a performance
question through the "no advice" path while still producing numbers.

---

## 6. RAG pipeline requirements

All six stages must be implemented and demonstrable. Stage order is fixed by the brief.

### 6.1 Loading

- Fetch the scoped pages **once**, ahead of time, and store local snapshots
  (HTML→text, plus PDFs such as factsheets).
- Record, per source: canonical URL, fetch timestamp, and local snapshot path.
- **Rationale:** the demo must not depend on live network at query time, and citations need a
  real "last updated" date. Snapshots are committed to the repo; the index is derived from them.

### 6.2 Chunking

- Strategy is **to be decided after inspecting the real data**, per the brief — not fixed here.
- Constraints the chosen strategy must satisfy:
  - Chunks must not split a fee table row or a lock-in sentence across boundaries.
  - Every chunk must retain its **source URL and section heading** as metadata, because class A
    and D both depend on a resolvable citation.
  - **Embedding-model limit:** `sentence-transformers/all-MiniLM-L6-v2` truncates at **256
    word-piece tokens**. Chunk targets must sit below that, or content is silently dropped from
    the vector. This rules out the 500–800 token sizing used in generic RAG demos.
- Output of this stage: a written chunking decision (rationale + measured effect), since the
  brief makes it an explicit deliverable step.

### 6.3 Embedding

- Model: **`sentence-transformers/all-MiniLM-L6-v2`** (fixed by the brief).
- Runs **locally**; no embedding API key or cost. CPU is sufficient for a 5-page corpus.
- Embeddings are cached to disk keyed by content hash, so a one-page edit re-embeds one page.

### 6.4 Vector store

- **ChromaDB**, persisted on disk (fixed by the brief).
- Collection name includes the embedding-model name, so a model change forces a rebuild
  instead of silently mixing vector spaces.

### 6.5 Retrieval

- Embed the question → return top-k chunks with scores.
- `k` and the similarity score must be visible in the UI or logs, for teaching.
- A relevance threshold drives class B: below it, the assistant does not call the LLM at all.
  Refusing at the retrieval layer, rather than trusting a prompt, is what keeps class B from
  producing a plausible-sounding answer.

### 6.6 Generation

- System prompt: use only the retrieved context; cite the single best source; stay within
  three sentences; if the context is insufficient, say so.
- Low temperature (0–0.3) for a stable demo.
- Every answer carries exactly one source link and the "Last updated from sources:" footer.
- Post-generation validation: the cited URL must be one the system actually retrieved. A
  citation the model invented is stripped and the answer is marked unverified.

---

## 7. UI requirements (tiny, per brief)

- Welcome line, **3 example questions**, and the note **"Facts-only. No investment advice."**
- Chat input with message history.
- One source link under every answered message.
- Retrieved chunks available in an expandable view, so the class can see retrieval working.
- Status indicator: indexing / retrieving / generating.
- Four visually distinct answer states (§5), so a refusal is never mistakable for an answer.

---

## 8. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Scope and snapshot the 5 scheme pages + named supporting official pages | P0 |
| F2 | Load HTML pages and PDF factsheets into a unified document form | P0 |
| F3 | Chunk with a data-justified strategy, preserving source URL + heading | P0 |
| F4 | Embed locally with `all-MiniLM-L6-v2` | P0 |
| F5 | Persist vectors in on-disk ChromaDB; rebuildable from snapshots | P0 |
| F6 | Answer from retrieved context only, ≤3 sentences | P0 |
| F7 | Exactly one source link per answered message | P0 |
| F8 | Refuse out-of-corpus questions without fabricating a source | P0 |
| F9 | Refuse advice questions with an educational link | P0 |
| F10 | Refuse performance questions, quote no figures, link the factsheet | P0 |
| F11 | Scrub PII from user input before embedding, prompting, or logging | P0 |
| F12 | Show "Last updated from sources: &lt;fetch date&gt;" on every answer | P1 |
| F13 | Tiny UI with welcome line, 3 example questions, disclaimer note | P1 |
| F14 | Expandable retrieved-chunk view; visible `k` and score | P1 |
| F15 | Configurable `k` and chunk size via env or sidebar | P1 |
| F16 | Multi-turn chat with short history | P2 |
| F17 | Labeled eval set with per-class accuracy (A–D) | P2 |
| F18 | Streamed/hosted prototype link, or demo video fallback | P1 |

---

## 9. Non-functional requirements

| Area | Requirement |
|------|-------------|
| Answer length | ≤3 sentences, enforced by test, not by hope |
| Citations | Exactly one per answer; URL must resolve to a retrieved source |
| Transparency | "Last updated from sources:" on every answer |
| **No PII** | PAN, Aadhaar, account numbers, OTPs, emails, and phone numbers are neither stored nor forwarded. Patterns are redacted at the input boundary and excluded from logs |
| **No performance claims** | No return figure is ever computed, compared, or quoted in an answer |
| Latency | End-to-end answer in a few seconds on a 5-page corpus; local embedding makes retrieval near-instant |
| Secrets | No LLM key is required for retrieval; any LLM key lives in `.env` and is never committed |
| Explainability | All six pipeline stages observable; retrieved chunks visible during the demo |
| Reproducibility | Pinned dependencies; committed snapshots so the corpus cannot drift under the demo |
| Scope discipline | Corpus limited to the scoped AMC/schemes — no crawling beyond the source list |

---

## 10. Constraints (from the brief, restated as rules)

1. **Public sources only.** No screenshots of an app back-end; no third-party blogs as sources.
2. **No PII** accepted or stored in any form.
3. **No performance claims.** Do not compute or compare returns; link to the official factsheet.
4. **Clarity and transparency.** Answers ≤3 sentences, with the "Last updated from sources:" line.
5. **No advice.** Opinionated and portfolio questions get a polite facts-only message.

---

## 11. Deliverables

| # | Deliverable | Done when |
|---|-------------|-----------|
| D1 | Working prototype link (app or notebook), or a ≤3-minute demo video | Runs from a clean clone, or the video is uploaded |
| D2 | Source list (CSV or MD) of the 5 URLs used | One row per source: scheme, URL, publisher, fetch date, local snapshot path |
| D3 | README: setup, scope (AMC + schemes), known limits | A stranger can run it following the README alone |
| D4 | Sample Q&A file, 5–10 queries with answers and links | Covers all four classes in §5, not just class A |
| D5 | Disclaimer snippet used in the UI | "Facts-only. No investment advice." present in the running app, not just the README |

D4 is the deliverable most likely to be submitted as all-class-A questions. It must include
at least one class B, one class C, and one class D example, because those are the
requirements that distinguish a RAG system from a document lookup.

---

## 12. Acceptance criteria

- [ ] All six pipeline stages implemented and demonstrable in the stated order
- [ ] Corpus scoped to HDFC AMC and the 5 listed schemes; source list has exactly 5 rows
- [ ] Each of the six in-scope topics answered correctly with exactly one source link
- [ ] Answers are ≤3 sentences and carry the "Last updated from sources:" footer
- [ ] An out-of-corpus question produces no fabricated citation
- [ ] An advice question is refused politely with an educational link
- [ ] A performance question quotes no figures and links the official factsheet
- [ ] PII patterns are redacted at the input boundary and absent from logs
- [ ] Index rebuilds from the snapshot folder with one command
- [ ] All five deliverables in §11 submitted
- [ ] Presenter can walk ingest → retrieve → generate in under three minutes

---

## 13. Open questions

| # | Question | Why it matters | Recommendation |
|---|----------|----------------|----------------|
| Q1 | Are the groww.in pages acceptable sources, given the brief also says "AMC/SEBI/AMFI" and bans third-party blogs? | Affects D2 and the credibility of every citation | **RESOLVED 2026-09-27: yes.** Use the 5 as primary corpus, disclose the publisher per row, and cite official AMC/SEBI/AMFI documents wherever they cover the same fact |
| Q2 | Which LLM generates the answer? The brief fixes embeddings but names no LLM. | Drives latency, hosting, and whether an API key exists on demo day | Local Ollama (`llama3.5`/`qwen2.5:7b`) as default so the demo cannot fail on network; hosted model behind the same interface as an upgrade |
| Q3 | Is the corpus limited to exactly 5 URLs, or 5 scheme pages plus official reference pages? | "How to download a capital-gains statement" and riskometer methodology are unlikely to be on the 5 scheme pages | Allow official AMC/SEBI/AMFI reference pages as a documented extension; list every URL actually used, and state the count plainly in the README |
| Q4 | Is the page content fetchable, and are figures current at fetch time? | Fee figures change; a stale snapshot is a correctness problem, not just a freshness one | Snapshot at build time, stamp the fetch date, and state in the README that figures are as-of that date |
| Q5 | Factsheet PDFs — required or optional? | Adds a PDF parsing stage | Optional; treat as P1. If skipped, class D links to the page's factsheet link rather than a local PDF |
| Q6 | Is multi-turn chat worth the added risk? | F16 is P2 and history can degrade retrieval | Ship single-turn, add history only if time remains |
| Q7 | Hosting available, or is the video fallback the realistic path? | Changes the effort split between engineering and presentation | Decide early; the video path needs a scripted, repeatable demo |

---

## 14. Document history

| Date | Change |
|------|--------|
| 2026-09-27 | Initial PRD for the class-demo RAG chatbot, written while `problemstatement.txt` was empty; content derived from project intent. | 
| 2026-09-27 | Rewritten from the populated `problemstatement.txt` (mutual fund FAQ assistant). Re-scoped from generic RAG to facts-only MF FAQ; fixed embeddings (`all-MiniLM-L6-v2`) and vector store (ChromaDB); added the four-class answer model, PII and performance constraints, the six-stage pipeline requirements, and the five deliverables. Raised 7 open questions, including the groww.in vs AMC/SEBI/AMFI sourcing conflict. |
