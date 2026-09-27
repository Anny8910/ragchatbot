# Implementation Guide — Mutual Fund FAQ Assistant

**How to use this file:** one phase at a time, in order. Each phase is a self-contained
Cursor prompt. Paste the prompt, review the diff, run the verify commands, commit, then move
on. Do not run two phases in parallel — the data contracts in Phase 0 propagate into every
later phase, and a type change in P3 silently breaks P8.

**Source of truth:** `docs/architecture.md` (design), `docs/PRD.md` (requirements).
This file adds the *how*: exact files, exact signatures, exact verification.

| This doc | Architecture |
|----------|--------------|
| P0 Scaffold + contracts | (new — prerequisite for everything) |
| P1 Sources + fetch | §22 phase 0 |
| P2 Sanitize | §22 phase 1 |
| P3 Loaders + chunker + eval labels | §22 phase 2 |
| P4 Embedding + Chroma + builder | §22 phase 3 |
| P5 Retriever + gate | §22 phase 4 |
| P6 PII scrub | §22 phase 5 |
| P7 Triage router | §22 phase 6 |
| P8 Generation + validation + assembly | §22 phase 7 |
| P9 UI | §22 phase 8 |
| P10 Eval + chunking decision + D4 | §22 phase 9 |
| P11 README + demo + hardening | §22 phase 10 |

---

## 0. Ground rules (apply to every phase)

**Rules to restate in every Cursor prompt.** They prevent the two failure modes that actually
cost time: invented scope, and silent design drift.

1. **Only create or modify the files listed in the phase.** No extras, no "while I'm here."
2. **If something is unspecified, stop and ask.** Do not invent a field, a config key, a
   default, or a dependency. An invented default is worse than a missing feature, because it
   propagates.
3. **No new third-party dependencies** beyond the pinned `requirements.txt`. If a phase seems to
   need one, say so and stop.
4. **No network calls** outside `sources/fetch.py`. Tests and the app run offline.
5. **Every module gets a module docstring** naming the architecture section it implements, e.g.
   `"""Stage 1 — Loading and snapshotting. Implements architecture §7."""`
6. **Type hints on every public function.** Dataclasses from `rag_bot/types.py`, never dicts
   crossing a module boundary.
7. **No `print()`** outside CLI modules and the trace logger. The UI and pipeline return values.
8. **Every phase ends green:** the verify commands must pass before the next phase starts.

**Definition of done, every phase:**

- [ ] Verify commands pass, output pasted into the conversation
- [ ] New code has type hints and a module docstring
- [ ] No file created outside the phase's file list
- [ ] Any deviation from this document is written down and justified
- [ ] `git commit` with the phase name

**Commit message format:** `P<n>: <what landed>`, e.g. `P5: retriever + relevance gate`.

---

## 1. Preflight (once, before P0)

Do these by hand. They are environment facts, not code.

```bash
python3 --version                 # need >= 3.10 for `X | None` syntax
git --version                     # the whole design depends on committed snapshots
curl -sSf https://ollama.com/install.sh | sh   # or install Ollama.app
ollama pull llama3.5:8b           # ~4.7 GB; do this BEFORE the demo, not on demo day
ollama pull qwen2.5:7b            # fallback if llama3.5 is too slow on the demo machine
```

**Pre-download the embedding model now** and confirm the word-piece cap, because two later
phases hard-fail on it:

```bash
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('max_seq_length:', m.max_seq_length)
print('dim:', m.get_sentence_embedding_dimension())
print(m.tokenizer.tokenize('The expense ratio is 1.05% per annum.'))
"
```

Expected: `max_seq_length: 256`, `dim: 384`. If `max_seq_length` is not 256, stop and raise it
— P3 and P4 both depend on this number. Then keep the model in the local cache
(`~/.cache/huggingface`) so demo day needs no network.

**Decide Q1 and Q4 before P1 finishes.** See §"Blocking questions" at the end of this file.

---

## P0 — Scaffold and shared contracts

**Goal:** a repo that imports, a pinned dependency set, `git` initialised, and every shared
dataclass defined once so no later phase invents its own types.

**Files to create**

```
.gitignore
.env.example
requirements.txt
rag_bot/__init__.py
rag_bot/types.py          # the contracts below, verbatim
rag_bot/config.py
tests/__init__.py
tests/test_config.py
```

**.gitignore — must contain, at minimum**

```
.env
data/index/
data/logs/
__pycache__/
*.pyc
.venv/
```

`data/corpus/snapshots/` must **not** be ignored — committed snapshots are the reproducibility
guarantee (architecture D4).

**`rag_bot/types.py` — use this content as given.** Every phase imports from here.

```python
"""Shared data contracts for the whole pipeline.

Implements the data model in architecture §5. Nothing else in the codebase
defines a dataclass that crosses a module boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Topic = Literal[
    "expense_ratio", "exit_load", "min_sip", "lock_in",
    "riskometer", "benchmark", "statement", "other",
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
    text: str                        # sanitized; the only text sent to the LLM
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
```

**`rag_bot/config.py` — required shape**

```python
"""Configuration: the single source of truth for env vars (architecture §15)."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULTS: dict[str, str] = {
    "RAG_PROVIDER": "ollama",
    "RAG_LLM_MODEL": "llama3.5:8b",
    "RAG_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RAG_OLLAMA_HOST": "http://localhost:11434",
    "RAG_TEMPERATURE": "0.1",
    "RAG_MAX_SENTENCES": "3",
    "RAG_TOP_K": "3",
    "RAG_MIN_SCORE": "0.35",
    "RAG_CHUNK_TOKENS": "180",
    "RAG_CHUNK_OVERLAP": "40",
    "RAG_CORPUS_DIR": "data/corpus",
    "RAG_INDEX_DIR": "data/index",
    "RAG_SOURCES_FILE": "rag_bot/sources/sources.yaml",
    "RAG_TRIAGE_LLM": "true",
    "RAG_PII_REDACT": "true",
    "RAG_SANITIZE_PERF": "true",
    "RAG_HISTORY_TURNS": "0",
    "RAG_LOG_TRACE": "true",
}

@dataclass(frozen=True)
class Config:
    provider: str
    llm_model: str
    embed_model: str
    ollama_host: str
    temperature: float
    max_sentences: int
    top_k: int
    min_score: float
    chunk_tokens: int
    chunk_overlap: int
    corpus_dir: str
    index_dir: str
    sources_file: str
    triage_llm: bool
    pii_redact: bool
    sanitize_perf: bool
    history_turns: int
    log_trace: bool


def load() -> Config:
    """Read DEFAULTS overlaid with os.environ, coerce types, no validation."""

def validate(cfg: Config) -> list[str]:
    """Return a list of human-readable problems. Empty list == usable.

    Check: provider is known; ollama_host is a URL; min_score in [0,1];
    chunk_tokens <= 224 (headroom under the 256 word-piece cap, architecture §9.2);
    chunk_overlap < chunk_tokens. Never raises -- callers decide what to do.
    """
```

`.env` loading: read `.env` if present using `python-dotenv`, but **never require it** — the
default build must run with no `.env` at all (architecture D3).

**Verify**

```bash
git init && git add -A && git commit -m "P0: scaffold, contracts, config"
python3 -c "from rag_bot.config import load, validate; c=load(); print(validate(c) or 'config ok')"
python3 -c "from rag_bot.types import Answer, Outcome; print(Outcome.A_ANSWERED.value)"
pytest tests/ -q
```

**Cursor prompt**

```
Create the scaffold for a RAG chatbot. Implement Phase P0 only.

Files to create, and nothing else:
  .gitignore, .env.example, requirements.txt,
  rag_bot/__init__.py, rag_bot/types.py, rag_bot/config.py,
  tests/__init__.py, tests/test_config.py

1. rag_bot/types.py must contain EXACTLY the dataclasses and enum in the spec below,
   including module docstring, `from __future__ import annotations`, and the Literal
   type aliases. Do not add fields. Do not add convenience methods.

2. rag_bot/config.py must contain DEFAULTS, a frozen Config dataclass, load(), and
   validate() per the spec below. load() reads .env via python-dotenv if present but
   must work with no .env. validate() returns a list of strings, never raises, and
   must flag chunk_tokens > 224.

3. requirements.txt: pin exact versions for chromadb, sentence-transformers,
   streamlit, beautifulsoup4, lxml, pyyaml, python-dotenv, requests, pypdf,
   pytest, and the ollama HTTP client. Use `pip freeze`-style exact pins (==).
   Pick versions that are mutually compatible on Python 3.11; state which
   chromadb major version you targeted in a comment.

4. .gitignore must ignore .env, data/index/, data/logs/, __pycache__/, *.pyc, .venv/
   and must NOT ignore data/corpus/snapshots/.

5. .env.example documents every key in DEFAULTS with a placeholder value and a
   one-line comment. Note in a comment that the default build needs no API key.

6. tests/test_config.py: assert defaults load, assert validate() flags
   chunk_tokens=300, assert validate() accepts defaults, assert an env var
   overrides a default. No network, no model download.

Spec — rag_bot/types.py:
<paste the types.py block from this document>

Spec — rag_bot/config.py:
<paste the config.py block from this document>

Run `pytest tests/ -q` and paste the output. Do not create any file not listed above.
```

---

## P1 — Sources registry and fetch/snapshot **\[GATE\]**

**Goal:** five committed page snapshots with real fetch timestamps, and a generated
`manifest.csv`. **This is deliverable D2 and the project's biggest risk.** If the pages cannot
be fetched, stop and escalate rather than continuing.

**Files to create**

```
rag_bot/sources/__init__.py
rag_bot/sources/sources.yaml
rag_bot/sources/fetch.py
rag_bot/sources/manifest.py
data/corpus/snapshots/       (populated by running the fetcher)
data/corpus/manifest.csv     (generated)
```

**`sources.yaml` shape** — the five brief URLs verbatim from the PRD, plus empty official refs:

```yaml
sources:
  - source_id: hdfc_large_cap_growth
    url: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
    publisher: groww.in
    source_tier: brief
    scheme_id: S1
    scheme_name: "HDFC Large Cap Fund – Direct Growth"
    factsheet_url: null          # filled from the page in P3 if present
    aliases: ["hdfc large cap", "large cap fund", "large cap", "s1"]
  # ... S2 flexi cap (hdfc-equity-fund), S3 ELSS (hdfc-elss-tax-saver),
  #     S4 small cap (hdfc-small-cap-fund), S5 balanced advantage
```

**Alias rules (critical, architecture §14.4).** Aliases are matched **exactly** after
lowercasing and collapsing whitespace. Never match on a bare category word:
- S2 is "HDFC Equity Fund" (flexi cap) — but "equity" also appears in S1's and S4's category
  labels, so **"equity" alone must never be an alias for S2**.
- Aliases must be mutually exclusive across the five schemes. Write a test that asserts this.
- Include the scheme_id itself (`"s1"`) as an alias.

**`fetch.py` requirements**

```python
def fetch_source(src: Source, *, refresh: bool, timeout: int) -> tuple[str | None, str]:
    """Return (html_text_or_None, status). status in {"ok","fetch_failed","blocked"}.

    Rules:
    - One request per source. No link following, no crawling, no retries beyond one.
    - Real, identifiable User-Agent. Sequential, with a sleep between sources.
    - If snapshot exists and content_hash matches and refresh is False: skip, status "ok".
    - Write the snapshot to data/corpus/snapshots/{source_id}.html
    - Stamp fetched_at as ISO-8601 with local timezone.
    - A non-200, a robots-block, or a JS-only shell (body text < 200 chars after
      tag stripping) returns status "blocked", not an exception.
    """

def fetch_all(path: str, *, refresh: bool = False) -> list[Source]:
    """Fetch every source; never raise. Returns updated Source objects."""
```

**`manifest.py`**

```python
def write_manifest(sources: list[Source], out_path: str) -> None:
    """Write CSV with header:
    source_id,scheme_id,scheme_name,url,publisher,source_tier,fetched_at,
    snapshot_path,content_hash,status
    Sorted by scheme_id. This file IS deliverable D2."""

def read_manifest(path: str) -> list[Source]:
    """Inverse of write_manifest."""
```

**Verify**

```bash
python3 -m rag_bot.sources.fetch
cat data/corpus/manifest.csv
wc -c data/corpus/snapshots/*.html
```

Then open **one** snapshot and read it. You are checking for: real scheme content (not a
cookie wall, not a login redirect, not a JS shell), a readable fee table, and a findable
factsheet link. If any source returns `blocked`, stop — see "Blocking questions".

**Cursor prompt**

```
Implement Phase P1: the source registry and one-time page snapshotting.
Implement P1 only. Create exactly these files:
  rag_bot/sources/__init__.py, rag_bot/sources/sources.yaml,
  rag_bot/sources/fetch.py, rag_bot/sources/manifest.py

1. sources.yaml: five entries, one per URL below, with the exact fields
   source_id, url, publisher, source_tier, scheme_id, scheme_name,
   factsheet_url, aliases. scheme_id S1..S5 in this order:
   S1 https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
   S2 https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
   S3 https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
   S4 https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth
   S5 https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth
   All five: publisher groww.in, source_tier brief.
   scheme_name: "HDFC Large Cap Fund – Direct Growth",
   "HDFC Equity Fund – Direct Growth",
   "HDFC ELSS Tax Saver Fund – Direct Growth",
   "HDFC Small Cap Fund – Direct Growth",
   "HDFC Balanced Advantage Fund – Direct Growth".
   aliases: lowercase, mutually exclusive, never a bare category word.
   "equity" must NOT be an alias for S2 because it collides with the other
   schemes' category labels. Explain each alias choice in a YAML comment.

2. fetch.py: fetch_source() and fetch_all() per the signatures in the spec.
   One request per source, identifiable User-Agent, no crawling, no link
   following. Idempotent: skip when the snapshot exists and content_hash matches
   unless refresh=True. Stamp fetched_at. Treat a non-200 or a JS shell (under
   200 chars of visible text) as status "blocked", not an exception.
   fetch_all() must never raise.

3. manifest.py: write_manifest() / read_manifest() with the exact CSV header
   specified. Sorted by scheme_id.

4. Run `python3 -m rag_bot.sources.fetch` and paste the manifest output and the
   byte size of each snapshot. If any source is "blocked", say so plainly and
   stop -- do not invent substitute content.

Do not create any other file. Do not parse or clean the HTML yet (that is P3).
```

---

## P2 — Sanitize snapshots

**Goal:** remove performance and contact data from the corpus *before* it is ever chunked or
indexed, so the no-claims constraint is enforced by data absence (architecture §8).

**Files to create**

```
rag_bot/sources/sanitize.py
tests/test_sanitize.py
data/corpus/sanitize_report.json    (generated)
```

**Sanitization rules — apply in this order, whole-section first**

1. **Section removal by heading.** Drop any section whose heading matches, case-insensitively:
   `returns?`, `performance`, `nav`, `historical nav`, `growth of \d+`, `ranking`,
   `fund ranking`, `peer group`, `riskometer score` (keep the *level* text, drop the score),
   `sip calculator`, `calculator`, `returns calculator`, `portfolio`.
2. **Inline figure removal** (fallback for figures outside a matching section). Remove a
   sentence or table row when a return/NAV keyword occurs within the same line or the
   adjacent line, together with a number:
   - keywords: `1 ?year|3 ?year|5 ?year|since inception|cagr|return|returns|gain|performance|p\.?a\.?`
   - NAV keywords: `nav|nav as of|price`
   - ranking keywords: `rank(ed)?|percentile|peer group|top ?\d+`
   - the number: a percentage (`\d+(\.\d+)?\s*%`) or a currency amount (`₹[\d,]+` / `Rs\.? ?[\d,]+`)
3. **Contact/PII removal from the page:** email addresses, phone numbers, "enter your PAN"
   prompts, agent contact blocks, registration CTAs.
4. **Replacement, not deletion, for removed performance sections:** leave the single sentence
   "Performance figures for this scheme are available in the official factsheet." so a
   class-D question has something honest to retrieve and the assistant can point somewhere.

**Hard constraint: never remove a fee.** Expense ratio, exit load, minimum SIP, minimum
investment, and lock-in figures are the product. Add an explicit guard list of
must-preserve keywords (`expense ratio`, `exit load`, `minimum sip`, `lock-in`, `lock in`,
`benchmark`, `riskometer level`, `direct plan`, `growth plan`) — a line matching one of these is
never removed, even if it contains a percentage. Write a test for this: a table row reading
`Expense ratio (Direct Growth): 1.05%` survives sanitization untouched.

**Report shape** (`sanitize_report.json`)

```json
{"sources": [{"source_id": "hdfc_large_cap_growth",
  "before_tokens": 0, "after_tokens": 0,
  "rules_fired": {"section_heading": 4, "inline_figure": 27, "contact": 3},
  "removed_pct": 0.0,
  "preserved_fee_lines": 11,
  "warning": null}]}
```

Set `"warning": "sanitization removed X% of this source"` when `removed_pct > 40`. That
warning is the tripwire for over-aggressive stripping.

**`sanitize.py` required shape**

```python
def sanitize_text(text: str, *, source_id: str) -> tuple[str, dict]:
    """Return (sanitized_text, stats). Never raises."""

def sanitize_source(src: Source, text: str) -> tuple[str, dict]:
    """Apply sanitize_text and record preserved fee lines. Never raises."""

def write_report(reports: list[dict], out_path: str) -> None: ...
```

**Verify**

```bash
pytest tests/test_sanitize.py -q
python3 -m rag_bot.sources.sanitize            # if a CLI entry is added
python3 -c "import json;r=json.load(open('data/corpus/sanitize_report.json'));\
print([(s['source_id'],s['removed_pct'],s['preserved_fee_lines']) for s in r['sources']])"
rg -in "cagr|since inception|1 year return|3 year return|\bnav\b" data/corpus/snapshots/ || echo "no returns left"
rg -in "expense ratio" data/corpus/snapshots/ | head        # must still find fee tables
```

The last two commands are the real check: the first must find nothing, the second must find
plenty. If the first finds matches, sanitization is incomplete. If the second finds nothing,
you have over-stripped and the product cannot answer anything.

**Cursor prompt**

```
Implement Phase P2: corpus sanitization. Implement P2 only.
Create exactly: rag_bot/sources/sanitize.py, tests/test_sanitize.py

Goal: strip performance, NAV, ranking, and contact data from page snapshots
BEFORE they are chunked or indexed, so the assistant structurally cannot quote
a return figure. See docs/architecture.md section 8.

1. sanitize_text(text, *, source_id) -> (str, stats). Apply rules in this order:
   a) whole-section removal by heading, matching (case-insensitive):
      returns?, performance, nav, historical nav, growth of, ranking, fund ranking,
      peer group, sip calculator, calculator, portfolio
      Drop the riskometer *score* but keep the riskometer *level* text.
   b) inline figure removal: drop a line when a return/NAV/ranking keyword AND a
      number (percentage or currency) co-occur on the same or adjacent line.
   c) contact removal: emails, phone numbers, "enter your PAN", agent contact
      blocks, registration CTAs.
   d) wherever a performance section was removed, leave one sentence:
      "Performance figures for this scheme are available in the official factsheet."

2. CRITICAL GUARD: never remove a line containing any of these:
   expense ratio, exit load, minimum sip, minimum investment, lock-in, lock in,
   benchmark, riskometer level, direct plan, growth plan
   even if that line contains a percentage. Test that the line
   "Expense ratio (Direct Growth): 1.05%" survives untouched.

3. Return stats: before_tokens, after_tokens, rules_fired counts, removed_pct,
   preserved_fee_lines. Add a warning key when removed_pct > 40.

4. tests/test_sanitize.py, using realistic fixture strings, not toy input:
   - a returns table is removed
   - an expense-ratio table survives
   - an exit-load table survives
   - "CAGR 14.2%" and "1 year return 12.4%" are removed
   - a rank/percentile row is removed
   - an email and a phone number are removed
   - a riskometer LEVEL sentence survives while its SCORE row is removed
   - sanitizing twice is idempotent

Do not create other files. Do not fetch anything. Do not touch loaders.py
(that is P3). Run pytest and paste the output.
```

---

## P3 — Loaders, chunker, and eval labels

**Goal:** snapshots become clean, structured, correctly-sized chunks; and the eval set exists
**before** chunking is finalized, so the §9.3 experiment is meaningful.

**Files to create**

```
rag_bot/ingest/__init__.py
rag_bot/ingest/loaders.py
rag_bot/ingest/chunker.py
rag_bot/eval/__init__.py
rag_bot/eval/eval_set.yaml
tests/test_loaders.py
tests/test_chunker.py
```

**`loaders.py` — the document text format is a contract with the chunker**

Loaders emits plain text with this structure, and the chunker parses exactly this:

- Headings on their own line as `## Heading` (levels `##` and `###`).
- **Tables flattened row-wise as `Label: value` lines**, consecutive. Example:
  ```
  ## Fees and charges
  Expense ratio (Direct Growth): 1.05%
  Exit load (0-7 days): 1.00%
  Minimum SIP: ₹500
  ```
  A table shredded into fragments is unanswerable; a table flattened this way is answerable.
  This is the single highest-leverage transformation in the whole loader.
- Paragraphs separated by a blank line.
- Non-table list items as `- item`.

Required behaviour:

```python
def load_source(src: Source) -> Document:
    """Read src.snapshot_path, return a Document with the structured text above.

    - Strip nav, header, footer, aside, script, style, noscript, cookie/consent
      dialogs, and ad containers before extracting text.
    - Convert every <table> to `Label: value` lines, carrying the header row as
      context on each row when the table has column headers
      (e.g. "Expense ratio | Direct Growth | 1.05%" ->
       "Expense ratio (Direct Growth): 1.05%").
    - Emit headings as `## ` / `### ` lines, preserving nesting.
    - Scan the raw HTML for a factsheet PDF link and return it on the Source
      (set factsheet_url if it is currently null). This is how class D gets
      its link (architecture section 13.3).
    - If a .pdf snapshot exists, extract per-page text and prefix each page
      with a `## Page N` heading; set page numbers in heading text.
    - Raise nothing. A source that cannot be loaded returns a Document with
      empty text and logs why.
    """

def load_all(sources: list[Source]) -> list[Document]: ...
```

**`chunker.py` — algorithm, in order**

```python
def chunk_document(doc: Document, *, target_tokens: int, overlap_tokens: int,
                   tokenizer) -> list[Chunk]:
    """Section-aware, table-aware chunking (architecture section 9.2).

    1. Split doc.text into sections at `## ` / `### ` lines. Never chunk across
       a section boundary.
    2. Within a section, build units in priority order:
       a. runs of consecutive `Label: value` lines  (table row-groups)
       b. blank-line-separated paragraphs
       c. sentences
       d. words
    3. Accumulate units into a chunk until adding the next unit would exceed
       target_tokens. Emit. Start the next chunk with the last
       overlap_tokens worth of units from the previous chunk.
    4. A table row-group keeps its leading header line with its rows, so
       "Expense ratio" is never separated from its value.
    5. Forward-progress guarantee: if a single unit exceeds the target, hard-split
       it on word boundaries and log a warning. The loop must ALWAYS advance.
    6. Any chunk whose tokenizer count exceeds 256 word-pieces is hard-split.
    7. topic is assigned per chunk from heading + keyword rules, first match wins:
       expense_ratio, exit_load, min_sip, lock_in, riskometer, benchmark,
       statement, other.
    8. id = f"{source.source_id}::{ordinal}", ordinal from 0 per document.
    9. content_hash = sha256 of the chunk text.
    """

def chunk_all(docs: list[Document], **kw) -> list[Chunk]: ...
```

**The tokenizer is injected, never imported inside the chunker.** P3 must be testable without
downloading a model, so tests pass a whitespace-token stub that returns `len(text.split())`.
P4 supplies the real word-piece tokenizer.

**`eval_set.yaml` — must exist before chunking is finalized**

30 class-A rows (6 topics × 5 schemes) plus B/C/D rows plus edge cases:

```yaml
- id: a-expense-S1
  question: "What is the expense ratio of HDFC Large Cap Fund?"
  outcome: A
  scheme_id: S1
  topic: expense_ratio
  expect_source_id: hdfc_large_cap_growth
  expect_keywords: ["expense ratio"]
  forbidden_patterns: ["cagr", "return of", "% return"]

- id: b-nav
  question: "What was the NAV of HDFC Large Cap Fund yesterday?"
  outcome: B
  forbidden_patterns: ["http", "groww.in/mutual-funds"]   # no citation allowed

- id: c-advice-S3
  question: "Should I buy the HDFC ELSS Tax Saver Fund?"
  outcome: C
  require_link: true

- id: d-performance
  question: "Which of these HDFC funds has the best 1-year return?"
  outcome: D
  forbidden_patterns: ["%", "cagr", "nav"]                # zero figures allowed
```

Required coverage: 6 topics × 5 schemes for A; ≥3 each for B, C, D; 2 multi-scheme
("expense ratio of large cap and ELSS?"); 2 scheme-ambiguous ("what is the exit load?" with
no scheme named).

> **Write the questions by hand, not by template.** Thirty templated rows like
> "What is the {topic} of {scheme}?" inflate hit-rate and make the chunking experiment
> meaningless. Vary the phrasing. The labels must be written *before* any retrieval score is
> seen, or the numbers mean nothing.

**Verify**

```bash
pytest tests/ -q
python3 -c "
from rag_bot.config import load
from rag_bot.ingest.loaders import load_all
from rag_bot.sources.manifest import read_manifest
from rag_bot.ingest.chunker import chunk_all
c = load()
docs = load_all(read_manifest(c.corpus_dir + '/manifest.csv'))
print('docs:', len(docs), 'chars:', sum(len(d.text) for d in docs))
chunks = chunk_all(docs, target_tokens=c.chunk_tokens, overlap_tokens=c.chunk_overlap,
                   tokenizer=lambda t: len(t.split()))
print('chunks:', len(chunks))
print('max words:', max(ch.n_tokens for ch in chunks))
from collections import Counter
print(Counter(ch.topic for ch in chunks))
"
```

Check by eye: print one chunk whose topic is `expense_ratio` and confirm the fee figure and
its label are in the same chunk. That is the demo's most important chunk.

**Cursor prompt**

```
Implement Phase P3: snapshot loaders, the chunker, and the eval set.
Create exactly:
  rag_bot/ingest/__init__.py, rag_bot/ingest/loaders.py, rag_bot/ingest/chunker.py,
  rag_bot/eval/__init__.py, rag_bot/eval/eval_set.yaml,
  tests/test_loaders.py, tests/test_chunker.py

Read docs/architecture.md sections 7, 8, and 9 first.

1. loaders.py -- load_source(src) and load_all(sources) per the spec.
   The output text format is a CONTRACT with the chunker:
     - headings on their own line as "## Heading" or "### Heading"
     - tables flattened row-wise as "Label: value" lines, with the column header
       folded into the label: "Expense ratio | Direct Growth | 1.05%" becomes
       "Expense ratio (Direct Growth): 1.05%"
     - paragraphs separated by a blank line
   Strip nav/header/footer/aside/script/style/noscript/cookie dialogs/ads first.
   Scan raw HTML for a factsheet PDF link and set src.factsheet_url if null.
   Handle .pdf snapshots by emitting "## Page N" headings.
   Never raise. A failed load returns a Document with empty text and a logged reason.

2. chunker.py -- chunk_document(doc, *, target_tokens, overlap_tokens, tokenizer).
   Section-aware and table-aware, per the algorithm in the spec:
   cut only at section boundaries; build units as table-row-groups, then
   paragraphs, then sentences, then words; accumulate to target_tokens; start the
   next chunk with the previous chunk's last overlap_tokens; keep a table's
   header line with its rows; GUARANTEE the loop always advances (hard-split
   oversized units and log); hard-split anything over 256 tokenizer units.
   Assign topic per chunk from heading+keyword rules (expense_ratio, exit_load,
   min_sip, lock_in, riskometer, benchmark, statement, other).
   id = "{source_id}::{ordinal}". content_hash = sha256 of chunk text.
   The tokenizer is INJECTED, never imported here -- tests pass
   `lambda t: len(t.split())`. Do not import sentence_transformers in this file.

3. eval_set.yaml -- write it BY HAND with varied phrasing, never templated:
   30 class-A rows (6 topics x 5 schemes), 3+ each of B/C/D, 2 multi-scheme,
   2 scheme-ambiguous. Each row has id, question, outcome, and where relevant
   scheme_id, topic, expect_source_id, expect_keywords, forbidden_patterns.
   Class-D rows must forbid "%", "cagr", and "nav". Class-B rows must forbid
   any citation URL.

4. tests: table flattening, heading extraction, factsheet link extraction,
   chunk never crosses a section boundary, table label stays with its value,
   forward progress on a pathological input, id stability, topic assignment.

Run pytest and paste output. Do not create other files. Do not download a
model. Do not build an index yet (that is P4).
```

---

## P4 — Embedding, Chroma store, builder

**Goal:** a rebuildable on-disk index, and proof that no chunk is silently truncated.

**Files to create**

```
rag_bot/providers/__init__.py
rag_bot/providers/base.py
rag_bot/providers/minilm.py
rag_bot/providers/fake.py
rag_bot/index/__init__.py
rag_bot/index/store.py
rag_bot/index/ids.py
rag_bot/ingest/builder.py
tests/test_store.py
tests/test_truncation.py
```

**`providers/base.py`**

```python
class EmbeddingProvider(Protocol):
    name: str
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def count_tokens(self, text: str) -> int: ...   # word pieces, for the 256 cap

class LLMProvider(Protocol):
    name: str
    def generate(self, system: str, user: str, *, temperature: float,
                 max_tokens: int) -> str: ...
    def classify(self, system: str, user: str, *, labels: list[str]) -> str: ...
```

**`providers/minilm.py`** — the 256 cap is enforced, not assumed:

```python
MAX_WORD_PIECES = 256

class MiniLMProvider:
    def __init__(self, model_name: str = ..., cache_dir: str | None = None):
        # lazy-load the model; do not download at import time
    def count_tokens(self, text: str) -> int:
        # tokenize WITHOUT truncation and return the true length
    def assert_fits(self, text: str) -> None:
        # raise ValueError if count_tokens(text) > MAX_WORD_PIECES
    def embed_documents(self, texts): # batch, show progress, normalize
    def embed_query(self, text): ...
```

**`providers/fake.py`** — deterministic hashed embeddings so the whole suite runs offline with
no model download. Use a fixed-seed hash of word tokens projected to 384 dims, L2-normalised.
It is *not* semantic; it exists so tests can run. Note that in the module docstring.

**`index/store.py`**

```python
COLLECTION_PREFIX = "mf_faq"

def collection_name(embed_model: str, corpus_version: str) -> str:
    """'mf_faq__<slug(embed_model)>__<corpus_version>' (architecture D5)."""

class Store:
    def __init__(self, index_dir: str, embed_model: str, corpus_version: str): ...
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Chroma metadata must be FLAT SCALARS. Build it with a helper that
        flattens Chunk fields; never pass a nested dict or a list."""

    def query(self, vector: list[float], *, k: int,
              scheme_id: str | None = None, topic: str | None = None
              ) -> list[ScoredChunk]:
        """n_results = k + 2 (overfetch, architecture section 12).
        Optional where-filter on scheme_id / topic.
        RETURN SCORES AS COSINE SIMILARITY and verify the convention empirically
        (see Watch out below)."""

    def count(self) -> int: ...
    def drop(self) -> None: ...
```

**`ingest/builder.py`** — the one-command rebuild:

```
python3 -m rag_bot.ingest.builder --rebuild     # full rebuild
python3 -m rag_bot.ingest.builder               # incremental
python3 -m rag_bot.ingest.builder --dry-run     # chunk + token counts, no index write
```

Pipeline: load manifest → `load_all` → `sanitize` (in memory; P2's rules) → `chunk_all` →
`assert_fits` every chunk (**hard fail, never index a truncated chunk**) → embed in batches →
upsert → delete orphans (chunks whose `source_id` is no longer in the manifest) → print a
summary: sources, chunks, per-topic counts, max word-pieces, collection name.

Embeddings are cached on disk keyed by chunk `content_hash`, so re-running on an unchanged
corpus does zero embedding work. Print how many chunks were reused vs recomputed.

**Watch out — the score convention.** The whole class-B gate rests on `RAG_MIN_SCORE = 0.35`
being a cosine similarity. Chroma's `distances` are *not* guaranteed to be `1 - cos` across
versions. Before trusting the threshold, verify empirically:

```python
import numpy as np
v = store.embed_query("What is the expense ratio?")
# embed the same text as a document, then query with it
# the returned distance should be ~0.0, implying score = 1 - distance
```

If the retrieved distance for an identical text is not ~0, fix the conversion in `query()`
and record the Chroma version and the convention in a comment. A gate threshold applied to
the wrong scale either never fires or always fires.

**Verify**

```bash
pytest tests/ -q
python3 -m rag_bot.ingest.builder --dry-run     # inspect chunk stats, no index
python3 -m rag_bot.ingest.builder --rebuild      # build the index
python3 -m rag_bot.ingest.builder               # must report all-reused, 0 new embeddings
python3 -c "
from rag_bot.config import load
from rag_bot.index.store import Store
s = Store(load().index_dir, load().embed_model, 'v1')
print('chunks:', s.count())
"
```

**Cursor prompt**

```
Implement Phase P4: local embedding, the Chroma store, and the index builder.
Create exactly:
  rag_bot/providers/__init__.py, rag_bot/providers/base.py,
  rag_bot/providers/minilm.py, rag_bot/providers/fake.py,
  rag_bot/index/__init__.py, rag_bot/index/store.py, rag_bot/index/ids.py,
  rag_bot/ingest/builder.py, tests/test_store.py, tests/test_truncation.py

Read docs/architecture.md sections 9, 10, and 11 first.

1. providers/base.py: EmbeddingProvider and LLMProvider Protocols exactly as in
   the spec. Protocol members only, no implementations.

2. providers/minilm.py: MiniLMProvider wrapping
   sentence-transformers/all-MiniLM-L6-v2. MAX_WORD_PIECES = 256. Load the model
   LAZILY -- importing this module must not trigger a download.
   count_tokens() must tokenize WITHOUT truncation and return the true length.
   assert_fits() raises ValueError above 256. Embeddings L2-normalised.

3. providers/fake.py: deterministic fake embedding provider using a fixed-seed
   token hash projected to 384 dims and normalised, so the test suite runs with
   no model download and no network. State in the docstring that it is NOT
   semantic and exists only so tests can run.

4. index/store.py: collection_name() -> "mf_faq__<slug>__<version>" and a Store
   class with upsert/query/count/drop per the spec. Chroma metadata must be FLAT
   SCALARS -- write a helper that flattens Chunk fields and never pass nested
   structures. query() takes n_results = k + 2 and optional where-filters on
   scheme_id and topic.
   IMPORTANT: empirically verify Chroma's distance convention before converting.
   Embed a sentence, query it with the same sentence, and check the distance is
   ~0. If it is not, compute cosine similarity yourself from the stored
   embeddings instead of assuming 1 - distance. Record the Chroma version and
   the convention you settled on in a comment.

5. ingest/builder.py: CLI with --rebuild, --dry-run, and default incremental.
   Flow: read manifest -> load_all -> sanitize in memory -> chunk_all ->
   assert_fits EVERY chunk (hard fail the build; never index a truncated chunk) ->
   embed in batches -> upsert -> delete orphan chunks whose source_id is gone
   from the manifest -> print summary (sources, chunks, per-topic counts, max
   word-pieces, collection name).
   Cache embeddings on disk keyed by chunk content_hash so an unchanged corpus
   re-runs with zero embedding work. Report reused vs recomputed counts.

6. tests: collection name format, flat metadata, upsert/query round trip with the
   fake provider, and a truncation test asserting a 300-word-piece chunk raises.

Run pytest and paste output. Then run the builder and paste the summary. Do not
create other files.
```

---

## P5 — Retriever and relevance gate

**Goal:** rank chunks, and refuse class B before any LLM is called.

**Files to create**

```
rag_bot/retrieve/__init__.py
rag_bot/retrieve/retriever.py
rag_bot/retrieve/gate.py
tests/test_gate.py
```

**`retriever.py`**

```python
def retrieve(query: str, *, k: int, scheme_id: str | None = None,
             embedder: EmbeddingProvider, store: Store) -> list[ScoredChunk]:
    """Embed the query, return top-(k+2) scored chunks sorted descending.
    An empty index returns []. Never raises."""
```

**`gate.py`** — the class-B decision, and the only place the LLM is skipped:

```python
@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str | None      # None | "empty_index" | "no_candidates" | "low_score"
    message: str | None     # user-facing, class-B copy
    weak_evidence: bool     # within 0.05 of the threshold
    top_score: float | None

def evaluate(chunks: list[ScoredChunk], *, min_score: float,
             index_empty: bool, covered_topics: list[str]) -> GateResult:
    """Checks in order (architecture section 12.1):
      1. index_empty                 -> reject "empty_index"
      2. not chunks                  -> reject "no_candidates"
      3. chunks[0].score < min_score -> reject "low_score"
      4. chunks[0].score < min_score + 0.05 -> accept with weak_evidence=True
    Class-B message names the six covered topics and never includes a URL."""
```

The class-B message must list what the assistant *can* answer — expense ratio, exit load,
minimum SIP, ELSS lock-in, riskometer/benchmark, statement download. A refusal that says what
it knows reads as scoped; one that says "not found" reads as broken.

**Watch out — no relative thresholds.** Do not implement "top result much better than the
rest". With a five-scheme corpus an off-topic question returns five equally bad matches, and a
relative rule cheerfully picks the best of them.

**Verify**

```bash
pytest tests/test_gate.py -q
python3 -c "
from rag_bot.config import load
from rag_bot.index.store import Store
from rag_bot.retrieve.retriever import retrieve
from rag_bot.providers.minilm import MiniLMProvider
c = load(); e = MiniLMProvider(); s = Store(c.index_dir, c.embed_model, 'v1')
for q in ['What is the expense ratio of HDFC Large Cap Fund?',
          'What is the ELSS lock-in period?',
          'What is the exit load?',
          'Who won the 1994 FIFA World Cup?',
          'What was the NAV yesterday?',
          'How do I cook pasta?']:
    r = retrieve(q, k=3, embedder=e, store=s)
    top = f'{r[0].score:.3f} {r[0].chunk.source_id}' if r else 'none'
    print(f'{top:>45}  <- {q}')
"
```

You are looking for a **clear gap** between on-topic scores (expect roughly 0.45–0.75) and
off-topic ones (expect roughly 0.05–0.20). If in-scope and off-topic scores overlap, do not
lower the threshold — tighten the chunking or go back to P3. Write the observed score
distribution into `docs/chunking-decision.md`; it is the evidence for the `RAG_MIN_SCORE`
choice in P10.

**Cursor prompt**

```
Implement Phase P5: retrieval and the relevance gate. Implement P5 only.
Create exactly:
  rag_bot/retrieve/__init__.py, rag_bot/retrieve/retriever.py,
  rag_bot/retrieve/gate.py, tests/test_gate.py

Read docs/architecture.md section 12 first.

1. retriever.retrieve(query, *, k, scheme_id=None, embedder, store) -> list[ScoredChunk]
   Embed the query, query with n_results = k + 2 (overfetch), return sorted
   descending by score. Optional scheme_id where-filter. An empty index returns
   []. Never raises.

2. gate.py: GateResult dataclass and evaluate() per the spec. Checks in this
   order: empty index, no candidates, top score below min_score, borderline
   within 0.05 of min_score. The class-B message must name the six covered
   topics (expense ratio, exit load, minimum SIP, ELSS lock-in,
   riskometer/benchmark, statement download) and must never contain a URL.

3. Do NOT implement any relative or margin-based threshold. Only the absolute
   min_score comparison. A relative rule picks the best of five equally bad
   matches and defeats the gate.

4. tests: empty index rejects with "empty_index"; no candidates rejects with
   "no_candidates"; a 0.20 score against min_score 0.35 rejects with "low_score";
   a 0.36 score accepts with weak_evidence True; a 0.80 score accepts with
   weak_evidence False; the class-B message contains no "http".

Run pytest and paste output. Do not create other files.
```

---

## P6 — PII scrub

**Goal:** no PAN, Aadhaar, account number, OTP, email, or phone number ever reaches the
embedder, the LLM, the logs, or the screen (architecture §6).

**Files to create**

```
rag_bot/safety/__init__.py
rag_bot/safety/pii.py
tests/test_pii.py
```

**`pii.py`**

```python
@dataclass(frozen=True)
class ScrubResult:
    text: str                 # the REDACTED string -- this is what flows onward
    rules_fired: list[str]    # e.g. ["pan", "email"]
    clean: bool               # True when nothing fired

def scrub(text: str) -> ScrubResult:
    """Redact PII. Must never raise.

    Rules, each a separate named function so tuning one does not disturb others:
      pan        10 chars, [A-Z]{5}[0-9]{4}[A-Z]           -> "[PAN redacted]"
      aadhaar    12 digits, spaces or X allowed            -> "[Aadhaar redacted]"
      email      standard pattern                          -> "[email redacted]"
      phone      10 digits, optional +91/0 prefix          -> "[phone redacted]"
      account    8-16 digits but ONLY with an "a/c",
                 "account", "acct", "folio" keyword within
                 40 chars                                    -> "[account redacted]"
      otp        4-6 digits but ONLY with "otp", "code",
                 "pin" within 40 chars                      -> "[OTP redacted]"

    HARD RULES:
    - NEVER redact a bare run of digits without a PII keyword nearby.
    - NEVER redact a percentage (1.05%) or a currency amount (Rs 500, Rs 500).
      These are the product's actual answers.
    """

def scrub_for_log(text: str) -> str:
    """Same rules, returns a plain string. Used by the trace logger in P9."""
```

**Fixture tables are the deliverable of this phase.** `tests/fixtures/pii_must_redact.txt` and
`pii_must_not_redact.txt` — real strings, not toy ones:

*Must redact:* a real-format PAN, a space-separated Aadhaar, `my email is a@b.com`, a
10-digit phone, `a/c no 1234567890`, `otp 482913`.

*Must **not** redact:* `The expense ratio is 1.05%.`, `Exit load is 1% for < 7 days.`,
`Minimum SIP is Rs 500.`, `Lock-in is 3 years.`, `The lock-in period for ELSS is 3 years.`,
`expense ratio 1.05 exit load 1.00 benchmark NIFTY 50 riskometer level 4`.

A scrubber that eats "1.05%" has broken the product more thoroughly than one that leaks an
email, because it breaks silently. Both tables are asserted, in both directions.

**Verify**

```bash
pytest tests/test_pii.py -q
python3 -c "
from rag_bot.safety.pii import scrub
for t in ['My PAN is ABCDE1234F, expense ratio?',
          'expense ratio is 1.05% and exit load 1%',
          'call me 9876543210 about minimum SIP Rs 500',
          'a/c no 1234567890 and otp 482913']:
    r = scrub(t); print(r.rules_fired, '|', r.text)
"
```

Read every line of that output. The first must redact the PAN and keep "expense ratio". The
second must redact nothing. The third must redact the phone and keep "Rs 500".

**Cursor prompt**

```
Implement Phase P6: PII scrubbing at the input boundary. Implement P6 only.
Create exactly:
  rag_bot/safety/__init__.py, rag_bot/safety/pii.py, tests/test_pii.py,
  tests/fixtures/pii_must_redact.txt, tests/fixtures/pii_must_not_redact.txt

Read docs/architecture.md section 6 first.

1. pii.py: ScrubResult dataclass, scrub(text) -> ScrubResult, and
   scrub_for_log(text) -> str, per the spec. Each rule is a separate named
   function. Never raises.

2. HARD CONSTRAINTS, both are release gates:
   - Never redact a bare run of digits without a PII keyword nearby.
   - Never redact a percentage (1.05%) or a currency amount (Rs 500 / Rs 500).
   These are the answers the product exists to give. A scrubber that eats
   "1.05%" is a worse bug than one that leaks an email, because it is silent.

3. tests/fixtures/pii_must_redact.txt: real-format PAN, space-separated Aadhaar,
   an email, a 10-digit phone, "a/c no 1234567890", "otp 482913". One per line.
   tests/fixtures/pii_must_not_redact.txt: "The expense ratio is 1.05%.",
   "Exit load is 1% for less than 7 days.", "Minimum SIP is Rs 500.",
   "Lock-in is 3 years.", "The lock-in period for ELSS is 3 years.",
   "expense ratio 1.05 exit load 1.00 benchmark NIFTY 50 riskometer level 4".

4. tests/test_pii.py reads BOTH fixture files and asserts every line is
   redacted / not redacted respectively. Also unit-test each rule in isolation
   and the keyword-proximity requirement for account and OTP.

Run pytest and paste the output. Do not create other files.
```

---

## P7 — Triage router

**Goal:** classes C and D decided by deterministic rules *before* retrieval, with a zero
false-positive rate on the six in-scope topics.

**Files to create**

```
rag_bot/answer/__init__.py
rag_bot/answer/triage.py
rag_bot/answer/schemes.py
rag_bot/answer/refusals.py
tests/test_triage.py
tests/test_schemes.py
```

**`schemes.py`** — exact-match scheme resolution (architecture §14.4):

```python
def resolve_scheme(question: str, sources: list[Source]) -> tuple[str | None, bool]:
    """Return (scheme_id or None, ambiguous).

    Matching is EXACT against the alias table after lowercasing and collapsing
    whitespace. NEVER fuzzy-match. "equity" alone must not resolve to S2 --
    it collides with other schemes' category labels. If exactly one scheme's
    aliases match, return it. If two or more match, return (None, True).
    If none match, return (None, False)."""
```

**`triage.py`**

```python
RULES_C: tuple[tuple[str, str], ...]   # (rule_name, compiled_pattern)
RULES_D: tuple[tuple[str, str], ...]

def classify_rules(question: str) -> TriageResult | None:
    """Layer 1. Return a TriageResult with outcome C or D, or None if unsure.
    D is checked BEFORE C: "should I buy the fund with the best returns" is a
    performance question first."""

def classify_llm(question: str, provider: LLMProvider) -> TriageResult | None:
    """Layer 2. One cheap call, labels ["A_or_B", "C_advice", "D_performance"].
    Return None if the model is unusable. Never raises."""

def classify(question: str, *, sources: list[Source],
             provider: LLMProvider | None = None,
             use_llm: bool = True) -> TriageResult:
    """Layer 1 first. If it returns a result, use it with layer="rules".
    Else if use_llm and provider: layer 2, layer="llm".
    Else: TriageResult(outcome=A, reason="no_rule_matched", layer=None).

    Always resolve the scheme via schemes.resolve_scheme and set scheme_id and
    scheme_ambiguous on the result."""
```

**Rule content — this list is the release gate, so be precise.**

Class D (performance): `best performing`, `1 year return`, `3 year return`, `5 year return`,
`since inception`, `cagr`, `returns of`, `how much did .* return`, `nav of`, `current nav`,
`which performed`, `top performer`, `ranking`, `ranked`, `percentile`, `peer group rank`,
`growth of`, `portfolio return`.

Class C (advice): `should i`, `should we`, `which is better for me`, `is it safe to`,
`i recommend`, `recommend`, `good time to`, `worth buying`, `suitable for me`,
`best scheme for`, `how much should i`, `portfolio allocation`, `is now a good time`,
`can i exit`.

**Banned bare keywords: `risk`, `better`, `return`, `perform`, `growth`, `invest`, `good`.**
Reason: `riskometer level` is an in-scope *fact* question; `return` appears in "returns
period" as an exit-load synonym; `better` appears in comparative fee questions. Each banned
word must appear only inside a longer, specific phrase.

**Mandatory test:** assert `classify()` returns outcome A (or None → A) for all 30 class-A
questions in `eval_set.yaml`. Any false positive here is a release blocker, because it means
the assistant refuses a question it is required to answer.

**`refusals.py`** — pre-written templates, never model-generated:

```python
def refusal_c(question: str, educational_url: str) -> tuple[str, str | None]:
    """(text, url). One sentence naming the facts-only limit, plus a relevant
    educational link. Must NOT restate the premise in advisory language."""

def refusal_d(scheme_name: str, factsheet_url: str | None) -> tuple[str, str | None]:
    """(text, url). States that returns are not provided or compared, quotes NO
    figure of any kind, links the official factsheet."""

def refusal_b(covered_topics: list[str]) -> tuple[str, None]:
    """(text, None). Names the six covered topics. Never contains a URL."""
```

Add a test asserting neither `refusal_c` nor `refusal_d` output contains a digit followed by
`%`, and that `refusal_c` output contains no advisory verb (`should`, `recommend`, `best`).

**Verify**

```bash
pytest tests/test_triage.py tests/test_schemes.py -q
python3 -c "
import yaml
from rag_bot.answer.triage import classify
rows = yaml.safe_load(open('rag_bot/eval/eval_set.yaml'))
bad = [r for r in rows if r['outcome']=='A' and classify(r['question']).outcome != 'A']
print('class-A false positives:', len(bad))
for r in bad: print('  ', r['question'])
for r in rows:
    if r['outcome'] in 'CD':
        t = classify(r['question'])
        print(r['outcome'], '->', t.outcome, t.reason, t.layer)
"
```

`class-A false positives: 0` is the number that matters. If it is not zero, tighten the rules
by *lengthening* patterns, never by adding a banned bare keyword.

**Cursor prompt**

```
Implement Phase P7: the pre-retrieval triage router. Implement P7 only.
Create exactly:
  rag_bot/answer/__init__.py, rag_bot/answer/triage.py,
  rag_bot/answer/schemes.py, rag_bot/answer/refusals.py,
  tests/test_triage.py, tests/test_schemes.py

Read docs/architecture.md sections 13 and 14.4 first.

1. schemes.py: resolve_scheme(question, sources) -> (scheme_id|None, ambiguous).
   EXACT alias matching only, after lowercasing and collapsing whitespace.
   NEVER fuzzy match. "equity" alone must not resolve to S2 because it collides
   with other schemes' category labels.

2. triage.py: RULES_C, RULES_D, classify_rules(), classify_llm(), classify()
   per the spec. Layer 1 (rules) always first and costs no LLM call. Layer 2
   (LLM classifier) runs only when layer 1 is unsure and use_llm is true.
   Check D before C: "should I buy the fund with the best returns" is a
   performance question first. classify() always resolves the scheme and sets
   scheme_id and scheme_ambiguous.

3. Use exactly the rule phrases in the spec. BANNED as standalone keywords:
   risk, better, return, perform, growth, invest, good. Each may appear only
   inside a longer specific phrase. Reason: "riskometer level" is an in-scope
   fact question, and "return" appears in "returns period" as an exit-load
   synonym.

4. refusals.py: refusal_c(), refusal_d(), refusal_b() -- pre-written templates,
   never model-generated. refusal_d must quote NO figure and link the official
   factsheet. refusal_c must not restate the premise in advisory language.
   refusal_b must never contain a URL.

5. tests/test_triage.py MUST include a release-gate test: load
   rag_bot/eval/eval_set.yaml, and assert that classify() does NOT return C or D
   for any row whose outcome is A. Zero false positives is a release blocker.
   Also test that each of the four prepared demo questions hits layer 1 with
   layer == "rules" and zero LLM calls.
   tests/test_schemes.py: exact match works, "equity" alone does not resolve,
   two matching schemes sets ambiguous, no match returns (None, False).

6. tests for refusals: neither refusal_c nor refusal_d contains a digit
   followed by "%"; refusal_c contains no advisory verb (should/recommend/best).

Run pytest and paste the output, including the class-A false-positive count.
Do not create other files.
```

---

## P8 — Generation, validation, assembly

**Goal:** class A answers that are ≤3 sentences, cite exactly one URL from retrieval
metadata, match the asked scheme, and contain no return figure.

**Files to create**

```
rag_bot/answer/prompts.py
rag_bot/answer/generator.py
rag_bot/answer/validate.py
rag_bot/answer/assemble.py
rag_bot/providers/ollama.py
rag_bot/providers/openai.py
rag_bot/pipeline.py
tests/test_sentences.py
tests/test_validate.py
tests/test_assemble.py
```

**`prompts.py`**

```python
SYSTEM_PROMPT = """..."""   # verbatim, per the spec below

def build_context(chunks: list[ScoredChunk]) -> str:
    """Numbered blocks, each carrying its citation material:
    [1] <scheme_name> - source: <publisher> - section: "<heading>"
        <chunk text>
    Return the numbered blocks only. No trailing commentary."""

def build_user_prompt(context: str, question: str) -> str:
    """Context, blank line, 'Question: <question>'."""
```

Verbatim system prompt:

```
You are a facts-only assistant for HDFC mutual fund scheme pages.

Rules:
1. Answer only from the numbered context. Never use outside knowledge.
2. Maximum 3 sentences. No preamble, no closing pleasantries.
3. Do not give investment advice. If asked whether to buy, hold, or sell,
   say that you only provide facts.
4. Do not state, compare, or estimate returns, NAV, or performance.
   None are present in the context.
5. End with exactly: "Source: <block number>"
6. If the context does not contain the answer, reply with exactly: NOT_IN_INDEX
```

**`validate.py`** — the structural layer. Every function returns data, never raises:

```python
def split_sentences(text: str) -> list[str]:
    """Decimal- and abbreviation-aware. "The ratio is 1.05%." is ONE sentence,
    not two. "Rs. 500" is one sentence. Never split on a period between digits."""

def enforce_max_sentences(text: str, limit: int) -> tuple[str, bool]:
    """(truncated_text, ok). Truncate on a sentence boundary."""

def parse_citation(text: str) -> int | None:
    """Extract the block number from a trailing "Source: [3]". None if absent."""

def check_single_url(chunks: list[ScoredChunk], cited_n: int | None) -> tuple[str | None, bool]:
    """(url, ok). Take the URL from the CITED CHUNK'S METADATA, never from model
    text. If cited_n is None, use rank 1. ok is False if more than one distinct
    source_id was cited -- keep rank 1, the rest belong in the chunk expander."""

def check_scheme(chunks, cited_n: int | None, asked_scheme_id: str | None) -> bool:
    """False when the cited chunk's scheme_id differs from the asked scheme."""

def find_return_figures(text: str) -> list[str]:
    """Return every performance/NAV figure found.

    A percentage is flagged ONLY when a return/NAV keyword appears in the SAME
    sentence. "1.05% expense ratio" is a required answer. "12.4% 1 year return"
    is not. Also flag currency amounts next to return keywords, NAV phrasing,
    and percentile/rank claims. A blanket number ban would break class A."""

def validate(raw: str, chunks: list[ScoredChunk], *, limit: int,
             asked_scheme_id: str | None) -> tuple[str, Validation, int | None]:
    """Run every check, return (final_text, Validation, cited_block_number)."""
```

`validate()` orchestration order: sentence limit → citation parse → single URL → scheme match
→ figure scan (strip offending sentences; if nothing survives, signal class D) →
`NOT_IN_INDEX` check.

**`assemble.py`**

```python
def assemble(outcome: Outcome, body: str, chunks: list[ScoredChunk],
             *, k: int, top_score: float | None, validation: Validation,
             latency: dict[str, float], triage_layer: str | None,
             reason: str | None) -> Answer:
    """Build the final Answer.

    For outcome A, append the footer EXACTLY:
      Source: <publisher> - <scheme_name>
      Last updated from sources: <DD Mon YYYY>

    The date is the fetched_at of the CITED source, formatted from the ISO
    string. The footer is assembled HERE, not by the model, so the model cannot
    omit or misdate it."""

def format_footer(fetched_at_iso: str | None) -> str:
    """'Last updated from sources: 27 Sep 2026'. Missing fetched_at renders as
    'Last updated from sources: unknown' -- never a fabricated date."""
```

**`pipeline.py`** — the orchestration, in exactly this order, one function per stage:

```python
def answer_question(question: str, *, cfg: Config, store: Store,
                    embedder: EmbeddingProvider, llm: LLMProvider,
                    sources: list[Source]) -> Answer:
    """1. scrub PII (P6)              -> redacted question
       2. triage (P7)                 -> C or D short-circuits here, no retrieval
       3. resolve scheme (P7)         -> scheme_id or ambiguity
       4. retrieve (P5), scheme-scoped
       5. gate (P5)                   -> class B short-circuits, LLM never called
       6. generate (P8)
       7. validate + assemble (P8)

    EVERY Answer carries retrieved_chunks, including refusals -- the class must
    see what retrieval found even when the assistant declines to answer.
    Every stage wrapped in a latency timer. Never raises: catch, and return an
    Answer with outcome=ERROR that still carries retrieved_chunks."""
```

Retrieval-failure versus generation-failure asymmetry: if generation raises, return an ERROR
answer that still shows the chunks. The demo keeps working, and no citation is fabricated
because citations come from chunk metadata rather than model text.

**Verify**

```bash
pytest tests/ -q
ollama list                                    # confirm the model is present
python3 -c "
from rag_bot.config import load
from rag_bot.pipeline import answer_question
from rag_bot.index.store import Store
from rag_bot.providers.minilm import MiniLMProvider
from rag_bot.providers.ollama import OllamaProvider
from rag_bot.sources.manifest import read_manifest
c = load()
store = Store(c.index_dir, c.embed_model, 'v1')
srcs = read_manifest(c.corpus_dir + '/manifest.csv')
llm = OllamaProvider(host=c.ollama_host, model=c.llm_model)
for q in ['What is the expense ratio of HDFC Large Cap Fund?',
          'What is the ELSS lock-in period?',
          'What was the NAV of HDFC Large Cap yesterday?',
          'Should I buy the HDFC Small Cap Fund?',
          'Which HDFC fund has the best 1-year return?',
          'What is the exit load?']:
    a = answer_question(q, cfg=c, store=store, embedder=MiniLMProvider(), llm=llm, sources=srcs)
    print('=' * 70)
    print(a.outcome.value, '|', a.scheme_name, '|', a.source_url, '|', a.last_updated)
    print(a.text)
    print('  validation:', a.validation)
"
```

You are checking, for each of the six: the outcome letter is right, class A has exactly one
URL, class A is ≤3 sentences, class D contains no `%` anywhere, and the footer date is real.
Run it **twice** and diff — answers should be near-identical at temperature 0.1. If they are
not, raise a specific question rather than lowering the temperature further.

**Cursor prompt**

```
Implement Phase P8: generation, structural validation, and answer assembly.
Implement P8 only. Create exactly:
  rag_bot/answer/prompts.py, rag_bot/answer/generator.py,
  rag_bot/answer/validate.py, rag_bot/answer/assemble.py,
  rag_bot/providers/ollama.py, rag_bot/providers/openai.py,
  rag_bot/pipeline.py, tests/test_sentences.py, tests/test_validate.py,
  tests/test_assemble.py

Read docs/architecture.md section 14 first.

1. prompts.py: the SYSTEM_PROMPT verbatim as specified in the phase doc,
   build_context(), build_user_prompt(). Context blocks are numbered and each
   carries scheme name, publisher, and heading.

2. validate.py: split_sentences() MUST be decimal- and abbreviation-aware.
   "The ratio is 1.05%." is ONE sentence. "Rs. 500" is one sentence. Never
   split on a period between digits. Then enforce_max_sentences(), parse_citation(),
   check_single_url(), check_scheme(), find_return_figures(), validate() per the
   spec. Every function returns data and NEVER raises.
   CRITICAL: find_return_figures() flags a percentage ONLY when a return/NAV
   keyword appears in the SAME sentence. "1.05% expense ratio" is a required
   answer and must NOT be flagged. A blanket number ban would break class A.
   CRITICAL: check_single_url() takes the URL from the cited chunk's METADATA,
   never from model text.
   validate() order: sentence limit, citation parse, single URL, scheme match,
   figure scan, NOT_IN_INDEX check.

3. assemble.py: assemble() and format_footer() per the spec. The "Source:" and
   "Last updated from sources:" footer is assembled in CODE, never by the model.
   The date is the fetched_at of the CITED source. A missing fetched_at must
   render "unknown" -- never a fabricated date.

4. providers/ollama.py and providers/openai.py implement the LLMProvider
   Protocol from P0. Ollama uses the local HTTP API and is the default.

5. pipeline.answer_question() per the spec: scrub PII, triage, resolve scheme,
   retrieve scheme-scoped, gate, generate, validate+assemble. Classes C and D
   short-circuit at triage WITHOUT retrieval. Class B short-circuits at the gate
   WITHOUT calling the LLM. Every Answer carries retrieved_chunks even for
   refusals and errors. Time every stage. Never raise -- return an ERROR answer
   that still carries retrieved_chunks.

6. tests: decimal sentence splitting, one-URL enforcement, a figure in a return
   sentence flagged, a percentage in an expense-ratio sentence NOT flagged,
   scheme mismatch detected, footer date from the cited source, NOT_IN_INDEX
   converted to a clean class B.

Run pytest and paste output. Do not create other files.
```

---

## P9 — UI and trace logging

**Goal:** the tiny Streamlit app with four visually distinct answer states, the chunk
expander, the disclaimer, and redacted trace logs.

**Files to create**

```
app.py
rag_bot/ui/__init__.py
rag_bot/ui/components.py
rag_bot/ui/sidebar.py
rag_bot/trace.py
deliverables/disclaimer.txt
```

**`deliverables/disclaimer.txt`** — deliverable D5, read by the app at startup so the file and
the UI cannot drift:

```
Facts-only. No investment advice.
```

**`ui/components.py`**

```python
def render_answer(answer: Answer) -> None:
    """Render one message. FOUR visually distinct states -- this is a hard
    requirement, not styling preference:
      A  normal card, a source link, the last-updated footer
      B  muted, no source, a reason chip reading "Not in sources", lists the
         six covered topics
      C  muted, no source, chip "Advice not provided"
      D  muted, no source, chip "Returns not provided", factsheet link only
    A refusal must never be mistakable for an answer."""

def render_chunks(answer: Answer) -> None:
    """Expandable "Retrieved chunks (N)". One row per chunk:
    [rank] <scheme> - "<heading>" - score - n_tokens. Expands to the full
    verbatim chunk text. Populated for EVERY outcome, including refusals and
    errors -- showing what retrieval found is the teaching moment."""

def render_timing(answer: Answer) -> None:
    """scrub / triage / retrieve / generate ms, plus triage_layer."""

def render_welcome(example_questions: list[str]) -> None:
    """Welcome line, the disclaimer line, and 3 example questions as buttons."""
```

**`ui/sidebar.py`** — corpus status line (files, chunks, indexed date), Rebuild and Reset
buttons, and the `k` / `min_score` sliders. A **warning banner** naming any source whose
manifest status is `fetch_failed` or `blocked` — a silently missing scheme is how a demo ends
up citing the wrong fund.

**`app.py`** — the only file importing `streamlit`. Chat input, message history, status
indicator (`retrieving` → `generating` → `done`), and calls into `pipeline.answer_question`.
A Streamlit rerun must not duplicate an answer: key each cached result on a hash of
`(question, k, min_score)`.

**`trace.py`**

```python
def log_query(answer: Answer, question: str) -> None:
    """Append one JSONL line per query to data/logs/run.jsonl.

    THE QUESTION MUST BE REDACTED with safety.pii.scrub_for_log before writing.
    A committed log file is the easiest way to leak a PAN into a public repo.
    Fields: ts, outcome, scheme, top_score, k, cited_url, url_count, sentences,
    figures, pii, triage_layer, latency."""
```

The six pipeline stages are surfaced as **in-app text panels** — chunk lists, scores, the
validation report — because the brief forbids back-end screenshots (PRD §10).

**Verify**

```bash
pytest tests/ -q
streamlit run app.py
```

Then click through all six questions from the P8 verify block and confirm: the four states are
visually distinct, the expander opens for a refusal, the disclaimer is visible, and
`data/logs/run.jsonl` shows `pii=clean` or `pii=redacted:<rules>` and never a raw PAN.

**Cursor prompt**

```
Implement Phase P9: the Streamlit UI and trace logging. Implement P9 only.
Create exactly: app.py, rag_bot/ui/__init__.py, rag_bot/ui/components.py,
rag_bot/ui/sidebar.py, rag_bot/trace.py, deliverables/disclaimer.txt

1. deliverables/disclaimer.txt contains exactly one line:
   "Facts-only. No investment advice."
   The app reads this file at startup. Do not hardcode the string in Python.

2. ui/components.py: render_answer(), render_chunks(), render_timing(),
   render_welcome() per the spec. FOUR visually distinct answer states is a hard
   requirement: A normal with a source link and footer; B/C/D muted, no source,
   each with its own reason chip. A refusal must never look like an answer.
   render_chunks() must work for EVERY outcome including refusals and errors.

3. ui/sidebar.py: corpus status (files, chunks, indexed date), Rebuild and Reset
   buttons, k and min_score sliders, and a WARNING BANNER naming any source
   whose manifest status is fetch_failed or blocked.

4. app.py is the only file that imports streamlit. Chat input, history, a
   status indicator (retrieving -> generating -> done), and a call into
   pipeline.answer_question. Cache each result on a hash of
   (question, k, min_score) so a Streamlit rerun does not duplicate an answer.

5. trace.py: log_query() appends one JSONL line to data/logs/run.jsonl with the
   fields in the spec. THE QUESTION IS REDACTED via safety.pii.scrub_for_log
   before writing. Never write a raw query.

6. Surface the pipeline stages as in-app text panels (chunk lists, scores, the
   validation report). Do NOT add screenshots or back-end tooling -- the brief
   forbids back-end screenshots.

Do not create other files. Do not modify P0-P8 modules except to import from them.
```

---

## P10 — Eval harness, chunking decision, sample Q&A **\[DELIVERABLE D4\]**

**Goal:** measured per-class accuracy, a written chunking decision, and a generated
`sample_qa.md` covering all four classes.

**Files to create**

```
rag_bot/eval/classes.py
rag_bot/eval/run_eval.py
rag_bot/eval/report.py
docs/chunking-decision.md
deliverables/sample_qa.md     (generated)
```

**`classes.py`** — the shared A/B/C/D contract, used by both eval and tests:

```python
def classify_answer(answer: Answer, row: dict) -> tuple[bool, list[str]]:
    """Return (passed, failures) for one eval row.

    For outcome A: cited source_id == expect_source_id; scheme matches;
                   <= 3 sentences; exactly one distinct URL;
                   every expect_keyword present in the text;
                   find_return_figures(text) is empty.
    For outcome B: outcome is B; answer.source_url is None;
                  no "http" anywhere in the text.
    For outcome C: outcome is C; source_url is None; a link is present if
                   the row sets require_link.
    For outcome D: outcome is D; source_url is None;
                  find_return_figures(text) is empty AND no "%" in the text.

    Check every forbidden_pattern as a case-insensitive substring, EXCEPT the
    class-D "%" rule which is handled by find_return_figures plus an explicit
    percent check."""
```

**`run_eval.py`** — reports **per class**, never one blended number:

```
python3 -m rag_bot.eval.run_eval                      # default config
python3 -m rag_bot.eval.run_eval --chunk-strategy B   # re-run the chunking experiment
python3 -m rag_bot.eval.run_eval --calibrate          # sweep min_score, print the ROC-ish table
```

Pass bars (architecture §18.1): class A ≥90%; classes B, C, D = **100%**; router
false-positives on class-A rows = **0**; PII fixtures = 100%; no chunk >256 word-pieces.

`--calibrate` sweeps `RAG_MIN_SCORE` from 0.20 to 0.60 and prints, for each value, how many
in-corpus and out-of-corpus questions it would accept or reject. Pick the value with the
widest gap and record the chosen number **with the model name** in
`docs/chunking-decision.md`. A similarity threshold without its model name is meaningless.

**`docs/chunking-decision.md`** — the brief asks for a data-driven chunking decision, so write
it with real numbers:

1. The five candidate strategies from architecture §9.3.
2. The 256-word-piece cap and why it disqualifies any larger target.
3. Observed per-candidate topic hit-rate over the 30 labeled questions.
4. The `RAG_MIN_SCORE` calibration table and chosen value.
5. The chosen strategy, its measured effect, and its cost.
6. An honest note on what the experiment does not cover (a 5-page corpus; results will not
   transfer to a large corpus).

**`report.py`** — also generates `deliverables/sample_qa.md` from the same run, with a hard
assertion that at least one B, one C, and one D row appear. The failure mode for this
submission is a Q&A file containing fifteen variations of "here is the expense ratio."

**Verify**

```bash
pytest tests/ -q
python3 -m rag_bot.eval.run_eval --calibrate
python3 -m rag_bot.eval.run_eval
python3 -m rag_bot.eval.run_eval --chunk-strategy A
python3 -m rag_bot.eval.run_eval --chunk-strategy C
cat deliverables/sample_qa.md
```

The three `100%` classes are gates, not targets. Do not lower a gate to make a report look
better — a fabricated citation, an advice answer, or a quoted return figure is a correctness
failure, not a metric to trade off. If a gate fails, fix the cause and say so in the report.

**Cursor prompt**

```
Implement Phase P10: the eval harness, the chunking-decision write-up, and the
generated sample Q&A. Implement P10 only. Create exactly:
  rag_bot/eval/classes.py, rag_bot/eval/run_eval.py, rag_bot/eval/report.py,
  docs/chunking-decision.md, deliverables/sample_qa.md (generated, not handwritten)

1. classes.py: classify_answer(answer, row) -> (passed, failures) per the spec,
   covering outcomes A, B, C, D. Use answer.validate.find_return_figures for the
   figure checks rather than inventing a new detector.

2. run_eval.py: CLI with a default run, --chunk-strategy, and --calibrate.
   Report PER CLASS, never one blended number, because one number would hide a
   perfect class A alongside a failing class D.
   Pass bars: class A >= 90%; classes B, C, D = 100%; router false-positives on
   class-A rows = 0; PII fixtures 100%; no chunk over 256 word-pieces.
   --calibrate sweeps RAG_MIN_SCORE 0.20..0.60 and prints how many in-corpus and
   out-of-corpus questions each value would accept or reject. Recommend the
   value with the widest gap.

3. report.py writes deliverables/sample_qa.md from the same run, with a hard
   assertion that at least one B, one C, and one D row appear in the output.
   Raise if any class is missing rather than writing an incomplete file.

4. docs/chunking-decision.md: write it with REAL numbers from your runs, not
   placeholders. Include the five candidate strategies, the 256 word-piece cap
   and why it disqualifies larger targets, per-candidate topic hit-rate over the
   30 labeled questions, the RAG_MIN_SCORE calibration table and the chosen value
   WITH the embedding model name, the chosen strategy with its measured effect and
   cost, and an honest note that results from a 5-page corpus do not transfer to a
   large corpus. If you do not have real numbers yet, say so in the file and mark
   the section TODO -- do not invent figures.

5. Do NOT lower a pass gate to make a report look better. If a gate fails, fix
   the cause and state it in the report.

Run the eval and paste the full per-class output. Do not create other files.
```

---

## P11 — README, demo script, hardening

**Goal:** the submission runs from a clean clone, and the demo is repeatable.

**Files to create / modify**

```
README.md                              (create)
deliverables/demo_script.md            (create)
requirements.txt                       (pin exactly, final)
.env.example                           (final)
```

**README must contain, per deliverable D3**

1. One-paragraph statement of what this is and the facts-only limit.
2. Scope: HDFC AMC, the five schemes by name, the source count, and the **fetch date** —
   with an explicit sentence that figures are as-of that date and may have changed since.
3. Setup: Python version, `pip install -r requirements.txt`, `ollama pull <model>`, and the
   one command to build the index and one to run the app.
4. The disclaimer: "Facts-only. No investment advice."
5. **Known limits**, stated plainly: five schemes only; no returns by design; no advice by
   design; the citation is a page snapshot, not a live lookup; MiniLM is a sentence-similarity
   model and can misrank similar schemes; answers are as-of the fetch date.
6. Rebuild and reset commands.
7. An explicit line on the sourcing question: which publisher each of the five sources
   belongs to, and that this is disclosed rather than presented as official AMC/SEBI/AMFI
   material (PRD Q1).

**`demo_script.md`** — a ≤3-minute walkthrough, written to be followed literally, with the
exact questions and the expected outcome for each:

| # | Do this | Say this | Expect |
|---|---------|----------|--------|
| 1 | Show the sidebar status line and Rebuild | "Five HDFC schemes, snapshot once, indexed locally" | files, chunks, fetch date |
| 2 | Ask "What is the expense ratio of HDFC Large Cap Fund?" | "Five hundred chunks, we take the top three" | class A, one link, footer date |
| 3 | Open the chunk expander | "This is the actual text the model was given" | verbatim sanitized chunk |
| 4 | Ask "What is the ELSS lock-in period?" | "Notice the scheme filter — five similar funds" | class A, correct scheme |
| 5 | Ask "Which HDFC fund has the best 1-year return?" | "Returns are not in the index at all — removed at load" | class D, no figures, factsheet link |
| 6 | Ask "Should I buy the small cap fund?" | "Facts only. Here's the relevant education page" | class C, educational link |
| 7 | Ask "What was the NAV yesterday?" | "Not in our sources — no citation invented" | class B, no link |
| 8 | Paste a PAN into the chat | "Scrubbed before anything else ran" | `[PAN redacted]`, log shows pii=redacted |

Steps 5 and 8 are the two that distinguish this from a document lookup. Rehearse until they
are as reliable as the happy path.

**Final hardening checklist**

- [ ] `git clone` to a clean directory; follow the README; app runs
- [ ] No `.env` needed for the default configuration
- [ ] `ollama list` shows the exact model named in the README
- [ ] `pytest` green with no network and no model download
- [ ] `git status` clean; `data/index/` and `data/logs/` ignored; `data/corpus/snapshots/` committed
- [ ] `git log --all -p | grep -iE "pan|aadhaar|otp"` finds no secrets or PII
- [ ] All four classes in `deliverables/sample_qa.md`
- [ ] All five deliverables present
- [ ] Every PRD §12 acceptance box ticked, honestly
- [ ] The demo was run twice end to end with the same results

**Cursor prompt**

```
Implement Phase P11: README, demo script, and final hardening. Implement P11
only. Create exactly: README.md, deliverables/demo_script.md. Update
requirements.txt to exact pins and .env.example if anything drifted.

1. README.md must cover, in this order: what this is and the facts-only limit;
   scope (HDFC AMC, the five schemes by name, source count, and the FETCH DATE,
   with an explicit sentence that figures are as-of that date); setup (Python
   version, pip install, ollama pull <model>, one command to build the index,
   one to run the app); the disclaimer "Facts-only. No investment advice.";
   KNOWN LIMITS stated plainly (five schemes only, no returns by design, no
   advice by design, citation is a page snapshot not a live lookup, MiniLM is a
   sentence-similarity model that can misrank similar schemes, answers are
   as-of the fetch date); rebuild and reset commands; and an explicit SOURCING
   NOTE naming which publisher each of the five sources belongs to, disclosed
   rather than presented as official AMC/SEBI/AMFI material.

2. deliverables/demo_script.md: a <=3-minute walkthrough with a table of
   step / do this / say this / expect, using the exact questions and expected
   outcomes specified in the phase doc. Steps 5 (performance refusal) and 8
   (PII scrub) are the two that distinguish this from a document lookup and must
   be rehearsed to the same reliability as the happy path.

3. Then verify and report on every item of the final hardening checklist in the
   phase doc: clean-clone run, no .env needed, model present, pytest green with
   no network, gitignore correct, no secrets or PII in git history, all four
   classes in sample_qa.md, all five deliverables present, and the demo run
   twice with the same results.

Report honestly. If a check fails, say which one and why. Do not claim a
check passed that you did not actually run.
```

---

## Blocking questions

Answer these before finishing P1. Both are cheap to answer and both can stop the project.

**Q1 — Are the groww.in pages acceptable sources?**
The brief says to collect pages from AMC/SEBI/AMFI and bans third-party blogs, but supplies
five groww.in URLs. groww.in is a broker, not the AMC or a regulator. Ask the instructor.
Whatever the answer, `publisher` and `source_tier` already exist per source, so applying it
is a filter in `sources.yaml` rather than a code change.

**Q4 — Are the pages fetchable, and are the figures current?**
P1 answers this empirically in about ten minutes. The failure modes to watch for: a robots
block, a JavaScript-rendered shell with no static text, a login or region wall, or a consent
dialog standing where the content should be. If all five are blocked, the fallback is
official AMC/SEBI/AMFI pages — which is the brief's own stated preference, so that outcome is
recoverable rather than fatal.

**Q7 — Hosting or the video fallback?**
Decide before P11 so the effort split is right. The video path needs a scripted, repeatable
demo, which is why `demo_script.md` is written to be followed literally.

---

## Global release gates

All of these must hold before submission. They are gates, not targets, and none of them may
be lowered to make a report look better.

| Gate | Bar |
|------|-----|
| Class B never fabricates a citation | 100% |
| Class C always refuses with an educational link | 100% |
| Class D always refuses and quotes zero figures | 100% |
| Router false-positives on the six in-scope topics | 0 |
| PII must-redact and must-not-redact fixtures | 100% |
| No chunk exceeds 256 word-pieces | 100% (build-time assert) |
| Every class-A answer has exactly one source URL | 100% |
| Class-A accuracy | ≥90% |
| Retrieval topic hit-rate | ≥85% |
| `pytest` green with no network and no model download | required |
| No PII or secrets in git history | required |
| All five deliverables present | required |
