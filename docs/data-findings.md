# Source data findings

Empirical findings from probing the brief's five scheme pages on 2026-09-27. These
findings drove three deviations from `implementation.md`, each recorded in
**Deviations** below. Nothing here is speculative: every claim is reproducible with
the commands in each section.

---

## 1. One of the brief's five URLs is dead

| Source | URL from the brief | Status |
|---|---|---|
| S1 | `hdfc-large-cap-fund-direct-growth` | 200 |
| S2 | `hdfc-equity-fund-direct-growth` | 200 |
| S3 | `hdfc-elss-tax-saver-fund-direct-plan-growth` | 200 |
| S4 | `hdfc-small-cap-fund-direct-growth` | 200 |
| S5 | `hdfc-balanced-advantage-fund-direct-plan-growth` | **404** |
| S5 | `hdfc-balanced-advantage-fund-direct-growth` | **200** |

The brief appears to have copied S3's `-direct-plan-growth` suffix onto S5, which
has no `-plan-` slug. Confirmed twice: by direct fetch, and via Groww's own search
endpoint, which returns `hdfc-balanced-advantage-fund-direct-growth` for the fund.

> **Deviation 1.** P1 registers the corrected slug and records the substitution in
> the manifest `notes` column. The brief's URL is not silently replaced.

### Neither heuristic catches this — only the status check does

The P1 spec detects a dead source by checking for "under 200 characters of visible
text". Measured against the brief's dead S5 URL:

| Probe | Result | Verdict |
|---|---|---|
| HTTP status | `404` | **the only thing that catches it** |
| Visible text after tag stripping | 1,919 chars | spec's "JS shell < 200 chars" rule **passes it** |
| `__NEXT_DATA__` payload present | `True` | liveness probe **passes it** |
| Response body size | 38,949 bytes | size rule **passes it** |

The page is a fully-rendered "not found" page: nav, footer, and readable body text,
served with an error status. So Groww serves a *soft-looking* 404 with enough
structure to fool content-based checks. `fetch.py` therefore checks the status code
first and treats it as authoritative; the content checks are only defence in depth
and are not sufficient on their own.

> **Deviation 2.** `fetch.py` must check the HTTP status code. Content-size and
> `__NEXT_DATA__` checks cannot substitute for it.

---

## 2. The in-scope facts live in JSON, not in the HTML tables

Each page embeds a `__NEXT_DATA__` script tag containing
`props.pageProps.mfServerSideData`: **97 keys**, identical schema across all five
pages. This is where every in-scope fact lives.

The four server-rendered `<table>` elements are, in order: **SIP returns, portfolio
holdings, category rank, peer returns**. Three of the four are performance data that
architecture §8.1 wants removed, and **none** contains fees, lock-in, benchmark, or
riskometer. BeautifulSoup table-flattening as originally specified in P3 would have
extracted **zero** of the in-scope topics while ingesting the returns tables.

> **Deviation 3.** P2's loader extracts a fixed allowlist of fields from
> `mfServerSideData` rather than scraping and flattening HTML tables. See §4 for why
> this is a correctness fix, not just a simplification.

---

## 3. Field map (97 keys)

Only these 22 fields are forbidden. They are the never-indexed set enforced by
`FORBIDDEN_FIELDS` in P2.

### In scope — the allowlist

| Topic | Field(s) | S3 (ELSS) value |
|---|---|---|
| identity | `fund_name` | `HDFC ELSS - Tax Saver Fund` |
| identity | `scheme_name` | `HDFC ELSS Tax Saver Fund Direct Plan Growth` |
| identity | `category` / `sub_category` | `Equity` / `ELSS` |
| identity | `amc` / `plan_type` / `scheme_type` | `HDFC` / `Direct` / `Growth` |
| identity | `isin` | `INF179K01YS4` |
| identity | `launch_date` / `registrar_agent` | `01-Jan-2013` / `CAMS` |
| expense_ratio | `expense_ratio` | `1.21` |
| expense_ratio | `base_expense_ratio` | `0.97` |
| exit_load | `exit_load` | `Nil` |
| min_sip | `min_sip_investment` | `500` |
| min_sip | `min_investment_amount` | `500` |
| lock_in | `lock_in` (dict) | `{"years": 3, "months": 0, "days": 0}` |
| riskometer | `nfo_risk` | `Moderately High` |
| benchmark | `benchmark_name` | `NIFTY 500 Total Return Index` |
| — | `description` | clean mandate text, no performance language |

A second, independent lock-in source exists at `additional_details.lock_in_yrs` (= 3),
which the P2 tests should cross-check `lock_in` against.

### Never indexed — `FORBIDDEN_FIELDS`

`nav` `nav_date` `return_stats` `simple_return` `sip_return` `stats`
`peerComparison` `analysis` `holdings` `aum` `meta_desc` `meta_title` `amc_info`
`rta_details` `stp_details` `swp_details` `groww_rating` `crisil_rating`
`portfolio_turnover` `historic_fund_expense` `historic_exit_loads`

Notes on the more dangerous ones:

- `nav` is a bare float (`1447.383`) and `nav_date` is `25-Sep-2026`.
- `return_stats`, `simple_return`, `sip_return` use camelCase keys (`return1d`,
  `return1w`, `return1m`, `return3y`) with unlabelled decimal values.
- `stats` is `{"type": "FUND_RETURN", "stat_1y": -6.51, "stat_3y": 12.47, ...}`.
- `analysis` holds **analyst recommendations** (`"analysis_type": "CONS"` =
  consolidate). This is investment advice, not a scheme fact. Excluding it is also
  what keeps the class-C refusal rule clean.
- `amc_info` and `rta_details` carry postal addresses and phone numbers, so they are
  the only corpus-side PII carriers — and excluding them removes the need for any
  corpus-side PII scrubber (the P5 input-boundary scrub is unaffected).

---

## 4. Why the denylist scrubber had to go

A denylist regex was run over all 97 fields of the S3 page to test the original P2
design. It failed in **both** directions.

### False positives — it would have deleted in-scope facts

| Field | Value | Matched | Why the match is wrong |
|---|---|---|---|
| `benchmark_name` | `NIFTY 500 Total Return Index` | `return` | "Total Return Index" is the benchmark's proper name. Benchmark is an in-scope topic. |
| `stamp_duty` | `0.005% (from July 1st, 2020)` | `%` | A tax fee, not a return. |
| `category_info` | "Invests in stocks as per the scheme's mandate…" | `performance`-family word | Mandate description, not a figure. |

### False negatives — it would have let returns through

| Field | Value | Why it was missed |
|---|---|---|
| `nav` | `1447.383` | Bare float: no keyword, no `%`. |
| `return_stats` | `{"return1d": 0.38, "return1w": -1.22, ...}` | `\breturn\b` cannot match `return1d` (no word boundary before a digit), and the values carry no `%`. |
| `simple_return` | same shape | same reason |
| `sip_return` | same shape | same reason |

So the original design would have **stripped the benchmark name** — a graded topic —
while **passing NAV and three return payloads** into the index. Under allowlist
extraction both failure modes are impossible by construction, and
`ALLOWED_FIELDS ∩ FORBIDDEN_FIELDS == ∅` becomes an assertable structural invariant
rather than an aspiration.

The `removed_pct > 40%` over-stripping tripwire is also retired: under allowlist
extraction nothing is "removed", so the metric is meaningless. It is replaced by a
sharper gate — the builder **hard-fails** if any forbidden field or pattern appears
in a chunk.

---

## 5. Open scope gaps

### `statement` has no source — dropped from scope

PRD §4 listed "how to download a statement" as in scope. **Zero** of the 97 keys on
any of the five pages mentions a statement, download, or capital gains; it is an
investor-portal help-centre topic, not a fund attribute. Per decision, it is dropped
and documented as a known limit rather than sourced from an invented page.

> **Deviation 4.** `Topic` in `rag_bot/types.py` drops `"statement"`.

### "Six topics" was never verifiably six

PRD §4 read: *"the six named in the brief: expense ratio · exit load · minimum SIP ·
ELSS lock-in period · riskometer **and** benchmark · how to download a statement."*
That is six listed items, but "riskometer and benchmark" is two distinct facts from
two distinct fields, so the list actually held seven. The docs' coverage maths
(`6 topics × 5 = 30` rows) then quietly disagreed with the `Topic` literal, which had
seven entries.

The resolved position: **riskometer and benchmark are separate topics**, and with
`statement` dropped there are exactly **6** topics, so the 30-row class-A eval set is
unchanged. The count is now verifiable by enumeration rather than by coincidence.
PRD §4 should list the six separately.

### Class D has no factsheet link

`brochure_link` is `None` on all five pages and there are **zero** PDF links anywhere.
P3's original instruction to "scan the raw HTML for a factsheet PDF link and set
`factsheet_url` if null" cannot succeed against these sources.

Partial substitute: `sid_url` is the official AMC domain (`https://www.hdfcfund.com`
for S3), which is at least an official publisher, but it is a homepage and not a
factsheet. `scheme_info_link` is `null` everywhere. **Unresolved** — see below.

---

## 6. Operational notes for P1

- **Fetches are slow and have timed out.** One probe took 8.6s; a later 30s timeout
  aborted at 112KB of a 453KB page. Use a ≥120s timeout and one retry.
- **Send a browser User-Agent.** Requests get redirected/served differently without
  one.
- **Size range:** 453–497KB for S1–S4, but **816KB for S5**, so a rough page-size
  expectation must not be used as a sanity check. All five carry the same 97-key
  schema.
- `nav_date` is `25-Sep-2026` on all five, two days before this probe. This **answers
  PRD Q4**: the fee figures are current as of the fetch date, so a provenance footer
  showing `fetched_at` is accurate.
- `nfo_risk` formatting is **inconsistent across pages**: S1/S4/S5 yield
  `Moderately High Riskometer`, S2/S3 yield `Moderately High`. Normalise by stripping
  a trailing `Riskometer` or two schemes will be phrased differently in the corpus.
- **`exit_load` has three distinct shapes** across the five schemes, so P2 must not
  assume a single format:
  - S1/S2/S4: `Exit load of 1% if redeemed within 1 year` (some carry a trailing `\r\n`)
  - S3: `Nil`
  - S5: `Exit Load for units in excess of 15% of the investment, 1% will be charged
    for redemption within …` — a *partial* load keyed to a 15% threshold, truncated
    in this field
- **`lock_in` is null on four of five schemes**; only S3 has
  `{"years": 3, "months": 0, "days": 0}`. Render null as an explicit "None" so the
  corpus can answer "is there a lock-in?" negatively rather than being silent.
- **S2 has been rebranded.** The brief and PRD call it "HDFC Equity Fund"; the live
  page reports `fund_name = "HDFC Flexi Cap Direct Plan-Growth"` and
  `sub_category = "Flexi Cap"`. The alias table covers both names, and
  `tests/test_sources.py::test_live_fund_name_resolves_for_every_snapshot` fails if
  the live name stops being resolvable.
- **Alias design is constrained by real values.** `amc` is `HDFC` for all five, so
  bare `"HDFC"` must never be an alias (ambiguous across all five sources). S1 and S4
  carry `Equity` category labels while S2 is named `* Equity Fund` in the brief, so
  bare `"equity"` must not be an alias either. S5's `sub_category` is
  `Dynamic Asset Allocation` (its `category` is `Hybrid`), so bare
  `"balanced advantage"` is safe as a *fund-name* fragment but not as a category.

---

## 7. Index findings (P3)

These are measured on the real corpus with real `all-MiniLM-L6-v2` vectors, not
asserted. All three contradict something the spec assumed, so each is recorded
with the measurement that forced the change.

### 7.1 chromadb 1.5.9 does NOT use `1 - cosine`

`architecture.md` §12 sets `RAG_MIN_SCORE = 0.35` as a cosine floor, and the
P3 brief assumed a returned distance converts with `score = 1 - d`. Measured
against analytically-known cosines of normalised vectors:

| metric | max err of `1 - d` | `1 - d/2` | `1 - d²/2` |
|---|---|---|---|
| default (squared L2) | **1.005** | 1.1e-7 | 1.016 |
| `hnsw:space = cosine` | 6e-8 | 0.503 | 0.500 |

So on the default metric the correct conversion is `1 - d/2`, and the spec's
formula is wrong there. Left uncorrected, a completely unrelated document scores
**-1.01** and a merely-adjacent one **-0.59** — both far below 0.35 — so *every*
question would be refused as class B. `Store` now sets `hnsw:space: "cosine"`
explicitly on every collection, which makes `1 - d` correct and makes the
conversion independent of chromadb changing its default.

### 7.2 A chunk must name its scheme to be retrievable

§5.2 defines `Chunk.text` as the fact text alone, e.g. `Expense ratio: 1.03`, and
§14.1 puts the scheme name in the citation header *above* it. That is right for
the LLM and wrong for the retriever: the five chunks of one topic are then five
near-identical strings differing only in a number, and the embedding has almost
no scheme-discriminating signal.

Measured on 10 scheme-specific questions, gold = the named scheme's chunk:

| embedded text | correct scheme ranked 1st | median rank | top-to-bottom spread |
|---|---|---|---|
| bare fact text | **3/10** | 3 of 5 | 0.01 – 0.03 (noise) |
| `scheme_name` + `heading` + fact text | **8/10** | 1 of 5 | 0.10 – 0.23 |

`Store.embedding_text()` adds the prefix for embedding only; the clean text is
kept in metadata under `body` and restored on read, so §5.2's "the only field
sent to the LLM" still holds. Two consequences that were easy to get wrong:

- the 256 word-piece cap must be checked on the **prefixed** string, because that
  is what MiniLM actually encodes (max in this corpus: 88, not 75);
- the embedding cache is keyed on the **embedded text**, not `Chunk.content_hash`.
  Keyed on the content hash it would have served the pre-prefix vectors for ever.

The two remaining misses are `exit load of HDFC Equity Flexi Cap` and
`minimum SIP for HDFC Equity Flexi Cap` — both rank S1 first. S2's live name is
`HDFC Flexi Cap Direct Plan-Growth` while the brief calls it HDFC Equity Fund, so
the query's wording is nearest to a different scheme. Alias resolution is a
retriever concern (§14.4), not an embedding fix.

### 7.3 `RAG_MIN_SCORE = 0.35` is not a calibrated threshold

Probe of 9 in-domain and 10 out-of-domain questions against the real index:

- in-domain: 0.82 / 0.77 / 0.74 / 0.73 / 0.61 / 0.61 / 0.60 / **0.27** / **0.04**
- out-of-domain: 0.01 … 0.20, then **0.30** and **0.37**

**The two distributions overlap** (`What is the TER?` is in-domain at 0.04;
`stock market tips for tomorrow` is out-of-domain at 0.37), so no threshold
separates them. 0.35 happens to pass 7/9 in-domain and 1/10 out-of-domain, but
that is a coincidence of this corpus, not a property of the metric.

Two causes, both actionable in P4:

1. **Vocabulary gaps are invisible to the score.** The corpus says "expense
   ratio" and never "TER", so a correct question scores 0.04. A synonym/alias
   map is needed, not a lower threshold — lowering it would also admit
   `stock market tips`.
2. **The score measures "does the query name a scheme we have", not "is the
   topic right".** Because the prefix put every scheme name in, any
   scheme-naming query scores 0.6–0.82 regardless of topic. Topic therefore has
   to come from routing (§14.4 rules + LLM classifier), with the score used only
   as a weak floor. This is a direct argument for keeping the two-layer triage
   rather than trusting a single score gate.

### 7.4 The value-overlap guard needs evidence-grade thresholds

P2's `guard()` is a no-op in practice: it compares forbidden field *names*
(`nav`) against rendered *labels* (`NAV`), two sets that cannot intersect by
construction. The builder therefore adds a second layer that looks for
forbidden fields' **values** inside the rendered text.

The first version matched raw substrings with a 4-character floor and **failed
the build on the clean S5 snapshot**: `peerComparison` is a *list of other
funds' records*, whose categorical fields contain `"High"`, and `"High"` occurs
legitimately in `Riskometer level: Moderately High`. Current rules, both forced
by real payloads:

- strings shorter than 12 chars are ignored — short categorical words (`High`,
  `Medium`, `Open-ended`) carry no evidence of a leak;
- numerics shorter than 4 chars are ignored — a rating of `3` or a tenure of `1`
  appears in ordinary text by chance;
- both classes match on **word boundaries**, so `1.2` cannot match inside `1.21`
  and `High` cannot match inside `Moderately High`;
- any value also carried by an allowlisted field is ignored, which is what keeps
  `category_info`'s repeated `Large Cap` from flagging the legitimate
  `category: Large Cap`.

Residual limitation, stated plainly: a leak of a short categorical value (a bare
`"High"`) would still pass. That is acceptable only because the allowlist
extraction makes such a leak structurally impossible; this layer is defence in
depth, not the guarantee.

---
## Unresolved — needs a human decision

_Nothing outstanding. The class-D factsheet question was the last open item and is
resolved below (2026-09-27)._

## Resolved

1. **Class D factsheet link — RESOLVED 2026-09-27.** The five groww pages carry no
   factsheet (`brochure_link` null, zero PDF links), so the link is the AMC's own
   factsheet page: `https://www.hdfcfund.com/mutual-funds/factsheets`, set as
   `factsheet_url` on all five sources.

   Chosen over the two alternatives for three reasons. It is the **official
   publisher**, so class D points at the AMC rather than the broker. It is **not
   month-stamped** — the per-scheme PDFs are `Fund Facts - <Scheme>_July 26.pdf`
   and would rot within a month, and a dead link in a refusal is worse than no
   link. And it is architecturally the right destination: **the corpus
   deliberately contains zero returns**, so handing the user the official
   performance document is the honest exit rather than an evasion. A refusal that
   says "returns are not in the indexed sources — here is the AMC's factsheet"
   is exactly the behaviour §13 asks for.

   **Verification caveat, stated rather than glossed.** An automated fetch of that
   URL returns **HTTP 403** (AMC bot protection), not 200. Existence was confirmed
   from the search index, which returned the page's real content, and a browser
   opens it normally. Per the P1 lesson — content heuristics are not a status
   check — the 403 is recorded as the observed status rather than asserting 200.
   The corpus-verified `sid_url` (`https://www.hdfcfund.com`) remains the fallback.

   This also fixed a silent data-loss bug: `read_manifest` hardcoded
   `factsheet_url=None` and `aliases=()`, so registry metadata never reached
   chunks. The manifest has no column for either because it records what was
   *fetched* and this page deliberately is not; `read_manifest` now takes the
   registry and joins on `source_id`. The link is verified present in all 45
   chunk records.

2. **Q1: is `groww.in` an acceptable publisher? YES — resolved 2026-09-27.** The five
   groww.in pages are acceptable and form the primary corpus. The reasoning: Groww is
   a regulated broker publishing standardised scheme data, which is materially
   different from the third-party blog/forum commentary the brief's ban targets. The
   `source_tier` stays `"brief"` for all five so that no row is ever presented as an
   official AMC/SEBI/AMFI document, and every manifest row discloses its `publisher`.
   Any genuine AMC/SEBI/AMFI page added later uses `source_tier: "official_ref"`.
