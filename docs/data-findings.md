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

### The "JS shell" heuristic would not have caught this

The P1 spec detects a dead source by checking for "under 200 characters of visible
text". The Groww 404 page returns **38,950 bytes** of perfectly readable text
("Page not found", nav, footer), so that heuristic passes it. Only an explicit HTTP
status check catches this case.

> **Deviation 2.** `fetch.py` must check the HTTP status code, not just content size.

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
- **Size range:** ~450–500KB per page, so ~2.3MB of committed HTML for five
  snapshots.
- `nav_date` is `25-Sep-2026`, two days before this probe. This **answers PRD Q4**:
  the fee figures are current as of the fetch date, so a provenance footer showing
  `fetched_at` is accurate.
- `nfo_risk` formatting is **inconsistent across pages**: S1 yields
  `Moderately High Riskometer`, S3 yields `Moderately High`. Normalise by stripping a
  trailing `Riskometer` or the two schemes will be phrased differently in the corpus.
- **Alias design is constrained by real values.** `amc` is `HDFC` for all five, so
  bare `"HDFC"` must never be an alias (ambiguous across all five sources). S2 is
  named `HDFC Equity Fund` while the others carry `Equity` category labels, so bare
  `"equity"` must not be an alias either. Derive aliases from the observed `fund_name`
  values, not from guesswork.

---

## Unresolved — needs a human decision

1. **Q1: is `groww.in` an acceptable publisher?** All five sources are
   `source_tier: brief`. The PRD lists a conflict between the brief's URLs and the
   AMC/SEBI/AMFI requirement, and this gates the PRD's own acceptance criteria. The
   data cannot settle it.
2. **Class D factsheet link.** Options: point at `sid_url` (AMC homepage, official but
   not a factsheet), add a real SEBI/AMFI factsheet source, or drop the link
   requirement. Cannot be decided from the five pages.
