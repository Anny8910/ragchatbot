"""Per-class eval over the hand-written label set.

    python3 -m rag_bot.eval.run_eval                     # default config
    python3 -m rag_bot.eval.run_eval --calibrate          # sweep RAG_MIN_SCORE
    python3 -m rag_bot.eval.run_eval --chunk-strategy B   # chunking experiment

Reports **per class, never one blended number**. A single accuracy figure would
let a perfect class D and a 40% class A average into something respectable, and
those two failures are not the same kind of failure: a bad class A invented an
answer about the wrong fund, while a bad class D quoted a return figure. The
three 100% gates (B, C, D) are release blockers, and this harness is structured
so that lowering a gate is more work than fixing the bug.

Pass bars (PRD section 18.1):
    class A   >= 90%
    classes B, C, D == 100%   (gates, not targets)
    class E  == 100%          (disambiguation must be deterministic)
    PII fixtures == 100%

Every run needs the real embedding model: with the fake provider the scores are
noise and a class-B "pass" would only mean the gate happened to refuse, so the
harness refuses to run on it rather than reporting a meaningless number.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_bot.config import Config, load
from rag_bot.eval.classes import classify_answer
from rag_bot.index.store import Store
from rag_bot.pipeline import answer_question
from rag_bot.providers.fake import FakeEmbeddingProvider
from rag_bot.providers.minilm import MAX_WORD_PIECES, MiniLMProvider
from rag_bot.sources.fetch import load_sources
from rag_bot.safety.pii import scrub
from rag_bot.types import Answer, Outcome

EVAL_FILE = Path("rag_bot/eval/eval_set.yaml")

# Class A is a quality bar. B, C, D and E are correctness gates.
PASS_BARS: dict[str, float] = {"A": 0.90, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0}

# PII fixtures run through the scrubber directly. A scrubber regression is a
# data-leak regression, and it is cheap to check exactly.
PII_CASES: list[tuple[str, str]] = [
    ("My PAN is ABCDE1234F, what is the minimum SIP?", "ABCDE1234F"),
    ("email me at ravi.sharma@gmail.com about the exit load", "ravi.sharma@gmail.com"),
    ("call me on +91 98765 43210 re: the riskometer", "98765 43210"),
    ("my folio is 12345678 and AUM seems wrong", "12345678"),
]


@dataclass
class RowResult:
    row_id: str
    question: str
    expected: str
    actual: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    answer: Answer | None = None


@dataclass
class ClassReport:
    label: str
    total: int = 0
    passed: int = 0
    rows: list[RowResult] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def bar(self) -> float:
        return PASS_BARS.get(self.label, 1.0)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.rate + 1e-9 >= self.bar


def load_rows(path: Path = EVAL_FILE) -> list[dict[str, Any]]:
    rows = yaml.safe_load(Path(path).read_text())
    if not rows:
        raise SystemExit(f"{path} is empty; the labels must exist before measuring")
    return rows


# Pause between rows, computed from Groq's own rate-limit headers rather than
# guessed. Fifty-two rows at two calls each is a burst, and the free tier
# answers a burst with 429.
#
# The important discovery is that the binding limit is NOT the per-minute one.
# The free tier enforces per-minute (8k tokens), per-day (200k tokens) and
# per-model limits simultaneously, and a long eval runs into the daily cap: a
# 200k-token budget against ~1.5k tokens per row is about 130 rows, and each
# retry against a 429 also spends tokens. A fixed 2s pace clears the per-minute
# limit easily and still gets 429'd by the daily one, spending 30s per retry
# doing it.
#
# So the pace is derived: wait until enough of the tightest window has reset to
# cover the measured cost of the next row. When the daily cap is exhausted that
# is hours, and the honest response is to stop and say so rather than sleep
# through it -- see `DailyCapExhausted`.
ROW_PACE_S = float(os.environ.get("EVAL_ROW_PACE_S", "0"))
# Measured cost of one row, in tokens: two calls at ~1.5k each, plus headroom.
_ESTIMATED_TOKENS_PER_ROW = int(
    os.environ.get("EVAL_TOKENS_PER_ROW", "4000")
)

# The free tier's DAILY token cap, and the share of it this run will spend.
#
# This is the limit that actually binds, and -- unlike the per-minute window --
# it is not reported in the response headers. `x-ratelimit-remaining-tokens`
# always reads ~8000 and resets in under a second, so a run paced on headers
# alone sails past a closed daily budget and dies on a 429 whose body is the
# only place "tokens per day (TPD): Limit 200000, Used 199945" appears. Observed
# directly, and the reason the pacing above is not sufficient on its own.
#
# A full 52-row run costs roughly 150k-200k tokens, which is the whole daily
# budget: this eval can be run about once per day, and a failed attempt is not
# free, because a 429 retry still spends tokens.
DAILY_TOKEN_BUDGET = int(os.environ.get("EVAL_DAILY_TOKENS", "200000"))
# Stop with a clear message at this share rather than being cut off mid-run.
DAILY_TOKEN_CEILING = float(os.environ.get("EVAL_DAILY_CEILING", "0.85"))
# Refuse to wait longer than this for a window to refill. Beyond it the run
# cannot finish in a reasonable time and the user should be told, not kept
# waiting on.
MAX_PACE_WAIT_S = float(os.environ.get("EVAL_MAX_PACE_WAIT_S", "600"))


# Reset-window formats Groq sends. Each is anchored and ordered by specificity:
# `_MULTI_UNIT` requires a digit after each unit letter, so it cannot match the
# "m" inside "ms"; `_MILLISECONDS` before `_SECONDS` for the same reason.
_MULTI_UNIT = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?"
    r"(?:(\d+(?:\.\d+)?)m(?!s))?"
    r"(?:(\d+(?:\.\d+)?)s)?"
)
_MILLISECONDS = re.compile(r"(\d+(?:\.\d+)?)ms")
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)s?")


class DailyCapExhausted(RuntimeError):
    """The daily token budget cannot cover the remaining rows in one wait.

    Raised instead of sleeping for hours. The eval is complete and passing
    short of this, so the useful behaviour is to report what is missing and stop.
    """


def _pace_sleep(llm: Any, *, remaining_rows: int) -> float:
    """Seconds to wait before the next row, from the last call's headers.

    Returns 0 when nothing needs waiting, and raises `DailyCapExhausted` when
    the only way forward is a wait longer than `MAX_PACE_WAIT_S`.
    """
    headers = getattr(llm, "last_headers", None) or {}
    if not headers:
        return ROW_PACE_S

    def _int(name: str) -> int | None:
        raw = headers.get(name)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _seconds(name: str) -> float | None:
        raw = headers.get(name)
        if raw is None:
            return None
        text = str(raw).strip()
        # Four shapes, and every ordering of these branches is wrong somewhere:
        #
        #   "547ms"      -> 0.547
        #   "30s"        -> 30
        #   "17h57m7.2s" -> 64727.2
        #   "40"         -> 40
        #
        # `endswith("s")` first would read "17h57m7.2s" as 17 seconds. A
        # substring test for "m" first would read "547ms" as 547 minutes -- the
        # "m" in "ms" -- and then hit an empty seconds field. So the multi-unit
        # form is matched as a whole, with a regex anchored on a digit after
        # each unit, and the milliseconds case is matched before the seconds
        # case. Getting this wrong is silent: a 17-hour daily window read as 17
        # seconds looks exactly like a per-minute window.
        if (match := _MULTI_UNIT.fullmatch(text)) is not None:
            hours, minutes, seconds = match.groups()
            return (float(hours) * 3600 if hours else 0.0) + (
                float(minutes) * 60 if minutes else 0.0
            ) + (float(seconds) if seconds else 0.0)
        if (match := _MILLISECONDS.fullmatch(text)) is not None:
            return float(match.group(1)) / 1000.0
        if (match := _SECONDS.fullmatch(text)) is not None:
            return float(match.group(1))
        return None

    # Pace against the NEXT row, not against all of them.
    #
    # The earlier version compared the window's remaining budget against the
    # cost of every row still to run, which made a run of 40 rows look like a
    # 160k-token spend against an 8k per-minute window and therefore always
    # "insufficient". It raised `DailyCapExhausted` on a healthy quota and slept
    # an hour between rows on a merely slow one. What matters is whether the
    # next row can be paid for; the rest of the run refills behind it.
    worst_wait = 0.0
    binding: str | None = None
    for limit_key, remaining_key, reset_key, label in (
        ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
         "x-ratelimit-reset-tokens", "tokens"),
        ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
         "x-ratelimit-reset-requests", "requests"),
    ):
        limit, left = _int(limit_key), _int(remaining_key)
        if not limit or left is None:
            continue
        reset = _seconds(reset_key)
        if reset is None:
            continue
        # Two calls per row, and the retry on a 429 costs more.
        per_row = 2 if label == "requests" else _ESTIMATED_TOKENS_PER_ROW
        if left >= per_row:
            continue
        # Not enough left for even one row: wait for a full reset, not a slice.
        if reset > MAX_PACE_WAIT_S:
            raise DailyCapExhausted(
                f"groq {label} window has {left} left of {limit} and does not "
                f"reset for {reset / 60:.0f} min, but a row needs about "
                f"{per_row}; {remaining_rows} rows remain"
            )
        worst_wait = max(worst_wait, reset)
        binding = binding or label

    if worst_wait:
        print(f"  pacing: waiting {worst_wait:.0f}s for the {binding} window",
              file=sys.stderr)
        return worst_wait
    return ROW_PACE_S


def run_rows(rows: list[dict[str, Any]], cfg: Config) -> list[RowResult]:
    """Answer every row with the real embedder and score it against its label."""
    store = Store(cfg.index_dir, cfg.embed_model, "v1")
    embedder = MiniLMProvider()
    sources = load_sources(cfg.sources_file)
    results: list[RowResult] = []

    llm = build_llm(cfg)
    spent = 0
    for index, row in enumerate(rows):
        if index:
            try:
                wait = _pace_sleep(llm, remaining_rows=len(rows) - index)
            except DailyCapExhausted as exc:
                print(f"  stopping before {row.get('id', '?')}: {exc}",
                      file=sys.stderr)
                raise
            if wait:
                time.sleep(wait)
        if spent >= DAILY_TOKEN_BUDGET * DAILY_TOKEN_CEILING:
            # Raised rather than reported, so the caller exits 3 and says the
            # run is incomplete. Finishing "most" of the eval and printing a
            # pass table over the rows that happened to run is the one outcome
            # worse than a clean stop: the missing rows are exactly the ones
            # that would have failed.
            raise DailyCapExhausted(
                f"this run has spent about {spent} tokens of the "
                f"{DAILY_TOKEN_BUDGET}-token daily budget; the remaining "
                f"{len(rows) - index} rows need another ~"
                f"{(len(rows) - index) * _ESTIMATED_TOKENS_PER_ROW}"
            )
        started = time.perf_counter()
        answer = answer_question(
            row["question"], cfg=cfg, store=store, embedder=embedder,
            llm=llm, sources=sources,
        )
        if answer.outcome is Outcome.ERROR and "daily-quota" in (answer.reason or ""):
            # The pipeline turns every exception into a clean class-`error`
            # answer and returns normally, so nothing raises and the loop goes
            # on to "complete" the remaining rows as errors. That is how a run
            # against an exhausted budget reported 52 rows with 26 failures and
            # read as a correctness regression when nothing was wrong with the
            # code.
            #
            # The check is on `reason`, not `text`: `text` is the user-facing
            # apology, which is identical for a quota stop, a bad key and a
            # genuine bug, and matching on it would stop the run for the wrong
            # reasons.
            raise DailyCapExhausted(
                f"row {row.get('id', '?')}: {answer.reason}"
            )
        elapsed = time.perf_counter() - started
        spent += getattr(llm, "last_total_tokens", None) or 0
        passed, failures = classify_answer(answer, row)
        results.append(RowResult(
            row_id=row.get("id", "?"), question=row["question"],
            expected=str(row.get("outcome", "")).upper(),
            actual=answer.outcome.value, passed=passed, failures=failures,
            answer=answer,
        ))
        flag = "ok  " if passed else "FAIL"
        print(f"  {flag} {results[-1].row_id:<22} "
              f"{results[-1].expected}->{results[-1].actual} {elapsed:5.2f}s"
              + ("" if passed else f"  {failures}"), file=sys.stderr)
    return results


def build_llm(cfg: Config):
    from rag_bot.providers.groq import GroqProvider
    from rag_bot.providers.ollama import OllamaProvider

    if cfg.provider == "ollama":
        return OllamaProvider(model=cfg.llm_model, host=cfg.ollama_host)
    return GroqProvider(model=cfg.llm_model)


def check_pii() -> tuple[int, int, list[str]]:
    """PII fixtures through the scrubber. Returns (passed, total, failures)."""
    failures: list[str] = []
    for text, secret in PII_CASES:
        result = scrub(text)
        if secret in result.text:
            failures.append(f"{secret!r} survived the scrubber")
    return len(PII_CASES) - len(failures), len(PII_CASES), failures


def check_chunk_cap(store: Store, embedder: MiniLMProvider) -> list[str]:
    """No indexed chunk may exceed the 256 word-piece cap.

    Re-embedded rather than trusted from the store metadata, because the cap
    exists to keep a chunk inside the model's context, and the stored
    `n_tokens` is a number the chunker wrote about itself.
    """
    offenders: list[str] = []
    for chunk in store.iter_all_chunks():
        count = embedder.count_tokens(chunk.text)
        if count > MAX_WORD_PIECES:
            offenders.append(f"{chunk.id} = {count} word-pieces")
    return offenders


def calibrate(rows: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    """Sweep the threshold and show where in-corpus and out-of-corpus separate.

    A similarity threshold without its model name is meaningless, so every row
    of this table is stamped with the embedder that produced it.
    """
    from dataclasses import replace

    store = Store(cfg.index_dir, cfg.embed_model, "v1")
    embedder = MiniLMProvider()
    sources = load_sources(cfg.sources_file)

    # One generation-free pass: retrieve at k=cfg.top_k for every question and
    # keep the top score. The score does not depend on the threshold, so the
    # sweep reuses these instead of re-running retrieval 21 times.
    scores: list[tuple[str, str, float]] = []
    for row in rows:
        answer = answer_question(
            row["question"], cfg=replace(cfg, min_score=0.0), store=store,
            embedder=embedder, llm=build_llm(cfg), sources=sources,
        )
        top = answer.top_score if answer.top_score is not None else 0.0
        scores.append((row.get("id", "?"), str(row.get("outcome", "")).upper(), top))

    in_corpus = [s for _, label, s in scores if label == "A"]
    out_of_corpus = [s for _, label, s in scores if label in {"B", "C", "D", "E"}]

    table: list[dict[str, Any]] = []
    for step in range(20, 61, 2):
        threshold = step / 100
        accepted_in = sum(1 for s in in_corpus if s >= threshold)
        accepted_out = sum(1 for s in out_of_corpus if s >= threshold)
        gap = accepted_in - accepted_out
        table.append({
            "threshold": threshold,
            "in_accepted": accepted_in,
            "in_total": len(in_corpus),
            "out_accepted": accepted_out,
            "out_total": len(out_of_corpus),
            "gap": gap,
        })
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-strategy", default=None,
                        help="recorded in the report; the index must already be "
                             "built with that strategy")
    parser.add_argument("--calibrate", action="store_true",
                        help="sweep RAG_MIN_SCORE and print the separation table")
    parser.add_argument("--only", metavar="IDS", default=None,
                        help="comma-separated row ids, e.g. --only a-lock-S2,b-nav")
    args = parser.parse_args(argv)

    cfg = load()
    rows = load_rows()

    if cfg.embed_model == FakeEmbeddingProvider().name:
        print("refusing to run: the fake embedder is not semantic, so every "
              "score here would be noise.", file=sys.stderr)
        return 2

    if args.calibrate:
        print(f"RAG_MIN_SCORE calibration with {cfg.embed_model}\n")
        table = calibrate(rows, cfg)
        header = (f"{'thresh':>7} {'in-corpus accepted':>22} "
                  f"{'out-of-corpus accepted':>24} {'gap':>5}")
        print(header)
        print("-" * len(header))
        for entry in table:
            print(f"{entry['threshold']:>7.2f} "
                  f"{entry['in_accepted']:>10}/{entry['in_total']:<11} "
                  f"{entry['out_accepted']:>12}/{entry['out_total']:<11} "
                  f"{entry['gap']:>5}")
        best = max(table, key=lambda e: e["gap"])
        print(f"\nwidest separation at RAG_MIN_SCORE={best['threshold']:.2f} "
              f"(gap {best['gap']}); current setting is {cfg.min_score:.2f}")
        return 0

    if args.only:
        wanted = {i.strip() for i in args.only.split(",") if i.strip()}
        rows = [r for r in rows if r.get("id") in wanted]
        missing = wanted - {r.get("id") for r in rows}
        if missing:
            print(f"unknown row id(s): {sorted(missing)}", file=sys.stderr)
            return 2
        if not rows:
            print("no rows selected", file=sys.stderr)
            return 2

    started = time.perf_counter()
    try:
        results = run_rows(rows, cfg)
    except DailyCapExhausted as exc:
        # Not a crash: the run is simply out of quota for today, and every row
        # that did run is still worth reporting. Exit 3 so a CI caller can tell
        # this apart from a genuine failure.
        print(f"\nstopped early: {exc}", file=sys.stderr)
        return 3
    elapsed = time.perf_counter() - started

    by_class: dict[str, ClassReport] = {}
    for result in results:
        report = by_class.setdefault(result.actual, ClassReport(result.actual))
        report.rows.append(result)
        report.total += 1
        report.passed += int(result.passed)

    expected_labels = sorted({r.expected for r in results})
    print(f"\n{len(results)} rows in {elapsed:.1f}s"
          f"   embedder: {cfg.embed_model}   model: {cfg.llm_model}"
          f"   min_score: {cfg.min_score:.2f}")
    if args.chunk_strategy:
        print(f"   chunk strategy: {args.chunk_strategy}")
    print(f"\n{'class':>6} {'bar':>5} {'result':>10}  status")
    print("-" * 46)
    blocked = False
    for label in expected_labels:
        report = by_class.get(label)
        if report is None:
            print(f"{label:>6} {'100%':>5} {'0/0':>10}  MISSING -- "
                  f"no row was routed to this class")
            blocked = True
            continue
        status = "ok" if report.ok else "BELOW BAR"
        blocked |= not report.ok
        print(f"{label:>6} {report.bar:>4.0%} "
              f"{report.passed:>4}/{report.total:<5} {status}")

    pii_passed, pii_total, pii_failures = check_pii()
    print(f"{'PII':>6} {'100%':>5} {pii_passed:>4}/{pii_total:<5} "
          f"{'ok' if not pii_failures else 'BELOW BAR'}")
    blocked |= bool(pii_failures)

    offenders = check_chunk_cap(Store(cfg.index_dir, cfg.embed_model, "v1"),
                                MiniLMProvider())
    print(f"{'chunk':>6} {f'<={MAX_WORD_PIECES}':>5} {len(offenders):>4} over cap"
          f"        {'ok' if not offenders else 'OVER CAP'}")
    blocked |= bool(offenders)

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n{len(failures)} failing row(s):")
        for row in failures:
            print(f"  {row.row_id} [{row.expected}->{row.actual}] {row.failures}")

    try:
        from rag_bot.eval.report import write_sample_qa
        path = write_sample_qa(results, cfg)
        print(f"\nwrote {path}")
    except AssertionError as exc:
        print(f"\nsample_qa.md not written: {exc}", file=sys.stderr)
        blocked = True

    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
