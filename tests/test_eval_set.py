"""Eval-set integrity. The labels must agree with the corpus they evaluate.

`eval_set.yaml` was written by hand BEFORE any retrieval score was seen, which is
the point of architecture §9.3. These tests do not touch retrieval at all; they
only check the label file is internally consistent and factually correct against
the committed snapshots.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from rag_bot.config import load
from rag_bot.ingest.loaders import load_from_manifest

EVAL_PATH = Path(load().source_config_path) if hasattr(load(), "source_config_path") else None
EVAL_FILE = Path(__file__).resolve().parent.parent / "rag_bot" / "eval" / "eval_set.yaml"
MANIFEST = f"{load().corpus_dir}/manifest.csv"

TOPICS = {
    "expense_ratio", "exit_load", "min_sip",
    "lock_in", "riskometer", "benchmark",
}
SCHEMES = {"S1", "S2", "S3", "S4", "S5"}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    data = yaml.safe_load(EVAL_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data
    return data


def test_has_thirty_class_a_rows(rows):
    a = [r for r in rows if r["outcome"] == "A"]
    assert len(a) >= 30, f"only {len(a)} class-A rows; spec requires 6 topics x 5 schemes = 30"


def test_class_a_covers_every_topic_and_scheme(rows):
    a = [r for r in rows if r["outcome"] == "A" and r.get("topic") and r.get("scheme_id")]
    pairs = Counter((r["topic"], r["scheme_id"]) for r in a)
    for topic in TOPICS:
        for scheme in SCHEMES:
            assert pairs.get((topic, scheme), 0) >= 1, (
                f"no class-A row for {topic} x {scheme}"
            )
    assert sum(pairs.values()) >= 30


def test_class_a_topics_are_all_in_scope(rows):
    for r in rows:
        if r["outcome"] == "A" and r.get("topic"):
            assert r["topic"] in TOPICS, f"{r['id']}: unknown topic {r['topic']}"


def test_statement_is_not_an_in_scope_topic(rows):
    """`statement` was dropped from scope; nothing may be labelled with it."""
    for r in rows:
        assert r.get("topic") != "statement", r["id"]


def test_class_d_has_at_least_three_and_forbids_all_figures(rows):
    d = [r for r in rows if r["outcome"] == "D"]
    assert len(d) >= 3
    for r in d:
        pats = [p.lower() for p in r.get("forbidden_patterns", [])]
        for required in ("%", "cagr", "nav"):
            assert required in pats, f"{r['id']}: class D must forbid {required!r}"


def test_class_b_forbids_citations(rows):
    b = [r for r in rows if r["outcome"] == "B"]
    assert len(b) >= 3
    for r in b:
        pats = [p.lower() for p in r.get("forbidden_patterns", [])]
        assert any("http" in p or "groww" in p for p in pats), (
            f"{r['id']}: class B must forbid any citation"
        )


def test_class_c_requires_a_link(rows):
    c = [r for r in rows if r["outcome"] == "C"]
    assert len(c) >= 3
    for r in c:
        assert r.get("require_link") is True, f"{r['id']}: class C must link the AMC"


def test_multi_scheme_rows_exist(rows):
    multi = [r for r in rows if r.get("expect_source_ids")]
    assert len(multi) >= 2, "the spec requires 2 multi-scheme rows"
    for r in multi:
        assert len(r["expect_source_ids"]) >= 2
        assert isinstance(r["expect_source_ids"], list)


def test_scheme_ambiguous_rows_exist(rows):
    amb = [
        r for r in rows
        if r.get("expect_behavior") == "ask_which_scheme"
        and r.get("scheme_id") is None
    ]
    assert len(amb) >= 2, "the spec requires 2 scheme-ambiguous rows"


def test_questions_are_varied_not_templated(rows):
    """Thirty rows of 'What is the {topic} of {scheme}?' would make the §9.3
    chunking experiment meaningless."""
    questions = [r["question"] for r in rows]
    assert len(set(questions)) == len(questions), "duplicate questions in the eval set"

    # no single question stem should dominate
    stems = Counter(q.split()[0].lower() for q in questions)
    assert max(stems.values()) <= 6, f"over-templated opening words: {stems.most_common(3)}"

    # A-class questions must not all be the same shape
    a_q = [r["question"] for r in rows if r["outcome"] == "A" and r.get("topic") in TOPICS]
    normalised = {" ".join(q.lower().split()[-2:]) for q in a_q}
    assert len(normalised) > len(a_q) * 0.4, "class-A phrasing is too uniform"


def test_ids_are_unique(rows):
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)


def test_every_row_has_required_fields(rows):
    for r in rows:
        assert r.get("id"), r
        assert r.get("question"), r["id"]
        assert r.get("outcome") in {"A", "B", "C", "D", "E"}, r["id"]


# --------------------------------------------------------------------------
# The labels must match the real corpus
# --------------------------------------------------------------------------
def test_class_a_expected_keywords_appear_in_their_source(rows):
    """A hand-written label that disagrees with the corpus is worse than no
    label, because it silently poisons the hit-rate."""
    docs = load_from_manifest(MANIFEST)
    if not any(d.text for d in docs):
        pytest.skip("no snapshots; run the fetcher")
    by_id = {d.source.source_id: d.text.lower() for d in docs}

    checked = 0
    for r in rows:
        if r["outcome"] != "A":
            continue
        source_id = r.get("expect_source_id")
        if not source_id:
            continue
        assert source_id in by_id, f"{r['id']}: unknown source_id {source_id}"
        text = by_id[source_id]
        for kw in r.get("expect_keywords", []):
            assert kw.lower() in text, (
                f"{r['id']}: expected keyword {kw!r} is not in {source_id}'s text"
            )
            checked += 1
    assert checked >= 50, f"only checked {checked} keywords"


def test_class_a_scheme_ids_exist(rows):
    for r in rows:
        sid = r.get("scheme_id")
        if sid:
            assert sid in SCHEMES, f"{r['id']}: bad scheme_id {sid}"
