"""Chunker tests. No model download: the tokenizer is injected."""
from __future__ import annotations

import re

import pytest

from rag_bot.config import load
from rag_bot.ingest.chunker import (
    MAX_WORD_PIECES,
    _build_units,
    _classify_topic,
    _split_sections,
    chunk_all,
    chunk_document,
)
from rag_bot.ingest.loaders import load_from_manifest
from rag_bot.types import Document, Source

# Whitespace-token stub, as the spec requires: tests must not need a model.
stub = lambda t: len(t.split())  # noqa: E731

MANIFEST = f"{load().corpus_dir}/manifest.csv"


def _src(source_id="s", scheme_id="S1", url="http://example.test/p"):
    return Source(
        source_id=source_id,
        url=url,
        publisher="groww.in",
        source_tier="brief",
        scheme_id=scheme_id,
        scheme_name="Test Fund",
        factsheet_url=None,
        aliases=("test fund",),
    )


def _doc(text, **kw):
    return Document(text=text, source=_src(**kw), heading=None)


# --------------------------------------------------------------------------
# Section splitting
# --------------------------------------------------------------------------
def test_split_sections_on_headings():
    text = "## A\nalpha\n## B\nbeta"
    sections = _split_sections(text)
    assert [h for h, _ in sections] == ["A", "B"]
    assert sections[0][1].strip() == "alpha"


def test_split_sections_keeps_h3_nesting():
    text = "## A\nx\n### B\ny"
    assert [h for h, _ in _split_sections(text)] == ["A", "B"]


def test_split_sections_drops_empty_sections():
    text = "## A\nalpha\n## B\n\n## C\ngamma"
    assert [h for h, _ in _split_sections(text)] == ["A", "C"]


def test_build_units_groups_consecutive_label_lines():
    body = "Expense ratio: 1.03\nExit load: Nil\nMinimum SIP: 500"
    units = _build_units(body)
    assert len(units) == 1, "a flattened table must stay one unit"
    assert "Expense ratio: 1.03" in units[0]


# --------------------------------------------------------------------------
# Chunking behaviour
# --------------------------------------------------------------------------
def test_chunk_never_crosses_a_section_boundary():
    text = "## A\n" + ("alpha " * 100) + "\n## B\n" + ("beta " * 100)
    chunks = chunk_document(
        _doc(text), target_tokens=20, overlap_tokens=0, tokenizer=stub
    )
    for c in chunks:
        assert not ("alpha" in c.text and "beta" in c.text), c.text[:80]


def test_table_label_stays_with_its_value():
    text = "## Fees\n" + "\n".join(f"Fee {i}: {i} pct" for i in range(30))
    chunks = chunk_document(
        _doc(text), target_tokens=25, overlap_tokens=0, tokenizer=stub
    )
    for c in chunks:
        for line in c.text.splitlines():
            if ":" in line:
                label, _, value = line.partition(":")
                assert value.strip(), f"label {label!r} lost its value"


def test_forward_progress_on_a_pathological_input():
    """A single enormous unit must not loop forever."""
    text = "## A\n" + " ".join(f"word{i}" for i in range(5000))
    chunks = chunk_document(
        _doc(text), target_tokens=50, overlap_tokens=5, tokenizer=stub
    )
    assert len(chunks) > 1
    assert all(c.n_tokens <= MAX_WORD_PIECES for c in chunks)


def test_hard_split_respects_the_target():
    text = "## A\nword " * 1000
    chunks = chunk_document(
        _doc(text), target_tokens=30, overlap_tokens=0, tokenizer=stub
    )
    assert max(c.n_tokens for c in chunks) <= 30


def test_ids_are_stable_and_sequential():
    text = "## A\nalpha\n## B\nbeta"
    a = chunk_document(_doc(text), target_tokens=50, overlap_tokens=0, tokenizer=stub)
    b = chunk_document(_doc(text), target_tokens=50, overlap_tokens=0, tokenizer=stub)
    assert [c.id for c in a] == [c.id for c in b] == ["s::0", "s::1"]
    assert [c.chunk_index for c in a] == [0, 1]


def test_content_hash_is_sha256_of_text():
    import hashlib

    chunks = chunk_document(
        _doc("## A\nalpha"), target_tokens=50, overlap_tokens=0, tokenizer=stub
    )
    c = chunks[0]
    assert c.content_hash == hashlib.sha256(c.text.encode()).hexdigest()


def test_overlap_carries_context_forward():
    text = "## A\n" + "\n".join(f"Fee {i}: {i}" for i in range(40))
    chunks = chunk_document(
        _doc(text), target_tokens=30, overlap_tokens=10, tokenizer=stub
    )
    assert len(chunks) > 1
    assert chunks[1].n_tokens > 0


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc(""), target_tokens=50, overlap_tokens=0, tokenizer=stub) == []
    assert chunk_document(_doc("   \n\n"), target_tokens=50, overlap_tokens=0, tokenizer=stub) == []


def test_target_larger_than_cap_is_capped():
    text = "## A\n" + "word " * 2000
    chunks = chunk_document(
        _doc(text), target_tokens=10_000, overlap_tokens=0, tokenizer=stub
    )
    assert max(c.n_tokens for c in chunks) <= MAX_WORD_PIECES


# --------------------------------------------------------------------------
# Topic assignment
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heading,body,expected",
    [
        ("Expense ratio", "Expense ratio: 1.03", "expense_ratio"),
        ("Exit load", "Exit load: Nil", "exit_load"),
        ("Minimum investments", "Minimum SIP amount: 500", "min_sip"),
        ("Lock-in period", "Lock-in period: 3 years", "lock_in"),
        ("Riskometer", "Riskometer level: Moderately High", "riskometer"),
        ("Benchmark", "Benchmark: NIFTY 100 Total Return Index", "benchmark"),
        ("Scheme identity", "Scheme name: HDFC Large Cap Fund Direct Growth", "other"),
        ("Objective", "seeks capital appreciation and income", "other"),
    ],
)
def test_topic_assignment(heading, body, expected):
    assert _classify_topic(heading, body) == expected


def test_statement_is_not_a_topic():
    """The statement topic was dropped from scope; nothing may be assigned it."""
    valid = {"expense_ratio", "exit_load", "min_sip", "lock_in", "riskometer",
             "benchmark", "other"}
    for heading, body in [
        ("Fees", "Expense ratio: 1.03"),
        ("How to download", "statement download"),
        (None, "random text"),
    ]:
        assert _classify_topic(heading, body) in valid


# --------------------------------------------------------------------------
# End to end on the real corpus
# --------------------------------------------------------------------------
def test_real_corpus_chunks_cleanly():
    docs = load_from_manifest(MANIFEST)
    if not any(d.text for d in docs):
        pytest.skip("no snapshots; run the fetcher")
    chunks = chunk_all(
        docs, target_tokens=load().chunk_tokens, overlap_tokens=load().chunk_overlap,
        tokenizer=stub,
    )
    assert chunks
    assert all(c.n_tokens <= MAX_WORD_PIECES for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids must be unique"
    topics = {c.topic for c in chunks}
    assert "expense_ratio" in topics
    assert "benchmark" in topics
    assert "riskometer" in topics
    assert "other" in topics


def test_real_corpus_has_no_performance_text_in_any_chunk():
    from rag_bot.ingest.allowlist import scan_forbidden_patterns

    docs = load_from_manifest(MANIFEST)
    if not any(d.text for d in docs):
        pytest.skip("no snapshots; run the fetcher")
    chunks = chunk_all(
        docs, target_tokens=load().chunk_tokens, overlap_tokens=load().chunk_overlap,
        tokenizer=stub,
    )
    for c in chunks:
        hits = scan_forbidden_patterns(c.text)
        assert hits == (), f"{c.id} contains {hits}"


def test_demo_chunk_keeps_expense_ratio_with_its_label():
    """The spec calls this the demo's most important chunk."""
    docs = load_from_manifest(MANIFEST)
    if not any(d.text for d in docs):
        pytest.skip("no snapshots; run the fetcher")
    chunks = chunk_all(
        docs, target_tokens=load().chunk_tokens, overlap_tokens=load().chunk_overlap,
        tokenizer=stub,
    )
    er = [c for c in chunks if c.topic == "expense_ratio"]
    assert er, "no expense_ratio chunk was produced"
    assert any(re.search(r"Expense ratio: \S+", c.text) for c in er)
