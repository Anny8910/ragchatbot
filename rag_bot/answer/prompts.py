"""Prompt construction (architecture 14.1).

The system prompt is specified verbatim in the phase doc and is reproduced here
character for character. It is not paraphrased or "improved": rule 4 is the
belt to the corpus's braces, and rule 6 is the escape hatch that lets the model
decline rather than guess. Both only work if the model is told them literally.
"""
from __future__ import annotations

from rag_bot.types import ScoredChunk

SYSTEM_PROMPT = """You are a facts-only assistant for HDFC mutual fund scheme pages.

Rules:
1. Answer only from the numbered context. Never use outside knowledge.
2. Maximum 3 sentences. No preamble, no closing pleasantries.
3. Do not give investment advice. If asked whether to buy, hold, or sell,
   say that you only provide facts.
4. Do not state, compare, or estimate returns, NAV, or performance.
   None are present in the context.
5. End with exactly: "Source: <block number>"
6. If the context does not contain the answer, reply with exactly: NOT_IN_INDEX
"""

# The sentinel the model emits when the context does not answer the question
# (rule 6). `validate()` converts it to a clean class B with no citation, which is
# the one place the model is trusted to decide something structural.
NOT_IN_INDEX = "NOT_IN_INDEX"

# What the model writes at the end of a real answer (rule 5). Note the brackets:
# the phase doc's own example in architecture 14.2 is "Source: [n]", while the
# prompt says "Source: <block number>". `parse_citation()` accepts both, because
# an 8B model reliably produces one of them and losing the citation over a bracket
# would be a self-inflicted wound.
SOURCE_PREFIX = "Source:"


def build_context(chunks: list[ScoredChunk]) -> str:
    """Numbered blocks, each carrying its citation material. No trailing commentary.

    Numbering is 1-based and matches the block number the model is asked to cite,
    so the number in the answer is a pointer into this exact string and nothing
    else has to be trusted.
    """
    if not chunks:
        return ""

    blocks: list[str] = []
    for i, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        heading = chunk.heading or "General"
        blocks.append(
            f"[{i}] {chunk.scheme_name} - source: {chunk.publisher} "
            f'- section: "{heading}"\n{chunk.text}'
        )
    return "\n\n".join(blocks)


def build_user_prompt(context: str, question: str) -> str:
    """Context, blank line, 'Question: <question>'."""
    return f"{context}\n\nQuestion: {question}"

# --- class E: which scheme? -------------------------------------------------
#
# Fixed copy, not model-generated, for the same reason the C and D refusals are
# fixed copy: a model asked "which fund do you mean?" will sometimes answer its
# own question by picking one, and a figure attributed to the wrong one of five
# similarly named funds is the PRD's own named failure mode.
#
# No figure, no URL, no advisory verb. The scheme list is passed in from the
# source registry rather than hardcoded, so the five names cannot drift out of
# sync with what is actually indexed.

_DISAMBIGUATION_PREFIX = (
    "Which scheme do you mean? I answer about one scheme at a time, and each of "
    "these carries its own figures, so I don't want to attribute one fund's "
    "number to another:"
)


def build_disambiguation(scheme_names: list[str]) -> str:
    """The class E question body. Empty when the registry is empty.

    Topic-agnostic on purpose: the same question arrives for expense ratio,
    lock-in and benchmark as for exit load, so the copy must not assert anything
    specific to one of them. The names come from the source registry rather than
    being hardcoded, so the list cannot drift out of sync with what is indexed.
    """
    names = [n for n in scheme_names if n]
    if not names:
        return ("Which scheme do you mean? I answer about one HDFC scheme at a "
                "time, so tell me which one and I'll look up the fact.")
    listed = ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]
    return f"{_DISAMBIGUATION_PREFIX} {listed}?"
