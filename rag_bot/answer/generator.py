"""Answer generation (architecture 14.1, phase P7).

Thin by design. The prompt lives in `prompts.py`, the checks live in
`validate.py`, and the footer lives in `assemble.py`; this module's only jobs are
to build the two prompt strings from the retrieved chunks and to call the
provider.

The one piece of policy here is the single retry. Architecture 14.2 asks for one
retry with a stricter instruction when a truncated answer lost its citation, which
is the one generation failure worth spending a second call on: a lost citation
means the answer's provenance is guesswork, and the pipeline's fallback for that
is to mark the answer `unverified`, which is a worse user experience than one
extra second of latency on a demo.
"""
from __future__ import annotations

import logging

from rag_bot.answer.prompts import (
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)
from rag_bot.providers.base import LLMProvider
from rag_bot.types import ScoredChunk

logger = logging.getLogger(__name__)

# Appended on the retry only. The first attempt uses SYSTEM_PROMPT verbatim.
_STRICTER_SUFFIX = (
    "\n\nYour previous reply was too long and lost its citation. "
    "Reply with at most 2 sentences and end with exactly: Source: <block number>"
)


def generate_answer(
    llm: LLMProvider,
    question: str,
    chunks: list[ScoredChunk],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """One generation call. Returns the raw model text, unvalidated.

    No `try` here. The pipeline owns error handling, because only the pipeline
    knows which stages have already run and what the latency record should say
    when one of them fails.
    """
    system, user = build_prompts(question, chunks)
    raw = llm.generate(system, user, temperature=temperature,
                       max_tokens=max_tokens)
    return raw if isinstance(raw, str) else ""


def build_prompts(question: str, chunks: list[ScoredChunk],
                  *, strict: bool = False) -> tuple[str, str]:
    """(system, user) for one generation call.

    Exposed separately from `generate_answer` so the retry can rebuild the prompts
    with the stricter system prompt while reusing the identical context block.
    """
    system = SYSTEM_PROMPT + _STRICTER_SUFFIX if strict else SYSTEM_PROMPT
    return system, build_user_prompt(build_context(chunks), question)


def retry_with_stricter_instruction(
    llm: LLMProvider,
    question: str,
    chunks: list[ScoredChunk],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """One retry after a truncated answer lost its citation (architecture 14.2).

    Deliberately NOT a loop. A second failure returns whatever the retry produced
    and lets `validate` mark the answer `unverified`; an unbounded retry loop
    against a local 8B model is a way to turn a citation problem into a hung demo.
    """
    system, user = build_prompts(question, chunks, strict=True)
    try:
        raw = llm.generate(system, user, temperature=temperature,
                           max_tokens=max_tokens)
    except Exception:
        logger.warning("strict retry failed; keeping the unverified first answer")
        return ""
    return raw if isinstance(raw, str) else ""
