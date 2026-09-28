"""Exact scheme resolution (architecture 14.4).

The PRD's own problem statement is that LLMs "mix up figures across similarly
named schemes", so the scheme named in a question is resolved *before* retrieval
and retrieval is scoped to it. This module is where that resolution happens, and
it is deliberately the dumbest possible implementation.

EXACT MATCHING ONLY. Never fuzzy. "equity" appears in "HDFC Equity Fund" and in
the category labels of S1 ("Equity Large Cap") and S4 ("Equity Small Cap"), so a
fuzzy or substring match here silently attributes one fund's figure to another
fund. A wrong scheme_id is worse than no scheme_id, because a wrong one scopes
retrieval confidently to the wrong fund. Bare category words ("large cap",
"small cap", "flexi cap", "balanced advantage") are excluded from the alias
tables for the same reason -- see rag_bot/sources/sources.yaml.
"""
from __future__ import annotations

import re

from rag_bot.types import Source


def _normalise(text: str) -> str:
    """Lowercase, collapse all whitespace runs to single spaces, trim."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# "Flexi cap scheme", "large cap plan" and "HDFC Equity Fund" all name the same
# kind of thing, and which one a person happens to use says nothing about which
# fund they mean. Canonicalised so an alias written one way still matches a
# question written another.
#
# This is a head-noun substitution, not a fuzzy match: the *distinguishing* words
# ("hdfc", "flexi", "cap", "elss") are untouched and still have to match as whole
# tokens, so the module's "exact matching only" rule holds. What it buys is
# recall on a question the corpus answers perfectly well -- "redeeming the flexi
# cap scheme" used to resolve to no scheme at all, and an unresolvable scheme
# turns a class-A question into a class-E "which fund do you mean?".
_HEAD_NOUNS = {"scheme", "schemes", "plan", "mf", "funds"}


def _canonical_head_nouns(text: str) -> str:
    """Rewrite interchangeable mutual-fund head nouns to "fund".

    Only whole tokens, so "planning" is untouched and "scheme" inside
    "schemesector" is not rewritten.
    """
    return re.sub(
        r"(?<![a-z0-9])(scheme|schemes|plan|funds|mf)(?![a-z0-9])",
        "fund", text,
    )


def _mentions_alias(normalised_question: str, alias: str) -> bool:
    """Whole-token containment of an alias inside the question.

    `(?<![a-z0-9])` / `(?![a-z0-9])` rather than `\\b`, because the aliases
    contain punctuation ("hdfc elss tax saver fund" is fine, but "a/c" style
    aliases are not) and `\\b` would misbehave around non-word characters. The
    effect that matters is that "s1" does not match inside "s1234" and "large cap
    fund" does not match inside "large cap fundsector".

    Tried against both the question as written and its head-noun canonical form,
    so the alias table stays readable ("hdfc flexi cap fund") without going
    blind to how people actually write the question.
    """
    needle = _normalise(alias)
    if not needle:
        return False
    canonical = _canonical_head_nouns(normalised_question)
    for haystack in (normalised_question, canonical):
        if re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack
        ) is not None:
            return True
    return False


def resolve_scheme(question: str, sources: list[Source]) -> tuple[str | None, bool]:
    """Return (scheme_id or None, ambiguous).

    If exactly one scheme's aliases match, return it. If two or more *distinct*
    schemes match, return (None, True) -- architecture 14.4 item 6: the assistant
    answers the primary scheme and says it handles one scheme per question,
    rather than guessing across schemes. If none match, return (None, False).

    Several aliases of the SAME scheme matching is not ambiguity: "hdfc large cap
    fund" and "large cap fund" both appear in "the expense ratio of the HDFC
    Large Cap Fund", and that is one scheme, not two.
    """
    if not question or not sources:
        return (None, False)

    normalised = _normalise(question)
    matched: list[str] = []
    for source in sources:
        if source.scheme_id is None:      # cross-scheme page: cannot scope to one fund
            continue
        if any(_mentions_alias(normalised, alias) for alias in source.aliases):
            if source.scheme_id not in matched:
                matched.append(source.scheme_id)

    if not matched:
        return (None, False)
    if len(matched) > 1:
        return (None, True)
    return (matched[0], False)
