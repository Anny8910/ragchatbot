"""Answer assembly (architecture 14.3).

The footer is built HERE, in code, never by the model. That is the whole point of
this module: a model that writes its own "Last updated from sources: 27 Sep 2026"
can omit the date, invent a plausible one, or contradict the snapshot it was
given. The date here is read from the `fetched_at` of the cited chunk, so it is
either real or "unknown" -- there is no third option in which it is a guess.

The date is rendered from a fixed month table rather than ``strftime("%b")``
because the footer is user-visible and ``%b`` follows the machine's locale.
"""
from __future__ import annotations

from datetime import datetime

from rag_bot.types import Answer, Outcome, ScoredChunk, TriageLayer, Validation

# Fixed so the rendered footer cannot change with the machine's locale.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_FOOTER_PREFIX = "Last updated from sources: "
_UNKNOWN = "unknown"


def parse_fetched_at(fetched_at_iso: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp. None on anything unparseable.

    "Z" is handled explicitly because ``fromisoformat`` only accepted it from
    3.11, and an unparseable timestamp must degrade to "unknown" rather than
    raise inside a footer.
    """
    if not fetched_at_iso or not isinstance(fetched_at_iso, str):
        return None
    text = fetched_at_iso.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_date(fetched_at_iso: str | None) -> str:
    """'27 Sep 2026', or 'unknown'. Never fabricates a date."""
    parsed = parse_fetched_at(fetched_at_iso)
    if parsed is None:
        return _UNKNOWN
    return f"{parsed.day:02d} {_MONTHS[parsed.month - 1]} {parsed.year:04d}"


def format_footer(fetched_at_iso: str | None) -> str:
    """'Last updated from sources: 27 Sep 2026'.

    A missing or unparseable fetched_at renders 'Last updated from sources:
    unknown' -- never a fabricated date, and never an omitted footer. An omitted
    date reads as "we know and are not telling you"; "unknown" reads as what it
    is, which is the only honest thing to say about a snapshot with no timestamp.
    """
    return _FOOTER_PREFIX + format_date(fetched_at_iso)


def cited_chunk(
    chunks: list[ScoredChunk], validation: Validation
) -> ScoredChunk | None:
    """The chunk the answer is actually about.

    Located by matching ``validation.cited_url`` -- which `validate` derived from
    chunk metadata, never from model text -- back to its chunk. Falls back to
    rank 1, matching `check_single_url`'s documented behaviour, and to None when
    retrieval found nothing.
    """
    if not chunks:
        return None
    if validation.cited_url:
        for scored in chunks:
            if scored.chunk.url == validation.cited_url:
                return scored
    return chunks[0]


def build_footer(scored: ScoredChunk | None) -> str:
    """The two footer lines for a class A answer.

    'Source: <publisher> - <scheme_name>' then 'Last updated from sources: <date>'.
    """
    if scored is None:
        return f"{_FOOTER_PREFIX}{_UNKNOWN}"
    chunk = scored.chunk
    return (f"Source: {chunk.publisher} - {chunk.scheme_name}\n"
            f"{format_footer(chunk.fetched_at)}")


def assemble(
    outcome: Outcome,
    body: str,
    chunks: list[ScoredChunk],
    *,
    k: int,
    top_score: float | None,
    validation: Validation,
    latency: dict[str, float],
    triage_layer: TriageLayer | None,
    reason: str | None,
) -> Answer:
    """Build the final Answer.

    Only outcome A gets a source, a publisher and a footer. B, C, D, E and error
    answers carry `source_url=None` on purpose: F7 gives class A exactly one URL
    and the other classes none, so a refusal can never be mistaken for a sourced
    answer. Class E is a question back to the user, so a source on it would be
    precisely the invented provenance 14.4.5 forbids.

    `retrieved_chunks` is passed through untouched for EVERY outcome. A class D
    refusal that shows what retrieval found is auditable; one that hides it looks
    identical to a refusal that never looked.
    """
    text = (body or "").strip()
    scored = cited_chunk(chunks, validation)

    if outcome is Outcome.A_ANSWERED:
        # A footer under an empty body would render as a bare source line, which
        # reads like an answer. An empty A body should not reach here: `validate`
        # empties the body only when every sentence carried a return figure, and
        # the pipeline converts that case to class D first.
        text = f"{text}\n{build_footer(scored)}" if text else text
        source_url = validation.cited_url
        publisher = scored.chunk.publisher if scored else None
        scheme_name = scored.chunk.scheme_name if scored else None
        last_updated = (format_date(scored.chunk.fetched_at) if scored else None)
    else:
        source_url = None
        publisher = None
        # Class E deliberately reports no scheme. Its whole text is "which scheme
        # do you mean?", and naming the top-ranked fund next to that question is
        # the answer it just asked the user not to guess. For B and D the top
        # chunk's scheme is kept, because there it is audit information: it is
        # what retrieval actually surfaced.
        scheme_name = (
            None if outcome is Outcome.E_NEEDS_SCHEME
            else (scored.chunk.scheme_name if scored else None)
        )
        last_updated = None

    if top_score is None and chunks:
        top_score = chunks[0].score

    return Answer(
        outcome=outcome,
        text=text,
        source_url=source_url,
        publisher=publisher,
        scheme_name=scheme_name,
        retrieved_chunks=list(chunks),
        top_score=top_score,
        k=k,
        last_updated=last_updated,
        validation=validation,
        latency=dict(latency),
        triage_layer=triage_layer,
        reason=reason,
    )
