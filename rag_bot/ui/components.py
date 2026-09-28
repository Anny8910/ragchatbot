"""Streamlit renderers for one answer.

**The four states must never be mistakable for each other.** That is a
correctness property, not a styling preference: a user who reads a refusal as an
answer has been told something the corpus does not support. So class A gets a
full card with its source link, and B/C/D are muted, carry no source link, and
each gets its own reason chip. Colour is reinforced with an explicit chip label
and an icon, because a colour alone fails for a colour-blind reader.
"""
from __future__ import annotations

import streamlit as st

from rag_bot.types import Answer, Outcome

# (icon, chip label) per outcome. The chip text is the accessible signal; the
# colour is only reinforcement.
_CHIP = {
    Outcome.A_ANSWERED: ("✅", "From sources"),
    Outcome.B_NOT_IN_CORPUS: ("🚫", "Not in sources"),
    Outcome.C_ADVICE_REFUSED: ("🚫", "Advice not provided"),
    Outcome.D_PERFORMANCE_REFUSED: ("🚫", "Returns not provided"),
    Outcome.E_NEEDS_SCHEME: ("❓", "Which scheme?"),
    Outcome.ERROR: ("⚠️", "Error"),
}

_MUTED = {"B", "C", "D", "E"}


def _is_refusal(answer: Answer) -> bool:
    return answer.outcome.value[:1] in _MUTED


def render_answer(answer: Answer) -> None:
    """Render one message in one of four visually distinct states."""
    icon, chip = _CHIP.get(answer.outcome, ("•", answer.outcome.value))

    if _is_refusal(answer):
        # Muted container, no source link. `st.info` is deliberately avoided:
        # it reads as neutral information, and a refusal is not neutral.
        with st.container(border=True):
            st.caption(f"{icon} {chip}")
            st.markdown(answer.text)
            if answer.outcome is Outcome.D_PERFORMANCE_REFUSED and answer.scheme_name:
                # A factsheet link is a real link, but it is NOT a source for this
                # answer -- labelled as such, or it reads as a citation.
                factsheet = _factsheet(answer)
                if factsheet:
                    st.caption(f"Factsheet (not a source for this answer): {factsheet}")
            st.caption(f"reason: {answer.reason or 'n/a'}")
    else:
        with st.container(border=True):
            st.caption(f"{icon} {chip}")
            st.markdown(answer.text)
            if answer.source_url:
                st.markdown(f"[Source: {answer.publisher or 'source'}]({answer.source_url})")
            if answer.last_updated:
                st.caption(f"Last updated from sources: {answer.last_updated}")
            if answer.outcome is Outcome.ERROR:
                st.error("This answer is unverified. Do not rely on it.")


def _factsheet(answer: Answer) -> str | None:
    for scored in answer.retrieved_chunks or []:
        if scored.chunk.factsheet_url:
            return scored.chunk.factsheet_url
    return None


def render_chunks(answer: Answer) -> None:
    """Expandable 'Retrieved chunks (N)'. Populated for EVERY outcome.

    Including refusals and errors -- showing what retrieval actually found is the
    teaching moment, and it is how a reviewer checks a wrong answer by hand.
    """
    chunks = answer.retrieved_chunks or []
    label = f"Retrieved chunks ({len(chunks)})"
    if not chunks:
        st.caption(f"{label}: none -- retrieval was not reached for this outcome.")
        return

    with st.expander(label):
        for rank, scored in enumerate(chunks, start=1):
            chunk = scored.chunk
            heading = chunk.heading or "(no heading)"
            st.markdown(
                f"**[{rank}]** {chunk.scheme_name} — “{heading}” — "
                f"score `{scored.score:.3f}` — {chunk.n_tokens} word-pieces"
            )
            with st.container(border=True):
                st.markdown(chunk.text)
            if chunk.url:
                st.caption(f"page: {chunk.url}")


def render_timing(answer: Answer) -> None:
    """Per-stage milliseconds plus which triage layer decided the outcome."""
    with st.expander("Pipeline trace"):
        latency = answer.latency or {}
        if latency:
            st.write(
                " · ".join(f"{stage} {ms:.0f}ms" for stage, ms in latency.items())
            )
        else:
            st.caption("no stages recorded")
        st.write(
            {
                "triage_layer": answer.triage_layer or "n/a",
                "reason": answer.reason or "n/a",
                "top_score": round(answer.top_score, 4)
                if answer.top_score is not None
                else None,
                "k": answer.k,
            }
        )
        report = answer.validation
        if report is not None:
            st.write(
                {
                    "sentences_ok": report.sentences_ok,
                    "single_url": report.single_url,
                    "scheme_match": report.scheme_match,
                    "no_figures": report.no_figures,
                    "unverified": report.unverified,
                    "notes": report.notes,
                }
            )


def render_welcome(example_questions: list[str], disclaimer: str) -> None:
    """Welcome line, the disclaimer read from disk, and example buttons."""
    st.title("HDFC mutual fund FAQ")
    st.markdown(
        "Answers come only from five indexed HDFC scheme pages. "
        "Figures are not in the index by design."
    )
    st.info(disclaimer)
    st.caption("Try one of these:")
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}"):
            st.session_state["pending_question"] = question
