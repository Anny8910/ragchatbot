"""The whole app. This is the only module that imports streamlit.

Everything reusable lives in `rag_bot.ui`, so the renderers can be imported by
tests without a running Streamlit server.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from rag_bot.config import Config, load  # noqa: E402
from rag_bot.index.store import Store  # noqa: E402
from rag_bot.pipeline import answer_question  # noqa: E402
from rag_bot.providers.base import LLMProvider  # noqa: E402
from rag_bot.providers.groq import GroqProvider  # noqa: E402
from rag_bot.providers.minilm import MiniLMProvider  # noqa: E402
from rag_bot.providers.ollama import OllamaProvider  # noqa: E402
from rag_bot.providers.openai import OpenAIProvider  # noqa: E402
from rag_bot.sources.fetch import load_sources  # noqa: E402
from rag_bot.trace import log_query  # noqa: E402
from rag_bot.ui import components, sidebar  # noqa: E402
from rag_bot.types import Answer  # noqa: E402

DISCLAIMER_PATH = Path(__file__).parent / "deliverables" / "disclaimer.txt"

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the exit load on the flexi cap fund?",
    "Which HDFC fund has the best 1-year return?",
    "Should I buy the small cap fund?",
]


def read_disclaimer() -> str:
    """Deliverable D5, read from disk so the file and the UI cannot drift."""
    try:
        return DISCLAIMER_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "Facts-only. No investment advice."


def build_llm(cfg: Config) -> LLMProvider:
    """Turn RAG_PROVIDER into a live provider.

    Temperature is not passed here: both providers take it per call, and the
    pipeline supplies it from config, so setting it in two places would let them
    disagree.

    The hosted providers read their own credentials from the environment
    (`OPENAI_API_KEY`, `GROQ_API_KEY`) rather than from `Config`, so a secret is
    never carried in a settings object that gets logged or printed.

    Raises a named error for a setting that cannot work, rather than failing
    later inside a request: `fake` is accepted by config validation but there is
    no fake LLM implementation, so it can only ever be a mistake.
    """
    if cfg.provider == "ollama":
        return OllamaProvider(host=cfg.ollama_host, model=cfg.llm_model)
    if cfg.provider == "openai":
        return OpenAIProvider(model=cfg.llm_model)
    if cfg.provider == "groq":
        return GroqProvider(model=cfg.llm_model)
    if cfg.provider == "none":
        raise RuntimeError(
            "RAG_PROVIDER=none has no generator. Class A questions cannot be "
            "answered; B/C/D refusals still work. Set RAG_PROVIDER=groq or ollama."
        )
    raise RuntimeError(
        f"RAG_PROVIDER={cfg.provider!r} has no implementation. Valid values with a "
        f"generator are 'ollama', 'openai' and 'groq'."
    )


@st.cache_resource(show_spinner="loading index and model...")
def get_store(cfg: Config) -> Store:
    return Store(cfg.index_dir, cfg.embed_model, "v1")


@st.cache_resource(show_spinner="loading embedding model...")
def get_embedder(cfg: Config) -> MiniLMProvider:
    return MiniLMProvider()


@st.cache_resource(show_spinner="loading language model...")
def get_llm(cfg: Config) -> LLMProvider:
    return build_llm(cfg)


@st.cache_data(show_spinner=False)
def cached_answer(question: str, k: int, min_score: float, cfg: Config) -> Answer:
    """One answer per (question, k, min_score).

    Streamlit reruns this script on every widget change and on every keystroke
    commit, so without a cache keyed on the inputs a single question would be
    embedded, retrieved, generated, and logged several times -- inflating the
    trace log and making a demo look non-deterministic.
    """
    tuned = dataclasses.replace(cfg, top_k=k, min_score=min_score)
    return answer_question(
        question,
        cfg=tuned,
        store=get_store(tuned),
        embedder=get_embedder(tuned),
        llm=get_llm(tuned),
        sources=load_sources(tuned.sources_file),
    )


def main() -> None:
    cfg = load()
    st.set_page_config(page_title="HDFC mutual fund FAQ", layout="centered")

    try:
        chunk_count = get_store(cfg).count()
    except Exception as exc:  # noqa: BLE001 - the app must explain, not crash
        st.error(
            f"Cannot open the index at {cfg.index_dir!r}: {exc}\n\n"
            "Build it first:  python -m rag_bot.ingest.builder --rebuild"
        )
        return

    k, min_score, status_line = sidebar.render_sidebar(cfg, chunk_count)
    st.session_state.setdefault("messages", [])

    if not st.session_state["messages"]:
        components.render_welcome(EXAMPLE_QUESTIONS, read_disclaimer())
        st.session_state["status_line"] = status_line

    for message in st.session_state["messages"]:
        with st.chat_message("user"):
            st.markdown(message["question"])
        with st.chat_message("assistant"):
            components.render_answer(message["answer"])
            components.render_chunks(message["answer"])
            components.render_timing(message["answer"])

    pending = st.session_state.pop("pending_question", None)
    typed = st.chat_input("Ask about expense ratio, exit load, SIP, lock-in, riskometer, benchmark")
    question = typed or pending
    if not question:
        return

    st.session_state["messages"].append({"question": question, "answer": None})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.caption("retrieving...")
        try:
            answer = cached_answer(question, k, min_score, cfg)
        except Exception as exc:  # noqa: BLE001 - surface it, do not crash the app
            st.error(f"The model could not be reached: {exc}")
            st.session_state["messages"].pop()
            return
        placeholder.caption("generating...")
        placeholder.empty()
        components.render_answer(answer)
        components.render_chunks(answer)
        components.render_timing(answer)

    if cfg.log_trace:
        log_query(answer, question)
    st.session_state["messages"][-1]["answer"] = answer
    with st.expander("Corpus"):
        st.caption(st.session_state.get("status_line", ""))


if __name__ == "__main__":
    main()
