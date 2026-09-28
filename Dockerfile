# syntax=docker/dockerfile:1
#
# Render builds this image. Two things about it are not the default for a
# Streamlit app, and both are load-bearing:
#
# 1. The CPU-only torch index. The default PyPI torch wheel for linux/x86_64
#    bundles the CUDA libraries, which add ~2.5GB to the image for a service
#    that never touches a GPU. The build pulls ~400MB instead.
# 2. The index is baked into the image, not built at start-up. The embedder and
#    the Chroma collection are both deterministic artefacts of committed
#    snapshots, so building them per-deploy would add a couple of minutes and a
#    failure mode to every cold start, and would make the running service
#    depend on being able to reach Hugging Face.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Streamlit renders its own port; 10000 is Render's convention for web
    # services and is what render.yaml's healthCheckPath expects.
    STREAMLIT_SERVER_PORT=10000 \
    STREAMLIT_SERVER_HEADLESS=true \
    # The browser opens a websocket back to this host. Without it Streamlit
    # guesses, and behind a proxy the guess is wrong, so the UI renders and then
    # fails on every interaction.
    STREAMLIT_SERVER_ENABLE_CORS=false \
    # Keep the container quiet: the trace log goes to stdout for Render.
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Build deps for chromadb's native pieces, removed in the same layer they are
# used so they do not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# torch first, from the CPU index, so pip does not resolve and download the CUDA
# wheel before the index constraint applies.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple -r requirements.txt

# The rest of the app, then the committed snapshots. `data/corpus/` is copied
# because the index is built from it inside the image, not because it is read
# at runtime: a served answer cites the URL in the chunk metadata, never the
# file.
COPY rag_bot/ ./rag_bot/
COPY data/corpus/ ./data/corpus/
COPY app.py streamlit_app.toml ./

# The index is gitignored (`data/index/`), so it is produced here rather than
# copied. Building it in the image rather than at start-up means a cold start
# does not need to reach Hugging Face, and it fails the deploy instead of the
# first user request.
RUN python -m rag_bot.ingest.builder

# Fail the build rather than the first user request if the index is empty.
RUN python -c "\
from rag_bot.config import load;\
from rag_bot.index.store import Store;\
c = load();\
n = Store(c.index_dir, c.embed_model, 'v1').count();\
print('indexed chunks:', n);\
raise SystemExit(0 if n >= 40 else 1)"

# Non-root. The service only ever reads its own files.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:10000/_stcore/health || exit 1

# exec form, so Streamlit is PID 1 and receives SIGTERM directly rather than
# through a shell that ignores it.
CMD ["streamlit", "run", "app.py", \
     "--server.port=10000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
