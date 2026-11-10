# docgen's web app. compose.yaml runs it next to the database; see the README.
FROM python:3.12-slim-bookworm

# git clones GitHub repos (docgen/repos.py).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

# A non-root user. /data is where compose mounts a volume for clones; a new
# volume takes this folder's owner, so docgen can write to it.
RUN useradd --create-home docgen && mkdir /data && chown docgen /data
USER docgen
WORKDIR /app

# Dependencies first, in their own layer, so editing code doesn't reinstall them.
COPY --chown=docgen pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Bake the models into the image (embeddings and reranker, about 210 MB), so
# the first question doesn't wait on a download.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-12-v2')"

COPY --chown=docgen docgen ./docgen
# The tree-sitter grammars docgen splits code with, so chunking works offline.
RUN python -c "from tree_sitter_language_pack import prefetch; from docgen.chunking import STRUCTURED_GRAMMARS; \
prefetch(sorted(STRUCTURED_GRAMMARS))"

ENV DOCGEN_WEB_HOST=0.0.0.0 \
    DOCGEN_NO_BROWSER=1 \
    PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["python", "-m", "docgen.web"]
