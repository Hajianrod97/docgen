"""Embedding function selection, set by the DOCGEN_EMBEDDINGS environment variable.

- unset: Chroma's built-in all-MiniLM-L6-v2 (downloaded on first use, runs
  locally on the CPU). Small and quick, but trained on English sentences, and
  it reads only the first 256 word pieces of each chunk.
- "ollama" or "ollama:<model>": an embedding model served by a local Ollama
  server, nomic-embed-text by default (`ollama pull nomic-embed-text`). It
  reads up to 8,192 tokens, so whole chunks are embedded.
- "hash": a small deterministic offline embedder for tests and network-
  restricted environments. It matches on shared identifier words, not on
  meaning, so retrieval quality is noticeably worse than a real model.

Changing the choice rebuilds a repo's search index on its next update, since
vectors from different models can't be compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request

from chromadb import Documents, EmbeddingFunction, Embeddings

_TOKEN_RE = re.compile(r"[A-Za-z][a-z0-9]*|[A-Z]+(?![a-z])|\d+")


def _tokenize(text: str) -> list[str]:
    """Split code into lowercase word pieces (snake_case and camelCase aware)."""
    tokens: list[str] = []
    for word in re.split(r"[^A-Za-z0-9]+", text):
        if not word:
            continue
        tokens.extend(t.lower() for t in _TOKEN_RE.findall(word))
    return tokens


class HashingEmbedding(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words embedding via the hashing trick."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for doc in input:
            vec = [0.0] * self.dim
            for tok in _tokenize(doc):
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "docgen-hashing"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> HashingEmbedding:
        return HashingEmbedding(dim=config.get("dim", 384))


class EmbeddingError(Exception):
    """The embedding model couldn't be reached or gave an unusable reply."""


DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Some embedding models are trained with a prefix that says whether the text is
# a stored document or a search query; leaving it out makes them noticeably worse.
_TASK_PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}


class OllamaEmbedding(EmbeddingFunction[Documents]):
    """Embeddings from a local Ollama server's /api/embed endpoint."""

    def __init__(self, model: str = DEFAULT_OLLAMA_EMBED_MODEL, host: str | None = None,
                 num_ctx: int = 8192, timeout: int = 300):
        from docgen.llm import _normalize_ollama_host

        self.model = model
        self.host = _normalize_ollama_host(host)
        self.num_ctx = num_ctx
        self.timeout = timeout
        self._prefixes = _TASK_PREFIXES.get(model.split(":")[0], ("", ""))
        # Straight to the server, not through any system proxy (as in docgen.llm).
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def docgen_id(self) -> str:
        """Names the model in the index manifest, so switching models rebuilds."""
        return f"ollama:{self.model}"

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed([self._prefixes[0] + text for text in input])

    def embed_query(self, input: Documents) -> Embeddings:
        return self._embed([self._prefixes[1] + text for text in input])

    def _embed(self, texts: list[str]) -> Embeddings:
        body = json.dumps({"model": self.model, "input": texts, "truncate": True,
                           "options": {"num_ctx": self.num_ctx}}).encode("utf-8")
        request = urllib.request.Request(f"{self.host}/api/embed", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise EmbeddingError(f"Ollama has no model named {self.model!r}. "
                                     f"Download it with: ollama pull {self.model}") from e
            raise EmbeddingError(f"Ollama returned HTTP {e.code} for embeddings") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise EmbeddingError(f"Could not reach Ollama at {self.host} for embeddings. "
                                 f"Is it running? ({e})") from e
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama's embedding reply did not have one vector per input")
        return vectors

    @staticmethod
    def name() -> str:
        return "docgen-ollama"

    def get_config(self) -> dict:
        return {"model": self.model, "host": self.host}

    @staticmethod
    def build_from_config(config: dict) -> OllamaEmbedding:
        return OllamaEmbedding(model=config.get("model", DEFAULT_OLLAMA_EMBED_MODEL), host=config.get("host"))


def get_embedding_function():
    """Return the configured embedding function (None = Chroma's default model)."""
    choice = os.environ.get("DOCGEN_EMBEDDINGS", "").strip()
    if choice.lower() == "hash":
        return HashingEmbedding()
    if choice.lower() == "ollama" or choice.lower().startswith("ollama:"):
        model = choice.split(":", 1)[1] if ":" in choice else DEFAULT_OLLAMA_EMBED_MODEL
        return OllamaEmbedding(model=model)
    return None
