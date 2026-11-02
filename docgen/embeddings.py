"""Embedding function selection, set by the DOCGEN_EMBEDDINGS environment variable.

- unset: all-MiniLM-L6-v2, run locally on the CPU by fastembed (downloaded on
  first use, about 90 MB). Small and quick, but trained on English sentences,
  and it reads only the first 256 word pieces of each chunk.
- "hash": a small deterministic offline embedder for tests and network-
  restricted environments. It matches on shared identifier words, not on
  meaning, so retrieval quality is noticeably worse than a real model.

Changing the choice rebuilds a repo's search index on its next update, since
vectors from different models can't be compared.

Each choice is a LangChain `Embeddings`: embed_documents for what is stored,
embed_query for questions.
"""

from __future__ import annotations

import hashlib
import math
import os
import re

from langchain_core.embeddings import Embeddings

from docgen.chunking import word_parts


def _tokenize(text: str) -> list[str]:
    """Split code into lowercase word pieces (snake_case and camelCase aware)."""
    return [part for word in re.findall(r"[A-Za-z0-9_]+", text) for part in word_parts(word)]


class HashingEmbedding(Embeddings):
    """Deterministic bag-of-words embedding via the hashing trick."""

    docgen_id = "docgen-hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for doc in texts:
            vec = [0.0] * self.dim
            for tok in _tokenize(doc):
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedding(Embeddings):
    """all-MiniLM-L6-v2 through fastembed (ONNX Runtime, no PyTorch). The
    model is loaded once per process, on first use."""

    docgen_id = f"fastembed:{DEFAULT_MODEL}"
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            from fastembed import TextEmbedding

            cls._model = TextEmbedding(DEFAULT_MODEL)
        return cls._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_function() -> Embeddings:
    """Return the configured embedding model."""
    choice = os.environ.get("DOCGEN_EMBEDDINGS", "").strip().lower()
    if choice == "hash":
        return HashingEmbedding()
    if choice:
        raise ValueError(f"Unknown DOCGEN_EMBEDDINGS {choice!r}: leave it unset for MiniLM, or use 'hash'")
    return MiniLMEmbedding()
