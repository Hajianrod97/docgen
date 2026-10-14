"""Embedding function selection.

By default docgen uses Chroma's built-in all-MiniLM-L6-v2 model (downloaded on
first use, runs locally, no API key). Set DOCGEN_EMBEDDINGS=hash to use a small
deterministic offline embedder instead. The hashing embedder is for tests and
network-restricted environments only: it matches on shared identifier words,
not on meaning, so retrieval quality is noticeably worse than the real model.
"""

from __future__ import annotations

import hashlib
import math
import os
import re

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
    def build_from_config(config: dict) -> "HashingEmbedding":
        return HashingEmbedding(dim=config.get("dim", 384))


def get_embedding_function():
    """Return the configured embedding function (None = Chroma's default model)."""
    if os.environ.get("DOCGEN_EMBEDDINGS", "").lower() == "hash":
        return HashingEmbedding()
    return None
