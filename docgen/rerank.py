"""Reranking: a cross-encoder scores each search hit against the question, reading
both together. The model is ms-marco-MiniLM-L-12-v2 via fastembed
(DOCGEN_RERANKER to change it; see docs/evaluation.md for how it was chosen).
"""

from __future__ import annotations

import dataclasses
import os

from docgen.chunking import tokenize

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-12-v2"
CANDIDATES = 20  # how many search hits the reranker reorders


class _HashScorer:
    """Offline stand-in: the share of the question's words found in the text."""

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        words = set(tokenize(question))
        return [len(words & set(tokenize(t))) / (len(words) or 1) for t in texts]


class _CrossEncoderScorer:
    _models: dict = {}  # model name -> loaded model, shared by every instance

    def __init__(self, model: str):
        self.model = model

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        if self.model not in self._models:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._models[self.model] = TextCrossEncoder(self.model)
        return [float(s) for s in self._models[self.model].rerank(question, texts)]


def get_scorer():
    """The configured reranker: a callable (question, texts) -> scores."""
    choice = os.environ.get("DOCGEN_RERANKER", "").strip()
    if choice.lower() == "hash":
        return _HashScorer()
    return _CrossEncoderScorer(choice or DEFAULT_RERANKER)


def _pair_text(hit) -> str:
    """What the reranker reads for a hit: where it is, what it's called, its text."""
    return f"{hit.citation} {hit.label}\n{hit.text}"


def rerank(question: str, hits: list, scorer=None) -> list:
    """The hits, best first by the reranker, each with its rerank_score set."""
    if not hits:
        return []
    scorer = scorer or get_scorer()
    scores = scorer(question, [_pair_text(h) for h in hits])
    scored = [dataclasses.replace(h, rerank_score=round(s, 4)) for h, s in zip(hits, scores, strict=True)]
    return sorted(scored, key=lambda h: -h.rerank_score)
