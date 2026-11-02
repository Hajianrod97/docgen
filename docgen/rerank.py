"""Reranking: a cross-encoder rescores search hits against the question.

Search compares the question with each chunk separately: the chunk's vector
was made long before the question was asked, and BM25 only counts shared
words. A cross-encoder reads the question and one chunk *together* and
scores how well that chunk answers it, which is much more accurate but far
too slow to run over a whole repo. So the two are chained: hybrid search
narrows thousands of chunks to a few dozen candidates, and the reranker
puts those in order.

The default model is ms-marco-MiniLM-L-12-v2, run locally by fastembed (ONNX
Runtime, no PyTorch; about 120 MB, downloaded on first use). Of four tried
on the docgen eval set it was the only one that helped (top-5 file hit 94%
to 100%, MRR 0.68 to 0.75); bge-reranker-base and jina-reranker-v1-turbo
made ranking worse. On the held-out gitsummarize set it was about even
(chunk hit 79% to 83%, file MRR 0.78 to 0.74), so Quick mode doesn't use
it; the agent does, mostly for its score. That score is a raw relevance
logit: higher is better, and its scale depends on the model.
DOCGEN_RERANKER picks another fastembed reranker, or "hash" for an offline
stand-in used by tests (it counts shared words, so it means little).
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
