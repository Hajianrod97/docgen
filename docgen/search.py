"""Hybrid search over the repo index: vector search and BM25 keyword search, fused.

The two halves fail differently, which is why both are used:

- Vector search (pgvector: cosine distance to the question's embedding,
  all-MiniLM-L6-v2 by default) matches meaning, so
  "how are repos downloaded" can find code that says `git clone`. It is weak
  on exact names: a question about "the cost estimate" may not find a method
  called `estimate`.
- BM25 matches words, weighted so rare words count more than common ones and
  a word repeated in a long chunk counts less than in a short one. It finds
  `estimate` at once but knows nothing about synonyms. The scoring is
  ParadeDB pg_search's; the words it scores come from docgen's `tokenize`,
  which splits identifiers.

Their rankings are merged with reciprocal rank fusion (RRF): each result
scores 1 / (RRF_K + rank) in every list it appears in, and the scores are
added. RRF only looks at ranks, so it needs no tuning to make cosine
similarities and BM25 scores comparable, which they aren't.

Plain RRF trusts both lists equally, so a result only one search found loses
to results both found further down. `alpha` weights the vector list against
the keyword list: 0.5 is plain RRF, 1.0 is vector only, 0.0 keyword only.

Both halves are SQL queries against the repo's table (docgen.store.ranked).

`RepoRetriever` offers the same search as a LangChain retriever, for code
built on LangChain (such as a LangGraph agent).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from docgen import store

RRF_K = 60  # the usual constant; larger values flatten the difference between ranks
HYBRID_ALPHA = 0.5  # weight of the vector list in hybrid fusion (0.5 = plain RRF)
CANDIDATES = 30  # how many results each half contributes before fusion
MODES = ("hybrid", "vector", "keyword")


def rrf(rankings: list[list[str]], k: int = RRF_K,
        weights: list[float] | None = None) -> dict[str, float]:
    """Reciprocal rank fusion of several ranked id lists, each optionally weighted."""
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights or [1.0] * len(rankings), strict=True):
        for rank, doc_id in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused


@dataclass
class SearchHit:
    id: str
    type: str  # "code" or "summary"
    path: str
    start_line: int | None  # code only
    end_line: int | None
    label: str  # symbol, heading path, or summary section
    text: str  # without the header that was added for embedding
    score: float
    vector_rank: int | None  # 1-based rank in each half, None if not in its top CANDIDATES
    keyword_rank: int | None
    rerank_score: float | None = None  # set by docgen.rerank

    @property
    def citation(self) -> str:
        if self.type == "code":
            return f"{self.path}:{self.start_line}-{self.end_line}"
        return f"{self.path or '(repo root)'} (summary)"


def _strip_header(document: str) -> str:
    _, sep, rest = document.partition("\n\n")
    return rest if sep else document


def search(repo_path: str | Path, query: str, k: int = 8, mode: str = "hybrid",
           types: tuple[str, ...] | None = None,
           exclude_paths: frozenset[str] = frozenset(),
           alpha: float = HYBRID_ALPHA) -> list[SearchHit]:
    """The k best index entries for a question. `mode` picks hybrid, vector
    only or keyword only (useful for comparing them); `types` limits results
    to "code" or "summary" entries; entries of `exclude_paths` are left out;
    `alpha` weights vector against keyword results in hybrid mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown search mode {mode!r}; use one of {', '.join(MODES)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, not {alpha}")
    repo = Path(repo_path).resolve()
    vector, keyword, entries = store.ranked(repo, query, mode, types, exclude_paths, CANDIDATES)
    if not vector and not keyword:
        return []

    # Weights 2·alpha and 2·(1 - alpha), so alpha = 0.5 gives plain RRF's scores.
    fused = rrf([vector, keyword], weights=[2 * alpha, 2 * (1 - alpha)])
    vector_rank = {doc_id: n for n, doc_id in enumerate(vector, 1)}
    keyword_rank = {doc_id: n for n, doc_id in enumerate(keyword, 1)}
    best = sorted(fused, key=lambda i: (-fused[i], i))[:k]

    hits = []
    for doc_id in best:
        meta, document = entries[doc_id]
        is_code = meta["type"] == "code"
        hits.append(SearchHit(
            id=doc_id,
            type=meta["type"],
            path=meta["path"],
            start_line=meta["start_line"] if is_code else None,
            end_line=meta["end_line"] if is_code else None,
            label=meta.get("symbol", "") if is_code else (meta.get("section") or meta.get("level", "")),
            text=_strip_header(document),
            score=round(fused[doc_id], 5),
            vector_rank=vector_rank.get(doc_id),
            keyword_rank=keyword_rank.get(doc_id),
        ))
    return hits


class RepoRetriever(BaseRetriever):
    """`search` as a LangChain retriever. Each Document is one hit: its text,
    with the rest of the hit (path, lines, citation, ranks) as metadata."""

    repo_path: str
    k: int = 8
    mode: str = "hybrid"
    types: tuple[str, ...] | None = None
    alpha: float = HYBRID_ALPHA

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hits = search(self.repo_path, query, k=self.k, mode=self.mode, types=self.types, alpha=self.alpha)
        return [Document(page_content=h.text, id=h.id,
                         metadata={k: v for k, v in dataclasses.asdict(h).items() if k != "text"}
                         | {"citation": h.citation})
                for h in hits]
