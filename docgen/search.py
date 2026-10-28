"""Hybrid search over the repo index: vector search and BM25 keyword search, fused.

The two halves fail differently, which is why both are used:

- Vector search (ChromaDB, all-MiniLM-L6-v2 by default) matches meaning, so
  "how are repos downloaded" can find code that says `git clone`. It is weak
  on exact names: a question about "the cost estimate" may not find a method
  called `estimate`.
- BM25 matches words, weighted so rare words count more than common ones and
  a word repeated in a long chunk counts less than in a short one. It finds
  `estimate` at once but knows nothing about synonyms.

Their rankings are merged with reciprocal rank fusion (RRF): each result
scores 1 / (RRF_K + rank) in every list it appears in, and the scores are
added. RRF only looks at ranks, so it needs no tuning to make cosine
similarities and BM25 scores comparable, which they aren't.

Plain RRF trusts both lists equally, so a result only one search found loses
to results both found further down. `alpha` weights the vector list against
the keyword list: 0.5 is plain RRF, 1.0 is vector only, 0.0 keyword only.

The BM25 index is built in memory from the documents already stored in
ChromaDB, and rebuilt whenever the index's manifest changes.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from docgen.repo_index import manifest_path, open_repo_collection

RRF_K = 60  # the usual constant; larger values flatten the difference between ranks
HYBRID_ALPHA = 0.5  # weight of the vector list in hybrid fusion (0.5 = plain RRF)
CANDIDATES = 30  # how many results each half contributes before fusion
MODES = ("hybrid", "vector", "keyword")

_STOPWORDS = frozenset("""
a an and are as at be by can do does for from has have how i if in into is it its
of on or so that the their then there these this to was what when where which who
why will with you your
""".split())
_WORD = re.compile(r"[A-Za-z][a-z0-9]*|[A-Z]+(?![a-z])|\d+")


def _stem(token: str) -> str:
    """A deliberately crude stemmer, so "cloned", "clones" and "clone" match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Words for BM25. Identifiers are split at underscores and camelCase, and
    also kept whole, so `make_hub_models` matches "hub models" and itself."""
    tokens = []
    for word in re.findall(r"[A-Za-z0-9_]+", text):
        parts = [p.lower() for p in _WORD.findall(word)]
        if len(parts) > 1:
            tokens.append(word.lower())
        tokens += parts
    return [_stem(t) for t in tokens if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_length = (sum(self.lengths) / len(docs)) if docs else 0.0
        df = Counter(term for c in self.counts for term in c)
        n = len(docs)
        # The "+1" form of IDF never goes negative, even for very common words.
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for counts, length in zip(self.counts, self.lengths, strict=True):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
            for term in set(query):
                tf = counts.get(term)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out.append(score)
        return out


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

    @property
    def citation(self) -> str:
        if self.type == "code":
            return f"{self.path}:{self.start_line}-{self.end_line}"
        return f"{self.path or '(repo root)'} (summary)"


class _Corpus:
    """Everything in the index, plus a BM25 index over it."""

    def __init__(self, collection):
        got = collection.get(include=["documents", "metadatas"])
        self.ids = got["ids"]
        self.documents = dict(zip(got["ids"], got["documents"], strict=True))
        self.metadatas = dict(zip(got["ids"], got["metadatas"], strict=True))
        self.bm25 = BM25([tokenize(d) for d in got["documents"]])


_corpora: dict[str, tuple[tuple, _Corpus]] = {}
_corpora_lock = threading.Lock()


def _corpus(repo: Path, collection) -> _Corpus:
    manifest = manifest_path(repo)
    stat = manifest.stat()
    stamp = (stat.st_mtime_ns, stat.st_size, collection.count())
    with _corpora_lock:
        cached = _corpora.get(str(repo))
        if cached and cached[0] == stamp:
            return cached[1]
    corpus = _Corpus(collection)
    with _corpora_lock:
        _corpora[str(repo)] = (stamp, corpus)
    return corpus


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
    collection = open_repo_collection(repo)
    corpus = _corpus(repo, collection)
    if not corpus.ids:
        return []
    def allowed(doc_id: str) -> bool:
        meta = corpus.metadatas[doc_id]
        return (not types or meta["type"] in types) and meta["path"] not in exclude_paths

    vector: list[str] = []
    if mode in ("hybrid", "vector"):
        where = {"type": {"$in": list(types)}} if types else None
        # Ask for extra so excluded entries don't leave the list short.
        n = min(CANDIDATES + 10 * len(exclude_paths), len(corpus.ids))
        result = collection.query(query_texts=[query], n_results=n, where=where)
        vector = [i for i in result["ids"][0] if allowed(i)][:CANDIDATES]

    keyword: list[str] = []
    if mode in ("hybrid", "keyword"):
        scores = corpus.bm25.scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        keyword = [corpus.ids[i] for i in ranked if scores[i] > 0 and allowed(corpus.ids[i])][:CANDIDATES]

    # Weights 2·alpha and 2·(1 - alpha), so alpha = 0.5 gives plain RRF's scores.
    fused = rrf([vector, keyword], weights=[2 * alpha, 2 * (1 - alpha)])
    vector_rank = {doc_id: n for n, doc_id in enumerate(vector, 1)}
    keyword_rank = {doc_id: n for n, doc_id in enumerate(keyword, 1)}
    best = sorted(fused, key=lambda i: (-fused[i], i))[:k]

    hits = []
    for doc_id in best:
        meta = corpus.metadatas[doc_id]
        is_code = meta["type"] == "code"
        hits.append(SearchHit(
            id=doc_id,
            type=meta["type"],
            path=meta["path"],
            start_line=meta.get("start_line") if is_code else None,
            end_line=meta.get("end_line") if is_code else None,
            label=meta.get("symbol", "") if is_code else (meta.get("section") or meta.get("level", "")),
            text=_strip_header(corpus.documents[doc_id]),
            score=round(fused[doc_id], 5),
            vector_rank=vector_rank.get(doc_id),
            keyword_rank=keyword_rank.get(doc_id),
        ))
    return hits
