"""Measures how well search finds the right files, on a set of questions with known answers.

An eval file is JSON:

    {"repo": "..",   # relative to the eval file, or absolute, or a GitHub URL
     "questions": [{"question": "...", "files": ["path/that/answers/it.py"]}]}

For each question and search mode, a question is a *hit* if any of its files
appears among the top k results (a code chunk of the file, or the file's hub
summary). Two numbers come out per mode:

- hit rate @k: the share of questions with a hit. The headline number.
- MRR (mean reciprocal rank): 1/rank of the first right result, averaged, so
  finding it at rank 1 beats finding it at rank 5.

Change one thing (chunk size, the embedding model, how results are fused),
run this again, and compare. Without numbers like these, a retrieval change
is a guess.

With --rewrite, two more modes run, which need a model (Claude Haiku by
default; rewrites are cached, so a rerun is free):

- rw-keyword: a model turns the question into likely identifiers first, then
  keyword search looks for the question plus those terms. Exact search with
  a model in front of it, the way a coding agent searches.
- rw-hybrid: the same rewritten query through hybrid search.

Run: python -m docgen.evaluate evals/docgen.json [--k 5] [--mode hybrid] [--rewrite]
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from docgen.search import HYBRID_ALPHA, MODES, SearchHit, search

# Rewrite modes, and the search mode each one runs on the rewritten query.
REWRITE_MODES = {"rw-keyword": "keyword", "rw-hybrid": "hybrid"}


@dataclass
class QuestionResult:
    question: str
    files: list[str]
    rank: int | None  # 1-based rank of the first right result, None if missed
    found: list[str] = field(default_factory=list)  # citations of the top results


@dataclass
class ModeResult:
    mode: str
    k: int
    questions: list[QuestionResult]

    @property
    def hit_rate(self) -> float:
        return sum(q.rank is not None for q in self.questions) / len(self.questions)

    @property
    def mrr(self) -> float:
        return sum(1 / q.rank for q in self.questions if q.rank) / len(self.questions)


def _matches(hit: SearchHit, files: list[str]) -> bool:
    # A folder summary doesn't count: it names the file at best in passing.
    return hit.path in files and (hit.type == "code" or hit.id.startswith("summary:file:"))


def load_eval(path: str | Path) -> tuple[str, list[dict]]:
    """(repo, questions) from an eval file; a relative repo is resolved
    against the eval file's folder."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    repo = data.get("repo", ".")
    if "://" not in repo and not repo.startswith("github.com"):
        repo = str((path.parent / repo).resolve())
    return repo, data["questions"]


def _relative_to(path: Path, repo: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None


def evaluate(repo: str | Path, questions: list[dict], k: int = 5, mode: str = "hybrid",
             eval_file: str | Path | None = None,
             rewrite: Callable[[str], str] | None = None,
             alpha: float = HYBRID_ALPHA) -> ModeResult:
    """Run every question through search. If the eval file sits inside the
    repo, it's left out of the results: it contains the questions word for
    word, so it would otherwise "answer" all of them. A rewrite mode needs
    `rewrite`, which turns a question into extra search terms."""
    search_mode = REWRITE_MODES.get(mode, mode)
    if mode in REWRITE_MODES and rewrite is None:
        raise ValueError(f"Mode {mode} needs a rewrite function")
    exclude = frozenset()
    if eval_file and (rel := _relative_to(Path(eval_file), Path(repo))):
        exclude = frozenset({rel})
    results = []
    for q in questions:
        query = q["question"]
        if mode in REWRITE_MODES:
            query = f"{query} {rewrite(query)}"
        hits = search(repo, query, k=k, mode=search_mode, exclude_paths=exclude, alpha=alpha)
        rank = next((n for n, h in enumerate(hits, 1) if _matches(h, q["files"])), None)
        results.append(QuestionResult(q["question"], q["files"], rank, [h.citation for h in hits]))
    return ModeResult(mode, k, results)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from docgen.llm import ProviderError
    from docgen.repo_index import update_index
    from docgen.repos import RepoError, resolve_repo

    parser = argparse.ArgumentParser(prog="python -m docgen.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("eval_file")
    parser.add_argument("--k", type=int, default=5, help="how many results count (default 5)")
    parser.add_argument("--alpha", type=float, default=HYBRID_ALPHA,
                        help=f"weight of vector results in hybrid search, 0-1 (default {HYBRID_ALPHA})")
    parser.add_argument("--mode", choices=MODES + ("all",), default="all")
    parser.add_argument("--misses", action="store_true", help="list the questions each mode missed")
    parser.add_argument("--rewrite", action="store_true",
                        help="also run the query-rewriting modes (needs ANTHROPIC_API_KEY, or --provider ollama)")
    parser.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic",
                        help="who rewrites queries (default anthropic)")
    parser.add_argument("--rewrite-model", help="model that rewrites queries")
    args = parser.parse_args(argv)

    repo_text, questions = load_eval(args.eval_file)
    try:
        repo = resolve_repo(repo_text).path
    except RepoError as e:
        print(e)
        return 1
    stats = update_index(repo)  # cheap when nothing changed
    if stats.entries_added:
        print(f"Updated the search index first ({stats.files_indexed} files, {stats.summaries_indexed} summaries).")

    modes = MODES if args.mode == "all" else (args.mode,)
    rewrite, cost = None, None
    if args.rewrite:
        rewrite, cost = _make_rewriter(repo, args.provider, args.rewrite_model)
        modes = modes + tuple(REWRITE_MODES)
    try:
        results = [evaluate(repo, questions, args.k, m, eval_file=args.eval_file, rewrite=rewrite, alpha=args.alpha)
                   for m in modes]
    except ProviderError as e:
        print(f"Stopped: {e}")
        return 1
    print(f"{len(questions)} questions, top {args.k} results\n")
    print(f"{'mode':<11} {'hit rate':>9} {'MRR':>6}")
    for r in results:
        print(f"{r.mode:<11} {r.hit_rate:>8.0%} {r.mrr:>6.2f}")
    if cost:
        spent = cost()
        print(f"\nRewriting cost {'nothing (all cached or free)' if not spent else f'${spent:.4f}'}.")

    if args.misses or len(results) == 1:
        for r in results:
            missed = [q for q in r.questions if q.rank is None]
            if not missed:
                continue
            print(f"\nMissed by {r.mode}:")
            for q in missed:
                print(f"  - {q.question}\n    wanted {', '.join(q.files)}; got {', '.join(q.found[:3])}")
    if args.misses and rewrite:
        print("\nRewrites (cached in .docgen/rewrites.json):")
        for q in questions:
            print(f"  - {q['question']}\n    -> {rewrite(q['question'])}")
    return 0


def _make_rewriter(repo: Path, provider: str, model: str | None):
    """(question -> terms, () -> dollars spent so far). A reply without the
    expected tags falls back to no extra terms, so one bad reply can't stop
    the eval."""
    from docgen.hub import model_cost
    from docgen.llm import (
        DEFAULT_OLLAMA_MODEL,
        GenerationError,
        UsageMeter,
        make_anthropic_completer,
        make_ollama_completer,
    )
    from docgen.rewrite import REWRITE_MODEL, rewrite_query

    meter = None
    if provider == "anthropic":
        model = model or REWRITE_MODEL
        meter = UsageMeter()
        complete = make_anthropic_completer(model, effort="low", meter=meter)
    else:
        model = model or DEFAULT_OLLAMA_MODEL
        complete = make_ollama_completer(model, max_tokens=256)

    def rewrite(question: str) -> str:
        try:
            return rewrite_query(repo, question, complete, model)
        except GenerationError:
            return ""

    def cost() -> float | None:
        return model_cost(model, meter.input_tokens, meter.output_tokens) if meter else 0.0

    return rewrite, cost


if __name__ == "__main__":
    raise SystemExit(main())
