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

Run: python -m docgen.evaluate evals/docgen.json [--k 5] [--mode hybrid]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from docgen.search import MODES, SearchHit, search


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
             eval_file: str | Path | None = None) -> ModeResult:
    """Run every question through search. If the eval file sits inside the
    repo, it's left out of the results: it contains the questions word for
    word, so it would otherwise "answer" all of them."""
    exclude = frozenset()
    if eval_file and (rel := _relative_to(Path(eval_file), Path(repo))):
        exclude = frozenset({rel})
    results = []
    for q in questions:
        hits = search(repo, q["question"], k=k, mode=mode, exclude_paths=exclude)
        rank = next((n for n, h in enumerate(hits, 1) if _matches(h, q["files"])), None)
        results.append(QuestionResult(q["question"], q["files"], rank, [h.citation for h in hits]))
    return ModeResult(mode, k, results)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from docgen.repo_index import update_index
    from docgen.repos import RepoError, resolve_repo

    parser = argparse.ArgumentParser(prog="python -m docgen.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("eval_file")
    parser.add_argument("--k", type=int, default=5, help="how many results count (default 5)")
    parser.add_argument("--mode", choices=MODES + ("all",), default="all")
    parser.add_argument("--misses", action="store_true", help="list the questions each mode missed")
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
    results = [evaluate(repo, questions, args.k, m, eval_file=args.eval_file) for m in modes]
    print(f"{len(questions)} questions, top {args.k} results\n")
    print(f"{'mode':<9} {'hit rate':>9} {'MRR':>6}")
    for r in results:
        print(f"{r.mode:<9} {r.hit_rate:>8.0%} {r.mrr:>6.2f}")

    if args.misses or len(results) == 1:
        for r in results:
            missed = [q for q in r.questions if q.rank is None]
            if not missed:
                continue
            print(f"\nMissed by {r.mode}:")
            for q in missed:
                print(f"  - {q.question}\n    wanted {', '.join(q.files)}; got {', '.join(q.found[:3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
