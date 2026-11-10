"""Grades docgen's answers, not just its search: two checks from the eval labels
(does the answer cite a source, and a right file) and optional LLM-as-judge
scores from openevals (groundedness, helpfulness, retrieval relevance). The
judge scores are uncalibrated; see docs/evaluation.md.

Run: python -m docgen.answer_eval evals/gitsummarize.json --provider ollama --no-judge [--agent] [--limit 10]
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docgen.ask import Answer

JUDGED = ("groundedness", "helpfulness", "retrieval_relevance")
CHECKED = ("cites", "right_file_cited")
JUDGE_MODELS = {"anthropic": "claude-sonnet-5-5", "ollama": "qwen2.5-coder:7b"}
JUDGE_MAX_TOKENS = 1024  # a verdict is a few sentences
JUDGE_TIMEOUT = 180  # seconds; don't wait forever on a stuck server

Answerer = Callable[[str], tuple[Answer, list[str]]]  # question -> (answer, steps)
Judge = Callable[..., dict]  # an openevals evaluator: (inputs=, outputs=, context=) -> {"score", "comment"}


@dataclass
class AnswerResult:
    question: str
    files: list[str]
    answer: str
    sources: list[str]  # citations of the sources the answer was given
    cited: list[str]  # citations of the sources it cited
    steps: list[str]
    seconds: float
    scores: dict[str, bool | None] = field(default_factory=dict)  # None: the judge gave no usable verdict
    reasons: dict[str, str] = field(default_factory=dict)  # why the judge scored as it did


def make_judges(provider: str, model: str | None = None) -> dict[str, Judge]:
    """The three openevals RAG judges, run by one chat model."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import RAG_GROUNDEDNESS_PROMPT, RAG_HELPFULNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT

    model = model or JUDGE_MODELS[provider]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model=model, max_tokens=4096, default_request_timeout=JUDGE_TIMEOUT)
    else:
        from langchain_ollama import ChatOllama

        from docgen.llm import OLLAMA_NUM_CTX, _normalize_ollama_host

        chat = ChatOllama(model=model, base_url=_normalize_ollama_host(None), num_ctx=OLLAMA_NUM_CTX,
                          temperature=0, num_predict=JUDGE_MAX_TOKENS,
                          sync_client_kwargs={"trust_env": False, "timeout": JUDGE_TIMEOUT})
    prompts = {"groundedness": RAG_GROUNDEDNESS_PROMPT, "helpfulness": RAG_HELPFULNESS_PROMPT,
               "retrieval_relevance": RAG_RETRIEVAL_RELEVANCE_PROMPT}
    return {key: create_llm_as_judge(prompt=prompt, judge=chat, feedback_key=key) for key, prompt in prompts.items()}


def _context(answer: Answer) -> str:
    """The sources as the judge reads them: numbered, with where each came from."""
    return "\n\n".join(f"[{n}] {h.citation}\n{h.text}" for n, h in enumerate(answer.sources, 1))


def grade(question: dict, answerer: Answerer, judges: dict[str, Judge]) -> AnswerResult:
    """Answer one question and score the answer."""
    started = time.monotonic()
    answer, steps = answerer(question["question"])
    seconds = time.monotonic() - started
    cited = [answer.sources[n - 1] for n in answer.cited]
    result = AnswerResult(question["question"], question["files"], answer.text,
                          [h.citation for h in answer.sources], [h.citation for h in cited], steps, round(seconds, 1))
    result.scores["cites"] = bool(cited)
    result.scores["right_file_cited"] = any(h.path in question["files"] for h in cited)
    context = _context(answer)
    for key, judge in judges.items():
        try:
            verdict = judge(inputs=question["question"], outputs=answer.text, context=context)
            result.scores[key] = bool(verdict["score"])
            result.reasons[key] = verdict.get("comment") or ""
        except (KeyError, ValueError, TypeError) as e:  # a reply without the verdict's fields
            result.scores[key] = None
            result.reasons[key] = f"judge failed: {e!r}"
    return result


def summarize(results: list[AnswerResult]) -> dict[str, tuple[float, int]]:
    """(share passing, how many were scored) for each check. Answers the
    judge failed to score are left out of that check's share."""
    summary = {}
    for key in (*CHECKED, *JUDGED):
        scored = [r.scores[key] for r in results if r.scores.get(key) is not None]
        if scored:
            summary[key] = (sum(scored) / len(scored), len(scored))
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    from docgen.agent import ask_agent
    from docgen.ask import CONTEXT_CHARS, ask, make_ask_completer
    from docgen.evaluate import load_eval
    from docgen.llm import ProviderError
    from docgen.repo_index import update_index
    from docgen.repos import RepoError, resolve_repo

    parser = argparse.ArgumentParser(prog="python -m docgen.answer_eval", description=__doc__.splitlines()[0])
    parser.add_argument("eval_file")
    parser.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic", help="who answers")
    parser.add_argument("--model", help="the answering model")
    parser.add_argument("--agent", action="store_true", help="answer in Agentic mode instead of Quick")
    parser.add_argument("--judge-provider", choices=["anthropic", "ollama"], help="who judges (default: --provider)")
    parser.add_argument("--judge-model", help=f"the judging model (default {JUDGE_MODELS})")
    parser.add_argument("--no-judge", action="store_true", help="only the checks that need no judge model")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--out", help="save every answer, score and judge's reason to this JSON file")
    args = parser.parse_args(argv)

    repo_text, questions = load_eval(args.eval_file)
    questions = questions[: args.limit] if args.limit else questions
    try:
        repo = resolve_repo(repo_text).path
    except RepoError as e:
        print(e)
        return 1
    update_index(repo)
    complete, model, meter = make_ask_completer(args.provider, args.model)
    budget = CONTEXT_CHARS[args.provider]

    def answerer(question: str) -> tuple[Answer, list[str]]:
        if args.agent:
            return ask_agent(repo, question, complete, model, meter, budget=budget)
        return ask(repo, question, complete, model, meter, budget=budget), []

    judge_provider = args.judge_provider or args.provider
    judges = {} if args.no_judge else make_judges(judge_provider, args.judge_model)
    mode = "Agentic" if args.agent else "Quick"
    judged_by = "no judge" if args.no_judge else \
        f"judged by {args.judge_model or JUDGE_MODELS[judge_provider]} (uncalibrated)"
    print(f"{len(questions)} questions, {mode} mode, answered by {model}, {judged_by}\n")
    results = []
    try:
        for n, q in enumerate(questions, 1):
            r = grade(q, answerer, judges)
            results.append(r)
            marks = " ".join(f"{k}={'?' if v is None else '✓' if v else '✗'}" for k, v in r.scores.items())
            print(f"[{n}/{len(questions)}] {r.seconds:.0f}s  {marks}\n    {q['question']}")
            if args.out:  # saved after every question, so a stopped run keeps what it has
                Path(args.out).write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
    except ProviderError as e:
        print(f"Stopped: {e}")
        if not results:
            return 1
    print()
    for key, (share, scored) in summarize(results).items():
        note = "" if scored == len(results) else f"  (judge failed on {len(results) - scored})"
        print(f"{key:<20} {share:>5.0%}{note}")
    print(f"{'median seconds':<20} {sorted(r.seconds for r in results)[len(results) // 2]:>5.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
