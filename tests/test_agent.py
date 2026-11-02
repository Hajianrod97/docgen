"""Tests for the reranker and the LangGraph agent. The reranker is a fake that
returns chosen scores, and the model a fake with scripted replies, so these
check the graph's routing (when it rewrites, retries, gives up), not quality."""

import shutil

import pytest
from test_hub import MULTILANG

from docgen import agent as agentmod
from docgen.agent import MAX_REWRITES, REWRITE_SYSTEM, ask_agent
from docgen.repo_index import update_index
from docgen.rerank import _HashScorer, get_scorer, rerank
from docgen.search import SearchHit


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")
    monkeypatch.setenv("DOCGEN_RERANKER", "hash")


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    update_index(dest)
    return dest


class Scripted:
    """A fake model: rewrite prompts get the next query, answer prompts the next answer."""

    def __init__(self, answers=("<answer>Tasks live in TaskStore [1].</answer>",), queries=("TaskStore",)):
        self.answers, self.queries = list(answers), list(queries)
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append((system, user))
        if system == REWRITE_SYSTEM:
            return f"<terms>{self.queries.pop(0) if len(self.queries) > 1 else self.queries[0]}</terms>"
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def scores(*values):
    """A fake reranker scoring every hit of the n-th search values[n] (the last repeats)."""
    calls = []

    def scorer(question, texts):
        value = values[min(len(calls), len(values) - 1)]
        calls.append(question)
        return [value - i * 0.01 for i in range(len(texts))]
    return scorer


def hit(n, text="x"):
    return SearchHit(id=f"code:f{n}.py:1-2", type="code", path=f"f{n}.py", start_line=1, end_line=2,
                     label="", text=text, score=1.0, vector_rank=n, keyword_rank=None)


# -------------------------------------------------------------- reranker


def test_rerank_orders_by_score_and_records_it():
    hits = [hit(1, "nothing here"), hit(2, "task store adds tasks")]
    ranked = rerank("how does the task store add tasks", hits, _HashScorer())
    assert [h.id for h in ranked] == ["code:f2.py:1-2", "code:f1.py:1-2"]
    assert ranked[0].rerank_score > ranked[1].rerank_score
    assert hits[0].rerank_score is None  # the originals are left alone
    assert rerank("anything", []) == []


def test_reranker_setting(monkeypatch):
    assert isinstance(get_scorer(), _HashScorer)
    monkeypatch.delenv("DOCGEN_RERANKER")
    assert get_scorer().model == "Xenova/ms-marco-MiniLM-L-12-v2"


# ------------------------------------------------------------------ agent


def test_relevant_sources_go_straight_to_an_answer(repo):
    model = Scripted()
    answer, steps = ask_agent(repo, "Where are tasks stored?", model, scorer=scores(5.0))
    assert [s.split(" ")[0] for s in steps] == ["retrieve", "generate:"]
    assert answer.cited == [1] and answer.sources
    assert all(system != REWRITE_SYSTEM for system, _ in model.prompts)


def test_weak_sources_get_a_rewritten_query(repo):
    model = Scripted(queries=("TaskStore add",))
    answer, steps = ask_agent(repo, "Where do the to-dos go?", model, scorer=scores(-5.0, 5.0))
    assert [s.split(" ")[0] for s in steps] == ["retrieve", "rewrite:", "retrieve", "generate:"]
    assert "'Where do the to-dos go? TaskStore add'" in steps[2]  # the question plus the guessed terms
    rewrite_prompt = next(user for system, user in model.prompts if system == REWRITE_SYSTEM)
    assert "Where do the to-dos go?" in rewrite_prompt and "<found>" in rewrite_prompt
    assert "backend/store.py" in rewrite_prompt  # it sees the repo's file paths


def test_rewrites_stop_after_the_limit_and_answer_from_the_best_sources(repo):
    model = Scripted(queries=("first try", "second try"))
    answer, steps = ask_agent(repo, "Something vague", model, scorer=scores(-5.0, -1.0, -3.0), relevant_score=0.0)
    assert sum(s.startswith("retrieve") for s in steps) == MAX_REWRITES + 1
    assert steps[-1].startswith("generate")
    # The second search scored best (-1.0), so its sources are the ones used.
    assert "-1.00" in [s for s in steps if s.startswith("retrieve")][1]
    assert answer.sources


def test_a_rewrite_with_no_terms_answers_instead_of_searching_again(repo):
    model = Scripted(queries=("",))
    answer, steps = ask_agent(repo, "Something vague", model, scorer=scores(-5.0))
    assert [s.split(" ")[0] for s in steps] == ["retrieve", "rewrite:", "generate:"]
    assert answer.sources


def test_an_answer_without_citations_gets_one_retry(repo):
    model = Scripted(answers=("<answer>Somewhere.</answer>", "<answer>In TaskStore [1].</answer>"))
    answer, steps = ask_agent(repo, "Where are tasks stored?", model, scorer=scores(5.0))
    assert [s.split(" ")[0] for s in steps] == ["retrieve", "generate:", "generate"]
    assert answer.cited == [1]
    assert "cited no sources" in model.prompts[-1][1]


def test_bad_citations_are_retried_only_once(repo):
    model = Scripted(answers=("<answer>See [99].</answer>",))
    answer, steps = ask_agent(repo, "Where are tasks stored?", model, scorer=scores(5.0))
    assert sum(s.startswith("generate") for s in steps) == 2
    assert answer.unknown_citations == [99]
    assert "[99]" in model.prompts[-1][1]


def test_no_sources_at_all_gives_up_without_answering(repo, monkeypatch):
    monkeypatch.setattr(agentmod, "search", lambda *a, **kw: [])
    model = Scripted()
    answer, steps = ask_agent(repo, "Anything?", model, scorer=scores(5.0))
    assert steps[-1].startswith("give up") and answer.sources == []
    assert "couldn't find" in answer.text
    assert all(system == REWRITE_SYSTEM for system, _ in model.prompts)  # never asked to answer
