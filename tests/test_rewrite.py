"""Tests for query rewriting and the eval's rewrite modes. The model is a fake."""

import json
import shutil

import pytest

from docgen import evaluate as evalmod
from docgen.evaluate import evaluate
from docgen.llm import GenerationError
from docgen.repo_index import update_index
from docgen.rewrite import build_prompt, parse_terms, rewrite_query
from test_hub import MULTILANG


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    update_index(dest)
    return dest


class FakeModel:
    def __init__(self, reply="<terms>TaskStore add_task store</terms>"):
        self.reply = reply
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.reply


def test_prompt_lists_the_repo_files_and_the_question(repo):
    system, user = build_prompt(repo, "Where do tasks live?")
    assert "search terms" in system
    assert "backend/store.py" in user and "frontend/src/api.ts" in user
    assert user.index("</paths>") < user.index("<question>\nWhere do tasks live?\n</question>")


def test_parse_terms():
    assert parse_terms("<terms>a_b  cD\n e</terms>") == "a_b cD e"
    with pytest.raises(GenerationError):
        parse_terms("no tags here")


def test_rewrites_are_cached_per_model_and_question(repo):
    fake = FakeModel()
    assert rewrite_query(repo, "Where do tasks live?", fake, "m1") == "TaskStore add_task store"
    assert rewrite_query(repo, "Where do tasks live?", fake, "m1") == "TaskStore add_task store"
    assert len(fake.calls) == 1
    rewrite_query(repo, "Where do tasks live?", fake, "m2")
    rewrite_query(repo, "Something else?", fake, "m1")
    assert len(fake.calls) == 3
    cache = json.loads((repo / ".docgen" / "rewrites.json").read_text())
    assert {e["model"] for e in cache.values()} == {"m1", "m2"}


def test_rewriting_rescues_a_question_without_the_code_s_words(repo):
    questions = [{"question": "zzz qqq where do the to-dos sit?", "files": ["backend/store.py"]}]
    plain = evaluate(repo, questions, k=1, mode="keyword")
    rewritten = evaluate(repo, questions, k=1, mode="rw-keyword",
                         rewrite=lambda q: "asdict next_id")
    assert plain.questions[0].rank is None
    assert rewritten.questions[0].rank == 1 and rewritten.mode == "rw-keyword"


def test_rewrite_modes_need_a_rewriter(repo):
    with pytest.raises(ValueError, match="needs a rewrite"):
        evaluate(repo, [{"question": "x", "files": ["a"]}], mode="rw-hybrid")


def test_eval_command_with_rewrite(repo, tmp_path, monkeypatch, capsys):
    f = tmp_path / "q.json"
    f.write_text(json.dumps({"repo": str(repo), "questions": [
        {"question": "where do the to-dos sit?", "files": ["backend/store.py"]}]}))
    monkeypatch.setattr(evalmod, "_make_rewriter", lambda repo, provider, model: (
        lambda q: "TaskStore", lambda: 0.0012))
    assert evalmod.main([str(f), "--rewrite", "--misses"]) == 0
    out = capsys.readouterr().out
    assert "rw-keyword" in out and "rw-hybrid" in out
    assert "Rewriting cost $0.0012." in out
    assert "-> TaskStore" in out


def test_a_bad_rewrite_reply_falls_back_to_the_plain_question(repo, monkeypatch):
    import docgen.llm as llm

    monkeypatch.setattr(llm, "make_ollama_completer", lambda model, max_tokens: FakeModel("no tags"))
    rewrite, cost = evalmod._make_rewriter(repo, "ollama", None)
    assert rewrite("anything") == "" and cost() == 0.0
