"""Tests for answering questions from the search index. The model is a fake
that records its prompt, so these check retrieval, prompting and citation
handling, not answer quality."""

import shutil

import pytest
from fastapi.testclient import TestClient

from docgen import web
from docgen.ask import ASK_MODEL, Answer, ask, build_prompt, make_ask_completer, parse_answer
from docgen.llm import ProviderError, UsageMeter
from docgen.repo_index import update_index
from docgen.search import SearchHit
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
    def __init__(self, reply="<answer>Tasks live in `TaskStore` [1], served by the API [2][2].</answer>"):
        self.reply = reply
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append((system, user))
        return self.reply


def hit(n, text="x = 1", **extra):
    fields = dict(id=f"code:f{n}.py:1-2", type="code", path=f"f{n}.py", start_line=1, end_line=2,
                  label="", text=text, score=1.0, vector_rank=n, keyword_rank=None)
    fields.update(extra)
    return SearchHit(**fields)


# ------------------------------------------------------------------ pieces


def test_parse_answer_finds_citations_and_ignores_code():
    text, cited, unknown = parse_answer(
        "<answer>See [1] and [3][7]. `items[2]` is an index, so is\n```\nx[4]\n```\nand [a](b) a link.</answer>", 5)
    assert text.startswith("See [1]")
    assert cited == [1, 3] and unknown == [7]


def test_parse_answer_without_tags_uses_the_whole_reply():
    assert parse_answer("Just text [1].", 1) == ("Just text [1].", [1], [])
    assert parse_answer("<answer>Cut off mid-way [2]", 2)[0] == "Cut off mid-way [2]"


def test_prompt_numbers_sources_and_keeps_the_question_last():
    system, user, used = build_prompt("How does it work?", [hit(1), hit(2, label='Say "hi"')], 10_000)
    assert "only the numbered sources" in system
    assert '<source n="1" citation="f1.py:1-2" type="code">' in user
    assert "label=\"Say 'hi'\"" in user  # quotes can't break the attribute
    assert user.index("</sources>") < user.index("<question>\nHow does it work?\n</question>")
    assert used == [hit(1), hit(2, label='Say "hi"')]


def test_prompt_stops_adding_sources_at_the_budget_but_keeps_the_first():
    big = "y\n" * 3_000
    _, user, used = build_prompt("q", [hit(1, big), hit(2, big), hit(3)], 5_000)
    assert len(used) == 1
    assert "more characters]" in user  # the first source was cut to SOURCE_CHARS


def test_make_ask_completer_defaults(monkeypatch):
    import anthropic

    created = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.messages = self

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    _, model, meter = make_ask_completer("anthropic", api_key="sk-test")
    assert model == ASK_MODEL and isinstance(meter, UsageMeter)
    assert created == {"api_key": "sk-test"}
    _, model, meter = make_ask_completer("ollama")
    assert model == "qwen2.5-coder:7b" and meter is None
    with pytest.raises(ValueError):
        make_ask_completer("gpt")


# --------------------------------------------------------------------- ask


def test_ask_retrieves_then_answers_with_citations(repo):
    fake = FakeModel()
    meter = UsageMeter()
    meter.add(10_000, 1_000)
    answer = ask(repo, "Where are tasks stored? TaskStore", fake, "claude-sonnet-5-5", meter, k=5)
    assert isinstance(answer, Answer)
    assert answer.cited == [1, 2] and answer.unknown_citations == []
    assert answer.sources[0].path == "backend/store.py"
    _, user = fake.prompts[0]
    assert "backend/store.py" in user and "Where are tasks stored?" in user
    assert answer.cost_usd == pytest.approx((10_000 * 2 + 1_000 * 10) / 1e6)


def test_ask_reports_citations_to_missing_sources(repo):
    answer = ask(repo, "TaskStore", FakeModel("<answer>Yes [1][40].</answer>"), k=3)
    assert answer.cited == [1] and answer.unknown_citations == [40]


def test_ask_with_an_empty_index_does_not_call_the_model(tmp_path):
    (tmp_path / "blank.txt").write_text("\n")
    update_index(tmp_path)
    fake = FakeModel()
    answer = ask(tmp_path, "anything", fake)
    assert "empty" in answer.text and fake.prompts == []


# --------------------------------------------------------------------- web


@pytest.fixture
def client(monkeypatch):
    fake = FakeModel()
    meter = UsageMeter()
    monkeypatch.setattr(web, "make_ask_completer", lambda provider, model, api_key: (fake, model or "fake", meter))
    c = TestClient(web.app)
    c.fake = fake
    return c


def post(client, repo, **extra):
    body = {"repo": str(repo), "question": "Where are tasks stored? TaskStore", "api_key": "sk-test", **extra}
    return client.post("/api/ask", json=body)


def test_web_ask_returns_answer_sources_and_citations(client, repo):
    data = post(client, repo, k=4).json()
    assert data["answer"].startswith("Tasks live in")
    assert len(data["sources"]) == 4 and data["cited"] == [1, 2]
    first = data["sources"][0]
    assert first["citation"] == f"backend/store.py:{first['start_line']}-{first['end_line']}"
    assert "vector_rank" in first and "keyword_rank" in first


def test_web_ask_updates_the_index_first(client, repo):
    (repo / "backend" / "zoo.py").write_text("def zebrafish_feeder():\n    return 'TaskStore'\n")
    data = post(client, repo, question="zebrafish feeder", mode="keyword").json()
    assert data["sources"][0]["path"] == "backend/zoo.py"


@pytest.mark.parametrize("extra, message", [
    ({"question": "  "}, "Type a question"),
    ({"mode": "magic"}, "Unknown search mode"),
    ({"api_key": None}, "API key"),
    ({"provider": "gpt"}, "Unknown provider"),
])
def test_web_ask_rejects_bad_requests(client, repo, extra, message):
    r = post(client, repo, **extra)
    assert r.status_code == 400 and message in r.json()["detail"]


def test_web_ask_needs_an_index(client, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    r = post(client, tmp_path)
    assert r.status_code == 400 and "search index" in r.json()["detail"]


def test_web_ask_reports_provider_errors(client, repo, monkeypatch):
    def broken(system, user):
        raise ProviderError("The model declined to answer this request.")

    monkeypatch.setattr(web, "make_ask_completer", lambda *a: (broken, "fake", None))
    r = post(client, repo)
    assert r.status_code == 502 and "declined" in r.json()["detail"]
