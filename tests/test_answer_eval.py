"""Tests for the answer evals. The answerer and the judges are fakes, so these
check what gets scored and how, not judge quality."""

from docgen.answer_eval import grade, summarize
from docgen.ask import Answer
from docgen.search import SearchHit


def hit(path, text="def f(): pass"):
    return SearchHit(id=f"code:{path}:1-2", type="code", path=path, start_line=1, end_line=2,
                     label="", text=text, score=1.0, vector_rank=1, keyword_rank=None)


def answerer(cited):
    sources = [hit("backend/store.py", "class TaskStore: ..."), hit("frontend/api.ts")]
    return lambda question: (Answer("Tasks live in TaskStore [1].", sources, cited), ["retrieve"])


class FakeJudge:
    def __init__(self, score):
        self.score, self.calls = score, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"key": "x", "score": self.score, "comment": "because"}


QUESTION = {"question": "Where are tasks stored?", "files": ["backend/store.py"]}


def test_grade_checks_citations_against_the_labelled_files():
    r = grade(QUESTION, answerer([1]), {})
    assert r.scores == {"cites": True, "right_file_cited": True}
    assert r.cited == ["backend/store.py:1-2"] and r.steps == ["retrieve"]
    assert grade(QUESTION, answerer([2]), {}).scores == {"cites": True, "right_file_cited": False}
    assert grade(QUESTION, answerer([]), {}).scores == {"cites": False, "right_file_cited": False}


def test_judges_see_the_question_the_answer_and_numbered_sources():
    judge = FakeJudge(True)
    r = grade(QUESTION, answerer([1]), {"groundedness": judge})
    sent = judge.calls[0]
    assert sent["inputs"] == "Where are tasks stored?" and sent["outputs"].startswith("Tasks live")
    assert sent["context"].startswith("[1] backend/store.py:1-2\nclass TaskStore")
    assert "[2] frontend/api.ts:1-2" in sent["context"]
    assert r.scores["groundedness"] is True and r.reasons["groundedness"] == "because"


def test_summary_is_the_share_passing_each_check():
    results = [grade(QUESTION, answerer([1]), {"groundedness": FakeJudge(True)}),
               grade(QUESTION, answerer([]), {"groundedness": FakeJudge(False)})]
    assert summarize(results) == {"cites": (0.5, 2), "right_file_cited": (0.5, 2), "groundedness": (0.5, 2)}


def test_a_verdict_without_a_score_is_recorded_not_fatal():
    def broken(**kwargs):
        raise KeyError("score")
    r = grade(QUESTION, answerer([1]), {"groundedness": broken, "helpfulness": FakeJudge(True)})
    assert r.scores["groundedness"] is None and "judge failed" in r.reasons["groundedness"]
    assert summarize([r])["helpfulness"] == (1.0, 1) and "groundedness" not in summarize([r])
