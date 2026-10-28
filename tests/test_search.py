"""Tests for hybrid search and the eval command. Uses the offline hashing
embedder, so the vector half here is lexical too; these check the mechanics
(ranking, fusion, filters, scoring), not real search quality."""

import json
import math
import shutil
from pathlib import Path

import pytest
from test_hub import MULTILANG, FakeModel, models_for

from docgen import search as searchmod
from docgen.evaluate import evaluate, load_eval, parse_chunk_label
from docgen.evaluate import main as eval_main
from docgen.hub import build_hub
from docgen.repo_index import update_index
from docgen.search import RepoRetriever, keyword_index, keyword_scores, rrf, search, tokenize

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    update_index(dest)
    return dest


# ------------------------------------------------------------------ pieces


def test_tokenize_splits_identifiers_and_keeps_them_whole():
    assert tokenize("make_hub_models") == ["make_hub_model", "make", "hub", "model"]
    assert tokenize("getTaskById") == ["gettaskbyid", "get", "task", "id"]  # "by" is a stopword
    assert tokenize("How are the repos cloned?") == ["repo", "clon"]


def test_stemming_matches_word_forms():
    assert tokenize("cloned")[0] == tokenize("clones")[0] == tokenize("cloning")[0]
    assert tokenize("is") == []  # stopword
    assert tokenize("bus") == ["bus"]  # too short to strip


def test_bm25_prefers_rare_words_and_short_documents():
    docs = [["the", "store", "task"], ["the", "api", "client"], ["the"] * 20 + ["store"]]
    index = keyword_index(docs)
    scores = keyword_scores(index, ["store"])
    assert scores[0] > scores[2] > 0 and scores[1] == 0  # same term, shorter doc wins
    # A word in every document still scores above zero ("+1" IDF), but less than a rare one.
    assert 0 < keyword_scores(index, ["the"])[1] < keyword_scores(index, ["api"])[1]


def test_bm25_counts_a_repeated_query_word_once():
    index = keyword_index([["store"], ["api"]])
    assert list(keyword_scores(index, ["store", "store"])) == list(keyword_scores(index, ["store"]))


def test_bm25_scores_a_query_with_no_words_as_zero():
    index = keyword_index([["store"], ["api"]])
    assert list(keyword_scores(index, [])) == [0, 0]
    assert list(keyword_scores(index, ["zebra"])) == [0, 0]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]])
    assert max(fused, key=fused.get) == "b"  # 2nd and 1st beats 1st and absent
    assert math.isclose(fused["a"], 1 / 61)
    assert math.isclose(fused["b"], 1 / 62 + 1 / 61)


# ------------------------------------------------------------------ search


def test_search_returns_cited_hits_with_both_ranks(repo):
    hits = search(repo, "TaskStore add task", k=5)
    assert hits and hits[0].path == "backend/store.py"
    top = hits[0]
    assert top.type == "code" and top.citation == f"backend/store.py:{top.start_line}-{top.end_line}"
    assert top.vector_rank is not None and top.keyword_rank is not None
    assert not top.text.startswith("File:")  # the embedding header is stripped
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.parametrize("mode", ["vector", "keyword"])
def test_single_modes_only_use_their_half(repo, mode):
    hits = search(repo, "TaskStore", k=3, mode=mode)
    assert hits
    if mode == "vector":
        assert all(h.keyword_rank is None for h in hits)
    else:
        assert all(h.vector_rank is None for h in hits)


def test_unknown_mode_is_an_error(repo):
    with pytest.raises(ValueError, match="Unknown search mode"):
        search(repo, "x", mode="magic")


def test_type_filter_and_excluded_paths(repo):
    build_hub(repo, models_for(FakeModel()))
    update_index(repo)
    summaries = search(repo, "backend store", k=5, types=("summary",))
    assert summaries and all(h.type == "summary" for h in summaries)
    assert summaries[0].citation.endswith("(summary)")
    hits = search(repo, "TaskStore", k=5, exclude_paths=frozenset({"backend/store.py"}))
    assert all(h.path != "backend/store.py" for h in hits)


def test_keyword_index_is_rebuilt_after_the_index_changes(repo):
    assert not search(repo, "zebrafish", mode="keyword")
    (repo / "backend" / "zoo.py").write_text("def zebrafish():\n    return 1\n")
    update_index(repo)
    hits = search(repo, "zebrafish", mode="keyword")
    assert hits and hits[0].path == "backend/zoo.py"


# -------------------------------------------------------------------- eval


def write_eval(path, repo, questions):
    path.write_text(json.dumps({"repo": str(repo), "questions": questions}))
    return path


def test_evaluate_scores_hits_and_misses(repo):
    questions = [
        {"question": "TaskStore add task", "files": ["backend/store.py"]},
        {"question": "zebrafish aquarium", "files": ["backend/store.py"]},
    ]
    result = evaluate(repo, questions, k=3, mode="hybrid")
    assert result.questions[0].rank == 1
    assert result.questions[1].rank is None
    assert result.hit_rate == 0.5 and result.mrr == 0.5


def test_the_eval_file_inside_the_repo_is_not_counted(repo):
    eval_file = write_eval(repo / "questions.json", ".", [
        {"question": "zebrafish aquarium penguin", "files": ["questions.json"]},
    ])
    eval_file.write_text(eval_file.read_text().replace(str(repo), "."))
    update_index(repo)
    with_file = evaluate(repo, load_eval(eval_file)[1], k=3)
    without = evaluate(repo, load_eval(eval_file)[1], k=3, eval_file=eval_file)
    assert with_file.questions[0].rank == 1  # it "answers" its own question
    assert without.questions[0].rank is None


def test_load_eval_resolves_the_repo_against_the_eval_file(tmp_path):
    (tmp_path / "evals").mkdir()
    f = tmp_path / "evals" / "q.json"
    f.write_text(json.dumps({"repo": "..", "questions": []}))
    assert load_eval(f)[0] == str(tmp_path.resolve())
    f.write_text(json.dumps({"repo": "https://github.com/o/r", "questions": []}))
    assert load_eval(f)[0] == "https://github.com/o/r"


def test_evaluate_grades_chunks_as_well_as_files(repo):
    questions = [
        {"question": "TaskStore add task", "files": ["backend/store.py"],
         "chunks": ["backend/store.py :: def add"]},
        {"question": "TaskStore add task", "files": ["backend/store.py"],
         "chunks": ["backend/store.py :: text that is in no chunk"]},
        {"question": "TaskStore add task", "files": ["backend/store.py"]},  # no chunk label
    ]
    result = evaluate(repo, questions, k=3, mode="hybrid")
    assert [q.rank for q in result.questions] == [1, 1, 1]
    assert result.questions[0].chunk_rank is not None
    assert result.questions[1].chunk_rank is None  # right file, wrong chunk
    assert len(result.labelled) == 2 and result.chunk_hit_rate == 0.5


def test_chunk_labels_need_a_path_and_text():
    assert parse_chunk_label("a/b.py :: def f(x)") == ("a/b.py", "def f(x)")
    for bad in ("a/b.py", "a/b.py ::", " :: def f"):
        with pytest.raises(ValueError):
            parse_chunk_label(bad)


def test_eval_command_prints_a_table(repo, tmp_path, capsys):
    f = write_eval(tmp_path / "q.json", repo, [{"question": "TaskStore", "files": ["backend/store.py"]}])
    assert eval_main([str(f), "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert "hybrid" in out and "vector" in out and "keyword" in out and "100%" in out
    assert "chunk" not in out  # no chunk labels, no chunk columns


def test_eval_command_adds_chunk_columns(repo, tmp_path, capsys):
    f = write_eval(tmp_path / "q.json", repo, [
        {"question": "TaskStore", "files": ["backend/store.py"], "chunks": ["backend/store.py :: class TaskStore"]},
    ])
    assert eval_main([str(f), "--k", "3", "--mode", "hybrid"]) == 0
    assert "chunk hit" in capsys.readouterr().out


def test_docgen_eval_set_is_well_formed():
    repo, questions = load_eval(ROOT / "evals" / "docgen.json")
    assert Path(repo) == ROOT and len(questions) >= 50
    for q in questions:
        assert q["question"].strip() and q["files"]
        for f in q["files"]:
            assert (ROOT / f).is_file(), f
        for label in q.get("chunks", []):
            path, text = parse_chunk_label(label)
            assert path in q["files"], label
            assert text in (ROOT / path).read_text(encoding="utf-8"), label


def test_web_search_endpoint(repo):
    from fastapi.testclient import TestClient

    from docgen import web

    client = TestClient(web.app)
    data = client.get("/api/search", params={"repo": str(repo), "q": "TaskStore add task", "k": 3}).json()
    assert data["hits"][0]["path"] == "backend/store.py"
    assert data["hits"][0]["citation"].startswith("backend/store.py:")
    r = client.get("/api/search", params={"repo": str(repo), "q": "x", "mode": "magic"})
    assert r.status_code == 400


def test_web_search_needs_an_index(tmp_path):
    from fastapi.testclient import TestClient

    from docgen import web

    (tmp_path / "a.py").write_text("x = 1\n")
    r = TestClient(web.app).get("/api/search", params={"repo": str(tmp_path), "q": "x"})
    assert r.status_code == 400 and "search index" in r.json()["detail"]


def test_a_question_of_only_stopwords_still_gets_vector_results(repo):
    hits = search(repo, "how is it?", k=3)
    assert hits and all(h.keyword_rank is None for h in hits)


def test_repo_retriever_returns_documents_with_citations(repo):
    docs = RepoRetriever(repo_path=str(repo), k=3).invoke("TaskStore add task")
    assert len(docs) == 3
    first = docs[0]
    assert first.metadata["path"] == "backend/store.py"
    assert first.metadata["citation"].startswith("backend/store.py:")
    assert "text" not in first.metadata and "class TaskStore" in first.page_content
    assert first.id == search(repo, "TaskStore add task", k=3)[0].id


def test_corpus_cache_is_per_repo(repo, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "solo.py").write_text("def penguin():\n    return 1\n")
    update_index(other)
    assert search(other, "penguin", mode="keyword")[0].path == "solo.py"
    assert not search(repo, "penguin", mode="keyword")
    assert str(repo.resolve()) in searchmod._corpora


def test_weighted_rrf_and_alpha():
    assert rrf([["a", "b"], ["b", "c"]], weights=[2.0, 0.0]) == pytest.approx({"a": 2 / 61, "b": 2 / 62, "c": 0.0})
    # Equal weights of 1.0 are plain RRF.
    assert rrf([["a"], ["a"]], weights=[1.0, 1.0]) == rrf([["a"], ["a"]])


def test_alpha_shifts_hybrid_toward_one_half(repo):
    plain = search(repo, "TaskStore add task", k=5)
    assert [h.id for h in search(repo, "TaskStore add task", k=5, alpha=0.5)] == [h.id for h in plain]
    vector_only = [h.id for h in search(repo, "TaskStore add task", k=5, mode="vector")]
    assert [h.id for h in search(repo, "TaskStore add task", k=5, alpha=1.0)][:len(vector_only)] == vector_only
    with pytest.raises(ValueError, match="alpha"):
        search(repo, "x", alpha=1.5)
