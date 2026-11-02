"""Tests for the search index. Uses the offline hashing embedder, so these check
what gets stored and when, not how good the search is."""

import shutil

import pytest
from test_hub import MULTILANG, FakeModel, models_for

from docgen import repo_index
from docgen.hub import build_hub
from docgen.repo_index import index_status, load_index, load_manifest, update_index
from docgen.search import search


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    return dest


def entries(repo, **where):
    stored = load_index(repo)
    return {i: m for i, m in zip(stored.ids, stored.metadatas, strict=True)
            if all(m.get(k) == v for k, v in where.items())}


def test_first_run_indexes_every_file(repo):
    stats = update_index(repo)
    assert not stats.rebuilt
    assert stats.files_indexed == 8 and stats.files_unchanged == 0
    assert stats.summaries_indexed == 0  # no hub yet
    code = entries(repo, type="code")
    assert stats.total_entries == len(code) == stats.by_type["code"]
    paths = {m["path"] for m in code.values()}
    assert {"backend/store.py", "frontend/src/api.ts", "README.md", "Dockerfile"} <= paths
    assert "frontend/package-lock.json" not in paths  # the walker's skip rules apply
    store = [m for m in code.values() if m["path"] == "backend/store.py"]
    assert any(m["unit"] == "class" and m["symbol"] == "TaskStore" for m in store)
    assert all(m["start_line"] >= 1 and m["end_line"] >= m["start_line"] for m in code.values())


def test_second_run_with_no_changes_embeds_nothing(repo, monkeypatch):
    update_index(repo)
    stats = update_index(repo)
    assert stats.files_indexed == 0 and stats.files_unchanged == 8
    assert stats.entries_added == 0 and stats.entries_removed == 0


def test_only_changed_and_removed_files_are_redone(repo):
    update_index(repo)
    before = entries(repo)
    store = repo / "backend" / "store.py"
    store.write_text(store.read_text() + "\n\ndef extra():\n    return 1\n")
    (repo / "Dockerfile").unlink()

    stats = update_index(repo)
    assert stats.files_indexed == 1 and stats.files_removed == 1 and stats.files_unchanged == 6
    after = entries(repo)
    assert not any(m["path"] == "Dockerfile" for m in after.values())
    assert any(m.get("symbol") == "extra" for m in after.values())
    untouched = {i for i, m in before.items() if m["path"] not in ("backend/store.py", "Dockerfile")}
    assert untouched <= set(after)
    assert stats.total_entries == len(after) == sum(stats.by_type.values())


def test_hub_summaries_are_indexed_as_their_own_type(repo):
    build_hub(repo, models_for(FakeModel()))
    stats = update_index(repo)
    assert stats.summaries_indexed == 1 + 4 + 1  # store.py, four folders, the overview
    summaries = entries(repo, type="summary")
    assert {m["level"] for m in summaries.values()} == {"file", "dir", "overview"}
    assert any(i.startswith("summary:file:backend/store.py#") for i in summaries)
    overview = [m for m in summaries.values() if m["level"] == "overview"]
    assert overview and overview[0]["section"] == "What it does"
    stored = load_index(repo)
    text = stored.texts[stored.ids.index("summary:file:backend/store.py#0")]
    assert text.startswith("Summary of the file backend/store.py")

    # Rebuilding the hub with new text re-indexes just those summaries.
    hub_file = repo / ".docgen" / "hub.json"
    hub_file.write_text(hub_file.read_text().replace("About backend/store.py.", "Stores tasks in memory."))
    stats = update_index(repo)
    assert stats.summaries_indexed == 1 and stats.files_indexed == 0


def test_summaries_of_deleted_files_are_dropped(repo):
    build_hub(repo, models_for(FakeModel()))
    update_index(repo)
    (repo / "backend" / "store.py").unlink()
    stats = update_index(repo)
    assert stats.summaries_removed == 1
    assert not any(i.startswith("summary:file:backend/store.py") for i in entries(repo))


def test_changing_the_embedding_model_rebuilds(repo, monkeypatch):
    update_index(repo)
    monkeypatch.setattr(repo_index, "embedding_name", lambda: "another-model")
    stats = update_index(repo)
    assert stats.rebuilt and stats.files_indexed == 8
    assert load_manifest(repo)["embedding"] == "another-model"


def test_missing_index_with_a_manifest_rebuilds(repo):
    update_index(repo)
    shutil.rmtree(repo / repo_index.INDEX_DIRNAME)
    stats = update_index(repo)
    assert stats.rebuilt and stats.files_indexed == 8


def test_status_reports_what_changed_without_embedding(repo):
    assert index_status(repo)["indexed"] is False
    update_index(repo)
    assert index_status(repo) == {"indexed": True, "files_changed": 0, "summaries_changed": 0,
                                  "needs_rebuild": False, "entries": load_entries_count(repo)}
    (repo / "README.md").write_text("changed\n")
    build_hub(repo, models_for(FakeModel()))
    status = index_status(repo)
    assert status["files_changed"] == 1 and status["summaries_changed"] == 6


def load_entries_count(repo):
    manifest = load_manifest(repo)
    return sum(len(r["ids"]) for section in ("files", "summaries") for r in manifest[section].values())


def test_progress_is_reported(repo):
    seen = []
    update_index(repo, on_progress=lambda done, total: seen.append((done, total)))
    assert seen[-1][0] == seen[-1][1] > 0


def test_query_finds_matching_code(repo):
    update_index(repo)
    hits = search(repo, "TaskStore add task", k=3, mode="vector")
    assert any(h.path == "backend/store.py" for h in hits)


def test_web_endpoint_updates_and_reports(repo):
    from fastapi.testclient import TestClient

    from docgen import web

    client = TestClient(web.app)
    data = client.post("/api/search-index", json={"repo": str(repo)}).json()
    assert data["files_indexed"] == 8 and data["status"]["files_changed"] == 0
    hub = client.get("/api/hub", params={"repo": str(repo)}).json()
    assert hub["search_index"]["indexed"] is True
