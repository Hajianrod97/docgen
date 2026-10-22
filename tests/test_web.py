"""Tests for the web UI backend (docgen/web.py).

Uses FastAPI's TestClient, which calls the ASGI app in-process: no real HTTP
server, no real model. This is what would have caught the path-resolution bug
found by hand-testing (GET /api/plan and POST /api/apply 500ing whenever the
repo path was given as relative, e.g. "example_repo" instead of an absolute
path) before it ever reached a browser.
"""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgen import web
from docgen.generator import ProviderError, ProviderSetupError
from docgen.parser import parse_repo

EXAMPLE = Path(__file__).resolve().parent.parent / "example_repo"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(EXAMPLE, dest, ignore=shutil.ignore_patterns(".docgen", "__pycache__", "docs"))
    return dest


@pytest.fixture
def client():
    return TestClient(web.app)


def fake_complete(system, user):
    tag = "endpoint_doc" if "<endpoint_doc>" in user else "docstring"
    return f"<{tag}>Generated.</{tag}>"


@pytest.fixture
def fake_completer(monkeypatch):
    """Stand in for both providers so /api/generate never makes a network call."""
    def fake_get_completer(provider, model, api_key):
        if provider == "anthropic" and not api_key:
            from fastapi import HTTPException
            raise HTTPException(400, "Anthropic provider needs an API key.")
        if provider not in ("anthropic", "ollama"):
            from fastapi import HTTPException
            raise HTTPException(400, f"Unknown provider: {provider}")
        return fake_complete, (model or "fake-model")

    monkeypatch.setattr(web, "_get_completer", fake_get_completer)


# ------------------------------------------------------------------ serving


def test_root_serves_the_page(client):
    result = client.get("/")
    assert result.status_code == 200
    assert "docgen" in result.text


# ------------------------------------------------------------------ /scan


def test_scan_relative_and_absolute_paths_agree(client, repo, monkeypatch):
    """The bug: relative and absolute paths to the same repo must not diverge."""
    monkeypatch.chdir(repo.parent)
    rel = client.get("/api/scan", params={"repo": repo.name}).json()
    abs_ = client.get("/api/scan", params={"repo": str(repo)}).json()
    assert rel["indexed"] == abs_["indexed"] is False
    assert {c["id"] for c in rel["chunks"]} == {c["id"] for c in abs_["chunks"]}
    assert len(rel["chunks"]) == 9


def test_scan_reports_files_by_language_and_skips(client):
    multilang = EXAMPLE.parent / "example_multilang"
    files = client.get("/api/scan", params={"repo": str(multilang)}).json()["files"]
    assert files["languages"]["typescript"] == 3
    assert files["languages"]["python"] == 2
    assert files["count"] == sum(files["languages"].values())
    assert files["skipped"] == [{"path": "frontend/package-lock.json", "reason": "lockfile"}]


def test_scan_rejects_a_missing_path(client):
    result = client.get("/api/scan", params={"repo": "/definitely/not/a/real/path"})
    assert result.status_code == 400


def test_scan_shows_generated_and_stale_status(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    chunks = parse_repo(repo)
    target = next(c for c in chunks if c.name == "load_file")

    from docgen.generator import save_generated, source_hash

    save_generated(repo, {"model": "x", "items": {
        target.id: {"kind": "docstring", "text": "Doc.", "file_path": target.file_path,
                    "start_line": target.start_line, "source_hash": source_hash(target)},
    }})
    result = client.get("/api/scan", params={"repo": repo.name}).json()
    row = next(c for c in result["chunks"] if c["id"] == target.id)
    assert row["generated"] is True and row["stale"] is False

    (repo / "utils.py").write_text(
        (repo / "utils.py").read_text().replace("f.read()", "f.read().strip()")
    )
    row = next(c for c in client.get("/api/scan", params={"repo": repo.name}).json()["chunks"]
               if c["id"] == target.id)
    assert row["generated"] is False and row["stale"] is True


# ----------------------------------------------------------------- /index


def test_index_then_context_round_trip(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    result = client.post("/api/index", json={"repo": repo.name})
    assert result.status_code == 200
    assert result.json()["count"] == 9

    ctx = client.get(
        "/api/context", params={"repo": repo.name, "target": "api.py::reload_config"}
    ).json()
    assert ctx["chunk"]["name"] == "reload_config"
    reasons = [i["reason"] for i in ctx["items"]]
    assert "called by target" in reasons
    assert any("called by parse_config" in r for r in reasons)  # depth > 1 reaches the prompt
    assert "<endpoint_doc>" in ctx["prompt_preview"]


def test_index_on_an_empty_repo_is_a_clean_error_not_a_500(client, tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)
    result = client.post("/api/index", json={"repo": "empty"})
    assert result.status_code == 400


def test_context_without_an_index_is_a_clean_error(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    result = client.get(
        "/api/context", params={"repo": repo.name, "target": "api.py::reload_config"}
    )
    assert result.status_code == 400


def test_context_for_an_unknown_target_is_404(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    result = client.get("/api/context", params={"repo": repo.name, "target": "nope::nope"})
    assert result.status_code == 404


# -------------------------------------------------------------- /generate


def test_generate_one_target_saves_it(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    result = client.post("/api/generate", json={
        "repo": repo.name, "provider": "ollama", "target": "api.py::reload_config",
    })
    assert result.status_code == 200
    body = result.json()
    assert body["results"] == [{"id": "api.py::reload_config", "ok": True, "text": "Generated."}]

    from docgen.generator import load_generated
    saved = load_generated(repo)
    assert saved["items"]["api.py::reload_config"]["kind"] == "endpoint_doc"


def test_generate_all_respects_limit_and_skips_documented_chunks(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    result = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "limit": 3})
    body = result.json()
    assert len(body["results"]) == 3  # 8 undocumented chunks exist, but limit=3
    assert all(r["ok"] for r in body["results"])


def test_generate_skips_up_to_date_chunks_unless_forced(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "target": "api.py::reload_config"})

    again = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    ids = [r["id"] for r in again.json()["results"]]
    assert "api.py::reload_config" not in ids  # already up to date

    forced = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "target": "api.py::reload_config", "force": True})
    assert forced.json()["results"][0]["ok"] is True


def test_generate_all_saves_every_undocumented_chunk_and_resumes(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    calls = []

    def counting_get_completer(provider, model, api_key):
        def complete(system, user):
            calls.append(user)
            return fake_complete(system, user)
        return complete, "fake-model"

    monkeypatch.setattr(web, "_get_completer", counting_get_completer)
    client.post("/api/index", json={"repo": repo.name})
    result = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "limit": 0})
    assert result.status_code == 200

    from docgen.generator import load_generated
    saved = load_generated(repo)
    # 9 chunks, 1 already documented (normalize) -> 8 generated
    assert len(saved["items"]) == 8
    assert saved["items"]["api.py::reload_config"]["kind"] == "endpoint_doc"
    assert saved["items"]["utils.py::load_file"]["kind"] == "docstring"
    assert len(calls) == 8

    # Second run: nothing changed, so no new model calls.
    again = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "limit": 0})
    assert again.json()["results"] == []
    assert len(calls) == 8

    # Editing one function regenerates only that one.
    utils = repo / "utils.py"
    utils.write_text(utils.read_text().replace("f.read()", "f.read().lstrip()"))
    client.post("/api/index", json={"repo": repo.name})
    changed = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "limit": 0})
    assert [r["id"] for r in changed.json()["results"]] == ["utils.py::load_file"]
    assert len(calls) == 9


def test_generate_without_index_is_a_clean_error(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    result = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    assert result.status_code == 400


def test_generate_missing_anthropic_key_is_400_not_500(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    result = client.post("/api/generate", json={"repo": repo.name, "provider": "anthropic"})
    assert result.status_code == 400


def test_generate_unreachable_provider_stops_and_saves_partial_progress(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})

    calls = []

    def flaky_get_completer(provider, model, api_key):
        def complete(system, user):
            calls.append(user)
            if len(calls) == 2:
                raise ProviderSetupError("Could not reach Ollama. Is it running?")
            return fake_complete(system, user)
        return complete, "fake-model"

    monkeypatch.setattr(web, "_get_completer", flaky_get_completer)
    result = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    assert result.status_code == 502
    assert "Is it running" in result.json()["detail"]

    from docgen.generator import load_generated
    saved = load_generated(repo)
    assert len(saved["items"]) == 1  # the one success before the setup error was saved, not lost


def test_generate_per_chunk_failure_does_not_abort_the_rest(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})

    def flaky_get_completer(provider, model, api_key):
        def complete(system, user):
            # Match only the <target ...> line, not retrieved context: load_file's
            # source legitimately appears inside other chunks' context too.
            if 'description="function load_file"' in user:
                raise ProviderError("model returned garbage")
            return fake_complete(system, user)
        return complete, "fake-model"

    monkeypatch.setattr(web, "_get_completer", flaky_get_completer)
    result = client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"}).json()
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["utils.py::load_file"]["ok"] is False
    assert "garbage" in by_id["utils.py::load_file"]["error"]
    assert by_id["api.py::reload_config"]["ok"] is True  # other chunks still went through


# ----------------------------------------------------------- /plan, /apply


def test_plan_is_read_only(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    before = (repo / "utils.py").read_bytes()

    result = client.get("/api/plan", params={"repo": repo.name})
    assert result.status_code == 200
    body = result.json()
    assert body["changes"] and body["api"]["is_new"] is True
    assert (repo / "utils.py").read_bytes() == before
    assert not (repo / "docs").exists()


def test_plan_and_apply_agree_on_relative_paths(client, repo, monkeypatch, fake_completer):
    """Regression test for the bug found by hand: relative repo path must not
    make plan.api.path.relative_to(repo) raise inside the response builder."""
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})

    plan = client.get("/api/plan", params={"repo": repo.name})
    assert plan.status_code == 200
    assert plan.json()["api"]["path"] == "docs/API.md"

    applied = client.post("/api/apply", json={"repo": repo.name})
    assert applied.status_code == 200
    assert applied.json()["docstring_count"] == 6
    assert (repo / "docs" / "API.md").exists()

    import ast
    for f in ("utils.py", "api.py"):
        ast.parse((repo / f).read_text())  # still valid Python after the web-driven apply


def test_full_pipeline_leaves_a_working_repo(client, repo, monkeypatch, fake_completer):
    """index -> generate (fake model) -> apply, then the repo still works."""
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama", "limit": 0})
    assert client.post("/api/apply", json={"repo": repo.name}).status_code == 200

    for py in repo.glob("*.py"):
        compile(py.read_text(), str(py), "exec")

    undocumented = {c.name for c in parse_repo(repo) if not c.docstring}
    assert undocumented == {"get_config", "reload_config"}  # endpoints live in API.md
    assert "## POST /config/reload" in (repo / "docs" / "API.md").read_text()


def test_apply_is_idempotent(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    client.post("/api/apply", json={"repo": repo.name})

    again = client.post("/api/apply", json={"repo": repo.name})
    assert again.json()["changes"] == []
    assert again.json()["api"] is None


def test_apply_no_api_md_flag_skips_it(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    result = client.post("/api/apply", json={"repo": repo.name, "api_md": False})
    assert result.json()["api"] is None
    assert not (repo / "docs").exists()


def test_apply_respects_docs_dir_escape_guard(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/index", json={"repo": repo.name})
    client.post("/api/generate", json={"repo": repo.name, "provider": "ollama"})
    result = client.post("/api/apply", json={"repo": repo.name, "docs_dir": "../outside"})
    assert result.status_code == 400


# --------------------------------------------------------------- /api/docs


def seed_generated(repo, texts):
    """texts: {chunk name: (kind, text)}"""
    import json as _json

    from docgen.generator import source_hash

    chunks = parse_repo(repo)
    from docgen.indexer import ensure_unique_ids as _euid
    _euid(chunks)
    by_name = {c.name: c for c in chunks}
    items = {}
    for name, (kind, text) in texts.items():
        c = by_name[name]
        items[c.id] = {
            "kind": kind, "text": text, "file_path": c.file_path,
            "start_line": c.start_line, "source_hash": source_hash(c),
        }
    d = repo / ".docgen"
    d.mkdir(exist_ok=True)
    (d / "generated.json").write_text(_json.dumps({"model": "x", "items": items}))


def test_docs_endpoint_is_read_only_and_groups_by_file(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    seed_generated(repo, {
        "load_file": ("docstring", "Read a file."),
        "parse_config": ("docstring", "Load and normalize a config."),
        "get_config": ("endpoint_doc", "Return the settings.\n\n**Response**\n\nJSON settings."),
    })
    before = (repo / "utils.py").read_bytes()

    result = client.get("/api/docs", params={"repo": repo.name})
    assert result.status_code == 200
    body = result.json()

    assert set(body["files"].keys()) == {"utils.py"}
    names = [e["name"] for e in body["files"]["utils.py"]]
    assert names == ["parse_config", "load_file"]  # sorted by line, not insertion order
    assert "## GET /config" in body["api_md"]
    assert "Return the settings." in body["api_md"]

    assert (repo / "utils.py").read_bytes() == before  # never touches source
    assert not (repo / "docs").exists()  # never writes API.md to disk either


def test_docs_endpoint_with_nothing_generated_yet(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    result = client.get("/api/docs", params={"repo": repo.name})
    assert result.status_code == 200
    assert result.json() == {"files": {}, "api_md": None, "overview": None, "overview_stale": False}


def test_docs_endpoint_flags_stale_entries(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    seed_generated(repo, {"load_file": ("docstring", "Read a file.")})
    (repo / "utils.py").write_text(
        (repo / "utils.py").read_text().replace("f.read()", "f.read().strip()")
    )
    entry = client.get("/api/docs", params={"repo": repo.name}).json()["files"]["utils.py"][0]
    assert entry["stale"] is True


def test_docs_endpoint_skips_entries_for_deleted_chunks(client, repo, monkeypatch):
    """A chunk removed from the source since generation must not crash the endpoint."""
    monkeypatch.chdir(repo.parent)
    seed_generated(repo, {"load_file": ("docstring", "Read a file.")})
    (repo / "utils.py").write_text("x = 1\n")  # load_file no longer exists
    result = client.get("/api/docs", params={"repo": repo.name})
    assert result.status_code == 200
    assert result.json() == {"files": {}, "api_md": None, "overview": None, "overview_stale": False}


def test_docs_endpoint_only_has_api_md_when_endpoints_exist(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    seed_generated(repo, {"load_file": ("docstring", "Read a file.")})
    result = client.get("/api/docs", params={"repo": repo.name})
    assert result.json()["api_md"] is None


# ------------------------------------------------------------- /api/overview


@pytest.fixture
def fake_overview_completer(monkeypatch):
    def fake_get_completer(provider, model, api_key):
        def complete(system, user):
            return "<overview>## Overview\n\nA config service.\n\n## How it fits together\n\n- utils.py, api.py.</overview>"
        return complete, (model or "fake-model")
    monkeypatch.setattr(web, "_get_completer", fake_get_completer)


def test_overview_endpoint_generates_and_saves(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    result = client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    assert result.status_code == 200
    body = result.json()
    assert body["cached"] is False
    assert "A config service" in body["text"]

    from docgen.generator import load_generated
    saved = load_generated(repo)
    assert "A config service" in saved["overview"]["text"]
    assert saved["overview"]["structure_hash"]


def test_overview_endpoint_is_cached_on_second_call(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    again = client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    assert again.json()["cached"] is True


def test_overview_endpoint_force_regenerates(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    forced = client.post("/api/overview", json={"repo": repo.name, "provider": "ollama", "force": True})
    assert forced.json()["cached"] is False


def test_overview_endpoint_regenerates_when_structure_changes(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    (repo / "utils.py").write_text((repo / "utils.py").read_text() + "\ndef new_fn():\n    return 1\n")
    result = client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    assert result.json()["cached"] is False


def test_overview_endpoint_bad_reply_is_a_clean_error(client, repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    monkeypatch.setattr(web, "_get_completer", lambda p, m, k: (lambda s, u: "no tags", "fake"))
    result = client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    assert result.status_code == 502


def test_overview_endpoint_missing_key_is_400(client, repo, monkeypatch, fake_completer):
    monkeypatch.chdir(repo.parent)
    result = client.post("/api/overview", json={"repo": repo.name, "provider": "anthropic"})
    assert result.status_code == 400


def test_docs_endpoint_includes_overview_and_its_staleness(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    fresh = client.get("/api/docs", params={"repo": repo.name}).json()
    assert "A config service" in fresh["overview"]
    assert fresh["overview_stale"] is False

    (repo / "utils.py").write_text((repo / "utils.py").read_text() + "\ndef new_fn():\n    return 1\n")
    stale = client.get("/api/docs", params={"repo": repo.name}).json()
    assert stale["overview_stale"] is True
    assert stale["overview"] == fresh["overview"]  # still shown, just flagged


def test_plan_and_apply_include_overview(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})

    planned = client.get("/api/plan", params={"repo": repo.name}).json()
    assert planned["overview"]["path"] == "docs/OVERVIEW.md"
    assert not (repo / "docs").exists()  # plan is read-only

    applied = client.post("/api/apply", json={"repo": repo.name}).json()
    assert applied["overview"]["path"] == "docs/OVERVIEW.md"
    assert (repo / "docs" / "OVERVIEW.md").exists()


def test_apply_no_overview_flag_skips_it(client, repo, monkeypatch, fake_overview_completer):
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    result = client.post("/api/apply", json={"repo": repo.name, "overview": False})
    assert result.json()["overview"] is None
    assert not (repo / "docs").exists()


def test_apply_with_only_overview_and_no_docstrings_still_works(client, repo, monkeypatch, fake_overview_completer):
    """Regression: the CLI had a bug here; confirm the web endpoint never had it."""
    monkeypatch.chdir(repo.parent)
    client.post("/api/overview", json={"repo": repo.name, "provider": "ollama"})
    result = client.post("/api/apply", json={"repo": repo.name})
    assert result.status_code == 200
    assert result.json()["changes"] == []
    assert result.json()["overview"] is not None
