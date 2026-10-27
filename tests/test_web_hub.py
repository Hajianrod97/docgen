"""Tests for the web UI's hub endpoints. The model is a fake, and builds run in
the real background thread, so these check the job handling end to end."""

import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

from docgen import repos, web
from docgen.generator import ProviderSetupError
from test_hub import MULTILANG, FakeModel, models_for
from test_repos import fake_github  # noqa: F401  (fixture)


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    return dest


@pytest.fixture
def client():
    return TestClient(web.app)


@pytest.fixture
def fake(monkeypatch):
    """Replace the real completers; records what the endpoint asked for."""
    model = FakeModel()
    model.requested = []

    def make(provider, file_model=None, top_model=None, api_key=None):
        model.requested.append((provider, file_model, top_model, api_key))
        return models_for(model)

    monkeypatch.setattr(web, "make_hub_models", make)
    return model


def wait_for_build(client, repo, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get("/api/hub/status", params={"repo": str(repo)}).json()["job"]
        if job and not job["running"]:
            return job
        time.sleep(0.02)
    raise AssertionError("build did not finish")


def build(client, repo, **extra):
    body = {"repo": str(repo), "provider": "anthropic", "api_key": "sk-test", **extra}
    return client.post("/api/hub/build", json=body)


def test_hub_view_before_any_build(client, repo):
    data = client.get("/api/hub", params={"repo": str(repo)}).json()
    assert data["name"] == "multilang"
    assert data["overview"] is None and data["overview_status"] == "missing"
    assert data["tree"]["status"] == "missing"
    assert [c["name"] for c in data["tree"]["children"]][:2] == ["backend", "frontend"]
    assert data["job"] is None
    assert data["export_dir"] == str(repo / "docs" / "hub")


def test_estimate_uses_default_models(client, repo):
    est = client.get("/api/hub/estimate", params={"repo": str(repo)}).json()
    assert (est["file_model"], est["top_model"]) == ("claude-haiku-5-5", "claude-opus-5-5")
    assert est["total_calls"] == 6 and est["todo_files"] == 1 and est["todo_dirs"] == 4
    assert est["cost_usd"] > 0

    est = client.get("/api/hub/estimate", params={"repo": str(repo), "provider": "ollama"}).json()
    assert est["file_model"] == est["top_model"] == "qwen2.5-coder:7b"
    assert est["cost_usd"] == 0

    est = client.get("/api/hub/estimate", params={
        "repo": str(repo), "file_model": "claude-sonnet-5-5", "top_model": "claude-haiku-5-5",
    }).json()
    assert (est["file_model"], est["top_model"]) == ("claude-sonnet-5-5", "claude-haiku-5-5")


def test_estimate_rejects_unknown_provider(client, repo):
    r = client.get("/api/hub/estimate", params={"repo": str(repo), "provider": "gpt"})
    assert r.status_code == 400


def test_build_needs_an_api_key_for_anthropic(client, repo, fake):
    r = client.post("/api/hub/build", json={"repo": str(repo), "provider": "anthropic"})
    assert r.status_code == 400 and "API key" in r.json()["detail"]
    assert fake.requested == []


def test_build_runs_in_the_background_and_fills_the_hub(client, repo, fake):
    r = build(client, repo, file_model="claude-sonnet-5-5")
    assert r.status_code == 200 and r.json()["running"] is True
    job = wait_for_build(client, repo)
    assert job["error"] is None and not job["stopped"]
    assert job["progress"]["stage"] == "done"
    assert job["progress"]["done"] == job["progress"]["total"] == 6
    assert fake.requested == [("anthropic", "claude-sonnet-5-5", "claude-opus-5-5", "sk-test")]
    assert "stop" not in job  # internal only

    data = client.get("/api/hub", params={"repo": str(repo)}).json()
    assert data["overview_status"] == "done"
    assert data["overview"]["diagram"].startswith("flowchart TD")
    assert data["files"]["backend/store.py"]["purpose"] == "Purpose of backend/store.py."
    assert data["tree"]["status"] == "done"


def test_failures_are_listed_on_the_job(client, repo, fake):
    fake.fail_on.add("backend/store.py")
    build(client, repo)
    job = wait_for_build(client, repo)
    assert job["error"] is None
    assert job["progress"]["errors"] == 1
    assert job["failures"][0]["item"] == "backend/store.py"


def test_setup_error_ends_the_job_with_a_message(client, repo, monkeypatch):
    def broken(system, user):
        raise ProviderSetupError("Authentication failed. Check the API key.")

    monkeypatch.setattr(web, "make_hub_models", lambda *a, **k: models_for(broken))
    build(client, repo)
    job = wait_for_build(client, repo)
    assert job["error"] == "Authentication failed. Check the API key."


class BlockingModel(FakeModel):
    """Waits on `release` before every reply, so a test can act mid-build."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, system, user):
        self.started.set()
        assert self.release.wait(10)
        return super().__call__(system, user)


def test_one_build_per_repo_and_stop(client, repo, monkeypatch):
    model = BlockingModel()
    monkeypatch.setattr(web, "make_hub_models", lambda *a, **k: models_for(model))
    assert build(client, repo).status_code == 200
    assert model.started.wait(5)

    second = build(client, repo)
    assert second.status_code == 409

    stopped = client.post("/api/hub/stop", json={"repo": str(repo)}).json()["job"]
    assert stopped["running"] is True  # the in-flight call still finishes
    model.release.set()
    job = wait_for_build(client, repo)
    assert job["stopped"] is True and job["error"] is None
    assert job["progress"]["done"] < job["progress"]["total"]

    # Once stopped, a new build may start and picks up where it left off.
    assert build(client, repo).status_code == 200
    job = wait_for_build(client, repo)
    assert job["progress"]["stage"] == "done"


def test_export_needs_a_built_hub_then_writes_markdown(client, repo, fake):
    r = client.post("/api/hub/export", json={"repo": str(repo)})
    assert r.status_code == 400

    build(client, repo)
    wait_for_build(client, repo)
    data = client.post("/api/hub/export", json={"repo": str(repo)}).json()
    assert data["dir"] == str(repo / "docs" / "hub")
    assert "backend/README.md" in data["files"]
    assert (repo / "docs" / "hub" / "README.md").exists()


def test_scan_and_hub_accept_a_github_url(client, fake_github, tmp_path):  # noqa: F811
    url = "https://github.com/owner/repo"
    data = client.get("/api/scan", params={"repo": url}).json()
    assert data["source"]["github"] == "owner/repo"
    assert data["source"]["path"] == str(tmp_path / "cache" / "owner" / "repo")
    assert data["files"]["languages"] == {"python": 1}

    hub = client.get("/api/hub", params={"repo": url}).json()
    assert hub["name"] == "owner/repo"
    assert hub["source_urls"]["app.py"] == "https://github.com/owner/repo/blob/HEAD/app.py"


def test_bad_repo_is_a_clear_error(client, tmp_path):
    r = client.get("/api/hub", params={"repo": str(tmp_path / "nope")})
    assert r.status_code == 400 and "Not a directory" in r.json()["detail"]


def test_failed_clone_is_a_clear_error(client, fake_github, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(repos, "clone_url", lambda owner, name: (tmp_path / "missing").as_uri())
    r = client.get("/api/scan", params={"repo": "https://github.com/owner/private"})
    assert r.status_code == 400 and "public GitHub repo" in r.json()["detail"]
