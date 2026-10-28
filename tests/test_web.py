"""Tests for the web UI's general endpoints: the page itself and Scan.

Uses FastAPI's TestClient, which calls the ASGI app in-process: no real HTTP
server, no real model. The hub, search index and Ask endpoints have their own
test files.
"""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgen import web

MULTILANG = Path(__file__).resolve().parent.parent / "example_multilang"


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    return dest


@pytest.fixture
def client():
    return TestClient(web.app)


def test_root_serves_the_page(client):
    result = client.get("/")
    assert result.status_code == 200
    assert "docgen" in result.text and 'id="main-ask"' in result.text
    assert "main-docstrings" not in result.text


def test_scan_relative_and_absolute_paths_agree(client, repo, monkeypatch):
    """Relative and absolute paths to the same repo must not diverge."""
    monkeypatch.chdir(repo.parent)
    rel = client.get("/api/scan", params={"repo": repo.name}).json()
    abs_ = client.get("/api/scan", params={"repo": str(repo)}).json()
    assert rel == abs_
    assert rel["source"] == {"path": str(repo.resolve()), "github": None, "warning": None}


def test_scan_reports_files_by_language_and_skips(client):
    files = client.get("/api/scan", params={"repo": str(MULTILANG)}).json()["files"]
    assert files["languages"]["typescript"] == 3
    assert files["languages"]["python"] == 2
    assert files["count"] == sum(files["languages"].values())
    assert files["skipped"] == [{"path": "frontend/package-lock.json", "reason": "lockfile"}]


def test_scan_rejects_a_missing_path(client):
    result = client.get("/api/scan", params={"repo": "/definitely/not/a/real/path"})
    assert result.status_code == 400


@pytest.mark.parametrize("path", ["/api/index", "/api/generate", "/api/plan", "/api/docs"])
def test_old_docstring_endpoints_are_gone(client, path):
    assert client.get(path).status_code in (404, 405)
