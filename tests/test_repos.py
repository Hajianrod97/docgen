"""Tests for turning a path or GitHub URL into a folder. Cloning is tested
against a local git repo standing in for GitHub, so nothing touches the network."""

import subprocess

import pytest

from docgen import repos
from docgen.repos import RepoError, parse_github_url, resolve_repo


@pytest.mark.parametrize("text, expected", [
    ("https://github.com/owner/repo", ("owner", "repo")),
    ("https://github.com/owner/repo.git", ("owner", "repo")),
    ("https://github.com/owner/repo/", ("owner", "repo")),
    ("https://github.com/owner/repo/tree/main/src", ("owner", "repo")),
    ("http://www.github.com/owner/my.repo-2", ("owner", "my.repo-2")),
    ("github.com/owner/repo", ("owner", "repo")),
    ("  https://github.com/owner/repo?tab=readme  ", ("owner", "repo")),
])
def test_parse_github_url_accepts_repo_links(text, expected):
    assert parse_github_url(text) == expected


@pytest.mark.parametrize("text", [
    "https://gitlab.com/owner/repo",
    "owner/repo",
    "https://github.com/owner",
    "https://github.com/../repo",
    "https://github.com/owner/..",
    "~/code/repo",
    "https://github.com/owner/repo;rm -rf",
])
def test_parse_github_url_rejects_everything_else(text):
    assert parse_github_url(text) is None


def test_local_path_resolves_without_cloning(tmp_path):
    source = resolve_repo(str(tmp_path))
    assert source.path == tmp_path.resolve()
    assert not source.is_clone and source.github is None
    assert source.source_url("a.py") is None


def test_missing_local_path_is_an_error(tmp_path):
    with pytest.raises(RepoError, match="Not a directory"):
        resolve_repo(str(tmp_path / "nope"))


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def fake_github(tmp_path, monkeypatch):
    """A local repo served as github.com/owner/repo, and a temporary cache."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    (origin / "app.py").write_text("print('v1')\n")
    git(origin, "add", ".")
    git(origin, "commit", "-q", "-m", "v1")
    monkeypatch.setenv("DOCGEN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(repos, "clone_url", lambda owner, name: origin.as_uri())
    return origin


def test_github_url_is_cloned_into_the_cache(fake_github, tmp_path):
    source = resolve_repo("https://github.com/owner/repo")
    assert source.path == tmp_path / "cache" / "owner" / "repo"
    assert source.github == "owner/repo"
    assert (source.path / "app.py").read_text() == "print('v1')\n"
    assert source.source_url("src/a.py") == "https://github.com/owner/repo/blob/HEAD/src/a.py"


def test_fetch_pulls_new_commits_and_keeps_the_hub_cache(fake_github):
    source = resolve_repo("https://github.com/owner/repo")
    (source.path / ".docgen").mkdir()
    (source.path / ".docgen" / "hub.json").write_text("{}")

    (fake_github / "app.py").write_text("print('v2')\n")
    git(fake_github, "commit", "-q", "-am", "v2")

    resolve_repo("https://github.com/owner/repo")  # no fetch: the old copy stays
    assert (source.path / "app.py").read_text() == "print('v1')\n"

    again = resolve_repo("https://github.com/owner/repo", fetch=True)
    assert again.warning is None
    assert (source.path / "app.py").read_text() == "print('v2')\n"
    assert (source.path / ".docgen" / "hub.json").read_text() == "{}"


def test_failed_pull_falls_back_to_the_existing_clone(fake_github, monkeypatch, tmp_path):
    resolve_repo("https://github.com/owner/repo")
    monkeypatch.setattr(repos, "clone_url", lambda owner, name: (tmp_path / "gone").as_uri())
    subprocess.run(["git", "remote", "set-url", "origin", (tmp_path / "gone").as_uri()],
                   cwd=tmp_path / "cache" / "owner" / "repo", check=True)
    source = resolve_repo("https://github.com/owner/repo", fetch=True)
    assert source.warning and "older copy" in source.warning
    assert (source.path / "app.py").exists()


def test_failed_clone_explains_the_likely_cause(fake_github, monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "clone_url", lambda owner, name: (tmp_path / "missing").as_uri())
    with pytest.raises(RepoError, match="Is it a public GitHub repo"):
        resolve_repo("https://github.com/owner/private")
    assert not (tmp_path / "cache" / "owner" / "private" / ".git").exists()


def test_missing_git_is_reported(fake_github, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(repos.subprocess, "run", no_git)
    with pytest.raises(RepoError, match="git is not installed"):
        resolve_repo("https://github.com/owner/repo")
