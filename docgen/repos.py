"""Turns a local path or public GitHub URL into a folder. GitHub repos are
shallow-cloned into ~/.cache/docgen/<owner>/<repo> (or $DOCGEN_CACHE_DIR) and
updated in place, with git prompts off so a private repo fails fast.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_TIMEOUT = 300  # seconds; a first clone of a large repo can be slow

_GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?(?:[/?#].*)?$"
)


class RepoError(Exception):
    """The path doesn't exist, or the repo couldn't be cloned."""


@dataclass
class RepoSource:
    path: Path
    owner: str | None = None  # set for GitHub clones
    name: str | None = None
    warning: str | None = None  # e.g. the pull failed and an older clone is used

    @property
    def is_clone(self) -> bool:
        return self.owner is not None

    @property
    def github(self) -> str | None:
        return f"{self.owner}/{self.name}" if self.is_clone else None

    def source_url(self, rel_path: str) -> str | None:
        """A link to a file on GitHub, for clones."""
        if not self.is_clone:
            return None
        return f"https://github.com/{self.owner}/{self.name}/blob/HEAD/{rel_path}"


def parse_github_url(text: str) -> tuple[str, str] | None:
    """(owner, repo) for a github.com URL, else None. Extra path parts such as
    /tree/main/src are ignored: the whole repo is always cloned."""
    match = _GITHUB_URL.match(text.strip())
    if not match:
        return None
    owner, name = match.groups()
    if owner in (".", "..") or name in (".", ".."):
        return None
    return owner, name


def cache_dir() -> Path:
    return Path(os.environ.get("DOCGEN_CACHE_DIR") or "~/.cache/docgen").expanduser()


def clone_url(owner: str, name: str) -> str:
    return f"https://github.com/{owner}/{name}.git"


def _git(args: list[str], cwd: Path | None = None) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.run(
            ["git", *args], cwd=cwd, env=env, check=True,
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
        )
    except FileNotFoundError as e:
        raise RepoError("git is not installed, so GitHub repos can't be cloned.") from e
    except subprocess.TimeoutExpired as e:
        raise RepoError(f"git {args[0]} took longer than {GIT_TIMEOUT}s and was stopped.") from e
    except subprocess.CalledProcessError as e:
        lines = (e.stderr or "").strip().splitlines()
        raise RepoError(lines[-1] if lines else f"git {args[0]} failed") from e


def sync_clone(owner: str, name: str, dest: Path) -> None:
    """Clone the repo into dest, or update an existing clone to the latest commit.

    Untracked files are left alone, so dest/.docgen (the hub cache) is kept.
    """
    if (dest / ".git").is_dir():
        _git(["fetch", "--depth", "1", "--quiet", "origin", "HEAD"], cwd=dest)
        _git(["reset", "--hard", "--quiet", "FETCH_HEAD"], cwd=dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(["clone", "--depth", "1", "--quiet", clone_url(owner, name), str(dest)])
    except RepoError as e:
        raise RepoError(f"Could not clone {owner}/{name}. Is it a public GitHub repo? ({e})") from e


def resolve_repo(text: str, fetch: bool = False) -> RepoSource:
    """The folder to read for a local path or GitHub URL.

    A GitHub repo is cloned the first time it's asked for. With `fetch`, an
    existing clone is updated; if that fails (offline, say), the old clone is
    used and `warning` says so.
    """
    gh = parse_github_url(text)
    if gh is None:
        path = Path(text).expanduser().resolve()
        if not path.is_dir():
            raise RepoError(f"Not a directory: {text}")
        return RepoSource(path)

    owner, name = gh
    dest = cache_dir() / owner / name
    source = RepoSource(dest, owner, name)
    if not (dest / ".git").is_dir():
        sync_clone(owner, name, dest)
    elif fetch:
        try:
            sync_clone(owner, name, dest)
        except RepoError as e:
            source.warning = f"Could not pull the latest commit, so an older copy is shown. ({e})"
    return source
