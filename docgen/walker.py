"""Walks a repo and decides which files docgen reads.

Every stage (the hub, the search index) gets its file list from
here, so the skip rules live in one place:

- `.gitignore` and `.docgenignore` patterns, including ones in subdirectories
  (gitignore syntax; a deeper file overrides a shallower one, as in git).
- A fixed set of noise directories, skipped even without a .gitignore, since a
  downloaded or non-git folder can still contain node_modules or a venv, plus
  docs/hub, where the hub writes its own Markdown export.
- Lockfiles, minified bundles and source maps: large, generated, and useless
  to summarize.
- Symlinks (never followed, so a walk can't leave the repo), binaries, files
  that aren't UTF-8, and files over MAX_FILE_BYTES.

Ignore-rule matches are dropped quietly (a node_modules folder alone can hold
thousands of files). Every other skip is reported with a reason so the UI can
show what was left out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

MAX_FILE_BYTES = 200_000

ALWAYS_SKIP_DIRS = {
    ".git", ".docgen", "__pycache__", "node_modules", "venv", ".venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
}

# Paths (relative to the repo root) skipped in every repo: the hub's own
# Markdown export, so a rebuild never summarizes the previous build's output.
ALWAYS_SKIP_PATHS = {"docs/hub"}

IGNORE_FILES = (".gitignore", ".docgenignore")

LOCKFILE_NAMES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "bun.lockb", "poetry.lock", "Pipfile.lock", "uv.lock", "Cargo.lock",
    "Gemfile.lock", "composer.lock", "go.sum", "mix.lock", "pubspec.lock",
    "Podfile.lock", "flake.lock", "packages.lock.json",
}

# Extension -> language name. Used for display now, and later to pick import
# patterns and splitting rules per language.
LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".rb": "ruby", ".php": "php", ".cs": "csharp", ".swift": "swift",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".m": "objective-c", ".dart": "dart", ".lua": "lua", ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hs": "haskell", ".clj": "clojure", ".r": "r", ".jl": "julia",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell", ".ps1": "powershell",
    ".sql": "sql", ".graphql": "graphql", ".proto": "protobuf",
    ".html": "html", ".htm": "html", ".vue": "vue", ".svelte": "svelte",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".xml": "xml", ".tf": "terraform",
}

LANGUAGE_BY_NAME = {
    "Dockerfile": "dockerfile", "Containerfile": "dockerfile", "Makefile": "make",
    "Jenkinsfile": "groovy", "Gemfile": "ruby", "Rakefile": "ruby", "Procfile": "text",
}


@dataclass
class RepoFile:
    """A file docgen will read."""

    path: str  # relative to the repo root, always with forward slashes
    abs_path: Path
    language: str
    size: int  # bytes
    lines: int


@dataclass
class WalkResult:
    files: list[RepoFile] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def detect_language(name: str) -> str:
    """Best-effort language name for a file; "text" when unknown."""
    if name in LANGUAGE_BY_NAME:
        return LANGUAGE_BY_NAME[name]
    if name.startswith("Dockerfile."):
        return "dockerfile"
    return LANGUAGE_BY_SUFFIX.get(Path(name).suffix.lower(), "text")


def _is_generated_name(name: str) -> str | None:
    """Reason to skip a file based on its name alone, or None."""
    if name in LOCKFILE_NAMES or name.endswith(".lock"):
        return "lockfile"
    if ".min." in name:
        return "minified"
    if name.endswith(".map"):
        return "source map"
    return None


def _load_specs(dir_path: Path) -> list[pathspec.GitIgnoreSpec]:
    specs = []
    for ignore_name in IGNORE_FILES:
        f = dir_path / ignore_name
        if f.is_file():
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            specs.append(pathspec.GitIgnoreSpec.from_lines(lines))
    return specs


def _is_ignored(rel_path: str, specs: list[tuple[str, pathspec.GitIgnoreSpec]]) -> bool:
    """Apply the ignore files from the root down; the last one that matches decides.

    `rel_path` ends in "/" for directories, so patterns like `build/` only
    match directories.
    """
    ignored = False
    for base, spec in specs:
        if base:
            if not rel_path.startswith(base + "/"):
                continue
            local = rel_path[len(base) + 1:]
        else:
            local = rel_path
        result = spec.check_file(local)
        if result.include is not None:
            ignored = result.include
    return ignored


def _read_text(path: Path) -> str | None:
    """Return the file's text, or None if it looks binary or isn't UTF-8."""
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def walk_repo(repo_path: str | Path) -> WalkResult:
    """Collect every file docgen should read, sorted by path."""
    root = Path(repo_path).resolve()
    result = WalkResult()
    # Ignore specs in effect for each directory: (dir relative to root, spec).
    specs_for: dict[str, list[tuple[str, pathspec.GitIgnoreSpec]]] = {}

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dir_abs = Path(dirpath)
        rel_dir = dir_abs.relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        parent = rel_dir.rsplit("/", 1)[0] if "/" in rel_dir else ""
        inherited = specs_for.get(parent, []) if rel_dir else []
        specs = inherited + [(rel_dir, s) for s in _load_specs(dir_abs)]
        specs_for[rel_dir] = specs

        def rel(name: str) -> str:
            return f"{rel_dir}/{name}" if rel_dir else name

        # Prune in place so os.walk never descends into skipped directories.
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in ALWAYS_SKIP_DIRS
            and rel(d) not in ALWAYS_SKIP_PATHS
            and not (dir_abs / d).is_symlink()
            and not _is_ignored(rel(d) + "/", specs)
        )

        for name in sorted(filenames):
            rel_path = rel(name)
            if name == ".docgenignore" or _is_ignored(rel_path, specs):
                continue
            abs_path = dir_abs / name
            if abs_path.is_symlink():
                result.skipped.append((rel_path, "symlink"))
                continue
            reason = _is_generated_name(name)
            if reason:
                result.skipped.append((rel_path, reason))
                continue
            size = abs_path.stat().st_size
            if size > MAX_FILE_BYTES:
                result.skipped.append((rel_path, f"over {MAX_FILE_BYTES // 1000} KB"))
                continue
            text = _read_text(abs_path)
            if text is None:
                result.skipped.append((rel_path, "binary or not UTF-8"))
                continue
            result.files.append(RepoFile(
                path=rel_path,
                abs_path=abs_path,
                language=detect_language(name),
                size=size,
                lines=len(text.splitlines()),
            ))

    result.files.sort(key=lambda f: f.path)
    result.skipped.sort()
    return result
