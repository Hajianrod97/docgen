"""Tests for the repo walker: ignore files, skip rules, language detection."""

import os
from pathlib import Path

import pytest

from docgen.walker import MAX_FILE_BYTES, detect_language, walk_repo

MULTILANG = Path(__file__).resolve().parent.parent / "example_multilang"


def write(root: Path, rel: str, content: str | bytes = "x\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def paths(root: Path) -> list[str]:
    return [f.path for f in walk_repo(root).files]


def skipped(root: Path) -> dict[str, str]:
    return dict(walk_repo(root).skipped)


def test_root_gitignore_files_and_dirs(tmp_path):
    write(tmp_path, ".gitignore", "*.log\nout/\n")
    write(tmp_path, "app.py")
    write(tmp_path, "debug.log")
    write(tmp_path, "out/result.txt")
    write(tmp_path, "src/out/also.txt")  # out/ matches at any depth
    assert paths(tmp_path) == [".gitignore", "app.py"]


def test_directory_only_pattern_does_not_match_a_file(tmp_path):
    write(tmp_path, ".gitignore", "build/\n")
    write(tmp_path, "build")  # a file named build, not a directory
    assert "build" in paths(tmp_path)


def test_nested_gitignore_is_relative_to_its_directory(tmp_path):
    write(tmp_path, "pkg/.gitignore", "/secret.txt\n")
    write(tmp_path, "pkg/secret.txt")
    write(tmp_path, "secret.txt")  # same name at the root is not affected
    assert paths(tmp_path) == ["pkg/.gitignore", "secret.txt"]


def test_deeper_gitignore_can_reinclude(tmp_path):
    write(tmp_path, ".gitignore", "*.txt\n")
    write(tmp_path, "docs/.gitignore", "!keep.txt\n")
    write(tmp_path, "docs/keep.txt")
    write(tmp_path, "docs/drop.txt")
    assert "docs/keep.txt" in paths(tmp_path)
    assert "docs/drop.txt" not in paths(tmp_path)


def test_docgenignore_applies_and_is_not_itself_read(tmp_path):
    write(tmp_path, ".docgenignore", "fixtures/\n")
    write(tmp_path, "fixtures/big.json")
    write(tmp_path, "main.go")
    assert paths(tmp_path) == ["main.go"]


def test_noise_dirs_skipped_without_gitignore(tmp_path):
    for d in ("node_modules", ".venv", ".git", "__pycache__", ".docgen"):
        write(tmp_path, f"{d}/inner.js")
    write(tmp_path, "index.js")
    assert paths(tmp_path) == ["index.js"]


def test_generated_files_are_reported(tmp_path):
    write(tmp_path, "package-lock.json", "{}")
    write(tmp_path, "Cargo.lock")
    write(tmp_path, "custom.lock")
    write(tmp_path, "vendor.min.js")
    write(tmp_path, "app.js.map")
    write(tmp_path, "app.js")
    assert paths(tmp_path) == ["app.js"]
    assert skipped(tmp_path) == {
        "package-lock.json": "lockfile",
        "Cargo.lock": "lockfile",
        "custom.lock": "lockfile",
        "vendor.min.js": "minified",
        "app.js.map": "source map",
    }


def test_large_binary_and_non_utf8_files_are_reported(tmp_path):
    write(tmp_path, "big.txt", "a" * (MAX_FILE_BYTES + 1))
    write(tmp_path, "edge.txt", "a" * MAX_FILE_BYTES)  # exactly at the limit is kept
    write(tmp_path, "logo.png", b"\x89PNG\r\n\x1a\n\0\0\0")
    write(tmp_path, "latin1.txt", "caf\xe9".encode("latin-1"))
    assert paths(tmp_path) == ["edge.txt"]
    reasons = skipped(tmp_path)
    assert reasons["big.txt"].startswith("over")
    assert reasons["logo.png"] == "binary or not UTF-8"
    assert reasons["latin1.txt"] == "binary or not UTF-8"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_symlinks_are_not_followed(tmp_path):
    outside = tmp_path / "outside"
    write(outside, "secret.py")
    repo = tmp_path / "repo"
    write(repo, "real.py")
    (repo / "linked_dir").symlink_to(outside, target_is_directory=True)
    (repo / "linked.py").symlink_to(outside / "secret.py")
    assert paths(repo) == ["real.py"]
    assert skipped(repo) == {"linked.py": "symlink"}


def test_file_details(tmp_path):
    write(tmp_path, "src/util.ts", "a\nb\nc\n")
    (f,) = walk_repo(tmp_path).files
    assert f.path == "src/util.ts"
    assert f.abs_path == (tmp_path / "src/util.ts").resolve()
    assert (f.language, f.size, f.lines) == ("typescript", 6, 3)


@pytest.mark.parametrize("name, language", [
    ("app.py", "python"),
    ("Component.TSX", "typescript"),
    ("Dockerfile", "dockerfile"),
    ("Dockerfile.dev", "dockerfile"),
    ("Makefile", "make"),
    ("config.yml", "yaml"),
    ("NOTES", "text"),
    ("data.unknownext", "text"),
])
def test_detect_language(name, language):
    assert detect_language(name) == language


def test_multilang_sample():
    result = walk_repo(MULTILANG)
    languages = {f.path: f.language for f in result.files}
    assert languages["backend/app.py"] == "python"
    assert languages["frontend/src/api.ts"] == "typescript"
    assert languages["Dockerfile"] == "dockerfile"
    assert ".docgenignore" not in languages
    assert dict(result.skipped) == {"frontend/package-lock.json": "lockfile"}
