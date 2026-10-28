"""Tests for splitting files into search-index chunks."""

from pathlib import Path

import pytest

from docgen.chunking import MAX_CHUNK_LINES, WINDOW_LINES, WINDOW_OVERLAP, chunk_file
from docgen.walker import walk_repo

ROOT = Path(__file__).resolve().parent.parent


def covered(chunks):
    lines = set()
    for c in chunks:
        lines.update(range(c.start_line, c.end_line + 1))
    return lines


def nonblank(text):
    return {n for n, line in enumerate(text.splitlines(), 1) if line.strip()}


def units(chunks):
    return [(c.unit, c.symbol, c.start_line, c.end_line) for c in chunks]


PY = '''"""Module docstring."""
import os

LIMIT = 3


# Loads the config.
@cache
def load(path):
    return open(path).read()


class Store:
    """Holds things."""

    def add(self, x):
        return x


async def main():
    pass

if __name__ == "__main__":
    main()
'''


def test_python_splits_at_definitions_and_keeps_code_between():
    chunks = chunk_file("app.py", "python", PY)
    assert units(chunks) == [
        ("module", "", 1, 4),
        ("function", "load", 7, 10),  # the comment and decorator come along
        ("class", "Store", 13, 17),
        ("function", "main", 20, 21),
        ("module", "", 23, 24),
    ]
    assert chunks[1].text.startswith("# Loads the config.\n@cache\ndef load")
    assert covered(chunks) >= nonblank(PY)


def test_long_python_class_is_split_into_methods():
    methods = "".join(f"    def m{i}(self):\n" + "        x = 1\n" * 10 + "\n" for i in range(12))
    text = "class Big:\n    \"\"\"Doc.\"\"\"\n    size = 1\n\n" + methods
    chunks = chunk_file("big.py", "python", text)
    assert chunks[0].unit == "class body" and chunks[0].symbol == "Big"
    assert chunks[0].text.startswith("class Big:")
    assert [c.symbol for c in chunks[1:]] == [f"Big.m{i}" for i in range(12)]
    assert all(c.unit == "method" for c in chunks[1:])
    assert covered(chunks) >= nonblank(text)


def test_long_python_function_is_windowed_with_overlap():
    text = "def huge():\n" + "".join(f"    x{i} = {i}\n" for i in range(MAX_CHUNK_LINES + 50))
    chunks = chunk_file("huge.py", "python", text)
    assert len(chunks) > 1
    assert all(c.symbol == "huge" and c.unit == "function" for c in chunks)
    assert all(c.end_line - c.start_line + 1 <= WINDOW_LINES for c in chunks)
    assert chunks[1].start_line == chunks[0].end_line - WINDOW_OVERLAP + 1
    assert covered(chunks) >= nonblank(text)


def test_python_that_does_not_parse_falls_back_to_windows():
    text = "def broken(:\n    pass\n"
    chunks = chunk_file("bad.py", "python", text)
    assert units(chunks) == [("window", "", 1, 2)]


MD = """Intro line before any heading.

# Guide

Welcome.

## Install

```bash
# not a heading
pip install x
```

### macOS

brew install x

## Usage

Run it.
"""


def test_markdown_splits_at_headings_with_their_path():
    chunks = chunk_file("README.md", "markdown", MD)
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("", 1, 1),
        ("Guide", 3, 5),
        ("Guide > Install", 7, 12),
        ("Guide > Install > macOS", 14, 16),
        ("Guide > Usage", 18, 20),
    ]
    assert "# not a heading" in chunks[2].text
    assert covered(chunks) >= nonblank(MD)


def test_other_languages_use_overlapping_windows():
    text = "".join(f"const a{i} = {i};\n" for i in range(150))
    chunks = chunk_file("big.ts", "typescript", text)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 150)]
    assert all(c.unit == "window" for c in chunks)


def test_short_file_is_one_chunk_and_blank_file_none():
    assert units(chunk_file("a.ts", "typescript", "\n\nexport const a = 1;\n\n")) == [("window", "", 3, 3)]
    assert chunk_file("empty.ts", "typescript", "\n  \n") == []


def test_embedding_text_names_the_file_and_lines():
    chunk = chunk_file("app.py", "python", PY)[1]
    assert chunk.id == "code:app.py:7-10"
    assert chunk.embedding_text().startswith("File: app.py lines 7-10 (python)\nfunction load\n\n# Loads")


@pytest.mark.parametrize("folder", ["example_multilang", "docgen"])
def test_every_nonblank_line_of_real_files_is_covered(folder):
    for f in walk_repo(ROOT / folder).files:
        text = f.abs_path.read_text(encoding="utf-8")
        chunks = chunk_file(f.path, f.language, text)
        assert covered(chunks) >= nonblank(text), f.path
        assert len({c.id for c in chunks}) == len(chunks), f.path
