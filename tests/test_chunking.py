"""Tests for splitting files into search-index chunks."""

from pathlib import Path

import pytest

from docgen import chunking
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


def test_files_without_a_parser_use_overlapping_windows():
    text = "".join(f"setting_{i} = {i}\n" for i in range(150))
    chunks = chunk_file("big.cfg", "text", text)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 150)]
    assert all(c.unit == "window" for c in chunks)


def test_short_file_is_one_chunk_and_blank_file_none():
    assert units(chunk_file("a.cfg", "text", "\n\nkey = 1\n\n")) == [("window", "", 3, 3)]
    assert chunk_file("empty.ts", "typescript", "\n  \n") == []


def test_embedding_text_names_the_file_and_lines():
    chunk = chunk_file("app.py", "python", PY)[1]
    assert chunk.id == "code:app.py:7-10"
    assert chunk.embedding_text().startswith("File: app.py lines 7-10 (python)\nfunction load\n\n# Loads")


GITSUMMARIZE = ROOT.parent / "gitsummarize"


@pytest.mark.parametrize("folder", [ROOT / "example_multilang", ROOT / "docgen", GITSUMMARIZE])
def test_every_nonblank_line_of_real_files_is_covered(folder):
    if not folder.is_dir():
        pytest.skip(f"{folder} isn't on this machine")
    for f in walk_repo(folder).files:
        text = f.abs_path.read_text(encoding="utf-8")
        chunks = chunk_file(f.path, f.language, text)
        assert covered(chunks) >= nonblank(text), f.path
        assert len({c.id for c in chunks}) == len(chunks), f.path


# -------------------------------------------------------------- tree-sitter


def needs_grammar(name):
    """Skip unless the grammar is already downloaded, so tests never fetch one."""
    try:
        import tree_sitter_language_pack as pack
    except ImportError:
        pytest.skip("tree-sitter-language-pack is not installed")
    if name not in pack.downloaded_languages():
        pytest.skip(f"the {name} grammar isn't downloaded (it downloads on first real use)")


TS = """import { cache } from "react";

/** Adds two numbers. */
export function add(a: number, b: number): number {
  return a + b;
}

export interface Task {
  id: number;
  title: string;
}

type Status = "open" | "done";

const LIMIT = 3;

export const getCount = cache(async () => {
  return 1;
});

export const table = createTable("tasks", {
  id: 1,
  title: "x",
  done: false,
}, (t) => t);

export class Store {
  add(task: Task) {
    return task;
  }
}
"""


def test_typescript_splits_at_definitions():
    needs_grammar("typescript")
    assert units(chunk_file("app.ts", "typescript", TS)) == [
        ("module", "", 1, 1),
        ("function", "add", 3, 6),  # with its doc comment
        ("interface", "Task", 8, 11),
        ("type", "Status", 13, 13),
        ("module", "", 15, 15),  # a one-line constant stays with the code between definitions
        ("function", "getCount", 17, 19),  # cache(async () => ...) is a function
        ("declaration", "table", 21, 25),  # a call that merely takes a callback isn't
        ("class", "Store", 27, 31),
    ]
    assert covered(chunk_file("app.ts", "typescript", TS)) >= nonblank(TS)


def test_long_definitions_are_split_into_what_they_contain():
    needs_grammar("tsx")
    handlers = "".join(
        f"  const handle{i} = () => {{\n" + "    step();\n" * 12 + "  };\n\n" for i in range(10)
    )
    text = ("export default function Page() {\n  const [x, setX] = useState(0);\n\n"
            + handlers + "  return (\n    <div>{x}</div>\n  );\n}\n")
    chunks = chunk_file("page.tsx", "typescript", text)
    assert chunks[0].unit == "body" and chunks[0].symbol == "Page"
    assert chunks[0].text.startswith("export default function Page() {")
    assert [c.symbol for c in chunks if c.unit == "function"] == [f"Page.handle{i}" for i in range(10)]
    assert chunks[-1].unit == "body" and "<div>{x}</div>" in chunks[-1].text
    assert covered(chunks) >= nonblank(text)


def test_long_class_methods_are_labeled_methods():
    needs_grammar("typescript")
    methods = "".join(f"  m{i}() {{\n" + "    work();\n" * 12 + "  }\n\n" for i in range(10))
    text = "export class Big {\n  size = 1;\n\n" + methods + "}\n"
    chunks = chunk_file("big.ts", "typescript", text)
    assert [(c.unit, c.symbol) for c in chunks[1:]] == [("method", f"Big.m{i}") for i in range(10)]
    assert chunks[0].text.startswith("export class Big {")  # the closing } alone isn't a chunk


def test_go_and_rust_definitions():
    needs_grammar("go")
    needs_grammar("rust")
    go = "package store\n\n// Task is a to-do.\ntype Task struct {\n\tID int\n}\n\nfunc (s *Store) Add(t Task) {\n}\n\nfunc New() *Store {\n\treturn nil\n}\n"
    assert units(chunk_file("store.go", "go", go)) == [
        ("module", "", 1, 1), ("type", "Task", 3, 6), ("method", "Add", 8, 9), ("function", "New", 11, 13),
    ]
    rust = "use std::fmt;\n\n#[derive(Debug)]\nstruct Task {\n    id: u32,\n}\n\nimpl Task {\n    fn new() -> Self { Task { id: 0 } }\n}\n\nfn main() {}\n"
    assert units(chunk_file("main.rs", "rust", rust)) == [
        ("module", "", 1, 1), ("struct", "Task", 3, 6), ("impl", "", 8, 10), ("function", "main", 12, 12),
    ]


def test_falls_back_to_windows_when_tree_sitter_is_off(monkeypatch):
    monkeypatch.setenv("DOCGEN_TREE_SITTER", "0")
    assert {c.unit for c in chunk_file("app.ts", "typescript", TS)} == {"window"}


def test_falls_back_when_line_counts_would_disagree():
    needs_grammar("typescript")
    text = "const a = 1;\x0cconst b = 2;\nfunction f() {}\n"  # a form feed splits a line for Python only
    assert {c.unit for c in chunk_file("odd.ts", "typescript", text)} == {"window"}


def test_large_files_parse_without_crashing():
    """py-tree-sitter 0.26.0 crashes when a point's .row is read; line numbers
    come from byte offsets instead. A big file is where that crash showed up."""
    needs_grammar("tsx")
    text = "".join(f"export const C{i} = () => {{\n  return <p>{i}</p>;\n}};\n\n" for i in range(400))
    chunks = chunk_file("many.tsx", "typescript", text)
    assert len([c for c in chunks if c.unit == "function"]) == 400
    assert covered(chunks) >= nonblank(text)


def test_unknown_suffix_uses_windows():
    assert chunking._tree_sitter_chunks("x.unknown", "a\n", ["a"]) is None
