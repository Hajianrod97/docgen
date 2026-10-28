"""Splits repo files into chunks for the search index, with a rule per kind of file.

- Python is split with `ast`: each top-level function and class is a chunk,
  with its decorators and the comment lines right above it. A class too long
  for one chunk is split into its methods plus the rest of its body. The code
  between definitions (imports, constants, scripts) becomes "module" chunks,
  so every line of the file lands in some chunk. A file that doesn't parse
  falls back to line windows.
- Markdown is split at headings. Each chunk knows its heading path, such as
  "Install > macOS", which says more about it than the text alone.
- Everything else is split into windows of WINDOW_LINES lines that overlap by
  WINDOW_OVERLAP, so something that sits on a window edge still appears whole
  in one of them.

Any chunk still longer than MAX_CHUNK_LINES is cut into windows too. One
vector has to stand for a whole chunk, so a huge chunk ends up a blur of many
things, and the default embedding model (all-MiniLM-L6-v2) only reads its first
256 word pieces anyway. Chunk size is a trade-off worth experimenting with.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

CHUNKER_VERSION = "1"  # bump when the rules change, so the index is rebuilt

WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_CHUNK_LINES = 120

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass
class CodeChunk:
    path: str
    language: str
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str
    unit: str  # function, class, method, module, section, window
    symbol: str = ""  # e.g. "TaskStore.add", or a heading path for Markdown

    @property
    def id(self) -> str:
        return f"code:{self.path}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        """What gets embedded: a header naming where the chunk is, then its text."""
        where = f"{self.path} lines {self.start_line}-{self.end_line} ({self.language})"
        label = f"{self.unit} {self.symbol}" if self.symbol else self.unit
        return f"File: {where}\n{label}\n\n{self.text}"


def _windows(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """(first, last) line ranges of WINDOW_LINES covering start..end, overlapping."""
    if end - start + 1 <= WINDOW_LINES:
        return [(start, end)]
    ranges = []
    first = start
    while True:
        last = min(first + WINDOW_LINES - 1, end)
        ranges.append((first, last))
        if last >= end:
            return ranges
        first = last - WINDOW_OVERLAP + 1


def _blank(lines: list[str], start: int, end: int) -> bool:
    return all(not line.strip() for line in lines[start - 1:end])


def _trim(lines: list[str], start: int, end: int) -> tuple[int, int] | None:
    """Drop blank lines at both ends of a range; None if nothing is left."""
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    return (start, end) if start <= end else None


# ------------------------------------------------------------------- Python


def _def_start(node: ast.AST, lines: list[str]) -> int:
    """First line of a definition, including decorators and the comments right above it."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return start


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _python_units(
    body: list[ast.stmt], lines: list[str], span: tuple[int, int], parent: str | None,
) -> list[tuple[int, int, str, str]]:
    """(start, end, unit, symbol) for the definitions in `body` and the code
    between them, covering every non-blank line of `span`."""
    units = []
    cursor = span[0]
    gap_unit = "class body" if parent else "module"
    gap_symbol = parent or ""

    def gap(until: int) -> None:
        trimmed = _trim(lines, cursor, until)
        if trimmed:
            units.append((*trimmed, gap_unit, gap_symbol))

    for node in body:
        if not isinstance(node, _DEFS):
            continue
        start = max(_def_start(node, lines), cursor)
        end = node.end_lineno
        gap(start - 1)
        name = f"{parent}.{node.name}" if parent else node.name
        if isinstance(node, ast.ClassDef) and end - start + 1 > MAX_CHUNK_LINES:
            units += _python_units(node.body, lines, (start, end), name)
        elif isinstance(node, ast.ClassDef):
            units.append((start, end, "class", name))
        else:
            units.append((start, end, "method" if parent else "function", name))
        cursor = end + 1
    gap(span[1])
    return units


def _python_chunks(text: str, lines: list[str]) -> list[tuple[int, int, str, str]] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    return _python_units(tree.body, lines, (1, len(lines)), None)


# ----------------------------------------------------------------- Markdown


def _markdown_chunks(lines: list[str]) -> list[tuple[int, int, str, str]]:
    headings = []  # (line number, level, title), outside code fences
    in_code = False
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        elif not in_code and (m := _HEADING.match(line)):
            headings.append((n, len(m.group(1)), m.group(2)))

    units = []
    if not headings or headings[0][0] > 1:
        first = headings[0][0] - 1 if headings else len(lines)
        trimmed = _trim(lines, 1, first)
        if trimmed:
            units.append((*trimmed, "section", ""))
    stack: list[tuple[int, str]] = []  # (level, title) of the enclosing headings
    for i, (n, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[i + 1][0] - 1 if i + 1 < len(headings) else len(lines)
        trimmed = _trim(lines, n, end)
        if trimmed:
            units.append((*trimmed, "section", " > ".join(t for _, t in stack)))
    return units


# --------------------------------------------------------------------- main


def chunk_file(path: str, language: str, text: str) -> list[CodeChunk]:
    """Split one file into chunks. Every non-blank line is in at least one."""
    lines = text.splitlines()
    if not lines or _blank(lines, 1, len(lines)):
        return []

    units = None
    if language == "python":
        units = _python_chunks(text, lines)
    elif language == "markdown":
        units = _markdown_chunks(lines)
    if units is None:
        trimmed = _trim(lines, 1, len(lines))
        units = [(*trimmed, "window", "")] if trimmed else []
    limit = MAX_CHUNK_LINES if language in ("python", "markdown") else WINDOW_LINES

    chunks = []
    for start, end, unit, symbol in units:
        ranges = _windows(lines, start, end) if end - start + 1 > limit else [(start, end)]
        for first, last in ranges:
            chunks.append(CodeChunk(
                path=path, language=language, start_line=first, end_line=last,
                text="\n".join(lines[first - 1:last]), unit=unit, symbol=symbol,
            ))
    return chunks
