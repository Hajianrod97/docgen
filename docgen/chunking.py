"""Splits repo files into chunks for the search index, with a rule per kind of file.

- Python is split with `ast`: each top-level function and class is a chunk,
  with its decorators and the comment lines right above it. A class too long
  for one chunk is split into its methods plus the rest of its body. The code
  between definitions (imports, constants, scripts) becomes "module" chunks,
  so every line of the file lands in some chunk. A file that doesn't parse
  falls back to line windows.
- Other code (TypeScript, JavaScript, Go, Rust, Java, C#, Ruby, ...) is parsed
  with tree-sitter, which gives a syntax tree in the same shape for every
  language, and split the same way as Python: functions, classes, interfaces
  and so on become chunks named after what they define, including the
  `const Page = () => {...}` style used for React components. A definition
  too long for one chunk is split into what it contains (a class into its
  methods, a component into its inner handlers), and the rest of its body is
  kept as "body" chunks. Grammars are downloaded on first use for each
  language; if one can't be had (offline, say), the file falls back to windows.
- Markdown is split at headings. Each chunk knows its heading path, such as
  "Install > macOS", which says more about it than the text alone.
- Everything else (and any file a parser can't handle) is split into windows
  of WINDOW_LINES lines that overlap by WINDOW_OVERLAP, so something that sits
  on a window edge still appears whole in one of them.

Any chunk still longer than MAX_CHUNK_LINES is cut into windows too. One
vector has to stand for a whole chunk, so a huge chunk ends up a blur of many
things, and the default embedding model (all-MiniLM-L6-v2) only reads its first
256 word pieces anyway. Chunk size is a trade-off worth experimenting with.
"""

from __future__ import annotations

import ast
import bisect
import os
import re
from dataclasses import dataclass

CHUNKER_VERSION = "2"  # bump when the rules change, so the index is rebuilt

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
    unit: str  # function, class, method, interface, type, module, body, section, window, ...
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


# -------------------------------------------------------------- tree-sitter

# File suffix -> tree-sitter grammar. TSX needs its own grammar for the JSX.
TREE_SITTER_GRAMMARS = {
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".rb": "ruby", ".php": "php", ".cs": "csharp", ".swift": "swift",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".lua": "lua", ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
}

# What a node type defines, matched by substring on the node's type name, first
# match wins. Grammars name things differently (function_declaration in TS,
# function_item in Rust, func_literal in Go), but the words are shared.
_KIND_WORDS = (
    ("method", "method"), ("constructor", "method"),
    ("interface", "interface"), ("trait", "trait"), ("impl", "impl"),
    ("class", "class"), ("struct", "struct"), ("enum", "enum"), ("object_declaration", "class"),
    ("type_alias", "type"), ("type_declaration", "type"), ("type_item", "type"),
    ("function", "function"), ("func", "function"),
    ("mod_item", "module"), ("namespace", "module"), ("internal_module", "module"),
)
# Type names that contain those words but aren't definitions.
_NOT_DEFINITIONS = ("call", "invocation", "expression", "reference", "parameter", "argument",
                    "identifier", "body", "_type", "signature_help")
# Siblings that belong to the definition after them, like Python's decorators.
_ATTACHED = ("comment", "attribute", "decorator", "annotation")
_CLASS_KINDS = ("class", "interface", "trait", "impl", "struct", "module")
_FUNCTION_VALUES = ("arrow_function", "function_expression", "function", "generator_function")

_CLOSERS = re.compile(r"\s*([)\]}>;,]|end)*\s*")

_parsers: dict = {}


def _tree_sitter_parser(grammar: str):
    """A cached parser for the grammar, or None if tree-sitter is turned off
    (DOCGEN_TREE_SITTER=0) or the grammar can't be loaded."""
    if os.environ.get("DOCGEN_TREE_SITTER", "1") == "0":
        return None
    if grammar not in _parsers:
        try:
            from tree_sitter_language_pack import get_parser

            _parsers[grammar] = get_parser(grammar)
        except Exception:  # not installed, unknown grammar, or the download failed
            _parsers[grammar] = None
    return _parsers[grammar]


@dataclass
class _Source:
    """The parsed file's bytes, plus where each line starts, for turning a
    node's byte offsets into line numbers. Line numbers come from byte offsets
    rather than the nodes' start_point/end_point: in py-tree-sitter 0.26.0,
    reading `.row` from those points crashes the interpreter."""

    data: bytes
    line_starts: list[int]

    @classmethod
    def of(cls, data: bytes) -> "_Source":
        starts = [0]
        starts += [i + 1 for i, b in enumerate(data) if b == 0x0A]
        return cls(data, starts)


def _text(node, source: _Source) -> str:
    return source.data[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _lines_of(node, source: _Source) -> tuple[int, int]:
    """1-based first and last line of a node."""
    first = bisect.bisect_right(source.line_starts, node.start_byte)
    last = bisect.bisect_right(source.line_starts, max(node.end_byte - 1, node.start_byte))
    return first, last


def _unwrap(node):
    """The declaration inside an `export ...` or similar wrapper."""
    if node.type in ("export_statement", "decorated_definition", "template_declaration"):
        inner = node.child_by_field_name("declaration") or node.child_by_field_name("definition")
        if inner is None:
            inner = next((c for c in node.named_children
                          if not any(word in c.type for word in _ATTACHED)), None)
        if inner is not None:
            return inner
    return node


def _name(node, source: _Source) -> str:
    """What a definition is called, as best the tree says."""
    found = node.child_by_field_name("name")
    if found is None:
        for child in node.named_children:  # const X = ..., Go's type X struct
            if child.type in ("variable_declarator", "type_spec", "const_spec", "var_spec"):
                found = child.child_by_field_name("name")
                break
    if found is None:  # C/C++: the name hides inside nested declarators
        declarator = node.child_by_field_name("declarator")
        while declarator is not None and found is None:
            if "identifier" in declarator.type:
                found = declarator
            declarator = declarator.child_by_field_name("declarator")
    if found is None:
        return ""
    name = " ".join(_text(found, source).split())
    return name if len(name) <= 80 else ""


def _declared_value(node):
    """The value of `const x = <value>` in JS/TS, else None."""
    if node.type not in ("lexical_declaration", "variable_declaration"):
        return None
    declarator = next((c for c in node.named_children if c.type == "variable_declarator"), None)
    return declarator.child_by_field_name("value") if declarator is not None else None


def _wrapped_function(call):
    """The function in `wrapper(() => ...)`, as in cache(...) or React.memo(...):
    a call whose first argument is a function. None otherwise, so a call that
    merely takes a callback somewhere, like createTable("x", {...}, (t) => ...),
    doesn't make its result a function."""
    if call.type != "call_expression":
        return None
    args = call.child_by_field_name("arguments")
    first = args.named_children[0] if args is not None and args.named_child_count else None
    return first if first is not None and first.type in _FUNCTION_VALUES else None


def _kind(node, source: _Source) -> str | None:
    """What a top-level node defines (function, class, ...), or None for
    everything else, which stays in the code between definitions."""
    value = _declared_value(node)
    if value is not None:
        if value.type in _FUNCTION_VALUES:
            return "function"
        if value.type == "class":
            return "class"
        if _wrapped_function(value) is not None:
            return "function"  # const f = cache(async () => ...), React.memo(...)
        first, last = _lines_of(node, source)
        return "declaration" if last - first >= 4 else None  # a big config object or table
    t = node.type
    if any(word in t for word in _NOT_DEFINITIONS):
        return None
    for word, kind in _KIND_WORDS:
        if word in t:
            return kind
    return None


def _body(node):
    """The node holding what a definition contains (class body, function body,
    or for `const f = () => {...}` the arrow function's body), or None."""
    value = _declared_value(node)
    if value is not None:
        node = value if value.type in _FUNCTION_VALUES + ("class",) else node
        node = _wrapped_function(value) or node
    body = node.child_by_field_name("body")
    if body is None:
        body = next((c for c in node.named_children
                     if c.type.endswith(("body", "block", "declaration_list", "field_declaration_list"))), None)
    return body if body is not None and body.named_child_count else None


def _tree_units(
    nodes: list, source: _Source, lines: list[str], span: tuple[int, int],
    parent: str | None, parent_kind: str | None,
) -> list[tuple[int, int, str, str]]:
    """(start, end, unit, symbol) for the definitions among `nodes` and the
    code between them, covering every non-blank line of `span`."""
    units = []
    cursor = span[0]
    gap_unit = "body" if parent else "module"

    def gap(until: int) -> None:
        trimmed = _trim(lines, cursor, until)
        # A run of closing brackets (the `}` ending a long class) is noise on its own.
        if trimmed and not all(_CLOSERS.fullmatch(line) for line in lines[trimmed[0] - 1:trimmed[1]]):
            units.append((*trimmed, gap_unit, parent or ""))

    for i, node in enumerate(nodes):
        inner = _unwrap(node)
        kind = _kind(inner, source)
        if kind is None:
            continue
        start, end = _lines_of(node, source)
        j = i  # take the comments, decorators and attributes right above it
        while (j > 0 and any(word in nodes[j - 1].type for word in _ATTACHED)
               and _lines_of(nodes[j - 1], source)[1] >= start - 1):
            j -= 1
            start = _lines_of(nodes[j], source)[0]
        start = max(start, cursor)
        if start > end:
            continue
        gap(start - 1)
        name = _name(inner, source)
        symbol = f"{parent}.{name}" if parent and name else (name or parent or "")
        if kind == "function" and parent_kind in _CLASS_KINDS:
            kind = "method"
        body = _body(inner) if end - start + 1 > MAX_CHUNK_LINES else None
        if body is not None:
            units += _tree_units(body.named_children, source, lines, (start, end), symbol or None, kind)
        else:
            units.append((start, end, kind, symbol))
        cursor = end + 1
    gap(span[1])
    return units


def _tree_sitter_chunks(path: str, text: str, lines: list[str]) -> list[tuple[int, int, str, str]] | None:
    suffix = os.path.splitext(path)[1].lower()
    grammar = TREE_SITTER_GRAMMARS.get(suffix)
    parser = _tree_sitter_parser(grammar) if grammar else None
    if parser is None:
        return None
    # tree-sitter counts lines at "\n" only; skip files where splitlines() would
    # disagree (a stray form feed or Unicode line separator), rather than misplace chunks.
    if len(text.replace("\r\n", "\n").split("\n")) - text.endswith(("\n", "\r\n")) != len(lines):
        return None
    source = _Source.of(text.encode("utf-8"))
    try:
        tree = parser.parse(source.data)
    except Exception:
        return None
    # `tree` stays referenced while its nodes are in use, since they point into it.
    return _tree_units(tree.root_node.named_children, source, lines, (1, len(lines)), None, None)


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
    else:
        units = _tree_sitter_chunks(path, text, lines)
    structured = units is not None  # split along the file's own structure
    if units is None:
        trimmed = _trim(lines, 1, len(lines))
        units = [(*trimmed, "window", "")] if trimmed else []
    limit = MAX_CHUNK_LINES if structured else WINDOW_LINES

    chunks = []
    for start, end, unit, symbol in units:
        ranges = _windows(lines, start, end) if end - start + 1 > limit else [(start, end)]
        for first, last in ranges:
            chunks.append(CodeChunk(
                path=path, language=language, start_line=first, end_line=last,
                text="\n".join(lines[first - 1:last]), unit=unit, symbol=symbol,
            ))
    return chunks
