"""Extracts documentable chunks from a repo's Python files via the ast module.

Which files get read is decided by docgen.walker. Detects plain functions,
class methods, classes, and Flask/FastAPI-style route decorators as API
endpoints. Other languages are read as whole files by the hub instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

from docgen.models import Chunk, ChunkType
from docgen.walker import walk_repo

# Decorator name fragments that mark a function as an HTTP route.
# Matched loosely (substring) so both `@app.route` and `@router.get` work.
ROUTE_DECORATOR_HINTS = {
    "route": "GET",  # Flask's generic @app.route defaults to GET
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "patch": "PATCH",
    "delete": "DELETE",
}


def _decorator_name(dec: ast.expr) -> str:
    """Best-effort flattening of a decorator node to a dotted name string."""
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    if isinstance(dec, ast.Attribute):
        return f"{_decorator_name(dec.value)}.{dec.attr}"
    if isinstance(dec, ast.Name):
        return dec.id
    return ""


def _methods_kwarg(dec: ast.Call) -> str | None:
    """Extract methods=["POST", ...] from a decorator call, if present."""
    for kw in dec.keywords:
        if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
            methods = [
                elt.value for elt in kw.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
            if methods:
                # Prefer a non-GET method when multiple are listed, since
                # that's usually the more informative one for docs.
                for m in methods:
                    if m.upper() != "GET":
                        return m.upper()
                return methods[0].upper()
    return None


def _route_info(node: ast.FunctionDef) -> tuple[str | None, str | None]:
    """Return (http_method, route_path) if this function looks like a route handler."""
    for dec in node.decorator_list:
        name = _decorator_name(dec).lower()
        for hint, default_method in ROUTE_DECORATOR_HINTS.items():
            if name.endswith(f".{hint}") or name == hint:
                route = None
                method = default_method
                if isinstance(dec, ast.Call):
                    if dec.args:
                        first = dec.args[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            route = first.value
                    # @app.route defaults to GET unless methods=[...] says otherwise;
                    # @app.post/@router.post etc. already imply the method from `hint`.
                    if hint == "route":
                        method = _methods_kwarg(dec) or default_method
                return method, route
    return None, None


def _collect_calls(node: ast.FunctionDef) -> list[str]:
    """Collect names of functions called within a node's body (decorators excluded)."""
    calls: list[str] = []
    for stmt in node.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Name):
                    calls.append(func.id)
                elif isinstance(func, ast.Attribute):
                    calls.append(func.attr)
    # de-dupe while preserving order
    seen = set()
    ordered = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def parse_file(file_path: Path, repo_root: Path) -> list[Chunk]:
    """Parse a single Python file into a list of Chunks."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    source_lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    chunks: list[Chunk] = []

    def make_id(name: str) -> str:
        return f"{rel_path}::{name}"

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            chunks.append(Chunk(
                id=make_id(node.name),
                name=node.name,
                type=ChunkType.CLASS,
                file_path=rel_path,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                source=_get_source_segment(source_lines, node),
                docstring=ast.get_docstring(node),
                calls=[],
            ))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(Chunk(
                        id=make_id(f"{node.name}.{item.name}"),
                        name=item.name,
                        type=ChunkType.METHOD,
                        file_path=rel_path,
                        start_line=item.lineno,
                        end_line=getattr(item, "end_lineno", item.lineno),
                        source=_get_source_segment(source_lines, item),
                        signature=f"def {item.name}({ast.unparse(item.args)})",
                        docstring=ast.get_docstring(item),
                        calls=_collect_calls(item),
                        parent=node.name,
                    ))

    # Top-level functions (not nested inside a class already captured above)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method, route = _route_info(node)
            chunks.append(Chunk(
                id=make_id(node.name),
                name=node.name,
                type=ChunkType.API_ENDPOINT if route or method else ChunkType.FUNCTION,
                file_path=rel_path,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                source=_get_source_segment(source_lines, node),
                signature=f"def {node.name}({ast.unparse(node.args)})",
                docstring=ast.get_docstring(node),
                calls=_collect_calls(node),
                http_method=method,
                route=route,
            ))

    return chunks


def parse_repo(repo_path: str | Path) -> list[Chunk]:
    """Parse every Python file in a repo into a flat list of Chunks.

    Files come from the walker, so .gitignore, .docgenignore and the size and
    binary checks apply here too.
    """
    repo_root = Path(repo_path).resolve()
    all_chunks: list[Chunk] = []
    for f in walk_repo(repo_root).files:
        if f.path.endswith(".py"):
            all_chunks.extend(parse_file(f.abs_path, repo_root))
    return all_chunks
