"""Finds which repo files import which, roughly: regular expressions over import
lines, for Python and JavaScript/TypeScript only, keeping only imports of files
inside the repo.
"""

from __future__ import annotations

import posixpath
import re

from docgen.walker import RepoFile

_PY_IMPORT = re.compile(r"^\s*(?:from\s+(\.*[\w.]*)\s+import\s+([\w*, ()]+)|import\s+([\w., ]+))", re.M)
_JS_IMPORT = re.compile(
    r"""(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]"""
    r"""|import\s*['"]([^'"]+)['"]"""
    r"""|(?:require|import)\s*\(\s*['"]([^'"]+)['"]\s*\)"""
)
_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")


def _python_candidates(module: str, file_dir: str) -> list[str]:
    """Repo paths a Python module name could refer to, most specific first."""
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        base = file_dir
        for _ in range(dots - 1):
            base = posixpath.dirname(base)
        rest = module[dots:]
        stem = posixpath.join(base, *rest.split(".")) if rest else base
    else:
        stem = posixpath.join(*module.split("."))
    return [f"{stem}.py", f"{stem}/__init__.py"]


def _resolve_python(module: str, file_dir: str, known: set[str]) -> str | None:
    for candidate in _python_candidates(module, file_dir):
        if candidate in known:
            return candidate
    return None


def _python_imports(text: str, file_dir: str, known: set[str]) -> set[str]:
    found = set()
    for m in _PY_IMPORT.finditer(text):
        if m.group(1) is not None:
            module = m.group(1)
            # `from pkg import mod` may import submodules rather than names;
            # only fall back to the package itself when none of them is a file.
            prefix = module if module.endswith(".") else module + "."
            names = [n.strip(" ()").split(" as ")[0].strip() for n in m.group(2).split(",")]
            subs = {_resolve_python(prefix + n, file_dir, known) for n in names if n and n != "*"}
            subs.discard(None)
            if subs:
                found |= subs
            elif (target := _resolve_python(module, file_dir, known)):
                found.add(target)
        else:
            for name in m.group(3).split(","):
                target = _resolve_python(name.strip().split(" as ")[0].strip(), file_dir, known)
                if target:
                    found.add(target)
    return found


def _js_imports(text: str, file_dir: str, known: set[str]) -> set[str]:
    found = set()
    for m in _JS_IMPORT.finditer(text):
        spec = next(g for g in m.groups() if g)
        if not spec.startswith("."):
            continue  # a package, not a repo file
        stem = posixpath.normpath(posixpath.join(file_dir, spec))
        candidates = [stem] + [stem + ext for ext in _JS_EXTENSIONS]
        candidates += [f"{stem}/index{ext}" for ext in _JS_EXTENSIONS]
        for candidate in candidates:
            if candidate in known:
                found.add(candidate)
                break
    return found


def import_graph(files: list[RepoFile], texts: dict[str, str]) -> dict[str, set[str]]:
    """Map each file path to the repo files it imports."""
    known = {f.path for f in files}
    graph: dict[str, set[str]] = {}
    for f in files:
        file_dir = posixpath.dirname(f.path)
        text = texts[f.path]
        if f.language == "python":
            edges = _python_imports(text, file_dir, known)
        elif f.language in ("javascript", "typescript", "vue", "svelte"):
            edges = _js_imports(text, file_dir, known)
        else:
            edges = set()
        edges.discard(f.path)
        graph[f.path] = edges
    return graph


def reverse_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """Map each file path to the repo files that import it."""
    reverse: dict[str, set[str]] = {path: set() for path in graph}
    for path, targets in graph.items():
        for target in targets:
            reverse.setdefault(target, set()).add(path)
    return reverse
