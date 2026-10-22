"""Local web UI for docgen.

A thin FastAPI layer over the existing pipeline (parser, indexer, retriever,
generator, writer) - it adds no new logic of its own, only JSON endpoints and
a static page that call them. Binds to 127.0.0.1
only: this is a local tool for one person's own machine, not a public service.

Run with: python -m docgen.web
Then open: http://127.0.0.1:8765
"""

from __future__ import annotations

import dataclasses
import webbrowser
from collections import Counter
from pathlib import Path
from threading import Timer

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docgen.generator import (
    GenerationError,
    ProviderError,
    ProviderSetupError,
    build_prompt,
    generate_for_chunk,
    generate_overview,
    generated_path,
    load_generated,
    make_anthropic_completer,
    make_ollama_completer,
    output_kind,
    repo_structure_hash,
    save_generated,
    source_hash,
)
from docgen.indexer import build_index, ensure_unique_ids, index_path, open_collection
from docgen.models import Chunk
from docgen.parser import parse_repo
from docgen.retriever import ContextItem, get_context
from docgen.walker import walk_repo
from docgen.writer import ApplyPlan, diff_lines, plan_apply, write_plan

app = FastAPI(title="docgen")
STATIC_DIR = Path(__file__).parent / "web_static"

DEFAULT_LIMIT = 20  # generate-all safety cap: a browser click shouldn't accidentally
                     # trigger hundreds of paid API calls on a large repo


# --------------------------------------------------------------- JSON shapes


def _chunk_dict(c: Chunk, generated: dict) -> dict:
    entry = generated.get("items", {}).get(c.id)
    stale = bool(entry) and entry.get("source_hash") != source_hash(c)
    return {
        "id": c.id,
        "name": c.name,
        "type": c.type.value,
        "file_path": c.file_path,
        "parent": c.parent,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "has_docstring": bool(c.docstring),
        "http_method": c.http_method,
        "route": c.route,
        "generated": bool(entry) and not stale,
        "stale": stale,
    }


def _context_item_dict(item: ContextItem) -> dict:
    d = dataclasses.asdict(item)
    d.pop("calls", None)  # internal only; not useful to display
    return d


# ------------------------------------------------------------------- errors


def _resolve_repo(repo: str) -> Path:
    path = Path(repo).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {repo}")
    return path


def _get_completer(provider: str, model: str | None, api_key: str | None):
    if provider == "anthropic":
        if not api_key:
            raise HTTPException(400, "Anthropic provider needs an API key.")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        def complete(system: str, user: str) -> str:
            try:
                resp = client.messages.create(
                    model=model or "claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            except anthropic.AuthenticationError as e:
                raise ProviderSetupError("Authentication failed. Check the API key.") from e
            except anthropic.APIError as e:
                raise ProviderError(str(e)) from e

        return complete, (model or "claude-haiku-4-5-20251001")
    elif provider == "ollama":
        from docgen.generator import DEFAULT_OLLAMA_MODEL

        m = model or DEFAULT_OLLAMA_MODEL
        return make_ollama_completer(m), m
    raise HTTPException(400, f"Unknown provider: {provider}")


# --------------------------------------------------------------------- scan


@app.get("/api/scan")
def scan(repo: str):
    path = _resolve_repo(repo)
    chunks = parse_repo(path)
    ensure_unique_ids(chunks)
    generated = load_generated(path)
    walked = walk_repo(path)
    return {
        "chunks": [_chunk_dict(c, generated) for c in chunks],
        "indexed": index_path(path).exists(),
        "files": {
            "count": len(walked.files),
            "languages": dict(Counter(f.language for f in walked.files).most_common()),
            "skipped": [{"path": p, "reason": r} for p, r in walked.skipped],
        },
    }


# -------------------------------------------------------------------- index


class IndexRequest(BaseModel):
    repo: str


@app.post("/api/index")
def index_repo(req: IndexRequest):
    path = _resolve_repo(req.repo)
    chunks = parse_repo(path)
    if not chunks:
        raise HTTPException(400, "No Python chunks found in this repo.")
    ensure_unique_ids(chunks)
    collection = build_index(chunks, path)
    return {"count": collection.count()}


# ------------------------------------------------------------------ context


@app.get("/api/context")
def context(repo: str, target: str):
    path = _resolve_repo(repo)
    chunks = parse_repo(path)
    ensure_unique_ids(chunks)
    chunk = next((c for c in chunks if c.id == target), None)
    if chunk is None:
        raise HTTPException(404, f"No chunk with id {target!r}")
    try:
        collection = open_collection(path)
    except Exception:
        raise HTTPException(400, "No index found. Build the index first.")
    ctx = get_context(collection, chunk)
    _, prompt_preview = build_prompt(chunk, ctx)
    return {
        "chunk": {"id": chunk.id, "name": chunk.name, "source": chunk.source},
        "items": [_context_item_dict(i) for i in ctx.all_items()],
        "prompt_preview": prompt_preview,
    }


# ----------------------------------------------------------------- generate


class GenerateRequest(BaseModel):
    repo: str
    provider: str = "anthropic"
    model: str | None = None
    api_key: str | None = None  # anthropic only; used for this request only, never stored
    target: str | None = None  # one chunk id, or None for "all undocumented"
    limit: int = DEFAULT_LIMIT
    force: bool = False


@app.post("/api/generate")
def generate(req: GenerateRequest):
    path = _resolve_repo(req.repo)
    chunks = parse_repo(path)
    ensure_unique_ids(chunks)
    by_id = {c.id: c for c in chunks}

    try:
        collection = open_collection(path)
    except Exception:
        raise HTTPException(400, "No index found. Build the index first.")

    if req.target:
        chunk = by_id.get(req.target)
        if chunk is None:
            raise HTTPException(404, f"No chunk with id {req.target!r}")
        targets = [chunk]
    else:
        targets = [c for c in chunks if not c.docstring]

    data = load_generated(path)
    if not req.force:
        targets = [
            c for c in targets
            if data["items"].get(c.id, {}).get("source_hash") != source_hash(c)
        ]
    targets = targets[: req.limit] if req.limit else targets

    complete, resolved_model = _get_completer(req.provider, req.model, req.api_key)
    data["model"] = f"{req.provider}:{resolved_model}"

    results = []
    for chunk in targets:
        ctx = get_context(collection, chunk)
        try:
            text = generate_for_chunk(chunk, ctx, complete)
        except ProviderSetupError as e:
            # Unreachable server / bad key: stop rather than fail every remaining chunk the same way.
            save_generated(path, data)
            raise HTTPException(502, str(e))
        except (GenerationError, ProviderError) as e:
            results.append({"id": chunk.id, "ok": False, "error": str(e)})
            continue
        data["items"][chunk.id] = {
            "kind": output_kind(chunk),
            "text": text,
            "file_path": chunk.file_path,
            "start_line": chunk.start_line,
            "source_hash": source_hash(chunk),
        }
        save_generated(path, data)
        results.append({"id": chunk.id, "ok": True, "text": text})

    return {"results": results, "model": resolved_model}


@app.get("/api/generated")
def generated(repo: str):
    path = _resolve_repo(repo)
    return load_generated(path)


class OverviewRequest(BaseModel):
    repo: str
    provider: str = "anthropic"
    model: str | None = None
    api_key: str | None = None
    force: bool = False


@app.post("/api/overview")
def generate_overview_endpoint(req: OverviewRequest):
    path = _resolve_repo(req.repo)
    chunks = parse_repo(path)
    if not chunks:
        raise HTTPException(400, "No Python chunks found in this repo.")
    ensure_unique_ids(chunks)

    data = load_generated(path)
    struct_hash = repo_structure_hash(chunks)
    existing = data.get("overview")
    if existing and existing.get("structure_hash") == struct_hash and not req.force:
        return {"text": existing["text"], "cached": True}

    complete, _ = _get_completer(req.provider, req.model, req.api_key)
    try:
        text = generate_overview(chunks, complete, data.get("items", {}))
    except ProviderSetupError as e:
        raise HTTPException(502, str(e))
    except (GenerationError, ProviderError) as e:
        raise HTTPException(502, str(e))

    data["overview"] = {"text": text, "structure_hash": struct_hash}
    save_generated(path, data)
    return {"text": text, "cached": False}


@app.get("/api/docs")
def docs(repo: str):
    """The generated documentation as a reader would want to see it: docstrings
    grouped by file, and the API reference markdown, both pulled straight from
    generated.json (no apply/write needed). Used by the web UI's Documentation tab.
    """
    path = _resolve_repo(repo)
    data = load_generated(path)
    chunks = {c.id: c for c in parse_repo(path)}
    ensure_unique_ids(chunks_list := list(chunks.values()))
    chunks = {c.id: c for c in chunks_list}

    by_file: dict[str, list[dict]] = {}
    endpoints: list[tuple[Chunk, str]] = []
    for id_, item in data.get("items", {}).items():
        chunk = chunks.get(id_)
        if chunk is None:
            continue
        if item["kind"] == "docstring":
            by_file.setdefault(chunk.file_path, []).append({
                "id": chunk.id,
                "name": chunk.name,
                "type": chunk.type.value,
                "parent": chunk.parent,
                "start_line": chunk.start_line,
                "text": item["text"],
                "stale": item.get("source_hash") != source_hash(chunk),
            })
        elif item["kind"] == "endpoint_doc":
            endpoints.append((chunk, item["text"]))

    for entries in by_file.values():
        entries.sort(key=lambda e: e["start_line"])

    api_md = None
    if endpoints:
        from docgen.writer import _build_api_md

        probe = ApplyPlan()
        _build_api_md(path, endpoints, "docs", force=True, plan=probe)
        api_md = probe.api.new_text if probe.api else None

    overview_entry = data.get("overview")
    overview_text = overview_entry["text"] if overview_entry else None
    overview_stale = (
        bool(overview_entry) and overview_entry.get("structure_hash") != repo_structure_hash(chunks_list)
    )

    return {
        "files": by_file,
        "api_md": api_md,
        "overview": overview_text,
        "overview_stale": overview_stale,
    }


# --------------------------------------------------------------- plan/apply


def _plan_dict(plan: ApplyPlan, repo: Path) -> dict:
    return {
        "overview": (
            {
                "path": plan.overview.path.relative_to(repo).as_posix(),
                "content": plan.overview.new_text,
                "is_new": plan.overview.old_text is None,
            }
            if plan.overview and plan.overview.changed
            else None
        ),
        "changes": [
            {
                "path": c.rel,
                "diff": diff_lines(c),
                "docstring_count": len(c.applied),
            }
            for c in plan.changes
        ],
        "api": (
            {
                "path": plan.api.path.relative_to(repo).as_posix(),
                "content": plan.api.new_text,
                "is_new": plan.api.old_text is None,
                "endpoints": plan.api.endpoints,
            }
            if plan.api and plan.api.changed
            else None
        ),
        "skipped": [{"id": s.id, "reason": s.reason} for s in plan.skipped],
        "docstring_count": plan.docstring_count,
    }


@app.get("/api/plan")
def plan(
    repo: str,
    docs_dir: str = "docs",
    api_md: bool = True,
    overview: bool = True,
    force: bool = False,
):
    path = _resolve_repo(repo)
    data = load_generated(path)
    try:
        p = plan_apply(
            path, data, docs_dir=docs_dir, include_api_md=api_md,
            include_overview=overview, force=force,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _plan_dict(p, path)


class ApplyRequest(BaseModel):
    repo: str
    docs_dir: str = "docs"
    api_md: bool = True
    overview: bool = True
    force: bool = False


@app.post("/api/apply")
def apply(req: ApplyRequest):
    path = _resolve_repo(req.repo)
    data = load_generated(path)
    try:
        p = plan_apply(
            path, data, docs_dir=req.docs_dir, include_api_md=req.api_md,
            include_overview=req.overview, force=req.force,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    write_plan(p)
    return _plan_dict(p, path)


# --------------------------------------------------------------- static UI

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass  # no browser available (e.g. a headless machine); the URL is printed either way


def main():
    import os

    import uvicorn

    port = int(os.environ.get("DOCGEN_WEB_PORT", 8765))
    url = f"http://127.0.0.1:{port}"
    print(f"docgen web UI running at {url} (Ctrl+C to stop)")
    if not os.environ.get("DOCGEN_NO_BROWSER"):
        Timer(1.0, _open_browser, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
