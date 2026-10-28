"""Local web UI for docgen.

A thin FastAPI layer over the existing pipeline (parser, indexer, retriever,
generator, writer, and the hub) - it adds no new logic of its own, only JSON
endpoints and a static page that call them. The repo can be a local path or a
public GitHub URL, which is cloned into a cache folder (see docgen/repos.py). Binds to 127.0.0.1
only: this is a local tool for one person's own machine, not a public service.

Run with: python -m docgen.web
Then open: http://127.0.0.1:8765
"""

from __future__ import annotations

import dataclasses
import threading
import time
import webbrowser
from collections import Counter
from pathlib import Path
from threading import Timer

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docgen.generator import (
    DEFAULT_MODEL,
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
from docgen.hub import (
    HubStopped,
    build_hub,
    default_models,
    hub_tree,
    load_hub,
    make_hub_models,
    overview_status,
    plan_hub,
)
from docgen.hub_export import default_export_dir, export_hub
from docgen.indexer import build_index, ensure_unique_ids, index_path, open_collection
from docgen.models import Chunk
from docgen.parser import parse_repo
from docgen.repo_index import index_status, update_index
from docgen.repos import RepoError, RepoSource, resolve_repo
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


def _resolve_source(repo: str, fetch: bool = False) -> RepoSource:
    """A local path, or a GitHub URL (cloned on first use; pulled again when `fetch`)."""
    try:
        return resolve_repo(repo, fetch=fetch)
    except RepoError as e:
        raise HTTPException(400, str(e))


def _resolve_repo(repo: str) -> Path:
    return _resolve_source(repo).path


def _get_completer(provider: str, model: str | None, api_key: str | None):
    if provider == "anthropic":
        if not api_key:
            raise HTTPException(400, "Anthropic provider needs an API key.")
        m = model or DEFAULT_MODEL
        return make_anthropic_completer(m, api_key=api_key), m
    elif provider == "ollama":
        from docgen.generator import DEFAULT_OLLAMA_MODEL

        m = model or DEFAULT_OLLAMA_MODEL
        return make_ollama_completer(m), m
    raise HTTPException(400, f"Unknown provider: {provider}")


# --------------------------------------------------------------------- scan


@app.get("/api/scan")
def scan(repo: str):
    source = _resolve_source(repo, fetch=True)
    path = source.path
    chunks = parse_repo(path)
    ensure_unique_ids(chunks)
    generated = load_generated(path)
    walked = walk_repo(path)
    return {
        "chunks": [_chunk_dict(c, generated) for c in chunks],
        "indexed": index_path(path).exists(),
        "source": {"path": str(path), "github": source.github, "warning": source.warning},
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


# ---------------------------------------------------------------------- hub
#
# A hub build can take minutes, so it runs in a background thread and the page
# polls /api/hub/status. One build per repo at a time; the API key lives only
# in that thread's completers and is gone when the build ends.

_hub_jobs: dict[str, dict] = {}
_hub_jobs_lock = threading.Lock()
MAX_REPORTED_FAILURES = 50


def _job_view(job: dict | None) -> dict | None:
    if job is None:
        return None
    return {k: job[k] for k in ("running", "progress", "error", "stopped", "failures", "started_at", "finished_at")}


def _hub_plan(path: Path, regenerate: bool = False):
    try:
        return plan_hub(path, regenerate=regenerate)
    except GenerationError as e:
        raise HTTPException(400, str(e))


@app.get("/api/hub")
def hub_view(repo: str):
    source = _resolve_source(repo)
    plan = _hub_plan(source.path)
    hub = load_hub(source.path)
    overview = hub["overview"]
    return {
        "name": source.github or source.path.name,
        "github": source.github,
        "tree": hub_tree(plan, hub),
        "overview": dict(overview, status=overview_status(plan, hub)) if overview else None,
        "overview_status": overview_status(plan, hub),
        "files": hub["files"],
        "dirs": hub["dirs"],
        "errors": hub["errors"],
        "source_urls": {p: source.source_url(p) for p in plan.files} if source.is_clone else None,
        "export_dir": str(default_export_dir(source)),
        "job": _job_view(_hub_jobs.get(str(source.path))),
        "search_index": index_status(source.path),
    }


def _models_or_400(provider: str, file_model: str | None, top_model: str | None):
    try:
        default_file, default_top = default_models(provider)
    except ValueError as e:
        raise HTTPException(400, str(e))
    file_model = file_model or default_file
    top_model = top_model or (default_top if provider == "anthropic" else file_model)
    return file_model, top_model


@app.get("/api/hub/estimate")
def hub_estimate(
    repo: str,
    provider: str = "anthropic",
    file_model: str | None = None,
    top_model: str | None = None,
    regenerate: bool = False,
):
    path = _resolve_repo(repo)
    file_model, top_model = _models_or_400(provider, file_model, top_model)
    plan = _hub_plan(path, regenerate=regenerate)
    est = plan.estimate(file_model, top_model)
    return dict(
        est,
        file_model=file_model,
        top_model=top_model,
        files=len(plan.files),
        todo_files=len(plan.todo_files),
        todo_dirs=len(plan.todo_dirs),
        todo_overview=plan.todo_overview,
    )


class HubBuildRequest(BaseModel):
    repo: str
    provider: str = "anthropic"
    file_model: str | None = None
    top_model: str | None = None
    api_key: str | None = None  # anthropic only; used for this build only, never stored
    regenerate: bool = False


@app.post("/api/hub/build")
def hub_build(req: HubBuildRequest):
    path = _resolve_repo(req.repo)
    if req.provider == "anthropic" and not req.api_key:
        raise HTTPException(400, "Anthropic provider needs an API key.")
    file_model, top_model = _models_or_400(req.provider, req.file_model, req.top_model)
    models = make_hub_models(req.provider, file_model, top_model, api_key=req.api_key)
    plan = _hub_plan(path, regenerate=req.regenerate)

    key = str(path)
    stop = threading.Event()
    job = {
        "running": True, "progress": None, "error": None, "stopped": False,
        "failures": [], "started_at": time.time(), "finished_at": None, "stop": stop,
    }
    with _hub_jobs_lock:
        current = _hub_jobs.get(key)
        if current and current["running"]:
            raise HTTPException(409, "A hub build is already running for this repo.")
        _hub_jobs[key] = job

    def on_progress(event: dict) -> None:
        job["progress"] = event
        if event.get("error") and len(job["failures"]) < MAX_REPORTED_FAILURES:
            job["failures"].append({"stage": event["stage"], "item": event.get("item"), "error": event["error"]})

    def run() -> None:
        try:
            build_hub(path, models, plan, on_progress=on_progress,
                      should_stop=stop.is_set, regenerate=req.regenerate)
        except HubStopped:
            job["stopped"] = True
        except ProviderError as e:
            job["error"] = str(e)
        except Exception as e:  # surface anything unexpected in the UI instead of a silent thread death
            job["error"] = f"Unexpected error: {type(e).__name__}: {e}"
        finally:
            job["finished_at"] = time.time()
            job["running"] = False

    threading.Thread(target=run, name=f"hub-build:{path.name}", daemon=True).start()
    return _job_view(job)


@app.get("/api/hub/status")
def hub_status(repo: str):
    path = _resolve_repo(repo)
    return {"job": _job_view(_hub_jobs.get(str(path)))}


class HubRepoRequest(BaseModel):
    repo: str


@app.post("/api/hub/stop")
def hub_stop(req: HubRepoRequest):
    path = _resolve_repo(req.repo)
    job = _hub_jobs.get(str(path))
    if job and job["running"]:
        job["stop"].set()  # in-flight calls finish; queued ones are skipped
    return {"job": _job_view(job)}


@app.post("/api/hub/export")
def hub_export(req: HubRepoRequest):
    source = _resolve_source(req.repo)
    plan = _hub_plan(source.path)
    hub = load_hub(source.path)
    if not hub["files"] and not hub["dirs"] and not hub["overview"]:
        raise HTTPException(400, "Nothing to export yet. Build the hub first.")
    out_dir = default_export_dir(source)
    written = export_hub(plan, hub, out_dir, source)
    return {"dir": str(out_dir), "files": [p.relative_to(out_dir).as_posix() for p in written]}


# ------------------------------------------------------------- search index


class SearchIndexRequest(BaseModel):
    repo: str
    rebuild: bool = False


@app.post("/api/search-index")
def search_index(req: SearchIndexRequest):
    """Index every file's chunks and the hub's summaries for Ask. Only what
    changed since the last run is embedded again."""
    path = _resolve_repo(req.repo)
    stats = update_index(path, rebuild=req.rebuild)
    return dict(dataclasses.asdict(stats), status=index_status(path))


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
