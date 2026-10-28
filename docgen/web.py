"""Local web UI for docgen.

A thin FastAPI layer over the pipeline (the hub, the search index, search and
Ask) - it adds no logic of its own, only JSON endpoints and a static page that
call them. The repo can be a local path or a public GitHub URL, which is
cloned into a cache folder (see docgen/repos.py). Binds to 127.0.0.1 only:
this is a local tool for one person's own machine, not a public service.

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

from docgen.ask import CONTEXT_CHARS, DEFAULT_K, ask, make_ask_completer
from docgen.embeddings import EmbeddingError
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
from docgen.llm import GenerationError, ProviderError
from docgen.repo_index import index_status, update_index
from docgen.repos import RepoError, RepoSource, resolve_repo
from docgen.search import MODES
from docgen.search import search as search_repo
from docgen.walker import walk_repo

app = FastAPI(title="docgen")
STATIC_DIR = Path(__file__).parent / "web_static"


# -------------------------------------------------------------------- repos


def _resolve_source(repo: str, fetch: bool = False) -> RepoSource:
    """A local path, or a GitHub URL (cloned on first use; pulled again when `fetch`)."""
    try:
        return resolve_repo(repo, fetch=fetch)
    except RepoError as e:
        raise HTTPException(400, str(e))


def _resolve_repo(repo: str) -> Path:
    return _resolve_source(repo).path


# --------------------------------------------------------------------- scan


@app.get("/api/scan")
def scan(repo: str):
    """Resolve the repo (cloning or pulling a GitHub URL) and list what the
    walker reads and skips."""
    source = _resolve_source(repo, fetch=True)
    path = source.path
    walked = walk_repo(path)
    return {
        "source": {"path": str(path), "github": source.github, "warning": source.warning},
        "files": {
            "count": len(walked.files),
            "languages": dict(Counter(f.language for f in walked.files).most_common()),
            "skipped": [{"path": p, "reason": r} for p, r in walked.skipped],
        },
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
    try:
        stats = update_index(path, rebuild=req.rebuild)
    except EmbeddingError as e:
        raise HTTPException(502, str(e))
    return dict(dataclasses.asdict(stats), status=index_status(path))



@app.get("/api/search")
def search_endpoint(repo: str, q: str, k: int = 8, mode: str = "hybrid", type: str | None = None):
    """Hybrid search over the search index; the Ask tab builds on this."""
    path = _resolve_repo(repo)
    if mode not in MODES:
        raise HTTPException(400, f"Unknown search mode {mode!r}; use one of {', '.join(MODES)}")
    if not index_status(path)["indexed"]:
        raise HTTPException(400, "No search index yet. Click \"Update search index\" first.")
    types = (type,) if type else None
    return {"hits": [dataclasses.asdict(h) | {"citation": h.citation}
                     for h in search_repo(path, q, k=k, mode=mode, types=types)]}


# ---------------------------------------------------------------------- ask


class AskRequest(BaseModel):
    repo: str
    question: str
    provider: str = "anthropic"
    model: str | None = None
    api_key: str | None = None  # anthropic only; used for this request only, never stored
    k: int = DEFAULT_K
    mode: str = "hybrid"


@app.post("/api/ask")
def ask_endpoint(req: AskRequest):
    """Answer a question from the search index, citing the sources it used.
    The index is brought up to date first, which is quick when little changed."""
    path = _resolve_repo(req.repo)
    if not req.question.strip():
        raise HTTPException(400, "Type a question first.")
    if req.mode not in MODES:
        raise HTTPException(400, f"Unknown search mode {req.mode!r}; use one of {', '.join(MODES)}")
    if req.provider not in CONTEXT_CHARS:
        raise HTTPException(400, f"Unknown provider: {req.provider}")
    if req.provider == "anthropic" and not req.api_key:
        raise HTTPException(400, "Anthropic provider needs an API key.")
    if not index_status(path)["indexed"]:
        raise HTTPException(400, "No search index yet. Click \"Update search index\" in the Hub tab first.")
    try:
        complete, model, meter = make_ask_completer(req.provider, req.model, req.api_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        update_index(path)
        answer = ask(path, req.question, complete, model, meter, k=max(1, min(req.k, 20)),
                     mode=req.mode, budget=CONTEXT_CHARS[req.provider])
    except (ProviderError, EmbeddingError) as e:
        raise HTTPException(502, str(e))
    return {
        "answer": answer.text,
        "sources": [dataclasses.asdict(h) | {"citation": h.citation} for h in answer.sources],
        "cited": answer.cited,
        "unknown_citations": answer.unknown_citations,
        "model": answer.model,
        "cost_usd": answer.cost_usd,
    }


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
