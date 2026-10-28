"""Builds the search index behind "Ask the repo": every file's chunks, plus the hub's summaries.

The index is a ChromaDB collection in <repo>/.docgen/chroma. Each entry has
a `type` in its metadata:

- "code": a chunk of a file, from docgen.chunking. Its metadata has the path
  and line range, which is what answers cite as `file:line`.
- "summary": a piece of a hub summary (a file, a folder or the overview),
  split at its headings. Summaries answer "what is this for" questions well,
  since they say in plain words what the code only implies.

Updates are incremental. <repo>/.docgen/repo_index.json records each file's
content hash and the ids of its chunks; a run re-chunks and re-embeds only
files whose hash changed, deletes the chunks of removed files, and does the
same for summaries. Changing the chunking rules or the embedding model
rebuilds everything, since vectors from different models can't be mixed.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from docgen.chunking import CHUNKER_VERSION, chunk_file
from docgen.embeddings import get_embedding_function
from docgen.hub import ROOT, load_hub
from docgen.walker import walk_repo

COLLECTION_NAME = "docgen_repo"
INDEX_DIRNAME = ".docgen/chroma"
MANIFEST_FILENAME = ".docgen/repo_index.json"
INDEX_VERSION = f"{CHUNKER_VERSION}/1"  # chunking rules / document layout
_BATCH = 128


@dataclass
class Entry:
    """One document to store: its id, the text to embed, and its metadata."""

    id: str
    text: str
    metadata: dict


@dataclass
class IndexStats:
    rebuilt: bool = False
    files_indexed: int = 0  # new or changed files
    files_removed: int = 0
    files_unchanged: int = 0
    summaries_indexed: int = 0
    summaries_removed: int = 0
    entries_added: int = 0
    entries_removed: int = 0
    total_entries: int = 0
    by_type: dict = field(default_factory=dict)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embedding_name() -> str:
    fn = get_embedding_function()
    if fn is None:
        return "chroma-default:all-MiniLM-L6-v2"
    return getattr(fn, "docgen_id", None) or fn.name()


def get_client(repo_path: str | Path):
    path = Path(repo_path).resolve() / INDEX_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def manifest_path(repo_path: str | Path) -> Path:
    return Path(repo_path).resolve() / MANIFEST_FILENAME


def _empty_manifest() -> dict:
    return {"version": INDEX_VERSION, "embedding": embedding_name(), "files": {}, "summaries": {}}


def load_manifest(repo_path: str | Path) -> dict | None:
    path = manifest_path(repo_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(repo_path: str | Path, manifest: dict) -> None:
    path = manifest_path(repo_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- documents


def code_entries(path: str, language: str, text: str) -> list[Entry]:
    entries = []
    seen = set()
    for chunk in chunk_file(path, language, text):
        if chunk.id in seen:  # two units on the same lines: keep the first
            continue
        seen.add(chunk.id)
        entries.append(Entry(chunk.id, chunk.embedding_text(), {
            "type": "code", "path": path, "language": language,
            "start_line": chunk.start_line, "end_line": chunk.end_line,
            "unit": chunk.unit, "symbol": chunk.symbol,
        }))
    return entries


def _split_markdown(md: str) -> list[tuple[str, str]]:
    """(heading, text) parts of a summary, split at its headings."""
    parts: list[tuple[str, list[str]]] = [("", [])]
    in_code = False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        stripped = line.lstrip("#")
        if not in_code and line.startswith("#") and stripped.startswith(" "):
            parts.append((stripped.strip(), []))
        else:
            parts[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in parts if "\n".join(body).strip()]


def summary_sources(hub: dict, existing_files: set[str], existing_dirs: set[str]) -> dict[str, tuple[str, str, str]]:
    """source id -> (level, path, full text) for every hub summary of something
    that still exists."""
    sources = {}
    for path, entry in hub["files"].items():
        if path in existing_files and entry.get("summary"):
            text = f"Purpose: {entry.get('purpose', '')}\n\n{entry['summary']}"
            sources[f"file:{path}"] = ("file", path, text)
    for d, entry in hub["dirs"].items():
        if d in existing_dirs and entry.get("summary"):
            small = "\n".join(f"- {name}: {desc}" for name, desc in sorted(entry.get("small_files", {}).items()))
            text = f"Purpose: {entry.get('purpose', '')}\n\n{entry['summary']}"
            if small:
                text += f"\n\n### Small files\n{small}"
            sources[f"dir:{d}"] = ("dir", d, text)
    overview = hub.get("overview")
    if overview and overview.get("summary"):
        sources["overview"] = ("overview", ROOT, overview["summary"])
    return sources


def summary_entries(source_id: str, level: str, path: str, text: str) -> list[Entry]:
    subject = {
        "file": f"Summary of the file {path}",
        "dir": f"Summary of the folder {path or '(repo root)'}/",
        "overview": "Overview of the whole repository",
    }[level]
    entries = []
    for i, (heading, body) in enumerate(_split_markdown(text)):
        title = f"{subject} - {heading}" if heading else subject
        entries.append(Entry(f"summary:{source_id}#{i}", f"{title}\n\n{body}", {
            "type": "summary", "level": level, "path": path, "section": heading,
        }))
    return entries


# ------------------------------------------------------------------- update


def _all_dirs(paths) -> set[str]:
    """Every folder holding one of these files, up to the repo root."""
    dirs = {ROOT}
    for p in paths:
        d = posixpath.dirname(p)
        while d:
            dirs.add(d)
            d = posixpath.dirname(d)
    return dirs


def _collection_exists(client) -> bool:
    try:
        client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False


def open_repo_collection(repo_path: str | Path):
    """Open the search index (raises if it hasn't been built)."""
    return get_client(repo_path).get_collection(
        name=COLLECTION_NAME, embedding_function=get_embedding_function()
    )


ProgressFn = Callable[[int, int], None]  # (entries embedded so far, entries to embed)


def update_index(repo_path: str | Path, on_progress: ProgressFn | None = None,
                 rebuild: bool = False) -> IndexStats:
    """Bring the search index up to date with the repo and its hub summaries."""
    repo = Path(repo_path).resolve()
    stats = IndexStats()
    walked = walk_repo(repo)
    texts = {f.path: f.abs_path.read_text(encoding="utf-8") for f in walked.files}
    languages = {f.path: f.language for f in walked.files}
    dirs = _all_dirs(texts)

    client = get_client(repo)
    manifest = load_manifest(repo)
    if (
        rebuild or manifest is None
        or manifest.get("version") != INDEX_VERSION
        or manifest.get("embedding") != embedding_name()
        or not _collection_exists(client)
    ):
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete yet
        stats.rebuilt = manifest is not None or rebuild
        manifest = _empty_manifest()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    # Work out what changed.
    stale_ids: list[str] = []
    to_add: list[tuple[str, str, str, list[Entry]]] = []  # (manifest section, key, hash, entries)
    for path, record in list(manifest["files"].items()):
        if path not in texts:
            stale_ids += record["ids"]
            del manifest["files"][path]
            stats.files_removed += 1
    for path, text in texts.items():
        digest = _sha(text)
        record = manifest["files"].get(path)
        if record and record["hash"] == digest:
            stats.files_unchanged += 1
            continue
        if record:
            stale_ids += record["ids"]
        to_add.append(("files", path, digest, code_entries(path, languages[path], text)))
        stats.files_indexed += 1

    sources = summary_sources(load_hub(repo), set(texts), dirs)
    for source_id, record in list(manifest["summaries"].items()):
        if source_id not in sources:
            stale_ids += record["ids"]
            del manifest["summaries"][source_id]
            stats.summaries_removed += 1
    for source_id, (level, path, text) in sources.items():
        digest = _sha(text)
        record = manifest["summaries"].get(source_id)
        if record and record["hash"] == digest:
            continue
        if record:
            stale_ids += record["ids"]
        to_add.append(("summaries", source_id, digest, summary_entries(source_id, level, path, text)))
        stats.summaries_indexed += 1

    # Apply: deletions first, then additions, saving the manifest as we go so
    # an interrupted run only redoes what wasn't recorded.
    for i in range(0, len(stale_ids), _BATCH):
        collection.delete(ids=stale_ids[i:i + _BATCH])
    stats.entries_removed = len(stale_ids)
    _save_manifest(repo, manifest)

    total = sum(len(entries) for *_, entries in to_add)
    done = 0
    pending: list[Entry] = []
    pending_records: list[tuple[str, str, str, list[str]]] = []

    def flush() -> None:
        nonlocal done
        if pending:
            collection.upsert(
                ids=[e.id for e in pending],
                documents=[e.text for e in pending],
                metadatas=[e.metadata for e in pending],
            )
            done += len(pending)
        for section, key, digest, ids in pending_records:
            manifest[section][key] = {"hash": digest, "ids": ids}
        _save_manifest(repo, manifest)
        pending.clear()
        pending_records.clear()
        if on_progress:
            on_progress(done, total)

    for section, key, digest, entries in to_add:
        pending.extend(entries)
        pending_records.append((section, key, digest, [e.id for e in entries]))
        if len(pending) >= _BATCH:
            flush()
    flush()

    stats.entries_added = total
    stats.total_entries = collection.count()
    counts = {"code": 0, "summary": 0}
    for record in manifest["files"].values():
        counts["code"] += len(record["ids"])
    for record in manifest["summaries"].values():
        counts["summary"] += len(record["ids"])
    stats.by_type = counts
    return stats


def index_status(repo_path: str | Path) -> dict:
    """Whether the search index exists and how far behind the repo it is,
    without embedding anything."""
    repo = Path(repo_path).resolve()
    manifest = load_manifest(repo)
    if manifest is None:
        return {"indexed": False, "files_changed": None, "summaries_changed": None, "entries": 0}
    walked = walk_repo(repo)
    texts = {f.path: f.abs_path.read_text(encoding="utf-8") for f in walked.files}
    changed = sum(1 for p, t in texts.items() if manifest["files"].get(p, {}).get("hash") != _sha(t))
    changed += sum(1 for p in manifest["files"] if p not in texts)
    dirs = _all_dirs(texts)
    sources = summary_sources(load_hub(repo), set(texts), dirs)
    summaries_changed = sum(
        1 for sid, (_, _, text) in sources.items()
        if manifest["summaries"].get(sid, {}).get("hash") != _sha(text)
    ) + sum(1 for sid in manifest["summaries"] if sid not in sources)
    outdated_setup = manifest.get("version") != INDEX_VERSION or manifest.get("embedding") != embedding_name()
    entries = sum(len(r["ids"]) for r in manifest["files"].values()) + sum(
        len(r["ids"]) for r in manifest["summaries"].values())
    return {
        "indexed": True,
        "files_changed": changed,
        "summaries_changed": summaries_changed,
        "needs_rebuild": outdated_setup,
        "entries": entries,
    }


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """Update the search index from the command line, and try a query against it."""
    import argparse

    from docgen.repos import RepoError, resolve_repo

    parser = argparse.ArgumentParser(prog="python -m docgen.repo_index", description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="a local path or a public GitHub URL")
    parser.add_argument("--rebuild", action="store_true", help="throw the index away and start over")
    parser.add_argument("--query", help="after updating, print the closest entries to this text")
    parser.add_argument("-k", type=int, default=5, help="how many entries --query prints")
    args = parser.parse_args(argv)

    try:
        source = resolve_repo(args.repo)
    except RepoError as e:
        print(e)
        return 1

    def progress(done: int, total: int) -> None:
        if total:
            print(f"\rEmbedding {done}/{total}", end="", flush=True)

    stats = update_index(source.path, on_progress=progress, rebuild=args.rebuild)
    if stats.entries_added:
        print()
    print(f"{stats.files_indexed} files indexed, {stats.files_unchanged} unchanged, {stats.files_removed} removed; "
          f"{stats.summaries_indexed} summaries indexed, {stats.summaries_removed} removed.")
    print(f"Index holds {stats.total_entries} entries "
          f"({stats.by_type.get('code', 0)} code chunks, {stats.by_type.get('summary', 0)} summary parts).")

    if args.query:
        result = open_repo_collection(source.path).query(query_texts=[args.query], n_results=args.k)
        for doc_id, meta, dist in zip(result["ids"][0], result["metadatas"][0], result["distances"][0], strict=True):
            print(f"{1 - dist:.3f}  {doc_id}  [{meta['type']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
