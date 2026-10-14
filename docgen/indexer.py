"""Phase 3: embed chunks and store them in a local ChromaDB collection.

The index lives at <repo>/.docgen/chroma so it travels with the repo it
describes (add `.docgen/` to that repo's .gitignore). v1 rebuilds the whole
collection on every run; incremental re-indexing is a Phase 6 item.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from docgen.embeddings import get_embedding_function
from docgen.models import Chunk

COLLECTION_NAME = "docgen_chunks"
INDEX_DIRNAME = ".docgen/chroma"
_BATCH = 256


def index_path(repo_path: str | Path) -> Path:
    return Path(repo_path).resolve() / INDEX_DIRNAME


def ensure_unique_ids(chunks: list[Chunk]) -> None:
    """Chroma rejects duplicate ids, and a file can define the same name twice
    (conditional definitions, property setters). Suffix collisions with the line."""
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id in seen:
            chunk.id = f"{chunk.id}@L{chunk.start_line}"
        seen.add(chunk.id)


def _metadata(chunk: Chunk) -> dict:
    """Chroma metadata must be str/int/float/bool, so None becomes ''."""
    return {
        "name": chunk.name,
        "type": chunk.type.value,
        "file_path": chunk.file_path,
        "parent": chunk.parent or "",
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "calls": ",".join(chunk.calls),
        "http_method": chunk.http_method or "",
        "route": chunk.route or "",
        "has_docstring": bool(chunk.docstring),
    }


def get_client(repo_path: str | Path):
    path = index_path(repo_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def open_collection(repo_path: str | Path):
    """Open an existing index (raises if `docgen index` hasn't been run)."""
    client = get_client(repo_path)
    return client.get_collection(
        name=COLLECTION_NAME, embedding_function=get_embedding_function()
    )


def build_index(chunks: list[Chunk], repo_path: str | Path):
    """(Re)build the vector index for a repo and return the collection."""
    client = get_client(repo_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ensure_unique_ids(chunks)
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i : i + _BATCH]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.to_embedding_text() for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )
    return collection
