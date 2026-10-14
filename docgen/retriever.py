"""Phase 3: gather cross-file context for a chunk.

Two complementary strategies:

1. Call graph (exact): follow the calls the chunk makes, and the calls those
   make, up to `depth` levels. Precise and deterministic, but names are matched
   bare, so a call like `data.get()` on a dict can wrongly match a repo
   function named `get`. Library calls (open, jsonify) find nothing and are
   skipped. Depth matters because a fact the model needs (say, that a file is
   opened and can be missing) is often a few calls below the function being
   documented.
2. Semantic (fuzzy): embed the chunk and pull its nearest neighbors. Catches
   related code the call graph can't see (sibling endpoints, similar helpers).

Callers get structured results so Phase 4 can decide how to lay them out in
the prompt and how much of each to include.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from docgen.models import Chunk

DEFAULT_DEPTH = 3  # target -> callee -> callee's callee -> ...
MAX_CALLEES = 5  # direct callees of the target
MAX_INDIRECT = 6  # everything deeper, combined, so fan-out can't blow up the prompt
MAX_PER_NAME = 2


@dataclass
class ContextItem:
    id: str
    name: str
    type: str
    file_path: str
    source: str
    reason: str  # "parent class", "called by target", "called by <name>", "semantically similar"
    calls: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    parent: ContextItem | None = None
    callees: list[ContextItem] = field(default_factory=list)  # called directly by the target
    indirect: list[ContextItem] = field(default_factory=list)  # called by those, and so on
    similar: list[ContextItem] = field(default_factory=list)

    def all_items(self) -> list[ContextItem]:
        """In priority order: what the prompt should keep first if space runs out."""
        items = [self.parent] if self.parent else []
        return items + self.callees + self.indirect + self.similar


def _strip_header(document: str) -> str:
    """Stored documents are '<header>\nFile: ...\n\n<source>'; keep the source."""
    _, _, source = document.partition("\n\n")
    return source or document


def _item(id_: str, meta: dict, document: str, reason: str) -> ContextItem:
    return ContextItem(
        id=id_,
        name=meta["name"],
        type=meta["type"],
        file_path=meta["file_path"],
        source=_strip_header(document),
        reason=reason,
        calls=[c for c in meta.get("calls", "").split(",") if c],
    )


def _resolve(collection, name: str, anchor_file: str, seen: set[str]) -> list[tuple]:
    """Find chunks a call to `name` could refer to, as (id, metadata, document).

    A bare call resolves to the same module first, so if the name is defined in
    `anchor_file` (the file the call was made from), other files' same-named
    chunks are not candidates. Classes are skipped: calling one constructs it,
    and the parent-class lookup already covers class context for methods.
    """
    res = collection.get(where={"name": name}, include=["documents", "metadatas"])
    rows = sorted(
        zip(res["ids"], res["metadatas"], res["documents"]),
        key=lambda r: r[1]["file_path"] != anchor_file,
    )
    if rows and rows[0][1]["file_path"] == anchor_file:
        rows = [r for r in rows if r[1]["file_path"] == anchor_file]
    rows = [r for r in rows if r[0] not in seen and r[1]["type"] != "class"]
    return rows[:MAX_PER_NAME]


def get_context(
    collection,
    chunk: Chunk,
    k_semantic: int = 3,
    depth: int = DEFAULT_DEPTH,
) -> RetrievedContext:
    ctx = RetrievedContext()
    seen: set[str] = {chunk.id}

    # 1a. Parent class, so a method is documented in light of its class.
    if chunk.parent:
        parent_id = f"{chunk.file_path}::{chunk.parent}"
        res = collection.get(ids=[parent_id], include=["documents", "metadatas"])
        if res["ids"]:
            ctx.parent = _item(
                res["ids"][0], res["metadatas"][0], res["documents"][0], "parent class"
            )
            seen.add(parent_id)

    # 1b. Call graph, breadth-first. Each frontier entry is
    # (names being called, file they are called from, who is calling).
    frontier = [(chunk.calls, chunk.file_path, "target")]
    for level in range(1, depth + 1):
        bucket = ctx.callees if level == 1 else ctx.indirect
        cap = MAX_CALLEES if level == 1 else MAX_INDIRECT
        next_frontier = []
        for names, anchor_file, caller in frontier:
            for name in names:
                if len(bucket) >= cap:
                    break
                for id_, meta, doc in _resolve(collection, name, anchor_file, seen):
                    if len(bucket) >= cap:
                        break
                    item = _item(id_, meta, doc, f"called by {caller}")
                    bucket.append(item)
                    seen.add(id_)
                    next_frontier.append((item.calls, item.file_path, item.name))
        frontier = next_frontier
        if not frontier:
            break

    # 2. Semantic neighbors, skipping anything already included.
    if k_semantic > 0:
        res = collection.query(
            query_texts=[chunk.to_embedding_text()],
            n_results=k_semantic + len(seen),
            include=["documents", "metadatas"],
        )
        for id_, meta, doc in zip(res["ids"][0], res["metadatas"][0], res["documents"][0]):
            if id_ in seen:
                continue
            ctx.similar.append(_item(id_, meta, doc, "semantically similar"))
            seen.add(id_)
            if len(ctx.similar) >= k_semantic:
                break

    return ctx
