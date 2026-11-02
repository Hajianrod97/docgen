"""The search index in Postgres: pgvector for vector search, ParadeDB's
pg_search for BM25 keyword search.

Each repo gets its own table (`table_name`), so BM25's word statistics are
the repo's own and a repo can use an embedding model of any vector size. A
row is one index entry:

    id         text primary key   "code:docgen/hub.py:40-72"
    text       text               what was embedded, header included
    keywords   text               tokenize(text) joined by spaces, for BM25
    type, path text               for filtering
    metadata   jsonb              lines, symbol, section, ...
    embedding  vector(<size>)     the entry's embedding

`keywords` holds docgen's own tokens rather than the raw text, and the BM25
index splits them only at spaces (pg_search's whitespace tokenizer), so
`make_hub_models` still matches "hub models" the way it did before Postgres.

`docgen_repos` holds each repo's manifest (the hash of every indexed file
and summary). It is saved in the same transaction as the rows it describes,
so the two can't disagree after a crash.

Connects to DOCGEN_DATABASE_URL, or the local docgen-db container by default.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from docgen.chunking import tokenize
from docgen.embeddings import get_embedding_function

DEFAULT_DATABASE_URL = "postgresql://docgen:docgen@localhost:5432/docgen"
REPOS_TABLE = "docgen_repos"


class DatabaseError(Exception):
    """Postgres couldn't be reached."""


def database_url() -> str:
    return os.environ.get("DOCGEN_DATABASE_URL", "").strip() or DEFAULT_DATABASE_URL


def connect() -> psycopg.Connection:
    """A new connection, with docgen's tables created if they're missing."""
    try:
        conn = psycopg.connect(database_url())
    except psycopg.OperationalError as e:
        raise DatabaseError(
            "Could not reach Postgres. Is the database running? Start it with: docker start docgen-db"
        ) from e
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {REPOS_TABLE} (
            repo_path text PRIMARY KEY,
            manifest jsonb NOT NULL)""")
    return conn


def table_name(repo: Path) -> str:
    """The repo's table: its folder name plus a hash of its full path, so two
    checkouts named alike don't share one."""
    resolved = str(Path(repo).resolve())
    name = re.sub(r"[^a-z0-9]+", "_", Path(resolved).name.lower()).strip("_")[:30]
    return f"repo_{name}_{hashlib.sha256(resolved.encode()).hexdigest()[:8]}"


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()[0]


def load_manifest(repo: Path) -> dict | None:
    """The repo's manifest, or None if it has never been indexed."""
    with connect() as conn:
        row = conn.execute(f"SELECT manifest FROM {REPOS_TABLE} WHERE repo_path = %s",
                           (str(Path(repo).resolve()),)).fetchone()
    return row[0] if row else None


class PostgresStore:
    """One repo's table, used by `update_index`. It calls prepare() once,
    then delete() and upsert() any number of times, each batch followed by
    save_manifest(), which commits. So a batch's rows and the manifest
    entries that list them are saved together or not at all."""

    def __init__(self, repo: Path):
        self.repo_path = str(Path(repo).resolve())
        self.table = sql.Identifier(table_name(repo))  # use as {table} in sql.SQL(...)
        self.embeddings = get_embedding_function()
        self.conn = connect()

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *exc) -> None:
        self.conn.close()  # anything not yet committed is rolled back

    def exists(self) -> bool:
        return _table_exists(self.conn, table_name(Path(self.repo_path)))

    def load_manifest(self) -> dict | None:
        row = self.conn.execute(f"SELECT manifest FROM {REPOS_TABLE} WHERE repo_path = %s",
                                (self.repo_path,)).fetchone()
        return row[0] if row else None

    def save_manifest(self, manifest: dict) -> None:
        """Save the manifest and commit everything since the last commit."""
        self.conn.execute(f"""INSERT INTO {REPOS_TABLE} (repo_path, manifest) VALUES (%s, %s)
            ON CONFLICT (repo_path) DO UPDATE SET manifest = EXCLUDED.manifest""",
                          (self.repo_path, Jsonb(manifest)))
        self.conn.commit()

    def prepare(self, reset: bool) -> None:
        """Create the table and its two search indexes if they don't exist;
        with reset, drop it first. The vector size comes from the embedding
        model, so changing models (which resets) can change it."""
        name = table_name(Path(self.repo_path))
        if reset:
            self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self.table))
        if not _table_exists(self.conn, name):
            dims = len(self.embeddings.embed_query("dimension probe"))
            self.conn.execute(sql.SQL("""CREATE TABLE {table} (
                id text PRIMARY KEY,
                text text NOT NULL,
                keywords text NOT NULL,
                type text NOT NULL,
                path text NOT NULL,
                metadata jsonb NOT NULL,
                embedding vector({dims}) NOT NULL)""").format(table=self.table, dims=sql.Literal(dims)))
            # BM25 over `keywords`, split only at spaces (docgen tokenized it already).
            self.conn.execute(sql.SQL("CREATE INDEX {index} ON {table} USING bm25 (id, (keywords::pdb.whitespace))")
                              .format(index=sql.Identifier(f"{name}_bm25"), table=self.table))
            # HNSW: approximate nearest neighbours by cosine distance.
            self.conn.execute(sql.SQL("CREATE INDEX {index} ON {table} USING hnsw (embedding vector_cosine_ops)")
                              .format(index=sql.Identifier(f"{name}_hnsw"), table=self.table))

    def count(self) -> int:
        return self.conn.execute(sql.SQL("SELECT count(*) FROM {}").format(self.table)).fetchone()[0]

    # ------------------------------------------------------------------
    # Writing entries. Neither commits: save_manifest() does.

    def delete(self, ids: list[str]) -> None:
        """Delete the rows with these ids. Ids that aren't there are fine."""
        statement = sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(self.table)
        self.conn.execute(statement, (ids,))

    def upsert(self, entries: list) -> None:
        """Insert these entries (docgen.repo_index.Entry), replacing any row
        with the same id."""
        vectors = self.embeddings.embed_documents([e.text for e in entries])
        statement = sql.SQL("""
            INSERT INTO {} (id, text, keywords, type, path, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                keywords = EXCLUDED.keywords,
                type = EXCLUDED.type,
                path = EXCLUDED.path,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding""").format(self.table)
        rows = [
            (e.id, e.text, " ".join(tokenize(e.text)), e.metadata["type"], e.metadata["path"],
             Jsonb(e.metadata), vector)
            for e, vector in zip(entries, vectors, strict=True)
        ]
        self.conn.cursor().executemany(statement, rows)


def ranked(repo: Path, query: str, mode: str, types: tuple[str, ...] | None,
           exclude_paths: frozenset[str], candidates: int):
    """(vector ranking, keyword ranking, {id: (metadata, text)}) for a
    question: the ids of the best `candidates` rows by each search, best
    first. Raises LookupError if the repo has no index."""
    table = table_name(repo)
    params = {"types": list(types) if types else None, "exclude": sorted(exclude_paths), "n": candidates}
    where = "(%(types)s::text[] IS NULL OR type = ANY(%(types)s)) AND NOT (path = ANY(%(exclude)s::text[]))"
    entries: dict[str, tuple[dict, str]] = {}
    vector: list[str] = []
    keyword: list[str] = []
    with connect() as conn:
        if not _table_exists(conn, table):
            raise LookupError(f"No search index for {repo}")
        if mode in ("hybrid", "vector"):
            # Keep scanning the HNSW graph until enough rows pass the filter.
            conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            params["question"] = get_embedding_function().embed_query(query)
            rows = conn.execute(sql.SQL(f"""SELECT id, metadata, text FROM {{table}} WHERE {where}
                ORDER BY embedding <=> %(question)s::vector LIMIT %(n)s""").format(table=sql.Identifier(table)),
                                params).fetchall()
            for doc_id, meta, text in rows:
                entries[doc_id] = (meta, text)
                vector.append(doc_id)
        words = " ".join(dict.fromkeys(tokenize(query)))  # a repeated word counts once
        if mode in ("hybrid", "keyword") and words:
            params["words"] = words
            # |||: rows with any of the words; pdb.score: their BM25 score.
            rows = conn.execute(sql.SQL(f"""SELECT id, metadata, text FROM {{table}}
                WHERE keywords ||| %(words)s AND {where}
                ORDER BY pdb.score(id) DESC LIMIT %(n)s""").format(table=sql.Identifier(table)),
                                params).fetchall()
            for doc_id, meta, text in rows:
                entries.setdefault(doc_id, (meta, text))
                keyword.append(doc_id)
    return vector, keyword, entries
