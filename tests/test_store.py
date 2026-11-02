"""Tests for the Postgres store. The ones marked "hand-written" check
PostgresStore.delete and PostgresStore.upsert, and fail until those two are
implemented. Uses the offline hashing embedder."""

import pytest
from psycopg import sql

from docgen import store as storemod
from docgen.chunking import tokenize
from docgen.repo_index import Entry
from docgen.store import DatabaseError, PostgresStore, connect, ranked, table_name


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


@pytest.fixture
def store(tmp_path):
    with PostgresStore(tmp_path) as s:
        s.prepare(reset=False)
        s.save_manifest({})  # commit the new table
        yield s


def entries(n=3, text="return"):
    return [Entry(f"code:f{i}.py:1-2", f"File: f{i}.py\n\ndef handler_{i}(): {text} {i}",
                  {"type": "code", "path": f"f{i}.py", "start_line": 1, "end_line": 2, "symbol": f"handler_{i}"})
            for i in range(n)]


def rows(s):
    """id -> (text, keywords, type, path, metadata), read over a separate
    connection, so only what was committed shows."""
    with connect() as conn:
        found = conn.execute(sql.SQL("SELECT id, text, keywords, type, path, metadata FROM {}").format(s.table))
        return {r[0]: r[1:] for r in found}


# ------------------------------------------------------------ setup


def test_table_name_is_per_checkout(tmp_path):
    a, b = tmp_path / "a" / "My Repo", tmp_path / "b" / "My Repo"
    assert table_name(a) == table_name(a) != table_name(b)
    assert table_name(a).startswith("repo_my_repo_")


def test_an_unreachable_database_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("DOCGEN_DATABASE_URL", "postgresql://docgen:docgen@127.0.0.1:1/docgen")
    with pytest.raises(DatabaseError, match="docker start docgen-db"):
        connect()


def test_prepare_creates_the_table_with_both_indexes(store):
    with connect() as conn:
        indexes = {r[0] for r in conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = %s",
                                              (table_name(store.repo_path),))}
    assert any("USING bm25" in i for i in indexes) and any("USING hnsw" in i for i in indexes)
    assert store.exists() and store.count() == 0


def test_manifest_round_trip(store):
    store.save_manifest({"files": {"a.py": {"hash": "x", "ids": ["code:a.py:1-2"]}}})
    assert storemod.load_manifest(store.repo_path)["files"]["a.py"]["ids"] == ["code:a.py:1-2"]


def test_search_without_an_index_is_a_lookup_error(tmp_path):
    with pytest.raises(LookupError):
        ranked(tmp_path / "never-indexed", "x", "hybrid", None, frozenset(), 5)


# ------------------------------------------------------ hand-written


def test_upsert_writes_every_column(store):
    store.upsert(entries())
    store.save_manifest({})
    got = rows(store)
    assert sorted(got) == [f"code:f{i}.py:1-2" for i in range(3)]
    text, keywords, kind, path, meta = got["code:f0.py:1-2"]
    assert text.startswith("File: f0.py") and keywords == " ".join(tokenize(text))
    assert kind == "code" and path == "f0.py"
    assert meta["start_line"] == 1 and meta["symbol"] == "handler_0"
    assert store.count() == 3


def test_upsert_replaces_a_row_with_the_same_id(store):
    store.upsert(entries())
    store.upsert(entries(1, text="yield"))
    store.save_manifest({})
    got = rows(store)
    assert len(got) == 3 and "yield" in got["code:f0.py:1-2"][0]


def test_upsert_waits_for_save_manifest_to_commit(store):
    store.upsert(entries())
    assert rows(store) == {}  # not committed yet, so other connections can't see it
    store.save_manifest({})
    assert len(rows(store)) == 3


def test_upserted_rows_are_found_by_both_searches(store):
    store.upsert(entries())
    store.save_manifest({})
    vector, keyword, found = ranked(store.repo_path, "handler_1", "hybrid", None, frozenset(), 5)
    assert vector[0] == keyword[0] == "code:f1.py:1-2"
    assert found["code:f1.py:1-2"][0]["path"] == "f1.py"


def test_delete_removes_only_those_rows(store):
    store.upsert(entries())
    store.delete(["code:f1.py:1-2", "code:not-there:1-1"])
    store.save_manifest({})
    assert sorted(rows(store)) == ["code:f0.py:1-2", "code:f2.py:1-2"]


def test_reset_starts_from_an_empty_table(store):
    store.upsert(entries())
    store.save_manifest({})
    store.prepare(reset=True)
    store.save_manifest({})
    assert store.count() == 0 and rows(store) == {}


def test_filters_reach_both_searches(store):
    store.upsert(entries())
    store.save_manifest({})
    vector, keyword, _ = ranked(store.repo_path, "handler_1", "hybrid", None, frozenset({"f1.py"}), 5)
    assert "code:f1.py:1-2" not in vector + keyword
    assert ranked(store.repo_path, "handler_1", "hybrid", ("summary",), frozenset(), 5)[:2] == ([], [])


def test_a_question_of_only_stopwords_skips_keyword_search(store):
    store.upsert(entries())
    store.save_manifest({})
    vector, keyword, _ = ranked(store.repo_path, "how is it?", "hybrid", None, frozenset(), 5)
    assert vector and keyword == []
