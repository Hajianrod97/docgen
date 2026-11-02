"""Every test run uses its own database, docgen_test, created fresh at the
start, so tests never touch the real index. It lives on the same Postgres
server as DOCGEN_DATABASE_URL (the local docgen-db container by default)."""

import os

import psycopg
import pytest

from docgen import store


@pytest.fixture(scope="session", autouse=True)
def test_database():
    server_url = store.database_url()
    try:
        with psycopg.connect(server_url, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS docgen_test WITH (FORCE)")
            admin.execute("CREATE DATABASE docgen_test")
    except psycopg.OperationalError:
        pass  # tests that need Postgres fail with store.DatabaseError's advice
    os.environ["DOCGEN_DATABASE_URL"] = server_url.rsplit("/", 1)[0] + "/docgen_test"
    yield
