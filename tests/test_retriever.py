"""Tests for call-graph retrieval depth, caps, and name resolution."""

import shutil
from pathlib import Path

import pytest

from docgen.indexer import build_index, ensure_unique_ids
from docgen.parser import parse_repo
from docgen.retriever import DEFAULT_DEPTH, MAX_CALLEES, MAX_INDIRECT, get_context

EXAMPLE = Path(__file__).resolve().parent.parent / "example_repo"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


def index_files(tmp_path, files):
    for name, body in files.items():
        (tmp_path / name).write_text(body)
    chunks = parse_repo(tmp_path)
    ensure_unique_ids(chunks)
    return chunks, build_index(chunks, tmp_path)


def chunk_named(chunks, name):
    return next(c for c in chunks if c.name == name)


def ids(items):
    return [i.id for i in items]


CHAIN = {
    "app.py": (
        "def top():\n    return mid()\n\n"
        "def mid():\n    return low()\n\n"
        "def low():\n    return leaf()\n\n"
        "def leaf():\n    return 1\n"
    )
}


def test_depth_controls_how_far_calls_are_followed(tmp_path):
    chunks, coll = index_files(tmp_path, CHAIN)
    top = chunk_named(chunks, "top")
    one = get_context(coll, top, k_semantic=0, depth=1)
    two = get_context(coll, top, k_semantic=0, depth=2)
    three = get_context(coll, top, k_semantic=0, depth=3)
    assert ids(one.callees) == ["app.py::mid"] and one.indirect == []
    assert ids(two.indirect) == ["app.py::low"]
    assert ids(three.indirect) == ["app.py::low", "app.py::leaf"]


def test_default_depth_reaches_three_levels(tmp_path):
    assert DEFAULT_DEPTH == 3
    chunks, coll = index_files(tmp_path, CHAIN)
    ctx = get_context(coll, chunk_named(chunks, "top"), k_semantic=0)
    assert "app.py::leaf" in ids(ctx.indirect)


def test_each_item_says_who_called_it(tmp_path):
    chunks, coll = index_files(tmp_path, CHAIN)
    ctx = get_context(coll, chunk_named(chunks, "top"), k_semantic=0)
    assert [i.reason for i in ctx.callees + ctx.indirect] == [
        "called by target",
        "called by mid",
        "called by low",
    ]


def test_mutual_recursion_terminates_without_duplicates_or_the_target(tmp_path):
    chunks, coll = index_files(
        tmp_path, {"loop.py": "def a():\n    return b()\n\ndef b():\n    return a()\n"}
    )
    ctx = get_context(coll, chunk_named(chunks, "a"), k_semantic=0, depth=10)
    assert ids(ctx.callees) == ["loop.py::b"]
    assert ctx.indirect == []
    assert "loop.py::a" not in ids(ctx.all_items())


def test_self_recursion_does_not_include_the_target(tmp_path):
    chunks, coll = index_files(tmp_path, {"r.py": "def f(n):\n    return f(n - 1)\n"})
    ctx = get_context(coll, chunk_named(chunks, "f"), k_semantic=0)
    assert ctx.all_items() == []


def test_deeper_levels_prefer_the_callers_own_file(tmp_path):
    chunks, coll = index_files(
        tmp_path,
        {
            "a.py": "def helper():\n    return 1\n",
            "b.py": "def helper():\n    return 2\n\ndef helper2():\n    return helper()\n",
            "c.py": "def top():\n    return helper2()\n",
        },
    )
    ctx = get_context(coll, chunk_named(chunks, "top"), k_semantic=0)
    assert ids(ctx.callees) == ["b.py::helper2"]
    assert ids(ctx.indirect) == ["b.py::helper"]  # not a.py's unrelated helper


def test_fan_out_is_capped(tmp_path):
    n = 10
    src = "def top():\n    return " + " + ".join(f"f{i}()" for i in range(n)) + "\n\n"
    for i in range(n):
        src += f"def f{i}():\n    return g{i}() + h{i}()\n\n"
        src += f"def g{i}():\n    return 1\n\ndef h{i}():\n    return 2\n\n"
    chunks, coll = index_files(tmp_path, {"wide.py": src})
    ctx = get_context(coll, chunk_named(chunks, "top"), k_semantic=0, depth=3)
    assert len(ctx.callees) == MAX_CALLEES
    assert len(ctx.indirect) == MAX_INDIRECT


def test_library_calls_are_skipped(tmp_path):
    chunks, coll = index_files(
        tmp_path, {"lib.py": "def f(p):\n    with open(p) as fh:\n        return fh.read().strip()\n"}
    )
    assert get_context(coll, chunk_named(chunks, "f"), k_semantic=0).all_items() == []


def test_k_semantic_zero_returns_no_neighbors_and_priority_order_holds(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(EXAMPLE, dest, ignore=shutil.ignore_patterns(".docgen", "__pycache__"))
    chunks = parse_repo(dest)
    ensure_unique_ids(chunks)
    coll = build_index(chunks, dest)
    method = next(c for c in chunks if c.id == "utils.py::ConfigManager.reload")

    assert get_context(coll, method, k_semantic=0).similar == []

    ctx = get_context(coll, method, k_semantic=3)
    reasons = [i.reason for i in ctx.all_items()]
    assert reasons[0] == "parent class"
    assert reasons.index("called by target") < reasons.index("called by parse_config")
    assert reasons.index("called by parse_config") < reasons.index("semantically similar")


def test_a_helper_shared_by_two_callees_appears_once(tmp_path):
    chunks, coll = index_files(
        tmp_path,
        {
            "d.py": (
                "def top():\n    return x() + y()\n\n"
                "def x():\n    return shared()\n\n"
                "def y():\n    return shared()\n\n"
                "def shared():\n    return 1\n"
            )
        },
    )
    ctx = get_context(coll, chunk_named(chunks, "top"), k_semantic=0)
    everything = ids(ctx.all_items())
    assert everything.count("d.py::shared") == 1
    assert len(everything) == len(set(everything))
