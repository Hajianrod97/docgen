"""Tests for Phase 5. No network or model needed."""

import ast
import difflib
import inspect
import json
import shutil
import textwrap
from pathlib import Path

import pytest

from docgen import writer
from docgen.generator import source_hash
from docgen.indexer import ensure_unique_ids
from docgen.models import ChunkType
from docgen.parser import parse_repo
from docgen.generator import repo_structure_hash
from docgen.writer import API_MARKER, plan_apply, write_plan

EXAMPLE = Path(__file__).resolve().parent.parent / "example_repo"


def make_generated(repo, texts):
    """Build a generated.json-shaped dict for the given {chunk id: text}."""
    chunks = parse_repo(repo)
    ensure_unique_ids(chunks)
    items = {}
    for c in chunks:
        if c.id in texts:
            kind = "endpoint_doc" if c.type == ChunkType.API_ENDPOINT else "docstring"
            items[c.id] = {
                "kind": kind,
                "text": texts[c.id],
                "file_path": c.file_path,
                "start_line": c.start_line,
                "source_hash": source_hash(c),
            }
    return {"model": "test", "items": items}


def write_src(tmp_path, body, name="mod.py", newline="\n"):
    path = tmp_path / name
    text = textwrap.dedent(body).lstrip("\n")
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


def apply_all(repo, texts, **kw):
    plan = plan_apply(repo, make_generated(repo, texts), **kw)
    write_plan(plan)
    return plan


def only_additions(old: str, new: str) -> bool:
    ops = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines()).get_opcodes()
    return all(tag in ("equal", "insert") for tag, *_ in ops)


def docstring_of(path, name):
    tree = ast.parse(path.read_bytes().decode("utf-8"))
    node = next(n for n in ast.walk(tree) if getattr(n, "name", None) == name)
    return ast.get_docstring(node)


# ------------------------------------------------------------- insertion


def test_inserts_function_method_and_class_with_correct_indentation(tmp_path):
    src = write_src(tmp_path, """
        def top(a):
            return a

        class Box:
            def __init__(self):
                self.x = 1

            def get(self):
                return self.x
    """)
    before = src.read_text()
    plan = apply_all(tmp_path, {
        "mod.py::top": "Return a.",
        "mod.py::Box": "A box.",
        "mod.py::Box.get": "Return x.",
    })
    after = src.read_text()
    assert plan.docstring_count == 3
    assert docstring_of(src, "top") == "Return a."
    assert docstring_of(src, "Box") == "A box."
    assert docstring_of(src, "get") == "Return x."
    assert only_additions(before, after)
    assert '    def get(self):\n        """Return x."""\n        return self.x' in after
    compile(after, "mod.py", "exec")


def test_class_docstring_is_followed_by_a_blank_line(tmp_path):
    src = write_src(tmp_path, """
        class Box:
            def get(self):
                return 1
    """)
    apply_all(tmp_path, {"mod.py::Box": "A box."})
    assert src.read_text() == (
        "class Box:\n"
        '    """A box."""\n'
        "\n"
        "    def get(self):\n"
        "        return 1\n"
    )


def test_multiline_docstring_layout(tmp_path):
    write_src(tmp_path, """
        class C:
            def run(self, n):
                return n
    """)
    apply_all(tmp_path, {"mod.py::C.run": "Run it.\n\nArgs:\n    n: The count.\n\nReturns:\n    n back."})
    expected = (
        "class C:\n"
        "    def run(self, n):\n"
        '        """Run it.\n'
        "\n"
        "        Args:\n"
        "            n: The count.\n"
        "\n"
        "        Returns:\n"
        "            n back.\n"
        '        """\n'
        "        return n\n"
    )
    assert (tmp_path / "mod.py").read_text() == expected


def test_docstring_goes_above_decorators_of_a_nested_first_statement(tmp_path):
    src = write_src(tmp_path, """
        import functools

        def outer():
            @functools.lru_cache(maxsize=None)
            def inner(n):
                return n
            return inner
    """)
    apply_all(tmp_path, {"mod.py::outer": "Build a cached identity function."})
    after = src.read_text()
    assert docstring_of(src, "outer") == "Build a cached identity function."
    assert docstring_of(src, "inner") is None  # not stolen by the inner def
    assert after.index('"""') < after.index("@functools")


def test_one_liner_is_skipped_and_file_untouched(tmp_path):
    src = write_src(tmp_path, """
        def tiny(): return 1

        def normal():
            return 2
    """)
    before = src.read_bytes()
    plan = apply_all(tmp_path, {"mod.py::tiny": "Tiny."})
    assert src.read_bytes() == before
    assert plan.changes == []
    assert "one-liner" in plan.skipped[0].reason


def test_one_liner_skip_does_not_block_other_functions_in_same_file(tmp_path):
    src = write_src(tmp_path, """
        def tiny(): return 1

        def normal():
            return 2
    """)
    plan = apply_all(tmp_path, {"mod.py::tiny": "Tiny.", "mod.py::normal": "Normal."})
    assert docstring_of(src, "normal") == "Normal."
    assert docstring_of(src, "tiny") is None
    assert plan.docstring_count == 1 and len(plan.skipped) == 1


def test_multiline_signature_and_leading_comment(tmp_path):
    src = write_src(tmp_path, """
        def far(
            a,
            b=2,
        ):
            # explain the sum
            return a + b
    """)
    apply_all(tmp_path, {"mod.py::far": "Add two numbers."})
    assert docstring_of(src, "far") == "Add two numbers."
    ns = {}
    exec(compile(src.read_text(), "mod.py", "exec"), ns)
    assert ns["far"](1) == 3


def test_async_function(tmp_path):
    src = write_src(tmp_path, """
        async def fetch():
            return 1
    """)
    apply_all(tmp_path, {"mod.py::fetch": "Fetch it."})
    assert docstring_of(src, "fetch") == "Fetch it."


def test_crlf_line_endings_are_preserved(tmp_path):
    src = write_src(
        tmp_path,
        """
        def f():
            return 1

        def g():
            return 2
    """,
        newline="\r\n",
    )
    apply_all(tmp_path, {"mod.py::f": "Doc for f.\n\nMore.", "mod.py::g": "Doc for g."})
    raw = src.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n")  # no bare \n anywhere
    assert docstring_of(src, "f") == "Doc for f.\n\nMore."


@pytest.mark.parametrize(
    "text",
    [
        r"Match a Windows path like C:\temp\new.",
        'Explain the """triple quote""" problem.',
        'Ends with a quote: "hello"',
        "Tabs\tand unicode: caf\u00e9 \u2192 ok",
        "Trailing spaces   \nlines   ",
    ],
)
def test_awkward_text_round_trips(tmp_path, text):
    src = write_src(tmp_path, """
        def f():
            return 1
    """)
    plan = apply_all(tmp_path, {"mod.py::f": text})
    assert plan.docstring_count == 1, plan.skipped
    expected = "\n".join(line.rstrip() for line in text.strip().split("\n"))
    # get_docstring() cleans like help() does (which expands tabs), so compare
    # against the cleaned expectation, and check the raw file kept any tab.
    assert docstring_of(src, "f") == inspect.cleandoc(expected)
    if "\t" in text:
        assert "\t" in src.read_text()


# --------------------------------------------------------------- safety


def test_existing_docstring_is_never_overwritten(tmp_path):
    src = write_src(tmp_path, '''
        def f():
            """Original."""
            return 1
    ''')
    before = src.read_bytes()
    gen = make_generated(tmp_path, {})
    chunk = next(c for c in parse_repo(tmp_path))
    gen["items"][chunk.id] = {
        "kind": "docstring", "text": "New.", "file_path": "mod.py",
        "start_line": 1, "source_hash": "irrelevant",
    }
    plan = plan_apply(tmp_path, gen)
    assert plan.changes == []
    assert plan.skipped[0].reason == "already has a docstring"
    assert src.read_bytes() == before


def test_stale_docs_are_skipped_when_code_changed(tmp_path):
    src = write_src(tmp_path, """
        def f():
            return 1
    """)
    gen = make_generated(tmp_path, {"mod.py::f": "Returns one."})
    src.write_text(src.read_text().replace("return 1", "return 2"))
    plan = plan_apply(tmp_path, gen)
    assert plan.changes == []
    assert "code changed" in plan.skipped[0].reason


def test_deleted_definition_is_skipped(tmp_path):
    src = write_src(tmp_path, """
        def f():
            return 1
    """)
    gen = make_generated(tmp_path, {"mod.py::f": "Doc."})
    src.write_text("x = 1\n")
    plan = plan_apply(tmp_path, gen)
    assert "no longer exists" in plan.skipped[0].reason


def test_applying_twice_is_idempotent(tmp_path):
    src = write_src(tmp_path, """
        def f():
            return 1
    """)
    gen = make_generated(tmp_path, {"mod.py::f": "Doc."})
    write_plan(plan_apply(tmp_path, gen))
    after_first = src.read_bytes()
    plan = plan_apply(tmp_path, gen)
    write_plan(plan)
    assert plan.changes == []
    assert src.read_bytes() == after_first


def test_failed_verification_leaves_file_untouched(tmp_path, monkeypatch):
    src = write_src(tmp_path, """
        def f():
            return 1
    """)
    before = src.read_bytes()
    monkeypatch.setattr(writer, "_format_docstring", lambda *a: ['    """never closed\n'])
    plan = apply_all(tmp_path, {"mod.py::f": "Doc."})
    assert src.read_bytes() == before
    assert plan.changes == []
    assert "verification failed" in plan.skipped[0].reason


def test_verification_catches_a_docstring_on_the_wrong_definition(tmp_path, monkeypatch):
    """A syntactically valid but misplaced insertion must still be rejected."""
    src = write_src(tmp_path, """
        def f():
            return 1
    """)
    before = src.read_bytes()
    monkeypatch.setattr(
        writer, "_format_docstring", lambda text, indent, nl: [f'{indent}"wrong text"{nl}']
    )
    plan = apply_all(tmp_path, {"mod.py::f": "Doc."})
    assert src.read_bytes() == before
    assert "did not read back" in plan.skipped[0].reason


def test_syntax_error_file_is_skipped_not_crashed(tmp_path):
    good = write_src(tmp_path, "def f():\n    return 1\n", name="good.py")
    gen = make_generated(tmp_path, {"good.py::f": "Doc."})
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    plan = plan_apply(tmp_path, gen)
    assert docstring_of(good, "f") is None  # planning alone writes nothing
    assert plan.docstring_count == 1


# ----------------------------------------------------------------- API.md


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(EXAMPLE, dest, ignore=shutil.ignore_patterns(".docgen", "__pycache__", "docs"))
    return dest


ENDPOINT_TEXTS = {
    "api.py::get_config": "Return the current settings.\n\n**Response**: JSON settings.",
    "api.py::reload_config": "Reload the config from disk.",
}


def test_api_md_is_built_sorted_with_marker(repo):
    plan = apply_all(repo, ENDPOINT_TEXTS)
    md = (repo / "docs" / "API.md").read_text()
    assert md.startswith(API_MARKER)
    assert md.index("## GET /config\n") < md.index("## POST /config/reload")
    assert "_Handler: `reload_config` in `api.py`_" in md
    assert plan.api.endpoints == 2
    # endpoint docs live in API.md only; the function source is untouched
    assert docstring_of(repo / "api.py", "get_config") is None


def test_api_md_refuses_to_overwrite_a_hand_written_file(repo):
    (repo / "docs").mkdir()
    hand = repo / "docs" / "API.md"
    hand.write_text("# My careful hand-written docs\n")
    plan = apply_all(repo, ENDPOINT_TEXTS)
    assert hand.read_text() == "# My careful hand-written docs\n"
    assert plan.api is None
    assert "not generated by docgen" in plan.skipped[0].reason

    apply_all(repo, ENDPOINT_TEXTS, force=True)
    assert hand.read_text().startswith(API_MARKER)


def test_api_md_can_be_regenerated_over_its_own_output(repo):
    apply_all(repo, ENDPOINT_TEXTS)
    plan = apply_all(repo, {**ENDPOINT_TEXTS, "api.py::reload_config": "Reload, revised."})
    assert plan.api.changed
    assert "Reload, revised." in (repo / "docs" / "API.md").read_text()


def test_docs_dir_cannot_escape_the_repo(repo):
    with pytest.raises(ValueError):
        plan_apply(repo, make_generated(repo, ENDPOINT_TEXTS), docs_dir="../outside")


def test_no_api_md_flag_skips_it(repo):
    plan = apply_all(repo, ENDPOINT_TEXTS, include_api_md=False)
    assert plan.api is None
    assert not (repo / "docs").exists()


# --------------------------------------------------------------- OVERVIEW.md


def make_overview(repo, text):
    chunks = parse_repo(repo)
    ensure_unique_ids(chunks)
    return {"model": "test", "items": {}, "overview": {
        "text": text, "structure_hash": repo_structure_hash(chunks),
    }}


def test_overview_md_is_written_with_marker(repo):
    gen = make_overview(repo, "## Overview\n\nA config service.")
    plan = plan_apply(repo, gen)
    write_plan(plan)
    content = (repo / "docs" / "OVERVIEW.md").read_text()
    assert content.startswith(API_MARKER)  # same marker constant/text as API.md
    assert "# Project Overview" in content
    assert "## Overview\n\nA config service." in content
    assert plan.overview.path.name == "OVERVIEW.md"


def test_overview_md_refuses_to_overwrite_hand_written_file(repo):
    (repo / "docs").mkdir()
    hand = repo / "docs" / "OVERVIEW.md"
    hand.write_text("# Hand-written notes\n")
    gen = make_overview(repo, "## Overview\n\nA config service.")
    plan = plan_apply(repo, gen)
    assert plan.overview is None
    assert any("OVERVIEW.md" in s.id and "not generated by docgen" in s.reason for s in plan.skipped)
    assert hand.read_text() == "# Hand-written notes\n"

    write_plan(plan_apply(repo, gen, force=True))
    assert hand.read_text().startswith(API_MARKER)


def test_overview_is_skipped_when_repo_structure_changed_since_generation(repo):
    gen = make_overview(repo, "## Overview\n\nStale.")
    (repo / "utils.py").write_text((repo / "utils.py").read_text() + "\ndef new_fn():\n    return 1\n")
    plan = plan_apply(repo, gen)
    assert plan.overview is None
    assert any("overview" == s.id and "structure changed" in s.reason for s in plan.skipped)


def test_overview_can_be_disabled(repo):
    gen = make_overview(repo, "## Overview\n\nText.")
    plan = plan_apply(repo, gen, include_overview=False)
    assert plan.overview is None
    assert plan.skipped == []
    write_plan(plan)
    assert not (repo / "docs").exists()


def test_overview_reapplied_is_idempotent(repo):
    gen = make_overview(repo, "## Overview\n\nText.")
    write_plan(plan_apply(repo, gen))
    before = (repo / "docs" / "OVERVIEW.md").read_bytes()
    plan = plan_apply(repo, gen)
    assert plan.overview.changed is False  # same contract as plan.api: always built, .changed says whether to act
    write_plan(plan)
    assert (repo / "docs" / "OVERVIEW.md").read_bytes() == before


def test_no_overview_generated_yet_means_no_overview_file(repo):
    plan = plan_apply(repo, {"model": None, "items": {}})
    assert plan.overview is None
