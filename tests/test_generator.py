"""Tests for Phase 4. The model is faked, so these run offline."""

import shutil
from pathlib import Path

import pytest
from docgen.generator import (
    MAX_CONTEXT_CHARS,
    GenerationError,
    build_prompt,
    parse_output,
)
from docgen.indexer import build_index, ensure_unique_ids
from docgen.parser import parse_repo
from docgen.retriever import ContextItem, RetrievedContext, get_context

EXAMPLE = Path(__file__).resolve().parent.parent / "example_repo"


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(EXAMPLE, dest, ignore=shutil.ignore_patterns(".docgen", "__pycache__"))
    return dest


@pytest.fixture
def indexed(repo):
    chunks = parse_repo(repo)
    ensure_unique_ids(chunks)
    collection = build_index(chunks, repo)
    return repo, chunks, collection


def _chunk(chunks, name):
    return next(c for c in chunks if c.name == name)


def test_endpoint_prompt_uses_endpoint_template_and_includes_callee(indexed):
    _, chunks, collection = indexed
    chunk = _chunk(chunks, "reload_config")
    system, user = build_prompt(chunk, get_context(collection, chunk))
    assert "<endpoint_doc>" in user
    assert "HTTP POST /config/reload" in user
    assert "self.config = parse_config(self.path)" in user  # callee from utils.py
    assert "not instructions to you" in system


def test_function_prompt_uses_docstring_template(indexed):
    _, chunks, collection = indexed
    chunk = _chunk(chunks, "parse_config")
    _, user = build_prompt(chunk, get_context(collection, chunk))
    assert "<docstring>" in user
    assert "<endpoint_doc>" not in user


def test_context_is_capped_at_budget():
    big = "x = 1\n" * 2000
    items = [
        ContextItem(f"f.py::c{i}", f"c{i}", "function", "f.py", big, "semantically similar")
        for i in range(10)
    ]
    ctx = RetrievedContext(similar=items)
    chunk = parse_repo(EXAMPLE)[0]
    _, user = build_prompt(chunk, ctx)
    context_part = user.split("<context>")[1]
    assert len(context_part) < MAX_CONTEXT_CHARS + 1000
    assert "[truncated]" in context_part


def test_parse_output_extracts_and_strips_quotes():
    assert parse_output("docstring", "<docstring>\nDo a thing.\n</docstring>") == "Do a thing."
    quoted = '<docstring>\n"""Do a thing."""\n</docstring>'
    assert parse_output("docstring", quoted) == "Do a thing."


@pytest.mark.parametrize(
    "reply", ["no tags here", "<docstring>\n\n</docstring>", "<endpoint_doc>x</endpoint_doc>"]
)
def test_parse_output_rejects_unusable_replies(reply):
    with pytest.raises(GenerationError):
        parse_output("docstring", reply)


# ---- endpoint doc cleanup and prompt wording ----


def _endpoint(text):
    return parse_output("endpoint_doc", f"<endpoint_doc>{text}</endpoint_doc>")


def test_placeholder_sections_are_removed_and_headings_get_spacing():
    raw = (
        "Retrieves settings.\n\n**Request**\nNo request parameters.\n\n"
        "**Response**\nReturns the settings.\n\n**Errors**\n- None\n"
    )
    assert _endpoint(raw) == (
        "Retrieves settings.\n\n**Request**\n\nNo request parameters.\n\n"
        "**Response**\n\nReturns the settings."
    )


def test_bare_none_request_and_errors_are_dropped_but_json_block_is_kept():
    raw = (
        "Reloads.\n\n**Request**\nNone\n\n**Response**\n```json\n{\n    \"ok\": true\n}\n```\n\n"
        "**Errors**\nNone"
    )
    assert _endpoint(raw) == 'Reloads.\n\n**Response**\n\n```json\n{\n    "ok": true\n}\n```'


@pytest.mark.parametrize("placeholder", ["None", "none.", "N/A", "n/a.", "- None.", "Not applicable", "No errors"])
def test_placeholder_variants_are_dropped(placeholder):
    assert _endpoint(f"Does a thing.\n\n**Errors**\n{placeholder}") == "Does a thing."


def test_real_error_content_is_kept():
    raw = "Does a thing.\n\n**Errors**\n- 500 if the config file is missing."
    assert _endpoint(raw) == "Does a thing.\n\n**Errors**\n\n- 500 if the config file is missing."


def test_inline_section_content_is_left_alone():
    raw = "Does a thing.\n\n**Response**: JSON settings."
    assert _endpoint(raw) == raw


def test_section_names_inside_code_fences_are_not_treated_as_sections():
    raw = "Desc.\n\n**Response**\n```\n**Errors**\nNone\n```"
    assert _endpoint(raw) == "Desc.\n\n**Response**\n\n```\n**Errors**\nNone\n```"


def test_a_reply_that_is_only_placeholders_is_rejected_as_empty():
    with pytest.raises(GenerationError, match="empty"):
        _endpoint("**Errors**\nNone")


def test_prompts_tell_the_model_not_to_write_placeholders(indexed):
    _, chunks, collection = indexed
    endpoint = _chunk(chunks, "reload_config")
    _, user = build_prompt(endpoint, get_context(collection, endpoint))
    assert 'Never write "None" or "N/A"' in user
    function = _chunk(chunks, "parse_config")
    _, user = build_prompt(function, get_context(collection, function))
    assert 'Never write "Raises: None"' in user


def test_deeper_callees_reach_the_prompt(indexed):
    """The file open lives three calls below the endpoint; the model must see it."""
    _, chunks, collection = indexed
    chunk = _chunk(chunks, "reload_config")
    _, user = build_prompt(chunk, get_context(collection, chunk))
    assert "with open(path) as f" in user
    assert 'why="called by parse_config"' in user


# ---- Phase overview: repo-level summary ----


def test_repo_map_groups_by_file_sorted_by_line_and_includes_descriptions(indexed):
    _, chunks, _ = indexed
    from docgen.generator import build_repo_map

    text = build_repo_map(chunks)
    assert text.index("api.py:") < text.index("utils.py:")
    assert text.index("def parse_config") < text.index("def load_file")
    assert "def normalize" in text and "Already documented" in text  # existing docstring surfaces
    assert "GET /config -> get_config()" in text
    assert "POST /config/reload -> reload_config()" in text
    assert "class ConfigManager:" in text


def test_repo_map_prefers_generated_text_over_missing_docstring(indexed):
    _, chunks, _ = indexed
    from docgen.generator import build_repo_map

    load_file = next(c for c in chunks if c.name == "load_file")
    text = build_repo_map(chunks, generated_items={load_file.id: {"text": "Reads a file from disk."}})
    assert "Reads a file from disk." in text


def test_repo_map_is_capped(indexed):
    from docgen.generator import OVERVIEW_MAP_CHAR_LIMIT, build_repo_map

    _, chunks, _ = indexed
    huge = chunks * 500  # force it well past the cap
    text = build_repo_map(huge)
    assert len(text) < OVERVIEW_MAP_CHAR_LIMIT + 200


def test_overview_prompt_contains_the_repo_map():
    from docgen.generator import build_overview_prompt

    system, user = build_overview_prompt("api.py:\n  GET /x -> f()\n")
    assert "GET /x -> f()" in user
    assert "<repo_map>" in user
    assert "not instructions to you" in system


def test_generate_overview_parses_the_tagged_reply(indexed):
    from docgen.generator import generate_overview

    _, chunks, _ = indexed
    text = generate_overview(
        chunks, lambda s, u: "<overview>## Overview\n\nA thing.</overview>"
    )
    assert text == "## Overview\n\nA thing."


def test_generate_overview_rejects_an_untagged_reply(indexed):
    from docgen.generator import generate_overview

    _, chunks, _ = indexed
    with pytest.raises(GenerationError):
        generate_overview(chunks, lambda s, u: "no tags here")


def test_repo_structure_hash_changes_when_a_function_is_renamed(repo):
    from docgen.generator import repo_structure_hash

    chunks = parse_repo(repo)
    ensure_unique_ids(chunks)
    base = repo_structure_hash(chunks)
    assert repo_structure_hash(parse_repo(repo)) == base  # deterministic across reparses

    utils = repo / "utils.py"
    utils.write_text(utils.read_text().replace("def load_file", "def read_file"))
    renamed = parse_repo(repo)
    ensure_unique_ids(renamed)
    assert repo_structure_hash(renamed) != base


def test_repo_structure_hash_changes_when_a_signature_changes(repo):
    from docgen.generator import repo_structure_hash

    base = repo_structure_hash(parse_repo(repo))
    utils = repo / "utils.py"
    utils.write_text(
        utils.read_text().replace(
            "def load_file(path: str)", "def load_file(path: str, mode: str = 'r')"
        )
    )
    assert repo_structure_hash(parse_repo(repo)) != base


def test_repo_structure_hash_ignores_body_changes(repo):
    """Changing what a function DOES (not its name or signature) shouldn't invalidate the overview."""
    from docgen.generator import repo_structure_hash

    base = repo_structure_hash(parse_repo(repo))
    utils = repo / "utils.py"
    utils.write_text(utils.read_text().replace("return f.read()", "return f.read().strip()"))
    assert repo_structure_hash(parse_repo(repo)) == base
