"""Tests for the hub's Markdown export and the status tree it relies on."""

import re
import shutil

import pytest

from docgen.hub import build_hub, hub_tree, load_hub, plan_hub
from docgen.hub_export import MARKER, default_export_dir, export_hub, shift_headings, slug
from docgen.repos import RepoSource
from test_hub import MULTILANG, FakeModel, models_for


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    return dest


@pytest.fixture
def built(repo):
    build_hub(repo, models_for(FakeModel()))
    return repo


def export(repo, out, source=None):
    plan = plan_hub(repo)
    return export_hub(plan, load_hub(repo), out, source or RepoSource(repo))


def links(text):
    return re.findall(r"\]\(([^)]+)\)", text)


# -------------------------------------------------------------- status tree


def statuses(node, out=None):
    out = {} if out is None else out
    out[(node["kind"], node["path"])] = node["status"]
    for child in node.get("children", []):
        statuses(child, out)
    return out


def test_tree_reports_status_of_every_node(repo):
    tree = statuses(hub_tree(plan_hub(repo), load_hub(repo)))
    assert tree[("dir", "")] == "missing"
    assert tree[("file", "backend/store.py")] == "missing"
    assert tree[("file", "backend/app.py")] == "tiny"

    build_hub(repo, models_for(FakeModel()))
    tree = statuses(hub_tree(plan_hub(repo), load_hub(repo)))
    assert tree[("dir", "")] == "done"
    assert tree[("dir", "frontend/src")] == "done"
    assert tree[("file", "backend/store.py")] == "done"

    (repo / "backend" / "store.py").write_text((repo / "backend" / "store.py").read_text() + "\n# edit\n")
    tree = statuses(hub_tree(plan_hub(repo), load_hub(repo)))
    assert tree[("file", "backend/store.py")] == "stale"
    assert tree[("dir", "backend")] == "stale"
    assert tree[("dir", "frontend")] == "done"


def test_tree_marks_failures(repo):
    fake = FakeModel()
    fake.fail_on.add("backend/store.py")
    build_hub(repo, models_for(fake))
    tree = statuses(hub_tree(plan_hub(repo), load_hub(repo)))
    assert tree[("file", "backend/store.py")] == "error"


def test_tree_lists_folders_before_files(repo):
    root = hub_tree(plan_hub(repo), load_hub(repo))
    kinds = [c["kind"] for c in root["children"]]
    assert kinds == sorted(kinds)  # "dir" < "file"
    assert [c["name"] for c in root["children"] if c["kind"] == "dir"] == ["backend", "frontend"]


# ------------------------------------------------------------------- export


def test_export_writes_one_readme_per_folder(built, tmp_path):
    out = tmp_path / "out"
    written = export(built, out)
    assert sorted(p.relative_to(out).as_posix() for p in written) == [
        "README.md", "backend/README.md", "frontend/README.md", "frontend/src/README.md",
    ]
    assert all(p.read_text().startswith(MARKER) for p in written)


def test_root_readme_has_overview_diagram_and_layout(built, tmp_path):
    out = tmp_path / "out"
    export(built, out)
    text = (out / "README.md").read_text()
    assert "# multilang" in text
    assert "## What it does" in text
    assert "```mermaid\nflowchart TD" in text
    assert "## Repository layout" in text
    assert "### Contents" in text
    assert "| [backend/](backend/README.md) | Purpose of dir backend. |" in text


def test_dir_readme_has_role_contents_and_file_sections(built, tmp_path):
    out = tmp_path / "out"
    export(built, out)
    text = (out / "backend" / "README.md").read_text()
    assert "[Up to the overview](../README.md)" in text
    assert "_Purpose of dir backend._" in text
    assert "## Role" in text and "## Contents" in text and "## Files" in text
    # A summarized file links to its own section; its headings sit under it.
    assert "| [store.py](#storepy) | Purpose of backend/store.py. |" in text
    assert "### store.py" in text and "#### Purpose" in text
    # A small file gets the folder's one-line description and a source link.
    assert re.search(r"\| \[app\.py\]\([^)]*backend/app\.py\) \| small file app\.py \|", text)
    assert "### app.py" not in text


def test_source_links_resolve_to_the_repo_files(built, tmp_path):
    out = tmp_path / "out"
    export(built, out)
    readme = out / "backend" / "README.md"
    sources = [l for l in links(readme.read_text()) if l.endswith(".py")]
    assert sources
    for link in sources:
        assert (readme.parent / link).resolve().is_file()


def test_clone_export_links_to_github(built, tmp_path):
    out = tmp_path / "out"
    export(built, out, RepoSource(built, "owner", "repo"))
    text = (out / "backend" / "README.md").read_text()
    assert "(https://github.com/owner/repo/blob/HEAD/backend/store.py)" in text
    assert (out / "README.md").read_text().count("# owner/repo") == 1


def test_export_before_building_says_what_is_missing(repo, tmp_path):
    out = tmp_path / "out"
    export(repo, out)
    assert "No overview yet" in (out / "README.md").read_text()
    assert "_Not summarized yet._" in (out / "backend" / "README.md").read_text()


def test_export_flags_stale_summaries(built, tmp_path):
    store = built / "backend" / "store.py"
    store.write_text(store.read_text() + "\n# edit\n")
    out = tmp_path / "out"
    export(built, out)
    text = (out / "backend" / "README.md").read_text()
    assert "may be out of date: the code changed" in text


def test_export_removes_its_own_leftovers_only(built, tmp_path):
    out = tmp_path / "out"
    (out / "old").mkdir(parents=True)
    (out / "old" / "README.md").write_text(MARKER + "\nfrom a folder that was deleted\n")
    (out / "notes").mkdir()
    (out / "notes" / "README.md").write_text("someone's own notes\n")
    export(built, out)
    assert not (out / "old" / "README.md").exists()
    assert (out / "notes" / "README.md").exists()


def test_default_export_dir(tmp_path, monkeypatch):
    assert default_export_dir(RepoSource(tmp_path)) == tmp_path / "docs" / "hub"
    monkeypatch.chdir(tmp_path)
    assert default_export_dir(RepoSource(tmp_path / "x", "owner", "repo")) == tmp_path / "output" / "owner-repo"


def test_exported_docs_are_not_summarized_on_the_next_build(built):
    out = default_export_dir(RepoSource(built))
    export(built, out)
    plan = plan_hub(built)
    assert not any(p.startswith("docs/hub") for p in plan.files)
    assert plan.todo_files == [] and plan.todo_dirs == [] and not plan.todo_overview


# ------------------------------------------------------------------ helpers


def test_shift_headings_keeps_relative_levels_and_skips_code():
    md = "### Purpose\ntext\n#### Detail\n```\n# not a heading\n```"
    assert shift_headings(md, 2) == "## Purpose\ntext\n### Detail\n```\n# not a heading\n```"
    assert shift_headings("no headings", 2) == "no headings"


def test_slug_matches_github_anchors():
    assert slug("store.py") == "storepy"
    assert slug("How it fits together") == "how-it-fits-together"
    assert slug("my_file-2.ts") == "my_file-2ts"
