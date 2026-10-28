"""Tests for the hub builder. A fake model answers by recognizing which prompt
it was sent, so these run offline and check the pipeline, not summary quality."""

import json
import re
import shutil
import threading
from pathlib import Path

import pytest

from docgen import hub as hubmod
from docgen.llm import GenerationError, ProviderError, ProviderSetupError
from docgen.hub import (
    HubModels,
    HubStopped,
    build_hub,
    check_mermaid,
    load_hub,
    plan_hub,
    split_sections,
)
from docgen.imports import import_graph, reverse_graph
from docgen.walker import walk_repo

MULTILANG = Path(__file__).resolve().parent.parent / "example_multilang"
GOOD_DIAGRAM = "flowchart TD\n  ui[Frontend] --> api[Backend]\n  api --> store[(Task store)]"


class FakeModel:
    """Replies in the format each hub prompt asks for, and records every call."""

    def __init__(self, diagram=GOOD_DIAGRAM, fixed_diagram=GOOD_DIAGRAM):
        self.calls = []  # (kind, subject)
        self.diagram = diagram
        self.fixed_diagram = fixed_diagram
        self.fail_on = set()  # subjects whose reply is unusable
        self.lock = threading.Lock()

    def __call__(self, system, user):
        kind, subject = self.classify(user)
        with self.lock:
            self.calls.append((kind, subject))
        if subject in self.fail_on:
            return "Sorry, I lost track of the format."
        if kind == "section":
            return "<notes>- defines things</notes>"
        if kind == "file":
            return f"<purpose>Purpose of {subject}.</purpose><summary>### Purpose\nAbout {subject}.</summary>"
        if kind == "dir":
            small = re.findall(r"^--- (.+) ---$", user, re.M)
            lines = "\n".join(f"- `{n}`: small file {n}" for n in small)
            return (f"<purpose>Purpose of dir {subject}.</purpose><summary>### Role\nDir {subject}.</summary>"
                    + (f"<small_files>\n{lines}\n</small_files>" if small else ""))
        if kind == "overview":
            return f"<overview>## What it does\nA to-do app.</overview><diagram>{self.diagram}</diagram>"
        if kind == "diagram_fix":
            return f"<diagram>{self.fixed_diagram}</diagram>"
        raise AssertionError(f"unexpected prompt: {user[:200]}")

    @staticmethod
    def classify(user):
        if "too long to summarize in one go" in user:
            return "section", re.search(r"lines \d+-\d+ of (\S+),", user).group(1)
        if user.startswith("Summarize the file"):
            return "file", re.match(r"Summarize the file (\S+) ", user).group(1)
        if user.startswith("Summarize the directory"):
            return "dir", re.match(r"Summarize the directory (.+?) for", user).group(1)
        if "front page of a documentation hub" in user:
            return "overview", "overview"
        if "does not render" in user:
            return "diagram_fix", "diagram"
        return "unknown", user[:40]

    def kinds(self):
        return sorted(k for k, _ in self.calls)


def models_for(fake, workers=1, context_chars=100_000):
    return HubModels(fake, "claude-haiku-5-5", fake, "claude-opus-5-5",
                     workers=workers, context_chars=context_chars)


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "multilang"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen"))
    return dest


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def long_file(lines=40, tag="x"):
    return "".join(f"line {i} {tag}\n" for i in range(lines))


# ------------------------------------------------------------------ planning


def test_plan_separates_tiny_files_and_orders_dirs_deepest_first(repo):
    plan = plan_hub(repo)
    # Only store.py reaches 30 lines; the rest are covered by their directory.
    assert plan.todo_files == ["backend/store.py"]
    assert [p for p in plan.files if not plan.is_tiny(p)] == ["backend/store.py"]
    assert plan.todo_dirs == ["frontend/src", "backend", "frontend", ""]
    assert plan.todo_overview is True
    assert plan.dirs[""].subdirs == ["backend", "frontend"]
    assert sorted(plan.dirs["frontend/src"].files) == [
        "frontend/src/api.ts", "frontend/src/config.ts", "frontend/src/main.ts",
    ]


def test_estimate_counts_calls_and_prices_known_models(tmp_path):
    write(tmp_path, "big.py", long_file(40))
    write(tmp_path, "pkg/small.py", "x = 1\n")
    plan = plan_hub(tmp_path)
    est = plan.estimate("claude-haiku-5-5", "claude-opus-5-5")
    assert est["calls"] == {"file": 1, "section": 0, "dir": 2, "overview": 1}
    assert est["total_calls"] == 4
    assert est["tiny_files"] == 1
    assert est["cost_usd"] > 0
    assert plan.estimate("qwen2.5-coder:7b", "qwen2.5-coder:7b")["cost_usd"] == 0
    assert plan.estimate("claude-future-9", "claude-opus-5-5")["cost_usd"] is None


def test_empty_repo_is_an_error(tmp_path):
    with pytest.raises(GenerationError):
        plan_hub(tmp_path)


# --------------------------------------------------------------------- build


def test_build_summarizes_every_level_and_saves(repo):
    write(repo, "backend/service.py", long_file(40))
    fake = FakeModel()
    hub = build_hub(repo, models_for(fake))

    assert fake.kinds() == ["dir"] * 4 + ["file"] * 2 + ["overview"]
    assert hub["files"]["backend/service.py"]["purpose"] == "Purpose of backend/service.py."
    assert set(hub["dirs"]) == {"", "backend", "frontend", "frontend/src"}
    # Tiny files get their one-liner from their directory's reply.
    assert hub["dirs"]["frontend/src"]["small_files"]["api.ts"] == "small file api.ts"
    assert hub["overview"]["summary"].startswith("## What it does")
    assert hub["overview"]["diagram"] == GOOD_DIAGRAM
    assert hub["errors"] == {}
    assert json.loads((repo / ".docgen/hub.json").read_text()) == hub


def test_second_build_uses_the_cache(repo):
    build_hub(repo, models_for(FakeModel()))
    fake = FakeModel()
    plan = plan_hub(repo)
    assert plan.estimate("claude-haiku-5-5", "claude-opus-5-5")["total_calls"] == 0
    build_hub(repo, models_for(fake), plan)
    assert fake.calls == []


def test_editing_a_file_redoes_only_it_and_the_dirs_above(repo):
    write(repo, "backend/service.py", long_file(40))
    write(repo, "frontend/src/big.ts", long_file(40))
    build_hub(repo, models_for(FakeModel()))

    write(repo, "frontend/src/big.ts", long_file(40, tag="changed"))
    fake = FakeModel()
    plan = plan_hub(repo)
    assert plan.todo_files == ["frontend/src/big.ts"]
    assert plan.todo_dirs == ["frontend/src", "frontend", ""]
    build_hub(repo, models_for(fake), plan)
    assert fake.calls == [
        ("file", "frontend/src/big.ts"), ("dir", "frontend/src"), ("dir", "frontend"),
        ("dir", "(repo root)"), ("overview", "overview"),
    ]


def test_editing_a_tiny_file_redoes_its_dirs_but_no_file_call(repo):
    build_hub(repo, models_for(FakeModel()))
    write(repo, "backend/app.py", (repo / "backend/app.py").read_text() + "# note\n")
    fake = FakeModel()
    build_hub(repo, models_for(fake))
    assert fake.calls == [("dir", "backend"), ("dir", "(repo root)"), ("overview", "overview")]


def test_regenerate_redoes_everything(repo):
    build_hub(repo, models_for(FakeModel()))
    fake = FakeModel()
    build_hub(repo, models_for(fake), regenerate=True)
    assert fake.kinds() == ["dir"] * 4 + ["file", "overview"]


def test_deleted_files_and_dirs_are_dropped(repo):
    write(repo, "extra/thing.py", long_file(40))
    build_hub(repo, models_for(FakeModel()))
    shutil.rmtree(repo / "extra")
    hub = build_hub(repo, models_for(FakeModel()))
    assert "extra/thing.py" not in hub["files"]
    assert "extra" not in hub["dirs"]


def test_parallel_build_matches_sequential(repo):
    for i in range(6):
        write(repo, f"backend/mod{i}.py", long_file(40, tag=str(i)))
    fake = FakeModel()
    hub = build_hub(repo, models_for(fake, workers=4))
    assert len([s for k, s in fake.calls if k == "file" and "mod" in s]) == 6
    assert all(f"backend/mod{i}.py" in hub["files"] for i in range(6))


# -------------------------------------------------------------- long files


def test_split_sections_overlaps_and_covers_everything():
    text = "".join(f"{i:05d} " + "x" * 94 + "\n" for i in range(400))  # 100 chars per line
    sections = split_sections(text)
    assert len(sections) > 1
    assert sections[0][0] == 1 and sections[-1][1] == 400
    for (_, end, _), (start, _, _) in zip(sections, sections[1:]):
        assert start == end - hubmod.SECTION_OVERLAP_LINES + 1
    assert all(len(t) <= hubmod.SECTION_CHARS for _, _, t in sections)
    assert split_sections("short\n") == [(1, 1, "short\n")]


def test_long_file_is_summarized_in_sections_then_merged(tmp_path):
    write(tmp_path, "huge.py", "".join(f"value_{i} = {'x' * 80}\n" for i in range(400)))
    fake = FakeModel()
    hub = build_hub(tmp_path, models_for(fake))
    sections = [c for c in fake.calls if c[0] == "section"]
    assert len(sections) == hub["files"]["huge.py"]["sections"] > 1
    assert ("file", "huge.py") in fake.calls


# ---------------------------------------------------------------- failures


def test_unusable_reply_is_recorded_and_retried_next_time(repo):
    write(repo, "backend/service.py", long_file(40))
    fake = FakeModel()
    fake.fail_on = {"backend/service.py"}
    hub = build_hub(repo, models_for(fake))
    assert "file:backend/service.py" in hub["errors"]
    assert "backend/service.py" not in hub["files"]
    assert "backend" in hub["dirs"]  # the directory was still summarized

    fake = FakeModel()
    hub = build_hub(repo, models_for(fake))
    # The file now succeeds, and every directory that was built without it is redone.
    assert fake.calls[:3] == [("file", "backend/service.py"), ("dir", "backend"), ("dir", "(repo root)")]
    assert hub["errors"] == {}


def test_setup_error_stops_the_build(repo):
    write(repo, "backend/a.py", long_file(40, "a"))
    write(repo, "backend/b.py", long_file(40, "b"))
    calls = []

    def broken(system, user):
        calls.append(user)
        raise ProviderSetupError("Could not reach Ollama")

    with pytest.raises(ProviderSetupError):
        build_hub(repo, HubModels(broken, "m", broken, "m", workers=1))
    assert len(calls) == 1  # no further calls after the first setup error


def test_stop_keeps_finished_work_and_resumes(repo):
    for i in range(3):
        write(repo, f"backend/mod{i}.py", long_file(40, tag=str(i)))
    fake = FakeModel()
    with pytest.raises(HubStopped):
        build_hub(repo, models_for(fake), should_stop=lambda: len(fake.calls) >= 2)
    saved = load_hub(repo)
    assert len(saved["files"]) == 2 and saved["dirs"] == {}

    fake = FakeModel()
    build_hub(repo, models_for(fake))
    redone = [s for k, s in fake.calls if k == "file"]
    assert len(redone) == 2 and not set(redone) & set(saved["files"])  # 4 files in all


def test_progress_events(repo):
    events = []
    build_hub(repo, models_for(FakeModel()), on_progress=events.append)
    assert events[0]["stage"] == "start" and events[-1]["stage"] == "done"
    assert events[-1]["done"] == events[-1]["total"] == 6  # 1 file, 4 dirs, overview


# ---------------------------------------------------------------- diagrams


@pytest.mark.parametrize("source, ok", [
    (GOOD_DIAGRAM, True),
    ('flowchart LR\n  a["Label (with parens"] --> b', True),
    ("graph TD\n  a --> b", True),
    ("", False),
    ("```mermaid\nflowchart TD\n a-->b\n```", False),
    ("This diagram shows a --> b", False),
    ("flowchart TD\n  a[Unclosed --> b", False),
    ("flowchart TD\n  a] --> b", False),
])
def test_check_mermaid(source, ok):
    assert (check_mermaid(source) is None) is ok


def test_bad_diagram_is_fixed_once(repo):
    fake = FakeModel(diagram="Here is the diagram: a --> b")
    hub = build_hub(repo, models_for(fake))
    assert ("diagram_fix", "diagram") in fake.calls
    assert hub["overview"]["diagram"] == GOOD_DIAGRAM


def test_diagram_left_out_if_still_broken(repo):
    fake = FakeModel(diagram="nope", fixed_diagram="still nope")
    hub = build_hub(repo, models_for(fake))
    assert hub["overview"]["diagram"] is None
    assert "first line" in hub["overview"]["diagram_problem"]
    assert hub["overview"]["summary"]  # the overview text is kept


def test_fenced_diagram_is_accepted_without_a_fix_call(repo):
    fake = FakeModel(diagram=f"```mermaid\n{GOOD_DIAGRAM}\n```")
    hub = build_hub(repo, models_for(fake))
    assert hub["overview"]["diagram"] == GOOD_DIAGRAM
    assert ("diagram_fix", "diagram") not in fake.calls


# ----------------------------------------------------------------- prompts


def test_file_prompt_includes_import_context(tmp_path):
    write(tmp_path, "app/main.py", "from app.db import connect\nfrom .util import helper\n" + long_file(40))
    write(tmp_path, "app/db.py", long_file(40))
    write(tmp_path, "app/util.py", long_file(40))
    plan = plan_hub(tmp_path)
    _, user = hubmod.build_file_prompt(plan, "app/main.py")
    assert "Imports from this repo: app/db.py, app/util.py" in user
    _, user = hubmod.build_file_prompt(plan, "app/db.py")
    assert "Imported by: app/main.py" in user
    assert "Other files in its folder: main.py, util.py" in user


def test_dir_prompt_is_clipped_to_budget(tmp_path):
    for i in range(20):
        write(tmp_path, f"f{i}.py", "x = 1\n" * 20)
    plan = plan_hub(tmp_path)
    hub = load_hub(tmp_path)
    _, user = hubmod.build_dir_prompt(plan, hub, "", budget=2_000)
    assert "more characters not shown" in user
    assert len(user) < 4_000


def test_overview_prompt_falls_back_to_directories_only(tmp_path):
    for i in range(80):
        write(tmp_path, f"pkg{i}/mod.py", "x = 1\n")
    plan = plan_hub(tmp_path)
    _, user = hubmod.build_overview_prompt(plan, load_hub(tmp_path), budget=1_000)
    tree = re.search(r"<tree>\n(.*?)\n</tree>", user, re.S).group(1)
    assert "mod.py" not in tree


# ----------------------------------------------------------------- imports


def test_import_graph_python_and_typescript(tmp_path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/a.py", "from . import b\nfrom .sub.c import thing\nimport os\n")
    write(tmp_path, "pkg/b.py", "import pkg.a\n")
    write(tmp_path, "pkg/sub/c.py", "from ..b import x\n")
    write(tmp_path, "web/index.ts", 'import { x } from "./lib";\nimport y from "react";\n')
    write(tmp_path, "web/lib/index.ts", 'export * from "../util.js";\n')
    write(tmp_path, "web/util.js", 'const z = require("./index");\n')
    walked = walk_repo(tmp_path)
    texts = {f.path: f.abs_path.read_text() for f in walked.files}
    graph = import_graph(walked.files, texts)
    assert graph["pkg/a.py"] == {"pkg/b.py", "pkg/sub/c.py"}
    assert graph["pkg/b.py"] == {"pkg/a.py"}
    assert graph["pkg/sub/c.py"] == {"pkg/b.py"}
    assert graph["web/index.ts"] == {"web/lib/index.ts"}
    assert graph["web/lib/index.ts"] == {"web/util.js"}
    assert graph["web/util.js"] == {"web/index.ts"}
    assert reverse_graph(graph)["pkg/b.py"] == {"pkg/a.py", "pkg/sub/c.py"}


# ------------------------------------------------------- anthropic completer


class FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 20})()


def fake_anthropic(monkeypatch, response):
    import anthropic

    sent = {}

    class Client:
        def __init__(self, **kwargs):
            self.messages = self

        def create(self, **kwargs):
            sent.update(kwargs)
            return response

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    return sent


def test_anthropic_completer_sends_effort_and_meters_usage(monkeypatch):
    from docgen.llm import UsageMeter, make_anthropic_completer

    sent = fake_anthropic(monkeypatch, FakeResponse("hi"))
    meter = UsageMeter()
    complete = make_anthropic_completer("claude-haiku-5-5", api_key="k", effort="low", meter=meter)
    assert complete("sys", "user") == "hi"
    assert sent["output_config"] == {"effort": "low"}
    assert sent["max_tokens"] >= 16000
    assert "temperature" not in sent
    assert (meter.input_tokens, meter.output_tokens) == (100, 20)
    models = HubModels(complete, "claude-haiku-5-5", complete, "claude-haiku-5-5", file_meter=meter)
    assert models.cost_so_far() == pytest.approx((100 * 0.10 + 20 * 0.50) / 1e6, abs=1e-4)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_anthropic_completer_rejects_unusable_stops(monkeypatch, stop_reason):
    from docgen.llm import make_anthropic_completer

    fake_anthropic(monkeypatch, FakeResponse("partial", stop_reason))
    with pytest.raises(ProviderError):
        make_anthropic_completer("claude-haiku-5-5", api_key="k")("sys", "user")
