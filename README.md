# docgen

AI-powered documentation generator. Point it at a repo, and it documents your code using RAG over the whole codebase so the LLM understands cross-file context.

## Status

- [x] Phase 1: Repo walker
- [x] Phase 2: Chunk the codebase (functions, methods, classes, Flask/FastAPI endpoints)
- [x] Phase 3: Embed and index (ChromaDB) + cross-file retrieval
- [x] Phase 4: Retrieval-augmented generation (Claude API)
- [x] Phase 5: Write output (docstrings + API.md + OVERVIEW.md)
- [x] Phase 6a: Local web UI (the way to run docgen)

Next: a documentation hub for whole repos in any language (file, directory and repo summaries), then an "Ask the repo" Q&A feature built on RAG.

- [x] Hub phase 1: repo walker for every language, with `.gitignore` / `.docgenignore` and skip rules
- [x] Hub phase 2: bottom-up summaries (files, directories, overview + diagram) with a cache, from the command line
- [x] Hub phase 3: Hub tab in the web UI, Markdown export, public GitHub URLs
- [x] Hub phase 4: search index of every file's chunks and the hub's summaries (for "Ask the repo")
- [x] Hub phase 5: hybrid search (vector + BM25) and an eval set to measure it

## Usage

Requires Python 3.10+. Run all commands from this folder. On macOS the system `python3` may be older (3.9); use a newer one explicitly, for example Homebrew's `python3.12 -m venv .venv`.

```bash
python -m venv .venv
# activate: .venv\Scripts\activate (Windows)  or  source .venv/bin/activate (macOS/Linux)
pip install -r requirements.txt

python -m docgen.web
```

This opens `http://127.0.0.1:8765` in your browser. It runs only on your own machine (binds to 127.0.0.1, nothing public). Then:

1. **Scan** a repo by path (relative to this folder, or absolute) to list every function, class, and endpoint, and whether it already has docs.
2. **Build index** to embed the chunks (stored in `path/to/repo/.docgen/chroma`; the first run downloads the embedding model). Clicking a chunk shows the cross-file context the model would get for it.
3. **Generate** docs, one chunk at a time or **Generate all undocumented** (capped by the *limit* box, 20 by default, so one click can't run up a large bill). Pick Anthropic with a pasted API key, or Ollama with no key (see below).
4. **Generate overview** on the Documentation tab for a high-level "what does this repo do" summary, then read everything compiled in that tab.
5. **Preview changes** on the Review & apply tab to see a diff of every docstring plus `docs/API.md` and `docs/OVERVIEW.md`, then **Apply changes**.

Your Anthropic key is sent only with each generate request and is never written to disk. Change the port with `DOCGEN_WEB_PORT`, and skip the auto-opened browser tab with `DOCGEN_NO_BROWSER=1` (useful over SSH or in a container).

The overview is generated from a repo-wide map of signatures and routes, not full source, so it stays a reasonable size even on large repos — which also means it can miss nuance that only shows up in a function's actual implementation.

- Endpoint docs are tidied after generation: a section the model can't support (for example `Errors: None`) is removed rather than left in as a claim that nothing can fail, and section names get the blank line Markdown needs.

Generated docs are saved to `path/to/repo/.docgen/generated.json` for review; nothing in your source files changes until you click **Apply changes**. Re-running skips chunks whose source hasn't changed, so you only pay for new or edited code. Tick *regenerate existing* to redo everything, and use the *Model* box to pick a different Claude model (default: `claude-haiku-5-5`, chosen for cost across many chunks).

Try it on the included sample: `example_repo`.

## Tests

```bash
python -m pytest tests -q
```

The tests fake the models (including a small fake Ollama server) and drive the web UI through FastAPI's TestClient (no real browser, no real server process), so everything runs offline and costs nothing. They cover prompt construction, output parsing, the context budget, generation's save/resume behavior, and the writer's edge cases (decorators, one-liners, CRLF files, awkward text, stale docs, verification failures, API.md safety).

## Look and feel

The Documentation tab reads `generated.json` and renders it for humans; it never touches your source files or writes `docs/API.md` on its own.

Styled as a plain dark dev-tool (GitHub Dark-style grays and blue), deliberately generic rather than personally branded, and self-contained: no external fonts or scripts, and the Markdown rendering is a small renderer built into the page rather than a CDN library, so the Documentation tab still works with zero internet access — consistent with the Ollama provider's whole point.

## Using a free local model (Ollama)

You don't need an Anthropic key. [Ollama](https://ollama.com) runs open models on your own machine, free and offline after the model is downloaded.

```bash
# 1. Install Ollama from ollama.com (it runs in the background after install)
# 2. Download a model once (several GB):
ollama pull qwen2.5-coder:7b
# 3. In the web UI, choose "ollama" as the provider and generate a couple of chunks first.
```

- **Model choice:** the default is `qwen2.5-coder:7b`, which runs on most machines. With a GPU that has 24 GB of memory, a bigger model such as `qwen2.5-coder:32b` gives noticeably better docs (`ollama pull qwen2.5-coder:32b`, then type `qwen2.5-coder:32b` in the *Model* box). Newer coding models appear often; browse ollama.com/library.
- **Server location:** defaults to `http://localhost:11434`. Set the `OLLAMA_HOST` environment variable to use another machine.
- **First request is slow** while the model loads into memory; later ones are faster.
- **Quality:** small open models are more likely than Claude to describe things the code doesn't show, or to ignore the required output format (those chunks are skipped and the start of the reply is shown so you can see why). Always read **Preview changes** before applying.
- docgen asks Ollama for an 8192-token context window, because Ollama's small default would silently cut off long prompts.

## What applying does

- Inserts each generated docstring directly above the first statement of its function, method, or class.
- Writes `docs/API.md` from the generated endpoint docs (endpoint functions themselves are not given a docstring; their reference lives in API.md). Untick *write docs/API.md* to skip it.
- Writes `docs/OVERVIEW.md` from the generated repo overview, if you've generated one. Skipped automatically if the repo's structure has changed since the overview was generated (generate it again first).
- Never overwrites an existing docstring, and never overwrites an `API.md` or `OVERVIEW.md` that docgen did not create. (The `/api/apply` endpoint accepts `force`, `docs_dir`, and `overview` options for this, but the page doesn't expose them.)

Safety checks, since this step edits your code:

1. **Nothing is written until everything is planned and verified.** You see the diff first, and nothing changes until you click Apply.
2. **Stale docs are skipped.** Each generated doc records a hash of the code it described; if the code has since changed, it is skipped and you are told to re-run `generate`.
3. **Every edited file is re-checked** before it is written: it must still parse, gain exactly the expected number of AST nodes, and read each new docstring back correctly. A file that fails is left untouched and reported.
4. **Formatting is preserved:** Windows (CRLF) line endings stay as they are, decorators stay attached to their functions, and awkward text (backslashes, triple quotes) is escaped so it reads back exactly.

Run it on a git repo so you can review and undo with `git diff` / `git checkout`.

## Building the hub

The hub explains a whole repo in any language, bottom-up:

1. **Files.** Each file gets a summary (purpose, key components, dependencies, what uses it). Its context is the repo files it imports, the files that import it, and its folder neighbors; imports are found for Python and JavaScript/TypeScript. Files too long for one prompt are summarized in overlapping sections, then merged. Files under 30 lines get no call of their own; their text goes into their directory's prompt.
2. **Directories**, deepest first, from their files' and subdirectories' one-line purposes.
3. **Overview**: what the repo does, stack, architecture, entry points, data flow, business rules, how to run, plus a Mermaid diagram. A diagram that fails a basic syntax check gets one repair attempt, and is left out (with a note) if that fails too.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m docgen.hub path/to/repo                      # Claude: claude-haiku-5-5 for files, claude-opus-5-5 above
python -m docgen.hub path/to/repo --provider ollama    # local and free, one call at a time
```

It prints the number of model calls and a rough cost estimate (leaning high, since the models' thinking can't be predicted) and asks before spending anything; at the end it prints what was actually spent, from the API's token counts. Use `--file-model` / `--top-model` to pick models, `--yes` to skip the question, and `--regenerate` to redo everything.

Results go to `path/to/repo/.docgen/hub.json`. Each summary is saved as soon as it's done, under a key built from what it was made from, so:

- Running again only redoes files whose text changed, plus the directories above them and the overview.
- Ctrl+C keeps everything finished so far; run again to continue.
- A reply in the wrong format is recorded under `errors` and retried on the next run; a setup problem (bad key, unknown model, Ollama not running) stops the run.

### In the web UI

Start `python -m docgen.web`, enter a path or a public GitHub URL, and click Scan. The **Hub** tab shows the repo as a tree, with a dot per file and folder: green is summarized, yellow means the code changed since, red means the last attempt failed, and hollow is a small file covered by its folder's summary. Click anything to read its summary; the root shows the overview and the diagram's Mermaid source (paste it into [mermaid.live](https://mermaid.live) to see it drawn).

**Build hub** shows the call count and estimated cost and asks before starting. The build runs in the background with a progress bar and a **Stop** button; stopping lets the calls already sent finish, and keeps everything done so far. Leave the two model fields empty for the defaults.

### GitHub repos

A public `https://github.com/owner/repo` URL is shallow-cloned into `~/.cache/docgen/owner/repo` (or `$DOCGEN_CACHE_DIR`), and each Scan pulls the latest commit. The hub cache lives in the clone, so a rebuild after new commits only redoes what changed. If the pull fails (offline, say), the existing copy is used with a warning. The CLI takes URLs too.

### Markdown export

**Export Markdown** (or `--export` on the command line) writes one `README.md` per folder: the top one has the overview, the diagram as a `mermaid` code block (GitHub draws these) and the root folder; each folder's README has its role, a table of what's in it, how it fits together, and its files' summaries, with links to the source. A local repo exports to `docs/hub/`, which the walker always skips so the hub never summarizes its own output. A GitHub repo exports to `./output/owner-repo/`. Re-exporting removes READMEs it wrote earlier for folders that no longer exist, and nothing else.

## The search index (for "Ask the repo")

Questions about a repo are answered from a search index: a ChromaDB collection in `.docgen/chroma`, separate from the docstring index. It holds two types of entry:

- **Code chunks** from every file the walker reads. Python is split with `ast` into top-level functions and classes (with their decorators and leading comments), with long classes split into methods and the code between definitions kept as "module" chunks. Markdown is split at headings, and each chunk keeps its heading path, such as `Install > macOS`. Everything else is split into 60-line windows that overlap by 10 lines. Every chunk records its file and line range, which answers will cite.
- **Summary parts**: the hub's file, folder and overview summaries, split at their headings.

Click **Update search index** in the Hub tab, or run:

```bash
python -m docgen.repo_index path/to/repo                       # update
python -m docgen.repo_index path/to/repo --query "how are GitHub repos cloned"
```

Only files and summaries whose text changed since the last run are embedded again (`.docgen/repo_index.json` keeps each one's hash). Changing the chunking rules or the embedding model rebuilds the whole index. Run it again after building the hub so the new summaries are included. `--query` prints the closest entries, which is a quick way to see what vector search finds and what it misses.

### Hybrid search

`docgen/search.py` runs two searches and merges them. Vector search matches meaning ("downloaded" finds `git clone`) but is weak on exact names; BM25 keyword search finds `estimate` at once but knows no synonyms. The two rankings are merged with reciprocal rank fusion, which only looks at ranks, so the two kinds of score never need to be made comparable. The web server exposes it at `GET /api/search?repo=...&q=...&mode=hybrid|vector|keyword`.

### Measuring it

`evals/docgen.json` has 20 questions about docgen's own code, each with the files that answer it. Run:

```bash
python -m docgen.evaluate evals/docgen.json --misses
```

It updates the index, runs every question through each search mode, and prints the hit rate (the share of questions where a right file is in the top k) and MRR (how high the first right file ranks). On docgen itself, top 5:

| mode | hit rate | MRR |
|---|---|---|
| hybrid | 100% | 0.75 |
| vector | 85% | 0.64 |
| keyword | 85% | 0.54 |

The questions were written by someone who knew the code, so treat these as optimistic. To evaluate another repo, copy the file, point `"repo"` at it (a path or GitHub URL) and write questions whose answers you know. Then change one thing at a time, such as chunk size or the embedding model, and compare.

## Which files are read

docgen walks the whole repo and reads every text file, in any language, except:

- Anything matched by `.gitignore`, or by a `.docgenignore` (same syntax, for things you want in git but not in the docs). Both are honored in subdirectories too, as in git.
- `node_modules`, `.venv`, `venv`, `dist`, `build`, `.git`, `.docgen` and cache folders, even without a `.gitignore`.
- Lockfiles (`package-lock.json`, `poetry.lock`, `*.lock`, ...), minified files (`*.min.*`) and source maps.
- Binaries, files that aren't UTF-8, files over 200 KB, and symlinks (never followed).

**Scan** lists how many files were read per language and, under *skipped*, every file left out by the last two rules with the reason. Files matched by ignore rules are left out without being listed.

Docstring generation still works on Python files only; the other languages are for the upcoming hub. `example_multilang` is a small Python + TypeScript sample for trying this.

## How retrieval works

For a target chunk, context comes from three sources:

1. **Parent class**, if the target is a method.
2. **Call graph**: chunks the target calls, then the chunks those call, up to three levels deep (each item is labeled with who called it). Capped at 5 direct and 6 deeper chunks so a wide call tree can't flood the prompt. A same-file match wins over other files, mirroring Python's name resolution, and deeper levels resolve names against the caller's own file.
3. **Semantic neighbors**: nearest chunks in the vector index.

## Known limitations

- Call-following stops at three levels and is capped, so a fact hidden deeper than that (or beyond the caps) is invisible to the model. Docs describe only what the model was shown.
- One-line definitions (`def f(): return 1`) are skipped by `apply` rather than rewritten.
- Docstring line length is not adjusted; run your formatter (black, ruff) after applying if you enforce one.
- Neither model client has been run against the real service by the author of this scaffold (no API key and no Ollama install were available); both are tested against fakes that follow the documented protocols. The web UI's own logic (routes, path handling, error responses) was hand-tested end to end against a real local Ollama install and a real browser during development, in addition to its automated tests; only the two model clients underneath it remain untested against the real services. Generate two or three chunks first and read the output before generating at scale.
- Generated docs describe only what the retrieved context shows. If the call-graph match misses a dependency, the doc may describe it vaguely; click the chunk in the web UI to see the context it got when a doc looks thin.
- Docstring generation is Python only (`ast`). Other languages are scanned but not yet documented.
- Call matching is by bare name. `obj.get()` on a plain dict can match a repo function named `get`, and a call resolved to another file can collide if two files define the same name.
- The index is rebuilt from scratch every time you click Build index.
- `DOCGEN_EMBEDDINGS=hash` selects an offline test embedder that matches on shared words, not meaning. Leave it unset to use the real model (downloaded on first run).
