# docgen

Point it at a repo, in any language, and it explains it. It writes a documentation hub of file, folder and overview summaries with an architecture diagram, and it answers questions about the code using RAG (retrieval-augmented generation), with `file:line` citations you can check.

## Status

- [x] Phase 1: repo walker for every language, with `.gitignore` / `.docgenignore` and skip rules
- [x] Phase 2: bottom-up summaries (files, directories, overview + diagram) with a cache
- [x] Phase 3: Hub tab in the web UI, Markdown export, public GitHub URLs
- [x] Phase 4: search index of every file's chunks and the hub's summaries
- [x] Phase 5: hybrid search (vector + BM25) and an eval set to measure it
- [x] Phase 6: Ask tab, questions answered from search results with `file:line` citations

## Usage

Requires Python 3.10+. Run all commands from this folder. On macOS the system `python3` may be older (3.9); use a newer one explicitly, for example Homebrew's `python3.12 -m venv .venv`.

```bash
python -m venv .venv
# activate: .venv\Scripts\activate (Windows)  or  source .venv/bin/activate (macOS/Linux)
pip install -r requirements.txt

python -m docgen.web
```

This opens `http://127.0.0.1:8765` in your browser. It runs only on your own machine (binds to 127.0.0.1, nothing public). Then:

1. **Scan** a repo: a path (relative to this folder, or absolute) or a public GitHub URL.
2. Pick a provider: Anthropic with a pasted API key, or Ollama with no key (see below).
3. **Build hub** on the Hub tab to summarize the repo. It shows the cost estimate and asks first.
4. **Update search index**, then ask questions on the **Ask** tab.

Your Anthropic key is sent only with each hub build or question and is never written to disk. Change the port with `DOCGEN_WEB_PORT`, and skip the auto-opened browser tab with `DOCGEN_NO_BROWSER=1` (useful over SSH or in a container).

Try it on the included sample, `example_multilang`, a small Python + TypeScript app. Everything docgen stores for a repo goes in that repo's `.docgen/` folder; add it to the repo's `.gitignore`.

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

It prints the number of model calls and a rough cost estimate (leaning high, since the models' thinking can't be predicted) and asks before spending anything; at the end it prints what was actually spent, from the API's token counts. Use `--file-model` / `--top-model` to pick models, `--yes` to skip the question, `--regenerate` to redo everything, and `--export` to write Markdown afterwards.

Results go to `path/to/repo/.docgen/hub.json`. Each summary is saved as soon as it's done, under a key built from what it was made from, so:

- Running again only redoes files whose text changed, plus the directories above them and the overview.
- Ctrl+C keeps everything finished so far; run again to continue.
- A reply in the wrong format is recorded under `errors` and retried on the next run; a setup problem (bad key, unknown model, Ollama not running) stops the run.

### In the web UI

The **Hub** tab shows the repo as a tree, with a dot per file and folder: green is summarized, yellow means the code changed since, red means the last attempt failed, and hollow is a small file covered by its folder's summary. Click anything to read its summary; the root shows the overview and the diagram's Mermaid source (paste it into [mermaid.live](https://mermaid.live) to see it drawn).

**Build hub** shows the call count and estimated cost and asks before starting. The build runs in the background with a progress bar and a **Stop** button; stopping lets the calls already sent finish, and keeps everything done so far. Leave the two model fields empty for the defaults.

### GitHub repos

A public `https://github.com/owner/repo` URL is shallow-cloned into `~/.cache/docgen/owner/repo` (or `$DOCGEN_CACHE_DIR`), and each Scan pulls the latest commit. The hub cache lives in the clone, so a rebuild after new commits only redoes what changed. If the pull fails (offline, say), the existing copy is used with a warning. The command-line tools take URLs too.

### Markdown export

**Export Markdown** (or `--export` on the command line) writes one `README.md` per folder: the top one has the overview, the diagram as a `mermaid` code block (GitHub draws these) and the root folder; each folder's README has its role, a table of what's in it, how it fits together, and its files' summaries, with links to the source. A local repo exports to `docs/hub/`, which the walker always skips so the hub never summarizes its own output. A GitHub repo exports to `./output/owner-repo/`. Re-exporting removes READMEs it wrote earlier for folders that no longer exist, and nothing else.

## The search index

Questions are answered from a search index: a ChromaDB collection in `.docgen/chroma`. It holds two types of entry:

- **Code chunks** from every file the walker reads, split along the code's own structure. Every chunk records its file and line range, which answers cite.
  - **Python** is split with `ast` into top-level functions and classes (with their decorators and leading comments). Long classes are split into methods, and the code between definitions is kept as "module" chunks.
  - **TypeScript, JavaScript, Go, Rust, Java, C#, Ruby, PHP, Kotlin, Swift, C/C++** and a few more are parsed with [tree-sitter](https://tree-sitter.github.io/) and split the same way: functions, classes, interfaces, types, structs and `const Page = () => {...}` components become chunks named after what they define. A definition over 120 lines is split into what it contains, such as a React component's inner handlers (`DocsPage.generateDocumentation`). Each language's grammar downloads on first use (a few MB) into `~/Library/Caches/tree-sitter-language-pack/` on macOS; set `DOCGEN_TREE_SITTER=0` to turn this off.
  - **Markdown** is split at headings, and each chunk keeps its heading path, such as `Install > macOS`.
  - **Everything else**, and any file a parser can't handle (or whose grammar couldn't be downloaded), is split into 60-line windows that overlap by 10 lines.
- **Summary parts**: the hub's file, folder and overview summaries, split at their headings.

Click **Update search index** in the Hub tab, or run:

```bash
python -m docgen.repo_index path/to/repo                       # update
python -m docgen.repo_index path/to/repo --query "how are GitHub repos cloned"
```

Only files and summaries whose text changed since the last run are embedded again (`.docgen/repo_index.json` keeps each one's hash). Changing the chunking rules or the embedding model rebuilds the whole index. Run it again after building the hub so the new summaries are included. `--query` prints the closest entries, which is a quick way to see what vector search finds and what it misses.

The embedding model is all-MiniLM-L6-v2, run locally by ChromaDB and downloaded on first use (about 80 MB). It reads only the first 256 word pieces of a chunk. `DOCGEN_EMBEDDINGS=ollama` uses nomic-embed-text through a local Ollama server instead (`ollama pull nomic-embed-text`; `ollama:<model>` picks another), which reads whole chunks. `DOCGEN_EMBEDDINGS=hash` swaps in an offline embedder that matches shared words rather than meaning; the tests use it. Changing the choice rebuilds the index on its next update, since vectors from different models can't be compared.

### Hybrid search

`docgen/search.py` runs two searches and merges them. Vector search matches meaning ("downloaded" finds `git clone`) but is weak on exact names; BM25 keyword search finds `estimate` at once but knows no synonyms. The two rankings are merged with reciprocal rank fusion, which only looks at ranks, so the two kinds of score never need to be made comparable. `HYBRID_ALPHA` weights the two lists (0.5, the default, is plain RRF; 1.0 is vector only); see below for why it stays at 0.5. The web server exposes it at `GET /api/search?repo=...&q=...&mode=hybrid|vector|keyword`.

### Measuring it

`evals/docgen.json` has 20 questions about docgen's own code, each with the files that answer it. Run:

```bash
python -m docgen.evaluate evals/docgen.json --misses
```

It updates the index, runs every question through each search mode, and prints the hit rate (the share of questions where a right file is in the top k) and MRR (how high the first right file ranks). On docgen itself:

| mode | hit rate, top 5 | hit rate, top 3 | MRR, top 5 |
|---|---|---|---|
| hybrid | 95% | 85% | 0.66 |
| vector | 95% | 80% | 0.59 |
| keyword | 80% | 65% | 0.50 |

`evals/gitsummarize.json` has 18 questions about [gitsummarize](https://github.com/antarixxx/gitsummarize), mostly about its TypeScript frontend, for measuring chunking in languages other than Python. Switching TypeScript from 60-line windows to tree-sitter chunks changed it like this (top 5):

| mode | windows | tree-sitter |
|---|---|---|
| hybrid | 94%, MRR 0.76 | 94%, MRR 0.81 |
| vector | 61%, MRR 0.49 | **83%**, MRR 0.65 |
| keyword | 100%, MRR 0.82 | 94%, MRR 0.81 |

**Embedding models.** Hybrid search over both question sets (38 questions), MiniLM against nomic-embed-text:

| model | hit rate, top 5 | hit rate, top 3 |
|---|---|---|
| all-MiniLM-L6-v2 | 37/38 | 34/38 |
| nomic-embed-text | 38/38 | 32/38 |

nomic did better on gitsummarize in every mode but ranked worse in the top 3 on docgen. With this few questions one question is worth 5–6 points, so neither is a clear winner and MiniLM stays the default.

**Weighting the two halves.** Plain RRF trusts both lists equally, so a result only one half finds gets pushed down: on docgen, `imports.py` was rank 1 in vector search but rank 4 after fusion, and on 8 of 20 questions hybrid ranked the right file lower than vector search alone. `--alpha` weights the vector list more. Tuned on docgen, α ≈ 0.6–0.8 was best (MiniLM at α = 0.8: top-3 hit rate 100%, MRR 0.73). Checked on gitsummarize, which it wasn't tuned on, every α above 0.5 was *worse* for both models (MiniLM top-5 hit rate 94% → 89%, MRR 0.81 → 0.75). One weight doesn't carry over between repos, so the default stays at plain RRF; reranking the fused results is the better fix.

**Retrieval-augmented answers against a tool-using agent.** The same local model (qwen2.5-coder:7b) answered the 18 gitsummarize questions two ways, graded by hand against the code:

| approach | correct | partly | wrong | median time |
|---|---|---|---|---|
| Ask (hybrid search, 8 sources in the prompt) | 16 | 1 | 1 | 35 s |
| agent with `list_files` / `grep` / `read_file` tools, up to 10 turns | 3 | 4 | 11 | 45 s |

Ask retrieved the right file for 17 of 18 questions, but only 9 answers included citations. The agent gave up twice at the turn limit, made up code or files four times, and used up to 44K input tokens on one question. A 7B model isn't good at choosing what to read next; fixed retrieval chooses for it. On a repo small enough to fit (`example_multilang`, about 1.3K tokens), putting the whole repo in the prompt was best: 4 of 4 correct, with exact citations.

The questions were written by someone who knew the code, so treat these as optimistic. To evaluate another repo, copy the file, point `"repo"` at it (a path or GitHub URL) and write questions whose answers you know. Then change one thing at a time, such as chunk size or the embedding model, and compare.

## Asking questions

The **Ask** tab answers questions about the repo. The question goes through hybrid search; the best results (8 by default) go into the prompt as numbered sources, and the model is told to answer only from them and to cite each claim like `[2]`. Each citation becomes a link such as `store.py:12-30`; clicking it opens that file in the Hub with the cited lines shown. **Retrieved sources** lists everything search found, which ones were cited, and each one's rank in the vector and keyword halves, so you can see why an answer was good or bad: usually the right code either was or wasn't retrieved.

The search index is updated before each question (quick when little changed), but it has to exist first: click **Update search index** in the Hub tab once. The default model is `claude-sonnet-5-5`; a question costs a few cents, shown under the answer. With Ollama it's free, with a smaller prompt to fit the local model's context.

From the command line:

```bash
python -m docgen.ask path/to/repo "How are GitHub repos cloned?" --sources
```

Each question is answered on its own, with no memory of earlier ones. If an answer cites a source number that doesn't exist, or cites nothing, the UI says so.

## Using a free local model (Ollama)

You don't need an Anthropic key. [Ollama](https://ollama.com) runs open models on your own machine, free and offline after the model is downloaded.

```bash
# 1. Install Ollama from ollama.com (it runs in the background after install)
# 2. Download a model once (several GB):
ollama pull qwen2.5-coder:7b
# 3. In the web UI, choose "Ollama" as the provider.
```

- **Model choice:** the default is `qwen2.5-coder:7b`, which runs on most machines. With a GPU that has 24 GB of memory, a bigger model such as `qwen2.5-coder:32b` gives noticeably better results (`ollama pull qwen2.5-coder:32b`, then type it in the model fields). Newer coding models appear often; browse ollama.com/library.
- **Server location:** defaults to `http://localhost:11434`. Set the `OLLAMA_HOST` environment variable to use another machine.
- **One call at a time:** hub builds with Ollama run sequentially, and prompts are kept smaller to fit the local model's context. docgen asks Ollama for an 8192-token context window, because Ollama's small default would silently cut off long prompts.
- **First request is slow** while the model loads into memory; later ones are faster.
- **Quality:** small open models are more likely than Claude to describe things the code doesn't show, or to ignore the required output format (those summaries are recorded as errors and retried on the next build).

## Which files are read

docgen walks the whole repo and reads every text file, in any language, except:

- Anything matched by `.gitignore`, or by a `.docgenignore` (same syntax, for things you want in git but not in the docs). Both are honored in subdirectories too, as in git.
- `node_modules`, `.venv`, `venv`, `dist`, `build`, `.git`, `.docgen` and cache folders, even without a `.gitignore`, plus `docs/hub` (the hub's own export).
- Lockfiles (`package-lock.json`, `poetry.lock`, `*.lock`, ...), minified files (`*.min.*`) and source maps.
- Binaries, files that aren't UTF-8, files over 200 KB, and symlinks (never followed).

**Scan** lists how many files were read per language and, under *skipped*, every file left out by the last two rules with the reason. Files matched by ignore rules are left out without being listed.

## Tests

```bash
python -m pytest tests -q
```

The tests fake the models (including a small fake Ollama server), use the offline embedder, clone from a local git repo standing in for GitHub, and drive the web UI through FastAPI's TestClient, so everything runs offline and costs nothing.

## Look and feel

Styled as a plain dark dev-tool (GitHub Dark-style grays and blue), and self-contained: no external fonts or scripts, and the Markdown rendering is a small renderer built into the page rather than a CDN library, so the UI still works with no internet access, which is the point of the Ollama provider.

## Known limitations

- The Ollama client has been run against a real server (qwen2.5-coder:7b and nomic-embed-text), which is where the measurements above come from. The Anthropic client has only been tested against a fake that follows the documented protocol, since no API key was available. Build the hub for a small repo and ask a couple of questions first, and read the output, before using it at scale.
- With a 7B local model, Ask often leaves out citations, the hub summaries can crowd code out of the sources for broad questions, and an answer takes 18–40 seconds.
- Summaries and answers describe only what the model was shown, and can still be wrong. Answers cite their sources so you can check them; summaries don't.
- Imports are found with patterns for Python and JavaScript/TypeScript only. Files in other languages get their folder neighbors as context instead.
- The embedding model reads only the first 256 word pieces of a chunk, so the end of a long chunk is barely searchable.
- Line numbers for tree-sitter chunks come from byte offsets, because reading a node's `start_point.row` crashes Python with py-tree-sitter 0.26.0.
- Test files are indexed like everything else and often take up search result slots.
- **Update search index** runs inside the request, so the page waits while a large repo is embedded for the first time.
- The diagram is shown as Mermaid source in the web UI, not drawn.
