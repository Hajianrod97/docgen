# How docgen was measured

Every retrieval and answering choice in docgen was measured before it was kept. This page has the numbers; the [README](../README.md) has the summary.

## Design decisions

Each was measured on a tuning question set (docgen's own code) and checked on a held-out one (gitsummarize). The sections below have the details.

| Decision | Evidence |
|---|---|
| **Hybrid search, not vector only** | Held-out top-5 file hit: hybrid 88%, vector 81%, keyword 83%. Each half covers the other's blind spot. |
| **Postgres (pgvector + ParadeDB pg_search), not Pinecone or Chroma** | One database for vectors, keywords and the manifest, with transactions keeping them consistent. Every held-out hit rate stayed the same after the move; hybrid MRR went from 0.74 to 0.78. Runs locally in Docker. |
| **Chunk by syntax, not fixed windows** | Switching TypeScript to tree-sitter chunks raised vector search's hit rate from 61% to 83%. |
| **Plain RRF, not tuned weights** | Weights tuned on one repo made the other worse (hit rate 94% → 89%). |
| **A 120 MB reranker, not the planned 1 GB one** | Of four rerankers, bge-reranker-base (the plan's pick) made ranking worse; ms-marco-MiniLM-L-12 raised tuning-set hit rate from 94% to 100%. It was about even on the held-out set, so only Agentic mode uses it. |
| **Agentic mode decides with rules, not the model** | A tool-using agent on the local 7B model got 3 of 18 answers right, against 16 of 18 for plain RAG: small models choose badly what to read. So the graph's decisions are a reranker score and a citation count; the model only writes. |
| **A citation check in Agentic mode** | Answers citing a source: 50% (Quick) → 100% (Agentic); citing a right file: 50% → 90%, for 8 seconds more. |
| **openevals, not Ragas** | Ragas had no release since January 2026, an open security advisory, and LangChain 0.3 pins. |
| **Tune on one repo, report on another** | The weighting and reranker results both looked good on the tuning set and didn't hold up on the held-out one. |

## The question sets

Two sets, each question labelled with the files that answer it and, more precisely, the chunks: a path plus a piece of text the right chunk contains, such as `docgen/hub.py :: def _tiny_description`.

- `evals/docgen.json`: 51 questions about docgen's own Python code. **Used for tuning.**
- `evals/gitsummarize.json`: 48 questions about [gitsummarize](https://github.com/antarixxx/gitsummarize), a Python backend with a TypeScript frontend. **Held out**: settings are chosen on docgen and only checked here, so a change can't be fitted to one repo.

With about 50 questions, one question is worth 2 points: differences smaller than that are noise. The questions were written with knowledge of the code, so treat the numbers as optimistic.

## Search

```bash
python -m docgen.evaluate evals/docgen.json --misses [--rerank]
```

It updates the index, runs every question through each search mode, and prints the **hit rate** (the share of questions with a right file in the top k), **MRR** (how high the first right result ranks: 1 for first place, 0.5 for second, ...), and the same two for the right *chunk*. The file numbers are lenient: any chunk of an 880-line file counts, even one about something else.

Top 5, MiniLM embeddings, plain RRF:

| mode | docgen file | docgen chunk | gitsummarize file | gitsummarize chunk |
|---|---|---|---|---|
| **hybrid** | 86%, MRR 0.61 | 49%, MRR 0.37 | **88%, MRR 0.78** | **79%, MRR 0.62** |
| vector | 84%, MRR 0.61 | 47%, MRR 0.32 | 81%, MRR 0.65 | 75%, MRR 0.48 |
| keyword | 71%, MRR 0.45 | 41%, MRR 0.25 | 83%, MRR 0.71 | 81%, MRR 0.63 |

The docgen columns move whenever docgen's own code or docs change, since that is the repo being searched (see [Comments are search material](#comments-are-search-material)); the evals don't pull new gitsummarize commits, so its columns stay put.

Hybrid search beats either half on its own: vector search misses exact names (`estimate`), BM25 misses synonyms ("downloaded" for `git clone`). Search usually finds the right file but often not the right part of it, especially in docgen's long modules.

### Query rewriting

A local model (qwen2.5-coder:7b) turning the question into likely identifiers first, with the repo's file paths to go on:

| | docgen file | gitsummarize file |
|---|---|---|
| hybrid | 96%, MRR 0.69 | 88%, MRR 0.77 |
| hybrid, question + guessed identifiers | 87%, MRR 0.61 | **96%**, MRR 0.76 |

It helped on one repo and hurt on the other, so search doesn't rewrite on its own. Agentic mode does, only when the sources it found look irrelevant.

### Chunking

Switching TypeScript from 60-line windows to tree-sitter chunks along the code's own definitions (first, smaller gitsummarize set, 18 questions, top 5):

| mode | windows | tree-sitter |
|---|---|---|
| hybrid | 94%, MRR 0.76 | 94%, MRR 0.81 |
| vector | 61%, MRR 0.49 | **83%**, MRR 0.65 |
| keyword | 100%, MRR 0.82 | 94%, MRR 0.81 |

### Embedding models

Hybrid search over both early sets (38 questions), MiniLM against nomic-embed-text:

| model | hit rate, top 5 | hit rate, top 3 |
|---|---|---|
| all-MiniLM-L6-v2 | 37/38 | 34/38 |
| nomic-embed-text | 38/38 | 32/38 |

With one question worth 5–6 points, neither is a clear winner. MiniLM stayed: nomic needed Ollama running for every index update and search, for no measured gain.

### Weighting the two halves

Plain RRF trusts both lists equally, so a result only one half finds gets pushed down: on docgen, `imports.py` was rank 1 in vector search but rank 4 after fusion. Tuned on docgen, weighting vector results more (α ≈ 0.6–0.8) was best (top-3 hit rate 100%, MRR 0.73). On gitsummarize, which it wasn't tuned on, every α above 0.5 was *worse* (top-5 hit rate 94% → 89%, MRR 0.81 → 0.75). One weight doesn't carry over between repos, so the default stays at plain RRF.

### Comments are search material

docgen's module docstrings were cut from about 320 lines to 85, keeping what each file does and moving results and history into these docs. Measured on the docgen set (top 5, hybrid), with nothing else changed:

| | file hit | chunk hit |
|---|---|---|
| long docstrings | 94% | 51% |
| short docstrings | 86–88% | 47–49% |

About 3 of 51 questions stopped finding their file. A docstring that says in plain words what a file does is exactly what a question phrased in plain words matches; without it, search has to match the question against the code alone. Several of the misses found the right explanation in `docs/` instead, which the labels don't count. Adding `docs/guide.md` beforehand had moved hybrid from 96% to 94% the same way, by competing with the code for the top results. The trim was kept for readability: in a code search system, how much to comment is a trade-off between the people reading the code and the search reading it.

### Changing the database

Swapping the vector store and keyword index (ChromaDB + in-memory BM25 → Postgres with pgvector and ParadeDB's pg_search) left every gitsummarize hit rate unchanged, and hybrid MRR went from 0.74 to 0.78.

## Reranking

Four fastembed cross-encoders reordering hybrid search's top 20, on the docgen set (top 5):

| reranker | size | file hit | MRR | chunk hit |
|---|---|---|---|---|
| none (hybrid only) | | 94–96% | 0.68–0.70 | 58% |
| bge-reranker-base | 1 GB | 94% | 0.64 | 50% |
| **ms-marco-MiniLM-L-12-v2** | 120 MB | **100%** | **0.75** | 58% |
| jina-reranker-v1-turbo-en | 150 MB | 90% | 0.70 | 52% |

The plan had picked bge-reranker-base; it made ranking worse, likely because it was trained on prose rather than code. On the held-out set MiniLM-L-12 was about even with hybrid search (file hit 88% both, MRR 0.78 → 0.74, chunk hit 79% → 83%), so Quick mode doesn't rerank. Agentic mode uses it mostly for its *score*, as a relevance check. Its threshold of -2.0 comes from the docgen set, where every question has an answer: it sends 4 of 52 through a rewrite they didn't need.

## Answers

```bash
python -m docgen.answer_eval evals/gitsummarize.json --provider ollama --no-judge --limit 10 [--agent]
```

Search evals check that the right file was *found*; this checks what the model then *said*. Two checks need no judge, from the eval labels: does the answer cite a source, and does it cite one of the files the question lists.

First 10 gitsummarize questions, answered by qwen2.5-coder:7b:

| | Quick | Agentic |
|---|---|---|
| answer cites a source | 50% | **100%** |
| cites a right file | 50% | **90%** |
| median time | 23 s | 31 s |

The agent rewrote no queries here (retrieval was already good). The gain came from its citation check, which sent 5 of the 10 answers back once with "cite your sources" and fixed all 5. The uncited Quick answers were often correct, but nothing tied them to the code.

### The judge

openevals' LLM-as-judge prompts (groundedness, helpfulness, retrieval relevance) are wired in, but no judge is trusted yet. Ragas, the usual choice, was passed over: no release since January 2026, an open security advisory, and install docs pinned to LangChain 0.3. With qwen2.5-coder:7b as the judge, openevals called the sources of three correct, well-retrieved answers irrelevant, once contradicted its own written reason, and sometimes returned no verdict at all. A stronger judge (`--judge-provider anthropic`) is the next step, and its scores need checking against a person's grades before they mean anything.

## RAG against a tool-using agent

Before Agentic mode existed, the same local model (qwen2.5-coder:7b) answered 18 gitsummarize questions two ways: retrieval-augmented, and as an agent with `list_files` / `grep` / `read_file` tools and up to 10 turns. The answers were graded against the code by Claude Code, not by a person, so treat these as indicative:

| approach | correct | partly | wrong | median time |
|---|---|---|---|---|
| Ask (hybrid search, 8 sources in the prompt) | 16 | 1 | 1 | 35 s |
| tool-using agent | 3 | 4 | 11 | 45 s |

The agent grepped for phrases instead of names, gave up twice at the turn limit, made up code or files four times, and used up to 44K input tokens on one question. A 7B model isn't good at choosing what to read next; fixed retrieval chooses for it. That's why docgen's Agentic mode keeps its decisions in code (a reranker score, a citation count) and uses the model only to write. With a strong model, the result could well flip, and that comparison is the most interesting one still to run.

On a repo small enough to fit (`example_multilang`, about 1.3K tokens), putting the whole repo in the prompt was best: 4 of 4 correct, with exact citations.

## Evaluating another repo

Copy an eval file, point `"repo"` at the repo (a path or GitHub URL), and write questions whose answers you know, with the files (and ideally a piece of the chunk) that answer each. Then change one thing at a time, such as chunk size or the embedding model, and compare.
