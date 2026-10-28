"""Tests for the Ollama provider.

A tiny fake server speaks Ollama's documented /api/chat protocol, so these
verify our client's behavior (request shape, error handling) offline. They do
not prove anything about a real Ollama install or a real model's output.
"""

import json
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from docgen import web
from docgen.hub import SYSTEM_PROMPT
from docgen.llm import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    ProviderError,
    ProviderSetupError,
    _normalize_ollama_host,
    make_ollama_completer,
)
from docgen.repo_index import update_index
from test_hub import MULTILANG


class FakeOllama:
    def __init__(self):
        self.requests = []  # (path, parsed json body)
        self.status = 200
        self.raw_body = None  # bytes; overrides the normal reply
        self.delay = 0.0

    def respond(self, body):
        if "input" in body:  # /api/embed: a small vector per input, from its length
            return {"embeddings": [[float(len(t)), 1.0, 0.5] for t in body["input"]]}
        return {"message": {"role": "assistant", "content": "<answer>Locally answered [1].</answer>"}, "done": True}


@pytest.fixture
def ollama():
    fake = FakeOllama()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fake.requests.append((self.path, body))
            if fake.delay:
                time.sleep(fake.delay)
            payload = (
                fake.raw_body
                if fake.raw_body is not None
                else json.dumps(fake.respond(body)).encode()
            )
            self.send_response(fake.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{server.server_port}"
    yield fake
    server.shutdown()
    server.server_close()


def closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"  # nothing is listening any more


# -------------------------------------------------------------- the client


def test_request_matches_ollamas_chat_protocol(ollama):
    complete = make_ollama_completer("mymodel:7b", host=ollama.url)
    reply = complete("SYSTEM TEXT", "USER TEXT")
    assert reply == "<answer>Locally answered [1].</answer>"

    path, body = ollama.requests[0]
    assert path == "/api/chat"
    assert body["model"] == "mymodel:7b"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    # Without this Ollama's small default window would silently cut our prompt.
    assert body["options"]["num_ctx"] == OLLAMA_NUM_CTX >= 8192


def test_missing_model_is_a_setup_error_with_the_fix(ollama):
    ollama.status = 404
    ollama.raw_body = json.dumps({"error": "model 'nomodel' not found"}).encode()
    complete = make_ollama_completer("nomodel", host=ollama.url)
    with pytest.raises(ProviderSetupError, match="ollama pull nomodel"):
        complete("s", "u")


def test_server_error_is_a_per_request_error_not_a_setup_error(ollama):
    ollama.status = 500
    ollama.raw_body = json.dumps({"error": "out of memory"}).encode()
    complete = make_ollama_completer("m", host=ollama.url)
    with pytest.raises(ProviderError, match="out of memory") as exc:
        complete("s", "u")
    assert not isinstance(exc.value, ProviderSetupError)


def test_unreachable_server_is_a_setup_error():
    complete = make_ollama_completer("m", host=closed_port_url())
    with pytest.raises(ProviderSetupError, match="Is it running"):
        complete("s", "u")


@pytest.mark.parametrize("raw", [b"<html>not json</html>", b'{"unexpected": true}', b'{"message": null}'])
def test_malformed_replies_are_per_request_errors(ollama, raw):
    ollama.raw_body = raw
    complete = make_ollama_completer("m", host=ollama.url)
    with pytest.raises(ProviderError) as exc:
        complete("s", "u")
    assert not isinstance(exc.value, ProviderSetupError)


def test_slow_reply_times_out_cleanly(ollama):
    ollama.delay = 1.0
    complete = make_ollama_completer("m", host=ollama.url, timeout=0.3)
    with pytest.raises(ProviderError, match="did not answer"):
        complete("s", "u")


@pytest.mark.parametrize(
    "given, expected",
    [
        (None, "http://localhost:11434"),
        ("localhost:11434", "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434"),
        ("https://gpu-box.local:8443", "https://gpu-box.local:8443"),
    ],
)
def test_host_normalization(monkeypatch, given, expected):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _normalize_ollama_host(given) == expected


def test_ollama_host_env_var_is_honoured(monkeypatch, ollama):
    monkeypatch.setenv("OLLAMA_HOST", ollama.url.replace("http://", ""))  # no scheme, like Ollama's own docs
    assert make_ollama_completer("m")("s", "u").startswith("<answer>")


def test_proxy_settings_do_not_interfere_with_local_calls(monkeypatch, ollama):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # a dead proxy
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    assert make_ollama_completer("m", host=ollama.url)("s", "u").startswith("<answer>")


def test_system_prompt_is_sent_unchanged(ollama):
    make_ollama_completer("m", host=ollama.url)(SYSTEM_PROMPT, "u")
    assert ollama.requests[0][1]["messages"][0]["content"] == SYSTEM_PROMPT


# ------------------------------------------------------- through the web app


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # ollama must not need it
    dest = tmp_path / "repo"
    shutil.copytree(MULTILANG, dest, ignore=shutil.ignore_patterns(".docgen", "docs"))
    update_index(dest)
    return dest


def test_ask_with_ollama_needs_no_api_key(repo, ollama, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", ollama.url)
    r = TestClient(web.app).post("/api/ask", json={
        "repo": str(repo), "question": "Where are tasks stored?", "provider": "ollama",
    })
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "Locally answered [1]." and r.json()["cited"] == [1]
    body = ollama.requests[0][1]
    assert body["model"] == DEFAULT_OLLAMA_MODEL
    assert "<sources>" in body["messages"][1]["content"]


def test_unreachable_ollama_is_a_clear_error(repo, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", closed_port_url())
    r = TestClient(web.app).post("/api/ask", json={
        "repo": str(repo), "question": "Where are tasks stored?", "provider": "ollama",
    })
    assert r.status_code == 502 and "Could not reach Ollama" in r.json()["detail"]


# ------------------------------------------------------------ embeddings


def test_ollama_embeddings_prefix_documents_and_queries_differently(ollama):
    from docgen.embeddings import OllamaEmbedding

    fn = OllamaEmbedding(host=ollama.url)
    # Chroma wraps embedding functions and returns numpy arrays.
    assert [list(map(float, v)) for v in fn(["def a(): pass", "x = 1"])] == [[30.0, 1.0, 0.5], [22.0, 1.0, 0.5]]
    fn.embed_query(["how?"])
    (path, docs), (_, query) = ollama.requests
    assert path == "/api/embed" and docs["model"] == "nomic-embed-text"
    assert docs["input"] == ["search_document: def a(): pass", "search_document: x = 1"]
    assert query["input"] == ["search_query: how?"]
    assert docs["options"]["num_ctx"] == 8192


def test_models_without_known_prefixes_get_none(ollama):
    from docgen.embeddings import OllamaEmbedding

    OllamaEmbedding("mxbai-embed-large", host=ollama.url)(["text"])
    assert ollama.requests[0][1]["input"] == ["text"]


def test_embedding_errors_are_clear(ollama):
    from docgen.embeddings import EmbeddingError, OllamaEmbedding

    ollama.status = 404
    with pytest.raises(EmbeddingError, match="ollama pull nomic-embed-text"):
        OllamaEmbedding(host=ollama.url)(["x"])
    with pytest.raises(EmbeddingError, match="Could not reach Ollama"):
        OllamaEmbedding(host=closed_port_url())(["x"])
    ollama.status = 200
    ollama.raw_body = json.dumps({"embeddings": []}).encode()
    with pytest.raises(EmbeddingError, match="one vector per input"):
        OllamaEmbedding(host=ollama.url)(["x"])


def test_env_var_picks_the_embedding_model(monkeypatch):
    from docgen.embeddings import HashingEmbedding, OllamaEmbedding, get_embedding_function
    from docgen.repo_index import embedding_name

    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "ollama")
    assert isinstance(get_embedding_function(), OllamaEmbedding)
    assert embedding_name() == "ollama:nomic-embed-text"
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "ollama:mxbai-embed-large")
    assert embedding_name() == "ollama:mxbai-embed-large"
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "hash")
    assert isinstance(get_embedding_function(), HashingEmbedding)
    monkeypatch.delenv("DOCGEN_EMBEDDINGS")
    assert embedding_name() == "chroma-default:all-MiniLM-L6-v2"


def test_index_and_search_with_ollama_embeddings(tmp_path, ollama, monkeypatch):
    from docgen.search import search

    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", ollama.url)
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
    stats = update_index(tmp_path)
    assert stats.files_indexed == 1
    assert search(tmp_path, "alpha", k=1, mode="vector")[0].path == "a.py"
    assert any(b["input"][0].startswith("search_query: ") for _, b in ollama.requests)


def test_unreachable_embedding_server_is_a_clear_web_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCGEN_EMBEDDINGS", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", closed_port_url())
    (tmp_path / "a.py").write_text("x = 1\n")
    r = TestClient(web.app).post("/api/search-index", json={"repo": str(tmp_path)})
    assert r.status_code == 502 and "for embeddings" in r.json()["detail"]
