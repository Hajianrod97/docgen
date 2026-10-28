"""Model providers: Claude (Anthropic API) and a local Ollama server.

Every model call in docgen goes through a plain `complete(system, user) -> str`
callable, so the hub and Ask can be tested with fakes and don't care which
provider is behind it. The errors are split by what the caller should do:
ProviderSetupError (bad key, unknown model, Ollama not running) means stop,
since every later call would fail the same way; ProviderError means this one
call failed; GenerationError means the reply came back but wasn't usable.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Callable

Completer = Callable[[str, str], str]

# Current Claude models think before answering, and that thinking counts
# toward max_tokens, so leave plenty of room or replies get cut off.
ANTHROPIC_MAX_TOKENS = 16000
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"
# Ollama's default context window is small and it truncates silently, which
# would cut off our instructions, so ask for room explicitly.
OLLAMA_NUM_CTX = 8192
OLLAMA_TIMEOUT = 600  # seconds; a local model's first reply includes load time


class GenerationError(Exception):
    """The model reply could not be used (missing tags, empty output)."""


class ProviderError(Exception):
    """One request to a model provider failed; other chunks may still work."""


class ProviderSetupError(ProviderError):
    """The provider is unreachable or misconfigured; retrying other chunks won't help."""


class UsageMeter:
    """Thread-safe running total of tokens used, as reported by the API."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens


def make_anthropic_completer(
    model: str,
    api_key: str | None = None,
    max_tokens: int = ANTHROPIC_MAX_TOKENS,
    effort: str | None = None,
    meter: UsageMeter | None = None,
) -> Completer:
    """Build a completer backed by the Anthropic API.

    Uses `api_key` if given, else the SDK's usual credential lookup
    (ANTHROPIC_API_KEY and friends). `effort` ("low", "medium", "high", ...)
    trades thoroughness for cost; None keeps the model's default. Token usage
    is added to `meter` when one is given.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    extra = {"output_config": {"effort": effort}} if effort else {}

    def complete(system: str, user: str) -> str:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **extra,
            )
        except anthropic.AuthenticationError as e:
            raise ProviderSetupError("Authentication failed. Check the API key.") from e
        except anthropic.NotFoundError as e:
            raise ProviderSetupError(f"Unknown Claude model {model!r}.") from e
        except anthropic.APIError as e:
            raise ProviderError(str(e)) from e
        if meter:
            meter.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            raise ProviderError("The model declined to answer this request.")
        if response.stop_reason == "max_tokens":
            raise ProviderError(f"The reply hit the {max_tokens}-token limit and was cut off.")
        return "".join(b.text for b in response.content if b.type == "text")

    return complete


def _normalize_ollama_host(host: str | None) -> str:
    host = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_URL).strip()
    if "://" not in host:
        host = "http://" + host
    # 0.0.0.0 is a bind address (what a server listens on), not somewhere to connect.
    host = host.replace("//0.0.0.0", "//127.0.0.1")
    return host.rstrip("/")


def make_ollama_completer(
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str | None = None,
    max_tokens: int = 1024,
    timeout: int = OLLAMA_TIMEOUT,
) -> Completer:
    """Build a completer backed by a local Ollama server (free, no API key).

    The host comes from `host`, else the OLLAMA_HOST environment variable
    (Ollama's own convention), else http://localhost:11434.
    """
    base = _normalize_ollama_host(host)
    # An empty ProxyHandler makes requests go straight to the server instead of
    # through any system proxy, which would break localhost on managed machines.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def complete(system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                pass
            if e.code == 404:
                raise ProviderSetupError(
                    f"Ollama has no model named {model!r}. Download it with: ollama pull {model}"
                ) from e
            raise ProviderError(f"Ollama returned HTTP {e.code}: {detail or e.reason}") from e
        except TimeoutError as e:
            raise ProviderError(f"Ollama did not answer within {timeout}s") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ProviderError(f"Ollama did not answer within {timeout}s") from e
            raise ProviderSetupError(
                f"Could not reach Ollama at {base}. Is it running? ({e.reason})"
            ) from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ProviderError("Ollama sent a reply that was not valid JSON") from e

        try:
            return payload["message"]["content"]
        except (KeyError, TypeError) as e:
            raise ProviderError("Ollama's reply did not have the expected shape") from e

    return complete
