"""Model providers behind one `complete(system, user) -> str` callable:
ChatAnthropic or ChatOllama. The errors say what to do: ProviderSetupError
means stop (bad key, unknown model, Ollama not running), ProviderError means
this call failed, and GenerationError means the reply came back unusable.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

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
    from langchain_anthropic import ChatAnthropic

    options = {"api_key": api_key} if api_key else {}
    if effort:
        options["output_config"] = {"effort": effort}
    chat = ChatAnthropic(model=model, max_tokens=max_tokens, max_retries=2, **options)

    def complete(system: str, user: str) -> str:
        try:
            reply = chat.invoke([SystemMessage(system), HumanMessage(user)])
        # LangChain re-raises the SDK's errors as subclasses of them.
        except anthropic.AuthenticationError as e:
            raise ProviderSetupError("Authentication failed. Check the API key.") from e
        except anthropic.NotFoundError as e:
            raise ProviderSetupError(f"Unknown Claude model {model!r}.") from e
        except anthropic.APIError as e:
            raise ProviderError(str(e)) from e
        except TypeError as e:  # no API key anywhere
            raise ProviderSetupError(str(e)) from e
        if meter and reply.usage_metadata:
            meter.add(reply.usage_metadata["input_tokens"], reply.usage_metadata["output_tokens"])
        stop = reply.response_metadata.get("stop_reason")
        if stop == "refusal":
            raise ProviderError("The model declined to answer this request.")
        if stop == "max_tokens":
            raise ProviderError(f"The reply hit the {max_tokens}-token limit and was cut off.")
        return reply.text

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
    import httpx
    import ollama
    from langchain_ollama import ChatOllama

    base = _normalize_ollama_host(host)
    chat = ChatOllama(
        model=model, base_url=base, num_ctx=OLLAMA_NUM_CTX, num_predict=max_tokens, temperature=0.2,
        # trust_env=False sends requests straight to the server instead of through
        # any system proxy, which would break localhost on managed machines.
        sync_client_kwargs={"timeout": timeout, "trust_env": False},
    )

    def complete(system: str, user: str) -> str:
        try:
            reply = chat.invoke([SystemMessage(system), HumanMessage(user)])
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ProviderSetupError(
                    f"Ollama has no model named {model!r}. Download it with: ollama pull {model}"
                ) from e
            raise ProviderError(f"Ollama returned HTTP {e.status_code}: {e.error}") from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"Ollama did not answer within {timeout}s") from e
        except (ConnectionError, httpx.ConnectError) as e:
            raise ProviderSetupError(f"Could not reach Ollama at {base}. Is it running? ({e})") from e
        except ValueError as e:  # a reply that isn't valid JSON or lacks the message
            raise ProviderError("Ollama sent a reply that could not be read") from e
        return reply.text

    return complete
