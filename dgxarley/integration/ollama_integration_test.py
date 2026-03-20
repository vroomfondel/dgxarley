"""Integration tests for the standalone Ollama instance.

Tests the Ollama API directly or via LiteLLM (OpenAI-compatible proxy).
Covers health, model availability, chat completions (streaming +
non-streaming), and embeddings.

Usage::

    # Direct Ollama API (default):
    python ollama_integration_test.py

    # Via LiteLLM (OpenAI-compatible):
    python ollama_integration_test.py --via-litellm
    LITELLM_URL=https://litellm.example.com python ollama_integration_test.py --via-litellm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# Repo root is 2 levels above __file__ (integration/ -> dgxarley/ -> repo-root)
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env from repo root (does not override existing env vars)
_env_files: list[Path] = [_REPO_ROOT / ".env", _REPO_ROOT / ".env.local"]
for _env_file in _env_files:
    if _env_file.is_file():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _value = _line.partition("=")
            _key = _key.strip()
            _value = _value.strip().strip("\"'")
            os.environ.setdefault(_key, _value)

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "https://ollama.example.com")
LITELLM_URL: str = os.environ.get("LITELLM_URL", "https://litellm.example.com")
TIMEOUT: tuple[int, int] = (10, 120)

# Models expected to be available.
# Ollama native: from ollama_preload_models in defaults
# LiteLLM: from litellm_model_list in defaults (only Ollama-backed models)
EXPECTED_MODELS_OLLAMA: list[str] = ["bge-m3", "qwen2.5-coder:latest"]
EXPECTED_MODELS_LITELLM: list[str] = ["bge-m3", "qwen2.5-coder:latest"]
EMBEDDING_MODEL_OLLAMA: str = "bge-m3"
EMBEDDING_MODEL_LITELLM: str = "bge-m3"
CHAT_MODEL_OLLAMA: str = "qwen2.5-coder:latest"
CHAT_MODEL_LITELLM: str = "qwen2.5-coder:latest"

# Runtime mode — set by CLI --via-litellm or env var USE_LITELLM=1
USE_LITELLM: bool = os.environ.get("USE_LITELLM", "").lower() in ("1", "true", "yes")



def _base_url() -> str:
    """Return the active base URL depending on mode."""
    return LITELLM_URL if USE_LITELLM else OLLAMA_URL


def _mode_label() -> str:
    return "LiteLLM (OpenAI)" if USE_LITELLM else "Ollama (native)"


def _expected_models() -> list[str]:
    return EXPECTED_MODELS_LITELLM if USE_LITELLM else EXPECTED_MODELS_OLLAMA


def _embedding_model() -> str:
    return EMBEDDING_MODEL_LITELLM if USE_LITELLM else EMBEDDING_MODEL_OLLAMA


def _chat_model() -> str:
    return CHAT_MODEL_LITELLM if USE_LITELLM else CHAT_MODEL_OLLAMA


class TestResult:
    """Result of a single integration test."""

    def __init__(self, name: str, passed: bool, duration: float, detail: str = "") -> None:
        self.name: str = name
        self.passed: bool = passed
        self.duration: float = duration
        self.detail: str = detail

    def __str__(self) -> str:
        status = "\033[32mPASS\033[0m" if self.passed else "\033[31mFAIL\033[0m"
        result = f"  [{status}] {self.name} ({self.duration:.2f}s)"
        if self.detail:
            result += f" — {self.detail}"
        return result


def test_health() -> TestResult:
    """Verify that the backend is reachable."""
    t0 = time.monotonic()
    try:
        url = _base_url()
        if USE_LITELLM:
            resp = requests.get(f"{url}/health", timeout=TIMEOUT)
            ok = resp.status_code == 200
        else:
            resp = requests.get(url, timeout=TIMEOUT)
            ok = resp.status_code == 200 and "Ollama is running" in resp.text
        return TestResult("health", ok, time.monotonic() - t0)
    except Exception as e:
        return TestResult("health", False, time.monotonic() - t0, str(e))


def test_list_models() -> TestResult:
    """Verify that all expected models appear in the model list."""
    t0 = time.monotonic()
    try:
        url = _base_url()
        if USE_LITELLM:
            resp = requests.get(f"{url}/v1/models", timeout=TIMEOUT)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
        else:
            resp = requests.get(f"{url}/api/tags", timeout=TIMEOUT)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
        missing = [e for e in _expected_models() if not any(e in m for m in models)]
        ok = len(missing) == 0
        detail = f"found: {models}" if ok else f"missing: {missing}, found: {models}"
        return TestResult("list_models", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("list_models", False, time.monotonic() - t0, str(e))


def test_model_info() -> TestResult:
    """Verify that model metadata can be retrieved for the chat model."""
    t0 = time.monotonic()
    if USE_LITELLM:
        # OpenAI API has no model info endpoint beyond /v1/models — skip
        return TestResult("model_info", True, time.monotonic() - t0, "skipped (not available via OpenAI API)")
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": _chat_model()},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        ok = "modelfile" in data or "parameters" in data or "template" in data
        details = data.get("details", {})
        detail = f"family={details.get('family', '?')}, params={details.get('parameter_size', '?')}"
        return TestResult("model_info", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("model_info", False, time.monotonic() - t0, str(e))


def test_embeddings() -> TestResult:
    """Verify that the embedding endpoint returns well-formed vectors."""
    t0 = time.monotonic()
    try:
        url = _base_url()
        if USE_LITELLM:
            resp = requests.post(
                f"{url}/v1/embeddings",
                json={
                    "model": _embedding_model(),
                    "input": ["Hello world", "Integration test embedding"],
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [e["embedding"] for e in data.get("data", [])]
        else:
            resp = requests.post(
                f"{url}/api/embed",
                json={
                    "model": _embedding_model(),
                    "input": ["Hello world", "Integration test embedding"],
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
        ok = (
            len(embeddings) == 2
            and all(isinstance(e, list) and len(e) > 0 for e in embeddings)
            and all(isinstance(v, float) for v in embeddings[0])
        )
        dims = [len(e) for e in embeddings] if embeddings else []
        detail = f"{len(embeddings)} embeddings, dims={dims}"
        return TestResult("embeddings", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("embeddings", False, time.monotonic() - t0, str(e))


def test_chat_non_streaming() -> TestResult:
    """Verify that a non-streaming chat completion returns a complete response."""
    t0 = time.monotonic()
    try:
        url = _base_url()
        if USE_LITELLM:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": _chat_model(),
                    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                    "stream": False,
                    "max_tokens": 16,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            ok = len(content) > 0
        else:
            resp = requests.post(
                f"{url}/api/chat",
                json={
                    "model": _chat_model(),
                    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                    "stream": False,
                    "options": {"num_predict": 16},
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            ok = len(content) > 0 and data.get("done", False)
        detail = f"response: {content!r}"
        return TestResult("chat_non_streaming", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("chat_non_streaming", False, time.monotonic() - t0, str(e))


def test_chat_streaming() -> TestResult:
    """Verify that a streaming chat completion delivers tokens incrementally."""
    t0 = time.monotonic()
    try:
        url = _base_url()
        if USE_LITELLM:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": _chat_model(),
                    "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
                    "stream": True,
                    "max_tokens": 32,
                },
                stream=True,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            content = ""
            chunk_count = 0
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                decoded = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not decoded.startswith("data: "):
                    continue
                payload = decoded[6:]
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                content += delta
                if delta:
                    chunk_count += 1
        else:
            resp = requests.post(
                f"{url}/api/chat",
                json={
                    "model": _chat_model(),
                    "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
                    "stream": True,
                    "options": {"num_predict": 32},
                },
                stream=True,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            content = ""
            chunk_count = 0
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                msg = chunk.get("message", {}).get("content", "")
                content += msg
                if msg:
                    chunk_count += 1
                if chunk.get("done"):
                    break
        ok = chunk_count > 0 and len(content.strip()) > 0
        detail = f"{chunk_count} chunks, response: {content.strip()!r}"
        return TestResult("chat_streaming", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("chat_streaming", False, time.monotonic() - t0, str(e))


def main() -> None:
    """Run all Ollama integration tests and exit with an appropriate status code."""
    global USE_LITELLM  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="Ollama integration tests")
    parser.add_argument(
        "--via-litellm", action="store_true",
        help="Test via LiteLLM proxy (OpenAI-compatible API) instead of Ollama native API",
    )
    args = parser.parse_args()
    USE_LITELLM = args.via_litellm or os.environ.get("USE_LITELLM", "").lower() in ("1", "true", "yes")

    print(f"Ollama integration tests — {_base_url()} ({_mode_label()})\n")

    tests = [
        test_health,
        test_list_models,
        test_model_info,
        test_embeddings,
        test_chat_non_streaming,
        test_chat_streaming,
    ]

    results: list[TestResult] = []
    for test_fn in tests:
        result = test_fn()
        print(result)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)
    print(f"\n{passed}/{total} passed in {total_time:.1f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
