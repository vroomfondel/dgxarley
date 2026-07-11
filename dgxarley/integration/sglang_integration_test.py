"""Direct SGLang integration tests (bypasses OpenWebUI).

Uses the same LLMClient base class and test functions from
openwebui_integration_test.py, but with SGLangClient which:
- Hits the SGLang OpenAI-compatible API directly (/v1/chat/completions)
- No auth header required
- Flattens extra_body into top-level payload (SGLang expects top_k,
  chat_template_kwargs etc. as direct fields, not nested in extra_body)
- No OpenWebUI features (web_search etc.)

Usage::

    # Requires SGLANG_URL env var
    SGLANG_URL=https://sglang.dgx.example.com python sglang_integration_test.py all

    # Specific tests
    python sglang_integration_test.py thinking coding presets

    # Without reasoning (any test)
    python sglang_integration_test.py --no-think thinking
    python sglang_integration_test.py --no-think parallel

    # Parallel load test (4 concurrent requests)
    python sglang_integration_test.py parallel -n 4

    # Parallel with thinking budget cap
    python sglang_integration_test.py -v --thinking-budget 2048 -n 4 parallel

    # Parallel with custom prompt and 8 requests
    python sglang_integration_test.py parallel -n 8 --prompt "Explain quantum computing"
"""

import asyncio
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import requests as httplib

from .streaming_repetition_guard import RepetitionGuard, GuardConfig, FeedResult, StopReason
from .thinking_parser import ThinkingParser

from .openwebui_integration_test import (
    SGLangClient,
    _dgx_defaults,
    get_random_xkcd_image,
    get_random_xkcd_image_url,
    litellm_api_key,
    load_sampling_presets,
    pick_default_preset,
    print_ascii_representation_of_image,
    test_all_presets,
    test_non_thinking_mode,
    test_sampling_params_passthrough,
    test_thinking_coding,
    test_thinking_mode,
)

from dgxarley import configure_logging, glogger, print_banner

os.environ.setdefault("LOGURU_LEVEL", "DEBUG")
configure_logging()
glogger.enable("dgxarley")

from loguru import logger

# Default model from Ansible defaults (the model currently deployed to SGLang)
_CONFIGURED_MODEL: str = _dgx_defaults.get("sglang_model", "")  # type: ignore[assignment]


def _serialize_logit_processor(module: str, cls_name: str) -> str:
    """Build an SGLang custom_logit_processor JSON string from a module path and class name.

    SGLang expects a dill-serialized class reference, but for importable classes
    this is just a standard pickle of the module+qualname (no code, no state).
    The pickle bytes are constructed directly so that neither dill nor sglang
    need to be installed as a dependency.

    Args:
        module: The fully qualified module path containing the processor class.
        cls_name: The class name of the logit processor within the module.

    Returns:
        A JSON string with a single ``"callable"`` key whose value is the
        hex-encoded pickle of the class reference, suitable for passing as
        ``custom_logit_processor`` in an SGLang API request payload.
    """
    import io
    import struct

    buf = io.BytesIO()
    buf.write(b"\x80\x04\x95")  # PROTO 4 + FRAME
    mod, cls = module.encode(), cls_name.encode()
    frame = (
        b"\x8c"
        + bytes([len(mod)])
        + mod  # SHORT_BINUNICODE module
        + b"\x94\x8c"
        + bytes([len(cls)])
        + cls  # MEMOIZE + SHORT_BINUNICODE class
        + b"\x94\x93\x94\x2e"
    )  # MEMOIZE + STACK_GLOBAL + MEMOIZE + STOP
    buf.write(struct.pack("<Q", len(frame)))
    buf.write(frame)
    return json.dumps({"callable": buf.getvalue().hex()})


# Thinking budget logit processors — keyed by model family.
# The processor forces </think> after N thinking tokens by manipulating logits.
_THINKING_BUDGET_PROCESSORS: dict[str, str] = {
    "qwen3": _serialize_logit_processor(
        "sglang.srt.sampling.custom_logit_processor",
        "Qwen3ThinkingBudgetLogitProcessor",
    ),
    "deepseek-r1": _serialize_logit_processor(
        "sglang.srt.sampling.custom_logit_processor",
        "DeepSeekR1ThinkingBudgetLogitProcessor",
    ),
}

# Default prompts for parallel load testing — varied to avoid prefix cache hits.
# Each prompt includes a role definition and a multi-part question to increase
# complexity and input token count.
PARALLEL_PROMPTS: list[str] = [
    (
        "You are a senior network engineer with 15 years of experience designing "
        "enterprise-grade infrastructure. You have deep expertise in protocol design, "
        "packet-level analysis, and performance tuning for high-throughput systems.\n\n"
        "A junior colleague is confused about when to use TCP vs UDP and keeps defaulting "
        "to TCP for everything, including real-time game state updates. Explain the main "
        "differences between TCP and UDP, covering reliability guarantees, ordering, "
        "congestion control, and overhead. Provide concrete examples of when each protocol "
        "is the right choice and explain the trade-offs involved. Include a brief discussion "
        "of QUIC and how it blurs the line between the two."
    ),
    (
        "You are a computer science professor who specializes in algorithms and "
        "computational complexity. You are known for making dry topics engaging by "
        "connecting them to real-world applications and historical anecdotes.\n\n"
        "A student has asked you to explain the Sieve of Eratosthenes. Write a Python "
        "function that finds all prime numbers up to N using this algorithm, then walk "
        "through the code step by step. Discuss the time and space complexity, compare it "
        "to trial division, and mention the Sieve of Atkin as an alternative. Bonus: show "
        "how to adapt the sieve for segmented operation when N is very large."
    ),
    (
        "You are a science communicator who has spent a decade writing for popular science "
        "magazines. You have a gift for using vivid analogies and everyday objects to make "
        "abstract physics concepts accessible to young readers.\n\n"
        "A curious 12-year-old has asked you: 'What is quantum entanglement and why does "
        "Einstein call it spooky?' Explain the concept in a way that is accurate but uses "
        "analogies they can relate to. Cover what entanglement is, how scientists create "
        "entangled particles, what Bell's theorem tells us, and why this matters for "
        "technologies like quantum computing and quantum cryptography. Avoid oversimplifying "
        "to the point of being misleading."
    ),
    (
        "You are a historian specializing in 18th-century European political revolutions. "
        "You have published extensively on the social and economic underpinnings of "
        "revolutionary movements and you are passionate about drawing parallels to "
        "contemporary political dynamics.\n\n"
        "A graduate student is preparing for their comprehensive exams and needs a thorough "
        "overview of the French Revolution. Cover the key causes — economic crisis, social "
        "inequality, Enlightenment philosophy, and the fiscal mismanagement of the Ancien "
        "Régime. Trace the major phases from the Estates-General through the Terror to "
        "Napoleon's rise. Discuss the lasting consequences for European politics, the concept "
        "of human rights, and the ripple effects on subsequent revolutions worldwide."
    ),
    (
        "You are a seasoned backend architect who has designed APIs for products serving "
        "millions of users. You are opinionated about REST best practices, consistent error "
        "handling, and pragmatic versioning strategies.\n\n"
        "A startup team is building their first production API for a collaborative todo-list "
        "application. Design a REST API that covers CRUD operations for users, lists, and "
        "items, plus sharing and collaboration features. Include endpoint paths, HTTP methods, "
        "request/response payloads with example JSON, pagination strategy, authentication "
        "approach, and error response format. Discuss trade-offs you considered, such as "
        "nested vs flat resource URLs and optimistic vs pessimistic concurrency control."
    ),
    (
        "You are a polyglot software engineer who has shipped production code in Haskell, "
        "Scala, Java, Python, and TypeScript. You are frequently invited to speak at "
        "conferences about paradigm trade-offs and you maintain a popular blog on the topic.\n\n"
        "A bootcamp graduate who learned JavaScript is trying to understand the difference "
        "between functional programming and object-oriented programming. Compare and contrast "
        "the two paradigms, covering core principles (immutability, first-class functions, "
        "encapsulation, inheritance), state management, testability, and real-world suitability. "
        "Use concrete code examples in Python or JavaScript to illustrate key differences. "
        "Discuss hybrid approaches and when blending paradigms makes practical sense."
    ),
    (
        "You are a deep learning researcher at a top AI lab. You contributed to the original "
        "implementation of several attention variants and have taught a graduate seminar on "
        "sequence modeling for three consecutive years.\n\n"
        "A machine learning engineer who is comfortable with CNNs and RNNs but new to "
        "transformers has asked you to explain how a transformer neural network works. Walk "
        "through the architecture step by step: input embeddings, positional encoding, "
        "multi-head self-attention (including the Q/K/V projections and scaled dot-product "
        "attention), feed-forward layers, layer normalization, and the encoder-decoder "
        "structure. Explain why attention replaced recurrence and discuss recent developments "
        "like rotary positional embeddings, flash attention, and mixture-of-experts layers."
    ),
    (
        "You are a mathematical logician and philosopher of mathematics. You hold a chair "
        "at a research university and have spent your career studying the foundations of "
        "mathematics, formal systems, and the limits of provability.\n\n"
        "A philosophy student with solid but not expert-level math background has asked you "
        "about Gödel's incompleteness theorems. Explain the first and second theorems, their "
        "proofs at a high level (Gödel numbering, the diagonal lemma, self-reference), and "
        "their philosophical significance. Discuss implications for Hilbert's program, the "
        "relationship to Turing's halting problem, and common misconceptions — for example, "
        "why the theorems do not mean 'math is broken' or 'there are truths we can never know.'"
    ),
    (
        "You are a senior DevOps engineer and SRE who manages infrastructure for a "
        "large-scale SaaS platform. You are pragmatic, favor simple and composable shell "
        "scripts over complex tooling, and always think about failure modes and alerting.\n\n"
        "Write a bash script that monitors disk usage across all mounted partitions and sends "
        "an alert when any partition exceeds 90%% utilization. The script should be idempotent, "
        "log its findings with timestamps, support both email and webhook-based alerting, and "
        "handle edge cases like read-only filesystems and tmpfs mounts. Include inline comments "
        "explaining your design choices and discuss how you would schedule and test this script "
        "in a production environment."
    ),
    (
        "You are a cybersecurity consultant who advises Fortune 500 companies on their "
        "encryption strategies. You hold CISSP and OSCP certifications and have a talent for "
        "explaining complex cryptographic concepts to non-technical executives.\n\n"
        "A CTO without a security background needs to understand the difference between "
        "symmetric and asymmetric encryption. Explain both approaches, covering key generation, "
        "performance characteristics, and typical use cases. Provide real-world examples: TLS "
        "handshakes, PGP email, disk encryption, and digital signatures. Discuss hybrid "
        "approaches (e.g., envelope encryption), key management challenges, and the looming "
        "impact of quantum computing on current cryptographic standards."
    ),
    (
        "You are a programming language runtime engineer who has contributed to the CPython "
        "interpreter, the HotSpot JVM, and the Rust compiler. You are deeply familiar with "
        "the trade-offs between manual memory management, tracing GC, and ownership systems.\n\n"
        "A systems programmer who is evaluating languages for a new latency-sensitive service "
        "wants to understand how garbage collection works in Java, Python, and Rust. Compare "
        "the approaches: Java's generational GC with G1/ZGC, Python's reference counting plus "
        "cycle detector, and Rust's ownership/borrowing model with no runtime GC. Discuss "
        "pause times, throughput overhead, memory fragmentation, and how each approach affects "
        "application design. Include practical advice on when each model shines and struggles."
    ),
    (
        "You are a distributed systems architect who has built and operated globally "
        "distributed databases. You worked on infrastructure at a FAANG company and have "
        "first-hand experience with the pain of partition events in production.\n\n"
        "A backend engineer designing their first multi-region service has asked about the "
        "CAP theorem. Explain what CAP states, define consistency, availability, and partition "
        "tolerance precisely, and discuss why 'pick two out of three' is an oversimplification. "
        "Cover the PACELC extension, give concrete examples of CP and AP systems (e.g., "
        "ZooKeeper vs Cassandra), and explain how modern systems like CockroachDB and Spanner "
        "navigate these trade-offs. Offer practical guidance for choosing the right consistency "
        "model based on business requirements."
    ),
    (
        "You are a mathematics professor who specializes in probability theory and Bayesian "
        "reasoning. You are known for your engaging lecture style and for using interactive "
        "demonstrations to build intuition about counterintuitive results.\n\n"
        "A student is frustrated because the Monty Hall problem 'doesn't make sense' to them — "
        "they insist it should be 50/50 after a door is opened. Explain the problem clearly, "
        "prove why switching doors gives a 2/3 probability of winning, and address the common "
        "intuitive objections. Use multiple explanations: enumeration of all outcomes, Bayesian "
        "updating, and the generalization to N doors. Discuss why human intuition fails here "
        "and connect this to broader lessons about conditional probability."
    ),
    (
        "You are a principal software architect who has led the migration of a major "
        "e-commerce platform from a monolith to microservices — and later consolidated some "
        "services back. You have strong opinions informed by hard-won experience about when "
        "each approach is appropriate.\n\n"
        "A VP of Engineering at a mid-size startup (50 engineers, Series B) is deciding whether "
        "to decompose their growing monolith into microservices. Present a balanced analysis of "
        "the pros and cons of microservices vs monolithic architecture. Cover deployment "
        "complexity, team autonomy, data consistency, operational overhead, debugging difficulty, "
        "and organizational fit (Conway's Law). Include a decision framework for when to stay "
        "monolithic, when to adopt microservices, and when a modular monolith is the pragmatic "
        "middle ground."
    ),
    (
        "You are a staff infrastructure engineer who has operated authoritative and recursive "
        "DNS infrastructure at scale. You have debugged countless 'DNS is always the problem' "
        "incidents and you understand every layer from stub resolvers to root servers.\n\n"
        "A frontend developer who only knows that 'DNS turns names into IPs' wants a deeper "
        "understanding. Explain the full DNS resolution process from the moment a user types a "
        "URL in their browser to when the page starts loading. Cover the browser cache, OS stub "
        "resolver, recursive resolver, root/TLD/authoritative queries, DNSSEC validation, "
        "caching and TTLs, and the role of anycast. Discuss common failure modes, how CDNs use "
        "DNS for load balancing, and emerging standards like DNS-over-HTTPS and DNS-over-QUIC."
    ),
]


def validate_model(sglang_url: str, model_id: str) -> None:
    """Query /v1/models and warn if the running model does not match model_id.

    Exits the process with code 1 if the server is reachable but serves a
    different model than requested. Prints a warning and returns normally if
    the endpoint cannot be reached.

    Args:
        sglang_url: Base URL of the SGLang server (e.g. ``https://sglang.example.com``).
        model_id: The model identifier that is expected to be served.

    Raises:
        SystemExit: If the server responds but does not serve ``model_id``.
    """
    try:
        _key = litellm_api_key()
        _hdrs = {"Authorization": f"Bearer {_key}"} if _key else None
        resp = httplib.get(f"{sglang_url.rstrip('/')}/v1/models", timeout=5, headers=_hdrs)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        served_ids = [m.get("id", "") for m in data]
    except Exception as e:
        print(f"\033[33m⚠ Could not query /v1/models: {e}\033[0m")
        return

    if not served_ids:
        print("\033[33m⚠ /v1/models returned no models\033[0m")
        return

    if model_id in served_ids:
        print(f"✓ Model confirmed: {model_id}")
    else:
        print(f"\033[31m✗ MODEL MISMATCH — requested '{model_id}' but server serves: {served_ids}\033[0m")
        print(f"\033[31m  Hint: set MODEL_ID={served_ids[0]} or omit to use Ansible default\033[0m")
        raise SystemExit(1)


def resolve_model_id() -> str:
    """Resolve the model ID from environment or Ansible defaults.

    Resolution order: ``MODEL_ID`` environment variable, then the
    ``sglang_model`` key from the Ansible defaults loaded at import time.

    Returns:
        The resolved model identifier string.

    Raises:
        ValueError: If neither the environment variable nor the Ansible
            defaults contain a model ID.
    """
    model_id = os.environ.get("MODEL_ID", "")
    if model_id:
        return model_id
    if _CONFIGURED_MODEL:
        return _CONFIGURED_MODEL
    raise ValueError("No MODEL_ID env var and no sglang_model in Ansible defaults")


def create_sglang_client(verbose: bool = False, reasoning_effort: str | None = None) -> SGLangClient:
    """Create an SGLangClient from environment variables.

    Reads ``SGLANG_URL`` and ``MODEL_ID`` (or falls back to the Ansible
    default), validates the model against the live server, and returns a
    configured client.

    Args:
        verbose: If ``True``, the client will print request payloads and
            reasoning tokens during test runs.
        reasoning_effort: If set, every request the client sends carries a
            top-level ``reasoning_effort`` field. Use ``"high"`` to turn ON
            reasoning for Mistral Large-3 / Medium-3.5 in the sequential tests
            (they default to none). ``None`` leaves the model default.

    Returns:
        A configured :class:`SGLangClient` ready for use.

    Raises:
        ValueError: If ``SGLANG_URL`` is not set or the model ID cannot be
            resolved.
        SystemExit: If the server serves a different model than requested.
    """
    model_id = resolve_model_id()
    use_litellm = os.environ.get("USE_LITELLM", "").lower() in ("1", "true", "yes")
    if use_litellm:
        sglang_url = os.environ.get("LITELLM_URL", "")
        if not sglang_url:
            raise ValueError("USE_LITELLM is set but LITELLM_URL is missing")
        label = "LiteLLM proxy"
    else:
        sglang_url = os.environ.get("SGLANG_URL", "")
        if not sglang_url:
            raise ValueError("Set SGLANG_URL environment variable (e.g. https://sglang.dgx.example.com)")
        label = "SGLang direct"
    validate_model(sglang_url, model_id)
    print(f"[{label}] {sglang_url} model={model_id}")
    return SGLangClient(sglang_url, model_id, verbose=verbose, reasoning_effort=reasoning_effort)


# ---------------------------------------------------------------------------
# Parallel load test
# ---------------------------------------------------------------------------


@dataclass
class RequestStats:
    """Tracks timing and output statistics for a single parallel streaming request.

    Attributes:
        request_id: 1-based index identifying this request within a parallel batch.
        prompt: The user prompt sent to the model.
        status: Current lifecycle state: ``"pending"``, ``"streaming"``,
            ``"done"``, or ``"error"``.
        output: Accumulated assistant content tokens received so far.
        thinking: Accumulated reasoning/thinking tokens received so far.
        ttft: Time-to-first-token in seconds (0.0 until the first token arrives).
        total_time: Total wall-clock duration in seconds from request start to
            completion or error.
        output_tokens: Number of output tokens as reported by the usage field;
            estimated from character count if the server omits usage.
        prompt_tokens: Number of prompt tokens as reported by the usage field.
        finish_reason: The ``finish_reason`` value from the final SSE chunk
            (e.g. ``"stop"``, ``"length"``).
        error: Human-readable error description if ``status == "error"``.
        _start: Monotonic timestamp recorded when streaming begins (internal).
        _first_token: Flag indicating whether the first token has been received (internal).
    """

    request_id: int
    prompt: str
    status: str = "pending"  # pending, streaming, done, error
    output: str = ""
    thinking: str = ""
    ttft: float = 0.0  # time to first token
    total_time: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: str = ""
    error: str = ""
    repetition_stopped: bool = False
    repetition_reason: str = ""
    repetition_diagnostics: dict[str, object] = field(default_factory=dict)
    clean_output: str = ""
    _start: float = field(default=0.0, repr=False)
    _first_token: bool = field(default=False, repr=False)
    _last_token_time: float = field(default=0.0, repr=False)

    @property
    def tokens_per_sec(self) -> float:
        """Compute the current token throughput in tokens per second.

        Uses ``output_tokens`` if available, otherwise estimates from the
        accumulated character lengths divided by 4.

        Returns:
            Tokens per second, or ``0.0`` if elapsed time or token count
            is zero.
        """
        if self.status == "done" or self.status == "error":
            t = self.total_time
        else:
            t = (time.monotonic() - self._start) if self._start else 0
        tokens = self.output_tokens if self.output_tokens > 0 else (len(self.output) + len(self.thinking)) // 4
        if t > 0 and tokens > 0:
            return tokens / t
        return 0.0

    def output_tail(self, max_chars: int = 1200) -> str:
        """Return the last ``max_chars`` characters of the accumulated output.

        Args:
            max_chars: Maximum number of characters to return from the end
                of ``self.output``.

        Returns:
            The tail of the output string, prefixed with ``"..."`` if truncated.
        """
        if len(self.output) <= max_chars:
            return self.output
        return "..." + self.output[-max_chars:]


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
    stats: RequestStats,
    no_guard: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Stream a single chat completion and update a RequestStats object in-place.

    Opens a POST request to ``url`` with ``payload`` as JSON, reads the
    server-sent event stream, and accumulates content/reasoning tokens into
    ``stats``. Token counts, timing fields, and the finish reason are updated
    as chunks arrive.

    A :class:`RepetitionGuard` monitors both content and reasoning streams;
    if repetition is detected the stream is aborted and ``stats`` is marked
    with ``repetition_stopped=True``. Pass ``no_guard`` to disable the guard
    for ``"content"``, ``"reasoning"``, or ``"both"`` streams.

    Args:
        session: An active :class:`aiohttp.ClientSession` to use for the request.
        url: The fully qualified endpoint URL (e.g.
            ``https://sglang.example.com/v1/chat/completions``).
        payload: The JSON-serialisable request body. Keys and value types
            vary by request; heterogeneous values are typed as ``object``.
        stats: The :class:`RequestStats` instance to update throughout
            streaming. Modified in place.
        headers: Request headers to send. When ``None``, only
            ``Content-Type: application/json`` is sent (SGLang direct, no auth).
            Pass a dict carrying ``Authorization: Bearer …`` to reach the
            auth-gated LiteLLM proxy.
    """
    _skip_content = no_guard in ("content", "both")
    _skip_reasoning = no_guard in ("reasoning", "both")
    content_guard = None if _skip_content else RepetitionGuard()
    reasoning_guard = None if _skip_reasoning else RepetitionGuard()
    tp = ThinkingParser()

    stats.status = "streaming"
    stats._start = time.monotonic()
    stats._last_token_time = stats._start
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers or {"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=1800, connect=10),
        ) as resp:
            if resp.status != 200:
                stats.status = "error"
                stats.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                stats.total_time = time.monotonic() - stats._start
                return

            async for raw_line in resp.content:
                decoded = raw_line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                u = chunk.get("usage")
                if u:
                    stats.output_tokens = u.get("completion_tokens", 0)
                    stats.prompt_tokens = u.get("prompt_tokens", 0)
                choice = (chunk.get("choices") or [None])[0]
                if choice and choice.get("finish_reason"):
                    stats.finish_reason = choice["finish_reason"]
                delta = (choice or {}).get("delta", {})
                parsed = tp.feed(
                    content=delta.get("content", ""),
                    reasoning_content=delta.get("reasoning_content", ""),
                )
                if parsed.thinking or parsed.content:
                    stats._last_token_time = time.monotonic()
                    if not stats._first_token:
                        stats._first_token = True
                        stats.ttft = time.monotonic() - stats._start
                if parsed.thinking:
                    stats.thinking += parsed.thinking
                    if reasoning_guard:
                        result = reasoning_guard.feed(parsed.thinking)
                        if result.should_stop:
                            stats.status = "error"
                            stats.repetition_stopped = True
                            stats.repetition_reason = result.detail
                            stats.repetition_diagnostics = {
                                "source": "reasoning",
                                **result.diagnostics,
                                "guard_stats": reasoning_guard.get_stats(),
                            }
                            assert result.reason is not None
                            stats.error = f"repetition: {result.reason.name} — {result.detail}"
                            stats.clean_output = reasoning_guard.get_clean_text()
                            return
                if parsed.content:
                    stats.output += parsed.content
                    if content_guard:
                        result = content_guard.feed(parsed.content)
                        if result.should_stop:
                            stats.status = "error"
                            stats.repetition_stopped = True
                            stats.repetition_reason = result.detail
                            stats.repetition_diagnostics = {
                                "source": "content",
                                **result.diagnostics,
                                "guard_stats": content_guard.get_stats(),
                            }
                            assert result.reason is not None
                            stats.error = f"repetition: {result.reason.name} — {result.detail}"
                            stats.clean_output = content_guard.get_clean_text()
                            return

    except Exception as e:
        import traceback

        stats.status = "error"
        stats.error = f"{e}\n{traceback.format_exc()}"

    stats.total_time = time.monotonic() - stats._start
    if stats.status != "error":
        stats.status = "done"
    # Estimate tokens from output length if usage wasn't reported
    if stats.output_tokens == 0 and (stats.output or stats.thinking):
        stats.output_tokens = (len(stats.output) + len(stats.thinking)) // 4  # rough estimate


def _tok_detail(s: RequestStats) -> str:
    """Format a thinking/content token breakdown string for panel headers.

    Estimates thinking token count (``T~N``) and content token count
    (``C~N``) from character lengths divided by 4, and returns a parenthesised
    slash-separated string. Returns an empty string if both counts are zero.

    Args:
        s: The :class:`RequestStats` instance whose token counts to format.

    Returns:
        A string of the form ``" (T~N/C~M)"``, ``" (T~N)"``, ``" (C~M)"``,
        or ``""`` if there are no tokens to report.
    """
    t_est = len(s.thinking) // 4 if s.thinking else 0
    c_est = len(s.output) // 4 if s.output else 0
    if not t_est and not c_est:
        return ""
    parts = []
    if t_est:
        parts.append(f"T~{t_est}")
    if c_est:
        parts.append(f"C~{c_est}")
    return f" ({'/'.join(parts)})"


def _tail_lines(text: str, max_lines: int, wrap_width: int) -> tuple[str, bool]:
    """Return the tail of *text* that fits within *max_lines* rendered lines.

    Accounts for terminal line-wrapping at *wrap_width*.  Returns
    ``(truncated_text, was_truncated)`` so callers can prepend an ellipsis
    indicator when needed.
    """
    if max_lines <= 0 or not text:
        return "", bool(text)
    source_lines = text.rstrip("\n").split("\n")
    taken: list[str] = []
    used = 0
    for line in reversed(source_lines):
        wrapped = max(1, -(-len(line) // wrap_width)) if wrap_width > 0 else 1
        if used + wrapped > max_lines:
            break
        taken.append(line)
        used += wrapped
    taken.reverse()
    return "\n".join(taken), len(taken) < len(source_lines)


def _wrapped_line_count(text: str, wrap_width: int) -> int:
    """Estimate rendered line count after terminal word-wrap."""
    if not text:
        return 0
    return sum(max(1, -(-len(l) // wrap_width)) for l in text.split("\n"))


def build_live_display(
    all_stats: list[RequestStats],
    verbose: bool = False,
    header: str = "",
) -> Table:
    """Build a Rich Table showing the live state of all parallel requests.

    Renders a two-column grid of :class:`rich.panel.Panel` objects — one per
    request — plus a summary row at the top. Panel height is computed from
    the terminal height so that all rows fit on screen simultaneously.

    Args:
        all_stats: List of :class:`RequestStats` objects, one per in-flight
            or completed request.
        verbose: If ``True``, the thinking/reasoning tokens are included in
            the panel body alongside content tokens.
        header: Optional multi-line header text displayed above the summary
            row (e.g. URL, model, params).  Included in the height
            calculation so panels don't push it off-screen.

    Returns:
        A :class:`rich.table.Table` suitable for passing to
        :class:`rich.live.Live`.
    """
    # Summary stats at the top
    done = [s for s in all_stats if s.status == "done"]
    streaming = [s for s in all_stats if s.status == "streaming"]
    errors = [s for s in all_stats if s.status == "error"]
    pending = [s for s in all_stats if s.status == "pending"]

    summary = Table.grid(padding=(0, 2))
    summary.add_row(
        f"[bold]Requests:[/] {len(all_stats)}",
        f"[green]Done:[/] {len(done)}",
        f"[yellow]Streaming:[/] {len(streaming)}",
        f"[dim]Pending:[/] {len(pending)}",
        f"[red]Errors:[/] {len(errors)}",
    )
    if done:
        agg_tokens = sum(s.output_tokens for s in done)
        agg_time = max(s.total_time for s in done) if done else 0
        avg_ttft = sum(s.ttft for s in done) / len(done)
        avg_tps = sum(s.tokens_per_sec for s in done) / len(done)
        agg_think = sum(len(s.thinking) // 4 for s in done)
        agg_content = sum(len(s.output) // 4 for s in done)
        summary.add_row(
            f"[bold]Aggregate:[/] {agg_tokens} tok [dim](T~{agg_think}/C~{agg_content})[/]",
            f"[bold]Elapsed:[/] {agg_time:.1f}s",
            f"[bold]Avg TTFT:[/] {avg_ttft:.2f}s",
            f"[bold]Avg tok/s:[/] {avg_tps:.1f}",
            f"[bold]Total tok/s:[/] {agg_tokens / agg_time:.1f}" if agg_time > 0 else "",
        )

    # Per-request panels — size to fill terminal
    console_width = Console().width
    console_height = Console().height
    col_width = (console_width - 1) // 2
    n_rows = (len(all_stats) + 1) // 2
    # Reserve lines for header text + summary row, split remaining height across panel rows
    header_lines = header.count("\n") + 1 if header else 0
    reserved = 4 + header_lines
    panel_height = max(8, (console_height - reserved) // n_rows) if n_rows > 0 else 16
    # Usable lines/width inside panel (subtract borders + padding)
    inner_width = col_width - 4
    inner_lines = panel_height - 2
    # Conservative: assume avg ~45% line fill due to word wrap and short lines
    max_chars = int(inner_width * inner_lines * 0.45)

    panels = []
    for s in all_stats:
        if s.status == "pending":
            style = "dim"
            panel_title = f"[dim]#{s.request_id} pending[/]"
            body = Text(s.prompt[:80] + "...", style="dim")
        elif s.status == "streaming":
            elapsed = time.monotonic() - s._start
            style = "yellow"
            tps = f" {s.tokens_per_sec:.1f} t/s" if s._first_token else ""
            tok_detail = _tok_detail(s)
            panel_title = f"[yellow]#{s.request_id} streaming {elapsed:.1f}s{tps}{tok_detail}[/]"
            if verbose and s.thinking:
                content_text = s.output_tail(max_chars)
                has_content = bool(content_text.strip())
                if has_content:
                    c_budget = min(_wrapped_line_count(content_text, inner_width), inner_lines // 2)
                    content_text, c_trunc = _tail_lines(content_text, c_budget, inner_width)
                    if c_trunc:
                        content_text, _ = _tail_lines(content_text, max(1, c_budget - 1), inner_width)
                        content_text = "...\n" + content_text
                    c_lines = _wrapped_line_count(content_text, inner_width)
                    t_budget = max(1, inner_lines - c_lines - 2)  # -2 for [thinking]/[/thinking] markers
                    t_tail, truncated = _tail_lines(s.thinking, t_budget, inner_width)
                    if truncated:
                        # recompute with -3 to reserve a line for the "..." prefix
                        t_budget = max(1, inner_lines - c_lines - 3)
                        t_tail, truncated = _tail_lines(s.thinking, t_budget, inner_width)
                    prefix = "...\n" if truncated else ""
                    display = f"[thinking]\n{prefix}{t_tail}\n[/thinking]\n{content_text}"
                else:
                    t_tail, truncated = _tail_lines(s.thinking, inner_lines - 1, inner_width)
                    if truncated:
                        t_tail, truncated = _tail_lines(s.thinking, inner_lines - 2, inner_width)
                    prefix = "...\n" if truncated else ""
                    display = f"[thinking]\n{prefix}{t_tail}"
            else:
                display = s.output_tail(max_chars)
            body = Text(display, style="white")
        elif s.status == "done":
            style = "green"
            tok_detail = _tok_detail(s)
            panel_title = (
                f"[green]#{s.request_id} done[/] "
                f"TTFT={s.ttft:.2f}s | {s.total_time:.1f}s | "
                f"{s.output_tokens} tok{tok_detail} | {s.tokens_per_sec:.1f} t/s"
            )
            if verbose and s.thinking:
                content_text = s.output_tail(max_chars)
                c_budget = min(_wrapped_line_count(content_text, inner_width), inner_lines // 2)
                content_text, c_trunc = _tail_lines(content_text, c_budget, inner_width)
                if c_trunc:
                    content_text, _ = _tail_lines(content_text, max(1, c_budget - 1), inner_width)
                    content_text = "...\n" + content_text
                c_lines = _wrapped_line_count(content_text, inner_width)
                t_budget = max(1, inner_lines - c_lines - 2)
                t_tail, truncated = _tail_lines(s.thinking, t_budget, inner_width)
                if truncated:
                    t_budget = max(1, inner_lines - c_lines - 3)
                    t_tail, truncated = _tail_lines(s.thinking, t_budget, inner_width)
                prefix = "...\n" if truncated else ""
                display = f"[thinking]\n{prefix}{t_tail}\n[/thinking]\n{content_text}"
            else:
                display = s.output_tail(max_chars)
            body = Text(display, style="white")
        elif s.repetition_stopped:
            style = "yellow"
            panel_title = f"[yellow]#{s.request_id} REPETITION — {s.repetition_reason[:60]}[/]"
            display = s.clean_output[-max_chars:] if s.clean_output else s.output_tail(max_chars)
            body = Text(display, style="white")
        else:
            style = "red"
            panel_title = f"[red]#{s.request_id} ERROR[/]"
            body = Text(s.error[:max_chars], style="red")

        panels.append(Panel(body, title=panel_title, border_style=style, height=panel_height))
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=col_width)
    grid.add_column(width=col_width)
    for i in range(0, len(panels), 2):
        left = panels[i]
        right = panels[i + 1] if i + 1 < len(panels) else ""
        grid.add_row(left, right)

    outer = Table.grid()
    if header:
        outer.add_row(Text.from_markup(header))
    outer.add_row(summary)
    outer.add_row(grid)
    return outer


def print_final_summary(all_stats: list[RequestStats], wall_time: float, verbose: bool = False) -> None:
    """Print a final results table to the console after all requests complete.

    Renders a per-request table with timing, token, and finish-reason
    columns, followed by an aggregate statistics table. If ``verbose`` is
    ``True``, also prints the full thinking and output content for each
    request in individual panels.

    Args:
        all_stats: List of :class:`RequestStats` objects for all completed
            or failed requests.
        wall_time: Total elapsed wall-clock time in seconds from the start
            of the parallel run to completion.
        verbose: If ``True``, print the full thinking and output text for
            each request after the summary tables.
    """
    console = Console()
    console.print()

    table = Table(title="Parallel Request Results", show_lines=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("TTFT", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Think tok", justify="right")
    table.add_column("Content tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("Finish", justify="center")
    table.add_column("Prompt", max_width=40)

    for s in all_stats:
        if s.repetition_stopped:
            status = "[yellow]REP[/]"
        elif s.status == "done":
            status = "[green]OK[/]"
        else:
            status = f"[red]{s.status}[/]"
        finish = s.finish_reason or "-"
        if s.repetition_stopped:
            finish = f"[yellow]repetition: {s.repetition_reason[:40]}[/]"
        elif s.thinking and not s.output:
            finish = f"[yellow]{finish} (thinking only!)[/]"
        elif s.finish_reason == "length":
            finish = f"[yellow]{finish}[/]"
        think_est = len(s.thinking) // 4 if s.thinking else 0
        content_est = len(s.output) // 4 if s.output else 0
        table.add_row(
            str(s.request_id),
            status,
            f"{s.ttft:.2f}s" if s.ttft > 0 else "-",
            f"{s.total_time:.1f}s",
            str(s.prompt_tokens),
            f"~{think_est}" if think_est else "-",
            f"~{content_est}" if content_est else "-",
            str(s.output_tokens),
            f"{s.tokens_per_sec:.1f}" if s.tokens_per_sec > 0 else "-",
            finish,
            s.prompt[:40] + ("..." if len(s.prompt) > 40 else ""),
        )

    # Full output log for each request
    console.print()
    for s in all_stats:
        if s.thinking and verbose:
            console.print(
                Panel(
                    Text(s.thinking, style="dim cyan"),
                    title=f"[bold]#{s.request_id} thinking[/]",
                    border_style="cyan",
                    expand=True,
                )
            )
        if s.repetition_stopped and s.clean_output:
            console.print(
                Panel(
                    Text(s.clean_output + "\n\n[truncated — repetition detected]"),
                    title=f"[bold]#{s.request_id} clean output[/] (repetition stopped)",
                    border_style="yellow",
                    expand=True,
                )
            )
        if s.repetition_stopped and s.repetition_diagnostics:
            import json as _json

            diag_text = _json.dumps(s.repetition_diagnostics, indent=2, ensure_ascii=False, default=str)
            console.print(
                Panel(
                    Text(diag_text),
                    title=f"[bold]#{s.request_id} repetition diagnostics[/]",
                    border_style="yellow",
                    expand=True,
                )
            )
        elif s.output:
            console.print(
                Panel(
                    Text(s.output),
                    title=f"[bold]#{s.request_id} full output[/] ({s.status})",
                    border_style="green" if s.status == "done" else "red",
                    expand=True,
                )
            )
        elif s.error:
            console.print(
                Panel(
                    Text(s.error, style="red"),
                    title=f"[bold red]#{s.request_id} error[/]",
                    border_style="red",
                    expand=True,
                )
            )

    # Results table + aggregate stats at the end (after details, so no scrolling needed)
    console.print(table)
    done = [s for s in all_stats if s.status == "done"]
    if done:
        total_out = sum(s.output_tokens for s in done)
        total_prompt = sum(s.prompt_tokens for s in done)
        total_think_est = sum(len(s.thinking) // 4 for s in done)
        total_content_est = sum(len(s.output) // 4 for s in done)
        avg_ttft = sum(s.ttft for s in done) / len(done)
        avg_tps = sum(s.tokens_per_sec for s in done) / len(done)
        p50_ttft = sorted(s.ttft for s in done)[len(done) // 2]
        p50_tps = sorted(s.tokens_per_sec for s in done)[len(done) // 2]

        agg = Table(title="Aggregate Stats", show_lines=True)
        agg.add_column("Metric", style="bold")
        agg.add_column("Value", justify="right")
        agg.add_row("Wall time", f"{wall_time:.1f}s")
        agg.add_row("Successful requests", str(len(done)))
        agg.add_row("Failed requests", str(len(all_stats) - len(done)))
        agg.add_row("Total prompt tokens", str(total_prompt))
        agg.add_row("Total output tokens", str(total_out))
        agg.add_row("  Think tokens (est.)", f"~{total_think_est}")
        agg.add_row("  Content tokens (est.)", f"~{total_content_est}")
        agg.add_row("Aggregate throughput", f"{total_out / wall_time:.1f} tok/s")
        agg.add_row("Avg TTFT", f"{avg_ttft:.2f}s")
        agg.add_row("P50 TTFT", f"{p50_ttft:.2f}s")
        agg.add_row("Avg per-request tok/s", f"{avg_tps:.1f}")
        agg.add_row("P50 per-request tok/s", f"{p50_tps:.1f}")
        console.print(agg)


async def run_parallel_test(
    n: int,
    sglang_url: str,
    model_id: str,
    preset: str | None,
    prompts: list[str],
    max_tokens: int | None,
    verbose: bool = False,
    thinking_budget: int | None = None,
    no_think: bool = False,
    reasoning_effort: str | None = None,
    no_guard: str | None = None,
    result_file: str | None = None,
    extra_info: dict[str, object] | None = None,
    stall_timeout: float = 120.0,
    abort_event: threading.Event | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> None:
    """Run ``n`` parallel streaming requests with a live Rich display.

    Constructs one payload per request by combining the resolved sampling
    preset, optional thinking-budget logit processor, and any per-request
    overrides. All requests are fired concurrently via asyncio and
    :mod:`aiohttp`. A :class:`rich.live.Live` display is refreshed four times
    per second until all requests finish. A final summary is printed when
    complete.

    Args:
        n: Number of parallel requests to issue.
        sglang_url: Base URL of the SGLang server.
        model_id: The model identifier to pass in each request payload.
        preset: Name of the sampling preset to apply, or ``None`` to use the
            model's default preset.
        prompts: Pool of prompt strings to draw from. Requests are assigned
            prompts round-robin (``prompts[i % len(prompts)]``).
        max_tokens: Maximum output tokens per request, or ``None`` to let the
            server apply its default.
        verbose: If ``True``, display thinking tokens in live panels and print
            full output after completion.
        thinking_budget: Maximum number of thinking tokens allowed, enforced
            via an SGLang custom logit processor. ``None`` means no cap
            beyond the model profile default.
        no_think: If ``True``, disable reasoning for all requests — sets
            ``enable_thinking=False`` (Qwen family) AND ``reasoning_effort="none"``
            (Mistral family) so the switch works regardless of model.
        reasoning_effort: If set, ENABLE/level reasoning per request via SGLang's
            native top-level ``reasoning_effort`` field (e.g. ``"high"`` for
            Mistral Large-3 / Medium-3.5, which default to no reasoning). Mistral
            accepts only ``"none"``/``"high"``; other families also take
            ``low``/``medium``/``max``. Ignored when ``no_think`` is set.
    """
    # Build payload template
    presets = load_sampling_presets(model_id)
    default_preset = pick_default_preset(presets)
    # Resolve the thinking budget logit processor for this model's reasoning_parser
    _reasoning_parser = _dgx_defaults.get("sglang_model_profiles", {}).get(model_id, {}).get("reasoning_parser", "")  # type: ignore[attr-defined]
    _thinking_processor = _THINKING_BUDGET_PROCESSORS.get(_reasoning_parser)
    if preset is None:
        preset = default_preset

    all_stats: list[RequestStats] = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        all_stats.append(RequestStats(request_id=i + 1, prompt=prompt))

    # Build payloads
    payloads: list[dict[str, object]] = []
    for s in all_stats:
        messages: list[dict[str, str]] = [{"role": "user", "content": s.prompt}]
        payload: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if no_think:
            # Disable reasoning across families: Qwen reads enable_thinking,
            # Mistral reads reasoning_effort. Set both — each model honours its
            # own switch and ignores the other.
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False  # type: ignore[index]
            payload["reasoning_effort"] = "none"
        elif reasoning_effort is not None:
            # Enable/level reasoning via SGLang's native top-level field;
            # serving_chat maps it into the chat template's reasoning_effort kwarg
            # (Mistral [THINK] blocks → reasoning_content via reasoning_parser=mistral).
            payload["reasoning_effort"] = reasoning_effort
        # thinking_budget: CLI arg overrides profile default
        _profile = _dgx_defaults.get("sglang_model_profiles", {}).get(model_id, {})  # type: ignore[attr-defined]
        _effective_budget = thinking_budget if thinking_budget is not None else _profile.get("thinking_budget")
        if _effective_budget is not None and _thinking_processor:
            # thinking_budget uses SGLang's custom logit processor, NOT chat_template_kwargs.
            # The processor forces </think> after N thinking tokens by manipulating logits.
            payload["custom_logit_processor"] = _thinking_processor
            payload["custom_params"] = {"thinking_budget": _effective_budget}
        # Apply preset sampling params
        if preset and preset in presets:
            p = presets[preset]
            for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in p:
                    payload[k] = p[k]
            extra = p.get("extra_body", {})
            payload.update(extra)  # type: ignore[arg-type]
        # Per-run CLI overrides win over the profile preset (e.g. raise temperature
        # for a reasoning_effort=high run without editing the model profile).
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        payloads.append(payload)

    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    # Auth header for the LiteLLM proxy (LITELLM_API_KEY / OPENAI_API_KEY);
    # absent → plain Content-Type, i.e. the keyless SGLang-direct case.
    req_headers = {"Content-Type": "application/json"}
    _key = litellm_api_key()
    if _key:
        req_headers["Authorization"] = f"Bearer {_key}"
    console = Console()
    if no_think:
        think_info = " | Thinking: OFF"
    elif reasoning_effort is not None:
        think_info = f" | reasoning_effort: {reasoning_effort}"
    elif thinking_budget is not None:
        think_info = f" | Thinking budget: {thinking_budget}"
    else:
        think_info = ""
    if temperature is not None:
        think_info += f" | temperature: {temperature} (override)"
    if top_p is not None:
        think_info += f" | top_p: {top_p} (override)"
    _header_lines = [
        f"[bold]Starting {n} parallel requests to {url}[/]",
        f"[dim]Model: {model_id} | Preset: {preset} | Max tokens: {max_tokens}{think_info}[/]",
    ]
    if extra_info:
        params = "  ".join(f"{k}={v}" for k, v in extra_info.items())
        _header_lines.append(f"[dim]Params: {params}[/]")
    _live_header = "\n".join(_header_lines)

    wall_start = time.monotonic()

    # "q" key listener: pressing q cancels all pending requests and prints
    # the summary with whatever has been collected so far.
    abort_requested = False

    def _on_stdin_ready() -> None:
        nonlocal abort_requested
        ch = sys.stdin.read(1)
        if ch.lower() == "q":
            abort_requested = True

    loop = asyncio.get_event_loop()
    # Only register stdin reader if stdin is a TTY (not piped)
    if sys.stdin.isatty():
        import tty, termios

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())  # char-at-a-time, no echo
        loop.add_reader(sys.stdin.fileno(), _on_stdin_ready)

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                stream_request(session, url, payloads[i], all_stats[i], no_guard=no_guard, headers=req_headers)
                for i in range(n)
            ]

            # Live display updates while requests stream
            with Live(
                build_live_display(all_stats, verbose, header=_live_header), console=console, refresh_per_second=4
            ) as live:
                # Start all tasks
                pending: set[asyncio.Task[None]] = set()
                for t in tasks:
                    pending.add(asyncio.ensure_future(t))

                while pending:
                    done_tasks, pending = await asyncio.wait(pending, timeout=0.25, return_when=asyncio.FIRST_COMPLETED)
                    live.update(build_live_display(all_stats, verbose, header=_live_header))

                    if abort_requested or (abort_event is not None and abort_event.is_set()):
                        reason = (
                            "external abort (pod failure)"
                            if (abort_event is not None and abort_event.is_set())
                            else "user (q pressed)"
                        )
                        for t in pending:  # type: ignore[assignment]
                            t.cancel()  # type: ignore[attr-defined]
                        # Wait for cancellations to propagate
                        if pending:
                            await asyncio.wait(pending, timeout=2)
                        for s in all_stats:
                            if s.status == "streaming":
                                s.status = "aborted"
                                s.total_time = time.monotonic() - s._start
                        console.print(f"\n[bold yellow]Aborted: {reason}[/]")
                        break

                    # Stall detection: if ALL streaming requests have
                    # received no token for stall_timeout seconds, abort.
                    if stall_timeout > 0:
                        now_mono = time.monotonic()
                        streaming = [s for s in all_stats if s.status == "streaming"]
                        if streaming and all((now_mono - s._last_token_time) > stall_timeout for s in streaming):
                            for t in pending:  # type: ignore[assignment]
                                t.cancel()  # type: ignore[attr-defined]
                            if pending:
                                await asyncio.wait(pending, timeout=2)
                            for s in streaming:
                                s.status = "error"
                                s.error = f"stall: no tokens for {stall_timeout:.0f}s"
                                s.total_time = now_mono - s._start
                            console.print(f"\n[bold red]All streams stalled for {stall_timeout:.0f}s — aborting[/]")
                            break

                # Final update
                live.update(build_live_display(all_stats, verbose))
    finally:
        if sys.stdin.isatty():
            loop.remove_reader(sys.stdin.fileno())
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    wall_time = time.monotonic() - wall_start
    print_final_summary(all_stats, wall_time, verbose)

    if result_file:
        _write_result_json(all_stats, wall_time, model_id, preset, result_file)


def _write_result_json(
    all_stats: list[RequestStats],
    wall_time: float,
    model_id: str,
    preset: str | None,
    result_file: str,
) -> None:
    """Write per-request results and aggregate stats to a JSON file."""
    requests_out = []
    # Capture leading + trailing snippets of generated text so correctness
    # regressions (e.g. word-salad / repetition loops) are visible directly
    # in TESTRESULTS_*.json without having to dig through the bench-job log.
    SNIPPET_HEAD = 1000
    SNIPPET_TAIL = 500
    for s in all_stats:
        think_est = len(s.thinking) // 4 if s.thinking else 0
        content_est = len(s.output) // 4 if s.output else 0
        full_text = (s.thinking or "") + (s.output or "")
        if len(full_text) > SNIPPET_HEAD + SNIPPET_TAIL:
            response_snippet = (
                full_text[:SNIPPET_HEAD]
                + f"\n…[truncated {len(full_text) - SNIPPET_HEAD - SNIPPET_TAIL} chars]…\n"
                + full_text[-SNIPPET_TAIL:]
            )
        else:
            response_snippet = full_text
        requests_out.append(
            {
                "request_id": s.request_id,
                "status": "repetition" if s.repetition_stopped else s.status,
                "ttft": round(s.ttft, 4) if s.ttft > 0 else None,
                "total_time": round(s.total_time, 2),
                "prompt_tokens": s.prompt_tokens,
                "think_tokens_est": think_est or None,
                "content_tokens_est": content_est or None,
                "output_tokens": s.output_tokens,
                "tokens_per_sec": round(s.tokens_per_sec, 2) if s.tokens_per_sec > 0 else None,
                "finish_reason": s.finish_reason or None,
                "prompt": s.prompt[:120],
                "response_snippet": response_snippet or None,
                "response_total_chars": len(full_text) or None,
            }
        )

    done = [s for s in all_stats if s.status == "done"]
    aggregate: dict[str, object] = {
        "wall_time": round(wall_time, 2),
        "successful_requests": len(done),
        "failed_requests": len(all_stats) - len(done),
    }
    if done:
        total_out = sum(s.output_tokens for s in done)
        total_prompt = sum(s.prompt_tokens for s in done)
        total_think_est = sum(len(s.thinking) // 4 for s in done)
        total_content_est = sum(len(s.output) // 4 for s in done)
        avg_ttft = sum(s.ttft for s in done) / len(done)
        avg_tps = sum(s.tokens_per_sec for s in done) / len(done)
        p50_ttft = sorted(s.ttft for s in done)[len(done) // 2]
        p50_tps = sorted(s.tokens_per_sec for s in done)[len(done) // 2]
        aggregate.update(
            {
                "total_prompt_tokens": total_prompt,
                "total_output_tokens": total_out,
                "think_tokens_est": total_think_est,
                "content_tokens_est": total_content_est,
                "aggregate_throughput": round(total_out / wall_time, 2) if wall_time > 0 else 0,
                "avg_ttft": round(avg_ttft, 4),
                "p50_ttft": round(p50_ttft, 4),
                "avg_per_request_tps": round(avg_tps, 2),
                "p50_per_request_tps": round(p50_tps, 2),
            }
        )

    result = {
        "model": model_id,
        "preset": preset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requests": requests_out,
        "aggregate": aggregate,
    }
    Path(result_file).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Console().print(f"[dim]Results written to {result_file}[/]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse command-line arguments and run the selected integration tests.

    Supports both sequential tests (xkcd, briefing, thinking, coding,
    sampling, presets) and the async parallel load test. The ``"all"``
    shorthand expands to all sequential tests. The ``"parallel"`` test is
    handled separately and launched via :func:`asyncio.run`.

    Raises:
        ValueError: If ``SGLANG_URL`` is not set or the model ID cannot be
            resolved when running the parallel test.
        SystemExit: If argument parsing fails or the model validation check
            fails.
    """
    print_banner(module=Path(__file__).stem)
    import argparse

    parser = argparse.ArgumentParser(description="Direct SGLang integration tests")

    # Default: run named tests
    parser.add_argument(
        "tests",
        nargs="*",
        default=["thinking"],
        help="Tests to run: xkcd, briefing, thinking, coding, sampling, presets, "
        "parallel, all (default: thinking). Combine with --no-think to "
        "disable reasoning on any test.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show thinking/reasoning tokens and full payloads",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=4,
        help="Number of parallel requests (for 'parallel' test, default: 4)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Sampling preset for parallel test (default: model's default)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for all parallel requests (default: varied prompts)",
    )
    parser.add_argument(
        "--max-tokens",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=8192,
        help="Max output tokens per request for parallel test (default: 8192, 'none' for model default)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Max thinking tokens (caps reasoning length so content tokens aren't exhausted)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking/reasoning (Qwen: enable_thinking=false; Mistral: reasoning_effort=none)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["none", "no_think", "low", "medium", "high", "max"],
        help="Enable/level reasoning per request via SGLang's native reasoning_effort field. "
        "Required to turn ON reasoning for Mistral Large-3 / Medium-3.5 (use 'high'; they "
        "default to no reasoning and accept only none/high). "
        "⚠ Valid values are MODEL-FAMILY-SPECIFIC and unvalidated ones raise a chat-template "
        "error server-side: Hy3/Hunyuan accepts ONLY high/low/no_think (its OFF value is "
        "'no_think', NOT 'none' — 'none'/'medium'/'max' error); Mistral takes none/high. "
        "Ignored with --no-think (which sends reasoning_effort=none — wrong for Hy3; use "
        "--reasoning-effort no_think to turn Hy3 thinking OFF).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature for the parallel test (wins over the model "
        "profile preset). Use e.g. 0.7 when turning reasoning ON via --reasoning-effort high.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override top_p for the parallel test (wins over the model profile preset).",
    )
    parser.add_argument(
        "--no-guard",
        nargs="?",
        const="both",
        default=None,
        metavar="STREAM",
        help="Disable streaming repetition guard. "
        "Values: content, reasoning, both (default: both if flag given without value)",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Write parallel test results (per-request + aggregate) to a JSON file",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    verbose: bool = args.verbose
    no_think: bool = args.no_think
    tests: set[str] = set(args.tests)
    if "all" in tests:
        tests = {"xkcd", "briefing", "thinking", "coding", "sampling", "presets"}

    # Show config summary and wait for confirmation
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    config_summary = {
        "tests": sorted(tests),
        "verbose": verbose,
        "no_think": no_think,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "thinking_budget": args.thinking_budget,
        "num_requests": args.num_requests,
        "preset": args.preset,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "no_guard": args.no_guard,
    }
    Console().print(
        Panel(
            Syntax(json.dumps(config_summary, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title="[bold]Test Configuration[/]",
            border_style="cyan",
        )
    )
    if not args.yes:
        try:
            input("[Enter to run tests, Ctrl+C to abort] (press 'q' during parallel tests to stop early) ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    # Handle parallel test separately (async)
    if "parallel" in tests:
        tests.discard("parallel")
        model_id = resolve_model_id()
        use_litellm = os.environ.get("USE_LITELLM", "").lower() in ("1", "true", "yes")
        sglang_url = os.environ.get("LITELLM_URL", "") if use_litellm else os.environ.get("SGLANG_URL", "")
        if not sglang_url:
            raise ValueError("Set SGLANG_URL (or LITELLM_URL with USE_LITELLM=true)")
        validate_model(sglang_url, model_id)
        prompts: list[str] = (
            [args.prompt] * args.num_requests if args.prompt else random.sample(PARALLEL_PROMPTS, len(PARALLEL_PROMPTS))
        )
        asyncio.run(
            run_parallel_test(
                n=args.num_requests,
                sglang_url=sglang_url,
                model_id=model_id,
                preset=args.preset,
                prompts=prompts,
                max_tokens=args.max_tokens,
                verbose=verbose,
                thinking_budget=args.thinking_budget,
                no_think=no_think,
                reasoning_effort=args.reasoning_effort,
                no_guard=args.no_guard,
                result_file=args.result_file,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    # Sequential tests
    if tests:
        # --no-think forces reasoning off across families (Mistral: reasoning_effort
        # none) even for the always-thinking coding test; otherwise honour the
        # explicit --reasoning-effort (e.g. "high" to turn Mistral reasoning ON).
        _seq_effort = "none" if no_think else args.reasoning_effort
        client = create_sglang_client(verbose=verbose, reasoning_effort=_seq_effort)
        if "xkcd" in tests:
            image = get_random_xkcd_image(get_random_xkcd_image_url())
            print_ascii_representation_of_image(image)
            client.explain_image(image, print_thinking=verbose and not no_think)

        if "briefing" in tests:
            print(f"\n{'*' * 80}")
            t0 = time.monotonic()
            client.get_daily_briefing(print_thinking=verbose and not no_think)
            elapsed = time.monotonic() - t0
            label = "non-thinking" if no_think else "thinking"
            print(f"\n--- Daily Briefing ({label}) completed in {elapsed:.1f}s ---")

        if "thinking" in tests:
            if no_think:
                test_non_thinking_mode(client)
            else:
                test_thinking_mode(client, print_thinking=verbose)

        if "coding" in tests:
            test_thinking_coding(client)

        if "sampling" in tests:
            test_sampling_params_passthrough(client)

        if "presets" in tests:
            test_all_presets(client)


if __name__ == "__main__":
    main()
