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
    python sglang_integration_test.py parallel -n 8 --prompt "Erkläre Quantencomputing"
"""

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field

import aiohttp
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import requests as httplib

from openwebui_integration_test import (
    SGLangClient,
    _dgx_defaults,
    get_random_xkcd_image,
    get_random_xkcd_image_url,
    load_sampling_presets,
    pick_default_preset,
    print_ascii_representation_of_image,
    test_all_presets,
    test_non_thinking_mode,
    test_sampling_params_passthrough,
    test_thinking_coding,
    test_thinking_mode,
)

# Default model from Ansible defaults (the model currently deployed to SGLang)
_CONFIGURED_MODEL: str = _dgx_defaults.get("sglang_model", "")

def _serialize_logit_processor(module: str, cls_name: str) -> str:
    """Build SGLang custom_logit_processor JSON from module path + class name.

    SGLang expects a dill-serialized class reference, but for importable classes
    this is just a standard pickle of the module+qualname (no code, no state).
    We construct the pickle bytes directly — no dill/sglang dependency needed.
    """
    import io
    import struct
    buf = io.BytesIO()
    buf.write(b'\x80\x04\x95')                                     # PROTO 4 + FRAME
    mod, cls = module.encode(), cls_name.encode()
    frame = (b'\x8c' + bytes([len(mod)]) + mod +                    # SHORT_BINUNICODE module
             b'\x94\x8c' + bytes([len(cls)]) + cls +                # MEMOIZE + SHORT_BINUNICODE class
             b'\x94\x93\x94\x2e')                                   # MEMOIZE + STACK_GLOBAL + MEMOIZE + STOP
    buf.write(struct.pack('<Q', len(frame)))
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
PARALLEL_PROMPTS = [
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
        "You are a senior data engineer and analytics consultant who has optimized SQL queries "
        "for data warehouses processing petabytes of data. You think carefully about query "
        "plans, index usage, and the practical differences between database engines.\n\n"
        "A junior analyst needs to write a SQL query that finds the top 3 customers by total "
        "spend per month for the last year. Write the query using window functions and CTEs, "
        "explain each clause, and discuss performance considerations. Then show alternative "
        "approaches: a correlated subquery version and a LATERAL JOIN version. Compare their "
        "execution plans conceptually and advise on indexing strategy. Assume a PostgreSQL "
        "database with tables `customers(id, name)` and `orders(id, customer_id, amount, "
        "created_at)`."
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
    """Query /v1/models and warn if the running model doesn't match MODEL_ID."""
    try:
        resp = httplib.get(f"{sglang_url.rstrip('/')}/v1/models", timeout=5)
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
    """Resolve MODEL_ID: env var > Ansible defaults > error."""
    model_id = os.environ.get("MODEL_ID", "")
    if model_id:
        return model_id
    if _CONFIGURED_MODEL:
        return _CONFIGURED_MODEL
    raise ValueError("No MODEL_ID env var and no sglang_model in Ansible defaults")


def create_sglang_client(verbose: bool = False) -> SGLangClient:
    """Create a SGLang client from environment variables."""
    model_id = resolve_model_id()
    sglang_url = os.environ.get("SGLANG_URL", "")
    if not sglang_url:
        raise ValueError(
            "Set SGLANG_URL environment variable (e.g. https://sglang.dgx.example.com)"
        )
    validate_model(sglang_url, model_id)
    print(f"[SGLang direct] {sglang_url} model={model_id}")
    return SGLangClient(sglang_url, model_id, verbose=verbose)


# ---------------------------------------------------------------------------
# Parallel load test
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    """Tracks stats for a single parallel request."""
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
    _start: float = field(default=0.0, repr=False)
    _first_token: bool = field(default=False, repr=False)

    @property
    def tokens_per_sec(self) -> float:
        if self.status == "done" or self.status == "error":
            t = self.total_time
        else:
            t = (time.monotonic() - self._start) if self._start else 0
        tokens = self.output_tokens if self.output_tokens > 0 else (len(self.output) + len(self.thinking)) // 4
        if t > 0 and tokens > 0:
            return tokens / t
        return 0.0

    def output_tail(self, max_chars: int = 1200) -> str:
        """Last N chars of output for display."""
        if len(self.output) <= max_chars:
            return self.output
        return "..." + self.output[-max_chars:]


def _detect_repetition(text: str, min_output: int = 800, max_ratio: float = 0.20) -> bool:
    """Detect degenerate repetition via compression ratio.

    Repetitive text compresses extremely well.  Normal prose/code compresses
    to ~30-50% of its original size (zlib); degenerate loops compress to <20%.
    This catches all repetition patterns regardless of exact/near-match and
    block alignment — subword loops, sentence repetition, paragraph repetition.

    Only checks the tail (last 1500 chars) to detect when the model *starts*
    looping, not penalise early legitimate structure.
    """
    import zlib
    if len(text) < min_output:
        return False
    tail = text[-1500:] if len(text) > 1500 else text
    compressed = zlib.compress(tail.encode())
    ratio = len(compressed) / len(tail.encode())
    return ratio < max_ratio


ENABLE_REPETTITION_DETECTION: bool = False


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    stats: RequestStats,
) -> None:
    """Stream a single chat completion and update stats in-place."""
    stats.status = "streaming"
    stats._start = time.monotonic()
    try:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
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
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    if not stats._first_token:
                        stats._first_token = True
                        stats.ttft = time.monotonic() - stats._start
                    stats.thinking += reasoning
                content = delta.get("content", "")
                if content:
                    if not stats._first_token:
                        stats._first_token = True
                        stats.ttft = time.monotonic() - stats._start
                    stats.output += content
                    # Abort on degenerate repetition loops
                    if ENABLE_REPETTITION_DETECTION and _detect_repetition(stats.output):
                        stats.status = "error"
                        stats.error = "aborted: degenerate repetition detected"
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
    """Format thinking/content token breakdown for panel headers."""
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


def build_live_display(all_stats: list[RequestStats], verbose: bool = False) -> Table:
    """Build a rich Table showing all parallel request states."""
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
        summary.add_row(
            f"[bold]Aggregate:[/] {agg_tokens} tokens",
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
    # Reserve 4 lines for summary header, split remaining height across panel rows
    panel_height = max(8, (console_height - 4) // n_rows) if n_rows > 0 else 16
    # Usable lines/width inside panel (subtract borders + padding)
    inner_width = col_width - 4
    inner_lines = panel_height - 2
    # Conservative: assume avg ~45% line fill due to word wrap and short lines
    max_chars = int(inner_width * inner_lines * 0.45)

    panels = []
    for s in all_stats:
        if s.status == "pending":
            style = "dim"
            header = f"[dim]#{s.request_id} pending[/]"
            body = Text(s.prompt[:80] + "...", style="dim")
        elif s.status == "streaming":
            elapsed = time.monotonic() - s._start
            style = "yellow"
            tps = f" {s.tokens_per_sec:.1f} t/s" if s._first_token else ""
            tok_detail = _tok_detail(s)
            header = f"[yellow]#{s.request_id} streaming {elapsed:.1f}s{tps}{tok_detail}[/]"
            if verbose and s.thinking:
                display = f"[thinking]\n{s.thinking[-max_chars // 2:]}\n[/thinking]\n{s.output_tail(max_chars // 2)}"
            else:
                display = s.output_tail(max_chars)
            body = Text(display, style="white")
        elif s.status == "done":
            style = "green"
            tok_detail = _tok_detail(s)
            header = (
                f"[green]#{s.request_id} done[/] "
                f"TTFT={s.ttft:.2f}s | {s.total_time:.1f}s | "
                f"{s.output_tokens} tok{tok_detail} | {s.tokens_per_sec:.1f} t/s"
            )
            if verbose and s.thinking:
                display = f"[thinking]\n{s.thinking[-max_chars // 2:]}\n[/thinking]\n{s.output_tail(max_chars // 2)}"
            else:
                display = s.output_tail(max_chars)
            body = Text(display, style="white")
        else:
            style = "red"
            header = f"[red]#{s.request_id} ERROR[/]"
            body = Text(s.error[:max_chars], style="red")

        panels.append(Panel(body, title=header, border_style=style, height=panel_height))
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=col_width)
    grid.add_column(width=col_width)
    for i in range(0, len(panels), 2):
        left = panels[i]
        right = panels[i + 1] if i + 1 < len(panels) else ""
        grid.add_row(left, right)

    outer = Table.grid()
    outer.add_row(summary)
    outer.add_row(grid)
    return outer


def print_final_summary(all_stats: list[RequestStats], wall_time: float, verbose: bool = False) -> None:
    """Print a final results table after all requests complete."""
    console = Console()
    console.print()

    table = Table(title="Parallel Request Results", show_lines=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("TTFT", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("Finish", justify="center")
    table.add_column("Prompt", max_width=40)

    for s in all_stats:
        status = "[green]OK[/]" if s.status == "done" else f"[red]{s.status}[/]"
        finish = s.finish_reason or "-"
        if s.thinking and not s.output:
            finish = f"[yellow]{finish} (thinking only!)[/]"
        elif s.finish_reason == "length":
            finish = f"[yellow]{finish}[/]"
        table.add_row(
            str(s.request_id),
            status,
            f"{s.ttft:.2f}s" if s.ttft > 0 else "-",
            f"{s.total_time:.1f}s",
            str(s.prompt_tokens),
            str(s.output_tokens),
            f"{s.tokens_per_sec:.1f}" if s.tokens_per_sec > 0 else "-",
            finish,
            s.prompt[:40] + ("..." if len(s.prompt) > 40 else ""),
        )

    console.print(table)

    done = [s for s in all_stats if s.status == "done"]
    if done:
        total_out = sum(s.output_tokens for s in done)
        total_prompt = sum(s.prompt_tokens for s in done)
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
        agg.add_row("Aggregate throughput", f"{total_out / wall_time:.1f} tok/s")
        agg.add_row("Avg TTFT", f"{avg_ttft:.2f}s")
        agg.add_row("P50 TTFT", f"{p50_ttft:.2f}s")
        agg.add_row("Avg per-request tok/s", f"{avg_tps:.1f}")
        agg.add_row("P50 per-request tok/s", f"{p50_tps:.1f}")
        console.print(agg)

    # Full output log for each request
    console.print()
    for s in all_stats:
        if s.thinking and verbose:
            console.print(Panel(
                Text(s.thinking, style="dim cyan"),
                title=f"[bold]#{s.request_id} thinking[/]",
                border_style="cyan",
                expand=True,
            ))
        if s.output:
            console.print(Panel(
                Text(s.output),
                title=f"[bold]#{s.request_id} full output[/] ({s.status})",
                border_style="green" if s.status == "done" else "red",
                expand=True,
            ))
        elif s.error:
            console.print(Panel(
                Text(s.error, style="red"),
                title=f"[bold red]#{s.request_id} error[/]",
                border_style="red",
                expand=True,
            ))


async def run_parallel_test(
    n: int,
    sglang_url: str,
    model_id: str,
    preset: str | None,
    prompts: list[str],
    max_tokens: int|None,
    verbose: bool = False,
    thinking_budget: int|None = None,
    no_think: bool = False,
) -> None:
    """Run n parallel streaming requests with live display."""
    # Build payload template
    presets = load_sampling_presets(model_id)
    default_preset = pick_default_preset(presets)
    # Resolve the thinking budget logit processor for this model's reasoning_parser
    _reasoning_parser = _dgx_defaults.get("sglang_model_profiles", {}).get(model_id, {}).get("reasoning_parser", "")
    _thinking_processor = _THINKING_BUDGET_PROCESSORS.get(_reasoning_parser)
    if preset is None:
        preset = default_preset

    all_stats: list[RequestStats] = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        all_stats.append(RequestStats(request_id=i + 1, prompt=prompt))

    # Build payloads
    payloads = []
    for s in all_stats:
        messages = [{"role": "user", "content": s.prompt}]
        payload: dict = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if no_think:
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        # thinking_budget: CLI arg overrides profile default
        _profile = _dgx_defaults.get("sglang_model_profiles", {}).get(model_id, {})
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
            payload.update(extra)
        payloads.append(payload)

    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    console = Console()
    console.print(f"[bold]Starting {n} parallel requests to {url}[/]")
    think_info = " | Thinking: OFF" if no_think else (f" | Thinking budget: {thinking_budget}" if thinking_budget is not None else "")
    console.print(f"[dim]Model: {model_id} | Preset: {preset} | Max tokens: {max_tokens}{think_info}[/]\n")

    wall_start = time.monotonic()

    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_request(session, url, payloads[i], all_stats[i])
            for i in range(n)
        ]

        # Live display updates while requests stream
        with Live(build_live_display(all_stats, verbose), console=console, refresh_per_second=4) as live:
            # Start all tasks
            pending = set()
            for t in tasks:
                pending.add(asyncio.ensure_future(t))

            while pending:
                done_tasks, pending = await asyncio.wait(
                    pending, timeout=0.25, return_when=asyncio.FIRST_COMPLETED
                )
                live.update(build_live_display(all_stats, verbose))

            # Final update
            live.update(build_live_display(all_stats, verbose))

    wall_time = time.monotonic() - wall_start
    print_final_summary(all_stats, wall_time, verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
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
        "-v", "--verbose",
        action="store_true",
        help="Show thinking/reasoning tokens and full payloads",
    )
    parser.add_argument(
        "-n", "--num-requests",
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
        help="Disable thinking/reasoning (sets enable_thinking=false via chat_template_kwargs)",
    )
    args = parser.parse_args()

    verbose = args.verbose
    no_think = args.no_think
    tests = set(args.tests)
    if "all" in tests:
        tests = {"xkcd", "briefing", "thinking", "coding", "sampling", "presets"}

    # Handle parallel test separately (async)
    if "parallel" in tests:
        tests.discard("parallel")
        model_id = resolve_model_id()
        sglang_url = os.environ.get("SGLANG_URL", "")
        if not sglang_url:
            raise ValueError("Set SGLANG_URL environment variable")
        validate_model(sglang_url, model_id)
        prompts = [args.prompt] * args.num_requests if args.prompt else random.sample(PARALLEL_PROMPTS, len(PARALLEL_PROMPTS))
        asyncio.run(run_parallel_test(
            n=args.num_requests,
            sglang_url=sglang_url,
            model_id=model_id,
            preset=args.preset,
            prompts=prompts,
            max_tokens=args.max_tokens,
            verbose=verbose,
            thinking_budget=args.thinking_budget,
            no_think=no_think,
        ))

    # Sequential tests
    if tests:
        client = create_sglang_client(verbose=verbose)
        # --no-think: select the non_thinking preset variant if available,
        # otherwise fall back to default preset (chat_template_kwargs is
        # already set by load_sampling_presets for non_thinking* presets).
        think_preset = "non_thinking" if no_think else None  # None = use client default

        if "xkcd" in tests:
            image = get_random_xkcd_image(get_random_xkcd_image_url())
            print_ascii_representation_of_image(image)
            client.explain_image(image, print_thinking=verbose and not no_think, preset=think_preset)

        if "briefing" in tests:
            print(f"\n{'*' * 80}")
            t0 = time.monotonic()
            client.get_daily_briefing(print_thinking=verbose and not no_think, preset=think_preset or "thinking")
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
