#!/usr/bin/env python3
"""Raw SGLang SSE stream viewer with dual-panel Rich display.

Upper panel: interpreted output (thinking + content) with live token stats.
Lower panel: raw SSE JSON chunks with syntax highlighting.

Module-level configuration is loaded from Ansible defaults via
:func:`openwebui_integration_test._dgx_defaults`.

Usage::

    python sglang_raw.py "Your prompt here"
    python sglang_raw.py --max-tokens 4096 "Explain TCP vs UDP"
    python sglang_raw.py --thinking-budget 2048 "What is 2+2?"
    python sglang_raw.py --no-think "Capital of France?"

    # Use default prompt (TCP vs UDP)
    python sglang_raw.py
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import cast

import requests

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .openwebui_integration_test import _dgx_defaults, load_sampling_presets, pick_default_preset
from .sglang_integration_test import _THINKING_BUDGET_PROCESSORS
from .streaming_repetition_guard import RepetitionGuard, GuardConfig, FeedResult
from .thinking_parser import ThinkingParser

from dgxarley import configure_logging, glogger, print_banner

os.environ.setdefault("LOGURU_LEVEL", "DEBUG")
configure_logging()
glogger.enable("dgxarley")

from loguru import logger

_CONFIGURED_MODEL: str = str(_dgx_defaults.get("sglang_model", ""))
"""Default model ID loaded from Ansible role defaults."""

_MODEL_PROFILES: dict[str, object] = _dgx_defaults.get("sglang_model_profiles", {})  # type: ignore[assignment]
"""Per-model profile configuration (YAML-loaded, heterogeneous values)."""

DEFAULT_PROMPT: str = (
    "You are a senior network engineer with 15 years of experience designing "
    "enterprise-grade infrastructure. You have deep expertise in protocol design, "
    "packet-level analysis, and performance tuning for high-throughput systems.\n\n"
    "A junior colleague is confused about when to use TCP vs UDP and keeps defaulting "
    "to TCP for everything, including real-time game state updates. Explain the main "
    "differences between TCP and UDP, covering reliability guarantees, ordering, "
    "congestion control, and overhead. Provide concrete examples of when each protocol "
    "is the right choice and explain the trade-offs involved. Include a brief discussion "
    "of QUIC and how it blurs the line between the two."
)
"""Fallback prompt used when none is supplied on the command line."""


def build_payload(
    model_id: str,
    prompt: str,
    max_tokens: int,
    thinking_budget: int | None,
    no_think: bool,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, object]:
    """Build the OpenAI-compatible chat-completions request payload.

    Sampling parameters are seeded from the model's named profile (loaded via
    :func:`load_sampling_presets`).  Any explicitly supplied keyword argument
    overrides the corresponding profile value so that CLI flags always win.

    If ``thinking_budget`` is not given on the CLI the value is taken from the
    model profile, if present.  When a budget is available and the profile
    defines a ``reasoning_parser``, the matching custom logit-processor is
    serialised into the payload.

    Args:
        model_id: Identifier of the model to query (e.g. a HuggingFace repo
            string or a short alias recognised by SGLang).
        prompt: The user turn text to send as the sole message.
        max_tokens: Hard upper limit on generated tokens.
        thinking_budget: Optional cap on reasoning/thinking tokens applied via
            a custom logit processor.  ``None`` falls back to the profile
            default, or no budget if the profile also lacks one.
        no_think: When ``True``, sets ``chat_template_kwargs.enable_thinking``
            to ``False``, suppressing the reasoning phase entirely.
        temperature: Sampling temperature override.  ``None`` means use the
            profile default.
        top_p: Nucleus-sampling probability mass override.
        top_k: Top-K sampling override.
        min_p: Minimum-probability sampling override.
        presence_penalty: Presence-penalty override.
        frequency_penalty: Frequency-penalty override.
        repetition_penalty: Repetition-penalty override.

    Returns:
        A dictionary suitable for JSON-serialisation as the request body of a
        ``POST /v1/chat/completions`` call with ``stream=True``.
    """
    payload: dict[str, object] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }

    # Sampling from model profile as baseline
    presets = load_sampling_presets(model_id)
    preset = pick_default_preset(presets)
    if preset and preset in presets:
        p = presets[preset]
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if k in p:
                payload[k] = p[k]
        extra = p.get("extra_body", {})
        payload.update(extra)  # type: ignore[arg-type]

    # CLI overrides (win over profile defaults)
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty

    if no_think:
        cast("dict[str, object]", payload.setdefault("chat_template_kwargs", {}))["enable_thinking"] = False

    # thinking_budget: CLI arg overrides profile default
    profile: dict[str, object] = _MODEL_PROFILES.get(model_id, {})  # type: ignore[assignment]
    effective_budget: int | None = (
        thinking_budget if thinking_budget is not None else cast("int | None", profile.get("thinking_budget"))
    )
    if effective_budget is not None:
        reasoning_parser: str = cast(str, profile.get("reasoning_parser", ""))
        processor = _THINKING_BUDGET_PROCESSORS.get(reasoning_parser)
        if processor:
            payload["custom_logit_processor"] = processor
            payload["custom_params"] = {"thinking_budget": effective_budget}

    return payload


def stream_and_display(
    url: str, payload: dict[str, object], raw_json: bool = False, guard_enabled: bool = True
) -> None:
    """Stream a chat-completions SSE response and render it in a two-panel Rich display.

    The upper panel shows the interpreted output (thinking tokens in cyan,
    content tokens in white) together with a live status line carrying elapsed
    time, TTFT, token counts, and repetition-guard state.  The lower panel
    shows either a structured table of SSE chunks (default) or raw JSON lines
    (when ``raw_json=True``).

    The function blocks until the stream ends, the ``[DONE]`` sentinel is
    received, or the repetition guard triggers an early stop.

    Args:
        url: Full URL of the ``/v1/chat/completions`` endpoint.
        payload: Request body dict as produced by :func:`build_payload`.
        raw_json: When ``True``, the lower panel displays raw JSON lines
            instead of the structured chunk table.
        guard_enabled: When ``True``, wraps each stream (thinking and content)
            with a :class:`~streaming_repetition_guard.RepetitionGuard` that
            aborts the request if a runaway repetition loop is detected.
    """
    console = Console()
    thinking_text: str = ""
    content_text: str = ""
    # Lower panel gets 2/5 of height; subtract 4 for overhead
    lower_max_rows: int = max(5, (console.height * 2 // 5) - 4)
    raw_chunks: deque[str] = deque(maxlen=lower_max_rows)
    chunk_rows: deque[dict[str, object]] = deque(maxlen=lower_max_rows)
    t_start: float = time.monotonic()
    t_first_token: float | None = None
    tp = ThinkingParser()
    finish_reason: str = ""
    usage_info: dict[str, object] = {}
    chunk_count: int = 0
    guard_status: str = ""
    thinking_guard: RepetitionGuard | None = RepetitionGuard() if guard_enabled else None
    content_guard: RepetitionGuard | None = RepetitionGuard() if guard_enabled else None

    def _status_line() -> str:
        """Build the one-line status string for the upper panel title.

        Reads closure variables ``t_start``, ``t_first_token``,
        ``thinking_tokens_est``, ``content_tokens_est``, ``usage_info``,
        ``finish_reason``, ``guard_status``, and ``guard_enabled``.

        Returns:
            A human-readable status string covering elapsed time, time-to-first
            token, token counts (estimated or from usage chunk), throughput in
            tokens/s, finish reason, and guard state.
        """
        elapsed = time.monotonic() - t_start
        ttft_str = f"{t_first_token - t_start:.2f}s" if t_first_token else "-"
        total_est = tp.total_tokens_est
        if usage_info:
            prompt_tok = usage_info.get("prompt_tokens", 0)
            completion_tok = usage_info.get("completion_tokens", 0)
            tok_info = f"prompt={prompt_tok} completion={completion_tok}"
        else:
            tok_info = f"T~{tp.thinking_tokens_est} C~{tp.content_tokens_est} total~{total_est}"
        tps = total_est / elapsed if elapsed > 0 and total_est > 0 else 0
        finish_str = f" | finish={finish_reason}" if finish_reason else ""
        guard_str = f" | guard: {guard_status}" if guard_status else (" | guard: ok" if guard_enabled else "")
        return f"{elapsed:.1f}s | TTFT={ttft_str} | {tok_info} | {tps:.1f} tok/s{finish_str}{guard_str}"

    def _interpreted_panel() -> Panel:
        """Render the upper Rich panel containing thinking and content text.

        Allocates three-fifths of the terminal height to the upper panel and
        tail-truncates whichever text sections would overflow.  The panel
        border colour reflects the current state: red for guard stop, green
        for a clean finish, yellow while still streaming.

        Returns:
            A :class:`rich.panel.Panel` ready to be placed in a
            :class:`rich.layout.Layout`.
        """
        # Calculate available lines: upper panel gets 3/5 of terminal height,
        # minus 2 for panel border/title
        avail_lines: int = max(5, (console.height * 3 // 5) - 3)
        panel_width: int = console.width - 4  # border + padding

        def _split_visual(text: str) -> list[str]:
            """Split text into visual lines, breaking long lines at panel_width.

            Args:
                text: Multi-line text to wrap.

            Returns:
                A list of strings where each element represents one visual line
                at most ``panel_width`` characters wide.
            """
            visual: list[str] = []
            for line in text.split("\n"):
                if not line:
                    visual.append("")
                else:
                    while len(line) > panel_width:
                        visual.append(line[:panel_width])
                        line = line[panel_width:]
                    visual.append(line)
            return visual

        def _tail_lines(text: str, max_lines: int) -> tuple[str, bool]:
            """Return the last ``max_lines`` visual lines worth of text.

            Takes word-wrap into account by first expanding ``text`` into
            visual lines via :func:`_split_visual`.

            Args:
                text: Source text, potentially containing newlines.
                max_lines: Maximum number of visual lines to retain.

            Returns:
                A 2-tuple of ``(tail_text, truncated)`` where ``tail_text`` is
                the retained portion joined by newlines and ``truncated`` is
                ``True`` when lines were dropped from the beginning.
            """
            visual = _split_visual(text)
            if len(visual) <= max_lines:
                return "\n".join(visual), False
            return "\n".join(visual[-max_lines:]), True

        interpreted = Text()
        if thinking_text and not content_text:
            # Still thinking — show tail of thinking
            tail, truncated = _tail_lines(thinking_text, avail_lines - 2)
            interpreted.append("[thinking]\n", style="dim cyan")
            if truncated:
                interpreted.append("...\n", style="dim")
            interpreted.append(tail, style="cyan")
        elif thinking_text and content_text:
            # Both — split space: a few lines of thinking tail, rest for content
            think_lines = max(3, avail_lines // 4)
            content_lines = avail_lines - think_lines - 2
            t_tail, t_trunc = _tail_lines(thinking_text, think_lines)
            c_tail, c_trunc = _tail_lines(content_text, content_lines)
            interpreted.append("[thinking] ", style="dim cyan")
            if t_trunc:
                interpreted.append("...", style="dim")
            interpreted.append(t_tail, style="cyan")
            interpreted.append(" [/thinking]\n", style="dim cyan")
            if c_trunc:
                interpreted.append("...\n", style="dim")
            interpreted.append(c_tail, style="white")
        elif content_text:
            tail, truncated = _tail_lines(content_text, avail_lines)
            if truncated:
                interpreted.append("...\n", style="dim")
            interpreted.append(tail, style="white")
        else:
            interpreted.append("waiting for tokens...", style="dim")
        if guard_status:
            border = "red"
        elif finish_reason:
            border = "green"
        else:
            border = "yellow"
        return Panel(
            interpreted,
            title=f"[bold]{_status_line()}[/]",
            border_style=border,
        )

    def _chunk_table() -> Table:
        """Build the structured SSE chunk table for the lower panel.

        Each row in ``chunk_rows`` becomes one table row.  Row style is keyed
        on the ``type`` field: ``think`` → cyan, ``content`` → white,
        ``usage`` → green, ``done``/``guard`` → bold green/red.  Empty rows
        are appended to keep the table height constant at ``lower_max_rows``.

        The ``from rich.table import Table as RichTable`` import is
        intentional — it avoids a name clash with the module-level ``Table``
        import alias used elsewhere.

        Returns:
            A :class:`rich.table.Table` populated with the current chunk rows.
        """
        from rich.table import Table as RichTable

        tbl = RichTable(
            show_lines=False,
            expand=True,
            pad_edge=False,
            title="[bold]SSE chunks[/]",
            border_style="blue",
        )
        tbl.add_column("#", justify="right", style="dim", width=5)
        tbl.add_column("Type", width=8)
        tbl.add_column("Content", ratio=1, overflow="ellipsis", no_wrap=True)
        tbl.add_column("Finish", width=8)
        tbl.add_column("Tokens", justify="right", width=8)
        for row in chunk_rows:
            n = str(row["n"])
            typ = str(row["type"])
            content = str(row["content"])
            fin = str(row.get("finish", ""))
            tokens = str(row.get("tokens", ""))
            if typ == "think":
                style = "cyan"
            elif typ == "content":
                style = "white"
            elif typ == "usage":
                style = "green"
            elif typ == "done":
                style = "bold green"
            elif typ == "guard":
                style = "bold red"
            else:
                style = "dim"
            tbl.add_row(n, typ, content, fin, tokens, style=style)
        # Pad with empty rows so the table always fills the panel
        for _ in range(lower_max_rows - len(chunk_rows)):
            tbl.add_row("", "", "", "", "")
        return tbl

    def make_display() -> Layout:
        """Assemble the full two-panel Rich layout for the current frame.

        The upper panel (ratio 3) is produced by :func:`_interpreted_panel`.
        The lower panel (ratio 2) is either a raw-JSON syntax panel (when
        ``raw_json`` is ``True``) or the structured chunk table from
        :func:`_chunk_table`.

        Returns:
            A :class:`rich.layout.Layout` with ``upper`` and ``lower``
            sub-layouts ready to pass to :class:`rich.live.Live`.
        """
        layout = Layout()
        lower: Panel | Table
        if raw_json:
            # Pad with empty lines so the panel is always full height
            padded = list(raw_chunks) + [""] * (lower_max_rows - len(raw_chunks))
            raw_display = "\n".join(padded)
            lower = Panel(
                Syntax(raw_display, "json", theme="monokai", word_wrap=True),
                title="[bold]Raw SSE chunks[/]",
                border_style="blue",
            )
        else:
            lower = _chunk_table()
        layout.split_column(
            Layout(_interpreted_panel(), name="upper", ratio=3),
            Layout(lower, name="lower", ratio=2),
        )
        return layout

    console.print(f"[bold]POST[/] {url}")
    console.print(f"[dim]{json.dumps({k: v for k, v in payload.items() if k != 'messages'}, indent=2)}[/]\n")

    with requests.post(url, json=payload, stream=True, timeout=1800) as resp:
        if resp.status_code != 200:
            console.print(f"[red]HTTP {resp.status_code}: {resp.text[:500]}[/]")
            return

        with Live(make_display(), console=console, refresh_per_second=8) as live:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:]
                if data == "[DONE]":
                    raw_chunks.append("[DONE]")
                    chunk_rows.append({"n": chunk_count + 1, "type": "done", "content": "[DONE]"})
                    live.update(make_display())
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                chunk_count += 1
                # Compact JSON for raw display
                raw_chunks.append(json.dumps(chunk, ensure_ascii=False))

                # Usage (typically in final chunk)
                u = chunk.get("usage")
                if u:
                    usage_info = u
                    chunk_rows.append(
                        {
                            "n": chunk_count,
                            "type": "usage",
                            "content": json.dumps(u),
                            "tokens": str(u.get("completion_tokens", "")),
                        }
                    )

                choice = (chunk.get("choices") or [None])[0]
                if choice:
                    fr = choice.get("finish_reason") or ""
                    if fr:
                        finish_reason = fr

                    delta = choice.get("delta", {})
                    result = tp.feed(
                        content=delta.get("content", ""),
                        reasoning_content=delta.get("reasoning_content", ""),
                    )

                    if result.thinking or result.content:
                        if t_first_token is None:
                            t_first_token = time.monotonic()

                    if result.thinking:
                        thinking_text += result.thinking
                        chunk_rows.append(
                            {
                                "n": chunk_count,
                                "type": "think",
                                "content": repr(result.thinking),
                                "finish": fr,
                            }
                        )
                        if thinking_guard:
                            gr: FeedResult = thinking_guard.feed(result.thinking)
                            if gr.should_stop:
                                guard_status = f"STOPPED ({gr.reason.name if gr.reason is not None else 'UNKNOWN'})"
                                finish_reason = "repetition_guard"
                                chunk_rows.append(
                                    {
                                        "n": chunk_count,
                                        "type": "guard",
                                        "content": f"thinking: {gr.detail}",
                                        "finish": "guard",
                                    }
                                )
                                if gr.diagnostics:
                                    chunk_rows.append(
                                        {
                                            "n": chunk_count,
                                            "type": "diag",
                                            "content": json.dumps(gr.diagnostics, ensure_ascii=False, default=str),
                                            "finish": "",
                                        }
                                    )
                                live.update(make_display())
                                break

                    if result.content:
                        content_text += result.content
                        chunk_rows.append(
                            {
                                "n": chunk_count,
                                "type": "content",
                                "content": repr(result.content),
                                "finish": fr,
                            }
                        )
                        if content_guard:
                            gr = content_guard.feed(result.content)
                            if gr.should_stop:
                                guard_status = f"STOPPED ({gr.reason.name if gr.reason is not None else 'UNKNOWN'})"
                                finish_reason = "repetition_guard"
                                chunk_rows.append(
                                    {
                                        "n": chunk_count,
                                        "type": "guard",
                                        "content": f"content: {gr.detail}",
                                        "finish": "guard",
                                    }
                                )
                                if gr.diagnostics:
                                    chunk_rows.append(
                                        {
                                            "n": chunk_count,
                                            "type": "diag",
                                            "content": json.dumps(gr.diagnostics, ensure_ascii=False, default=str),
                                            "finish": "",
                                        }
                                    )
                                live.update(make_display())
                                break

                    # Chunk with only finish_reason, no content
                    if fr and not result.thinking and not result.content and not u:
                        chunk_rows.append(
                            {
                                "n": chunk_count,
                                "type": "finish",
                                "content": "",
                                "finish": fr,
                            }
                        )

                live.update(make_display())

    # Final summary
    elapsed = time.monotonic() - t_start
    console.print(f"\n[bold green]Done[/] in {elapsed:.1f}s | {chunk_count} chunks")
    if usage_info:
        console.print(f"[bold]Usage:[/] {json.dumps(usage_info)}")
    console.print(
        f"[bold]Estimate:[/] estimate_thinking_tokens={tp.thinking_tokens_est}"
        f" estimate_content_tokens={tp.content_tokens_est}"
        f" estimate_total_tokens={tp.total_tokens_est}"
    )
    if guard_status:
        console.print(f"[bold red]Repetition guard:[/] {guard_status}")
        for label, g in [("thinking", thinking_guard), ("content", content_guard)]:
            if g and cast(int, g.get_stats()["worst_ngram_count"]) > 0:
                console.print(f"  [dim]{label}:[/] {g.get_stats()}")


def main() -> None:
    """Parse command-line arguments and run the SGLang SSE stream viewer.

    Reads ``SGLANG_URL`` from the environment, builds the request payload via
    :func:`build_payload`, optionally prompts the user for confirmation, and
    then delegates to :func:`stream_and_display` which handles the live Rich
    rendering loop.

    Exits with status 1 when a required argument or environment variable is
    missing.  Exits with status 0 on a ``KeyboardInterrupt`` at the
    confirmation prompt.
    """
    print_banner(module=Path(__file__).stem)
    parser = argparse.ArgumentParser(
        description="Raw SGLang SSE stream viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sampling args override model profile defaults. Omit to use profile values.",
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Prompt to send")

    # Generation control
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=8192)
    gen.add_argument("--thinking-budget", type=int, default=None, help="Cap thinking tokens (custom logit processor)")
    gen.add_argument("--no-think", action="store_true", help="Disable thinking (enable_thinking=false)")
    gen.add_argument("--no-guard", action="store_true", help="Disable streaming repetition guard")
    gen.add_argument("--model", default=None, help=f"Model ID (default: {_CONFIGURED_MODEL})")

    # Sampling parameters
    smp = parser.add_argument_group("sampling (override profile defaults)")
    smp.add_argument("--temperature", "-t", type=float, default=None)
    smp.add_argument("--top-p", type=float, default=None)
    smp.add_argument("--top-k", type=int, default=None)
    smp.add_argument("--min-p", type=float, default=None)
    smp.add_argument("--presence-penalty", "--pp", type=float, default=None)
    smp.add_argument("--frequency-penalty", "--fp", type=float, default=None)
    smp.add_argument("--repetition-penalty", "--rp", type=float, default=None)

    # Display
    parser.add_argument("--raw", action="store_true", help="Show raw JSON chunks instead of table")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    model_id = args.model or _CONFIGURED_MODEL
    if not model_id:
        print("No model configured. Set --model or sglang_model in Ansible defaults.", file=sys.stderr)
        sys.exit(1)

    sglang_url = os.environ.get("SGLANG_URL", "")
    if not sglang_url:
        print("Set SGLANG_URL environment variable", file=sys.stderr)
        sys.exit(1)

    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    payload = build_payload(
        model_id,
        args.prompt,
        args.max_tokens,
        args.thinking_budget,
        args.no_think,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    # Show payload and confirm
    c = Console()
    display_payload = {k: v for k, v in payload.items() if k not in ("messages", "custom_logit_processor")}
    # Show human-readable processor name instead of hex blob
    if "custom_logit_processor" in payload:
        blob = payload["custom_logit_processor"]
        # Extract class name from the dill-serialized reference
        try:
            parts = bytes.fromhex(json.loads(cast(str, blob))["callable"]).split(b"\x94")
            cls_name = parts[1].lstrip(b"\x8c\x21\x20").decode()
        except Exception:
            cls_name = "(serialized)"
        display_payload["custom_logit_processor"] = cls_name
    display_payload["prompt"] = args.prompt[:120] + ("..." if len(args.prompt) > 120 else "")
    c.print(
        Panel(
            Syntax(json.dumps(display_payload, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title=f"[bold]POST {url}[/]",
            border_style="cyan",
        )
    )
    if not args.yes:
        try:
            input("[Enter to send, Ctrl+C to abort] ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    stream_and_display(url, payload, raw_json=args.raw, guard_enabled=not args.no_guard)


if __name__ == "__main__":
    main()
