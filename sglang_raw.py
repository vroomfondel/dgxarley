#!/usr/bin/env python3
"""Raw SGLang SSE stream viewer with dual-panel Rich display.

Upper panel: interpreted output (thinking + content) with live token stats.
Lower panel: raw SSE JSON chunks with syntax highlighting.

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

import requests

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from openwebui_integration_test import _dgx_defaults, load_sampling_presets, pick_default_preset
from sglang_integration_test import _THINKING_BUDGET_PROCESSORS

_CONFIGURED_MODEL: str = _dgx_defaults.get("sglang_model", "")
_MODEL_PROFILES: dict = _dgx_defaults.get("sglang_model_profiles", {})

DEFAULT_PROMPT = (
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
) -> dict:
    payload: dict = {
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
        payload.update(extra)

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
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    if thinking_budget is not None:
        reasoning_parser = _MODEL_PROFILES.get(model_id, {}).get("reasoning_parser", "")
        processor = _THINKING_BUDGET_PROCESSORS.get(reasoning_parser)
        if processor:
            payload["custom_logit_processor"] = processor
            payload["custom_params"] = {"thinking_budget": thinking_budget}

    return payload


def stream_and_display(url: str, payload: dict, raw_json: bool = False) -> None:
    console = Console()
    thinking_text = ""
    content_text = ""
    raw_chunks: deque[str] = deque(maxlen=30)  # keep last N chunks for display
    chunk_rows: deque[dict] = deque(maxlen=50)  # structured rows for table mode
    t_start = time.monotonic()
    t_first_token: float | None = None
    thinking_tokens_est = 0
    content_tokens_est = 0
    finish_reason = ""
    usage_info: dict = {}
    chunk_count = 0

    def _status_line() -> str:
        elapsed = time.monotonic() - t_start
        ttft_str = f"{t_first_token - t_start:.2f}s" if t_first_token else "-"
        total_est = thinking_tokens_est + content_tokens_est
        if usage_info:
            prompt_tok = usage_info.get("prompt_tokens", 0)
            completion_tok = usage_info.get("completion_tokens", 0)
            tok_info = f"prompt={prompt_tok} completion={completion_tok}"
        else:
            tok_info = f"T~{thinking_tokens_est} C~{content_tokens_est} total~{total_est}"
        tps = total_est / elapsed if elapsed > 0 and total_est > 0 else 0
        finish_str = f" | finish={finish_reason}" if finish_reason else ""
        return f"{elapsed:.1f}s | TTFT={ttft_str} | {tok_info} | {tps:.1f} tok/s{finish_str}"

    def _interpreted_panel() -> Panel:
        # Calculate available lines: upper panel gets 3/5 of terminal height,
        # minus 2 for panel border/title
        avail_lines = max(5, (console.height * 3 // 5) - 2)
        panel_width = console.width - 4  # border + padding

        def _tail_lines(text: str, max_lines: int) -> tuple[str, bool]:
            """Return last max_lines worth of text, and whether it was truncated."""
            lines = text.split("\n")
            if len(lines) <= max_lines:
                return text, False
            return "\n".join(lines[-max_lines:]), True

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
        return Panel(
            interpreted,
            title=f"[bold]{_status_line()}[/]",
            border_style="green" if finish_reason else "yellow",
        )

    def _chunk_table() -> Table:
        from rich.table import Table as RichTable
        tbl = RichTable(
            show_lines=False, expand=True, pad_edge=False,
            title="[bold]SSE chunks[/]", border_style="blue",
        )
        tbl.add_column("#", justify="right", style="dim", width=5)
        tbl.add_column("Type", width=8)
        tbl.add_column("Content", ratio=1, overflow="ellipsis", no_wrap=True)
        tbl.add_column("Finish", width=8)
        tbl.add_column("Tokens", justify="right", width=8)
        for row in chunk_rows:
            n = str(row["n"])
            typ = row["type"]
            content = row["content"]
            fin = row.get("finish", "")
            tokens = row.get("tokens", "")
            if typ == "think":
                style = "cyan"
            elif typ == "content":
                style = "white"
            elif typ == "usage":
                style = "green"
            elif typ == "done":
                style = "bold green"
            else:
                style = "dim"
            tbl.add_row(n, typ, content, fin, tokens, style=style)
        return tbl

    def make_display() -> Layout:
        layout = Layout()
        if raw_json:
            raw_display = "\n".join(raw_chunks)
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
                    chunk_rows.append({
                        "n": chunk_count, "type": "usage", "content": json.dumps(u),
                        "tokens": str(u.get("completion_tokens", "")),
                    })

                choice = (chunk.get("choices") or [None])[0]
                if choice:
                    fr = choice.get("finish_reason") or ""
                    if fr:
                        finish_reason = fr

                    delta = choice.get("delta", {})
                    reasoning = delta.get("reasoning_content", "")
                    content = delta.get("content", "")

                    if reasoning:
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        thinking_text += reasoning
                        thinking_tokens_est = len(thinking_text) // 4
                        chunk_rows.append({
                            "n": chunk_count, "type": "think",
                            "content": repr(reasoning), "finish": fr,
                        })

                    if content:
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        content_text += content
                        content_tokens_est = len(content_text) // 4
                        chunk_rows.append({
                            "n": chunk_count, "type": "content",
                            "content": repr(content), "finish": fr,
                        })

                    # Chunk with only finish_reason, no content
                    if fr and not reasoning and not content and not u:
                        chunk_rows.append({
                            "n": chunk_count, "type": "finish",
                            "content": "", "finish": fr,
                        })

                live.update(make_display())

    # Final summary
    elapsed = time.monotonic() - t_start
    console.print(f"\n[bold green]Done[/] in {elapsed:.1f}s | {chunk_count} chunks")
    if usage_info:
        console.print(f"[bold]Usage:[/] {json.dumps(usage_info)}")


def main():
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
        model_id, args.prompt, args.max_tokens, args.thinking_budget, args.no_think,
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
            parts = bytes.fromhex(json.loads(blob)["callable"]).split(b"\x94")
            cls_name = parts[1].lstrip(b"\x8c\x21\x20").decode()
        except Exception:
            cls_name = "(serialized)"
        display_payload["custom_logit_processor"] = cls_name
    display_payload["prompt"] = args.prompt[:120] + ("..." if len(args.prompt) > 120 else "")
    c.print(Panel(
        Syntax(json.dumps(display_payload, indent=2, ensure_ascii=False), "json", theme="monokai"),
        title=f"[bold]POST {url}[/]",
        border_style="cyan",
    ))
    if not args.yes:
        try:
            input("[Enter to send, Ctrl+C to abort] ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    stream_and_display(url, payload, raw_json=args.raw)


if __name__ == "__main__":
    main()
