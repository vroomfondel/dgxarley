#!/usr/bin/env python3
"""Non-interactive SGLang parameter test harness for the Ralph tuning loop.

Sends a streaming chat-completions request, monitors for repetition in
real-time via RepetitionGuard, runs post-hoc analysis via detect_repetition(),
and outputs a single JSON result to stdout.

Usage::

    python ops/qwen235b-tuning/test_params.py \\
      --prompt analytical --max-tokens 4096 \\
      -t 0.6 --top-p 0.95 --top-k 40 --min-p 0.1 \\
      --presence-penalty 1.0 --frequency-penalty 0.0 --repetition-penalty 0.0

    # Run all 3 prompts:
    python ops/qwen235b-tuning/test_params.py --prompt all --max-tokens 4096

    # Save full output text:
    python ops/qwen235b-tuning/test_params.py --prompt creative --save-output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from dgxarley.integration.sglang_raw import build_payload, _CONFIGURED_MODEL
from dgxarley.integration.streaming_repetition_guard import RepetitionGuard, GuardConfig
from dgxarley.integration.repetition_detector import detect_repetition

# ──────────────────────────────────────────────────────────────
# Test Prompts
# ──────────────────────────────────────────────────────────────

TEST_PROMPTS: dict[str, str] = {
    "analytical": (
        "You are a principal software architect at a large technology company. "
        "A VP has asked you to write a comprehensive analysis comparing "
        "microservices architecture vs monolithic architecture for their next "
        "platform rewrite. Cover: (1) operational complexity and deployment, "
        "(2) team autonomy and organizational scaling, (3) data consistency "
        "and transaction management, (4) performance characteristics and "
        "latency, (5) testing strategies, (6) migration paths from one to "
        "the other. For each dimension, discuss concrete trade-offs with "
        "real-world examples. Conclude with a decision framework that helps "
        "teams choose based on their specific constraints."
    ),
    "creative": (
        "Write a short story (2000-3000 words) set in a cyberpunk city where "
        "an underground courier discovers that the encrypted packages they've "
        "been delivering contain fragments of a banned AI's consciousness. "
        "The story should have: a vivid opening scene, at least two distinct "
        "characters with dialogue, a moral dilemma, and an ambiguous ending. "
        "Use sensory details — neon reflections on wet pavement, the hum of "
        "electromagnetic fields, the taste of recycled air."
    ),
    "technical": (
        "Write an in-depth technical deep dive on the complete lifecycle of "
        "an HTTP request, from the moment a user types a URL in a browser to "
        "the final rendered page. Cover: DNS resolution (recursive vs "
        "iterative, caching layers, DNSSEC), TCP handshake (SYN/SYN-ACK/ACK, "
        "window scaling, congestion control algorithms), TLS 1.3 negotiation "
        "(key exchange, 0-RTT, certificate validation, OCSP stapling), "
        "HTTP/2 framing (HPACK, stream multiplexing, server push, flow "
        "control), server-side processing (load balancing, reverse proxy, "
        "application server, database queries, caching strategies), and "
        "browser rendering (DOM construction, CSSOM, render tree, layout, "
        "paint, compositing). Include packet-level details where relevant."
    ),
}


# ──────────────────────────────────────────────────────────────
# Streaming + Guard
# ──────────────────────────────────────────────────────────────

def run_test(
    url: str,
    payload: dict,
    prompt_name: str,
    save_output: bool = False,
    results_dir: Path | None = None,
) -> dict:
    """Send a streaming request and return structured results.

    Args:
        url: Full /v1/chat/completions endpoint URL.
        payload: Request body dict (with stream=True already set).
        prompt_name: Label for the prompt (analytical/creative/technical).
        save_output: Whether to save full output text to a file.
        results_dir: Directory for saved output files.

    Returns:
        A dict matching the output schema documented in the plan.
    """
    thinking_text = ""
    content_text = ""
    t_start = time.monotonic()
    t_first_token: float | None = None
    finish_reason = ""
    usage_info: dict = {}
    guard_triggered = False
    guard_reason: str | None = None

    # Use default GuardConfig — tuned for outputs up to ~16k tokens
    thinking_guard = RepetitionGuard()
    content_guard = RepetitionGuard()

    try:
        with requests.post(url, json=payload, stream=True, timeout=1800) as resp:
            if resp.status_code != 200:
                return {
                    "prompt": prompt_name,
                    "params": _extract_params(payload),
                    "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                    "elapsed_s": round(time.monotonic() - t_start, 1),
                }

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Usage (typically in final chunk)
                u = chunk.get("usage")
                if u:
                    usage_info = u

                choice = (chunk.get("choices") or [None])[0]
                if not choice:
                    continue

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
                    gr = thinking_guard.feed(reasoning)
                    if gr.should_stop:
                        guard_triggered = True
                        guard_reason = f"{gr.reason.name}: thinking: {gr.detail}"
                        finish_reason = "repetition_guard"
                        resp.close()
                        break

                if content:
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                    content_text += content
                    gr = content_guard.feed(content)
                    if gr.should_stop:
                        guard_triggered = True
                        guard_reason = f"{gr.reason.name}: content: {gr.detail}"
                        finish_reason = "repetition_guard"
                        resp.close()
                        break

    except requests.exceptions.ConnectionError as e:
        return {
            "prompt": prompt_name,
            "params": _extract_params(payload),
            "error": f"Connection error: {e}",
            "elapsed_s": round(time.monotonic() - t_start, 1),
        }
    except requests.exceptions.Timeout:
        return {
            "prompt": prompt_name,
            "params": _extract_params(payload),
            "error": "Request timed out (1800s)",
            "elapsed_s": round(time.monotonic() - t_start, 1),
        }

    elapsed = time.monotonic() - t_start

    # Post-hoc repetition analysis on content text
    rep_report = None
    if content_text and len(content_text) > 50:
        rep_report = detect_repetition(content_text)

    # Estimate tokens
    completion_tokens = usage_info.get("completion_tokens", 0)
    if not completion_tokens:
        # Rough estimate: ~4 chars per token
        completion_tokens = (len(thinking_text) + len(content_text)) // 4
    reasoning_tokens = usage_info.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if not reasoning_tokens:
        reasoning_tokens = len(thinking_text) // 4

    result: dict = {
        "prompt": prompt_name,
        "params": _extract_params(payload),
        "elapsed_s": round(elapsed, 1),
        "completion_tokens": completion_tokens,
        "reasoning_tokens_est": reasoning_tokens,
        "finish_reason": finish_reason or "unknown",
        "guard_triggered": guard_triggered,
        "guard_reason": guard_reason,
        "content_chars": len(content_text),
    }

    if rep_report:
        result["repetition"] = {
            "overall": round(rep_report.overall_score, 3),
            "severity": rep_report.severity,
            "ngram": round(rep_report.ngram_score, 3),
            "sentence": round(rep_report.sentence_score, 3),
            "loop": round(rep_report.loop_score, 3),
        }
        result["summary"] = rep_report.summary()
    else:
        result["repetition"] = {
            "overall": 0.0,
            "severity": "none",
            "ngram": 0.0,
            "sentence": 0.0,
            "loop": 0.0,
        }
        result["summary"] = "[NONE] score=0.00 — insufficient content for analysis"

    # Save output text if requested
    if save_output and results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_file = results_dir / f"{ts}-{prompt_name}.txt"
        with open(out_file, "w") as f:
            if thinking_text:
                f.write("<thinking>\n")
                f.write(thinking_text)
                f.write("\n</thinking>\n\n")
            f.write(content_text)
        result["output_file"] = str(out_file)

    return result


def _extract_params(payload: dict) -> dict:
    """Extract sampling parameters from the payload for the result."""
    keys = [
        "max_tokens", "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "frequency_penalty", "repetition_penalty",
    ]
    return {k: payload[k] for k in keys if k in payload}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-interactive SGLang parameter test harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Prompt selection
    parser.add_argument(
        "--prompt", "-p",
        choices=["analytical", "creative", "technical", "all"],
        default="analytical",
        help="Test prompt to use (default: analytical)",
    )

    # Generation control
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=4096)
    gen.add_argument("--thinking-budget", type=int, default=1024)
    gen.add_argument("--no-think", action="store_true")
    gen.add_argument("--model", default=None)

    # Sampling parameters — explicit defaults match the current baseline so that
    # every request has ALL params set, regardless of server-side generation_config.
    # This is critical for controlled sweeps: omitting a param would silently
    # fall back to the server default, making comparisons unreliable.
    smp = parser.add_argument_group("sampling (baseline defaults, always sent)")
    smp.add_argument("--temperature", "-t", type=float, default=0.6)
    smp.add_argument("--top-p", type=float, default=0.95)
    smp.add_argument("--top-k", type=int, default=40)
    smp.add_argument("--min-p", type=float, default=0.1)
    smp.add_argument("--presence-penalty", "--pp", type=float, default=1.0)
    smp.add_argument("--frequency-penalty", "--fp", type=float, default=0.0)
    smp.add_argument("--repetition-penalty", "--rp", type=float, default=0.0)

    # Output control
    parser.add_argument("--save-output", action="store_true", help="Save full output text to results/")

    args = parser.parse_args()

    model_id = args.model or _CONFIGURED_MODEL
    if not model_id:
        print(json.dumps({"error": "No model configured"}), file=sys.stderr)
        sys.exit(1)

    sglang_url = os.environ.get("SGLANG_URL", "")
    if not sglang_url:
        print(json.dumps({"error": "SGLANG_URL not set"}), file=sys.stderr)
        sys.exit(1)

    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    # Determine which prompts to run
    if args.prompt == "all":
        prompt_names = list(TEST_PROMPTS.keys())
    else:
        prompt_names = [args.prompt]

    results = []
    for prompt_name in prompt_names:
        prompt_text = TEST_PROMPTS[prompt_name]

        payload = build_payload(
            model_id, prompt_text, args.max_tokens, args.thinking_budget, args.no_think,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            repetition_penalty=args.repetition_penalty,
        )

        # Print progress to stderr (stdout is reserved for JSON)
        print(f"[{prompt_name}] Testing with max_tokens={args.max_tokens} ...", file=sys.stderr)
        result = run_test(url, payload, prompt_name, args.save_output, results_dir)
        results.append(result)

        status = "GUARD" if result.get("guard_triggered") else result.get("finish_reason", "?")
        rep = result.get("repetition", {})
        print(
            f"[{prompt_name}] {status} | "
            f"{result.get('elapsed_s', 0)}s | "
            f"rep={rep.get('overall', 0):.3f} ({rep.get('severity', '?')})",
            file=sys.stderr,
        )

    # Output JSON to stdout
    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
