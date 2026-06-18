#!/usr/bin/env python3
# sglang_bench.py — thin launcher around `python -m sglang.bench_serving` that
# pins the auto-downloaded ShareGPT dataset to a PERSISTENT cache dir and adds a
# concurrency-sweep mode.
#
# Why this exists
# ---------------
# sglang.bench_serving hard-codes its dataset download to /tmp and the download
# target is NOT reliably steerable via --dataset-path:
#
#   * download_and_cache_file(url, filename=None): when filename is None it uses
#     os.path.join("/tmp", basename) — the literal string "/tmp", so TMPDIR does
#     NOT redirect it. /tmp is wiped on reboot → re-download every cold start.
#   * --dataset-name random calls download_and_cache_file(SHAREGPT_URL) WITHOUT a
#     filename, so it downloads to /tmp and ignores --dataset-path for the download
#     target (it only reads --dataset-path if that file already exists + is valid).
#   * --dataset-name sharegpt only auto-downloads when --dataset-path == ""; a
#     non-existent path is opened directly → FileNotFoundError.
#
# So the robust fix is to PRE-SEED the ShareGPT file at a persistent path and then
# hand bench_serving --dataset-path <that file>. Both `random` and `sharegpt` then
# find valid JSON and skip the /tmp download entirely.
#
# We reuse SGLang's own SHAREGPT_URL + download_and_cache_file so the URL and the
# JSON-validity check stay in sync with whatever sglang version is installed.
#
# Fixed-length runs
# -----------------
# For `--dataset-name random`, bench_serving's default --random-range-ratio is 0.0,
# which spreads input/output lengths UNIFORMLY from 1..max (so "1024/512" actually
# measures ~512/256 avg with wide variance). This wrapper defaults range-ratio to 1.0
# (lengths pinned exactly) so runs are reproducible/comparable. Pass it explicitly to
# override.
#
# Sweep mode
# ----------
# --sweep "1,4,8,16,32" runs one benchmark per concurrency level (each as its own
# subprocess so bench_serving global state can't leak between runs), streams each run's
# output live, then prints a compact summary table. num-prompts scales as
# concurrency * --sweep-prompts-per-conc unless you pass a fixed --num-prompts.
# The table reports BOTH aggregate output tok/s AND per-stream tok/s (1000/median-ITL,
# the per-user decode speed) since the aggregate number alone hides single-stream UX.
#
# Usage
# -----
# Installed as the `sglang-bench` console script (or run via
# `python -m dgxarley.integration.sglang_bench`):
#
#   sglang-bench --backend sglang-oai \
#       --base-url https://sglang.dgx.elasticc.io \
#       --dataset-name random --random-input-len 1024 --random-output-len 512 \
#       --num-prompts 200 --max-concurrency 32
#
#   sglang-bench --sweep 1,4,8,16,32,64 --backend sglang-oai \
#       --base-url https://sglang.dgx.elasticc.io \
#       --dataset-name random --random-input-len 1024 --random-output-len 512
#
# All UNKNOWN flags pass straight through to sglang.bench_serving. Wrapper-owned flags:
# --cache-dir, --sweep, --sweep-prompts-per-conc. If you pass your own --dataset-path
# it is respected and no pre-seed happens (use that for e.g. --dataset-name mooncake).
#
# Requires the `sglang-bench` extra in the active venv:  pip install -e '.[sglang-bench]'

import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

SHAREGPT_BASENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Metric name -> regex over a bench_serving "Serving Benchmark Result" block.
SUMMARY_PATTERNS = {
    "n_ok": r"Successful requests:\s+(\d+)",
    "req_s": r"Request throughput \(req/s\):\s+([\d.]+)",
    "out_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "tot_tok_s": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "med_ttft": r"Median TTFT \(ms\):\s+([\d.]+)",
    "med_itl": r"Median ITL \(ms\):\s+([\d.]+)",
    "p99_itl": r"P99 ITL \(ms\):\s+([\d.]+)",
}

# allow_extra_args + ignore_unknown_options → every flag we don't define ourselves
# is collected verbatim into ctx.args and handed through to bench_serving.
app = typer.Typer(add_completion=False)


def _has_flag(passthrough: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in passthrough)


def _flag_value(passthrough: list[str], name: str) -> Optional[str]:
    for i, a in enumerate(passthrough):
        if a == name and i + 1 < len(passthrough):
            return passthrough[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _strip_flags(passthrough: list[str], names: set[str]) -> list[str]:
    """Drop the given flags (and their values) from a passthrough list."""
    out: list[str] = []
    i = 0
    while i < len(passthrough):
        a = passthrough[i]
        if a in names:
            i += 2  # flag + value
            continue
        if any(a.startswith(nm + "=") for nm in names):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _seed_dataset(cache_dir: Path, passthrough: list[str]) -> list[str]:
    """Pre-seed ShareGPT to the persistent cache; inject --dataset-path unless the
    user already supplied one."""
    if _has_flag(passthrough, "--dataset-path"):
        return passthrough
    from sglang.bench_serving import SHAREGPT_URL, download_and_cache_file

    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / SHAREGPT_BASENAME
    # download_and_cache_file returns immediately if target is already valid JSON.
    download_and_cache_file(SHAREGPT_URL, str(target))
    return ["--dataset-path", str(target), *passthrough]


def _default_range_ratio(passthrough: list[str]) -> list[str]:
    """Pin random lengths to exact values (range-ratio 1.0) unless overridden."""
    name = _flag_value(passthrough, "--dataset-name") or "sharegpt"
    if name == "random" and not _has_flag(passthrough, "--random-range-ratio"):
        typer.echo(
            "[sglang_bench] defaulting --random-range-ratio 1.0 (fixed lengths); pass it to override",
            err=True,
        )
        return [*passthrough, "--random-range-ratio", "1.0"]
    return passthrough


def _run_single(passthrough: list[str]) -> None:
    """Run bench_serving in-process so signals / exit codes propagate cleanly."""
    sys.argv = ["sglang.bench_serving", *passthrough]
    runpy.run_module("sglang.bench_serving", run_name="__main__")


def _run_one_capture(passthrough: list[str]) -> dict[str, Optional[float]]:
    """Run bench_serving as a subprocess, tee its stdout live, parse the summary."""
    # Force a C locale so the summary block stays ASCII-formatted (decimal points,
    # untranslated metric labels) and SUMMARY_PATTERNS keeps matching regardless of
    # the caller's LANG/LC_*.
    env = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "sglang.bench_serving", *passthrough],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    text = "".join(captured)
    row: dict[str, Optional[float]] = {}
    for key, pat in SUMMARY_PATTERNS.items():
        m = re.search(pat, text)
        row[key] = float(m.group(1)) if m else None
    return row


def _print_sweep_table(rows: list[dict[str, Optional[float]]]) -> None:
    def fmt(x: Optional[float], nd: int = 2) -> str:
        return f"{x:.{nd}f}" if x is not None else "-"

    cols = (
        f"{'conc':>5} {'n_ok':>5} {'req/s':>7} {'out tok/s':>10} "
        f"{'tot tok/s':>10} {'tok/s/req':>10} {'med TTFT':>9} {'med ITL':>8} {'p99 ITL':>8}"
    )
    print("\n================ Sweep summary ================")
    print("out tok/s = aggregate (total/wall);  tok/s/req = per-stream decode (1000/med-ITL)")
    print(cols)
    print("-" * len(cols))
    for r in rows:
        med_itl = r.get("med_itl")
        per_stream = 1000.0 / med_itl if med_itl else None
        n_ok = r.get("n_ok")
        print(
            f"{int(r['conc']):>5} "  # type: ignore[arg-type]
            f"{int(n_ok) if n_ok is not None else '-':>5} "
            f"{fmt(r.get('req_s')):>7} {fmt(r.get('out_tok_s')):>10} "
            f"{fmt(r.get('tot_tok_s')):>10} {fmt(per_stream):>10} "
            f"{fmt(r.get('med_ttft'), 0):>9} {fmt(med_itl, 0):>8} {fmt(r.get('p99_itl'), 0):>8}"
        )
    print("=" * len(cols))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def bench(
    ctx: typer.Context,
    cache_dir: Path = typer.Option(
        Path.home() / ".cache" / "sglang" / "benchmark",
        "--cache-dir",
        envvar="SGLANG_BENCH_CACHE_DIR",
        help="Persistent dir for the pre-seeded ShareGPT dataset.",
    ),
    sweep: Optional[str] = typer.Option(
        None,
        "--sweep",
        help="Comma-separated concurrency levels, e.g. '1,4,8,16,32'. Runs one benchmark per level.",
    ),
    sweep_prompts_per_conc: int = typer.Option(
        8,
        "--sweep-prompts-per-conc",
        help="num-prompts = concurrency * this (unless --num-prompts is set explicitly).",
    ),
) -> None:
    """Run sglang.bench_serving with a persistently-cached ShareGPT dataset.

    All other options are passed straight through to sglang.bench_serving.
    """
    passthrough = list(ctx.args)

    try:
        import sglang.bench_serving  # noqa: F401
    except ImportError as exc:  # sglang not importable → nothing to wrap
        typer.echo(f"error: sglang not installed in this interpreter: {exc}", err=True)
        raise typer.Exit(code=1)

    passthrough = _seed_dataset(cache_dir, passthrough)
    passthrough = _default_range_ratio(passthrough)

    if not sweep:
        _run_single(passthrough)
        return

    levels = [int(x) for x in sweep.replace(" ", "").split(",") if x]
    fixed_num = _flag_value(passthrough, "--num-prompts")
    base = _strip_flags(passthrough, {"--max-concurrency", "--num-prompts"})

    rows: list[dict[str, Optional[float]]] = []
    for conc in levels:
        num = fixed_num or str(max(conc * sweep_prompts_per_conc, sweep_prompts_per_conc))
        args = [*base, "--max-concurrency", str(conc), "--num-prompts", num]
        typer.echo(f"\n===== sweep: concurrency={conc}, num-prompts={num} =====", err=True)
        row = _run_one_capture(args)
        row["conc"] = float(conc)
        rows.append(row)

    _print_sweep_table(rows)


def main() -> None:
    """Console-script entry point (see [project.scripts] `sglang-bench`)."""
    app()


if __name__ == "__main__":
    main()
