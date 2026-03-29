"""Download HuggingFace models and optionally rsync them to remote hosts.

Models to download are read from the ``HF_PRELOAD_MODELS`` environment variable
(comma-separated repo IDs) or from ``SGLANG_MODEL`` (single repo ID).  After a
successful download the model directory can be synchronised to one or more remote
hosts via rsync over SSH by setting ``RSYNC_TARGETS`` (space-separated hostnames)
and ``HF_CACHE_HOST_PATH`` (destination base path on the remote hosts).

Environment variables:
    HF_PRELOAD_MODELS: Comma-separated list of HuggingFace repo IDs to download.
    SGLANG_MODEL: Single HuggingFace repo ID (fallback when HF_PRELOAD_MODELS is
        unset or empty).
    RSYNC_TARGETS: Space-separated list of hostnames to rsync each model to after
        download.
    HF_CACHE_HOST_PATH: Destination base path on remote rsync targets (e.g.
        ``/data/huggingface``).
"""

import os, sys, threading
from typing import IO
from huggingface_hub import snapshot_download, HfApi
import huggingface_hub

huggingface_hub.logging.set_verbosity_info()  # type: ignore[no-untyped-call]

cache_dir = "/root/.cache/huggingface/hub"
_stop_monitor = threading.Event()
_state: dict[str, int | str | bool] = {"total": 0, "model_path": "", "active": True}


def _cache_size(path: str) -> int:
    """Return the total byte size of all files under *path*.

    Args:
        path: Root directory to measure.

    Returns:
        Sum of ``os.path.getsize`` for every regular file found via
        ``os.walk``.  Returns ``0`` if *path* does not exist or any
        OS-level error occurs.
    """
    try:
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns)
    except Exception:
        return 0


def _monitor(path: str, interval: int = 10) -> None:
    """Periodically log download progress to stdout.

    Intended to run in a background daemon thread.  Sleeps for *interval*
    seconds between iterations and skips reporting when the global
    ``_state["active"]`` flag is ``False`` or when no ``_state["model_path"]``
    has been set yet.

    Args:
        path: Root HuggingFace hub cache directory (used only as the base for
            constructing the blobs sub-path; the actual sub-path is taken from
            ``_state["model_path"]`` at each iteration).
        interval: Number of seconds to wait between progress reports.
    """
    while not _stop_monitor.is_set():
        _stop_monitor.wait(interval)
        if not _state["active"]:
            continue
        model_path = str(_state.get("model_path", ""))
        if not model_path:
            continue
        downloaded = _cache_size(os.path.join(model_path, "blobs"))
        total_size = int(_state["total"])
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(
                f"  [progress] {downloaded / 1e9:.1f} / {total_size / 1e9:.1f} GB" f" ({pct:.0f}%)",
                flush=True,
            )
        else:
            print(
                f"  [progress] {downloaded / 1e9:.1f} GB downloaded",
                flush=True,
            )


def _stream(pipe: "IO[bytes] | None", prefix: str = "") -> None:
    """Stream bytes from *pipe* to stdout, printing one line at a time.

    Reads the pipe one byte at a time and flushes a decoded line to stdout
    whenever a newline or carriage-return character is encountered.  Any
    remaining buffered bytes are flushed when the pipe reaches EOF.  Designed
    to be run in a dedicated daemon thread so that stdout and stderr of a
    subprocess can be streamed concurrently without blocking.

    Args:
        pipe: A readable binary I/O object (e.g. ``subprocess.Popen.stdout``
            or ``.stderr``).  If ``None`` the function returns immediately.
        prefix: Optional string prepended to every printed line, useful for
            distinguishing stderr output (e.g. ``"[rsync stderr] "``).
    """
    if pipe is None:
        return
    buf = b""
    while True:
        ch = pipe.read(1)
        if not ch:
            if buf:
                print(f"{prefix}{buf.decode().rstrip()}", flush=True)
            break
        if ch in (b"\n", b"\r"):
            if buf:
                print(f"{prefix}{buf.decode().rstrip()}", flush=True)
                buf = b""
        else:
            buf += ch


# Accept models from HF_PRELOAD_MODELS (comma-separated) or SGLANG_MODEL (single)
raw = os.environ.get("HF_PRELOAD_MODELS", "") or os.environ.get("SGLANG_MODEL", "")
models = [m.strip() for m in raw.split(",") if m.strip()]

if not models:
    print("No models specified, nothing to do.", flush=True)
    sys.exit(0)

rsync_targets = os.environ.get("RSYNC_TARGETS", "").split()
hf_cache_host_path = os.environ.get("HF_CACHE_HOST_PATH", "")

api = HfApi()
failed = []

monitor = threading.Thread(target=_monitor, args=(cache_dir,), daemon=True)
monitor.start()

for model_id in models:
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_id}", flush=True)
    try:
        info = api.model_info(model_id, files_metadata=True)
        siblings = info.siblings or []
        total_size = sum(s.size or 0 for s in siblings)
        n_files = len(siblings)
        print(f"Files: {n_files}, Total size: {total_size / 1e9:.1f} GB", flush=True)
        print(f"Downloading to {cache_dir} ...", flush=True)
        _state["total"] = total_size
        model_dir = "models--" + model_id.replace("/", "--")
        _state["model_path"] = os.path.join(cache_dir, model_dir)
        _state["active"] = True
        snapshot_download(repo_id=model_id, cache_dir=cache_dir)
        print(f"Model ready: {model_id}", flush=True)
        _state["active"] = False
        if rsync_targets:
            import subprocess

            model_dir = "models--" + model_id.replace("/", "--")
            src = f"/root/.cache/huggingface/hub/{model_dir}/"
            ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
            for target in rsync_targets:
                dst = f"root@{target}:{hf_cache_host_path}/hub/{model_dir}/"
                print(f"Syncing {model_dir} to {target}:{hf_cache_host_path}/hub/ ...", flush=True)
                proc = subprocess.Popen(
                    ["rsync", "-ah", "--info=progress2", "--inplace", "-e", "ssh " + " ".join(ssh_opts), src, dst],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                t_out = threading.Thread(target=_stream, args=(proc.stdout,), daemon=True)
                t_err = threading.Thread(target=_stream, args=(proc.stderr, "[rsync stderr] "), daemon=True)
                t_out.start()
                t_err.start()
                t_out.join()
                t_err.join()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"rsync to {target} failed with exit code {rc}")
                print(f"Sync to {target} complete.", flush=True)
    except Exception as e:
        print(f"FAILED: {model_id}: {e}", flush=True)
        failed.append(model_id)

_stop_monitor.set()
print(f"\n{'='*60}", flush=True)
if failed:
    print(f"Failed models: {failed}", flush=True)
    sys.exit(1)
print("All models ready.", flush=True)
