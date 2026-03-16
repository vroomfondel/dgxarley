import os, sys, threading
from huggingface_hub import snapshot_download, HfApi
import huggingface_hub

huggingface_hub.logging.set_verbosity_info()

cache_dir = "/root/.cache/huggingface/hub"
_stop_monitor = threading.Event()
_state = {"total": 0, "model_path": "", "active": True}


def _cache_size(path):
    try:
        return sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(path)
            for f in fns
        )
    except Exception:
        return 0


def _monitor(path, interval=10):
    while not _stop_monitor.is_set():
        _stop_monitor.wait(interval)
        if not _state["active"]:
            continue
        model_path = _state.get("model_path", "")
        if not model_path:
            continue
        downloaded = _cache_size(os.path.join(model_path, "blobs"))
        total_size = _state["total"]
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(
                f"  [progress] {downloaded / 1e9:.1f} / {total_size / 1e9:.1f} GB"
                f" ({pct:.0f}%)",
                flush=True,
            )
        else:
            print(
                f"  [progress] {downloaded / 1e9:.1f} GB downloaded",
                flush=True,
            )


def _stream(pipe, prefix=""):
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
        total_size = sum(s.size or 0 for s in info.siblings)
        n_files = len(info.siblings)
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
                proc = subprocess.Popen([
                    "rsync", "-ah", "--info=progress2", "--inplace",
                    "-e", "ssh " + " ".join(ssh_opts),
                    src, dst
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                t_out = threading.Thread(target=_stream, args=(proc.stdout,), daemon=True)
                t_err = threading.Thread(target=_stream, args=(proc.stderr, "[rsync stderr] "), daemon=True)
                t_out.start(); t_err.start()
                t_out.join(); t_err.join()
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
