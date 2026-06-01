"""DSV4 unified-memory load probe (GB10 / SM121).

Activated only when env DSV4_MEMPROBE=1 (sglang_launch.sh sets it from
SGLANG_MEMPROBE). Goal: on GB10 there is ONE unified pool — no host<->device
relocation — so the ~2x footprint AFTER all weight shards are read is a real
in-allocator duplication caused by some specific post-load action. This probe
pinpoints which one.

It logs torch.cuda.memory_allocated()/reserved() + process RSS + node swap:
  - bracketing ModelRunner.load_model (BEGIN = start of load, END = all layers
    read + post-processed) and init_memory_pool / cuda-graph capture,
  - the cuda-alloc DELTA of every Fp8(MoE|Linear)Method.process_weights_after_loading
    call (the prime suspect — per-layer weight repack/requant/contiguous copies),
  - plus a 0.2s background ticker that prints whenever cuda_alloc or RSS jumps
    >0.5 GB, so any unwrapped action that doubles memory is still caught.

All output goes to stderr -> pod log -> Loki. Grep:  |~ "\\[memprobe"
No-op (zero overhead, not even imported side effects) unless DSV4_MEMPROBE=1.
Wired via .pth in sglang_launch.sh; wraps targets lazily as their modules import
(works in the spawned scheduler/TP-worker processes, not just the launcher).
"""

import os
import sys
import threading
import time
from typing import Any

if os.environ.get("DSV4_MEMPROBE", "0") in ("1", "true", "yes"):
    import importlib.abc
    import importlib.util

    # Full snapshot to find the swap-out TRIGGER: which memory category grows
    # while anon swaps out. torch's allocator view (cuda_alloc) is NOT enough on
    # unified memory — NCCL/driver/other allocations show only in the device
    # mem_get_info(used) and in the /proc/meminfo categories. /proc/vmstat gives
    # the real swap-out + reclaim counters (diff across ticks for rates). Values
    # in GB except *_ctr (raw cumulative page counters).
    def _snap() -> "dict[str, float]":
        d: "dict[str, float]" = {}
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            d["cuda_alloc"] = torch.cuda.memory_allocated() / 1e9
            d["cuda_resv"] = torch.cuda.memory_reserved() / 1e9
            d["dev_used"] = (total - free) / 1e9  # whole-device used (incl NCCL/driver/other)
            d["dev_free"] = free / 1e9
        except Exception:
            d.update(cuda_alloc=-1.0, cuda_resv=-1.0, dev_used=-1.0, dev_free=-1.0)
        try:
            mi = {}
            for ln in open("/proc/meminfo"):
                k, v = ln.split(":", 1)
                mi[k] = int(v.split()[0])  # kB

            def gb(key: str) -> float:
                return mi.get(key, 0) / 1024.0 / 1024.0

            d.update(
                memfree=gb("MemFree"),
                cached=gb("Cached"),
                anon=gb("AnonPages"),
                mapped=gb("Mapped"),
                shmem=gb("Shmem"),
                srecl=gb("SReclaimable"),
                sunrecl=gb("SUnreclaim"),
                pagetbl=gb("PageTables"),
                swapused=(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1024.0 / 1024.0,
            )
        except Exception:
            pass
        try:
            vs = {}
            for ln in open("/proc/vmstat"):
                p = ln.split()
                if len(p) == 2:
                    vs[p[0]] = int(p[1])
            d["pswpout_ctr"] = vs.get("pswpout", 0)
            d["pgst_kswapd_ctr"] = vs.get("pgsteal_kswapd", 0)
            d["pgst_direct_ctr"] = vs.get("pgsteal_direct", 0)
        except Exception:
            pass
        try:
            for ln in open("/proc/self/status"):
                if ln.startswith("VmSwap:"):
                    d["self_vmswap"] = int(ln.split()[1]) / 1024.0 / 1024.0
                    break
        except Exception:
            pass
        return d

    _ORDER = [
        "cuda_alloc",
        "cuda_resv",
        "dev_used",
        "dev_free",
        "memfree",
        "cached",
        "anon",
        "mapped",
        "shmem",
        "srecl",
        "sunrecl",
        "pagetbl",
        "swapused",
        "self_vmswap",
        "pswpout_ctr",
        "pgst_kswapd_ctr",
        "pgst_direct_ctr",
    ]

    def _fmt(d: "dict[str, float]") -> str:
        out = []
        for k in _ORDER:
            v = d.get(k)
            if v is None:
                continue
            out.append(("%s=%d" % (k, v)) if k.endswith("_ctr") else ("%s=%.2f" % (k, v)))
        return " ".join(out)

    def _emit(tag: str) -> None:
        sys.stderr.write("[memprobe] %-40s %s\n" % (tag, _fmt(_snap())))
        sys.stderr.flush()

    _ticking = threading.Event()

    def _ticker() -> None:
        # Detailed line every ~1.5s (whole NCCL-phase trajectory lands in Loki),
        # plus immediately on a >0.5G cuda or >1G swap move. Runs only in the
        # loading process (started from the load_model wrapper).
        n = 0
        lca = lsw = -99.0
        while True:
            d = _snap()
            n += 1
            ca = d.get("cuda_alloc", -1.0)
            sw = d.get("swapused", -1.0)
            if n % 3 == 0 or abs(ca - lca) > 0.5 or abs(sw - lsw) > 1.0:
                sys.stderr.write("[memprobe.tick] %s\n" % _fmt(d))
                sys.stderr.flush()
                lca, lsw = ca, sw
            time.sleep(0.5)

    def _start_ticker_once() -> None:
        if not _ticking.is_set():
            _ticking.set()
            threading.Thread(target=_ticker, name="memprobe", daemon=True).start()
            _emit("ticker-started")

    def _wrap_bracket(cls: Any, name: str, label: str, start_ticker: bool = False) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "__memprobe__", False):
            return

        def w(*a: Any, **k: Any) -> Any:
            if start_ticker:
                _start_ticker_once()
            _emit("BEGIN " + label)
            try:
                return orig(*a, **k)
            finally:
                _emit("END   " + label)

        setattr(w, "__memprobe__", True)
        setattr(cls, name, w)

    def _wrap_delta(cls: Any, name: str, label: str) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "__memprobe__", False):
            return

        def w(*a: Any, **k: Any) -> Any:
            a0 = None
            try:
                import torch

                a0 = torch.cuda.memory_allocated()
            except Exception:
                pass
            try:
                return orig(*a, **k)
            finally:
                if a0 is not None:
                    try:
                        import torch

                        d = (torch.cuda.memory_allocated() - a0) / 1e9
                        if abs(d) > 0.2:
                            sys.stderr.write("[memprobe.delta] %-40s d_cuda_alloc=%+7.2fG\n" % (label, d))
                            sys.stderr.flush()
                    except Exception:
                        pass

        setattr(w, "__memprobe__", True)
        setattr(cls, name, w)

    # module -> [(class, method, kind)]; kind: bracket_tick | bracket | delta
    _TARGETS = {
        "sglang.srt.model_executor.model_runner": [
            ("ModelRunner", "load_model", "bracket_tick"),
            ("ModelRunner", "init_memory_pool", "bracket"),
            ("ModelRunner", "init_attention_backend", "bracket"),
        ],
        "sglang.srt.layers.quantization.fp8": [
            ("Fp8MoEMethod", "process_weights_after_loading", "delta"),
            ("Fp8MoEMethod", "process_weights_after_loading_block_quant", "delta"),
            ("Fp8LinearMethod", "process_weights_after_loading", "delta"),
            ("Fp8LinearMethod", "process_weights_after_loading_block_quant", "delta"),
        ],
        "sglang.srt.model_executor.cuda_graph_runner": [
            ("CudaGraphRunner", "capture", "bracket"),
        ],
    }

    def _apply(modname: str, mod: Any) -> None:
        for cls_name, meth, kind in _TARGETS.get(modname, []):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            label = cls_name + "." + meth
            if kind == "bracket_tick":
                _wrap_bracket(cls, meth, label, start_ticker=True)
            elif kind == "bracket":
                _wrap_bracket(cls, meth, label)
            else:
                _wrap_delta(cls, meth, label)
        sys.stderr.write("[memprobe] wrapped %s\n" % modname)
        sys.stderr.flush()

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
            if name not in _TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec and spec.loader:
                real = spec.loader.exec_module

                def exec_module(module: Any, _r: Any = real, _n: str = name) -> None:
                    _r(module)
                    try:
                        _apply(_n, module)
                    except Exception as e:  # noqa: BLE001
                        sys.stderr.write("[memprobe] wrap-failed %s: %s\n" % (_n, e))
                        sys.stderr.flush()

                spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    # modules already imported when the probe loads -> wrap immediately
    for _m in list(_TARGETS):
        if _m in sys.modules:
            try:
                _apply(_m, sys.modules[_m])
            except Exception:
                pass
    sys.meta_path.insert(0, _Hook())
    sys.stderr.write("[memprobe] armed (DSV4_MEMPROBE=1)\n")
    sys.stderr.flush()
