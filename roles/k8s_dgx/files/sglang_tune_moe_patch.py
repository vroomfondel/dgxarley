# Patches the upstream fused-MoE tuning scripts in place: progress prints, env overrides, incremental save + resume.

import re
import sys
import textwrap
from pathlib import Path
from typing import Callable

Builder = Callable[[str], str]


def _block(indent: str, text: str) -> str:
    return textwrap.indent(textwrap.dedent(text).lstrip("\n"), indent)


def _apply(src: str, name: str, marker: str, pattern: str, build: Builder) -> str:
    if marker in src:
        print(f"Patch {name}: already applied")
        return src
    m = re.search(pattern, src, re.M)
    if m is None:
        print(f"WARNING: patch {name}: anchor not found, upstream may have changed")
        return src
    print(f"Patch {name}: applied")
    return src[: m.start()] + build(m.group("i")) + src[m.end() :]


def patch_common_utils(path: Path) -> None:
    src = path.read_text()
    src = _apply(
        src,
        "group_size",
        "if group_size is not None:",
        r'^(?P<i>[ ]*)group_size = weights_config\.get\("group_size"\)\n[ ]*block_shape = \[0, group_size\]\n[ ]*assert len\(block_shape\) == 2\n',
        lambda i: _block(
            i,
            """
            group_size = weights_config.get("group_size")
            if group_size is not None:
                block_shape = [0, group_size]
                assert len(block_shape) == 2
            """,
        ),
    )
    path.write_text(src)


_HELPERS = """
def _tune_load_partial(path):
    done = {}
    if path and _tune_os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                done[int(rec["batch_size"])] = rec["config"]
    return done


def _tune_record_partial(path, num_tokens, config, kernel_time):
    _tune_os.makedirs(_tune_os.path.dirname(path) or ".", exist_ok=True)
    torn = False
    if _tune_os.path.exists(path) and _tune_os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            torn = f.read(1) != b"\\n"
    with open(path, "a") as f:
        if torn:
            f.write("\\n")
        f.write(json.dumps({"batch_size": num_tokens, "config": config, "time_us": kernel_time}) + "\\n")
        f.flush()
        _tune_os.fsync(f.fileno())
    done = _tune_load_partial(path)
    snap = (path[: -len(".jsonl")] if path.endswith(".jsonl") else path) + ".json"
    with open(snap + ".tmp", "w") as f:
        json.dump({M: sort_config(c) for M, c in sorted(done.items())}, f, indent=4)
        f.write("\\n")
    _tune_os.replace(snap + ".tmp", snap)
    print(f"Partial result saved: batch_size={num_tokens} ({len(done)} done) -> {snap}", flush=True)


"""


def patch_tuner(path: Path) -> None:
    src = path.read_text()
    src = _apply(
        src,
        "os-import",
        "import os as _tune_os",
        r"^(?P<i>)import argparse\n",
        lambda i: "import argparse\nimport os as _tune_os\n",
    )
    src = _apply(
        src,
        "progress",
        "_cfg_idx",
        r"^(?P<i>[ ]*)for config in tqdm\(search_space\):\n",
        lambda i: _block(
            i,
            """
            print(flush=True)
            print(f"--- batch_size={num_tokens}: tuning {len(search_space)} configs ---", flush=True)
            _total = len(search_space)
            for _cfg_idx, config in enumerate(search_space, 1):
                if _cfg_idx == 1 or _cfg_idx % 50 == 0 or _cfg_idx == _total:
                    _best_str = f"best={best_time:.1f}us" if best_time < float("inf") else "searching..."
                    print(f"  batch_size={num_tokens}: [{_cfg_idx}/{_total}] ({_cfg_idx / _total * 100:.0f}%) {_best_str}", flush=True)
            """,
        ),
    )
    src = _apply(
        src,
        "record-partial",
        "_tune_record_partial(_tune_partial",
        r'^(?P<i>[ ]*)print\(f"\{now\.ctime\(\)\}\] Completed tuning for batch_size=\{num_tokens\}"\)\n',
        lambda i: _block(
            i,
            """
            print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}", flush=True)
            _tune_partial = _tune_os.environ.get("MOE_TUNE_PARTIAL", "")
            if _tune_partial and best_config is not None:
                _tune_record_partial(_tune_partial, num_tokens, best_config, best_time)
            """,
        ),
    )
    src = _apply(
        src,
        "partial-helpers",
        "def _tune_load_partial(",
        r"^(?P<i>)def main\(",
        lambda i: _HELPERS.lstrip("\n") + "def main(",
    )
    src = _apply(
        src,
        "ray-env",
        'runtime_env={"env_vars"',
        r"^(?P<i>[ ]*)ray\.init\(\)\n",
        lambda i: _block(
            i,
            """
            ray.init(
                runtime_env={"env_vars": {k: _tune_os.environ[k] for k in ("MOE_TUNE_PARTIAL",) if k in _tune_os.environ}}
            )
            """,
        ),
    )
    src = _apply(
        src,
        "existing-check",
        "_tune_banner",
        r"^(?P<i>[ ]*)ray\.init\(",
        lambda i: _block(
            i,
            """
            _tune_existing_dir = _tune_os.environ.get("MOE_TUNE_EXISTING_DIR", "")
            if args.tune and _tune_existing_dir:
                _tune_target = get_config_filename(
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                )
                _tune_target_path = _tune_os.path.join(_tune_existing_dir, _tune_target)
                _tune_force = _tune_os.environ.get("FORCE_RETUNE", "false") == "true"
                _tune_subset = args.batch_sizes is not None or args.batch_size is not None
                _tune_requested = args.batch_sizes or ([args.batch_size] if args.batch_size is not None else [])
                _tune_hint = "ansible -e sglang_force_retune=true  |  Job env FORCE_RETUNE=true"

                def _tune_banner(*lines):
                    bar = "!" * 100
                    print(bar, *[f"!!! {line}" for line in lines], bar, sep="\\n", flush=True)

                if not _tune_os.path.exists(_tune_target_path):
                    print(f"No tuned config yet: {_tune_target}", flush=True)
                else:
                    with open(_tune_target_path) as _f:
                        _tune_have = sorted(int(k) for k in json.load(_f))
                    _tune_missing = [M for M in _tune_requested if M not in _tune_have]
                    if not _tune_force and _tune_subset:
                        _tune_banner(
                            f"SKIPPING: batch-size subset {_tune_requested} was requested, but the target file already exists:",
                            f"  {_tune_target_path}",
                            f"  it holds batch sizes {_tune_have}",
                            f"  of the requested ones it LACKS {_tune_missing}" if _tune_missing else "  it already holds every requested batch size",
                            f"To tune the subset anyway, override with: {_tune_hint}",
                            "  then the result holds ONLY the subset; the job backs the old file up (.bak-<ts>) and prints a merge command",
                        )
                        raise SystemExit(3)
                    if not _tune_force:
                        print(f"Tuned config already exists: {_tune_target_path} (batch sizes {_tune_have})", flush=True)
                        print(f"Not re-tuning; override with: {_tune_hint}", flush=True)
                        raise SystemExit(3)
                    if _tune_subset:
                        _tune_banner(
                            f"FORCE_RETUNE with batch-size subset {_tune_requested}: the existing file will be REPLACED",
                            f"  {_tune_target_path}",
                            f"  currently holds batch sizes {_tune_have}; the new file will hold ONLY {_tune_requested}",
                            "  the job backs the old file up (.bak-<ts>) and prints a merge command at the end",
                        )
                    else:
                        print(f"FORCE_RETUNE: re-tuning {_tune_target_path}; the old file is backed up (.bak-<ts>)", flush=True)
            """,
        )
        + i
        + "ray.init(",
    )
    src = _apply(
        src,
        "block-shape-override",
        "TUNE_BLOCK_SHAPE",
        r'^(?P<i>[ ]*)block_shape = model_config\["block_shape"\]\n',
        lambda i: _block(
            i,
            """
            block_shape = model_config["block_shape"]
            _bs_env = _tune_os.environ.get("TUNE_BLOCK_SHAPE", "").strip()
            if _bs_env:
                block_shape = None if _bs_env.lower() == "none" else [int(x) for x in _bs_env.split(",")]
                print(f"TUNE_BLOCK_SHAPE override: block_shape={block_shape}", flush=True)
            """,
        ),
    )
    src = _apply(
        src,
        "fp8-autodetect",
        "Auto-detect FP8 from model quantization_config",
        r"^(?P<i>[ ]*)per_channel_quant = args\.per_channel_quant\n",
        lambda i: _block(
            i,
            """
            per_channel_quant = args.per_channel_quant
            # Auto-detect FP8 from model quantization_config when --dtype auto
            if args.dtype == "auto":
                _qcfg = getattr(get_config(args.model, trust_remote_code=True), "quantization_config", None)
                if isinstance(_qcfg, dict) and _qcfg.get("quant_method") == "fp8":
                    use_fp8_w8a8 = True
            """,
        ),
    )
    src = _apply(
        src,
        "resume",
        "_tune_all_batch_sizes",
        r"^(?P<i>[ ]*)configs = _distribute\(\n",
        lambda i: _block(
            i,
            """
            _tune_all_batch_sizes = list(batch_sizes)
            _tune_done = {
                M: c
                for M, c in _tune_load_partial(_tune_os.environ.get("MOE_TUNE_PARTIAL", "")).items()
                if M in _tune_all_batch_sizes
            }
            if _tune_done:
                print(f"Resuming: {len(_tune_done)} batch sizes already tuned {sorted(_tune_done)}", flush=True)
            batch_sizes = [M for M in _tune_all_batch_sizes if M not in _tune_done]
            print(f"Tuning plan: {len(batch_sizes)} batch sizes {batch_sizes} x {len(search_space)} configs", flush=True)
            configs = _distribute(
            """,
        ),
    )
    src = _apply(
        src,
        "resume-merge",
        "_tune_new = dict(",
        r"^(?P<i>[ ]*)best_configs = \{\n[ ]*M: sort_config\(config\) for M, config in zip\(batch_sizes, configs\)\n[ ]*\}\n",
        lambda i: _block(
            i,
            """
            _tune_new = dict(zip(batch_sizes, configs))
            best_configs = {
                M: sort_config(_tune_new[M] if M in _tune_new else _tune_done[M]) for M in _tune_all_batch_sizes
            }
            """,
        ),
    )
    path.write_text(src)


def main() -> None:
    workdir = Path(sys.argv[1])
    patch_common_utils(workdir / "common_utils.py")
    patch_tuner(workdir / "tuning_fused_moe_triton.py")


if __name__ == "__main__":
    main()
