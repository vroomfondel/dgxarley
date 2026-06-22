#!/bin/bash
set -e

# Install ping for ARP priming (not included in sglang image)
apt-get update -qq && apt-get install -y -qq tini iproute2 iputils-ping net-tools curl ethtool >/dev/null 2>&1

# accelerate: required by SGLang's ModelOptModelLoader
# (srt/model_loader/loader.py → _load_modelopt_base_model). Triggered by:
#   - GLM-5-NVFP4 (modelopt base model load path)
#   - EAGLE3/speculative decoding with a modelopt-quantized target model
#     (e.g. nvidia/Qwen3-235B-A22B-NVFP4 + lmsys EAGLE3 draft) — the draft
#     worker loads the target's embeddings via _load_modelopt_base_model
#     and hits ImportError if accelerate is missing.
# Upstream scitrera/dgx-spark-sglang image does NOT ship accelerate.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]] || [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate
fi

# GLM-5 specific: transformers upgrade + mem_get_info patch.
# Only needed for glm_moe_dsa models — skip for MiniMax, Qwen, etc.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]]; then
  echo "GLM-5 model detected — applying GLM-5 specific patches..."

  # transformers ≥5.3.0: required for glm_moe_dsa model type.
  # Must also pull huggingface_hub >=1.3.0 (transformers 5.3.0 dependency).
  python3 -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'glm_moe_dsa' in CONFIG_MAPPING" 2>/dev/null \
    || pip install transformers==5.3.0 huggingface_hub==1.3.0

  # Patch _cuda_mem_fallback: transformers 5.x + huggingface_hub >=1.3.0
  # triggers a CUDA context init during import that breaks torch.cuda.mem_get_info()
  # on GB10 (cudaErrorMemoryAllocation). nvidia-smi also can't report memory on GB10.
  # Fix: fall back to /proc/meminfo (GB10 unified memory = system RAM).
  COMMON_PY="/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/common.py"
  if grep -q '_cuda_mem_fallback' "$COMMON_PY" 2>/dev/null; then
    python3 << 'PATCH_MEM_FALLBACK_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/common.py"
with open(f) as fh:
    code = fh.read()
old = '    raise RuntimeError(\n        f"Failed to get GPU memory capacity from nvidia-smi. "'
if old in code:
    new = '''    # GB10 unified memory: read total from /proc/meminfo
    import logging as _logging
    try:
        with open("/proc/meminfo") as _mf:
            for _line in _mf:
                if _line.startswith("MemTotal:"):
                    _mem_mib = int(_line.split()[1]) // 1024  # kB → MiB
                    break
        _logging.getLogger(__name__).warning(
            f"nvidia-smi and torch.cuda.mem_get_info() both failed. "
            f"Falling back to /proc/meminfo MemTotal: {_mem_mib} MiB."
        )
        return _mem_mib
    except Exception as _e:
        _logging.getLogger(__name__).warning(f"/proc/meminfo fallback also failed: {_e}")
    raise RuntimeError(
        f"Failed to get GPU memory capacity from nvidia-smi. "'''
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched _cuda_mem_fallback: /proc/meminfo fallback for unified memory (GB10)")
else:
    print("_cuda_mem_fallback: patch target not found, skipping")
PATCH_MEM_FALLBACK_EOF
  fi
else
  echo "SKIPPING GLM-5 specific patches..."
fi

# Patch SGLang get_config() to convert dict sub_configs after loading (transformers 5.5.0 bug).
# Transformers 5.x auto-generates __init__ for PretrainedConfig subclasses with sub_configs,
# bypassing dict→config conversion. from_pretrained() also bypasses __post_init__.
# Both vision_config and text_config arrive as raw dicts → AttributeError on .hidden_size etc.
# Fix: patch get_config() to convert dict sub-configs after loading for any config with sub_configs.
HF_UTILS="/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers_utils.py"
if grep -q 'return config' "$HF_UTILS" 2>/dev/null && ! grep -q 'sub_configs dict fix' "$HF_UTILS" 2>/dev/null; then
  python3 << 'PATCH_GET_CONFIG_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers_utils.py"
with open(f) as fh:
    code = fh.read()
# Find the final "return config" in get_config() and add sub_configs conversion before it.
# The function ends with:
#     return config
# We insert a conversion block just before.
old = '''    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config'''
new = '''    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    # [patch] sub_configs dict fix — transformers 5.x from_pretrained() leaves sub-configs
    # as raw dicts instead of converting to their declared config classes.
    _sub_cfgs = getattr(config, "sub_configs", None)
    if _sub_cfgs:
        for _key, _cls in _sub_cfgs.items():
            _val = getattr(config, _key, None)
            if isinstance(_val, dict):
                try:
                    setattr(config, _key, _cls(**_val))
                except Exception:
                    pass  # non-critical: some sub-configs may not accept all dict keys

    # [patch] Qwen3.5 MoE: text_config lacks norm_topk_prob (Qwen2MoeSparseMoeBlock expects it).
    # Qwen3.5 uses softmax routing — renormalize=True is correct default.
    _tc = getattr(config, "text_config", None)
    if _tc is not None and not isinstance(_tc, dict) and not hasattr(_tc, "norm_topk_prob"):
        _tc.norm_topk_prob = True

    return config'''
if old in code:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched get_config(): sub_configs dict→config conversion after loading")
else:
    print("get_config(): patch target not found (code changed?)")
PATCH_GET_CONFIG_EOF
fi

# Prime ARP table on the QSFP link before NCCL tries to connect.
# Without this, the first TCP SYNs get dropped until ARP resolves,
# causing ~230s delay in "Init torch distributed".
# Full-mesh: every node pings ALL other nodes so NCCL's ring/tree
# topology can communicate immediately over any path.
IFS=',' read -ra peers <<< "$QSFP_PEER_IPS"
pids=()
for peer in "${peers[@]}"; do
  (
    echo "Waiting for QSFP peer ${peer} ..."
    while true; do
      ping -c3 -W1 "$peer" 2>&1 | sed "s/^/[${peer}] /"
      [[ ${PIPESTATUS[0]} -eq 0 ]] && break
      sleep 1
    done
    echo "QSFP peer ${peer} reachable."
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "All ${#peers[@]} QSFP peers reachable."

# Patch CUTLASS BlockScaledMmaOp to support SM121 (DGX Spark GB10) for FP4 operations.
# Upstream CUTLASS restricts FP4 tensor ops to sm_100a only (issue NVIDIA/cutlass#2800).
# SM121 has native FP4 Tensor Core support but is not in admissible_archs → the JIT-compiled
# nvfp4_blockwise_moe kernel falls back to an incompatible code path → device-side assert.
# Fix: add sm_120a + sm_121a to admissible_archs in both CUTLASS DSL copies.
# External validation: BTankut/dgx-spark-sglang-moe-configs achieved 356 TFLOPS NVFP4 on GB10.
for mma_py in \
  /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py \
  /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py; do
  if [ -f "$mma_py" ] && grep -q 'admissible_archs = \[' "$mma_py" 2>/dev/null; then
    if ! grep -q 'sm_121a' "$mma_py" 2>/dev/null; then
      sed -i 's/Arch\.sm_100a,/Arch.sm_100a, Arch.sm_120a, Arch.sm_121a,/' "$mma_py"
      echo "Patched $(basename $(dirname $(dirname $(dirname "$mma_py"))))/mma.py: added sm_120a + sm_121a to BlockScaledMmaOp.admissible_archs"
    else
      echo "$(basename "$mma_py"): sm_121a already present"
    fi
  fi
done

# NOTE: nvfp4_blockwise_moe.cuh SM121 tile fix was attempted but cannot be solved
# via runtime patching. The CUTLASS FP4 grouped GEMM kernel requires SM121-specific
# sgl-kernel build with proper TMA tile shapes — not achievable by editing the .cuh
# source at startup. See CUTLASS_NVFP4_SM121_PRD.md for full analysis.
# For NVFP4 models on SM121: use flashinfer_cutlass MoE runner (avoids cutlass_moe_fp4).

# ─────────────────────────────────────────────────────────────────────────────
# Patch sglang Transformers fallback: add Gemma-4 MoE config attribute names.
#
# Gemma-4 (Gemma4ForConditionalGeneration) has no native SGLang implementation
# and falls through to the Transformers backend. The MoEMixin.recursive_replace()
# method looks up top_k via ("num_experts_per_tok", "top_k") — Gemma-4 uses
# "top_k_experts" instead → AssertionError: Cannot determine top_k from config.
#
# Fix: add "top_k_experts" to the _getattr_first lookup tuple on line 1197.
#
# Note: SGLang v0.5.11 has native Gemma-4 support (PR #21952 + follow-ups
# #22079, #24048, #22842), so this patch is a no-op there (the grep guard
# inside makes it idempotent — pattern not found → skip).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_TRANSFORMERS_TOPK_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/transformers.py")
if not p.exists():
    print("sglang/srt/models/transformers.py: not found, skipping")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_gemma4_topk_"
    old = '("num_experts_per_tok", "top_k")'
    new = '("num_experts_per_tok", "top_k", "top_k_experts")'
    if marker in src:
        print("sglang/srt/models/transformers.py: already patched (top_k_experts), skipping")
    elif old not in src:
        print("sglang/srt/models/transformers.py: top_k lookup pattern not found, skipping")
    else:
        # Replace the tuple inline. Marker goes on a NEW line above to avoid
        # breaking the closing paren of _getattr_first(...).
        src = src.replace(old, new, 1)
        # Insert marker as a comment on the line before the patched line
        src = src.replace(
            "top_k = _getattr_first(text_config, " + new,
            marker + "\n        top_k = _getattr_first(text_config, " + new,
            1,
        )
        p.write_text(src)
        print("Patched sglang/srt/models/transformers.py: added top_k_experts to MoE config lookup")
PATCH_TRANSFORMERS_TOPK_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch flashinfer.jit.cpp_ext.get_cuda_version: avoid subprocess from inside a
# torch.compile/dynamo trace.
#
# Symptom (GLM-4.7-NVFP4 EP=1, piecewise CUDA graphs + fi_cudnn FP4):
#   flashinfer/quantization/fp4_quantization.py:170 build_and_load()
#   → gen_fp4_quantization_sm120f_module
#   → flashinfer/jit/cpp_ext.py:91 is_cuda_version_at_least("12.8")
#   → cpp_ext.py:73 subprocess.check_output([nvcc, "--version"])
#   → subprocess/threading.Lock() under torch/_dynamo/polyfills:392 getattr_and_trace
#   → child process sigquit → pod restart → startup_crash
#
# Why: the JIT build is triggered on the first forward pass, which for piecewise
# CUDA graphs happens inside a torch.compile trace. Dynamo can't polyfill
# subprocess.Popen (it does fork/threading internals), so the call blows up even
# though nvcc is present and works fine from a normal shell.
#
# Fix: short-circuit get_cuda_version() with torch.version.cuda, which is always
# available at import time and matches what nvcc reports for the same install.
# The original function already had this as a fallback path on exception — we
# just promote it to run first. This keeps the subprocess path as the fallback
# for pytorch builds without a CUDA version (none of ours).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_FI_CUDA_VER_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py")
if not p.exists():
    print("flashinfer/jit/cpp_ext.py: not found, skipping")
else:
    src = p.read_text()
    marker = "# [patch] _fi_cuda_ver_subprocess_bypass_"
    if marker in src:
        print("flashinfer/jit/cpp_ext.py: get_cuda_version already patched, skipping")
    else:
        target = (
            "@functools.cache\n"
            "def get_cuda_version() -> Version:\n"
            "    # Try to query nvcc for CUDA version; if nvcc is unavailable, "
            "fall back to torch.version.cuda\n"
            "    try:"
        )
        replacement = (
            "@functools.cache\n"
            "def get_cuda_version() -> Version:\n"
            "    " + marker + "\n"
            "    # Short-circuit with torch.version.cuda to avoid spawning a `nvcc --version`\n"
            "    # subprocess from inside a torch.compile/dynamo trace context. See the\n"
            "    # sglang_launch.sh header block above this patch for the full rationale.\n"
            "    if torch.version.cuda is not None:\n"
            "        return Version(torch.version.cuda)\n"
            "    # Try to query nvcc for CUDA version; if nvcc is unavailable, "
            "fall back to torch.version.cuda\n"
            "    try:"
        )
        if target in src:
            p.write_text(src.replace(target, replacement, 1))
            print("Patched flashinfer/jit/cpp_ext.py:get_cuda_version to bypass subprocess")
        else:
            print("flashinfer/jit/cpp_ext.py: get_cuda_version target pattern not found, skipping")
PATCH_FI_CUDA_VER_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch flashinfer.quantization.fp4_quantization: register `fp4_quantize` as
# an opaque leaf op via `torch.compiler.allow_in_graph`, so dynamo emits a
# single FX graph node for the call without tracing its body.
#
# Symptom chain (GLM-4.7-NVFP4 EP=1, piecewise CUDA graphs):
#   sglang/srt/compilation/compile.py:183 _ensure_compiled
#   → torch._dynamo compile_wrapper
#   → sglang/.../modelopt_quant.py:1482 fp4_quantize(x, layer.input_scale_inv)
#   → flashinfer/quantization/fp4_quantization.py:700 fp4_quantize
#
# Dynamo hits a whole family of un-traceable things as it recurses into the
# FP4 path:
#   (a) get_fp4_quantization_module → JitSpec.is_aot → Path.exists → os.stat
#       → "Attempted to call function marked as skipped: posix.stat"
#   (b) fp4_quantize_sm100 (line 222) → `module.fp4_quantize(...)` is a
#       torch.autograd.Function → "Unsupported method call: Function.__call__"
#   (c) almost certainly more further in.
#
# Whack-a-mole fixes (functools.cache wrapping, pre-resolving the module
# lookup into a module-level constant, etc.) each unblock one layer and
# then hit the next. `@torch.compiler.disable` also doesn't work: sglang's
# piecewise compile path treats skipped calls as a hard error (gb0098:
# "Skip calling torch.compiler.disable()d function") rather than a normal
# graph break.
#
# Correct approach: `torch.compiler.allow_in_graph`. This tells dynamo to
# emit a single opaque call node for `fp4_quantize` in the FX graph without
# tracing through its body at all — the JIT lookup, os.stat, and
# autograd.Function.__call__ all execute at real runtime (outside any
# trace), which is completely fine for all of them.
#
# Contract for allow_in_graph: the function must take/return tensors
# (or pytrees of tensors) and must be deterministic in output dtype/shape
# given input dtype/shape. `fp4_quantize(input, global_scale)` returns
# (x_q, sf) — both tensors — and its output shapes are a deterministic
# function of input shape + sf_vec_size. Contract satisfied.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_FI_FP4_ALLOW_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py")
if not p.exists():
    print("flashinfer/quantization/fp4_quantization.py: not found, skipping")
else:
    src = p.read_text()
    marker = "# [patch] _fi_fp4_allow_in_graph_"
    # Undo any remnants from earlier patch revisions in the same container
    # (e.g. after a crash loop re-execs sglang_launch.sh without a fresh
    # image) so the source stays clean.
    stale_const_call = "_SGLANG_FP4_MOD.fp4_quantize_sm100("
    if stale_const_call in src:
        src = src.replace(
            stale_const_call,
            'get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100(',
        )
    src = src.replace(
        "@torch.compiler.disable\n@flashinfer_api\ndef fp4_quantize(\n",
        "@flashinfer_api\ndef fp4_quantize(\n",
    )
    for old_marker in (
        "# [patch] _fi_fp4_cache_and_prewarm_",
        "# [patch] _fi_fp4_prewarm_const_",
        "# [patch] _fi_fp4_compiler_disable_",
    ):
        idx = src.find("\n\n# " + old_marker)
        if idx != -1:
            src = src[:idx].rstrip() + "\n"
    if marker in src:
        print("flashinfer/quantization/fp4_quantization.py: already patched (allow_in_graph), writing back only cleanup")
        p.write_text(src)
    elif "def fp4_quantize(" not in src:
        print("flashinfer/quantization/fp4_quantization.py: fp4_quantize not found, skipping")
    else:
        # The registration must run at module import time AFTER fp4_quantize
        # has been defined. We append a trailing block that imports torch
        # and rebinds the module-level name through allow_in_graph. Rebinding
        # the name is safe because allow_in_graph returns a wrapper with the
        # exact same calling contract, and any subsequent `from flashinfer...
        # import fp4_quantize` inside sglang picks up the wrapped version.
        append_block = (
            "\n\n"
            "# " + marker + "\n"
            "# Appended by sglang_launch.sh runtime patch. Registers fp4_quantize\n"
            "# as an opaque leaf op so dynamo emits a single FX graph node for it\n"
            "# instead of tracing into the body (which hits os.stat during the JIT\n"
            "# lookup and torch.autograd.Function.__call__ inside fp4_quantize_sm100).\n"
            "# See sglang_launch.sh header for full rationale.\n"
            "try:\n"
            "    import torch as _sglang_t\n"
            "    fp4_quantize = _sglang_t.compiler.allow_in_graph(fp4_quantize)\n"
            "    import sys as _sglang_sys\n"
            "    print('[fp4_quantization] fp4_quantize registered via allow_in_graph', file=_sglang_sys.stderr)\n"
            "except Exception as _sglang_e:\n"
            "    import sys as _sglang_sys\n"
            "    print(f'[fp4_quantization] allow_in_graph registration failed: {_sglang_e}', file=_sglang_sys.stderr)\n"
        )
        p.write_text(src + append_block)
        print("Patched flashinfer/quantization/fp4_quantization.py: fp4_quantize → allow_in_graph")
PATCH_FI_FP4_ALLOW_EOF

# Version gate: warn if the container image changed — patches below may need review.
# Dev builds report __version__=0.0.0 (no setuptools-scm), so we check the image
# tag (injected as SGLANG_IMAGE env var by Ansible) instead of the Python version.
# The grep guards still prevent patching if the target code has changed.
SGLANG_EXPECTED_IMAGE_PATTERN="xomoxcc/dgx-spark-sglang:.*-sm121-dev1"
if [ -n "$SGLANG_IMAGE" ] && ! echo "$SGLANG_IMAGE" | grep -qE "^${SGLANG_EXPECTED_IMAGE_PATTERN}$"; then
  echo "WARNING: SGLang image does not match expected pattern ${SGLANG_EXPECTED_IMAGE_PATTERN} (got ${SGLANG_IMAGE})."
  echo "         Monkey-patches may no longer apply or may need updating."
fi

# When load_format is sharded_state, wait for the shard marker then use sharded path
model_path="$SGLANG_MODEL"
if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
  model_slug=$(echo "$SGLANG_MODEL" | sed 's|/|--|g')
  shard_suffix="sglang-TP${TP}"
  if [ -n "$EP" ] && [ "$EP" != "1" ]; then
    shard_suffix="${shard_suffix}-EP${EP}"
  fi
  if [ -n "$SGLANG_QUANTIZATION" ]; then
    shard_suffix="${shard_suffix}-${SGLANG_QUANTIZATION}"
  fi
  if [ -n "$SGLANG_MOE_RUNNER_BACKEND" ]; then
    shard_suffix="${shard_suffix}-${SGLANG_MOE_RUNNER_BACKEND}"
  fi
  sharded_path="/root/.cache/huggingface/sharded/${model_slug}-${shard_suffix}"
  marker="${sharded_path}/model.safetensors.index.json"
  echo "Waiting for sharded checkpoint at ${marker} ..."
  while [ ! -f "$marker" ]; do
    echo "  $(date '+%H:%M:%S') shard not ready yet, waiting 30s ..."
    sleep 30
  done
  model_path="$sharded_path"
  echo "Using pre-sharded model at ${model_path}"
fi

# Patch weight loading iterators to log progress per shard file.
# tqdm writes directly to sys.stderr in TP worker subprocesses — this output is
# NOT forwarded by SGLang's logger infrastructure, so it never appears in kubectl logs.
# Additionally, BAR_FORMAT lacks a trailing \n, so tqdm uses \r (carriage return)
# which is invisible in non-TTY kubectl logs.
# Fix: replace tqdm loops with logger.info() calls that go through the logging pipeline.
#
# v0.5.10: enable_multithread_load defaults to True, so the default code path is
# buffered_multi_thread_safetensors_weights_iterator (not the old single-thread one).
# We patch both to cover all cases.
WEIGHT_UTILS="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
if [ -f "$WEIGHT_UTILS" ]; then
  python3 << 'PATCH_SAFETENSORS_TQDM_EOF'
import os
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
with open(f) as fh:
    code = fh.read()
patched = False
# Add logger import if missing
if "\nlogger = " not in code and "\nlogger=" not in code:
    code = code.replace(
        "from tqdm.auto import tqdm",
        "import logging\nfrom tqdm.auto import tqdm\nlogger = logging.getLogger(__name__)",
        1)

# --- Patch 1: single-thread safetensors_weights_iterator (v0.5.10rc0 path) ---
old_single = '''    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):'''
new_single = '''    _total = len(hf_weights_files)
    for _i, st_file in enumerate(hf_weights_files, 1):
        if enable_tqdm:
            logger.info(f"Loading safetensors shard {_i}/{_total}: {os.path.basename(st_file)}")'''
if old_single in code:
    code = code.replace(old_single, new_single, 1)
    patched = True
    print("Patched safetensors_weights_iterator: tqdm → logger.info")

# --- Patch 2: buffered_multi_thread_safetensors_weights_iterator (v0.5.10 default) ---
# Replace tqdm progress bar in the sliding-window loop with logger.info per shard.
old_buffered = '''        with tqdm(
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm,
            bar_format=BAR_FORMAT,
            position=tqdm._get_free_pos(),
        ) as pbar:
            while pending:
                future = pending.popleft()
                state_dict = future.result()
                del future  # let GC reclaim the Future's internal result

                # Replenish: submit the next file to keep the buffer full.
                next_file = next(file_iter, None)
                if next_file is not None:
                    pending.append(executor.submit(_load_file, next_file))

                for name in sorted(state_dict.keys()):
                    yield name, state_dict[name]
                del state_dict
                pbar.update(1)'''
new_buffered = '''        _shard_done = 0
        _shard_total = len(hf_weights_files)
        while pending:
            future = pending.popleft()
            state_dict = future.result()
            del future  # let GC reclaim the Future's internal result
            _shard_done += 1

            if enable_tqdm:
                logger.info(f"Loading shard {_shard_done}/{_shard_total} ({len(state_dict)} tensors)")

            # Replenish: submit the next file to keep the buffer full.
            next_file = next(file_iter, None)
            if next_file is not None:
                pending.append(executor.submit(_load_file, next_file))

            for name in sorted(state_dict.keys()):
                yield name, state_dict[name]
            del state_dict'''
if old_buffered in code:
    code = code.replace(old_buffered, new_buffered, 1)
    patched = True
    print("Patched buffered_multi_thread_safetensors_weights_iterator: tqdm → logger.info")

# --- Patch 3: multi_thread_safetensors_weights_iterator (non-buffered variant) ---
old_mt = '''        if enable_tqdm:
            futures_iter = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(hf_weights_files),
                desc="Multi-thread loading shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
            )
        else:
            futures_iter = concurrent.futures.as_completed(futures)

        for future in futures_iter:
            state_dict = future.result()
            for name, param in state_dict.items():
                yield name, param'''
new_mt = '''        _mt_done = 0
        _mt_total = len(hf_weights_files)
        for future in concurrent.futures.as_completed(futures):
            state_dict = future.result()
            _mt_done += 1
            if enable_tqdm:
                logger.info(f"Loading shard {_mt_done}/{_mt_total} ({len(state_dict)} tensors)")
            for name, param in state_dict.items():
                yield name, param'''
if old_mt in code:
    code = code.replace(old_mt, new_mt, 1)
    patched = True
    print("Patched multi_thread_safetensors_weights_iterator: tqdm → logger.info")

if patched:
    if "import os" not in code:
        code = "import os\n" + code
    with open(f, 'w') as fh:
        fh.write(code)
else:
    print("weight_utils: no tqdm patch targets found (already patched or code changed)")
PATCH_SAFETENSORS_TQDM_EOF
fi

# Patch ShardedStateLoader to log progress per shard file (no progress bar by default)
LOADER="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/loader.py"
if grep -q 'for path in filepaths:' "$LOADER" 2>/dev/null; then
  sed -i 's/for path in filepaths:/for _shard_i, path in enumerate(filepaths, 1):/' "$LOADER"
  sed -i '/_shard_i, path in enumerate(filepaths/a\                logger.info(f"Loading shard {_shard_i}\/{len(filepaths)}: {os.path.basename(path)}")' "$LOADER"
  echo "Patched ShardedStateLoader for per-file progress logging"
fi

# Patch moe_wna16 weight loader for EP-aware qzeros handling (SGLang 0.5.9 bug).
# Two bugs in the w13_qzeros/w2_qzeros branches:
#   1. Uses raw global expert_id (0-127) instead of local EP index (0-63)
#   2. Uses global tp_rank for TP-slice, but moe_tp_size=tp/ep — need tp_rank % moe_tp_size
# Safe no-op when ep_size=1 (identity mapping, tp_rank unchanged).
MOE_WNA16="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"
if grep -q 'param\.data\[expert_id' "$MOE_WNA16" 2>/dev/null; then
  python3 << 'PATCH_QZEROS_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"
with open(f) as fh:
    code = fh.read()
old_w13 = '''            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor'''
new_w13 = '''            if "w13_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[_moe_tp_rank]
                if shard_id == "w1":
                    param.data[_local_id, : shard_size // 2] = tensor
                else:
                    param.data[_local_id, shard_size // 2 :] = tensor'''
old_w2 = '''            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, tp_rank]'''
new_w2 = '''            elif "w2_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                param.data[_local_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, _moe_tp_rank]'''
if old_w13 not in code:
    print("w13_qzeros: already patched or source changed, skipping")
    sys.exit(0)
if old_w2 not in code:
    print("w2_qzeros: already patched or source changed, skipping")
    sys.exit(0)
code = code.replace(old_w13, new_w13, 1)
code = code.replace(old_w2, new_w2, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched moe_wna16.py: EP-aware expert_id + tp_rank remapping for qzeros")
PATCH_QZEROS_EOF
fi

# Patch quantization/utils.py: dot-boundary matching in is_layer_skipped()
# (GitHub issue #23687, PR #23467, commit 4323fce, 2026-04-22).
# Naive `ignored in prefix` substring check causes `mlp.gate` (from Qwen3.6-FP8
# modules_to_not_convert) to match `mlp.gate_up_proj`, silently bypassing FP8
# weight_scale_inv registration and producing garbage logits / token salad.
# Fix: replace all four bare substring checks with _module_path_match() which
# requires dot-boundary separation, and add _FALLBACK_FUSED_SHARDS for configs
# that don't ship packed_modules_mapping.
#
# Note: SGLang v0.5.11 has this fix upstream (PR #23467 was merged into main
# pre-v0.5.11 cut). On v0.5.11 the grep guard below short-circuits — the
# `def _module_path_match` symbol is already present. On older images
# (v0.5.10 / dev1) the guard fires through and applies the patch. Idempotent.
QUANT_UTILS="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/utils.py"
if [ -f "$QUANT_UTILS" ] && ! grep -q 'def _module_path_match' "$QUANT_UTILS" 2>/dev/null; then
  python3 << 'PATCH_QUANT_UTILS_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/utils.py"
with open(f) as fh:
    code = fh.read()

# Idempotency check (belt-and-suspenders after the shell grep guard)
if "def _module_path_match" in code:
    print("quantization/utils.py: already patched (_module_path_match present), skipping")
    sys.exit(0)

# Hunk 1: inject _module_path_match() and _FALLBACK_FUSED_SHARDS just before
# is_layer_skipped() — anchor on the function signature line.
old1 = '''def is_layer_skipped(
    prefix: str,
    ignored_layers: List[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> bool:'''
new1 = '''def _module_path_match(ignored: str, prefix: str) -> bool:
    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
    # that collide with fused dense MLP names by plain substring.
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")


# Known fused-linear -> shard names. Used as a fallback when the quant
# config doesn't ship packed_modules_mapping (typical for HF FP8 configs).
_FALLBACK_FUSED_SHARDS: dict[str, list[str]] = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
}


def is_layer_skipped(
    prefix: str,
    ignored_layers: List[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> bool:'''

if old1 not in code:
    print("quantization/utils.py: is_layer_skipped signature not found, skipping")
    sys.exit(0)
code = code.replace(old1, new1, 1)

# Hunk 2: inside the fused-mapping branch — use _FALLBACK_FUSED_SHARDS when
# proj_name is not in fused_mapping, and replace bare substring check with
# _module_path_match().
old2 = '''    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                ignored in shard_prefix for ignored in ignored_layers
            )'''
new2 = '''    effective_fused = (
        fused_mapping if proj_name in fused_mapping else _FALLBACK_FUSED_SHARDS
    )
    if proj_name in effective_fused:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in effective_fused[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                _module_path_match(ignored, shard_prefix) for ignored in ignored_layers
            )'''

if old2 not in code:
    print("quantization/utils.py: fused-mapping branch not found, skipping")
    sys.exit(0)
code = code.replace(old2, new2, 1)

# Hunk 3: the else-branch bare substring check → _module_path_match().
old3 = '''    else:
        is_skipped = any(ignored in prefix for ignored in ignored_layers)
        if "gate_up_proj" in prefix:'''
new3 = '''    else:
        is_skipped = any(
            _module_path_match(ignored, prefix) for ignored in ignored_layers
        )
        if "gate_up_proj" in prefix:'''

if old3 not in code:
    print("quantization/utils.py: else-branch substring check not found, skipping")
    sys.exit(0)
code = code.replace(old3, new3, 1)

with open(f, 'w') as fh:
    fh.write(code)
print("Patched quantization/utils.py: dot-boundary _module_path_match + _FALLBACK_FUSED_SHARDS in is_layer_skipped()")
PATCH_QUANT_UTILS_EOF
fi

# Patch modelopt_quant.py: EP-aware input_scale slicing (SGLang 0.5.9 bug).
# In process_weights_after_loading(), the fallback else-branch computes
# w13_input_scale and w2_input_scale with shape (num_experts,) but multiplies
# them with w13_weight_scale_2 / w2_weight_scale_2 which are (num_local_experts,).
# With EP=2 on MiniMax-M2.5 (256 experts): (256,) * (128,) → RuntimeError.
# The cutedsl branch has _slice_scale() but the else-branch is missing it.
# Fix: slice input_scale to local experts in the else-branch.
MODELOPT_QUANT="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py"
if grep -q 'w13_input_scale\.max(dim=-1)' "$MODELOPT_QUANT" 2>/dev/null; then
  python3 << 'PATCH_MODELOPT_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py"
with open(f) as fh:
    code = fh.read()
old = '''        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale'''
new = '''        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale
            # EP-aware slicing: input_scale has shape (num_experts,) but must match
            # weight_scale_2 which is (num_local_experts,). No-op when ep_size=1.
            if layer.moe_ep_size > 1:
                _ep_start = layer.moe_ep_rank * layer.num_local_experts
                _ep_end = _ep_start + layer.num_local_experts
                w13_input_scale = w13_input_scale[_ep_start:_ep_end]
                w2_input_scale = w2_input_scale[_ep_start:_ep_end]'''
if old not in code:
    print("modelopt_quant.py else-branch: already patched or source changed, skipping")
    sys.exit(0)
code = code.replace(old, new, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched modelopt_quant.py: EP-aware input_scale slicing in else-branch")
PATCH_MODELOPT_EOF
fi

# Patch modelopt_quant.py: CutlassMoEParams uses layer.num_experts (global=256)
# instead of layer.num_local_experts (128 with EP=2). The cutlass_moe_fp4 forward
# function then asserts num_experts == weight expert dim, which fails because
# weights are EP-sliced to num_local_experts. Fix: use num_local_experts.
if grep -q 'num_experts=layer.num_experts,  # global num experts' "$MODELOPT_QUANT" 2>/dev/null; then
  sed -i 's/num_experts=layer\.num_experts,  # global num experts/num_experts=layer.num_local_experts,  # EP-aware: use local expert count/' "$MODELOPT_QUANT"
  sed -i 's/existing_params\.num_experts != layer\.num_experts/existing_params.num_experts != layer.num_local_experts/' "$MODELOPT_QUANT"
  echo "Patched modelopt_quant.py: CutlassMoEParams uses num_local_experts for EP"
fi

# Patch cutlass_moe.py: cutlass_moe_fp4 EP correctness fix. Two independent
# bugs in the same function, both triggered by `topk_ids == -1` non-local
# sentinels that `StandardDispatcher.local_expert_mapping` writes under EP>1.
#
# BUG 1: a_map / c_map allocated with torch.empty (uninitialized). The native
# kernel prepare_moe_input.compute_arg_sorts iterates blockIdx.x over
# [0, num_experts) and only writes a_map[slot] / c_map[slot] where
# topk_ids[i] == blockIdx.x. The -1 entries match no block and leave those
# slots with torch.empty garbage. Downstream a.index_select(0, a_map) in
# _shuffle_rows_torch reads the garbage as row indices and trips torch's
# vectorized_gather_kernel bounds check — surfaces as device-side assert at
# nvfp4_blockwise_moe.cuh:78 (via next cudaMallocAsync sync point).
# Fix 1: zero-init. Zero is always a valid row index into `a`.
#
# BUG 2: topk_weights are NOT zeroed for -1 slots. Fix 1 alone eliminates the
# crash but produces garbage output. The dispatcher remaps topk_ids to local
# IDs + -1 sentinels, but leaves topk_weights carrying the original softmax
# weights. After the grouped GEMM path, shuffle_rows(c2, c_map, ...) at
# line 493 finds c_map[slot] == 0 for non-local slots (from our Fix 1
# zero-init) and reads c2[0] — the first ACTIVE expert's output for the
# first local token — into those slots. The non-local slots then go into
# `c2 * topk_weights.view(m, num_topk, 1)` carrying real-but-wrong finite
# values multiplied by real (non-zero) weights, and `sum(dim=1)` aggregates
# them into the output alongside the correct local-slot contributions.
# Fix 2: mask topk_weights where topk_ids < 0 at the start of
# cutlass_moe_fp4 — a .masked_fill before any math runs propagates through
# both the `apply_router_weight_on_input=False` path (line 496 final
# multiply) and the `=True` path (weights baked into input earlier).
#
# Two separate PATCH_*_EOF blocks so each patch's grep guard runs
# independently and failure of one doesn't prevent the other.
#
# Upstream PR #20869 is the adjacent work but only fixes the first two bugs
# in this chain (input-scale slicing + num_local_experts for CutlassMoEParams);
# it then sidesteps the third+fourth bugs here by auto-routing SM120 to
# flashinfer_cutlass. Our monkey-patches are the first real fix for the
# cutlass_moe_fp4 codepath under EP that we are aware of. See
# SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md for the full debug ordeal.
CUTLASS_MOE="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/moe/cutlass_moe.py"
if grep -q 'a_map = torch.empty((topk_ids.numel())' "$CUTLASS_MOE" 2>/dev/null; then
  python3 << 'PATCH_CUTLASS_MOE_ZEROINIT_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/moe/cutlass_moe.py"
with open(f) as fh:
    code = fh.read()
# Only patch the cutlass_moe_fp4 call site (line ~436) — there is also a
# torch.empty at ~line 145 inside cutlass_fused_experts_fp8 which is the
# FP8 MoE path (not affected by this bug; leave it alone). The second
# occurrence is the one we need, discriminated by the surrounding
# num_topk = topk_ids.shape[1] line that exists only in cutlass_moe_fp4.
old = '''    num_topk = topk_ids.shape[1]
    device = a.device
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)'''
new = '''    num_topk = topk_ids.shape[1]
    device = a.device
    # EP-aware: mask topk_weights where topk_ids == -1 (non-local sentinels
    # from StandardDispatcher.local_expert_mapping). Without this, non-local
    # slots carry real softmax weights into the final c2 * topk_weights
    # multiply and pollute the output reduction with the wrong expert's
    # values (see Fix 2 in the patch header).
    topk_weights = topk_weights.masked_fill(topk_ids < 0, 0)
    # EP-aware: zero-init instead of torch.empty. prepare_moe_input only
    # writes slots for non-(-1) topk_ids; -1 entries leave slots as garbage
    # and the downstream a.index_select(0, a_map) trips torch's bounds
    # check. Zero is a valid row index into `a`; the fake-gathered rows
    # then multiply by our newly-masked (zero) topk_weights above and
    # vanish in the reduction (see Fix 1 in the patch header).
    a_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)'''
if old not in code:
    print("cutlass_moe.py: already patched or source changed, skipping")
    sys.exit(0)
code = code.replace(old, new, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched cutlass_moe.py: a_map/c_map zero-init + topk_weights mask for EP")
PATCH_CUTLASS_MOE_ZEROINIT_EOF
fi

# Patch ModelOptModelLoader to support load_format=sharded_state (SGLang 0.5.9 bug).
# ModelOptModelLoader inherits DefaultModelLoader whose _prepare_weights() doesn't
# handle LoadFormat.SHARDED_STATE → "Unknown load_format" error. Fix: for pre-quantized
# models with sharded_state, delegate to ShardedStateLoader instead of super().load_model().
LOADER="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/loader.py"
if grep -q 'class ModelOptModelLoader' "$LOADER" 2>/dev/null; then
  python3 << 'PATCH_MODELOPT_SHARDED_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/loader.py"
with open(f) as fh:
    code = fh.read()
old = '''        if model_config._is_already_quantized():
            logger.info("Model is already quantized, loading directly...")
            # Use default loading for pre-quantized models
            return super().load_model(
                model_config=model_config, device_config=device_config
            )'''
new = '''        if model_config._is_already_quantized():
            logger.info("Model is already quantized, loading directly...")
            # Sharded state: delegate to ShardedStateLoader (which calls
            # process_weights_after_loading and loads per-rank shard files).
            # DefaultModelLoader._prepare_weights doesn't handle SHARDED_STATE.
            if self.load_config.load_format == LoadFormat.SHARDED_STATE:
                logger.info("Using ShardedStateLoader for pre-quantized sharded model")
                _sharded_loader = ShardedStateLoader(self.load_config)
                return _sharded_loader.load_model(
                    model_config=model_config, device_config=device_config
                )
            # Use default loading for pre-quantized models
            return super().load_model(
                model_config=model_config, device_config=device_config
            )'''
if old not in code:
    print("ModelOptModelLoader: already patched or source changed, skipping")
    sys.exit(0)
code = code.replace(old, new, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched ModelOptModelLoader: sharded_state support for pre-quantized models")
PATCH_MODELOPT_SHARDED_EOF
fi

# Patch MiniMaxM2ForCausalLM: add set_embed_and_head for NEXTN speculative decoding.
# The model has get_embed_and_head but is missing the setter, which eagle_worker.py
# calls to share the target model's embed/head weights with the draft model.
# Every other NEXTN-capable model (DeepSeek, GLM, Llama) has this method.
MINIMAX_M2="/usr/local/lib/python3.12/dist-packages/sglang/srt/models/minimax_m2.py"
if [ -f "$MINIMAX_M2" ] && grep -q 'def get_embed_and_head' "$MINIMAX_M2" && ! grep -q 'def set_embed_and_head' "$MINIMAX_M2"; then
  python3 << 'PATCH_MINIMAX_NEXTN_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/minimax_m2.py"
with open(f) as fh:
    code = fh.read()
old = "    def get_embed_and_head(self):"
new = """    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed_and_head(self):"""
code = code.replace(old, new, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched MiniMaxM2ForCausalLM: added set_embed_and_head for NEXTN speculative decoding")
PATCH_MINIMAX_NEXTN_EOF
else
  echo "MiniMax NEXTN patch: not needed or already applied, skipping"
fi

# Patch DeepseekV3Config.kv_lora_rank to allow None for the DeepSeek-V4-Flash
# variant (config.json has kv_lora_rank: null — Flash uses q-LoRA + o-LoRA + GQA,
# NO MLA KV compression). The class sglang actually instantiates for model_type
# "deepseek_v4" is _DeepseekV4ConfigAlias in sglang/srt/utils/hf_transformers/
# common.py, which SUBCLASSES transformers' DeepseekV3Config — so the strict
# dataclass field `kv_lora_rank: int` (and its huggingface_hub @strict validator)
# is declared in transformers/models/deepseek_v3/configuration_deepseek_v3.py,
# NOT in sglang's own configs/deepseek_v4.py (that file is never used for this
# model_type). Under transformers 5.x the null value fails at startup with:
#   StrictDataclassFieldValidationError: Field 'kv_lora_rank' expected int,
#   got NoneType (value: None)
# Widening the annotation to `int | None` BEFORE import makes @strict build a
# Union validator that accepts None. This is a SAFE widening: DeepSeek-V3 / V3.2
# / Kimi-K2 (the other models sharing this config) always supply an int, so they
# are unaffected; only V4-Flash's null now passes. We keep None rather than
# coercing to an int (an int would push modeling onto the MLA KV-LoRA path the
# Flash weights don't have). pyc is timestamp-invalidated, so the edit takes on
# reimport. NOTE: clears the config-parse blocker only — Flash serving may still
# hit further upstream issues downstream (sglang #25165 / #23743).
DEEPSEEK_V3_CFG="/usr/local/lib/python3.12/dist-packages/transformers/models/deepseek_v3/configuration_deepseek_v3.py"
if [ -f "$DEEPSEEK_V3_CFG" ] && grep -q 'kv_lora_rank: int = 512' "$DEEPSEEK_V3_CFG"; then
  python3 << 'PATCH_DSV4_KVLORA_EOF'
f = "/usr/local/lib/python3.12/dist-packages/transformers/models/deepseek_v3/configuration_deepseek_v3.py"
with open(f) as fh:
    code = fh.read()
old = "    kv_lora_rank: int = 512"
new = "    kv_lora_rank: int | None = 512"
if old not in code:
    print("DeepseekV3Config kv_lora_rank patch: marker not found, skipping")
else:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched DeepseekV3Config: kv_lora_rank now int|None (DeepSeek-V4-Flash kv_lora_rank=null support)")
PATCH_DSV4_KVLORA_EOF
else
  echo "DeepseekV3Config kv_lora_rank patch: not needed or already applied, skipping"
fi

# DeepSeek-V4-Flash C4-indexer torch-fallback seq_lens shape fix (SM121).
# With SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 (our profile — DeepGEMM ships no sm_121
# paged_mqa_logits kernel, §6) forward_c4_indexer() unconditionally unsqueezes
# c4_seq_lens to 2-D (batch,1) for the deep_gemm/tilelang kernels, but the torch
# fallback fp8_paged_mqa_logits_torch() asserts 1-D `seq_lens.shape==(batch_size,)`
# → AssertionError on the FIRST multi-token forward (not just EAGLE). Squeeze a
# trailing singleton before the assert (no-op when already 1-D). This is sglang's
# own gap: the vendored 0xSero _patch_sglang_indexer_fallbacks targets the OLD
# nsa/compressed module paths and does NOT apply on v0.5.12.post1 (the indexer
# moved to attention/dsv4/indexer.py). See UPSTREAM_DSV4_BUGS.md §6.
DSV4_INDEXER="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/indexer.py"
if [ -f "$DSV4_INDEXER" ] && grep -q '    assert seq_lens.shape == (batch_size,)' "$DSV4_INDEXER"; then
  python3 << 'PATCH_DSV4_INDEXER_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/indexer.py"
with open(f) as fh:
    code = fh.read()
old = "    assert seq_lens.shape == (batch_size,)\n"
new = ("    if seq_lens.dim() == 2 and seq_lens.shape[-1] == 1:\n"
       "        seq_lens = seq_lens.squeeze(-1)\n"
       "    assert seq_lens.shape == (batch_size,)\n")
if new in code:
    print("DSV4 indexer seq_lens patch: already applied, skipping")
elif old not in code:
    print("DSV4 indexer seq_lens patch: marker not found, skipping")
else:
    code = code.replace(old, new, 1)
    with open(f, "w") as fh:
        fh.write(code)
    print("Patched dsv4/indexer.py: fp8_paged_mqa_logits_torch tolerates 2-D seq_lens")
PATCH_DSV4_INDEXER_EOF
else
  echo "DSV4 indexer seq_lens patch: not needed or already applied, skipping"
fi

# fastsafetensors loader: make it usable on multi-node TP + no-GDS GB10 so the
# weight load STREAMS disk→device through a bounded bounce buffer instead of
# accumulating full shards in host memory (which is what swaps — confirmed by
# memray/smaps: top allocator _load_file weight_utils.py:1060, swapped mapping
# = safetensors-mmap). sglang's fastsafetensors_weights_iterator does a WORLD
# collective load (→ Gloo connectFullMesh timeout across our 4 nodes) onto
# cuda:{world_rank} (invalid on 1-GPU worker nodes). Rewrite to:
#   - SingleGroup() : each rank loads its files independently (no collective)
#   - device "cuda" : the local current device (not cuda:{world_rank})
#   - nogds=True    : 16 MB bounce-buffer streaming (no GPU Direct Storage here)
# TP slicing is unchanged — the per-param weight_loader slices the full tensors,
# exactly as the normal safetensors iterator yields them. Inert unless
# load_format=fastsafetensors. See UPSTREAM_DSV4_BUGS.md.
FST_F="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
if [ -f "$FST_F" ] && grep -q 'device = torch.device(f"cuda:{rank}")' "$FST_F"; then
  python3 << 'PATCH_DSV4_FST_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
code = open(f).read()
old1 = (
    '    if torch.distributed.is_initialized():\n'
    '        pg = torch.distributed.group.WORLD\n'
    '    else:\n'
    '        pg = SingleGroup()\n'
    '\n'
    '    try:\n'
    '        rank = pg.rank()\n'
    '    except Exception:\n'
    '        rank = 0\n'
    '\n'
    '    device = torch.device(f"cuda:{rank}")'
)
new1 = (
    '    # dgxarley: per-rank independent load (no WORLD collective → no Gloo\n'
    '    # connectFullMesh timeout across nodes) onto the LOCAL device (explicit\n'
    '    # index — fastsafetensors set_device rejects bare "cuda"), nogds\n'
    '    # bounce-buffer streaming (no GDS on GB10) → no host full-shard pileup.\n'
    '    pg = SingleGroup()\n'
    '    device = torch.device("cuda", torch.cuda.current_device())'
)
old2 = "        loader = SafeTensorsFileLoader(pg, device)"
new2 = "        loader = SafeTensorsFileLoader(pg, device, nogds=True)"
changed = False
if "torch.cuda.current_device())" in code and "pg = SingleGroup()\n    device" in code:
    print("fastsafetensors patch: already applied, skipping")
else:
    if old1 in code:
        code = code.replace(old1, new1, 1); changed = True
    else:
        print("fastsafetensors patch: pg/device marker not found")
    if old2 in code:
        code = code.replace(old2, new2, 1); changed = True
    else:
        print("fastsafetensors patch: SafeTensorsFileLoader marker not found")
    if changed:
        open(f, "w").write(code)
        print("Patched fastsafetensors_weights_iterator: SingleGroup + local device + nogds")
PATCH_DSV4_FST_EOF
else
  echo "fastsafetensors patch: not needed or already applied, skipping"
fi

# DeepSeek-V4-Flash FlashMLA sparse-decode hook activation (sm_121a / GB10).
# The image bakes deepseek_v4_kernel, but its sitecustomize.py is SHADOWED:
# Ubuntu ships /usr/lib/python3.12/sitecustomize.py (apport) earlier on sys.path,
# and Python imports only the FIRST sitecustomize it finds — so the hook never
# ran and V4-Flash died with "Unsupported architecture for sparse decode fwd".
# A .pth is immune: site.py runs EVERY import-line in EVERY .pth across ALL site
# dirs (no "first wins"), in main AND every spawned sglang worker.
#
# CRITICAL — install ONLY the flash_mla wrapper (_patch_flash_mla_pkg), NOT the
# kernel's patch_flash_mla()/install(). install() also runs
# _patch_sglang_indexer_fallbacks(), which imports sglang…nsa.tilelang_kernel →
# loads tilelang's libcudart_stub.so. At site-init that stub loads BEFORE
# flashinfer.comm, so flashinfer's find_loaded_library("libcudart") grabs the
# stub (no cudaDeviceReset) → hard AttributeError at import (NOT caught by
# sglang's `except ImportError`). sglang imports tilelang itself LATER (after
# flashinfer), so the indexer fallback is not ours to bootstrap. Guarded: a
# broken kernel just falls through to stock flash_mla. Idempotent (also written
# by dockerfile-dsv4-flashmla.patch on rebuilt images). See UPSTREAM_DSV4_BUGS.md §7.
DSV4_DP="/usr/local/lib/python3.12/dist-packages"
if [ -d "$DSV4_DP/deepseek_v4_kernel" ]; then
  cat > "$DSV4_DP/dsv4_autopatch.py" <<'DSV4_AUTOPATCH_EOF'
import os, sys
if os.environ.get("DSV4_KERNEL_DISABLE", "0") not in ("1", "true", "yes"):
    try:
        from deepseek_v4_kernel._patch import _patch_flash_mla_pkg
        _patch_flash_mla_pkg()
    except Exception as exc:
        print("[dsv4_autopatch] flash_mla patch skipped:", exc, file=sys.stderr)
DSV4_AUTOPATCH_EOF
  echo 'import dsv4_autopatch' > "$DSV4_DP/zz_dsv4_autopatch.pth"
  echo "Installed DSV4 FlashMLA autopatch (flash_mla wrapper only): $DSV4_DP/zz_dsv4_autopatch.pth"
else
  echo "DSV4 FlashMLA kernel not present in image, skipping autopatch"
fi

# DSV4 unified-memory load probe (diagnostic, gated by SGLANG_MEMPROBE=1).
# Copies /scripts/dsv4_memprobe.py into dist-packages and drops a .pth so every
# sglang worker arms it at startup (env DSV4_MEMPROBE=1, read by the module). It
# brackets ModelRunner.load_model / init_memory_pool / cuda-graph capture and the
# per-call cuda-alloc delta of Fp8(MoE|Linear)Method.process_weights_after_loading,
# plus a 0.2s ticker — to find which post-load action doubles GB10 unified memory.
# Output → stderr → Loki (grep "[memprobe"). Inert unless SGLANG_MEMPROBE=1.
if [ "${SGLANG_MEMPROBE:-0}" = "1" ] && [ -f /scripts/dsv4_memprobe.py ]; then
  cp /scripts/dsv4_memprobe.py "$DSV4_DP/dsv4_memprobe.py"
  echo 'import dsv4_memprobe' > "$DSV4_DP/zz_dsv4_memprobe.pth"
  export DSV4_MEMPROBE=1
  # memray for the HOST native+mmap allocation profiler (the probe uses it if
  # importable). Best-effort install; absence just disables host profiling.
  pip install -q memray >/dev/null 2>&1 && echo "memprobe: memray installed" || echo "memprobe: memray install failed (host profiling off)"
  echo "Installed DSV4 memprobe (DSV4_MEMPROBE=1): $DSV4_DP/zz_dsv4_memprobe.pth"
else
  rm -f "$DSV4_DP/zz_dsv4_memprobe.pth" "$DSV4_DP/dsv4_memprobe.py" 2>/dev/null || true
fi

# Manual pipeline-stage layer boundaries. SGLang reads SGLANG_PP_LAYER_PARTITION
# directly from the env (os.getenv in get_pp_indices), NOT as a CLI flag. We pass
# our own PP_LAYER_PARTITION and promote it ONLY when non-empty: an empty value
# would make SGLang parse int("") and crash. Empty = SGLang default even split.
if [ -n "${PP_LAYER_PARTITION:-}" ]; then
  export SGLANG_PP_LAYER_PARTITION="$PP_LAYER_PARTITION"
  echo "PP layer partition (manual): SGLANG_PP_LAYER_PARTITION=$SGLANG_PP_LAYER_PARTITION"
fi

args=(
  tini -s --
  python3 -m sglang.launch_server
  --model-path "$model_path"
  --context-length "$SGLANG_CONTEXT_LENGTH"
  --kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE"
  --mem-fraction-static "$SGLANG_MEM_FRACTION"
  --tp-size "$TP"
  --pp-size "$PP"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --nccl-init-addr "${QSFP_IP_SPARK1}:${NCCL_PORT}"
  --port "$SGLANG_PORT"
)
# PP async micro-batching: overlap forward passes across pipeline stages.
if [ -n "$PP_ASYNC_BATCH_DEPTH" ] && [ "$PP_ASYNC_BATCH_DEPTH" != "0" ]; then
  args+=(--pp-async-batch-depth "$PP_ASYNC_BATCH_DEPTH")
fi
# Expert parallelism: partitions the TP group for MoE layers.
# EP=TP → MoE uses all-to-all, attention stays tensor-parallel.
if [ -n "$EP" ] && [ "$EP" != "1" ]; then
  args+=(--expert-parallel-size "$EP")
fi
if [ "$SGLANG_ENABLE_EPLB" = "true" ]; then
  args+=(--enable-eplb)
fi
if [ "$SGLANG_ENABLE_EXPERT_DISTRIBUTION_METRICS" = "true" ]; then
  args+=(--enable-expert-distribution-metrics)
fi
# Prometheus exporter: serves /metrics on the HTTP server port (only effective on
# the head; workers run no HTTP server). Gated per-instance via SGLANG_ENABLE_METRICS.
if [ "$SGLANG_ENABLE_METRICS" = "true" ]; then
  args+=(--enable-metrics)
fi
if [ -n "$SGLANG_HOST" ]; then
  args+=(--host "$SGLANG_HOST")
fi
if [ -n "$SGLANG_LOAD_FORMAT" ] && [ "$SGLANG_LOAD_FORMAT" != "auto" ]; then
  args+=(--load-format "$SGLANG_LOAD_FORMAT")
fi
# Diagnostic/tuning knob for the weight-load read-buffer concurrency. The default
# loader uses buffered_multi_thread_safetensors_weights_iterator with 8 workers
# (DEFAULT_NUM_THREADS) — up to 8 shards buffered at once on top of the resident
# weights. Pass e.g. {"enable_multithread_load": false} (no buffering pool) or
# {"num_threads": 1} to shrink the load-time source-buffer peak. NOTE: prefetch
# (weight_loader_prefetch_checkpoints) is OFF by default, so its num_threads is
# inert — THIS is the active buffering control.
if [ -n "$SGLANG_MODEL_LOADER_EXTRA_CONFIG" ]; then
  args+=(--model-loader-extra-config "$SGLANG_MODEL_LOADER_EXTRA_CONFIG")
fi
if [ -n "$SGLANG_QUANTIZATION" ]; then
  args+=(--quantization "$SGLANG_QUANTIZATION")
fi
if [ "$SGLANG_TRUST_REMOTE_CODE" = "true" ]; then
  args+=(--trust-remote-code)
fi
if [ -n "$SGLANG_JSON_MODEL_OVERRIDE_ARGS" ]; then
  args+=(--json-model-override-args "$SGLANG_JSON_MODEL_OVERRIDE_ARGS")
fi
if [ -n "$SGLANG_MOE_RUNNER_BACKEND" ]; then
  args+=(--moe-runner-backend "$SGLANG_MOE_RUNNER_BACKEND")
fi
if [ -n "$SGLANG_REASONING_PARSER" ]; then
  args+=(--reasoning-parser "$SGLANG_REASONING_PARSER")
fi
if [ -n "$SGLANG_TOOL_CALL_PARSER" ]; then
  args+=(--tool-call-parser "$SGLANG_TOOL_CALL_PARSER")
fi
if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  args+=(--speculative-algo "$SGLANG_SPECULATIVE_ALGO")
  args+=(--speculative-num-steps "$SGLANG_SPECULATIVE_NUM_STEPS")
  args+=(--speculative-eagle-topk "$SGLANG_SPECULATIVE_EAGLE_TOPK")
  args+=(--speculative-num-draft-tokens "$SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS")
  # External draft model (EAGLE/EAGLE3): use speculative_draft_model_path from profile.
  if [ -n "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH" ]; then
    args+=(--speculative-draft-model-path "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH")
  fi
  # Draft model quantization override. By default SGLang inherits the target's
  # quantization for the draft, which breaks when the target is modelopt-
  # quantized (NVFP4) but the draft ships as plain BF16 (typical for external
  # EAGLE3 drafts). Setting "unquant" forces the draft to load without
  # quantization, bypassing the modelopt loader and its Qwen3MoE state-dict
  # shape mismatch against the single-layer EAGLE3 checkpoint.
  if [ -n "$SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION" ]; then
    args+=(--speculative-draft-model-quantization "$SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION")
  fi
  # DSV4/SM121: the nextn draft MoE hardcodes an sm100 trtllm kernel that crashes
  # on GB10. Force marlin (SM80+) via this arg — requires the modelopt_quant
  # marlin-branch in sglang-dsv4-nvfp4-pr25820.patch (image rebuild).
  if [ -n "$SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND" ]; then
    args+=(--speculative-moe-runner-backend "$SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND")
  fi
  # WORKAROUND (SGLang 0.5.9): sharded_state + speculative decoding crash.
  # The draft model's ModelRunner inherits load_format=sharded_state from
  # server_args. ShardedStateLoader then fails with KeyError because the
  # per-rank shard files don't contain the draft/MTP model weight keys.
  # Fix: force auto load format for the draft model and point it to the
  # original HF model ID (resolved from HF cache) instead of the shard dir.
  # See SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md for details.
  if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
    args+=(--speculative-draft-load-format auto)
    # Only override draft model path if not already set by profile
    if [ -z "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH" ]; then
      args+=(--speculative-draft-model-path "$SGLANG_MODEL")
    fi
  fi
  # Adaptive Spec V2 (SGLang ≥0.5.12, PR #23336). Dynamically retunes
  # num_steps / num_draft_tokens at runtime. Only meaningful with
  # EAGLE/EAGLE3 + speculative_eagle_topk=1 — SGLang silently disables
  # otherwise (adaptive_unsupported_reason() in
  # srt/speculative/adaptive_spec_params.py). NEXTN is NOT supported.
  if [ "$SGLANG_SPECULATIVE_ADAPTIVE" = "true" ]; then
    args+=(--speculative-adaptive)
    if [ -n "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" ] \
        && [ "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" != "{}" ] \
        && [ "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" != "null" ]; then
      printf '%s' "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" \
        > /tmp/speculative_adaptive_config.json
      args+=(--speculative-adaptive-config /tmp/speculative_adaptive_config.json)
    fi
  fi
fi
if [ -n "$SGLANG_MAMBA_SCHEDULER_STRATEGY" ]; then
  args+=(--mamba-scheduler-strategy "$SGLANG_MAMBA_SCHEDULER_STRATEGY")
fi
# Mamba state-cache pool sizing (hybrid SSM models). Empty = SGLang auto-fit.
# max_mamba_cache_size // mamba_ratio is the parallelism ceiling on hybrid models.
if [ -n "$SGLANG_MAMBA_FULL_MEMORY_RATIO" ]; then
  args+=(--mamba-full-memory-ratio "$SGLANG_MAMBA_FULL_MEMORY_RATIO")
fi
if [ -n "$SGLANG_MAX_MAMBA_CACHE_SIZE" ] && [ "$SGLANG_MAX_MAMBA_CACHE_SIZE" != "0" ]; then
  args+=(--max-mamba-cache-size "$SGLANG_MAX_MAMBA_CACHE_SIZE")
fi
if [ -n "$SGLANG_MAX_RUNNING_REQUESTS" ] && [ "$SGLANG_MAX_RUNNING_REQUESTS" != "0" ]; then
  args+=(--max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
fi
# Absolute KV-cache pool cap (tokens). Unset/0 -> sized by mem_fraction_static.
# Used to pin a co-located instance's memory footprint deterministically.
if [ -n "$SGLANG_MAX_TOTAL_TOKENS" ] && [ "$SGLANG_MAX_TOTAL_TOKENS" != "0" ]; then
  args+=(--max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS")
fi
if [ -n "$SGLANG_SCHEDULE_POLICY" ]; then
  args+=(--schedule-policy "$SGLANG_SCHEDULE_POLICY")
fi
if [ -n "$SGLANG_CHUNKED_PREFILL_SIZE" ] && [ "$SGLANG_CHUNKED_PREFILL_SIZE" != "0" ]; then
  args+=(--chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE")
fi
if [ -n "$SGLANG_DIST_TIMEOUT" ]; then
  args+=(--dist-timeout "$SGLANG_DIST_TIMEOUT")
fi
if [ -n "$SGLANG_WATCHDOG_TIMEOUT" ]; then
  args+=(--watchdog-timeout "$SGLANG_WATCHDOG_TIMEOUT")
fi
if [ -n "$SGLANG_ATTENTION_BACKEND" ]; then
  args+=(--attention-backend "$SGLANG_ATTENTION_BACKEND")
fi
# Diffusion-LLM (dLLM) decode path — when SGLANG_DLLM_ALGORITHM is set (e.g.
# "Gemma4Renoise" for DiffusionGemma) launch_server runs the block-diffusion
# scheduler instead of the autoregressive one. SGLang's _handle_dllm_inference
# auto-forces triton attention, eager mode (cuda graph disabled), and unchunked
# prefill for Gemma4Renoise, so the autoregressive cuda-graph / attention flags
# above are overridden internally. Empty for all autoregressive models → no flag
# is added, zero impact. Requires the 0.5.13-gemmadiffusion image (PR #28054
# baked); other images reject --dllm-algorithm for Gemma4.
if [ -n "$SGLANG_DLLM_ALGORITHM" ]; then
  args+=(--dllm-algorithm "$SGLANG_DLLM_ALGORITHM")
fi
# Server log level. Empty → no flag → SGLang's built-in default ('info'). Set
# SGLANG_LOG_LEVEL=debug (via sglang_log_level in defaults/main.yml or a model
# profile) to surface SGLang's logger.debug diagnostics — notably the Frozen-KV
# MTP draft-bind skip ("Draft model <class> does not implement ... skipping
# frozen-kv bind."), which names the class a Gemma-4 assistant draft loads as.
if [ -n "$SGLANG_LOG_LEVEL" ]; then
  args+=(--log-level "$SGLANG_LOG_LEVEL")
fi
if [ -n "$SGLANG_FP8_GEMM_RUNNER_BACKEND" ] && [ "$SGLANG_FP8_GEMM_RUNNER_BACKEND" != "auto" ]; then
  args+=(--fp8-gemm-backend "$SGLANG_FP8_GEMM_RUNNER_BACKEND")
fi
if [ -n "$SGLANG_FP4_GEMM_BACKEND" ] && [ "$SGLANG_FP4_GEMM_BACKEND" != "auto" ]; then
  args+=(--fp4-gemm-backend "$SGLANG_FP4_GEMM_BACKEND")
fi
if [ "$SGLANG_DISABLE_CUDA_GRAPH" = "true" ] || [ "$SGLANG_CUDA_GRAPH_MAX_BS" = "0" ]; then
  args+=(--disable-cuda-graph)
elif [ -n "$SGLANG_CUDA_GRAPH_MAX_BS" ] && [ "$SGLANG_CUDA_GRAPH_MAX_BS" != "256" ]; then
  args+=(--cuda-graph-max-bs "$SGLANG_CUDA_GRAPH_MAX_BS")
fi
if [ "$SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH" = "true" ]; then
  args+=(--disable-piecewise-cuda-graph)
fi
if [ "$SGLANG_WEIGHT_LOADER_DISABLE_MMAP" = "true" ]; then
  args+=(--weight-loader-disable-mmap)
fi
if [ "$SGLANG_WEIGHT_LOADER_DROP_CACHE_AFTER_LOAD" = "true" ]; then
  args+=(--weight-loader-drop-cache-after-load)
fi
if [ "$SGLANG_DISABLE_OVERLAP_SCHEDULE" = "true" ]; then
  args+=(--disable-overlap-schedule)
fi
if [ "$SGLANG_DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER" = "true" ]; then
  args+=(--disable-flashinfer-cutlass-moe-fp4-allgather)
fi
if [ -n "$SGLANG_SERVED_MODEL_NAME" ]; then
  args+=(--served-model-name "$SGLANG_SERVED_MODEL_NAME")
fi
# Chat template kwargs (enable_thinking, etc.)
# These control Jinja2 chat template rendering — NOT sampling parameters.
# Sampling defaults (temperature, top_p, ...) are set via generation_config.json
# overlay above, because SGLang has no CLI flags for individual sampling params.
# NOTE: thinking_budget does NOT go here — it uses SGLang's custom logit processor
# system (--enable-custom-logit-processor), not the chat template.
if [ -n "$SGLANG_CHAT_TEMPLATE_KWARGS" ] && [ "$SGLANG_CHAT_TEMPLATE_KWARGS" != "{}" ]; then
  args+=(--chat-template-kwargs "$SGLANG_CHAT_TEMPLATE_KWARGS")
fi
# Enable custom logit processors (required for per-request thinking_budget via
# Qwen3ThinkingBudgetLogitProcessor). Safe to always enable — no-op if unused.
args+=(--enable-custom-logit-processor)

# Echo the exact launch command + relevant ENV to stdout so the head/worker
# pod logs (and Loki) capture them verbatim. printf '%q ' produces a shell-safe,
# copy-pasteable form (handles spaces, quotes, JSON args). Logged before exec
# so it appears even if the server crashes during startup.
#
# ENV filter: SGLANG_*, NCCL_*, FLASHINFER_*, TORCH*, CUDA_*, HF_*, plus a few
# named knobs that gate behavior at runtime (mamba/spec/JIT). Excludes generic
# vars (PATH, HOME, K8S_*, KUBERNETES_*, POD_*) to keep output focused.
printf '=== sglang launch ENV (filtered, secrets redacted) ===\n'
env | grep -E '^(SGLANG_|NCCL_|FLASHINFER_|TORCH(_|INDUCTOR_)|CUDA_|HF_|GLOO_|UCX_|RDMAV_|MASTER_|RANK=|WORLD_SIZE=|LOCAL_RANK=|NODE_RANK=|NNODES=|DIST_INIT_ADDR=|MAMBA_|SPEC_V2)' \
  | grep -vE '^(SGLANG_EXPECTED_IMAGE_PATTERN=|HF_HUB_OFFLINE_PATH=)' \
  | sed -E 's/^([A-Z_0-9]*(TOKEN|SECRET|KEY|PASSWORD|PASS|API|CREDENTIAL)[A-Z_0-9]*)=.*/\1=***REDACTED***/' \
  | LC_ALL=C sort
printf '=== end sglang launch ENV ===\n'

printf '=== sglang launch command (%d args) ===\n' "${#args[@]}"
printf '%q ' "${args[@]}"
printf '\n=== end sglang launch command ===\n'

exec "${args[@]}"
