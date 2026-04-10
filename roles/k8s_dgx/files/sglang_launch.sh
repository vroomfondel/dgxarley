#!/bin/bash
set -e

# Install ping for ARP priming (not included in sglang image)
apt-get update -qq && apt-get install -y -qq tini iproute2 iputils-ping net-tools curl ethtool >/dev/null 2>&1

# GLM-5 specific: transformers upgrade + mem_get_info patch.
# Only needed for glm_moe_dsa models — skip for MiniMax, Qwen, etc.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]]; then
  echo "GLM-5 model detected — applying GLM-5 specific patches..."

  # accelerate: required by ModelOptModelLoader for GLM-5-NVFP4.
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate

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

# Version gate: warn if the container image changed — patches below may need review.
# Dev builds report __version__=0.0.0 (no setuptools-scm), so we check the image
# tag (injected as SGLANG_IMAGE env var by Ansible) instead of the Python version.
# The grep guards still prevent patching if the target code has changed.
SGLANG_EXPECTED_IMAGE="scitrera/dgx-spark-sglang:0.5.10"
if [ -n "$SGLANG_IMAGE" ] && [ "$SGLANG_IMAGE" != "$SGLANG_EXPECTED_IMAGE" ]; then
  echo "WARNING: SGLang image changed (expected ${SGLANG_EXPECTED_IMAGE}, got ${SGLANG_IMAGE})."
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
if [ -n "$SGLANG_HOST" ]; then
  args+=(--host "$SGLANG_HOST")
fi
if [ -n "$SGLANG_LOAD_FORMAT" ] && [ "$SGLANG_LOAD_FORMAT" != "auto" ]; then
  args+=(--load-format "$SGLANG_LOAD_FORMAT")
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
fi
if [ -n "$SGLANG_MAX_RUNNING_REQUESTS" ] && [ "$SGLANG_MAX_RUNNING_REQUESTS" != "0" ]; then
  args+=(--max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
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
exec "${args[@]}"
