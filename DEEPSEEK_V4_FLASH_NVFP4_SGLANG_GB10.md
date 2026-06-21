# SGLang + DeepSeek-V4-Flash-NVFP4 on 4× DGX Spark GB10 — fixes, patches, and interim numbers

**TL;DR:** `nvidia/DeepSeek-V4-Flash-NVFP4` runs under **SGLang** on four DGX Spark (GB10 / sm_121) nodes. The vLLM path is better documented elsewhere; this document records what it took on the SGLang side and links the patches so the setup is reproducible. The short version: three SM121-specific fixes are needed before the model will even boot with MTP, and a fourth is needed to escape a hard decode throughput floor caused by the missing DeepGEMM sm121 kernel.

All patches referenced below live in [`scripts/patches/`](scripts/patches/); the serving config lives in the `nvidia/DeepSeek-V4-Flash-NVFP4` model profile under `roles/k8s_dgx/model_profiles/`.

---

## Setup

| Item | Value |
|---|---|
| Hardware | 4× DGX Spark (ASUS Ascent GX10), 120 GB unified memory / node |
| Arch | sm_121 (GB10, Grace ARM64) |
| Interconnect | QSFP56-200G, RoCE v2 (NCCL over SR-IOV VFs) |
| Engine | SGLang v0.5.13, custom image `xomoxcc/dgx-spark-sglang:0.5.13-sm121` |
| Model | `nvidia/DeepSeek-V4-Flash-NVFP4` (MIXED_PRECISION: NVFP4 routed experts, block-FP8 base, MTP head unquantized) |
| Serving config | TP=4, EP=4, context 262 144, CUDA-graph on (`max_bs=32`), EAGLE-MTP |
| Orchestration | K3s cluster (SGLang head on spark1, workers on spark2-4) |

The base image is the `sgl-kernel` SM121 build from `scitrera/cuda-containers`; the model-serving patches are layered on top. The model is a direct HuggingFace pull; `hf_quant_config.json` uses `MIXED_PRECISION` with NVFP4 on all `layers.*.ffn.experts` and block-FP8 on the base — identical layout to `nvidia/DeepSeek-V4-Pro-NVFP4`.

---

## SM121-specific fixes

### Fix 1 — PR #25820 IndentationError

SGLang does not yet natively support the NVFP4 MoE path for the DeepSeek-V4 architecture. The unmerged PR [sgl-project/sglang#25820](https://github.com/sgl-project/sglang/pull/25820) adds it via `HybridFp8NvFp4Config`. We cherry-pick it as a patch in the image build.

On rebase the patch had a one-line placement error: the `gemm1_clamp_limit=` kwarg was placed _after_ the closing `)` of the constructor call rather than _inside_ it. This causes:

```
IndentationError: unexpected indent (modelopt_quant.py, line 2337)
```

…at import time → immediate `CrashLoopBackOff`. Fix: move the kwarg line before the `)`. The corrected patch is [`scripts/patches/sglang-dsv4-nvfp4-pr25820.patch`](scripts/patches/sglang-dsv4-nvfp4-pr25820.patch).

### Fix 2 — MTP/EAGLE draft MoE crashes on GB10

With EAGLE-MTP enabled, the nextn/draft MoE layers go through `HybridFp8NvFp4Config.get_quant_method()`. Upstream PR #25820 hardcodes `Mxfp4FlashinferTrtllmMoEMethod` for those layers. That method dispatches a TVM-compiled kernel built for SM100 (`bmm_..._sm100f`). On GB10:

```
tvm.error.InternalError: Check failed: ... sm100f kernel not supported on sm121
```

**Fix:** add a `marlin` branch in `get_quant_method()` that returns `Mxfp4MarlinMoEMethod` when `speculative_moe_runner_backend == "marlin"`. Marlin targets SM80+, runs correctly on GB10.

SGLang flag: `--speculative-moe-runner-backend marlin`

Diff (added to [`scripts/patches/sglang-dsv4-nvfp4-pr25820.patch`](scripts/patches/sglang-dsv4-nvfp4-pr25820.patch)):

```diff
+            if get_moe_runner_backend().is_marlin():
+                return Mxfp4MarlinMoEMethod(self, layer)
             return Mxfp4FlashinferTrtllmMoEMethod(self, layer)
```

Also important: `speculative_algorithm` **must be `EAGLE`**, not `NEXTN`. The `deepseek_v4_hook.py` asserts this at arg validation — `NEXTN` will crash before the model loads. `spec_v2` is auto-managed by the hook; do not set it manually.

### Fix 3 — CUDA-graph max batch size

Default `cuda_graph_max_bs=8` causes SGLang to fall back to **eager** decode above 8 concurrent requests, and throughput collapses (we measured ~6.5 tok/s aggregate at 16 concurrent vs ~18 on-graph). Raising to 32 keeps decode on the CUDA graph across typical concurrency. (Separately, the `cutlass_moe_fp4` eager path is a documented NaN risk on sm121 — another reason to stay on-graph.)

```
SGLANG_CUDA_GRAPH_MAX_BS=32
```

### Fix 4 — The decode throughput floor (TileLang DSA indexer)

This was the hardest to find. After fixing the above, throughput was stuck at **~13.5 tok/s aggregate regardless of concurrency**. The culprit:

DeepSeek-V4-Flash uses a DSA (Lightning-Indexer) attention mechanism. The hot kernel is `fp8_paged_mqa_logits`, which runs on **every decode step across all 20 DSA layers, outside the CUDA graph**. SGLang normally dispatches this to DeepGEMM. But DeepGEMM hard-blocks GB10:

1. `deep_gemm/_C.so` has an SM allowlist (`SM100`, `SM120`) — SM121 returns `Unsupported architecture (attention.hpp:219)`.
2. The SM100 kernel uses `tcgen05`, UMMA, and TMEM instructions absent on GB10; no sm121 impl exists in `deepseek-ai/DeepGEMM` main.
3. [DeepGEMM PR #324](https://github.com/deepseek-ai/DeepGEMM/pull/324) ("feat: add sm120 support") exists and explicitly targets sm120/sm121, but it is not merged as of this writing and is against a `nv_dev` branch of the deepseek-ai fork — not the `sgl-project/DeepGEMM` fork used by SGLang.

SGLang falls back to a **pure-torch dispatch per step** — outside the CUDA graph, taking ~18 ms/step for 20 layers. This creates a hard ~18 tok/s ceiling regardless of batch size or MTP.

**The fix: SGLang's TileLang DSA indexer.**

SGLang has an alternative dispatch path: `SGLANG_OPT_USE_TILELANG_INDEXER=1`. This uses a TileLang-compiled kernel for `fp8_paged_mqa_logits` that:
- runs correctly on GB10 (no SM allowlist)
- is CUDA-graph-capturable
- is bit-correct vs the torch reference

However, the kernel was written for TileLang 0.1.6/0.1.7, and the image ships TileLang 0.1.8. Attempting to compile:

```
tvm.error.InternalError: Check failed: (buffer->shape.size() >= 2):
Buffer "k_smem_u8" shape [8192] should be ≥2D (inject_permuted_layout)
```

TileLang 0.1.8's `inject_permuted_layout` rejects 1D shared-buffer allocations. The fix is three lines in `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`:

```diff
--- a/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py
+++ b/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py
@@ ... @@
-                k_smem_u8 = T.alloc_shared((B * D,), UINT8)
+                k_smem_u8 = T.alloc_shared((1, B * D), UINT8)  # TL0.1.8: 2D required
                 T.copy(kvcache_u8[page, 0:SCALE_OFFSET], k_smem_u8)
                 k_smem = T.view(k_smem_u8, (B, D), FP8)
-                k_s_smem_u8 = T.alloc_shared((B * 4,), UINT8)
+                k_s_smem_u8 = T.alloc_shared((1, B * 4), UINT8)  # TL0.1.8: 2D required
                 T.copy(kvcache_u8[page, SCALE_OFFSET:BLOCK_BYTES], k_s_smem_u8)
                 k_s_smem = T.view(k_s_smem_u8, (B,), FP32)
@@ ... @@
-    kvcache_u8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
+    kvcache_u8 = kvcache_fp8.view(torch.uint8).view(-1, block_size * (head_dim + 4))  # TL0.1.8: uint8 view required
```

The patch is [`scripts/patches/sglang-tilelang-018-indexer-compat.patch`](scripts/patches/sglang-tilelang-018-indexer-compat.patch).

**Throughput impact:**

| Config | n=1 | 12 concurrent |
|---|---|---|
| Torch fallback (no TileLang) | ~13.5 tok/s | ~18 tok/s (flat ceiling) |
| TileLang indexer (patched) | **~18.7 tok/s** | **~42 tok/s** (~2.3x, scales with load) |

MTP accept length ~1.8 tokens/step. The model's nextn head is unquantized (FP16/BF16), so MTP runs without the NVFP4 path and adds minimal overhead.

One additional small win: the `topk_transform_512` native JIT kernel (`topk_v1.cuh`) compiles and runs correctly on SM121 — bit-identical to the PyTorch fallback. Drop the torch fallback:

```
SGLANG_TOPK_TRANSFORM_512_TORCH=0
```

---

## Long-context — single-stream, cold cache

Re-measured with **unique prompts per context** (distinct topic + nonce → no shared prefix → no radix-cache contamination) and `ignore_eos`:

| Context (prompt tokens) | TTFT (cold prefill) | Decode |
|---|---|---|
| 20,057 | 145 s | 9.7 tok/s |
| 38,456 | 290 s | 9.9 tok/s |
| 73,655 | 622 s | 10.2 tok/s |

Two findings:
- **Decode is essentially flat at ~10 tok/s from 20K → 74K context** — no progressive cliff. It's a single step down from the ~18.7 tok/s short-context rate, then stable.
- **Cold prefill runs at ~120–140 tok/s**, so TTFT scales ~linearly with input length (74K tokens ≈ 10 min). This is the universal long-context wall — prefill is compute-bound on input length regardless of engine. (An earlier run showed a misleading ~5 s TTFT at 32K; that was a radix-cache hit from a shared prompt prefix, now eliminated.)

---

## Comparison to the vLLM path

For reference, [jasl9187's thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-2-nodes/368916) covers the vLLM+FP8 approach on 2× DGX Spark (TP=2):

| Metric | vLLM FP8, 2× GB10, TP=2 | SGLang NVFP4, 4× GB10, TP=4 |
|---|---|---|
| n=1 decode | ~21 tok/s | ~18.7 tok/s |
| 4-concurrent | ~42 tok/s | ~42 tok/s @ 12-conc |
| Long-context decode | ~16-17 tok/s @ 64K | ~10 tok/s (20-74K, flat) |
| CUDA graph stable? | No (sm12.x instability → `--enforce-eager`, ~4× penalty) | Yes (max_bs=32) |
| Quant | FP8 (sgl-project/FP8 checkpoint) | NVFP4 (nvidia official) |
| MTP | Not reported | EAGLE-MTP, ~1.8 accept |

These are not apples-to-apples (different topology, different quantization, different number of nodes). This setup uses 2× the hardware, so the per-node throughput efficiency is lower. The vLLM path is simpler to get running on GB10 (no SM121-specific patches). The SGLang path gives stable CUDA-graph across high concurrency and the NVFP4 quant may have a quality advantage at long context.

---

## Summary of env/flags

For a 4-node SGLang deployment serving `nvidia/DeepSeek-V4-Flash-NVFP4` on GB10:

```bash
# MoE kernel (main model)
SGLANG_MOE_RUNNER_BACKEND=flashinfer_cutlass

# MoE kernel (MTP/draft layers — must be marlin on GB10)
SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND=marlin

# DSA indexer — use TileLang, not DeepGEMM (DeepGEMM blocks sm121)
SGLANG_OPT_USE_TILELANG_INDEXER=1
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1  # safe fallback if tilelang flag off

# topk: native JIT works on sm121
SGLANG_TOPK_TRANSFORM_512_TORCH=0

# CUDA graph
SGLANG_CUDA_GRAPH_MAX_BS=32

# DeepGEMM: disabled (no sm121 support)
SGLANG_DISABLE_DEEP_GEMM=1
SGLANG_OPT_DEEPGEMM_HC_PRENORM=0

# Speculative decoding
--speculative-algorithm EAGLE
--speculative-num-steps 1
--speculative-eagle-topk 1
--speculative-num-draft-tokens 2
--speculative-moe-runner-backend marlin
```

And the two image patches needed:
1. [`scripts/patches/sglang-dsv4-nvfp4-pr25820.patch`](scripts/patches/sglang-dsv4-nvfp4-pr25820.patch) — NVFP4 MoE path (PR #25820 + IndentationError fix + Marlin branch)
2. [`scripts/patches/sglang-tilelang-018-indexer-compat.patch`](scripts/patches/sglang-tilelang-018-indexer-compat.patch) — TileLang 0.1.8 1D→2D shared-buffer fix

---

## Open questions / what's next

- **Clean long-context numbers pending** — will update once the re-run (unique prompts, `ignore_eos`, more output tokens) completes.
- **DeepGEMM PR #324** — if this lands and gets pulled into `sgl-project/DeepGEMM`, the TileLang path would no longer be needed and throughput might improve further.
- **opt_fp8_wo_a_gemm** — untested on this setup; may offer an additional throughput gain for the FP8 base GEMM.

---

*Maintained as part of the dgxarley 4-node DGX Spark K3s cluster for distributed LLM inference.*
