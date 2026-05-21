# TurboQuant — KV Cache Quantization for Extended Context

## What It Is

TurboQuant is a **KV cache quantization** method from Google Research (Zandieh et al., ICLR 2026).
It compresses the attention KV cache at runtime to 3-4 bits — this is **not weight quantization**,
it targets the memory consumed by cached key/value tensors during inference.

- Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Google blog: [research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## How It Works

1. **Random orthogonal rotation** (Hadamard/FWHT) spreads energy across all dimensions
2. **Lloyd-Max scalar quantization** per rotated coordinate (3-4 bit)
3. **Optional 1-bit QJL residual correction** for unbiased inner product estimation
4. **Outlier-aware channel allocation**: top ~15% high-variance channels kept at bf16

Key properties:
- Data-oblivious — no calibration dataset or fine-tuning required
- Applied online during inference, not at model conversion time
- No pre-quantized model files needed

## Performance

Benchmarked on Qwen2.5-7B-Instruct / H100 80GB (vLLM PR #38280):

| Metric             | fp8_e4m3 (baseline) |   TurboQuant 4-bit |
|--------------------|--------------------:|-------------------:|
| KV cache tokens    |                427K | 1,152K (**2.69x**) |
| PPL                |                7.53 |       7.69 (+2.2%) |
| TTFT               |             13.1 ms |     15.4 ms (+18%) |
| Needle-in-Haystack |                PASS |               PASS |

General claims from the paper: 2.7-5x memory savings, <2.5% PPL degradation.

## Why This Matters for Our Cluster

Each DGX Spark has 128 GB unified memory. With current NVFP4 weights + fp8_e4m3 KV cache,
the KV cache is the binding constraint for context length on large MoE models.

TurboQuant **stacks on top of weight quantization** — they compress different things:
- NVFP4: model weights (loaded once)
- TurboQuant: KV cache (grows with context length)

This could extend effective context length by 3-4x at the same memory budget,
or allow running larger models at current context lengths.

## Framework Integration Status

As of 2026-05-21 — **vLLM merged TurboQuant on 2026-04-15** (PR #38479), shipping in v0.21.0 (2026-05-15) with a follow-up perf fix (PR #40941, shared dequant buffers, merged 2026-04-27). **But:** several crash-level bugs are open against the v0.21.0 implementation (see vLLM section below) — the merge is real, production-readiness for non-vanilla configs is not. llama.cpp merged the rotation baseline back in April; CPU TBQ types (#21089) still open with renewed activity (2026-05-17, NexusQuant asymmetric-K/V analysis). **SGLang is still the laggard**: no merge in v0.5.11 (2026-05-05) or v0.5.12 (2026-05-16). PR #23135 (fused Triton rewrite, last touched 2026-05-13) remains the most credible path; PR #22048 was revived 2026-05-19 with community H200 testing on Gemma-4-31B.

### SGLang

| PR                                                           | Title                                                                                       | Status                                                                                                                                  |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [#23135](https://github.com/sgl-project/sglang/pull/23135)   | [KVCache] TurboQuant: fused Triton KV cache compression (3.88x, 93–105 % decode throughput) | **Currently most active**, opened 2026-04-18, last touched 2026-05-13. Rewrite addressing perf + CUDA-graph blockers of all earlier PRs. New issue surfaced 2026-04-26: MLA-incompatible (forced `decode_attention_backend=triton` breaks MLA models; `_maybe_fuse_tq_output_rotation` needs MLA guard) |
| [#22048](https://github.com/sgl-project/sglang/pull/22048)   | feat(quantization): add TurboQuant KV cache quantization (arXiv:2504.19874)                 | **Revived 2026-05-19** after stalling at 2026-04-15. Community testing on H200 (TP=2, Gemma-4-31B) reports `--kv-cache-dtype turboquant` working; new bug found: TTFT explosion with HiCache at >90 % KV-fill. Author (@YanGaev2) responding actively                         |
| [#21419](https://github.com/sgl-project/sglang/pull/21419)   | Add TurboQuant KV cache compression (3-4 bit, ICLR 2026)                                    | Open, last touched 2026-04-29 (light activity). No maintainer approval                                                                  |
| [#21617](https://github.com/sgl-project/sglang/pull/21617)   | feat(kv-cache): Add TurboQuant KV cache quantization (WIP)                                  | **Dead**, no activity since 2026-04-03. Qwen3 IndexError bug                                                                            |
| [#23133](https://github.com/sgl-project/sglang/pull/23133)   | [KVCache] TurboQuant v2: fused Triton KV cache compression                                  | Closed 2026-04-18 — superseded by #23135 (same author, branch reset)                                                                    |
| [#21628](https://github.com/sgl-project/sglang/pull/21628)   | [AMD] Add TurboQuant KV cache compression for ROCm                                          | Closed 2026-03-29 — abandoned after review                                                                                              |
| [#23134](https://github.com/sgl-project/sglang/issues/23134) | Tracking issue for fused Triton variant (#23135)                                            | Open, opened 2026-04-18                                                                                                                 |
| [#21618](https://github.com/sgl-project/sglang/issues/21618) | Original feature request                                                                    | Open, low activity since 2026-03-29                                                                                                     |

Usage (once merged): `--kv-cache-dtype turboquant_4bit_uniform` (per #23135) or `--kv-cache-dtype turboquant`.

**Most relevant: PR #23135** (liuhuijiayou, branch `feat/turboquant-kvcache`). Architectural rewrite of the
earlier PRs. Key technical claims:
- **Fused Triton decode/extend kernels read packed uint8 KV directly during attention** — no
  dequant buffer. This was the single biggest bottleneck in #21419/#21617/#22048: the older PRs
  dequantized the entire KV pool on every attention call, yielding ~0.03 tok/s on H200. The fused
  kernel approach eliminates that staging buffer entirely.
- **N-way split dot product** — no full decompressed vector materialization.
- **WHT (Walsh–Hadamard) rotation fused into `W_O` weights at init** — zero decode-time rotation cost.
- **Uniform dequant**: 1 FMA replaces 15 `tl.where` codebook lookups (+15 % decode).
- **FP8 model compatibility** with auto rotation-fusion skip.
- **Codebook + uniform modes** via `--kv-cache-dtype turboquant_4bit_uniform`.

Reported results (H200, Llama-3.1-8B-Instruct, Triton backend):
- Decode throughput **93–105 % of bf16** (vs ~1 % in original)
- KV cache compression **3.88×** vs bf16
- GSM8K 79.3 % vs bf16 79.2 %, MMLU 68.3 % == bf16
- **Full CUDA graph support**

Known limitations:
- Prefill + decode locked to **Triton attention backend** (fa3 extend path incompatible with packed KV pool)
- Prefill degrades with long inputs (P5 4096→1: 64 % of bf16) — needs further kernel fusion
- Only validated on H200 — smaller GPUs (A30 etc.) and SM121 (GB10) unverified

**Older PRs (#21419, #21617, #22048)**: all attempted fused Triton kernels + `MHATokenToKVPoolTurboQuant`
memory pool, but each retained a dequant staging buffer somewhere on the hot path. #21617 skipped
quantization during CUDA-graph capture (decode runs at full precision — memory savings only during
prefill); #21419 disabled CUDA graph entirely; #22048 claimed CUDA-graph support but stalled in review.
#23135's "no dequant buffer" approach is the architectural break that addresses the underlying issue.

**Quality concerns (older SGLang PRs)**: GPQA accuracy on Qwen3-8B showed ~10 % degradation vs bf16/fp8 KV
cache. #23135 has not yet been benchmarked on GPQA but reports MMLU/GSM8K parity with bf16.

### vLLM

**vLLM merged TurboQuant on 2026-04-15** — PR [#38479](https://github.com/vllm-project/vllm/pull/38479)
("[Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity"). A follow-up perf
fix (PR #40941, shared dequant buffers, eliminates `float16_copy`) merged 2026-04-27 and shipped
in **v0.21.0 (2026-05-15)**. Verify the `scitrera/dgx-spark-vllm` image base contains v0.21.0+ before
relying on the optimized path.

**Production-readiness caveat (re-verified 2026-05-21):** despite the merge, TurboQuant in v0.21.0
has multiple **crash-level bugs open** that affect exactly the model classes we run:

- [#40124](https://github.com/vllm-project/vllm/issues/40124) — TQ + Hybrid MoE (Qwen3.6-35B-A3B) broken on Ampere (SM 80–86), 13-patch proposal
- [#41403](https://github.com/vllm-project/vllm/issues/41403) — TQ + Gemma-4 multimodal: 5-blocker stack
- [#41413](https://github.com/vllm-project/vllm/issues/41413) — TQ fails on non-power-of-2 `head_dim`
- [#41726](https://github.com/vllm-project/vllm/issues/41726) — TQ crashes on large chunked-continuation prefill with Hybrid-Attention (Qwen3.5-9B)
- [#42808](https://github.com/vllm-project/vllm/issues/42808) — TQ + MTP speculative decoding workspace-assertion
- [#42215](https://github.com/vllm-project/vllm/issues/42215) — `[Bugfix][V1][TurboQuant] Warm up decode kernels` (OPEN — listed as mandatory by tracking issue #40069)
- [#40798](https://github.com/vllm-project/vllm/issues/40798) — Share decode scratch workspace across layers (OPEN — also mandatory per #40069)
- [#40069](https://github.com/vllm-project/vllm/issues/40069) — central tracking issue: says outright that TurboQuant "does not work without" #40798 and #42215

For our cluster (NVFP4 MoE on SM121/GB10), every one of these is in the danger zone. Hybrid-MoE,
Gemma-4 multimodal, and MTP spec-decode are all production paths we use or want to use. Treat
vLLM TQ as **experimental** until those issues close.

| PR / Issue                                                  | Title                                                                       | Status                                                              |
|-------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| [#38479](https://github.com/vllm-project/vllm/pull/38479)   | [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity | **MERGED 2026-04-15**                                               |
| [#40941](https://github.com/vllm-project/vllm/pull/40941)   | TurboQuant shared dequant buffers (perf fix)                                | **MERGED 2026-04-27**, shipped in v0.21.0                           |
| [#40069](https://github.com/vllm-project/vllm/issues/40069) | Tracking issue: TQ housekeeping (FA3/FA4, Hybrid-Attn, MLA)                 | OPEN — lists #40798 and #42215 as prerequisites                     |
| [#42215](https://github.com/vllm-project/vllm/issues/42215) | Warm up decode kernels                                                      | OPEN — mandatory bugfix                                             |
| [#40798](https://github.com/vllm-project/vllm/issues/40798) | Share decode scratch workspace across layers                                | OPEN — mandatory bugfix                                             |
| [#40124](https://github.com/vllm-project/vllm/issues/40124) | TQ + Hybrid MoE broken on Ampere                                            | OPEN (2026-05-20)                                                   |
| [#41403](https://github.com/vllm-project/vllm/issues/41403) | TQ + Gemma-4 multimodal 5-blocker stack                                     | OPEN (2026-05-13)                                                   |
| [#41413](https://github.com/vllm-project/vllm/issues/41413) | TQ fails on non-pow-2 head_dim                                              | OPEN (2026-05-19)                                                   |
| [#41726](https://github.com/vllm-project/vllm/issues/41726) | TQ crash on chunked prefill + Hybrid-Attention                              | OPEN (2026-05-17)                                                   |
| [#42808](https://github.com/vllm-project/vllm/issues/42808) | TQ + MTP spec-decode workspace assertion                                    | OPEN (2026-05-20)                                                   |
| [#38280](https://github.com/vllm-project/vllm/pull/38280)   | [Quantization] Add TurboQuant dynamic KV cache compression                  | Closed 2026-04-06 — author out of bandwidth, preserved as reference |
| [#38662](https://github.com/vllm-project/vllm/pull/38662)   | [Kernel] TurboQuant KV cache (PolarQuant + QJL)                             | Closed 2026-04-07 in favor of #38479                                |
| [#39008](https://github.com/vllm-project/vllm/pull/39008)   | [Quant] Add TurboQuant 4-bit (tq4) KV cache quantization                    | Closed 2026-04-05, short-lived                                      |
| [#39050](https://github.com/vllm-project/vllm/pull/39050)   | [Draft] AttentionPack-style KV compression scaffold                         | Draft, alternative path                                             |
| [#38273](https://github.com/vllm-project/vllm/pull/38273)   | Turbo Quant (docs/ROCm)                                                     | Open, low-quality stub                                              |
| [#38171](https://github.com/vllm-project/vllm/issues/38171) | Feature request / RFC                                                       | Open, kept as discussion forum                                      |

Usage: `--kv-cache-dtype turboquant` (works on vanilla configs; expect breakage on MoE/multimodal/MTP/spec-decode in v0.21.0)

Key findings from community ablations on issue #38171:
- `tq3` (2-bit MSE + FP8 values) matches FP16 quality on Qwen3
- `tq4` (3-bit MSE) produces garbage due to a byte-spanning packing bug
- Ampere/Ada GPUs need `fp8e4b15` instead of `fp8e4nv`
- fp8 KV cache on H100 is essentially free (0.97x throughput vs fp16, confirmed on Qwen2.5-7B 2026-04-13)

Stopgap: `pip install turboquant-vllm[vllm]` — out-of-tree package, Qwen3 validated (cosine sim ~0.9951 at K4/V4).
Community fork [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) (vLLM 0.18.1rc1) updated 2026-04-03.

### llama.cpp

**Merged (2026-04-01):**
- [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) — `llama: rotate activations for better quantization` —
  by Gerganov. Adds backend-agnostic Hadamard rotation of Q/K/V before quantization. Not full TurboQuant
  (no new types), but the rotation step alone gives dramatic PPL improvements on existing types
  (e.g., q5_1 on Qwen3 0.6B: PPL 61.7 → 14.1). Disable with `LLAMA_ATTN_ROT_DISABLE`.

**Open:**
- [PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) — CPU TurboQuant KV cache types
  (`TBQ3_0` / `TBQ4_0`). CPU-only, awaiting review. Active again 2026-05-17 with NexusQuant
  cross-reference (#21591) — discussion of asymmetric K/V quantization, softmax error floor at
  ~3-bit K, per-head fp16 masking of bottom 2 % KV heads. TBQ4_0: 4.06 bits/elem (3.94x
  compression), PPL nearly identical to q4_0. TBQ3_0: 3.06 bits/elem (5.22x compression).
  CUDA/ROCm planned as follow-up.

**Closed (AI policy):** 9 TurboQuant PRs rejected for AI-generated code policy violations. The volume
of vibe-coded submissions prompted Gerganov to merge the rotation baseline (#21038) himself.

- [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977) — Feature request, active discussion (updated 2026-04-03)

### HuggingFace

- No official `transformers` PR yet
- Third-party drop-in: `pip install turboquant` v0.2.0 — but [back2matching/turboquant](https://github.com/back2matching/turboquant) GitHub is at **v0.3.1** (tagged 2026-04-16, last commit 2026-04-21: transformers 5.x cache-API backward-compat shims). PyPI still serves v0.2.0; install from GitHub for the latest
- 7 models tagged `turboquant` on HuggingFace Hub (community uploads, Qwen3 variants)
- Google has **not released official code** yet (expected Q2 2026)

## What To Do

1. **vLLM has it — try it there first, but expect breakage.** PR #38479 is merged and v0.21.0
   (2026-05-15) ships the shared-dequant-buffer perf fix (#40941). Confirm the
   `scitrera/dgx-spark-vllm` image base is v0.21.0+ or rebuild against current `main`. **Read the
   vLLM section above first** — there are multiple open crash-level bugs that hit exactly our
   workloads (Hybrid MoE, Gemma-4 multimodal, MTP spec-decode, Ampere). For a one-off TurboQuant
   smoke test, stick to vanilla configs (single-modal, no spec-decode, power-of-2 head_dim). **SM121
   caveat:** the merged backend was developed and validated on H100/H200 (Hopper) — Ampere/Ada need
   `fp8e4b15` instead of `fp8e4nv` per community ablations on issue #38171, and SM121 specifics
   are unknown. Expect to debug.
2. **Watch SGLang PR #23135** — currently the most credible SGLang path. Architectural rewrite with
   no dequant staging buffer, full CUDA graph support, claimed 93–105 % decode throughput vs bf16
   on H200. Key open questions before adopting: (a) does it survive review and merge, (b) does the
   "Triton-backend-only" constraint hurt us on SM121 (FlashInfer is normally faster on this cluster),
   (c) does prefill performance hold up at our typical batch sizes, (d) **MLA incompatibility**
   surfaced 2026-04-26 (forced `decode_attention_backend=triton` breaks MLA) — must be fixed before
   merge or we lose MLA models. Check tracking issue #23134 for activity.
3. **SGLang PR #22048 revived 2026-05-19** — was previously stalled but is back: community testing
   on H200 with TP=2 Gemma-4-31B confirms `--kv-cache-dtype turboquant` works, author responding
   actively. New bug found: TTFT explodes with HiCache at >90 % KV-fill. If we can wait for the
   HiCache fix, this might land before #23135.
4. **Older SGLang PRs (#21419, #21617) remain dead.** #21617 hasn't moved since 2026-04-03; #21419
   has light activity but no maintainer engagement.
5. **llama.cpp rotation is usable now** — PR #21038 merged 2026-04-01, existing KV cache types
   benefit from Hadamard rotation. Not directly relevant to our SGLang/vLLM stack but validates the
   underlying approach.
6. **Once a usable build lands**: change `kv_cache_dtype` in the model profile from `fp8_e4m3` to
   `turboquant` (or `turboquant_4bit_uniform` for SGLang #23135):
   ```yaml
   kv_cache_dtype: "turboquant"  # was: fp8_e4m3
   ```
7. **Benchmark** against current fp8_e4m3 KV cache:
   - Context length capacity (how many tokens fit in 128 GB unified mem)
   - TTFT overhead (~18 % on H100; SM121 unverified)
   - PPL / output quality on our standard prompts
   - Needle-in-Haystack at extended context
   - GPQA accuracy (older SGLang PRs reported ~10 % degradation on Qwen3-8B; #23135 claims parity
     on MMLU/GSM8K but no GPQA numbers yet)

## Comparison With Current KV Cache Options

| KV dtype      | Bits | Compression   | Quality    | Overhead  | Available                                                             |
|---------------|------|---------------|------------|-----------|-----------------------------------------------------------------------|
| `auto` (bf16) | 16   | 1x (baseline) | Perfect    | None      | Yes                                                                   |
| `fp8_e4m3`    | 8    | 2x            | Negligible | Minimal   | Yes (current)                                                         |
| `fp8_e5m2`    | 8    | 2x            | Negligible | Minimal   | Yes                                                                   |
| `turboquant`  | 3-4  | 4-5x          | <2.5% PPL  | ~18% TTFT | **vLLM: yes in v0.21.0 (merged 2026-04-15), but crash bugs open for MoE/multimodal/MTP; SGLang: no (PR #23135 most active, #22048 revived 2026-05-19)** |

## References

- [Paper: arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [HuggingFace paper page](https://huggingface.co/papers/2504.19874)
- [Heise article (DE)](https://www.heise.de/en/news/TurboQuant-Google-aims-to-curb-the-memory-hunger-of-large-LLMs-11225521.html)
- [SGLang PR #23135](https://github.com/sgl-project/sglang/pull/23135) (currently most active, fused Triton rewrite)
- [SGLang issue #23134](https://github.com/sgl-project/sglang/issues/23134) (tracking issue for #23135)
- [SGLang PR #22048](https://github.com/sgl-project/sglang/pull/22048) (stalled since 2026-04-15)
- [SGLang PR #21419](https://github.com/sgl-project/sglang/pull/21419)
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) (**MERGED 2026-04-15**)
- [vLLM feature request #38171](https://github.com/vllm-project/vllm/issues/38171) (community ablation data)
- [llama.cpp PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) (merged rotation baseline)
- [llama.cpp PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) (CPU TBQ3/TBQ4 types)
- [turboquant PyPI package](https://pypi.org/project/turboquant/) (v0.2.0, HuggingFace drop-in)
- [turboquant-vllm PyPI package](https://pypi.org/project/turboquant-vllm/) (vLLM out-of-tree stopgap)
