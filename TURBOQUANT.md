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

| Metric | fp8_e4m3 (baseline) | TurboQuant 4-bit |
|--------|--------------------:|------------------:|
| KV cache tokens | 427K | 1,152K (**2.69x**) |
| PPL | 7.53 | 7.69 (+2.2%) |
| TTFT | 13.1 ms | 15.4 ms (+18%) |
| Needle-in-Haystack | PASS | PASS |

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

As of 2026-04-03 — llama.cpp merged a rotation baseline; SGLang/vLLM still have no merged implementations.

### SGLang

| PR | Title | Status |
|----|-------|--------|
| [#21419](https://github.com/sgl-project/sglang/pull/21419) | Add TurboQuant KV cache compression (3-4 bit, ICLR 2026) | Open, most traction (33 comments), no maintainer approval |
| [#21617](https://github.com/sgl-project/sglang/pull/21617) | feat(kv-cache): Add TurboQuant KV cache quantization | Open, WIP draft, Qwen3 IndexError bug (2026-04-02) |
| [#21628](https://github.com/sgl-project/sglang/pull/21628) | [AMD] Add TurboQuant KV cache compression for ROCm | Closed (not merged, 2026-03-29) — abandoned after review |
| [#21618](https://github.com/sgl-project/sglang/issues/21618) | Feature request: Add TurboQuant KV Cache Quantization | Open issue, low activity |

Usage (once merged): `--kv-cache-dtype turboquant`

PR #21419 adds fused Triton kernels for FWHT + bit-packing and a new `MHATokenToKVPoolTurboQuant`
memory pool. Tested on Qwen3-0.6B and Qwen2.5-1.5B. Active kernel optimization (fused gather/unpack/norm).

**Critical blocker (both PRs)**: CUDA graph incompatibility — TurboQuant's boolean indexing
(`tensor[mask]`) hits `cudaStreamCaptureUnsupported`. PR #21617 bypasses this by skipping quantization
during CUDA graph capture, meaning **decode KV cache runs at full precision** (memory savings only during
prefill). PR #21419 disables CUDA graph entirely. Neither approach is acceptable for production.

**Quality concerns**: GPQA accuracy on Qwen3-8B shows ~10% degradation vs bf16/fp8 KV cache.
Author notes TurboQuant targets memory-constrained long-context scenarios, not general replacement.

### vLLM

| PR | Title | Status |
|----|-------|--------|
| [#38280](https://github.com/vllm-project/vllm/pull/38280) | [Quantization] Add TurboQuant dynamic KV cache compression | Open, needs-rebase, OOM bugs reported on H20 |
| [#38479](https://github.com/vllm-project/vllm/pull/38479) | [Attention Backend] TurboQuant: 2-bit KV cache with 4x capacity | Open, most active implementation, merge conflicts |
| [#38662](https://github.com/vllm-project/vllm/pull/38662) | [Kernel] TurboQuant KV cache (PolarQuant + QJL) | Open, needs-rebase, 4th competing PR |
| [#38273](https://github.com/vllm-project/vllm/pull/38273) | TurboQuant (docs/ROCm) | Open, low-quality stub, likely superseded |
| [#38171](https://github.com/vllm-project/vllm/issues/38171) | Feature request | Open RFC, very active community discussion |

Usage (once merged): `--kv-cache-dtype turboquant`

**Four competing PRs**, none merged, all with merge conflicts. Community explicitly asking authors
to consolidate. Maintainer `mgoin` paused review of #38280 ("a bit hacked together"). Key findings
from community ablations (issue #38171):
- `tq3` (2-bit MSE + FP8 values) matches FP16 quality on Qwen3
- `tq4` (3-bit MSE) produces garbage due to a byte-spanning packing bug
- Ampere/Ada GPUs need `fp8e4b15` instead of `fp8e4nv`

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
  (`TBQ3_0` / `TBQ4_0`). CPU-only, awaiting review. TBQ4_0: 4.06 bits/elem (3.94x compression),
  PPL nearly identical to q4_0. TBQ3_0: 3.06 bits/elem (5.22x compression). CUDA/ROCm planned as follow-up.

**Closed (AI policy):** 9 TurboQuant PRs rejected for AI-generated code policy violations. The volume
of vibe-coded submissions prompted Gerganov to merge the rotation baseline (#21038) himself.

- [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977) — Feature request, active discussion (updated 2026-04-03)

### HuggingFace

- No official `transformers` PR yet
- Third-party drop-in: `pip install turboquant` v0.2.0 ([back2matching/turboquant](https://github.com/back2matching/turboquant), updated 2026-04-02)
- 7 models tagged `turboquant` on HuggingFace Hub (community uploads, Qwen3 variants)
- Google has **not released official code** yet (expected Q2 2026)

## What To Do

1. **Wait for CUDA graph blocker to be resolved** — both SGLang PRs currently disable CUDA graph
   or skip quantization during decode, negating the primary memory benefit. This is the gating issue.
2. **Monitor SGLang PR #21419** — most likely to merge first, but needs maintainer review + CUDA graph fix
3. **llama.cpp rotation is usable now** — PR #21038 merged, existing KV cache types benefit from
   Hadamard rotation. Not directly relevant to our SGLang/vLLM stack, but validates the approach.
4. **Wait for new `scitrera/dgx-spark-sglang` image** containing the merged code
5. **Test**: change `kv_cache_dtype` in the model profile from `fp8_e4m3` to `turboquant`:
   ```yaml
   kv_cache_dtype: "turboquant"  # was: fp8_e4m3
   ```
6. **Benchmark** against current fp8_e4m3 KV cache:
   - Context length capacity (how many tokens fit in 128 GB)
   - TTFT overhead (expect ~18%)
   - PPL / output quality on our standard prompts
   - Needle-in-Haystack at extended context
   - GPQA accuracy (community reported ~10% degradation on Qwen3-8B)

## Comparison With Current KV Cache Options

| KV dtype | Bits | Compression | Quality | Overhead | Available |
|----------|------|-------------|---------|----------|-----------|
| `auto` (bf16) | 16 | 1x (baseline) | Perfect | None | Yes |
| `fp8_e4m3` | 8 | 2x | Negligible | Minimal | Yes (current) |
| `fp8_e5m2` | 8 | 2x | Negligible | Minimal | Yes |
| `turboquant` | 3-4 | 4-5x | <2.5% PPL | ~18% TTFT | **Not yet** |

## References

- [Paper: arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [HuggingFace paper page](https://huggingface.co/papers/2504.19874)
- [Heise article (DE)](https://www.heise.de/en/news/TurboQuant-Google-aims-to-curb-the-memory-hunger-of-large-LLMs-11225521.html)
- [SGLang PR #21419](https://github.com/sgl-project/sglang/pull/21419)
- [vLLM PR #38280](https://github.com/vllm-project/vllm/pull/38280)
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) (most active vLLM implementation)
- [vLLM feature request #38171](https://github.com/vllm-project/vllm/issues/38171) (community ablation data)
- [llama.cpp PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) (merged rotation baseline)
- [llama.cpp PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) (CPU TBQ3/TBQ4 types)
- [turboquant PyPI package](https://pypi.org/project/turboquant/) (v0.2.0, HuggingFace drop-in)
- [turboquant-vllm PyPI package](https://pypi.org/project/turboquant-vllm/) (vLLM out-of-tree stopgap)
