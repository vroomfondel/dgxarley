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

As of 2026-04-02 — none merged yet. Both main PRs are in active review.

### SGLang

| PR | Title | Status |
|----|-------|--------|
| [#21419](https://github.com/sgl-project/sglang/pull/21419) | Add TurboQuant KV cache compression (3-4 bit, ICLR 2026) | Open, full implementation, 17 passing tests |
| [#21617](https://github.com/sgl-project/sglang/pull/21617) | feat(kv-cache): Add TurboQuant KV cache quantization | Open, WIP draft |
| [#21628](https://github.com/sgl-project/sglang/pull/21628) | [AMD] Add TurboQuant KV cache compression for ROCm | Closed (not merged, 2026-03-29) — awaiting main PR #21419 first |
| [#21618](https://github.com/sgl-project/sglang/issues/21618) | Feature request: Add TurboQuant KV Cache Quantization | Open issue |

Usage (once merged): `--kv-cache-dtype turboquant`

PR #21419 adds Triton kernels for FWHT + bit-packing, a new `MHATokenToKVPoolTurboQuant`
memory pool, and has been tested on Qwen3-0.6B and Qwen2.5-1.5B. Blocker: sgl_kernel SM86
incompatibility on the test machine — server E2E and long-context benchmarks pending.

### vLLM

| PR | Title | Status |
|----|-------|--------|
| [#38280](https://github.com/vllm-project/vllm/pull/38280) | [Quantization] Add TurboQuant KV cache quantization (Phase 1) | Open, benchmarked |
| [#38273](https://github.com/vllm-project/vllm/pull/38273) | TurboQuant (docs/ROCm) | Open, stub |
| [#38171](https://github.com/vllm-project/vllm/issues/38171) | Feature request | Open RFC |

Usage (once merged): `--kv-cache-dtype turboquant`

Phase 1 done (rotation + quantization + Triton kernels), Phase 2 WIP (packed uint8 storage).
A community fork [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) based on
vLLM 0.18.1rc1 is already functional.

### HuggingFace

- No official `transformers` PR yet
- Third-party drop-in: `pip install turboquant` ([back2matching/turboquant](https://github.com/back2matching/turboquant))
- 7 models tagged `turboquant` on HuggingFace Hub (community uploads, Qwen3 variants)
- Google has **not released official code** yet (expected Q2 2026)

### llama.cpp

- [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) and a community fork — no PR in main repo

## What To Do

1. **Wait for SGLang PR merge** — likely within weeks given quality of #21419
2. **Wait for new `scitrera/dgx-spark-sglang` image** containing the merged code
3. **Test**: change `kv_cache_dtype` in the model profile from `fp8_e4m3` to `turboquant`:
   ```yaml
   kv_cache_dtype: "turboquant"  # was: fp8_e4m3
   ```
4. **Benchmark** against current fp8_e4m3 KV cache:
   - Context length capacity (how many tokens fit in 128 GB)
   - TTFT overhead (expect ~18%)
   - PPL / output quality on our standard prompts
   - Needle-in-Haystack at extended context

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
- [turboquant PyPI package](https://pypi.org/project/turboquant/)
