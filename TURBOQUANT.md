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

As of 2026-04-13 — llama.cpp merged a rotation baseline; SGLang/vLLM still have no merged implementations, but both ecosystems have consolidated significantly since 2026-04-03. vLLM's four competing PRs collapsed to one primary effort (#38479); a new SGLang PR (#22048) claims to resolve the CUDA graph blocker.

### SGLang

| PR | Title | Status |
|----|-------|--------|
| [#22048](https://github.com/sgl-project/sglang/pull/22048) | feat(quantization): add TurboQuant KV cache quantization (arXiv:2504.19874) | **NEW 2026-04-03**, open, unreviewed. Claims **CUDA graph support** — the blocker from the older PRs |
| [#21419](https://github.com/sgl-project/sglang/pull/21419) | Add TurboQuant KV cache compression (3-4 bit, ICLR 2026) | Open, no maintainer approval, last touched 2026-04-05 |
| [#21617](https://github.com/sgl-project/sglang/pull/21617) | feat(kv-cache): Add TurboQuant KV cache quantization | Open WIP, last touched 2026-04-03, Qwen3 IndexError bug (2026-04-02) |
| [#21628](https://github.com/sgl-project/sglang/pull/21628) | [AMD] Add TurboQuant KV cache compression for ROCm | Closed 2026-03-29 — abandoned after review |
| [#21618](https://github.com/sgl-project/sglang/issues/21618) | Feature request: Add TurboQuant KV Cache Quantization | Open issue, low activity |

Usage (once merged): `--kv-cache-dtype turboquant`

**Most relevant: PR #22048** (YanGaev2, 2026-04-03). Full TurboQuant implementation: 1212 LOC core
(`turboquant.py`) + 196 LOC fused decode kernel + 221 LOC fused encode kernel + 543 LOC
`flashinfer_tq_backend.py`. Implements both TurboQuant_prod (Algorithm 2: (b-1)-bit MSE + 1-bit QJL
for keys) and TurboQuant_mse (Algorithm 1: b-bit MSE for values), parametric 1–4 bit via Triton
`constexpr`, mixed-precision mode with outlier channel allocation (Section 2.3), norm correction in
rotated domain (-1.17% PPL), and boundary layer protection (first/last N layers stored as bf16).
**Pitched as CUDA-graph-compatible**, directly addressing the blocker that killed the older PRs.
No comments or reviews yet — brand new and unvetted. **Watch this PR closely.**

**Older PRs (#21419, #21617)**: both fused Triton kernels + `MHATokenToKVPoolTurboQuant` memory pool,
tested on Qwen3-0.6B and Qwen2.5-1.5B. Both hit the CUDA graph blocker — TurboQuant's boolean indexing
(`tensor[mask]`) hits `cudaStreamCaptureUnsupported`. #21617 skipped quantization during capture (decode
KV cache runs at full precision — memory savings only during prefill), #21419 disabled CUDA graph
entirely. Neither approach is production-acceptable. Both have been quiet since 2026-04-05.

**Quality concerns (all SGLang PRs)**: GPQA accuracy on Qwen3-8B shows ~10% degradation vs bf16/fp8
KV cache. Authors note TurboQuant targets memory-constrained long-context scenarios, not general
replacement.

### vLLM

The "four competing PRs" situation from early April has consolidated: two were closed by their
authors after community push to converge. **#38479 is now the single primary effort.**

| PR | Title | Status |
|----|-------|--------|
| [#38479](https://github.com/vllm-project/vllm/pull/38479) | [Attention Backend] TurboQuant: 2-bit KV cache with 4x capacity | **Primary effort**, open, 91 comments, very active. Force-rebased 2026-04-11, cross-validation from multiple contributors (MidasMining, Alberto-Codes, jagmarques, domvox) converging on a hybrid-attention / SWA vs full-attention layer skip pattern — **directly relevant to Qwen3.5's GatedDeltaNet + GatedAttention hybrid architecture** |
| [#38280](https://github.com/vllm-project/vllm/pull/38280) | [Quantization] Add TurboQuant dynamic KV cache compression | **CLOSED 2026-04-06** by author — student out of bandwidth, preserved as reference. Had unresolved OOM bugs on H20 |
| [#38662](https://github.com/vllm-project/vllm/pull/38662) | [Kernel] TurboQuant KV cache (PolarQuant + QJL) | **CLOSED 2026-04-07** — maintainer asked author to close in favor of #38479 |
| [#39008](https://github.com/vllm-project/vllm/pull/39008) | [Quant] Add TurboQuant 4-bit (tq4) KV cache quantization | Closed 2026-04-05, short-lived |
| [#39050](https://github.com/vllm-project/vllm/pull/39050) | [Draft][Experimental][CUDA][VLM] Scaffold AttentionPack-style KV compression path | Draft/experimental alternative, opened 2026-04-07 |
| [#38273](https://github.com/vllm-project/vllm/pull/38273) | Turbo Quant (docs/ROCm) | Open, low-quality stub, no movement since 2026-03-26 |
| [#38171](https://github.com/vllm-project/vllm/issues/38171) | Feature request / RFC | Open, very active community discussion (last comment 2026-04-13) |

Usage (once merged): `--kv-cache-dtype turboquant`

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

1. **Watch SGLang PR #22048** — new entrant (2026-04-03) that claims to resolve the CUDA graph
   blocker the older PRs couldn't. If the claim holds up under review, this is the most likely
   path to production-ready TurboQuant in SGLang. Currently unreviewed — check back weekly.
2. **Monitor vLLM PR #38479** — vLLM's single consolidated effort after community convergence.
   Contributors are working out a hybrid-attention layer-skip pattern that is directly relevant
   to Qwen3.5 (GatedDeltaNet + GatedAttention). Not our primary runtime but worth tracking for
   the layer-skip intuition, which may transfer to SGLang.
3. **Older SGLang PRs (#21419, #21617) are stale** — quiet since 2026-04-05. Don't invest time here
   unless #22048 dies and they get revived.
4. **llama.cpp rotation is usable now** — PR #21038 merged 2026-04-01, existing KV cache types
   benefit from Hadamard rotation. Not directly relevant to our SGLang/vLLM stack, but validates
   the approach.
5. **Wait for new `scitrera/dgx-spark-sglang` image** containing the merged code
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
- [SGLang PR #22048](https://github.com/sgl-project/sglang/pull/22048) (newest, claims CUDA-graph compatibility)
- [SGLang PR #21419](https://github.com/sgl-project/sglang/pull/21419)
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) (vLLM's single consolidated effort, 91 comments)
- [vLLM feature request #38171](https://github.com/vllm-project/vllm/issues/38171) (community ablation data)
- [llama.cpp PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) (merged rotation baseline)
- [llama.cpp PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) (CPU TBQ3/TBQ4 types)
- [turboquant PyPI package](https://pypi.org/project/turboquant/) (v0.2.0, HuggingFace drop-in)
- [turboquant-vllm PyPI package](https://pypi.org/project/turboquant-vllm/) (vLLM out-of-tree stopgap)
