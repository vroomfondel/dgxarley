# DSA Decode Speedup on GB10 / SM121 (GlmMoeDsa, MLA + DeepSeek Sparse Attention)

Working memo. What throttles decode for DSA-class models (GlmMoeDsa, DeepSeek-V3.2/V4)
on the DGX Spark GB10 (SM121), what is verified vs hypothesised, and the concrete
levers to reach a fully accelerated decode path. Companion to `TURBOQUANT.md`.

Status: root cause VERIFIED from the live head log of `0xSero/glm-5.2-reap-504B-v2`
(TP=4, modelopt_fp4/NVFP4, image `xomoxcc/dgx-spark-sglang:0.5.15-sm121`), 2026-07-15.
Approaches A-F further below are the ORIGINAL 2026-07-15 hypotheses and are now SUPERSEDED
by the GPU-pod-verified conclusions in the UPDATE 2026-07-16 section directly below (in short:
the sparse indexer cannot run on SM121 via any hardware kernel; the torch fallback in
`dsalogitrework.md` is the one viable path). Read the UPDATE first.

## UPDATE 2026-07-16 (GPU-debug-pod verified) - the sparse path is a HARD SM121 block

A GPU-debug-pod test on GB10 (device capability (12,1)) settled the two candidate levers:

- **The sparse indexer is a HARD NO-GO on SM121.** `attention_backend="dsa"`
  (DeepseekSparseAttnBackend) dispatches paged-MQA-logits to DeepGEMM
  (`dsa_paged_mqa_logits_backend=auto -> DEEPGEMM`), and
  `deep_gemm.get_paged_mqa_logits_metadata()` throws a COMPILED C++ assert
  `Unsupported architecture` (deepgemm/csrc/apis/attention.hpp:227) on SM121. This runs
  on EVERY decode step, so `"dsa"` crashes the TARGET on its first token (not just the
  draft). `cutedsl` is gated behind `is_sm100_supported()`==False on GB10;
  `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` is wired only into dsv4/indexer.py, not the
  `dsa.dsa_indexer.Indexer` GlmMoeDsa uses; `aiter`=ROCm; tilelang covers only prefill.

- **The base model has been running DENSE MLA all along.** Under `"flashinfer"`,
  `FlashInferMLAAttnBackend.get_indexer_metadata()` returns None -> the sparse indexer is a
  SILENT no-op -> dense MLA on the fa2 decode path (Gap 1 below). So DSA sparse attention
  has never actually run here; the "throttle" is dense-MLA-on-fa2, and the "enable the
  indexer" idea does not apply - the indexer cannot run on SM121 at all right now.

- **Consequence for MTP/NEXTN: blocked by this same gap.** The draft always computes
  topk_indices (is_nextn), so it needs the indexer: flashinfer -> draft passes a
  topk_indices kwarg that forward_decode() rejects; dsa -> DeepGEMM SM121 assert. No
  working config until the gap closes. The MTP weight patch + mem config ARE validated and
  ready. See memory reference_glm52_dsa_indexer_deepgemm_sm121.

- **cutedsl: ALSO a HARD NO-GO (verified 2026-07-16, GPU pod).** Relaxing the
  `is_sm100_supported()` gate lets `resolve("cutedsl")` through, but the kernel then fails at
  `cute.compile()`: `_setup_mma` emits a `tcgen05.MmaF8F6F4Op`, a Tensor-Memory MMA that only
  exists on datacenter Blackwell (sm_100/sm_103), NOT consumer Blackwell (sm_120/sm_121=GB10).
  That is an ISA boundary, not a software gate - no flag brings back a missing instruction
  (`expects arch sm_100a/103a..., got sm_121a`). Independently, the schedule metadata
  (`get_paged_mqa_logits_metadata`) is DeepGEMM at TWO call sites regardless of the logits
  backend (also eager in `dsa_backend.py::init_forward_metadata`), so it would crash anyway.

- **DeepGEMM upstream: closed short-term.** PR #318 (Apr 2026) added SM12x kernels for exactly
  `paged_mqa_logits` and was explicitly REJECTED by the maintainer (no hardware to test/maintain);
  issue #372 (Jul 2026) is a second, independent SM12x block (`transform_sf_into_required_layout`,
  NVFP4 expert scales). No image-bump / version-pin path via DeepGEMM.

- **Warp-level cute-MMA: what our working kernels already run on, but NOT a quick DSA path**
  (verified 2026-07-16, GPU pod). Our FP4 GEMM for GlmMoeDsa dispatches to
  `flashinfer .../dense_blockscaled_gemm_sm120_b12x.py` (`@supported_compute_capability([120,121])`,
  uses `MmaMXF4NVF4Op` - the SAME warp-level family as the `MmaF16BF16Op` the cutedsl error
  suggests, not tcgen05). `MmaF16BF16Op` compiles + runs + passes a numeric check on GB10. BUT
  reimplementing the DSA paged-MQA-logits on it is from-scratch kernel authoring: the build has
  NO FP8 warp-MMA (`MmaMXF8Op` absent -> needs an fp8->bf16 dequant), and there is no
  paged-attention / top-k template on warp-MMA in the image. Keep it as a LATER perf option,
  after correctness via torch.

- **The one viable path = torch fallback.** SGLang already merged a CUDA-graph-safe torch
  paged-MQA-logits fallback for DeepSeek-V4 (`dsv4/indexer.py::fp8_paged_mqa_logits_torch_sm120`,
  PR #24692, in v0.5.13 -> present in our v0.5.15 image) with a matching signature + KV layout;
  it just is not wired into the generic `dsa/dsa_indexer.py` path GlmMoeDsa uses. The torch path
  discards the DeepGEMM schedule metadata (`_ = deep_gemm_metadata`), so both metadata call sites
  can simply be skipped. Port plan: `dsalogitrework.md`. Current prod state: stay on `"flashinfer"`
  (dense MLA, known-working/slow), MTP off, until the torch port lands + is GSM8K-validated against
  the dense baseline (a community torch fallback silently returned NaN/zeros on SM120 -> numeric
  verification is mandatory, not optional).

## KERNEL-PATH SURVEY 2026-07-16 (verdict: DSA-sparse on SM121 is NOT config-reachable)

After the torch indexer fallback was implemented + live-deployed, the boot got PAST the DeepGEMM
indexer assert (torch gating worked, all 3 metadata call sites) and PAST KV alloc, then crashed at
CUDA-graph autotune of the MLA decode attention: `TllmGenFmhaRunner ... Unsupported architecture`
(trtllm-gen FMHA). A full GPU-pod survey of every DSA decode/prefill attention backend:

| Backend (`dsa_decode/prefill_backend`) | Origin | SM121 status |
|---|---|---|
| `trtllm` (auto-default when kv fp8 + major>=10) | trtllm-gen FMHA (`TllmGenFmhaRunner`) | ISA wall, live crash |
| `flashmla_sparse` / `flashmla_kv` | `sgl_kernel.flash_mla` (`flashmla_ops`) | extension NOT built in image (ImportError; only `sm100/common_ops`) |
| `fa3` | flash-attention-v3 | hard gate `_is_fa3_supported()` = major==8 or 9; GB10=12 -> False |
| `tilelang` | `dsa/tilelang_kernel.py` (own JIT) | PROVEN DEAD (GPU-tested): smem/compile contradiction, see below |
| `aiter` | ROCm | N/A (AMD) |
| indexer `dsa_paged_mqa_logits_backend=torch` | our port | WORKS (this is what got the boot past the indexer) |

Root cause of the auto-default: `arg_groups/overrides.py::_dsa_split_backend_resolution` picks
`trtllm if major>=10` - the SAME `major>=10` bug that conflates datacenter Blackwell (SM100) with
consumer Blackwell (SM120/121), and it recurs structurally across `dsa_backend.py` (flashmla padding,
workspace-buffer gate, `_forward_standard_mha` dispatch), not a one-liner.

**Verdict: the torch indexer fix was necessary but NOT sufficient, and every attention-decode path is now
exhausted.** The MLA decode attention has the same datacenter-vs-consumer gap, and unlike the indexer there
is NO ready merged torch reference in the image to port.

The tilelang lead (the only backend that even compiles on SM121) was GPU-tested against the smem budget and
is **proven dead**, not merely uncertain. `sparse_attention_fwd_kernel_v2` smem =
`1160*H + 2306*block_I + 2*H*block_I` (H=64 for GLM-5.2), dominated by the 4x KV double-buffer (2-stage
pipeline), linear in block_I. There is a hard contradiction:

| block_I | smem | compiles? | launches (< 101376 B)? |
|---|---|---|---|
| 64 (default) | 231680 B | yes | NO |
| 16 | 114944 B | yes | NO (~13.5 KB over) |
| 8 | ~93712 B (fits) | **NO** ("Layout infer conflict m_i/alpha_local") | - |
| 4 | ~83976 B | **NO** ("N must be divisible by 8") | - |

The smallest COMPILABLE block_I (16) needs 114944 B (over budget); the smallest smem-FITTING block_I (8)
does not compile (the warp-specialized GEMM/layout choreography breaks at tiny tiles). No block_I satisfies
both. Shrinking below the budget would need a kernel REWRITE (single- instead of double-buffering, redoing
the 3-warpgroup barrier choreography), not a parameter change.

So DSA-sparse (and therefore NEXTN/MTP, which needs the indexer) is blocked on GB10/SM121 with NO config or
tuning path. The only two real options, both a project with no guarantee: (1) upstream adds consumer-Blackwell
kernels (trtllm-gen FMHA / flashmla / DeepGEMM for sm120/121), or (2) from-scratch kernel authoring on the
attention decode (a single-buffered tilelang rewrite, or a torch MLA-attention fallback like the indexer one).
Prod stays on `attention_backend="flashinfer"` (dense MLA, known-working). The torch indexer fallback + the
`--dsa-paged-mqa-logits-backend` wiring stay committed for when the attention side is unblocked upstream.

## Symptom (measured)

- Single-stream decode: ~2.5-4 tok/s (head-log `Decode batch ... #running-req: 1-2 ...
  gen throughput (token/s): 2.98-4.06`).
- Batched at concurrency 16: ~24 tok/s aggregate (`#running-req: 16 ... cuda graph: True
  ... gen throughput (token/s): 23.65`), i.e. ~1.5 tok/s per sequence.
- So decode is compute/kernel-bound per token, NOT concurrency- or KV-bound (KV pool
  `token usage: 0.01`, graphs captured up to bs=32). Batching amortises but the
  per-token cost is the wall.

## Root cause: two SM121 kernel-coverage gaps in the DECODE path

GB10 is **SM121 = consumer Blackwell**. The fast decode kernels were tuned/gated for
SM90 (Hopper) and SM100 (datacenter Blackwell, GB200 cluster-launch). SGLang's auto
heuristics do not route SM121 to those native paths, so decode falls back.

### Gap 1 (primary): MLA decode uses FlashAttention-2, not the Blackwell-native trtllm-gen kernel

GlmMoeDsa uses MLA (compressed latent KV). Verbatim head-log warning:

```
flashinfer_mla_backend.py:276: UserWarning: BatchMLAPagedAttentionWrapper: backend='auto'
selected 'fa2' on SM121, which is not Blackwell-native and gives poor MLA decode
performance. For decode, use flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
(Blackwell-native trtllm-gen); backend='cutlass' is the closest in-wrapper alternative
but may be slower than this fallback for decode shapes.
```

flashinfer's `BatchMLAPagedAttentionWrapper` with `backend='auto'` picks **fa2** on
SM121. fa2 maps the unusual MLA head geometry (large rope/nope split, absorbed KV)
poorly on Blackwell. This is the main per-token throttle.

### Gap 2: DeepGEMM JIT disabled, so the DSA indexer runs off the optimised FP8 GEMM path

`GlmMoeDsa` runs DeepSeek Sparse Attention: a per-token "lightning indexer"
(`fp8_paged_mqa_logits`) scores paged KV and selects top-k before the MLA attention.
Head-log: `Setting page size to 64 for DeepSeek DSA` / `index_topk=2048 for DeepSeek
with DSA` / `Set DSA backends for fp8_e4m3 KV Cache: prefill=trtllm, decode=trtllm`.

Image-baked env (observed):

| flag | value | meaning |
|------|-------|---------|
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `false` | DeepGEMM JIT FP8 GEMM kernels OFF (the indexer/MoE fast path) |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | `0` | indexer logits NOT on the slow torch fallback (native path) |
| `SGLANG_TOPK_TRANSFORM_512_TORCH` | `0` | top-k transform native (works on SM121, per prior finding) |
| `SGLANG_OPT_USE_TILELANG_INDEXER` | `0` | tilelang indexer path OFF |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM` | `1` | DeepGEMM high-contiguous prenorm opt (moot while JIT off) |
| (startup) | | `Patched cute/mma.py: added sm_120a + sm_121a to BlockScaledMmaOp.admissible_archs` |

DeepGEMM is off almost certainly because its codegen targets SM90 / SM100 (TMA,
cluster, `tcgen05`) and SM121 is not admitted, so enabling it in the image build either
failed to compile or produced wrong results (hence the explicit `false`). The indexer
therefore runs on a non-DeepGEMM native path (TORCH=0), which is functional but not the
tuned FP8 GEMM. TO VERIFY: whether that native `fp8_paged_mqa_logits` path is itself
fast or a second-tier fallback on SM121.

### Why FP4 GEMM works but decode does not

The image already patches CUTLASS to admit `sm_121a` for the FP4 block-scaled MMA
(`Patched cute/mma.py: added sm_120a + sm_121a to BlockScaledMmaOp.admissible_archs`).
That is why weights load and the MoE GEMMs run. The **decode attention** kernels
(trtllm-gen MLA, DeepGEMM indexer) got no equivalent SM121 admission, so they fall back.
This arch-admission patch is the template for the fixes below.

## Approaches to a fully accelerated decode

Ordered cheapest-first.

### A. Force MLA decode onto the trtllm-gen kernel (runtime flag, cheapest, test first)

The warning names the fix: `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla`.
SGLang exposes this as the `trtllm_mla` attention backend. Lever:

- Profile knob `attention_backend: flashinfer -> trtllm_mla` (or a decode-only
  `--decode-attention-backend` if we want to keep flashinfer for prefill). Currently the
  profile sets `attention_backend: "flashinfer"` and `decode_attention_backend` is unset
  (auto -> fa2). See `roles/k8s_dgx/model_profiles/0xsero-glm-5.2-reap-504b-v2.yml` and
  `sglang_attention_backend` in `roles/k8s_dgx/defaults/main/sglang.yml`.
- Auto picked fa2 because the SM121 heuristic excludes it, NOT (necessarily) because the
  kernel is absent. GB10 is Blackwell, so the trtllm-gen MLA decode kernel MAY already
  admit SM121. If so, this is a pure config win, no rebuild.

TO VERIFY before deploy: (1) that `trtllm_mla` is a valid backend token in this image,
(2) that it supports the DSA sparse path (MLA-over-selected-KV) and not only plain MLA,
(3) boot + correctness (same 50 GSM8K) + throughput A/B vs the fa2 baseline. Risk: its
own SM121 gaps or DSA incompatibility -> boot/accuracy check is mandatory, forward-fix.

### B. Enable DeepGEMM on SM121 for the indexer (build-level, the "JIT aktivieren" lever)

Flipping `SGLANG_ENABLE_JIT_DEEPGEMM=true` alone is NOT expected to work: DeepGEMM must
first ADMIT SM121 in its codegen (same class of problem the CUTLASS `sm_121a` patch
already solved for FP4). Concretely, to get the tuned FP8 indexer/MoE GEMMs:

1. Patch DeepGEMM's arch-admission / target list to include `sm_121a` (mirror the
   existing `cute/mma.py` sm_120a/sm_121a patch pattern), OR pull a DeepGEMM version that
   supports consumer Blackwell.
2. Confirm the SM121 feature set DeepGEMM assumes (TMA, `tcgen05`, cluster-launch) is
   available on GB10; where GB10 lacks a datacenter feature (e.g. multicast/cluster),
   the kernel needs an SM121 code path or it will mis-compile / mis-execute.
3. Then set `SGLANG_ENABLE_JIT_DEEPGEMM=true` and re-baseline. Watch for silent numeric
   errors (FP8 GEMM on the wrong arch can produce plausible-but-wrong output -> gate on
   GSM8K, not just "it booted").

This is a Dockerfile/recipe change (see `scripts/patches/`), not a runtime flip. It is
the deeper lever and the only one that addresses the indexer path.

### C. tilelang indexer (alternative to DeepGEMM for the DSA index step)

`SGLANG_OPT_USE_TILELANG_INDEXER=0` today. tilelang generates the indexer kernel via its
own compiler and may cover SM121 where DeepGEMM does not. Candidate to try if B stalls:
flip to `1`, verify it builds + runs on SM121 + accuracy holds. Independent of A.

### D. Concurrency amortisation (already applied, free)

conc 16 -> ~24 tok/s aggregate vs ~3 single. KV pool and captured graphs (bs up to 32)
leave headroom, so conc 24/32 should push aggregate further until the DSA kernels
saturate. Does nothing for per-sequence latency; use it for throughput-bound eval/batch.

### E. Upstream maturation

SM121 coverage in flashinfer / trtllm-gen / DeepGEMM / sgl-kernel is moving. A future
image bump may let `backend='auto'` route the native path with no local patch. Re-check
the fa2 warning after each image bump.

### F. Architectural residual (not removable)

The DSA top-k selection over paged KV is inherent per-token overhead on top of MLA. Even
with every kernel native, DSA decode will cost more per token than dense MHA. Some of the
gap is the algorithm, not the arch.

## Recommended sequence

1. Establish the fa2 baseline: current GSM8K (50, temp 0.6) + `gen throughput` numbers.
2. **A** first (cheap, runtime): `attention_backend: trtllm_mla`, redeploy, same GSM8K +
   throughput. If it boots, stays correct, and is faster -> biggest win for the least
   work.
3. If A is blocked or insufficient, **B** (DeepGEMM SM121 admission in the recipe), then
   optionally **C**. Always gate on GSM8K correctness, not just boot.
4. Keep **D** for any throughput-bound run regardless.

Every change is forward-fix on SGLang (no rollback, no vLLM); validate with the same
`eval_gsm8k.py` 50-sample harness + the head-log `gen throughput` under load.

## References

- Live evidence: head log of `sglang-head` (2026-07-15), the `flashinfer_mla_backend.py:276`
  fa2 warning and the `SGLANG_*` env block above.
- Profile: `roles/k8s_dgx/model_profiles/0xsero-glm-5.2-reap-504b-v2.yml`
  (`attention_backend`, `moe_runner_backend`, `kv_cache_dtype`).
- Backend wiring: `roles/k8s_dgx/defaults/main/sglang.yml` (`sglang_attention_backend`),
  `roles/k8s_dgx/tasks/sglang_instance.yml` (`SGLANG_ATTENTION_BACKEND`).
- Arch-admission patch precedent: image `cute/mma.py` sm_120a/sm_121a patch; recipe
  patches under `scripts/patches/`.
- Prior findings: the DSV4 DSA kernel notes (topk_transform native works;
  `fp8_paged_mqa_logits` DeepGEMM SM121 block) and the GB10 util/stuck-rank notes.
