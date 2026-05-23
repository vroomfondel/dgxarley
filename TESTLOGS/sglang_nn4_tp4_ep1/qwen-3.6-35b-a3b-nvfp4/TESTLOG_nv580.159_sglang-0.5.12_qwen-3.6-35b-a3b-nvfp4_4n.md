# SGLang Test Log — Qwen3.6 35B-A3B-NVFP4 (MoE), 4 Nodes, TP=4 EP=1, v0.5.12 (first NVFP4 contact)

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                              |
| Driver    | 580.159                                                                     |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                                          |
| Kernel    | 6.17.0-1018-nvidia                                                          |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.35.3+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `scitrera/dgx-spark-sglang:0.5.12`                                          |
| Model     | `RedHatAI/Qwen3.6-35B-A3B-NVFP4`                                            |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)                                       |
| Transport | **RoCE** via SR-IOV VF                                                      |
| AllReduce | Legacy (both `SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_..._V2=0`)         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4_n4_ep1.yaml`

**First NVFP4 contact for this model** — no prior baseline. Direct A/B references go to the FP8 sibling:
- `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/TESTLOG_nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-fp8_4n.md` (same driver/image, FP8 path).

NVFP4 candidate selection: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` chosen over `unsloth/...`, `mmangkad/...`, `sakamakismile/...` — 2.24M HF downloads as of 2026-05-22, orders of magnitude the most-pulled NVFP4 variant for this model, highest likelihood of community-vetted packaging.

Toolchain delta vs FP8 sibling: identical (same image). NVFP4-relevant 0.5.12 changes:
- FlashInfer 0.6.8.post1 → 0.6.11.post1 (PRs #24452, #25129, #25310, #25335)
- sgl-kernel 0.4.2 → 0.4.2.post2 (#24457, #25326)
- PR #23590 (Cute-DSL FP4 GEMM reland) + PR #23745 (Cute-DSL NVFP4 quant kernels) — relevant for Block D (fi_cutedsl MoE).
- `SGLANG_OPT_FP8_WO_A_GEMM` default-on (#25181) — irrelevant on this NVFP4 model (FP4 path).
- Spec V2 fixes (#25204 frozen-KV, #24635 stuck-MTP DSA, #24662 breakable CG bs>1) — Block E (MTP cases).
- JIT Custom All-Reduce default-on (#24363) — explicitly disabled via `sglang_jit_allreduce=false` + dual env-var injection (`SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`) to keep the collective path comparable across runs.

See `SGLANG_v0.5.12_VERSION_CHANGES.md` for the full release delta.

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). **NVFP4** quantization (~10–12 GB on-device weights, vs ~35 GB for the FP8 variant).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
- VL-capable (vision encoder), we run text-only — no special flags.
- **NVFP4-specific**: weights are ~3× smaller than FP8 → expect significantly more headroom in the KV cache for long-context, and FP4 tensor-core MMA paths become available (`cutlass`, `flashinfer_cutlass`, `flashinfer_cutedsl` MoE runners that all crash on the FP8 variant).

## Why this matrix exists

- FP8 baseline is the throughput champion of the cluster: 0.5.12 winners ~402–406 tok/s n=8 (no-MTP) / 426.76 peak (MTP s=2). The question is whether NVFP4 — with FP4 tensor cores on GB10/SM121 plus the ~3× smaller memory footprint — can match or beat that, and what it does for KV-cache headroom.
- Unlike FP8 on this model, NVFP4 unlocks `cutlass` and `flashinfer_cutlass` MoE runners (FP8 path 6/6 crashes there with `'Fp8MoEMethod' object has no attribute 'runner'`) — fresh axis to sweep.
- First-contact validation, so the matrix is wider (42 cases vs 24 on FP8) — 4×{moe_runner} × 2×{attention} × 2×{fp4_gemm_backend} × 3×{CG variant} with `cutedsl × cuDNN-FP4` cut as a too-experimental cross-product, plus 6 MTP cases (2 anchors + 4 num_steps sweep).

## What changes vs the FP8 sibling matrix

1. **NVFP4 MoE runners now in play.** FP8 was stuck on `triton` (only one that worked); NVFP4 sweeps `triton`, `flashinfer_cutlass`, `cutlass`, `flashinfer_cutedsl` — three more MoE backends to characterize, the most important being `flashinfer_cutlass` (matching the SM121-default for most other NVFP4 models per CLAUDE.md).
2. **`fp4_gemm_backend` axis added.** New axis (`flashinfer_cutlass` vs `flashinfer_cudnn`) — independent of `moe_runner_backend`. Per CLAUDE.md: `flashinfer_cutlass` works almost everywhere; cuDNN-FP4 is the more experimental of the two.
3. **No FP8 → NVFP4 quality migration baseline.** TTR_min / coherence floors are unknown on NVFP4. Per-case output-quality check is mandatory.
4. **`cutlass_moe_fp4` known-bad on SM121 for most NVFP4 models** (CLAUDE.md note: cutlass-direct + eager produces `!`-token collapse; cutlass-direct only works for `nvidia/Qwen3.5-397B-A17B-NVFP4`). On Qwen3.6-35B-NVFP4 the SM121 default is `flashinfer_cutlass` — Block C (`cutlass` direct, Tests 25–36) will likely produce either crashes or word-salad in eager rows (26, 29, 32, 35). Validate before relying on any cutlass-direct number.
5. **`flashinfer_cutedsl` MoE expected to work on NVFP4.** On FP8 it pre-checks fail-fast with `Invalid quantization 'None'. FlashInfer CuteDSL MOE currently supports only: 'modelopt_fp4'.` — the path is NVFP4-designed. Block D (Tests 37–42) is its first real test on our cluster.
6. **MTP sweep mirrors FP8 0.5.12 winners.** Anchors (43–44) reuse FP8 Tests 13–14 shapes; num_steps sweep (45–48) reuses FP8 winner shape (Test 03 / Tests 21–24) for direct A/B. If a different no-MTP shape wins on NVFP4, the sweep should be re-run on the actual winner.

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, context_length=262144, num_experts=256, enable_eplb=false` unless noted. NVFP4 → FP4 GEMM backend sweep instead of an FP8 one. `flashinfer_cutedsl × flashinfer_cudnn` cross-product cut (too-experimental).

### Block A — triton MoE (FP8-validated reliable baseline) — Tests 01–12

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status     | n=1 tok/s | n=4 peak | n=8 peak   |
|----|------------|-----------|--------------|----------------|---------------|------------|-----------|----------|------------|
| 01 | triton     | fi        | fi_cutlass   | false          | true          | ok         | 71.89     | 245.49   | 407.25     |
| 02 | triton     | fi        | fi_cutlass   | true           | true          | ok         | 15.07     | 84.94    | 169.44     |
| 03 | triton     | fi        | fi_cutlass   | false          | false         | ok⁺        | 75.43     | 244.98   | 353.06     |
| 04 | triton     | triton    | fi_cutlass   | false          | true          | ok         | 61.13     | 245.13   | 404.19     |
| 05 | triton     | triton    | fi_cutlass   | true           | true          | ok         | 14.60     | 84.12    | 165.28     |
| 06 | triton     | triton    | fi_cutlass   | false          | false         | ok⁺        | 56.69     | 245.53   | 403.03     |
| 07 | triton     | fi        | fi_cudnn     | false          | true          | **crash S**| —         | —        | —          |
| 08 | triton     | fi        | fi_cudnn     | true           | true          | **crash B**| —         | —        | —          |
| 09 | triton     | fi        | fi_cudnn     | false          | false         | **crash S**| —         | —        | —          |
| 10 | triton     | triton    | fi_cudnn     | false          | true          | **crash S**| —         | —        | —          |
| 11 | triton     | triton    | fi_cudnn     | true           | true          | **crash B**| —         | —        | —          |
| 12 | triton     | triton    | fi_cudnn     | false          | false         | **crash S**| —         | —        | —          |

### Block B — flashinfer_cutlass MoE (crashed on FP8) — Tests 13–24

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|-----------|----------|----------|
| 13 | fi_cutlass | fi        | fi_cutlass   | false          | true          | **crash S** | — | — | — |
| 14 | fi_cutlass | fi        | fi_cutlass   | true           | true          | **crash S** | — | — | — |
| 15 | fi_cutlass | fi        | fi_cutlass   | false          | false         | **crash S** | — | — | — |
| 16 | fi_cutlass | triton    | fi_cutlass   | false          | true          | **crash S** | — | — | — |
| 17 | fi_cutlass | triton    | fi_cutlass   | true           | true          | **crash S** | — | — | — |
| 18 | fi_cutlass | triton    | fi_cutlass   | false          | false         | **crash S** | — | — | — |
| 19 | fi_cutlass | fi        | fi_cudnn     | false          | true          | **crash S** | — | — | — |
| 20 | fi_cutlass | fi        | fi_cudnn     | true           | true          | **crash S** | — | — | — |
| 21 | fi_cutlass | fi        | fi_cudnn     | false          | false         | **crash S** | — | — | — |
| 22 | fi_cutlass | triton    | fi_cudnn     | false          | true          | **crash S** | — | — | — |
| 23 | fi_cutlass | triton    | fi_cudnn     | true           | true          | **crash S** | — | — | — |
| 24 | fi_cutlass | triton    | fi_cudnn     | false          | false         | **crash S** | — | — | — |

### Block C — cutlass MoE (direct, was N/A on FP8) — Tests 25–36

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|-----------|----------|----------|
| 25 | cutlass    | fi        | fi_cutlass   | false          | true          | **timeout** | — | — | — |
| 26 | cutlass    | fi        | fi_cutlass   | true           | true          | ok          | 15.33 | 84.36  | 169.60 |
| 27 | cutlass    | fi        | fi_cutlass   | false          | false         | ok          | 67.13 | 247.03 | 405.81 |
| 28 | cutlass    | triton    | fi_cutlass   | false          | true          | ok          | 55.20 | 237.66 | 401.06 |
| 29 | cutlass    | triton    | fi_cutlass   | true           | true          | ok          | 15.31 | 83.52  | 166.88 |
| 30 | cutlass    | triton    | fi_cutlass   | false          | false         | ok          | 74.66 | 242.67 | 404.53 |
| 31 | cutlass    | fi        | fi_cudnn     | false          | true          | **crash S** | — | — | — |
| 32 | cutlass    | fi        | fi_cudnn     | true           | true          | **crash B** | — | — | — |
| 33 | cutlass    | fi        | fi_cudnn     | false          | false         | **crash S** | — | — | — |
| 34 | cutlass    | triton    | fi_cudnn     | false          | true          | **crash S** | — | — | — |
| 35 | cutlass    | triton    | fi_cudnn     | true           | true          | **crash B** | — | — | — |
| 36 | cutlass    | triton    | fi_cudnn     | false          | false         | **crash S** | — | — | — |

### Block D — flashinfer_cutedsl MoE (NVFP4-only design, was crash B on FP8) — Tests 37–42

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|-----------|----------|----------|
| 37 | fi_cutedsl | fi        | fi_cutlass   | false          | true          | **crash S** | — | — | — |
| 38 | fi_cutedsl | fi        | fi_cutlass   | true           | true          | **crash S** | — | — | — |
| 39 | fi_cutedsl | fi        | fi_cutlass   | false          | false         | **crash S** | — | — | — |
| 40 | fi_cutedsl | triton    | fi_cutlass   | false          | true          | **crash S** | — | — | — |
| 41 | fi_cutedsl | triton    | fi_cutlass   | true           | true          | **crash S** | — | — | — |
| 42 | fi_cutedsl | triton    | fi_cutlass   | false          | false         | **crash S** | — | — | — |

### Block E — MTP (NEXTN) anchors + winner-shape num_steps sweep — Tests 43–48

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | spec      | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|-----------|--------|-----------|----------|----------|
| 43 | triton     | triton    | fi_cutlass   | false          | false         | NEXTN s=3 | ok          | 94.93 | 265.49 | 405.89     |
| 44 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=3 | ok          | 78.20 | 274.03 | 423.71     |
| 45 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=2 | **ok 🏆**   | 82.06 | 275.70 | **438.07** |
| 46 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=3 | ok          | 78.42 | 247.59 | 387.42     |
| 47 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=4 | ok          | 81.62 | 245.36 | 366.38     |
| 48 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=5 | ok          | 71.97 | 230.53 | 350.36     |

### Block F — flashinfer_trtllm MoE (added post-initial-sweep) — Tests 49–54

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status      | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|-------------|-----------|----------|----------|
| 49 | fi_trtllm  | fi        | fi_cutlass   | false          | true          | **crash S** | — | — | — |
| 50 | fi_trtllm  | fi        | fi_cutlass   | true           | true          | **crash S** | — | — | — |
| 51 | fi_trtllm  | fi        | fi_cutlass   | false          | false         | **crash S** | — | — | — |
| 52 | fi_trtllm  | triton    | fi_cutlass   | false          | true          | **crash S** | — | — | — |
| 53 | fi_trtllm  | triton    | fi_cutlass   | true           | true          | **crash S** | — | — | — |
| 54 | fi_trtllm  | triton    | fi_cutlass   | false          | false         | **crash S** | — | — | — |

### Block G — cuDNN-FP4 GEMM re-test (cuDNN-rebuilt image) — Test 55

| #  | moe_runner | attention | fp4_gemm  | dis_cuda_graph | dis_piecewise | spec      | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|-----------|----------------|---------------|-----------|--------|-----------|----------|----------|
| 55 | triton     | fi        | fi_cudnn  | false          | false         | NEXTN s=2 | ok     | 81.52     | 262.45   | 393.40   |

### Column Legend

| Column         | Description                                                                                                                    |
|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), `cutlass` (direct), `flashinfer_cutedsl` (`fi_cutedsl`), `flashinfer_trtllm` (`fi_trtllm`) |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                                                     |
| fp4_gemm       | `fp4_gemm_backend` — `fi_cutlass` = `flashinfer_cutlass`, `fi_cudnn` = `flashinfer_cudnn`                                      |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                                               |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = fixed-BS graphs only, false = piecewise variable-length graphs                         |
| spec           | speculative decoding — `NEXTN s=N` = MTP with `speculative_num_steps=N`, `eagle_topk=1`, `num_draft_tokens=N+1`                |

---

## Pre-run hypotheses

1. **Block A (triton MoE, Tests 01–12)**: most likely to all pass. The `fi_cutlass` FP4 GEMM (Tests 01–06) should be a stable baseline. The `fi_cudnn` FP4 GEMM (Tests 07–12) is the experimental axis — risk of CG-capture crash or output regression. Expect n=8 peak in the 350–450 tok/s ballpark — could go either way vs FP8 (FP4 tensor-core MMA is faster, but the FP4 dequant path has overhead).
2. **Block B (fi_cutlass MoE, Tests 13–24)**: should work on NVFP4 (the path that crashes on FP8 with `Fp8MoEMethod.runner` is now a real FP4 path). This is the CLAUDE.md SM121-default for most NVFP4 models — strong candidate for the overall winner. EP=1 sidesteps the `cutlass_moe_fp4` EP-assertion bug, and there's no MoE-in-eager case here that would trigger the `!`-token collapse outside of Block C.
3. **Block C (cutlass MoE direct, Tests 25–36)**: known-bad on SM121 for most NVFP4 models — `cutlass_moe_fp4` shared-mem and EP-assert crashes. EP=1 keeps the EP-assert bug latent, but eager rows (26, 29, 32, 35) are high-risk for `!`-collapse per CLAUDE.md. Even if they don't crash, output-quality check is the gate.
4. **Block D (fi_cutedsl MoE, Tests 37–42)**: first real test on our cluster. PR #23590 (Cute-DSL FP4 GEMM reland) + PR #23745 (Cute-DSL NVFP4 quant kernels) make this potentially viable. If it works, the eager case (38, 41) is worth a hard look — cutedsl is a re-implementation in DSL form, eager-mode bugs would be a new class of failure.
5. **Block E (MTP, Tests 43–48)**: anchors (43–44) tie back to FP8 0.5.12 Tests 13–14 (which landed in the 396–402 tok/s n=8 range). The num_steps sweep mirrors FP8 Tests 21–24 — on FP8 0.5.12 the sweet spot was s=2 (peak 426.76). NVFP4 mamba-state path is the same kernel as FP8 — expect similar shape, but absolute numbers depend entirely on how NVFP4 vs FP8 compute throughput nets out.
6. **Output-quality**: hybrid-mamba word-salad regression (0.5.11 FP8 issue) was fixed via the `0c2bdd4` profile patch (`is_layer_skipped` substring + `sampling_overrides={}`). The NVFP4 profile should inherit the same fix — verify by pattern-grep + TTR + tail-eyeball per case. NVFP4 weights can themselves degrade quality slightly vs FP8; baseline floor unknown.

---

## Results

**Matrix run complete (2026-05-22 ~17:22 → 2026-05-23 ~00:30 UTC+2, two phases).** Originally 48 cases (Tests 01–48); matrix YAML extended on 2026-05-22 evening with Block F (fi_trtllm MoE, 6 cases) + Block G (cuDNN-FP4 GEMM re-test on rebuilt image, 1 case) → **55 cases attempted total**, **18 ok**, **36 crashed/timeout**, **1 ok-with-quality-flag**.

### Crash legend

- **crash S** (`startup_crash`): SGLang head/worker pod restarts — never reaches inference. Read as "this kernel combination doesn't even compile/load on SM121 for this model".
- **crash B** (`bench_crash`): pod starts, but every benchmark request fails (n=1: 0/1, n=4: 0/4, n=8: 0/8). Inference reachable, but first forward pass errors out.
- **timeout**: `SGLang not ready after 900s` — pod neither restarts nor becomes ready in 15 min.

### Completed `ok` cases — Block A (triton MoE, fi_cutlass-fp4 GEMM only)

| #  | Config                                                  | n=1 tok/s | n=4 peak | n=8 peak | n=8 avg/req | n=8 ok | Finish reasons   | n=8 TTR_min | Output     |
|----|---------------------------------------------------------|----------:|---------:|---------:|------------:|--------|------------------|------------:|------------|
| 01 | triton-moe + fi-attn + fi_cutlass-fp4, CG on            |     71.89 |   245.49 |   407.25 |       50.91 | 8/8    | length×8         |       0.691 | clean ✓    |
| 02 | …+ **cuda_graph off**                                   |     15.07 |    84.94 |   169.44 |       21.18 | 8/8    | length×8         |       0.643 | clean ✓    |
| 03 | …+ **piecewise**                                        |     75.43 |   244.98 |   353.06 |       50.44 | 7/8    | length×6, stop×1 |       0.609 | **flag** ⚠ |
| 04 | triton-moe + **triton-attn** + fi_cutlass-fp4, CG on    |     61.13 |   245.13 |   404.19 |       50.52 | 8/8    | length×8         |       0.642 | clean ✓    |
| 05 | …+ **cuda_graph off**                                   |     14.60 |    84.12 |   165.28 |       20.66 | 8/8    | length×8         |       0.698 | clean ✓    |
| 06 | …+ **piecewise**                                        |     56.69 |   245.53 |   403.03 |       50.38 | 8/8    | length×7, stop×1 |       0.645 | clean ✓    |

### Completed `ok` cases — Block C (cutlass-direct MoE, fi_cutlass-fp4 GEMM only)

| #  | Config                                                  | n=1 tok/s | n=4 peak | n=8 peak | n=8 avg/req | n=8 ok | Finish reasons | n=8 TTR_min | Output  |
|----|---------------------------------------------------------|----------:|---------:|---------:|------------:|--------|----------------|------------:|---------|
| 26 | cutlass-moe + fi-attn + fi_cutlass-fp4, **eager**       |     15.33 |    84.36 |   169.60 |       21.20 | 8/8    | length×8       |       0.566 | clean ✓ |
| 27 | …+ piecewise                                            |     67.13 |   247.03 |   405.81 |       50.73 | 8/8    | length×8       |       0.531 | clean ✓ |
| 28 | cutlass-moe + triton-attn + fi_cutlass-fp4, CG on       |     55.20 |   237.66 |   401.06 |       50.13 | 8/8    | length×8       |       0.647 | clean ✓ |
| 29 | …+ **eager**                                            |     15.31 |    83.52 |   166.88 |       20.86 | 8/8    | length×8       |       0.624 | clean ✓ |
| 30 | …+ piecewise                                            |     74.66 |   242.67 |   404.53 |       50.57 | 8/8    | length×8       |       0.593 | clean ✓ |

### Completed `ok` cases — Block E (MTP / NEXTN sweep)

| #   | Config                                              | n=1 tok/s | n=4 peak | n=8 peak    | n=8 avg/req | n=8 ok | Finish reasons   | n=8 TTR_min | Output  |
|-----|-----------------------------------------------------|----------:|---------:|------------:|------------:|--------|------------------|------------:|---------|
| 43  | triton-moe + triton-attn + piecewise, **MTP s=3**   |     94.93 |   265.49 |     405.89  |       50.74 | 8/8    | length×8         |       0.615 | clean ✓ |
| 44  | triton-moe + fi-attn + piecewise, **MTP s=3**       |     78.20 |   274.03 |     423.71  |       52.96 | 8/8    | length×8         |       0.656 | clean ✓ |
| 45  | winner-shape + **MTP s=2** 🏆                       |     82.06 |   275.70 | **438.07**  |       54.76 | 8/8    | length×8         |       0.607 | clean ✓ |
| 46  | winner-shape + **MTP s=3**                          |     78.42 |   247.59 |     387.42  |       48.43 | 8/8    | length×8         |       0.715 | clean ✓ |
| 47  | winner-shape + **MTP s=4**                          |     81.62 |   245.36 |     366.38  |       45.80 | 7/8    | length×7, stop×1 |       0.665 | clean ✓ |
| 48  | winner-shape + **MTP s=5**                          |     71.97 |   230.53 |     350.36  |       43.80 | 8/8    | length×8         |       0.634 | clean ✓ |

### Crash summary

| Block | Cases | Status | Root cause |
|-------|-------|--------|------------|
| A (triton-MoE) × `fi_cudnn` FP4 GEMM | 07–12 | **6/6 crash** (4× S, 2× B) | `RuntimeError: cuDNN is not available` — Python `nvidia-cudnn-cu12` missing in image at run time. **Fixed after image rebuild** — see Test 55 |
| B (`fi_cutlass`-MoE), both FP4 GEMMs | 13–24 | **12/12 crash** (all S) | `_handle_moe_kernel_config` whitelist: `Invalid quantization 'compressed-tensors'` (only `modelopt_*` / bf16 accepted) |
| C (cutlass-direct MoE), `fi_cutlass` FP4 GEMM | 25 | timeout (head not ready in 900s) | CG-capture cold-start ~62 s/batch — 900 s startup deadline insufficient for `cutlass_moe_fp4` first-time autotune. Tests 27/30 with piecewise CG worked |
| C (cutlass-direct MoE) × `fi_cudnn` FP4 GEMM | 31–36 | **6/6 crash** (4× S, 2× B) | Same cuDNN-missing as Block A's cuDNN cases |
| D (`fi_cutedsl`-MoE) | 37–42 | **6/6 crash** (all S) | `_handle_moe_kernel_config` whitelist: `Invalid quantization 'compressed-tensors'` (only `modelopt_fp4` accepted — even tighter than B) |
| F (`fi_trtllm`-MoE) | 49–54 | **6/6 crash** (all S) | trtllm-MoE GEMM picks `sm100f`-suffixed Blackwell-Datacenter kernel (`bmm_E2m1_..._sm100f`); we're on **SM121/GB10** (Blackwell-Consumer). Architecture mismatch in trtllm's kernel selector — likely upstream-unfixable until trtllm adds an SM121 codepath |

### Findings (so far)

1. **`fp4_gemm_backend: flashinfer_cudnn` is completely broken on this NVFP4 model.** Every single case in Blocks A, C using the cuDNN-FP4 GEMM crashes (12/12) — pod startup or first forward pass. The crash signature is independent of MoE runner. cuDNN-FP4 is the more experimental of the two FP4 GEMM backends per the matrix design notes; on Qwen3.6-35B-NVFP4 it's a no-go. **Recommendation:** drop `fi_cudnn` from future first-contact matrices for this model class until upstream stabilizes it.
2. **`moe_runner_backend: flashinfer_cutlass` doesn't start on this NVFP4 model.** All 12 Block-B cases fail with `startup_crash`. This contradicts the pre-run hypothesis (CLAUDE.md "SM121-default for most NVFP4 models" — *most*, not *all*). The Qwen3.6 35B-NVFP4 architecture (Gated DeltaNet hybrid, expert-intermediate=512) appears not to be supported by the cutlass-FP4 MoE kernel as currently shipped in FlashInfer 0.6.11.post1 + sgl-kernel 0.4.2.post2. **Worth a head-log dive on Test 13 to confirm the exact error** before adding the model to a known-bad list.
3. **`moe_runner_backend: flashinfer_cutedsl` doesn't start either.** All 6 Block-D cases fail with `startup_crash`. PR #23590 (Cute-DSL FP4 GEMM reland) + PR #23745 (Cute-DSL NVFP4 quant kernels) made the path nominally viable, but apparently not for this MoE topology.
4. **`moe_runner_backend: cutlass` (direct) WORKS — including eager mode.** Tests 26 and 29 (cutlass-direct + `disable_cuda_graph: true`) finished cleanly with `length×8`, TTR_min 0.566 / 0.624, and visible-text spot-checks show diverse coherent content (programming, physics, mathematical logic) with **no `!`-token collapse**. This makes **Qwen3.6-35B-NVFP4 a second exception** alongside `nvidia/Qwen3.5-397B-A17B-NVFP4` to the CLAUDE.md note that "Eager mode is broken on ANY `cutlass_moe_fp4` path". The CLAUDE.md note should be downgraded from "ANY" to "most NVFP4 models — exceptions: Qwen3.5-397B-NVFP4, Qwen3.6-35B-NVFP4".
5. **No NVFP4 throughput win vs FP8 with any working MoE runner.** Best n=8 peak so far is Test 01 (triton-moe) at 407.25 — essentially tied with FP8 0.5.12 Test 01 (406.44). Block C's best (Test 27, cutlass-direct + piecewise) lands at 405.81 — also a tie. The ~3× smaller weights buy KV-cache headroom and faster cold-start, **not** throughput on GB10 at this concurrency.
6. **Working surface area is small: 11/42 cases.** All `ok` cases use `fp4_gemm_backend: flashinfer_cutlass`. The viable MoE backends are `triton` and `cutlass` (direct). Production candidate is the same shape as FP8: triton-MoE + fi-attn + fi_cutlass-fp4 GEMM + (full or piecewise) CG.
7. **Output quality clean across all 11 `ok` cases except Test 03's repetition flag.** TTR_min ≥ 0.531 (Test 27, the lowest). Test 06 has `length×7, stop×1` like Test 03 but **without** a repetition abort — so the `stop`-finish pattern alone is benign; Test 03's `repetition` flag stands alone and is worth re-running once to see if it's reproducible or a one-shot detector blip.
8. **Block E (MTP) — new cluster-wide n=8 peak record: Test 45 (winner-shape + MTP s=2) at 438.07 tok/s.** Surpasses
   - the best no-MTP NVFP4 result (Test 01 at 407.25) by **+7.6 %**, and
   - the FP8 0.5.12 winner Test 21 (peak 426.76) by **+2.6 %** — the first time on this cluster that NVFP4 *beats* FP8 on n=8 throughput.
   The sweet spot is unambiguously **s=2**: Test 45 = 438.07 > Test 46 (s=3) = 387.42 > Test 47 (s=4) = 366.38 > Test 48 (s=5, n=8 pending; n=1/n=4 already trending lower). Same shape as the FP8 0.5.12 sweep (peak at s=2 there too). All MTP cases also lift n=1 vs no-MTP (78–95 vs Test 01's 71.89) — draft pre-fill amortizes even under single-tenant load.
9. **Output quality clean across all Block E cases.** TTR_min 0.607–0.715, all finish `length×8` except Test 47 (1× natural `stop`, no repetition flag). The `0c2bdd4` profile fix carries over to the NVFP4 + MTP path as expected.
10. **Production recommendation**: NVFP4 profile flipped to **Test 45 shape** — triton-MoE + fi-attn + fi_cutlass-fp4 GEMM + piecewise CG + MTP NEXTN s=2 (`speculative_num_steps=2, eagle_topk=1, num_draft_tokens=3, mamba_scheduler_strategy=extra_buffer, enable_spec_v2=true`). Same shape as the active FP8 profile (per [[reference_testlog]] FP8 0.5.12 testlog) but on NVFP4 weights for KV-cache headroom + 7 % throughput. Profile file `roles/k8s_dgx/model_profiles/redhatai-qwen3.6-35b-a3b-nvfp4.yml` already carries these values — the seed config matched the eventual winner so only the comments needed an update.

### Findings — Block F (fi_trtllm MoE) + Block G (cuDNN-FP4 re-test)

Added on 2026-05-22 evening after the initial 48-case sweep finished, the matrix YAML was extended with two follow-ups.

11. **`fi_trtllm` MoE is unusable on GB10 (SM121).** All 6 Block-F cases passed the SGLang pre-check (`flashinfer_trtllm` evidently has no `compressed-tensors` blacklist, unlike Blocks B and D), got through model loading, NCCL init, KV-cache allocation, and even entered the first MoE forward — then crashed in the trtllm AutoTuner:
    ```
    flashinfer/fused_moe/core.py: trtllm_fp4_block_scale_moe →
    /workspace/csrc/trtllm_batched_gemm_runner.cu:278:
      RuntimeError: Error in function 'run' / Error occurred when running GEMM!
      Kernel: bmm_E2m1_..._sm100f
    ```
    The `sm100f` suffix in the kernel name is the give-away: that's a Blackwell-**Datacenter** (SM100) kernel. We're on **GB10 / SM121** (Blackwell-Consumer/Spark). FlashInfer's trtllm-MoE kernel selector dispatches to an SM100-only path with no SM121 fallback. This is a **hardware/architecture-level mismatch**, not a config issue — needs an upstream `trtllm_fp4_block_scale_moe` SM121 codepath, or a kernel selector that detects SM121 and refuses to dispatch instead of crashing.
12. **cuDNN-FP4 GEMM (Test 55) works on the rebuilt image, but is slower than fi_cutlass-FP4.** Re-test of the winner shape (triton-MoE + fi-attn + piecewise CG + MTP s=2) with `fp4_gemm_backend: flashinfer_cudnn`: **n=8 peak 393.40** vs Test 45's `flashinfer_cutlass` 438.07 → **−10.2 %**. Output quality clean: TTR_min 0.694, all 8/8 `length`, no salad triggers. Conclusion:
    - The cuDNN-image-rebuild (adding `nvidia-cudnn-cu12` + `nvidia-cudnn-frontend`) DID fix the Block-A/C `fi_cudnn` startup crashes — the path now goes through end-to-end.
    - But **cuDNN-FP4 GEMM is not a competitive backend on GB10** for this model class — `flashinfer_cutlass` wins by a comfortable 10 %. Production profile stays on `fp4_gemm_backend: flashinfer_cutlass`.
    - A full Block-A re-sweep with the rebuilt image would now succeed on Tests 07–12 and 31–36, but unless cuDNN-FP4 is needed for a specific other model, it's diminishing-returns work.

(Re-run via `kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4_n4_ep1.yaml`.)

---

## Action items after the matrix run

- [ ] Fill the four block tables with actual results
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball) — first NVFP4 contact, no prior quality floor
- [ ] Identify the no-MTP winner shape; if it diverges from the FP8 winner (triton-moe + fi-attn + piecewise CG), re-anchor the MTP sweep
- [ ] Compute Δ vs FP8 sibling on the corresponding 0.5.12 winners (Test 01 / 03 / 14 / 21 of FP8 ↔ closest NVFP4 shape)
- [ ] If `fi_cutlass` MoE (Block B) wins: candidate to flip the production profile (current FP8 profile uses triton MoE — see `model_profiles/Qwen--Qwen3.6-35B-A3B-FP8.yml`; an NVFP4 profile file would need to be added)
- [ ] If Block C `cutlass`-direct crashes or produces `!`-collapse: confirm CLAUDE.md note ("cutlass-direct only works for nvidia/Qwen3.5-397B-A17B-NVFP4") and document for this model explicitly
- [ ] If Block D `fi_cutedsl` works: this is the first datapoint on our cluster — log absolute n=8 peak and a TTR check for any DSL-vs-cutlass numerical divergence
