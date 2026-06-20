# SGLang Test Log — Qwen3.5 122B-A10B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.12 (base image)

> 🔄 **RUN IN PROGRESS** (started 2026-06-20 12:01) — 0 / 21 cases done. Matrix is running in numeric order (01 first). Case **01** booted clean (weights loaded, KV+Mamba cache allocated, mid CUDA-graph capture as of 12:06) — no boot crash. No throughput numbers yet; bench still polling readiness (HTTP 503). Updating this log as cases complete.

## Environment

| Component | Value                                                                               |
|-----------|-------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                      |
| Driver    | 580.159.03                                                                          |
| Kernel    | 6.17.0-1021-nvidia                                                                  |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                                                        |
| K3s       | v1.36.1+k3s1                                                                        |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                         |
| Image     | `scitrera/dgx-spark-sglang:0.5.12` (dgxarley default base — **no cuDNN-FP4 wheel**) |
| Model     | `nvidia/Qwen3.5-122B-A10B-NVFP4`                                                    |
| Transport | **RoCE** via SR-IOV VF                                                              |

> Matrix def: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.5-122b-a10b-nvfp4/nv580.159_sglang-0.5.12_qwen3.5-122b-nvfp4_n4_ep1.yaml`
> Raw results (after run): `kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-122b-a10b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-122b-nvfp4_4n_1pp_4tp_ep1.json`

---

## Why this matrix — SEED sweep, structural twin of the 397B-A17B run

The profile (`nvidia/Qwen3.5-122B-A10B-NVFP4`) is a **SEED** mirroring the validated `nvidia/Qwen3.5-397B-A17B-NVFP4` winner shape (fi_cutlass-MoE + triton-attn + fi_cutlass-FP4, EP=1) adapted to this model's 256-expert arch. Same `qwen3_5_moe` hybrid-attention architecture (Gated DeltaNet → SGLang "mamba" scheduler path) and a **1-layer MTP head** (`config.json: mtp_num_hidden_layers=1`, present in both the base `Qwen/Qwen3.5-122B-A10B` and this NVFP4 quant → NEXTN speculative decoding). NVFP4 footprint is well under the FP8 build's ~127 GB → fits with ample KV headroom at TP=4 (the card even runs TP=1).

Matrix structure is intentionally identical to the 397B base-image matrix so the two are **directly comparable** (cross-model NVFP4 scaling at the same parallelism).

Key structural points:
- **fp4_gemm axis collapses to `flashinfer_cutlass` only.** The base image ships no cuDNN-FP4 wheel → `flashinfer_cudnn` FP4 *should* crash "cuDNN is not available". Case **13** keeps ONE `fi_cudnn` probe to turn "absent" into a *recorded* crash. (Note: on the 397B base run this probe unexpectedly ran clean — watch for the same here.)
- **Standalone `cutlass` MoE runner is gone** (removed in 0.5.12 → `NotImplementedError` at load_model). MoE axis = `{triton, flashinfer_cutlass}`.
- `cuda_graph_max_bs=32` held across all cases (NB: the 397B run used 16).

**21 cases** = 12 no-spec (Block A) + 1 fi_cudnn crash-probe (13) + 3 trtllm probe (Block B) + 5 MTP (Block C).

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=1, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, cuda_graph_max_bs=32, fp4_gemm=flashinfer_cutlass` (except 13), MTP cases add `mem_fraction_static=0.75, mamba_scheduler_strategy=extra_buffer, SGLANG_ENABLE_SPEC_V2=1`. n=X = aggregate (sum per-request) tok/s at concurrency X.

| #  | moe_runner | attn   | fp4_gemm   | cg  | mtp     | Status        | n=1 | n=4 | n=8 | n=16 |
|----|------------|--------|------------|-----|---------|---------------|-----|-----|-----|------|
| 01 | triton     | fi     | fi_cutlass | on  | —       | PENDING       | —   | —   | —   | —    |
| 02 | triton     | fi     | fi_cutlass | off | —       | PENDING       | —   | —   | —   | —    |
| 03 | triton     | fi     | fi_cutlass | pw  | —       | PENDING       | —   | —   | —   | —    |
| 04 | triton     | triton | fi_cutlass | on  | —       | PENDING       | —   | —   | —   | —    |
| 05 | triton     | triton | fi_cutlass | off | —       | PENDING       | —   | —   | —   | —    |
| 06 | triton     | triton | fi_cutlass | pw  | —       | PENDING       | —   | —   | —   | —    |
| 07 | fi_cutlass | fi     | fi_cutlass | on  | —       | PENDING †     | —   | —   | —   | —    |
| 08 | fi_cutlass | fi     | fi_cutlass | off | —       | PENDING       | —   | —   | —   | —    |
| 09 | fi_cutlass | fi     | fi_cutlass | pw  | —       | PENDING       | —   | —   | —   | —    |
| 10 | fi_cutlass | triton | fi_cutlass | on  | —       | PENDING       | —   | —   | —   | —    |
| 11 | fi_cutlass | triton | fi_cutlass | off | —       | PENDING       | —   | —   | —   | —    |
| 12 | fi_cutlass | triton | fi_cutlass | pw  | —       | PENDING       | —   | —   | —   | —    |
| 13 | fi_cutlass | fi     | fi_cudnn   | on  | —       | PENDING ⚠️    | —   | —   | —   | —    |
| 14 | fi_trtllm  | fi     | fi_cutlass | on  | —       | PENDING ‡     | —   | —   | —   | —    |
| 15 | fi_trtllm  | fi     | fi_cutlass | pw  | —       | PENDING ‡     | —   | —   | —   | —    |
| 16 | fi_trtllm  | triton | fi_cutlass | on  | —       | PENDING ‡     | —   | —   | —   | —    |
| 17 | fi_cutlass | triton | fi_cutlass | on  | s1/d2   | PENDING       | —   | —   | —   | —    |
| 18 | fi_cutlass | triton | fi_cutlass | on  | s3/d4   | PENDING ★     | —   | —   | —   | —    |
| 19 | fi_cutlass | triton | fi_cutlass | on  | s5/d5   | PENDING       | —   | —   | —   | —    |
| 20 | fi_cutlass | triton | fi_cutlass | on  | s5/d7   | PENDING       | —   | —   | —   | —    |
| 21 | triton     | triton | fi_cutlass | on  | s3/d4   | PENDING       | —   | —   | —   | —    |

- **cg**: `on` = full CUDA graphs; `off` = eager; `pw` = piecewise. **mtp**: NEXTN `steps/draft`, `—` = off.
- **† Case 07** = the designated no-spec serving reference (fi_cutlass-MoE + fi-attn + full-CG).
- **‡ Probes** — **13** `fi_cudnn` FP4 *expected* to crash on the base image (no cuDNN wheel); **14–16** `flashinfer_trtllm` MoE (alt NVFP4 runner — crashed on the 397B run, likely crashes here too). Crash = recorded outcome, not a failure of the sweep.
- **★ Case 18** = COOKBOOK / SEED config (MTP s3/d4) — the headline number and the config currently pinned in the profile. Compare against the 397B winner (case 18 there: 40.1 / 95.1 / 125.4 / 172.6) for cross-model scaling.

---

## What to expect (hypotheses — to confirm/refute on run)

- **fi_cutlass-MoE should beat triton-MoE** at concurrency (held on the 397B run: +6.5% n=16). Best no-spec likely **case 07**.
- **CUDA graphs ON** worth the most at n=1, gap closing by n=16.
- **MTP s3/d4 (18) the sweet spot**; deeper draft chains (s5/d5, s5/d7) expected to REGRESS (397B showed monotonic falloff past d4). Cross-runner triton-MoE+MTP (21) may edge 18 at n=1 but trail at concurrency.
- **First-boot gates (SEED — verify before trusting numbers):**
  1. NVFP4 quant actually ships the MTP layer (else NEXTN init error) — the headline MTP block depends on it.
  2. No OOM at `mem_fraction_static=0.75` with MTP draft KV.
  3. `mamba_scheduler_strategy=extra_buffer` + spec_v2 boot clean (else "not compatible with radix cache" crash).
  4. EP=1 enforced (EP>1 crashes — StandardDispatcher combine bug).

## Observations

_None yet — run pending._

## Refresh

After the run, re-read the summary JSON and fill the table:
`kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-122b-a10b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-122b-nvfp4_4n_1pp_4tp_ep1.json`
