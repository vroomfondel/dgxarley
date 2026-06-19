# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.12 (base image)

> ⏳ **RUN IN PROGRESS** — 1 / 21 cases complete as of 2026-06-19 ~09:15. Case 02 server is up (2/2) and mid-benchmark (n1→n16, ~26 min/case); not yet written to the summary. Numbers for cases 02–21 are pending; this log will be filled as the matrix advances.

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.159.03                                         |
| Kernel    | 6.17.0-1021-nvidia                                 |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                       |
| K3s       | v1.36.1+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.12` (dgxarley default base — **no cuDNN-FP4 wheel**) |
| Model     | `nvidia/Qwen3.5-397B-A17B-NVFP4`                   |
| Transport | **RoCE** via SR-IOV VF                             |

> Matrix def: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.5-397b-a17b-nvfp4/nv580.159_sglang-0.5.12_qwen3.5-397b-nvfp4_n4_ep1.yaml`
> Raw results: `kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-397b-a17b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-397b-nvfp4_4n_1pp_4tp_ep1.json`

---

## Why this matrix — base image twin of the 0.5.12-cudnn run

Same model/topology as `TESTLOG_nv580.159_sglang-0.5.12-cudnn_...` but on the **plain** `scitrera/dgx-spark-sglang:0.5.12` (the dgxarley default base). Goal: does the base image match the cudnn build on the headline fi_cutlass-MoE + MTP config (the cudnn winner was Test 29 @ 39.1/89.7/120.7/163.6)?

Key structural difference from the cudnn matrix:
- **fp4_gemm axis collapses to `flashinfer_cutlass` only.** The base image ships no cuDNN-FP4 wheel → `flashinfer_cudnn` FP4 crashes "cuDNN is not available". So the no-spec cartesian halves to 12 (vs 24 on cudnn). Case **13** keeps ONE `fi_cudnn` probe to turn "absent" into a *recorded* crash.
- Everything else identical: standalone `cutlass` MoE is gone (removed in 0.5.12), `cuda_graph_max_bs=16`, the `flashinfer_trtllm` MoE probe (14–16), and the MTP depth sweep (17–21).

**21 cases** = 12 no-spec (Block A) + 1 fi_cudnn crash-probe (13) + 3 trtllm probe (Block B) + 5 MTP (Block C).

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=1, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, cuda_graph_max_bs=16, fp4_gemm=flashinfer_cutlass` (except 13), MTP cases add `mem_fraction_static=0.75, mamba_scheduler_strategy=extra_buffer, SGLANG_ENABLE_SPEC_V2=1`. n=X = aggregate (sum per-request) tok/s at concurrency X.

| #  | moe_runner | attn   | fp4_gemm   | cg  | mtp     | Status      | n=1      | n=4      | n=8      | n=16      |
|----|------------|--------|------------|-----|---------|-------------|----------|----------|----------|-----------|
| 01 | triton     | fi     | fi_cutlass | on  | —       | **DONE**    | 21.0     | 64.3     | 98.4     | 136.2     |
| 02 | triton     | fi     | fi_cutlass | off | —       | ⏳ running   | —        | —        | —        | —         |
| 03 | triton     | fi     | fi_cutlass | pw  | —       | pending     | —        | —        | —        | —         |
| 04 | triton     | triton | fi_cutlass | on  | —       | pending     | —        | —        | —        | —         |
| 05 | triton     | triton | fi_cutlass | off | —       | pending     | —        | —        | —        | —         |
| 06 | triton     | triton | fi_cutlass | pw  | —       | pending     | —        | —        | —        | —         |
| 07 | fi_cutlass | fi     | fi_cutlass | on  | —       | pending†    | —        | —        | —        | —         |
| 08 | fi_cutlass | fi     | fi_cutlass | off | —       | pending     | —        | —        | —        | —         |
| 09 | fi_cutlass | fi     | fi_cutlass | pw  | —       | pending     | —        | —        | —        | —         |
| 10 | fi_cutlass | triton | fi_cutlass | on  | —       | pending     | —        | —        | —        | —         |
| 11 | fi_cutlass | triton | fi_cutlass | off | —       | pending     | —        | —        | —        | —         |
| 12 | fi_cutlass | triton | fi_cutlass | pw  | —       | pending     | —        | —        | —        | —         |
| 13 | fi_cutlass | fi     | fi_cudnn   | on  | —       | pending ‡   | —        | —        | —        | —         |
| 14 | fi_trtllm  | fi     | fi_cutlass | on  | —       | pending ‡   | —        | —        | —        | —         |
| 15 | fi_trtllm  | fi     | fi_cutlass | pw  | —       | pending ‡   | —        | —        | —        | —         |
| 16 | fi_trtllm  | triton | fi_cutlass | on  | —       | pending ‡   | —        | —        | —        | —         |
| 17 | fi_cutlass | triton | fi_cutlass | on  | s1/d2   | pending     | —        | —        | —        | —         |
| 18 | fi_cutlass | triton | fi_cutlass | on  | s3/d4   | pending ★   | —        | —        | —        | —         |
| 19 | fi_cutlass | triton | fi_cutlass | on  | s5/d5   | pending     | —        | —        | —        | —         |
| 20 | fi_cutlass | triton | fi_cutlass | on  | s5/d7   | pending     | —        | —        | —        | —         |
| 21 | triton     | triton | fi_cutlass | on  | s3/d4   | pending     | —        | —        | —        | —         |

- **cg**: `on` = full CUDA graphs; `off` = eager; `pw` = piecewise. **mtp**: NEXTN `steps/draft`, `—` = off.
- **† Case 07** = the designated no-spec serving reference (fi_cutlass-MoE + fi-attn + full-CG).
- **‡ Probes** — expected/possible crashes: **13** `fi_cudnn` FP4 (no cuDNN wheel on this base image → should crash "cuDNN not available"); **14–16** `flashinfer_trtllm` MoE (crashed on the cudnn twin — likely crashes here too).
- **★ Case 18** = COOKBOOK config, the one to compare against the cudnn winner (Test 29: 39.1/89.7/120.7/163.6) and the pinned profile config.

---

## Observations so far

- **Case 01 (triton-MoE baseline) ≈ identical between base and cudnn images.** Base 0.5.12: 21.0 / 64.3 / 98.4 / 136.2 (n1/4/8/16) vs cudnn case 01: 21.4 / 65.9 / 98.0 / 135.9 — within noise. So the cuDNN build does **not** change the triton-MoE baseline; any advantage must come from the `fi_cudnn` FP4 path (absent here) or elsewhere. The decisive comparison is case **18** (fi_cutlass-MoE + MTP s3/d4) vs cudnn Test 29 — still pending.
- (more to follow as cases complete)

## Refresh

Re-read the summary JSON and update the table:
`kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-397b-a17b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-397b-nvfp4_4n_1pp_4tp_ep1.json`
