# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.12 (base image)

> ⏳ **RUN IN PROGRESS** — 17 / 21 cases complete as of 2026-06-19 ~15:07. All 3 trtllm probes (14–16) startup-crashed. MTP block started: 17 (s1/d2) = 31.4/82.2/109.7/144.5. Case 18 (cookbook s3/d4 — decisive vs cudnn Test 29) running. 18–21 pending.

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
| Model     | `nvidia/Qwen3.5-397B-A17B-NVFP4`                                                    |
| Transport | **RoCE** via SR-IOV VF                                                              |

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
| 02 | triton     | fi     | fi_cutlass | off | —       | **DONE**    | 14.3     | 62.6     | 95.5     | 136.3     |
| 03 | triton     | fi     | fi_cutlass | pw  | —       | **DONE**    | 21.2     | 64.2     | 100.3    | 138.8     |
| 04 | triton     | triton | fi_cutlass | on  | —       | **DONE**    | 20.8     | 66.4     | 99.4     | 138.6     |
| 05 | triton     | triton | fi_cutlass | off | —       | **DONE**    | 13.4     | 60.7     | 95.5     | 135.3     |
| 06 | triton     | triton | fi_cutlass | pw  | —       | **DONE**    | 20.7     | 64.5     | 99.4     | 136.4     |
| 07 | fi_cutlass | fi     | fi_cutlass | on  | —       | **DONE**    | 21.3     | 69.3     | 103.6    | 147.8     |
| 08 | fi_cutlass | fi     | fi_cutlass | off | —       | **DONE**    | 19.9     | 65.6     | 100.3    | 143.8     |
| 09 | fi_cutlass | fi     | fi_cutlass | pw  | —       | **DONE**    | 21.6     | 68.3     | 103.6    | 143.0     |
| 10 | fi_cutlass | triton | fi_cutlass | on  | —       | **DONE**    | 22.8     | 68.6     | 105.0    | 146.6     |
| 11 | fi_cutlass | triton | fi_cutlass | off | —       | **DONE**    | 21.0     | 67.2     | 100.9    | 144.7     |
| 12 | fi_cutlass | triton | fi_cutlass | pw  | —       | **DONE**    | 20.8     | 70.7     | 105.4    | 145.7     |
| 13 | fi_cutlass | fi     | fi_cudnn   | on  | —       | **DONE** ⚠️ | 23.2     | 69.4     | 104.2    | 147.8     |
| 14 | fi_trtllm  | fi     | fi_cutlass | on  | —       | **CRASH**   | —        | —        | —        | —         |
| 15 | fi_trtllm  | fi     | fi_cutlass | pw  | —       | **CRASH**   | —        | —        | —        | —         |
| 16 | fi_trtllm  | triton | fi_cutlass | on  | —       | **CRASH**   | —        | —        | —        | —         |
| 17 | fi_cutlass | triton | fi_cutlass | on  | s1/d2   | **DONE**    | 31.4     | 82.2     | 109.7    | 144.5     |
| 18 | fi_cutlass | triton | fi_cutlass | on  | s3/d4   | ⏳ running ★ | —        | —        | —        | —         |
| 19 | fi_cutlass | triton | fi_cutlass | on  | s5/d5   | pending     | —        | —        | —        | —         |
| 20 | fi_cutlass | triton | fi_cutlass | on  | s5/d7   | pending     | —        | —        | —        | —         |
| 21 | triton     | triton | fi_cutlass | on  | s3/d4   | pending     | —        | —        | —        | —         |

- **cg**: `on` = full CUDA graphs; `off` = eager; `pw` = piecewise. **mtp**: NEXTN `steps/draft`, `—` = off.
- **† Case 07** = the designated no-spec serving reference (fi_cutlass-MoE + fi-attn + full-CG).
- **‡ Probes** — **13** `fi_cudnn` FP4 was *expected* to crash ("no cuDNN wheel on the base image") but **ran fine** (⚠️ — the matrix premise is wrong; see observations); **14–16** `flashinfer_trtllm` MoE (crashed on the cudnn twin — likely crashes here too).
- **★ Case 18** = COOKBOOK config, the one to compare against the cudnn winner (Test 29: 39.1/89.7/120.7/163.6) and the pinned profile config.

---

## Observations so far

- **Case 01 (triton-MoE baseline) ≈ identical between base and cudnn images.** Base 0.5.12: 21.0 / 64.3 / 98.4 / 136.2 (n1/4/8/16) vs cudnn case 01: 21.4 / 65.9 / 98.0 / 135.9 — within noise. So the cuDNN build does **not** change the triton-MoE baseline; any advantage must come from the `fi_cudnn` FP4 path (absent here) or elsewhere. The decisive comparison is case **18** (fi_cutlass-MoE + MTP s3/d4) vs cudnn Test 29 — still pending.
- **CUDA graphs ON is worth ~+47% at n=1** (case 01 on: 21.0 vs case 02 off: 14.3); the gap closes by n=4 (64.3 vs 62.6) and vanishes at n=16 (~136 both). Same pattern as the cudnn run.
- **Piecewise graphs (03) ≈ best of the triton-MoE trio so far** — marginally ahead at n=8/n=16 (100.3 / 138.8 vs 98.4 / 136.2 full-CG), n=1 within noise. Tracks cudnn cases 1–3.
- **Block A triton-MoE (01–06) complete — mirrors the cudnn image 1:1.** Best triton-MoE config is piecewise (03: 138.8 n=16), same as cudnn. triton-vs-fi attn = wash; CG-on > no-CG at low concurrency only. As expected, since both images share the fi_cutlass FP4 path. The fi_cutlass-MoE (07–12) and MTP (17–21) cases are where a base-vs-cudnn delta could still appear.
- **fi_cutlass-MoE works on the base image too and beats triton-MoE** (case 07: 147.8 n=16 vs triton best 138.8, +6.5%) — comparable to the cudnn twin (its case 13: 144.3 n=16). So the cudnn FP4 wheel is not required for fi_cutlass-MoE; it only adds the (separate) fi_cudnn FP4 GEMM option. MTP cases (17–18) still the decisive comparison.
- **Block A complete (01–12, no-spec).** Ranking holds: all six fi_cutlass-MoE configs (07–12: 143–148 n=16) beat all six triton-MoE configs (01–06: 135–139 n=16). Best no-spec = **case 07** (fi-attn/full-CG, 147.8). attn and CG-variant are second-order within each MoE family. Whole block tracks the cudnn twin within ±3% (a wash, see the comparison handed over earlier). The MTP block (17–21) is what's left to decide a base-vs-cudnn winner.
- **⚠️ The fi_cudnn FP4 GEMM probe (13) did NOT crash on the base image** — contrary to the matrix premise ("scitrera/dgx-spark-sglang:0.5.12 ships no cuDNN-FP4 wheel"). It ran clean (23.2 / 69.4 / 104.2 / 147.8) and **ties case 07 for best no-spec** (highest n=1 of the whole run). Implications: (a) the matrix's "fp4 axis collapses to fi_cutlass only" assumption is false — the base image *does* have cuDNN-FP4; (b) this removes the only claimed cudnn-exclusive advantage, since fi_cudnn matches the cudnn twin's fi_cudnn cases (~148 n=16) and is available here too. The 21-vs-32-case split (vs the cudnn matrix) was therefore unnecessary. **Base vs cudnn is now looking like a full wash; the MTP headline (17–21 vs cudnn Test 29) is the last open question.**

## Refresh

Re-read the summary JSON and update the table:
`kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-397b-a17b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-397b-nvfp4_4n_1pp_4tp_ep1.json`
