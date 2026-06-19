# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.12 (base image)

> ✅ **RUN COMPLETE** — 21 / 21 cases done as of 2026-06-19 ~16:44 (14–16 trtllm probes crashed as expected). 🎯 Winner: cookbook MTP s3/d4 (**case 18**) = **40.1 / 95.1 / 125.4 / 172.6**, which **BEATS** cudnn Test 29 (39.1/89.7/120.7/163.6) by +3–6% across the board. MTP depth sweep peaks at s3/d4 then regresses (s5/d5=159.8, s5/d7=153.7 — classic draft overshoot). Cross-runner case 21 (triton-MoE + same MTP) tops everything at n=1 (42.5) but trails 18 at concurrency (161.8 vs 172.6 n=16) — fi_cutlass-MoE still wins the throughput end.

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
| 18 | fi_cutlass | triton | fi_cutlass | on  | s3/d4   | **DONE ★**  | **40.1** | **95.1** | **125.4**| **172.6** |
| 19 | fi_cutlass | triton | fi_cutlass | on  | s5/d5   | **DONE**    | 37.6     | 84.2     | 111.7    | 159.8     |
| 20 | fi_cutlass | triton | fi_cutlass | on  | s5/d7   | **DONE**    | 34.1     | 83.8     | 110.8    | 153.7     |
| 21 | triton     | triton | fi_cutlass | on  | s3/d4   | **DONE**    | 42.5     | 86.0     | 115.6    | 161.8     |

- **cg**: `on` = full CUDA graphs; `off` = eager; `pw` = piecewise. **mtp**: NEXTN `steps/draft`, `—` = off.
- **† Case 07** = the designated no-spec serving reference (fi_cutlass-MoE + fi-attn + full-CG).
- **‡ Probes** — **13** `fi_cudnn` FP4 was *expected* to crash ("no cuDNN wheel on the base image") but **ran fine** (⚠️ — the matrix premise is wrong; see observations); **14–16** `flashinfer_trtllm` MoE (crashed on the cudnn twin — likely crashes here too).
- **★ Case 18** = COOKBOOK config, the one to compare against the cudnn winner (Test 29: 39.1/89.7/120.7/163.6) and the pinned profile config.

---

## Observations so far

- **🎯 HEADLINE — the base image WINS the decisive MTP cookbook comparison.** Case 18 (fi_cutlass-MoE + triton-attn + fi_cutlass-FP4 + full-CG + MTP s3/d4) on the *base* `0.5.12` posts **40.1 / 95.1 / 125.4 / 172.6** (n1/4/8/16) vs the cudnn twin's Test 29 **39.1 / 89.7 / 120.7 / 163.6** — base is **+2.6% / +6.0% / +3.9% / +5.5%** ahead, identical config. It also tops the old 0.5.10 cutlass-direct (40.0 n=1). So plain 0.5.12 is not just "not slower" — on the winner config it is marginally **faster** than cudnn, and the n=1=31.7 I measured ad-hoc earlier was purely the `cuda_graph_max_bs=8` handicap (matrix uses 16).
  - **Config implication:** the profile currently pins `xomoxcc/dgx-spark-sglang:0.5.12-cudnn` (chosen earlier). This result argues the plain `scitrera/dgx-spark-sglang:0.5.12` default would be ≥ as fast on the pinned config — the pin could be dropped. (Flagged to the user; not changed here.)

- **Case 01 (triton-MoE baseline) ≈ identical between base and cudnn images.** Base 0.5.12: 21.0 / 64.3 / 98.4 / 136.2 (n1/4/8/16) vs cudnn case 01: 21.4 / 65.9 / 98.0 / 135.9 — within noise. So the cuDNN build does **not** change the triton-MoE baseline; any advantage must come from the `fi_cudnn` FP4 path (absent here) or elsewhere. The decisive comparison is case **18** (fi_cutlass-MoE + MTP s3/d4) vs cudnn Test 29 — still pending.
- **CUDA graphs ON is worth ~+47% at n=1** (case 01 on: 21.0 vs case 02 off: 14.3); the gap closes by n=4 (64.3 vs 62.6) and vanishes at n=16 (~136 both). Same pattern as the cudnn run.
- **Piecewise graphs (03) ≈ best of the triton-MoE trio so far** — marginally ahead at n=8/n=16 (100.3 / 138.8 vs 98.4 / 136.2 full-CG), n=1 within noise. Tracks cudnn cases 1–3.
- **Block A triton-MoE (01–06) complete — mirrors the cudnn image 1:1.** Best triton-MoE config is piecewise (03: 138.8 n=16), same as cudnn. triton-vs-fi attn = wash; CG-on > no-CG at low concurrency only. As expected, since both images share the fi_cutlass FP4 path. The fi_cutlass-MoE (07–12) and MTP (17–21) cases are where a base-vs-cudnn delta could still appear.
- **fi_cutlass-MoE works on the base image too and beats triton-MoE** (case 07: 147.8 n=16 vs triton best 138.8, +6.5%) — comparable to the cudnn twin (its case 13: 144.3 n=16). So the cudnn FP4 wheel is not required for fi_cutlass-MoE; it only adds the (separate) fi_cudnn FP4 GEMM option. MTP cases (17–18) still the decisive comparison.
- **Block A complete (01–12, no-spec).** Ranking holds: all six fi_cutlass-MoE configs (07–12: 143–148 n=16) beat all six triton-MoE configs (01–06: 135–139 n=16). Best no-spec = **case 07** (fi-attn/full-CG, 147.8). attn and CG-variant are second-order within each MoE family. Whole block tracks the cudnn twin within ±3% (a wash, see the comparison handed over earlier). The MTP block (17–21) is what's left to decide a base-vs-cudnn winner.
- **⚠️ The fi_cudnn FP4 GEMM probe (13) did NOT crash on the base image** — contrary to the matrix premise ("scitrera/dgx-spark-sglang:0.5.12 ships no cuDNN-FP4 wheel"). It ran clean (23.2 / 69.4 / 104.2 / 147.8) and **ties case 07 for best no-spec** (highest n=1 of the whole run). Implications: (a) the matrix's "fp4 axis collapses to fi_cutlass only" assumption is false — the base image *does* have cuDNN-FP4; (b) this removes the only claimed cudnn-exclusive advantage, since fi_cudnn matches the cudnn twin's fi_cudnn cases (~148 n=16) and is available here too. The 21-vs-32-case split (vs the cudnn matrix) was therefore unnecessary. **Base vs cudnn is a full wash; the only differentiator is the MTP headline, which the base image wins (case 18 > cudnn Test 29).**

- **🧪 MTP depth sweep (17–20) — s3/d4 is the sweet spot, deeper drafts REGRESS.** With fi_cutlass-MoE/triton-attn/full-CG fixed: s1/d2 (17) = 144.5, **s3/d4 (18) = 172.6 ★**, s5/d5 (19) = 159.8, s5/d7 (20) = 153.7 (n=16). Acceptance-rate gains from longer draft chains are outweighed by per-step verification cost past d4 — monotonic falloff after the s3/d4 peak at every concurrency. This confirms the cookbook s3/d4 pick was correct and there's no headroom in going deeper. Same shape on n=1 (40.1 → 37.6 → 34.1).

- **🔀 Cross-runner check (21) — MTP helps triton-MoE too, but fi_cutlass-MoE still wins at concurrency.** Triton-MoE + the same MTP s3/d4 (21) posts 42.5 / 86.0 / 115.6 / 161.8. It actually **edges out case 18 at n=1** (42.5 vs 40.1 — triton-MoE's lower per-token latency dominates when there's no batching), but falls behind from n=4 up (161.8 vs 172.6 at n=16, −6.3%). So the no-spec ranking holds under MTP: triton-MoE for single-stream latency, **fi_cutlass-MoE (case 18) for served throughput** — and 18 remains the overall winner and the config to pin.

## Refresh

Re-read the summary JSON and update the table:
`kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-397b-a17b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-397b-nvfp4_4n_1pp_4tp_ep1.json`
