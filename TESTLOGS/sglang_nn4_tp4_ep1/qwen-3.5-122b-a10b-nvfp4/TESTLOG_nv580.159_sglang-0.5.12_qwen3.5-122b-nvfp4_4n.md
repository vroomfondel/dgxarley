# SGLang Test Log — Qwen3.5 122B-A10B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.12 (base image)

> 🔄 **RUN IN PROGRESS** (started 2026-06-20 12:01) — **14 / 21 cases done**. Block A (01–12) all clean. **Probe results:** case **13** (fi_cudnn FP4) ran CLEAN (16/16) — no crash, just like the 397B run, contradicting the "base image has no cuDNN-FP4 wheel → crash" expectation. Case **14** (fi_trtllm MoE) **CRASHED at boot** as predicted — trtllm FP4 block-scale-MoE GEMM kernel fails on GB10. Case **15** (trtllm piecewise) booting/restarting at 15:28 (likely same crash). **No-spec leader: case 09 — n=16 269.0.** Then 16 (trtllm) + Block C (17–21, MTP). Peak = sum-of-per-request tok/s (not the summary JSON's `aggregate_throughput`, which is total/wall). Updating this log as cases complete.

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
| 01 | triton     | fi     | fi_cutlass | on  | —       | ✅ OK (16/16) | 34.3 | 112.6 | 176.0 | 259.8 |
| 02 | triton     | fi     | fi_cutlass | off | —       | ✅ OK (16/16) | 15.8 | 96.1 | 161.2 | 248.8 |
| 03 | triton     | fi     | fi_cutlass | pw  | —       | ✅ OK (16/16) | 30.7 | 110.1 | 169.8 | 254.6 |
| 04 | triton     | triton | fi_cutlass | on  | —       | ✅ OK (16/16) | 31.7 | 113.4 | 175.3 | 259.3 |
| 05 | triton     | triton | fi_cutlass | off | —       | ✅ OK (16/16) | 15.7 | 95.5 | 167.2 | 251.9 |
| 06 | triton     | triton | fi_cutlass | pw  | —       | ✅ OK (16/16) | 31.6 | 111.2 | 176.3 | 259.9 |
| 07 | fi_cutlass | fi     | fi_cutlass | on  | —       | ✅ OK (16/16) † | 33.8 | 117.7 | 183.7 | 266.5 |
| 08 | fi_cutlass | fi     | fi_cutlass | off | —       | ✅ OK (16/16) | 26.8 | 108.3 | 172.4 | 258.1 |
| 09 | fi_cutlass | fi     | fi_cutlass | pw  | —       | ✅ OK (16/16) | 34.6 | 117.9 | 180.9 | 269.0 |
| 10 | fi_cutlass | triton | fi_cutlass | on  | —       | ✅ OK (16/16) | 33.5 | 117.3 | 177.5 | 264.0 |
| 11 | fi_cutlass | triton | fi_cutlass | off | —       | ✅ OK (16/16) | 26.5 | 107.8 | 173.2 | 257.1 |
| 12 | fi_cutlass | triton | fi_cutlass | pw  | —       | ✅ OK (16/16) | 33.3 | 115.9 | 182.1 | 264.1 |
| 13 | fi_cutlass | fi     | fi_cudnn   | on  | —       | ✅ OK (16/16) ⚠️ | 34.7 | 115.4 | 179.2 | 262.6 |
| 14 | fi_trtllm  | fi     | fi_cutlass | on  | —       | 💥 CRASH (boot) ‡ | —   | —   | —   | —    |
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

- **Case 01 (triton-MoE / fi-attn / full-CG, no-spec):** clean boot, **16/16 ok** at every concurrency, all `finish_reason` ∈ {length, stop}. Peak tok/s **34.3 / 112.6 / 176.0 / 259.8** (n=1/4/8/16). Per-request tok/s degrades with load (34.3 → 16.2 at n=16) as expected; aggregate scales ~linearly to n=8 then sublinear. Boot ~15 min wall (weight load 244 s + Mamba/KV alloc + CUDA-graph capture of bs [1,2,4,8,12,16,24,32]). First SEED gates confirmed: NVFP4 loads as `Qwen3_5MoeForConditionalGeneration`, no OOM at `mem_fraction_static=0.75`, `mamba_scheduler_strategy=extra_buffer` boots clean, EP=1.
  - FYI: summary JSON's `aggregate_throughput` (n16=255.6) = total_tokens/wall_time; the table uses peak sum-of-per-request (n16=259.8). Close here because all requests run the full window.
- **Case 02 (same as 01 but eager / `disable_cuda_graph`):** clean boot, **16/16 ok**, faster startup (no CUDA-graph capture). Peak **15.8 / 96.1 / 161.2 / 248.8**. Confirms the hypothesis: **CUDA graphs are worth the most at n=1** (34.3 → 15.8, **−54%** going eager) and the gap closes under load (n=16: 259.8 → 248.8, only **−4%**). Note eager runs clean here because this is a `triton`-MoE path; eager is only broken on `cutlass_moe_fp4` runners (per TURBOQUANT) — watch cases 11 (fi_cutlass-MoE + eager).
- **Case 03 (same as 01 but piecewise-CG):** clean boot, **16/16 ok**. Peak **30.7 / 110.1 / 169.8 / 254.6** — sits **between** full-CG and eager but much closer to full-CG: n=1 only **−10%** vs full-CG (vs eager's −54%), n=16 within **−2%** (254.6 vs 259.8). So for triton-MoE, **full-CG (01) ≥ piecewise (03) > eager (02)** at every concurrency; piecewise recovers almost all of the CUDA-graph benefit. CG-mode ranking on triton-MoE+fi-attn settled.
- **Case 04 (triton-MoE, triton-attn, full-CG):** clean boot, **16/16 ok**. Peak **31.7 / 113.4 / 175.3 / 259.3** — **statistically identical to case 01 (fi-attn)** at concurrency (n=16: 259.3 vs 259.8; n=8/4 within noise). Only difference is n=1, where fi-attn edges triton-attn (34.3 vs 31.7, +8%). **Attention backend (fi vs triton) is near-irrelevant for throughput here**; fi-attn marginally better at low concurrency.
- **Case 05 (triton-MoE, triton-attn, eager):** clean boot, **16/16 ok**. Peak **15.7 / 95.5 / 167.2 / 251.9** — mirrors case 02 (fi-attn eager: 15.8 / 96.1 / 161.2 / 248.8) to within noise. Reconfirms attn-backend irrelevance, now also in the eager path. **triton-MoE half of Block A (01–06) shows a tight, well-behaved CG>piecewise>eager pattern independent of attn backend.**
- **Case 06 (triton-MoE, triton-attn, piecewise):** clean boot, **16/16 ok**. Peak **31.6 / 111.2 / 176.3 / 259.9** — mirrors case 03 (fi-attn piecewise). **Block A triton-MoE half (01–06) complete.** Summary across the 6: at n=16 all four full-CG/piecewise cases cluster at **254–260** (CG-mode and attn-backend differences wash out under load); the two eager cases trail at **249–252**. n=1 is where mode matters: full-CG/piecewise **31–34**, eager **~16** (half). No crashes, no failed requests anywhere in Block A's triton-MoE half.
- **Case 07 (fi_cutlass-MoE, fi-attn, full-CG — the no-spec serving reference):** clean boot, **16/16 ok**. Peak **33.8 / 117.7 / 183.7 / 266.5**. **Confirms the headline hypothesis: fi_cutlass-MoE beats triton-MoE at concurrency** — vs case 01 (triton-MoE, same attn+CG): n=4 **+4.5%** (117.7 vs 112.6), n=8 **+4.4%** (183.7 vs 176.0), n=16 **+2.6%** (266.5 vs 259.8); n=1 a tie (33.8 vs 34.3). Matches the 397B run's +6.5% n=16 direction (a bit smaller here). **Current overall leader.** Block A's best no-spec config is fi_cutlass-MoE + full-CG.
- **Case 08 (fi_cutlass-MoE, fi-attn, eager):** **booted and ran clean, 16/16 ok** — no crash. Peak **26.8 / 108.3 / 172.4 / 258.1**. Two findings: (1) **the "eager broken on `cutlass_moe_fp4`" caveat does NOT apply to the `flashinfer_cutlass` MoE runner** (different code path — the broken one is the now-removed standalone `cutlass` runner). (2) **fi_cutlass-MoE degrades far more gracefully under eager than triton-MoE**: n=1 only **−21%** vs its full-CG (26.8 vs 33.8), where triton-MoE eager lost **−54%** (case 02). At n≥8 fi_cutlass eager (172/258) still edges triton full-CG (176/260 ≈ tie). So on the fi_cutlass runner, the CUDA-graph penalty for eager is much smaller.
- **Case 09 (fi_cutlass-MoE, fi-attn, piecewise):** clean boot, **16/16 ok**. Peak **34.6 / 117.9 / 180.9 / 269.0** — **new overall no-spec leader**, edging case 07 (full-CG) at n=1 (34.6 vs 33.8) and n=16 (269.0 vs 266.5); n=8 a hair behind (180.9 vs 183.7). Within-noise tie with full-CG, but **piecewise is at least as good as full-CG on the fi_cutlass runner** — unlike the triton-MoE half where full-CG was strictly best. (n=1 single-request `finish_reason=stop`, so slightly shorter generation; rate still comparable.)
- **Case 10 (fi_cutlass-MoE, triton-attn, full-CG):** clean boot, **16/16 ok**. Peak **33.5 / 117.3 / 177.5 / 264.0** — vs case 07 (fi-attn, same MoE+CG): n=1/4 tied, n=8 −3.4% (177.5 vs 183.7), n=16 −0.9% (264.0 vs 266.5). Reconfirms **fi-attn ≥ triton-attn marginally**, now on the fi_cutlass runner too. attn-backend remains a near-zero throughput lever.
- **Case 11 (fi_cutlass-MoE, triton-attn, eager):** clean boot, **16/16 ok** — second confirmation that **cutlass-eager does not crash** here (independent of attn backend). Peak **26.5 / 107.8 / 173.2 / 257.1** — mirrors case 08 (fi-attn eager) to within noise. **Block A no-spec sweep (01–12, bar 12 still running) is uniformly crash-free; fi_cutlass-MoE wins, attn-backend ≈ irrelevant, eager penalty small on fi_cutlass / large on triton.**
- **Case 12 (fi_cutlass-MoE, triton-attn, piecewise):** clean boot, **16/16 ok**. Peak **33.3 / 115.9 / 182.1 / 264.1** — vs case 09 (fi-attn piecewise): triton-attn again a touch lower at n=1/4/16, dead-even at n=8. **Block A (01–12) DONE, zero crashes, zero failed requests.** No-spec ranking: **fi_cutlass-MoE + fi-attn + piecewise (09) ≈ full-CG (07) > triton-MoE equivalents > all eager.** Best no-spec n=16 = **269.0** (case 09).

### Probes (13–16)
- **Case 13 (fp4_gemm = `fi_cudnn`, EXPECT-CRASH-PROBE):** **ran CLEAN, 16/16 ok.** Peak **34.7 / 115.4 / 179.2 / 262.6** — statistically identical to the `fi_cutlass` FP4-GEMM cases (e.g. 07/09), so cuDNN-FP4 is either present in the base image or silently no-ops to the cutlass path. **Expectation refuted** (matrix note predicted "cuDNN is not available" crash); same surprise as the 397B base-image run. The ⚠️ probe is now a recorded *non*-crash.
- **Case 14 (moe_runner = `flashinfer_trtllm`, PROBE):** **CRASHED at boot** (`outcome=startup_crash`, head pod +1 restart). Root cause — the trtllm NVFP4 block-scale MoE GEMM kernel fails to execute on GB10:
  ```
  RuntimeError: Error in function 'run' at trtllm_batched_gemm_runner.cu:278: Error occurred when running GEMM!
  (numBatches: 256, GemmMNK: 1 512 3072, Kernel: bmm_E2m1_..._sm100f)
  ```
  in `flashinfer/fused_moe/core.py → trtllm_fp4_block_scale_moe`. The kernel is compiled for `sm100` (datacenter Blackwell); GB10 is `sm121`, and the batched-GEMM runner errors out at autotune. **`flashinfer_trtllm` MoE is unusable on this card** — confirms the 397B run's behaviour. (Distinct from the gated-MoE *padding* assert that trtllm normally avoids; here it's a kernel-exec failure.) Cases 15–16 (same runner) expected to crash identically.

## Refresh

After the run, re-read the summary JSON and fill the table:
`kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-122b-a10b-nvfp4/0.5.12/MATRIX_SUMMARY_nv580.159_sglang-0.5.12_qwen3.5-122b-nvfp4_4n_1pp_4tp_ep1.json`
