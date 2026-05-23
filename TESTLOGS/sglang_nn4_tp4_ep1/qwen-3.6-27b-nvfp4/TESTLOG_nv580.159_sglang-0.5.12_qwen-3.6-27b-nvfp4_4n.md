# SGLang Test Log — Qwen3.6 27B-NVFP4 (dense, mmangkad / modelopt), 4 Nodes, TP=4 EP=1, v0.5.12

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell-Consumer), 128 GB per node                     |
| Driver    | 580.159                                                                     |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                                          |
| Kernel    | 6.17.0-1018-nvidia                                                          |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.35.3+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `scitrera/dgx-spark-sglang:0.5.12` (Block A/C–F) / `xomoxcc/dgx-spark-sglang:0.5.12-cudnn` (Block B per-case override) |
| Model     | `mmangkad/Qwen3.6-27B-NVFP4` (**modelopt_fp4** quantization)                |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)                                       |
| Transport | **RoCE** via SR-IOV VF                                                      |
| AllReduce | Legacy (both `SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_..._V2=0`)         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-nvfp4/nv580.159_sglang-0.5.12_qwen-3.6-27b-nvfp4_n4_ep1.yaml`

Sister testlogs:
- FP8 sibling: `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-27b-fp8_4n.md` (winner = Case 10, **267.68 tok/s** @ n=8: fi-attn + CG on + piecewise off + MTP NEXTN s=3 / drafts=4 / topk=1).
- 35B-A3B-NVFP4 (MoE, RedHatAI variant): `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4/TESTLOG_nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4_4n.md`.
- 35B-A3B-NVFP4 (MoE, mmangkad/modelopt variant): `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4-mmangkad/TESTLOG_nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4-mmangkad_4n.md`.

## Why this matrix exists — first NVFP4 validation on a dense Qwen3.6

27B is **dense** (hybrid Gated DeltaNet + Gated Attention), so there is **no MoE-runner sweep** and the matrix is correspondingly slim. The two new axes vs the FP8 sibling are:

- `fp4_gemm_backend ∈ {flashinfer_cutlass, flashinfer_cudnn}` — primary FP4 dispatch question. cuDNN-FP4 needs the `:0.5.12-cudnn` image (per-case override on Block B); the upstream scitrera image is cuDNN-less.
- Quantization is `modelopt_fp4`. `mmangkad/Qwen3.6-27B-NVFP4` was selected as the primary candidate — no upstream `nvidia/Qwen3.6-27B-NVFP4` exists, and the mmangkad model card explicitly documents the SGLang serve command and `--quantization modelopt_fp4`. Other community variants (unsloth, sakamakismile, vrfai) lean on vLLM as primary serving path.

On-device weights ≈ 7–8 GB (vs ≈ 13 GB FP8) → roughly 2× the KV-cache headroom. The headline question: does the FP4 tensor-core path on GB10/SM121 match or beat the FP8 sibling's 267.68 tok/s @ n=8, and does the extra KV-cache headroom shift the MTP sweet spot?

Hybrid Gated-DeltaNet arch → all MTP cases (Blocks C–F) require:

```yaml
mamba_scheduler_strategy: extra_buffer
enable_spec_v2: true
```

(same constraint as the FP8 sibling and the 35B-A3B hybrid testlogs).

## Image policy

Default image is `scitrera/dgx-spark-sglang:0.5.12`. Block B (Tests 07–12, all `fp4_gemm_backend: flashinfer_cudnn`) overrides per-case to `xomoxcc/dgx-spark-sglang:0.5.12-cudnn` — without this, the cuDNN-FP4 path crashes at startup with `RuntimeError: cuDNN is not available` (confirmed empirically on the RedHatAI 35B-A3B-NVFP4 matrix Block A Tests 07–12, 12/12 startup_crash).

## Configuration Matrix (22 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, disable_deep_gemm=true, context_length=262144`. Dense → no MoE-runner sweep. Quantization: `modelopt_fp4`. All speculative cases (Tests 13–22) use NEXTN with `mamba_scheduler_strategy=extra_buffer + enable_spec_v2=true`.

### Block A — `fi_cutlass-fp4` GEMM backend × {fi-attn, triton-attn} × CG variants (Tests 01–06)

Most-validated FP4 GEMM path on SM121 (`flashinfer_cutlass`).

| #  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------------|----------------|---------------|--------|----------:|---------:|---------:|
| 01 | fi        | fi_cutlass   | false          | true          | ok     |     23.88 |    93.68 |   174.24 |
| 02 | fi        | fi_cutlass   | true           | true          | ok⁺    |     20.12 |    81.36 |   159.03 |
| 03 | fi        | fi_cutlass   | false          | false         | ok     |     23.75 |    94.76 |   176.80 |
| 04 | triton    | fi_cutlass   | false          | true          | ok     |     24.33 |    94.44 |   176.40 |
| 05 | triton    | fi_cutlass   | true           | true          | ok     |     17.95 |    79.72 |   156.32 |
| 06 | triton    | fi_cutlass   | false          | false         | ok     |     23.90 |    91.95 |   172.72 |

### Block B — `fi_cudnn-fp4` GEMM backend × {fi-attn, triton-attn} × CG variants (Tests 07–12)

All cases override the image to `xomoxcc/dgx-spark-sglang:0.5.12-cudnn`.

| #  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------------|----------------|---------------|--------|----------:|---------:|---------:|
| 07 | fi        | fi_cudnn     | false          | true          | ok     |     23.46 |    89.32 |   167.68 |
| 08 | fi        | fi_cudnn     | true           | true          | ok     |     17.61 |    72.21 |   144.95 |
| 09 | fi        | fi_cudnn     | false          | false         | ok     |     23.53 |    91.81 |   170.08 |
| 10 | triton    | fi_cudnn     | false          | true          | ok     |     23.92 |    91.28 |   170.64 |
| 11 | triton    | fi_cudnn     | true           | true          | ok     |     16.41 |    71.92 |   140.93 |
| 12 | triton    | fi_cudnn     | false          | false         | ok     |     21.86 |    90.04 |   168.72 |

### Block C — MTP (NEXTN) anchors on FP8-winner shape (Tests 13–14)

FP8-27B 0.5.11 winner = Case 10 = fi-attn + CG on + piecewise off + MTP NEXTN s=3 / drafts=4 / topk=1 → 267.68 tok/s @ n=8. Anchors mirror that shape (with `fi_cutlass-fp4` GEMM) for direct A/B against the FP8 sibling.

| #  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | spec                     | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------------|----------------|---------------|--------------------------|--------|----------:|---------:|---------:|
| 13 | fi        | fi_cutlass   | false          | true          | NEXTN s=3 / d=4 / topk=1 | ok     |     44.50 |   142.67 |   229.29 |
| 14 | triton    | fi_cutlass   | false          | true          | NEXTN s=3 / d=4 / topk=1 | ok     |     41.52 |   141.25 |   234.89 |

### Block D — `speculative_num_steps` sweep on FP8-winner shape (Tests 15–18)

fi-attn + CG on + piecewise off + `fi_cutlass-fp4` GEMM × `speculative_num_steps ∈ {2, 3, 4, 5}`. `num_draft_tokens` scales with `num_steps` (`num_steps + 1`). FP8 sibling found s=3 best (s=5 collapsed −10 %); the question is whether NVFP4 shifts the sweet spot.

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|----------:|---------:|---------:|
| 15 | 2         | 3      | 1    | ok     |     40.99 |   149.97 |   244.90 |
| 16 | 3         | 4      | 1    | ok⁺    |     42.21 |   146.77 |   242.28 |
| 17 | 4         | 5      | 1    | ok     |     40.61 |   150.96 |   243.80 |
| 18 | 5         | 6      | 1    | ok⁺    |     47.59 |   144.14 |   235.96 |

Test 16 is the direct A/B against FP8 Case 10 (267.68 tok/s).

### Block E — `speculative_num_draft_tokens` sweep on winner shape (Tests 19–20)

Same shape as Test 16 (num_steps=3, topk=1), drafts ∈ {6, 8}. FP8 sibling found drafts=4 best (monotonically worse at 6/8); NVFP4 has ~2× the KV-cache headroom, so the trade-off may flip.

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|----------:|---------:|---------:|
| 19 | 3         | 6      | 1    | ok     |     40.71 |   150.72 |   250.84 |
| 20 | 3         | 8      | 1    | ok⁺    |     44.08 |   149.68 |   251.40 |

### Block F — `speculative_eagle_topk` sweep on winner shape (Tests 21–22)

Same shape as Test 16 (num_steps=3, drafts=4), topk ∈ {2, 4}. FP8 sibling found topk=1 best (topk=2/4 monotonically worse @ n=8); re-tested in case FP4 verification cost changes the per-candidate trade-off.

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|----------:|---------:|---------:|
| 21 | 3         | 4      | 2    | **ok 🏆** | 44.10 |   148.06 | **256.23** |
| 22 | 3         | 4      | 4    | **crash B** | 45.52 |    —   |        — |

### Column Legend

| Column         | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                      |
| fp4_gemm       | `fp4_gemm_backend` — `fi_cutlass` = `flashinfer_cutlass`, `fi_cudnn` = `flashinfer_cudnn`       |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = fixed-BS graphs only, false = piecewise variable-length |
| num_steps      | `speculative_num_steps` — NEXTN draft depth                                                     |
| drafts         | `speculative_num_draft_tokens` — verified per step                                              |
| topk           | `speculative_eagle_topk` — candidates per step (1 = pure NEXTN)                                 |
| spec           | speculative decoding shorthand — `NEXTN s=N / d=K / topk=T`                                     |

---

## Results

**Matrix run complete (2026-05-23).** 22/22 cases attempted. **21 ok, 1 bench_crash** (Test 22 — spec-v2 topk=4 deadlock). Output quality clean across **all** 21 ok cases — 0 word-salad pattern hits across the entire matrix's 168 n=8 requests.

### Crash legend

- **crash B** (spec-v2 topk=4 hang): Test 22 — all 8/8 n=8 requests `status: error` after 600 s wall time, 0 output_tokens, no Python exception in the head log. Server started cleanly (CG capture finished, model loaded). Likely a Spec-V2 deadlock with `eagle_topk=4 × num_steps=3 × num_draft_tokens=4` — the draft-tree branching factor (topk × steps = 12 candidates) is much larger than `num_draft_tokens=4` can verify, putting the decode loop into a state the breakable-CG path doesn't recover from. Test 21 (topk=2) ran cleanly. Workaround: pair higher topk with proportionally higher `num_draft_tokens` (`topk × steps + 1` is the safe lower bound).
- **ok⁺**: case passed but had ≥ 1 atypical finish reason (`length×7, stop×1` instead of pure `length×8`). All ok⁺ cases verified clean by pattern-grep + TTR.

### Headline numbers

| Block | Best case | n=8 peak | Notes                                                                       |
|-------|-----------|---------:|-----------------------------------------------------------------------------|
| A (fi_cutlass-fp4, no-MTP) | Test 03 (fi + piecewise) | 176.80 | Best no-MTP baseline                                          |
| B (fi_cudnn-fp4, no-MTP)   | Test 10 (triton + CG)    | 170.64 | **−3.5 % vs Block A** — cuDNN-FP4 consistently slower (confirms 35B finding) |
| C (MTP anchors)            | Test 14 (triton-attn)    | 234.89 | +32.9 % over no-MTP best — MTP delivers                                     |
| D (num_steps sweep)        | Test 15 (s=2)            | 244.90 | s=2 ≈ s=3 ≈ s=4 (≤ 1 % spread); s=5 drops 3.6 %                            |
| E (drafts sweep)           | Test 20 (drafts=8)       | 251.40 | drafts=6/8 essentially tied, +3.7 % over drafts=4 baseline (Test 16)        |
| F (topk sweep)             | **Test 21 (topk=2)**     | **256.23 🏆** | **+5.7 % over the FP8-shape MTP baseline** — new local winner shape       |

### Per-block findings

1. **No-MTP comparison FP8 vs NVFP4 27B** — FP8 sibling 0.5.11 no-MTP cases topped at ~200 tok/s n=8; NVFP4 no-MTP best is 176.80 (Test 03). **NVFP4 is ~12 % slower than FP8 in the no-MTP regime** on this dense 27B model. The 2× smaller weights don't translate to throughput because GB10 dense MMA isn't memory-bandwidth-bound at this concurrency.
2. **cuDNN-FP4 GEMM is consistently slower than fi_cutlass-FP4** — Block B best 170.64 vs Block A best 176.80 → −3.5 %. Same shape as the 35B-A3B-NVFP4 Test 55 finding (where cuDNN-FP4 was −10 %). The cuDNN-FP4 path works (image rebuild did its job), but it's never the fastest backend on GB10. **The pending hypothesis "could fi_cudnn win here?" → resolved: no.**
3. **MTP delivers massively on this dense model**: Block C anchors at 229.29–234.89 tok/s → already +30–33 % over no-MTP best (176.80). MTP impact is much larger here than on the 35B-A3B MoE (where MTP was +7.6 % over the no-MTP best).
4. **`num_steps` is essentially flat at 2/3/4** — Tests 15/16/17 land at 244.90 / 242.28 / 243.80 (≤ 1 % spread). s=5 drops to 235.96 (−3.6 %). Markedly different shape vs the 35B-A3B-NVFP4 sweep where s=2 was a clear winner (438 vs 387 at s=3). On this dense model the choice between s=2/3/4 is essentially a tie — **s=2 is the lightest-overhead winner**.
5. **`num_draft_tokens=6/8 > 4`** — Tests 19/20 at 250.84/251.40 vs Test 16 at 242.28 → **+3.5–3.8 % gain from wider draft acceptance**. **Different from the FP8 sibling** where drafts=4 was best. Hypothesis: NVFP4's smaller verification cost per draft candidate makes wider trees economical. The "KV-cache headroom changes the trade-off" pre-run hypothesis is confirmed.
6. **`eagle_topk=2 > topk=1`** — Test 21 hits 256.23 vs Test 16 baseline 242.28 → **+5.7 % gain**. Also **different from FP8 27B** (where topk=1 was best). The pre-run hypothesis ("expect topk=1 to remain best") is refuted: combined with finding #5, this suggests the dense FP4 path absorbs wider draft trees that the FP8 path couldn't. NVFP4 verification cost per candidate is genuinely cheaper.
7. **`eagle_topk=4` deadlocks** — Test 22: server starts cleanly, accepts no requests successfully in 600 s. Probably a Spec-V2-tree-size invariant violation (topk × steps = 12 candidates with `num_draft_tokens=4` cap). Need a re-run with `num_draft_tokens=13` to disambiguate hardware-hang from invariant-violation.
8. **Best n=8 peak: Test 21 at 256.23 tok/s.** Compared to FP8-27B 0.5.11 Case 10 winner (267.68) → **−4.3 %**. **NVFP4 does NOT beat FP8 on this dense 27B model**, unlike the 35B-A3B MoE where NVFP4 + MTP s=2 beat FP8 + MTP s=2 by +2.6 %. **The NVFP4-vs-FP8 advantage on GB10 is MoE-specific** — FP4 tensor-core MMA benefit scales with expert-routing parallelism that dense models don't have.
9. **Output quality clean** across all 21 ok cases. TTR_min 0.549–0.733, all `length×8` or `length×7, stop×1` finish patterns, zero word-salad pattern matches.

### Production recommendation

For the 27B-NVFP4 dense model, the active profile should use the **Test 21 shape**:

```yaml
attention_backend: flashinfer
fp4_gemm_backend: flashinfer_cutlass
disable_cuda_graph: false
disable_piecewise_cuda_graph: true   # piecewise off — winner shape mirrors FP8 sibling
speculative_enabled: true
speculative_algo: NEXTN
speculative_num_steps: 3
speculative_eagle_topk: 2
speculative_num_draft_tokens: 4
mamba_scheduler_strategy: extra_buffer
enable_spec_v2: true
```

**Caveat:** the FP8 sibling is still faster (267.68 vs 256.23) — for raw 27B throughput on this cluster the **FP8 path remains the better default**. NVFP4 27B is worth using **when KV-cache headroom matters more than peak throughput** (long-context workloads) — the smaller weight footprint frees up 5–6 GB per node for the KV cache.

### Comparison to sibling matrices on this cluster

| Model                              | Best config                        | n=8 peak |
|------------------------------------|------------------------------------|---------:|
| 35B-A3B-FP8                        | Test 21 (winner + MTP s=2)         |   426.76 |
| 35B-A3B-NVFP4 (RedHatAI)           | Test 45 (winner + MTP s=2) 🏆      |   438.07 |
| 35B-A3B-NVFP4 (mmangkad/modelopt)  | Test 04 (fi_cutlass-MoE no-MTP)    |   406.68 |
| **27B-FP8 (0.5.11)**               | Case 10 (winner + MTP s=3)         |   267.68 |
| **27B-NVFP4 (this matrix)**        | Test 21 (winner + MTP s=3, topk=2) |   256.23 |

**Pattern: NVFP4 wins on MoE (35B-A3B: +2.6 % vs FP8), loses on dense (27B: −4.3 % vs FP8).** FP4 tensor-core MMA benefit scales with expert-routing parallelism — dense models don't have that channel.

---

## Action items after the matrix run

- [x] Fill the six block tables with actual results
- [x] Verify output quality on every `ok` case (pattern-grep across all 168 n=8 requests, 0 word-salad hits)
- [x] Compute Δ vs FP8 sibling — FP8 still wins (267.68 vs 256.23, −4.3 %)
- [ ] Decide on `mmangkad-qwen3.6-27b-nvfp4.yml` model profile: Test 21 shape if NVFP4 27B is to be deployed for long-context use, else skip (FP8 sibling remains active production)
- [x] Document whether `fi_cudnn-fp4` ever wins → no, −3.5 % vs fi_cutlass-FP4. Consistent with 35B-A3B Test 55 (−10 %)
- [ ] Re-run Test 22 (topk=4) with `num_draft_tokens=13` (= topk × num_steps + 1) to confirm whether the deadlock is purely a tree-size invariant violation or a deeper Spec-V2 bug
- [ ] Save the "NVFP4 wins MoE, loses dense" pattern to project memory — relevant for future quantization choices on this cluster
