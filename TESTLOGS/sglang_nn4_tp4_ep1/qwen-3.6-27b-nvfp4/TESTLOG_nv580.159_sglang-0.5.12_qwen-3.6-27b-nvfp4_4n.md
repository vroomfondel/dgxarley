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
| 01 | fi        | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 02 | fi        | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 03 | fi        | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 04 | triton    | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 05 | triton    | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 06 | triton    | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |

### Block B — `fi_cudnn-fp4` GEMM backend × {fi-attn, triton-attn} × CG variants (Tests 07–12)

All cases override the image to `xomoxcc/dgx-spark-sglang:0.5.12-cudnn`.

| #  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------------|----------------|---------------|--------|----------:|---------:|---------:|
| 07 | fi        | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 08 | fi        | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 09 | fi        | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |
| 10 | triton    | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 11 | triton    | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 12 | triton    | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |

### Block C — MTP (NEXTN) anchors on FP8-winner shape (Tests 13–14)

FP8-27B 0.5.11 winner = Case 10 = fi-attn + CG on + piecewise off + MTP NEXTN s=3 / drafts=4 / topk=1 → 267.68 tok/s @ n=8. Anchors mirror that shape (with `fi_cutlass-fp4` GEMM) for direct A/B against the FP8 sibling.

| #  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | spec                     | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------------|----------------|---------------|--------------------------|--------|----------:|---------:|---------:|
| 13 | fi        | fi_cutlass   | false          | true          | NEXTN s=3 / d=4 / topk=1 | TBD    | —         | —        | —        |
| 14 | triton    | fi_cutlass   | false          | true          | NEXTN s=3 / d=4 / topk=1 | TBD    | —         | —        | —        |

### Block D — `speculative_num_steps` sweep on FP8-winner shape (Tests 15–18)

fi-attn + CG on + piecewise off + `fi_cutlass-fp4` GEMM × `speculative_num_steps ∈ {2, 3, 4, 5}`. `num_draft_tokens` scales with `num_steps` (`num_steps + 1`). FP8 sibling found s=3 best (s=5 collapsed −10 %); the question is whether NVFP4 shifts the sweet spot.

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|----------:|---------:|---------:|
| 15 | 2         | 3      | 1    | TBD    | —         | —        | —        |
| 16 | 3         | 4      | 1    | TBD    | —         | —        | —        |
| 17 | 4         | 5      | 1    | TBD    | —         | —        | —        |
| 18 | 5         | 6      | 1    | TBD    | —         | —        | —        |

Test 16 is the direct A/B against FP8 Case 10 (267.68 tok/s).

### Block E — `speculative_num_draft_tokens` sweep on winner shape (Tests 19–20)

Same shape as Test 16 (num_steps=3, topk=1), drafts ∈ {6, 8}. FP8 sibling found drafts=4 best (monotonically worse at 6/8); NVFP4 has ~2× the KV-cache headroom, so the trade-off may flip.

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|----------:|---------:|---------:|
| 19 | 3         | 6      | 1    | TBD    | —         | —        | —        |
| 20 | 3         | 8      | 1    | TBD    | —         | —        | —        |

### Block F — `speculative_eagle_topk` sweep on winner shape (Tests 21–22)

Same shape as Test 16 (num_steps=3, drafts=4), topk ∈ {2, 4}. FP8 sibling found topk=1 best (topk=2/4 monotonically worse @ n=8); re-tested in case FP4 verification cost changes the per-candidate trade-off.

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|----------:|---------:|---------:|
| 21 | 3         | 4      | 2    | TBD    | —         | —        | —        |
| 22 | 3         | 4      | 4    | TBD    | —         | —        | —        |

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

**Matrix not yet started.** 22 cases planned. Run via:

```bash
kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-nvfp4/nv580.159_sglang-0.5.12_qwen-3.6-27b-nvfp4_n4_ep1.yaml
```

Results dir on completion: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/qwen-3.6-27b-nvfp4/0.5.12/`.

### Pending hypotheses

- **Block A vs B (fi_cutlass-fp4 vs fi_cudnn-fp4 GEMM)** — `fi_cutlass` is the most-validated FP4 GEMM path on SM121. Expect `fi_cudnn` to be at parity or slightly worse; the interesting outcome is if cuDNN-FP4 starts winning here (it has yet to on any GB10 matrix).
- **Block C vs FP8 sibling** — direct A/B at the FP8-winner shape. Smaller weights + same kernel quality should at minimum match 267.68 tok/s @ n=8; the upside scenario is that the KV-cache headroom lets MTP run wider (Blocks D/E) for a real lift.
- **Block D (num_steps sweep)** — FP8 found s=3 best. If NVFP4 KV-cache headroom shifts the sweet spot to s=4 / s=5 → it would mean the FP8 result was bottlenecked on KV, not on draft-acceptance.
- **Block E (drafts sweep)** — FP8 found drafts=4 best. NVFP4 with more KV-cache headroom may finally make drafts=6 / 8 viable.
- **Block F (topk sweep)** — Expect topk=1 to remain best (verification cost scales with topk; FP4 doesn't cheapen that).

---

## Action items after the matrix run

- [ ] Fill the six block tables with actual results
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball — same word-salad concern as the FP8 sibling and the 35B-A3B hybrids)
- [ ] Compute Δ vs FP8 sibling (`qwen-3.6-27b-fp8`) on the matching shapes
- [ ] If any Block C/D/E/F case beats FP8 Case 10 (267.68 tok/s @ n=8): create a `mmangkad-qwen3.6-27b-nvfp4.yml` model profile entry with the winner shape; otherwise keep the FP8 sibling as the active 27B production profile
- [ ] Document whether `fi_cudnn-fp4` ever wins on this dense + modelopt path (so far it has not on any GB10 matrix)
