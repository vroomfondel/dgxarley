# SGLang Test Log — Qwen3.6 27B-FP8 (dense), 4 Nodes, TP=4 EP=1, v0.5.11

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                 |
| Kernel    | 6.19.13-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.11`                 |
| Model     | `Qwen/Qwen3.6-27B-FP8`                             |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/nv580.142_sglang-0.5.11_qwen-3.6-27b-fp8_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Spec V2 with
Overlap-Scheduling is now baseline (PR #21062). See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 27B **dense** (NOT MoE), hybrid Gated DeltaNet + Gated Attention. Fine-grained FP8 (block 128).
- Architecture: 16 layers of (3× Gated DeltaNet → FFN) + (1× Gated Attention → FFN).
  - Gated DeltaNet: 48 linear-attn V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 24 Q-heads, 4 KV-heads, head_dim=256, RoPE dim=64.
  - FFN intermediate: 17 408.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration`-style hybrid.
- Same hybrid-mamba arch family as Qwen3.6-35B-A3B-FP8 — **inherits the same
  word-salad concurrency-race observed there in v0.5.11** (see
  `qwen-3.6-35b-a3b-fp8/TESTLOG_..._sglang-0.5.11_*` Correctness Debug Sweep).
  Verify output quality manually for n=4 and n=8.

## Configuration Matrix (18 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144`. Dense → no MoE-runner sweep. FP8 → no FP4 sweep. All speculative cases (7–18) use NEXTN with `mamba_scheduler_strategy=extra_buffer + enable_spec_v2=true`.

### Block A: backend baseline (no MTP, Tests 1–6)

| #  | attention | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|----------------|---------------|--------|-----------|----------|----------|
| 1  | fi        | false          | true          | tbd    | —         | —        | —        |
| 2  | fi        | true           | true          | tbd    | —         | —        | —        |
| 3  | fi        | false          | false         | tbd    | —         | —        | —        |
| 4  | triton    | false          | true          | tbd    | —         | —        | —        |
| 5  | triton    | true           | true          | tbd    | —         | —        | —        |
| 6  | triton    | false          | false         | tbd    | —         | —        | —        |

### Block B: MTP (NEXTN) baseline at num_steps=3, Tests 7–8

| #  | attention | dis_piecewise | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|---------------|-----------|--------|------|--------|-----------|----------|----------|
| 7  | fi        | true          | 3         | 4      | 1    | tbd    | —         | —        | —        |
| 8  | triton    | true          | 3         | 4      | 1    | tbd    | —         | —        | —        |

### Block C: winner-shape `speculative_num_steps` sweep (fi + CG on + piecewise off + MTP), Tests 9–12

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 9  | 2         | 4      | 1    | tbd    | —         | —        | —        |
| 10 | 3         | 4      | 1    | tbd    | —         | —        | —        |
| 11 | 4         | 4      | 1    | tbd    | —         | —        | —        |
| 12 | 5         | 4      | 1    | tbd    | —         | —        | —        |

Test 10 is a re-run of Test 7 at the same num_steps=3 to validate stability of the
sweet-spot inside this block.

### Block D: piecewise CG **ON** + MTP, Tests 13–14

| #  | attention | dis_piecewise | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|---------------|-----------|--------|------|--------|-----------|----------|----------|
| 13 | fi        | false         | 3         | 4      | 1    | tbd    | —         | —        | —        |
| 14 | triton    | false         | 3         | 4      | 1    | tbd    | —         | —        | —        |

### Block E: winner-shape `speculative_num_draft_tokens` sweep, Tests 15–16

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 15 | 3         | 6      | 1    | tbd    | —         | —        | —        |
| 16 | 3         | 8      | 1    | tbd    | —         | —        | —        |

### Block F: winner-shape `speculative_eagle_topk` sweep, Tests 17–18

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 17 | 3         | 4      | 2    | tbd    | —         | —        | —        |
| 18 | 3         | 4      | 4    | tbd    | —         | —        | —        |

### Column Legend

| Column         | Description |
|----------------|-------------|
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length |
| num_steps      | `speculative_num_steps` — NEXTN draft depth |
| drafts         | `speculative_num_draft_tokens` — verified per step |
| topk           | `speculative_eagle_topk` — candidates per step (1 = pure NEXTN) |

---

## Results

**Run pending.**

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/0.5.11/`.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_qwen-3.6-27b-fp8_4n.md`:

- **Non-MTP winner:** Test 3 (fi + CG on + piecewise on) — 22.0 / 84.3 / 158.6 tok/s @ n=1/4/8.
- **MTP winner:** Test 8 (triton + CG on + piecewise off + MTP num_steps=3) — 36.6 / 152.6 / **239.4** tok/s.
  Test 7 (fi-attn variant) was the n=1 leader at 44.4 tok/s and tied at n=8 (238.8).
- **MTP gain over best non-MTP:** +102 % n=1, +74 % n=4, +52 % n=8.

For 0.5.11 the same Block A/B cases plus 4 sweep blocks (C–F) explore whether the
new defaults (Spec V2 + Overlap-Scheduling, sgl-kernel 0.4.2, FlashInfer 0.6.8.post1,
Eagle3/DFLASH CUDA-Graph-Init fix #22836) shift the optimum away from `num_steps=3 /
drafts=4 / topk=1 / piecewise=off`. Populate each block's table after the run, then
add a delta section here covering:

1. Block A vs 0.5.10 Tests 1–6 (toolchain delta only).
2. Block B vs 0.5.10 Tests 7–8 (Spec V2 default, MTP baseline).
3. Block C: best `num_steps` for this model (model card recommends 3 — does the sweep agree?).
4. Block D: does piecewise-on combine constructively with MTP, or does it cannibalise the speedup?
5. Block E/F: do larger draft pools or higher eagle_topk pay off, or do they cost more than they save?

Pay particular attention to **output quality at n=4 and n=8** — same hybrid-mamba
arch family as Qwen3.6-35B-A3B-FP8, which exhibited the word-salad concurrency-race
in v0.5.11 (see `qwen-3.6-35b-a3b-fp8/TESTLOG_..._sglang-0.5.11_*` Correctness Debug
Sweep). Verify token coherence per case before recording a `STABLE` status.

### DFLASH (intentionally not tested)

DFLASH symbols are present in the 0.5.11 image but require a separate draft-model
path (like EAGLE3); Qwen3.6 only ships built-in NEXTN/MTP heads. To bench DFLASH
on this model, a compatible draft would need to be sourced first.
