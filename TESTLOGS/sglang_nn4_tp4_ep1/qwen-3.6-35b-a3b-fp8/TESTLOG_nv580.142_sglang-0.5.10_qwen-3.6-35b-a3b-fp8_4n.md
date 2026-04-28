# SGLang Test Log — Qwen3.6 35B-A3B-FP8 (MoE), 4 Nodes, TP=4 EP=1, v0.5.10

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2                                               |
| Kernel    | 6.19.13-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.10`                 |
| Model     | `Qwen/Qwen3.6-35B-A3B-FP8`                         |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.142_sglang-0.5.10_qwen-3.6-35b-a3b-fp8_n4_ep1.yaml`

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). Fine-grained FP8 (block 128).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
  - Hidden=2048, embedding/lm_head=248 320 (padded).
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
  Same arch class SGLang 0.5.10 already supports for Qwen3.5 MoE → no new image needed.
- VL-fähig (Vision-Encoder), wir fahren rein Text — keine speziellen Flags.

## Known caveats inherited from the Qwen3.5 MoE codepath

- **EPLB stays off** (`enable_eplb: false`): `Qwen3_5MoeForConditionalGeneration`
  lacks `routed_experts_weights_of_layer` — same crash-after-~1000-passes bug
  documented for Qwen3.5-122B-A10B. Confirmed broken on 0.5.9-dev2; PR #19767's
  fix is not effective in our 0.5.10 build.
- **`moe_runner_backend: cutlass` skipped** — `cutlass_moe_fp4` requires FP4
  tensors. FP8 weights → only `triton` and `flashinfer_cutlass` are valid here.

## MTP / speculative decoding

Model card recommends NEXTN with:
```
--speculative-algo NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
SGLang 0.5.10 ships `qwen3_5_mtp.py` so this should be supported out of the box.
Tests 13–14 cover MTP under both MoE runners with `attention=flashinfer` (the
shape that won on Qwen3.5-397B and GLM-4.7 at EP=1). Profile keeps it disabled
(commented-out block) until validated here.

## Expected behaviour

- `moe_runner_backend=triton`: the safe default on SM121; matches the Qwen3.5
  MoE family's stable shape. Expected to be the workhorse across attn variants.
- `moe_runner_backend=flashinfer_cutlass`: at EP=1 on Qwen3.5-397B-NVFP4 the
  fi_cutlass MoE was stable; on FP8 weights the codepath is exploratory.
  Crashes here would mirror the FP4 EP>1 instability and would not be a
  showstopper — `triton` MoE remains the fallback.
- Eager (`disable_cuda_graph: true`) is expected to be slower but stable; the
  `cutlass_moe_fp4` eager-mode `!`-token-collapse only applies to FP4.
- Piecewise CUDA graphs: previously winning on the dense Gemma-4 path; on FP8
  MoE the picture is mixed in the Qwen3.5 family — wait for results.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144, num_experts=256, enable_eplb=false` unless noted. FP8 → no FP4 sweep. `cutlass` MoE skipped (FP4-only).

| #  | nccl | moe_runner   | attention | dis_cuda_graph | dis_piecewise | spec | Status  | n=1 tok/s | n=4 peak | n=8 peak |
|----|------|--------------|-----------|----------------|---------------|------|---------|-----------|----------|----------|
| 1  | roce | triton       | fi        | false          | true          | —    | pending | —         | —        | —        |
| 2  | roce | triton       | fi        | true           | true          | —    | pending | —         | —        | —        |
| 3  | roce | triton       | fi        | false          | false         | —    | pending | —         | —        | —        |
| 4  | roce | triton       | triton    | false          | true          | —    | pending | —         | —        | —        |
| 5  | roce | triton       | triton    | true           | true          | —    | pending | —         | —        | —        |
| 6  | roce | triton       | triton    | false          | false         | —    | pending | —         | —        | —        |
| 7  | roce | fi_cutlass   | fi        | false          | true          | —    | pending | —         | —        | —        |
| 8  | roce | fi_cutlass   | fi        | true           | true          | —    | pending | —         | —        | —        |
| 9  | roce | fi_cutlass   | fi        | false          | false         | —    | pending | —         | —        | —        |
| 10 | roce | fi_cutlass   | triton    | false          | true          | —    | pending | —         | —        | —        |
| 11 | roce | fi_cutlass   | triton    | true           | true          | —    | pending | —         | —        | —        |
| 12 | roce | fi_cutlass   | triton    | false          | false         | —    | pending | —         | —        | —        |
| 13 | roce | triton       | fi        | false          | true          | NEXTN| pending | —         | —        | —        |
| 14 | roce | fi_cutlass   | fi        | false          | true          | NEXTN| pending | —         | —        | —        |

### Column Legend

| Column         | Description |
|----------------|-------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`roce` = RDMA via SR-IOV VF) |
| moe_runner     | `moe_runner_backend` — `triton` or `flashinfer_cutlass` (`fi_cutlass`) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| spec           | speculative decoding (`NEXTN` = MTP, num_steps=3, eagle_topk=1, num_draft_tokens=4) |

---

## Results

_Tests not yet run._
