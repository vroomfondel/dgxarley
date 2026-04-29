# SGLang Test Log — Qwen3.6 27B-FP8 (dense), 4 Nodes, TP=4 EP=1, v0.5.10

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
| Model     | `Qwen/Qwen3.6-27B-FP8`                             |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/nv580.142_sglang-0.5.10_qwen-3.6-27b-fp8_n4_ep1.yaml`

---

## Model Notes

- 27B **dense** (NOT MoE), hybrid Gated DeltaNet + Gated Attention. Fine-grained FP8 (block 128).
- Architecture: 16 layers of (3× Gated DeltaNet → FFN) + (1× Gated Attention → FFN).
  - Gated DeltaNet: 48 linear-attn V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 24 Q-heads, 4 KV-heads, head_dim=256, RoPE dim=64.
  - FFN intermediate: 17 408.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5ForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
  Same code path SGLang already supports for Qwen3.5 dense → no new image needed.
- MTP-trained — NEXTN speculative decoding available (model card recommends
  `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`).
  Disabled in profile pending validation; not in this matrix.
- VL-fähig (Vision-Encoder), wir fahren rein Text — keine speziellen Flags.

## Notes vs. the Qwen3.5 family

- Same `Qwen3_5ForConditionalGeneration` arch as Qwen3.5 dense → 0.5.10 supported out of the box.
- No EPLB question (dense model).
- FP8 (not FP4) → no `fp4_gemm_backend` sweep, no `cutlass_moe_fp4` codepath.
- Recommended `mem_fraction_static=0.80` per card; profile mirrors that.

## Expected behaviour

- `attention_backend=triton` is the profile default and the most likely-stable
  path on SM121 (matches Qwen3.5 dense / Gemma-4 patterns).
- `attention_backend=flashinfer` — head_dim=256 on the gated-attn path is
  nominally inside FlashInfer's dispatch table; expected to work but unverified
  on SM121 for this hybrid arch. If it crashes (`prefill.cuh` Invalid
  configuration), it would mirror the Gemma-4 head_dim=512 issue.
- Eager (`disable_cuda_graph: true`) is expected to be stable but slower —
  no `cutlass_moe_fp4` codepath here that breaks under eager.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144` unless noted. Dense → no MoE sweep. FP8 → no FP4 sweep.

| # | nccl | attention | dis_cuda_graph | dis_piecewise | Status               | n=1 tok/s | n=4 peak | n=8 peak |
|---|------|-----------|----------------|---------------|----------------------|-----------|----------|----------|
| 1 | roce | fi        | false          | true          | bench_crash†         | —         | —        | —        |
| 2 | roce | fi        | true           | true          | bench_crash†         | —         | —        | —        |
| 3 | roce | fi        | false          | false         | bench_crash†         | —         | —        | —        |
| 4 | roce | triton    | false          | true          | bench_crash†         | —         | —        | —        |
| 5 | roce | triton    | true           | true          | bench_crash†         | —         | —        | —        |
| 6 | roce | triton    | false          | false         | aborted (re-run)     | —         | —        | —        |

† All requests returned `status=repetition` — RepetitionGuard tripped on
chinesische n-gram floods im `<think>`-Stream. Server itself was healthy
(TTFT ~0.8s, model loads cleanly). Root cause: the model card's general
thinking-mode default `presence_penalty=0.0` is too lenient for this
arch. See "First run aborted" below.

### Column Legend

| Column         | Description |
|----------------|-------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`roce` = RDMA via SR-IOV VF) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |

---

## Results

### First run aborted (2026-04-29) — RepetitionGuard floods

Initial run kicked off 2026-04-29 with profile defaults. Tests 1–5 all came
back `bench_crash` with **every single request flagged `status=repetition`**;
test 6 was aborted before completing. Result dir:
`kikube/matrixtest/2026-04-29/results/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/0.5.10/`.

Diagnosis:
- Server is healthy across all 5 cases — model loads, attention works,
  TTFT ~0.8s on first request, CUDA-graph capture (where applicable) finishes.
- Each request emits `output_tokens=0`, `status=repetition` — i.e. the bench
  harness's RepetitionGuard tripped inside the `<think>` block before any
  visible output. Logs show **chinesische n-gram floods** in the thinking
  stream.
- Root cause: the model card's recommended `presence_penalty=0.0` for general
  thinking is too lenient for this hybrid Gated-DeltaNet arch on the
  bench-prompt mix.

Fix applied (profile, bench-only):

```yaml
# qwen-qwen3.6-27b-fp8.yml — recommended_sampling unchanged
sampling_overrides:
  presence_penalty: 1.5
  frequency_penalty: 0.5
  min_tokens: 4
```

`recommended_sampling` keeps the card defaults (so direct API users see the
documented values); `sampling_overrides` is merged only by the integration-
test bench. Same shape will also be applied to the 35B-A3B sibling for
symmetry.

### Second run — pending

Re-run not yet started.
