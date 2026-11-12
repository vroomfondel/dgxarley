# SGLang Test Log: Qwen3.8-Flash-Next-NVFP4 (NVIDIA export), 4 Nodes, TP=4 EP=4, v0.5.23-sm121, engine-log sweep

## Environment

| Component | Value                                                                                                                      |
|-----------|----------------------------------------------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified per node, driver 580.178.04 (spark `nvidia-smi`)                                       |
| Nodes     | 4x DGX Spark (spark1 head/rank0, spark2-4 workers), TP4/EP4                                                                |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.23-sm121` (head Deployment image)                                                            |
| Versions  | torch 2.14.1, triton 3.8.0, flashinfer 0.7.3, tokenizers 0.23.3 (imported in the head container)                           |
| Model     | `nvidia/Qwen3.8-Flash-Next-NVFP4` (quant `modelopt_mixed`)                                                                 |
| Engine    | `moe_runner_backend: flashinfer_cutlass`, draft MoE `triton`, `attention_backend: flashinfer`, `mamba_ssm_dtype: bfloat16` |
| MTP       | EAGLE, `speculative_adaptive: True` (steps 2, topk 1, draft tokens 3 initial)                                              |
| Memory    | `mem_fraction_static: 0.68` (0.72 on 0.5.21), `max_running_requests: 32`                                                   |
| MoE conf  | default Triton MoE config (no `triton_3_8_0` file; boot log: "Using default MoE kernel config")                            |
| Date      | 2026-11-11, sweep 21:44-21:58 UTC                                                                                          |

Verified from the head log `server_args` and the running pod: image tag, mem_fraction_static, max_running_requests, ep_size, moe_runner_backend, mamba_ssm_dtype, speculative_adaptive, MoE default-config warning; library versions by import inside `sglang-head`. `sglang-embed-head` (spark2) was running during the sweep, i.e. co-located with the worker on the GPUs.

## Method

Same as "Rerun with dual-rail and 4 NCCL channels (2026-10-19)" in `TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_4n.md`, one sweep, no redeploy or config change. Idle check before the sweep: no `Decode batch`/`Prefill batch`/POST lines in the head log for the preceding 3 min. Levels n1/n4/n8/n16/n32 with 8/32/64/128/256 questions (first N of GSM8K test), 20 s pause between levels, temperature 0, max_tokens 8192, harness `sglang-gsm8k run` over a port-forward to `svc/sglang`. Throughput is the highest `gen throughput (token/s)` of the head log `Decode batch` lines inside each level's time window; client-side numbers are not used.

## Results

engine max tok/s (2026-10-19 on 0.5.21 in parentheses):

| level | questions | window (UTC) | engine max tok/s | max `#running-req` | accept len mean | POSTs | eager batches |
|-------|----------:|--------------|-----------------:|-------------------:|----------------:|------:|--------------:|
| n1    |         8 | 21:44-21:45  |      89.5 (83.5) |                  1 |            3.19 |     8 |             0 |
| n4    |        32 | 21:45-21:48  |        256 (254) |                  4 |            2.88 |    32 |             0 |
| n8    |        64 | 21:48-21:49  |        350 (356) |                  8 |            3.23 |    64 |             0 |
| n16   |       128 | 21:49-21:53  |        554 (504) |                 16 |            2.98 |   128 |             0 |
| n32   |       256 | 21:53-21:58  |        737 (762) |                 32 |            2.96 |   256 |             0 |

n1 single stream: 74-77 tok/s on typical log lines (client `full_load_tok_s` 74.7), peak line 89.5.

## GSM8K / quality

| level | exact/n | wrong | truncated (8192) | char_runs | failed requests |
|-------|--------:|------:|-----------------:|-----------|----------------:|
| n1    |     8/8 |     0 |                0 | none      |               0 |
| n4    |   30/32 |     1 |                1 | none      |               0 |
| n8    |   63/64 |     1 |                0 | none      |               0 |
| n16   | 125/128 |     2 |                1 | none      |               0 |
| n32   | 250/256 |     3 |                3 | none      |               0 |

Within the 2026-10-12/19 range (n32 249-253/256, up to 4 truncations). The same few questions recur as failures (ids 12, 13, 93, 119).

## Observations

- Against 2026-10-19: n1 +7 %, n4 +1 %, n8 -2 %, n16 +10 %, n32 -3 %. Single runs; run-to-run spread on this cluster was ±15 % before, so only n16 is a candidate for a real gain and nothing here is a regression.
- Every level reached its full concurrency (`#running-req` = n, queue 0), POSTs match the question counts, 0 decode batches with `cuda graph: False`.
- Changes since the 0.5.21 run: slim base image with cubin prune, triton pinned to 3.8.0, tokenizers 0.23.3, `mem_fraction_static` 0.72 -> 0.68, `jit_max_jobs` 2, `sglang-embed-head` co-located and running. The MoE Triton config is still the default one, so the fused-MoE kernels run sub-optimal until a tuned `triton_3_8_0` file exists (tuning runs separately).
- Accept length mean 2.88-3.23, in line with the 0.5.21 profile (2.96 mean).

## Status

Measurement only, nothing redeployed. Reference for the tuned-MoE-config comparison.
