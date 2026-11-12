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
| MoE conf  | default for the baseline sweep ("Using default MoE kernel config"); tuned file: see A/B section below                      |
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

## MoE Triton config tuned (A/B)

### What changed

Only the MoE Triton kernel config. The SGLang pods (head, workers 1-3) were restarted at 09:51 UTC on 2026-11-12 with a tuned file in the HF cache; image (`xomoxcc/dgx-spark-sglang:0.5.23-sm121`), profile and `mem_fraction_static: 0.68` are unchanged. The JSON was produced on spark5 with the upstream tuner (`--tp-size 4 --ep-size 4 --dtype fp8_w8a8`, block_shape overridden to [128, 128] because the tuner misderives it for this ModelOpt MIXED_PRECISION checkpoint), 18 batch sizes 1..4096, about 17 h of tuning. No separate down-projection config exists, so the up config is reused without TMA.

### Verification

From the current head log (pod `sglang-head-7874bdfcf4-cw4nv`, started 09:51:27 UTC) and `kubectl get pod`:

- `Using MoE kernel config from /root/.cache/huggingface/moe_configs/configs/triton_3_8_0/E=128,N=640,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128, 128].json.` (09:58:32, TP0 EP0)
- `Down MoE config file not found at .../E=128,N=640,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128, 128]_down.json; reusing the tuned up-projection config without TMA.`
- `server_args`: `mem_fraction_static` 0.68, `max_running_requests` 32, `tp_size` 4, `ep_size` 4, `moe_runner_backend` flashinfer_cutlass, `speculative_moe_runner_backend` triton, `attention_backend` flashinfer, `speculative_algorithm` EAGLE, `speculative_adaptive` True, `mamba_ssm_dtype` bfloat16, `disable_cuda_graph` False.
- Image tag of head and workers `xomoxcc/dgx-spark-sglang:0.5.23-sm121`; all four pods started 09:51 UTC, 0 restarts. `sglang-embed-head` (spark2) still running (16 h age), as in the baseline.
- Idle check at 10:06 UTC: 0 `Decode batch`/`Prefill batch`/POST lines in the 3 min before the sweep.

### Results

Method identical to the baseline above (one sweep, 2026-11-12 10:06-10:20 UTC, same harness, questions and 20 s pauses). Engine max is the highest `gen throughput (token/s)` of the head log `Decode batch` lines in the level window; top-5 median is the median of the five highest such lines (less sensitive to a single line):

| level | baseline engine max | tuned engine max | delta % | baseline top-5 median | tuned top-5 median | delta % |
|-------|--------------------:|-----------------:|--------:|----------------------:|-------------------:|--------:|
| n1    |                89.5 |             89.4 |    -0.1 |                  87.9 |               83.7 |    -4.7 |
| n4    |               255.8 |            241.1 |    -5.7 |                 240.5 |              231.4 |    -3.8 |
| n8    |               350.1 |            337.3 |    -3.6 |                 331.8 |              325.2 |    -2.0 |
| n16   |               553.6 |            531.3 |    -4.0 |                 475.4 |              480.6 |    +1.1 |
| n32   |               737.0 |            706.2 |    -4.2 |                 619.4 |              625.3 |    +1.0 |

Window data:

| level | questions | window (UTC) | max `#running-req` | accept len mean (baseline) | accept len mean (tuned) | POSTs | eager batches |
|-------|----------:|--------------|-------------------:|---------------------------:|------------------------:|------:|--------------:|
| n1    |         8 | 10:06-10:07  |                  1 |                       3.19 |                    3.25 |     8 |             0 |
| n4    |        32 | 10:07-10:09  |                  4 |                       2.88 |                    3.03 |    32 |             0 |
| n8    |        64 | 10:09-10:10  |                  8 |                       3.23 |                    3.09 |    64 |             0 |
| n16   |       128 | 10:10-10:14  |                 16 |                       2.98 |                    2.97 |   128 |             0 |
| n32   |       256 | 10:14-10:20  |                 32 |                       2.96 |                    2.80 |   256 |             0 |

The baseline n16 and n32 maxima are single outlier lines (553.6 and 737.0) well above the next lines (492.5 and 622.6), so the engine max overstates those levels; the top-5 median is the steadier figure. The n1 and n8 maxima come from about 20 log lines.

### GSM8K / quality

| level | exact/n | wrong | correct, not exact | truncated (8192) | char_runs | failed requests | baseline exact/n |
|-------|--------:|------:|-------------------:|-----------------:|-----------|----------------:|-----------------:|
| n1    |     8/8 |     0 |                  0 |                0 | none      |               0 |              8/8 |
| n4    |   31/32 |     1 |                  0 |                0 | none      |               0 |            30/32 |
| n8    |   63/64 |     1 |                  0 |                0 | none      |               0 |            63/64 |
| n16   | 125/128 |     0 |                  1 |                2 | none      |               0 |          125/128 |
| n32   | 248/256 |     5 |                  1 |                2 | none      |               0 |          250/256 |

Failures in the tuned run: n4/n8 id 12 (known recurring failure); n16 truncated ids 13, 119; n32 wrong ids 12, 64, 119, 45, 182, truncated ids 100, 255. "Correct, not exact" is a numerically right answer whose format did not match exactly (one each at n16 and n32). Baseline n32 had 3 wrong and 3 truncated; the tuned run has 5 wrong and 2 truncated, at the lower edge of the 2026-10-12/19 range (249-253/256).

### Observations

- Engine max is lower than baseline at every level except n1 (equal): n4 -5.7 %, n8 -3.6 %, n16 -4.0 %, n32 -4.2 %. The top-5 median is mixed: n1 -4.7 %, n4 -3.8 %, n8 -2.0 %, n16 +1.1 %, n32 +1.0 %.
- All deltas are inside the known ±10-15 % day-to-day spread, so there is no measurable gain and no demonstrable regression either.
- The n32 accept length mean is lower (2.80 vs 2.96) and the n32 level took 308 s vs 284 s window time; with a single run per side this is not conclusive.
- Every level reached full concurrency (`#running-req` = n, queue 0), POSTs match the question counts, 0 decode batches with `cuda graph: False`.
- Reason for the null result (checked in the SGLang source, upstream main): only the Triton fused-MoE path reads these JSON files (`try_get_optimal_moe_config` / `get_moe_configs` in `moe_runner/triton_utils/`); the `flashinfer_cutlass` runner has no reference to them. The main model's MoE layers run on `flashinfer_cutlass` (NVFP4), so the tuned file only reaches the FP8 MTP draft head, which runs on `speculative_moe_runner_backend: triton`. Its `dtype=fp8_w8a8, block_shape=[128, 128]` shape matches that FP8 head. One draft MoE layer is too small a share of step time to show up end to end.

### Conclusion

The tuned `triton_3_8_0` MoE config loads, but it gives no throughput gain beyond noise on this setup (engine max -0.1 % to -5.7 %, top-5 median -4.7 % to +1.1 %), and GSM8K accuracy stays within the usual range (248/256 at n32 versus 250/256). Keeping or dropping the file makes no measurable difference for this profile: the main MoE runs on `flashinfer_cutlass`, which ignores Triton configs, and only the FP8 MTP draft head uses the tuned file. The file stays in the JuiceFS cache (harmless, silences the boot warning).

## Status

Measurement only, nothing redeployed. The baseline sections above are the reference for the tuned-MoE-config A/B.
