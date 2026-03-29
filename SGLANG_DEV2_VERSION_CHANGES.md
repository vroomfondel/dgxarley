# SGLang 0.5.9-dev2-acab24a7 Changes

Changes between stable `v0.5.9` (`bbe9c7ee`, 2026-02-23) and `acab24a7` (2026-03-11).
656 commits. Image: `scitrera/dgx-spark-sglang:0.5.9-dev2-acab24a7-t5`.

## Critical Bug Fixes

**Qwen-MoE VRAM memory duplication** (PR #18255, 2026-03-10)
`_cached_params_dict` held stale references after `torch.nn.Parameter()` reassignment during
weight loading, preventing garbage collection. Qwen3 MoE models consumed ~2x VRAM (measured:
110 GB instead of 57 GB on Qwen3-30B-A3B). Affects Qwen3.5 (same MoE stack). Fixed by dropping
the cache and computing `dict(self.named_parameters())` fresh per `load_weights()` call.

**Fused RoPE K writeback regression** (PR #19636, 2026-03-07)
The fused RoPE kernel wrote rotated K to the KV cache but not to the live K tensor. Paths reading
K directly in the same forward pass saw stale data. Caused near-zero GSM8K accuracy on Qwen3-Coder-30B
with flashinfer backend. Fixed by restoring the K writeback in `fused_rope_store_kernel`.

**`/health` probe hang during startup** (PR #19805, 2026-03-04) ⚠️ **Reverted 2026-03-12** (PR #20468)
Early socket prebinding called `listen()` during initialization, allowing TCP connections to queue
before FastAPI was ready. K8s startup/liveness probes would hang instead of getting 503. Fixed:
`bind()` only during init, `listen()` deferred until just before `uvicorn.run()`.
*Note: Both the prebinding feature (#17754) and this fix (#19805) were reverted one day after our
snapshot by PR #20468, restoring uvicorn's native bind. Present in our image, absent on current main.*

**Graceful OOM abort in retract_decode** (PR #19881, 2026-03-11)
When a single decode request exhausted the KV cache pool, `retract_decode` had no other requests
to retract and raised `ValueError`, crashing the scheduler. Now the last request is marked
`FINISH_ABORT` and returned cleanly.

**RadixCache hit_count never incremented** (PR #18843, 2026-03-08)
`node.hit_count` was never incremented in `match_prefix` or `insert`, making all frequency-based
eviction policies non-functional. Also adds SLRU (Segmented LRU) as an alternative eviction
policy — 7.6x tail latency improvement under scan workloads.

**MLA decode returning garbage/NaN rows** (PR #19902, 2026-03-06)
MLA decode kernel only writes the first `qo_indptr[-1]` rows to its output buffer, but the code
selected output rows with `o[q_mask]`, which could reference unwritten padded rows (uninitialized
memory / NaNs). Fixed by returning `o[:qo_indptr[-1]]`.

**DP attention pool over-allocation** (PR #20063, 2026-03-07)
With DP attention, `req_to_token_pool` was sized from global `max_running_requests` instead of
`max_running_requests // dp_size`, over-allocating KV memory per worker.

## Qwen3.5-Specific Fixes

**EPLB + MTP support** (PR #19767, 2026-03-09)
Adds `routed_experts_weights_of_layer` property to `Qwen3_5MoeForConditionalGeneration`.
Enables EPLB with MTP for Qwen3.5 and Qwen3-Next.

**MTP spec_v2 enabled** (PR #19391, 2026-03-04)
Makes MTP_v2 work for Qwen3.5 by passing `mm_input_embeds` to the MTP head. Fixes buffer
validation logic (MTP_v2 only requires `extra_buffer` when radix cache is enabled).

**Pipeline Parallelism** (PR #19670, 2026-03-07)
Previously Qwen3.5 crashed with PP. Adds `PPMissingLayer` placeholder for `embed_tokens`,
proper `start_layer`/`end_layer` distribution. GSM8K accuracy confirmed at 85.29%.

**GDN kernel fusion in verify_target** (PR #19775, 2026-03-06)
Fuses `fused_gdn_gating` and `fused_recurrent_gated_delta_rule_update` for speculative decoding
verify step. Measured **11% TTFT improvement** for 8k input / 1.5k output sequences.

## Performance Improvements

**Batch CUDA copies with `torch._foreach_copy_`** (PR #18558, 2026-03-09)
Replaces individual `copy_()` calls with batched `torch._foreach_copy_()` in
`GraphInputBuffers.populate_from_forward_batch`. GB200: decode throughput +84% at BS=32
(22,896 → 42,189 tok/s).

**Triton MoE padding adjustment** (PR #19174, 2026-03-05)
Changes `round_up` block size from `mxfp4_block` to 64 when Triton kernels are active.
~3.3% throughput improvement (TP=4, BS=64, 32K in / 2K out). Directly relevant to Qwen3.5 MoE.

**D2H operation reduction** (PR #19424, 2026-03-04)
Batches Device-to-Host copies in `compute_output` (59→49µs) and `all_gather` (51→30µs).
Overall D2H overhead halved (2.08% → 1.04%).

**Sliding window attention decode optimization** (PR #19655, 2026-03-05)
n_offset alignment, kBlockN=192, `use_one_mma_wg` for SWA decode layers.
+4.7% output throughput on gpt-oss-120b.

**Skip first delayer to maximize decode batch size** (PR #19836, 2026-03-06)
When running requests are near capacity, the first `merge_batch` delay is skipped.
P99 inter-token latency: 615ms → 278ms.

**FlashInfer v0.6.4** (PR #19537, 2026-03-10)
Integrates mxfp8 GEMM, MoE, and routed MoE kernels. Validated on Qwen-30B mxfp8 at
20,905 tok/s on B200.

**Piecewise CUDA graph unlocked for logprob_start_len=-1** (PR #19453, 2026-03-10)
Previously blocked PCG activation with no benefit; now allowed.

**Weight file sequential I/O optimization** (PR #20194, 2026-03-11)
`SGLANG_SORT_WEIGHT_FILES=1` sorts weight files alphabetically before loading.
~52s improvement on a 2055s load cycle.

## New Features

**SSL/TLS for HTTP and gRPC** (PR #18973, 2026-03-05)
`--ssl-keyfile`, `--ssl-certfile`, `--ssl-ca-certs`, `--ssl-keyfile-password`,
`--enable-ssl-refresh`. TLS can be terminated directly in SGLang without a sidecar.

**Auto NUMA node binding** (PR #15678, 2026-03-09)
Automatic NUMA affinity binding for multi-socket hosts.

**New Prometheus metrics** (PR #19982, 2026-03-07)
`sglang:gpu_overlap_wait_seconds_total` (GPU forward stream idle time),
`sglang:full_token_usage` (full-attention KV usage), split prefill/decode CUDA graph tracking.

**HTTP keep-alive timeout** (PR #19847, 2026-03-05)
`SGLANG_TIMEOUT_KEEP_ALIVE` env var (default: 5s).

**X-Data-Parallel-Rank header** (PR #19832, 2026-03-11)
HTTP header to pin a request to a specific DP rank.

**NIXL-EP MoE backend** (PR #19248, 2026-03-11)
NVIDIA NIXL framework as `--moe-a2a-backend` option for elastic RDMA/NVLink-based EP.

**Priority scheduling improvements** (PR #17026, 2026-03-04)
Configurable default priority, `disable_try_preemption_by_priority` toggle, per-priority
queue metrics.

**Structured JSON logging** (PR #19968, 2026-03-05)
`--json-log` flag on the router.

**`return_logprob` support for spec v2** (PR #19801, 2026-03-10)
Speculative v2 can now return log probabilities without breaking two-stream overlap.

## Distributed / DP Changes

**Ray actor scheduler for DP=1** (PR #17684, 2026-03-05)
Adds Ray actor support for scheduler process management when `dp_size=1`. Changes scheduler
startup behavior by allowing Ray-based process orchestration as an alternative.

**Sync point removal + prefill CUDA graph for DP** (PR #19190, 2026-02-28)
Removes CPU-GPU sync points and enables prefill CUDA graphs for data-parallel configurations.
Also disables cache reset during memory checks. Low-level timing change that may affect
probe/startup behavior.

**Distributed backend selection refactor** (PR #19202, 2026-02-25)
Extracts device-to-backend mapping into `get_default_distributed_backend`. Changes how
CUDA/NCCL vs other backends are automatically selected.

**gRPC streaming last-chunk fix** (PR #19895, 2026-03-04)
Fixes gRPC streaming to send the last chunk before signaling completion. Correctness fix for
streaming responses via gRPC.

## Other Notable Fixes

- **FP8 MTP layer without EP** (#18515) — graceful fallback when `ep_size=1` with a2a backend
- **Eagle v2 NaN/OOB crash** (#19807) — `qo_indptr` layout mismatch during CUDA graph replay
- **DeepSeekV3.2 128K seqlen crash** (#19319) — Triton kernel grid too large
- **PD disagg decode infinite loop** (#20371) — when prefill server went offline
- **Mamba verify update crash on idle batch** (#20167) — missing attention metadata
- **input_embeds retraction shape mismatch** (#14110) — output_ids not cleared on retraction
- **Health-check false positive timeout** (#20256) — multi-tokenizer-worker signal routing
- **FP4 GEMM NaN on Blackwell** (#20047) — missing GDC compile flags, default changed to `auto`
- **Async NaN/OOB detection in EAGLE** (#19899) — `torch._assert_async` for two-stream overlap
- **Decode throughput metric fix** (#19984) — single-batch completions reported near-zero throughput
