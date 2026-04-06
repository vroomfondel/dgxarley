# SGLang v0.5.10 Changes (vs. v0.5.10rc0)

Changes between `v0.5.10rc0` (tag `1115dbf2`, 2026-03-28) and release
`v0.5.10` (2026-04-06). 250 commits.

Previous delta doc: `SGLANG_v0.5.10rc0_VERSION_CHANGES.md` (acab24a7 -> v0.5.10rc0, 462 commits).

## Breaking Changes

**FlashInfer 0.6.6 -> 0.6.7.post2**
Routine bump, but includes 3 bugfixes (issues #19081, #18989, #18980) and new
`flashinfer_trtllm` mxfp8 GEMM backend. Required by `sglang-kernel` 0.4.1.

**`sglang-kernel` 0.4.0 -> 0.4.1**
Kernel package version bump. Legacy `sgl-kernel` import paths removed (#21528).

**`/get_server_info` deprecated** (PR #21463)
All internal callers migrated to `/server_info`. The old endpoint still works
(deprecation wrapper) but may be removed in a future release. If any external
tooling hits `/get_server_info`, update it now.

## Critical Fixes for Our Deployment

### reasoning_tokens Finally Counted (PR #15562, merged 2026-04-04)

The `reasoning_tokens` field in `/v1/chat/completions` `usage` was always 0
for thinking models (documented in `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md`).
Now properly populated. This was our longest-standing upstream bug report.

**Status change**: `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md` can be marked as
**FIXED** in v0.5.10. Was NOT in rc0.

### Subprocess Liveness Monitor (PR #18582)

A `SubprocessWatchdog` daemon thread polls `proc.is_alive()` every 1 second
on all scheduler and detokenizer subprocesses. When a C++-level crash occurs
(e.g., NCCL timeout triggering `std::terminate()`), the main process previously
became a "zombie service" -- accepting TCP connections but unable to process
them. The watchdog now sends `SIGQUIT` to trigger proper cleanup.

**Directly addresses the stale-service scenario from CLAUDE.md**: when the
head's NCCL connection breaks at the C++ level, the watchdog detects the
subprocess death and triggers self-termination. kubelet restarts the head, and
the worker's `livenessProbe` eventually detects the broken NCCL pipe.

Design:
- Multi-node: non-zero rank nodes get `None` watchdog (only rank 0 monitors)
- Ray backend: only monitors detokenizer, not actor schedulers
- Normal exit (exitcode=0) does NOT trigger SIGQUIT
- Overhead: one `is_alive()` poll per second per process (negligible)

Note: This PR was in the rc0 commit window (merged 2026-03-29) but is listed
here because it was not captured in `SGLANG_v0.5.10rc0_VERSION_CHANGES.md`
as a separate entry until confirmed in the final tag.

### Scheduler Launch Hang on Rank Death (PR #20287)

Fixes the head/worker deadlock pattern from CLAUDE.md: during multi-node
initialization, the startup wait loop only monitored the rank it was currently
reading from via `pipe.recv()`. If a different rank died (e.g., OOM-killed by
the kernel), the surviving rank hung indefinitely at `recv()` with no error
output.

Fix: monitor all ranks for liveness during the blocking recv loop, not just the
one being read. When any rank dies, the surviving ranks detect it and abort
cleanly rather than hanging.

### Fix Remote Weight Info for nnode>1 and dp>1 (PR #17389)

In multi-node setups, each node only knew about its local scheduler's transfer
engine info. If a caller queried `/get_remote_instance_transfer_engine_info?rank=N`
and rank N was on a different node, the request failed. In the dp>1 case,
`dp_controller` did not propagate `scheduler_info` upward.

Fix: complete refactor -- instead of propagating through
`ModelRunner -> Scheduler -> Engine -> HTTP Server -> Caller`, an
`EngineInfoBootstrapServer` is introduced where `ModelRunner` registers per-rank
engine information directly after the scheduler subprocess starts. The old
propagation path is removed entirely.

**Directly relevant to our 4-node TP setup**.

## Security Fixes

### CVE-2026-3059 / CVE-2026-3060: ZMQ Remote Code Execution (PR #21435, CVSS 9.8)

Part of CERT/CC VU#665416 ("ShadowMQ" advisory). ZMQ broker sockets in the
multimodal generation path and the disaggregation encoder receiver were bound to
`tcp://*:port` (all interfaces), making `pickle.loads()` / `recv_pyobj()` calls
reachable from any network peer without authentication -- enabling Remote Code
Execution.

Fix: default binding changed to `tcp://127.0.0.1:port`. Cross-machine
disaggregation call sites that genuinely need external access pass their host
explicitly.

### CVE-2026-3989: Unsafe pickle.loads (PR #20904, CVSS 9.8)

Defense-in-depth layer using `SafeUnpickler` (allowlist/denylist of allowed
module prefixes) as a drop-in replacement for raw `pickle.loads()`. Main
affected site: `scripts/playground/replay_request_dump.py`. Blocks known RCE
gadget chains (`os.system`, `subprocess.Popen`, `eval`, `exec`, etc.).
Negligible performance overhead (not on the hot inference path).

### Docker CVE Fixes (PR #21789)

Fixes CVEs in Docker image: pillow, linux-libc-dev, and broken
`sgl-model-gateway` build. Trivy vulnerability scanning added to nightly dev
Docker builds (#21772).

## Performance Improvements

### Fused Temperature + Softmax in Sampling (PR #20501)

Replaces the two-kernel decode sampling path (`logits.div_(temperatures)` then
`torch.softmax()`) with a single Triton kernel:
- Vocab <= 32768: single-pass variant, entire vocab in registers (1 read +
  1 write = 2 memory passes instead of 6)
- Larger vocabs (Llama3 128K, Qwen 152K): 2-pass online softmax with
  `@triton.autotune` (2 reads + 1 write = 3 memory passes)

Temperature division is fused into the load pass, eliminating a separate
division kernel launch. Meaningful for large-vocab models at decode latency
budget scale.

### topk Sampling: O(n log k) Instead of O(n log n) (PR #14702)

The speculative decoding / top-k sampling code used a full `sort` operation
when only the top-k elements were needed. Replaced with `topk`. For vocab
sizes of 128K-152K, the savings are substantial.

### Async Logprob Transfer: +26% Throughput (PR #20972)

The `return_logprob` path previously performed a synchronous GPU->CPU copy
that stalled the scheduler. Fix: compute logprobs on GPU -> async GPU->CPU
transfer in `copy_to_cpu()` -> convert CPU tensor to list after async transfer
completes. Overlaps logprob transfer with other work.

Measured: output token throughput for logprob-enabled requests improved from
**196.23 -> 246.91 tok/s** (+26%) at concurrency=1 with `top_logprobs_num=5`.

### Multi-Thread Weight Loading Now Default (PR #20289)

Multi-threaded weight loading in `DefaultModelLoader` was previously opt-in.
Now enabled by default. Reduces model startup time (relevant for our ~7-8 min
head startup).

### CUTLASS NVFP4 GEMM Improvement for SM120 (PR #21314)

Exhaustive CUTLASS profiler search for NVFP4xNVFP4 tile configurations on
SM120 (Blackwell consumer, GeForce RTX 50xx) -- distinct from SM100
(GB100/GB200). SM120 lacks multicast support and certain tile sizes that SM100
has. New heuristics approach cuDNN performance for M<=128 decode batch sizes.

Measured: M=16, N=6144, K=5120 (representative decode GEMM): **~20% speedup**
vs previous heuristics.

Not relevant for our DGX Sparks (Grace Hopper / GH200), but signals kernel
maturity.

### FlashInfer TRT-LLM MXFP8 GEMM Backend (PR #21576)

New `--fp8-gemm-backend flashinfer_trtllm` flag for MX-FP8 block-wise
quantization via FlashInfer v0.6.7's TRT-LLM kernel. Avoids storing both
swizzled and non-swizzled scaling factors.

Measured on Qwen3-4B MXFP8: `flashinfer_trtllm` -> 24525 tok/s output;
`flashinfer_cutlass` -> 24892 tok/s (within noise, similar accuracy 0.841
GSM8K).

### FlashMLA Rollback Reverted (PR #21922)

Previous rollback to an older FlashMLA version was reverted -- back on the
latest FlashMLA.

## Piecewise CUDA Graph Fixes

PCG is default since rc0 -- these are stability fixes landed between rc0 and
v0.5.10:

### Crash with `--enable-mixed-chunk` (PR #20441)

When running with both `--enable-mixed-chunk` and PCG, the scheduler sets
`forward_mode=ForwardMode.MIXED` for runtime batches, but PCG graphs are
captured under `ForwardMode.EXTEND`. TorchDynamo's strict guards triggered
unexpected recompilation. During recompilation, `get_pcg_capture_stream()`
returns `None` outside the dedicated capture phase -> `AssertionError`.

Fix: in `replay_prepare()`, normalize `ForwardMode.MIXED` -> `ForwardMode.EXTEND`
before replay so the pre-captured graph is used.

### qo_indptr Wrong Final Value (PR #21452)

Padding tokens caused `qo_indptr` to have an incorrect final value. Fix: append
a fake (bs+1)-th request with `pad_tokens` extend tokens whose KV indices all
point to scratch slot 0 -- this makes `qo_indptr[-1] = static_num_tokens`
without affecting causal masks for real requests.

Measured on Qwen3-14B: throughput improved from **2255 -> 2968 tok/s** when
enabling CUDA graph with this fix.

### Crash on Models Without `.layers` (PR #21565)

EAGLE3 draft model (`LlamaForCausalLMEagle3`) uses a single `midlayer`
attribute instead of a `layers` list. `init_piecewise_cuda_graphs()` tried to
iterate `language_model.model.layers`, crashed with `AttributeError`, and the
subprocess exited with code -9. Fix: `hasattr` guard -- draft model skips PCG
gracefully, main model still uses it.

### NemotronH PCG Silently Disabled (PR #21436)

Layer discovery loop only appended to `attention_layers` for attention/Mamba
layers. NemotronH's pure MLP layers were skipped, making
`len(attention_layers) < num_hidden_layers` and triggering the bail-out early
return.

Fix: append `None` as a positional placeholder for non-attention/non-Mamba
layers.

Measured on Nemotron-Nano-9B-v2: GSM8K throughput improved from **1507 ->
1763 tok/s** (~17% speedup) by enabling PCG.

### CUDA Graph Max BS Capture Upper Bound (PR #21005)

When users set `--cuda-graph-max-bs`, the generated list of captured batch
sizes did not reliably include that exact value at the top end. Requests at
exactly max-bs would fall through to non-CUDA-graph execution -> performance
regression. Fix: ensure captured batch sizes always include the configured
`cuda_graph_max_bs`.

### PCG + SWA Eviction (PR #21754)

Sliding Window Attention eviction now works with PCG enabled.

### PCG + torch.dynamo Recompile in mxfp8 Triton Path (PR #21888)

Fixes spurious recompilation in the mxfp8 Triton quantization path when PCG
is active.

## Model Fixes

### GLM-4.7 / GLM-V Gate Projection FP32 Cast (PR #21660)

GLM models require inputs to the gate projection to be cast to FP32 before the
operation to avoid numerical issues. Without this, inference produces degraded
output quality. Subset of PR #21258.

### Mistral Small 4 Startup Crash (PR #21620)

`mistralai/Mistral-Small-4-119B-2603` ships with both `params.json` and
`config.json`. The config-loading path matched `"mistral-small-4"` and parsed
`params.json` expecting native Mistral weight names, but the weight-loading
path saw both files and returned `is_mistral_native_format()=False`, loading
HF-format names instead. The remapping regex couldn't match -> all weights
skipped -> `kv_b_proj = None` -> crash.

Fix: in `_is_mistral_native_format()`, when both files exist, check if the
model name matches patterns that trigger the native format loader and return
`True`.

### MiniMax rope_theta After transformers v5 (PR #21241)

transformers v5 changed how MiniMax models expose their `rope_theta`
configuration, breaking SGLang's extraction. GSM8K accuracy verified at 0.959
post-fix.

### Qwen3.5 MoE Model Loading + Mamba Cache Sharding in PP Mode (PR #21448)

Two bugs in Pipeline Parallel mode for Qwen3.5 MoE:
1. Mamba cache allocated for all layers regardless of PP rank, wasting GPU
   memory on layers the rank doesn't own. Fix: filter `mamba_layer_ids` to
   only layers within `[start_layer, end_layer)`.
2. MoE expert weights for layers outside the current PP rank were attempted,
   causing `KeyError: 'model.layers.21.mlp.experts.w2_weight'`. Fix: skip
   loading expert weights for layers not belonging to the current PP rank.

### GLM-4V Bugfix (PR #17122)

Long-standing GLM-4V model fix.

### Llama EAGLE3 Bugfix (PR #21397)

Fix for EAGLE3 speculative decoding with Llama models.

### Qwen3.5 MoE Triton Tuning (PR #20232, also in rc0)

The `get_model_config()` crash documented in `SGLANG_MOE_TUNE_UPSTREAM_BUG.md`
is fixed upstream. Our monkey-patch in `sglang_tune_moe.sh` will gracefully
skip (target string no longer found).

## Tool Calling / API Compatibility

### Multi-Tool Streaming Fix (PR #20004)

When a Qwen2.5 model output multiple tool calls with `stream=True`, the
block-based format uses `</tool_call>\n<tool_call>\n` markup between calls.
`BaseFormatDetector.parse_streaming_increment` assumed JSON immediately
followed `tool_call_separator`, so the markup between calls caused
`MalformedJSON` -- all tool calls after the first were **silently dropped**.

Fix: when JSON parsing fails in the separator branch, search for `bot_token`
to skip past markup and locate the next JSON object. Broadens exception
handling to catch both `MalformedJSON` and `json.JSONDecodeError`.

### parallel_tool_calls Support (PR #20208)

A `maxItems: 1` JSON schema constraint in `get_json_schema_constraint` caused
models to stall on whitespace when `tool_choice` specified a function and the
prompt implied multiple calls. Fix: remove the unconditional `maxItems: 1`;
expose `parallel_tool_calls: bool = True` on `ChatCompletionRequest` (OpenAI-
compatible). When `false`, `maxItems: 1` is applied.

### Streaming Validation Errors Return HTTP 400 (PR #21900)

For oversized prompts (input tokens > context length), non-streaming requests
correctly returned HTTP 400, but streaming requests returned HTTP 200 with an
SSE error payload (matching neither OpenAI nor vLLM behavior). Root cause:
`_handle_streaming_request()` returned `StreamingResponse` immediately,
locking in HTTP 200 before validation ran.

Fix: `await __anext__()` on the generator first -- if validation fails before
the first yield, return HTTP 400 directly.

## Observability

### MFU Metrics in Prometheus (PR #19395)

Three new opt-in Prometheus counters:
- `sglang:estimated_flops_per_gpu_total`
- `sglang:estimated_read_bytes_per_gpu_total`
- `sglang:estimated_write_bytes_per_gpu_total`

Gated by `--enable-mfu-metrics` (requires `--enable-metrics` also set).
Estimation covers: linear-layer FLOPs, attention dot-product FLOPs, weight
read bytes, activation read/write bytes, and KV cache read/write bytes.

Benchmark on A100 (Qwen3-0.6B, 1000 prompts): no throughput regression between
gate-off and gate-on -- differences within 3-5% run-to-run variance.

**Useful for our Prometheus/Grafana monitoring stack** -- can be added to the
SGLang dashboard.

## Streaming / Memory Fixes

### Mamba Cache Leak on Failed PrefillAdder Add (PR #21404)

When `MambaRadixCache._match_post_processor` ran and `mamba_pool.alloc` was
called (step 1), then `PrefillAdder` tried to add the request and called
`req.init_next_round_input()` which also allocated (step 2), but the adder
then failed to add the request (`batch_is_full=True`) -- the mamba allocation
from step 2 was never freed. This caused a slow memory leak (1 page per failed
add) that accumulated until the scheduler's `check_memory()` raised
`ValueError: token_to_kv_pool_allocator memory leak detected!` and crashed.

Fix: add a free call in the failure path of the adder.

### Streaming Backlog Coalescing Scoped (PR #21037)

A previous PR (#19977) added queue processing logic to coalesce streaming
backlog, but it was applied unconditionally to all streaming requests. Now
scoped to only apply when `incremental_streaming_output` mode is active. The
spam log for accumulated queue is now only emitted when queue depth >= 20
chunks.

## Network / Infrastructure

### IPv6 Address Wrapping (PR #21236)

Several places constructed `host:port` strings with `f"{host}:{port}"`, which
is broken for IPv6 addresses (e.g., `http://::1:30000` instead of
`http://[::1]:30000`). Fix: use `NetworkAddress.to_host_port_str()` /
`to_url()` throughout bench_serving, disaggregation bootstrap/gRPC, and
health check log messages.

Not directly relevant (we use IPv4 on QSFP), but prevents future issues.

### NUMA Auto-Configuration (PR #19452)

Enables automatic NUMA configuration on NVIDIA systems. May improve memory
locality on multi-socket systems.

### Direct Model Loading from Object Storage (PR #17948)

New Runai Model Streamer integration for loading models directly from S3, GCS,
or Azure blob storage. Gated by new `runai` extra dependency.

Not relevant for our setup (local HF cache on NVMe).

## New Features

### Stronger Transformers Modeling Backend (PR #19163)

Generic modeling backend using HuggingFace `AutoModel.from_config()` directly.
Any model with a `tp_plan`/`pp_plan` and custom attention support can run on
SGLang without a dedicated model implementation. Architecture:

- `TransformersBase`: meta-device init, recursive Linear->TP replacement,
  attention injection, PP support, weight loading via `AutoWeightsLoader`
- `CausalMixin`: LM head + logits
- `MoEMixin`: auto-detects expert modules, replaces with
  `TransformersFusedMoE` with fused kernels and EPLB recording
- `MultiModalMixin`: vision/audio encoder dispatch, M-RoPE

Tested on H100/A100 with Qwen3-0.6B (TP1/2, torch compile), Qwen3-30B-A3B
MoE, Gemma3-4B-IT VLM, Qwen3-VL-2B VLM.

### LoRA for MoE Layers (PRs #21439, #21466, #21469, #21570)

Series of PRs expanding LoRA support for Mixture-of-Experts:
1. Auto-detect lora target modules (#21439)
2. Shared outer experts + Qwen3-30B-A3B support (#21466)
3. Qwen3-VL-30B-A3B-Instruct support (#21469)
4. GPT-OSS 20B support (#21570)

### Skip-Softmax Attention (PR #19089)

Support for skip-softmax attention -- a variant that omits the softmax
normalization in attention computation.

### CUDA Graph + Timestamp for Whisper (PR #21190)

Enables CUDA graph capture and timestamp support for Whisper speech models.

### MiMo-V2-Flash Reasoning Parser (PR #21414)

Adds a reasoning parser for MiMo-V2-Flash models.

## AMD / NPU / Platform-Specific

### AMD
- Fused rope KV store (#21315)
- MXFP4 Qwen3.5-397B-A17B support (#21234)
- Configurable KV transfer overlap for disaggregation (#20410)
- MoRI bumped to v0.1.0 (#21673)
- tgemm.mm for MoEGate router gemm (#21657)
- Optimized Qwen3-VL decode: fused QK-norm + 3D mRoPE + KV cache write
  (#21458)
- GLM-4.7 FP8 accuracy CI for MI35x (#21534)
- Fix performance regression with `--context-length 13824` (#21691)

### NPU / Ascend
- DeepSeek-V3.2 deployment docs (#21468)
- GLM-5 optimized with fused kernels (#18617)
- Parallel decoding for qwen-image (#20757)
- Ring attention with FA (#21383)
- GLM-4.7-Flash support (#21408)

### CPU
- `apply_rotary_pos_emb_cpu` for Qwen3-VL/Omni (#13121)
- MXFP4 GEMM kernels for Intel AMX (#14385)

### Intel GPU (XPU)
- DeepSeek R1 inference on XPU (#18461)

### Apple Silicon (MLX / MPS)
- Fix Triton stub sub-module imports on Python 3.12+ (#21551)

## Diffusion Model Changes

- Overlay model materialization (#21600) + enhanced overlay mechanism (#21648)
- Flux.2-Klein prompt tokenization fix (#21407)
- Flux.2 TP fix (#21664)
- Wan2.2-I2V-A14B video max size fix (#21390)
- MOVA support (#21633)
- `--strict-ports` for predictable port assignment (#21320)
- `--uvicorn-access-log-exclude-prefixes` to suppress noisy access logs (#20379)
- `ORJSONResponse` -> `orjson_response` deprecation (#21755)
- Unified TeaCacheParams (#20706)
- Ring SP performance benchmark page (#20998)
- Weight loading hooks refactored to dedicated file (#21366)

## JIT Kernel Changes

- Fused `qknorm_rope` JIT kernel (#19059)
- Migrate cast (downcast_fp8) from AOT to JIT (#19103)
- Optimized `qknorm_across_heads` CUDA kernel (#21503)
- Fused GDN `kkt + solve_tril` kernel (#21411)
- KDA fused `scaled_dot_kkt + solve_tril + recompute_w_u` kernel (#21604)
- Updated JIT rmsnorm (#21834)
- Optimized `fused_qknorm_rope`: deduplicated sincosf for interleave RoPE
  (#21654)
- Migrated ngram corpus from torch cpp_extension to TVM FFI jit_kernel
  (#21920)
- Fused temperature + softmax in sampling (#20501) -- see Performance section

## Disaggregated Serving

### GPU Staging Buffer with Dynamic Ring Allocator (PR #19890)

In disaggregated serving with mismatched TP layouts (e.g., prefill TP4 +
decode EP4), KV cache transfer required redistributing head slices. The
original per-token RDMA approach generated ~30,000 small RDMA requests per
request, severely bottlenecking throughput.

New `StagingBuffer` (backed by `cuMemCreate` for NVLink compatibility) with a
dynamic ring-buffer allocator gathers scattered head slices into contiguous
memory for a single bulk RDMA transfer. Reduces RDMA request count by ~1000x.
Decode-side async scatter overlaps with decode forward on a separate CUDA
stream. Opt-in via `SGLANG_DISAGG_STAGING_BUFFER=true`.

Measured: LongbenchV2 (128k ISL) total latency dropped from **6696s -> 1151s**
(5.8x speedup) with identical accuracy.

### PD Refactor and Hang Fix (PR #21299)

Refactored disaggregation connection layer and fixed hang with
`total_request/total_tokens` balancing.

## Speculative Decoding

- Ngram: removed `max_match_window_size` and `min_match_window_size`,
  matching all suffixes of the Trie (#21225)
- Fix spec v2 + logprob when `max_num_token` is set (#20799)
- Fix draft extend cuda graph when `spec_step=1` (#21709)
- Remove H2D for Qwen3.5 SpecV2 (#20864)
- Switch MooncakeSpec to EAGLE3 + Llama-3.1 (#21794)

## Internal / CI / Cleanup

- Subprocess watchdog logging reduced (#21968)
- `CustomTestCase` for CI retry support (#21650, #21830)
- Redundant PCG tests removed (#21554, #21485)
- Clean up detokenizer and dead multimodal_gen code (#21588)
- Clean up TokenizerManager dead code (#21639)
- `_wait_for_scheduler_ready` cleanup (#21626)
- Removed redundant allreduce fusion block for TP=1 (#20621)
- Removed deprecated `environs` (#21536)
- Removed obsolete sgl-kernel legacy paths (#21528)
- Removed flashinfer wheel cache cleanup that deletes other versions (#21711)
- JPEG input optimization for NVIDIA GPU (#19749)
- CUDA IPC caching for multimodal transfer (#21418)
- Fix CUDA IPC + MM splitting incompatibility (#19915)
- Reduce CPU peak memory in multimodal tensor hashing (#21123)
- VLM: change default mm-attention backend from triton_attn to fa4 on
  Blackwell (#21595)
- VLM: enable per-image MM splitting by default (#21899)
- ShmPointerMMData multi-pickle safety (#21465)
- Fix shared memory race condition in multi-GPU VLM (#21655)
- Added tokens config filter fix (#17905)
- HiMambaTree host lock optimization (#21750)
- HiRadixCache: removed TTL-based hard pin (#21884)
- HiCache: clone host indices to avoid memory leak (#21624)
- HiCache + PD: fixed cache hit breakdown in PD scenarios (#21764)
- Merge prohibition policy during CI maintenance mode (#21882)
- Slack upload timeouts (#21903)
- Trivy scanning (added #21772, CVE skip list #21905)

## Dependency Summary

| Package | v0.5.10rc0 | v0.5.10 |
|---------|------------|---------|
| sglang-kernel | 0.4.0 | 0.4.1 |
| FlashInfer | 0.6.6 | 0.6.7.post2 |
| mooncake | 0.3.10 | 0.3.10.post1 |
| runai-model-streamer | (not present) | >=0.15.7 (new `runai` extra) |
| polars | (not present) | added to `test` extra |

All other dependencies unchanged from rc0 (transformers 5.3.0, xgrammar
0.1.32, Flash Attention 4.x, diffusers 0.37.0).

## Status of Our Known Bugs

| Bug | Doc | Status in v0.5.10rc0 | Status in v0.5.10 |
|-----|-----|---------------------|-------------------|
| reasoning_tokens always 0 | `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md` | NOT FIXED | **FIXED** -- PR #15562 merged 2026-04-04 |
| moe_wna16 qzeros + EP | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | NOT FIXED | NOT FIXED -- vLLM PRs #35598/#36026 still open |
| EPLB + Qwen3 MoE | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | NOT FIXED | NOT FIXED -- PR #21822 still open (2026-04-05) |
| NVFP4 input_scale + EP | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | NOT FIXED | NOT FIXED -- PRs #20869/#21630/#20963 still open |
| ModelOptModelLoader + sharded_state | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | NOT FIXED | NOT FIXED -- PR #21612 still open |
| CutlassMoEParams global num_experts | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | NOT FIXED | NOT FIXED -- unreported |
| sharded_state + speculative | `SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md` | NOT FIXED | NOT FIXED -- unreported |
| MoE Triton tuning text_config | `SGLANG_MOE_TUNE_UPSTREAM_BUG.md` | FIXED (PR #20232) | FIXED |

## EADDRINUSE Sidecar Status

**Still needed in v0.5.10**. No fix for the EADDRINUSE bug in these 250
commits. The Scheduler subprocess still binds `<pod-ip>:<port>` for internal
communication, so uvicorn's `0.0.0.0:<same-port>` still conflicts. The
HAProxy sidecar (`haproxy:lts-alpine`, `0.0.0.0:{{ sglang_port }}` ->
`127.0.0.1:{{ sglang_internal_port }}`) remains in the deployment.

## Upgrade Action Items

1. **Update `sglang_image`** in `roles/k8s_dgx/defaults/main.yml` -- new image
   tag TBD (waiting for `scitrera` build)
2. **Update `SGLANG_EXPECTED_IMAGE`** in `sglang_launch.sh` and
   `sglang_shard_launch.sh` when image is available
3. **Consider `--enable-mfu-metrics`** -- adds FLOPs/bandwidth Prometheus
   counters to our Grafana dashboard (zero overhead when gated off)
4. **`/get_server_info` -> `/server_info`** -- update any external tooling
   that hits the old endpoint (sglang-test CLI, monitoring scripts)
5. **HAProxy sidecar still needed** -- EADDRINUSE NOT fixed
6. **Remaining monkey-patches** -- all still needed except MoE tuning
   (`sglang_tune_moe.sh` auto-skips since rc0):
   - moe_wna16 qzeros EP patch
   - CutlassMoEParams num_experts patch
   - modelopt_quant NVFP4 input_scale patch (if using NVFP4 + EP)
   - sharded_state + speculative workaround (CLI flags)
7. **`SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md`** -- can be marked as FIXED;
   verify `reasoning_tokens` is populated after image upgrade
8. **Verify multi-tool streaming** -- if using tool calling via OpenWebUI,
   test that multi-tool responses are now fully returned (#20004)
9. **Test `parallel_tool_calls`** parameter if tool calling is used (#20208)
10. **transformers 5.3.0 tokenizer** -- same as rc0, already validated
