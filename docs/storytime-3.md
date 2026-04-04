# GLM-4.7-NVFP4 on 4x DGX Spark — First Tokens

April 4, 2026. After a full day of systematic kernel testing across two SGLang image versions (v0.5.9-dev2 and v0.5.10rc0), 36-configuration test matrices, and dozens of startup crashes, GLM-4.7-NVFP4 finally produced its first tokens on the 4-node DGX Spark cluster.

---

## The Problem

NVIDIA's GLM-4.7-NVFP4 is a 358B-parameter MoE model (160 experts, top-8, sigmoid routing) quantized to FP4 via ModelOpt. The model card recommends 8x B200 GPUs (datacenter SM100) — we have 4x DGX Spark (consumer GB10, SM121, ARM64). Different silicon, different ISA, different kernel compatibility.

The TP=4/EP=4 topology (the natural fit for 4 GPUs) was systematically tested across all three MoE runner backends (`triton`, `cutlass`, `flashinfer_cutlass`) with every combination of attention backend, FP4 GEMM backend, and CUDA graph settings. **Every single configuration failed:**

- **v0.5.9-dev2**: `cutlass_moe_fp4` kernel crashes with `device-side assert triggered` at `nvfp4_blockwise_moe.cuh:78` — both during CUDA graph capture AND eager-mode inference. All three MoE backends route through this kernel for NVFP4. Total: 0 tokens across all tested configs.

- **v0.5.10rc0**: The device-side assert is gone, but the server starts and returns 0 tokens on every request — the `cutlass_moe_fp4` kernel silently fails during the forward pass. Configs with CUDA graphs crash during graph capture (OOMKilled or startup crash). Total: 0 tokens across 16 tested TP=4/EP=4 configs.

## The Breakthrough: PP=4

Pipeline Parallelism (PP=4, TP=1, EP=1) distributes the 92 transformer layers across 4 nodes sequentially instead of sharding experts across GPUs. Each GPU holds ~23 layers with all 160 experts — no expert parallelism needed, so the broken `cutlass_moe_fp4` EP code path is never hit.

**Config: PP=4, TP=1, EP=1, v0.5.10rc0, triton MoE, flashinfer attention, cuda_graph ON.**

### First successful inference

![GLM-4.7 PP=4 GPU usage and QSFP throughput](../media/Bildschirmfoto_2026-04-04_19-25-52.png)

GPU utilization across all four DGX Sparks (top four panels) alongside real-time QSFP 200 GbE switch throughput on the MikroTik CRS812 (bottom panel) during single-request inference of GLM-4.7-NVFP4. The pipeline-parallel topology distributes layers across nodes — each GPU processes its shard sequentially, visible as alternating activity bursts. Inter-node communication flows over the QSFP mesh at ~350 Mbps per direction.

[Demo video: GLM-4.7 PP=4 inference with GPU and network monitoring](../media/simplescreenrecorder-2026-04-04_19.24.42.mp4)

### Results

| Concurrency | Result | Throughput | TTFT | Tokens | Finish |
|---|---|---|---|---|---|
| n=1 | **success** | **5.64 tok/s** | 1.4s | 2891 (1644 think + 1472 content) | stop |
| n=4 | crash | — | — | — | FlashInfer `merge_state` invalid argument |
| n=8 | crash | — | — | — | same |

Single-request works perfectly. Concurrent requests crash in FlashInfer's `merge_state` cascade kernel — a PP-specific bug where merging attention states from multiple prefill chunks fails on SM121.

## What We Learned

1. **EP is broken for GLM-4.7 on SM121.** The `cutlass_moe_fp4` kernel at `nvfp4_blockwise_moe.cuh:78` fails on SM121 in both graph capture and eager mode. This affects all MoE runner backends (`triton`, `cutlass`, `flashinfer_cutlass`) since they all route through `cutlass_moe_fp4` for NVFP4 quantized models.

2. **PP=4 bypasses the EP bug entirely.** With TP=1/EP=1, each GPU runs all 160 experts locally — the broken EP code path is never invoked.

3. **v0.5.10rc0 fixed the device-side assert** that killed v0.5.9-dev2, but introduced a silent inference failure (0 tokens) on TP=4/EP=4 configs.

4. **FlashInfer cascade merge is broken under PP concurrency on SM121.** Single requests work, parallel requests crash at `merge_state`. Next test: `attention_backend: triton` to bypass FlashInfer cascade entirely.

5. **The scitrera Docker images are essential.** The NVIDIA-recommended `lmsysorg/sglang` image is x86_64/SM100 only — incompatible with DGX Spark's ARM64/SM121 architecture.

6. **5.64 tok/s on a 358B model across 4 consumer GPUs** — not fast, but it works. PP has inherent pipeline bubble overhead; concurrent requests (once the FlashInfer bug is resolved) should improve aggregate throughput significantly.
