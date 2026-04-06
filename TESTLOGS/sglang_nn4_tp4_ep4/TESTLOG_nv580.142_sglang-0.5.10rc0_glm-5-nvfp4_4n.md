# SGLang Test Log — GLM-5 NVFP4, 4 Nodes, v0.5.10rc0

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver | 580.142                                            |
| CUDA | 13.0                                               |
| Kernel | 6.17.0-1014-nvidia                                 |
| OS | Ubuntu 24.04.4 LTS (aarch64)                       |
| K3s | v1.35.3+k3s1                                       |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image | `scitrera/dgx-spark-sglang:0.5.10rc0` |
| Model | `nvidia/GLM-5-NVFP4`                         |

## Result: Does NOT fit on 4× DGX Spark (4× 128 GB)

GLM-5-NVFP4 (744B/40B-active) is too large for 4 nodes with 128 GB unified memory each.

**Tested configurations:**

- **EP=4, TP=4, PP=1** (`mem_fraction_static=0.80`): OOM during weight init.
  `torch.OutOfMemoryError` at `modelopt_quant.py:create_weights` — ~108 GB used,
  tried to allocate 384 MiB with only 755 MiB free (of 121 GB usable CUDA).
  Even with `mem_fraction_static=0.40`: same OOM — the model weights alone exceed
  available CUDA memory per GPU. All 78 layers' attention weights (BF16, not quantized)
  are on every GPU (TP-sharded but not PP-split), consuming the bulk of memory.

- **PP=4, TP=1, EP=1** (`mem_fraction_static=0.80`): OOM on node4.
  Each GPU gets ~20 layers with all 256 experts, but the per-layer weight footprint
  (256 experts × MoE FFN at FP4 + full attention at BF16) still exceeds ~121 GB.

**Why:** NVFP4 only quantizes MoE FFN weights; attention projections (q/k/v/o with
MLA kv_lora_rank=512, q_lora_rank=2048), DSA indexer, lm_head, and MTP layer remain
in BF16. The BF16 attention is the dominant memory consumer. Model card recommends
TP=8 on B300 (8 GPUs). `cpu_offload_gb` does not help on GB10 unified memory
(same physical RAM, CUDA and CPU allocators compete for the same pool).

**Conclusion:** GLM-5-NVFP4 requires more than 4× 128 GB. Test matrix abandoned.

---

## Configuration Matrix (abandoned)

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | OOM (not run) | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | OOM (not run) | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | OOM (not run) | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | OOM (not run) | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | OOM (not run) | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | OOM (not run) | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | OOM (not run) | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | OOM (not run) | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | OOM (not run) | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | OOM (not run) | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | OOM (not run) | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | OOM (not run) | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | OOM (not run) | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | OOM (not run) | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | OOM (not run) | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl_transport | `sglang_nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via IBext) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`flashinfer` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn; valid choices: auto, flashinfer_cudnn, flashinfer_cutlass, flashinfer_trtllm) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous). Irrelevant with PP=1 (TP-only), always 0 here. |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| 1∥ tok/s | Throughput with 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |
