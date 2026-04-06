# SGLang Test Log — MiniMax M2.5 NVFP4, 2 Nodes

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2 (1 GPU each) |

---

## Stable configuration

- **Image:** `scitrera/dgx-spark-sglang:0.5.9-dev2-acab24a7-t5`
- **Model:** `nvidia/MiniMax-M2.5-NVFP4`
- **Config:** `tp_size=2, pp_size=1, ep_size=2, quantization=modelopt_fp4, moe_runner_backend=flashinfer_cutlass, attention_backend=flashinfer, fp4_gemm_backend=auto`
- **Result:** Working. TP=2 splits KV heads (8/2=4 per GPU), EP=2 distributes 256 experts (128 per GPU). ~70 GB weights/GPU, ~58 GB free for KV cache.

This configuration was superseded when spark3 was added — see `TESTLOG_nv580.142_sglang-0.5.9-dev2_minimax-m2.5-nvfp4_3n.md`.
