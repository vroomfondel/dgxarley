# SGLang Test Log — GLM-5 NVFP4, 4 Nodes, v0.5.9-dev2

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
| Image | `scitrera/dgx-spark-sglang:0.5.9-dev2-acab24a7`    |
| Model | `nvidia/GLM-5-NVFP4`                               |

## Result: Does NOT fit on 4x DGX Spark (4x 128 GB)

Same conclusion as the v0.5.10rc0 test — see `TESTLOG_nv580.142_sglang-0.5.10rc0_glm-5-nvfp4_4n.md` for the detailed analysis.

Only 3 configurations were attempted before the test run was canceled:

| # | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | Stability |
|---|------------|-----------|----------|----------------|---------------|-----------|
| 13 | fi_cutlass | flashinfer | fi_cutlass | false | true | **startup_crash** |
| 14 | fi_cutlass | flashinfer | fi_cutlass | true | true | **startup_crash** |
| 15 | fi_cutlass | flashinfer | fi_cutlass | false | false | **deploy_failed** |

### #13 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash — head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-03 16:40:49–16:42:13 UTC
- **Patches:** moe=flashinfer_cutlass, attn=flashinfer, fp4=flashinfer_cutlass, disable_cuda_graph=false, disable_piecewise=true, ep=4, tp=4
- **Pod events:**
  - sglang-head-654cd86f59-6cnr4: +1 restart (total=1)
  - sglang-worker-1-c8fd8566d-5d8st: +1 restart (total=1)
  - sglang-worker-2-54749d87f-lrdkm: +1 restart (total=1)
  - sglang-worker-3-778b45d699-xkptx: +1 restart (total=1)

### #14 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** startup_crash — head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-03 16:42:48–16:44:12 UTC
- **Patches:** moe=flashinfer_cutlass, attn=flashinfer, fp4=flashinfer_cutlass, disable_cuda_graph=true, disable_piecewise=true, ep=4, tp=4
- **Pod events:**
  - sglang-head-654cd86f59-4jcjv: +1 restart (total=1)
  - sglang-worker-1-c8fd8566d-8kpqp: +1 restart (total=1)
  - sglang-worker-2-54749d87f-9n7v4: +1 restart (total=1)
  - sglang-worker-3-778b45d699-jfvt2: +1 restart (total=1)

### #15 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** deploy_failed — Ansible canceled
- **Time:** 2026-04-03 16:44:49–16:45:02 UTC
- **Patches:** moe=flashinfer_cutlass, attn=flashinfer, fp4=flashinfer_cutlass, disable_cuda_graph=false, disable_piecewise=false, ep=4, tp=4
