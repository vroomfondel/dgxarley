<!-- short: NVIDIA ModelOpt NVFP4 PTQ toolchain for DGX Spark (GB10, SM121, arm64): quantize + smoke-serve. -->

# dgx-spark-quant

**NVIDIA ModelOpt PTQ (NVFP4 / `modelopt_fp4`) toolchain**, layered on top of the
[`xomoxcc/dgx-spark-sglang`](https://hub.docker.com/r/xomoxcc/dgx-spark-sglang)
serving image for the **NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

It exists so a model can be **quantized directly ON a DGX Spark** (GB10, one GPU +
128 GB unified memory) instead of on a rented Hopper box — and then **smoke-served
from the same image on the same node**, with no checkpoint scp in between. ModelOpt
PTQ itself needs only torch + CUDA + modelopt (it runs fine on any recent GPU to
*produce* an FP4 checkpoint); the FP4 tensor cores are only needed to *serve* it,
which the shared serving base already does on GB10.

- **Source / build script**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
  (see [`scripts/build_dgx_spark_quant_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_dgx_spark_quant_image.sh) and
  [`scripts/patches/dgx-spark-quant-sm121.Dockerfile`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/dgx-spark-quant-sm121.Dockerfile))
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: same upstream licenses as the SGLang base + NVIDIA TensorRT-ModelOpt

## What's inside

Base = the **full `dgx-spark-sglang:<tag>` serving image, on purpose** — so
quantize *and* smoke-serve share ONE image on the same Spark. The serving base
already ships `nvidia-modelopt 0.45.0`, `torch 2.12.0/cu132`, `transformers 5.12.1`,
`datasets 5.0.0`, `huggingface-hub 1.23.0` (+ the `hf` CLI), `ninja`, `typer`. This
layer therefore adds **only the three genuinely-missing leaves** the quant scripts
need:

- **`accelerate`** — `device_map` weight placement / CPU offload for PTQ
- **`hf_transfer`** — fast model download (`HF_XET_HIGH_PERFORMANCE=1` is set)
- **`py-spy`** — self-contained Rust sampling profiler (no python deps) for
  tracing a running quantize / smoke-serve process

Everything else is reused as-is. To keep the add safe, the serving stack is first
frozen with `pip freeze --all` as a **constraints file**, so the install can only
ADD, never move, anything already pinned (torch / flashinfer / sgl-kernel / …).
This deliberately sidesteps the base's internally-inconsistent
`datasets 5.0.0` ⇄ `fsspec 2026.6.0` pin, which a naïve `pip install datasets`
would trip over (`ResolutionImpossible`).

Build also **asserts `modelopt >= 0.45`** (the release with the mixed-precision
`--recipe` system), **removes `deepspeed`** (its import aborts without a CUDA
toolkit and it is irrelevant to PTQ), and runs a **build-time import smoke** so the
whole quant stack (`torch` / `transformers` / `datasets` / `accelerate` /
`modelopt.torch.quantization`) is proven to import together before publish.

## How to use

Driven by the `quantizer/*.sh` scripts (model-agnostic; a YAML config is the first
arg), typically in three phases inside this container on a Spark:

1. `test_quant_dryrun.sh  configs/<model>.yaml` — mechanical trace/quant/export gate
2. `quantize_modelopt_nvfp4.sh configs/<model>.yaml` — real NVFP4 (W4A4) export
   (`qformat: nvfp4` full W4A4, or `nvfp4_mlp_only` = experts-only, safer quality /
   lower peak memory)
3. `smoke_sglang_spark.sh configs/<model>.yaml` — load + generate the fresh
   checkpoint with SGLang, from this same image

## Tags

| Tag            | Notes                                                                              |
|----------------|------------------------------------------------------------------------------------|
| `0.5.15-sm121` | ModelOpt PTQ toolchain on `dgx-spark-sglang:0.5.15-sm121` base, arm64 (current)    |
| `0.5.14-sm121` | ModelOpt PTQ toolchain on `dgx-spark-sglang:0.5.14-sm121` base, arm64 (rollback)   |

Tag tracks the serving base it layers on — bump both in lockstep. `linux/arm64`
only; the FP4 serving kernels are not useful on non-GB10 hardware.

## Why a separate image

- Keeps the serving image lean — PTQ-only deps (`accelerate`, `hf_transfer`) don't
  belong in every serving pod.
- Guarantees quantize + smoke-serve run against the **exact same** torch / CUDA /
  SGLang stack, so a checkpoint that quantizes clean here also loads clean when
  served — no cross-box ABI drift.
- If you only want to *serve* NVFP4 models, use
  [`xomoxcc/dgx-spark-sglang`](https://hub.docker.com/r/xomoxcc/dgx-spark-sglang)
  directly; this image is only for *producing* the checkpoints.

## Status / support

Built and exercised on a private 4-node DGX Spark cluster. Published in case
someone else has the same hardware. No commercial support; tags may be retagged or
removed without notice. Open an issue on the GitHub repo if something is broken.
