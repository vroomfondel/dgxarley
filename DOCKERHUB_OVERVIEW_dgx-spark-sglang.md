<!-- short: SGLang for NVIDIA DGX Spark / GB10 (SM121): sm_121-pruned kernels, NVFP4, DSv4/GLM MTP, arm64 -->

# dgx-spark-sglang

Custom [SGLang](https://github.com/sgl-project/sglang) container images for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

Upstream SGLang / sgl-kernel binaries do not target `sm_121` and silently fall
back to JIT or to kernels that never run on a GB10. These images carry a stack
of patches against `sgl-kernel` that arch-prune the build to `sm_121` and strip
the Hopper-only kernels (FA3, sm90 targets, FlashMLA) it otherwise tries to
compile, plus per-release source patches for whatever the current SGLang tag
still gets wrong on this arch.

The original headline fix, the CUTLASS NVFP4 blockwise-MoE patch for the GB10's
101 KB shared-memory budget (device-side assert at
`nvfp4_blockwise_moe.cuh:78`), is **history as of `0.5.16-sm121`**: SGLang
[PR #30448](https://github.com/sgl-project/sglang/pull/30448) deleted that
kernel together with `cutlass_moe_fp4()` and now dispatches NVFP4 MoE to
FlashInfer-CUTLASS on SM100 and SM120/121 alike, which is the backend the model
profiles pin anyway. So the patch was not merged, it became unreachable, and it
is gated off (`APPLY_SGL_KERNEL_SM121=0`) on `0.5.16-sm121` and later. It is
still applied on `0.5.15.post1-sm121` and below.

For the DeepSeek-V4-Flash path the older tags also carry the then-unmerged
DeepSeek-V4 NVFP4 MoE support (upstream
[PR #25820](https://github.com/sgl-project/sglang/pull/25820), rebased onto
v0.5.13) so the `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint can be served on
SM121 at all; that one landed upstream and is native from v0.5.14 on. `sm_121`
remains a compute-capability tier the DeepGEMM / FlashMLA / cutlass kernel
ecosystem is only slowly shipping kernels for.

- **Source**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: Apache-2.0 (same as SGLang)

## What's inside

- **SGLang** built from upstream tags (currently `v0.5.18`)
- **sgl-kernel** with SM121 build patches: arch-prune to `sm_121` only, FA3 /
  sm90 targets / FlashMLA stripped (the bundled FlashMLA is Hopper-only). From
  `0.5.17-sm121` on all four are repathed, because SGLang RFC #29630 relocated
  the whole top-level `sgl-kernel/` tree to `python/sglang/kernels/aot/`, and
  `0.5.18-sm121` carries its own patch variant again: upstream reshaped the
  Blackwell gencode block in `aot/CMakeLists.txt` (the explicit `sm_103a`
  gencode is gone, replaced by `sm_100f` on CUDA 12.9+ with `sm_100a` as the
  pre-12.9 fallback), so the v0.5.17 arch-prune hunk no longer applies. The
  pruned result is unchanged: `sm_121a` on aarch64 plus `--compress-mode=size`.
  The CUTLASS NVFP4 blockwise-MoE patch (`StageCount<1>` +
  `KernelPtrArrayTmaWarpSpecialized`) is present only on `0.5.15.post1-sm121`
  and below, see above.
- **DeepSeek-V4 EAGLE-MTP marlin + TileLang indexer compat**, still patched on
  the current tags: `HybridFp8NvFp4Config.get_quant_method` has no marlin branch
  for the MTP/nextn draft-MoE layers, and TileLang's uint8 arg check still bites
  on the DSA indexer kernel.
- **DeepSeek-V4-Flash NVFP4 MoE (sm_121, experimental)** — the `0.5.13-sm121`
  tag carries upstream [PR #25820](https://github.com/sgl-project/sglang/pull/25820)
  ("DeepSeek-V4 NVFP4 MoE", unmerged at build time) rebased onto v0.5.13, so the
  `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint can be served on GB10. PR #25820
  is upstream-validated only on B200 (SM100) and default-routes to
  `flashinfer_trtllm_routed`, which is not runnable on SM121
  ([#26324](https://github.com/sgl-project/sglang/issues/26324)) — the model
  profile pins `flashinfer_cutlass` explicitly. **First-contact / unvalidated.**
  V4's sparse-decode path itself no longer needs a vendored kernel as of
  v0.5.13: upstream [PR #24692](https://github.com/sgl-project/sglang/pull/24692)
  ships a native SM120/121 Triton path (`major==12`, covers GB10), so the
  sm_121a-retargeted [`0xSero/deepseek-v4-flash-sm120`](https://github.com/0xSero/deepseek-v4-flash-sm120)
  kernel bake that earlier tags (`0.5.12.post1-sm121` and below) used is
  **dropped** here. Full wall-by-wall breakdown — DeepGEMM `hc_prenorm` +
  `paged_mqa_logits` torch fallbacks, `wo_a` fp8→bf16
  (`SGLANG_OPT_FP8_WO_A_GEMM=0`), `mem_fraction_static`, node swap for the load
  peak — in
  [`UPSTREAM_DSV4_BUGS.md`](https://github.com/vroomfondel/dgxarley/blob/main/UPSTREAM_DSV4_BUGS.md).
- **flashinfer pinned to `0.6.17`** (as of `0.5.18-sm121` this is exactly
  upstream's own pyproject pin, so for the first time in this line the two
  agree; it was a deliberate bump over the v0.5.17 tag's `0.6.15.post1`, and
  `0.6.18` still only exists as an rc), deliberately paired with
  **nvidia-cutlass-dsl `4.6.2`** (`4.6.1` up to `0.5.17-sm121`; the bump follows
  upstream and its release notes credit it with fixing an FA4 startup
  regression on Blackwell). flashinfer only
  constrains cutlass-dsl `>=4.5.0` (open-ended), and NVIDIA has shipped
  internally-inconsistent `4.5.2`/`4.5.3` wheels that ICE on every fresh
  CuTe-DSL JIT compile at CUDA-graph warmup, so the explicit cutlass pin is not
  optional. What this pin buys on GB10:
  - [PR #3932](https://github.com/flashinfer-ai/flashinfer/pull/3932) fixes the
    **b12x FP4 quantization numerics** and adds `input_global_scale` to
    decouple weight from activation scale. Upstream's own summary is that W4A4
    serving on GB10 / SM120 / SM121 "now delivers the output quality its
    benchmark scores imply", with two NVFP4 quantization bugs fixed. This is
    the main reason for the `0.6.17` pin.
  - SM12x fused-MoE and FP4 GEMM resync:
    [#4253](https://github.com/flashinfer-ai/flashinfer/pull/4253) (`mm_fp4`
    SM120 NVFP4 dense GEMM to b12x HEAD),
    [#4130](https://github.com/flashinfer-ai/flashinfer/pull/4130) (cute SM120
    fused MoE, SwiGLU).
  - [#3897](https://github.com/flashinfer-ai/flashinfer/pull/3897) NVFP4
    attention enabled for SM121 and
    [#3960](https://github.com/flashinfer-ai/flashinfer/pull/3960) GDN CuteDSL
    as `sm_121a`, both inherited from the `0.6.16` pin, plus
    [#4117](https://github.com/flashinfer-ai/flashinfer/pull/4117) GDN WY
    decode on SM121 and
    [#3903](https://github.com/flashinfer-ai/flashinfer/pull/3903)
    `trtllm_allreduce` extended to SM12x.
  - Earlier in this line,
    [PR #3576](https://github.com/flashinfer-ai/flashinfer/pull/3576) added the
    `head_dim=512` dispatch for SM120/121. **Caveat for Gemma-4:** SGLang's own
    attention-backend allowlist hard-rejects `flashinfer` for the Gemma-4
    architecture (only `trtllm_mha | triton | ascend | intel_xpu` are
    accepted), so that fix stays moot for Gemma *attention*. The Gemma-4
    profiles still set `attention_backend=triton`, which is a SGLang allowlist
    constraint and not a flashinfer-version limitation.

  Roll back with `FLASHINFER_VERSION=0.6.16.post3` (what the first
  `0.5.17-sm121` build shipped), then `0.6.16` (the `0.5.16-sm121` pin), then
  `0.6.15.post1` (the v0.5.17 upstream pin, live-validated on
  `0.5.15.post1-sm121`).
- **transformers pinned to `5.12.1`** (exactly SGLang v0.5.17's pyproject
  pin, unchanged since v0.5.15; was `5.8.1` on v0.5.13/v0.5.14), required for
  the Gemma-4 `*-assistant` drafter checkpoints used by NEXTN/MTP speculative
  decoding (`google/gemma-4-{26B-A4B,31B}-it-assistant`).
  Earlier transformers releases don't know the drafter's config subclass
  and the SGLang head exits with `Unrecognized configuration class` during
  drafter weight-loading. **Exception:** the `0.5.14-gemmadiffusion-sm121`
  image pins `5.11.0` instead — `diffusion_gemma` is an unregistered
  `model_type` before then (AutoConfig `KeyError`), and 5.11.0 is the
  version DiffusionGemma's upstream PR #28054 pins.
- **Gemma-4 MTP (Frozen-KV) speculative-decoding patch** — the
  `0.5.11-gemma4-sm121` tag carries a cherry-pick of upstream
  [PR #24436](https://github.com/sgl-project/sglang/pull/24436)
  ("Gemma 4 — Adding MTP support", merged 2026-05-07, after the v0.5.11
  release tag). **Native in v0.5.12+** — the `0.5.12*` tags ship it from
  upstream and the cherry-pick is no longer applied. Adds the dedicated `Gemma4AssistantForCausalLM` model and a
  new `FROZEN_KV_MTP` speculative algorithm (recurrent hidden-state draft
  loop with frozen target KV cache). At runtime SGLang auto-promotes
  `--speculative-algorithm NEXTN → FROZEN_KV_MTP` once the drafter is
  detected. Without this patch the stock NEXTN/EAGLE worker crashes with
  `ValueError: No module or parameter named 'model.language_model' in
  TransformersMultiModalForCausalLM` during drafter weight-load.
  Verified working on the 4-node DGX Spark cluster — see the 31B-it
  TESTLOG, [Test 07 (`num_steps=2`, `num_draft_tokens=3`)](https://github.com/vroomfondel/dgxarley/blob/main/TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-31b-it/TESTLOG_nv580.142_sglang-0.5.11_gemma-4-31b-it_4n.md#mtp-sweep-tests-711--partial-15-cases-done):
  **+98 % at n=1** (10.49 → 20.83 tok/s), **+76 % at n=4** (44.06 → 77.67
  tok/s), drafter acceptance rate median ~0.68, 5/5 requests stopped on
  natural EOS. The 26B-A4B MoE sibling's MTP sweep is still in progress
  ([TESTLOG](https://github.com/vroomfondel/dgxarley/blob/main/TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/TESTLOG_nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it_4n.md)).
- **NemotronH MTP + radix cache (experimental)** — the
  `0.5.13-dev-nemotronh-mtp-sm121` tag carries upstream
  [PR #27998](https://github.com/sgl-project/sglang/pull/27998) (unmerged),
  which enables native MTP speculative decoding for
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` **without**
  `--disable-radix-cache`. It differs from the production `0.5.13-sm121` tag
  by exactly this one Python-only source patch so the MTP behaviour can be
  A/B'd against production. **First-contact / unvalidated** — validate
  `accept_len > 1` and no NaN logits (an NVFP4-MTP risk,
  [#27828](https://github.com/sgl-project/sglang/issues/27828)) on first boot.
- **DiffusionGemma dLLM (experimental, main-ahead)** — the
  `0.5.14-gemmadiffusion-sm121` tag is **the unified Gemma-4 image**: one
  build that serves all five Gemma-4 profiles — BF16 + FROZEN_KV_MTP,
  NVFP4, and the diffusion-LLM `nvidia/diffusiongemma-26B-A4B-it-NVFP4`. It
  is pinned forward to SGLang `main @ 3a1417a` (post-v0.5.13, 2026-06-12)
  rather than the v0.5.13 tag because (1)
  [PR #28081](https://github.com/sgl-project/sglang/pull/28081) fixes a
  short-lived broken `FrozenKVMTPCudaGraphRunner` that crashes Gemma-4 BF16
  MTP at boot on the v0.5.13 tag, and (2) the DiffusionGemma bake
  ([PR #28054](https://github.com/sgl-project/sglang/pull/28054), unmerged)
  applies far more cleanly main-vs-main. Carries the Gemma-4 GeGLU/FP4
  NaN-clamp source patch ([PR #22928](https://github.com/sgl-project/sglang/pull/22928))
  and uses the `-mainahead` sgl-kernel patch variants (one day of main drift
  shifted the mscclpp link lines). **First-contact / main-ahead, not a
  tagged release.**
- **Qwen4-Exp / Qwen3.8-Flash-Next (`0.5.18-sm121`)**: upstream
  [PR #36497](https://github.com/sgl-project/sglang/pull/36497) (still open,
  not in v0.5.18 and not in `main`) applied to the source before install. It is
  the only implementation of `Qwen4ExpForConditionalGeneration` / `model_type
  qwen4_exp` anywhere; without it the image refuses
  `RadixArk/Qwen3.8-Flash-Next-NVFP4` at load with *"has no SGLang
  implementation"*. The build is only half of it: the two GB10/SM121 QSA
  decode fixes are deliberately **runtime** patches in the Ansible role, one of
  them a defensive veto against an upstream gate expression that was respelled
  three times in three days, one spelling of which silently corrupts
  long-context output on GB10. An image built from the source patch alone
  crashes at backend init on a Spark.
- **Nemotron-3.5-Lightning speculative decoding (`0.5.18-sm121`)**: upstream
  [PR #36186](https://github.com/sgl-project/sglang/pull/36186) (merged
  2026-08-25, three days after the v0.5.18 tag) backported together with its
  two DFlash2 prerequisites,
  [#35371](https://github.com/sgl-project/sglang/pull/35371) and
  [#35496](https://github.com/sgl-project/sglang/pull/35496). Enables all three
  published drafters for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
  (in-checkpoint MTP plus the external `-DFlash` / `-DSpark` drafts, the latter
  being NVIDIA's own DGX Spark recommendation). Stock v0.5.18 aborts DSPARK /
  DFLASH at backend setup with *"implements neither
  set_dspark_layers_to_capture nor set_dflash_layers_to_capture"*. Serving the
  target **unspeculated** does not need this patch. It expires on the next
  SGLang tag, which will contain the merged code.
- Built on a CUDA 13.2 + PyTorch 2.13 + NCCL 2.30.7 base for the GB10 codegen
  path (CUDA 13.1 / PyTorch 2.10 fallback is ~45 % slower end-to-end). **Known
  issue:** the NCCL 2.30.x NVLS path has a regression that can silently hang
  high-expert-count MoE weight loads (≥256 experts) on GB10/RoCE
  ([NVIDIA/nccl#2167](https://github.com/NVIDIA/nccl/issues/2167)) — set
  `NCCL_NVLS_ENABLE=0` when running these (free on non-NVLink hardware; the
  Ansible role does this for you)

## Tags

| Tag                                 | Notes                                                                       |
|-------------------------------------|------------------------------------------------------------------------------|
| `0.5.18-sm121`                      | SGLang v0.5.18 + SM121 patches (own arch-prune variant: upstream reshaped the Blackwell gencode block); adds two source backports absent from the tag: Qwen4-Exp / `qwen4_exp` (open PR #36497, the only implementation of that architecture) and Nemotron-3.5-Lightning speculative decoding (merged PR #36186 + DFlash2 prerequisites #35371/#35496); DSV4 EAGLE-MTP marlin + TileLang remainder still patched; PyTorch 2.13 base, flashinfer 0.6.17 + cutlass-dsl 4.6.2 + transformers 5.12.1. **(current)** |
| `0.5.17-sm121`                      | SGLang v0.5.17 + SM121 patches, repathed for the RFC #29630 `sgl-kernel` → `python/sglang/kernels/aot` relocation; CUTLASS NVFP4 SM121 patch gated off (PR #30448); DSV4 EAGLE-MTP marlin + TileLang remainder still patched; PyTorch 2.12 base, flashinfer 0.6.17 + cutlass-dsl 4.6.1 + transformers 5.12.1. Rollback / A/B |
| `0.5.16-sm121`                      | SGLang v0.5.16 + SM121 patches; first tag where the CUTLASS NVFP4 SM121 patch is gated off (PR #30448 deleted its target); flashinfer 0.6.16 + cutlass-dsl 4.6.1. Rollback / A/B |
| `0.5.16-dev-sm121`                  | Same recipe line as `0.5.16-sm121` but built with flashinfer **0.6.16rc3**. Kept frozen as the measurement basis cited by the NVFP4-KV / uniform-q-len findings in the repo, do not expect it to be rebuilt |
| `0.5.15.post1-sm121`                | SGLang v0.5.15.post1 (source-only bugfix release, "mostly for GLM 5.2"); last tag that still carries the CUTLASS NVFP4 SM121 kernel patch; flashinfer 0.6.15.post1 + cutlass-dsl 4.6.1 |
| `0.5.15-sm121`                      | SGLang v0.5.15 + SM121 patches; NVFP4-MoE dispatch (PR #25820) **and** Qwen3.6 ModelOpt mixed / W4A16_NVFP4 (PR #27906) now **native**; GB10-only DSV4 EAGLE-MTP marlin + TileLang 0.1.8 remainder still patched; flashinfer 0.6.14 + cutlass-dsl 4.6.0 + transformers 5.12.1 |
| `0.5.14-sm121`                      | SGLang v0.5.14 + SM121 patches; native NVFP4-MoE dispatch (PR #25820), native MTP for Nemotron-3 Super 120B; Qwen3.6 W4A16_NVFP4 still **patched** (PR #27906, then unmerged); flashinfer 0.6.14. Rollback / A/B |
| `0.5.13-sm121`                      | SGLang v0.5.13 + SM121 patches + DeepSeek-V4 NVFP4 MoE (PR #25820); native SM120/121 FlashMLA (PR #24692), no vendored kernel; flashinfer 0.6.13rc2 |
| `0.5.14-gemmadiffusion-sm121`       | **Unified Gemma-4 image** — main-ahead (`3a1417a`, post-v0.5.13) serving all five Gemma-4 profiles incl. DiffusionGemma dLLM (PR #28054) + FROZEN_KV_MTP fix (PR #28081). First-contact |
| `0.5.13-gemmadiffusion-sm121`       | Gemma-4 diffusion build on v0.5.13 base (2026-06-19); rollback / A/B against `0.5.14-gemmadiffusion-sm121` |
| `0.5.13-dev-nemotronh-mtp-sm121`    | v0.5.13 + SM121 patches + NemotronH MTP/radix-cache (unmerged PR #27998); A/B against `0.5.13-sm121`. Experimental |
| `0.5.13-gemma4-sm121`               | v0.5.13 + SM121 patches + Gemma-4 NVFP4 source patch (PR #22928); for NVFP4 Gemma-4 on flashinfer 0.6.13rc2 (DSV4 deliberately omitted — mutually exclusive) |
| `0.5.12.post1-sm121`                | SGLang v0.5.12.post1 + SM121 patches + vendored sm_121 DeepSeek-V4-Flash FlashMLA kernel (previous, kept for rollback / A/B) |
| `0.5.12-sm121`                      | SGLang v0.5.12 + SM121 patches                                              |
| `0.5.12-gemma4-sm121`               | v0.5.12 + Gemma-4 NVFP4 source patches                                       |
| `0.5.11-sm121`                      | SGLang v0.5.11 + SM121 patches (kept for rollback / A/B)                     |
| `0.5.11-gemma4-sm121`               | v0.5.11 + unmerged Gemma-4 NVFP4 source patches (PRs #22929, #22928) + MTP cherry-pick |
| `0.5.10-20260429-sm121-dev1`        | Legacy v0.5.10 line, kept for rollback / A/B                                 |
| `0.5.10-20260429-gemma4-sm121-dev1` | Legacy v0.5.10 + Gemma-4 patches                                             |

All tags are **`linux/arm64` only** — these images are useless on x86_64 and on
non-GB10 NVIDIA hardware (the kernels are arch-pruned to `sm_121`).

## Build & deploy context

Everything that produces these images — Dockerfiles, sgl-kernel patches, recipe
files, the cross-arch podman build driver, and the Ansible roles that deploy
SGLang on a 4-node DGX Spark K3s cluster — lives in
[github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley).

Relevant entry points:

- [`scripts/build_sm121_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_sm121_image.sh)
  — remote-podman build driver (x86 control host → arm64 build runner)
- [`scripts/patches/sglang-0.5.18-sm121.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-0.5.18-sm121.recipe)
  — recipe pinned by the build (SGLang + flashinfer + cutlass-dsl + transformers
  pins, plus the per-release patch gates). One recipe per tag, they are kept
  rather than edited in place
- [`scripts/patches/sgl-kernel-sm121.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sgl-kernel-sm121.patch)
  — the core CUTLASS NVFP4 SM121 fix, applied up to `0.5.15.post1-sm121` and
  gated off since (`APPLY_SGL_KERNEL_SM121=0`)
- [`scripts/verify_sglang_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/verify_sglang_image.sh)
  — patch-set acceptance gate; the build driver runs it against the built image
  automatically (opt out with `--no-verify`), and no tag is pushed without it
- [`scripts/patches/sglang-dsv4-nvfp4-pr25820.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-dsv4-nvfp4-pr25820.patch)
  — DeepSeek-V4 NVFP4 MoE support (upstream PR #25820, rebased onto v0.5.13)
- `CUTLASS_NVFP4_SM121_PRD.md` — NVFP4 root cause + fix rationale (in repo)
- `UPSTREAM_DSV4_BUGS.md` — DeepSeek-V4-Flash sm_121 boot chain, wall by wall (in repo)
- [`roles/k8s_dgx/`](https://github.com/vroomfondel/dgxarley/tree/main/roles/k8s_dgx)
  — Ansible role that deploys SGLang head + workers (Multus + RoCE-over-SR-IOV
  NCCL, HAProxy sidecar for the head's EADDRINUSE workaround, model profiles)

## Status / support

These images are built and exercised on a private 4-node DGX Spark cluster
(`spark1`–`spark4`). They are published in case someone else has the same
hardware and runs into the same SM121 crashes — there is no commercial support
and tags may be retagged or removed without notice. Open an issue on the
GitHub repo if something is broken.
