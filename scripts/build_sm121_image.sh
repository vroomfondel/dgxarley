#!/usr/bin/env bash
#
# build_sm121_image.sh — Build xomoxcc/dgx-spark-sglang:*-sm121 images.
#
# Produces a custom SGLang image where cutlass_moe_fp4 (the triton/cutlass
# MoE runner codepath for NVFP4) is patched to fit SM121's 101 KB shared
# memory budget. The upstream scitrera/dgx-spark-sglang:0.5.10 image crashes
# any NVFP4 MoE config that uses moe_runner_backend=triton/cutlass with a
# device-side assert at nvfp4_blockwise_moe.cuh:78. This image fixes that
# by patching run_fp4_blockwise_scaled_group_mm_sm120() to use StageCount<1>
# + KernelPtrArrayTmaWarpSpecialized (non-pingpong).
#
# Full root cause + fix rationale: CUTLASS_NVFP4_SM121_PRD.md
# Reference implementation: TensorRT-LLM PR #12141
# CUTLASS root cause: NVIDIA/cutlass#3144
#
# Workflow (all steps run on the x86 control host)
# -------------------------------------------------
# 1. Preflight: verify patch files + podman + git + patch are available.
# 2. Ensure a registered podman connection to the arm64 build host (spark4)
#    that uses a dedicated unencrypted SSH key (Podman's Go SSH client cannot
#    use ssh-agent or encrypted keys). Create it on demand if missing.
# 3. Clone or update scitrera/cuda-containers locally on x86. Switch to a
#    local 'sm121' branch, hard-reset to origin/main (idempotent), drop in
#    the sgl-kernel patch + Dockerfile patch + recipe file.
# 4. Invoke `podman --connection <name> build` — the build context is
#    streamed from x86 to spark4 over the podman socket, the actual build
#    runs natively on arm64 (no QEMU), and the resulting image is stored
#    in spark4's local podman image store. The x86 host never writes
#    credentials to spark4.
# 5. Streamed `podman image save | load` to pull the built image from spark4 back to x86.
# 6. `podman push` from x86 using the x86 host's pre-existing registry
#    credentials. spark4 never has Docker Hub credentials.
#
# Prerequisites on the x86 control host
# --------------------------------------
# - podman (`apt install podman`)
# - An unencrypted SSH key for podman: generate with
#     ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N ""
#     ssh-copy-id -i ~/.ssh/id_podman root@spark4
#   The key MUST be unencrypted — podman's Go SSH client does not support
#   ssh-agent or encrypted keys. Override via BUILD_SM121_SSH_IDENTITY.
# - `podman login docker.io -u xomoxcc` already done on the x86 host.
# - ~10 GB free disk for the image after the local copy.
# - git, patch (cuda-containers clone + patch apply happens on x86).
#
# Prerequisites on spark4 (the build host)
# ----------------------------------------
# - podman (`apt install podman`)
# - podman.socket enabled as root:
#     systemctl enable --now podman.socket
#   This exposes /run/podman/podman.sock which the x86 client connects to.
# - ~50 GB free disk (sgl-kernel layers + final image in local image store).
# - NO credentials, NO clone, NO patches, NO local scripts. spark4 is a
#   dumb remote build runner.
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

CUDA_CONTAINERS_REPO="https://github.com/scitrera/cuda-containers.git"
# The clone lives on the x86 control host. Overridable via env.
CUDA_CONTAINERS_DIR="${BUILD_SM121_CC_DIR:-${HOME}/pythondev_workspace/cuda-containers}"

# Local-only scratch branch in the cuda-containers clone; never exists
# upstream and is never pushed. Created from origin/main on first run,
# hard-reset to origin/main on every subsequent run (see prepare_cuda_containers)
# so that re-applying the patch stack is idempotent and drift in the working
# tree from previous runs is discarded.
BRANCH_NAME="sm121"
# Recipe variants live in scripts/patches/. They differ only in (a) which
# SGLang source ref they pin and (b) whether the two unmerged Gemma-4-NVFP4
# source patches (PRs #22929/#22928) are also applied — the underlying
# build steps and SM121 sgl-kernel patches are identical.
#
# Current set (v0.5.14 line — DEFAULT):
#   sglang-0.5.14-sm121.recipe         — THE production image. SGLang v0.5.14 +
#                                        SM121 sgl-kernel patches (mainahead) +
#                                        flashinfer 0.6.13 + kernels 0.14.1. DSV4
#                                        NVFP4 MoE (PR #25820) is NATIVE in v0.5.14
#                                        → its patch is OFF (APPLY_DSV4_NVFP4_PR25820
#                                        =0). ONE stacked unmerged model PR remains:
#                                          · Qwen3.6 ModelOpt mixed NVFP4 PR #27906
#                                            (APPLY_QWEN36_MIXED_NVFP4_PR27906=1,
#                                            still OPEN) — modelopt_mixed +
#                                            W4A16_NVFP4 lm_head/MoE/linears +
#                                            wrapper-prefix + MTP. Trailing-context-
#                                            only dockerfile patch.
#                                        Serves nvidia/DeepSeek-V4-Flash/Pro-NVFP4
#                                        (native) + nvidia/Qwen3.6-35B-A3B-NVFP4.
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.14-sm121
#   sglang-0.5.14-gemma4-diffusion-sm121.recipe — THE unified Gemma-4 image, now
#                                        pinned to the v0.5.14 TAG (was main-ahead
#                                        3a1417a): FROZEN_KV_MTP #28081 native +
#                                        DiffusionGemma #28054 (still OPEN, patched)
#                                        + gemma4-NVFP4 patch. Serves ALL five
#                                        Gemma-4 profiles (BF16 MTP, NVFP4, diffusion).
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.14-gemmadiffusion-sm121
#   sglang-0.5.14-gemma4-sm121.recipe  — gemma4-NVFP4 standalone. DO NOT BUILD
#                                        PROACTIVELY (superseded by the unified
#                                        image; NVFP4 Gemma-4 PRs blocked upstream).
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.14-gemma4-sm121
#
# Previous set (v0.5.13 line — kept for rollback):
#   sglang-0.5.13-sm121.recipe         — prior production image. SGLang v0.5.13 +
#                                        DSV4 NVFP4 (PR #25820, patched — unmerged
#                                        at the time) + Qwen3.6 #27906 + flashinfer
#                                        0.6.13 + kernels 0.12.3.
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.13-sm121
#   sglang-0.5.13-gemma4-sm121.recipe  — v0.5.13 gemma4-NVFP4 standalone (same
#                                        upstream-blocked caveat).
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.13-gemma4-sm121
#   sglang-0.5.13-dev-nemotronh-mtp-sm121.recipe — v0.5.13 + PR #27998 (MTP +
#                                        radix cache) experiment. NOT bumped to
#                                        v0.5.14 (MTP enablement #24955 is native
#                                        in v0.5.14; test that first).
#                                        Tag: …:0.5.13-dev-nemotronh-mtp-sm121
#
# Previous set (v0.5.12 line):
#   sglang-0.5.12-sm121.recipe         — SGLang v0.5.12 + six SM121 sgl-kernel
#                                        patches + flashinfer 0.6.11.post1.
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.12-sm121
#   sglang-0.5.12-gemma4-sm121.recipe  — same + Gemma-4 NVFP4 source patches.
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.12-gemma4-sm121
#                                        DO NOT BUILD PROACTIVELY (blocked upstream).
#
# Previous set (v0.5.11 line — kept for rollback / A/B comparison, and as the
# only working build for Gemma-4 BF16 vs the still-blocked NVFP4 variant):
#   sglang-0.5.11-sm121.recipe         — SGLang v0.5.11 + same six SM121
#                                        sgl-kernel patches + flashinfer 0.6.11.
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.11-sm121
#   sglang-0.5.11-gemma4-sm121.recipe  — same + Gemma-4 NVFP4 source patches.
#                                        Tag: xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121
#                                        Production target for Gemma-4 NVFP4
#                                        models until upstream PRs land.
#
# Legacy set (0.5.10-dev1 line) REMOVED 2026-06-19 — both sglang-sm121-dev1.recipe
# and sglang-gemma4-sm121-dev1.recipe deleted (superseded by the 0.5.11+ native
# Gemma-4 path; flashinfer 0.6.8.post1 era). The 0.5.10-20260429-*-sm121-dev1
# images remain on Docker Hub for rollback; recover the recipes from git history.
#
# apply_patches() gates the Gemma-4 source patches and the gemma4 Dockerfile
# patch by `RECIPE_NAME == *gemma4*`. The Gemma-4 MTP cherry-pick (PR #24436)
# is additionally version-gated and SKIPPED on SGLang >= v0.5.12, where
# PR #24436 is merged into the release. The DSV4 NVFP4 patch (PR #25820) is
# gated by the recipe variable APPLY_DSV4_NVFP4_PR25820=1 instead of a name
# pattern — see apply_patches().
#
# RESOLVED (2026-06-29): the DSV4 NVFP4 gate is now OFF in the v0.5.14 base recipe
# — PR #25820 shipped in the v0.5.14 release (2026-06-26), so the new
# sglang-0.5.14-sm121.recipe sets APPLY_DSV4_NVFP4_PR25820=0 (re-applying a merged
# patch fails the in-container dry-run). The rebased sglang-dsv4-nvfp4-pr25820
# .patch stays on disk for the 0.5.13 recipe + git history. Ref: UPSTREAM_DSV4_BUGS.md.
RECIPE_NAME="sglang-0.5.14-sm121"
IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.14-sm121"

# Rollback: previous production line (v0.5.13). DSV4 NVFP4 (#25820) was still
# patched there; it is native in v0.5.14 and the patch is OFF in the 0.5.14 recipe.
#RECIPE_NAME="sglang-0.5.13-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.13-sm121"

# gemma4-NVFP4 standalone — DO NOT BUILD PROACTIVELY (superseded by the unified
# gemma-diffusion image below; SM120/121 Gemma-4 NVFP4 PRs still blocked upstream).
#RECIPE_NAME="sglang-0.5.14-gemma4-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.14-gemma4-sm121"
#RECIPE_NAME="sglang-0.5.13-gemma4-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.13-gemma4-sm121"

# THE UNIFIED GEMMA-4 IMAGE: now pinned to the v0.5.14 TAG (was main-ahead commit
# 3a1417a until 2026-06-29) → FROZEN_KV_MTP fix (#28081) native + DiffusionGemma
# dLLM bake (#28054, still OPEN) + gemma4-NVFP4 patch. Serves ALL five Gemma-4
# profiles (BF16 MTP, NVFP4, diffusion) off one build. See the recipe header for
# the sgl-kernel-patch re-validation caveat. SCOPE: DIFFUSIONGEMMA_SGLANG_SCOPE.md.
#RECIPE_NAME="sglang-0.5.14-gemma4-diffusion-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.14-gemmadiffusion-sm121"

# NemotronH MTP experiment: NOT bumped to v0.5.14 (deferred 2026-06-29). The MTP
# enablement (#24955) is now native in v0.5.14, so test native MTP + radix cache
# on the production 0.5.14-sm121 image FIRST; only re-cut this dev recipe against
# v0.5.14 if the spec-v2 crash (PR #27998, still OPEN) actually reappears.
#RECIPE_NAME="sglang-0.5.13-dev-nemotronh-mtp-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121"

# NB: Qwen3.6 mixed-NVFP4 support (PR #27906, still OPEN) is NOT a separate
# recipe — it is baked into the production sglang-0.5.14-sm121 recipe above
# (APPLY_QWEN36_MIXED_NVFP4_PR27906=1). On v0.5.14 the DSV4 patch is OFF (#25820
# merged), so qwen36 is the only stacked model patch. Target:
# nvidia/Qwen3.6-35B-A3B-NVFP4.

#RECIPE_NAME="sglang-0.5.12-sm121"
## IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.12-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121"
#RECIPE_NAME="sglang-0.5.12-gemma4-sm121"
#IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.12.post1-gemma4-sm121"

# Remote build host (spark4, arm64). Uses a registered podman connection
# with a dedicated unencrypted SSH key. The connection name is derived from
# this value by stripping the user@ prefix so that `podman system connection
# list` shows a clean "spark4" entry.
#
# Both can also be set by flags (--remote-host / --podman-connection); the
# derivation of the connection name from the host happens after argparse
# so that a late --remote-host override propagates.
REMOTE_HOST="${BUILD_SM121_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_SM121_PODMAN_CONNECTION:-}"
PODMAN_SSH_IDENTITY="${BUILD_SM121_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Build-time parallelism. scitrera's Dockerfile.sglang-nightly defaults to
# ARG BUILD_JOBS=2 which uses only 2 of the DGX Spark GB10's 20 ARM cores
# (10%). MAX_JOBS is set from this ARG and propagates to sgl-kernel,
# flashinfer, and any Python extension build that honors it.
#
# GB10 topology: 10 Cortex-X925 + 10 Cortex-A725, 128 GB unified memory
# (of which ~113 GiB is typically free on an otherwise-idle spark host).
#
# EMPIRICAL RESULT (2026-04-10): BUILD_JOBS=16 → OOM-killed during the
# CUTLASS template instantiation phase of sgl-kernel. CUTLASS TU peaks are
# heavier than the "~5 GB per TU" rule of thumb suggests — closer to
# 7-10 GB when multiple heavy TUs hit template expansion simultaneously.
# 16 × ~7 GB = ~112 GB → no margin for buffers/caches → OOM.
#
# 8 is the verified safe default on this hardware. If you want to push:
#   - 10 should still fit (10 × 7 = 70 GB worst case, ~40 GB headroom)
#   - 12 is the theoretical ceiling (12 × 7 = 84 GB, ~30 GB headroom)
#   - 16+ has been confirmed to OOM-kill, don't try again without first
#     reducing per-TU memory pressure (e.g. SGL_KERNEL_COMPILE_THREADS=1
#     in the CMake args — which the Dockerfile already sets for nvcc).
BUILD_JOBS="${BUILD_SM121_BUILD_JOBS:-8}"

PUSH_IMAGE=1

# When set to 1 (via --no-local-copy), skip the ~15 min save|load transfer
# of the built image from the remote build host back to this control host.
# Implies PUSH_IMAGE=0 because `run_push` reads from the local podman store
# which would be empty. Intended use: immediately follow the build with a
# `distrsm121image.sh --source <host>` call that distributes from the
# build host's podman store directly via a throwaway registry:2.
NO_LOCAL_COPY=0

# Base image selection. The recipe ships with a default BASE_IMAGE
# (currently our custom xomoxcc 2.12/cu132 build); --base lets you swap
# it at build time without editing the recipe. Supported aliases:
#
#   xomoxcc   xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132
#             Our locally-built 2.12/cu132 base (scripts/build_pytorch_base_image.sh).
#             Only present on spark4's podman store — never published.
#             This is the recipe default and what you want for performance.
#
#   scitrera  scitrera/dgx-spark-pytorch-dev:2.12.0-v1-cu132
#             scitrera's published upstream base. Pulled from Docker Hub.
#             Bumped 2026-06-19 from 2.10.0-v2-cu131 → 2.12.0-v1-cu132
#             (scitrera shipped 2.12/cu132 on 2026-06-09). Now the SAME
#             torch/cuda as our xomoxcc base, so the old ~45% codegen
#             regression (torch 2.10/cu131 vs 2.12/cu132,
#             reference_sm121_build_base_regression memory) NO LONGER applies.
#             Any residual delta vs xomoxcc would come from our custom build
#             tuning (cuBLAS-Blackwell workspaces, SVE/CMake), NOT measured —
#             xomoxcc stays the tested default; use scitrera for A/B only.
#
# Any other --base VALUE is passed through verbatim as the BASE_IMAGE.
# BUILD_SM121_BASE_IMAGE env var overrides --base for scripting.
BASE_XOMOXCC_IMAGE="xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132"
BASE_SCITRERA_IMAGE="scitrera/dgx-spark-pytorch-dev:2.12.0-v1-cu132"
BASE_IMAGE_ALIAS=""
BASE_IMAGE_OVERRIDE="${BUILD_SM121_BASE_IMAGE:-}"
EFFECTIVE_BASE_IMAGE=""
BASE_IMAGE_SOURCE=""

# sgl-kernel source patch toggles. These map to build-args consumed by the
# patched Dockerfile (APPLY_SGL_KERNEL_ARCH_PRUNE / APPLY_SGL_KERNEL_DISABLE_FA3
# / APPLY_SGL_KERNEL_SKIP_SM90_TARGET) and are applied conditionally inside
# the builder stage BEFORE the sgl-kernel wheel build. The patch FILES
# themselves are always copied into the build context by apply_patches() —
# only the in-container `patch` invocation is gated, which keeps the build
# deterministic regardless of toggle state.
#
# All three default to ON. Disable switches:
#   --no-arch-prune         Keep the upstream arch list. Needed only if
#                           you plan to deploy the wheel on non-GB10
#                           Blackwell / Hopper hardware.
#   --keep-fa3              Keep FlashAttention-3 (Hopper-only) in the
#                           build. Default skips it since is_fa3_supported()
#                           returns False on GB10 anyway and __init__.py
#                           never imports flash_attn unconditionally. See
#                           patch header for the evidence chain.
#   --keep-sm90-target      Keep the common_ops_sm90_build target. Default
#                           skips it since load_utils._load_architecture_
#                           specific_ops() on GB10 always loads from
#                           sgl_kernel/sm100/, never sm90/.
APPLY_ARCH_PRUNE=1
APPLY_DISABLE_FA3=1
APPLY_SKIP_SM90_TARGET=1
APPLY_SKIP_FLASHMLA=1

# sm121-debug: opt-in diagnostic that adds a cudaStreamSynchronize +
# cudaGetLastError check immediately after gemm_op.run() in
# run_fp4_blockwise_scaled_group_mm_sm120(). Gated at runtime by the
# SGL_SM121_DEBUG_CUTLASS env var so it can be toggled without a rebuild,
# but the patch itself needs to be compiled into the image first (via
# --sm121-debug below). Default OFF.
APPLY_SM121_DEBUG=0

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--base xomoxcc|scitrera|<image>]
                        [--remote-host user@host] [--podman-connection NAME]
                        [--no-arch-prune] [--keep-fa3] [--keep-sm90-target]
                        [--keep-flashmla] [--sm121-debug]
                        [--no-local-copy] [--no-push] [--help]

Builds ${IMAGE_TAG} on the remote build host via podman socket, copies
the result back to this host (unless --no-local-copy), and pushes it
from here (unless --no-push or --no-local-copy).

Options:
  --base VALUE Select the PyTorch dev base image this build sits on:
                 xomoxcc   ${BASE_XOMOXCC_IMAGE}
                           (recipe default; locally-built 2.11/cu132; fast path)
                 scitrera  ${BASE_SCITRERA_IMAGE}
                           (upstream published; 2.10/cu131; ~45% slower)
                 <image>   arbitrary image reference, passed verbatim.
               Omitted → recipe default is used.
  --remote-host user@host
               Remote arm64 build host reachable via SSH + podman socket.
               Default: ${REMOTE_HOST}
  --podman-connection NAME
               Registered podman connection name to use (or create). If
               omitted, derived from --remote-host (strip user@ and domain).
  --no-arch-prune
               Skip sgl-kernel-arch-prune.patch. Default is to apply it.
               Disable only if you need a build that runs on non-GB10
               Blackwell / Hopper hardware.
  --keep-fa3   Skip sgl-kernel-disable-fa3.patch. Default IS to apply
               (i.e. FA3 is disabled by default since it is Hopper-only
               and cannot run on GB10). Use this to keep FA3 symbols in
               the build for testing on Hopper GPUs elsewhere.
  --keep-sm90-target
               Skip sgl-kernel-skip-sm90-target.patch. Default IS to apply
               (i.e. common_ops_sm90_build target is dead-coded since
               GB10 always loads sgl_kernel/sm100/common_ops.*). Use this
               if you plan to run the wheel on an actual Hopper GPU.
  --keep-flashmla
               Skip sgl-kernel-skip-flashmla.patch. Default IS to apply
               (i.e. the entire flashmla_ops target is dead-coded — its
               ~25 TUs target sm_90a / sm_100a and none run on GB10).
               Use this if you plan to run the wheel on Hopper/B200
               silicon for MLA-based inference (DeepSeek-V3, Kimi K2,
               etc.) — those models need the sm90 FlashMLA kernels.
  --sm121-debug
               Apply sgl-kernel-sm121-debug.patch on top of the primary
               sm121 patch. Default is NOT to apply. When applied, the
               JIT-compiled sm120 GEMM path gets a post-launch
               cudaStreamSynchronize + cudaGetLastError diagnostic block,
               gated at runtime by the SGL_SM121_DEBUG_CUTLASS env var.
               Set that env var on the sglang pod to turn the diagnostic
               on at runtime without any additional rebuild.
  --no-local-copy
               Skip the 'podman save | podman load' transfer of the built
               image from the remote build host back to this control host.
               Use when you will distribute the image directly via
               scripts/distrsm121image.sh, which runs a temporary registry
               on the build host and lets all K3s nodes pull from there —
               the local copy would be a pure ~15-minute time sink.
               Implies --no-push (you cannot push without a local copy;
               the build host has no Docker Hub credentials by design).
  --no-push    Skip 'podman push' after build + local copy.
  --help       Show this help.

Environment overrides:
  BUILD_SM121_REMOTE_HOST        user@host for spark4 SSH.
                                 Default: ${REMOTE_HOST}
  BUILD_SM121_PODMAN_CONNECTION  Registered podman connection name.
                                 Default: derived from --remote-host
                                 (strip user@ and domain suffix).
  BUILD_SM121_SSH_IDENTITY       Unencrypted SSH private key for podman.
                                 Default: ${PODMAN_SSH_IDENTITY}
  BUILD_SM121_CC_DIR             Local cuda-containers clone path (on x86).
                                 Default: ${CUDA_CONTAINERS_DIR}
  BUILD_SM121_BUILD_JOBS         Parallel compile jobs on the build host
                                 (--build-arg BUILD_JOBS, sets MAX_JOBS env).
                                 Upstream Dockerfile default is 2 — useless
                                 on GB10's 20-core CPU. Push higher if more
                                 memory headroom is available.
                                 Default: ${BUILD_JOBS}
  BUILD_SM121_BASE_IMAGE         Direct BASE_IMAGE override. Wins over --base
                                 and over the recipe default. Use for
                                 scripting when --base aliases are too coarse.

The entire script runs on the x86 control host. spark4 is used purely as
a remote podman build runner — it holds no credentials, no clone, and no
state between runs (except the local podman image store and layer cache,
which persist and accelerate rebuilds).
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push) PUSH_IMAGE=0; shift ;;
        --no-local-copy) NO_LOCAL_COPY=1; PUSH_IMAGE=0; shift ;;
        --base)
            shift
            [[ $# -gt 0 ]] || die "--base requires an argument (xomoxcc|scitrera|<image>)"
            BASE_IMAGE_ALIAS="$1"
            shift
            ;;
        --base=*)
            BASE_IMAGE_ALIAS="${1#--base=}"
            shift
            ;;
        --remote-host)
            shift
            [[ $# -gt 0 ]] || die "--remote-host requires an argument (user@host)"
            REMOTE_HOST="$1"
            shift
            ;;
        --remote-host=*)
            REMOTE_HOST="${1#--remote-host=}"
            shift
            ;;
        --podman-connection)
            shift
            [[ $# -gt 0 ]] || die "--podman-connection requires an argument"
            PODMAN_CONNECTION="$1"
            shift
            ;;
        --podman-connection=*)
            PODMAN_CONNECTION="${1#--podman-connection=}"
            shift
            ;;
        --no-arch-prune)
            APPLY_ARCH_PRUNE=0
            shift
            ;;
        --keep-fa3)
            APPLY_DISABLE_FA3=0
            shift
            ;;
        --keep-sm90-target)
            APPLY_SKIP_SM90_TARGET=0
            shift
            ;;
        --keep-flashmla)
            APPLY_SKIP_FLASHMLA=0
            shift
            ;;
        --sm121-debug)
            APPLY_SM121_DEBUG=1
            shift
            ;;
        --help|-h) usage; exit 0 ;;
        *)         die "Unknown argument: $1 (use --help)" ;;
    esac
done

# Derive the podman connection name from REMOTE_HOST if the user didn't
# set it explicitly. "root@spark4.local" -> "spark4".
if [[ -z "${PODMAN_CONNECTION}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    # Shorten a DNS name to its first label (spark4.local -> spark4), but keep an
    # IPv4 address whole ("192.168.0.5" must NOT collapse to "192").
    if [[ ! "${PODMAN_CONNECTION}" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
    fi
fi

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    local required_files=(
        sgl-kernel-sm121.patch
        sgl-kernel-sm121-debug.patch
        sgl-kernel-arch-prune.patch
        sgl-kernel-disable-fa3.patch
        sgl-kernel-skip-sm90-target.patch
        sgl-kernel-skip-flashmla.patch
        dockerfile-sm121.patch
        dockerfile-dsv4-flashmla.patch
        build-image-sh-podman.patch
        "${RECIPE_NAME}.recipe"
    )
    # Gemma-4 MTP cherry-pick is only required when the recipe pins SGLang
    # < v0.5.12 (where PR #24436 is not yet upstream). The full SGLANG_VERSION
    # comparison happens in apply_patches(); preflight is content with the
    # coarser RECIPE_NAME pattern (any *0.5.12*, *0.5.13*, ... is post-PR).
    local _recipe_sver=""
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]]; then
        _recipe_sver="$(grep -E '^SGLANG_VERSION=' "${PATCHES_DIR}/${RECIPE_NAME}.recipe" \
            | head -1 | cut -d= -f2- | tr -d '"' || true)"
    fi
    if [[ -z "${_recipe_sver}" ]] \
        || ! printf '0.5.12\n%s\n' "${_recipe_sver}" | sort -V -C 2>/dev/null; then
        required_files+=(
            dockerfile-gemma4-mtp.patch
            sglang-gemma4-mtp-pr24436.patch
        )
    fi
    # Gemma-4 NVFP4 patches are only required when building the gemma4 recipe variant.
    # See apply_patches() for the matching gating logic.
    if [[ "${RECIPE_NAME}" == *gemma4* ]]; then
        required_files+=(
            dockerfile-gemma4-nvfp4.patch
            sglang-gemma4-geglu-nan-clamp.patch
        )
    fi
    # DSV4 NVFP4 patches (PR #25820) + TileLang 0.1.8 compat are only required
    # when the recipe opts in via APPLY_DSV4_NVFP4_PR25820=1. See apply_patches()
    # for the matching gate. The TileLang patch is folded into this gate because
    # dockerfile-dsv4-nvfp4.patch now adds both COPY+RUN steps.
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_DSV4_NVFP4_PR25820=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        required_files+=(
            dockerfile-dsv4-nvfp4.patch
            sglang-dsv4-nvfp4-pr25820.patch
            sglang-tilelang-018-indexer-compat.patch
        )
    fi
    # DSV4 EAGLE-MTP marlin branch + TileLang 0.1.8 compat — the v0.5.14+
    # remainder of the above once PR #25820's base NVFP4-MoE support went
    # native (APPLY_DSV4_NVFP4_PR25820=0). Gated by its own recipe variable
    # so the two states (pre-merge cherry-pick vs. post-merge remainder) never
    # collide. See apply_patches() for the matching gate.
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_DSV4_MTP_MARLIN_TILELANG=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        required_files+=(
            dockerfile-dsv4-mtp-marlin-tilelang.patch
            sglang-dsv4-mtp-marlin-v0.5.14.patch
            sglang-tilelang-018-indexer-compat.patch
        )
    fi
    # DiffusionGemma patches (PR #28054) — only when the recipe opts in via
    # APPLY_DIFFUSIONGEMMA_PR28054=1. See apply_patches() for the matching gate.
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_DIFFUSIONGEMMA_PR28054=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        required_files+=(
            dockerfile-diffusiongemma.patch
            sglang-diffusiongemma-pr28054.patch
        )
    fi
    # NemotronH MTP patches (PR #27998) are only required when the recipe opts
    # in via APPLY_NEMOTRONH_MTP_PR27998=1. See apply_patches() for the gate.
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_NEMOTRONH_MTP_PR27998=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        required_files+=(
            dockerfile-nemotronh-mtp.patch
            sglang-nemotronh-mtp-pr27998.patch
        )
    fi
    # Qwen3.6 mixed NVFP4 patches (PR #27906) are only required when the recipe
    # opts in via APPLY_QWEN36_MIXED_NVFP4_PR27906=1. See apply_patches() for the gate.
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_QWEN36_MIXED_NVFP4_PR27906=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        required_files+=(
            dockerfile-qwen36-mixed-nvfp4.patch
            sglang-qwen36-mixed-nvfp4-pr27906.patch
        )
    fi

    local missing=0
    for f in "${required_files[@]}"; do
        if [[ ! -f "${PATCHES_DIR}/${f}" ]]; then
            warn "Missing patch file: ${PATCHES_DIR}/${f}"
            missing=1
        fi
    done
    (( missing == 0 )) || die "Required patch files missing (see warnings above)"

    for tool in git patch podman; do
        command -v "${tool}" >/dev/null || die "Required tool not found: ${tool}"
    done

    if [[ ! -f "${PODMAN_SSH_IDENTITY}" ]]; then
        cat >&2 <<EOF

ERROR: SSH identity '${PODMAN_SSH_IDENTITY}' not found.

Podman's Go SSH client cannot use the ssh-agent or encrypted keys, so a
dedicated unencrypted key is required. Create it with:

  ssh-keygen -t ed25519 -f ${PODMAN_SSH_IDENTITY} -N ""
  ssh-copy-id -i ${PODMAN_SSH_IDENTITY} ${REMOTE_HOST}

Then re-run this script.
EOF
        exit 1
    fi

    echo "Patch files present, tools available, SSH identity found"
}

# ============================================================================
# Podman connection to the remote build host
# ============================================================================

ensure_podman_connection() {
    log "Ensuring podman connection '${PODMAN_CONNECTION}' → ${REMOTE_HOST}"

    if podman system connection list --format '{{.Name}}' | grep -qxF "${PODMAN_CONNECTION}"; then
        echo "Connection '${PODMAN_CONNECTION}' already registered"
    else
        echo "Registering new podman connection..."

        # Resolve the remote podman socket path. For root on spark4 this is
        # /run/podman/podman.sock; for rootless it would be
        # /run/user/<uid>/podman/podman.sock. We always SSH as root to the
        # cluster (per project convention), so root-socket is the default.
        local remote_uid
        remote_uid="$(ssh -i "${PODMAN_SSH_IDENTITY}" -o BatchMode=yes -o ConnectTimeout=5 \
            "${REMOTE_HOST}" id -u 2>/dev/null)" \
            || die "SSH to ${REMOTE_HOST} failed — verify the key is authorized"

        local sock_path
        if [[ "${remote_uid}" == "0" ]]; then
            sock_path="/run/podman/podman.sock"
        else
            sock_path="/run/user/${remote_uid}/podman/podman.sock"
        fi

        podman system connection add "${PODMAN_CONNECTION}" \
            "ssh://${REMOTE_HOST}${sock_path}" \
            --identity "${PODMAN_SSH_IDENTITY}" \
            || die "Failed to register podman connection '${PODMAN_CONNECTION}'"
    fi

    echo "Validating connection..."
    if ! podman --connection "${PODMAN_CONNECTION}" info >/dev/null 2>&1; then
        cat >&2 <<EOF

ERROR: Podman connection '${PODMAN_CONNECTION}' is not responding.

Check on ${REMOTE_HOST}:
  systemctl status podman.socket
  systemctl enable --now podman.socket

And that the socket path matches what this script registered:
  podman system connection list
EOF
        exit 1
    fi

    local remote_arch
    remote_arch="$(podman --connection "${PODMAN_CONNECTION}" info --format '{{.Host.Arch}}')"
    if [[ "${remote_arch}" != "arm64" && "${remote_arch}" != "aarch64" ]]; then
        die "Remote host is ${remote_arch}, expected arm64/aarch64 (scitrera base image is arm64)"
    fi
    echo "Remote podman is reachable (arch=${remote_arch})"
}

# ============================================================================
# Verify the pytorch dev base image is present in the remote podman store
# ============================================================================
#
# The recipes default to a locally-built xomoxcc base, e.g.
#   BASE_IMAGE=xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132
# which is NOT on Docker Hub — it's built locally via
# scripts/build_pytorch_base_image.sh and kept in spark4's podman store.
# If the sglang build runs before the base image exists, podman will try
# to pull from docker.io and fail with a 404 after a long retry cycle.
# Fail fast here instead with a clear diagnostic pointing at the fix.

resolve_base_image() {
    # Resolves the effective BASE_IMAGE. Order of precedence:
    #   1. BUILD_SM121_BASE_IMAGE env var (highest — scripting override)
    #   2. --base <value> CLI flag (xomoxcc/scitrera alias or verbatim image)
    #   3. Recipe default (lowest)
    # Fills the globals EFFECTIVE_BASE_IMAGE and BASE_IMAGE_SOURCE.
    if [[ -n "${EFFECTIVE_BASE_IMAGE}" ]]; then
        return 0   # already resolved
    fi

    if [[ -n "${BASE_IMAGE_OVERRIDE}" ]]; then
        EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_OVERRIDE}"
        BASE_IMAGE_SOURCE="BUILD_SM121_BASE_IMAGE env"
        return 0
    fi

    case "${BASE_IMAGE_ALIAS}" in
        xomoxcc)
            EFFECTIVE_BASE_IMAGE="${BASE_XOMOXCC_IMAGE}"
            BASE_IMAGE_SOURCE="--base xomoxcc"
            return 0
            ;;
        scitrera)
            EFFECTIVE_BASE_IMAGE="${BASE_SCITRERA_IMAGE}"
            BASE_IMAGE_SOURCE="--base scitrera"
            return 0
            ;;
        "")
            ;;   # fall through to recipe default
        *)
            EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_ALIAS}"
            BASE_IMAGE_SOURCE="--base (verbatim)"
            return 0
            ;;
    esac

    # Fall-through: no override, use recipe default.
    local recipe_file="${PATCHES_DIR}/${RECIPE_NAME}.recipe"
    if [[ -f "${recipe_file}" ]]; then
        EFFECTIVE_BASE_IMAGE="$(grep -E '^BASE_IMAGE=' "${recipe_file}" | head -1 | cut -d= -f2-)"
        BASE_IMAGE_SOURCE="recipe default"
    fi
}

ensure_base_image_present() {
    resolve_base_image
    local base_image="${EFFECTIVE_BASE_IMAGE}"
    [[ -n "${base_image}" ]] || return 0   # no base image declared, nothing to check

    log "Verifying base image '${base_image}' is present on '${PODMAN_CONNECTION}' (from ${BASE_IMAGE_SOURCE})"

    # Check both short-name and docker.io/ FQN forms. build_pytorch_base_image.sh
    # tags with both, but a hand-built image or earlier script version might
    # only have one. Either is acceptable for the downstream Dockerfile's
    # FROM resolution.
    if podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${base_image}" 2>/dev/null; then
        echo "Base image found on ${PODMAN_CONNECTION} as docker.io/${base_image}"
        return 0
    fi
    if podman --connection "${PODMAN_CONNECTION}" image exists "${base_image}" 2>/dev/null; then
        warn "Base image exists as short name only ('${base_image}'), not as docker.io/${base_image}."
        warn "The Dockerfile FROM step may normalize to docker.io/... and fail to find it."
        warn "Recommend retagging: podman --connection ${PODMAN_CONNECTION} tag ${base_image} docker.io/${base_image}"
        return 0
    fi

    # Not found locally — try pulling from Docker Hub before giving up.
    echo "Base image not found locally on ${PODMAN_CONNECTION}. Attempting pull from Docker Hub..."
    if podman --connection "${PODMAN_CONNECTION}" pull "docker.io/${base_image}" 2>&1; then
        echo "Base image pulled successfully from Docker Hub."
        return 0
    fi

    # Pull failed. Give a specific diagnostic for xomoxcc images (which may
    # not be published), generic hint for everything else.
    case "${base_image}" in
        xomoxcc/dgx-spark-pytorch-dev:*)
            cat >&2 <<EOF

ERROR: Base image '${base_image}' is not present on ${PODMAN_CONNECTION}
and could not be pulled from Docker Hub.

If the image has not been pushed to Docker Hub, build it first:
  bash ${SCRIPT_DIR}/build_pytorch_base_image.sh

That build takes approximately 3-5 hours (cold) or 30-60 min (with warm
ccache from a prior run). It produces the CUDA 13.2 + PyTorch 2.11 base
that this sglang build depends on.
EOF
            exit 1
            ;;
        *)
            die "Base image '${base_image}' not found locally and pull from Docker Hub failed."
            ;;
    esac
}

# ============================================================================
# cuda-containers clone + branch management (runs on x86)
# ============================================================================

prepare_cuda_containers() {
    log "Preparing cuda-containers at ${CUDA_CONTAINERS_DIR}"

    if [[ ! -d "${CUDA_CONTAINERS_DIR}/.git" ]]; then
        echo "Cloning scitrera/cuda-containers..."
        mkdir -p "$(dirname "${CUDA_CONTAINERS_DIR}")"
        git clone "${CUDA_CONTAINERS_REPO}" "${CUDA_CONTAINERS_DIR}"
    fi

    cd "${CUDA_CONTAINERS_DIR}"
    echo "Fetching upstream..."
    git fetch origin

    # Create or reset local sm121 branch from origin/main so re-runs are idempotent.
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        git checkout "${BRANCH_NAME}"
        git reset --hard origin/main
        git clean -fd
    else
        git checkout -b "${BRANCH_NAME}" origin/main
    fi

    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "HEAD:   $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
}

# ============================================================================
# Apply patches into the cuda-containers clone (runs on x86)
# ============================================================================

apply_patches() {
    log "Applying SM121 patches"

    cd "${CUDA_CONTAINERS_DIR}"

    # Determine recipe variant once. The "*gemma4*" pattern matches our
    # sglang-0.5.{11,12}-gemma4-sm121.recipe filename and any future gemma4 spinoff.
    local apply_gemma4_patches=0
    if [[ "${RECIPE_NAME}" == *gemma4* ]]; then
        apply_gemma4_patches=1
    fi

    # Determine whether the Gemma-4 MTP cherry-pick (PR #24436) is needed.
    # The PR is merged into upstream as of v0.5.12; applying it on top of
    # v0.5.12+ would fail with "already applied" (or silently corrupt the
    # tree, depending on patch detection). Source the recipe to read
    # SGLANG_VERSION and gate accordingly. Falls back to "apply" on
    # recipes without a parseable SGLANG_VERSION (e.g. legacy main-branch
    # recipes pinned to a commit SHA before #24436, so the patch still
    # applies cleanly there).
    local recipe_sglang_version=""
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]]; then
        recipe_sglang_version="$(grep -E '^SGLANG_VERSION=' "${PATCHES_DIR}/${RECIPE_NAME}.recipe" \
            | head -1 | cut -d= -f2- | tr -d '"' || true)"
    fi
    local apply_gemma4_mtp_patch=1
    if [[ -n "${recipe_sglang_version}" ]] \
        && printf '0.5.12\n%s\n' "${recipe_sglang_version}" \
        | sort -V -C 2>/dev/null; then
        # recipe_sglang_version >= 0.5.12 → MTP cherry-pick is already in
        # the source tree, do not re-apply.
        apply_gemma4_mtp_patch=0
    fi

    # DSV4 NVFP4 (PR #25820) — gated by an explicit recipe variable rather
    # than a name pattern or version gate: the patch must be dropped the
    # moment the PR lands in the pinned SGLANG_REF (re-applying a merged
    # patch fails the in-container dry-run and aborts the build), and an
    # explicit APPLY_DSV4_NVFP4_PR25820=0/absent keeps that state visible
    # in the recipe itself instead of in script heuristics.
    local apply_dsv4_nvfp4_patch=0
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_DSV4_NVFP4_PR25820=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        apply_dsv4_nvfp4_patch=1
    fi

    # DSV4 EAGLE-MTP marlin branch + TileLang 0.1.8 compat — gated separately
    # from apply_dsv4_nvfp4_patch (mutually exclusive: this fires once the base
    # NVFP4-MoE class is native and only the GB10-specific MTP/indexer
    # remainder still needs patching). Drop APPLY_DSV4_MTP_MARLIN_TILELANG once
    # the marlin branch and the TileLang 0.1.8 fix both land upstream.
    local apply_dsv4_mtp_marlin_patch=0
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_DSV4_MTP_MARLIN_TILELANG=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        apply_dsv4_mtp_marlin_patch=1
    fi

    # DiffusionGemma (PR #28054) — gated like DSV4 by an explicit recipe
    # variable. Adds the dLLM Gemma4Renoise model/sampler. The matching
    # dockerfile-diffusiongemma.patch ANCHORS AFTER the gemma4-nvfp4 block, so
    # this REQUIRES a *gemma4* recipe variant (apply_gemma4_patches=1). Drop
    # APPLY_DIFFUSIONGEMMA_PR28054 once PR #28054 lands in the pinned SGLANG_REF
    # (re-applying a merged patch aborts the build).
    local apply_diffusiongemma_patch=0
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_DIFFUSIONGEMMA_PR28054=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        apply_diffusiongemma_patch=1
    fi

    # NemotronH MTP (PR #27998) — gated like DSV4 by an explicit recipe
    # variable. Drop APPLY_NEMOTRONH_MTP_PR27998 the moment the PR lands in the
    # pinned SGLANG_REF (re-applying a merged patch fails the in-container
    # dry-run and aborts the build). Anchors on the same Dockerfile region as
    # the DSV4/gemma4 patches, so the two are mutually exclusive per recipe.
    local apply_nemotronh_mtp_patch=0
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_NEMOTRONH_MTP_PR27998=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        apply_nemotronh_mtp_patch=1
    fi

    # Qwen3.6 ModelOpt mixed NVFP4 (PR #27906) — gated like DSV4 by an explicit
    # recipe variable. Drop APPLY_QWEN36_MIXED_NVFP4_PR27906 the moment the PR
    # lands in the pinned SGLANG_REF (re-applying a merged patch fails the
    # in-container dry-run and aborts the build). UNLIKE the DSV4/NemotronH/gemma4
    # patches, dockerfile-qwen36-mixed-nvfp4.patch uses trailing-context-only and
    # therefore STACKS after them — the production recipe runs it together with
    # DSV4 (qwen36 dockerfile step 2f applies after dsv4 step 2c).
    local apply_qwen36_mixed_nvfp4_patch=0
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^APPLY_QWEN36_MIXED_NVFP4_PR27906=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        apply_qwen36_mixed_nvfp4_patch=1
    fi

    # 1. Drop sgl-kernel source patches into the build context.
    # The Dockerfile COPY steps read from container-build/patches/ and the
    # in-container `patch` invocations are conditionally gated by the
    # APPLY_SGL_KERNEL_* build-args — we always copy the sgl-kernel files
    # so the build context is deterministic regardless of toggle state.
    # Gemma-4 source patches are copied only for the gemma4 recipe variant
    # because the matching dockerfile-gemma4-nvfp4.patch (which adds the
    # `RUN patch -p1 < /tmp/sglang-gemma4-*.patch` steps to the Dockerfile)
    # is itself gated by ${apply_gemma4_patches} below — copying the source
    # patch files in the non-gemma4 variant would have no effect but would
    # bloat the build context unnecessarily.
    mkdir -p container-build/patches
    # Source patches always copied (applied unconditionally in every build).
    # Includes sgl-kernel patches (CMakeLists / JIT header edits, gated at
    # build time by APPLY_SGL_KERNEL_* build-args) and the Gemma-4 MTP
    # cherry-pick of upstream PR #24436 (inert until --speculative-algorithm
    # FROZEN_KV_MTP or a gemma4_assistant drafter is loaded).
    # Patches always copied (Dockerfile patch step is itself gated by
    # build-args / version gates below — copying the file in regardless
    # keeps the build context deterministic).
    local always_source_patches=(
        sgl-kernel-sm121.patch
        sgl-kernel-sm121-debug.patch
        sgl-kernel-arch-prune.patch
        sgl-kernel-disable-fa3.patch
        sgl-kernel-skip-sm90-target.patch
        sgl-kernel-skip-flashmla.patch
    )
    # Gemma-4 MTP cherry-pick — version-gated. Copy only when we will
    # actually apply it, to avoid bloating the build context on 0.5.12+.
    local mtp_source_patches=(
        sglang-gemma4-mtp-pr24436.patch
    )
    local gemma4_source_patches=(
        sglang-gemma4-geglu-nan-clamp.patch
    )
    # DSV4 NVFP4 source patch (PR #25820 rebased onto v0.5.13) + TileLang 0.1.8
    # compat fix — both copied only when the recipe opts in via
    # APPLY_DSV4_NVFP4_PR25820=1, same rationale as the gemma4 patches.
    # The TileLang patch is folded into the same gate because it is only
    # needed for DSV4-Flash (tilelang_kernel.py is the DSA indexer path), and
    # dockerfile-dsv4-nvfp4.patch now adds both COPY+RUN steps.
    local dsv4_nvfp4_source_patches=(
        sglang-dsv4-nvfp4-pr25820.patch
        sglang-tilelang-018-indexer-compat.patch
    )
    # DSV4 EAGLE-MTP marlin branch (v0.5.14+ remainder, rebased standalone
    # against the now-native HybridFp8NvFp4Config) + the same TileLang 0.1.8
    # compat fix (unchanged file, listed again here since it must be copied
    # under THIS gate too — APPLY_DSV4_NVFP4_PR25820=0 on v0.5.14 means the
    # entry above is not copied). Copied only when the recipe opts in via
    # APPLY_DSV4_MTP_MARLIN_TILELANG=1.
    local dsv4_mtp_marlin_source_patches=(
        sglang-dsv4-mtp-marlin-v0.5.14.patch
        sglang-tilelang-018-indexer-compat.patch
    )
    # DiffusionGemma source patch (PR #28054) — copied only when the recipe
    # opts in, same rationale as the dsv4/gemma4 patches.
    local diffusiongemma_source_patches=(
        sglang-diffusiongemma-pr28054.patch
    )
    # NemotronH MTP source patch (PR #27998) — copied only when the recipe
    # opts in, same rationale as the dsv4/gemma4 patches.
    local nemotronh_mtp_source_patches=(
        sglang-nemotronh-mtp-pr27998.patch
    )
    # Qwen3.6 mixed NVFP4 source patch (PR #27906) — copied only when the recipe
    # opts in, same rationale as the dsv4/nemotronh patches.
    local qwen36_mixed_nvfp4_source_patches=(
        sglang-qwen36-mixed-nvfp4-pr27906.patch
    )
    local patches_to_copy=( "${always_source_patches[@]}" )
    if (( apply_gemma4_mtp_patch )); then
        patches_to_copy+=( "${mtp_source_patches[@]}" )
    fi
    if (( apply_gemma4_patches )); then
        patches_to_copy+=( "${gemma4_source_patches[@]}" )
    fi
    if (( apply_dsv4_nvfp4_patch )); then
        patches_to_copy+=( "${dsv4_nvfp4_source_patches[@]}" )
    fi
    if (( apply_dsv4_mtp_marlin_patch )); then
        patches_to_copy+=( "${dsv4_mtp_marlin_source_patches[@]}" )
    fi
    if (( apply_diffusiongemma_patch )); then
        patches_to_copy+=( "${diffusiongemma_source_patches[@]}" )
    fi
    if (( apply_nemotronh_mtp_patch )); then
        patches_to_copy+=( "${nemotronh_mtp_source_patches[@]}" )
    fi
    if (( apply_qwen36_mixed_nvfp4_patch )); then
        patches_to_copy+=( "${qwen36_mixed_nvfp4_source_patches[@]}" )
    fi
    # sgl-kernel patch variant: a main-ahead SGLANG_REF (post-v0.5.13) can drift
    # the sgl-kernel CMakeLists out from under the SM121 sgl-kernel patches (e.g.
    # the mscclpp link refactor that broke sgl-kernel-skip-sm90-target.patch on
    # commit 3a1417a). A recipe pinned forward sets SGL_KERNEL_MAINAHEAD=1; for
    # any patch that has a "<base>-mainahead.patch" sibling we then install THAT
    # under the canonical name, so the Dockerfile applies it transparently while
    # the v0.5.13 recipes keep using the originals.
    local sgl_kernel_variant=""
    if [[ -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]] \
        && grep -qE '^SGL_KERNEL_MAINAHEAD=1' "${PATCHES_DIR}/${RECIPE_NAME}.recipe"; then
        sgl_kernel_variant="-mainahead"
    fi
    for p in "${patches_to_copy[@]}"; do
        local src="${PATCHES_DIR}/${p}"
        if [[ -n "${sgl_kernel_variant}" \
            && -f "${PATCHES_DIR}/${p%.patch}${sgl_kernel_variant}.patch" ]]; then
            src="${PATCHES_DIR}/${p%.patch}${sgl_kernel_variant}.patch"
        fi
        if [[ -f "${src}" ]]; then
            install -m 0644 "${src}" "container-build/patches/${p}"
            echo "Installed container-build/patches/${p} (from $(basename "${src}"))"
        fi
    done

    # 2. Patch the Dockerfile (adds `patch` apt-get dependency + COPY/RUN step).
    #    Dry-run first so any upstream drift fails early with a clear diagnostic
    #    rather than half-applying.
    echo "Applying dockerfile-sm121.patch..."
    patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-sm121.patch" \
        || die "Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-sm121.patch"
    patch -p1 < "${PATCHES_DIR}/dockerfile-sm121.patch"
    grep -q 'patches/sgl-kernel-sm121.patch' container-build/Dockerfile.sglang-nightly \
        || die "Dockerfile patch verification failed"
    echo "Dockerfile patched"

    # 2-kernels. Pin huggingface `kernels` to a transformers-5.x-compatible
    #     version (default 0.12.3). Workaround for transformers/integrations/
    #     hub_kernels.py:89 calling LayerRepository(...) without the revision/
    #     version arg that kernels>=0.13 now requires — SGLang startup fails
    #     with ValueError at first transformers.activations import.
    #     The patch adds an `ARG KERNELS_VERSION` + force-reinstall RUN step
    #     after the transformers install; recipe sets KERNELS_VERSION.
    #     Always applied — no-op when KERNELS_VERSION is empty in the recipe.
    if [[ -f "${PATCHES_DIR}/dockerfile-kernels-pin.patch" ]]; then
        echo "Applying dockerfile-kernels-pin.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-kernels-pin.patch" \
            || die "kernels-pin Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-kernels-pin.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-kernels-pin.patch"
        grep -q 'ARG KERNELS_VERSION' container-build/Dockerfile.sglang-nightly \
            || die "kernels-pin Dockerfile patch verification failed"
        echo "kernels-pin Dockerfile patched"
    fi

    # 2-audio. Add an `ARG AUDIO_DEPS` + gated install RUN step (librosa etc.)
    #     for omni models that decode audio at runtime. Anchors on the kernels-pin
    #     block above, so it MUST run after it. Always applied — no-op when the
    #     recipe leaves AUDIO_DEPS empty. See patches/dockerfile-audio-deps.patch.
    if [[ -f "${PATCHES_DIR}/dockerfile-audio-deps.patch" ]]; then
        echo "Applying dockerfile-audio-deps.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-audio-deps.patch" \
            || die "audio-deps Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-audio-deps.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-audio-deps.patch"
        grep -q 'ARG AUDIO_DEPS' container-build/Dockerfile.sglang-nightly \
            || die "audio-deps Dockerfile patch verification failed"
        echo "audio-deps Dockerfile patched"
    fi

    # 2-cutlass. Add an `ARG CUTLASS_DSL_VERSION` + gated force-reinstall RUN
    #     step pinning nvidia-cutlass-dsl (+ its libs-base/libs-cu13 wheels) to
    #     an exact version. Root cause 2026-07-09: NVIDIA shipped different
    #     wheel content under the SAME 4.5.2 version — the newer variant's
    #     tvm_ffi_provider passes data=[] to an llvm.mlir_global_dtors() binding
    #     without that param → "ICE ... unexpected keyword argument 'data'" on
    #     every fresh CuTe-DSL JIT compile (flashinfer rmsnorm_cute in CUDA-graph
    #     warmup crashes ANY model). flashinfer pins only >=4.5.0, so rebuilds
    #     re-resolve this transitively. Anchors on the audio-deps block above,
    #     so it MUST run after it. Always applied — no-op when the recipe leaves
    #     CUTLASS_DSL_VERSION empty. See patches/dockerfile-cutlass-dsl-pin.patch.
    if [[ -f "${PATCHES_DIR}/dockerfile-cutlass-dsl-pin.patch" ]]; then
        echo "Applying dockerfile-cutlass-dsl-pin.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-cutlass-dsl-pin.patch" \
            || die "cutlass-dsl-pin Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-cutlass-dsl-pin.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-cutlass-dsl-pin.patch"
        grep -q 'ARG CUTLASS_DSL_VERSION' container-build/Dockerfile.sglang-nightly \
            || die "cutlass-dsl-pin Dockerfile patch verification failed"
        echo "cutlass-dsl-pin Dockerfile patched"
    fi

    # 2-accelerate. Add an `ARG ACCELERATE_DEPS` (default "accelerate") + gated
    #     install RUN step. SGLang's ModelOptModelLoader imports accelerate, which
    #     the upstream base image omits (GLM-5-NVFP4 + EAGLE/speculative-decode
    #     against a modelopt-quantized target need it). Baking it turns the
    #     sglang_launch.sh runtime `pip install accelerate` guard into a no-op.
    #     Anchors on the `# SM121 shared memory fix` block, so it MUST run AFTER
    #     the cutlass-dsl-pin block above (which inserts just above the same
    #     anchor). Always applied — recipe sets ACCELERATE_DEPS="" to opt out.
    #     See patches/dockerfile-accelerate.patch.
    if [[ -f "${PATCHES_DIR}/dockerfile-accelerate.patch" ]]; then
        echo "Applying dockerfile-accelerate.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-accelerate.patch" \
            || die "accelerate Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-accelerate.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-accelerate.patch"
        grep -q 'ARG ACCELERATE_DEPS' container-build/Dockerfile.sglang-nightly \
            || die "accelerate Dockerfile patch verification failed"
        echo "accelerate Dockerfile patched"
    fi

    # 2-dsv4. DeepSeek-V4-Flash FlashMLA sparse-decode kernel (sm_121a).
    #     Adds an ARG + RUN step in the builder (before the dist-packages split)
    #     that installs stock flash_mla and builds 0xSero/deepseek-v4-flash-sm120
    #     retargeted to DSV4_KERNEL_ARCH. The Dockerfile patch is applied
    #     unconditionally; the RUN itself is gated on DSV4_KERNEL_REPO (recipe
    #     sets it) and is FATAL (a build failure aborts the image — clear
    #     DSV4_KERNEL_REPO in the recipe to opt out).
    #     No source-patch copy needed — the kernel is git-cloned at build time.
    if [[ -f "${PATCHES_DIR}/dockerfile-dsv4-flashmla.patch" ]]; then
        echo "Applying dockerfile-dsv4-flashmla.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-dsv4-flashmla.patch" \
            || die "dsv4-flashmla Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-dsv4-flashmla.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-dsv4-flashmla.patch"
        grep -q 'DSV4_KERNEL_REPO' container-build/Dockerfile.sglang-nightly \
            || die "dsv4-flashmla Dockerfile patch verification failed"
        echo "dsv4-flashmla Dockerfile patched"
    fi

    # 2a. Gemma-4 MTP (PR #24436) Dockerfile patch — version-gated.
    #     Adds a COPY + RUN step that cherry-picks upstream PR #24436 into
    #     the SGLang source before `uv pip install ./python`. Inert for
    #     non-Gemma-4 workloads (the new files only activate when
    #     --speculative-algorithm FROZEN_KV_MTP or a Gemma-4 drafter is loaded).
    #     PR #24436 is merged into v0.5.12 — on that and later releases we
    #     skip the cherry-pick entirely (re-applying would fail).
    if (( apply_gemma4_mtp_patch )); then
        echo "Applying dockerfile-gemma4-mtp.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-gemma4-mtp.patch" \
            || die "Gemma-4 MTP Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-gemma4-mtp.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-gemma4-mtp.patch"
        grep -q 'sglang-gemma4-mtp-pr24436.patch' container-build/Dockerfile.sglang-nightly \
            || die "Gemma-4 MTP Dockerfile patch verification failed"
        echo "Gemma-4 MTP Dockerfile patched"
    else
        echo "Skipping dockerfile-gemma4-mtp.patch (recipe SGLANG_VERSION='${recipe_sglang_version}' >= 0.5.12 — PR #24436 already upstream)"
    fi

    # 2b. Gemma-4 NVFP4 Dockerfile patch — only for the gemma4 recipe variant.
    #     Adds COPY + apply step for PR #22928 (GEGLU activation + NaN clamp).
    #     PR #22929 (per-expert weight loading) was dropped 2026-06-19 — landed
    #     upstream. Stacks on top of the sm121 patch.
    if (( apply_gemma4_patches )); then
        echo "Applying dockerfile-gemma4-nvfp4.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-gemma4-nvfp4.patch" \
            || die "Gemma4 Dockerfile patch dry-run failed — regenerate dockerfile-gemma4-nvfp4.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-gemma4-nvfp4.patch"
        grep -q 'sglang-gemma4-geglu-nan-clamp.patch' container-build/Dockerfile.sglang-nightly \
            || die "Gemma4 Dockerfile patch verification failed"
        echo "Gemma4 Dockerfile patched"
    else
        echo "Skipping dockerfile-gemma4-nvfp4.patch (RECIPE_NAME='${RECIPE_NAME}' is not a gemma4 variant)"
    fi

    # 2c. DSV4 NVFP4 (PR #25820) Dockerfile patch — recipe-gated (see the
    #     apply_dsv4_nvfp4_patch determination above). Adds the COPY + RUN
    #     step that applies sglang-dsv4-nvfp4-pr25820.patch to the sglang
    #     source before `uv pip install ./python`. NOTE: anchors on the same
    #     Dockerfile region as the gemma4-nvfp4 patch — not combinable with
    #     a gemma4 recipe without regenerating the context (no such recipe
    #     exists; the dry-run below catches it if that ever changes).
    if (( apply_dsv4_nvfp4_patch )); then
        echo "Applying dockerfile-dsv4-nvfp4.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-dsv4-nvfp4.patch" \
            || die "DSV4 NVFP4 Dockerfile patch dry-run failed — regenerate dockerfile-dsv4-nvfp4.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-dsv4-nvfp4.patch"
        grep -q 'sglang-dsv4-nvfp4-pr25820.patch' container-build/Dockerfile.sglang-nightly \
            || die "DSV4 NVFP4 Dockerfile patch verification failed"
        echo "DSV4 NVFP4 Dockerfile patched"
    else
        echo "Skipping dockerfile-dsv4-nvfp4.patch (recipe does not set APPLY_DSV4_NVFP4_PR25820=1)"
    fi

    # 2c-bis. DSV4 EAGLE-MTP marlin + TileLang 0.1.8 compat Dockerfile patch —
    #     recipe-gated (see the apply_dsv4_mtp_marlin_patch determination
    #     above). v0.5.14+ remainder of 2c above: applies the small marlin
    #     draft-MoE branch on top of the now-native HybridFp8NvFp4Config, plus
    #     the still-needed TileLang 0.1.8 indexer compat fix. Same anchor as
    #     2c/2b — not combinable with a gemma4 or the pre-merge DSV4 recipe
    #     without regenerating context (the dry-run below catches it).
    if (( apply_dsv4_mtp_marlin_patch )); then
        echo "Applying dockerfile-dsv4-mtp-marlin-tilelang.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-dsv4-mtp-marlin-tilelang.patch" \
            || die "DSV4 MTP marlin/tilelang Dockerfile patch dry-run failed — regenerate dockerfile-dsv4-mtp-marlin-tilelang.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-dsv4-mtp-marlin-tilelang.patch"
        grep -q 'sglang-dsv4-mtp-marlin-v0.5.14.patch' container-build/Dockerfile.sglang-nightly \
            || die "DSV4 MTP marlin/tilelang Dockerfile patch verification failed"
        echo "DSV4 MTP marlin/tilelang Dockerfile patched"
    else
        echo "Skipping dockerfile-dsv4-mtp-marlin-tilelang.patch (recipe does not set APPLY_DSV4_MTP_MARLIN_TILELANG=1)"
    fi

    # 2d. NemotronH MTP (PR #27998) Dockerfile patch — recipe-gated (see the
    #     apply_nemotronh_mtp_patch determination above). Adds the COPY + RUN
    #     step that applies sglang-nemotronh-mtp-pr27998.patch to the sglang
    #     source before `uv pip install ./python`. NOTE: anchors on the same
    #     Dockerfile region as the dsv4/gemma4 patches — mutually exclusive per
    #     recipe; the dry-run below catches it if that ever changes.
    if (( apply_nemotronh_mtp_patch )); then
        echo "Applying dockerfile-nemotronh-mtp.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-nemotronh-mtp.patch" \
            || die "NemotronH MTP Dockerfile patch dry-run failed — regenerate dockerfile-nemotronh-mtp.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-nemotronh-mtp.patch"
        grep -q 'sglang-nemotronh-mtp-pr27998.patch' container-build/Dockerfile.sglang-nightly \
            || die "NemotronH MTP Dockerfile patch verification failed"
        echo "NemotronH MTP Dockerfile patched"
    else
        echo "Skipping dockerfile-nemotronh-mtp.patch (recipe does not set APPLY_NEMOTRONH_MTP_PR27998=1)"
    fi

    # 2e. DiffusionGemma (PR #28054) Dockerfile patch — recipe-gated (see the
    #     apply_diffusiongemma_patch determination above). Adds the COPY + RUN
    #     step that applies sglang-diffusiongemma-pr28054.patch. NOTE: this patch
    #     ANCHORS AFTER the gemma4-nvfp4 block, so it requires a *gemma4* recipe
    #     variant (the gemma4 patch must run first); the dry-run below catches a
    #     missing anchor.
    if (( apply_diffusiongemma_patch )); then
        echo "Applying dockerfile-diffusiongemma.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-diffusiongemma.patch" \
            || die "DiffusionGemma Dockerfile patch dry-run failed — needs a gemma4 variant / regenerate dockerfile-diffusiongemma.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-diffusiongemma.patch"
        grep -q 'sglang-diffusiongemma-pr28054.patch' container-build/Dockerfile.sglang-nightly \
            || die "DiffusionGemma Dockerfile patch verification failed"
        echo "DiffusionGemma Dockerfile patched"
    else
        echo "Skipping dockerfile-diffusiongemma.patch (recipe does not set APPLY_DIFFUSIONGEMMA_PR28054=1)"
    fi

    # 2f. Qwen3.6 mixed NVFP4 (PR #27906) Dockerfile patch — recipe-gated (see the
    #     apply_qwen36_mixed_nvfp4_patch determination above). Adds the COPY + RUN
    #     step that applies sglang-qwen36-mixed-nvfp4-pr27906.patch to the sglang
    #     source before `uv pip install ./python`. This patch uses trailing-context-
    #     only, so it STACKS after the dsv4 step (2c) when both are enabled in the
    #     production recipe — and also applies standalone. Must run AFTER 2c.
    if (( apply_qwen36_mixed_nvfp4_patch )); then
        echo "Applying dockerfile-qwen36-mixed-nvfp4.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-qwen36-mixed-nvfp4.patch" \
            || die "Qwen3.6 mixed NVFP4 Dockerfile patch dry-run failed — regenerate dockerfile-qwen36-mixed-nvfp4.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-qwen36-mixed-nvfp4.patch"
        grep -q 'sglang-qwen36-mixed-nvfp4-pr27906.patch' container-build/Dockerfile.sglang-nightly \
            || die "Qwen3.6 mixed NVFP4 Dockerfile patch verification failed"
        echo "Qwen3.6 mixed NVFP4 Dockerfile patched"
    else
        echo "Skipping dockerfile-qwen36-mixed-nvfp4.patch (recipe does not set APPLY_QWEN36_MIXED_NVFP4_PR27906=1)"
    fi

    # 3. Drop in the recipe file. run_build() parses it inline and calls
    #    `podman build` directly, bypassing container-build/build-image.sh
    #    (which uses `docker buildx build` — podman has no buildx subcommand).
    install -m 0644 "${PATCHES_DIR}/${RECIPE_NAME}.recipe" \
        "container-recipes/${RECIPE_NAME}.recipe"
    echo "Installed container-recipes/${RECIPE_NAME}.recipe"

    # 4. Defense-in-depth: also patch build-image.sh to replace
    #    `docker buildx build` with `docker build`. Our run_build() bypasses
    #    build-image.sh entirely, so this is not strictly required — but the
    #    patched clone is then also usable for any other recipe via podman,
    #    and re-runs of this script hard-reset the branch so there's no
    #    accumulation cost.
    echo "Applying build-image-sh-podman.patch..."
    patch --dry-run -p1 < "${PATCHES_DIR}/build-image-sh-podman.patch" \
        || die "build-image.sh patch dry-run failed — upstream build-image.sh drifted; regenerate build-image-sh-podman.patch"
    patch -p1 < "${PATCHES_DIR}/build-image-sh-podman.patch"
    grep -q '^BUILD_CMD=(docker build)$' container-build/build-image.sh \
        || die "build-image.sh patch verification failed"
    echo "build-image.sh patched"
}

# ============================================================================
# Build via remote podman socket
# ============================================================================

run_build() {
    log "Running podman build on '${PODMAN_CONNECTION}' (60–90 minutes expected)"
    cd "${CUDA_CONTAINERS_DIR}"

    # Source the recipe in a subshell so we can read the build-time values
    # without polluting our own environment, then pass them as --build-arg.
    # The script's own IMAGE_TAG is verified against the recipe for consistency.
    local recipe_file="container-recipes/${RECIPE_NAME}.recipe"
    [[ -f "${recipe_file}" ]] || die "Recipe not found: ${recipe_file}"

    local R_DOCKERFILE R_TARGET R_BASE_IMAGE R_FLASHINFER_VERSION
    local R_TRANSFORMERS_VERSION R_KERNELS_VERSION R_CUTLASS_DSL_VERSION R_AUDIO_DEPS R_ACCELERATE_DEPS R_SGLANG_VERSION R_SGLANG_REF R_IMAGE_TAG
    local R_FLASH_MLA_REPO R_FLASH_MLA_REF R_DSV4_KERNEL_REPO R_DSV4_KERNEL_REF R_DSV4_KERNEL_ARCH
    # shellcheck disable=SC1090
    source <(
        set -a
        # shellcheck disable=SC1090
        source "${recipe_file}"
        echo "R_DOCKERFILE='${DOCKERFILE}'"
        echo "R_TARGET='${TARGET}'"
        echo "R_BASE_IMAGE='${BASE_IMAGE}'"
        echo "R_FLASHINFER_VERSION='${FLASHINFER_VERSION}'"
        echo "R_TRANSFORMERS_VERSION='${TRANSFORMERS_VERSION}'"
        echo "R_KERNELS_VERSION='${KERNELS_VERSION:-}'"
        echo "R_CUTLASS_DSL_VERSION='${CUTLASS_DSL_VERSION:-}'"
        echo "R_AUDIO_DEPS='${AUDIO_DEPS:-}'"
        # default-on: `-` (not `:-`) so an unset recipe still bakes accelerate,
        # while an explicit ACCELERATE_DEPS="" in a recipe opts out (stays empty).
        echo "R_ACCELERATE_DEPS='${ACCELERATE_DEPS-accelerate}'"
        echo "R_SGLANG_VERSION='${SGLANG_VERSION}'"
        echo "R_SGLANG_REF='${SGLANG_REF}'"
        echo "R_FLASH_MLA_REPO='${FLASH_MLA_REPO:-}'"
        echo "R_FLASH_MLA_REF='${FLASH_MLA_REF:-}'"
        echo "R_DSV4_KERNEL_REPO='${DSV4_KERNEL_REPO:-}'"
        echo "R_DSV4_KERNEL_REF='${DSV4_KERNEL_REF:-main}'"
        echo "R_DSV4_KERNEL_ARCH='${DSV4_KERNEL_ARCH:-}'"
        echo "R_IMAGE_TAG='${IMAGE_TAG}'"
    )

    if [[ "${R_IMAGE_TAG}" != "${IMAGE_TAG}" ]]; then
        die "Recipe IMAGE_TAG (${R_IMAGE_TAG}) does not match script IMAGE_TAG (${IMAGE_TAG})"
    fi

    # Resolve the effective BASE_IMAGE once (may already be populated from
    # ensure_base_image_present). If nothing overrode the recipe, fall back
    # to the recipe's BASE_IMAGE value so the --build-arg is always set.
    resolve_base_image
    local effective_base_image="${EFFECTIVE_BASE_IMAGE:-${R_BASE_IMAGE}}"
    local effective_base_source="${BASE_IMAGE_SOURCE:-recipe default}"

    echo "Recipe values:"
    echo "  DOCKERFILE           = ${R_DOCKERFILE}"
    echo "  TARGET               = ${R_TARGET}"
    echo "  BASE_IMAGE (recipe)  = ${R_BASE_IMAGE}"
    echo "  BASE_IMAGE (in use)  = ${effective_base_image}  [${effective_base_source}]"
    echo "  FLASHINFER_VERSION   = ${R_FLASHINFER_VERSION}"
    echo "  TRANSFORMERS_VERSION = ${R_TRANSFORMERS_VERSION}"
    echo "  KERNELS_VERSION      = ${R_KERNELS_VERSION:-<unset, skipped>}"
    echo "  CUTLASS_DSL_VERSION  = ${R_CUTLASS_DSL_VERSION:-<unset, skipped>}"
    echo "  AUDIO_DEPS           = ${R_AUDIO_DEPS:-<unset, skipped>}"
    echo "  ACCELERATE_DEPS      = ${R_ACCELERATE_DEPS:-<empty, opted out>}"
    echo "  SGLANG_VERSION       = ${R_SGLANG_VERSION}"
    echo "  SGLANG_REF           = ${R_SGLANG_REF}"
    echo "  FLASH_MLA_REPO       = ${R_FLASH_MLA_REPO:-<unset>}"
    echo "  FLASH_MLA_REF        = ${R_FLASH_MLA_REF:-<unset>}"
    echo "  DSV4_KERNEL_REPO     = ${R_DSV4_KERNEL_REPO:-<unset, V4 FlashMLA kernel skipped>}"
    echo "  DSV4_KERNEL_REF      = ${R_DSV4_KERNEL_REF:-main}"
    echo "  DSV4_KERNEL_ARCH     = ${R_DSV4_KERNEL_ARCH:-<unset>}"
    echo "  IMAGE_TAG            = ${IMAGE_TAG}"
    echo "  BUILD_JOBS           = ${BUILD_JOBS} (overrides Dockerfile ARG default of 2)"
    echo "  sgl-kernel patches:"
    echo "    sm121 JIT kernel   = ALWAYS (late stage, cheap to re-apply)"
    echo "    sm121-debug        = $([ ${APPLY_SM121_DEBUG} -eq 1 ] && echo APPLY || echo skip)  (--sm121-debug opts in; runtime-gated by SGL_SM121_DEBUG_CUTLASS env)"
    echo "    arch-prune         = $([ ${APPLY_ARCH_PRUNE} -eq 1 ] && echo APPLY || echo skip)  (--no-arch-prune opts out)"
    echo "    disable-fa3        = $([ ${APPLY_DISABLE_FA3} -eq 1 ] && echo APPLY || echo skip)  (--keep-fa3 opts out)"
    echo "    skip-sm90-target   = $([ ${APPLY_SKIP_SM90_TARGET} -eq 1 ] && echo APPLY || echo skip)  (--keep-sm90-target opts out)"
    echo "    skip-flashmla      = $([ ${APPLY_SKIP_FLASHMLA} -eq 1 ] && echo APPLY || echo skip)  (--keep-flashmla opts out)"
    echo "  source patches (in apply_patches stage on x86):"
    echo "    gemma4-mtp PR24436 = (handled in apply_patches() — version-gated, see log above)"
    echo "    dsv4-nvfp4 PR25820 = (handled in apply_patches() — recipe-gated via APPLY_DSV4_NVFP4_PR25820, see log above)"
    echo "    tilelang-018-compat= (handled in apply_patches() — folded into APPLY_DSV4_NVFP4_PR25820 gate, see log above)"

    # The build context is container-build/ (contains Dockerfile + patches/
    # subdir). Podman streams it to the remote build host over the socket;
    # the build runs natively on arm64 and the result lands in the remote
    # host's local image store.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "container-build/${R_DOCKERFILE}" \
        --target "${R_TARGET}" \
        --build-arg "BASE_IMAGE=${effective_base_image}" \
        --build-arg "FLASHINFER_VERSION=${R_FLASHINFER_VERSION}" \
        --build-arg "TRANSFORMERS_VERSION=${R_TRANSFORMERS_VERSION}" \
        --build-arg "KERNELS_VERSION=${R_KERNELS_VERSION:-}" \
        --build-arg "CUTLASS_DSL_VERSION=${R_CUTLASS_DSL_VERSION:-}" \
        --build-arg "AUDIO_DEPS=${R_AUDIO_DEPS:-}" \
        --build-arg "ACCELERATE_DEPS=${R_ACCELERATE_DEPS}" \
        --build-arg "SGLANG_VERSION=${R_SGLANG_VERSION}" \
        --build-arg "SGLANG_REF=${R_SGLANG_REF}" \
        --build-arg "FLASH_MLA_REPO=${R_FLASH_MLA_REPO:-}" \
        --build-arg "FLASH_MLA_REF=${R_FLASH_MLA_REF:-}" \
        --build-arg "DSV4_KERNEL_REPO=${R_DSV4_KERNEL_REPO:-}" \
        --build-arg "DSV4_KERNEL_REF=${R_DSV4_KERNEL_REF:-main}" \
        --build-arg "DSV4_KERNEL_ARCH=${R_DSV4_KERNEL_ARCH:-}" \
        --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
        --build-arg "APPLY_SGL_KERNEL_ARCH_PRUNE=${APPLY_ARCH_PRUNE}" \
        --build-arg "APPLY_SGL_KERNEL_DISABLE_FA3=${APPLY_DISABLE_FA3}" \
        --build-arg "APPLY_SGL_KERNEL_SKIP_SM90_TARGET=${APPLY_SKIP_SM90_TARGET}" \
        --build-arg "APPLY_SGL_KERNEL_SKIP_FLASHMLA=${APPLY_SKIP_FLASHMLA}" \
        --build-arg "APPLY_SGL_KERNEL_SM121_DEBUG=${APPLY_SM121_DEBUG}" \
        -t "${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG}" \
        container-build/

    if ! podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${IMAGE_TAG}"; then
        die "Build finished but docker.io/${IMAGE_TAG} not present in remote image store — check podman build output above"
    fi
    echo "Remote build complete: ${IMAGE_TAG} (also tagged as docker.io/${IMAGE_TAG})"
}

# ============================================================================
# Transfer built image from remote to local
# ============================================================================

transfer_image_from_remote() {
    log "Copying docker.io/${IMAGE_TAG} from ${PODMAN_CONNECTION} to local image store"

    # If an older local copy exists under any of the possible tags, remove
    # it first so the save→load pipeline doesn't silently keep stale layers
    # around. `localhost/` is included because `podman load` of a short-name
    # RepoTag normalizes the reference to `localhost/...` — that tag would
    # otherwise linger across runs and keep old image layers dangling.
    podman image rm "${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "docker.io/${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "localhost/${IMAGE_TAG}" 2>/dev/null || true

    # Stream save → load so pv can show progress. Use the docker.io/ FQN
    # so the loaded image lands under the same name the downstream tools
    # (containerd, ansible, k3s) expect.
    #
    # Image size comes from `podman image inspect --format '{{.Size}}'` and
    # is passed to pv as -s so the progress bar can compute percent + ETA.
    # The raw byte total is echoed first — pv's -ptebar format does not
    # print the absolute target itself, only the running count and percent,
    # so without this line the user has no way to tell up-front how big
    # the transfer will be.
    local size size_human
    size=$(podman --connection "${PODMAN_CONNECTION}" image inspect \
            --format '{{.Size}}' "docker.io/${IMAGE_TAG}" 2>/dev/null || echo "")
    if [[ -n "${size}" ]] && command -v numfmt >/dev/null 2>&1; then
        size_human=$(numfmt --to=iec --suffix=B "${size}")
        echo "Transfer target: ${size_human} (${size} bytes)"
    elif [[ -n "${size}" ]]; then
        echo "Transfer target: ${size} bytes"
    else
        warn "Could not determine image size on ${PODMAN_CONNECTION}; pv will run without ETA"
    fi

    if command -v pv >/dev/null 2>&1; then
        local pv_args=(-ptebar)
        [[ -n "${size}" ]] && pv_args+=(-s "${size}")
        set -o pipefail
        podman --connection "${PODMAN_CONNECTION}" image save "docker.io/${IMAGE_TAG}" \
            | pv "${pv_args[@]}" \
            | podman image load \
            || die "streamed image transfer failed"
    else
        set -o pipefail
        podman --connection "${PODMAN_CONNECTION}" image save "docker.io/${IMAGE_TAG}" \
            | podman image load \
            || die "streamed image transfer failed"
    fi

    # `podman image save docker.io/${IMAGE_TAG} | podman image load` does NOT
    # preserve the `docker.io/` prefix on the receiving side: podman's loader
    # strips the registry component and re-applies the short name, which then
    # gets normalized to `localhost/${IMAGE_TAG}`. Retag unconditionally so
    # downstream `podman push docker.io/${IMAGE_TAG}` finds the image. `podman
    # tag` atomically moves the target tag, so this is safe even if the tag
    # already points elsewhere from a prior run.
    if podman image exists "localhost/${IMAGE_TAG}"; then
        podman tag "localhost/${IMAGE_TAG}" "docker.io/${IMAGE_TAG}"
    elif podman image exists "${IMAGE_TAG}"; then
        podman tag "${IMAGE_TAG}" "docker.io/${IMAGE_TAG}"
    fi

    podman image inspect "docker.io/${IMAGE_TAG}" >/dev/null \
        || die "Image not present locally after transfer — check podman output"
    echo "Image transferred: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Push (from x86, using x86's credentials)
# ============================================================================

run_push() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping push (--no-push)"
        return
    fi

    log "Pushing docker.io/${IMAGE_TAG} to Docker Hub from x86"

    # Podman looks for auth in $REGISTRY_AUTH_FILE or $XDG_RUNTIME_DIR/containers/auth.json
    # (rootless) or /run/containers/<uid>/auth.json. Do a soft check against
    # the most common location; if missing, let `podman push` produce the
    # definitive error.
    local auth_file="${REGISTRY_AUTH_FILE:-${XDG_RUNTIME_DIR:-/run}/containers/auth.json}"
    if [[ ! -f "${auth_file}" ]]; then
        # Legacy fallback: docker config, which podman also reads.
        auth_file="${HOME}/.docker/config.json"
    fi
    if [[ ! -f "${auth_file}" ]]; then
        warn "No registry auth file found (checked \$REGISTRY_AUTH_FILE and ~/.docker/config.json)"
        echo "Run 'podman login docker.io -u xomoxcc' on this host, then re-run."
        die "Registry authentication missing"
    fi

    podman push "docker.io/${IMAGE_TAG}"
    echo "Image pushed: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Next steps
# ============================================================================

print_next_steps() {
    if (( NO_LOCAL_COPY == 1 )); then
        cat <<EOF

$(log "Remote-only build complete")

Image: ${IMAGE_TAG}
Location: docker.io/${IMAGE_TAG} in ${PODMAN_CONNECTION}'s podman store only
          (NOT on this control host, NOT pushed to Docker Hub)

Next step — distribute to all 4 K3s nodes via the throwaway registry on
the build host (uses QSFP 200 GbE for the heavy transfers, source-fast-
path avoids the registry roundtrip for the build host itself):

  ./scripts/distrsm121image.sh --source ${REMOTE_HOST#*@} \\
      --registry-host <QSFP-IP of ${REMOTE_HOST#*@}>

Then redeploy:
  ansible-playbook k8s_dgx.yml --tags sglang

EOF
        return
    fi

    cat <<EOF

$(log "Build + push complete")

Image: ${IMAGE_TAG}

Next steps (on the x86 control host in ~/pythondev_workspace/dgxarley):

1. Bump default_sglang_image in roles/k8s_dgx/defaults/main/sglang.yml:
     sglang_image: "${IMAGE_TAG}"

2. Bump SGLANG_EXPECTED_IMAGE in roles/k8s_dgx/files/sglang_launch.sh:
     SGLANG_EXPECTED_IMAGE="${IMAGE_TAG}"

3. Smoke-test against a stable profile FIRST (recommended: Qwen3.6-35B-A3B-FP8,
   our Hermes/LiteLLM default). Confirms image boots + NCCL/RoCE init works
   + token output is coherent (pattern-grep + token distribution + tail
   eyeball; see feedback_output_quality_evidence memory).

4. Then run the NVFP4-MoE matrix per the current matrix doc (TODO_0.5.12.md
   until a 0.5.13 successor exists):
     glm-4.7 → glm-5 → qwen3-235b → nemotron-3
     → qwen3.5-397b-a17b-nvfp4 (crash candidate, re-test with flashinfer_cutlass)
     → minimax-m2.5 PP=4    (crash candidate, re-test with flashinfer_cutlass)
   For the two crash models also try moe_runner_backend=flashinfer_cute_dsl
   (Cute-DSL FP4 GEMM, reland #23590).

   If this is a *gemma4* and/or flashinfer-bumped image: ALSO run the
   head_dim=512 / Gemma-4 flashinfer-attention test (FLASHINFER_0.6.12_TODO.md
   §7) — flip attention_backend triton→flashinfer on the Gemma-4 profiles and
   verify boot + BOTH CUDA-graph paths (CG-on and --disable-cuda-graph) before
   dropping the triton workaround.

5. Deploy:
     ansible-playbook k8s_dgx.yml --tags sglang

6. Watch logs:
     kubectl --context=ht@dgxarley -n sglang logs -f <head-pod>

7. Record results in a TESTLOG named per reference_testlog memory
   (driver + image tag + model), e.g.:
     TESTLOG_nv<driver>_sglang-${IMAGE_TAG##*:}_<model>.md
   <driver> = the actual NVIDIA driver on the node (nvidia-smi). Note the
   exact flashinfer version in the log body — the image tag alone does not
   disambiguate flashinfer bumps. Peak throughput, not aggregate (peak_not_agg).

8. On any SGLang version bump, re-check launch scripts / profiles for renamed
   SGLANG_* env vars before rollout (e.g. the 0.5.11→0.5.12 rename
   SGLANG_USE_JIT_ALL_REDUCE → SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2, #24297).

If a profile regresses vs the previous baseline, roll back sglang_image to the
last known-good image for that workload:
  - lean:          xomoxcc/dgx-spark-sglang:0.5.11-sm121
  - gemma4-NVFP4:  xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121
  - or upstream scitrera/dgx-spark-sglang:0.5.11 / 0.5.12 if SM121 patches
    aren't critical for that workload.

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    preflight
    ensure_podman_connection
    ensure_base_image_present
    prepare_cuda_containers
    apply_patches
    run_build
    if (( NO_LOCAL_COPY == 0 )); then
        transfer_image_from_remote
        run_push
    else
        log "Skipping local copy + push (--no-local-copy) — image stays on ${PODMAN_CONNECTION}"
    fi
    print_next_steps
}

main "$@"
