#!/usr/bin/env bash
#
# verify_sglang_image.sh — image acceptance gate for xomoxcc/dgx-spark-sglang:*
#
# Runs the dgxarley runtime patch set exactly the way sglang_launch.sh runs it
# (roles/k8s_dgx/files/sglang_patches/p[0-9][0-9]_*.py, filename order), then
# asks the two questions the patch run itself CANNOT answer:
#
#   1. Did any patch report ANCHOR-DRIFT?
#      -> an anchor moved; the fix silently does not happen any more.
#   2. Does SGLang's model registry still import cleanly?
#      -> THE check that matters. On 2026-07-28 p30 generated a module importing
#         sglang.srt.layers.quantization.fp8_kernel, which RFC #29630 moved in
#         v0.5.16. The patch run said "ok". The registry swallowed the
#         ImportError ("Ignore import error when loading ...") and 31 model
#         classes were silently disabled: deepseek_v2, deepseek_v4, glm4_moe,
#         kimi_*, mistral_large_3, pixtral, ... Nothing crashed, nothing warned
#         at the top level, and the models simply were not there any more.
#
#   3. (qwen4_exp images only) Does the Qwen3.8-Flash-Next architecture import,
#      AND is the SM121 trtllm veto from p65 actually live?
#      -> the veto is the one check here that guards CORRECTNESS rather than
#         availability. With it missing, a GB10 pod serves fine on short
#         prompts and returns runs of "!" past roughly 120k tokens of context,
#         silently and stochastically (upstream #36716 / #36558; measured 4/4
#         corrupt at 210k on 2x Spark with real weights). Nothing in the patch
#         log or the model registry can see that, which is why it is its own
#         check. Set EXPECT_QWEN4EXP=1 to make the ABSENCE of qwen4_exp a
#         failure too (use it for the 0.5.18-sm121 image, whose recipe sets
#         APPLY_QWEN4EXP_PR36497=1); left unset, the section self-skips on
#         images that were never meant to carry the architecture.
#
# "Applies cleanly" is not "works". Run this before promoting any image, and
# after every change under sglang_patches/.
#
# No GPU and no k3s needed: plain podman, CPU only, ~1 minute.
#
# Usage:
#   scripts/verify_sglang_image.sh <image> [podman-connection]
#
#   scripts/verify_sglang_image.sh xomoxcc/dgx-spark-sglang:0.5.16-sm121
#   scripts/verify_sglang_image.sh xomoxcc/dgx-spark-sglang:0.5.16-sm121 spark5
#   EXPECT_QWEN4EXP=1 scripts/verify_sglang_image.sh \
#       xomoxcc/dgx-spark-sglang:0.5.18-sm121 spark5
#
# With a connection name the checks run on that remote podman host (the images
# live in spark5's store), otherwise locally.
#
# Exit codes: 0 = clean, 1 = drift or registry damage, 2 = usage/plumbing error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_SRC="${REPO_ROOT}/roles/k8s_dgx/files/sglang_patches"

IMAGE="${1:-}"
CONNECTION="${2:-}"
[[ -n "${IMAGE}" ]] || { sed -n '2,40p' "$0"; exit 2; }
[[ -d "${PATCH_SRC}" ]] || { echo "ERROR: patch dir not found: ${PATCH_SRC}" >&2; exit 2; }

# Scenarios mirror the model-name gates in the patch set (gate_model()), so the
# model-specific patches are exercised too, not just the ungated majority.
SCENARIOS=(
    "bare::"
    "glm:SGLANG_MODEL=zai-org/GLM-5.2-NVFP4:SGLANG_DSA_INDEXER_TRITON=1"
    "hy3:SGLANG_MODEL=tencent/Hy3-NVFP4:SGLANG_HUNYUAN_TOKEN_SUFFIX=1"
)

run_podman() {
    if [[ -n "${CONNECTION}" ]]; then
        podman --connection "${CONNECTION}" "$@"
    else
        podman "$@"
    fi
}

# The patch set has to be inside the podman host's filesystem. With a remote
# connection, ship it over first (tar over ssh, same host the connection names).
REMOTE_PATCHES="/tmp/dgxarley-verify-patches"
if [[ -n "${CONNECTION}" ]]; then
    ssh_host="$(podman system connection list --format '{{.Name}} {{.URI}}' \
        | awk -v c="${CONNECTION}" '$1==c {print $2}' \
        | sed -E 's#^ssh://([^/]+)/.*#\1#; s#:[0-9]+$##')"
    [[ -n "${ssh_host}" ]] || { echo "ERROR: cannot resolve ssh host for podman connection '${CONNECTION}'" >&2; exit 2; }
    tar -C "${PATCH_SRC}/.." -cf - "$(basename "${PATCH_SRC}")" \
        | ssh -o BatchMode=yes "${ssh_host}" \
            "rm -rf ${REMOTE_PATCHES} && mkdir -p ${REMOTE_PATCHES} && tar -C /tmp -xf - \
             && cp /tmp/$(basename "${PATCH_SRC}")/*.py ${REMOTE_PATCHES}/ \
             && rm -rf ${REMOTE_PATCHES}/__pycache__"
    PATCH_MOUNT="${REMOTE_PATCHES}"
else
    PATCH_MOUNT="${PATCH_SRC}"
fi

echo "=== verifying ${IMAGE}${CONNECTION:+ (on ${CONNECTION})}"
failures=0

# BASELINE: which model classes does the UNPATCHED image already fail to import?
# Upstream ships modules with optional dependencies (bailing_moe_* import vllm,
# which we do not install), so a raw count would fail every image forever. Only
# the DELTA introduced by our patch set is a defect, so measure it as a delta.
registry_failures() { # stdin: registry output -> stdout: sorted module names
    grep 'Ignore import error' | sed -E 's/.*loading (sglang[^:]*):.*/\1/' | sort -u
}
baseline="$(run_podman run --rm "${IMAGE}" python3 -c 'import sglang.srt.models.registry' 2>&1 \
    | registry_failures || true)"
if [[ -n "${baseline}" ]]; then
    echo "  note: $(wc -l <<< "${baseline}") model class(es) already fail to import WITHOUT our patches"
    sed 's/^/        /' <<< "${baseline}"
    echo "        (upstream optional deps — excluded from the check below)"
fi

for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r label env1 env2 <<< "${scenario}"
    env_args=()
    [[ -n "${env1}" ]] && env_args+=(-e "${env1}")
    [[ -n "${env2}" ]] && env_args+=(-e "${env2}")

    echo
    echo "--- scenario: ${label}"
    out="$(run_podman run --rm "${env_args[@]}" -v "${PATCH_MOUNT}:/patches:ro" "${IMAGE}" bash -c '
        for p in /patches/p[0-9][0-9]_*.py; do python3 "$p" 2>&1; done
        echo "###REGISTRY###"
        python3 -c "import sglang.srt.models.registry" 2>&1
    ' 2>/dev/null)" || true

    patch_phase="${out%%###REGISTRY###*}"
    registry_phase="${out#*###REGISTRY###}"

    drift="$(grep -c 'ANCHOR-DRIFT' <<< "${patch_phase}" || true)"

    if [[ "${drift}" -gt 0 ]]; then
        echo "  FAIL  ${drift} ANCHOR-DRIFT:"
        grep 'ANCHOR-DRIFT' <<< "${patch_phase}" | sed 's/^/        /'
        failures=$((failures + 1))
    else
        echo "  ok    0 ANCHOR-DRIFT"
    fi

    patched_failures="$(registry_failures <<< "${registry_phase}" || true)"
    regressions="$(comm -13 <(echo "${baseline}") <(echo "${patched_failures}") || true)"

    if [[ -n "${regressions}" ]]; then
        echo "  FAIL  $(wc -l <<< "${regressions}") model class(es) broken BY OUR PATCHES:"
        sed 's/^/        /' <<< "${regressions}"
        echo "        (a generated/patched module is unimportable — these are GONE at runtime)"
        failures=$((failures + 1))
    else
        echo "  ok    model registry: no regression vs the unpatched image"
    fi
done

# ---------------------------------------------------------------------------
# qwen4_exp / Qwen3.8-Flash-Next gate.
#
# Two questions the loop above structurally cannot answer. The registry check
# only reports modules that FAIL to import, so an architecture that was never
# built into the image looks exactly like one that is fine. And ANCHOR-DRIFT
# says an anchor matched, not that the resulting code does the right thing —
# p65's SM121 veto is a correctness guard whose absence is invisible until a
# real 190k-token request comes back as exclamation marks.
#
# Runs the patch set once more with the model gate set to the Flash-Next
# checkpoint, then probes the PATCHED module in-process. The trtllm probe
# stubs flashinfer and monkeypatches the capability helpers, so it needs no GPU
# and no flashinfer install: it asserts the resolver returns None on cc (12,1)
# and a callable on cc (12,0), which is exactly the split upstream #36806 and
# our veto encode.
echo
echo "--- qwen4_exp / Qwen3.8-Flash-Next"
qwen4_out="$(run_podman run --rm \
    -e SGLANG_MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    -v "${PATCH_MOUNT}:/patches:ro" "${IMAGE}" bash -c '
    for p in /patches/p[0-9][0-9]_*.py; do python3 "$p" 2>&1; done
    echo "###PROBE###"
    python3 - <<"PY" 2>&1
import importlib, sys, types

def emit(tag, ok, detail=""):
    print(f"{tag}={'1' if ok else '0'} {detail}")

try:
    importlib.import_module("sglang.srt.models.qwen4_exp")
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration  # noqa
    emit("ARCH", True)
except Exception as exc:  # noqa: BLE001
    emit("ARCH", False, f"{type(exc).__name__}: {exc}")
    sys.exit(0)

try:
    from sglang.srt.models.registry import ModelRegistry
    names = set(getattr(ModelRegistry, "models", {}) or {})
    emit("REGISTRY", "Qwen4ExpForConditionalGeneration" in names,
         "" if "Qwen4ExpForConditionalGeneration" in names else "not in ModelRegistry.models")
except Exception as exc:  # noqa: BLE001
    emit("REGISTRY", False, f"{type(exc).__name__}: {exc}")

# Functional probe of the decode resolver: no GPU, no flashinfer needed.
try:
    sentinel = object()
    fi = types.ModuleType("flashinfer")
    fid = types.ModuleType("flashinfer.decode")
    fid.trtllm_batch_decode_with_kv_cache = sentinel
    fi.decode = fid
    sys.modules.setdefault("flashinfer", fi)
    sys.modules["flashinfer.decode"] = fid

    qsa = importlib.import_module(
        "sglang.srt.layers.attention.qwen_sparse_attn_backend")
    import sglang.srt.utils as U
    import sglang.srt.utils.common as C

    def force(cap):
        for mod in (U, C):
            for name, val in (
                ("is_sm100_supported", lambda: False),
                ("is_sm120_supported", lambda: cap[0] == 12),
                ("is_sm120", lambda: cap == (12, 0)),
                ("is_sm121", lambda: cap == (12, 1)),
            ):
                if hasattr(mod, name):
                    setattr(mod, name, val)
        qsa._resolve_trtllm_sparse_decode.cache_clear()
        return qsa._resolve_trtllm_sparse_decode()

    on_121 = force((12, 1))
    on_120 = force((12, 0))
    emit("VETO_SM121", on_121 is None,
         "" if on_121 is None else "resolver returned a kernel on cc (12,1)")
    emit("SM120_KEPT", on_120 is sentinel,
         "" if on_120 is sentinel else f"resolver returned {on_120!r} on cc (12,0)")
except Exception as exc:  # noqa: BLE001
    emit("VETO_SM121", False, f"{type(exc).__name__}: {exc}")
PY
' 2>/dev/null)" || true

qwen4_patch_phase="${qwen4_out%%###PROBE###*}"
qwen4_probe="${qwen4_out#*###PROBE###}"
probe_val() { grep -oP "(?<=^$1=)[01]" <<< "${qwen4_probe}" | head -1; }

if [[ "$(probe_val ARCH)" != "1" ]]; then
    if [[ "${EXPECT_QWEN4EXP:-0}" == "1" ]]; then
        echo "  FAIL  qwen4_exp is NOT in this image, but EXPECT_QWEN4EXP=1"
        grep -E "^ARCH=" <<< "${qwen4_probe}" | sed 's/^/        /'
        echo "        (recipe must set APPLY_QWEN4EXP_PR36497=1 — see"
        echo "         scripts/patches/sglang-qwen4exp-pr36497.patch)"
        failures=$((failures + 1))
    else
        echo "  skip  no qwen4_exp in this image (set EXPECT_QWEN4EXP=1 to require it)"
    fi
else
    echo "  ok    qwen4_exp imports"

    if [[ "$(probe_val REGISTRY)" == "1" ]]; then
        echo "  ok    Qwen4ExpForConditionalGeneration is in the model registry"
    else
        echo "  FAIL  Qwen4ExpForConditionalGeneration missing from the model registry"
        grep -E "^REGISTRY=" <<< "${qwen4_probe}" | sed 's/^/        /'
        failures=$((failures + 1))
    fi

    # p65 must have RUN, not been gated out: the file exists in this image, so
    # "gate not matched" here means target_contains failed, i.e. the resolver
    # was renamed and the veto is silently absent.
    if grep -qE "gate not matched" <<< "$(grep -i "qsa" <<< "${qwen4_patch_phase}")"; then
        echo "  FAIL  p65 reported 'gate not matched' on an image that HAS qwen4_exp"
        failures=$((failures + 1))
    fi

    if [[ "$(probe_val VETO_SM121)" == "1" ]]; then
        echo "  ok    trtllm sparse decode is vetoed on cc (12,1) / GB10"
    else
        echo "  FAIL  trtllm sparse decode is REACHABLE on cc (12,1) — silent"
        echo "        long-context corruption (token-0 runs) on every Spark."
        grep -E "^VETO_SM121=" <<< "${qwen4_probe}" | sed 's/^/        /'
        echo "        (p65_qsa_sm121_sparse_decode.py edit 1 did not take effect)"
        failures=$((failures + 1))
    fi

    if [[ "$(probe_val SM120_KEPT)" == "1" ]]; then
        echo "  ok    cc (12,0) still reaches the trtllm kernel (veto is not too wide)"
    else
        echo "  warn  cc (12,0) no longer reaches the trtllm kernel"
        grep -E "^SM120_KEPT=" <<< "${qwen4_probe}" | sed 's/^/        /'
        echo "        (not fatal on this cluster — no SM120 part here — but it"
        echo "         means the shipped gate changed shape; re-read p65)"
    fi
fi

echo
if [[ "${failures}" -eq 0 ]]; then
    echo "RESULT: clean — ${IMAGE} passes the patch-set acceptance gate"
    exit 0
fi
echo "RESULT: ${failures} failing check(s) — do NOT promote ${IMAGE}"
exit 1
