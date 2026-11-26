#!/bin/bash
# Runs dgxarley.k3shelperstuff.drift (pin / keel) from the dgxarley package against this repo and the dgxarley cluster.
#
#   scripts/check_pin_drift.sh --help                       # both tools; pin --help / keel --help for their options
#   scripts/check_pin_drift.sh [pin] [pin-drift options]    # version pins in pin_drift.yml vs. upstream releases
#   scripts/check_pin_drift.sh keel [keel-drift options]    # running pod digests vs. registry digests (cluster)
#
# Env overrides:
#   DGXARLEY_PYTHON   interpreter with dgxarley installed     (default: <repo>/.venv/bin/python, else uvx dgxarley from PyPI)
#   DGXARLEY_SPEC     uvx package spec for the PyPI fallback  (default: dgxarley[k3s])
#   KUBE_CONTEXT      kubeconfig context for keel + pin's running column (default: ht@dgxarley)
#   PIN_DRIFT_CONFIG  pin only; pin declarations             (default: <repo>/pin_drift.yml)
#   GITHUB_TOKEN      pin only; falls back to GH_TOKEN, then `gh auth token`
set -euo pipefail

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly KUBE_CONTEXT="${KUBE_CONTEXT:-ht@dgxarley}"
readonly SPEC="${DGXARLEY_SPEC:-dgxarley[k3s]}"
export KUBE_CONTEXT
export DRIFT_PROG_NAME=scripts/check_pin_drift.sh
readonly module=dgxarley.k3shelperstuff.drift

tool=pin
case "${1:-}" in
  pin|keel) tool="$1"; shift ;;
  -h|--help|help) tool=help; shift ;;
esac

case "${tool}" in
  pin)
    if [ -z "${GITHUB_TOKEN:-}" ] && [ -z "${GH_TOKEN:-}" ] && command -v gh >/dev/null; then
      GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
    fi
    export GITHUB_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
    [ -n "${GITHUB_TOKEN}" ] || echo "WARN: no GitHub token, anonymous API limit is 60 requests/h" >&2
    export PIN_DRIFT_CONFIG="${PIN_DRIFT_CONFIG:-${REPO_DIR}/pin_drift.yml}"
    ;;
esac

[ "${tool}" = help ] && set -- --help || set -- "${tool}" "$@"

python="${DGXARLEY_PYTHON:-${REPO_DIR}/.venv/bin/python}"
if [ -x "${python}" ] && "${python}" -c "import ${module}" 2>/dev/null; then
  exec "${python}" -m "${module}" "$@"
fi

command -v uvx >/dev/null || { echo "ERROR: ${python} lacks dgxarley and uvx is not installed" >&2; exit 2; }
echo "INFO: using ${SPEC} from PyPI via uvx" >&2
exec uvx --from "${SPEC}" python -m "${module}" "$@"
