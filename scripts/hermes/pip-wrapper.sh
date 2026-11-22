#!/bin/bash
# pip for the hermes venv: installs into the per-user target on /opt/data that hermes_user_site.pth appends to sys.path.
set -euo pipefail

PY=/opt/hermes/.venv/bin/python
UV=/usr/local/libexec/uv
CONSTRAINTS=/usr/local/share/hermes-pip/constraints.txt
BIN_DIR=/opt/data/.local/bin

if [ "$(id -u)" = "0" ]; then
    echo "pip: run as the hermes user, not root (the target on /opt/data must stay user-owned)" >&2
    exit 1
fi

T="$("$PY" -c 'import hermes_user_site; print(hermes_user_site.path)')"
mkdir -p "$T" "$BIN_DIR"

cmd="${1:-help}"
[ $# -gt 0 ] && shift

args=()
for a in "$@"; do
    case "$a" in
        --user|-y|--yes|--no-warn-script-location|--disable-pip-version-check|--root-user-action=*|--no-input) ;;
        *) args+=("$a") ;;
    esac
done

relink_bins() {
    find "$BIN_DIR" -maxdepth 1 -type l -lname "$T/bin/*" ! -exec test -e {} \; -delete
    if [ -d "$T/bin" ]; then
        for f in "$T"/bin/*; do
            [ -e "$f" ] && ln -sfn "$f" "$BIN_DIR/$(basename "$f")"
        done
    fi
    return 0
}

case "$cmd" in
    install)
        "$UV" pip install --python "$PY" --target "$T" -c "$CONSTRAINTS" "${args[@]}"
        relink_bins
        ;;
    uninstall)
        "$UV" pip uninstall --python "$PY" --target "$T" "${args[@]}"
        relink_bins
        ;;
    list|freeze|show)
        "$UV" pip "$cmd" --python "$PY" --target "$T" "${args[@]}"
        ;;
    -V|--version)
        echo "pip shim ($("$UV" --version)), target $T"
        ;;
    *)
        cat >&2 <<EOF
pip shim for the Hermes Python ($("$PY" -V)); backed by uv, no root needed.
  pip install <pkg>...      installs into $T (persistent, importable right away)
  pip uninstall <pkg>...
  pip list | freeze | show
Packages already shipped with Hermes keep their version (install fails with a resolver error on a conflict).
'python -m pip' does not exist here, use 'pip'.
EOF
        [ "$cmd" = "help" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ] || exit 2
        ;;
esac
