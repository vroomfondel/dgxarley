#!/bin/sh
# Defaults the prefix to the persistent /opt/data and `install` to the base env, so `micromamba install X` just works.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/opt/data/.micromamba}"

if [ "${1:-}" = "install" ] || [ "${1:-}" = "remove" ] || [ "${1:-}" = "update" ]; then
    case " $* " in
        *" -n "*|*" --name "*|*" -p "*|*" --prefix "*) ;;
        *) cmd="$1"; shift; set -- "$cmd" -n base "$@" ;;
    esac
fi

exec /usr/local/libexec/micromamba "$@"
