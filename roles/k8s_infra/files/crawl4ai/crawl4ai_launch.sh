#!/usr/bin/env bash
# Container start for Crawl4AI with the usermap hooks. Runs as root, patches /app in place, drops to appuser, execs the
# image's own start command. See README.md. Never starts the server as root.
set -u

APP_DIR="${CRAWL4AI_APP_DIR:-/app}"
PATCH_DIR="${CRAWL4AI_PATCH_DIR:-/patches}"
RUN_AS="${CRAWL4AI_RUN_AS:-appuser}"
PY="${CRAWL4AI_PYTHON:-python3}"
CHECKED_VERSION="0.9.5"
HOOKED="api.py auth_gate.py llm_broker.py mcp_bridge.py server.py"

log() { echo "[crawl4ai-launch] $*" >&2; }

usermap_on() {
  case "$(printf '%s' "${CRAWL4AI_USERMAP:-off}" | tr '[:upper:]' '[:lower:]')" in
    on | true | 1 | yes) return 0 ;;
    *) return 1 ;;
  esac
}

# Runs every patch script against $1 and the verifier; 0 only if nothing drifted.
patch_tree() {
  local root="$1" rc=0 n=0 p out
  for p in "$PATCH_DIR"/c[0-9][0-9]_*.py; do
    [ -e "$p" ] || continue
    n=$((n + 1))
    out=$(CRAWL4AI_PATCH_ROOT="$root" PYTHONPATH="$PATCH_DIR" "$PY" "$p" 2>&1) || rc=1
    printf '%s\n' "$out" | sed 's/^/[crawl4ai-launch]   /' >&2
    case "$out" in *ANCHOR-DRIFT*) rc=1 ;; esac
  done
  if [ "$n" -eq 0 ]; then
    log "no patch scripts (c[0-9][0-9]_*.py) in $PATCH_DIR"
    rc=1
  fi
  "$PY" "$PATCH_DIR/crawl4ai_verify_hooks.py" "$root" >&2 || rc=1
  return "$rc"
}

restore_originals() {
  local f
  for f in $HOOKED; do cp "$1/orig/$f" "$APP_DIR/$f" 2>/dev/null; done
}

# Dry run on a scratch copy first; only if every anchor matches is /app written. Failure after the real write restores
# the originals, so /app is never left half patched.
apply_hooks() {
  local tmp dirmode f
  tmp=$(mktemp -d) || return 1
  mkdir "$tmp/dry" "$tmp/orig" || return 1
  cp "$APP_DIR"/*.py "$tmp/orig/" || return 1
  cp "$tmp"/orig/*.py "$tmp/dry/"
  chmod -R u+w "$tmp/dry" "$tmp/orig"
  cp "$PATCH_DIR/crawl4ai_usermap.py" "$tmp/dry/" || return 1

  log "dry run on a scratch copy"
  if ! patch_tree "$tmp/dry"; then
    rm -rf "$tmp"
    return 1
  fi

  dirmode=$(stat -c %a "$APP_DIR")
  if ! chmod u+w "$APP_DIR"; then
    log "cannot make $APP_DIR writable (not running as its owner?)"
    rm -rf "$tmp"
    return 1
  fi
  for f in $HOOKED crawl4ai_usermap.py; do chmod u+w "$APP_DIR/$f" 2>/dev/null; done

  log "applying to $APP_DIR"
  if ! cp "$PATCH_DIR/crawl4ai_usermap.py" "$APP_DIR/crawl4ai_usermap.py" || ! patch_tree "$APP_DIR"; then
    log "real run failed after a clean dry run, restoring the original files"
    restore_originals "$tmp"
    rm -f "$APP_DIR/crawl4ai_usermap.py"
    for f in $HOOKED; do chmod a-w "$APP_DIR/$f" 2>/dev/null; done
    chmod "$dirmode" "$APP_DIR"
    rm -rf "$tmp"
    return 1
  fi

  for f in $HOOKED crawl4ai_usermap.py; do chmod 444 "$APP_DIR/$f"; done
  chmod "$dirmode" "$APP_DIR"
  rm -rf "$tmp"
  return 0
}

installed=$("$PY" -c 'import importlib.metadata as m; print(m.version("Crawl4AI"))' 2>/dev/null || echo unknown)
if [ "$installed" != "$CHECKED_VERSION" ]; then
  log "WARNING: Crawl4AI $installed running; hooks were last checked against $CHECKED_VERSION"
fi

if apply_hooks; then
  log "usermap hooks applied (CRAWL4AI_USERMAP=${CRAWL4AI_USERMAP:-off})"
elif usermap_on; then
  log "ERROR: CRAWL4AI_USERMAP is on but the hooks do not apply (see ANCHOR-DRIFT / HOOK-MISSING lines above). Not starting."
  exit 1
else
  log "WARNING: hooks do not apply (see lines above); starting UNPATCHED like upstream (CRAWL4AI_USERMAP is off)"
fi

home_dir=$(getent passwd "$RUN_AS" 2>/dev/null | cut -d: -f6)
home_dir="${home_dir:-/home/$RUN_AS}"
read -r -a START <<<"${CRAWL4AI_START_CMD:-bash entrypoint.sh}"
if [ -n "${CRAWL4AI_DROP_CMD:-}" ]; then
  read -r -a DROP <<<"$CRAWL4AI_DROP_CMD"
elif [ "$(id -u)" = "0" ]; then
  if command -v setpriv >/dev/null 2>&1; then
    DROP=(setpriv "--reuid=$RUN_AS" "--regid=$RUN_AS" --init-groups)
  elif command -v runuser >/dev/null 2>&1; then
    DROP=(runuser -u "$RUN_AS" --)
  else
    log "ERROR: running as root and neither setpriv nor runuser exists; refusing to start the server as root"
    exit 1
  fi
  # supervisord re-opens /dev/stdout and /dev/stderr for redis and gunicorn (stdout_logfile=/dev/stdout). Started as
  # root, those are pipes owned by root, so after the drop the open fails with EACCES ("spawnerr: unknown error
  # making dispatchers", seen on the first rollout). root owns the pipes, so a plain chmod needs no capability.
  # /proc/$$/fd, not /proc/self/fd: "self" would be the chmod process, whose fd 2 is the 2>/dev/null redirect below.
  for _fd in 1 2; do
    chmod a+w "/proc/$$/fd/$_fd" 2>/dev/null || log "WARNING: could not make fd $_fd writable for $RUN_AS"
  done
else
  DROP=()
fi

cd "$APP_DIR" || exit 1
exec "${DROP[@]}" env "HOME=$home_dir" "USER=$RUN_AS" "LOGNAME=$RUN_AS" "${START[@]}"
