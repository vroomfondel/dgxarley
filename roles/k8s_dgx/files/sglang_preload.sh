#!/usr/bin/env bash
# SGLang model-preload initContainer script (ConfigMap-mounted at /scripts).
#
# Extracted from the inline `bash -c` heredoc in sglang_instance.yml so the SAME
# logic is shared by the head + worker Deployments (no duplicated heredoc, no
# Jinja `{% %}` at column 0). The one template-time value it needs is passed as
# an env var (JFS_CACHE); everything else is pure bash reading env + mounts.
#
# Ensures the active model(s) are PRESENT for this pod; it does NOT warm the
# local JuiceFS blockcache -- that is the next initContainer, jfs-warmup, which
# blocks-until-100% WITH live progress.
#
# Env:
#   JFS_CACHE=yes|no      is the HF hub cache on the shared JuiceFS mount?
#   SGLANG_ROLE=head|worker
#   HF_PRELOAD_MODELS     comma-list: primary model (+ MTP/draft path)
# Mounts: /scripts/download_models.py, /root/.cache/huggingface (hub),
#         /usr/local/bin/juicefs (only when JFS_CACHE=yes).
set -u
MARKDIR=/root/.cache/huggingface/hub/.preload-done

if [ "${JFS_CACHE:-no}" != "yes" ]; then
  # Local per-node cache: every pod downloads itself.
  exec python3 /scripts/download_models.py
fi

if [ "${SGLANG_ROLE:-}" = "head" ]; then
  # Shared JuiceFS: the HEAD is the single downloader (one writer to the shared
  # dir), then drops a per-model done-marker for the workers.
  python3 /scripts/download_models.py || exit $?
  mkdir -p "$MARKDIR"
  for m in $(printf '%s' "$HF_PRELOAD_MODELS" | tr ',' ' '); do
    touch "$MARKDIR/$(printf '%s' "$m" | sed 's#/#--#g')"
  done
  echo "[preload] head download complete; markers written under $MARKDIR/"
  # Warming THIS node's local JuiceFS cache is deferred to the dedicated
  # jfs-warmup initContainer (runs next), which warms WITH live cache-fill %
  # progress and BLOCKS until 100%. Markers are written BEFORE we exit so the
  # workers unblock and warm in parallel.
  exit 0
fi

# Worker on shared JuiceFS: do NOT download (avoids a concurrent-write race on
# the shared dir + redundant work). Just WAIT for the head's per-model done-
# markers, kicking a BACKGROUND warmup so warming overlaps the wait and printing
# the cache-fill % each round. The BLOCKING warm-until-100% guarantee (with
# progress) is the NEXT initContainer, jfs-warmup. Non-fatal 30min marker
# timeout -> jfs-warmup / sglang then fail loudly on a genuinely missing model.
MODELS=$(printf '%s' "$HF_PRELOAD_MODELS" | tr ',' ' ')
echo "[preload-wait] shared JuiceFS -> waiting for head preload marker(s) under ${MARKDIR}/ (<=30min); background-warming"
for m in $MODELS; do
  echo "[preload-wait]   marker for '$m' -> ${MARKDIR}/$(printf '%s' "$m" | sed 's#/#--#g')"
done
start=$(date +%s); deadline=$(( start + 1800 ))
HAVE_JFS=no; [ -x /usr/local/bin/juicefs ] && HAVE_JFS=yes
if [ "$HAVE_JFS" = yes ]; then
  for m in $MODELS; do
    d="/root/.cache/huggingface/hub/models--$(printf '%s' "$m" | sed 's#/#--#g')"
    [ -d "$d" ] && /usr/local/bin/juicefs warmup --threads 1 --background "$d" >/dev/null 2>&1 || true
  done
fi
while :; do
  now=$(date +%s); elapsed=$(( now - start )); pending=
  for m in $MODELS; do
    mk="$MARKDIR/$(printf '%s' "$m" | sed 's#/#--#g')"
    [ -e "$mk" ] || pending="$pending $mk"
  done
  [ -z "$pending" ] && { echo "[preload-wait] all markers present after ${elapsed}s"; break; }
  [ "$now" -ge "$deadline" ] && { echo "[preload-wait] TIMEOUT after ${elapsed}s; still missing marker(s):$pending (non-fatal)"; break; }
  if [ "$HAVE_JFS" = yes ]; then
    for m in $MODELS; do
      d="/root/.cache/huggingface/hub/models--$(printf '%s' "$m" | sed 's#/#--#g')"
      [ -d "$d" ] || continue
      pct=$(/usr/local/bin/juicefs warmup --check "$d" 2>&1 | grep -oE '\([0-9.]+%\)' | tail -1)
      echo "[preload-wait] ${elapsed}s; missing marker(s):$pending; warming $(basename "$d") cached=${pct:-?}"
    done
  else
    echo "[preload-wait] ${elapsed}s; missing marker(s):$pending"
  fi
  sleep 20
done
# Warm-until-100% is enforced by the jfs-warmup initContainer (next), which
# reports live progress -- no silent blocking warmup here.
exit 0
