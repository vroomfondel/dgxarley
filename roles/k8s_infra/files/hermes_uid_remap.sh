#!/bin/sh
# Entrypoint for every hermes-agent-image container sharing /opt/data: pre-sets the
# `hermes` UID/GID so cont-init stage2-hook.sh skips `usermod -u`, whose chown walk
# of the shared home races the sibling container and aborts cont-init (exit 12).
# Then hands over to the image ENTRYPOINT, which must stay PID 1 (s6-overlay).
set -eu
sed -i -E "s/^(hermes:[^:]*:)[0-9]+:[0-9]+:/\1${HERMES_UID}:${HERMES_GID}:/" /etc/passwd
sed -i -E "s/^(hermes:[^:]*:)[0-9]+:/\1${HERMES_GID}:/" /etc/group
exec /opt/hermes/docker/entrypoint-dispatch.sh "$@"
