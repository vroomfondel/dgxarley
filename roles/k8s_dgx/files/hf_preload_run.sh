#!/bin/bash
set -e
if [ -n "$RSYNC_TARGETS" ]; then
  apt-get update -qq && apt-get install -y -qq tini rsync openssh-client net-tools iputils-ping iproute2 curl >/dev/null 2>&1
fi

exec tini -- python3 /scripts/download_models.py
