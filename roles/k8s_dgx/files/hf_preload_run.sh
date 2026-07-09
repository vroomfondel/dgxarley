#!/bin/bash
set -e

apt-get update -qq && apt-get install -y -qq tini net-tools iputils-ping iproute2 curl >/dev/null 2>&1

if [ -n "$RSYNC_TARGETS" ]; then
  apt-get install -y -qq rsync openssh-client >/dev/null 2>&1
fi

exec tini -- python3 /scripts/download_models.py
