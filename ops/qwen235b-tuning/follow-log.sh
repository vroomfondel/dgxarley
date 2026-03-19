#!/bin/bash
# Follow a ralph iteration log with colored output
# Usage: ./ops/ceph-monitor/follow-log.sh <logfile>
#        ./ops/ceph-monitor/follow-log.sh latest

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"

if [ -z "$1" ] || [ "$1" = "latest" ]; then
  LOG_FILE=$(ls -t "$LOGS_DIR"/iteration-*.log 2>/dev/null | head -1)
  if [ -z "$LOG_FILE" ]; then
    echo "No log files found in $LOGS_DIR"
    exit 1
  fi
else
  LOG_FILE="$1"
fi

echo "Following: $LOG_FILE"
echo "---"

tail -f "$LOG_FILE" | jq -r '
  if .type == "assistant" then
    (.message.content[]? |
      if .type == "text" then "\u001b[32m>>> \(.text)\u001b[0m"
      elif .type == "tool_use" then "\u001b[33m--- \(.name): \(.input | tostring | .[0:200])\u001b[0m"
      else empty end)
  elif .type == "result" then
    "\u001b[36m=== Result: \(.result // .error | tostring | .[0:300])\u001b[0m"
  else empty end
'
