#!/bin/bash
# Qwen 235B Parameter Tuning Loop (Ralph-style)
# Usage: ./ops/qwen235b-tuning/ralph-qwen235b-tuning.sh [max_iterations]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/CLAUDE.md"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
PRD_FILE="$SCRIPT_DIR/prd.json"
LOGS_DIR="$SCRIPT_DIR/logs"
MAX_ITERATIONS="${1:-20}"

# Check required tools
MISSING=""
for tool in claude tini jq; do
  command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
if [ -n "$MISSING" ]; then
  echo "ERROR: Missing required tools:$MISSING"
  exit 1
fi

cd "$REPO_ROOT"

# Activate .venv for ansible-playbook and python dependencies
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  source "$REPO_ROOT/.venv/bin/activate"
  echo "Activated virtualenv: $VIRTUAL_ENV"
else
  echo "WARNING: No .venv found at $REPO_ROOT/.venv — ansible-playbook may fail"
fi

# Check SGLANG_URL is set
if [ -z "$SGLANG_URL" ]; then
  echo "ERROR: SGLANG_URL not set. Export it before running, e.g.:"
  echo "  export SGLANG_URL=http://192.168.191.x:8000"
  exit 1
fi
echo "SGLANG_URL=$SGLANG_URL"

# Cleanup all child processes on exit (Ctrl+C, errors, etc.)
WATCHER_PID=""
TINI_PID=""
cleanup() {
  echo ""
  echo "Cleaning up child processes..."
  # tini -g forwards signals to entire process group — just kill tini
  [ -n "$TINI_PID" ] && kill "$TINI_PID" 2>/dev/null
  sleep 1
  [ -n "$TINI_PID" ] && kill -9 "$TINI_PID" 2>/dev/null
  [ -n "$WATCHER_PID" ] && kill "$WATCHER_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Qwen 235B Parameter Tuning Progress" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

mkdir -p "$LOGS_DIR"

echo "Starting Qwen 235B Parameter Tuning Loop - Max iterations: $MAX_ITERATIONS"
echo "Working directory: $REPO_ROOT"
echo "Logs directory: $LOGS_DIR"

for i in $(seq 1 "$MAX_ITERATIONS"); do
  echo ""
  echo "==============================================================="
  echo "  Qwen Tuning Iteration $i of $MAX_ITERATIONS"
  echo "==============================================================="

  LOG_FILE="$LOGS_DIR/iteration-$(printf '%03d' "$i")-$(date +%Y%m%d-%H%M%S).log"
  echo "  Log: $LOG_FILE"
  echo "  Follow: $SCRIPT_DIR/follow-log.sh $LOG_FILE"

  # Snapshot passes count before iteration to detect changes during run
  PREV_PASSES=$(jq -r '[.stories[] | select(.passes)] | map(.id) | join(",")' "$PRD_FILE" 2>/dev/null || echo "")

  # Background watcher: print when a story flips to passes during iteration
  (
    while kill -0 $$ 2>/dev/null; do
      sleep 15
      CUR_PASSES=$(jq -r '[.stories[] | select(.passes)] | map(.id) | join(",")' "$PRD_FILE" 2>/dev/null || echo "")
      if [ "$CUR_PASSES" != "$PREV_PASSES" ]; then
        # Find newly passed stories
        for sid in $(jq -r '.stories[] | select(.passes) | .id' "$PRD_FILE" 2>/dev/null); do
          if ! echo ",$PREV_PASSES," | grep -q ",$sid,"; then
            TITLE=$(jq -r --arg id "$sid" '.stories[] | select(.id == $id) | .title' "$PRD_FILE")
            echo "  >>> PASSED: $sid - $TITLE"
          fi
        done
        PREV_PASSES="$CUR_PASSES"
      fi
    done
  ) &
  WATCHER_PID=$!

  # tini -s (subreaper): adopts orphaned grandchildren (kubectl port-forwards etc.)
  # tini -g (process group): forwards signals to entire process group
  tini -sg -- claude --dangerously-skip-permissions --output-format stream-json --verbose --print < "$PROMPT_FILE" > "$LOG_FILE" 2>&1 &
  TINI_PID=$!
  wait $TINI_PID 2>/dev/null || true
  TINI_PID=""

  kill $WATCHER_PID 2>/dev/null; wait $WATCHER_PID 2>/dev/null

  # Check for completion signal
  if grep -q "<promise>COMPLETE</promise>" "$LOG_FILE"; then
    echo ""
    echo "All tuning stories completed!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    exit 0
  fi

  # Show story status after iteration
  echo ""
  echo "  Story status:"
  jq -r '.stories[] | "    \(if .passes then "PASS" else "    " end)  \(.id) - \(.title)"' "$PRD_FILE"
  PASSED=$(jq '[.stories[] | select(.passes)] | length' "$PRD_FILE")
  TOTAL=$(jq '.stories | length' "$PRD_FILE")
  echo "  Progress: $PASSED/$TOTAL passed"
  echo ""
  echo "Iteration $i complete."

  # Check for stop file — allows graceful stop between iterations
  if [ -f "$SCRIPT_DIR/STOP" ]; then
    echo "Stop file detected ($SCRIPT_DIR/STOP). Stopping after iteration $i."
    rm -f "$SCRIPT_DIR/STOP"
    exit 0
  fi

  echo "Continuing in 5s... (touch $SCRIPT_DIR/STOP to stop after next iteration)"
  sleep 5
done

echo ""
echo "Reached max iterations ($MAX_ITERATIONS) without completing all stories."
echo "Check $PROGRESS_FILE for status."
exit 1
