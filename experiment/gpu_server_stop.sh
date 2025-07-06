#!/usr/bin/env bash
# gpu_server_stop.sh
# Terminates all RPC server processes started by gpu_server_start.sh.
# This script looks for the PID list stored in logs/gpu_server_pids.txt and
# sends SIGINT to each process.  If a process does not exit within 5 seconds
# it will be force-killed with SIGKILL.

set -eu

PID_FILE="logs/gpu_server_pids.txt"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE – nothing to stop." >&2
  exit 0
fi

while read -r pid; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping rpc_server process with PID $pid…"
    kill "$pid"
    # Wait up to 5 seconds
    for _ in {1..5}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    # Force kill if still alive
    if kill -0 "$pid" 2>/dev/null; then
      echo "Process $pid did not exit gracefully – killing."
      kill -9 "$pid" || true
    fi
  fi
done < "$PID_FILE"

# Cleanup
rm -f "$PID_FILE"

echo "All GPU server processes stopped." 