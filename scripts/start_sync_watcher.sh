#!/usr/bin/env bash
set -euo pipefail

cd /home/seshu/NueralRetail_Solo
mkdir -p /tmp

if pgrep -f "python3 scripts/watch_sync_to_windows.py" >/dev/null 2>&1; then
  echo "Sync watcher is already running."
  exit 0
fi

nohup python3 scripts/watch_sync_to_windows.py >/tmp/neuralretail_sync_watcher.log 2>&1 &
echo "Started sync watcher with PID $!"
