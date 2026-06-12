#!/usr/bin/env bash
# Kill any process holding port 5000 before starting the server.
INODE=$(awk '$4=="0A" && $2~/1388$/ {print $10}' /proc/net/tcp /proc/net/tcp6 2>/dev/null)
if [ -n "$INODE" ]; then
  for pid in /proc/[0-9]*; do
    p=$(basename "$pid")
    ls -la "$pid/fd" 2>/dev/null | grep -q "socket:\[$INODE\]" && kill -9 "$p" 2>/dev/null
  done
  sleep 1
fi
exec uvicorn src.api:app --host 0.0.0.0 --port 5000 --reload
