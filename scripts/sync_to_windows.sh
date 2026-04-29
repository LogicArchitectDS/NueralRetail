#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="/home/seshu/NueralRetail_Solo/"
TARGET_DIR="${1:-/mnt/c/Users/ll/OneDrive/Desktop/AMDOX/NueralRetail_Solo/}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Target directory not found: $TARGET_DIR" >&2
  exit 1
fi

rsync \
  -rltv \
  --delete \
  --omit-dir-times \
  --no-perms \
  --no-owner \
  --no-group \
  --exclude ".git/" \
  --exclude ".planning/" \
  --exclude ".venv/" \
  --exclude "node_modules/" \
  --exclude "data/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude "*.pyc" \
  --exclude ".env" \
  "$SOURCE_DIR" "$TARGET_DIR"

echo "Synced Ubuntu repo to Windows mirror: $TARGET_DIR"
