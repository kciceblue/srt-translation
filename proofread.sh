#!/usr/bin/env bash
set -euo pipefail

# ── Configuration (edit these) ───────────────────────────────────────
ENDPOINT="http://127.0.0.1:8089/v1/chat/completions"
INPUT="subs/"
OUT_DIR="out"
SOURCE_LANG="Japanese"
TARGET_LANG="Simplified Chinese"
EXTRA='{"model":"local/qwen3.5","temperature":0.2}'
# ─────────────────────────────────────────────────────────────────────

# Requires tmp/ from a prior run (with --debug or individual steps)
python3 cli.py proofread \
  --endpoint "$ENDPOINT" \
  --out-dir "$OUT_DIR" \
  --source-lang "$SOURCE_LANG" \
  --target-lang "$TARGET_LANG" \
  --extra-payload "$EXTRA" \
  --debug
