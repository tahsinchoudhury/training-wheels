#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPOCHS=3
SEQ_LEN=32
BATCH_SIZE=4
MAX_TRAIN_TEXTS=10
MAX_EVAL_TEXTS=5
MAX_TRAIN_TOKENS=2000
MAX_EVAL_TOKENS=800
DEVICE="cpu"

PYTHON_BIN="python"
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" therapml/phase2/part1/run.py \
  --epochs "$EPOCHS" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --max-train-texts "$MAX_TRAIN_TEXTS" \
  --max-eval-texts "$MAX_EVAL_TEXTS" \
  --max-train-tokens "$MAX_TRAIN_TOKENS" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --device "$DEVICE" \
  "$@"
