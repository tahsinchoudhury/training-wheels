#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="models/checkpoints/checkpoint_epoch_1.pth"
TOKENIZER_PATH="data/tinystories_bpe.json"
PROMPT=""
MAX_TOKENS=100
DEVICE=""

PYTHON_BIN="$ROOT_DIR"/.venv/bin/python

ARGS=(
  --model-path "$MODEL_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --max-tokens "$MAX_TOKENS"
)

if [[ -n "$PROMPT" ]]; then
  ARGS+=(--prompt "$PROMPT")
fi

if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi

exec "$PYTHON_BIN" therapml/phase2/part1/inference.py "${ARGS[@]}" "$@"
