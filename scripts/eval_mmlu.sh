#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="$ROOT_DIR/lm-evaluation-harness:$ROOT_DIR:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-models/checkpoints/mmlu_finetuned.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tinystories_bpe.json}"
TASKS="${TASKS:-mmlu_college_computer_science mmlu_elementary_mathematics mmlu_jurisprudence}"
GPT2_MODEL="${GPT2_MODEL:-gpt2-large}"
GPT2_MODEL_ARGS="${GPT2_MODEL_ARGS:-}"
GPT2_DTYPE="${GPT2_DTYPE:-float32}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GPT2_BATCH_SIZE="${GPT2_BATCH_SIZE:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT-25}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-64}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model checkpoint not found: $MODEL_PATH" >&2
  echo "Run scripts/fine_tune_mmlu.sh first, or set MODEL_PATH to an existing checkpoint." >&2
  exit 1
fi

read -r -a TASK_ARRAY <<< "$TASKS"

ARGS=(
  scripts/mmlu_eval.py
  --tasks "${TASK_ARRAY[@]}"
  --model-path "$MODEL_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --gpt2-model "$GPT2_MODEL"
  --gpt2-dtype "$GPT2_DTYPE"
  --num-fewshot "$NUM_FEWSHOT"
  --batch-size "$BATCH_SIZE"
  --max-gen-toks "$MAX_GEN_TOKS"
  --bootstrap-iters "$BOOTSTRAP_ITERS"
  --log-dir "$LOG_DIR"
)

if [[ -n "$GPT2_MODEL_ARGS" ]]; then
  ARGS+=(--gpt2-model-args "$GPT2_MODEL_ARGS")
fi
if [[ -n "$GPT2_BATCH_SIZE" ]]; then
  ARGS+=(--gpt2-batch-size "$GPT2_BATCH_SIZE")
fi
if [[ "$DEVICE" != "auto" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$OUTPUT_PATH" ]]; then
  ARGS+=(--output-path "$OUTPUT_PATH")
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
