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
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT-25}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-64}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
OUTPUT_PATH="${OUTPUT_PATH:-reports/mmlu_three_subjects_smoke_results.json}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model checkpoint not found: $MODEL_PATH" >&2
  echo "Run scripts/fine_tune_mmlu.sh first, or set MODEL_PATH to an existing checkpoint." >&2
  exit 1
fi

read -r -a TASK_ARRAY <<< "$TASKS"

ARGS=(
  -m therapml.phase2.part1.llama_eval_wrapper
  --tasks "${TASK_ARRAY[@]}"
  --model-path "$MODEL_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --num-fewshot "$NUM_FEWSHOT"
  --batch-size "$BATCH_SIZE"
  --max-gen-toks "$MAX_GEN_TOKS"
  --bootstrap-iters "$BOOTSTRAP_ITERS"
  --output-path "$OUTPUT_PATH"
)

if [[ "$DEVICE" != "auto" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
