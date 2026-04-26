#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DATASET_NAME="${DATASET_NAME:-cais/mmlu}"
DATASET_CONFIG="${DATASET_CONFIG:-all}"
TRAIN_SPLIT="${TRAIN_SPLIT:-auxiliary_train}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-models/checkpoints/checkpoint_epoch_5.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tinystories_bpe.json}"
OUTPUT_PATH="${OUTPUT_PATH:-models/checkpoints/mmlu_finetuned.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

DEVICE="${DEVICE:-auto}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
SEQ_LEN="${SEQ_LEN:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

MAX_LR="${MAX_LR:-3e-5}"
MIN_LR="${MIN_LR:-3e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
SEED="${SEED:-1337}"

ARGS=(
  -m therapml.phase2.part1.fine_tune_mmlu
  --dataset-name "$DATASET_NAME"
  --dataset-config "$DATASET_CONFIG"
  --train-split "$TRAIN_SPLIT"
  --eval-split "$EVAL_SPLIT"
  --checkpoint-path "$CHECKPOINT_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --output-path "$OUTPUT_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --context-length "$CONTEXT_LENGTH"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --max-lr "$MAX_LR"
  --min-lr "$MIN_LR"
  --warmup-steps "$WARMUP_STEPS"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP"
  --log-every-steps "$LOG_EVERY_STEPS"
  --eval-every-steps "$EVAL_EVERY_STEPS"
  --seed "$SEED"
)

if [[ -n "$SEQ_LEN" ]]; then
  ARGS+=(--seq-len "$SEQ_LEN")
fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi
if [[ -n "$MAX_TRAIN_EXAMPLES" ]]; then
  ARGS+=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "$MAX_EVAL_EXAMPLES" ]]; then
  ARGS+=(--max-eval-examples "$MAX_EVAL_EXAMPLES")
fi
if [[ -n "$MAX_EVAL_BATCHES" ]]; then
  ARGS+=(--max-eval-batches "$MAX_EVAL_BATCHES")
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
