#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOKENIZER="data/tinystories_bpe.json"
DATASET=""              # leave empty to use HF TinyStories
TEXT_FIELD="text"

SEQ_LEN=128
BATCH_SIZE=128
STRIDE=64
EPOCHS=2
MAX_LR=3e-5
MIN_LR=3e-8
WARMUP_STEPS=5000
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0

D_MODEL=768
NUM_LAYERS=6
NUM_HEADS=8
D_FF=1536
ROPE_THETA=10000.0

MAX_TRAIN_TEXTS=2119719
MAX_EVAL_TEXTS=21990
# MAX_TRAIN_TEXTS=211900
# MAX_EVAL_TEXTS=2190
MAX_TRAIN_TOKENS=468974258
MAX_EVAL_TOKENS=4714076
# MAX_TRAIN_TOKENS=5000
# MAX_EVAL_TOKENS=5000

DEVICE="cuda"

GENERATION_STEP_INTERVAL=500
PLOT_INTERVAL_STEPS=5000

CHECKPOINT_INTERVAL_EPOCHS=1
CHECKPOINT_DIR="models/checkpoints"

PYTHON_BIN="$ROOT_DIR"/.venv/bin/python

exec "$PYTHON_BIN" therapml/phase2/part1/run.py \
  --tokenizer "$TOKENIZER" \
  --dataset "$DATASET" \
  --text-field "$TEXT_FIELD" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --max-lr "$MAX_LR" \
  --min-lr "$MIN_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --d-ff "$D_FF" \
  --rope-theta "$ROPE_THETA" \
  --checkpoint-interval-epochs "$CHECKPOINT_INTERVAL_EPOCHS" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --max-train-texts "$MAX_TRAIN_TEXTS" \
  --max-eval-texts "$MAX_EVAL_TEXTS" \
  --max-train-tokens "$MAX_TRAIN_TOKENS" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --device "$DEVICE" \
  --generation-step-interval "$GENERATION_STEP_INTERVAL" \
  --plot-interval-steps "$PLOT_INTERVAL_STEPS" \
  --stride "$STRIDE" \
  "$@"