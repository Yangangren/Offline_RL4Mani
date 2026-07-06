#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Keep the robomimic environment boring and stable. These match the settings we
# have been using for Square RGB-DP evaluation/training.
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

PYTHON_BIN="${PYTHON_BIN:-/home/ryan/miniconda3/envs/robomimic_stable/bin/python}"
FEATURES="${FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-trained_models/square_success_potential/stage1_default_features}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260706}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.05}"
MONOTONIC_EPS="${MONOTONIC_EPS:-0.01}"
NUM_TIME_BINS="${NUM_TIME_BINS:-10}"

STEP_FEATURE_ARG="--no-include-step-feature"
if [[ "${INCLUDE_STEP_FEATURE:-0}" == "1" ]]; then
  STEP_FEATURE_ARG="--include-step-feature"
fi

BALANCED_ARG="--balanced-bce"
if [[ "${BALANCED_BCE:-1}" == "0" ]]; then
  BALANCED_ARG="--no-balanced-bce"
fi

"${PYTHON_BIN}" -B scripts/train_square_success_potential_stage1.py \
  --features "${FEATURES}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --total-steps "${TOTAL_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-every "${EVAL_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --dropout "${DROPOUT}" \
  --monotonic-eps "${MONOTONIC_EPS}" \
  --num-time-bins "${NUM_TIME_BINS}" \
  "${STEP_FEATURE_ARG}" \
  "${BALANCED_ARG}"
