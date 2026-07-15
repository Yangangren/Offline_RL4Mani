#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "can" || "${1:-}" == "Can" \
   || "${1:-}" == "square" || "${1:-}" == "Square" \
   || "${1:-}" == "transport" || "${1:-}" == "Transport" \
   || "${1:-}" == "tool_hang" || "${1:-}" == "ToolHang" \
   || "${1:-}" == "toolhang" || "${1:-}" == "tool-hang" ]]; then
  TASK=$1
  shift
fi

TASK=${TASK:-${RGB_DP_TASK:-square}}
TASK=${TASK,,}
TASK=${TASK//-/_}
if [[ "$TASK" == "toolhang" ]]; then
  TASK=tool_hang
fi

case "$TASK" in
  can|square|transport|tool_hang)
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use can, square, transport, or tool_hang." >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_${TASK}_baseline_rgb_dp_pycache_${USER}_$$"
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=${ROBOMIMIC_PYTHON:-/home/ryan/miniconda3/envs/robomimic_stable/bin/python}
export ROBOMIMIC_PYTHON="$PYTHON"
SCRIPT=scripts/run_rgb_dp_baseline.py

STAGE=${1:-prepare}
RECIPE=${RECIPE:-official}
DATASET_TYPE=${DATASET_TYPE:-ph}
TARGET_EPOCH=${TARGET_EPOCH:-2000}
EPOCHS=${EPOCHS:-$TARGET_EPOCH}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-100}
BATCH_SIZE=${BATCH_SIZE:-100}
LR=${LR:-1e-4}
SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS:-50}
EVAL_ROLLOUTS=${EVAL_ROLLOUTS:-100}
EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE:-10}
MAX_TRAIN_ATTEMPTS=${MAX_TRAIN_ATTEMPTS:-100}
MAX_EVAL_RETRIES=${MAX_EVAL_RETRIES:-5}
read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS:-0 1 2 3 4}"

BASE_ARGS=(
  --task "$TASK"
  --dataset-type "$DATASET_TYPE"
)
if [[ -n "${DATASET:-}" ]]; then
  BASE_ARGS+=(--dataset "$DATASET")
fi
if [[ -n "${RAW_DATASET:-}" ]]; then
  BASE_ARGS+=(--raw-dataset "$RAW_DATASET")
fi
if [[ -n "${CAMERA_SIZE:-}" ]]; then
  BASE_ARGS+=(--camera-size "$CAMERA_SIZE")
fi
if [[ -n "${CROP_SIZE:-}" ]]; then
  BASE_ARGS+=(--crop-size "$CROP_SIZE")
fi
if [[ -n "${CAMERA_NAMES:-}" ]]; then
  read -r -a CAMERA_NAME_ARRAY <<< "$CAMERA_NAMES"
  BASE_ARGS+=(--camera-names "${CAMERA_NAME_ARRAY[@]}")
fi
if [[ -n "${RGB_KEYS:-}" ]]; then
  read -r -a RGB_KEY_ARRAY <<< "$RGB_KEYS"
  BASE_ARGS+=(--rgb-keys "${RGB_KEY_ARRAY[@]}")
fi
if [[ -n "${LOW_DIM_KEYS:-}" ]]; then
  read -r -a LOW_DIM_KEY_ARRAY <<< "$LOW_DIM_KEYS"
  BASE_ARGS+=(--low-dim-keys "${LOW_DIM_KEY_ARRAY[@]}")
fi
if [[ -n "${HORIZON:-}" ]]; then
  BASE_ARGS+=(--horizon "$HORIZON")
fi

COMMON_RUN_ARGS=(
  --recipe "$RECIPE"
  --epochs "$EPOCHS"
  --target-epoch "$TARGET_EPOCH"
)
TRAIN_ARGS=(
  "${COMMON_RUN_ARGS[@]}"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --save-every-epochs "$SAVE_EVERY_EPOCHS"
  --max-train-attempts "$MAX_TRAIN_ATTEMPTS"
)
if [[ -n "${SEED:-}" ]]; then
  TRAIN_ARGS+=(--seed "$SEED")
fi
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--force-train)
fi
if [[ "${ENABLE_TRAIN_ROLLOUTS:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--enable-train-rollouts)
fi

EVAL_ARGS=(
  "${COMMON_RUN_ARGS[@]}"
  --eval-rollouts "$EVAL_ROLLOUTS"
  --eval-chunk-size "$EVAL_CHUNK_SIZE"
  --eval-seeds "${EVAL_SEED_ARRAY[@]}"
  --max-eval-retries "$MAX_EVAL_RETRIES"
)
if [[ -n "${CHECKPOINT:-}" ]]; then
  EVAL_ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ "${FORCE_EVAL:-0}" == "1" ]]; then
  EVAL_ARGS+=(--force-eval)
fi

run_stage() {
  "$PYTHON" -B "$SCRIPT" "${BASE_ARGS[@]}" "$@"
}

case "$STAGE" in
  dataset)
    dataset_args=()
    if [[ "${FORCE_DATASET:-0}" == "1" ]]; then
      dataset_args+=(--force-dataset)
    fi
    run_stage --stages dataset "${dataset_args[@]}"
    ;;

  prepare)
    run_stage --stages prepare "${TRAIN_ARGS[@]}"
    ;;

  train)
    run_stage --stages train "${TRAIN_ARGS[@]}"
    ;;

  train_resilient)
    run_stage --stages train "${TRAIN_ARGS[@]}" --resilient-train
    ;;

  eval)
    run_stage --stages eval "${EVAL_ARGS[@]}"
    ;;

  all)
    "$0" "$TASK" dataset
    "$0" "$TASK" prepare
    "$0" "$TASK" train
    "$0" "$TASK" eval
    ;;

  all_resilient)
    "$0" "$TASK" dataset
    "$0" "$TASK" prepare
    "$0" "$TASK" train_resilient
    "$0" "$TASK" eval
    ;;

  *)
    echo "Usage: $0 [can|square|transport|tool_hang] {dataset|prepare|train|train_resilient|eval|all|all_resilient}" >&2
    exit 2
    ;;
esac
