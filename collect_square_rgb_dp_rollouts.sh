#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_square_collect_pycache_${USER}_$$"
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=/home/ryan/miniconda3/envs/robomimic_stable/bin/python
export ROBOMIMIC_PYTHON="$PYTHON"

CHECKPOINT=${CHECKPOINT:-/home/ryan/Documents/robomimic/trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth}
OUTPUT_DIR=${OUTPUT_DIR:-rollouts/square_rgb_dp/epoch190_collection}
TOTAL_ROLLOUTS=${TOTAL_ROLLOUTS:-500}
ROLLOUTS_PER_SHARD=${ROLLOUTS_PER_SHARD:-10}
HORIZON=${HORIZON:-400}
SEED_BASE=${SEED_BASE:-0}
MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_SEED_OFFSET=${RETRY_SEED_OFFSET:-100000}
RAW_NAME=${RAW_NAME:-square_rgb_dp_rollouts_raw.hdf5}
RGB_NAME=${RGB_NAME:-square_rgb_dp_rollouts_rgb2.hdf5}
CAMERA_SIZE=${CAMERA_SIZE:-84}

if (( TOTAL_ROLLOUTS % ROLLOUTS_PER_SHARD != 0 )); then
  echo "TOTAL_ROLLOUTS must be divisible by ROLLOUTS_PER_SHARD" >&2
  exit 2
fi
NUM_SHARDS=$((TOTAL_ROLLOUTS / ROLLOUTS_PER_SHARD))

STAGE=${1:-collect}

collect_args=(
  --agent "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --merged-name "$RAW_NAME"
  --num-shards "$NUM_SHARDS"
  --rollouts-per-shard "$ROLLOUTS_PER_SHARD"
  --horizon "$HORIZON"
  --seed-base "$SEED_BASE"
  --max-retries "$MAX_RETRIES"
  --retry-seed-offset "$RETRY_SEED_OFFSET"
  --force-merge
)
if [[ "${FORCE_COLLECT:-0}" == "1" ]]; then
  collect_args+=(--force-shards)
fi
if [[ "${DATASET_OBS:-0}" == "1" ]]; then
  collect_args+=(--dataset-obs)
fi

case "$STAGE" in
  collect)
    "$PYTHON" -B scripts/collect_rollout_shards.py "${collect_args[@]}"
    ;;

  convert_rgb)
    "$PYTHON" -B robomimic/scripts/dataset_states_to_obs.py \
      --dataset "$OUTPUT_DIR/$RAW_NAME" \
      --output_name "$RGB_NAME" \
      --done_mode 2 \
      --copy_rewards \
      --copy_dones \
      --camera_names agentview robot0_eye_in_hand \
      --camera_height "$CAMERA_SIZE" \
      --camera_width "$CAMERA_SIZE" \
      --compress \
      --exclude-next-obs
    ;;

  all)
    "$PYTHON" -B scripts/collect_rollout_shards.py "${collect_args[@]}"
    "$PYTHON" -B robomimic/scripts/dataset_states_to_obs.py \
      --dataset "$OUTPUT_DIR/$RAW_NAME" \
      --output_name "$RGB_NAME" \
      --done_mode 2 \
      --copy_rewards \
      --copy_dones \
      --camera_names agentview robot0_eye_in_hand \
      --camera_height "$CAMERA_SIZE" \
      --camera_width "$CAMERA_SIZE" \
      --compress \
      --exclude-next-obs
    ;;

  *)
    echo "Usage: $0 {collect|convert_rgb|all}" >&2
    echo "Common overrides: TOTAL_ROLLOUTS=500 ROLLOUTS_PER_SHARD=10 CHECKPOINT=... OUTPUT_DIR=... FORCE_COLLECT=1" >&2
    exit 2
    ;;
esac
