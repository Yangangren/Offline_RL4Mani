#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_collect_pycache_${USER}_$$"
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

CHECKPOINT=${CHECKPOINT:-/home/ryan/Documents/robomimic/trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth}
OUTPUT_DIR=${OUTPUT_DIR:-rollouts/square_rgb_dp/epoch190_collection}
TOTAL_ROLLOUTS=${TOTAL_ROLLOUTS:-500}
HORIZON=${HORIZON:-400}
SEED_BASE=${SEED_BASE:-0}
POLICY_SEEDS=${POLICY_SEEDS:-}
NUM_ENV_SEEDS=${NUM_ENV_SEEDS:-}
if [[ -n "$POLICY_SEEDS" ]]; then
  ROLLOUTS_PER_SHARD=${ROLLOUTS_PER_SHARD:-1}
else
  ROLLOUTS_PER_SHARD=${ROLLOUTS_PER_SHARD:-10}
fi
MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_SEED_OFFSET=${RETRY_SEED_OFFSET:-100000}
RAW_NAME=${RAW_NAME:-square_rgb_dp_rollouts_raw.hdf5}
RGB_NAME=${RGB_NAME:-square_rgb_dp_rollouts_rgb2.hdf5}
CAMERA_SIZE=${CAMERA_SIZE:-84}
CAMERA_NAMES=${CAMERA_NAMES:-agentview robot0_eye_in_hand}
read -r -a CAMERA_NAME_ARRAY <<< "$CAMERA_NAMES"
if (( ${#CAMERA_NAME_ARRAY[@]} == 0 )); then
  echo "CAMERA_NAMES must contain at least one camera name" >&2
  exit 2
fi

if [[ -n "$POLICY_SEEDS" ]]; then
  if (( ROLLOUTS_PER_SHARD != 1 )); then
    echo "Split env/policy seed collection requires ROLLOUTS_PER_SHARD=1" >&2
    exit 2
  fi
  read -r -a POLICY_SEED_ARRAY <<< "$POLICY_SEEDS"
  if (( ${#POLICY_SEED_ARRAY[@]} == 0 )); then
    echo "POLICY_SEEDS must contain at least one integer seed" >&2
    exit 2
  fi
  if [[ -z "$NUM_ENV_SEEDS" ]]; then
    if (( TOTAL_ROLLOUTS % ${#POLICY_SEED_ARRAY[@]} != 0 )); then
      echo "TOTAL_ROLLOUTS must be divisible by number of POLICY_SEEDS" >&2
      exit 2
    fi
    NUM_ENV_SEEDS=$((TOTAL_ROLLOUTS / ${#POLICY_SEED_ARRAY[@]}))
  fi
  NUM_SHARDS=$((NUM_ENV_SEEDS * ${#POLICY_SEED_ARRAY[@]}))
  if (( NUM_SHARDS != TOTAL_ROLLOUTS )); then
    echo "TOTAL_ROLLOUTS must equal NUM_ENV_SEEDS * number of POLICY_SEEDS" >&2
    exit 2
  fi
else
  if (( TOTAL_ROLLOUTS % ROLLOUTS_PER_SHARD != 0 )); then
    echo "TOTAL_ROLLOUTS must be divisible by ROLLOUTS_PER_SHARD" >&2
    exit 2
  fi
  NUM_SHARDS=$((TOTAL_ROLLOUTS / ROLLOUTS_PER_SHARD))
fi

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
if [[ -n "$POLICY_SEEDS" ]]; then
  collect_args+=(--num-env-seeds "$NUM_ENV_SEEDS")
  collect_args+=(--policy-seeds "${POLICY_SEED_ARRAY[@]}")
fi
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
    convert_rgb_args=(
      --dataset "$OUTPUT_DIR/$RAW_NAME"
      --output_name "$RGB_NAME"
      --done_mode 2
      --copy_rewards
      --camera_names "${CAMERA_NAME_ARRAY[@]}"
      --camera_height "$CAMERA_SIZE"
      --camera_width "$CAMERA_SIZE"
      --compress
      --exclude-next-obs
      --resume
      --reuse-identical-models
    )
    if [[ "${RESTART_RGB:-0}" == "1" ]]; then
      convert_rgb_args+=(--restart)
    fi
    if [[ "${OVERWRITE_RGB:-0}" == "1" ]]; then
      convert_rgb_args+=(--overwrite)
    fi
    "$PYTHON" -B robomimic/scripts/dataset_states_to_obs.py "${convert_rgb_args[@]}"
    ;;

  *)
    echo "Usage: $0 {collect|convert_rgb}" >&2
    echo "Common overrides: TOTAL_ROLLOUTS=500 ROLLOUTS_PER_SHARD=10 HORIZON=... CHECKPOINT=... OUTPUT_DIR=... RAW_NAME=... FORCE_COLLECT=1" >&2
    echo "RGB conversion overrides: RGB_NAME=... CAMERA_NAMES=\"camera0 camera1\" CAMERA_SIZE=84" >&2
    echo "Split seed grid: POLICY_SEEDS=\"0 1\" NUM_ENV_SEEDS=250 ROLLOUTS_PER_SHARD=1 TOTAL_ROLLOUTS=500" >&2
    exit 2
    ;;
esac
