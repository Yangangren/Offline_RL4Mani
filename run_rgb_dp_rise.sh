#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

first_arg=${1:-}
normalized_first=${first_arg,,}
normalized_first=${normalized_first//-/_}
case "$normalized_first" in
  square|can|transport|tool_hang)
    TASK=$normalized_first
    shift
    ;;
esac
TASK=${TASK:-square}
TASK=${TASK,,}
TASK=${TASK//-/_}
STAGE=${1:-train_resilient}

case "$STAGE" in
  build_dataset|train|train_resilient|eval|eval_grid_resilient)
    ;;
  *)
    echo "Usage: $0 [square|can|transport|tool_hang] {build_dataset|train|train_resilient|eval|eval_grid_resilient}" >&2
    exit 2
    ;;
esac

case "$TASK" in
  square)
    DATA_STEM=200demo_406success_94failure
    ;;
  can)
    DATA_STEM=200demo_467success_33failure
    ;;
  transport)
    DATA_STEM=200demo_422success_78failure
    ;;
  tool_hang)
    DATA_STEM=200demo_132success_168failure
    EXPERT_DATASET=${EXPERT_DATASET:-datasets/tool_hang/ph/image_v15.rebuilt.hdf5}
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use square, can, transport, or tool_hang." >&2
    exit 2
    ;;
esac

# Resolve the requested training world size explicitly. The RISE-specific
# spelling is preferred; IDQL_NUM_GPUS and NUM_GPUS remain supported aliases.
RISE_RESOLVED_NUM_GPUS=
RISE_GPU_COUNT_SOURCE=default
for gpu_var in RISE_NUM_GPUS IDQL_NUM_GPUS NUM_GPUS; do
  if [[ -v "$gpu_var" && -n "${!gpu_var}" ]]; then
    if [[ -n "$RISE_RESOLVED_NUM_GPUS" && "$RISE_RESOLVED_NUM_GPUS" != "${!gpu_var}" ]]; then
      echo "Conflicting GPU counts: $RISE_GPU_COUNT_SOURCE=$RISE_RESOLVED_NUM_GPUS and $gpu_var=${!gpu_var}." >&2
      exit 2
    fi
    RISE_RESOLVED_NUM_GPUS=${!gpu_var}
    RISE_GPU_COUNT_SOURCE=$gpu_var
  fi
done
if [[ -z "$RISE_RESOLVED_NUM_GPUS" ]]; then
  if [[
    ( "$STAGE" == "train" || "$STAGE" == "train_resilient" )
    && -v CHUNK_NUM_GPUS
    && -n "$CHUNK_NUM_GPUS"
  ]]; then
    echo "CHUNK_NUM_GPUS does not configure the separate RISE baseline; use RISE_NUM_GPUS=$CHUNK_NUM_GPUS." >&2
    exit 2
  fi
  RISE_RESOLVED_NUM_GPUS=1
fi
if ! [[ "$RISE_RESOLVED_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "$RISE_GPU_COUNT_SOURCE must be a positive integer; got $RISE_RESOLVED_NUM_GPUS." >&2
  exit 2
fi

RISE_SPECTRAL_PENALTY_WEIGHT=${RISE_SPECTRAL_PENALTY_WEIGHT:-0.1}
RISE_SPECTRAL_HIDDEN_DIMS=${RISE_SPECTRAL_HIDDEN_DIMS:-512 512}
RISE_BETA_TAG=${RISE_SPECTRAL_PENALTY_WEIGHT//./p}

export TASK
export EXPERT_DATASET
export SUCCESS_MASK
export RISE_NUM_GPUS=$RISE_RESOLVED_NUM_GPUS
export IDQL_NUM_GPUS=$RISE_RESOLVED_NUM_GPUS
export IDQL_REWARD_MODE=rise_source_binary
export IDQL_TRAIN_SCRIPT=scripts/train_rgb_dp_rise.py
export IDQL_EVAL_SCRIPT=scripts/eval_rgb_dp_rise.py
export IDQL_TRAIN_EXTRA_ARGS="--spectral-penalty-weight $RISE_SPECTRAL_PENALTY_WEIGHT --spectral-hidden-dims $RISE_SPECTRAL_HIDDEN_DIMS --expected-world-size $RISE_RESOLVED_NUM_GPUS"
export IDQL_DATASET=${IDQL_DATASET:-datasets/$TASK/rise/${TASK}_rgb_dp_rise_${DATA_STEM}_source_binary.hdf5}
export IDQL_OUTPUT_DIR=${IDQL_OUTPUT_DIR:-trained_models/${TASK}_rgb_dp/rise/${DATA_STEM}_source_binary_spectral_beta${RISE_BETA_TAG}_one_step}
export EVAL_OUTPUT=${EVAL_OUTPUT:-rollouts/${TASK}_rgb_dp/rise/${DATA_STEM}_source_binary_spectral_beta${RISE_BETA_TAG}_one_step}

if [[ "$STAGE" == "train" || "$STAGE" == "train_resilient" ]]; then
  echo "[rgb_dp_rise] requested training GPUs=$RISE_RESOLVED_NUM_GPUS source=$RISE_GPU_COUNT_SOURCE" >&2
fi

exec bash run_rgb_dp_idql.sh "$TASK" "$STAGE"
