#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

first_arg=${1:-}
first_arg=${first_arg,,}
first_arg=${first_arg//-/_}
if [[ "$first_arg" == "square" || "$first_arg" == "can" || "$first_arg" == "transport" || "$first_arg" == "tool_hang" ]]; then
  TASK=$first_arg
  shift
fi
TASK=${TASK:-square}
TASK=${TASK,,}

case "$TASK" in
  square)
    TASK_DP_CHECKPOINT=trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_DEMO_DATASET=datasets/square/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=50
    TASK_GT_CHUNKS=rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_gt_good_failure_chunks.hdf5
    TASK_OUTPUT_DIR=trained_models/square_rgb_dp/gt_good_failure_imitation/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/square_rgb_dp/gt_good_failure_imitation/200demo_100success_50failure
    TASK_HORIZON=400
    ;;
  can)
    TASK_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    TASK_DEMO_DATASET=datasets/can/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/can_rgb_dp/epoch50_collection/can_rgb_dp_rollouts_rgb2.hdf5
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=33
    TASK_GT_CHUNKS=rollouts/can_rgb_dp/epoch50_collection/can_rgb_dp_gt_good_failure_chunks.hdf5
    TASK_OUTPUT_DIR=trained_models/can_rgb_dp/gt_good_failure_imitation/200demo_100success_33failure
    TASK_EVAL_OUTPUT=rollouts/can_rgb_dp/gt_good_failure_imitation/200demo_100success_33failure
    TASK_HORIZON=400
    ;;
  transport)
    TASK_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_DEMO_DATASET=datasets/transport/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/transport_rgb_dp/epoch200_collection/transport_rgb_dp_rollouts_rgb4.hdf5
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=50
    TASK_GT_CHUNKS=rollouts/transport_rgb_dp/epoch200_collection/transport_rgb_dp_gt_good_failure_chunks.hdf5
    TASK_OUTPUT_DIR=trained_models/transport_rgb_dp/gt_good_failure_imitation/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/transport_rgb_dp/gt_good_failure_imitation/200demo_100success_50failure
    TASK_HORIZON=700
    ;;
  tool_hang)
    TASK_DP_CHECKPOINT=trained_models/tool_hang_rgb_dp/tool_hang_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_DEMO_DATASET=datasets/tool_hang/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/tool_hang_rgb_dp/epoch200_collection/tool_hang_rgb_dp_rollouts_rgb2.hdf5
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=50
    TASK_GT_CHUNKS=rollouts/tool_hang_rgb_dp/epoch200_collection/tool_hang_rgb_dp_gt_good_failure_chunks.hdf5
    TASK_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/gt_good_failure_imitation/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/gt_good_failure_imitation/200demo_100success_50failure
    TASK_HORIZON=700
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use square, can, transport, or tool_hang." >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_${TASK}_gt_good_failure_pycache_${USER}_$$"

PYTHON=${ROBOMIMIC_PYTHON:-/home/ryan/miniconda3/envs/robomimic_stable/bin/python}
DP_CHECKPOINT=${DP_CHECKPOINT:-$TASK_DP_CHECKPOINT}
DEMO_DATASET=${DEMO_DATASET:-$TASK_DEMO_DATASET}
ROLLOUT_DATASET=${ROLLOUT_DATASET:-$TASK_ROLLOUT_DATASET}
SUCCESS_MASK=${SUCCESS_MASK:-success_100}
FAILURE_MASK=${FAILURE_MASK:-$TASK_FAILURE_MASK}
FAILURE_COUNT=${FAILURE_COUNT:-$TASK_FAILURE_COUNT}
GT_GOOD_FAILURE_CHUNKS=${GT_GOOD_FAILURE_CHUNKS:-$TASK_GT_CHUNKS}
GT_GOOD_FAILURE_OUTPUT_DIR=${GT_GOOD_FAILURE_OUTPUT_DIR:-$TASK_OUTPUT_DIR}
EVAL_OUTPUT=${EVAL_OUTPUT:-$TASK_EVAL_OUTPUT}
HORIZON=${HORIZON:-$TASK_HORIZON}

build_chunks() {
  local -a overwrite_args=()
  if [[ "${OVERWRITE_GT_GOOD_FAILURE_CHUNKS:-0}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  "$PYTHON" -B scripts/build_rgb_dp_failure_filter_dataset.py \
    --task "$TASK" \
    --source "$ROLLOUT_DATASET" \
    --output "$GT_GOOD_FAILURE_CHUNKS" \
    --success-mask "$SUCCESS_MASK" \
    --failure-mask "$FAILURE_MASK" \
    --prediction-horizon "${GT_GOOD_PREDICTION_HORIZON:-16}" \
    --stride "${GT_GOOD_STRIDE:-1}" \
    --min-start-step "${GT_GOOD_MIN_START_STEP:-0}" \
    --max-chunks-per-failure "${GT_GOOD_MAX_CHUNKS_PER_FAILURE:-4}" \
    --minimum-spacing "${GT_GOOD_MINIMUM_SPACING:-8}" \
    --prefer "${GT_GOOD_PREFER:-progress}" \
    --min-goal-progress "${GT_GOOD_MIN_GOAL_PROGRESS:-0.05}" \
    --min-normalized-displacement "${GT_GOOD_MIN_NORMALIZED_DISPLACEMENT:-0.05}" \
    --goal-progress-weight "${GT_GOOD_PROGRESS_WEIGHT:-1.0}" \
    --displacement-weight "${GT_GOOD_DISPLACEMENT_WEIGHT:-0.1}" \
    --position-scale-floor "${GT_GOOD_POSITION_SCALE_FLOOR:-0.01}" \
    "${overwrite_args[@]}"
}

ensure_chunks() {
  if [[ ! -f "$GT_GOOD_FAILURE_CHUNKS" ]]; then
    echo "[gt_good_failure task=$TASK] building missing chunks: $GT_GOOD_FAILURE_CHUNKS" >&2
    build_chunks
  fi
}

run_imitation() {
  local stage=$1
  AUTO_PREPARE_FILTERS=0 \
  DP_CHECKPOINT="$DP_CHECKPOINT" \
  DEMO_DATASET="$DEMO_DATASET" \
  ROLLOUT_DATASET="$ROLLOUT_DATASET" \
  SUCCESS_DATASET="$ROLLOUT_DATASET" \
  FAILURE_DATASET="$GT_GOOD_FAILURE_CHUNKS" \
  SUCCESS_FILTER_KEY="$SUCCESS_MASK" \
  MIXED_IMITATION_SUCCESS_FILTER_KEY="$SUCCESS_MASK" \
  MIXED_IMITATION_FAILURE_FILTER_KEY=gt_good_failure \
  FAILURE_FILTER_SIZE="$FAILURE_COUNT" \
  MIXED_IMITATION_FAILURE_DEMO_START_ONLY=1 \
  MIXED_IMITATION_FAILURE_SAMPLE_START_OFFSET=1 \
  MIXED_IMITATION_FAILURE_ANTI_FAILURE_LABEL=0.0 \
  MIXED_IMITATION_OUTPUT_DIR="$GT_GOOD_FAILURE_OUTPUT_DIR" \
  IMITATION_OUTPUT_DIR="$GT_GOOD_FAILURE_OUTPUT_DIR" \
  MIXED_IMITATION_MODE_NAME=gt_good_failure_mixed_imitation_learning \
  MIXED_IMITATION_EXPERIMENT_NAME="${TASK}_rgb_dp_gt_good_failure_imitation" \
  EVAL_OUTPUT="$EVAL_OUTPUT" \
  HORIZON="$HORIZON" \
  ./run_rgb_dp_mixed_imitation.sh "$TASK" "$stage"
}

STAGE=${1:-train_resilient}
case "$STAGE" in
  build_chunks)
    build_chunks
    ;;
  train)
    ensure_chunks
    run_imitation train_mixed_imitation
    ;;
  train_resilient)
    ensure_chunks
    run_imitation train_mixed_imitation_resilient
    ;;
  check)
    ensure_chunks
    run_imitation check_mixed
    ;;
  eval_grid_resilient)
    run_imitation eval_mixed_grid_resilient
    ;;
  all)
    ensure_chunks
    run_imitation train_mixed_imitation_resilient
    ;;
  *)
    echo "Usage: $0 [square|can|transport|tool_hang] {build_chunks|check|train|train_resilient|eval_grid_resilient|all}" >&2
    exit 2
    ;;
esac
