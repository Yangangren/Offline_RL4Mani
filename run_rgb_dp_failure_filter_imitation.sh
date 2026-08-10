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
export NUMBA_DISABLE_JIT=1

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
if [[ "$TASK" == "can" ]]; then
  EXPECTED_FILTER_VERSION=can_privileged_stage_v1
  DEFAULT_FILTER_MINIMUM_SPACING=16
elif [[ "$TASK" == "transport" ]]; then
  EXPECTED_FILTER_VERSION=transport_privileged_stage_v1
  DEFAULT_FILTER_MINIMUM_SPACING=16
elif [[ "$TASK" == "tool_hang" ]]; then
  EXPECTED_FILTER_VERSION=tool_hang_privileged_stage_v1
  DEFAULT_FILTER_MINIMUM_SPACING=16
else
  EXPECTED_FILTER_VERSION=goal_endpoint_progress_v1
  DEFAULT_FILTER_MINIMUM_SPACING=8
fi

validate_chunks() {
  "$PYTHON" -B - \
    "$GT_GOOD_FAILURE_CHUNKS" \
    "$ROLLOUT_DATASET" \
    "$TASK" \
    "$SUCCESS_MASK" \
    "$FAILURE_MASK" \
    "${GT_GOOD_PREDICTION_HORIZON:-16}" \
    "$EXPECTED_FILTER_VERSION" <<'PYCHECK'
import sys
from pathlib import Path

import h5py

(
    artifact,
    source,
    task,
    success_mask,
    failure_mask,
    prediction_horizon,
    filter_version,
) = sys.argv[1:]
artifact_path = Path(artifact).expanduser().resolve()
if not artifact_path.is_file():
    raise FileNotFoundError(artifact_path)
with h5py.File(artifact_path, "r") as handle:
    expected = {
        "task": task,
        "source_path": str(Path(source).expanduser().resolve()),
        "source_success_mask": success_mask,
        "source_failure_mask": failure_mask,
        "prediction_horizon": int(prediction_horizon),
        "filter_version": filter_version,
    }
    actual = {key: handle.attrs.get(key) for key in expected}
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise ValueError(
            f"incompatible filtered chunk artifact {artifact_path}: {mismatches}"
        )
    if "mask/gt_good_failure" not in handle:
        raise KeyError(f"{artifact_path} has no mask/gt_good_failure")
    if len(handle["mask/gt_good_failure"]) == 0:
        raise ValueError(f"{artifact_path} contains no filtered chunks")
    chunk_count = len(handle["mask/gt_good_failure"])
print(
    f"[validated chunks] {artifact_path}: "
    f"version={filter_version} chunks={chunk_count}"
)
PYCHECK
}

build_chunks() {
  local -a overwrite_args=()
  local -a task_filter_args=()
  if [[ "${OVERWRITE_GT_GOOD_FAILURE_CHUNKS:-0}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  if [[ "$TASK" == "can" ]]; then
    task_filter_args=(
      --can-success-calibration-limit "${CAN_SUCCESS_CALIBRATION_LIMIT:-100}"
      --can-max-chunks-per-stage-per-failure "${CAN_MAX_CHUNKS_PER_STAGE_PER_FAILURE:-2}"
      --can-safe-min-start-distance "${CAN_SAFE_MIN_START_DISTANCE:-0.10}"
      --can-safe-min-distance "${CAN_SAFE_MIN_DISTANCE:-0.05}"
      --can-safe-min-reach-gain "${CAN_SAFE_MIN_REACH_GAIN:-0.04}"
      --can-safe-min-progress-fraction "${CAN_SAFE_MIN_PROGRESS_FRACTION:-0.75}"
      --can-safe-progress-tolerance "${CAN_SAFE_PROGRESS_TOLERANCE:-0.001}"
      --can-safe-max-regression "${CAN_SAFE_MAX_REGRESSION:-0.015}"
      --can-safe-max-can-displacement "${CAN_SAFE_MAX_CAN_DISPLACEMENT:-0.01}"
      --can-grasp-min-frames "${CAN_GRASP_MIN_FRAMES:-6}"
      --can-grasp-min-lift-gain "${CAN_GRASP_MIN_LIFT_GAIN:-0.025}"
      --can-grasp-max-drop "${CAN_GRASP_MAX_DROP:-0.015}"
      --can-transport-min-grasp-fraction "${CAN_TRANSPORT_MIN_GRASP_FRACTION:-0.75}"
      --can-transport-min-lift-height "${CAN_TRANSPORT_MIN_LIFT_HEIGHT:-0.04}"
      --can-transport-min-bin-progress "${CAN_TRANSPORT_MIN_BIN_PROGRESS:-0.04}"
      --can-transport-min-progress-fraction "${CAN_TRANSPORT_MIN_PROGRESS_FRACTION:-0.65}"
      --can-transport-progress-tolerance "${CAN_TRANSPORT_PROGRESS_TOLERANCE:-0.002}"
      --can-transport-max-regression "${CAN_TRANSPORT_MAX_REGRESSION:-0.025}"
      --can-transport-max-drop "${CAN_TRANSPORT_MAX_DROP:-0.025}"
    )
  elif [[ "$TASK" == "transport" ]]; then
    task_filter_args=(
      --transport-success-calibration-limit "${TRANSPORT_SUCCESS_CALIBRATION_LIMIT:-100}"
      --transport-max-chunks-per-stage-per-failure "${TRANSPORT_MAX_CHUNKS_PER_STAGE_PER_FAILURE:-1}"
      --transport-safe-min-start-distance "${TRANSPORT_SAFE_MIN_START_DISTANCE:-0.10}"
      --transport-safe-min-distance "${TRANSPORT_SAFE_MIN_DISTANCE:-0.05}"
      --transport-safe-min-reach-gain "${TRANSPORT_SAFE_MIN_REACH_GAIN:-0.04}"
      --transport-safe-min-progress-fraction "${TRANSPORT_SAFE_MIN_PROGRESS_FRACTION:-0.70}"
      --transport-safe-progress-tolerance "${TRANSPORT_SAFE_PROGRESS_TOLERANCE:-0.001}"
      --transport-safe-max-regression "${TRANSPORT_SAFE_MAX_REGRESSION:-0.015}"
      --transport-lid-min-clearance "${TRANSPORT_LID_MIN_CLEARANCE:-0.10}"
      --transport-lid-min-clearance-gain "${TRANSPORT_LID_MIN_CLEARANCE_GAIN:-0.04}"
      --transport-lid-max-clearance-regression "${TRANSPORT_LID_MAX_CLEARANCE_REGRESSION:-0.03}"
      --transport-lid-max-drop "${TRANSPORT_LID_MAX_DROP:-0.04}"
      --transport-grasp-min-frames "${TRANSPORT_GRASP_MIN_FRAMES:-6}"
      --transport-grasp-min-lift-gain "${TRANSPORT_GRASP_MIN_LIFT_GAIN:-0.025}"
      --transport-grasp-max-drop "${TRANSPORT_GRASP_MAX_DROP:-0.02}"
      --transport-target-min-grasp-fraction "${TRANSPORT_TARGET_MIN_GRASP_FRACTION:-0.75}"
      --transport-target-min-lift-height "${TRANSPORT_TARGET_MIN_LIFT_HEIGHT:-0.04}"
      --transport-target-min-bin-progress "${TRANSPORT_TARGET_MIN_BIN_PROGRESS:-0.04}"
      --transport-target-min-progress-fraction "${TRANSPORT_TARGET_MIN_PROGRESS_FRACTION:-0.65}"
      --transport-target-progress-tolerance "${TRANSPORT_TARGET_PROGRESS_TOLERANCE:-0.002}"
      --transport-target-max-regression "${TRANSPORT_TARGET_MAX_REGRESSION:-0.025}"
      --transport-target-max-drop "${TRANSPORT_TARGET_MAX_DROP:-0.025}"
      --transport-place-min-bin-progress "${TRANSPORT_PLACE_MIN_BIN_PROGRESS:-0.02}"
      --transport-place-score-bonus "${TRANSPORT_PLACE_SCORE_BONUS:-0.10}"
      --transport-secondary-max-static-displacement "${TRANSPORT_SECONDARY_MAX_STATIC_DISPLACEMENT:-0.015}"
      --transport-secondary-min-grasp-fraction "${TRANSPORT_SECONDARY_MIN_GRASP_FRACTION:-0.50}"
    )
  elif [[ "$TASK" == "tool_hang" ]]; then
    task_filter_args=(
      --tool-hang-success-calibration-limit "${TOOL_HANG_SUCCESS_CALIBRATION_LIMIT:-100}"
      --tool-hang-max-chunks-per-stage-per-failure "${TOOL_HANG_MAX_CHUNKS_PER_STAGE_PER_FAILURE:-1}"
      --tool-hang-safe-min-start-distance "${TOOL_HANG_SAFE_MIN_START_DISTANCE:-0.10}"
      --tool-hang-safe-min-distance "${TOOL_HANG_SAFE_MIN_DISTANCE:-0.05}"
      --tool-hang-safe-min-reach-gain "${TOOL_HANG_SAFE_MIN_REACH_GAIN:-0.04}"
      --tool-hang-safe-min-progress-fraction "${TOOL_HANG_SAFE_MIN_PROGRESS_FRACTION:-0.70}"
      --tool-hang-safe-max-regression "${TOOL_HANG_SAFE_MAX_REGRESSION:-0.015}"
      --tool-hang-progress-tolerance "${TOOL_HANG_PROGRESS_TOLERANCE:-0.001}"
      --tool-hang-grasp-min-frames "${TOOL_HANG_GRASP_MIN_FRAMES:-6}"
      --tool-hang-grasp-min-lift-gain "${TOOL_HANG_GRASP_MIN_LIFT_GAIN:-0.025}"
      --tool-hang-grasp-max-drop "${TOOL_HANG_GRASP_MAX_DROP:-0.02}"
      --tool-hang-transport-min-grasp-fraction "${TOOL_HANG_TRANSPORT_MIN_GRASP_FRACTION:-0.75}"
      --tool-hang-transport-min-lift-height "${TOOL_HANG_TRANSPORT_MIN_LIFT_HEIGHT:-0.04}"
      --tool-hang-transport-min-progress "${TOOL_HANG_TRANSPORT_MIN_PROGRESS:-0.04}"
      --tool-hang-transport-min-progress-fraction "${TOOL_HANG_TRANSPORT_MIN_PROGRESS_FRACTION:-0.65}"
      --tool-hang-transport-max-regression "${TOOL_HANG_TRANSPORT_MAX_REGRESSION:-0.025}"
      --tool-hang-transport-max-drop "${TOOL_HANG_TRANSPORT_MAX_DROP:-0.025}"
      --tool-hang-frame-insert-max-distance "${TOOL_HANG_FRAME_INSERT_MAX_DISTANCE:-0.05}"
      --tool-hang-frame-insert-score-bonus "${TOOL_HANG_FRAME_INSERT_SCORE_BONUS:-0.10}"
      --tool-hang-align-max-endpoint-distance "${TOOL_HANG_ALIGN_MAX_ENDPOINT_DISTANCE:-0.06}"
      --tool-hang-align-min-error-progress "${TOOL_HANG_ALIGN_MIN_ERROR_PROGRESS:-0.015}"
      --tool-hang-align-min-progress-fraction "${TOOL_HANG_ALIGN_MIN_PROGRESS_FRACTION:-0.60}"
      --tool-hang-align-max-regression "${TOOL_HANG_ALIGN_MAX_REGRESSION:-0.02}"
      --tool-hang-hook-contact-score-bonus "${TOOL_HANG_HOOK_CONTACT_SCORE_BONUS:-0.05}"
      --tool-hang-max-static-object-displacement "${TOOL_HANG_MAX_STATIC_OBJECT_DISPLACEMENT:-0.01}"
    )
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
    --minimum-spacing "${GT_GOOD_MINIMUM_SPACING:-$DEFAULT_FILTER_MINIMUM_SPACING}" \
    --prefer "${GT_GOOD_PREFER:-progress}" \
    --min-goal-progress "${GT_GOOD_MIN_GOAL_PROGRESS:-0.05}" \
    --min-normalized-displacement "${GT_GOOD_MIN_NORMALIZED_DISPLACEMENT:-0.05}" \
    --goal-progress-weight "${GT_GOOD_PROGRESS_WEIGHT:-1.0}" \
    --displacement-weight "${GT_GOOD_DISPLACEMENT_WEIGHT:-0.1}" \
    --position-scale-floor "${GT_GOOD_POSITION_SCALE_FLOOR:-0.01}" \
    "${task_filter_args[@]}" \
    "${overwrite_args[@]}"
  validate_chunks
}

ensure_chunks() {
  if [[ ! -f "$GT_GOOD_FAILURE_CHUNKS" ]]; then
    echo "[gt_good_failure task=$TASK] building missing chunks: $GT_GOOD_FAILURE_CHUNKS" >&2
    build_chunks
  fi
  validate_chunks
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
  ACTOR_UNIFORM_SAMPLE_POOL="${GT_GOOD_ACTOR_UNIFORM_SAMPLE_POOL:-0}" \
  ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE="${GT_GOOD_ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE:-1}" \
  MIXED_IMITATION_DEMO_WEIGHT="${GT_GOOD_IMITATION_DEMO_WEIGHT:-1.0}" \
  MIXED_IMITATION_SUCCESS_WEIGHT="${GT_GOOD_IMITATION_SUCCESS_WEIGHT:-1.0}" \
  MIXED_IMITATION_FAILURE_WEIGHT="${GT_GOOD_IMITATION_FAILURE_WEIGHT:-0.02}" \
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

  *)
    echo "Usage: $0 [square|can|transport|tool_hang] {build_chunks|check|train|train_resilient|eval_grid_resilient}" >&2
    exit 2
    ;;
esac
