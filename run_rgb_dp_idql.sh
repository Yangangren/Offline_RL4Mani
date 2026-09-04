#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

USER_REAL_ROBOT_ROLLOUT_SOURCE_ROOT_SET=${REAL_ROBOT_ROLLOUT_SOURCE_ROOT+x}

first_arg=${1:-}
first_arg=${first_arg,,}
first_arg=${first_arg//-/_}
if [[ "$first_arg" == "square" || "$first_arg" == "can" || "$first_arg" == "transport" || "$first_arg" == "tool_hang" || "$first_arg" == "pick_cup" || "$first_arg" == "stack_cup" ]]; then
  TASK=$first_arg
  shift
fi
TASK=${TASK:-square}
TASK=${TASK,,}
TASK_DEFAULT_IDQL_REWARD_MODE=terminal_success
TASK_REAL_ROBOT=0
TASK_REAL_ROBOT_HUMAN_DATASETS=
TASK_REAL_ROBOT_ROLLOUT_SOURCE_ROOT=
TASK_REAL_ROBOT_ROLLOUT_BUILDER=
TASK_REAL_ROBOT_MIXED_BUILDER=
TASK_REAL_ROBOT_VALIDATION_DATASET=
TASK_REAL_ROBOT_VALIDATION_HUMAN_TRANSITIONS=-1

case "$TASK" in
  square)
    TASK_DP_CHECKPOINT=trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/square/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/square/idql/square_rgb_dp_idql_200demo_406success_94failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/square_rgb_dp/idql/200demo_406success_94failure
    TASK_EVAL_OUTPUT=rollouts/square_rgb_dp/idql/200demo_406success_94failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=1
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    TASK_TERMINAL_SUCCESS_DATASET=datasets/square/idql/square_rgb_dp_idql_200demo_406success_94failure_terminal_success.hdf5
    ;;
  can)
    TASK_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    TASK_EXPERT_DATASET=datasets/can/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/can_rgb_dp/epoch50_collection/can_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_467success_33failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/can_rgb_dp/idql/200demo_467success_33failure
    TASK_EVAL_OUTPUT=rollouts/can_rgb_dp/idql/200demo_467success_33failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    TASK_TERMINAL_SUCCESS_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_467success_33failure_terminal_success.hdf5
    ;;
  transport)
    TASK_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/transport/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/transport_rgb_dp/epoch200_collection/transport_rgb_dp_rollouts_rgb4.hdf5
    TASK_IDQL_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_422success_78failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/transport_rgb_dp/idql/200demo_422success_78failure
    TASK_EVAL_OUTPUT=rollouts/transport_rgb_dp/idql/200demo_422success_78failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos,robot1_gripper_qpos
    TASK_TERMINAL_SUCCESS_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_422success_78failure_terminal_success_reward.hdf5
    ;;
  tool_hang)
    TASK_DP_CHECKPOINT=trained_models/tool_hang_rgb_dp/tool_hang_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/tool_hang/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/tool_hang_rgb_dp/epoch200_collection/tool_hang_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_132success_168failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/idql/200demo_132success_168failure
    TASK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/idql/200demo_132success_168failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    TASK_TERMINAL_SUCCESS_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_132success_168failure_terminal_success.hdf5
    ;;
  pick_cup)
    TASK_DP_CHECKPOINT=trained_models/real_robot/pick_cup_rgb_dp/pick_cup_rgb_dp_ddim_s1/20260816144749/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/real_robot/pick_cup/round1_rgb.hdf5
    TASK_ROLLOUT_DATASET=datasets/real_robot/pick_cup/idql/pick_cup_epoch200_20hz_rollouts.hdf5
    TASK_IDQL_DATASET=datasets/real_robot/pick_cup/idql/pick_cup_chunk_idql_65demo_23success_11failure_terminal_success.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/real_robot/pick_cup_rgb_dp/idql/65demo_23success_11failure_terminal_success
    TASK_EVAL_OUTPUT=rollouts/real_robot/pick_cup/idql/65demo_23success_11failure_terminal_success
    TASK_EXPERT_MASK=train
    TASK_EXPERT_COUNT=65
    TASK_SUCCESS_MASK=success_train
    TASK_SUCCESS_COUNT=23
    TASK_FAILURE_MASK=failure_train
    TASK_FAILURE_COUNT=11
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_state
    TASK_DEFAULT_IDQL_REWARD_MODE=terminal_success
    TASK_REAL_ROBOT=1
    TASK_REAL_ROBOT_HUMAN_DATASETS="datasets/real_robot/pick_cup/round1_rgb.hdf5 datasets/real_robot/pick_cup/round2_rgb.hdf5"
    TASK_REAL_ROBOT_ROLLOUT_SOURCE_ROOT=/home/ryan/datasets/pick_cup/rollout
    TASK_REAL_ROBOT_ROLLOUT_BUILDER=scripts/real_robot/build_pick_cup_rollout_hdf5.py
    TASK_REAL_ROBOT_MIXED_BUILDER=scripts/real_robot/build_pick_cup_chunk_idql_dataset.py
    ;;
  stack_cup)
    TASK_DP_CHECKPOINT=trained_models/real_robot/stack_cup_rgb_dp/stack_cup_rgb_dp_ddim_s1/20260902111545/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/real_robot/stack_cup/stack_cup_rgb.hdf5
    TASK_ROLLOUT_DATASET=datasets/real_robot/stack_cup/idql/stack_cup_epoch200_ddim100_20hz_rollouts.hdf5
    TASK_IDQL_DATASET=datasets/real_robot/stack_cup/idql/stack_cup_idql_44demo_20success_10failure_ddim100_terminal_success.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/real_robot/stack_cup_rgb_dp/idql/44demo_20success_10failure_ddim100_terminal_success_rise_temporal_v2_dynamics
    TASK_EVAL_OUTPUT=rollouts/real_robot/stack_cup/idql/44demo_20success_10failure_ddim100_terminal_success_rise_temporal_v2_dynamics
    TASK_EXPERT_MASK=train
    TASK_EXPERT_COUNT=44
    TASK_SUCCESS_MASK=success_train
    TASK_SUCCESS_COUNT=20
    TASK_FAILURE_MASK=failure_train
    TASK_FAILURE_COUNT=10
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=600
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_state
    TASK_DEFAULT_IDQL_REWARD_MODE=terminal_success
    TASK_REAL_ROBOT=1
    TASK_REAL_ROBOT_HUMAN_DATASETS=$TASK_EXPERT_DATASET
    TASK_REAL_ROBOT_ROLLOUT_SOURCE_ROOT=/home/ryan/datasets/stack_cup/rollout
    TASK_REAL_ROBOT_ROLLOUT_BUILDER=scripts/real_robot/build_stack_cup_processed_rollout_hdf5.py
    TASK_REAL_ROBOT_MIXED_BUILDER=scripts/real_robot/build_stack_cup_chunk_idql_dataset.py
    TASK_REAL_ROBOT_VALIDATION_DATASET=datasets/real_robot/stack_cup/idql/stack_cup_idql_validation_5demo_6success_4failure_ddim100_terminal_success.hdf5
    TASK_REAL_ROBOT_VALIDATION_HUMAN_TRANSITIONS=2286
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use square, can, transport, tool_hang, pick_cup, or stack_cup." >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_${TASK}_rgb_dp_idql_pycache_${USER}_$$"
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

PYTHON=${ROBOMIMIC_PYTHON:-/home/ryan/miniconda3/envs/robomimic_stable/bin/python}
export ROBOMIMIC_PYTHON="$PYTHON"
IDQL_NUM_GPUS=${IDQL_NUM_GPUS:-1}
if (( IDQL_NUM_GPUS > 1 )) && [[ "${DEVICE:-cuda}" != "cuda" ]]; then
  echo "IDQL_NUM_GPUS>1 requires DEVICE=cuda." >&2
  exit 2
fi
EVAL_GPU_ARGS=()
if [[ -n "${EVAL_NUM_GPUS:-}" ]]; then
  EVAL_GPU_ARGS+=(--num-gpus "$EVAL_NUM_GPUS")
fi
if [[ -n "${EVAL_GPU_IDS:-}" ]]; then
  read -r -a eval_gpu_id_args <<< "$EVAL_GPU_IDS"
  EVAL_GPU_ARGS+=(--gpu-ids "${eval_gpu_id_args[@]}")
fi

DP_CHECKPOINT=${DP_CHECKPOINT:-$TASK_DP_CHECKPOINT}
EXPERT_DATASET=${EXPERT_DATASET:-$TASK_EXPERT_DATASET}
ROLLOUT_DATASET=${ROLLOUT_DATASET:-$TASK_ROLLOUT_DATASET}
REAL_ROBOT_HUMAN_DATASETS=${REAL_ROBOT_HUMAN_DATASETS:-$TASK_REAL_ROBOT_HUMAN_DATASETS}
REAL_ROBOT_ROLLOUT_SOURCE_ROOT=${REAL_ROBOT_ROLLOUT_SOURCE_ROOT:-$TASK_REAL_ROBOT_ROLLOUT_SOURCE_ROOT}
REAL_ROBOT_VALIDATION_DATASET=${REAL_ROBOT_VALIDATION_DATASET:-$TASK_REAL_ROBOT_VALIDATION_DATASET}
REAL_ROBOT_VALIDATION_HUMAN_TRANSITIONS=${REAL_ROBOT_VALIDATION_HUMAN_TRANSITIONS:-$TASK_REAL_ROBOT_VALIDATION_HUMAN_TRANSITIONS}
IDQL_REWARD_MODE=${IDQL_REWARD_MODE:-$TASK_DEFAULT_IDQL_REWARD_MODE}
case "$IDQL_REWARD_MODE" in
  task)
    DEFAULT_IDQL_DATASET=${TASK_IDQL_DATASET%.hdf5}_task_reward.hdf5
    DEFAULT_IDQL_OUTPUT_DIR=${TASK_IDQL_OUTPUT_DIR}_task_reward
    DEFAULT_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_task_reward
    DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_pretrained_dp_actor_task_reward
    ;;
  terminal_success)
    DEFAULT_IDQL_DATASET=${TASK_TERMINAL_SUCCESS_DATASET:-${TASK_IDQL_DATASET%.hdf5}_terminal_success_reward.hdf5}
    DEFAULT_IDQL_OUTPUT_DIR=${TASK_IDQL_OUTPUT_DIR}_terminal_success_reward
    DEFAULT_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_terminal_success_reward
    DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_pretrained_dp_actor_terminal_success_reward
    ;;
  rise)
    DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
    DEFAULT_IDQL_OUTPUT_DIR=$TASK_IDQL_OUTPUT_DIR
    DEFAULT_EVAL_OUTPUT=$TASK_EVAL_OUTPUT
    DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_pretrained_dp_actor
    ;;
  *)
    echo "Unsupported IDQL_REWARD_MODE=$IDQL_REWARD_MODE. Use task, terminal_success, or rise." >&2
    exit 2
    ;;
esac
if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
  if [[ "$IDQL_REWARD_MODE" != "terminal_success" ]]; then
    echo "TASK=$TASK requires IDQL_REWARD_MODE=terminal_success." >&2
    exit 2
  fi
  # One-step and chunk IDQL deliberately consume the same preselected mixed
  # dataset so their comparison changes the learning horizon, not the data.
  DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
  DEFAULT_IDQL_OUTPUT_DIR=$TASK_IDQL_OUTPUT_DIR
  DEFAULT_EVAL_OUTPUT=$TASK_EVAL_OUTPUT
  DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_pretrained_dp_actor
fi
IDQL_DATASET=${IDQL_DATASET:-$DEFAULT_IDQL_DATASET}
IDQL_OUTPUT_DIR=${IDQL_OUTPUT_DIR:-$DEFAULT_IDQL_OUTPUT_DIR}
IDQL_CHECKPOINT=${IDQL_CHECKPOINT:-$IDQL_OUTPUT_DIR/last.pt}
EVAL_OUTPUT=${EVAL_OUTPUT:-$DEFAULT_EVAL_OUTPUT}
COMPOSED_DP_CHECKPOINT=${COMPOSED_DP_CHECKPOINT:-$TASK_DP_CHECKPOINT}
COMPOSED_CHUNK_EVAL_OUTPUT=${COMPOSED_CHUNK_EVAL_OUTPUT:-$DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT}
EVAL_HORIZON=${HORIZON:-$TASK_EVAL_HORIZON}
CRITIC_LATE_FUSION_KEY=${CRITIC_LATE_FUSION_KEY:-$TASK_CRITIC_LATE_FUSION_KEY}
EXPERT_MASK=${EXPERT_MASK:-$TASK_EXPERT_MASK}
EXPERT_COUNT=${EXPERT_COUNT:-$TASK_EXPERT_COUNT}
SUCCESS_MASK=${SUCCESS_MASK:-$TASK_SUCCESS_MASK}
SUCCESS_COUNT=${SUCCESS_COUNT:-$TASK_SUCCESS_COUNT}
FAILURE_MASK=${FAILURE_MASK:-$TASK_FAILURE_MASK}
FAILURE_COUNT=${FAILURE_COUNT:-$TASK_FAILURE_COUNT}

DEFAULT_IDQL_DYNAMICS_WEIGHT=0.0
DEFAULT_IDQL_ACTOR_UNET_LR=${ACTOR_LR:-1e-4}
DEFAULT_IDQL_ACTOR_OBS_ENCODER_LR=${ACTOR_LR:-1e-4}
DEFAULT_IDQL_ACTOR_OBS_ENCODER_FREEZE_STEPS=0
DEFAULT_IDQL_CRITIC_ENCODER_FREEZE_STEPS=0
DEFAULT_IDQL_VF_ENCODER_FREEZE_STEPS=0
if [[ "$TASK" == "stack_cup" ]]; then
  DEFAULT_IDQL_DYNAMICS_WEIGHT=0.05
  DEFAULT_IDQL_ACTOR_UNET_LR=1e-5
  DEFAULT_IDQL_ACTOR_OBS_ENCODER_LR=1e-5
  DEFAULT_IDQL_ACTOR_OBS_ENCODER_FREEZE_STEPS=1000
  DEFAULT_IDQL_CRITIC_ENCODER_FREEZE_STEPS=1000
  DEFAULT_IDQL_VF_ENCODER_FREEZE_STEPS=1000
fi

PIN_MEMORY_ARG=--pin-memory
if [[ "${PIN_MEMORY:-1}" == "0" ]]; then
  PIN_MEMORY_ARG=--no-pin-memory
fi
PERSISTENT_WORKERS_ARG=--persistent-workers
if [[ "${PERSISTENT_WORKERS:-1}" == "0" ]]; then
  PERSISTENT_WORKERS_ARG=--no-persistent-workers
fi
SPARSE_ONE_STEP_LOADER_ARG=--sparse-one-step-loader
if [[ "${IDQL_SPARSE_ONE_STEP_LOADER:-1}" == "0" ]]; then
  SPARSE_ONE_STEP_LOADER_ARG=--no-sparse-one-step-loader
fi
CRITIC_GROUP_NORM=${CRITIC_GROUP_NORM:-$TASK_CRITIC_GROUP_NORM}
CRITIC_GROUP_NORM_ARG=--no-critic-group-norm
if [[ "$CRITIC_GROUP_NORM" == "1" ]]; then
  CRITIC_GROUP_NORM_ARG=--critic-group-norm
fi
USE_HUBER_ARG=--no-use-huber
if [[ "${USE_HUBER:-0}" == "1" ]]; then
  USE_HUBER_ARG=--use-huber
fi
MAX_GRADIENT_NORM_ARGS=()
if [[ -n "${MAX_GRADIENT_NORM:-}" ]]; then
  MAX_GRADIENT_NORM_ARGS=(--max-gradient-norm "$MAX_GRADIENT_NORM")
fi
COMPOSED_CONDITION_ARGS=(
  --no-require-success-condition-adapter
  --forbid-success-condition-adapter
)
if [[ "${COMPOSED_CONDITIONED_ACTOR:-0}" == "1" ]]; then
  COMPOSED_CONDITION_ARGS=(
    --require-success-condition-adapter
    --no-forbid-success-condition-adapter
    --inference-success-condition 1.0
    --inference-condition-mask 1.0
  )
fi

require_simulation_stage_task() {
  local stage_name=$1
  if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
    echo "[rgb_dp_idql task=$TASK] stage '$stage_name' is simulation-only and cannot create or control the real robot." >&2
    echo "Use the task's dedicated 20 Hz real-robot deployment path for shadow evaluation or guarded execution." >&2
    exit 2
  fi
}

ensure_real_robot_rollout_dataset() {
  if [[ -f "$ROLLOUT_DATASET" && -s "$ROLLOUT_DATASET" ]]; then
    if [[ "${REAL_ROBOT_ROLLOUT_OUTPUT_ONLY_VALIDATION:-0}" == "1" ]]; then
      echo "[rgb_dp_idql task=$TASK] output-only rollout validation was explicitly requested: $ROLLOUT_DATASET" >&2
      "$PYTHON" -B "$TASK_REAL_ROBOT_ROLLOUT_BUILDER" \
        --output "$ROLLOUT_DATASET" \
        --validate-output-only
    elif [[ -d "$REAL_ROBOT_ROLLOUT_SOURCE_ROOT" || -n "$USER_REAL_ROBOT_ROLLOUT_SOURCE_ROOT_SET" ]]; then
      echo "[rgb_dp_idql task=$TASK] validating converted rollout provenance: $ROLLOUT_DATASET" >&2
      "$PYTHON" -B "$TASK_REAL_ROBOT_ROLLOUT_BUILDER" \
        --source-root "$REAL_ROBOT_ROLLOUT_SOURCE_ROOT" \
        --output "$ROLLOUT_DATASET" \
        --validate-only
    else
      echo "[rgb_dp_idql task=$TASK] raw rollout source is unavailable; validating the HDF5 and its embedded immutable manifest: $ROLLOUT_DATASET" >&2
      "$PYTHON" -B "$TASK_REAL_ROBOT_ROLLOUT_BUILDER" \
        --output "$ROLLOUT_DATASET" \
        --validate-output-only
    fi
    return
  fi
  if [[ ! -d "$REAL_ROBOT_ROLLOUT_SOURCE_ROOT" ]]; then
    echo "[rgb_dp_idql task=$TASK] raw rollout root does not exist: $REAL_ROBOT_ROLLOUT_SOURCE_ROOT" >&2
    echo "Set REAL_ROBOT_ROLLOUT_SOURCE_ROOT to the organized success/failure rollout directory." >&2
    exit 1
  fi
  echo "[rgb_dp_idql task=$TASK] converting real-robot rollouts: $ROLLOUT_DATASET" >&2
  "$PYTHON" -B "$TASK_REAL_ROBOT_ROLLOUT_BUILDER" \
    --source-root "$REAL_ROBOT_ROLLOUT_SOURCE_ROOT" \
    --output "$ROLLOUT_DATASET"
  if [[ ! -f "$ROLLOUT_DATASET" || ! -s "$ROLLOUT_DATASET" ]]; then
    echo "[rgb_dp_idql task=$TASK] rollout conversion did not create a non-empty dataset: $ROLLOUT_DATASET" >&2
    exit 1
  fi
}

run_real_robot_mixed_builder() {
  local validation_only=${1:-0}
  local -a human_datasets=()
  local -a human_args=()
  local -a mode_args=()
  read -r -a human_datasets <<< "$REAL_ROBOT_HUMAN_DATASETS"
  if (( ${#human_datasets[@]} == 0 )); then
    echo "REAL_ROBOT_HUMAN_DATASETS must contain at least one human HDF5 path." >&2
    exit 2
  fi
  for dataset_path in "${human_datasets[@]}"; do
    if [[ ! -f "$dataset_path" || ! -s "$dataset_path" ]]; then
      echo "[rgb_dp_idql task=$TASK] human dataset does not exist or is empty: $dataset_path" >&2
      exit 1
    fi
    human_args+=(--human-dataset "$dataset_path")
  done
  ensure_real_robot_rollout_dataset
  if [[ "$validation_only" == "1" || ( -f "$IDQL_DATASET" && "${OVERWRITE_DATASET:-0}" != "1" ) ]]; then
    mode_args=(--validate-only)
  elif [[ "${OVERWRITE_DATASET:-0}" == "1" ]]; then
    mode_args=(--overwrite)
  fi
  "$PYTHON" -B "$TASK_REAL_ROBOT_MIXED_BUILDER" \
    --task "$TASK" \
    "${human_args[@]}" \
    --rollout-dataset "$ROLLOUT_DATASET" \
    --output "$IDQL_DATASET" \
    --human-mask "$EXPERT_MASK" \
    --human-count "$EXPERT_COUNT" \
    --success-mask "$SUCCESS_MASK" \
    --success-count "$SUCCESS_COUNT" \
    --failure-mask "$FAILURE_MASK" \
    --failure-count "$FAILURE_COUNT" \
    --reward-mode "$IDQL_REWARD_MODE" \
    --actor-condition-mode human_only \
    --seed "${DATASET_SEED:-0}" \
    "${mode_args[@]}"
}

run_real_robot_validation_builder() {
  local validation_only=${1:-0}
  local -a human_datasets=()
  local -a human_args=()
  local -a mode_args=()
  read -r -a human_datasets <<< "$REAL_ROBOT_HUMAN_DATASETS"
  if (( ${#human_datasets[@]} == 0 )); then
    echo "REAL_ROBOT_HUMAN_DATASETS must contain at least one human HDF5 path." >&2
    exit 2
  fi
  if [[ -z "$REAL_ROBOT_VALIDATION_DATASET" ]]; then
    return
  fi
  for dataset_path in "${human_datasets[@]}"; do
    if [[ ! -f "$dataset_path" || ! -s "$dataset_path" ]]; then
      echo "[rgb_dp_idql task=$TASK] human dataset does not exist or is empty: $dataset_path" >&2
      exit 1
    fi
    human_args+=(--human-dataset "$dataset_path")
  done
  if [[ ! -f "$ROLLOUT_DATASET" || ! -s "$ROLLOUT_DATASET" ]]; then
    echo "[rgb_dp_idql task=$TASK] rollout dataset does not exist or is empty: $ROLLOUT_DATASET" >&2
    exit 1
  fi
  if [[ "$validation_only" == "1" || ( -f "$REAL_ROBOT_VALIDATION_DATASET" && "${OVERWRITE_DATASET:-0}" != "1" ) ]]; then
    mode_args=(--validate-only)
  elif [[ "${OVERWRITE_DATASET:-0}" == "1" ]]; then
    mode_args=(--overwrite)
  fi
  "$PYTHON" -B "$TASK_REAL_ROBOT_MIXED_BUILDER" \
    --task "$TASK" \
    "${human_args[@]}" \
    --rollout-dataset "$ROLLOUT_DATASET" \
    --output "$REAL_ROBOT_VALIDATION_DATASET" \
    --selection-role validation \
    --human-mask valid \
    --human-count -1 \
    --expected-human-transitions "$REAL_ROBOT_VALIDATION_HUMAN_TRANSITIONS" \
    --success-mask success_valid \
    --success-count -1 \
    --failure-mask failure_valid \
    --failure-count -1 \
    --reward-mode "$IDQL_REWARD_MODE" \
    --actor-condition-mode human_only \
    --seed "${DATASET_SEED:-0}" \
    "${mode_args[@]}"
}

build_dataset() {
  local overwrite_args=()
  if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
    run_real_robot_mixed_builder 0
    run_real_robot_validation_builder 0
    return
  fi
  if [[ ! -f "$EXPERT_DATASET" ]]; then
    echo "[rgb_dp_idql task=$TASK] expert dataset does not exist: $EXPERT_DATASET" >&2
    exit 1
  fi
  if [[ ! -f "$ROLLOUT_DATASET" ]]; then
    echo "[rgb_dp_idql task=$TASK] rollout dataset does not exist: $ROLLOUT_DATASET" >&2
    echo "Collect and label success_100 / failure rollouts, or override ROLLOUT_DATASET." >&2
    exit 1
  fi
  if [[ "${OVERWRITE_DATASET:-0}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  "$PYTHON" -B scripts/build_rgb_dp_idql_dataset.py \
    --task "$TASK" \
    --expert-dataset "$EXPERT_DATASET" \
    --rollout-dataset "$ROLLOUT_DATASET" \
    --output "$IDQL_DATASET" \
    --expert-mask "$EXPERT_MASK" \
    --expert-count "$EXPERT_COUNT" \
    --success-mask "$SUCCESS_MASK" \
    --success-count "$SUCCESS_COUNT" \
    --failure-mask "$FAILURE_MASK" \
    --failure-count "$FAILURE_COUNT" \
    --reward-mode "$IDQL_REWARD_MODE" \
    --seed "${DATASET_SEED:-0}" \
    "${overwrite_args[@]}"
}

ensure_dataset() {
  if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
    run_real_robot_mixed_builder 0
    run_real_robot_validation_builder 0
    return
  fi
  if [[ ! -f "$IDQL_DATASET" ]]; then
    echo "[rgb_dp_idql] building missing mixed dataset: $IDQL_DATASET" >&2
    build_dataset
  fi
}

run_train() {
  local resume_path="${1:-}"
  local resume_args=()
  local steps_per_epoch_args=()
  local distributed_args=()
  local heldout_args=()
  local train_launcher=("$PYTHON" -B)
  if [[ -n "$resume_path" ]]; then
    resume_args=(--resume-checkpoint "$resume_path")
  fi
  if [[ -n "${STEPS_PER_EPOCH:-}" ]]; then
    steps_per_epoch_args=(--steps-per-epoch "$STEPS_PER_EPOCH")
  fi
  if [[ "$TASK_REAL_ROBOT" == "1" && -n "$REAL_ROBOT_VALIDATION_DATASET" ]]; then
    heldout_args=(
      --validation-dataset "$REAL_ROBOT_VALIDATION_DATASET"
      --validation-seed "${IDQL_VALIDATION_SEED:-10000}"
    )
  fi
  if (( IDQL_NUM_GPUS > 1 )); then
    train_launcher=(
      "$PYTHON" -B -m torch.distributed.run
      --standalone
      --nnodes=1
      "--nproc-per-node=$IDQL_NUM_GPUS"
    )
    distributed_args=(
      --distributed
      --distributed-backend "${IDQL_DISTRIBUTED_BACKEND:-auto}"
      --gradient-bucket-cap-mb "${IDQL_GRADIENT_BUCKET_CAP_MB:-100}"
    )
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    echo "[rgb_dp_idql] distributed training: GPUs=$IDQL_NUM_GPUS per-rank-batch=${BATCH_SIZE:-64}" >&2
  fi
  "${train_launcher[@]}" scripts/train_rgb_dp_idql.py \
    --task "$TASK" \
    "${distributed_args[@]}" \
    --dataset "$IDQL_DATASET" \
    --checkpoint "$DP_CHECKPOINT" \
    --output-dir "$IDQL_OUTPUT_DIR" \
    "${heldout_args[@]}" \
    "${resume_args[@]}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-0}" \
    --epochs "${EPOCHS:-50}" \
    "${steps_per_epoch_args[@]}" \
    --schedule-reference-batch-size "${BATCH_SIZE:-100}" \
    --batch-size "${BATCH_SIZE:-100}" \
    --num-workers "${NUM_WORKERS:-6}" \
    --prefetch-factor "${PREFETCH_FACTOR:-2}" \
    "$PIN_MEMORY_ARG" \
    "$PERSISTENT_WORKERS_ARG" \
    "$SPARSE_ONE_STEP_LOADER_ARG" \
    --hdf5-cache-mode "${HDF5_CACHE_MODE:-low_dim}" \
    --reward-mode "$IDQL_REWARD_MODE" \
    --discount "${DISCOUNT:-0.99}" \
    --expectile "${EXPECTILE:-0.9}" \
    --target-tau "${TARGET_TAU:-0.01}" \
    --actor-lr "${ACTOR_LR:-1e-4}" \
    --actor-unet-lr "${ACTOR_UNET_LR:-$DEFAULT_IDQL_ACTOR_UNET_LR}" \
    --actor-obs-encoder-lr "${ACTOR_OBS_ENCODER_LR:-$DEFAULT_IDQL_ACTOR_OBS_ENCODER_LR}" \
    --actor-obs-encoder-freeze-steps "${ACTOR_OBS_ENCODER_FREEZE_STEPS:-$DEFAULT_IDQL_ACTOR_OBS_ENCODER_FREEZE_STEPS}" \
    --critic-lr "${CRITIC_LR:-1e-4}" \
    --encoder-lr "${ENCODER_LR:-1e-5}" \
    --encoder-freeze-steps "${ENCODER_FREEZE_STEPS:-$DEFAULT_IDQL_CRITIC_ENCODER_FREEZE_STEPS}" \
    --vf-lr "${VF_LR:-1e-4}" \
    --vf-encoder-freeze-steps "${VF_ENCODER_FREEZE_STEPS:-$DEFAULT_IDQL_VF_ENCODER_FREEZE_STEPS}" \
    --lr-scheduler "${LR_SCHEDULER:-cosine}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS:-500}" \
    --lr-num-cycles "${LR_NUM_CYCLES:-0.5}" \
    --critic-hidden-dims ${CRITIC_HIDDEN_DIMS:-300 400 300} \
    --critic-observation-horizon "${IDQL_CRITIC_OBSERVATION_HORIZON:-2}" \
    --latent-dim "${IDQL_LATENT_DIM:-300}" \
    --action-hidden-dim "${IDQL_ACTION_HIDDEN_DIM:-128}" \
    --num-attention-heads "${IDQL_NUM_ATTENTION_HEADS:-4}" \
    --num-action-conv-layers "${IDQL_NUM_ACTION_CONV_LAYERS:-2}" \
    --dropout "${IDQL_DROPOUT:-0.0}" \
    --temporal-num-layers "${IDQL_TEMPORAL_NUM_LAYERS:-2}" \
    --temporal-num-heads "${IDQL_TEMPORAL_NUM_HEADS:-6}" \
    --temporal-feedforward-dim "${IDQL_TEMPORAL_FEEDFORWARD_DIM:-600}" \
    --temporal-dropout "${IDQL_TEMPORAL_DROPOUT:-0.0}" \
    --rise-v2-fusion-mode "${IDQL_RISE_V2_FUSION_MODE:-film}" \
    --dynamics-weight "${DYNAMICS_WEIGHT:-$DEFAULT_IDQL_DYNAMICS_WEIGHT}" \
    --num-critics "${NUM_CRITICS:-2}" \
    "$CRITIC_GROUP_NORM_ARG" \
    --critic-late-fusion-key "$CRITIC_LATE_FUSION_KEY" \
    "$USE_HUBER_ARG" \
    "${MAX_GRADIENT_NORM_ARGS[@]}" \
    --log-every "${LOG_EVERY:-100}" \
    --save-every-epochs "${SAVE_EVERY_EPOCHS:-1}" \
    --snapshot-every-epochs "${SNAPSHOT_EVERY_EPOCHS:-10}"
}

STAGE=${1:-train_resilient}
case "$STAGE" in
  build_dataset)
    build_dataset
    ;;

  train)
    ensure_dataset
    run_train "${RESUME_CHECKPOINT:-}"
    ;;

  train_resilient)
    ensure_dataset
    max_restarts=${MAX_RESTARTS:-20}
    retry_sleep=${RETRY_SLEEP:-5}
    resume_path=${RESUME_CHECKPOINT:-}
    if [[ -z "$resume_path" && -f "$IDQL_OUTPUT_DIR/latest.pt" ]]; then
      resume_path="$IDQL_OUTPUT_DIR/latest.pt"
    fi
    attempt=1
    while (( attempt <= max_restarts )); do
      echo "[rgb_dp_idql attempt=$attempt/$max_restarts] resume=${resume_path:-none}" >&2
      set +e
      run_train "$resume_path"
      status=$?
      set -e
      if [[ "$status" -eq 0 ]]; then
        exit 0
      fi
      echo "[rgb_dp_idql attempt=$attempt] exited with status $status" >&2
      if [[ ! -f "$IDQL_OUTPUT_DIR/latest.pt" ]]; then
        echo "[rgb_dp_idql] no latest.pt is available for recovery" >&2
        exit "$status"
      fi
      resume_path="$IDQL_OUTPUT_DIR/latest.pt"
      attempt=$((attempt + 1))
      sleep "$retry_sleep"
    done
    echo "[rgb_dp_idql] exhausted $max_restarts attempts" >&2
    exit 1
    ;;

  eval)
    require_simulation_stage_task "$STAGE"
    "$PYTHON" -B scripts/eval_rgb_dp_idql.py \
      --idql-checkpoint "$IDQL_CHECKPOINT" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
      --actor-source hybrid_dp_chunk_actor \
      --critic-source "${CRITIC_SOURCE:-online}" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "$EVAL_HORIZON" \
      --seed "${EVAL_SEED:-0}" \
      --num-candidates "${N:-16}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --execution-horizon "${EXECUTION_HORIZON:-8}" \
      --selection "${SELECTION:-argmax}" \
      --clip-actions
    ;;

  eval_grid_resilient)
    require_simulation_stage_task "$STAGE"
    read -r -a candidate_args <<< "${EVAL_NUM_CANDIDATES:-1 4 8 16 32 64}"
    read -r -a seed_args <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    "$PYTHON" -B scripts/run_rgb_dp_idql_eval_grid.py \
      --idql-checkpoint "$IDQL_CHECKPOINT" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
      "${EVAL_GPU_ARGS[@]}" \
      --actor-source hybrid_dp_chunk_actor \
      --critic-source "${CRITIC_SOURCE:-online}" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "$EVAL_HORIZON" \
      --num-candidates "${candidate_args[@]}" \
      --seeds "${seed_args[@]}" \
      --rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-10}" \
      --inter-chunk-sleep "${EVAL_INTER_CHUNK_SLEEP:-0}" \
      --max-retries "${EVAL_MAX_RETRIES:-3}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --execution-horizon "${EXECUTION_HORIZON:-8}" \
      --selection "${SELECTION:-argmax}" \
      --clip-actions
    ;;

  eval_composed_chunk_grid_resilient)
    require_simulation_stage_task "$STAGE"
    read -r -a candidate_args <<< "${EVAL_NUM_CANDIDATES:-4 8 12 16}"
    read -r -a seed_args <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    "$PYTHON" -B scripts/run_rgb_dp_idql_eval_grid.py \
      --idql-checkpoint "$IDQL_CHECKPOINT" \
      --dp-checkpoint "$COMPOSED_DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$COMPOSED_CHUNK_EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
      "${EVAL_GPU_ARGS[@]}" \
      --actor-source external_dp_chunk_critic \
      --critic-source "${CRITIC_SOURCE:-online}" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "$EVAL_HORIZON" \
      --num-candidates "${candidate_args[@]}" \
      --seeds "${seed_args[@]}" \
      --rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-25}" \
      --inter-chunk-sleep "${EVAL_INTER_CHUNK_SLEEP:-0}" \
      --max-retries "${EVAL_MAX_RETRIES:-3}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --execution-horizon "${EXECUTION_HORIZON:-8}" \
      --selection "${SELECTION:-argmax}" \
      "${COMPOSED_CONDITION_ARGS[@]}" \
      --clip-actions
    ;;

  *)
    echo "Usage: $0 [square|can|transport|tool_hang|pick_cup|stack_cup] {build_dataset|train|train_resilient|eval|eval_grid_resilient|eval_composed_chunk_grid_resilient}" >&2
    exit 2
    ;;
esac
