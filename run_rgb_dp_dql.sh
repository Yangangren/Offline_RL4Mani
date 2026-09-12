#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

first_arg=${1:-}
first_arg=${first_arg,,}
first_arg=${first_arg//-/_}
case "$first_arg" in
  square|can|transport|tool_hang|pick_cup|pickup|stack_cup|stackcup|move_spoon|movespoon)
    TASK=$first_arg
    shift
    ;;
esac
TASK=${TASK:-square}
TASK=${TASK,,}
case "$TASK" in
  pickup) TASK=pick_cup ;;
  stackcup) TASK=stack_cup ;;
  movespoon) TASK=move_spoon ;;
esac
TASK_REAL_ROBOT=0
TASK_VALIDATION_DATASET=
TASK_EXPECTED_TRAIN_VALID_ROWS=-1
TASK_EXPECTED_VALIDATION_VALID_ROWS=-1
TASK_EXPECTED_TRAIN_SOURCE_COUNTS=
TASK_EXPECTED_VALIDATION_SOURCE_COUNTS=

# Keep these paths and selections identical to run_rgb_dp_idql.sh and
# run_rgb_dp_chunk_idql.sh. DQL consumes the same built mixed HDF5 directly.
case "$TASK" in
  square)
    TASK_DP_CHECKPOINT=trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/square/idql/square_rgb_dp_idql_200demo_406success_94failure.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=datasets/square/idql/square_rgb_dp_idql_200demo_406success_94failure_terminal_success.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/square_rgb_dp/dql/200demo_406success_94failure
    TASK_EVAL_OUTPUT=rollouts/square_rgb_dp/dql/200demo_406success_94failure
    TASK_CRITIC_GROUP_NORM=1
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  can)
    TASK_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    TASK_IDQL_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_467success_33failure.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_467success_33failure_terminal_success.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/can_rgb_dp/dql/200demo_467success_33failure
    TASK_EVAL_OUTPUT=rollouts/can_rgb_dp/dql/200demo_467success_33failure
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  transport)
    TASK_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_422success_78failure.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_422success_78failure_terminal_success_reward.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/transport_rgb_dp/dql/200demo_422success_78failure
    TASK_EVAL_OUTPUT=rollouts/transport_rgb_dp/dql/200demo_422success_78failure
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos,robot1_gripper_qpos
    ;;
  tool_hang)
    TASK_DP_CHECKPOINT=trained_models/tool_hang_rgb_dp/tool_hang_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_132success_168failure.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_132success_168failure_terminal_success.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/dql/200demo_132success_168failure
    TASK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/dql/200demo_132success_168failure
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  pick_cup)
    TASK_REAL_ROBOT=1
    TASK_DP_CHECKPOINT=trained_models/real_robot/pick_cup_rgb_dp/pick_cup_rgb_dp_ddim_s1/20260908111825/models/model_epoch_50.pth
    TASK_IDQL_DATASET=datasets/real_robot/pick_cup/idql/pick_cup_idql_episode_layout_v1_45demo_24success_8failure_terminal_success.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=$TASK_IDQL_DATASET
    TASK_VALIDATION_DATASET=datasets/real_robot/pick_cup/idql/pick_cup_idql_episode_layout_v1_validation_5demo_6success_2failure_terminal_success.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/real_robot/pick_cup_rgb_dp/dql/45demo_24success_8failure_terminal_success_rise_temporal_v2_episode_layout_v1_ddim100_eta0p01_minq_sampled_safe_v1
    TASK_EVAL_OUTPUT=rollouts/real_robot/pick_cup/dql/45demo_24success_8failure_terminal_success_rise_temporal_v2_episode_layout_v1_ddim100_eta0p01_minq_sampled_safe_v1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=600
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_state
    TASK_EXPECTED_TRAIN_VALID_ROWS=25032
    TASK_EXPECTED_VALIDATION_VALID_ROWS=3630
    TASK_EXPECTED_TRAIN_SOURCE_COUNTS=45,24,8
    TASK_EXPECTED_VALIDATION_SOURCE_COUNTS=5,6,2
    ;;
  stack_cup)
    TASK_REAL_ROBOT=1
    TASK_DP_CHECKPOINT=trained_models/real_robot/stack_cup_rgb_dp/stack_cup_rgb_dp_ddim_s1/20260902111545/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/real_robot/stack_cup/idql/stack_cup_idql_episode_layout_v1_45demo_20success_10failure_terminal_success.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=$TASK_IDQL_DATASET
    TASK_VALIDATION_DATASET=datasets/real_robot/stack_cup/idql/stack_cup_idql_episode_layout_v1_validation_5demo_6success_4failure_terminal_success.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/real_robot/stack_cup_rgb_dp/dql/45demo_20success_10failure_terminal_success_rise_temporal_v2_episode_layout_v1_ddim100_eta0p01_minq_sampled_safe_v1
    TASK_EVAL_OUTPUT=rollouts/real_robot/stack_cup/dql/45demo_20success_10failure_terminal_success_rise_temporal_v2_episode_layout_v1_ddim100_eta0p01_minq_sampled_safe_v1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=600
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_state
    TASK_EXPECTED_TRAIN_VALID_ROWS=34621
    TASK_EXPECTED_VALIDATION_VALID_ROWS=7585
    TASK_EXPECTED_TRAIN_SOURCE_COUNTS=45,20,10
    TASK_EXPECTED_VALIDATION_SOURCE_COUNTS=5,6,4
    ;;
  move_spoon)
    TASK_REAL_ROBOT=1
    TASK_DP_CHECKPOINT=trained_models/real_robot/move_spoon_rgb_dp/move_spoon_rgb_dp_ddim_s1/20260903104112/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/real_robot/move_spoon/idql/move_spoon_idql_episode_layout_v1_45demo_20success_11failure_terminal_success.hdf5
    TASK_TERMINAL_SUCCESS_DATASET=$TASK_IDQL_DATASET
    TASK_VALIDATION_DATASET=datasets/real_robot/move_spoon/idql/move_spoon_idql_episode_layout_v1_validation_5demo_5success_4failure_terminal_success.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/real_robot/move_spoon_rgb_dp/dql/45demo_20success_11failure_terminal_success_rise_temporal_v2_episode_layout_v1_ddim100_eta0p01_minq_sampled_safe_v1
    TASK_EVAL_OUTPUT=rollouts/real_robot/move_spoon/dql/45demo_20success_11failure_terminal_success_rise_temporal_v2_episode_layout_v1_ddim100_eta0p01_minq_sampled_safe_v1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=600
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_state
    TASK_EXPECTED_TRAIN_VALID_ROWS=34810
    TASK_EXPECTED_VALIDATION_VALID_ROWS=6811
    TASK_EXPECTED_TRAIN_SOURCE_COUNTS=45,20,11
    TASK_EXPECTED_VALIDATION_SOURCE_COUNTS=5,5,4
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use square, can, transport, tool_hang, pick_cup, stack_cup, or move_spoon." >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_${TASK}_rgb_dp_dql_pycache_${USER}_$$"
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
DQL_NUM_GPUS=${DQL_NUM_GPUS:-1}
if [[ ! "$DQL_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DQL_NUM_GPUS must be a positive integer; got '$DQL_NUM_GPUS'." >&2
  exit 2
fi
if (( DQL_NUM_GPUS > 1 )) && [[ "${DEVICE:-cuda}" != "cuda" ]]; then
  echo "DQL_NUM_GPUS>1 requires DEVICE=cuda." >&2
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
DQL_REWARD_MODE=${DQL_REWARD_MODE:-terminal_success}
case "$DQL_REWARD_MODE" in
  task)
    DEFAULT_IDQL_DATASET=${TASK_IDQL_DATASET%.hdf5}_task_reward.hdf5
    DEFAULT_DQL_OUTPUT_DIR=${TASK_DQL_OUTPUT_DIR}_task_reward
    DEFAULT_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_task_reward
    ;;
  terminal_success)
    DEFAULT_IDQL_DATASET=$TASK_TERMINAL_SUCCESS_DATASET
    DEFAULT_DQL_OUTPUT_DIR=${TASK_DQL_OUTPUT_DIR}_terminal_success_reward
    DEFAULT_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_terminal_success_reward
    ;;
  rise)
    DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
    DEFAULT_DQL_OUTPUT_DIR=$TASK_DQL_OUTPUT_DIR
    DEFAULT_EVAL_OUTPUT=$TASK_EVAL_OUTPUT
    ;;
  *)
    echo "Unsupported DQL_REWARD_MODE=$DQL_REWARD_MODE. Use task, terminal_success, or rise." >&2
    exit 2
    ;;
esac
if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
  if [[ "$DQL_REWARD_MODE" != "terminal_success" ]]; then
    echo "TASK=$TASK requires DQL_REWARD_MODE=terminal_success." >&2
    exit 2
  fi
  # The real-robot mixed HDF5 is already the canonical terminal-success
  # dataset shared with one-step IDQL; do not derive another filename suffix.
  DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
  DEFAULT_DQL_OUTPUT_DIR=$TASK_DQL_OUTPUT_DIR
  DEFAULT_EVAL_OUTPUT=$TASK_EVAL_OUTPUT
fi
IDQL_DATASET=${IDQL_DATASET:-$DEFAULT_IDQL_DATASET}
DQL_VALIDATION_DATASET=${DQL_VALIDATION_DATASET:-$TASK_VALIDATION_DATASET}
DQL_OUTPUT_DIR=${DQL_OUTPUT_DIR:-$DEFAULT_DQL_OUTPUT_DIR}
DQL_CHECKPOINT=${DQL_CHECKPOINT:-$DQL_OUTPUT_DIR/last.pt}
EVAL_OUTPUT=${EVAL_OUTPUT:-$DEFAULT_EVAL_OUTPUT}
EVAL_HORIZON=${HORIZON:-$TASK_EVAL_HORIZON}
CRITIC_LATE_FUSION_KEY=${CRITIC_LATE_FUSION_KEY:-$TASK_CRITIC_LATE_FUSION_KEY}

DEFAULT_DQL_ACTOR_LR=1e-4
DEFAULT_DQL_ACTOR_OBS_ENCODER_FREEZE_STEPS=0
DEFAULT_DQL_CRITIC_ENCODER_FREEZE_STEPS=0
DEFAULT_DQL_NUM_INFERENCE_STEPS=5
DEFAULT_DQL_ETA=1.0
DEFAULT_DQL_Q_HEAD=random
if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
  DEFAULT_DQL_ACTOR_LR=1e-5
  DEFAULT_DQL_ACTOR_OBS_ENCODER_FREEZE_STEPS=1000
  DEFAULT_DQL_CRITIC_ENCODER_FREEZE_STEPS=1000
  DEFAULT_DQL_NUM_INFERENCE_STEPS=100
  # Keep Q guidance conservative on the small, sparse-reward real-robot
  # datasets, and avoid selecting a single optimistic critic head.
  DEFAULT_DQL_ETA=0.01
  DEFAULT_DQL_Q_HEAD=min
fi

PIN_MEMORY_ARG=--pin-memory
if [[ "${PIN_MEMORY:-1}" == "0" ]]; then
  PIN_MEMORY_ARG=--no-pin-memory
fi
PERSISTENT_WORKERS_ARG=--persistent-workers
if [[ "${PERSISTENT_WORKERS:-1}" == "0" ]]; then
  PERSISTENT_WORKERS_ARG=--no-persistent-workers
fi
SPARSE_DQL_LOADER_ARG=--sparse-dql-loader
if [[ "${DQL_SPARSE_LOADER:-1}" == "0" ]]; then
  SPARSE_DQL_LOADER_ARG=--no-sparse-dql-loader
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
DQL_CLIP_ACTIONS_ARG=--dql-clip-actions
if [[ "${DQL_CLIP_ACTIONS:-1}" == "0" ]]; then
  DQL_CLIP_ACTIONS_ARG=--no-dql-clip-actions
fi

check_real_robot_contract() {
  "$PYTHON" -B - \
    "$TASK" \
    "$DP_CHECKPOINT" \
    "$IDQL_DATASET" \
    "$DQL_VALIDATION_DATASET" \
    "$TASK_EXPECTED_TRAIN_VALID_ROWS" \
    "$TASK_EXPECTED_VALIDATION_VALID_ROWS" \
    "$TASK_EXPECTED_TRAIN_SOURCE_COUNTS" \
    "$TASK_EXPECTED_VALIDATION_SOURCE_COUNTS" <<'PYCHECK'
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

(
    task,
    checkpoint_path,
    training_path,
    validation_path,
    expected_training_valid,
    expected_validation_valid,
    expected_training_counts,
    expected_validation_counts,
) = sys.argv[1:9]

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
if isinstance(config, str):
    config = json.loads(config)
horizon = config["algo"]["horizon"]
actual_horizons = (
    int(horizon["observation_horizon"]),
    int(horizon["action_horizon"]),
    int(horizon["prediction_horizon"]),
)
if actual_horizons != (2, 8, 16):
    raise ValueError(
        f"{checkpoint_path} has DP horizons={actual_horizons}; expected (2, 8, 16)"
    )
modalities = config["observation"]["modalities"]["obs"]
checkpoint_obs_keys = set(modalities["low_dim"]) | set(modalities["rgb"])
expected_obs_keys = {
    "main_image",
    "wrist_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
}
if checkpoint_obs_keys != expected_obs_keys:
    raise ValueError(
        f"{checkpoint_path} observation keys={sorted(checkpoint_obs_keys)}; "
        f"expected {sorted(expected_obs_keys)}"
    )
if config["train"]["action_config"] != {
    "actions": {"normalization": None}
}:
    raise ValueError(
        f"{checkpoint_path} must use identity-normalized actions"
    )


def validate_dataset(path, expected_valid, expected_counts, role):
    expected_sources = dict(
        zip(
            ("expert", "non_expert_success", "non_expert_failure"),
            (int(value) for value in expected_counts.split(",")),
        )
    )
    source_counts = {key: 0 for key in expected_sources}
    valid_rows = 0
    raw_rows = 0
    source_records = set()
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("task", "")) != task:
            raise ValueError(f"{path} task attribute does not equal {task}")
        if str(handle.attrs.get("reward_mode", "")) != "terminal_success":
            raise ValueError(f"{path} is not a terminal_success dataset")
        for demo_key, demo in handle["data"].items():
            source = demo.attrs.get("rise_source", "")
            if isinstance(source, bytes):
                source = source.decode("utf-8")
            source = str(source)
            if source not in source_counts:
                raise ValueError(f"{path}:data/{demo_key} has source={source!r}")
            source_counts[source] += 1
            num_samples = int(demo.attrs["num_samples"])
            raw_rows += num_samples
            validity = np.asarray(
                demo["one_step_critic_valid"][:], dtype=np.uint8
            ).reshape(-1)
            if validity.shape != (num_samples,) or np.any(
                ~np.isin(validity, (0, 1))
            ):
                raise ValueError(
                    f"{path}:data/{demo_key}/one_step_critic_valid is invalid"
                )
            valid_rows += int(validity.sum())
            if source != "expert":
                if not bool(demo.attrs.get("one_step_aligned", 0)):
                    raise ValueError(
                        f"{path}:data/{demo_key} is not one_step_aligned"
                    )
                one_step_obs = demo.get("one_step_obs")
                if one_step_obs is None or set(one_step_obs) != expected_obs_keys:
                    raise ValueError(
                        f"{path}:data/{demo_key}/one_step_obs keys are invalid"
                    )
                for obs_key in expected_obs_keys:
                    if one_step_obs[obs_key].shape[:2] != (num_samples, 2):
                        raise ValueError(
                            f"{path}:data/{demo_key}/one_step_obs/{obs_key} "
                            "does not have a two-frame history"
                        )
            source_file = demo.attrs.get("rise_source_file", "")
            source_demo = demo.attrs.get("rise_source_demo", "")
            if isinstance(source_file, bytes):
                source_file = source_file.decode("utf-8")
            if isinstance(source_demo, bytes):
                source_demo = source_demo.decode("utf-8")
            if not source_file or not source_demo:
                raise ValueError(
                    f"{path}:data/{demo_key} is missing source provenance"
                )
            record = (str(Path(str(source_file)).resolve()), str(source_demo))
            if record in source_records:
                raise ValueError(
                    f"{path}:data/{demo_key} has invalid source provenance"
                )
            source_records.add(record)
    if source_counts != expected_sources:
        raise ValueError(
            f"{path} source counts={source_counts}; expected {expected_sources}"
        )
    if valid_rows != int(expected_valid):
        raise ValueError(
            f"{path} valid rows={valid_rows}; expected {expected_valid}"
        )
    return {
        "path": path,
        "role": role,
        "source_counts": source_counts,
        "raw_rows": raw_rows,
        "one_step_valid_rows": valid_rows,
    }, source_records


training, training_records = validate_dataset(
    training_path,
    expected_training_valid,
    expected_training_counts,
    "train",
)
validation, validation_records = validate_dataset(
    validation_path,
    expected_validation_valid,
    expected_validation_counts,
    "validation",
)
overlap = training_records.intersection(validation_records)
if overlap:
    raise ValueError(
        f"training and validation datasets overlap: {sorted(overlap)[:8]}"
    )
print(
    json.dumps(
        {
            "task": task,
            "checkpoint": checkpoint_path,
            "horizons": actual_horizons,
            "training": training,
            "validation": validation,
            "source_episode_overlap": 0,
        },
        indent=2,
        sort_keys=True,
    )
)
PYCHECK
}

require_training_data() {
  if [[ ! -f "$DP_CHECKPOINT" ]]; then
    echo "[rgb_dp_dql task=$TASK] DP checkpoint does not exist: $DP_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -f "$IDQL_DATASET" ]]; then
    echo "[rgb_dp_dql task=$TASK] shared IDQL dataset does not exist: $IDQL_DATASET" >&2
    echo "Build it with: IDQL_REWARD_MODE=$DQL_REWARD_MODE ./run_rgb_dp_idql.sh $TASK build_dataset" >&2
    exit 1
  fi
  if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
    if [[ ! -f "$DQL_VALIDATION_DATASET" ]]; then
      echo "[rgb_dp_dql task=$TASK] validation dataset does not exist: $DQL_VALIDATION_DATASET" >&2
      exit 1
    fi
    echo "[rgb_dp_dql task=$TASK] validating one-step train and validation datasets" >&2
    check_real_robot_contract
  fi
}

run_train() {
  local resume_path=${1:-}
  local -a resume_args=()
  local -a steps_per_epoch_args=()
  local -a distributed_args=()
  local -a validation_args=()
  local -a sampled_validation_args=()
  local -a train_launcher=("$PYTHON" -B)
  if [[ -n "$resume_path" ]]; then
    resume_args=(--resume-checkpoint "$resume_path")
  fi
  if [[ -n "${STEPS_PER_EPOCH:-}" ]]; then
    steps_per_epoch_args=(--steps-per-epoch "$STEPS_PER_EPOCH")
  fi
  if [[ -n "$DQL_VALIDATION_DATASET" ]]; then
    validation_args=(
      --validation-dataset "$DQL_VALIDATION_DATASET"
      --validation-seed "${DQL_VALIDATION_SEED:-10000}"
    )
  fi
  if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
    sampled_validation_args=(
      --dql-sampled-validation-rows "${DQL_SAMPLED_VALIDATION_ROWS:-32}"
      --dql-sampled-validation-max-error-ratio "${DQL_SAMPLED_VALIDATION_MAX_ERROR_RATIO:-2.0}"
      --dql-sampled-validation-max-saturation-excess "${DQL_SAMPLED_VALIDATION_MAX_SATURATION_EXCESS:-0.25}"
    )
  fi
  if (( DQL_NUM_GPUS > 1 )); then
    train_launcher=(
      "$PYTHON" -B -m torch.distributed.run
      --standalone
      --nnodes=1
      "--nproc-per-node=$DQL_NUM_GPUS"
    )
    distributed_args=(
      --distributed
      --distributed-backend "${DQL_DISTRIBUTED_BACKEND:-auto}"
      --gradient-bucket-cap-mb "${DQL_GRADIENT_BUCKET_CAP_MB:-100}"
    )
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    echo "[rgb_dp_dql] distributed training: GPUs=$DQL_NUM_GPUS per-rank-batch=${BATCH_SIZE:-100} per-rank-q-batch=${DQL_Q_BATCH_SIZE:-8}" >&2
  fi
  "${train_launcher[@]}" scripts/train_rgb_dp_dql.py \
    --task "$TASK" \
    "${distributed_args[@]}" \
    --dataset "$IDQL_DATASET" \
    --checkpoint "$DP_CHECKPOINT" \
    --output-dir "$DQL_OUTPUT_DIR" \
    "${validation_args[@]}" \
    "${sampled_validation_args[@]}" \
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
    "$SPARSE_DQL_LOADER_ARG" \
    --hdf5-cache-mode "${HDF5_CACHE_MODE:-low_dim}" \
    --reward-mode "$DQL_REWARD_MODE" \
    --discount "${DISCOUNT:-0.99}" \
    --target-tau "${TARGET_TAU:-0.005}" \
    --actor-lr "${ACTOR_LR:-$DEFAULT_DQL_ACTOR_LR}" \
    --actor-obs-encoder-freeze-steps "${ACTOR_OBS_ENCODER_FREEZE_STEPS:-$DEFAULT_DQL_ACTOR_OBS_ENCODER_FREEZE_STEPS}" \
    --critic-lr "${CRITIC_LR:-1e-4}" \
    --encoder-lr "${ENCODER_LR:-1e-5}" \
    --encoder-freeze-steps "${ENCODER_FREEZE_STEPS:-$DEFAULT_DQL_CRITIC_ENCODER_FREEZE_STEPS}" \
    --lr-scheduler "${LR_SCHEDULER:-cosine}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS:-500}" \
    --lr-num-cycles "${LR_NUM_CYCLES:-0.5}" \
    --critic-hidden-dims ${CRITIC_HIDDEN_DIMS:-300 400 300} \
    --critic-observation-horizon "${DQL_CRITIC_OBSERVATION_HORIZON:-2}" \
    --latent-dim "${DQL_LATENT_DIM:-300}" \
    --action-hidden-dim "${DQL_ACTION_HIDDEN_DIM:-128}" \
    --num-attention-heads "${DQL_NUM_ATTENTION_HEADS:-4}" \
    --num-action-conv-layers "${DQL_NUM_ACTION_CONV_LAYERS:-2}" \
    --dropout "${DQL_DROPOUT:-0.0}" \
    --temporal-num-layers "${DQL_TEMPORAL_NUM_LAYERS:-2}" \
    --temporal-num-heads "${DQL_TEMPORAL_NUM_HEADS:-6}" \
    --temporal-feedforward-dim "${DQL_TEMPORAL_FEEDFORWARD_DIM:-600}" \
    --temporal-dropout "${DQL_TEMPORAL_DROPOUT:-0.0}" \
    --rise-v2-fusion-mode "${DQL_RISE_V2_FUSION_MODE:-film}" \
    --num-critics "${NUM_CRITICS:-2}" \
    "$CRITIC_GROUP_NORM_ARG" \
    --critic-late-fusion-key "$CRITIC_LATE_FUSION_KEY" \
    "$USE_HUBER_ARG" \
    --actor-max-gradient-norm "${ACTOR_MAX_GRADIENT_NORM:-1.0}" \
    --critic-max-gradient-norm "${CRITIC_MAX_GRADIENT_NORM:-10.0}" \
    --dql-eta "${DQL_ETA:-$DEFAULT_DQL_ETA}" \
    --dql-bc-weight "${DQL_BC_WEIGHT:-1.0}" \
    --dql-q-batch-size "${DQL_Q_BATCH_SIZE:-8}" \
    --dql-num-inference-steps "${DQL_NUM_INFERENCE_STEPS:-$DEFAULT_DQL_NUM_INFERENCE_STEPS}" \
    --dql-target-num-candidates "${DQL_TARGET_NUM_CANDIDATES:-1}" \
    --dql-q-head "${DQL_Q_HEAD:-$DEFAULT_DQL_Q_HEAD}" \
    --dql-q-denominator-floor "${DQL_Q_DENOMINATOR_FLOOR:-1.0}" \
    --dql-critic-warmup-steps "${DQL_CRITIC_WARMUP_STEPS:-1000}" \
    --dql-actor-ema-update-every "${DQL_ACTOR_EMA_UPDATE_EVERY:-5}" \
    "$DQL_CLIP_ACTIONS_ARG" \
    --log-every "${LOG_EVERY:-100}" \
    --save-every-epochs "${SAVE_EVERY_EPOCHS:-1}" \
    --snapshot-every-epochs "${SNAPSHOT_EVERY_EPOCHS:-10}"
}

STAGE=${1:-train_resilient}
case "$STAGE" in
  check)
    require_training_data
    ;;

  train)
    require_training_data
    run_train "${RESUME_CHECKPOINT:-}"
    ;;

  train_resilient)
    require_training_data
    max_restarts=${MAX_RESTARTS:-20}
    retry_sleep=${RETRY_SLEEP:-5}
    resume_path=${RESUME_CHECKPOINT:-}
    if [[ -z "$resume_path" && -f "$DQL_OUTPUT_DIR/latest.pt" ]]; then
      resume_path="$DQL_OUTPUT_DIR/latest.pt"
    fi
    attempt=1
    while (( attempt <= max_restarts )); do
      echo "[rgb_dp_dql task=$TASK attempt=$attempt/$max_restarts] resume=${resume_path:-none}" >&2
      set +e
      run_train "$resume_path"
      status=$?
      set -e
      if [[ "$status" -eq 0 ]]; then
        exit 0
      fi
      echo "[rgb_dp_dql task=$TASK attempt=$attempt] exited with status $status" >&2
      if [[ ! -f "$DQL_OUTPUT_DIR/latest.pt" ]]; then
        echo "[rgb_dp_dql task=$TASK] no latest.pt is available for recovery" >&2
        exit "$status"
      fi
      resume_path="$DQL_OUTPUT_DIR/latest.pt"
      attempt=$((attempt + 1))
      sleep "$retry_sleep"
    done
    echo "[rgb_dp_dql task=$TASK] exhausted $max_restarts attempts" >&2
    exit 1
    ;;

  eval)
    if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
      echo "[rgb_dp_dql task=$TASK] eval is simulation-only; use the dedicated real-robot deployment profile." >&2
      exit 2
    fi
    "$PYTHON" -B scripts/eval_rgb_dp_idql.py \
      --idql-checkpoint "$DQL_CHECKPOINT" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
      --actor-source hybrid_dp_chunk_actor \
      --critic-source "${CRITIC_SOURCE:-target}" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "$EVAL_HORIZON" \
      --seed "${EVAL_SEED:-0}" \
      --num-candidates "${N:-1}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --execution-horizon "${EXECUTION_HORIZON:-1}" \
      --selection "${SELECTION:-softmax}" \
      --softmax-temperature "${SOFTMAX_TEMPERATURE:-1.0}" \
      --clip-actions
    ;;

  eval_grid_resilient)
    if [[ "$TASK_REAL_ROBOT" == "1" ]]; then
      echo "[rgb_dp_dql task=$TASK] eval_grid_resilient is simulation-only; use the dedicated real-robot deployment profile." >&2
      exit 2
    fi
    read -r -a candidate_args <<< "${EVAL_NUM_CANDIDATES:-1 4 8 16 32 50}"
    read -r -a seed_args <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    "$PYTHON" -B scripts/run_rgb_dp_idql_eval_grid.py \
      --idql-checkpoint "$DQL_CHECKPOINT" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
      "${EVAL_GPU_ARGS[@]}" \
      --actor-source hybrid_dp_chunk_actor \
      --critic-source "${CRITIC_SOURCE:-target}" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "$EVAL_HORIZON" \
      --num-candidates "${candidate_args[@]}" \
      --seeds "${seed_args[@]}" \
      --rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-25}" \
      --inter-chunk-sleep "${EVAL_INTER_CHUNK_SLEEP:-0}" \
      --max-retries "${EVAL_MAX_RETRIES:-3}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --execution-horizon "${EXECUTION_HORIZON:-8}" \
      --selection "${SELECTION:-softmax}" \
      --softmax-temperature "${SOFTMAX_TEMPERATURE:-1.0}" \
      --clip-actions
    ;;

  *)
    echo "Usage: $0 [square|can|transport|tool_hang|pick_cup|stack_cup|move_spoon] {check|train|train_resilient|eval|eval_grid_resilient}" >&2
    exit 2
    ;;
esac
