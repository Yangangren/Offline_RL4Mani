#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "can" || "${1:-}" == "Can" || "${1:-}" == "square" || "${1:-}" == "Square" || "${1:-}" == "transport" || "${1:-}" == "Transport" ]]; then
  TASK=$1
  shift
fi
TASK=${TASK:-${RGB_DP_TASK:-can}}
TASK=${TASK,,}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_${TASK}_rgb_dp_imitation_pycache_${USER}_$$"
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

case "$TASK" in
  can)
    TASK_RGB_PREFIX=can_rgb_dp
    DEFAULT_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    DEFAULT_DEMO_DATASET=datasets/can/ph/image_v15.hdf5
    DEFAULT_ROLLOUT_DATASET=rollouts/can_rgb_dp/epoch50_collection/can_rgb_dp_rollouts_rgb2.hdf5
    DEFAULT_FAILURE_FILTER_SIZE=33
    DEFAULT_FAILURE_FILTER_KEY=failure
    DEFAULT_SELF_IMITATION_OUTPUT_DIR=trained_models/can_rgb_dp/mixed_imitation/200demo_100success
    DEFAULT_MIXED_IMITATION_OUTPUT_DIR=trained_models/can_rgb_dp/mixed_imitation/200demo_100success_33failure
    DEFAULT_CONDITIONED_IMITATION_OUTPUT_DIR=trained_models/can_rgb_dp/mixed_imitation/200demo_100success_33failure_conditioned
    DEFAULT_EVAL_OUTPUT=rollouts/can_rgb_dp/imitation_eval
    DEFAULT_HORIZON=400
    ;;
  square)
    TASK_RGB_PREFIX=square_rgb_dp
    DEFAULT_DP_CHECKPOINT=trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    DEFAULT_DEMO_DATASET=datasets/square/ph/image_v15.hdf5
    DEFAULT_ROLLOUT_DATASET=rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5
    DEFAULT_FAILURE_FILTER_SIZE=50
    DEFAULT_FAILURE_FILTER_KEY=failure_50
    DEFAULT_SELF_IMITATION_OUTPUT_DIR=trained_models/square_rgb_dp_self_imitation/200demo_100success
    DEFAULT_MIXED_IMITATION_OUTPUT_DIR=trained_models/square_rgb_dp/mixed_imitation/200demo_100success_50failure
    DEFAULT_CONDITIONED_IMITATION_OUTPUT_DIR=trained_models/square_rgb_dp/mixed_imitation/200demo_100success_50failure_human_condition
    DEFAULT_EVAL_OUTPUT=rollouts/square_rgb_dp/imitation_eval
    DEFAULT_HORIZON=400
    ;;
  transport)
    TASK_RGB_PREFIX=transport_rgb_dp
    DEFAULT_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    DEFAULT_DEMO_DATASET=datasets/transport/ph/image_v15.hdf5
    DEFAULT_ROLLOUT_DATASET=rollouts/transport_rgb_dp/epoch200_collection/transport_rgb_dp_rollouts_rgb4.hdf5
    DEFAULT_FAILURE_FILTER_SIZE=50
    DEFAULT_FAILURE_FILTER_KEY=failure_50
    DEFAULT_SELF_IMITATION_OUTPUT_DIR=trained_models/transport_rgb_dp/mixed_imitation/200demo_100success
    DEFAULT_MIXED_IMITATION_OUTPUT_DIR=trained_models/transport_rgb_dp/mixed_imitation/200demo_100success_50failure
    DEFAULT_CONDITIONED_IMITATION_OUTPUT_DIR=trained_models/transport_rgb_dp/mixed_imitation/200demo_100success_50failure_conditioned
    DEFAULT_EVAL_OUTPUT=rollouts/transport_rgb_dp/imitation_eval
    DEFAULT_HORIZON=700
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use TASK=can, TASK=square, or TASK=transport." >&2
    exit 2
    ;;
esac

STAGE=${1:-train_mixed_imitation_resilient}
IMITATION_KIND=${IMITATION_KIND:-mixed}
case "$STAGE" in
  train|train_mixed_imitation)
    STAGE=train
    IMITATION_KIND=mixed
        ;;
  train_resilient|train_mixed_imitation_resilient)
    STAGE=train_resilient
    IMITATION_KIND=mixed
    ;;
  train_self_imitation)
    STAGE=train
    IMITATION_KIND=self
    ;;
  train_self_imitation_resilient)
    STAGE=train_resilient
    IMITATION_KIND=self
    ;;
  check_self)
    STAGE=check
    IMITATION_KIND=self
    ;;
  check_mixed)
    STAGE=check
    IMITATION_KIND=mixed
    ;;
  check_conditioned|check_conditioned_mixed_imitation)
    STAGE=check
    IMITATION_KIND=mixed
    CONDITIONED_MIXED_IMITATION=1
    ;;
  prepare_self_filters|prepare_filters_self)
    STAGE=prepare_filters
    IMITATION_KIND=self
    ;;
  prepare_mixed_filters|prepare_filters_mixed)
    STAGE=prepare_filters
    IMITATION_KIND=mixed
    ;;
  all_self)
    STAGE=all
    IMITATION_KIND=self
    ;;
  all_mixed)
    STAGE=all
    IMITATION_KIND=mixed
    ;;
  train_conditioned|train_conditioned_mixed_imitation)
    STAGE=train
    IMITATION_KIND=mixed
    CONDITIONED_MIXED_IMITATION=1
    ;;
  train_conditioned_resilient|train_conditioned_imitation_resilient)
    STAGE=train_resilient
    IMITATION_KIND=mixed
    CONDITIONED_MIXED_IMITATION=1
    ;;
  eval_self|eval_self_grid_resilient)
    STAGE=eval_grid_resilient
    IMITATION_KIND=self
    ;;
  eval_mixed|eval_mixed_grid_resilient)
    STAGE=eval_grid_resilient
    IMITATION_KIND=mixed
    ;;
  eval_conditioned|eval_conditioned_grid_resilient|eval_conditioned_mixed_imitation_resilient)
    STAGE=eval_grid_resilient
    IMITATION_KIND=mixed
    CONDITIONED_MIXED_IMITATION=1
    ;;
esac

IMITATION_KIND=${IMITATION_KIND,,}
if [[ "$IMITATION_KIND" != "self" && "$IMITATION_KIND" != "mixed" ]]; then
  echo "Unsupported IMITATION_KIND=$IMITATION_KIND. Use self or mixed." >&2
  exit 2
fi

# post-training
DP_CHECKPOINT=${DP_CHECKPOINT:-$DEFAULT_DP_CHECKPOINT}
DEMO_DATASET=${DEMO_DATASET:-$DEFAULT_DEMO_DATASET}
ROLLOUT_DATASET=${ROLLOUT_DATASET:-$DEFAULT_ROLLOUT_DATASET}
SUCCESS_DATASET=${SUCCESS_DATASET:-$ROLLOUT_DATASET}
FAILURE_DATASET=${FAILURE_DATASET:-$ROLLOUT_DATASET}
HORIZON=${HORIZON:-$DEFAULT_HORIZON}

ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-100}
ACTOR_LR=${ACTOR_LR:-}
ACTOR_HDF5_CACHE_MODE=${ACTOR_HDF5_CACHE_MODE:-}
ACTOR_NUM_WORKERS=${ACTOR_NUM_WORKERS:-4}
ACTOR_UNIFORM_SAMPLE_POOL=${ACTOR_UNIFORM_SAMPLE_POOL:-1}
ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE=${ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE:-0}
CONDITION_LABEL_MODE=${CONDITION_LABEL_MODE:-human_only}
IMITATION_STEPS_PER_EPOCH=${IMITATION_STEPS_PER_EPOCH:-}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
LOG_EVERY=${LOG_EVERY:-100}

SELF_IMITATION_OUTPUT_DIR=${SELF_IMITATION_OUTPUT_DIR:-$DEFAULT_SELF_IMITATION_OUTPUT_DIR}
MIXED_IMITATION_OUTPUT_DIR=${MIXED_IMITATION_OUTPUT_DIR:-$DEFAULT_MIXED_IMITATION_OUTPUT_DIR}
CONDITIONED_IMITATION_OUTPUT_DIR=${CONDITIONED_IMITATION_OUTPUT_DIR:-$DEFAULT_CONDITIONED_IMITATION_OUTPUT_DIR}

SUCCESS_SOURCE_FILTER_KEY=${SUCCESS_SOURCE_FILTER_KEY:-success}
SUCCESS_FILTER_SIZE=${SUCCESS_FILTER_SIZE:-100}
SUCCESS_FILTER_SUFFIX=${SUCCESS_FILTER_SUFFIX:-100}
SUCCESS_SELECTION_SEED=${SUCCESS_SELECTION_SEED:-0}
FAILURE_SOURCE_FILTER_KEY=${FAILURE_SOURCE_FILTER_KEY:-failure}
FAILURE_FILTER_SIZE=${FAILURE_FILTER_SIZE:-$DEFAULT_FAILURE_FILTER_SIZE}
FAILURE_SELECTION_SEED=${FAILURE_SELECTION_SEED:-0}
DEMO_FILTER_KEY=${DEMO_FILTER_KEY:-}

if [[ "$IMITATION_KIND" == "self" ]]; then
  if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
    echo "Self-imitation only uses human and success demos; conditioned mixed imitation requires IMITATION_KIND=mixed." >&2
    exit 2
  fi
  IMITATION_OUTPUT_DIR=${IMITATION_OUTPUT_DIR:-$SELF_IMITATION_OUTPUT_DIR}
  IMITATION_EPOCHS=${IMITATION_EPOCHS:-${SELF_IMITATION_EPOCHS:-${MIXED_IMITATION_EPOCHS:-50}}}
  IMITATION_SAVE_EVERY_EPOCHS=${IMITATION_SAVE_EVERY_EPOCHS:-${SELF_IMITATION_SAVE_EVERY_EPOCHS:-10}}
  IMITATION_SAVE_LATEST_EVERY_EPOCHS=${IMITATION_SAVE_LATEST_EVERY_EPOCHS:-${SELF_IMITATION_SAVE_LATEST_EVERY_EPOCHS:-1}}
  IMITATION_SEED=${IMITATION_SEED:-${SELF_IMITATION_SEED:-${MIXED_IMITATION_SEED:-}}}
  SUCCESS_FILTER_KEY=${SELF_IMITATION_SUCCESS_FILTER_KEY:-${SUCCESS_FILTER_KEY:-${SUCCESS_SOURCE_FILTER_KEY}_${SUCCESS_FILTER_SUFFIX}}}
  FAILURE_FILTER_KEY=${FAILURE_FILTER_KEY:-failure}
  IMITATION_MODE_NAME_VALUE="${SELF_IMITATION_MODE_NAME:-${IMITATION_MODE_NAME:-self_imitation_learning}}"
  IMITATION_EXPERIMENT_NAME_VALUE="${SELF_IMITATION_EXPERIMENT_NAME:-${IMITATION_EXPERIMENT_NAME:-${TASK_RGB_PREFIX}_self_imitation_200demo_${SUCCESS_FILTER_SIZE}success}}"
  IMITATION_DEMO_WEIGHT_VALUE="${SELF_IMITATION_DEMO_WEIGHT:-${IMITATION_DEMO_WEIGHT:-1.0}}"
  IMITATION_SUCCESS_WEIGHT_VALUE="${SELF_IMITATION_SUCCESS_WEIGHT:-${IMITATION_SUCCESS_WEIGHT:-1.0}}"
  IMITATION_FAILURE_WEIGHT_VALUE=0.0
else
  if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
    IMITATION_OUTPUT_DIR=${IMITATION_OUTPUT_DIR:-$CONDITIONED_IMITATION_OUTPUT_DIR}
  else
    IMITATION_OUTPUT_DIR=${IMITATION_OUTPUT_DIR:-$MIXED_IMITATION_OUTPUT_DIR}
  fi
  IMITATION_EPOCHS=${IMITATION_EPOCHS:-${MIXED_IMITATION_EPOCHS:-50}}
  IMITATION_SAVE_EVERY_EPOCHS=${IMITATION_SAVE_EVERY_EPOCHS:-${MIXED_IMITATION_SAVE_EVERY_EPOCHS:-10}}
  IMITATION_SAVE_LATEST_EVERY_EPOCHS=${IMITATION_SAVE_LATEST_EVERY_EPOCHS:-${MIXED_IMITATION_SAVE_LATEST_EVERY_EPOCHS:-1}}
  IMITATION_SEED=${IMITATION_SEED:-${MIXED_IMITATION_SEED:-}}
  SUCCESS_FILTER_KEY=${MIXED_IMITATION_SUCCESS_FILTER_KEY:-${SUCCESS_FILTER_KEY:-${SUCCESS_SOURCE_FILTER_KEY}_${SUCCESS_FILTER_SUFFIX}}}
  FAILURE_FILTER_KEY=${MIXED_IMITATION_FAILURE_FILTER_KEY:-${FAILURE_FILTER_KEY:-$DEFAULT_FAILURE_FILTER_KEY}}
  if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
    IMITATION_MODE_NAME_VALUE="${MIXED_IMITATION_MODE_NAME:-${IMITATION_MODE_NAME:-success_conditioned_mixed_quality_imitation_learning}}"
    IMITATION_EXPERIMENT_NAME_VALUE="${MIXED_IMITATION_EXPERIMENT_NAME:-${IMITATION_EXPERIMENT_NAME:-${TASK_RGB_PREFIX}_human_conditioned_mixed_imitation_200demo_${SUCCESS_FILTER_SIZE}success_${FAILURE_FILTER_SIZE}failure}}"
  else
    IMITATION_MODE_NAME_VALUE="${MIXED_IMITATION_MODE_NAME:-${IMITATION_MODE_NAME:-mixed_quality_imitation_learning}}"
    IMITATION_EXPERIMENT_NAME_VALUE="${MIXED_IMITATION_EXPERIMENT_NAME:-${IMITATION_EXPERIMENT_NAME:-${TASK_RGB_PREFIX}_mixed_imitation_200demo_${SUCCESS_FILTER_SIZE}success_${FAILURE_FILTER_SIZE}failure}}"
  fi
  # Mixed-imitation weights are inclusion flags only. With the default uniform
  # sample pool, every selected sequence is shuffled without source weighting.
  IMITATION_DEMO_WEIGHT_VALUE=1.0
  IMITATION_SUCCESS_WEIGHT_VALUE=1.0
  IMITATION_FAILURE_WEIGHT_VALUE=1.0
fi
SUCCESS_SELECTION_MANIFEST=${SUCCESS_SELECTION_MANIFEST:-${ROLLOUT_DATASET%.hdf5}_${SUCCESS_FILTER_KEY}_selection.json}
FAILURE_SELECTION_MANIFEST=${FAILURE_SELECTION_MANIFEST:-${FAILURE_DATASET%.hdf5}_${FAILURE_FILTER_KEY}_selection.json}

# evaluation
EVAL_DP_CHECKPOINT=${EVAL_DP_CHECKPOINT:-$IMITATION_OUTPUT_DIR/models/model_epoch_${IMITATION_EPOCHS}.pth}
EVAL_OUTPUT=${EVAL_OUTPUT:-$DEFAULT_EVAL_OUTPUT}
ACTOR_SOURCE=${ACTOR_SOURCE:-plain_dp}
EVAL_NUM_CANDIDATES_VALUE=${EVAL_NUM_CANDIDATES:-"1"}
read -r -a EVAL_NUM_CANDIDATE_ARGS <<< "$EVAL_NUM_CANDIDATES_VALUE"
EVAL_SEEDS_VALUE=${EVAL_SEEDS:-"0 1 2 3 4"}
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS_VALUE"

PIN_MEMORY_ARGS=(--pin-memory)
if [[ "${PIN_MEMORY:-1}" == "0" ]]; then
  PIN_MEMORY_ARGS=(--no-pin-memory)
fi
PERSISTENT_WORKERS_ARGS=(--persistent-workers)
if [[ "${PERSISTENT_WORKERS:-1}" == "0" ]]; then
  PERSISTENT_WORKERS_ARGS=(--no-persistent-workers)
fi
DIFFUSION_CLIP_SAMPLE_ARGS=(--diffusion-clip-sample)
if [[ "${DIFFUSION_CLIP_SAMPLE:-1}" == "0" ]]; then
  DIFFUSION_CLIP_SAMPLE_ARGS=(--no-diffusion-clip-sample)
fi
ACTOR_BATCH_SIZE_ARGS=()
if [[ -n "$ACTOR_BATCH_SIZE" ]]; then
  ACTOR_BATCH_SIZE_ARGS=(--actor-batch-size "$ACTOR_BATCH_SIZE")
fi
ACTOR_LR_ARGS=()
if [[ -n "$ACTOR_LR" ]]; then
  ACTOR_LR_ARGS=(--actor-lr "$ACTOR_LR")
fi
IMITATION_SEED_ARGS=()
if [[ -n "$IMITATION_SEED" ]]; then
  IMITATION_SEED_ARGS=(--seed "$IMITATION_SEED")
fi
IMITATION_STEPS_PER_EPOCH_ARGS=()
if [[ -n "$IMITATION_STEPS_PER_EPOCH" ]]; then
  IMITATION_STEPS_PER_EPOCH_ARGS=(--steps-per-epoch "$IMITATION_STEPS_PER_EPOCH")
fi
ACTOR_HDF5_CACHE_MODE_ARGS=()
if [[ -n "$ACTOR_HDF5_CACHE_MODE" ]]; then
  ACTOR_HDF5_CACHE_MODE_ARGS=(--actor-hdf5-cache-mode "$ACTOR_HDF5_CACHE_MODE")
fi
ACTOR_NORMALIZE_WEIGHTS_ARGS=()
case "$ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE" in
  auto) ;;
  1) ACTOR_NORMALIZE_WEIGHTS_ARGS=(--actor-normalize-weights-by-ds-size) ;;
  0) ACTOR_NORMALIZE_WEIGHTS_ARGS=(--no-actor-normalize-weights-by-ds-size) ;;
  *)
    echo "ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE must be auto, 0, or 1." >&2
    exit 2
    ;;
esac
ACTOR_SAMPLE_POOL_ARGS=(--no-actor-uniform-sample-pool)
if [[ "$ACTOR_UNIFORM_SAMPLE_POOL" == "1" ]]; then
  ACTOR_SAMPLE_POOL_ARGS=(--actor-uniform-sample-pool)
fi

ACTOR_LR_SCHEDULER_ARGS=(--no-actor-disable-lr-scheduler)
if [[ "${ACTOR_DISABLE_LR_SCHEDULER:-0}" == "1" ]]; then
  ACTOR_LR_SCHEDULER_ARGS=(--actor-disable-lr-scheduler)
fi
DEFAULT_FAILURE_ANTI_FAILURE_LABEL=0.0
if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
  DEFAULT_FAILURE_ANTI_FAILURE_LABEL=1.0
fi
FAILURE_CHUNK_ARGS=(
  --actor-failure-sample-start-offset "${MIXED_IMITATION_FAILURE_SAMPLE_START_OFFSET:-0}"
  --actor-failure-anti-failure-label "${MIXED_IMITATION_FAILURE_ANTI_FAILURE_LABEL:-$DEFAULT_FAILURE_ANTI_FAILURE_LABEL}"
)
if [[ "${MIXED_IMITATION_FAILURE_DEMO_START_ONLY:-0}" == "1" ]]; then
  FAILURE_CHUNK_ARGS=(--actor-failure-demo-start-only "${FAILURE_CHUNK_ARGS[@]}")
else
  FAILURE_CHUNK_ARGS=(--no-actor-failure-demo-start-only "${FAILURE_CHUNK_ARGS[@]}")
fi

if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
  CONDITION_ARGS=(
    --conditioned-mixed-imitation
    --condition-label-mode "$CONDITION_LABEL_MODE"
    --condition-dropout "${MIXED_IMITATION_CONDITION_DROPOUT:-${CONDITION_DROPOUT:-0.0}}"
    --condition-hidden-dim "${MIXED_IMITATION_CONDITION_HIDDEN_DIM:-${CONDITION_HIDDEN_DIM:-128}}"
  )
else
  CONDITION_ARGS=(
    --no-conditioned-mixed-imitation
  )
fi

if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
  EVAL_CONDITION_ARGS=(
    --require-success-condition-adapter
    --no-forbid-success-condition-adapter
    --inference-success-condition 1.0
    --inference-condition-mask 1.0
  )
  EVAL_CONDITION_DESCRIPTION="condition=1 condition_mask=1"
else
  EVAL_CONDITION_ARGS=(
    --no-require-success-condition-adapter
    --forbid-success-condition-adapter
  )
  EVAL_CONDITION_DESCRIPTION="condition_input=none"
fi

RESUME_POINT_VALUE="${RESUME_POINT:-${RESUME_CHECKPOINT:-}}"
RESTART_WITHOUT_LAST=${RESTART_WITHOUT_LAST:-1}

prepare_fixed_filter() {
  local label=$1
  local dataset=$2
  local source_key=$3
  local output_key=$4
  local filter_size=$5
  local selection_seed=$6
  local manifest=$7
  local force_filter=$8
  local -a force_args=()
  if [[ "$force_filter" == "1" ]]; then
    force_args=(--force)
  fi
  echo "[prepare_filters] $label: mask/$output_key from mask/$source_key, size=$filter_size seed=$selection_seed" >&2
  "$PYTHON" -B scripts/prepare_rgb_dp_rollout_filter.py \
    --dataset "$dataset" \
    --source-filter-key "$source_key" \
    --output-filter-key "$output_key" \
    --num-demos "$filter_size" \
    --seed "$selection_seed" \
    --manifest "$manifest" \
    "${force_args[@]}"
}

prepare_success_filter() {
  prepare_fixed_filter \
    success \
    "$SUCCESS_DATASET" \
    "$SUCCESS_SOURCE_FILTER_KEY" \
    "$SUCCESS_FILTER_KEY" \
    "$SUCCESS_FILTER_SIZE" \
    "$SUCCESS_SELECTION_SEED" \
    "$SUCCESS_SELECTION_MANIFEST" \
    "${FORCE_SUCCESS_FILTER:-0}"
}

prepare_failure_filter() {
  prepare_fixed_filter \
    failure \
    "$FAILURE_DATASET" \
    "$FAILURE_SOURCE_FILTER_KEY" \
    "$FAILURE_FILTER_KEY" \
    "$FAILURE_FILTER_SIZE" \
    "$FAILURE_SELECTION_SEED" \
    "$FAILURE_SELECTION_MANIFEST" \
    "${FORCE_FAILURE_FILTER:-0}"
}

check_datasets() {
  "$PYTHON" -B - \
    "$DP_CHECKPOINT" \
    "$DEMO_DATASET" \
    "$ROLLOUT_DATASET" \
    "$FAILURE_DATASET" \
    "$DEMO_FILTER_KEY" \
    "$SUCCESS_FILTER_KEY" \
    "$FAILURE_FILTER_KEY" \
    "$IMITATION_DEMO_WEIGHT_VALUE" \
    "$IMITATION_SUCCESS_WEIGHT_VALUE" \
    "$IMITATION_FAILURE_WEIGHT_VALUE" \
    "$ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE" \
    "$SUCCESS_SELECTION_MANIFEST" \
    "$FAILURE_SELECTION_MANIFEST" \
    "$ACTOR_BATCH_SIZE" \
    "$ACTOR_LR" \
    "${ACTOR_DISABLE_LR_SCHEDULER:-0}" \
    "$IMITATION_SEED" \
    "$ACTOR_HDF5_CACHE_MODE" \
    "${IMITATION_STEPS_PER_EPOCH:-auto}" \
    "$IMITATION_EPOCHS" \
    "$ACTOR_UNIFORM_SAMPLE_POOL" \
    "$CONDITION_LABEL_MODE" \
    "${CONDITIONED_MIXED_IMITATION:-0}" <<'PYCHECK'
import json
import sys
from pathlib import Path

import h5py
import torch

(
    checkpoint,
    demo_dataset,
    rollout_dataset,
    failure_dataset,
    demo_filter,
    success_filter,
    failure_filter,
    demo_weight,
    success_weight,
    failure_weight,
    normalize_by_ds_size,
    success_selection_manifest,
    failure_selection_manifest,
    actor_batch_size,
    actor_lr,
    disable_lr_scheduler,
    imitation_seed,
    actor_hdf5_cache_mode,
    steps_per_epoch,
    epochs,
    actor_uniform_sample_pool,
    condition_label_mode,
    conditioned_mixed_imitation,
) = sys.argv[1:24]

checkpoint_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
checkpoint_config = json.loads(checkpoint_dict["config"])
checkpoint_train = checkpoint_config["train"]
checkpoint_policy_optim = checkpoint_config["algo"]["optim_params"]["policy"]

def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value

source_weights = {
    "human_demo": float(demo_weight),
    "success_rollout": float(success_weight),
    "failure_rollout": float(failure_weight),
}
uniform_sample_pool = actor_uniform_sample_pool == "1"
conditioned = conditioned_mixed_imitation == "1"
checkpoint_has_condition_adapter = any(
    key.startswith("policy.condition_adapter.")
    for key in checkpoint_dict["model"]["nets"]
)
if not conditioned and checkpoint_has_condition_adapter:
    raise ValueError(
        "standard mixed imitation requires an unconditioned checkpoint, but "
        f"{checkpoint} contains a condition adapter"
    )
active_source_count = sum(weight > 0.0 for weight in source_weights.values())
effective_hdf5_cache_mode = (
    actor_hdf5_cache_mode
    or (
        checkpoint_train["hdf5_cache_mode"]
        if active_source_count == 1
        else "low_dim"
    )
)
if uniform_sample_pool:
    effective_normalize_by_ds_size = False
elif normalize_by_ds_size == "auto":
    effective_normalize_by_ds_size = (
        bool(checkpoint_train["normalize_weights_by_ds_size"])
        if active_source_count == 1
        else True
    )
else:
    effective_normalize_by_ds_size = normalize_by_ds_size != "0"

source_conditions = None
source_condition_masks = None
if conditioned:
    source_conditions = {
        "human_demo": 1.0,
        "success_rollout": 0.0 if condition_label_mode == "human_only" else 1.0,
        "failure_rollout": 0.0,
    }
    source_condition_masks = {key: 1.0 for key in source_conditions}

def decode(items):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in items]

def count_data(path):
    with h5py.File(path, "r") as f:
        return len(f["data"])

def count_mask(path, key):
    if not key:
        return None
    with h5py.File(path, "r") as f:
        mask_path = f"mask/{key}"
        if mask_path not in f:
            return None
        return len(f[mask_path])

def masks(path):
    with h5py.File(path, "r") as f:
        if "mask" not in f:
            return {}
        return {k: len(f[f"mask/{k}"]) for k in sorted(f["mask"].keys())}

def selected_num_samples(path, key):
    with h5py.File(path, "r") as f:
        if key:
            demos = decode(f[f"mask/{key}"][:])
        else:
            demos = list(f["data"].keys())
        return sum(int(f[f"data/{demo}"].attrs["num_samples"]) for demo in demos)

source_num_sequences = {
    "human_demo": selected_num_samples(demo_dataset, demo_filter),
    "success_rollout": selected_num_samples(rollout_dataset, success_filter),
    "failure_rollout": selected_num_samples(failure_dataset, failure_filter),
}
pooled_num_sequences = sum(
    source_num_sequences[source]
    for source, weight in source_weights.items()
    if weight > 0.0
)
effective_batch_size = int(actor_batch_size or checkpoint_train["batch_size"])
drop_last = pooled_num_sequences >= effective_batch_size
auto_steps_per_epoch = (
    pooled_num_sequences // effective_batch_size
    if drop_last
    else int(pooled_num_sequences > 0)
)
effective_steps_per_epoch = (
    auto_steps_per_epoch if steps_per_epoch == "auto" else int(steps_per_epoch)
)

report = {
    "checkpoint": {"path": checkpoint, "exists": Path(checkpoint).exists()},
    "effective_dp_training_parameters": {
        "optimizer_type": checkpoint_policy_optim["optimizer_type"],
        "initial_learning_rate": float(
            actor_lr or checkpoint_policy_optim["learning_rate"]["initial"]
        ),
        "learning_rate_source": "environment_override" if actor_lr else "pretrained_dp_checkpoint",
        "scheduler_type": checkpoint_policy_optim["learning_rate"]["scheduler_type"],
        "scheduler_enabled": disable_lr_scheduler != "1",
        "scheduler_step_every_batch": bool(
            checkpoint_policy_optim["learning_rate"]["step_every_batch"]
        ),
        "scheduler_warmup_steps": int(
            checkpoint_policy_optim["learning_rate"]["warmup_steps"]
        ),
        "scheduler_num_cycles": float(
            checkpoint_policy_optim["learning_rate"]["num_cycles"]
        ),
        "weight_decay": float(checkpoint_policy_optim["regularization"]["L2"]),
        "batch_size": effective_batch_size,
        "batch_size_source": "environment_override" if actor_batch_size else "pretrained_dp_checkpoint",
        "seed": int(imitation_seed or checkpoint_train["seed"]),
        "seed_source": "environment_override" if imitation_seed else "pretrained_dp_checkpoint",
        "hdf5_cache_mode": effective_hdf5_cache_mode,
        "hdf5_cache_mode_source": (
            "environment_override"
            if actor_hdf5_cache_mode
            else ("pretrained_dp_checkpoint" if active_source_count == 1 else "multi_source_requirement")
        ),
        "pooled_num_training_sequences": pooled_num_sequences,
        "source_num_training_sequences": source_num_sequences,
        "drop_last": drop_last,
        "dropped_sequences_per_epoch": (
            pooled_num_sequences % effective_batch_size if drop_last else 0
        ),
        "steps_per_epoch": effective_steps_per_epoch,
        "steps_per_epoch_source": "environment_override" if steps_per_epoch != "auto" else "dataloader_length",
        "epochs": int(epochs),
        "total_training_steps": effective_steps_per_epoch * int(epochs),
    },
    "effective_dp_normalization": {
        "hdf5_normalize_obs": bool(checkpoint_train["hdf5_normalize_obs"]),
        "action_config": checkpoint_train["action_config"],
        "action_stats_source": "pretrained_dp_checkpoint",
        "action_normalization_stats": jsonable(
            checkpoint_dict.get("action_normalization_stats")
        ),
    },
    "observation_modalities": checkpoint_config["observation"]["modalities"]["obs"],
    "sampler": {
        "mode": (
            "uniform_sample_pool_without_replacement"
            if uniform_sample_pool
            else "weighted_random_sampler_with_replacement"
        ),
        "source_weighting_enabled": not uniform_sample_pool,
        "source_inclusion_values": source_weights,
        "source_weights": None if uniform_sample_pool else source_weights,
        "source_distribution": {},
        "normalize_weights_by_dataset_size": effective_normalize_by_ds_size,
    },
    "success_conditioning": {
        "enabled": conditioned,
        "condition_input_used": conditioned,
        "condition_adapter_in_pretrained_checkpoint": checkpoint_has_condition_adapter,
        "label_mode": condition_label_mode if conditioned else None,
        "source_conditions": source_conditions,
        "source_condition_masks": source_condition_masks,
        "inference_condition": 1.0 if conditioned else None,
        "inference_condition_mask": 1.0 if conditioned else None,
    },
    "success_selection_manifest": {
        "path": success_selection_manifest,
        "exists": Path(success_selection_manifest).exists(),
    },
    "failure_selection_manifest": {
        "path": failure_selection_manifest,
        "exists": Path(failure_selection_manifest).exists(),
    },
    "human_demos": {
        "path": demo_dataset,
        "total_demos": count_data(demo_dataset),
        "selected_filter_key": demo_filter,
        "selected_count": count_mask(demo_dataset, demo_filter) if demo_filter else count_data(demo_dataset),
        "masks": masks(demo_dataset),
    },
    "rollouts": {
        "path": rollout_dataset,
        "total_demos": count_data(rollout_dataset),
        "success_filter_key": success_filter,
        "success_count": count_mask(rollout_dataset, success_filter),
        "masks": masks(rollout_dataset),
    },
    "failure_rollouts": {
        "path": failure_dataset,
        "total_demos": count_data(failure_dataset),
        "failure_filter_key": failure_filter,
        "failure_count": count_mask(failure_dataset, failure_filter),
        "masks": masks(failure_dataset),
    },
}
if not uniform_sample_pool:
    weight_total = sum(source_weights.values())
    if weight_total > 0.0:
        report["sampler"]["source_distribution"] = {
            key: value / weight_total for key, value in source_weights.items()
        }
print(json.dumps(report, indent=2, sort_keys=True))
PYCHECK
}

maybe_prepare_filters() {
  if [[ "${AUTO_PREPARE_FILTERS:-1}" == "1" ]]; then
    prepare_success_filter
    if [[ "$IMITATION_KIND" == "mixed" ]]; then
      prepare_failure_filter
    fi
  fi
}

run_train() {
  local -a resume_args=("$@")
  "$PYTHON" -B scripts/train_rgb_dp_mixed_imitation.py \
    --checkpoint "$DP_CHECKPOINT" \
    --demo-dataset "$DEMO_DATASET" \
    --success-dataset "$SUCCESS_DATASET" \
    --failure-dataset "$FAILURE_DATASET" \
    --output-dir "$IMITATION_OUTPUT_DIR" \
    "${resume_args[@]}" \
    --device cuda \
    "${IMITATION_SEED_ARGS[@]}" \
    --epochs "$IMITATION_EPOCHS" \
    "${IMITATION_STEPS_PER_EPOCH_ARGS[@]}" \
    "${ACTOR_BATCH_SIZE_ARGS[@]}" \
    --actor-num-workers "$ACTOR_NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    "${PIN_MEMORY_ARGS[@]}" \
    "${PERSISTENT_WORKERS_ARGS[@]}" \
    "${ACTOR_HDF5_CACHE_MODE_ARGS[@]}" \
    --demo-filter-key "$DEMO_FILTER_KEY" \
    --success-filter-key "$SUCCESS_FILTER_KEY" \
    "${ACTOR_SAMPLE_POOL_ARGS[@]}" \
    --failure-filter-key "$FAILURE_FILTER_KEY" \
    --actor-demo-weight "$IMITATION_DEMO_WEIGHT_VALUE" \
    --actor-success-weight "$IMITATION_SUCCESS_WEIGHT_VALUE" \
    --actor-failure-weight "$IMITATION_FAILURE_WEIGHT_VALUE" \
    "${FAILURE_CHUNK_ARGS[@]}" \
    "${CONDITION_ARGS[@]}" \
    --mode-name "$IMITATION_MODE_NAME_VALUE" \
    --experiment-name "$IMITATION_EXPERIMENT_NAME_VALUE" \
    "${ACTOR_NORMALIZE_WEIGHTS_ARGS[@]}" \
    "${ACTOR_LR_ARGS[@]}" \
    "${ACTOR_LR_SCHEDULER_ARGS[@]}" \
    --save-every-epochs "$IMITATION_SAVE_EVERY_EPOCHS" \
    --save-latest-every-epochs "$IMITATION_SAVE_LATEST_EVERY_EPOCHS" \
    --log-every "$LOG_EVERY"
}

run_resilient_train() {
  local max_restarts="${MAX_RESTARTS:-20}"
  local retry_sleep="${RETRY_SLEEP:-5}"
  local attempt=1
  local resume_for_attempt="$RESUME_POINT_VALUE"
  local save_latest_every=5
  if [[ -z "$resume_for_attempt" && -f "$IMITATION_OUTPUT_DIR/last.pth" ]]; then
    resume_for_attempt="$IMITATION_OUTPUT_DIR/last.pth"
  fi
  while (( attempt <= max_restarts )); do
    local -a attempt_resume_args=()
    if [[ -n "$resume_for_attempt" ]]; then
      attempt_resume_args=(--resume-checkpoint "$resume_for_attempt")
    fi
    echo "[train_resilient task=$TASK kind=$IMITATION_KIND attempt=$attempt/$max_restarts] resume=${resume_for_attempt:-none} save_latest_every=${save_latest_every}" >&2
    set +e
    IMITATION_SAVE_LATEST_EVERY_EPOCHS="$save_latest_every" run_train "${attempt_resume_args[@]}"
    local status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      exit 0
    fi
    echo "[train_resilient attempt=$attempt] training exited with status $status" >&2
    if [[ -f "$IMITATION_OUTPUT_DIR/models/model_epoch_${IMITATION_EPOCHS}.pth" ]]; then
      echo "[train_resilient] final checkpoint exists; treating training as complete" >&2
      exit 0
    fi
    if [[ ! -f "$IMITATION_OUTPUT_DIR/last.pth" ]]; then
      if [[ "$RESTART_WITHOUT_LAST" == "1" ]]; then
        echo "[train_resilient] no last.pth available; retrying from scratch" >&2
        resume_for_attempt=""
        attempt=$((attempt + 1))
        sleep "$retry_sleep"
        continue
      fi
      echo "[train_resilient] no last.pth available to resume from" >&2
      exit "$status"
    fi
    resume_for_attempt="$IMITATION_OUTPUT_DIR/last.pth"
    attempt=$((attempt + 1))
    sleep "$retry_sleep"
  done
  echo "[train_resilient] exhausted $max_restarts attempts" >&2
  exit 1
}

run_eval_grid_resilient() {
  if [[ ! -f "$EVAL_DP_CHECKPOINT" ]]; then
    echo "[eval_grid_resilient] EVAL_DP_CHECKPOINT does not exist: $EVAL_DP_CHECKPOINT" >&2
    exit 2
  fi
  if [[ "$ACTOR_SOURCE" == "plain_dp" ]]; then
    for n in "${EVAL_NUM_CANDIDATE_ARGS[@]}"; do
      if [[ "$n" != "1" ]]; then
        echo "[eval_grid_resilient] ACTOR_SOURCE=plain_dp only supports EVAL_NUM_CANDIDATES=1" >&2
        exit 2
      fi
    done
    ckpt_name=$(basename "$EVAL_DP_CHECKPOINT" .pth)
  else
    ckpt_name=$(basename "${IDQL_CHECKPOINT:?IDQL_CHECKPOINT is required when ACTOR_SOURCE is not plain_dp}" .pt)
  fi
  grid_dir="${EVAL_OUTPUT}_${ckpt_name}"
  echo "[eval_grid_resilient] checkpoint=$EVAL_DP_CHECKPOINT output=$grid_dir seeds=${EVAL_SEED_ARGS[*]} rollouts_per_seed=${N_ROLLOUTS:-50} $EVAL_CONDITION_DESCRIPTION" >&2
  "$PYTHON" -B scripts/run_square_rgb_dp_one_step_idql_eval_grid.py \
    --idql-checkpoint "${IDQL_CHECKPOINT:-$EVAL_DP_CHECKPOINT}" \
    --dp-checkpoint "$EVAL_DP_CHECKPOINT" \
    --output-dir "$grid_dir" \
    --device cuda \
    --actor-source "$ACTOR_SOURCE" \
    --n-rollouts "${N_ROLLOUTS:-50}" \
    --horizon "$HORIZON" \
    --num-candidates "${EVAL_NUM_CANDIDATE_ARGS[@]}" \
    --seeds "${EVAL_SEED_ARGS[@]}" \
    --rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-10}" \
    --max-retries "${EVAL_MAX_RETRIES:-3}" \
    --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS:-100}" \
    --execution-horizon "${EXECUTION_HORIZON:-8}" \
    --selection "${SELECTION:-argmax}" \
    "${EVAL_CONDITION_ARGS[@]}" \
    "${DIFFUSION_CLIP_SAMPLE_ARGS[@]}"
}

case "$STAGE" in
  check)
    check_datasets
    ;;
  prepare_filters)
    prepare_success_filter
    if [[ "$IMITATION_KIND" == "mixed" ]]; then
      prepare_failure_filter
    fi
    check_datasets
    ;;
  train)
    maybe_prepare_filters
    resume_args=()
    if [[ -n "$RESUME_POINT_VALUE" ]]; then
      resume_args=(--resume-checkpoint "$RESUME_POINT_VALUE")
    fi
    run_train "${resume_args[@]}"
    ;;
  train_resilient)
    maybe_prepare_filters
    run_resilient_train
    ;;
  all)
    prepare_success_filter
    if [[ "$IMITATION_KIND" == "mixed" ]]; then
      prepare_failure_filter
    fi
    check_datasets
    run_resilient_train
    ;;
  eval_grid_resilient)
    run_eval_grid_resilient
    ;;
  *)
    echo "Usage: $0 [can|square|transport] {check|check_self|check_mixed|prepare_filters|prepare_self_filters|prepare_mixed_filters|train_self_imitation|train_self_imitation_resilient|train_mixed_imitation|train_mixed_imitation_resilient|train_conditioned|train_conditioned_resilient|train_conditioned_mixed_imitation|train_conditioned_imitation_resilient|eval_self_grid_resilient|eval_mixed_grid_resilient|eval_conditioned_grid_resilient|eval_grid_resilient|all|all_self|all_mixed}" >&2
    exit 2
    ;;
esac
