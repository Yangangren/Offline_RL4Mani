#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "can" || "${1:-}" == "Can" || "${1:-}" == "square" || "${1:-}" == "Square" ]]; then
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
    DEFAULT_SELF_IMITATION_OUTPUT_DIR=trained_models/can_rgb_dp/self_imitation/200demo_100success
    DEFAULT_MIXED_IMITATION_OUTPUT_DIR=trained_models/can_rgb_dp/mixed_imitation/200demo_100success_33failure
    DEFAULT_EVAL_OUTPUT=rollouts/can_rgb_dp/imitation_eval
    ;;
  square)
    TASK_RGB_PREFIX=square_rgb_dp
    DEFAULT_DP_CHECKPOINT=trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth
    DEFAULT_DEMO_DATASET=datasets/square/ph/image_v15.hdf5
    DEFAULT_ROLLOUT_DATASET=rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5
    DEFAULT_SELF_IMITATION_OUTPUT_DIR=trained_models/square_rgb_dp_self_imitation/200demo_100success
    DEFAULT_MIXED_IMITATION_OUTPUT_DIR=trained_models/square_rgb_dp_mixed_imitation/200demo_100success_94failure
    DEFAULT_EVAL_OUTPUT=rollouts/square_rgb_dp/imitation_eval
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use TASK=can or TASK=square, or pass can|square as the first argument." >&2
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
  train_conditioned_resilient|train_conditioned_mixed_imitation_resilient)
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

ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE:-100}
ACTOR_LR=${ACTOR_LR:-1e-4}
ACTOR_HDF5_CACHE_MODE=${ACTOR_HDF5_CACHE_MODE:-low_dim}
ACTOR_NUM_WORKERS=${ACTOR_NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
LOG_EVERY=${LOG_EVERY:-100}

SELF_IMITATION_OUTPUT_DIR=${SELF_IMITATION_OUTPUT_DIR:-$DEFAULT_SELF_IMITATION_OUTPUT_DIR}
MIXED_IMITATION_OUTPUT_DIR=${MIXED_IMITATION_OUTPUT_DIR:-$DEFAULT_MIXED_IMITATION_OUTPUT_DIR}

SUCCESS_SOURCE_FILTER_KEY=${SUCCESS_SOURCE_FILTER_KEY:-success}
SUCCESS_FILTER_SIZE=${SUCCESS_FILTER_SIZE:-100}
SUCCESS_FILTER_SUFFIX=${SUCCESS_FILTER_SUFFIX:-100}
DEMO_FILTER_KEY=${DEMO_FILTER_KEY:-}

if [[ "$IMITATION_KIND" == "self" ]]; then
  if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
    echo "Self-imitation only uses human and success demos; conditioned mixed imitation requires IMITATION_KIND=mixed." >&2
    exit 2
  fi
  IMITATION_OUTPUT_DIR=${IMITATION_OUTPUT_DIR:-$SELF_IMITATION_OUTPUT_DIR}
  IMITATION_EPOCHS=${IMITATION_EPOCHS:-${SELF_IMITATION_EPOCHS:-${MIXED_IMITATION_EPOCHS:-50}}}
  IMITATION_STEPS_PER_EPOCH=${IMITATION_STEPS_PER_EPOCH:-${SELF_IMITATION_STEPS_PER_EPOCH:-${MIXED_IMITATION_STEPS_PER_EPOCH:-100}}}
  IMITATION_SAVE_EVERY_EPOCHS=${IMITATION_SAVE_EVERY_EPOCHS:-${SELF_IMITATION_SAVE_EVERY_EPOCHS:-10}}
  IMITATION_SAVE_LATEST_EVERY_EPOCHS=${IMITATION_SAVE_LATEST_EVERY_EPOCHS:-${SELF_IMITATION_SAVE_LATEST_EVERY_EPOCHS:-1}}
  IMITATION_SEED=${IMITATION_SEED:-${SELF_IMITATION_SEED:-${MIXED_IMITATION_SEED:-20260710}}}
  SUCCESS_FILTER_KEY=${SELF_IMITATION_SUCCESS_FILTER_KEY:-${SUCCESS_FILTER_KEY:-${SUCCESS_SOURCE_FILTER_KEY}_${SUCCESS_FILTER_SUFFIX}}}
  FAILURE_FILTER_KEY=${FAILURE_FILTER_KEY:-failure}
  IMITATION_MODE_NAME_VALUE="${SELF_IMITATION_MODE_NAME:-${IMITATION_MODE_NAME:-self_imitation_learning}}"
  IMITATION_EXPERIMENT_NAME_VALUE="${SELF_IMITATION_EXPERIMENT_NAME:-${IMITATION_EXPERIMENT_NAME:-${TASK_RGB_PREFIX}_self_imitation_200demo_${SUCCESS_FILTER_SIZE}success}}"
  IMITATION_DEMO_WEIGHT_VALUE="${SELF_IMITATION_DEMO_WEIGHT:-${IMITATION_DEMO_WEIGHT:-1.0}}"
  IMITATION_SUCCESS_WEIGHT_VALUE="${SELF_IMITATION_SUCCESS_WEIGHT:-${IMITATION_SUCCESS_WEIGHT:-1.0}}"
  IMITATION_FAILURE_WEIGHT_VALUE=0.0
else
  IMITATION_OUTPUT_DIR=${IMITATION_OUTPUT_DIR:-$MIXED_IMITATION_OUTPUT_DIR}
  IMITATION_EPOCHS=${IMITATION_EPOCHS:-${MIXED_IMITATION_EPOCHS:-50}}
  IMITATION_STEPS_PER_EPOCH=${IMITATION_STEPS_PER_EPOCH:-${MIXED_IMITATION_STEPS_PER_EPOCH:-100}}
  IMITATION_SAVE_EVERY_EPOCHS=${IMITATION_SAVE_EVERY_EPOCHS:-${MIXED_IMITATION_SAVE_EVERY_EPOCHS:-10}}
  IMITATION_SAVE_LATEST_EVERY_EPOCHS=${IMITATION_SAVE_LATEST_EVERY_EPOCHS:-${MIXED_IMITATION_SAVE_LATEST_EVERY_EPOCHS:-1}}
  IMITATION_SEED=${IMITATION_SEED:-${MIXED_IMITATION_SEED:-20260710}}
  SUCCESS_FILTER_KEY=${MIXED_IMITATION_SUCCESS_FILTER_KEY:-${SUCCESS_FILTER_KEY:-${SUCCESS_SOURCE_FILTER_KEY}_${SUCCESS_FILTER_SUFFIX}}}
  FAILURE_FILTER_KEY=${MIXED_IMITATION_FAILURE_FILTER_KEY:-${FAILURE_FILTER_KEY:-failure}}
  if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
    IMITATION_MODE_NAME_VALUE="${MIXED_IMITATION_MODE_NAME:-${IMITATION_MODE_NAME:-success_conditioned_mixed_quality_imitation_learning}}"
    IMITATION_EXPERIMENT_NAME_VALUE="${MIXED_IMITATION_EXPERIMENT_NAME:-${IMITATION_EXPERIMENT_NAME:-${TASK_RGB_PREFIX}_success_conditioned_mixed_quality_imitation}}"
  else
    IMITATION_MODE_NAME_VALUE="${MIXED_IMITATION_MODE_NAME:-${IMITATION_MODE_NAME:-mixed_quality_imitation_learning}}"
    IMITATION_EXPERIMENT_NAME_VALUE="${MIXED_IMITATION_EXPERIMENT_NAME:-${IMITATION_EXPERIMENT_NAME:-${TASK_RGB_PREFIX}_mixed_imitation_200demo_${SUCCESS_FILTER_SIZE}success}}"
  fi
  IMITATION_DEMO_WEIGHT_VALUE="${MIXED_IMITATION_DEMO_WEIGHT:-${IMITATION_DEMO_WEIGHT:-0.4}}"
  IMITATION_SUCCESS_WEIGHT_VALUE="${MIXED_IMITATION_SUCCESS_WEIGHT:-${IMITATION_SUCCESS_WEIGHT:-0.4}}"
  IMITATION_FAILURE_WEIGHT_VALUE="${MIXED_IMITATION_FAILURE_WEIGHT:-${IMITATION_FAILURE_WEIGHT:-0.2}}"
fi
SUCCESS_SELECTION_MANIFEST=${SUCCESS_SELECTION_MANIFEST:-${ROLLOUT_DATASET%.hdf5}_${SUCCESS_FILTER_KEY}_selection.json}

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
ACTOR_NORMALIZE_WEIGHTS_ARGS=(--actor-normalize-weights-by-ds-size)
if [[ "${ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE:-1}" == "0" ]]; then
  ACTOR_NORMALIZE_WEIGHTS_ARGS=(--no-actor-normalize-weights-by-ds-size)
fi
ACTOR_LR_SCHEDULER_ARGS=(--actor-disable-lr-scheduler)
if [[ "${ACTOR_DISABLE_LR_SCHEDULER:-1}" == "0" ]]; then
  ACTOR_LR_SCHEDULER_ARGS=(--no-actor-disable-lr-scheduler)
fi
FAILURE_CHUNK_ARGS=(
  --actor-failure-sample-start-offset "${MIXED_IMITATION_FAILURE_SAMPLE_START_OFFSET:-0}"
  --actor-failure-anti-failure-label "${MIXED_IMITATION_FAILURE_ANTI_FAILURE_LABEL:-1.0}"
)
if [[ "${MIXED_IMITATION_FAILURE_DEMO_START_ONLY:-0}" == "1" ]]; then
  FAILURE_CHUNK_ARGS=(--actor-failure-demo-start-only "${FAILURE_CHUNK_ARGS[@]}")
else
  FAILURE_CHUNK_ARGS=(--no-actor-failure-demo-start-only "${FAILURE_CHUNK_ARGS[@]}")
fi

if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
  CONDITION_ARGS=(
    --conditioned-mixed-imitation
    --condition-dropout "${MIXED_IMITATION_CONDITION_DROPOUT:-${CONDITION_DROPOUT:-0.0}}"
    --condition-hidden-dim "${MIXED_IMITATION_CONDITION_HIDDEN_DIM:-${CONDITION_HIDDEN_DIM:-128}}"
  )
else
  CONDITION_ARGS=(
    --no-conditioned-mixed-imitation
    --condition-dropout "${MIXED_IMITATION_CONDITION_DROPOUT:-${CONDITION_DROPOUT:-0.0}}"
    --condition-hidden-dim "${MIXED_IMITATION_CONDITION_HIDDEN_DIM:-${CONDITION_HIDDEN_DIM:-128}}"
  )
fi

EVAL_CONDITION_ARGS=(
  --inference-success-condition 1.0
  --inference-condition-mask 1.0
)
if [[ "${CONDITIONED_MIXED_IMITATION:-0}" == "1" ]]; then
  EVAL_CONDITION_ARGS=(--require-success-condition-adapter "${EVAL_CONDITION_ARGS[@]}")
else
  EVAL_CONDITION_ARGS=(--no-require-success-condition-adapter "${EVAL_CONDITION_ARGS[@]}")
fi

RESUME_POINT_VALUE="${RESUME_POINT:-${RESUME_CHECKPOINT:-}}"
RESTART_WITHOUT_LAST=${RESTART_WITHOUT_LAST:-1}

filter_exists() {
  local dataset=$1
  local filter_key=$2
  "$PYTHON" -B - "$dataset" "$filter_key" <<'PYCHECK'
import sys
import h5py

dataset, filter_key = sys.argv[1], sys.argv[2]
with h5py.File(dataset, "r") as f:
    raise SystemExit(0 if f"mask/{filter_key}" in f else 1)
PYCHECK
}

write_success_selection_manifest() {
  "$PYTHON" -B - \
    "$ROLLOUT_DATASET" \
    "$SUCCESS_SOURCE_FILTER_KEY" \
    "$SUCCESS_FILTER_KEY" \
    "$SUCCESS_FILTER_SIZE" \
    "$SUCCESS_SELECTION_MANIFEST" <<'PYMANIFEST'
import json
import sys
from pathlib import Path

import h5py

dataset, source_key, filter_key, requested_size, manifest_path = sys.argv[1:6]

def decode(items):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in items]

with h5py.File(dataset, "r") as f:
    source_path = f"mask/{source_key}"
    filter_path = f"mask/{filter_key}"
    if source_path not in f:
        raise KeyError(f"{source_path} not found in {dataset}")
    if filter_path not in f:
        raise KeyError(f"{filter_path} not found in {dataset}")
    source_demos = decode(f[source_path][:])
    selected_demos = decode(f[filter_path][:])
    selected_lengths = {
        key: int(f["data"][key].attrs["num_samples"])
        for key in selected_demos
    }

manifest = {
    "dataset": dataset,
    "source_filter_key": source_key,
    "fixed_filter_key": filter_key,
    "requested_num_demos": int(requested_size),
    "source_num_demos": len(source_demos),
    "selected_num_demos": len(selected_demos),
    "selected_num_samples": int(sum(selected_lengths.values())),
    "selection_seed": 0,
    "selection_method": "robomimic/scripts/filter_dataset_size.py with np.random.seed(0)",
    "selection_storage": f"mask/{filter_key}",
    "physically_copied": False,
    "selected_demos": selected_demos,
    "selected_lengths": selected_lengths,
}

path = Path(manifest_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"[prepare_filters] wrote fixed success selection manifest to {path}", file=sys.stderr)
PYMANIFEST
}

prepare_success_filter() {
  if [[ "$SUCCESS_DATASET" != "$ROLLOUT_DATASET" ]]; then
    echo "[prepare_filters] SUCCESS_DATASET differs from ROLLOUT_DATASET; skipping automatic filter creation" >&2
    return
  fi
  if filter_exists "$ROLLOUT_DATASET" "$SUCCESS_FILTER_KEY" && [[ "${FORCE_SUCCESS_FILTER:-0}" != "1" ]]; then
    echo "[prepare_filters] mask/$SUCCESS_FILTER_KEY already exists in $ROLLOUT_DATASET" >&2
    write_success_selection_manifest
    return
  fi
  local expected_prefix="${SUCCESS_SOURCE_FILTER_KEY}_"
  if [[ "$SUCCESS_FILTER_KEY" != "$expected_prefix"* ]]; then
    echo "[prepare_filters] cannot infer output key suffix for SUCCESS_FILTER_KEY=$SUCCESS_FILTER_KEY" >&2
    echo "[prepare_filters] expected a key like ${SUCCESS_SOURCE_FILTER_KEY}_${SUCCESS_FILTER_SUFFIX}" >&2
    exit 2
  fi
  local output_suffix="${SUCCESS_FILTER_KEY#${expected_prefix}}"
  echo "[prepare_filters] creating mask/$SUCCESS_FILTER_KEY from mask/$SUCCESS_SOURCE_FILTER_KEY with $SUCCESS_FILTER_SIZE demos" >&2
  "$PYTHON" -B robomimic/scripts/filter_dataset_size.py \
    --dataset "$ROLLOUT_DATASET" \
    --input_filter_key "$SUCCESS_SOURCE_FILTER_KEY" \
    --num_demos "$SUCCESS_FILTER_SIZE" \
    --output_filter_key "$output_suffix"
  write_success_selection_manifest
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
    "${ACTOR_NORMALIZE_WEIGHTS_BY_DS_SIZE:-1}" \
    "$SUCCESS_SELECTION_MANIFEST" <<'PYCHECK'
import json
import sys
from pathlib import Path

import h5py

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
) = sys.argv[1:13]

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

report = {
    "checkpoint": {"path": checkpoint, "exists": Path(checkpoint).exists()},
    "sampler": {
        "source_weights": {
            "human_demo": float(demo_weight),
            "success_rollout": float(success_weight),
            "failure_rollout": float(failure_weight),
        },
        "source_distribution": {},
        "normalize_weights_by_dataset_size": normalize_by_ds_size != "0",
    },
    "success_selection_manifest": {
        "path": success_selection_manifest,
        "exists": Path(success_selection_manifest).exists(),
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
weight_total = sum(report["sampler"]["source_weights"].values())
if weight_total > 0.0:
    report["sampler"]["source_distribution"] = {
        key: value / weight_total for key, value in report["sampler"]["source_weights"].items()
    }
print(json.dumps(report, indent=2, sort_keys=True))
PYCHECK
}

maybe_prepare_filters() {
  if [[ "${AUTO_PREPARE_FILTERS:-1}" == "1" ]]; then
    prepare_success_filter
  fi
}

run_train() {
  local -a resume_args=("$@")
  "$PYTHON" -B scripts/train_rgb_dp_self_imitation.py \
    --checkpoint "$DP_CHECKPOINT" \
    --demo-dataset "$DEMO_DATASET" \
    --success-dataset "$SUCCESS_DATASET" \
    --failure-dataset "$FAILURE_DATASET" \
    --output-dir "$IMITATION_OUTPUT_DIR" \
    "${resume_args[@]}" \
    --device cuda \
    --seed "$IMITATION_SEED" \
    --epochs "$IMITATION_EPOCHS" \
    --steps-per-epoch "$IMITATION_STEPS_PER_EPOCH" \
    --actor-batch-size "$ACTOR_BATCH_SIZE" \
    --actor-num-workers "$ACTOR_NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    "${PIN_MEMORY_ARGS[@]}" \
    "${PERSISTENT_WORKERS_ARGS[@]}" \
    --actor-hdf5-cache-mode "$ACTOR_HDF5_CACHE_MODE" \
    --demo-filter-key "$DEMO_FILTER_KEY" \
    --success-filter-key "$SUCCESS_FILTER_KEY" \
    --failure-filter-key "$FAILURE_FILTER_KEY" \
    --actor-demo-weight "$IMITATION_DEMO_WEIGHT_VALUE" \
    --actor-success-weight "$IMITATION_SUCCESS_WEIGHT_VALUE" \
    --actor-failure-weight "$IMITATION_FAILURE_WEIGHT_VALUE" \
    "${FAILURE_CHUNK_ARGS[@]}" \
    "${CONDITION_ARGS[@]}" \
    --mode-name "$IMITATION_MODE_NAME_VALUE" \
    --experiment-name "$IMITATION_EXPERIMENT_NAME_VALUE" \
    "${ACTOR_NORMALIZE_WEIGHTS_ARGS[@]}" \
    --actor-lr "$ACTOR_LR" \
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
  local save_latest_every=1
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
  echo "[eval_grid_resilient] checkpoint=$EVAL_DP_CHECKPOINT output=$grid_dir seeds=${EVAL_SEED_ARGS[*]} rollouts_per_seed=${N_ROLLOUTS:-50} success_condition=1 condition_mask=1" >&2
  "$PYTHON" -B scripts/run_square_rgb_dp_one_step_idql_eval_grid.py \
    --idql-checkpoint "${IDQL_CHECKPOINT:-$EVAL_DP_CHECKPOINT}" \
    --dp-checkpoint "$EVAL_DP_CHECKPOINT" \
    --output-dir "$grid_dir" \
    --device cuda \
    --actor-source "$ACTOR_SOURCE" \
    --n-rollouts "${N_ROLLOUTS:-50}" \
    --horizon "${HORIZON:-400}" \
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
    check_datasets
    run_resilient_train
    ;;
  eval_grid_resilient)
    run_eval_grid_resilient
    ;;
  *)
    echo "Usage: $0 [can|square] {check|check_self|check_mixed|prepare_filters|prepare_self_filters|prepare_mixed_filters|train_self_imitation|train_self_imitation_resilient|train_mixed_imitation|train_mixed_imitation_resilient|train_conditioned|train_conditioned_resilient|train_conditioned_mixed_imitation|train_conditioned_mixed_imitation_resilient|eval_self_grid_resilient|eval_mixed_grid_resilient|eval_conditioned_grid_resilient|eval_grid_resilient|all|all_self|all_mixed}" >&2
    exit 2
    ;;
esac
