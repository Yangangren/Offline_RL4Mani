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
    TASK_EXPERT_DATASET=datasets/square/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/square/idql/square_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/square_rgb_dp/idql/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/square_rgb_dp/idql/200demo_100success_50failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=1
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  can)
    TASK_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    TASK_EXPERT_DATASET=datasets/can/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/can_rgb_dp/epoch50_collection/can_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_100success_33failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/can_rgb_dp/idql/200demo_100success_33failure
    TASK_EVAL_OUTPUT=rollouts/can_rgb_dp/idql/200demo_100success_33failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  transport)
    TASK_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/transport/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/transport_rgb_dp/epoch200_collection/transport_rgb_dp_rollouts_rgb4.hdf5
    TASK_IDQL_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/transport_rgb_dp/idql/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/transport_rgb_dp/idql/200demo_100success_50failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos,robot1_gripper_qpos
    ;;
  tool_hang)
    TASK_DP_CHECKPOINT=trained_models/tool_hang_rgb_dp/tool_hang_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/tool_hang/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/tool_hang_rgb_dp/epoch200_collection/tool_hang_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/idql/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/idql/200demo_100success_50failure
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=-1
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  *)
    echo "Unsupported TASK=$TASK. Use square, can, transport, or tool_hang." >&2
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
IDQL_REWARD_MODE=${IDQL_REWARD_MODE:-task}
case "$IDQL_REWARD_MODE" in
  task)
    DEFAULT_IDQL_DATASET=${TASK_IDQL_DATASET%.hdf5}_task_reward.hdf5
    DEFAULT_IDQL_OUTPUT_DIR=${TASK_IDQL_OUTPUT_DIR}_task_reward
    DEFAULT_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_task_reward
    ;;
  rise)
    DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
    DEFAULT_IDQL_OUTPUT_DIR=$TASK_IDQL_OUTPUT_DIR
    DEFAULT_EVAL_OUTPUT=$TASK_EVAL_OUTPUT
    ;;
  *)
    echo "Unsupported IDQL_REWARD_MODE=$IDQL_REWARD_MODE. Use task or rise." >&2
    exit 2
    ;;
esac
IDQL_DATASET=${IDQL_DATASET:-$DEFAULT_IDQL_DATASET}
IDQL_OUTPUT_DIR=${IDQL_OUTPUT_DIR:-$DEFAULT_IDQL_OUTPUT_DIR}
IDQL_CHECKPOINT=${IDQL_CHECKPOINT:-$IDQL_OUTPUT_DIR/last.pt}
EVAL_OUTPUT=${EVAL_OUTPUT:-$DEFAULT_EVAL_OUTPUT}
EVAL_HORIZON=${HORIZON:-$TASK_EVAL_HORIZON}
CRITIC_LATE_FUSION_KEY=${CRITIC_LATE_FUSION_KEY:-$TASK_CRITIC_LATE_FUSION_KEY}
EXPERT_MASK=${EXPERT_MASK:-$TASK_EXPERT_MASK}
EXPERT_COUNT=${EXPERT_COUNT:-$TASK_EXPERT_COUNT}
SUCCESS_MASK=${SUCCESS_MASK:-$TASK_SUCCESS_MASK}
SUCCESS_COUNT=${SUCCESS_COUNT:-$TASK_SUCCESS_COUNT}
FAILURE_MASK=${FAILURE_MASK:-$TASK_FAILURE_MASK}
FAILURE_COUNT=${FAILURE_COUNT:-$TASK_FAILURE_COUNT}

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

build_dataset() {
  local overwrite_args=()
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
  local train_launcher=("$PYTHON" -B)
  if [[ -n "$resume_path" ]]; then
    resume_args=(--resume-checkpoint "$resume_path")
  fi
  if [[ -n "${STEPS_PER_EPOCH:-}" ]]; then
    steps_per_epoch_args=(--steps-per-epoch "$STEPS_PER_EPOCH")
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
    "${resume_args[@]}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-0}" \
    --epochs "${EPOCHS:-50}" \
    "${steps_per_epoch_args[@]}" \
    --schedule-reference-batch-size "${IDQL_SCHEDULE_REFERENCE_BATCH_SIZE:-64}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --num-workers "${NUM_WORKERS:-4}" \
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
    --critic-lr "${CRITIC_LR:-1e-4}" \
    --vf-lr "${VF_LR:-1e-4}" \
    --lr-scheduler "${LR_SCHEDULER:-cosine}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS:-500}" \
    --lr-num-cycles "${LR_NUM_CYCLES:-0.5}" \
    --critic-hidden-dims ${CRITIC_HIDDEN_DIMS:-300 400 300} \
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

  *)
    echo "Usage: $0 [square|can|transport|tool_hang] {build_dataset|train|train_resilient|eval|eval_grid_resilient}" >&2
    exit 2
    ;;
esac
