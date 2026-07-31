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

# Keep these paths and selections identical to run_rgb_dp_idql.sh and
# run_rgb_dp_chunk_idql.sh. DQL consumes the same built mixed HDF5 directly.
case "$TASK" in
  square)
    TASK_DP_CHECKPOINT=trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/square/idql/square_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/square_rgb_dp/dql/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/square_rgb_dp/dql/200demo_100success_50failure
    TASK_CRITIC_GROUP_NORM=1
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  can)
    TASK_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    TASK_IDQL_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_100success_33failure.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/can_rgb_dp/dql/200demo_100success_33failure
    TASK_EVAL_OUTPUT=rollouts/can_rgb_dp/dql/200demo_100success_33failure
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  transport)
    TASK_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/transport_rgb_dp/dql/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/transport_rgb_dp/dql/200demo_100success_50failure
    TASK_CRITIC_GROUP_NORM=0
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos,robot1_gripper_qpos
    ;;
  tool_hang)
    TASK_DP_CHECKPOINT=trained_models/tool_hang_rgb_dp/tool_hang_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_IDQL_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_DQL_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/dql/200demo_100success_50failure
    TASK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/dql/200demo_100success_50failure
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

DP_CHECKPOINT=${DP_CHECKPOINT:-$TASK_DP_CHECKPOINT}
DQL_REWARD_MODE=${DQL_REWARD_MODE:-task}
case "$DQL_REWARD_MODE" in
  task)
    DEFAULT_IDQL_DATASET=${TASK_IDQL_DATASET%.hdf5}_task_reward.hdf5
    DEFAULT_DQL_OUTPUT_DIR=${TASK_DQL_OUTPUT_DIR}_task_reward
    DEFAULT_EVAL_OUTPUT=${TASK_EVAL_OUTPUT}_task_reward
    ;;
  rise)
    DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
    DEFAULT_DQL_OUTPUT_DIR=$TASK_DQL_OUTPUT_DIR
    DEFAULT_EVAL_OUTPUT=$TASK_EVAL_OUTPUT
    ;;
  *)
    echo "Unsupported DQL_REWARD_MODE=$DQL_REWARD_MODE. Use task or rise." >&2
    exit 2
    ;;
esac
IDQL_DATASET=${IDQL_DATASET:-$DEFAULT_IDQL_DATASET}
DQL_OUTPUT_DIR=${DQL_OUTPUT_DIR:-$DEFAULT_DQL_OUTPUT_DIR}
DQL_CHECKPOINT=${DQL_CHECKPOINT:-$DQL_OUTPUT_DIR/last.pt}
EVAL_OUTPUT=${EVAL_OUTPUT:-$DEFAULT_EVAL_OUTPUT}
EVAL_HORIZON=${HORIZON:-$TASK_EVAL_HORIZON}
CRITIC_LATE_FUSION_KEY=${CRITIC_LATE_FUSION_KEY:-$TASK_CRITIC_LATE_FUSION_KEY}

PIN_MEMORY_ARG=--pin-memory
if [[ "${PIN_MEMORY:-1}" == "0" ]]; then
  PIN_MEMORY_ARG=--no-pin-memory
fi
PERSISTENT_WORKERS_ARG=--persistent-workers
if [[ "${PERSISTENT_WORKERS:-1}" == "0" ]]; then
  PERSISTENT_WORKERS_ARG=--no-persistent-workers
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
}

run_train() {
  local resume_path=${1:-}
  local -a resume_args=()
  local -a steps_per_epoch_args=()
  if [[ -n "$resume_path" ]]; then
    resume_args=(--resume-checkpoint "$resume_path")
  fi
  if [[ -n "${STEPS_PER_EPOCH:-}" ]]; then
    steps_per_epoch_args=(--steps-per-epoch "$STEPS_PER_EPOCH")
  fi
  "$PYTHON" -B scripts/train_rgb_dp_dql.py \
    --task "$TASK" \
    --dataset "$IDQL_DATASET" \
    --checkpoint "$DP_CHECKPOINT" \
    --output-dir "$DQL_OUTPUT_DIR" \
    "${resume_args[@]}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-0}" \
    --epochs "${EPOCHS:-50}" \
    "${steps_per_epoch_args[@]}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --prefetch-factor "${PREFETCH_FACTOR:-2}" \
    "$PIN_MEMORY_ARG" \
    "$PERSISTENT_WORKERS_ARG" \
    --hdf5-cache-mode "${HDF5_CACHE_MODE:-low_dim}" \
    --reward-mode "$DQL_REWARD_MODE" \
    --discount "${DISCOUNT:-0.99}" \
    --target-tau "${TARGET_TAU:-0.005}" \
    --actor-lr "${ACTOR_LR:-1e-4}" \
    --critic-lr "${CRITIC_LR:-3e-4}" \
    --lr-scheduler "${LR_SCHEDULER:-cosine}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS:-500}" \
    --lr-num-cycles "${LR_NUM_CYCLES:-0.5}" \
    --critic-hidden-dims ${CRITIC_HIDDEN_DIMS:-300 400 300} \
    --num-critics "${NUM_CRITICS:-2}" \
    "$CRITIC_GROUP_NORM_ARG" \
    --critic-late-fusion-key "$CRITIC_LATE_FUSION_KEY" \
    "$USE_HUBER_ARG" \
    --max-gradient-norm "${MAX_GRADIENT_NORM:-10.0}" \
    --dql-eta "${DQL_ETA:-1.0}" \
    --dql-bc-weight "${DQL_BC_WEIGHT:-1.0}" \
    --dql-q-batch-size "${DQL_Q_BATCH_SIZE:-8}" \
    --dql-num-inference-steps "${DQL_NUM_INFERENCE_STEPS:-5}" \
    --dql-target-num-candidates "${DQL_TARGET_NUM_CANDIDATES:-1}" \
    --dql-q-head "${DQL_Q_HEAD:-random}" \
    --dql-q-denominator-floor "${DQL_Q_DENOMINATOR_FLOOR:-1e-6}" \
    "$DQL_CLIP_ACTIONS_ARG" \
    --log-every "${LOG_EVERY:-100}" \
    --save-every-epochs "${SAVE_EVERY_EPOCHS:-1}" \
    --snapshot-every-epochs "${SNAPSHOT_EVERY_EPOCHS:-10}"
}

STAGE=${1:-train_resilient}
case "$STAGE" in
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
    read -r -a candidate_args <<< "${EVAL_NUM_CANDIDATES:-1 4 8 16 32 50}"
    read -r -a seed_args <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    "$PYTHON" -B scripts/run_rgb_dp_idql_eval_grid.py \
      --idql-checkpoint "$DQL_CHECKPOINT" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
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
    echo "Usage: $0 [square|can|transport|tool_hang] {train|train_resilient|eval|eval_grid_resilient}" >&2
    exit 2
    ;;
esac
