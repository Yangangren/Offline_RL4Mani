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
    TASK_CHUNK_IDQL_OUTPUT_DIR=trained_models/square_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=-1
    TASK_CHUNK_INITIALIZATION=pretrained_dp_joint
    TASK_CHUNK_ACTOR_DEMO_ONLY=0
    TASK_CRITIC_GROUP_NORM=0
    TASK_VF_ENCODER_FREEZE_STEPS=1000
    TASK_ENCODER_FREEZE_STEPS=0
    TASK_CHUNK_EVAL_OUTPUT=rollouts/square_rgb_dp_visual/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition
    TASK_COMPOSED_DP_CHECKPOINT=$TASK_DP_CHECKPOINT
    TASK_COMPOSED_CHUNK_EVAL_OUTPUT=rollouts/square_rgb_dp_visual/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition_epoch200_actor
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  can)
    TASK_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_50.pth
    TASK_EXPERT_DATASET=datasets/can/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/can_rgb_dp/epoch50_collection/can_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/can/idql/can_rgb_dp_idql_200demo_100success_33failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/can_rgb_dp/idql/200demo_100success_33failure
    TASK_CHUNK_IDQL_OUTPUT_DIR=trained_models/can_rgb_dp/chunk_idql/200demo_100success_33failure_h8_dynamics_human_condition
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure
    TASK_FAILURE_COUNT=-1
    TASK_CHUNK_INITIALIZATION=pretrained_dp_joint
    TASK_CHUNK_ACTOR_DEMO_ONLY=0
    TASK_CRITIC_GROUP_NORM=0
    TASK_VF_ENCODER_FREEZE_STEPS=1000
    TASK_ENCODER_FREEZE_STEPS=1000
    TASK_CHUNK_EVAL_OUTPUT=rollouts/can_rgb_dp/chunk_idql/200demo_100success_33failure_h8_dynamics_human_condition
    TASK_COMPOSED_DP_CHECKPOINT=trained_models/can_rgb_dp/can_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_COMPOSED_CHUNK_EVAL_OUTPUT=rollouts/can_rgb_dp/chunk_idql/200demo_100success_33failure_h8_dynamics_human_condition_epoch200_actor
    TASK_EVAL_HORIZON=400
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos
    ;;
  transport)
    TASK_DP_CHECKPOINT=trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/transport/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/transport_rgb_dp/epoch200_collection/transport_rgb_dp_rollouts_rgb4.hdf5
    TASK_IDQL_DATASET=datasets/transport/idql/transport_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/transport_rgb_dp/idql/200demo_100success_50failure
    TASK_CHUNK_IDQL_OUTPUT_DIR=trained_models/transport_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=-1
    TASK_CHUNK_INITIALIZATION=pretrained_dp_joint
    TASK_CHUNK_ACTOR_DEMO_ONLY=0
    TASK_CRITIC_GROUP_NORM=0
    TASK_VF_ENCODER_FREEZE_STEPS=1000
    TASK_ENCODER_FREEZE_STEPS=1000
    TASK_CHUNK_EVAL_OUTPUT=rollouts/transport_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition
    TASK_COMPOSED_DP_CHECKPOINT=$TASK_DP_CHECKPOINT
    TASK_COMPOSED_CHUNK_EVAL_OUTPUT=rollouts/transport_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition_epoch200_actor
    TASK_EVAL_HORIZON=700
    TASK_CRITIC_LATE_FUSION_KEY=robot0_gripper_qpos,robot1_gripper_qpos
    ;;
  tool_hang)
    TASK_DP_CHECKPOINT=trained_models/tool_hang_rgb_dp/tool_hang_ph_rgb_dp_official_s1/models/model_epoch_200.pth
    TASK_EXPERT_DATASET=datasets/tool_hang/ph/image_v15.hdf5
    TASK_ROLLOUT_DATASET=rollouts/tool_hang_rgb_dp/epoch200_collection/tool_hang_rgb_dp_rollouts_rgb2.hdf5
    TASK_IDQL_DATASET=datasets/tool_hang/idql/tool_hang_rgb_dp_idql_200demo_100success_50failure.hdf5
    TASK_IDQL_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/idql/200demo_100success_50failure
    TASK_CHUNK_IDQL_OUTPUT_DIR=trained_models/tool_hang_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition
    TASK_EXPERT_MASK=
    TASK_EXPERT_COUNT=200
    TASK_SUCCESS_MASK=success_100
    TASK_SUCCESS_COUNT=-1
    TASK_FAILURE_MASK=failure_50
    TASK_FAILURE_COUNT=-1
    TASK_CHUNK_INITIALIZATION=pretrained_dp_joint
    TASK_CHUNK_ACTOR_DEMO_ONLY=0
    TASK_CRITIC_GROUP_NORM=0
    TASK_VF_ENCODER_FREEZE_STEPS=1000
    TASK_ENCODER_FREEZE_STEPS=1000
    TASK_CHUNK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition
    TASK_COMPOSED_DP_CHECKPOINT=$TASK_DP_CHECKPOINT
    TASK_COMPOSED_CHUNK_EVAL_OUTPUT=rollouts/tool_hang_rgb_dp/chunk_idql/200demo_100success_50failure_h8_dynamics_human_condition_epoch200_actor
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
export PYTHONPYCACHEPREFIX="/tmp/robomimic_${TASK}_rgb_dp_chunk_idql_pycache_${USER}_$$"
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
EXPERT_DATASET=${EXPERT_DATASET:-$TASK_EXPERT_DATASET}
ROLLOUT_DATASET=${ROLLOUT_DATASET:-$TASK_ROLLOUT_DATASET}
IDQL_REWARD_MODE=${IDQL_REWARD_MODE:-task}
case "$IDQL_REWARD_MODE" in
  task)
    DEFAULT_IDQL_DATASET=${TASK_IDQL_DATASET%.hdf5}_task_reward.hdf5
    DEFAULT_IDQL_OUTPUT_DIR=${TASK_IDQL_OUTPUT_DIR}_task_reward
    DEFAULT_CHUNK_IDQL_OUTPUT_DIR=${TASK_CHUNK_IDQL_OUTPUT_DIR}_task_reward
    DEFAULT_CHUNK_EVAL_OUTPUT=${TASK_CHUNK_EVAL_OUTPUT}_task_reward
    DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT=${TASK_COMPOSED_CHUNK_EVAL_OUTPUT}_task_reward
    ;;
  rise)
    DEFAULT_IDQL_DATASET=$TASK_IDQL_DATASET
    DEFAULT_IDQL_OUTPUT_DIR=$TASK_IDQL_OUTPUT_DIR
    DEFAULT_CHUNK_IDQL_OUTPUT_DIR=${TASK_CHUNK_IDQL_OUTPUT_DIR}
    DEFAULT_CHUNK_EVAL_OUTPUT=${TASK_CHUNK_EVAL_OUTPUT}
    DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT=${TASK_COMPOSED_CHUNK_EVAL_OUTPUT}
    ;;
  *)
    echo "Unsupported IDQL_REWARD_MODE=$IDQL_REWARD_MODE. Use task or rise." >&2
    exit 2
    ;;
esac
IDQL_DATASET=${IDQL_DATASET:-$DEFAULT_IDQL_DATASET}
IDQL_OUTPUT_DIR=${IDQL_OUTPUT_DIR:-$DEFAULT_IDQL_OUTPUT_DIR}
IDQL_CHECKPOINT=${IDQL_CHECKPOINT:-$IDQL_OUTPUT_DIR/last.pt}
SOURCE_IDQL_CHECKPOINT=${SOURCE_IDQL_CHECKPOINT:-$IDQL_CHECKPOINT}
CHUNK_IDQL_OUTPUT_DIR=${CHUNK_IDQL_OUTPUT_DIR:-$DEFAULT_CHUNK_IDQL_OUTPUT_DIR}
CHUNK_INITIALIZATION=${CHUNK_INITIALIZATION:-$TASK_CHUNK_INITIALIZATION}
CHUNK_ACTOR_DEMO_ONLY=${CHUNK_ACTOR_DEMO_ONLY:-${ACTOR_DEMO_ONLY:-$TASK_CHUNK_ACTOR_DEMO_ONLY}}
CHUNK_CONDITIONED_ACTOR=${CHUNK_CONDITIONED_ACTOR:-1}
CHUNK_IDQL_CHECKPOINT=${CHUNK_IDQL_CHECKPOINT:-$CHUNK_IDQL_OUTPUT_DIR/last.pt}
CHUNK_EVAL_OUTPUT=${CHUNK_EVAL_OUTPUT:-$DEFAULT_CHUNK_EVAL_OUTPUT}
COMPOSED_DP_CHECKPOINT=${COMPOSED_DP_CHECKPOINT:-$TASK_COMPOSED_DP_CHECKPOINT}
COMPOSED_CHUNK_EVAL_OUTPUT=${COMPOSED_CHUNK_EVAL_OUTPUT:-$DEFAULT_COMPOSED_CHUNK_EVAL_OUTPUT}
EVAL_HORIZON=${HORIZON:-$TASK_EVAL_HORIZON}
CRITIC_LATE_FUSION_KEY=${CRITIC_LATE_FUSION_KEY:-$TASK_CRITIC_LATE_FUSION_KEY}
VF_ENCODER_FREEZE_STEPS=${VF_ENCODER_FREEZE_STEPS:-$TASK_VF_ENCODER_FREEZE_STEPS}
ENCODER_FREEZE_STEPS=${ENCODER_FREEZE_STEPS:-$TASK_ENCODER_FREEZE_STEPS}
EXPERT_MASK=${EXPERT_MASK:-$TASK_EXPERT_MASK}
EXPERT_COUNT=${EXPERT_COUNT:-$TASK_EXPERT_COUNT}
SUCCESS_MASK=${SUCCESS_MASK:-$TASK_SUCCESS_MASK}
SUCCESS_COUNT=${SUCCESS_COUNT:-$TASK_SUCCESS_COUNT}
FAILURE_MASK=${FAILURE_MASK:-$TASK_FAILURE_MASK}
FAILURE_COUNT=${FAILURE_COUNT:-$TASK_FAILURE_COUNT}

CHUNK_STEPS_PER_EPOCH=${CHUNK_STEPS_PER_EPOCH:-}

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
CHUNK_ACTOR_DEMO_ONLY_ARG=--no-actor-demo-only
if [[ "$CHUNK_ACTOR_DEMO_ONLY" == "1" ]]; then
  CHUNK_ACTOR_DEMO_ONLY_ARG=--actor-demo-only
fi
if [[ "$CHUNK_INITIALIZATION" != "pretrained_dp_joint" ]]; then
  CHUNK_ACTOR_DEMO_ONLY_ARG=--no-actor-demo-only
fi
USE_HUBER_ARG=--use-huber
if [[ "${USE_HUBER:-1}" == "0" ]]; then
  USE_HUBER_ARG=--no-use-huber
fi
MAX_GRADIENT_NORM_ARGS=(--max-gradient-norm "${MAX_GRADIENT_NORM:-10.0}")
CHUNK_CONDITION_ARGS=(--conditioned-actor)
if [[ "$CHUNK_CONDITIONED_ACTOR" == "0" ]]; then
  CHUNK_CONDITION_ARGS=(--no-conditioned-actor)
fi
CHUNK_EVAL_CONDITION_ARGS=(
  --require-success-condition-adapter
  --no-forbid-success-condition-adapter
  --inference-success-condition 1.0
  --inference-condition-mask 1.0
)
if [[ "$CHUNK_CONDITIONED_ACTOR" == "0" ]]; then
  CHUNK_EVAL_CONDITION_ARGS=(
    --no-require-success-condition-adapter
    --forbid-success-condition-adapter
  )
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

build_dataset() {
  local overwrite_args=()
  if [[ ! -f "$EXPERT_DATASET" ]]; then
    echo "[rgb_dp_chunk_idql task=$TASK] expert dataset does not exist: $EXPERT_DATASET" >&2
    exit 1
  fi
  if [[ ! -f "$ROLLOUT_DATASET" ]]; then
    echo "[rgb_dp_chunk_idql task=$TASK] rollout dataset does not exist: $ROLLOUT_DATASET" >&2
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
    echo "[rgb_dp_chunk_idql] building missing mixed dataset: $IDQL_DATASET" >&2
    build_dataset
  fi
}

run_chunk_train() {
  local resume_path="${1:-}"
  local resume_args=()
  local initialization_args=()
  if [[ -n "$resume_path" ]]; then
    resume_args=(--resume-checkpoint "$resume_path")
  fi
  case "$CHUNK_INITIALIZATION" in
    pretrained_dp_joint)
      initialization_args=(
        --initialization pretrained_dp_joint
        --checkpoint "$DP_CHECKPOINT"
      )
      ;;
    pretrained_dp_frozen)
      initialization_args=(
        --initialization pretrained_dp_frozen
        --checkpoint "$DP_CHECKPOINT"
      )
      ;;
    source_idql_frozen)
      initialization_args=(
        --initialization source_idql_frozen
        --source-idql-checkpoint "$SOURCE_IDQL_CHECKPOINT"
      )
      ;;
    *)
      echo "Unsupported CHUNK_INITIALIZATION=$CHUNK_INITIALIZATION" >&2
      return 2
      ;;
  esac
  "$PYTHON" -B scripts/train_rgb_dp_chunk_idql.py \
    --task "$TASK" \
    "${initialization_args[@]}" \
    --dataset "$IDQL_DATASET" \
    --output-dir "$CHUNK_IDQL_OUTPUT_DIR" \
    "${resume_args[@]}" \
    --device "${DEVICE:-cuda}" \
    --seed "${CHUNK_SEED:-${SEED:-0}}" \
    --epochs "${CHUNK_EPOCHS:-50}" \
    --steps-per-epoch "$CHUNK_STEPS_PER_EPOCH" \
    --batch-size "${CHUNK_BATCH_SIZE:-${BATCH_SIZE:-64}}" \
    --num-workers "${CHUNK_NUM_WORKERS:-${NUM_WORKERS:-4}}" \
    --prefetch-factor "${CHUNK_PREFETCH_FACTOR:-${PREFETCH_FACTOR:-2}}" \
    "$PIN_MEMORY_ARG" \
    "$PERSISTENT_WORKERS_ARG" \
    --hdf5-cache-mode "${HDF5_CACHE_MODE:-low_dim}" \
    --reward-mode "$IDQL_REWARD_MODE" \
    --chunk-horizon "${CHUNK_HORIZON:-8}" \
    --discount "${DISCOUNT:-0.99}" \
    --expectile "${EXPECTILE:-0.9}" \
    --target-tau "${TARGET_TAU:-0.01}" \
    --dynamics-target-sync-interval "${DYNAMICS_TARGET_SYNC_INTERVAL:-1000}" \
    --actor-lr "${CHUNK_ACTOR_LR:-${ACTOR_LR:-1e-4}}" \
    --actor-lr-scheduler "${CHUNK_ACTOR_LR_SCHEDULER:-cosine}" \
    --actor-lr-warmup-steps "${CHUNK_ACTOR_LR_WARMUP_STEPS:-1000}" \
    --actor-lr-num-cycles "${CHUNK_ACTOR_LR_NUM_CYCLES:-0.5}" \
    "$CHUNK_ACTOR_DEMO_ONLY_ARG" \
    "${CHUNK_CONDITION_ARGS[@]}" \
    --condition-dropout "${CHUNK_CONDITION_DROPOUT:-${CONDITION_DROPOUT:-0.0}}" \
    --condition-hidden-dim "${CHUNK_CONDITION_HIDDEN_DIM:-${CONDITION_HIDDEN_DIM:-128}}" \
    --critic-lr "${CHUNK_CRITIC_LR:-${CRITIC_LR:-1e-4}}" \
    --encoder-lr "${CHUNK_ENCODER_LR:-1e-5}" \
    --vf-lr "${CHUNK_VF_LR:-${VF_LR:-1e-4}}" \
    --critic-vf-lr-scheduler "${CHUNK_CRITIC_VF_LR_SCHEDULER:-cosine}" \
    --critic-vf-lr-warmup-steps "${CHUNK_CRITIC_VF_LR_WARMUP_STEPS:-1000}" \
    --critic-vf-lr-num-cycles "${CHUNK_CRITIC_VF_LR_NUM_CYCLES:-0.5}" \
    --critic-hidden-dims ${CHUNK_CRITIC_HIDDEN_DIMS:-${CRITIC_HIDDEN_DIMS:-300 400 300}} \
    --latent-dim "${CHUNK_LATENT_DIM:-300}" \
    --action-hidden-dim "${CHUNK_ACTION_HIDDEN_DIM:-128}" \
    --num-attention-heads "${CHUNK_NUM_ATTENTION_HEADS:-4}" \
    --num-action-conv-layers "${CHUNK_NUM_ACTION_CONV_LAYERS:-2}" \
    --dropout "${CHUNK_DROPOUT:-0.0}" \
    --num-critics "${NUM_CRITICS:-2}" \
    "$CRITIC_GROUP_NORM_ARG" \
    --critic-late-fusion-key "$CRITIC_LATE_FUSION_KEY" \
    --dynamics-weight "${DYNAMICS_WEIGHT:-0.5}" \
    --dynamics-cosine-weight "${DYNAMICS_COSINE_WEIGHT:-0.5}" \
    --dynamics-warmup-steps "${DYNAMICS_WARMUP_STEPS:-1000}" \
    --encoder-freeze-steps "$ENCODER_FREEZE_STEPS" \
    --vf-encoder-freeze-steps "$VF_ENCODER_FREEZE_STEPS" \
    "$USE_HUBER_ARG" \
    "${MAX_GRADIENT_NORM_ARGS[@]}" \
    --log-every "${LOG_EVERY:-200}" \
    --save-every-epochs "${CHUNK_SAVE_EVERY_EPOCHS:-1}" \
    --snapshot-every-epochs "${CHUNK_SNAPSHOT_EVERY_EPOCHS:-5}"
}

STAGE=${1:-train_chunk_idql_resilient}
case "$STAGE" in
  train_chunk_idql)
    ensure_dataset
    run_chunk_train "${CHUNK_RESUME_CHECKPOINT:-}"
    ;;

  train_chunk_idql_resilient)
    ensure_dataset
    max_restarts=${MAX_RESTARTS:-20}
    retry_sleep=${RETRY_SLEEP:-5}
    resume_path=${CHUNK_RESUME_CHECKPOINT:-}
    if [[ -z "$resume_path" && -f "$CHUNK_IDQL_OUTPUT_DIR/latest.pt" ]]; then
      resume_path="$CHUNK_IDQL_OUTPUT_DIR/latest.pt"
    fi
    attempt=1
    while (( attempt <= max_restarts )); do
      echo "[rgb_dp_chunk_idql attempt=$attempt/$max_restarts] resume=${resume_path:-none}" >&2
      set +e
      run_chunk_train "$resume_path"
      status=$?
      set -e
      if [[ "$status" -eq 0 ]]; then
        exit 0
      fi
      echo "[rgb_dp_chunk_idql attempt=$attempt] exited with status $status" >&2
      if [[ ! -f "$CHUNK_IDQL_OUTPUT_DIR/latest.pt" ]]; then
        echo "[rgb_dp_chunk_idql] no latest.pt is available for recovery" >&2
        exit "$status"
      fi
      resume_path="$CHUNK_IDQL_OUTPUT_DIR/latest.pt"
      attempt=$((attempt + 1))
      sleep "$retry_sleep"
    done
    echo "[rgb_dp_chunk_idql] exhausted $max_restarts attempts" >&2
    exit 1
    ;;

  eval_chunk_grid_resilient)
    read -r -a candidate_args <<< "${EVAL_NUM_CANDIDATES:-1 8 16}"
    read -r -a seed_args <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    "$PYTHON" -B scripts/run_rgb_dp_idql_eval_grid.py \
      --idql-checkpoint "$CHUNK_IDQL_CHECKPOINT" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$CHUNK_EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
      --actor-source hybrid_dp_chunk_actor \
      --critic-source "${CRITIC_SOURCE:-online}" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "$EVAL_HORIZON" \
      --num-candidates "${candidate_args[@]}" \
      --seeds "${seed_args[@]}" \
      --rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-25}" \
      --inter-chunk-sleep "${EVAL_INTER_CHUNK_SLEEP:-0}" \
      --max-retries "${EVAL_MAX_RETRIES:-5}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --execution-horizon "${EXECUTION_HORIZON:-8}" \
      --selection "${SELECTION:-argmax}" \
      "${CHUNK_EVAL_CONDITION_ARGS[@]}" \
      --clip-actions
    ;;

  eval_composed_chunk_grid_resilient)
    read -r -a candidate_args <<< "${EVAL_NUM_CANDIDATES:-1 8 16}"
    read -r -a seed_args <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    "$PYTHON" -B scripts/run_rgb_dp_idql_eval_grid.py \
      --idql-checkpoint "$CHUNK_IDQL_CHECKPOINT" \
      --dp-checkpoint "$COMPOSED_DP_CHECKPOINT" \
      --expected-task "$TASK" \
      --output-dir "$COMPOSED_CHUNK_EVAL_OUTPUT" \
      --device "${DEVICE:-cuda}" \
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
    echo "Usage: $0 [square|can|transport|tool_hang] {train_chunk_idql|train_chunk_idql_resilient|eval_chunk_grid_resilient|eval_composed_chunk_grid_resilient}" >&2
    exit 2
    ;;
esac
