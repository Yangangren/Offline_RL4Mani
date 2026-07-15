#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_successor_critic_pycache_${USER}_$$"
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=${PYTHON:-/home/ryan/miniconda3/envs/robomimic_stable/bin/python}
ROLLOUTS=${ROLLOUTS:-rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5}
DP_CHECKPOINT=${DP_CHECKPOINT:-trained_models/square_rgb_dp_idql_visual/default_reward_dp_chunk_actor_iql/best_success_auc.pt}
MODEL_ARCH=${MODEL_ARCH:-v4}
OUTPUT_DIR=${OUTPUT_DIR:-trained_models/square_rgb_dp_successor_critic/iql_actor_q8_v3_frozen_v_gamma0p99}
ACTION_HORIZON=${ACTION_HORIZON:-8}
TOTAL_STEPS=${TOTAL_STEPS:-5000}
WARMUP_STEPS=${WARMUP_STEPS:-750}
Q_RAMP_STEPS=${Q_RAMP_STEPS:-500}
ENCODER_UNFREEZE_STEP=${ENCODER_UNFREEZE_STEP:-4000}
ENCODER_LR_SCALE=${ENCODER_LR_SCALE:-0.05}
ENCODER_REFERENCE_WEIGHT=${ENCODER_REFERENCE_WEIGHT:-0.1}
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-2}
EVAL_EPISODE_BATCH_SIZE=${EVAL_EPISODE_BATCH_SIZE:-2}
NORMALIZER_EPISODE_BATCH_SIZE=${NORMALIZER_EPISODE_BATCH_SIZE:-4}
EVAL_EVERY=${EVAL_EVERY:-250}
# GAMMA is the discount over one complete executed 8-step chunk. The Python
# trainer derives step_gamma=GAMMA^(1/ACTION_HORIZON), so a full chunk uses
# exactly this discount while a short terminal chunk is handled correctly.
GAMMA=${GAMMA:-0.99}
TARGET_LABEL_SMOOTHING=${TARGET_LABEL_SMOOTHING:-0.01}
STATE_WEIGHT=${STATE_WEIGHT:-1.0}
DYNAMICS_WEIGHT=${DYNAMICS_WEIGHT:-1.0}
DYNAMICS_COSINE_WEIGHT=${DYNAMICS_COSINE_WEIGHT:-0.1}
Q_WEIGHT=${Q_WEIGHT:-1.0}
VALUE_CONSISTENCY_WEIGHT=${VALUE_CONSISTENCY_WEIGHT:-0.0}
CONTRAST_WEIGHT=${CONTRAST_WEIGHT:-0.0}
SEED=${SEED:-20260714}
STAGE=${1:-train}

train_args=(
  --rollouts "$ROLLOUTS"
  --output-dir "$OUTPUT_DIR"
  --encoder-init-checkpoint "$DP_CHECKPOINT"
  --model-arch "$MODEL_ARCH"
  --device cuda
  --total-steps "$TOTAL_STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --q-ramp-steps "$Q_RAMP_STEPS"
  --encoder-unfreeze-step "$ENCODER_UNFREEZE_STEP"
  --encoder-lr-scale "$ENCODER_LR_SCALE"
  --encoder-reference-weight "$ENCODER_REFERENCE_WEIGHT"
  --episode-batch-size "$EPISODE_BATCH_SIZE"
  --eval-episode-batch-size "$EVAL_EPISODE_BATCH_SIZE"
  --normalizer-episode-batch-size "$NORMALIZER_EPISODE_BATCH_SIZE"
  --action-horizon "$ACTION_HORIZON"
  --eval-every "$EVAL_EVERY"
  --gamma "$GAMMA"
  --target-label-smoothing "$TARGET_LABEL_SMOOTHING"
  --state-weight "$STATE_WEIGHT"
  --dynamics-weight "$DYNAMICS_WEIGHT"
  --dynamics-cosine-weight "$DYNAMICS_COSINE_WEIGHT"
  --q-weight "$Q_WEIGHT"
  --value-consistency-weight "$VALUE_CONSISTENCY_WEIGHT"
  --contrast-weight "$CONTRAST_WEIGHT"
  --seed "$SEED"
)

case "$STAGE" in
  train|all)
    "$PYTHON" -B scripts/train_rgb_dp_successor_critic.py "${train_args[@]}"
    ;;
  *)
    echo "Usage: $0 {train|all}" >&2
    exit 2
    ;;
esac
