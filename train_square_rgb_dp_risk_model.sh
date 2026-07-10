#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_square_risk_pycache_${USER}_$$"
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=/home/ryan/miniconda3/envs/robomimic_stable/bin/python
export ROBOMIMIC_PYTHON="$PYTHON"

ROLLOUTS=${ROLLOUTS:-rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5}
DP_CHECKPOINT=${DP_CHECKPOINT:-/home/ryan/Documents/robomimic/trained_models/square_rgb_dp_idql_visual/default_reward_dp_chunk_actor_iql/best_success_auc.pt}
FEATURES=${FEATURES:-rollouts/square_rgb_dp/epoch190_collection/risk_model/chunk_features.npz}
OUTPUT_DIR=${OUTPUT_DIR:-trained_models/square_rgb_dp_causal_prefix_risk/epoch190}
ANALYSIS_DIR=${ANALYSIS_DIR:-rollouts/square_rgb_dp/epoch190_collection/risk_analysis}

# Square does not yet have privileged critical/safe labels. Passing missing
# files makes the feature extractor create all-false diagnostic labels; they
# are not used for optimization anyway.
CRITICAL_SUMMARY=${CRITICAL_SUMMARY:-rollouts/square_rgb_dp/epoch190_collection/not_available_critical.json}
SAFE_SUMMARY=${SAFE_SUMMARY:-rollouts/square_rgb_dp/epoch190_collection/not_available_safe.json}

ACTION_HORIZON=${ACTION_HORIZON:-8}
PREDICTION_HORIZON=${PREDICTION_HORIZON:-16}
Q_ACTION_HORIZON=${Q_ACTION_HORIZON:-$ACTION_HORIZON}
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-128}
TOTAL_STEPS=${TOTAL_STEPS:-5000}
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-16}
EVAL_EVERY=${EVAL_EVERY:-250}
SEED=${SEED:-20260630}
MAX_EPISODES=${MAX_EPISODES:-}
FORCE_FEATURES=${FORCE_FEATURES:-0}
OBJECTIVE=${OBJECTIVE:-two_stage_temporal_safe_anchor}
TARGET_OUTCOME=${TARGET_OUTCOME:-success}
SUCCESS_LOSS_MODE=${SUCCESS_LOSS_MODE:-chunk_bce_failure_mil}
MODEL_ARCH=${MODEL_ARCH:-v2}
ACTION_NUM_HEADS=${ACTION_NUM_HEADS:-4}
ACTION_CONV_LAYERS=${ACTION_CONV_LAYERS:-2}
PREFIX_CONV_LAYERS=${PREFIX_CONV_LAYERS:-1}
STAGE1_STEPS=${STAGE1_STEPS:-500}
if [[ -z "${INITIAL_STATE_LOGIT_BIAS+x}" ]]; then
  if [[ "$TARGET_OUTCOME" == "success" ]]; then
    # Success noisy-AND multiplies per-chunk probabilities across the whole
    # episode. A strongly negative initial bias makes the episode success
    # probability essentially zero before learning.
    INITIAL_STATE_LOGIT_BIAS=4.0
  else
    # Failure noisy-OR models rare bad chunks, so low initial risk is sensible.
    INITIAL_STATE_LOGIT_BIAS=-5.0
  fi
fi
INITIAL_ACTION_DELTA_BIAS=${INITIAL_ACTION_DELTA_BIAS:-0.0}
RESIDUAL_L1_WEIGHT=${RESIDUAL_L1_WEIGHT:-0.001}
SHUFFLED_RESIDUAL_WEIGHT=${SHUFFLED_RESIDUAL_WEIGHT:-0.02}
SMOOTHNESS_WEIGHT=${SMOOTHNESS_WEIGHT:-0.005}
SUCCESS_RESIDUAL_WEIGHT=${SUCCESS_RESIDUAL_WEIGHT:-0.0}
DECORRELATION_WEIGHT=${DECORRELATION_WEIGHT:-0.1}
PAIRWISE_RANK_WEIGHT=${PAIRWISE_RANK_WEIGHT:-0.0}
PAIRWISE_RANK_MARGIN=${PAIRWISE_RANK_MARGIN:-0.05}
PAIRWISE_RANK_NEGATIVES=${PAIRWISE_RANK_NEGATIVES:-4}
PROGRESS_CONSISTENCY_WEIGHT=${PROGRESS_CONSISTENCY_WEIGHT:-0.0}
PROGRESS_CONSISTENCY_SPACE=${PROGRESS_CONSISTENCY_SPACE:-logit}
DELTA_PROGRESS_WEIGHT=${DELTA_PROGRESS_WEIGHT:-0.2}
DELTA_PROGRESS_STRIDE=${DELTA_PROGRESS_STRIDE:-$Q_ACTION_HORIZON}
DELTA_PROGRESS_SPACE=${DELTA_PROGRESS_SPACE:-logit}
DELTA_PROGRESS_CLIP=${DELTA_PROGRESS_CLIP:-0.5}
DELTA_PROGRESS_HUBER_BETA=${DELTA_PROGRESS_HUBER_BETA:-0.05}
DELTA_PROGRESS_ABS_WEIGHT=${DELTA_PROGRESS_ABS_WEIGHT:-0.0}
CHECKPOINT_METRIC=${CHECKPOINT_METRIC:-mixed_action}
SAFE_ANCHOR_WEIGHT=${SAFE_ANCHOR_WEIGHT:-0.02}
SAFE_ANCHOR_EPSILON=${SAFE_ANCHOR_EPSILON:-0.0}
TEMPORAL_RISK_WEIGHT=${TEMPORAL_RISK_WEIGHT:-0.0}
TEMPORAL_RISK_MARGIN=${TEMPORAL_RISK_MARGIN:-0.05}
TEMPORAL_STRIDE=${TEMPORAL_STRIDE:-$Q_ACTION_HORIZON}
TEMPORAL_MIN_INCREASE=${TEMPORAL_MIN_INCREASE:-0.0}
TEMPORAL_NORMALIZE_WEIGHTS=${TEMPORAL_NORMALIZE_WEIGHTS:-0}

STAGE=${1:-all}

feature_args=(
  --rollouts "$ROLLOUTS"
  --dp-checkpoint "$DP_CHECKPOINT"
  --features "$FEATURES"
  --critical-summary "$CRITICAL_SUMMARY"
  --safe-summary "$SAFE_SUMMARY"
  --action-horizon "$ACTION_HORIZON"
  --prediction-horizon "$PREDICTION_HORIZON"
  --q-action-horizon "$Q_ACTION_HORIZON"
  --encoder-batch-size "$ENCODER_BATCH_SIZE"
  --device cuda
  --prepare-only
)
if [[ "$FORCE_FEATURES" == "1" ]]; then
  feature_args+=(--rebuild-features)
fi
if [[ -n "$MAX_EPISODES" ]]; then
  feature_args+=(--max-episodes "$MAX_EPISODES")
fi

train_args=(
  --features "$FEATURES"
  --output-dir "$OUTPUT_DIR"
  --device cuda
  --action-horizon "$ACTION_HORIZON"
  --total-steps "$TOTAL_STEPS"
  --episode-batch-size "$EPISODE_BATCH_SIZE"
  --eval-every "$EVAL_EVERY"
  --seed "$SEED"
  --objective "$OBJECTIVE"
  --target-outcome "$TARGET_OUTCOME"
  --success-loss-mode "$SUCCESS_LOSS_MODE"
  --model-arch "$MODEL_ARCH"
  --action-num-heads "$ACTION_NUM_HEADS"
  --action-conv-layers "$ACTION_CONV_LAYERS"
  --prefix-conv-layers "$PREFIX_CONV_LAYERS"
  --stage1-steps "$STAGE1_STEPS"
  --initial-state-logit-bias "$INITIAL_STATE_LOGIT_BIAS"
  --initial-action-delta-bias "$INITIAL_ACTION_DELTA_BIAS"
  --residual-l1-weight "$RESIDUAL_L1_WEIGHT"
  --shuffled-residual-weight "$SHUFFLED_RESIDUAL_WEIGHT"
  --smoothness-weight "$SMOOTHNESS_WEIGHT"
  --success-residual-weight "$SUCCESS_RESIDUAL_WEIGHT"
  --decorrelation-weight "$DECORRELATION_WEIGHT"
  --pairwise-rank-weight "$PAIRWISE_RANK_WEIGHT"
  --pairwise-rank-margin "$PAIRWISE_RANK_MARGIN"
  --pairwise-rank-negatives "$PAIRWISE_RANK_NEGATIVES"
  --progress-consistency-weight "$PROGRESS_CONSISTENCY_WEIGHT"
  --progress-consistency-space "$PROGRESS_CONSISTENCY_SPACE"
  --delta-progress-weight "$DELTA_PROGRESS_WEIGHT"
  --delta-progress-stride "$DELTA_PROGRESS_STRIDE"
  --delta-progress-space "$DELTA_PROGRESS_SPACE"
  --delta-progress-clip "$DELTA_PROGRESS_CLIP"
  --delta-progress-huber-beta "$DELTA_PROGRESS_HUBER_BETA"
  --delta-progress-abs-weight "$DELTA_PROGRESS_ABS_WEIGHT"
  --checkpoint-metric "$CHECKPOINT_METRIC"
  --safe-anchor-weight "$SAFE_ANCHOR_WEIGHT"
  --safe-anchor-epsilon "$SAFE_ANCHOR_EPSILON"
  --temporal-risk-weight "$TEMPORAL_RISK_WEIGHT"
  --temporal-risk-margin "$TEMPORAL_RISK_MARGIN"
  --temporal-stride "$TEMPORAL_STRIDE"
  --temporal-min-increase "$TEMPORAL_MIN_INCREASE"
)
if [[ "$TEMPORAL_NORMALIZE_WEIGHTS" == "1" ]]; then
  train_args+=(--temporal-normalize-weights)
fi

analysis_args=(
  --rollouts "$ROLLOUTS"
  --predictions "$OUTPUT_DIR/prefix_predictions.npz"
  --output-dir "$ANALYSIS_DIR"
  --top-k 50
)

case "$STAGE" in
  features)
    "$PYTHON" -B scripts/train_rgb_dp_hazard_mil.py "${feature_args[@]}"
    ;;

  train)
    "$PYTHON" -B scripts/train_rgb_dp_causal_prefix_risk.py "${train_args[@]}"
    ;;

  analyze)
    "$PYTHON" -B scripts/analyze_prefix_risk_predictions.py "${analysis_args[@]}"
    ;;

  all)
    "$PYTHON" -B scripts/train_rgb_dp_hazard_mil.py "${feature_args[@]}"
    "$PYTHON" -B scripts/train_rgb_dp_causal_prefix_risk.py "${train_args[@]}"
    "$PYTHON" -B scripts/analyze_prefix_risk_predictions.py "${analysis_args[@]}"
    ;;

  smoke)
    MAX_EPISODES=${MAX_EPISODES:-20}
    smoke_features="rollouts/square_rgb_dp/epoch190_collection/risk_model_smoke/chunk_features.npz"
    smoke_output="trained_models/square_rgb_dp_causal_prefix_risk/epoch190_smoke"
    smoke_analysis="rollouts/square_rgb_dp/epoch190_collection/risk_analysis_smoke"
    "$PYTHON" -B scripts/train_rgb_dp_hazard_mil.py \
      --rollouts "$ROLLOUTS" \
      --dp-checkpoint "$DP_CHECKPOINT" \
      --features "$smoke_features" \
      --critical-summary "$CRITICAL_SUMMARY" \
      --safe-summary "$SAFE_SUMMARY" \
      --action-horizon "$ACTION_HORIZON" \
      --prediction-horizon "$PREDICTION_HORIZON" \
      --encoder-batch-size "$ENCODER_BATCH_SIZE" \
      --max-episodes "$MAX_EPISODES" \
      --rebuild-features \
      --device cuda \
      --prepare-only
    "$PYTHON" -B scripts/train_rgb_dp_causal_prefix_risk.py \
      --features "$smoke_features" \
      --output-dir "$smoke_output" \
      --device cuda \
      --action-horizon "$ACTION_HORIZON" \
      --model-arch "$MODEL_ARCH" \
      --action-num-heads "$ACTION_NUM_HEADS" \
      --action-conv-layers "$ACTION_CONV_LAYERS" \
      --prefix-conv-layers "$PREFIX_CONV_LAYERS" \
      --total-steps 50 \
      --episode-batch-size 8 \
      --eval-every 25 \
      --seed "$SEED"
    "$PYTHON" -B scripts/analyze_prefix_risk_predictions.py \
      --rollouts "$ROLLOUTS" \
      --predictions "$smoke_output/prefix_predictions.npz" \
      --output-dir "$smoke_analysis" \
      --top-k 10
    ;;

  *)
    echo "Usage: $0 {features|train|analyze|all|smoke}" >&2
    echo "Common overrides: TOTAL_STEPS=5000 FORCE_FEATURES=1 MAX_EPISODES=100" >&2
    exit 2
    ;;
esac
