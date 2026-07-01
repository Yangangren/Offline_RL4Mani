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
DP_CHECKPOINT=${DP_CHECKPOINT:-/home/ryan/Documents/robomimic/trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth}
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
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-128}
TOTAL_STEPS=${TOTAL_STEPS:-5000}
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-16}
EVAL_EVERY=${EVAL_EVERY:-250}
SEED=${SEED:-20260630}
MAX_EPISODES=${MAX_EPISODES:-}
FORCE_FEATURES=${FORCE_FEATURES:-0}

STAGE=${1:-all}

feature_args=(
  --rollouts "$ROLLOUTS"
  --dp-checkpoint "$DP_CHECKPOINT"
  --features "$FEATURES"
  --critical-summary "$CRITICAL_SUMMARY"
  --safe-summary "$SAFE_SUMMARY"
  --action-horizon "$ACTION_HORIZON"
  --prediction-horizon "$PREDICTION_HORIZON"
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
)

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
