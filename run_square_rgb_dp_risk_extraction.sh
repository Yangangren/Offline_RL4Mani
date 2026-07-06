#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_square_risk_extract_pycache_${USER}_$$"
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

POLICY=${POLICY:-trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth}
RISK=${RISK:-trained_models/square_rgb_dp_causal_prefix_risk/epoch190_two_stage_temporal_safe_anchor/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-rollouts/square_rgb_dp/risk_extraction_eval}

NUM_CANDIDATES=${NUM_CANDIDATES:-"1 4 8 16"}
SCORE_MODES=${SCORE_MODES:-"positive_action_risk"}
SELECTIONS=${SELECTIONS:-"argmin"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
N_ROLLOUTS=${N_ROLLOUTS:-50}
ROLLOUTS_PER_CHUNK=${ROLLOUTS_PER_CHUNK:-1}
HORIZON=${HORIZON:-400}
CANDIDATE_BATCH_SIZE=${CANDIDATE_BATCH_SIZE:-16}
EXECUTE_HORIZON=${EXECUTE_HORIZON:-8}
ACTION_START_INDEX=${ACTION_START_INDEX:--1}
MAX_PREFIX_LEN=${MAX_PREFIX_LEN:-0}
MAX_RETRIES=${MAX_RETRIES:-3}
RISK_THRESHOLD=${RISK_THRESHOLD:-}
FORCE=${FORCE:-0}

STAGE=${1:-grid}

common_args=(
  --policy "$POLICY"
  --risk "$RISK"
  --output-dir "$OUTPUT_DIR"
  --device cuda
  --horizon "$HORIZON"
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE"
  --execute-horizon "$EXECUTE_HORIZON"
  --action-start-index "$ACTION_START_INDEX"
  --max-prefix-len "$MAX_PREFIX_LEN"
)
if [[ -n "$RISK_THRESHOLD" ]]; then
  common_args+=(--risk-threshold "$RISK_THRESHOLD")
fi

case "$STAGE" in
  smoke)
    "$PYTHON" -B scripts/eval_square_rgb_dp_risk_extraction.py \
      "${common_args[@]}" \
      --n-rollouts "${SMOKE_ROLLOUTS:-2}" \
      --seed "${SMOKE_SEED:-0}" \
      --num-candidates "${SMOKE_NUM_CANDIDATES:-2}" \
      --score-mode "${SMOKE_SCORE_MODE:-positive_action_risk}" \
      --selection "${SMOKE_SELECTION:-argmin}"
    ;;

  single)
    "$PYTHON" -B scripts/eval_square_rgb_dp_risk_extraction.py \
      "${common_args[@]}" \
      --n-rollouts "$N_ROLLOUTS" \
      --seed "${SEED:-0}" \
      --num-candidates "${N:-16}" \
      --score-mode "${SCORE_MODE:-positive_action_risk}" \
      --selection "${SELECTION:-argmin}"
    ;;

  grid)
    grid_args=(
      "${common_args[@]}"
      --num-candidates $NUM_CANDIDATES
      --score-modes $SCORE_MODES
      --selections $SELECTIONS
      --seeds $SEEDS
      --n-rollouts "$N_ROLLOUTS"
      --rollouts-per-chunk "$ROLLOUTS_PER_CHUNK"
      --max-retries "$MAX_RETRIES"
    )
    if [[ "$FORCE" == "1" ]]; then
      grid_args+=(--force)
    fi
    "$PYTHON" -B scripts/run_square_rgb_dp_risk_extraction_grid.py "${grid_args[@]}"
    ;;

  *)
    echo "Usage: $0 {smoke|single|grid}" >&2
    echo "Examples:" >&2
    echo "  $0 smoke" >&2
    echo "  N_ROLLOUTS=50 SEEDS='0 1 2 3 4' NUM_CANDIDATES='1 4 8 16' $0 grid" >&2
    echo "  SCORE_MODES='positive_action_risk action_delta_logodds' $0 grid" >&2
    exit 2
    ;;
esac
