#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

PYTHON_BIN="${PYTHON_BIN:-/home/ryan/miniconda3/envs/robomimic_stable/bin/python}"
BASE_FEATURES="${BASE_FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz}"
POTENTIAL_CHECKPOINT="${POTENTIAL_CHECKPOINT:-trained_models/square_success_potential/stage1_default_features/best.pt}"
ALPHA="${ALPHA:-0.5}"
PROGRESS_CLIP="${PROGRESS_CLIP:-0.1}"
REWARD_MODE="${REWARD_MODE:-sparse_plus_progress}"
TERMINAL_NEXT_MODE="${TERMINAL_NEXT_MODE:-feature}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8192}"

safe_alpha="${ALPHA//./p}"
safe_clip="${PROGRESS_CLIP//./p}"
OUTPUT="${OUTPUT:-rollouts/square_rgb_dp/epoch190_collection/idql/success_potential_${REWARD_MODE}_alpha${safe_alpha}_clip${safe_clip}_one_step_features.npz}"

"${PYTHON_BIN}" -B scripts/build_square_success_potential_reward_one_step_features.py \
  --base-features "${BASE_FEATURES}" \
  --potential-checkpoint "${POTENTIAL_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --alpha "${ALPHA}" \
  --progress-clip "${PROGRESS_CLIP}" \
  --reward-mode "${REWARD_MODE}" \
  --terminal-next-mode "${TERMINAL_NEXT_MODE}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}"
