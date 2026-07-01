#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_square_pycache_${USER}_$$"
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
SCRIPT=scripts/run_square_rgb_dp_baseline.py

# Choose exactly one stage per run. Keeping stages separate avoids carrying
# simulator / torch / h5py state across dataset creation, training, and eval.
STAGE=${1:-prepare}

case "$STAGE" in
  dataset)
    # 1) Build datasets/square/ph/image_v15.hdf5 from raw Square PH states.
    #    If raw demo_v15.hdf5 is missing, this first downloads it.
    "$PYTHON" -B "$SCRIPT" \
      --stages dataset
    ;;

  prepare)
    # 2) Write / refresh the RGB-DP config only.
    "$PYTHON" -B "$SCRIPT" \
      --stages prepare
    ;;

  train)
    # 3) Official-style Square PH RGB-DP training.
    #    Defaults match robomimic DP as closely as possible:
    #    large UNet, DDPM, batch 100, 100 steps / epoch.
    #
    #    Override TARGET_EPOCH to choose the training length, e.g.
    #      TARGET_EPOCH=200 ./train_square_rgb_dp.sh train
    TARGET_EPOCH=${TARGET_EPOCH:-2000}
    "$PYTHON" -B "$SCRIPT" \
      --stages train \
      --recipe official \
      --epochs "$TARGET_EPOCH" \
      --target-epoch "$TARGET_EPOCH" \
      --steps-per-epoch 100 \
      --batch-size 100 \
      --lr 1e-4
    ;;

  eval)
    # 4) Evaluate an official-style checkpoint with 5 seeds x 100 rollouts.
    #    Override TARGET_EPOCH to evaluate an intermediate checkpoint.
    #    Override CHECKPOINT to evaluate an explicit path, e.g.
    #      TARGET_EPOCH=190 CHECKPOINT=trained_models/.../last.pth ./train_square_rgb_dp.sh eval
    #    Override EVAL_ROLLOUTS / EVAL_SEEDS for quick smoke checks, e.g.
    #      EVAL_ROLLOUTS=10 EVAL_SEEDS="0" ./train_square_rgb_dp.sh eval
    TARGET_EPOCH=${TARGET_EPOCH:-2000}
    EVAL_ROLLOUTS=${EVAL_ROLLOUTS:-100}
    EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE:-10}
    read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS:-0 1 2 3 4}"
    EVAL_ARGS=(
      --stages eval \
      --recipe official \
      --target-epoch "$TARGET_EPOCH" \
      --eval-rollouts "$EVAL_ROLLOUTS" \
      --eval-chunk-size "$EVAL_CHUNK_SIZE" \
      --eval-seeds "${EVAL_SEED_ARRAY[@]}"
    )
    if [[ -n "${CHECKPOINT:-}" ]]; then
      EVAL_ARGS+=(--checkpoint "$CHECKPOINT")
    fi
    if [[ "${FORCE_EVAL:-0}" == "1" ]]; then
      EVAL_ARGS+=(--force-eval)
    fi
    "$PYTHON" -B "$SCRIPT" "${EVAL_ARGS[@]}"
    ;;

  train_fast)
    # Debug-only short recipe used by the earlier quick sanity run.
    "$PYTHON" -B "$SCRIPT" \
      --stages train \
      --recipe fast \
      --epochs 25 \
      --target-epoch 25 \
      --steps-per-epoch 500 \
      --batch-size 64 \
      --lr 1e-4
    ;;

  eval_fast)
    "$PYTHON" -B "$SCRIPT" \
      --stages eval \
      --recipe fast \
      --target-epoch 25 \
      --eval-rollouts 100 \
      --eval-seeds 0 1 2 3 4
    ;;

  *)
    echo "Usage: $0 {dataset|prepare|train|eval|train_fast|eval_fast}" >&2
    exit 2
    ;;
esac
