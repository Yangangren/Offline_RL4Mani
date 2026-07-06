#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/robomimic_one_step_idql_pycache_${USER}_$$"
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

DP_CHECKPOINT=${DP_CHECKPOINT:-trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth}
DEMO_DATASET=${DEMO_DATASET:-datasets/square/ph/image_v15.hdf5}
ROLLOUT_DATASET=${ROLLOUT_DATASET:-rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5}
FEATURES=${FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz}
RISK_CHECKPOINT=${RISK_CHECKPOINT:-trained_models/square_rgb_dp_causal_prefix_risk/epoch190_two_stage_temporal_safe_anchor/best.pt}
RISK_REWARD_FEATURES=${RISK_REWARD_FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/positive_action_risk_reward_one_step_features.npz}
HYBRID_REWARD_FEATURES=${HYBRID_REWARD_FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/hybrid_default_minus_0p1_positive_action_risk_one_step_features.npz}
SIGNED_RISK_REWARD_FEATURES=${SIGNED_RISK_REWARD_FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/sparse_signed_risk_lambda0p1_q95_one_step_features.npz}
FAILURE_ONLY_SIGNED_RISK_REWARD_FEATURES=${FAILURE_ONLY_SIGNED_RISK_REWARD_FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/failure_only_signed_risk_lambda0p1_q95_one_step_features.npz}
FAILURE_ONLY_POTENTIAL_RISK_REWARD_FEATURES=${FAILURE_ONLY_POTENTIAL_RISK_REWARD_FEATURES:-rollouts/square_rgb_dp/epoch190_collection/idql/failure_only_potential_risk_lambda0p1_one_step_features.npz}
OUTPUT_DIR=${OUTPUT_DIR:-trained_models/square_rgb_dp_idql/default_reward_one_step_idql_paper_faithful}
IDQL_CHECKPOINT=${IDQL_CHECKPOINT:-$OUTPUT_DIR/best_success_auc.pt}
EVAL_OUTPUT=${EVAL_OUTPUT:-rollouts/square_rgb_dp/one_step_idql_eval}
ACTOR_SOURCE=${ACTOR_SOURCE:-idql_target_one_step_mlp}
CRITIC_SOURCE=${CRITIC_SOURCE:-target}
EVAL_NUM_CANDIDATES_VALUE=${EVAL_NUM_CANDIDATES:-"1 4 8 16 32 64"}
read -r -a EVAL_NUM_CANDIDATE_ARGS <<< "$EVAL_NUM_CANDIDATES_VALUE"
EVAL_SEEDS_VALUE=${EVAL_SEEDS:-"0 1 2"}
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS_VALUE"

STAGE=${1:-train}
RESUME_POINT_VALUE="${RESUME_POINT:-${RESUME_CHECKPOINT:-}}"
RESUME_ARGS=()
if [[ -n "$RESUME_POINT_VALUE" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RESUME_POINT_VALUE")
fi
ROLLOUT_EVAL_SEEDS_VALUE=${ROLLOUT_EVAL_SEEDS:-"0 1 2"}
read -r -a ROLLOUT_EVAL_SEED_ARGS <<< "$ROLLOUT_EVAL_SEEDS_VALUE"
ROLLOUT_EVAL_EVERY_VALUE="${ROLLOUT_EVAL_EVERY:-0}"
if [[ "$STAGE" == "train_with_rollout_eval" ]]; then
  ROLLOUT_EVAL_EVERY_VALUE="${ROLLOUT_EVAL_EVERY:-5000}"
fi
ROLLOUT_EVAL_COMMON_ARGS=()
if [[ "$ROLLOUT_EVAL_EVERY_VALUE" != "0" ]]; then
  ROLLOUT_EVAL_COMMON_ARGS=(
    --rollout-eval-every "$ROLLOUT_EVAL_EVERY_VALUE"
    --rollout-eval-seeds "${ROLLOUT_EVAL_SEED_ARGS[@]}"
    --rollout-eval-n-rollouts "${ROLLOUT_EVAL_N_ROLLOUTS:-50}"
    --rollout-eval-horizon "${ROLLOUT_EVAL_HORIZON:-400}"
    --rollout-eval-num-candidates "${ROLLOUT_EVAL_N:-1}"
    --rollout-eval-candidate-batch-size "${ROLLOUT_EVAL_CANDIDATE_BATCH_SIZE:-16}"
    --rollout-eval-num-inference-steps "${ROLLOUT_EVAL_NUM_INFERENCE_STEPS:-100}"
    --rollout-eval-selection "${ROLLOUT_EVAL_SELECTION:-argmax}"
    --rollout-eval-device cuda
    --rollout-eval-retries "${ROLLOUT_EVAL_RETRIES:-3}"
    --rollout-eval-rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-5}"
  )
fi
NORMALIZE_ACTION_ARGS=(--no-normalize-actions)
if [[ "${NORMALIZE_ACTIONS:-0}" == "1" ]]; then
  NORMALIZE_ACTION_ARGS=(--normalize-actions)
fi
CLIP_SAMPLE_ARGS=(--clip-sample)
if [[ "${CLIP_SAMPLE:-1}" == "0" ]]; then
  CLIP_SAMPLE_ARGS=(--no-clip-sample)
fi
DIFFUSION_CLIP_SAMPLE_ARGS=(--diffusion-clip-sample)
if [[ "${DIFFUSION_CLIP_SAMPLE:-1}" == "0" ]]; then
  DIFFUSION_CLIP_SAMPLE_ARGS=(--no-diffusion-clip-sample)
fi
SIGNED_RISK_SCALE_ARGS=()
if [[ -n "${SIGNED_RISK_SCALE:-}" ]]; then
  SIGNED_RISK_SCALE_ARGS=(--signed-risk-scale "$SIGNED_RISK_SCALE")
fi

case "$STAGE" in
  build_features)
    "$PYTHON" -B scripts/build_square_rgb_dp_chunk_idql_features.py \
      --checkpoint "$DP_CHECKPOINT" \
      --demo-dataset "$DEMO_DATASET" \
      --rollout-dataset "$ROLLOUT_DATASET" \
      --output "$FEATURES" \
      --chunk-horizon 1 \
      --stride 1 \
      --observation-horizon 2 \
      --gamma "${GAMMA:-0.99}" \
      --encoder-batch-size "${ENCODER_BATCH_SIZE:-128}" \
      --device cuda
    ;;

  build_risk_reward_features)
    "$PYTHON" -B scripts/build_square_rgb_dp_risk_reward_one_step_features.py \
      --base-features "$FEATURES" \
      --demo-dataset "$DEMO_DATASET" \
      --rollout-dataset "$ROLLOUT_DATASET" \
      --risk-checkpoint "$RISK_CHECKPOINT" \
      --output "$RISK_REWARD_FEATURES" \
      --reward-mode risk_only \
      --risk-lambda "${RISK_LAMBDA:-1.0}" \
      --risk-threshold "${RISK_THRESHOLD:-0.014938089996576302}" \
      --reward-clip "${RISK_REWARD_CLIP:-1.0}" \
      --pad-mode "${RISK_PAD_MODE:-zero}" \
      --eval-batch-size "${RISK_EVAL_BATCH_SIZE:-64}" \
      --device cuda
    ;;

  build_hybrid_reward_features)
    "$PYTHON" -B scripts/build_square_rgb_dp_risk_reward_one_step_features.py \
      --base-features "$FEATURES" \
      --demo-dataset "$DEMO_DATASET" \
      --rollout-dataset "$ROLLOUT_DATASET" \
      --risk-checkpoint "$RISK_CHECKPOINT" \
      --output "$HYBRID_REWARD_FEATURES" \
      --reward-mode hybrid_default_minus_risk \
      --risk-lambda "${RISK_LAMBDA:-0.1}" \
      --risk-threshold "${RISK_THRESHOLD:-0.014938089996576302}" \
      --reward-clip "${RISK_REWARD_CLIP:-1.0}" \
      --pad-mode "${RISK_PAD_MODE:-zero}" \
      --eval-batch-size "${RISK_EVAL_BATCH_SIZE:-64}" \
      --device cuda
    ;;

  build_signed_risk_reward_features)
    "$PYTHON" -B scripts/build_square_rgb_dp_risk_reward_one_step_features.py \
      --base-features "$FEATURES" \
      --demo-dataset "$DEMO_DATASET" \
      --rollout-dataset "$ROLLOUT_DATASET" \
      --risk-checkpoint "$RISK_CHECKPOINT" \
      --output "$SIGNED_RISK_REWARD_FEATURES" \
      --reward-mode hybrid_default_signed_risk \
      --risk-lambda "${RISK_LAMBDA:-0.1}" \
      --signed-risk-quantile "${SIGNED_RISK_QUANTILE:-0.95}" \
      "${SIGNED_RISK_SCALE_ARGS[@]}" \
      --risk-threshold "${RISK_THRESHOLD:-0.014938089996576302}" \
      --reward-clip "${RISK_REWARD_CLIP:-1.0}" \
      --pad-mode "${RISK_PAD_MODE:-zero}" \
      --eval-batch-size "${RISK_EVAL_BATCH_SIZE:-64}" \
      --device cuda
    ;;

  build_failure_only_signed_risk_reward_features)
    "$PYTHON" -B scripts/build_square_rgb_dp_risk_reward_one_step_features.py \
      --base-features "$FEATURES" \
      --demo-dataset "$DEMO_DATASET" \
      --rollout-dataset "$ROLLOUT_DATASET" \
      --risk-checkpoint "$RISK_CHECKPOINT" \
      --output "$FAILURE_ONLY_SIGNED_RISK_REWARD_FEATURES" \
      --reward-mode failure_only_signed_risk \
      --risk-lambda "${RISK_LAMBDA:-0.1}" \
      --signed-risk-quantile "${SIGNED_RISK_QUANTILE:-0.95}" \
      "${SIGNED_RISK_SCALE_ARGS[@]}" \
      --risk-threshold "${RISK_THRESHOLD:-0.014938089996576302}" \
      --reward-clip "${RISK_REWARD_CLIP:-1.0}" \
      --pad-mode "${RISK_PAD_MODE:-zero}" \
      --eval-batch-size "${RISK_EVAL_BATCH_SIZE:-64}" \
      --device cuda
    ;;

  build_failure_only_potential_risk_reward_features)
    "$PYTHON" -B scripts/build_square_rgb_dp_risk_reward_one_step_features.py \
      --base-features "$FEATURES" \
      --demo-dataset "$DEMO_DATASET" \
      --rollout-dataset "$ROLLOUT_DATASET" \
      --risk-checkpoint "$RISK_CHECKPOINT" \
      --output "$FAILURE_ONLY_POTENTIAL_RISK_REWARD_FEATURES" \
      --reward-mode failure_only_potential_risk_shaping \
      --risk-lambda "${RISK_LAMBDA:-0.1}" \
      --potential-type "${POTENTIAL_TYPE:-probability}" \
      --terminal-risk-mode "${TERMINAL_RISK_MODE:-outcome}" \
      --risk-threshold "${RISK_THRESHOLD:-0.014938089996576302}" \
      --reward-clip "${RISK_REWARD_CLIP:-1.0}" \
      --pad-mode "${RISK_PAD_MODE:-zero}" \
      --eval-batch-size "${RISK_EVAL_BATCH_SIZE:-64}" \
      --device cuda
    ;;

  train)
    "$PYTHON" -B scripts/train_square_rgb_dp_one_step_idql.py \
      --features "$FEATURES" \
      --output-dir "$OUTPUT_DIR" \
      "${RESUME_ARGS[@]}" \
      --device cuda \
      --total-steps "${TOTAL_STEPS:-50000}" \
      --batch-size "${BATCH_SIZE:-512}" \
      --eval-every "${EVAL_EVERY:-1000}" \
      --log-every "${LOG_EVERY:-100}" \
      --expectile "${EXPECTILE:-0.7}" \
      --target-tau "${TARGET_TAU:-0.005}" \
      --actor-target-tau "${ACTOR_TARGET_TAU:-0.001}" \
      --critic-lr "${CRITIC_LR:-3e-4}" \
      --actor-lr "${ACTOR_LR:-3e-4}" \
      --reward-scale "${REWARD_SCALE:-1.0}" \
      --num-diffusion-steps "${NUM_DIFFUSION_STEPS:-100}" \
      "${NORMALIZE_ACTION_ARGS[@]}" \
      "${CLIP_SAMPLE_ARGS[@]}" \
      "${ROLLOUT_EVAL_COMMON_ARGS[@]}"
    ;;

  train_resilient)
    max_restarts="${MAX_RESTARTS:-20}"
    retry_sleep="${RETRY_SLEEP:-5}"
    attempt=1
    resume_for_attempt="$RESUME_POINT_VALUE"
    if [[ -z "$resume_for_attempt" && -f "$OUTPUT_DIR/latest.pt" ]]; then
      resume_for_attempt="$OUTPUT_DIR/latest.pt"
    fi
    while (( attempt <= max_restarts )); do
      attempt_resume_args=()
      if [[ -n "$resume_for_attempt" ]]; then
        attempt_resume_args=(--resume-checkpoint "$resume_for_attempt")
      fi
      echo "[train_resilient attempt=$attempt/$max_restarts] resume=${resume_for_attempt:-none}" >&2
      set +e
      "$PYTHON" -B scripts/train_square_rgb_dp_one_step_idql.py \
        --features "$FEATURES" \
        --output-dir "$OUTPUT_DIR" \
        "${attempt_resume_args[@]}" \
        --device cuda \
        --total-steps "${TOTAL_STEPS:-50000}" \
        --batch-size "${BATCH_SIZE:-512}" \
        --eval-every "${EVAL_EVERY:-1000}" \
        --log-every "${LOG_EVERY:-100}" \
        --expectile "${EXPECTILE:-0.7}" \
        --target-tau "${TARGET_TAU:-0.005}" \
        --actor-target-tau "${ACTOR_TARGET_TAU:-0.001}" \
        --critic-lr "${CRITIC_LR:-3e-4}" \
        --actor-lr "${ACTOR_LR:-3e-4}" \
        --reward-scale "${REWARD_SCALE:-1.0}" \
        --num-diffusion-steps "${NUM_DIFFUSION_STEPS:-100}" \
        "${NORMALIZE_ACTION_ARGS[@]}" \
        "${CLIP_SAMPLE_ARGS[@]}" \
        "${ROLLOUT_EVAL_COMMON_ARGS[@]}"
      status=$?
      set -e
      if [[ "$status" -eq 0 ]]; then
        exit 0
      fi
      echo "[train_resilient attempt=$attempt] training exited with status $status" >&2
      if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
        echo "[train_resilient] last.pt exists; treating training as complete" >&2
        exit 0
      fi
      if [[ ! -f "$OUTPUT_DIR/latest.pt" ]]; then
        echo "[train_resilient] no latest.pt available to resume from" >&2
        exit "$status"
      fi
      resume_for_attempt="$OUTPUT_DIR/latest.pt"
      attempt=$((attempt + 1))
      sleep "$retry_sleep"
    done
    echo "[train_resilient] exhausted $max_restarts attempts" >&2
    exit 1
    ;;

  train_with_rollout_eval)
    "$PYTHON" -B scripts/train_square_rgb_dp_one_step_idql.py \
      --features "$FEATURES" \
      --output-dir "$OUTPUT_DIR" \
      "${RESUME_ARGS[@]}" \
      --device cuda \
      --total-steps "${TOTAL_STEPS:-50000}" \
      --batch-size "${BATCH_SIZE:-512}" \
      --eval-every "${EVAL_EVERY:-1000}" \
      --log-every "${LOG_EVERY:-100}" \
      --expectile "${EXPECTILE:-0.7}" \
      --target-tau "${TARGET_TAU:-0.005}" \
      --actor-target-tau "${ACTOR_TARGET_TAU:-0.001}" \
      --critic-lr "${CRITIC_LR:-3e-4}" \
      --actor-lr "${ACTOR_LR:-3e-4}" \
      --reward-scale "${REWARD_SCALE:-1.0}" \
      --num-diffusion-steps "${NUM_DIFFUSION_STEPS:-100}" \
      "${NORMALIZE_ACTION_ARGS[@]}" \
      "${CLIP_SAMPLE_ARGS[@]}" \
      "${ROLLOUT_EVAL_COMMON_ARGS[@]}"
    ;;

  smoke_train)
    "$PYTHON" -B scripts/train_square_rgb_dp_one_step_idql.py \
      --features "$FEATURES" \
      --output-dir "${SMOKE_OUTPUT_DIR:-/tmp/one_step_idql_smoke}" \
      --device "${SMOKE_DEVICE:-cpu}" \
      --total-steps 2 \
      --batch-size 16 \
      --eval-every 1 \
      --log-every 1 \
      --critic-hidden-dims 64 64 \
      --actor-hidden-dims 64 64 \
      --num-diffusion-steps 10
    ;;

  eval)
    "$PYTHON" -B scripts/eval_square_rgb_dp_one_step_idql.py \
      --idql-checkpoint "$IDQL_CHECKPOINT" \
      --output-dir "$EVAL_OUTPUT" \
      --device cuda \
      --actor-source "$ACTOR_SOURCE" \
      --critic-source "$CRITIC_SOURCE" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "${HORIZON:-400}" \
      --seed "${SEED:-0}" \
      --num-candidates "${N:-16}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS:-100}" \
      --selection "${SELECTION:-argmax}" \
      "${DIFFUSION_CLIP_SAMPLE_ARGS[@]}"
    ;;


  eval_grid)
    ckpt_name=$(basename "$IDQL_CHECKPOINT" .pt)
    grid_dir="${EVAL_OUTPUT}_${ckpt_name}_grid"
    for n in "${EVAL_NUM_CANDIDATE_ARGS[@]}"; do
      for seed in "${EVAL_SEED_ARGS[@]}"; do
        "$PYTHON" -B scripts/eval_square_rgb_dp_one_step_idql.py \
          --idql-checkpoint "$IDQL_CHECKPOINT" \
          --output-dir "$grid_dir" \
          --device cuda \
          --actor-source "$ACTOR_SOURCE" \
          --critic-source "$CRITIC_SOURCE" \
          --n-rollouts "${N_ROLLOUTS:-50}" \
          --horizon "${HORIZON:-400}" \
          --seed "$seed" \
          --num-candidates "$n" \
          --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
          --num-inference-steps "${NUM_INFERENCE_STEPS:-100}" \
          --selection "${SELECTION:-argmax}" \
          "${DIFFUSION_CLIP_SAMPLE_ARGS[@]}"
      done
    done
    "$PYTHON" -B - <<PYGRID
import json
from pathlib import Path
out = Path("$grid_dir")
rows = []
for path in sorted(out.glob("one_step_idql_N*_seed*.json")):
    if path.name.endswith("_partial.json"):
        continue
    data = json.loads(path.read_text())
    avg = data["average_rollout_stats"]
    rows.append({
        "path": str(path),
        "checkpoint": data["idql_checkpoint"],
        "checkpoint_step": data.get("checkpoint_step"),
        "N": data["num_candidates"],
        "seed": data["seed"],
        "n_rollouts": data["n_rollouts"],
        "num_success": avg["Num_Success"],
        "success_rate": avg["Success_Rate"],
        "return": avg["Return"],
        "horizon": avg["Horizon"],
        "critic_used_for_action_selection": data.get("critic_used_for_action_selection"),
        "execution_horizon": data.get("execution_horizon"),
    })
by_n = {}
for row in rows:
    by_n.setdefault(row["N"], []).append(row)
summary = {"rows": rows, "by_N": {}}
for n, group in sorted(by_n.items()):
    total_rollouts = sum(int(r["n_rollouts"]) for r in group)
    total_success = sum(float(r["num_success"]) for r in group)
    summary["by_N"][str(n)] = {
        "num_seeds": len(group),
        "total_rollouts": total_rollouts,
        "total_success": total_success,
        "success_rate": total_success / max(total_rollouts, 1),
        "mean_return": sum(float(r["return"]) * int(r["n_rollouts"]) for r in group) / max(total_rollouts, 1),
        "mean_horizon": sum(float(r["horizon"]) * int(r["n_rollouts"]) for r in group) / max(total_rollouts, 1),
    }
summary_path = out / "one_step_idql_eval_grid_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["by_N"], indent=2))
print(f"Wrote {summary_path}")
PYGRID
    ;;

  eval_grid_resilient)
    ckpt_name=$(basename "$IDQL_CHECKPOINT" .pt)
    grid_dir="${EVAL_OUTPUT}_${ckpt_name}_grid"
    "$PYTHON" -B scripts/run_square_rgb_dp_one_step_idql_eval_grid.py \
      --idql-checkpoint "$IDQL_CHECKPOINT" \
      --output-dir "$grid_dir" \
      --device cuda \
      --actor-source "$ACTOR_SOURCE" \
      --critic-source "$CRITIC_SOURCE" \
      --n-rollouts "${N_ROLLOUTS:-50}" \
      --horizon "${HORIZON:-400}" \
      --num-candidates "${EVAL_NUM_CANDIDATE_ARGS[@]}" \
      --seeds "${EVAL_SEED_ARGS[@]}" \
      --rollouts-per-chunk "${ROLLOUTS_PER_CHUNK:-5}" \
      --max-retries "${EVAL_MAX_RETRIES:-3}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-16}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS:-100}" \
      --selection "${SELECTION:-argmax}" \
      "${DIFFUSION_CLIP_SAMPLE_ARGS[@]}"
    ;;

  smoke_eval)
    "$PYTHON" -B scripts/eval_square_rgb_dp_one_step_idql.py \
      --idql-checkpoint "$IDQL_CHECKPOINT" \
      --output-dir "${SMOKE_EVAL_OUTPUT:-/tmp/one_step_idql_eval_smoke}" \
      --device cuda \
      --actor-source "$ACTOR_SOURCE" \
      --critic-source "$CRITIC_SOURCE" \
      --n-rollouts 1 \
      --horizon 400 \
      --seed 0 \
      --num-candidates "${N:-2}" \
      --candidate-batch-size "${CANDIDATE_BATCH_SIZE:-2}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS:-100}" \
      --selection argmax \
      "${DIFFUSION_CLIP_SAMPLE_ARGS[@]}"
    ;;

  *)
    echo "Usage: $0 {build_features|build_risk_reward_features|build_hybrid_reward_features|build_signed_risk_reward_features|build_failure_only_signed_risk_reward_features|build_failure_only_potential_risk_reward_features|train|train_resilient|train_with_rollout_eval|smoke_train|eval|eval_grid|eval_grid_resilient|smoke_eval}" >&2
    exit 2
    ;;
esac
