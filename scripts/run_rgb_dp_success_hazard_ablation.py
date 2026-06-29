#!/usr/bin/env python3
"""Run demos + success BC with success-context hazard regularization.

This ablation uses no failed rollout segments. It tests whether the frozen
prefix-risk model can improve post-training by regularizing sampled DP chunks
on successful deployment states only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ryan/miniconda3/envs/robomimic_clean/bin/python")
BASELINE = (
    ROOT
    / "trained_models/rgb_dp_segment_posttrain"
    / "lift_rgb2_dp_baseline_s1/20260627122714/models/model_epoch_25.pth"
)
CONFIG_ROOT = ROOT / "robomimic/exps/templates/rgb_dp_segment_posttrain"
MODEL_ROOT = ROOT / "trained_models/rgb_dp_segment_posttrain"
DATASET = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_success_hazard_context_chunks.hdf5"
)
HAZARD_CHECKPOINT = ROOT / "trained_models/rgb_dp_causal_prefix_risk/best.pt"

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPYCACHEPREFIX": "/tmp/robomimic_success_hazard_pycache",
    "PYTHONNOUSERSITE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


def weight_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def experiment_tag(args) -> str:
    return (
        f"dw{weight_tag(args.hazard_data_weight)}"
        f"_lw{weight_tag(args.hazard_loss_weight)}"
        f"_m{weight_tag(args.hazard_margin)}"
        f"_k{args.action_samples}"
    )


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    return env


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=process_env(), check=True)


def paths(args) -> dict[str, Path]:
    tag = experiment_tag(args)
    return {
        "config": CONFIG_ROOT
        / f"posttrain_success_hazard_regularized_{tag}.json",
        "experiment": MODEL_ROOT
        / f"lift_rgb2_dp_success_hazard_regularized_{tag}_s{args.seed}",
        "evaluation": ROOT / f"rollouts/rgb_dp/success_hazard_regularized_{tag}_eval",
        "restarts": ROOT
        / f"rollouts/rgb_dp/success_hazard_regularized_{tag}_train_restarts",
        "summary": ROOT / f"rollouts/rgb_dp/success_hazard_regularized_{tag}_experiment.json",
    }


def build_dataset(args) -> dict:
    summary = DATASET.with_suffix(".summary.json")
    if DATASET.exists() and summary.exists() and not args.overwrite_data:
        print(f"[reuse dataset] {DATASET}", flush=True)
        return json.loads(summary.read_text())
    command = [
        str(PYTHON),
        "-B",
        "scripts/build_rgb_dp_success_hazard_contexts.py",
        "--output",
        str(DATASET),
        "--max-chunks-per-success",
        str(args.max_chunks_per_success),
        "--minimum-spacing",
        str(args.minimum_spacing),
        "--device",
        args.device,
        "--overwrite",
    ]
    run(command)
    return json.loads(summary.read_text())


def generate_config(args) -> Path:
    command = [
        str(PYTHON),
        "-B",
        "scripts/rgb_dp_segment_posttrain.py",
        "--baseline-checkpoint",
        str(args.baseline_checkpoint),
        "--posttrain-epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--hazard-checkpoint",
        str(args.hazard_checkpoint),
        "--hazard-data-weight",
        str(args.hazard_data_weight),
        "--hazard-success-data-weight",
        str(args.hazard_data_weight),
        "--hazard-loss-weight",
        str(args.hazard_loss_weight),
        "--hazard-margin",
        str(args.hazard_margin),
        "--hazard-positive-reference-weight",
        str(args.positive_reference_weight),
        "--hazard-negative-reference-weight",
        str(args.negative_reference_weight),
        "--hazard-warmup-steps",
        str(args.warmup_steps),
        "--hazard-ramp-steps",
        str(args.ramp_steps),
        "--hazard-sampling-steps",
        str(args.sampling_steps),
        "--hazard-action-samples",
        str(args.action_samples),
    ]
    run(command)
    config_path = paths(args)["config"]
    config = json.loads(config_path.read_text())
    datasets = config["train"]["data"]
    if len(datasets) != 3:
        raise RuntimeError("success-hazard config should contain three datasets")
    hazard_dataset = datasets[2]
    if Path(hazard_dataset["path"]) != DATASET:
        raise RuntimeError("success-hazard config uses the wrong hazard dataset")
    if not hazard_dataset.get("hazard_failure", False):
        raise RuntimeError("success-hazard context stream must be hazard-only")
    return config_path


def latest_run(experiment_root: Path) -> Path | None:
    candidates = sorted(path for path in experiment_root.glob("*") if path.is_dir())
    return candidates[-1] if candidates else None


def resumable_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    for name in ("last.pth", "last_bak.pth"):
        checkpoint = run_dir / name
        if checkpoint.exists():
            return checkpoint
    return None


def archive_stale_experiment(experiment_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d%H%M%S")
    archive = experiment_root.with_name(f"{experiment_root.name}_stale_{stamp}")
    index = 1
    while archive.exists():
        archive = experiment_root.with_name(
            f"{experiment_root.name}_stale_{stamp}_{index:02d}"
        )
        index += 1
    experiment_root.rename(archive)
    print(
        f"[archive stale run] moved incomplete experiment to {archive}",
        flush=True,
    )
    return archive


def completed_checkpoint(args) -> Path | None:
    run_dir = latest_run(paths(args)["experiment"])
    if run_dir is None:
        return None
    checkpoint = run_dir / f"models/model_epoch_{args.epochs}.pth"
    return checkpoint if checkpoint.exists() else None


def train(args) -> Path:
    checkpoint = completed_checkpoint(args)
    if checkpoint is not None:
        print(f"[reuse checkpoint] {checkpoint}", flush=True)
        return checkpoint

    experiment_paths = paths(args)
    run_dir = latest_run(experiment_paths["experiment"])
    if (
        run_dir is not None
        and not (run_dir / f"models/model_epoch_{args.epochs}.pth").exists()
        and resumable_checkpoint(run_dir) is None
    ):
        archive_stale_experiment(experiment_paths["experiment"])
        run_dir = None

    if run_dir is None:
        try:
            run(
                [
                    str(PYTHON),
                    "-B",
                    "-m",
                    "robomimic.scripts.train",
                    "--config",
                    str(experiment_paths["config"]),
                ]
            )
        except subprocess.CalledProcessError as exc:
            print(
                "[initial train exited before target] "
                f"returncode={exc.returncode}; switching to resilient retry",
                flush=True,
            )
        run_dir = latest_run(experiment_paths["experiment"])
        if run_dir is None:
            raise RuntimeError("training produced no experiment directory")

    checkpoint = run_dir / f"models/model_epoch_{args.epochs}.pth"
    if not checkpoint.exists():
        run(
            [
                str(PYTHON),
                "-B",
                "scripts/resilient_train.py",
                "--config",
                str(experiment_paths["config"]),
                "--checkpoint",
                str(checkpoint),
                "--log-dir",
                str(experiment_paths["restarts"]),
                "--max-attempts",
                str(args.max_train_restarts),
            ]
        )
        run_dir = latest_run(experiment_paths["experiment"])
        if run_dir is None:
            raise RuntimeError("resilient training produced no experiment directory")
        checkpoint = run_dir / f"models/model_epoch_{args.epochs}.pth"
    if not checkpoint.exists():
        raise RuntimeError(f"target checkpoint was not created: {checkpoint}")
    return checkpoint


def valid_evaluation(args, checkpoint: Path) -> bool:
    summary_path = paths(args)["evaluation"] / "stability_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        Path(summary["checkpoint"]).resolve() == checkpoint.resolve()
        and summary["total_rollouts"] == 500
    )


def evaluate(args, checkpoint: Path) -> dict:
    output_dir = paths(args)["evaluation"]
    if valid_evaluation(args, checkpoint) and not args.force_eval:
        print(f"[reuse evaluation] {output_dir}", flush=True)
        return json.loads((output_dir / "stability_summary.json").read_text())
    command = [
        str(PYTHON),
        "-B",
        "scripts/validate_epoch50_platform.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--n-rollouts",
        "100",
        "--horizon",
        "400",
        "--max-retries",
        "5",
        "--evaluate-only",
    ]
    if args.force_eval:
        command.append("--force")
    run(command)
    return json.loads((output_dir / "stability_summary.json").read_text())


def write_summary(args, dataset_summary: dict, checkpoint: Path, evaluation: dict) -> None:
    result = {
        "method": "demo + success BC with success-context hazard regularization",
        "dataset": str(DATASET),
        "retained_success_hazard_contexts": dataset_summary["retained_chunks"],
        "retained_source_rollouts": dataset_summary["retained_source_rollouts"],
        "checkpoint": str(checkpoint),
        "evaluation_summary": str(
            paths(args)["evaluation"] / "stability_summary.json"
        ),
        "total_success": evaluation["total_success"],
        "total_rollouts": evaluation["total_rollouts"],
        "pooled_success_rate": evaluation["pooled_success_rate"],
    }
    output = paths(args)["summary"]
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("build", "config", "train", "eval"),
        default=("build", "config", "train", "eval"),
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE)
    parser.add_argument(
        "--hazard-checkpoint",
        type=Path,
        default=HAZARD_CHECKPOINT,
    )
    parser.add_argument("--hazard-data-weight", type=float, default=0.1)
    parser.add_argument("--hazard-loss-weight", type=float, default=0.02)
    parser.add_argument("--hazard-margin", type=float, default=0.1)
    parser.add_argument("--positive-reference-weight", type=float, default=0.05)
    parser.add_argument("--negative-reference-weight", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--ramp-steps", type=int, default=500)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--action-samples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-restarts", type=int, default=20)
    parser.add_argument("--max-chunks-per-success", type=int, default=0)
    parser.add_argument("--minimum-spacing", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    args.hazard_checkpoint = args.hazard_checkpoint.resolve()

    stages = set(args.stages)
    dataset_summary = {}
    if "build" in stages:
        dataset_summary = build_dataset(args)
    elif DATASET.with_suffix(".summary.json").exists():
        dataset_summary = json.loads(DATASET.with_suffix(".summary.json").read_text())

    if "config" in stages:
        generate_config(args)

    checkpoint = completed_checkpoint(args)
    if "train" in stages:
        checkpoint = train(args)
    if "eval" in stages:
        if checkpoint is None:
            raise FileNotFoundError("no completed checkpoint; run train first")
        evaluation = evaluate(args, checkpoint)
        write_summary(args, dataset_summary, checkpoint, evaluation)
        print(
            f"[complete] {evaluation['total_success']}/"
            f"{evaluation['total_rollouts']} success "
            f"({100.0 * evaluation['pooled_success_rate']:.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
