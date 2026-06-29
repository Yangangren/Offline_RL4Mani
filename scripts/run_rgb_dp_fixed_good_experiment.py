#!/usr/bin/env python3
"""Reproduce the fixed-window RGB-DP failure-prefix experiment end to end.

Stages are restart-safe and reuse complete artifacts:

1. build privileged, pre-contact, fixed-length positive chunks;
2. verify every RGB history and action target against the source rollout;
3. generate the matched post-training config;
4. train (or resume) through epoch 25;
5. evaluate 500 rollouts over five independent seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ryan/miniconda3/envs/robomimic_clean/bin/python")
BASELINE = (
    ROOT
    / "trained_models/rgb_dp_segment_posttrain"
    / "lift_rgb2_dp_baseline_s1/20260627122714/models/model_epoch_25.pth"
)
SOURCE = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
FILTERED = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_good_chunks_fixed_window.hdf5"
)
CONFIG = (
    ROOT
    / "robomimic/exps/templates/rgb_dp_segment_posttrain"
    / "posttrain_fixed_good.json"
)
MODEL_ROOT = ROOT / "trained_models/rgb_dp_segment_posttrain"
EVAL_ROOT = ROOT / "rollouts/rgb_dp/fixed_good_eval"

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    return env


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=process_env(), check=True)


def build_dataset(overwrite: bool) -> None:
    if FILTERED.exists() and not overwrite:
        print(f"[reuse dataset] {FILTERED}", flush=True)
        return
    command = [
        str(PYTHON),
        "-B",
        "scripts/build_rgb_dp_good_action_chunks.py",
    ]
    if overwrite:
        command.append("--overwrite")
    run(command)


def audit_dataset() -> dict:
    summary_path = FILTERED.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    records = summary["chunks"]
    if not records:
        raise RuntimeError("fixed-window filter retained no chunks")

    with h5py.File(SOURCE, "r") as source, h5py.File(FILTERED, "r") as filtered:
        if len(filtered["data"]) != len(records):
            raise RuntimeError("summary and HDF5 chunk counts do not match")
        for record in records:
            output = filtered[f"data/{record['output_demo']}"]
            original = source[f"data/{record['source_demo']}"]
            boundary = int(record["decision_boundary"])
            indices = np.asarray(
                [max(0, boundary - 1)] + list(range(boundary, boundary + 16))
            )
            if int(output.attrs["num_samples"]) != 17:
                raise RuntimeError("a fixed-window demo does not contain 17 frames")
            for key in (
                "actions",
                "obs/agentview_image",
                "obs/robot0_eye_in_hand_image",
                "obs/robot0_eef_pos",
            ):
                if not np.array_equal(output[key][:], original[key][indices]):
                    raise RuntimeError(
                        f"source mismatch for {record['output_demo']} key={key}"
                    )

    if summary["contains_padded_target_actions"]:
        raise RuntimeError("fixed-window summary unexpectedly reports padded actions")
    print(
        "[audit passed] "
        f"{summary['retained_chunks']} chunks from "
        f"{summary['retained_source_rollouts']} failed rollouts; "
        f"stages={summary['stage_counts']}",
        flush=True,
    )
    return summary


def generate_config(args) -> None:
    run(
        [
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
            "--failure-weight",
            str(args.failure_weight),
        ]
    )


def experiment_root(seed: int) -> Path:
    return MODEL_ROOT / f"lift_rgb2_dp_fixed_good_posttrain_s{seed}"


def latest_run(seed: int) -> Path | None:
    root = experiment_root(seed)
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    return candidates[-1] if candidates else None


def completed_checkpoint(seed: int, epochs: int) -> Path | None:
    run_dir = latest_run(seed)
    if run_dir is None:
        return None
    checkpoint = run_dir / f"models/model_epoch_{epochs}.pth"
    return checkpoint if checkpoint.exists() else None


def train(args) -> Path:
    checkpoint = completed_checkpoint(args.seed, args.epochs)
    if checkpoint is not None:
        print(f"[reuse checkpoint] {checkpoint}", flush=True)
        return checkpoint

    run_dir = latest_run(args.seed)
    if run_dir is None:
        # robomimic catches training exceptions internally; inspect artifacts
        # afterward and let the restart wrapper recover if necessary.
        run(
            [
                str(PYTHON),
                "-B",
                "-m",
                "robomimic.scripts.train",
                "--config",
                str(CONFIG),
            ]
        )
        run_dir = latest_run(args.seed)
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
                str(CONFIG),
                "--checkpoint",
                str(checkpoint),
                "--log-dir",
                str(ROOT / "rollouts/rgb_dp/fixed_good_train_restarts"),
                "--max-attempts",
                str(args.max_train_restarts),
            ]
        )
    if not checkpoint.exists():
        raise RuntimeError(f"target checkpoint was not created: {checkpoint}")
    return checkpoint


def valid_evaluation(checkpoint: Path) -> bool:
    summary_path = EVAL_ROOT / "stability_summary.json"
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


def evaluate(checkpoint: Path, force: bool) -> dict:
    if valid_evaluation(checkpoint) and not force:
        print(f"[reuse evaluation] {EVAL_ROOT}", flush=True)
        return json.loads((EVAL_ROOT / "stability_summary.json").read_text())

    command = [
        str(PYTHON),
        "-B",
        "scripts/validate_epoch50_platform.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(EVAL_ROOT),
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
    if force:
        command.append("--force")
    run(command)
    return json.loads((EVAL_ROOT / "stability_summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("build", "audit", "config", "train", "eval"),
        default=("build", "audit", "config", "train", "eval"),
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--failure-weight", type=float, default=0.25)
    parser.add_argument("--max-train-restarts", type=int, default=20)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()

    stages = set(args.stages)
    if "build" in stages:
        build_dataset(overwrite=args.overwrite_data)
    if "audit" in stages:
        audit_dataset()
    if "config" in stages:
        generate_config(args)

    checkpoint = completed_checkpoint(args.seed, args.epochs)
    if "train" in stages:
        checkpoint = train(args)
    if "eval" in stages:
        if checkpoint is None:
            raise FileNotFoundError(
                "no completed checkpoint; include the train stage first"
            )
        result = evaluate(checkpoint, force=args.force_eval)
        print(
            f"[complete] {result['total_success']}/{result['total_rollouts']} "
            f"success ({100.0 * result['pooled_success_rate']:.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
