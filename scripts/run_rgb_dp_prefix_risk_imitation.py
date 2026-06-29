#!/usr/bin/env python3
"""Run matched RGB-DP imitation with learned pre-risk failure prefixes.

The positive data are original demonstrations, successful policy rollouts,
and exact 16-action chunks selected before a learned action-risk transition.
The script is restart-safe and uses the established five-seed, 500-rollout
evaluation protocol.
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
    / "lift_rgb_dp_prefix_risk_good_chunks.hdf5"
)
MODEL_ROOT = ROOT / "trained_models/rgb_dp_segment_posttrain"

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


def weight_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    return env


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=process_env(), check=True)


def paths(args) -> dict[str, Path]:
    tag = weight_tag(args.prefix_risk_weight)
    return {
        "config": (
            ROOT
            / "robomimic/exps/templates/rgb_dp_segment_posttrain"
            / f"posttrain_prefix_risk_good_w{tag}.json"
        ),
        "experiment": (
            MODEL_ROOT
            / f"lift_rgb2_dp_prefix_risk_good_w{tag}_posttrain_s{args.seed}"
        ),
        "evaluation": ROOT / f"rollouts/rgb_dp/prefix_risk_good_w{tag}_eval",
        "restarts": (
            ROOT / f"rollouts/rgb_dp/prefix_risk_good_w{tag}_train_restarts"
        ),
        "experiment_summary": (
            ROOT / f"rollouts/rgb_dp/prefix_risk_good_w{tag}_experiment.json"
        ),
    }


def build_dataset(overwrite: bool) -> None:
    if FILTERED.exists() and not overwrite:
        print(f"[reuse dataset] {FILTERED}", flush=True)
        return
    command = [
        str(PYTHON),
        "-B",
        "scripts/build_rgb_dp_prefix_risk_filtered_chunks.py",
    ]
    if overwrite:
        command.append("--overwrite")
    run(command)


def audit_dataset() -> dict:
    summary = json.loads(FILTERED.with_suffix(".summary.json").read_text())
    records = summary["chunks"]
    if not summary["quality_gate"]["passed"]:
        raise RuntimeError("prefix-risk dataset did not pass its quality gate")
    if summary["quality_gate_overridden"]:
        raise RuntimeError("refusing a dataset created by overriding a failed gate")
    if not records:
        raise RuntimeError("prefix-risk filter retained no chunks")
    if summary["selection"]["contains_padded_target_actions"]:
        raise RuntimeError("prefix-risk targets unexpectedly contain padding")

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
                raise RuntimeError("a prefix-risk demo does not contain 17 frames")
            if int(output.attrs["sample_start_offset"]) != 1:
                raise RuntimeError("incorrect observation-history offset")
            for key in (
                "actions",
                "obs/agentview_image",
                "obs/robot0_eye_in_hand_image",
                "obs/robot0_eef_pos",
            ):
                if not np.array_equal(output[key][:], original[key][:][indices]):
                    raise RuntimeError(
                        f"source mismatch for {record['output_demo']} key={key}"
                    )
    print(
        "[audit passed] "
        f"{summary['retained_chunks']} chunks from "
        f"{summary['retained_source_rollouts']} failed rollouts; "
        f"privileged overlap="
        f"{summary['privileged_overlap_audit']['target_overlaps_any_critical_window']}",
        flush=True,
    )
    return summary


def generate_config(args) -> Path:
    experiment_paths = paths(args)
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
            "--prefix-risk-weight",
            str(args.prefix_risk_weight),
        ]
    )
    config_path = experiment_paths["config"]
    config = json.loads(config_path.read_text())
    expected = str(FILTERED)
    matching = [
        entry
        for entry in config["train"]["data"]
        if entry["path"] == expected
    ]
    if len(matching) != 1:
        raise RuntimeError("generated config does not contain prefix-risk dataset")
    if float(matching[0]["weight"]) != args.prefix_risk_weight:
        raise RuntimeError("generated config contains the wrong prefix-risk weight")
    if not matching[0].get("demo_start_only"):
        raise RuntimeError("prefix-risk dataset must sample only exact demo starts")
    return config_path


def latest_run(experiment_root: Path) -> Path | None:
    candidates = sorted(path for path in experiment_root.glob("*") if path.is_dir())
    return candidates[-1] if candidates else None


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
    if run_dir is None:
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


def write_experiment_summary(
    args,
    dataset_summary: dict,
    checkpoint: Path,
    evaluation: dict,
) -> None:
    success_control_path = (
        ROOT / "rollouts/rgb_dp/posttrain_success_eval/stability_summary.json"
    )
    success_control = (
        json.loads(success_control_path.read_text())
        if success_control_path.exists()
        else None
    )
    result = {
        "method": "demos + successful rollouts + learned pre-risk prefixes",
        "prefix_risk_weight": args.prefix_risk_weight,
        "dataset_summary": str(FILTERED.with_suffix(".summary.json")),
        "retained_chunks": dataset_summary["retained_chunks"],
        "checkpoint": str(checkpoint),
        "evaluation_summary": str(
            paths(args)["evaluation"] / "stability_summary.json"
        ),
        "total_success": evaluation["total_success"],
        "total_rollouts": evaluation["total_rollouts"],
        "pooled_success_rate": evaluation["pooled_success_rate"],
        "success_only_control": success_control,
    }
    output = paths(args)["experiment_summary"]
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("build", "audit", "config", "train", "eval"),
        default=("build", "audit", "config", "train", "eval"),
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--prefix-risk-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-restarts", type=int, default=20)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()

    stages = set(args.stages)
    if "build" in stages:
        build_dataset(args.overwrite_data)
    dataset_summary = audit_dataset()
    if "config" in stages:
        generate_config(args)

    checkpoint = completed_checkpoint(args)
    if "train" in stages:
        checkpoint = train(args)
    if "eval" in stages:
        if checkpoint is None:
            raise FileNotFoundError("no completed checkpoint; run the train stage")
        evaluation = evaluate(args, checkpoint)
        write_experiment_summary(args, dataset_summary, checkpoint, evaluation)
        print(
            f"[complete] {evaluation['total_success']}/"
            f"{evaluation['total_rollouts']} success "
            f"({100.0 * evaluation['pooled_success_rate']:.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
