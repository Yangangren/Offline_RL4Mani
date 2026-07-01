#!/usr/bin/env python3
"""Post-train Square RGB-DP with success-only or low-risk failure chunks.

Variants:

* ``success_only``: original demos + successful policy rollouts.
* ``lowrisk``: original demos + successful policy rollouts + low-risk chunks
  selected from failed policy rollouts by the learned prefix-risk model.
* ``progress_lowrisk``: same as ``lowrisk``, but the selected failure chunks
  must also make privileged Square task progress.
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
PYTHON = Path(
    os.environ.get(
        "ROBOMIMIC_PYTHON",
        "/home/ryan/miniconda3/envs/robomimic_stable/bin/python",
    )
)
TEMPLATE = ROOT / "robomimic/exps/templates/diffusion_policy.json"
CONFIG_DIR = ROOT / "robomimic/exps/templates/square_rgb_dp_posttrain"
MODEL_ROOT = ROOT / "trained_models/square_rgb_dp_posttrain"

RGB_DEMOS = ROOT / "datasets/square/ph/image_v15.hdf5"
RGB_ROLLOUTS = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
LOW_RISK_FAILURE = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection"
    / "square_rgb_dp_low_risk_failure_chunks.hdf5"
)
PROGRESS_LOW_RISK_FAILURE = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection"
    / "square_rgb_dp_progress_low_risk_failure_chunks.hdf5"
)
BASELINE = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1"
    / "20260629231002/last.pth"
)

RGB_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "ROBOMIMIC_SAVE_LATEST_EVERY_N_EPOCHS": "10",
}


def process_env(cache_suffix: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_square_lowrisk_{cache_suffix}"
    env["ROBOMIMIC_PYTHON"] = str(PYTHON)
    return env


def run(command: list[str], *, cache_suffix: str) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=process_env(cache_suffix), check=True)


def weight_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def is_success_only(args) -> bool:
    return args.variant == "success_only" or args.failure_weight <= 0.0


def method_tag(args) -> str:
    if is_success_only(args):
        demo_tag = f"_{args.demo_filter_key}" if args.demo_filter_key else ""
        return f"success_only{demo_tag}"
    if args.variant == "progress_lowrisk":
        return f"progress_lowrisk_w{weight_tag(args.failure_weight)}"
    return f"lowrisk_w{weight_tag(args.failure_weight)}"


def method_name(args) -> str:
    if is_success_only(args):
        return "Square RGB-DP demos + successful rollouts"
    if args.variant == "progress_lowrisk":
        return (
            "Square RGB-DP demos + successful rollouts + "
            "privileged progress-aware low-risk failure chunks"
        )
    return "Square RGB-DP demos + successful rollouts + low-risk failure chunks"


def image_encoder_config() -> dict:
    return {
        "core_class": "VisualCore",
        "core_kwargs": {
            "feature_dimension": 64,
            "backbone_class": "ResNet18Conv",
            "backbone_kwargs": {
                "pretrained": False,
                "input_coord_conv": False,
            },
            "pool_class": "SpatialSoftmax",
            "pool_kwargs": {
                "num_kp": 32,
                "learnable_temperature": False,
                "temperature": 1.0,
                "noise_std": 0.0,
            },
        },
        "obs_randomizer_class": "CropRandomizer",
        "obs_randomizer_kwargs": {
            "crop_height": 76,
            "crop_width": 76,
            "num_crops": 1,
            "pos_enc": False,
        },
    }


def make_config(args) -> dict:
    config = json.loads(TEMPLATE.read_text())
    config["experiment"]["name"] = f"square_rgb_dp_{method_tag(args)}_posttrain_s{args.seed}"
    config["experiment"]["render"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["keep_all_videos"] = False
    config["experiment"]["validate"] = False
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["save"]["every_n_epochs"] = args.save_every_epochs
    config["experiment"]["save"]["epochs"] = sorted(
        set([min(25, args.epochs), min(50, args.epochs), args.epochs])
    )
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["ckpt_path"] = str(args.baseline_checkpoint.resolve())

    demo_entry = {"path": str(args.demo_dataset.resolve()), "weight": args.demo_weight}
    if args.demo_filter_key:
        demo_entry["filter_key"] = args.demo_filter_key
    data = [
        demo_entry,
        {
            "path": str(args.rollout_dataset.resolve()),
            "filter_key": "success",
            "weight": args.success_weight,
        },
    ]
    if not is_success_only(args):
        data.append(
            {
                "path": str(args.low_risk_dataset.resolve()),
                "weight": args.failure_weight,
                "demo_start_only": True,
                "sample_start_offset": 1,
            }
        )
    config["train"]["data"] = data
    config["train"]["output_dir"] = str(MODEL_ROOT)
    config["train"]["normalize_weights_by_ds_size"] = True
    config["train"]["num_data_workers"] = 0
    config["train"]["hdf5_cache_mode"] = args.hdf5_cache_mode
    config["train"]["hdf5_load_next_obs"] = False
    config["train"]["seq_length"] = 16
    config["train"]["frame_stack"] = 2
    config["train"]["batch_size"] = args.batch_size
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed

    config["algo"]["horizon"] = {
        "observation_horizon": 2,
        "action_horizon": 8,
        "prediction_horizon": 16,
    }
    config["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] = args.lr
    config["algo"]["optim_params"]["policy"]["learning_rate"]["warmup_steps"] = min(
        500, max(50, args.epochs * args.steps_per_epoch // 10)
    )

    config["observation"]["modalities"]["obs"] = {
        "low_dim": PROPRIO_KEYS,
        "rgb": RGB_KEYS,
        "depth": [],
        "scan": [],
    }
    config["observation"]["modalities"]["goal"] = {
        "low_dim": [],
        "rgb": [],
        "depth": [],
        "scan": [],
    }
    config["observation"]["encoder"]["rgb"] = image_encoder_config()
    return config


def config_path(args) -> Path:
    return CONFIG_DIR / f"posttrain_{method_tag(args)}_s{args.seed}.json"


def experiment_root(args) -> Path:
    return MODEL_ROOT / f"square_rgb_dp_{method_tag(args)}_posttrain_s{args.seed}"


def evaluation_root(args) -> Path:
    return ROOT / f"rollouts/square_rgb_dp/{method_tag(args)}_posttrain_eval"


def build_dataset(args) -> None:
    if is_success_only(args):
        print("[success-only] no failure-chunk dataset to build", flush=True)
        return
    if args.low_risk_dataset.exists() and not args.overwrite_data:
        print(f"[reuse low-risk dataset] {args.low_risk_dataset}", flush=True)
        return
    command = [
        str(PYTHON),
        "-B",
        "scripts/build_square_rgb_dp_low_risk_failure_chunks.py",
        "--source",
        str(args.rollout_dataset),
        "--predictions",
        str(args.risk_predictions),
        "--output",
        str(args.low_risk_dataset),
        "--prediction-horizon",
        str(args.prediction_horizon),
        "--max-chunks-per-failure",
        str(args.max_chunks_per_failure),
        "--minimum-spacing",
        str(args.minimum_spacing),
        "--prefer",
        args.prefer,
        "--target-peg-xy",
        *[str(value) for value in args.target_peg_xy],
        "--min-peg-xy-progress",
        str(args.min_peg_xy_progress),
        "--min-nut-displacement",
        str(args.min_nut_displacement),
        "--min-nut-z-gain",
        str(args.min_nut_z_gain),
        "--peg-progress-weight",
        str(args.peg_progress_weight),
        "--z-gain-weight",
        str(args.z_gain_weight),
        "--nut-displacement-weight",
        str(args.nut_displacement_weight),
        "--eef-approach-weight",
        str(args.eef_approach_weight),
        "--progress-risk-penalty",
        str(args.progress_risk_penalty),
    ]
    if args.threshold is not None:
        command += ["--threshold", str(args.threshold)]
    else:
        command += ["--success-quantile", str(args.success_quantile)]
        if args.use_stored_threshold:
            command.append("--use-stored-threshold")
        else:
            command.append("--no-use-stored-threshold")
    if args.overwrite_data:
        command.append("--overwrite")
    run(command, cache_suffix="build_chunks")


def audit_dataset(args) -> dict:
    if is_success_only(args):
        summary = {
            "variant": "success_only",
            "source_failure_rollouts": 0,
            "retained_chunks": 0,
            "retained_source_rollouts": 0,
            "selection": None,
            "risk_stats": None,
        }
        print("[success-only] no failure-chunk dataset to audit", flush=True)
        return summary
    summary_path = args.low_risk_dataset.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    if summary["retained_chunks"] <= 0:
        raise RuntimeError("low-risk failure filter retained no chunks")
    if summary["contains_padded_target_actions"]:
        raise RuntimeError("low-risk dataset unexpectedly contains padded actions")
    with h5py.File(args.low_risk_dataset, "r") as dataset:
        if len(dataset["data"]) != summary["retained_chunks"]:
            raise RuntimeError("summary and HDF5 chunk counts do not match")
        for demo_key in list(dataset["data"].keys())[: min(10, len(dataset["data"]))]:
            group = dataset[f"data/{demo_key}"]
            if int(group.attrs["num_samples"]) != args.prediction_horizon + 1:
                raise RuntimeError(f"{demo_key} has wrong number of samples")
            if int(group.attrs["sample_start_offset"]) != 1:
                raise RuntimeError(f"{demo_key} has wrong sample_start_offset")
    print(
        "[audit passed] "
        f"{summary['retained_chunks']} chunks from "
        f"{summary['retained_source_rollouts']} failed rollouts; "
        f"threshold={summary['selection']['threshold']:.4f}",
        flush=True,
    )
    return summary


def write_config(args) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config_path(args)
    config = make_config(args)
    path.write_text(json.dumps(config, indent=4))
    print(f"Wrote {path}", flush=True)
    return path


def latest_run(args) -> Path | None:
    root = experiment_root(args)
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    return candidates[-1] if candidates else None


def completed_checkpoint(args) -> Path | None:
    run_dir = latest_run(args)
    if run_dir is None:
        return None
    checkpoint = run_dir / f"models/model_epoch_{args.epochs}.pth"
    return checkpoint if checkpoint.exists() else None


def train(args) -> Path:
    checkpoint = completed_checkpoint(args)
    if checkpoint is not None:
        print(f"[reuse checkpoint] {checkpoint}", flush=True)
        return checkpoint

    path = config_path(args)
    run_dir = latest_run(args)
    if run_dir is None:
        run(
            [str(PYTHON), "-B", "-m", "robomimic.scripts.train", "--config", str(path)],
            cache_suffix="train_first",
        )
        run_dir = latest_run(args)
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
                str(path),
                "--checkpoint",
                str(checkpoint),
                "--log-dir",
                str(ROOT / f"rollouts/square_rgb_dp/{method_tag(args)}_posttrain_restarts"),
                "--max-attempts",
                str(args.max_train_restarts),
            ],
            cache_suffix="train_resume",
        )
    if not checkpoint.exists():
        raise RuntimeError(f"target checkpoint was not created: {checkpoint}")
    return checkpoint


def evaluate(args, checkpoint: Path) -> dict:
    output_dir = evaluation_root(args)
    summary_path = output_dir / "stability_summary.json"
    if summary_path.exists() and not args.force_eval:
        try:
            summary = json.loads(summary_path.read_text())
            if (
                Path(summary["checkpoint"]).resolve() == checkpoint.resolve()
                and summary["total_rollouts"] == args.eval_rollouts * len(args.eval_seeds)
            ):
                print(f"[reuse evaluation] {summary_path}", flush=True)
                return summary
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    command = [
        str(PYTHON),
        "-B",
        "scripts/validate_epoch50_platform.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(args.rollout_dataset),
        "--output-dir",
        str(output_dir),
        "--seeds",
        *[str(seed) for seed in args.eval_seeds],
        "--n-rollouts",
        str(args.eval_rollouts),
        "--chunk-size",
        str(args.eval_chunk_size),
        "--horizon",
        str(args.eval_horizon),
        "--max-retries",
        str(args.eval_max_retries),
        "--evaluate-only",
    ]
    if args.force_eval:
        command.append("--force")
    run(command, cache_suffix="eval")
    return json.loads(summary_path.read_text())


def write_experiment_summary(args, dataset_summary: dict, checkpoint: Path | None, evaluation: dict | None) -> None:
    output = ROOT / f"rollouts/square_rgb_dp/{method_tag(args)}_experiment_summary.json"
    result = {
        "method": method_name(args),
        "variant": args.variant,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "config": str(config_path(args)),
        "low_risk_dataset": None if is_success_only(args) else str(args.low_risk_dataset),
        "low_risk_dataset_summary": (
            None
            if is_success_only(args)
            else str(args.low_risk_dataset.with_suffix(".summary.json"))
        ),
        "dataset_summary": {
            key: dataset_summary.get(key)
            for key in (
                "source_failure_rollouts",
                "retained_chunks",
                "retained_source_rollouts",
                "selection",
                "risk_stats",
                "privileged_progress_stats",
            )
        },
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "evaluation_summary": (
            str(evaluation_root(args) / "stability_summary.json")
            if evaluation is not None
            else None
        ),
        "evaluation": evaluation,
    }
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("build", "audit", "config", "train", "eval"),
        default=("build", "audit", "config"),
    )
    parser.add_argument(
        "--variant",
        choices=("lowrisk", "progress_lowrisk", "success_only"),
        default="lowrisk",
        help=(
            "success_only uses demos + successful policy rollouts only; "
            "lowrisk adds selected low-risk failure chunks; "
            "progress_lowrisk additionally requires privileged task progress."
        ),
    )
    parser.add_argument("--demo-dataset", type=Path, default=RGB_DEMOS)
    parser.add_argument("--rollout-dataset", type=Path, default=RGB_ROLLOUTS)
    parser.add_argument("--low-risk-dataset", type=Path, default=None)
    parser.add_argument(
        "--risk-predictions",
        type=Path,
        default=(
            ROOT
            / "trained_models/square_rgb_dp_causal_prefix_risk/epoch190"
            / "prefix_predictions.npz"
        ),
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--demo-weight", type=float, default=1.0)
    parser.add_argument(
        "--demo-filter-key",
        type=str,
        default=None,
        help="Optional mask in the original demo dataset, e.g. 50_percent_train.",
    )
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hdf5-cache-mode", choices=("all", "low_dim", "none"), default="low_dim")
    parser.add_argument("--save-every-epochs", type=int, default=25)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--max-chunks-per-failure", type=int, default=2)
    parser.add_argument("--minimum-spacing", type=int, default=16)
    parser.add_argument(
        "--prefer",
        choices=("lowest", "latest", "earliest", "progress_aware"),
        default=None,
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--success-quantile", type=float, default=0.95)
    parser.add_argument("--target-peg-xy", type=float, nargs=2, default=(0.23, 0.10))
    parser.add_argument("--min-peg-xy-progress", type=float, default=0.02)
    parser.add_argument("--min-nut-displacement", type=float, default=0.02)
    parser.add_argument("--min-nut-z-gain", type=float, default=-0.01)
    parser.add_argument("--peg-progress-weight", type=float, default=2.0)
    parser.add_argument("--z-gain-weight", type=float, default=1.0)
    parser.add_argument("--nut-displacement-weight", type=float, default=0.25)
    parser.add_argument("--eef-approach-weight", type=float, default=0.0)
    parser.add_argument("--progress-risk-penalty", type=float, default=0.5)
    parser.add_argument("--use-stored-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--max-train-restarts", type=int, default=20)
    parser.add_argument("--eval-rollouts", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-chunk-size", type=int, default=10)
    parser.add_argument("--eval-horizon", type=int, default=400)
    parser.add_argument("--eval-max-retries", type=int, default=5)
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    if args.prefer is None:
        args.prefer = "progress_aware" if args.variant == "progress_lowrisk" else "lowest"
    if args.low_risk_dataset is None:
        args.low_risk_dataset = (
            PROGRESS_LOW_RISK_FAILURE
            if args.variant == "progress_lowrisk"
            else LOW_RISK_FAILURE
        )
    for key in (
        "demo_dataset",
        "rollout_dataset",
        "low_risk_dataset",
        "risk_predictions",
        "baseline_checkpoint",
    ):
        setattr(args, key, getattr(args, key).resolve())

    stages = set(args.stages)
    if "build" in stages:
        build_dataset(args)
    dataset_summary = audit_dataset(args)
    if "config" in stages:
        write_config(args)

    checkpoint = completed_checkpoint(args)
    if "train" in stages:
        checkpoint = train(args)
    evaluation = None
    if "eval" in stages:
        if checkpoint is None:
            raise FileNotFoundError("no completed checkpoint; run train first")
        evaluation = evaluate(args, checkpoint)
    write_experiment_summary(args, dataset_summary, checkpoint, evaluation)


if __name__ == "__main__":
    main()
