#!/usr/bin/env python3
"""Run frozen-hazard constrained RGB Diffusion Policy post-training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from robomimic.models.prefix_risk_nets import CausalPrefixRisk


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
    / "lift_rgb_dp_high_risk_constraint_chunks.hdf5"
)
HAZARD_CHECKPOINT = ROOT / "trained_models/rgb_dp_causal_prefix_risk/best.pt"
CONFIG_ROOT = ROOT / "robomimic/exps/templates/rgb_dp_segment_posttrain"
MODEL_ROOT = ROOT / "trained_models/rgb_dp_segment_posttrain"

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPYCACHEPREFIX": "/tmp/robomimic_hazard_wrapper_pycache",
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
    kind = "hazard_constraint" if args.variant == "constraint" else "hazard_control"
    return {
        "config": CONFIG_ROOT / f"posttrain_{kind}_{tag}.json",
        "experiment": MODEL_ROOT / f"lift_rgb2_dp_{kind}_{tag}_s{args.seed}",
        "evaluation": ROOT / f"rollouts/rgb_dp/{kind}_{tag}_eval",
        "restarts": ROOT / f"rollouts/rgb_dp/{kind}_{tag}_train_restarts",
        "diagnostic": (
            ROOT / f"rollouts/rgb_dp/{kind}_{tag}_reference_diagnostic.json"
        ),
        "summary": ROOT / f"rollouts/rgb_dp/{kind}_{tag}_experiment.json",
    }


def build_dataset(overwrite: bool) -> None:
    if FILTERED.exists() and not overwrite:
        print(f"[reuse dataset] {FILTERED}", flush=True)
        return
    command = [
        str(PYTHON),
        "-B",
        "scripts/build_rgb_dp_high_risk_constraints.py",
    ]
    if overwrite:
        command.append("--overwrite")
    run(command)


def load_hazard():
    checkpoint = torch.load(
        HAZARD_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    train_args = checkpoint["args"]
    model = CausalPrefixRisk(
        feature_dim=int(checkpoint["feature_dim"]),
        prediction_horizon=int(checkpoint["prediction_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(train_args["hidden_dim"]),
        action_hidden_dim=int(train_args["action_hidden_dim"]),
        dropout=float(train_args["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def audit_dataset() -> dict:
    summary = json.loads(FILTERED.with_suffix(".summary.json").read_text())
    records = summary["chunks"]
    if not summary["quality_gate"]["passed"]:
        raise RuntimeError("hazard-constraint data failed its quality gate")
    if summary["quality_gate_overridden"]:
        raise RuntimeError("refusing a dataset built by overriding a failed gate")
    if not records:
        raise RuntimeError("hazard filter retained no chunks")
    model, checkpoint = load_hazard()
    action_mean = checkpoint["stats"]["action_mean"]
    action_std = checkpoint["stats"]["action_std"]

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
                raise RuntimeError("constraint demo does not contain 17 frames")
            context = output["hazard_context"][:]
            if context.shape != (17, summary["hazard_context_dim"]):
                raise RuntimeError("invalid hazard_context shape")
            if not np.allclose(context, context[:1]):
                raise RuntimeError("hazard context must be constant within a chunk")
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

            future_actions = original["actions"][
                boundary : boundary + 16
            ].astype(np.float32)
            normalized = (future_actions - action_mean) / action_std
            delta = model.action_delta(
                torch.from_numpy(context[:1, None]),
                torch.from_numpy(normalized[None, None]),
            ).item()
            if not np.isclose(delta, record["action_delta"], atol=1e-4, rtol=1e-4):
                raise RuntimeError("stored context does not reproduce hazard score")

    print(
        "[audit passed] "
        f"{summary['retained_chunks']} high-risk transitions from "
        f"{summary['retained_source_rollouts']} failed rollouts; "
        f"privileged overlap="
        f"{summary['privileged_overlap_audit']['target_overlaps_any_critical_window']}",
        flush=True,
    )
    return summary


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
    hazard = config["algo"]["hazard_constraint"]
    if not hazard["enabled"]:
        raise RuntimeError("generated config did not enable hazard constraint")
    expected_weight = args.hazard_loss_weight if args.variant == "constraint" else 0.0
    if float(hazard["weight"]) != expected_weight:
        raise RuntimeError("generated config contains incorrect hazard weight")
    if config["train"]["dataset_key_shapes"]["hazard_context"] != 256:
        raise RuntimeError("generated config contains incorrect context dimension")
    if args.variant == "constraint":
        negatives = [
            entry
            for entry in config["train"]["data"]
            if entry.get("hazard_failure")
        ]
        if len(negatives) != 1 or negatives[0]["path"] != str(FILTERED):
            raise RuntimeError("generated config does not contain hazard data")
    return config_path


@torch.no_grad()
def diagnose_reference_policy(args) -> dict | None:
    if args.variant != "constraint":
        print("[diagnostic skipped] control has no failure constraints", flush=True)
        return None

    from torch.utils.data._utils.collate import default_collate

    from robomimic.algo import algo_factory
    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.train_utils as TrainUtils
    import robomimic.utils.torch_utils as TorchUtils

    raw = json.loads(paths(args)["config"].read_text())
    optim = raw["algo"]["optim_params"]["policy"]
    optim["num_train_batches"] = 1
    optim["num_epochs"] = 1
    config = config_factory("diffusion_policy", dic=raw)
    ObsUtils.initialize_obs_utils_with_config(config)
    shape = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=config.train.data[0],
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=False,
    )
    trainset, _ = TrainUtils.load_data_for_training(
        config,
        obs_keys=shape["all_obs_keys"],
    )
    hazard_dataset = trainset.datasets[2]
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model = algo_factory(
        config.algo_name,
        config,
        shape["all_shapes"],
        shape["ac_dim"],
        device,
    )
    baseline = FileUtils.load_dict_from_checkpoint(
        config.experiment.ckpt_path
    )
    model.deserialize(baseline["model"])
    model.set_eval()
    model.reference_nets.eval()

    all_risks = []
    for sample_seed in range(args.diagnostic_samples):
        torch.manual_seed(args.diagnostic_seed + sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.diagnostic_seed + sample_seed)
        for start in range(0, len(hazard_dataset), args.diagnostic_batch_size):
            samples = [
                hazard_dataset[index]
                for index in range(
                    start,
                    min(start + args.diagnostic_batch_size, len(hazard_dataset)),
                )
            ]
            batch = default_collate(samples)
            batch = model.process_batch_for_training(batch)
            batch = model.postprocess_batch_for_training(
                batch,
                obs_normalization_stats=None,
            )
            inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
            obs_cond = model._encode_obs(inputs, model.reference_nets)
            initial_noise = torch.randn(
                (
                    len(samples),
                    config.algo.horizon.prediction_horizon,
                    shape["ac_dim"],
                ),
                device=device,
            )
            trajectory = model._sample_ddim_trajectory(
                model.reference_nets,
                obs_cond,
                initial_noise,
                config.algo.hazard_constraint.sampling_steps,
            )
            delta = model._hazard_action_delta(
                batch["hazard_context"],
                trajectory,
            )
            all_risks.append(torch.relu(delta).cpu().numpy())

    risks = np.concatenate(all_risks).reshape(
        args.diagnostic_samples,
        len(hazard_dataset),
    )
    margin = float(config.algo.hazard_constraint.margin)
    hinge_at_initialization = np.minimum(risks, margin)
    result = {
        "checkpoint": str(args.baseline_checkpoint),
        "hazard_checkpoint": str(args.hazard_checkpoint),
        "num_constraint_states": len(hazard_dataset),
        "num_action_samples_per_state": args.diagnostic_samples,
        "sampling_steps": int(config.algo.hazard_constraint.sampling_steps),
        "margin": margin,
        "reference_positive_risk_fraction": float(np.mean(risks > 0)),
        "states_with_any_positive_reference_risk": float(
            np.mean(np.any(risks > 0, axis=0))
        ),
        "states_with_all_positive_reference_risk": float(
            np.mean(np.all(risks > 0, axis=0))
        ),
        "reference_positive_action_logodds": {
            "mean": float(np.mean(risks)),
            "median": float(np.median(risks)),
            "q90": float(np.quantile(risks, 0.9)),
            "maximum": float(np.max(risks)),
        },
        "initial_hazard_hinge": {
            "mean": float(np.mean(hinge_at_initialization)),
            "active_fraction": float(np.mean(hinge_at_initialization > 0)),
        },
    }
    output = paths(args)["diagnostic"]
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return result


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


def write_summary(args, dataset_summary, checkpoint, evaluation) -> None:
    success_path = ROOT / "rollouts/rgb_dp/posttrain_success_eval/stability_summary.json"
    result = {
        "method": (
            "frozen prefix-risk negative constraint"
            if args.variant == "constraint"
            else "matched positive/reference control"
        ),
        "variant": args.variant,
        "dataset_summary": str(FILTERED.with_suffix(".summary.json")),
        "retained_high_risk_chunks": dataset_summary["retained_chunks"],
        "checkpoint": str(checkpoint),
        "evaluation_summary": str(
            paths(args)["evaluation"] / "stability_summary.json"
        ),
        "total_success": evaluation["total_success"],
        "total_rollouts": evaluation["total_rollouts"],
        "pooled_success_rate": evaluation["pooled_success_rate"],
        "success_only_control": (
            json.loads(success_path.read_text()) if success_path.exists() else None
        ),
    }
    output = paths(args)["summary"]
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("build", "audit", "config", "diagnose", "train", "eval"),
        default=("build", "audit", "config", "diagnose", "train", "eval"),
    )
    parser.add_argument(
        "--variant",
        choices=("constraint", "control"),
        default="constraint",
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE)
    parser.add_argument(
        "--hazard-checkpoint",
        type=Path,
        default=HAZARD_CHECKPOINT,
    )
    parser.add_argument("--hazard-data-weight", type=float, default=0.1)
    parser.add_argument("--hazard-loss-weight", type=float, default=0.05)
    parser.add_argument("--hazard-margin", type=float, default=0.1)
    parser.add_argument("--positive-reference-weight", type=float, default=0.05)
    parser.add_argument("--negative-reference-weight", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--ramp-steps", type=int, default=500)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--action-samples", type=int, default=2)
    parser.add_argument("--diagnostic-samples", type=int, default=3)
    parser.add_argument("--diagnostic-batch-size", type=int, default=16)
    parser.add_argument("--diagnostic-seed", type=int, default=20260629)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-restarts", type=int, default=20)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    args.hazard_checkpoint = args.hazard_checkpoint.resolve()

    stages = set(args.stages)
    if "build" in stages:
        build_dataset(args.overwrite_data)
    dataset_summary = audit_dataset()
    if "config" in stages:
        generate_config(args)
    if "diagnose" in stages:
        diagnose_reference_policy(args)

    checkpoint = completed_checkpoint(args)
    if "train" in stages:
        checkpoint = train(args)
    if "eval" in stages:
        if checkpoint is None:
            raise FileNotFoundError("no completed checkpoint; run the train stage")
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
