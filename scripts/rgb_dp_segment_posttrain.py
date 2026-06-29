#!/usr/bin/env python3
"""Prepare matched RGB Diffusion Policy baseline and post-training configs.

The policy observes two RGB cameras plus robot proprioception. Simulator object
state is retained in the datasets only for offline labeling and is never listed
as a policy input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "robomimic/exps/templates/diffusion_policy.json"
CONFIG_DIR = ROOT / "robomimic/exps/templates/rgb_dp_segment_posttrain"
OUTPUT_DIR = ROOT / "trained_models/rgb_dp_segment_posttrain"

RGB_DEMOS = ROOT / "datasets/lift/ph/image_v15_rgb2.hdf5"
RGB_ROLLOUTS = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
RGB_FILTERED_FAILURE = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_failure_segments_stage_filtered.hdf5"
)
RGB_CRITICAL_FAILURE = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_critical_failure_chunks.hdf5"
)
RGB_FIXED_GOOD_FAILURE = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_good_chunks_fixed_window.hdf5"
)
RGB_PREFIX_RISK_GOOD_FAILURE = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_prefix_risk_good_chunks.hdf5"
)
RGB_HIGH_RISK_CONSTRAINT = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_high_risk_constraint_chunks.hdf5"
)
RGB_SUCCESS_HAZARD_CONTEXTS = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_success_hazard_context_chunks.hdf5"
)
RGB_PREFIX_RISK_CHECKPOINT = (
    ROOT / "trained_models/rgb_dp_causal_prefix_risk/best.pt"
)

RGB_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def weight_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


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


def make_config(
    *,
    name: str,
    data: list[dict],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seed: int,
    checkpoint: Path | None,
    learning_rate: float,
    reference_margin: dict | None = None,
    hazard_constraint: dict | None = None,
    dataset_key_shapes: dict | None = None,
) -> dict:
    config = json.loads(TEMPLATE.read_text())
    config["experiment"]["name"] = name
    config["experiment"]["render"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["keep_all_videos"] = False
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["save"]["every_n_epochs"] = 25
    config["experiment"]["save"]["epochs"] = sorted(
        set([min(25, epochs), min(50, epochs), min(100, epochs), epochs])
    )
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["epoch_every_n_steps"] = steps_per_epoch
    config["experiment"]["ckpt_path"] = (
        str(checkpoint.resolve()) if checkpoint is not None else None
    )

    config["train"]["data"] = data
    config["train"]["output_dir"] = str(OUTPUT_DIR)
    config["train"]["normalize_weights_by_ds_size"] = len(data) > 1
    config["train"]["num_data_workers"] = 0
    config["train"]["hdf5_cache_mode"] = "low_dim"
    config["train"]["hdf5_load_next_obs"] = False
    config["train"]["seq_length"] = 16
    config["train"]["frame_stack"] = 2
    config["train"]["batch_size"] = batch_size
    config["train"]["num_epochs"] = epochs
    config["train"]["seed"] = seed
    if dataset_key_shapes:
        dataset_keys = list(config["train"]["dataset_keys"])
        for key in dataset_key_shapes:
            if key not in dataset_keys:
                dataset_keys.append(key)
        config["train"]["dataset_keys"] = dataset_keys
        config["train"]["dataset_key_shapes"] = dataset_key_shapes

    config["algo"]["horizon"] = {
        "observation_horizon": 2,
        "action_horizon": 8,
        "prediction_horizon": 16,
    }
    config["algo"]["unet"] = {
        "enabled": True,
        "diffusion_step_embed_dim": 128,
        "down_dims": [128, 256, 512],
        "kernel_size": 5,
        "n_groups": 8,
    }
    config["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] = learning_rate
    config["algo"]["optim_params"]["policy"]["learning_rate"]["warmup_steps"] = min(
        500, max(50, epochs * steps_per_epoch // 10)
    )
    # DDIM preserves the same diffusion training objective but makes rollout
    # evaluation and deployment substantially faster than 100-step DDPM.
    config["algo"]["ddpm"]["enabled"] = False
    config["algo"]["ddim"]["enabled"] = True
    config["algo"]["ddim"]["num_train_timesteps"] = 100
    config["algo"]["ddim"]["num_inference_timesteps"] = 10
    if reference_margin is not None:
        config["algo"]["reference_margin"] = reference_margin
    if hazard_constraint is not None:
        config["algo"]["hazard_constraint"] = hazard_constraint

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


def write_configs(args) -> dict[str, Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    baseline = make_config(
        name=f"lift_rgb2_dp_baseline_s{args.seed}",
        data=[{"path": str(RGB_DEMOS)}],
        epochs=args.baseline_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint=None,
        learning_rate=args.baseline_lr,
    )
    paths = {"baseline": CONFIG_DIR / "baseline.json"}
    paths["baseline"].write_text(json.dumps(baseline, indent=4))

    if args.baseline_checkpoint is not None:
        success_data = [
            {"path": str(RGB_DEMOS), "weight": 1.0},
            {"path": str(RGB_ROLLOUTS), "filter_key": "success", "weight": 1.0},
        ]
        success_control = make_config(
            name=f"lift_rgb2_dp_success_posttrain_s{args.seed}",
            data=success_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
        )
        paths["posttrain_success"] = CONFIG_DIR / "posttrain_success.json"
        paths["posttrain_success"].write_text(json.dumps(success_control, indent=4))

        posttrain_data = [
            *success_data,
            {"path": str(RGB_FILTERED_FAILURE), "weight": args.failure_weight},
        ]
        posttrain = make_config(
            name=f"lift_rgb2_dp_stage_filtered_posttrain_s{args.seed}",
            data=posttrain_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
        )
        paths["posttrain"] = CONFIG_DIR / "posttrain_filtered.json"
        paths["posttrain"].write_text(json.dumps(posttrain, indent=4))

        fixed_good_data = [
            *success_data,
            {
                "path": str(RGB_FIXED_GOOD_FAILURE),
                "weight": args.failure_weight,
                "demo_start_only": True,
                "sample_start_offset": 1,
            },
        ]
        fixed_good = make_config(
            name=f"lift_rgb2_dp_fixed_good_posttrain_s{args.seed}",
            data=fixed_good_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
        )
        paths["posttrain_fixed_good"] = CONFIG_DIR / "posttrain_fixed_good.json"
        paths["posttrain_fixed_good"].write_text(json.dumps(fixed_good, indent=4))

        prefix_risk_weight_tag = weight_tag(args.prefix_risk_weight)
        prefix_risk_data = [
            *success_data,
            {
                "path": str(RGB_PREFIX_RISK_GOOD_FAILURE),
                "weight": args.prefix_risk_weight,
                "demo_start_only": True,
                "sample_start_offset": 1,
            },
        ]
        prefix_risk_good = make_config(
            name=(
                "lift_rgb2_dp_prefix_risk_good_"
                f"w{prefix_risk_weight_tag}_posttrain_s{args.seed}"
            ),
            data=prefix_risk_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
        )
        paths["posttrain_prefix_risk_good"] = (
            CONFIG_DIR
            / f"posttrain_prefix_risk_good_w{prefix_risk_weight_tag}.json"
        )
        paths["posttrain_prefix_risk_good"].write_text(
            json.dumps(prefix_risk_good, indent=4)
        )

        anti_failure_data = [
            {"path": str(RGB_DEMOS), "weight": 0.5},
            {
                "path": str(RGB_ROLLOUTS),
                "filter_key": "success",
                "weight": 0.4,
            },
            {
                "path": str(RGB_CRITICAL_FAILURE),
                "weight": args.negative_sampling_weight,
                "anti_failure": True,
                "demo_start_only": True,
                "sample_start_offset": 1,
            },
        ]
        reference_control = make_config(
            name=f"lift_rgb2_dp_reference_control_posttrain_s{args.seed}",
            data=anti_failure_data[:2],
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
            reference_margin={
                "enabled": True,
                "margin": args.margin,
                "failure_weight": 0.0,
                "reference_weight": args.reference_weight,
                "warmup_steps": args.anti_warmup_steps,
                "ramp_steps": args.anti_ramp_steps,
                "normalized_margin": True,
                "initialize_from_ema": True,
            },
        )
        paths["posttrain_reference_control"] = (
            CONFIG_DIR / "posttrain_reference_control.json"
        )
        paths["posttrain_reference_control"].write_text(
            json.dumps(reference_control, indent=4)
        )

        anti_failure = make_config(
            name=f"lift_rgb2_dp_reference_margin_posttrain_s{args.seed}",
            data=anti_failure_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
            reference_margin={
                "enabled": True,
                "margin": args.margin,
                "failure_weight": args.anti_failure_weight,
                "reference_weight": args.reference_weight,
                "warmup_steps": args.anti_warmup_steps,
                "ramp_steps": args.anti_ramp_steps,
                "normalized_margin": True,
                "initialize_from_ema": True,
            },
        )
        paths["posttrain_reference_margin"] = (
            CONFIG_DIR / "posttrain_reference_margin.json"
        )
        paths["posttrain_reference_margin"].write_text(
            json.dumps(anti_failure, indent=4)
        )

        hazard_tag = (
            f"dw{weight_tag(args.hazard_data_weight)}"
            f"_lw{weight_tag(args.hazard_loss_weight)}"
            f"_m{weight_tag(args.hazard_margin)}"
            f"_k{args.hazard_action_samples}"
        )
        hazard_config = {
            "enabled": True,
            "checkpoint": str(args.hazard_checkpoint.resolve()),
            "margin": args.hazard_margin,
            "weight": args.hazard_loss_weight,
            "positive_reference_weight": args.hazard_positive_reference_weight,
            "negative_reference_weight": args.hazard_negative_reference_weight,
            "warmup_steps": args.hazard_warmup_steps,
            "ramp_steps": args.hazard_ramp_steps,
            "sampling_steps": args.hazard_sampling_steps,
            "action_samples": args.hazard_action_samples,
            "action_start_index": 1,
            "initialize_from_ema": True,
        }
        hazard_control = make_config(
            name=f"lift_rgb2_dp_hazard_control_{hazard_tag}_s{args.seed}",
            data=success_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
            hazard_constraint={**hazard_config, "weight": 0.0},
            dataset_key_shapes={"hazard_context": args.hazard_context_dim},
        )
        paths["posttrain_hazard_control"] = (
            CONFIG_DIR / f"posttrain_hazard_control_{hazard_tag}.json"
        )
        paths["posttrain_hazard_control"].write_text(
            json.dumps(hazard_control, indent=4)
        )

        success_hazard_data = [
            *success_data,
            {
                "path": str(RGB_SUCCESS_HAZARD_CONTEXTS),
                "weight": args.hazard_success_data_weight,
                "hazard_failure": True,
                "demo_start_only": True,
                "sample_start_offset": 1,
            },
        ]
        success_hazard = make_config(
            name=(
                "lift_rgb2_dp_success_hazard_regularized_"
                f"{hazard_tag}_s{args.seed}"
            ),
            data=success_hazard_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
            hazard_constraint=hazard_config,
            dataset_key_shapes={"hazard_context": args.hazard_context_dim},
        )
        paths["posttrain_success_hazard_regularized"] = (
            CONFIG_DIR / f"posttrain_success_hazard_regularized_{hazard_tag}.json"
        )
        paths["posttrain_success_hazard_regularized"].write_text(
            json.dumps(success_hazard, indent=4)
        )

        hazard_data = [
            *success_data,
            {
                "path": str(RGB_HIGH_RISK_CONSTRAINT),
                "weight": args.hazard_data_weight,
                "hazard_failure": True,
                "demo_start_only": True,
                "sample_start_offset": 1,
            },
        ]
        hazard_posttrain = make_config(
            name=f"lift_rgb2_dp_hazard_constraint_{hazard_tag}_s{args.seed}",
            data=hazard_data,
            epochs=args.posttrain_epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint=args.baseline_checkpoint,
            learning_rate=args.posttrain_lr,
            hazard_constraint=hazard_config,
            dataset_key_shapes={"hazard_context": args.hazard_context_dim},
        )
        paths["posttrain_hazard_constraint"] = (
            CONFIG_DIR / f"posttrain_hazard_constraint_{hazard_tag}.json"
        )
        paths["posttrain_hazard_constraint"].write_text(
            json.dumps(hazard_posttrain, indent=4)
        )

    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-epochs", type=int, default=200)
    parser.add_argument("--posttrain-epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline-lr", type=float, default=1e-4)
    parser.add_argument("--posttrain-lr", type=float, default=5e-5)
    parser.add_argument("--failure-weight", type=float, default=0.25)
    parser.add_argument("--prefix-risk-weight", type=float, default=0.05)
    parser.add_argument("--negative-sampling-weight", type=float, default=0.1)
    parser.add_argument("--anti-failure-weight", type=float, default=0.02)
    parser.add_argument("--reference-weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--anti-warmup-steps", type=int, default=250)
    parser.add_argument("--anti-ramp-steps", type=int, default=500)
    parser.add_argument(
        "--hazard-checkpoint",
        type=Path,
        default=RGB_PREFIX_RISK_CHECKPOINT,
    )
    parser.add_argument("--hazard-context-dim", type=int, default=256)
    parser.add_argument("--hazard-data-weight", type=float, default=0.1)
    parser.add_argument("--hazard-success-data-weight", type=float, default=0.1)
    parser.add_argument("--hazard-loss-weight", type=float, default=0.05)
    parser.add_argument("--hazard-margin", type=float, default=0.1)
    parser.add_argument(
        "--hazard-positive-reference-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--hazard-negative-reference-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--hazard-warmup-steps", type=int, default=250)
    parser.add_argument("--hazard-ramp-steps", type=int, default=500)
    parser.add_argument("--hazard-sampling-steps", type=int, default=10)
    parser.add_argument("--hazard-action-samples", type=int, default=2)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    write_configs(args)


if __name__ == "__main__":
    main()
