#!/usr/bin/env python3
"""Actor-only post-training for RGB DiffusionPolicy checkpoints.

This entrypoint initializes from a deployed RGB-DP checkpoint and continues
training the full diffusion policy, including its image encoder, on selected
human-demo, successful-rollout, and failure-rollout datasets. It intentionally
trains no IQL critic.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils

from train_square_rgb_dp_chunk_actor_iql import (
    actor_train_step,
    actor_trainability_summary,
    build_actor_loader,
    configure_actor_optimizer,
    cycle,
    initialize_actor_from_deployed_ema,
    jsonable,
    write_json,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth"
)
DEFAULT_DEMOS = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_ROLLOUTS = ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
DEFAULT_OUTPUT = ROOT / "trained_models/square_rgb_dp_self_imitation/200demo_100success"


def numeric_mean(logs: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for item in logs for key in item.keys()})
    result = {}
    for key in keys:
        values = []
        for item in logs:
            value = item.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            result[key] = float(np.mean(values))
    return result


def add_scalars(writer, prefix: str, values: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def save_policy_checkpoint(
    *,
    path: Path,
    actor_algo,
    actor_config,
    dp_ckpt: dict,
    epoch: int,
    global_step: int,
    history: list[dict],
    mode_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variable_state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_valid_loss": None,
        "best_return": None,
        "best_success_rate": None,
        "self_imitation": True,
        "posttrain_mode": str(mode_name),
        "success_conditioned": str(mode_name) == "success_conditioned_mixed_quality_imitation_learning",
    }
    TrainUtils.save_model(
        model=actor_algo,
        config=actor_config,
        env_meta=dp_ckpt["env_metadata"],
        shape_meta=dp_ckpt["shape_metadata"],
        ckpt_path=str(path),
        variable_state=variable_state,
        action_normalization_stats=dp_ckpt.get("action_normalization_stats"),
    )


def infer_mode_name(args: argparse.Namespace) -> str:
    if args.mode_name:
        return str(args.mode_name)
    demo = float(args.actor_demo_weight) > 0.0
    success = float(args.actor_success_weight) > 0.0
    failure = float(args.actor_failure_weight) > 0.0
    if bool(args.conditioned_mixed_imitation):
        return "success_conditioned_mixed_quality_imitation_learning"
    if demo and success and failure:
        return "mixed_quality_imitation_learning"
    if demo and success and not failure:
        return "self_imitation_learning"
    return "actor_only_dp_posttraining"


def infer_description(args: argparse.Namespace) -> str:
    if args.description:
        return str(args.description)
    mode_name = infer_mode_name(args)
    if mode_name == "success_conditioned_mixed_quality_imitation_learning":
        return (
            "success-conditioned actor-only post-deployment DP training on human demos, "
            "successful rollouts, and failure rollouts"
        )
    if mode_name == "gt_good_failure_mixed_imitation_learning":
        return (
            "actor-only post-deployment DP training on human demos, successful rollouts, "
            "and privileged-GT good failure chunks"
        )
    if mode_name == "mixed_quality_imitation_learning":
        return "actor-only post-deployment DP training on human demos, successful rollouts, and failure rollouts"
    if mode_name == "self_imitation_learning":
        return "actor-only post-deployment DP training on human demos and successful rollouts"
    return "actor-only post-deployment DP training"


def make_summary(
    *,
    args: argparse.Namespace,
    actor_dataset,
    actor_config,
    trainability: dict,
    history: list[dict],
    last_checkpoint: Path | None,
) -> dict:
    mode_name = infer_mode_name(args)
    return {
        "mode": mode_name,
        "description": infer_description(args),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "epochs": int(args.epochs),
        "steps_per_epoch": int(args.steps_per_epoch),
        "actor_dataset_size": int(len(actor_dataset)),
        "actor_action_normalization_source": getattr(
            actor_dataset,
            "actor_action_normalization_source",
            None,
        ),
        "actor_initialized_from_deployed_ema": bool(
            getattr(args, "actor_initialized_from_deployed_ema", False)
        ),
        "actor_data": jsonable(actor_config.train.data),
        "actor_training_scope": "full_pretrained_dp_policy_including_obs_encoder",
        "actor_trainability": jsonable(trainability),
        "critic_trained": False,
        "loaded_pretrained_dp": True,
        "success_conditioning": {
            "enabled": bool(args.conditioned_mixed_imitation),
            "positive_sources": ["demo", "success"],
            "negative_sources": ["failure"],
            "source_conditions": {
                "human_demo": 1.0,
                "success_rollout": 1.0,
                "failure_rollout": 0.0,
            },
            "source_condition_masks": {
                "human_demo": 1.0,
                "success_rollout": 1.0,
                "failure_rollout": 1.0,
            },
            "inference_condition": 1.0,
            "dropout": float(args.condition_dropout),
            "dropout_semantics": "condition=0, mask=0 null token; real failure is condition=0, mask=1",
            "hidden_dim": int(args.condition_hidden_dim),
        },
        "last_checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
        "history": history,
    }


def train(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.conditioned_mixed_imitation):
        failure_label = float(args.actor_failure_anti_failure_label)
        if float(args.actor_failure_weight) > 0.0 and failure_label != 1.0:
            raise ValueError(
                "conditioned mixed imitation requires failure rollouts to use "
                f"actor_failure_anti_failure_label=1.0, got {failure_label}"
            )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")

    actor_policy, dp_ckpt = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
        verbose=False,
    )
    actor_policy.start_episode()
    actor_algo = actor_policy.policy
    args.actor_initialized_from_deployed_ema = False
    if args.resume_checkpoint is None:
        args.actor_initialized_from_deployed_ema = initialize_actor_from_deployed_ema(actor_algo)
    if not 0.0 <= float(args.condition_dropout) < 1.0:
        raise ValueError(f"condition_dropout must be in [0, 1), got {args.condition_dropout}")
    if bool(args.conditioned_mixed_imitation):
        if not hasattr(actor_algo, "install_success_condition_adapter"):
            raise RuntimeError("loaded policy does not support success-conditioned DP adapters")
        actor_algo.install_success_condition_adapter(hidden_dim=int(args.condition_hidden_dim))
        actor_algo.success_condition_dropout = float(args.condition_dropout)
    configure_actor_optimizer(actor_algo, args.actor_lr, args.actor_disable_lr_scheduler)

    actor_dataset, actor_loader, actor_config = build_actor_loader(
        args=args,
        actor_algo=actor_algo,
        checkpoint_dict=dp_ckpt,
    )
    with actor_config.values_unlocked():
        actor_config.experiment.name = args.experiment_name
        actor_config.experiment.epoch_every_n_steps = int(args.steps_per_epoch)
        actor_config.train.output_dir = str(args.output_dir)
        actor_config.train.num_epochs = int(args.epochs)
        actor_config.train.seed = int(args.seed)

    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if "model" not in ckpt:
            raise ValueError(f"resume checkpoint is not a robomimic policy checkpoint: {args.resume_checkpoint}")
        actor_algo.deserialize(ckpt["model"], load_optimizers=True)
        configure_actor_optimizer(actor_algo, args.actor_lr, args.actor_disable_lr_scheduler)
        variable_state = ckpt.get("variable_state", {}) or {}
        start_epoch = int(variable_state.get("epoch", 0))
        global_step = int(variable_state.get("global_step", start_epoch * int(args.steps_per_epoch)))
        summary_path = args.output_dir / "partial_summary.json"
        if summary_path.exists():
            try:
                history = list(json.loads(summary_path.read_text()).get("history", []))
            except Exception:
                history = []

    trainability = actor_trainability_summary(actor_algo)
    print(json.dumps({"actor_trainability": jsonable(trainability)}, indent=2), flush=True)
    print(
        json.dumps(
            {
                "mode": infer_mode_name(args),
                "actor_dataset_size": int(len(actor_dataset)),
                "actor_action_normalization_source": getattr(
                    actor_dataset,
                    "actor_action_normalization_source",
                    None,
                ),
                "actor_initialized_from_deployed_ema": bool(args.actor_initialized_from_deployed_ema),
                "actor_data": jsonable(actor_config.train.data),
                "success_conditioning": {
                    "enabled": bool(args.conditioned_mixed_imitation),
                    "source_conditions": {
                        "human_demo": 1.0,
                        "success_rollout": 1.0,
                        "failure_rollout": 0.0,
                    },
                    "source_condition_masks": {
                        "human_demo": 1.0,
                        "success_rollout": 1.0,
                        "failure_rollout": 1.0,
                    },
                    "dropout": float(args.condition_dropout),
                    "hidden_dim": int(args.condition_hidden_dim),
                },
                "resume_epoch": int(start_epoch),
            },
            indent=2,
        ),
        flush=True,
    )

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("TensorBoard requested but torch.utils.tensorboard is unavailable.", flush=True)
        else:
            log_dir = args.tensorboard_dir or (args.output_dir / "tensorboard")
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))

    actor_iterator = cycle(actor_loader)
    last_checkpoint: Path | None = args.resume_checkpoint
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        epoch_logs = []
        for _ in range(int(args.steps_per_epoch)):
            global_step += 1
            actor_batch = next(actor_iterator)
            actor_log = actor_train_step(
                actor_policy,
                actor_batch,
                global_step,
                obs_normalization_stats=actor_policy.obs_normalization_stats,
            )
            epoch_logs.append(actor_log)
            if global_step % int(args.log_every) == 0:
                payload = {"epoch": int(epoch), "global_step": int(global_step)}
                payload.update(actor_log)
                print(json.dumps(jsonable(payload), sort_keys=True), flush=True)

        metrics = numeric_mean(epoch_logs)
        record = {"epoch": int(epoch), "global_step": int(global_step), "train": metrics}
        history.append(record)
        add_scalars(writer, "train", metrics, global_step)
        if writer is not None:
            writer.flush()

        partial = make_summary(
            args=args,
            actor_dataset=actor_dataset,
            actor_config=actor_config,
            trainability=trainability,
            history=history,
            last_checkpoint=last_checkpoint,
        )
        write_json(args.output_dir / "partial_summary.json", partial)

        saved_epoch_checkpoint = None
        should_save_epoch = bool(args.save_checkpoints) and (
            epoch == int(args.epochs)
            or (int(args.save_every_epochs) > 0 and epoch % int(args.save_every_epochs) == 0)
        )
        if should_save_epoch:
            saved_epoch_checkpoint = models_dir / f"model_epoch_{epoch}.pth"
            save_policy_checkpoint(
                path=saved_epoch_checkpoint,
                actor_algo=actor_algo,
                actor_config=actor_config,
                dp_ckpt=dp_ckpt,
                epoch=epoch,
                global_step=global_step,
                history=history,
                mode_name=infer_mode_name(args),
            )

        should_save_latest = bool(args.save_checkpoints) and (
            epoch == int(args.epochs)
            or (int(args.save_latest_every_epochs) > 0 and epoch % int(args.save_latest_every_epochs) == 0)
        )
        if should_save_latest:
            latest = args.output_dir / "last.pth"
            if saved_epoch_checkpoint is not None:
                shutil.copyfile(saved_epoch_checkpoint, latest)
                print(f"copied checkpoint to {latest}", flush=True)
            else:
                save_policy_checkpoint(
                    path=latest,
                    actor_algo=actor_algo,
                    actor_config=actor_config,
                    dp_ckpt=dp_ckpt,
                    epoch=epoch,
                    global_step=global_step,
                    history=history,
                    mode_name=infer_mode_name(args),
                )
            shutil.copyfile(latest, args.output_dir / "last_bak.pth")
            last_checkpoint = latest

    if writer is not None:
        writer.close()
    close_fn = getattr(actor_dataset, "close", None)
    if callable(close_fn):
        close_fn()

    summary = make_summary(
        args=args,
        actor_dataset=actor_dataset,
        actor_config=actor_config,
        trainability=trainability,
        history=history,
        last_checkpoint=last_checkpoint,
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(jsonable({k: v for k, v in summary.items() if k != "history"}), indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--demo-dataset", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--success-dataset", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--failure-dataset", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default="rgb_dp_self_imitation")
    parser.add_argument("--mode-name", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--actor-batch-size", type=int, default=100)
    parser.add_argument("--actor-num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-seq-length", type=int, default=None)
    parser.add_argument("--actor-hdf5-cache-mode", type=str, default="low_dim")
    parser.add_argument("--demo-filter-key", type=str, default="")
    parser.add_argument("--success-filter-key", type=str, default="success_100")
    parser.add_argument("--failure-filter-key", type=str, default="failure")
    parser.add_argument("--actor-demo-weight", type=float, default=1.0)
    parser.add_argument("--actor-success-weight", type=float, default=1.0)
    parser.add_argument("--actor-failure-weight", type=float, default=0.0)
    parser.add_argument("--actor-failure-demo-start-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-failure-sample-start-offset", type=int, default=0)
    parser.add_argument("--actor-failure-anti-failure-label", type=float, default=1.0)
    parser.add_argument("--conditioned-mixed-imitation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument("--actor-normalize-weights-by-ds-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--actor-disable-lr-scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epochs", type=int, default=10)
    parser.add_argument("--save-latest-every-epochs", type=int, default=10)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    args = parser.parse_args()

    for key in ("checkpoint", "demo_dataset", "success_dataset", "failure_dataset", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.tensorboard_dir is not None:
        args.tensorboard_dir = args.tensorboard_dir.resolve()
    train(args)


if __name__ == "__main__":
    main()
