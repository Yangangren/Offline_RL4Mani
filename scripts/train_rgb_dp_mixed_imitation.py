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
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils

from rgb_dp_imitation_utils import (
    actor_train_step,
    actor_trainability_summary,
    build_actor_loader,
    configure_actor_optimizer,
    initialize_actor_from_deployed_ema,
    jsonable,
    write_json,
)
from rgb_dp_distributed import (
    DistributedContext,
    all_reduce_gradients,
    broadcast_module_buffers,
    broadcast_module_state,
    initialize_distributed,
    modules_have_mutable_batch_norm,
    seed_process,
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
    distributed_context: DistributedContext,
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
        "distributed_training": {
            "enabled": bool(distributed_context.enabled),
            "world_size": int(distributed_context.world_size),
            "backend": str(distributed_context.backend),
            "rank_zero_writes_only": True,
        },
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


def condition_source_values(args: argparse.Namespace) -> dict[str, float]:
    if str(args.condition_label_mode) == "human_only":
        return {
            "human_demo": 1.0,
            "success_rollout": 0.0,
            "failure_rollout": 0.0,
        }
    return {
        "human_demo": 1.0,
        "success_rollout": 1.0,
        "failure_rollout": 0.0,
    }


def success_condition_summary(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.conditioned_mixed_imitation):
        return {
            "enabled": False,
            "condition_input_used": False,
            "label_mode": None,
            "positive_sources": [],
            "negative_sources": [],
            "source_conditions": None,
            "source_condition_masks": None,
            "inference_condition": None,
            "inference_condition_mask": None,
            "dropout": None,
            "dropout_semantics": None,
            "hidden_dim": None,
        }
    source_conditions = condition_source_values(args)
    return {
        "enabled": True,
        "condition_input_used": True,
        "label_mode": str(args.condition_label_mode),
        "positive_sources": [
            source for source, value in source_conditions.items() if value == 1.0
        ],
        "negative_sources": [
            source for source, value in source_conditions.items() if value == 0.0
        ],
        "source_conditions": source_conditions,
        "source_condition_masks": {
            "human_demo": 1.0,
            "success_rollout": 1.0,
            "failure_rollout": 1.0,
        },
        "inference_condition": 1.0,
        "inference_condition_mask": 1.0,
        "dropout": float(args.condition_dropout),
        "dropout_semantics": "condition=0, mask=0 null token; real failure is condition=0, mask=1",
        "hidden_dim": int(args.condition_hidden_dim),
    }


def has_success_condition_adapter(actor_algo) -> bool:
    adapter_in_nets = "condition_adapter" in actor_algo.nets["policy"]
    adapter_in_ema = (
        actor_algo.ema is not None
        and "condition_adapter" in actor_algo.ema.averaged_model["policy"]
    )
    return bool(adapter_in_nets or adapter_in_ema)


def require_unconditioned_policy(actor_algo, checkpoint: Path) -> None:
    if has_success_condition_adapter(actor_algo):
        raise ValueError(
            "standard mixed imitation must not use a condition adapter, but one "
            f"was found in checkpoint {checkpoint}"
        )


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
    policy_optim = actor_config.algo.optim_params.policy
    return {
        "mode": mode_name,
        "description": infer_description(args),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "epochs": int(args.epochs),
        "steps_per_epoch": int(args.steps_per_epoch),
        "steps_per_epoch_source": str(args.steps_per_epoch_source),
        "drop_last": bool(args.actor_drop_last),
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
        "actor_optimization": {
            "optimizer_type": str(policy_optim.optimizer_type),
            "initial_learning_rate": float(policy_optim.learning_rate.initial),
            "scheduler_type": str(policy_optim.learning_rate.scheduler_type),
            "scheduler_enabled": not bool(args.actor_disable_lr_scheduler),
            "scheduler_step_every_batch": bool(policy_optim.learning_rate.step_every_batch),
            "scheduler_warmup_steps": int(policy_optim.learning_rate.warmup_steps),
            "scheduler_num_cycles": float(policy_optim.learning_rate.num_cycles),
            "weight_decay": float(policy_optim.regularization.L2),
            "num_train_batches": int(policy_optim.num_train_batches),
            "num_epochs": int(policy_optim.num_epochs),
            "batch_size": int(actor_config.train.batch_size),
            "seed": int(actor_config.train.seed),
        },
        "actor_normalization": {
            "hdf5_normalize_obs": bool(actor_config.train.hdf5_normalize_obs),
            "action_config": jsonable(actor_config.train.action_config),
            "action_stats_source": getattr(
                actor_dataset,
                "actor_action_normalization_source",
                None,
            ),
            "normalize_source_weights_by_dataset_size": bool(
                actor_config.train.normalize_weights_by_ds_size
            ),
        },
        "actor_sampling": {
            "mode": (
                "uniform_sample_pool_without_replacement"
                if args.actor_uniform_sample_pool
                else "weighted_random_with_replacement"
            ),
            "source_weighting_enabled": not bool(args.actor_uniform_sample_pool),
            "source_weights": None if args.actor_uniform_sample_pool else {
                "human_demo": float(args.actor_demo_weight),
                "success_rollout": float(args.actor_success_weight),
                "failure_rollout": float(args.actor_failure_weight),
            },
        },
        "actor_training_scope": "full_pretrained_dp_policy_including_obs_encoder",
        "actor_trainability": jsonable(trainability),
        "critic_trained": False,
        "loaded_pretrained_dp": True,
        "success_conditioning": success_condition_summary(args),
        "distributed_training": {
            "enabled": bool(getattr(args, "distributed", False)),
            "world_size": int(getattr(args, "distributed_world_size", 1)),
            "backend": str(getattr(args, "distributed_backend_resolved", "none")),
            "launcher": "torchrun" if bool(getattr(args, "distributed", False)) else "python",
            "batch_size_per_rank": int(actor_config.train.batch_size),
            "effective_global_batch_size": int(actor_config.train.batch_size)
            * int(getattr(args, "distributed_world_size", 1)),
            "gradient_sync": "bounded_async_bucketed_mean_all_reduce",
            "gradient_bucket_cap_mb": float(args.gradient_bucket_cap_mb),
            "rank_zero_writes_only": True,
        },
        "last_checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
        "history": history,
    }


def train(args: argparse.Namespace) -> dict:
    distributed = initialize_distributed(args)
    args.distributed = bool(distributed.enabled)
    args.distributed_rank = int(distributed.rank)
    args.distributed_local_rank = int(distributed.local_rank)
    args.distributed_world_size = int(distributed.world_size)
    args.distributed_backend_resolved = str(distributed.backend)
    if distributed.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    if distributed.is_main_process:
        models_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.conditioned_mixed_imitation):
        failure_label = float(args.actor_failure_anti_failure_label)
        if float(args.actor_failure_weight) > 0.0 and failure_label != 1.0:
            raise ValueError(
                "conditioned mixed imitation requires failure condition 0 "
                f"(actor_failure_anti_failure_label=1.0), got {failure_label}"
            )

    device = distributed.device

    dp_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config, _ = FileUtils.config_from_checkpoint(
        ckpt_dict=dp_ckpt,
        verbose=False,
    )
    if args.seed is None:
        args.seed = int(checkpoint_config.train.seed)
    # Construct identical policies on every rank before switching to independent
    # rank-local data and diffusion-noise streams.
    seed_process(args.seed, device)

    actor_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=dp_ckpt,
        device=device,
        verbose=False,
    )
    actor_policy.start_episode()
    actor_algo = actor_policy.policy
    if args.actor_batch_size is None:
        args.actor_batch_size = int(checkpoint_config.train.batch_size)
    if args.actor_lr is None:
        args.actor_lr = float(
            checkpoint_config.algo.optim_params.policy.learning_rate.initial
        )
    active_actor_sources = sum(
        float(weight) > 0.0
        for weight in (
            args.actor_demo_weight,
            args.actor_success_weight,
            args.actor_failure_weight,
        )
    )
    if args.actor_uniform_sample_pool:
        args.actor_normalize_weights_by_ds_size = False
    elif args.actor_normalize_weights_by_ds_size is None:
        args.actor_normalize_weights_by_ds_size = (
            bool(checkpoint_config.train.normalize_weights_by_ds_size)
            if active_actor_sources == 1
            else True
        )
    if args.actor_hdf5_cache_mode is None:
        args.actor_hdf5_cache_mode = (
            str(checkpoint_config.train.hdf5_cache_mode)
            if active_actor_sources == 1
            else "low_dim"
        )
    args.actor_initialized_from_deployed_ema = False
    if args.resume_checkpoint is None:
        args.actor_initialized_from_deployed_ema = initialize_actor_from_deployed_ema(actor_algo)
    if not 0.0 <= float(args.condition_dropout) < 1.0:
        raise ValueError(f"condition_dropout must be in [0, 1), got {args.condition_dropout}")
    if bool(args.conditioned_mixed_imitation):
        if not hasattr(actor_algo, "install_success_condition_adapter"):
            raise RuntimeError("loaded policy does not support success-conditioned DP adapters")
        actor_algo.install_success_condition_adapter(hidden_dim=int(args.condition_hidden_dim))
        actor_algo.set_inference_success_condition(
            success_condition=1.0,
            condition_mask=1.0,
        )
        actor_algo.success_condition_dropout = float(args.condition_dropout)
    else:
        require_unconditioned_policy(actor_algo, args.checkpoint)
    actor_dataset, actor_loader, actor_config = build_actor_loader(
        args=args,
        actor_algo=actor_algo,
        checkpoint_dict=dp_ckpt,
    )
    args.actor_drop_last = bool(actor_loader.drop_last)
    if args.steps_per_epoch is None:
        args.steps_per_epoch = len(actor_loader)
        args.steps_per_epoch_source = "dataloader_length"
    else:
        args.steps_per_epoch_source = "command_line_override"

    configure_actor_optimizer(
        actor_algo,
        args.actor_lr,
        args.actor_disable_lr_scheduler,
        num_train_batches=args.steps_per_epoch,
        num_epochs=args.epochs,
        reset_scheduler=True,
    )
    with actor_config.values_unlocked():
        actor_config.experiment.name = args.experiment_name
        actor_config.experiment.epoch_every_n_steps = int(args.steps_per_epoch)
        actor_config.train.output_dir = str(args.output_dir)
        actor_config.train.num_epochs = int(args.epochs)
        actor_config.train.seed = int(args.seed)
        actor_config.algo.optim_params.policy.learning_rate.initial = float(args.actor_lr)
        actor_config.algo.optim_params.policy.num_train_batches = int(args.steps_per_epoch)
        actor_config.algo.optim_params.policy.num_epochs = int(args.epochs)

    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if "model" not in ckpt:
            raise ValueError(f"resume checkpoint is not a robomimic policy checkpoint: {args.resume_checkpoint}")
        saved_distributed = (ckpt.get("variable_state", {}) or {}).get(
            "distributed_training",
            {},
        )
        if bool(saved_distributed.get("enabled", False)) and (
            not distributed.enabled
            or int(saved_distributed.get("world_size", 1)) != distributed.world_size
        ):
            raise ValueError(
                "distributed self-imitation checkpoints require the same world "
                f"size on resume: checkpoint={saved_distributed.get('world_size')} "
                f"requested={distributed.world_size}"
            )
        actor_algo.deserialize(ckpt["model"], load_optimizers=True)
        if not bool(args.conditioned_mixed_imitation):
            require_unconditioned_policy(actor_algo, args.resume_checkpoint)
        configure_actor_optimizer(
            actor_algo,
            args.actor_lr,
            args.actor_disable_lr_scheduler,
            num_train_batches=args.steps_per_epoch,
            num_epochs=args.epochs,
            preserve_current_lr=True,
        )
        variable_state = ckpt.get("variable_state", {}) or {}
        start_epoch = int(variable_state.get("epoch", 0))
        global_step = int(variable_state.get("global_step", start_epoch * int(args.steps_per_epoch)))
        summary_path = args.output_dir / "partial_summary.json"
        if distributed.is_main_process and summary_path.exists():
            try:
                history = list(json.loads(summary_path.read_text()).get("history", []))
            except Exception:
                history = []

    synchronized_modules: list[nn.Module] = [actor_algo.nets]
    if actor_algo.ema is not None:
        synchronized_modules.append(actor_algo.ema.averaged_model)
    broadcast_module_state(synchronized_modules, distributed)
    actor_algo.gradient_sync_fn = (
        (
            lambda parameters: all_reduce_gradients(
                parameters,
                distributed,
                bucket_cap_mb=float(args.gradient_bucket_cap_mb),
                preserve_unused_parameters=False,
            )
        )
        if distributed.enabled
        else None
    )
    synchronize_training_buffers = modules_have_mutable_batch_norm(
        synchronized_modules
    )
    if distributed.enabled:
        seed_process(int(args.seed) + distributed.rank, device)

    trainability = actor_trainability_summary(actor_algo)
    if distributed.is_main_process:
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
                    "actor_optimization": {
                        "optimizer_type": str(actor_config.algo.optim_params.policy.optimizer_type),
                        "initial_learning_rate": float(args.actor_lr),
                        "scheduler_type": str(
                            actor_config.algo.optim_params.policy.learning_rate.scheduler_type
                        ),
                        "scheduler_enabled": not bool(args.actor_disable_lr_scheduler),
                        "scheduler_step_every_batch": bool(
                            actor_config.algo.optim_params.policy.learning_rate.step_every_batch
                        ),
                        "scheduler_warmup_steps": int(
                            actor_config.algo.optim_params.policy.learning_rate.warmup_steps
                        ),
                        "weight_decay": float(
                            actor_config.algo.optim_params.policy.regularization.L2
                        ),
                        "batch_size": int(args.actor_batch_size),
                        "seed": int(args.seed),
                        "steps_per_epoch": int(args.steps_per_epoch),
                        "steps_per_epoch_source": str(args.steps_per_epoch_source),
                        "actor_dataset_size": int(len(actor_dataset)),
                        "drop_last": bool(args.actor_drop_last),
                        "epochs": int(args.epochs),
                    },
                    "actor_normalization": {
                        "hdf5_normalize_obs": bool(actor_config.train.hdf5_normalize_obs),
                        "action_config": jsonable(actor_config.train.action_config),
                        "action_stats_source": getattr(
                            actor_dataset,
                            "actor_action_normalization_source",
                            None,
                        ),
                        "normalize_source_weights_by_dataset_size": bool(
                            actor_config.train.normalize_weights_by_ds_size
                        ),
                    },
                    "actor_sampling": {
                        "mode": (
                            "uniform_sample_pool_without_replacement"
                            if args.actor_uniform_sample_pool
                            else "weighted_random_with_replacement"
                        ),
                        "source_weighting_enabled": not bool(args.actor_uniform_sample_pool),
                    },
                    "distributed_training": {
                        "enabled": bool(distributed.enabled),
                        "world_size": int(distributed.world_size),
                        "backend": str(distributed.backend),
                        "batch_size_per_rank": int(args.actor_batch_size),
                        "effective_global_batch_size": int(args.actor_batch_size)
                        * int(distributed.world_size),
                        "rank_zero_writes_only": True,
                    },
                    "success_conditioning": success_condition_summary(args),
                    "resume_epoch": int(start_epoch),
                },
                indent=2,
            ),
            flush=True,
        )

    writer = None
    if distributed.is_main_process and args.tensorboard:
        if SummaryWriter is None:
            print("TensorBoard requested but torch.utils.tensorboard is unavailable.", flush=True)
        else:
            log_dir = args.tensorboard_dir or (args.output_dir / "tensorboard")
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))

    last_checkpoint: Path | None = args.resume_checkpoint
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        if distributed.enabled and hasattr(actor_loader.sampler, "set_epoch"):
            actor_loader.sampler.set_epoch(epoch)
        actor_iterator = iter(actor_loader)
        epoch_logs = [] if distributed.is_main_process else None
        for _ in range(int(args.steps_per_epoch)):
            try:
                actor_batch = next(actor_iterator)
            except StopIteration:
                actor_iterator = iter(actor_loader)
                actor_batch = next(actor_iterator)
            global_step += 1
            if synchronize_training_buffers:
                broadcast_module_buffers(synchronized_modules, distributed)
            actor_log = actor_train_step(
                actor_policy,
                actor_batch,
                global_step,
                obs_normalization_stats=actor_policy.obs_normalization_stats,
                materialize_log=distributed.is_main_process,
            )
            if distributed.is_main_process:
                epoch_logs.append(actor_log)
            if distributed.is_main_process and global_step % int(args.log_every) == 0:
                payload = {"epoch": int(epoch), "global_step": int(global_step)}
                payload.update(actor_log)
                print(json.dumps(jsonable(payload), sort_keys=True), flush=True)

        metrics = numeric_mean(epoch_logs) if distributed.is_main_process else {}
        record = {"epoch": int(epoch), "global_step": int(global_step), "train": metrics}
        if distributed.is_main_process:
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
        should_save_epoch = distributed.is_main_process and bool(args.save_checkpoints) and (
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
                distributed_context=distributed,
            )

        should_save_latest = distributed.is_main_process and bool(args.save_checkpoints) and (
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
                    distributed_context=distributed,
                )
            shutil.copyfile(latest, args.output_dir / "last_bak.pth")
            last_checkpoint = latest

    if writer is not None:
        writer.close()
    close_fn = getattr(actor_dataset, "close", None)
    if callable(close_fn):
        close_fn()

    summary = {}
    if distributed.is_main_process:
        summary = make_summary(
            args=args,
            actor_dataset=actor_dataset,
            actor_config=actor_config,
            trainability=trainability,
            history=history,
            last_checkpoint=last_checkpoint,
        )
        write_json(args.output_dir / "summary.json", summary)
        print(
            json.dumps(
                jsonable({k: v for k, v in summary.items() if k != "history"}),
                indent=2,
            ),
            flush=True,
        )
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()
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
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable torchrun data-parallel training. This is also enabled "
            "automatically when WORLD_SIZE is greater than one."
        ),
    )
    parser.add_argument(
        "--distributed-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-bucket-cap-mb",
        type=float,
        default=100.0,
        help="Maximum size of each flat gradient all-reduce bucket in MiB.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help="Local process rank supplied by torchrun; the environment wins.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to train.seed from the pretrained DP checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=(
            "Override batches per epoch. By default, use len(train_loader): one "
            "shuffled pass over the selected pooled sequences per epoch."
        ),
    )
    parser.add_argument(
        "--actor-batch-size",
        type=int,
        default=None,
        help="Defaults to train.batch_size from the pretrained DP checkpoint.",
    )
    parser.add_argument("--actor-num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-seq-length", type=int, default=None)
    parser.add_argument(
        "--actor-hdf5-cache-mode",
        choices=("all", "low_dim"),
        default=None,
        help=(
            "Defaults to the checkpoint cache mode for one source and to low_dim "
            "for multi-source MetaDataset training."
        ),
    )
    parser.add_argument("--demo-filter-key", type=str, default="")
    parser.add_argument("--success-filter-key", type=str, default="success")
    parser.add_argument("--failure-filter-key", type=str, default="failure")
    parser.add_argument("--actor-demo-weight", type=float, default=1.0)
    parser.add_argument("--actor-success-weight", type=float, default=1.0)
    parser.add_argument("--actor-failure-weight", type=float, default=0.0)
    parser.add_argument(
        "--actor-uniform-sample-pool",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--actor-failure-demo-start-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-failure-sample-start-offset", type=int, default=0)
    parser.add_argument("--actor-failure-anti-failure-label", type=float, default=1.0)
    parser.add_argument("--conditioned-mixed-imitation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--condition-label-mode",
        choices=("outcome", "human_only"),
        default="outcome",
    )
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--actor-normalize-weights-by-ds-size",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Defaults to the checkpoint setting for one source and to true for "
            "multi-source training so source weights are sampling probabilities."
        ),
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=None,
        help="Defaults to the policy learning rate from the pretrained DP checkpoint.",
    )
    parser.add_argument(
        "--actor-disable-lr-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable the checkpoint DP learning-rate schedule; it remains enabled by default.",
    )
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
