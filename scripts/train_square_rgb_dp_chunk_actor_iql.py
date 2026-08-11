#!/usr/bin/env python3
"""Hybrid Square RGB-DP chunk actor + one-step IQL critic post-training.

This keeps the post-training actor as the original RGB DiffusionPolicy UNet:
it is initialized from the pretrained DP checkpoint and updated with the
standard diffusion BC objective over full action chunks. The critic is a
separate visual one-step IQL critic, trained on (o_t, a_t, r_t, o_{t+1}) where
a_t is the executed first action. At evaluation time the actor proposes action
chunks, and the critic scores only each chunk's first action.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils

from train_square_rgb_dp_one_step_idql import (
    ChunkIQLCritic,
    add_tensorboard_scalars,
    binary_average_precision,
    binary_roc_auc,
    soft_update,
    tensor_stats,
)
from train_square_rgb_dp_one_step_idql_visual_critic import (
    IndexedRawTransitionDataset,
    compute_visual_iql_losses,
    dataset_stats,
    encode_obs,
    get_policy_obs_encoder,
    load_feature_npz,
    make_loader,
    make_tensorboard_writer,
    raw_batch_to_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_INDEX = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz"
)
DEFAULT_DEMOS = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_ROLLOUTS = ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
DEFAULT_CHECKPOINT = (
    ROOT / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth"
)
DEFAULT_OUTPUT = ROOT / "trained_models/square_rgb_dp_idql_visual/default_reward_dp_chunk_actor_iql"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2))


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def initialize_actor_from_deployed_ema(actor_algo) -> bool:
    if actor_algo.ema is None:
        return False
    ema_state = copy.deepcopy(actor_algo.ema.averaged_model.state_dict())
    actor_algo.nets.load_state_dict(ema_state)
    actor_algo.ema.averaged_model.load_state_dict(actor_algo.nets.state_dict())
    if hasattr(actor_algo, "_refresh_ema_parameter_views"):
        actor_algo._refresh_ema_parameter_views()
    return True


def apply_pretrained_action_normalization(actor_dataset, checkpoint_dict: dict) -> str:
    action_stats = checkpoint_dict.get("action_normalization_stats")
    if action_stats is None:
        raise ValueError(
            "pretrained DP checkpoint is missing action_normalization_stats; "
            "refusing to normalize actor data with mixed-dataset statistics"
        )
    if not hasattr(actor_dataset, "set_action_normalization_stats"):
        raise TypeError("actor dataset does not support set_action_normalization_stats")
    actor_dataset.set_action_normalization_stats(copy.deepcopy(action_stats))
    source = "pretrained_dp_checkpoint_action_normalization_stats"
    setattr(actor_dataset, "actor_action_normalization_source", source)
    return source


def action_normalization_stats_match(left: dict, right: dict) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for action_key in left:
        if set(left[action_key].keys()) != set(right[action_key].keys()):
            return False
        for stat_key in left[action_key]:
            if not np.allclose(
                np.asarray(left[action_key][stat_key]),
                np.asarray(right[action_key][stat_key]),
            ):
                return False
    return True


def configure_actor_optimizer(
    actor_algo,
    lr: float | None,
    disable_lr_scheduler: bool,
    *,
    num_train_batches: int | None = None,
    num_epochs: int | None = None,
    reset_scheduler: bool = False,
    preserve_current_lr: bool = False,
) -> None:
    optim_params = actor_algo.optim_params["policy"]
    if num_train_batches is not None:
        optim_params.num_train_batches = int(num_train_batches)
    if num_epochs is not None:
        optim_params.num_epochs = int(num_epochs)
    configured_lr = float(
        optim_params.learning_rate.initial if lr is None else lr
    )
    if lr is not None:
        optim_params.learning_rate.initial = configured_lr

    optimizer = actor_algo.optimizers["policy"]
    optim_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    missing = [
        param
        for param in actor_algo.nets["policy"].parameters()
        if param.requires_grad and id(param) not in optim_param_ids
    ]
    if missing:
        optimizer.add_param_group({"params": missing, "lr": configured_lr})
    if not preserve_current_lr:
        for group in optimizer.param_groups:
            group["lr"] = configured_lr
            group["initial_lr"] = configured_lr
    if disable_lr_scheduler:
        actor_algo.lr_schedulers["policy"] = None
        actor_algo.step_lr_schedulers_every_batch["policy"] = False
    elif reset_scheduler:
        actor_algo.lr_schedulers["policy"] = TorchUtils.lr_scheduler_from_optim_params(
            net_optim_params=optim_params,
            net=actor_algo.nets["policy"],
            optimizer=optimizer,
        )
        actor_algo.step_lr_schedulers_every_batch["policy"] = bool(
            optim_params.learning_rate.get("step_every_batch", False)
        )


def actor_trainability_summary(actor_algo) -> dict[str, Any]:
    policy = actor_algo.nets["policy"]
    obs_encoder = policy["obs_encoder"]
    optim_param_ids = {
        id(param)
        for group in actor_algo.optimizers["policy"].param_groups
        for param in group["params"]
    }

    def module_summary(module: nn.Module) -> dict[str, Any]:
        params = list(module.named_parameters())
        total = sum(param.numel() for _, param in params)
        trainable = sum(param.numel() for _, param in params if param.requires_grad)
        optimized = sum(param.numel() for _, param in params if id(param) in optim_param_ids)
        frozen_names = [name for name, param in params if not param.requires_grad]
        missing_names = [name for name, param in params if id(param) not in optim_param_ids]
        return {
            "num_parameters": int(total),
            "num_trainable_parameters": int(trainable),
            "num_optimizer_parameters": int(optimized),
            "all_trainable": len(frozen_names) == 0,
            "all_in_optimizer": len(missing_names) == 0,
            "num_frozen_tensors": int(len(frozen_names)),
            "num_missing_optimizer_tensors": int(len(missing_names)),
            "first_frozen_tensors": frozen_names[:10],
            "first_missing_optimizer_tensors": missing_names[:10],
        }

    summary = {
        "scope": "full_pretrained_dp_policy",
        "policy": module_summary(policy),
        "obs_encoder": module_summary(obs_encoder),
    }
    if not summary["policy"]["all_trainable"] or not summary["policy"]["all_in_optimizer"]:
        raise RuntimeError(
            "DP actor is not fully trainable. "
            f"trainability={json.dumps(jsonable(summary), indent=2)}"
        )
    if not summary["obs_encoder"]["all_trainable"] or not summary["obs_encoder"]["all_in_optimizer"]:
        raise RuntimeError(
            "DP actor obs encoder is not fully trainable. "
            f"trainability={json.dumps(jsonable(summary), indent=2)}"
        )
    return summary


def actor_source_entries(args: argparse.Namespace) -> list[dict]:
    conditioned = bool(getattr(args, "conditioned_mixed_imitation", False))
    condition_label_mode = str(getattr(args, "condition_label_mode", "outcome"))
    human_only_condition = conditioned and condition_label_mode == "human_only"
    entries = []
    if float(args.actor_demo_weight) > 0.0:
        demo = {
            "path": str(args.demo_dataset),
            "weight": float(args.actor_demo_weight),
            "anti_failure": 0.0,
        }
        if args.demo_filter_key:
            demo["filter_key"] = args.demo_filter_key
        entries.append(demo)
    if float(args.actor_success_weight) > 0.0:
        entries.append(
            {
                "path": str(args.success_dataset),
                "filter_key": args.success_filter_key,
                "weight": float(args.actor_success_weight),
                "anti_failure": 1.0 if human_only_condition else 0.0,
            }
        )
    if float(args.actor_failure_weight) > 0.0:
        failure = {
            "path": str(args.failure_dataset),
            "weight": float(args.actor_failure_weight),
            "anti_failure": float(getattr(args, "actor_failure_anti_failure_label", 1.0)),
        }
        if args.failure_filter_key:
            failure["filter_key"] = args.failure_filter_key
        if bool(getattr(args, "actor_failure_demo_start_only", False)):
            failure["demo_start_only"] = True
        sample_start_offset = int(getattr(args, "actor_failure_sample_start_offset", 0))
        if sample_start_offset != 0:
            failure["sample_start_offset"] = sample_start_offset
        entries.append(failure)
    if not entries:
        raise ValueError("at least one actor dataset weight must be positive")
    return entries


def build_actor_loader(
    *,
    args: argparse.Namespace,
    actor_algo,
    checkpoint_dict: dict,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, Any]:
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint_dict, verbose=False)
    ObsUtils.initialize_obs_utils_with_config(config)
    prediction_horizon = int(actor_algo.algo_config.horizon.prediction_horizon)
    observation_horizon = int(actor_algo.algo_config.horizon.observation_horizon)
    seq_length = int(args.actor_seq_length or prediction_horizon)
    if seq_length < prediction_horizon:
        raise ValueError(
            f"actor_seq_length={seq_length} must be >= DP prediction_horizon={prediction_horizon}"
        )

    source_entries = actor_source_entries(args)
    if args.actor_hdf5_cache_mode == "all" and len(source_entries) > 1:
        raise ValueError(
            "actor_hdf5_cache_mode=all is unsupported for multi-source MetaDataset training; "
            "use low_dim or no cache"
        )

    with config.values_unlocked():
        config.train.data = source_entries
        config.train.normalize_weights_by_ds_size = bool(
            args.actor_normalize_weights_by_ds_size
            and not bool(getattr(args, "actor_uniform_sample_pool", False))
        )
        config.train.hdf5_cache_mode = args.actor_hdf5_cache_mode
        config.train.hdf5_load_next_obs = False
        config.train.seq_length = seq_length
        config.train.frame_stack = observation_horizon
        config.train.pad_seq_length = True
        config.train.pad_frame_stack = True
        config.train.batch_size = int(args.actor_batch_size)
        config.train.num_data_workers = int(args.actor_num_workers)

    dataset = TrainUtils.dataset_factory(config, obs_keys=list(actor_algo.obs_shapes.keys()))
    if args.actor_hdf5_cache_mode == "all":
        cached_action_stats = dataset.get_action_normalization_stats()
        checkpoint_action_stats = checkpoint_dict.get("action_normalization_stats")
        if checkpoint_action_stats is None or not action_normalization_stats_match(
            cached_action_stats,
            checkpoint_action_stats,
        ):
            raise ValueError(
                "actor_hdf5_cache_mode=all requires dataset action normalization "
                "to match the pretrained DP checkpoint"
            )
    action_normalization_source = apply_pretrained_action_normalization(dataset, checkpoint_dict)
    print(json.dumps({"actor_action_normalization_source": action_normalization_source}), flush=True)
    sampler = None
    if not bool(getattr(args, "actor_uniform_sample_pool", False)):
        sampler = dataset.get_dataset_sampler() if hasattr(dataset, "get_dataset_sampler") else None
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader_kwargs: dict[str, Any] = {}
    if int(args.actor_num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.actor_batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=len(dataset) >= int(args.actor_batch_size),
        num_workers=int(args.actor_num_workers),
        pin_memory=bool(args.pin_memory and actor_algo.device.type == "cuda"),
        generator=generator,
        **loader_kwargs,
    )
    return dataset, loader, config


def source_counts(dataset: IndexedRawTransitionDataset) -> dict[str, int]:
    return {str(source): int(np.sum(dataset.source == source)) for source in np.unique(dataset.source)}


def transition_sample_weights(dataset: IndexedRawTransitionDataset, args: argparse.Namespace) -> np.ndarray:
    source_weights = {
        "demo": float(args.critic_demo_weight),
        "rollout_success": float(args.critic_success_weight),
        "rollout_failure": float(args.critic_failure_weight),
    }
    weights = np.zeros(len(dataset), dtype=np.float64)
    for source, source_weight in source_weights.items():
        mask = dataset.source == source
        count = int(np.sum(mask))
        if count == 0 or source_weight <= 0.0:
            continue
        if args.critic_normalize_weights_by_source:
            weights[mask] = source_weight / float(count)
        else:
            weights[mask] = source_weight
    if not np.any(weights > 0.0):
        raise ValueError(
            "critic source weights select no samples; check critic_demo_weight, "
            "critic_success_weight, and critic_failure_weight"
        )
    return weights


def make_transition_train_loader(
    dataset: IndexedRawTransitionDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    weights = torch.as_tensor(transition_sample_weights(dataset, args), dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 17)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader_kwargs: dict[str, Any] = {}
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        shuffle=False,
        drop_last=len(dataset) >= int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        generator=generator,
        **loader_kwargs,
    )


def actor_train_step(actor_policy, raw_batch: dict, step: int, obs_normalization_stats) -> dict[str, float]:
    actor_algo = actor_policy.policy
    actor_algo.set_train()
    batch = actor_algo.process_batch_for_training(raw_batch)
    batch = actor_algo.postprocess_batch_for_training(
        batch,
        obs_normalization_stats=obs_normalization_stats,
    )
    info = actor_algo.train_on_batch(batch, epoch=step, validate=False)
    actor_algo.on_gradient_step()
    return {k: float(v) for k, v in actor_algo.log_info(info).items()}


@torch.no_grad()
def evaluate_critic_split(
    *,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    teacher_encoder: nn.Module,
    dataset: IndexedRawTransitionDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    gamma: float,
    expectile: float,
    use_huber: bool,
    aux_next_pred_weight: float,
    aux_next_pred_mode: str,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> dict[str, float | None]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    critic_encoder.eval()
    target_critic_encoder.eval()
    critic.eval()
    target_critic.eval()
    teacher_encoder.eval()
    q_values = []
    v_values = []
    rewards = []
    success = []
    dones = []
    critic_losses = []
    iql_losses = []
    next_pred_losses = []
    for batch_index, raw_batch in enumerate(loader):
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        batch = raw_batch_to_device(raw_batch, device)
        critic_loss, critic_info, _ = compute_visual_iql_losses(
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            teacher_encoder=teacher_encoder,
            batch=batch,
            gamma=gamma,
            expectile=expectile,
            use_huber=use_huber,
            aux_next_pred_weight=aux_next_pred_weight,
            aux_next_pred_mode=aux_next_pred_mode,
        )
        obs_z = encode_obs(critic_encoder, batch["obs"])
        q = critic.q_min(obs_z, batch["actions"]).reshape(-1)
        v = critic.value(obs_z).reshape(-1)
        q_values.append(q.cpu().numpy())
        v_values.append(v.cpu().numpy())
        rewards.append(batch["rewards"].cpu().numpy())
        success.append(batch["success"].cpu().numpy())
        dones.append(batch["dones"].cpu().numpy())
        critic_losses.append(float(critic_loss.cpu()))
        iql_losses.append(float(critic_info["iql_loss"].cpu()))
        next_pred_losses.append(float(critic_info["next_pred_loss"].cpu()))
    q_np = np.concatenate(q_values)
    v_np = np.concatenate(v_values)
    reward_np = np.concatenate(rewards)
    success_np = np.concatenate(success)
    done_np = np.concatenate(dones)
    return {
        "num_samples": int(len(q_np)),
        "critic_loss": float(np.mean(critic_losses)),
        "iql_loss": float(np.mean(iql_losses)),
        "next_pred_loss": float(np.mean(next_pred_losses)),
        "q_mean": float(np.mean(q_np)),
        "v_mean": float(np.mean(v_np)),
        "reward_mean": float(np.mean(reward_np)),
        "done_fraction": float(np.mean(done_np)),
        "success_auc": binary_roc_auc(success_np > 0.5, q_np),
        "success_ap": binary_average_precision(success_np > 0.5, q_np),
        "reward_auc": binary_roc_auc(reward_np > 0.0, q_np),
        "reward_ap": binary_average_precision(reward_np > 0.0, q_np),
    }


def best_from_history(history: list[dict]) -> dict:
    best = {
        "val_critic_loss": float("inf"),
        "val_success_auc": -float("inf"),
        "val_reward_auc": -float("inf"),
    }
    for record in history:
        val = record.get("val", {})
        if val.get("critic_loss") is not None:
            best["val_critic_loss"] = min(best["val_critic_loss"], float(val["critic_loss"]))
        if val.get("success_auc") is not None:
            best["val_success_auc"] = max(best["val_success_auc"], float(val["success_auc"]))
        if val.get("reward_auc") is not None:
            best["val_reward_auc"] = max(best["val_reward_auc"], float(val["reward_auc"]))
    return best


def save_checkpoint(
    path: Path,
    *,
    actor_algo,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    feature_dim: int,
    action_dim: int,
    gamma: float,
    step: int,
    history: list[dict],
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "hybrid_dp_chunk_actor_iql": True,
        "visual_critic_idql": True,
        "actor_model": actor_algo.serialize(),
        "critic_encoder": critic_encoder.state_dict(),
        "target_critic_encoder": target_critic_encoder.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "args": vars(args),
        "feature_dim": int(feature_dim),
        "action_dim": int(action_dim),
        "chunk_horizon": 1,
        "actor_prediction_horizon": int(actor_algo.algo_config.horizon.prediction_horizon),
        "actor_action_horizon": int(actor_algo.algo_config.horizon.action_horizon),
        "observation_horizon": int(args.observation_horizon),
        "gamma": float(gamma),
        "action_mean": data["action_mean"].astype(np.float32),
        "action_std": data["action_std"].astype(np.float32),
        "normalize_actions": bool(args.normalize_actions),
        "features": str(args.feature_index),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "critic_encoder_trainable": True,
        "critic_encoder_initialized_from_dp": True,
        "actor_initialized_from_dp": True,
        "actor_training_objective": "diffusion_bc_full_chunk",
        "actor_trainability": actor_trainability_summary(actor_algo),
        "actor_encoder_trainable": True,
        "critic_input_mode": "raw_hdf5_observations",
        "critic_feature_index_usage": "transition_metadata_actions_rewards_splits_only",
        "critic_uses_cached_latents": False,
        "feature_index_contains_cached_latents": bool("obs_features" in data),
        "actor_lr_scheduler_disabled": bool(args.actor_disable_lr_scheduler),
        "critic_training_objective": "one_step_iql",
        "actor_source_weights": {
            "demo": float(args.actor_demo_weight),
            "success": float(args.actor_success_weight),
            "failure": float(args.actor_failure_weight),
            "normalize_by_dataset_size": bool(args.actor_normalize_weights_by_ds_size),
        },
        "critic_source_weights": {
            "demo": float(args.critic_demo_weight),
            "success": float(args.critic_success_weight),
            "failure": float(args.critic_failure_weight),
            "normalize_by_source": bool(args.critic_normalize_weights_by_source),
        },
        "aux_next_pred_enabled": bool(float(args.aux_next_pred_weight) > 0.0),
        "aux_next_pred_mode": str(args.aux_next_pred_mode),
        "aux_next_pred_weight": float(args.aux_next_pred_weight),
        "step": int(step),
        "history": history,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def make_summary(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    actor_dataset,
    actor_config,
    critic_datasets: dict[str, IndexedRawTransitionDataset],
    feature_dim: int,
    action_dim: int,
    gamma: float,
    best: dict,
    history: list[dict],
) -> dict:
    return {
        "hybrid_dp_chunk_actor_iql": True,
        "feature_index": str(args.feature_index),
        "output_dir": str(args.output_dir),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "actor_data": jsonable(actor_config.train.data),
        "actor_dataset_size": int(len(actor_dataset)),
        "actor_action_normalization_source": getattr(
            actor_dataset,
            "actor_action_normalization_source",
            None,
        ),
        "actor_initialized_from_deployed_ema": bool(
            getattr(args, "actor_initialized_from_deployed_ema", False)
        ),
        "critic_num_train": int(len(critic_datasets["train"])),
        "critic_num_val": int(len(critic_datasets["val"])),
        "critic_num_test": int(len(critic_datasets["test"])),
        "critic_source_counts": {
            split: source_counts(dataset) for split, dataset in critic_datasets.items()
        },
        "feature_dim": int(feature_dim),
        "action_dim": int(action_dim),
        "gamma": float(gamma),
        "normalize_actions_for_critic": bool(args.normalize_actions),
        "actor_training_objective": "diffusion_bc_full_chunk",
        "actor_training_scope": "full_pretrained_dp_policy_including_obs_encoder",
        "actor_encoder_trainable": True,
        "actor_lr_scheduler_disabled": bool(args.actor_disable_lr_scheduler),
        "critic_training_objective": "one_step_iql",
        "critic_input_mode": "raw_hdf5_observations",
        "critic_feature_index_usage": "transition_metadata_actions_rewards_splits_only",
        "critic_uses_cached_latents": False,
        "feature_index_contains_cached_latents": bool("obs_features" in data),
        "eval_actor_source": "hybrid_dp_chunk_actor",
        "critic_encoder_initialized_from_dp": True,
        "aux_next_pred": {
            "enabled": bool(float(args.aux_next_pred_weight) > 0.0),
            "weight": float(args.aux_next_pred_weight),
            "mode": str(args.aux_next_pred_mode),
            "target": "frozen_dp_teacher",
        },
        "loader": {
            "actor_num_workers": int(args.actor_num_workers),
            "critic_num_workers": int(args.num_workers),
            "eval_num_workers": int(args.eval_num_workers),
            "pin_memory": bool(args.pin_memory),
            "prefetch_factor": int(args.prefetch_factor),
            "persistent_workers": bool(args.persistent_workers),
        },
        "best": best,
        "last_completed_eval_step": int(history[-1]["step"]) if history else None,
        "history": history,
        "checkpoints": {
            "latest": str(args.output_dir / "latest.pt"),
            "best_critic_loss": str(args.output_dir / "best_critic_loss.pt"),
            "best_success_auc": str(args.output_dir / "best_success_auc.pt"),
            "best_reward_auc": str(args.output_dir / "best_reward_auc.pt"),
            "last": str(args.output_dir / "last.pt"),
        },
    }


def train(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if float(args.aux_next_pred_weight) < 0.0:
        raise ValueError(f"aux_next_pred_weight must be non-negative, got {args.aux_next_pred_weight}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")

    data = load_feature_npz(args.feature_index)
    if int(data["chunk_horizon"]) != 1:
        raise ValueError(
            f"hybrid trainer requires one-step feature index; got chunk_horizon={int(data['chunk_horizon'])}"
        )
    gamma = float(data["gamma"])
    action_dim = int(data["action_dim"])
    observation_horizon = int(args.observation_horizon)
    if "observation_horizon" in data and int(data["observation_horizon"]) != observation_horizon:
        raise ValueError(
            f"feature index observation_horizon={int(data['observation_horizon'])}, "
            f"but args.observation_horizon={observation_horizon}"
        )
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
    configure_actor_optimizer(actor_algo, args.actor_lr, args.actor_disable_lr_scheduler)
    print(
        json.dumps({"actor_trainability": jsonable(actor_trainability_summary(actor_algo))}, indent=2),
        flush=True,
    )

    actor_dataset, actor_loader, actor_config = build_actor_loader(
        args=args,
        actor_algo=actor_algo,
        checkpoint_dict=dp_ckpt,
    )
    actor_iterator = cycle(actor_loader)

    critic_encoder = get_policy_obs_encoder(actor_policy).to(device)
    target_critic_encoder = copy.deepcopy(critic_encoder).to(device)
    teacher_encoder = get_policy_obs_encoder(actor_policy).to(device)
    teacher_encoder.eval().requires_grad_(False)
    feature_dim = int(critic_encoder.output_shape()[0]) * observation_horizon

    critic_datasets = {
        split: IndexedRawTransitionDataset(
            data,
            split=split,
            demo_dataset=args.demo_dataset,
            rollout_dataset=args.rollout_dataset,
            observation_horizon=observation_horizon,
            reward_scale=args.reward_scale,
            normalize_actions=args.normalize_actions,
        )
        for split in ("train", "val", "test")
    }
    stats = dataset_stats(critic_datasets)
    write_json(args.output_dir / "reward_decomposition_stats.json", stats)
    print(json.dumps({"reward_decomposition_stats": jsonable(stats)}, indent=2), flush=True)
    print(
        json.dumps(
            {
                "actor_dataset_size": int(len(actor_dataset)),
                "actor_data": jsonable(actor_config.train.data),
                "critic_source_counts": {
                    split: source_counts(dataset) for split, dataset in critic_datasets.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )

    critic = ChunkIQLCritic(
        feature_dim=feature_dim,
        action_dim=action_dim,
        chunk_horizon=1,
        hidden_dims=tuple(args.critic_hidden_dims),
        dropout=args.critic_dropout,
        aux_next_pred=bool(float(args.aux_next_pred_weight) > 0.0),
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    target_critic_encoder.eval().requires_grad_(False)
    target_critic.eval().requires_grad_(False)

    critic_optimizer = torch.optim.AdamW(
        [
            {"params": critic_encoder.parameters(), "lr": args.critic_encoder_lr},
            {"params": critic.parameters(), "lr": args.critic_lr},
        ],
        lr=args.critic_lr,
        weight_decay=args.critic_weight_decay,
    )

    history: list[dict] = []
    start_step = 0
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
        print(f"Resuming hybrid DP-chunk actor IQL from {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        actor_algo.deserialize(checkpoint["actor_model"], load_optimizers=True)
        configure_actor_optimizer(actor_algo, args.actor_lr, args.actor_disable_lr_scheduler)
        print(
            json.dumps({"actor_trainability_after_resume": jsonable(actor_trainability_summary(actor_algo))}, indent=2),
            flush=True,
        )
        critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        target_critic_encoder.load_state_dict(checkpoint.get("target_critic_encoder", checkpoint["critic_encoder"]))
        critic.load_state_dict(checkpoint["critic"])
        target_critic.load_state_dict(checkpoint.get("target_critic", checkpoint["critic"]))
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))

    writer = make_tensorboard_writer(args)
    if writer is not None:
        writer.add_text("config/feature_index", str(args.feature_index), 0)
        writer.add_text("config/checkpoint", str(args.checkpoint), 0)
        add_tensorboard_scalars(
            writer,
            "data",
            {f"{k}/{kk}/{kkk}": v for k, vv in stats.items() for kk, vvv in vv.items() for kkk, v in vvv.items()},
            0,
        )
        writer.flush()

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    critic_loader = make_transition_train_loader(critic_datasets["train"], args, device)
    critic_iterator = cycle(critic_loader)
    best = best_from_history(history)

    for step in range(start_step + 1, args.total_steps + 1):
        actor_batch = next(actor_iterator)
        actor_log = actor_train_step(
            actor_policy,
            actor_batch,
            step,
            obs_normalization_stats=actor_policy.obs_normalization_stats,
        )

        raw_batch = next(critic_iterator)
        batch = raw_batch_to_device(raw_batch, device)
        critic_encoder.train()
        critic.train()
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss, critic_info, _ = compute_visual_iql_losses(
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            teacher_encoder=teacher_encoder,
            batch=batch,
            gamma=gamma,
            expectile=args.expectile,
            use_huber=args.use_huber,
            aux_next_pred_weight=args.aux_next_pred_weight,
            aux_next_pred_mode=args.aux_next_pred_mode,
        )
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            list(critic_encoder.parameters()) + list(critic.parameters()),
            args.grad_clip,
        )
        critic_optimizer.step()
        soft_update(target_critic, critic, args.target_tau)
        soft_update(target_critic_encoder, critic_encoder, args.target_tau)

        if step % args.log_every == 0:
            payload = {
                "step": int(step),
                **{f"actor_{k}": float(v) for k, v in actor_log.items()},
                **{k: float(v.detach().cpu()) for k, v in critic_info.items()},
                "critic_grad_norm": float(critic_grad),
                "critic_lr": float(critic_optimizer.param_groups[-1]["lr"]),
                "critic_encoder_lr": float(critic_optimizer.param_groups[0]["lr"]),
                "actor_lr": float(actor_algo.optimizers["policy"].param_groups[-1]["lr"]),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "train", {k: v for k, v in payload.items() if k != "step"}, step)
                writer.flush()

        if step % args.eval_every == 0 or step == args.total_steps:
            val_metrics = evaluate_critic_split(
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                teacher_encoder=teacher_encoder,
                dataset=critic_datasets["val"],
                batch_size=args.eval_batch_size,
                device=device,
                max_batches=args.max_eval_batches,
                gamma=gamma,
                expectile=args.expectile,
                use_huber=args.use_huber,
                aux_next_pred_weight=args.aux_next_pred_weight,
                aux_next_pred_mode=args.aux_next_pred_mode,
                num_workers=args.eval_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
            )
            test_metrics = evaluate_critic_split(
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                teacher_encoder=teacher_encoder,
                dataset=critic_datasets["test"],
                batch_size=args.eval_batch_size,
                device=device,
                max_batches=args.max_eval_batches,
                gamma=gamma,
                expectile=args.expectile,
                use_huber=args.use_huber,
                aux_next_pred_weight=args.aux_next_pred_weight,
                aux_next_pred_mode=args.aux_next_pred_mode,
                num_workers=args.eval_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
            )
            record = {
                "step": int(step),
                "val": val_metrics,
                "test": test_metrics,
                "actor_train": actor_log,
            }
            history.append(record)
            print(json.dumps(jsonable(record), indent=2), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "val", val_metrics, step)
                add_tensorboard_scalars(writer, "test", test_metrics, step)
                add_tensorboard_scalars(writer, "actor_eval_step_train", actor_log, step)
                writer.flush()

            metrics = {"val": val_metrics, "test": test_metrics, "actor_train": actor_log}
            save_checkpoint(
                args.output_dir / "latest.pt",
                actor_algo=actor_algo,
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                critic_optimizer=critic_optimizer,
                args=args,
                data=data,
                feature_dim=feature_dim,
                action_dim=action_dim,
                gamma=gamma,
                step=step,
                history=history,
                metrics=metrics,
            )
            partial = make_summary(
                args,
                data,
                actor_dataset,
                actor_config,
                critic_datasets,
                feature_dim,
                action_dim,
                gamma,
                best,
                history,
            )
            write_json(args.output_dir / "partial_summary.json", partial)

            if float(val_metrics["critic_loss"]) < best["val_critic_loss"]:
                best["val_critic_loss"] = float(val_metrics["critic_loss"])
                save_checkpoint(
                    args.output_dir / "best_critic_loss.pt",
                    actor_algo=actor_algo,
                    critic_encoder=critic_encoder,
                    target_critic_encoder=target_critic_encoder,
                    critic=critic,
                    target_critic=target_critic,
                    critic_optimizer=critic_optimizer,
                    args=args,
                    data=data,
                    feature_dim=feature_dim,
                    action_dim=action_dim,
                    gamma=gamma,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            if val_metrics.get("success_auc") is not None and float(val_metrics["success_auc"]) > best["val_success_auc"]:
                best["val_success_auc"] = float(val_metrics["success_auc"])
                save_checkpoint(
                    args.output_dir / "best_success_auc.pt",
                    actor_algo=actor_algo,
                    critic_encoder=critic_encoder,
                    target_critic_encoder=target_critic_encoder,
                    critic=critic,
                    target_critic=target_critic,
                    critic_optimizer=critic_optimizer,
                    args=args,
                    data=data,
                    feature_dim=feature_dim,
                    action_dim=action_dim,
                    gamma=gamma,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            if val_metrics.get("reward_auc") is not None and float(val_metrics["reward_auc"]) > best["val_reward_auc"]:
                best["val_reward_auc"] = float(val_metrics["reward_auc"])
                save_checkpoint(
                    args.output_dir / "best_reward_auc.pt",
                    actor_algo=actor_algo,
                    critic_encoder=critic_encoder,
                    target_critic_encoder=target_critic_encoder,
                    critic=critic,
                    target_critic=target_critic,
                    critic_optimizer=critic_optimizer,
                    args=args,
                    data=data,
                    feature_dim=feature_dim,
                    action_dim=action_dim,
                    gamma=gamma,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            write_json(
                args.output_dir / "partial_summary.json",
                make_summary(
                    args,
                    data,
                    actor_dataset,
                    actor_config,
                    critic_datasets,
                    feature_dim,
                    action_dim,
                    gamma,
                    best,
                    history,
                ),
            )

    final_metrics = {
        split: evaluate_critic_split(
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            teacher_encoder=teacher_encoder,
            dataset=critic_datasets[split],
            batch_size=args.eval_batch_size,
            device=device,
            max_batches=args.max_eval_batches,
            gamma=gamma,
            expectile=args.expectile,
            use_huber=args.use_huber,
            aux_next_pred_weight=args.aux_next_pred_weight,
            aux_next_pred_mode=args.aux_next_pred_mode,
            num_workers=args.eval_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
        for split in ("train", "val", "test")
    }
    save_checkpoint(
        args.output_dir / "last.pt",
        actor_algo=actor_algo,
        critic_encoder=critic_encoder,
        target_critic_encoder=target_critic_encoder,
        critic=critic,
        target_critic=target_critic,
        critic_optimizer=critic_optimizer,
        args=args,
        data=data,
        feature_dim=feature_dim,
        action_dim=action_dim,
        gamma=gamma,
        step=args.total_steps,
        history=history,
        metrics=final_metrics,
    )
    summary = make_summary(
        args,
        data,
        actor_dataset,
        actor_config,
        critic_datasets,
        feature_dim,
        action_dim,
        gamma,
        best,
        history,
    )
    summary["final_metrics"] = final_metrics
    write_json(args.output_dir / "summary.json", summary)
    if writer is not None:
        for split, metrics in final_metrics.items():
            add_tensorboard_scalars(writer, f"final/{split}", metrics, args.total_steps)
        writer.close()
    for dataset in critic_datasets.values():
        dataset.close()
    close_fn = getattr(actor_dataset, "close", None)
    if callable(close_fn):
        close_fn()
    print(json.dumps(jsonable({k: v for k, v in summary.items() if k != "history"}), indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--feature-index", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--demo-dataset", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--rollout-dataset", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--success-dataset", type=Path, default=None)
    parser.add_argument("--failure-dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128, help="critic batch size")
    parser.add_argument("--actor-batch-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--num-workers", type=int, default=4, help="critic DataLoader workers")
    parser.add_argument("--actor-num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--actor-seq-length", type=int, default=None)
    parser.add_argument("--actor-hdf5-cache-mode", type=str, default="low_dim")
    parser.add_argument("--demo-filter-key", type=str, default="")
    parser.add_argument("--success-filter-key", type=str, default="success")
    parser.add_argument("--failure-filter-key", type=str, default="failure")
    parser.add_argument("--actor-demo-weight", type=float, default=1.0)
    parser.add_argument("--actor-success-weight", type=float, default=1.0)
    parser.add_argument("--actor-failure-weight", type=float, default=0.0)
    parser.add_argument("--actor-failure-demo-start-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-failure-sample-start-offset", type=int, default=0)
    parser.add_argument("--actor-failure-anti-failure-label", type=float, default=1.0)
    parser.add_argument("--actor-normalize-weights-by-ds-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-demo-weight", type=float, default=1.0)
    parser.add_argument("--critic-success-weight", type=float, default=1.0)
    parser.add_argument("--critic-failure-weight", type=float, default=1.0)
    parser.add_argument("--critic-normalize-weights-by-source", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--critic-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--actor-disable-lr-scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-weight-decay", type=float, default=0.0)
    parser.add_argument("--critic-dropout", type=float, default=0.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--aux-next-pred-weight", type=float, default=0.0)
    parser.add_argument("--aux-next-pred-mode", choices=("delta", "next"), default="delta")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    args = parser.parse_args()

    for key in ("checkpoint", "feature_index", "demo_dataset", "rollout_dataset", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.success_dataset = (args.success_dataset or args.rollout_dataset).resolve()
    args.failure_dataset = (args.failure_dataset or args.rollout_dataset).resolve()
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.tensorboard_dir is not None:
        args.tensorboard_dir = args.tensorboard_dir.resolve()
    train(args)


if __name__ == "__main__":
    main()
