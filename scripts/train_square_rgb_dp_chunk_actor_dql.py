#!/usr/bin/env python3
"""Hybrid Square RGB-DP chunk actor + one-step Diffusion Q-learning.

This is the DQL counterpart to ``train_square_rgb_dp_chunk_actor_iql.py``.
The actor is the pretrained RGB DiffusionPolicy chunk model and is optimized by
the usual diffusion BC loss plus a Q-maximization term through differentiable
sampling. The critic is a visual one-step twin-Q critic trained with TD targets.

At evaluation time the checkpoint is intentionally compatible with
``ACTOR_SOURCE=hybrid_dp_chunk_actor`` in ``eval_square_rgb_dp_one_step_idql.py``:
the actor proposes chunks, and the critic scores only the first action.
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
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from train_square_rgb_dp_chunk_actor_iql import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DEMOS,
    DEFAULT_FEATURE_INDEX,
    DEFAULT_ROLLOUTS,
    actor_trainability_summary,
    build_actor_loader,
    configure_actor_optimizer,
    cycle,
    initialize_actor_from_deployed_ema,
    jsonable,
    make_transition_train_loader,
    source_counts,
    write_json,
)
from train_square_rgb_dp_one_step_idql import (
    ChunkIQLCritic,
    add_tensorboard_scalars,
    binary_average_precision,
    binary_roc_auc,
    q_regression_loss,
    soft_update,
)
from train_square_rgb_dp_one_step_idql_visual_critic import (
    IndexedRawTransitionDataset,
    dataset_stats,
    encode_obs,
    get_policy_obs_encoder,
    make_loader,
    make_tensorboard_writer,
    next_prediction_teacher_target,
    raw_batch_to_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "trained_models/square_rgb_dp_dql_visual/default_reward_dp_chunk_actor_dql"


def flat_stat(stats: dict, key: str, name: str) -> np.ndarray:
    value = np.asarray(stats[key][name], dtype=np.float32)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value.reshape(-1)


def build_action_space(
    *,
    actor_algo,
    dp_ckpt: dict,
    data: dict[str, np.ndarray],
    action_std_floor: float,
    device: torch.device,
) -> dict[str, Any]:
    action_dim = int(data["action_dim"])
    action_keys = list(actor_algo.global_config.train.action_keys)
    dp_stats = dp_ckpt.get("action_normalization_stats")
    if dp_stats is None:
        dp_scale = np.ones((action_dim,), dtype=np.float32)
        dp_offset = np.zeros((action_dim,), dtype=np.float32)
        dp_source = "none_identity"
    else:
        scale_parts = []
        offset_parts = []
        for key in action_keys:
            if key not in dp_stats:
                raise KeyError(f"DP checkpoint action_normalization_stats missing key {key!r}")
            scale_parts.append(flat_stat(dp_stats, key, "scale"))
            offset_parts.append(flat_stat(dp_stats, key, "offset"))
        dp_scale = np.concatenate(scale_parts, axis=0).astype(np.float32)
        dp_offset = np.concatenate(offset_parts, axis=0).astype(np.float32)
        dp_source = "pretrained_dp_checkpoint_action_normalization_stats"
    if dp_scale.shape[0] != action_dim or dp_offset.shape[0] != action_dim:
        raise ValueError(
            f"DP action stats dimension mismatch: scale={dp_scale.shape}, "
            f"offset={dp_offset.shape}, expected action_dim={action_dim}"
        )
    critic_mean = data["action_mean"].astype(np.float32)
    critic_std_raw = data["action_std"].astype(np.float32)
    critic_std = np.maximum(critic_std_raw, float(action_std_floor)).astype(np.float32)
    return {
        "dp_scale": torch.as_tensor(dp_scale, device=device, dtype=torch.float32),
        "dp_offset": torch.as_tensor(dp_offset, device=device, dtype=torch.float32),
        "critic_mean": torch.as_tensor(critic_mean, device=device, dtype=torch.float32),
        "critic_std": torch.as_tensor(critic_std, device=device, dtype=torch.float32),
        "critic_mean_np": critic_mean,
        "critic_std_np": critic_std,
        "critic_std_raw_np": critic_std_raw,
        "critic_action_std_floor": float(action_std_floor),
        "dp_source": dp_source,
        "dp_is_identity": bool(np.allclose(dp_scale, 1.0) and np.allclose(dp_offset, 0.0)),
        "critic_std_min_raw": float(np.min(critic_std_raw)),
        "critic_std_min_effective": float(np.min(critic_std)),
    }


def module_requires_grad(module: nn.Module, requires_grad: bool) -> list[bool]:
    old = [param.requires_grad for param in module.parameters()]
    module.requires_grad_(requires_grad)
    return old


def restore_requires_grad(module: nn.Module, old: list[bool]) -> None:
    for param, requires_grad in zip(module.parameters(), old):
        param.requires_grad_(requires_grad)


def prepare_actor_batch(actor_policy, raw_batch: dict, obs_normalization_stats) -> dict:
    actor_algo = actor_policy.policy
    batch = actor_algo.process_batch_for_training(raw_batch)
    return actor_algo.postprocess_batch_for_training(
        batch,
        obs_normalization_stats=obs_normalization_stats,
    )


def actor_bc_loss_from_processed_batch(actor_algo, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    actions = batch["actions"]
    batch_size = actions.shape[0]
    inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
    for key in actor_algo.obs_shapes:
        if inputs["obs"][key].ndim - 2 != len(actor_algo.obs_shapes[key]):
            raise ValueError(f"actor obs key={key} has wrong shape {tuple(inputs['obs'][key].shape)}")

    obs_cond = actor_algo._encode_obs(inputs, actor_algo.nets)
    obs_cond, condition_stats = actor_algo._apply_success_condition(
        obs_cond,
        nets=actor_algo.nets,
        batch=batch,
        validate=False,
    )
    noise = torch.randn_like(actions)
    timesteps = torch.randint(
        0,
        actor_algo.noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=actor_algo.device,
    ).long()
    noisy_actions = actor_algo.noise_scheduler.add_noise(actions, noise, timesteps)
    noise_pred = actor_algo.nets["policy"]["noise_pred_net"](
        noisy_actions,
        timesteps,
        global_cond=obs_cond,
    )
    per_sample = F.mse_loss(noise_pred, noise, reduction="none").flatten(start_dim=1).mean(dim=1)
    info = {
        "bc_loss": per_sample.mean().detach(),
        "bc_energy_mean": per_sample.mean().detach(),
        "bc_energy_std": per_sample.std(unbiased=False).detach(),
    }
    if condition_stats is not None:
        for key, value in condition_stats.items():
            info[f"success_condition_{key}"] = value.detach()
    return per_sample.mean(), info


def scheduler_timesteps(actor_algo, num_inference_steps: int) -> torch.Tensor:
    scheduler = actor_algo.noise_scheduler
    train_steps = int(scheduler.config.num_train_timesteps)
    steps = max(1, min(int(num_inference_steps), train_steps))
    step_ratio = max(train_steps // steps, 1)
    timesteps = torch.arange(
        0,
        train_steps,
        step_ratio,
        device=actor_algo.device,
        dtype=torch.long,
    ).flip(0).contiguous()[:steps]
    scheduler.num_inference_steps = int(len(timesteps))
    scheduler.timesteps = timesteps
    return timesteps


def subset_obs(obs: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {key: value[:count] for key, value in obs.items()}


def repeat_obs(obs: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    return {
        key: value.repeat_interleave(int(repeats), dim=0)
        for key, value in obs.items()
    }


def sample_dp_action_chunks(
    *,
    actor_algo,
    obs: dict[str, torch.Tensor],
    nets: nn.Module,
    num_inference_steps: int,
    clip_actions: bool,
) -> torch.Tensor:
    """Differentiably sample the executable DP action chunk [B, Ta, A]."""
    inputs = {"obs": obs, "goal": None}
    obs_cond = actor_algo._encode_obs(inputs, nets)
    obs_cond, _ = actor_algo._apply_success_condition(
        obs_cond,
        nets=nets,
        success_condition=torch.ones(obs_cond.shape[0], device=actor_algo.device),
        condition_mask=torch.ones(obs_cond.shape[0], device=actor_algo.device),
        validate=True,
    )
    horizon = actor_algo.algo_config.horizon
    prediction_horizon = int(horizon.prediction_horizon)
    action_horizon = int(horizon.action_horizon)
    action_dim = int(actor_algo.ac_dim)
    trajectory = torch.randn(
        (obs_cond.shape[0], prediction_horizon, action_dim),
        device=actor_algo.device,
    )
    for timestep in scheduler_timesteps(actor_algo, num_inference_steps):
        noise_pred = nets["policy"]["noise_pred_net"](
            sample=trajectory,
            timestep=timestep,
            global_cond=obs_cond,
        )
        trajectory = actor_algo.noise_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=trajectory,
        ).prev_sample
    start = int(horizon.observation_horizon) - 1
    chunk = trajectory[:, start : start + action_horizon, :]
    if clip_actions:
        chunk = chunk.clamp(-1.0, 1.0)
    return chunk


def actor_action_to_critic_action(
    actions: torch.Tensor,
    action_space: dict[str, Any],
    normalize_actions: bool,
) -> torch.Tensor:
    """Convert DP-normalized sampled actions into the critic action space."""
    dtype = actions.dtype
    scale = action_space["dp_scale"].to(device=actions.device, dtype=dtype).view(1, 1, -1)
    offset = action_space["dp_offset"].to(device=actions.device, dtype=dtype).view(1, 1, -1)
    raw_actions = actions * scale + offset
    if not normalize_actions:
        return raw_actions
    mean = action_space["critic_mean"].to(device=actions.device, dtype=dtype).view(1, 1, -1)
    std = action_space["critic_std"].to(device=actions.device, dtype=dtype).view(1, 1, -1)
    return (raw_actions - mean) / std


def normalize_obs_with_stats(
    obs: dict[str, torch.Tensor],
    obs_normalization_stats,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if obs_normalization_stats is None:
        return obs
    stats = TensorUtils.to_float(
        TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), device)
    )
    return ObsUtils.normalize_dict(obs, normalization_stats=stats)


def raw_batch_to_device_consistent(
    batch: dict[str, Any],
    device: torch.device,
    obs_normalization_stats,
) -> dict[str, Any]:
    processed = raw_batch_to_device(batch, device)
    processed["obs"] = normalize_obs_with_stats(processed["obs"], obs_normalization_stats, device)
    processed["next_obs"] = normalize_obs_with_stats(processed["next_obs"], obs_normalization_stats, device)
    return processed


@torch.no_grad()
def target_actor_first_action(
    *,
    actor_algo,
    next_obs: dict[str, torch.Tensor],
    action_space: dict[str, Any],
    normalize_actions: bool,
    num_inference_steps: int,
    clip_actions: bool,
    num_candidates: int,
) -> torch.Tensor:
    nets = actor_algo.ema.averaged_model if actor_algo.ema is not None else actor_algo.nets
    if int(num_candidates) > 1:
        next_obs = repeat_obs(next_obs, int(num_candidates))
    chunks = sample_dp_action_chunks(
        actor_algo=actor_algo,
        obs=next_obs,
        nets=nets,
        num_inference_steps=num_inference_steps,
        clip_actions=clip_actions,
    )
    first = actor_action_to_critic_action(chunks[:, :1, :], action_space, normalize_actions)
    return first


def compute_dql_critic_loss(
    *,
    actor_algo,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    action_space: dict[str, Any],
    batch: dict[str, Any],
    gamma: float,
    normalize_actions: bool,
    use_huber: bool,
    num_target_candidates: int,
    num_inference_steps: int,
    clip_actions: bool,
    aux_next_pred_weight: float,
    aux_next_pred_mode: str,
    teacher_encoder: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    actions = batch["actions"]
    rewards = batch["rewards"].unsqueeze(-1)
    dones = batch["dones"].unsqueeze(-1)

    with torch.no_grad():
        next_actions = target_actor_first_action(
            actor_algo=actor_algo,
            next_obs=next_obs,
            action_space=action_space,
            normalize_actions=normalize_actions,
            num_inference_steps=num_inference_steps,
            clip_actions=clip_actions,
            num_candidates=num_target_candidates,
        )
        if int(num_target_candidates) > 1:
            repeated_next_obs = repeat_obs(next_obs, int(num_target_candidates))
            next_z = encode_obs(target_critic_encoder, repeated_next_obs)
            target_q = target_critic.q_min(next_z, next_actions).view(actions.shape[0], int(num_target_candidates))
            target_q = target_q.max(dim=1, keepdim=True)[0]
        else:
            next_z = encode_obs(target_critic_encoder, next_obs)
            target_q = target_critic.q_min(next_z, next_actions)
        backup = rewards + (1.0 - dones) * float(gamma) * target_q

    obs_z = encode_obs(critic_encoder, obs)
    q1, q2 = critic.q_values(obs_z, actions)
    q1_loss = q_regression_loss(q1, backup, use_huber)
    q2_loss = q_regression_loss(q2, backup, use_huber)
    td_loss = q1_loss + q2_loss

    next_pred_loss = q1.new_tensor(0.0)
    next_pred_weighted_loss = q1.new_tensor(0.0)
    if float(aux_next_pred_weight) > 0.0:
        target = next_prediction_teacher_target(
            teacher_encoder,
            obs,
            next_obs,
            aux_next_pred_mode,
        )
        pred = critic.next_prediction(obs_z, actions)
        next_pred_loss = F.mse_loss(pred, target)
        next_pred_weighted_loss = float(aux_next_pred_weight) * next_pred_loss

    loss = td_loss + next_pred_weighted_loss
    with torch.no_grad():
        q_min = torch.minimum(q1, q2)
    return loss, {
        "critic_loss": loss.detach(),
        "td_loss": td_loss.detach(),
        "q1_loss": q1_loss.detach(),
        "q2_loss": q2_loss.detach(),
        "next_pred_loss": next_pred_loss.detach(),
        "next_pred_weighted_loss": next_pred_weighted_loss.detach(),
        "q_mean": q_min.mean().detach(),
        "target_q_mean": target_q.mean().detach(),
        "backup_mean": backup.mean().detach(),
        "reward_mean": rewards.mean().detach(),
    }


def actor_q_loss(
    *,
    actor_algo,
    critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    action_space: dict[str, Any],
    obs: dict[str, torch.Tensor],
    normalize_actions: bool,
    num_inference_steps: int,
    clip_actions: bool,
    q_head: str,
    q_denom_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    chunks = sample_dp_action_chunks(
        actor_algo=actor_algo,
        obs=obs,
        nets=actor_algo.nets,
        num_inference_steps=num_inference_steps,
        clip_actions=clip_actions,
    )
    first = actor_action_to_critic_action(chunks[:, :1, :], action_space, normalize_actions)
    old_encoder = module_requires_grad(critic_encoder, False)
    old_critic = module_requires_grad(critic, False)
    try:
        with torch.no_grad():
            z = encode_obs(critic_encoder, obs)
        q1, q2 = critic.q_values(z, first)
        if q_head == "q1":
            q_loss = -q1.mean() / q2.detach().abs().mean().clamp_min(float(q_denom_floor))
            q_for_log = q1
        elif q_head == "q2":
            q_loss = -q2.mean() / q1.detach().abs().mean().clamp_min(float(q_denom_floor))
            q_for_log = q2
        elif q_head == "min":
            q_min = torch.minimum(q1, q2)
            q_loss = -q_min.mean() / q_min.detach().abs().mean().clamp_min(float(q_denom_floor))
            q_for_log = q_min
        elif q_head == "random":
            if torch.rand((), device=q1.device) > 0.5:
                q_loss = -q1.mean() / q2.detach().abs().mean().clamp_min(float(q_denom_floor))
                q_for_log = q1
            else:
                q_loss = -q2.mean() / q1.detach().abs().mean().clamp_min(float(q_denom_floor))
                q_for_log = q2
        else:
            raise ValueError(f"unknown dql_q_head={q_head}")
    finally:
        restore_requires_grad(critic_encoder, old_encoder)
        restore_requires_grad(critic, old_critic)
    return q_loss, {
        "ql_loss": q_loss.detach(),
        "actor_q_mean": q_for_log.mean().detach(),
        "actor_first_action_abs_mean": first.detach().abs().mean(),
        "actor_q_denom_floor": q_for_log.new_tensor(float(q_denom_floor)),
    }


def dql_actor_train_step(
    *,
    actor_policy,
    raw_batch: dict,
    critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    action_space: dict[str, Any],
    args: argparse.Namespace,
    step: int,
) -> dict[str, float]:
    actor_algo = actor_policy.policy
    actor_algo.set_train()
    batch = prepare_actor_batch(
        actor_policy,
        raw_batch,
        obs_normalization_stats=actor_policy.obs_normalization_stats,
    )
    bc_loss, bc_info = actor_bc_loss_from_processed_batch(actor_algo, batch)

    q_batch_size = min(int(args.dql_q_batch_size), int(next(iter(batch["obs"].values())).shape[0]))
    q_loss = bc_loss.new_tensor(0.0)
    q_info: dict[str, torch.Tensor] = {
        "ql_loss": q_loss.detach(),
        "actor_q_mean": q_loss.detach(),
        "actor_first_action_abs_mean": q_loss.detach(),
    }
    if float(args.dql_eta) != 0.0 and q_batch_size > 0:
        q_obs = subset_obs(batch["obs"], q_batch_size)
        q_loss, q_info = actor_q_loss(
            actor_algo=actor_algo,
            critic_encoder=critic_encoder,
            critic=critic,
            action_space=action_space,
            obs=q_obs,
            normalize_actions=bool(args.normalize_actions),
            num_inference_steps=int(args.dql_num_inference_steps),
            clip_actions=bool(args.dql_clip_actions_for_q),
            q_head=str(args.dql_q_head),
            q_denom_floor=float(args.dql_q_denom_floor),
        )

    actor_loss = float(args.dql_bc_weight) * bc_loss + float(args.dql_eta) * q_loss
    actor_algo.optimizers["policy"].zero_grad(set_to_none=True)
    actor_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(actor_algo.nets.parameters(), float(args.actor_grad_clip))
    actor_algo.optimizers["policy"].step()
    if actor_algo.ema is not None:
        actor_algo._update_ema()
    actor_algo.on_gradient_step()

    info = {
        "actor_loss": float(actor_loss.detach().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
        "ql_loss": float(q_loss.detach().cpu()),
        "actor_grad_norm": float(grad_norm),
        "actor_lr": float(actor_algo.optimizers["policy"].param_groups[-1]["lr"]),
        "dql_eta": float(args.dql_eta),
        "dql_q_batch_size": int(q_batch_size),
        "dql_num_inference_steps": int(args.dql_num_inference_steps),
    }
    for key, value in {**bc_info, **q_info}.items():
        info[key] = float(value.detach().cpu())
    return info


@torch.no_grad()
def evaluate_dql_split(
    *,
    actor_algo,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    teacher_encoder: nn.Module,
    action_space: dict[str, Any],
    dataset: IndexedRawTransitionDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    gamma: float,
    normalize_actions: bool,
    use_huber: bool,
    num_target_candidates: int,
    num_inference_steps: int,
    clip_actions: bool,
    aux_next_pred_weight: float,
    aux_next_pred_mode: str,
    obs_normalization_stats,
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
    rewards = []
    success = []
    dones = []
    losses = []
    td_losses = []
    next_pred_losses = []
    target_q_values = []
    for batch_index, raw_batch in enumerate(loader):
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        batch = raw_batch_to_device_consistent(raw_batch, device, obs_normalization_stats)
        critic_loss, info = compute_dql_critic_loss(
            actor_algo=actor_algo,
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            action_space=action_space,
            batch=batch,
            gamma=gamma,
            normalize_actions=normalize_actions,
            use_huber=use_huber,
            num_target_candidates=num_target_candidates,
            num_inference_steps=num_inference_steps,
            clip_actions=clip_actions,
            aux_next_pred_weight=aux_next_pred_weight,
            aux_next_pred_mode=aux_next_pred_mode,
            teacher_encoder=teacher_encoder,
        )
        z = encode_obs(critic_encoder, batch["obs"])
        q = critic.q_min(z, batch["actions"]).reshape(-1)
        q_values.append(q.cpu().numpy())
        rewards.append(batch["rewards"].cpu().numpy())
        success.append(batch["success"].cpu().numpy())
        dones.append(batch["dones"].cpu().numpy())
        losses.append(float(critic_loss.cpu()))
        td_losses.append(float(info["td_loss"].cpu()))
        next_pred_losses.append(float(info["next_pred_loss"].cpu()))
        target_q_values.append(float(info["target_q_mean"].cpu()))
    q_np = np.concatenate(q_values)
    reward_np = np.concatenate(rewards)
    success_np = np.concatenate(success)
    done_np = np.concatenate(dones)
    return {
        "num_samples": int(len(q_np)),
        "critic_loss": float(np.mean(losses)),
        "td_loss": float(np.mean(td_losses)),
        "next_pred_loss": float(np.mean(next_pred_losses)),
        "q_mean": float(np.mean(q_np)),
        "target_q_mean": float(np.mean(target_q_values)),
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
    action_space: dict[str, Any],
    feature_dim: int,
    action_dim: int,
    gamma: float,
    step: int,
    history: list[dict],
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "diffusion_q_learning": True,
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
        "action_mean": action_space["critic_mean_np"].astype(np.float32),
        "action_std": action_space["critic_std_np"].astype(np.float32),
        "raw_action_std": action_space["critic_std_raw_np"].astype(np.float32),
        "critic_action_std_floor": float(action_space["critic_action_std_floor"]),
        "dp_action_normalization_source": str(action_space["dp_source"]),
        "dp_action_normalization_identity": bool(action_space["dp_is_identity"]),
        "normalize_actions": bool(args.normalize_actions),
        "features": str(args.feature_index),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "actor_initialized_from_dp": True,
        "actor_training_objective": "diffusion_bc_plus_q_maximization",
        "critic_training_objective": "one_step_td_twin_q",
        "actor_trainability": actor_trainability_summary(actor_algo),
        "actor_encoder_trainable": True,
        "critic_encoder_trainable": True,
        "critic_encoder_initialized_from_dp": True,
        "critic_input_mode": "raw_hdf5_observations",
        "critic_feature_index_usage": "transition_metadata_actions_rewards_splits_only",
        "critic_uses_cached_latents": False,
        "feature_index_contains_cached_latents": bool("obs_features" in data),
        "dql_eta": float(args.dql_eta),
        "dql_bc_weight": float(args.dql_bc_weight),
        "dql_num_inference_steps": int(args.dql_num_inference_steps),
        "dql_target_num_candidates": int(args.dql_target_num_candidates),
        "dql_q_head": str(args.dql_q_head),
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
    action_space: dict[str, Any],
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
        "diffusion_q_learning": True,
        "hybrid_dp_chunk_actor_iql": True,
        "feature_index": str(args.feature_index),
        "output_dir": str(args.output_dir),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "actor_data": jsonable(actor_config.train.data),
        "actor_dataset_size": int(len(actor_dataset)),
        "actor_action_normalization_source": getattr(actor_dataset, "actor_action_normalization_source", None),
        "actor_initialized_from_deployed_ema": bool(getattr(args, "actor_initialized_from_deployed_ema", False)),
        "critic_num_train": int(len(critic_datasets["train"])),
        "critic_num_val": int(len(critic_datasets["val"])),
        "critic_num_test": int(len(critic_datasets["test"])),
        "critic_source_counts": {split: source_counts(dataset) for split, dataset in critic_datasets.items()},
        "feature_dim": int(feature_dim),
        "action_dim": int(action_dim),
        "gamma": float(gamma),
        "normalize_actions_for_critic": bool(args.normalize_actions),
        "action_space": {
            "actor_sample_space": "dp_normalized_action_space",
            "critic_action_space": "feature_standardized_raw_action_space"
            if bool(args.normalize_actions)
            else "raw_action_space",
            "dp_action_normalization_source": str(action_space["dp_source"]),
            "dp_action_normalization_identity": bool(action_space["dp_is_identity"]),
            "critic_action_std_floor": float(action_space["critic_action_std_floor"]),
            "critic_std_min_raw": float(action_space["critic_std_min_raw"]),
            "critic_std_min_effective": float(action_space["critic_std_min_effective"]),
        },
        "actor_training_objective": "diffusion_bc_plus_q_maximization",
        "critic_training_objective": "one_step_td_twin_q",
        "dql": {
            "eta": float(args.dql_eta),
            "bc_weight": float(args.dql_bc_weight),
            "q_batch_size": int(args.dql_q_batch_size),
            "num_inference_steps": int(args.dql_num_inference_steps),
            "target_num_candidates": int(args.dql_target_num_candidates),
            "q_head": str(args.dql_q_head),
            "q_denom_floor": float(args.dql_q_denom_floor),
            "clip_actions_for_q": bool(args.dql_clip_actions_for_q),
        },
        "eval_actor_source": "hybrid_dp_chunk_actor",
        "eval_note": "Use actor_source=hybrid_dp_chunk_actor; use selection=argmax because value head is not trained by DQL.",
        "critic_encoder_initialized_from_dp": True,
        "aux_next_pred": {
            "enabled": bool(float(args.aux_next_pred_weight) > 0.0),
            "weight": float(args.aux_next_pred_weight),
            "mode": str(args.aux_next_pred_mode),
            "target": "frozen_dp_teacher",
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
    if float(args.dql_eta) < 0.0:
        raise ValueError(f"dql_eta must be non-negative, got {args.dql_eta}")
    if int(args.dql_q_batch_size) < 0:
        raise ValueError(f"dql_q_batch_size must be non-negative, got {args.dql_q_batch_size}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")

    data = np.load(args.feature_index, allow_pickle=True)
    data = {key: data[key] for key in data.files}
    if int(data["chunk_horizon"]) != 1:
        raise ValueError(f"DQL trainer requires one-step feature index; got chunk_horizon={int(data['chunk_horizon'])}")
    gamma = float(data["gamma"])
    action_dim = int(data["action_dim"])
    observation_horizon = int(args.observation_horizon)

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
    print(json.dumps({"actor_trainability": jsonable(actor_trainability_summary(actor_algo))}, indent=2), flush=True)

    actor_dataset, actor_loader, actor_config = build_actor_loader(
        args=args,
        actor_algo=actor_algo,
        checkpoint_dict=dp_ckpt,
    )
    actor_iterator = cycle(actor_loader)
    action_space = build_action_space(
        actor_algo=actor_algo,
        dp_ckpt=dp_ckpt,
        data=data,
        action_std_floor=float(args.critic_action_std_floor),
        device=device,
    )
    print(
        json.dumps(
            {
                "dql_action_space": {
                    "dp_action_normalization_source": action_space["dp_source"],
                    "dp_action_normalization_identity": action_space["dp_is_identity"],
                    "critic_std_min_raw": action_space["critic_std_min_raw"],
                    "critic_std_min_effective": action_space["critic_std_min_effective"],
                    "critic_action_std_floor": action_space["critic_action_std_floor"],
                    "critic_normalize_actions": bool(args.normalize_actions),
                }
            },
            indent=2,
        ),
        flush=True,
    )

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
            action_std_floor=float(args.critic_action_std_floor),
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
                "critic_source_counts": {split: source_counts(dataset) for split, dataset in critic_datasets.items()},
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
        print(f"Resuming hybrid DP-chunk actor DQL from {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        actor_algo.deserialize(checkpoint["actor_model"], load_optimizers=True)
        configure_actor_optimizer(actor_algo, args.actor_lr, args.actor_disable_lr_scheduler)
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
        raw_critic_batch = next(critic_iterator)
        critic_batch = raw_batch_to_device_consistent(
            raw_critic_batch,
            device,
            actor_policy.obs_normalization_stats,
        )
        critic_encoder.train()
        critic.train()
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss, critic_info = compute_dql_critic_loss(
            actor_algo=actor_algo,
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            action_space=action_space,
            batch=critic_batch,
            gamma=gamma,
            normalize_actions=bool(args.normalize_actions),
            use_huber=bool(args.use_huber),
            num_target_candidates=int(args.dql_target_num_candidates),
            num_inference_steps=int(args.dql_num_inference_steps),
            clip_actions=bool(args.dql_clip_actions_for_q),
            aux_next_pred_weight=float(args.aux_next_pred_weight),
            aux_next_pred_mode=str(args.aux_next_pred_mode),
            teacher_encoder=teacher_encoder,
        )
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            list(critic_encoder.parameters()) + list(critic.parameters()),
            float(args.grad_clip),
        )
        critic_optimizer.step()
        soft_update(target_critic, critic, float(args.target_tau))
        soft_update(target_critic_encoder, critic_encoder, float(args.target_tau))

        raw_actor_batch = next(actor_iterator)
        actor_log = dql_actor_train_step(
            actor_policy=actor_policy,
            raw_batch=raw_actor_batch,
            critic_encoder=critic_encoder,
            critic=critic,
            action_space=action_space,
            args=args,
            step=step,
        )

        if step % args.log_every == 0:
            payload = {
                "step": int(step),
                **actor_log,
                **{k: float(v.detach().cpu()) for k, v in critic_info.items()},
                "critic_grad_norm": float(critic_grad),
                "critic_lr": float(critic_optimizer.param_groups[-1]["lr"]),
                "critic_encoder_lr": float(critic_optimizer.param_groups[0]["lr"]),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "train", {k: v for k, v in payload.items() if k != "step"}, step)
                writer.flush()

        if step % args.eval_every == 0 or step == args.total_steps:
            val_metrics = evaluate_dql_split(
                actor_algo=actor_algo,
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                teacher_encoder=teacher_encoder,
                action_space=action_space,
                dataset=critic_datasets["val"],
                batch_size=args.eval_batch_size,
                device=device,
                max_batches=args.max_eval_batches,
                gamma=gamma,
                normalize_actions=bool(args.normalize_actions),
                use_huber=bool(args.use_huber),
                num_target_candidates=int(args.dql_target_num_candidates),
                num_inference_steps=int(args.dql_num_inference_steps),
                clip_actions=bool(args.dql_clip_actions_for_q),
                aux_next_pred_weight=float(args.aux_next_pred_weight),
                aux_next_pred_mode=str(args.aux_next_pred_mode),
                obs_normalization_stats=actor_policy.obs_normalization_stats,
                num_workers=args.eval_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
            )
            test_metrics = evaluate_dql_split(
                actor_algo=actor_algo,
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                teacher_encoder=teacher_encoder,
                action_space=action_space,
                dataset=critic_datasets["test"],
                batch_size=args.eval_batch_size,
                device=device,
                max_batches=args.max_eval_batches,
                gamma=gamma,
                normalize_actions=bool(args.normalize_actions),
                use_huber=bool(args.use_huber),
                num_target_candidates=int(args.dql_target_num_candidates),
                num_inference_steps=int(args.dql_num_inference_steps),
                clip_actions=bool(args.dql_clip_actions_for_q),
                aux_next_pred_weight=float(args.aux_next_pred_weight),
                aux_next_pred_mode=str(args.aux_next_pred_mode),
                obs_normalization_stats=actor_policy.obs_normalization_stats,
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
                action_space=action_space,
                feature_dim=feature_dim,
                action_dim=action_dim,
                gamma=gamma,
                step=step,
                history=history,
                metrics=metrics,
            )
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
                    action_space=action_space,
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
                    action_space=action_space,
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
                    action_space=action_space,
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
                    action_space,
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
        split: evaluate_dql_split(
            actor_algo=actor_algo,
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            teacher_encoder=teacher_encoder,
            action_space=action_space,
            dataset=critic_datasets[split],
            batch_size=args.eval_batch_size,
            device=device,
            max_batches=args.max_eval_batches,
            gamma=gamma,
            normalize_actions=bool(args.normalize_actions),
            use_huber=bool(args.use_huber),
            num_target_candidates=int(args.dql_target_num_candidates),
            num_inference_steps=int(args.dql_num_inference_steps),
            clip_actions=bool(args.dql_clip_actions_for_q),
            aux_next_pred_weight=float(args.aux_next_pred_weight),
            aux_next_pred_mode=str(args.aux_next_pred_mode),
            obs_normalization_stats=actor_policy.obs_normalization_stats,
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
        action_space=action_space,
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
        action_space,
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
    parser.add_argument("--seed", type=int, default=20260712)
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
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--critic-action-std-floor", type=float, default=1e-3)
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--actor-grad-clip", type=float, default=10.0)
    parser.add_argument("--dql-eta", type=float, default=1.0)
    parser.add_argument("--dql-bc-weight", type=float, default=1.0)
    parser.add_argument("--dql-q-batch-size", type=int, default=8)
    parser.add_argument("--dql-num-inference-steps", type=int, default=5)
    parser.add_argument("--dql-target-num-candidates", type=int, default=1)
    parser.add_argument("--dql-q-head", choices=("random", "q1", "q2", "min"), default="random")
    parser.add_argument("--dql-q-denom-floor", type=float, default=1.0)
    parser.add_argument("--dql-clip-actions-for-q", action=argparse.BooleanOptionalAction, default=True)
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
