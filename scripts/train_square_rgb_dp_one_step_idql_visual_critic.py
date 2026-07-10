#!/usr/bin/env python3
"""One-step IDQL with a separate trainable visual critic encoder.

This is the visual-critic variant of train_square_rgb_dp_one_step_idql.py.
The actor and critic no longer share the cached frozen latent:

    actor encoder  -> one-step diffusion actor
    critic encoder -> IQL Q/V heads

The actor encoder is initialized from the pretrained RGB-DP checkpoint and is
frozen by default. The critic encoder is initialized from the same checkpoint
but is updated by the IQL critic loss and, optionally, the next-latent auxiliary
loss. The data index still comes from the one-step IDQL feature cache so the
train / val / test splits, sparse rewards, action normalization statistics, and
mixed-quality data composition exactly match the cached-latent baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from train_square_rgb_dp_one_step_idql import (
    ChunkIQLCritic,
    OneStepDiffusionActor,
    actor_bc_loss,
    add_tensorboard_scalars,
    batch_to_device,
    binary_average_precision,
    binary_roc_auc,
    expectile_loss,
    finite_float,
    make_scheduler,
    q_regression_loss,
    soft_update,
    tensor_stats,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_INDEX = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz"
)
DEFAULT_DEMOS = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_ROLLOUTS = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT / "trained_models/square_rgb_dp_idql/default_reward_one_step_idql_visual_critic"
)
OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def load_feature_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def decode_string_array(values: np.ndarray) -> np.ndarray:
    return values.astype(str)


def get_policy_obs_encoder(policy) -> nn.Module:
    algo = policy.policy
    nets = algo.ema.averaged_model if algo.ema is not None else algo.nets
    return copy.deepcopy(nets["policy"]["obs_encoder"])


def encode_obs(encoder: nn.Module, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    features = TensorUtils.time_distributed(
        {"obs": obs, "goal": None},
        encoder,
        inputs_as_kwargs=True,
    )
    if features.ndim != 3:
        raise ValueError(f"expected encoded features [B,T,D], got {tuple(features.shape)}")
    return features.flatten(start_dim=1)


def process_raw_obs(raw_obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    obs = TensorUtils.to_device(raw_obs, device)
    obs = TensorUtils.to_float(obs)
    return ObsUtils.process_obs_dict(obs)


class IndexedRawTransitionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        *,
        split: str,
        demo_dataset: Path,
        rollout_dataset: Path,
        observation_horizon: int,
        reward_scale: float,
        normalize_actions: bool,
    ):
        split_mask = decode_string_array(data["split"]) == split
        self.indices = np.flatnonzero(split_mask)
        self.demo_dataset = Path(demo_dataset)
        self.rollout_dataset = Path(rollout_dataset)
        self.observation_horizon = int(observation_horizon)
        self.steps = data["steps"][self.indices].astype(np.int64)
        self.next_steps = data["next_steps"][self.indices].astype(np.int64)
        self.source = decode_string_array(data["source"])[self.indices]
        self.demo = decode_string_array(data["demo"])[self.indices]
        actions = data["action_chunks"][self.indices].astype(np.float32)
        self.action_mean = data["action_mean"].astype(np.float32)
        self.action_std = np.maximum(data["action_std"].astype(np.float32), 1e-6)
        if normalize_actions:
            actions = (actions - self.action_mean.reshape(1, 1, -1)) / self.action_std.reshape(1, 1, -1)
        self.actions = actions.astype(np.float32)
        self.raw_rewards = data["chunk_returns"][self.indices].astype(np.float32)
        self.rewards = self.raw_rewards * float(reward_scale)
        self.dones = data["dones"][self.indices].astype(np.float32)
        self.success = data["episode_success"][self.indices].astype(np.float32)
        self._files: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return int(len(self.indices))

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def _path_for_source(self, source: str) -> Path:
        if source == "demo":
            return self.demo_dataset
        if source in ("rollout_success", "rollout_failure"):
            return self.rollout_dataset
        raise ValueError(f"unknown source={source}")

    def _file(self, path: Path) -> h5py.File:
        path = path.resolve()
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def _group(self, source: str, demo_key: str) -> h5py.Group:
        path = self._path_for_source(source)
        return self._file(path)[f"data/{demo_key}"]

    def _obs_at(self, group: h5py.Group, step: int) -> dict[str, np.ndarray]:
        n = int(group.attrs["num_samples"])
        start = int(step) - self.observation_horizon + 1
        indices = [min(max(j, 0), n - 1) for j in range(start, int(step) + 1)]
        obs = {}
        for key in OBS_KEYS:
            dataset = group[f"obs/{key}"]
            obs[key] = np.stack([np.asarray(dataset[j]) for j in indices], axis=0)
        return obs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source = str(self.source[idx])
        demo_key = str(self.demo[idx])
        group = self._group(source, demo_key)
        return {
            "obs": self._obs_at(group, int(self.steps[idx])),
            "next_obs": self._obs_at(group, int(self.next_steps[idx])),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "raw_rewards": torch.tensor(self.raw_rewards[idx], dtype=torch.float32),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32),
            "success": torch.tensor(self.success[idx], dtype=torch.float32),
        }


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {}
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
        **loader_kwargs,
    )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def raw_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "obs": process_raw_obs(batch["obs"], device),
        "next_obs": process_raw_obs(batch["next_obs"], device),
        "actions": batch["actions"].to(device),
        "rewards": batch["rewards"].to(device),
        "raw_rewards": batch["raw_rewards"].to(device),
        "dones": batch["dones"].to(device),
        "success": batch["success"].to(device),
    }


def next_prediction_teacher_target(
    teacher_encoder: nn.Module,
    obs: dict[str, torch.Tensor],
    next_obs: dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    with torch.no_grad():
        teacher_obs = encode_obs(teacher_encoder, obs)
        teacher_next = encode_obs(teacher_encoder, next_obs)
    if mode == "delta":
        return teacher_next - teacher_obs
    if mode == "next":
        return teacher_next
    raise ValueError(f"unknown aux_next_pred_mode={mode}")


def compute_visual_iql_losses(
    *,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    teacher_encoder: nn.Module,
    batch: dict[str, Any],
    gamma: float,
    expectile: float,
    use_huber: bool,
    aux_next_pred_weight: float,
    aux_next_pred_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    actions = batch["actions"]
    rewards = batch["rewards"].unsqueeze(-1)
    dones = batch["dones"].unsqueeze(-1)

    with torch.no_grad():
        target_obs_z = encode_obs(target_critic_encoder, obs)
        q_for_v = target_critic.q_min(target_obs_z, actions)
        next_z_for_backup = encode_obs(critic_encoder, next_obs)
        next_v = critic.value(next_z_for_backup)  # TODO: using target v
        q_backup = rewards + (1.0 - dones) * float(gamma) * next_v

    obs_z = encode_obs(critic_encoder, obs)
    v = critic.value(obs_z)
    v_loss = expectile_loss(q_for_v - v, float(expectile)).mean()

    q1, q2 = critic.q_values(obs_z, actions)
    q1_loss = q_regression_loss(q1, q_backup, use_huber)
    q2_loss = q_regression_loss(q2, q_backup, use_huber)
    iql_loss = q1_loss + q2_loss + v_loss

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

    loss = iql_loss + next_pred_weighted_loss
    with torch.no_grad():
        q_min = torch.minimum(q1, q2)
        adv = q_for_v - v
    return loss, {
        "critic_loss": loss.detach(),
        "iql_loss": iql_loss.detach(),
        "q1_loss": q1_loss.detach(),
        "q2_loss": q2_loss.detach(),
        "v_loss": v_loss.detach(),
        "next_pred_loss": next_pred_loss.detach(),
        "next_pred_weighted_loss": next_pred_weighted_loss.detach(),
        "q_mean": q_min.mean().detach(),
        "v_mean": v.mean().detach(),
        "adv_mean": adv.mean().detach(),
        "reward_mean": rewards.mean().detach(),
    }, obs_z


def dataset_stats(datasets: dict[str, IndexedRawTransitionDataset]) -> dict:
    result = {}
    for split, dataset in datasets.items():
        result[split] = {
            "reward": tensor_stats(dataset.raw_rewards),
            "reward_scaled": tensor_stats(dataset.rewards),
            "success": tensor_stats(dataset.success),
        }
    return result


def make_tensorboard_writer(args: argparse.Namespace):
    if not args.tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard logging requested, but torch.utils.tensorboard is unavailable.", flush=True)
        return None
    log_dir = args.tensorboard_dir or (args.output_dir / "tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to {log_dir}", flush=True)
    return writer


@torch.no_grad()
def evaluate_split(
    *,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    actor_encoder: nn.Module,
    actor: OneStepDiffusionActor,
    teacher_encoder: nn.Module,
    scheduler,
    dataset: IndexedRawTransitionDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    gamma: float,
    expectile: float,
    use_huber: bool,
    aux_next_pred_weight: float,
    aux_next_pred_mode: str,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
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
    actor_encoder.eval()
    actor.eval()
    teacher_encoder.eval()
    q_values = []
    v_values = []
    rewards = []
    success = []
    dones = []
    critic_losses = []
    iql_losses = []
    next_pred_losses = []
    actor_losses = []
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
        actor_z = encode_obs(actor_encoder, batch["obs"])
        actor_loss, _ = actor_bc_loss(actor, scheduler, actor_z, batch["actions"])
        critic_z = encode_obs(critic_encoder, batch["obs"])
        q = critic.q_min(critic_z, batch["actions"]).reshape(-1)
        v = critic.value(critic_z).reshape(-1)
        q_values.append(q.cpu().numpy())
        v_values.append(v.cpu().numpy())
        rewards.append(batch["rewards"].cpu().numpy())
        success.append(batch["success"].cpu().numpy())
        dones.append(batch["dones"].cpu().numpy())
        critic_losses.append(float(critic_loss.cpu()))
        iql_losses.append(float(critic_info["iql_loss"].cpu()))
        next_pred_losses.append(float(critic_info["next_pred_loss"].cpu()))
        actor_losses.append(float(actor_loss.cpu()))
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
        "actor_bc_loss": float(np.mean(actor_losses)),
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
        "val_loss": float("inf"),
        "val_success_auc": -float("inf"),
        "val_reward_auc": -float("inf"),
        "val_actor_loss": float("inf"),
    }
    for record in history:
        val = record.get("val", {})
        if "critic_loss" in val and "actor_bc_loss" in val:
            best["val_loss"] = min(best["val_loss"], float(val["critic_loss"]) + float(val["actor_bc_loss"]))
        if val.get("success_auc") is not None:
            best["val_success_auc"] = max(best["val_success_auc"], float(val["success_auc"]))
        if val.get("reward_auc") is not None:
            best["val_reward_auc"] = max(best["val_reward_auc"], float(val["reward_auc"]))
        if val.get("actor_bc_loss") is not None:
            best["val_actor_loss"] = min(best["val_actor_loss"], float(val["actor_bc_loss"]))
    return best


def save_checkpoint(
    path: Path,
    *,
    actor_encoder: nn.Module,
    critic_encoder: nn.Module,
    target_critic_encoder: nn.Module,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    actor: OneStepDiffusionActor,
    target_actor: OneStepDiffusionActor,
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    feature_dim: int,
    action_dim: int,
    gamma: float,
    step: int,
    history: list[dict],
    metrics: dict,
    critic_optimizer: torch.optim.Optimizer | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "visual_critic_idql": True,
        "actor_encoder": actor_encoder.state_dict(),
        "critic_encoder": critic_encoder.state_dict(),
        "target_critic_encoder": target_critic_encoder.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "actor": actor.state_dict(),
        "target_actor": target_actor.state_dict(),
        "args": vars(args),
        "feature_dim": int(feature_dim),
        "action_dim": int(action_dim),
        "chunk_horizon": 1,
        "observation_horizon": int(args.observation_horizon),
        "gamma": float(gamma),
        "action_mean": data["action_mean"].astype(np.float32),
        "action_std": data["action_std"].astype(np.float32),
        "normalize_actions": bool(args.normalize_actions),
        "features": str(args.feature_index),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "actor_encoder_trainable": bool(args.train_actor_encoder),
        "critic_encoder_trainable": True,
        "aux_next_pred_enabled": bool(float(args.aux_next_pred_weight) > 0.0),
        "aux_next_pred_mode": str(args.aux_next_pred_mode),
        "aux_next_pred_weight": float(args.aux_next_pred_weight),
        "step": int(step),
        "history": history,
        "metrics": metrics,
    }
    if critic_optimizer is not None:
        checkpoint["critic_optimizer"] = critic_optimizer.state_dict()
    if actor_optimizer is not None:
        checkpoint["actor_optimizer"] = actor_optimizer.state_dict()
    torch.save(checkpoint, path)


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


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
            f"visual critic trainer requires one-step feature index; got chunk_horizon={int(data['chunk_horizon'])}"
        )
    gamma = float(data["gamma"])
    action_dim = int(data["action_dim"])
    observation_horizon = int(args.observation_horizon)
    if "observation_horizon" in data and int(data["observation_horizon"]) != observation_horizon:
        raise ValueError(
            f"feature index observation_horizon={int(data['observation_horizon'])}, "
            f"but args.observation_horizon={observation_horizon}"
        )

    dp_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
        verbose=False,
    )
    dp_policy.start_episode()
    actor_encoder = get_policy_obs_encoder(dp_policy).to(device)
    critic_encoder = get_policy_obs_encoder(dp_policy).to(device)
    target_critic_encoder = copy.deepcopy(critic_encoder).to(device)
    teacher_encoder = get_policy_obs_encoder(dp_policy).to(device)
    teacher_encoder.eval().requires_grad_(False)
    if not args.train_actor_encoder:
        actor_encoder.eval().requires_grad_(False)
    feature_dim = int(actor_encoder.output_shape()[0]) * observation_horizon

    datasets = {
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
    stats = dataset_stats(datasets)
    write_summary(args.output_dir / "reward_decomposition_stats.json", stats)
    print(json.dumps({"reward_decomposition_stats": stats}, indent=2), flush=True)

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
    actor = OneStepDiffusionActor(
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.actor_hidden_dims),
        time_dim=args.time_dim,
        dropout=args.actor_dropout,
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_actor.eval().requires_grad_(False)
    scheduler = make_scheduler(args)

    critic_params = [
        {"params": critic_encoder.parameters(), "lr": args.critic_encoder_lr},
        {"params": critic.parameters(), "lr": args.critic_lr},
    ]
    critic_optimizer = torch.optim.AdamW(
        critic_params,
        lr=args.critic_lr,
        weight_decay=args.critic_weight_decay,
    )
    actor_params = [{"params": actor.parameters(), "lr": args.actor_lr}]
    if args.train_actor_encoder:
        actor_params.insert(0, {"params": actor_encoder.parameters(), "lr": args.actor_encoder_lr})
    actor_optimizer = torch.optim.AdamW(
        actor_params,
        lr=args.actor_lr,
        weight_decay=args.actor_weight_decay,
    )

    history: list[dict] = []
    start_step = 0
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
        print(f"Resuming visual-critic IDQL from {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        actor_encoder.load_state_dict(checkpoint["actor_encoder"])
        critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        target_critic_encoder.load_state_dict(checkpoint.get("target_critic_encoder", checkpoint["critic_encoder"]))
        critic.load_state_dict(checkpoint["critic"])
        target_critic.load_state_dict(checkpoint.get("target_critic", checkpoint["critic"]))
        actor.load_state_dict(checkpoint["actor"])
        target_actor.load_state_dict(checkpoint.get("target_actor", checkpoint["actor"]))
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "actor_optimizer" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))

    writer = make_tensorboard_writer(args)
    if writer is not None:
        writer.add_text("config/feature_index", str(args.feature_index), 0)
        writer.add_text("config/checkpoint", str(args.checkpoint), 0)
        add_tensorboard_scalars(writer, "data", {f"{k}/{kk}/{kkk}": v for k, vv in stats.items() for kk, vvv in vv.items() for kkk, v in vvv.items()}, 0)
        writer.flush()

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    train_loader = make_loader(
        datasets["train"],
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    iterator = cycle(train_loader)
    best = best_from_history(history)
    for record in history:
        val = record.get("val", {})
        if "critic_loss" in val and "actor_bc_loss" in val:
            best["val_loss"] = min(
                best["val_loss"],
                float(val["critic_loss"]) + args.actor_selection_weight * float(val["actor_bc_loss"]),
            )

    for step in range(start_step + 1, args.total_steps + 1):
        raw_batch = next(iterator)
        batch = raw_batch_to_device(raw_batch, device)
        critic_encoder.train()
        critic.train()
        if args.train_actor_encoder:
            actor_encoder.train()
        else:
            actor_encoder.eval()
        actor.train()

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

        actor_optimizer.zero_grad(set_to_none=True)
        if args.train_actor_encoder:
            actor_z = encode_obs(actor_encoder, batch["obs"])
        else:
            with torch.no_grad():
                actor_z = encode_obs(actor_encoder, batch["obs"])
        actor_loss, actor_info = actor_bc_loss(actor, scheduler, actor_z, batch["actions"])
        actor_loss.backward()
        actor_params_for_clip = list(actor.parameters())
        if args.train_actor_encoder:
            actor_params_for_clip += list(actor_encoder.parameters())
        actor_grad = torch.nn.utils.clip_grad_norm_(actor_params_for_clip, args.grad_clip)
        actor_optimizer.step()
        soft_update(target_actor, actor, args.actor_target_tau)

        if step % args.log_every == 0:
            payload = {
                "step": int(step),
                **{k: float(v.detach().cpu()) for k, v in critic_info.items()},
                **{k: float(v.detach().cpu()) for k, v in actor_info.items()},
                "critic_grad_norm": float(critic_grad),
                "actor_grad_norm": float(actor_grad),
                "critic_lr": float(critic_optimizer.param_groups[-1]["lr"]),
                "critic_encoder_lr": float(critic_optimizer.param_groups[0]["lr"]),
                "actor_lr": float(actor_optimizer.param_groups[-1]["lr"]),
            }
            if args.train_actor_encoder:
                payload["actor_encoder_lr"] = float(actor_optimizer.param_groups[0]["lr"])
            print(json.dumps(payload, sort_keys=True), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "train", {k: v for k, v in payload.items() if k != "step"}, step)
                writer.flush()

        if step % args.eval_every == 0 or step == args.total_steps:
            val_metrics = evaluate_split(
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                actor_encoder=actor_encoder,
                actor=actor,
                teacher_encoder=teacher_encoder,
                scheduler=scheduler,
                dataset=datasets["val"],
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
            test_metrics = evaluate_split(
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                actor_encoder=actor_encoder,
                actor=actor,
                teacher_encoder=teacher_encoder,
                scheduler=scheduler,
                dataset=datasets["test"],
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
            record = {"step": int(step), "val": val_metrics, "test": test_metrics}
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "val", val_metrics, step)
                add_tensorboard_scalars(writer, "test", test_metrics, step)
                writer.flush()

            metrics = {"val": val_metrics, "test": test_metrics}
            save_checkpoint(
                args.output_dir / "latest.pt",
                actor_encoder=actor_encoder,
                critic_encoder=critic_encoder,
                target_critic_encoder=target_critic_encoder,
                critic=critic,
                target_critic=target_critic,
                actor=actor,
                target_actor=target_actor,
                args=args,
                data=data,
                feature_dim=feature_dim,
                action_dim=action_dim,
                gamma=gamma,
                step=step,
                history=history,
                metrics=metrics,
                critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer,
            )
            partial = make_summary(args, data, datasets, feature_dim, action_dim, gamma, best, history)
            write_summary(args.output_dir / "partial_summary.json", partial)

            combined_val_loss = float(val_metrics["critic_loss"]) + args.actor_selection_weight * float(val_metrics["actor_bc_loss"])
            if combined_val_loss < best["val_loss"]:
                best["val_loss"] = combined_val_loss
                save_checkpoint(args.output_dir / "best_loss.pt", actor_encoder=actor_encoder, critic_encoder=critic_encoder, target_critic_encoder=target_critic_encoder, critic=critic, target_critic=target_critic, actor=actor, target_actor=target_actor, args=args, data=data, feature_dim=feature_dim, action_dim=action_dim, gamma=gamma, step=step, history=history, metrics=metrics)
            if val_metrics.get("success_auc") is not None and float(val_metrics["success_auc"]) > best["val_success_auc"]:
                best["val_success_auc"] = float(val_metrics["success_auc"])
                save_checkpoint(args.output_dir / "best_success_auc.pt", actor_encoder=actor_encoder, critic_encoder=critic_encoder, target_critic_encoder=target_critic_encoder, critic=critic, target_critic=target_critic, actor=actor, target_actor=target_actor, args=args, data=data, feature_dim=feature_dim, action_dim=action_dim, gamma=gamma, step=step, history=history, metrics=metrics)
            if val_metrics.get("reward_auc") is not None and float(val_metrics["reward_auc"]) > best["val_reward_auc"]:
                best["val_reward_auc"] = float(val_metrics["reward_auc"])
                save_checkpoint(args.output_dir / "best_reward_auc.pt", actor_encoder=actor_encoder, critic_encoder=critic_encoder, target_critic_encoder=target_critic_encoder, critic=critic, target_critic=target_critic, actor=actor, target_actor=target_actor, args=args, data=data, feature_dim=feature_dim, action_dim=action_dim, gamma=gamma, step=step, history=history, metrics=metrics)
            if float(val_metrics["actor_bc_loss"]) < best["val_actor_loss"]:
                best["val_actor_loss"] = float(val_metrics["actor_bc_loss"])
                save_checkpoint(args.output_dir / "best_actor_loss.pt", actor_encoder=actor_encoder, critic_encoder=critic_encoder, target_critic_encoder=target_critic_encoder, critic=critic, target_critic=target_critic, actor=actor, target_actor=target_actor, args=args, data=data, feature_dim=feature_dim, action_dim=action_dim, gamma=gamma, step=step, history=history, metrics=metrics)
            write_summary(args.output_dir / "partial_summary.json", make_summary(args, data, datasets, feature_dim, action_dim, gamma, best, history))

    final_metrics = {
        split: evaluate_split(
            critic_encoder=critic_encoder,
            target_critic_encoder=target_critic_encoder,
            critic=critic,
            target_critic=target_critic,
            actor_encoder=actor_encoder,
            actor=actor,
            teacher_encoder=teacher_encoder,
            scheduler=scheduler,
            dataset=datasets[split],
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
        actor_encoder=actor_encoder,
        critic_encoder=critic_encoder,
        target_critic_encoder=target_critic_encoder,
        critic=critic,
        target_critic=target_critic,
        actor=actor,
        target_actor=target_actor,
        args=args,
        data=data,
        feature_dim=feature_dim,
        action_dim=action_dim,
        gamma=gamma,
        step=args.total_steps,
        history=history,
        metrics=final_metrics,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
    )
    summary = make_summary(args, data, datasets, feature_dim, action_dim, gamma, best, history)
    summary["final_metrics"] = final_metrics
    write_summary(args.output_dir / "summary.json", summary)
    if writer is not None:
        for split, metrics in final_metrics.items():
            add_tensorboard_scalars(writer, f"final/{split}", metrics, args.total_steps)
        writer.close()
    for dataset in datasets.values():
        dataset.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)
    return summary


def make_summary(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    datasets: dict[str, IndexedRawTransitionDataset],
    feature_dim: int,
    action_dim: int,
    gamma: float,
    best: dict,
    history: list[dict],
) -> dict:
    return {
        "feature_index": str(args.feature_index),
        "output_dir": str(args.output_dir),
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "num_train": len(datasets["train"]),
        "num_val": len(datasets["val"]),
        "num_test": len(datasets["test"]),
        "feature_dim": int(feature_dim),
        "action_dim": int(action_dim),
        "gamma": float(gamma),
        "visual_critic_idql": True,
        "actor_encoder_trainable": bool(args.train_actor_encoder),
        "critic_encoder_trainable": True,
        "loader": {
            "num_workers": int(args.num_workers),
            "eval_num_workers": int(args.eval_num_workers),
            "pin_memory": bool(args.pin_memory),
            "prefetch_factor": int(args.prefetch_factor),
            "persistent_workers": bool(args.persistent_workers),
        },
        "normalize_actions": bool(args.normalize_actions),
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
            "best_loss": str(args.output_dir / "best_loss.pt"),
            "best_success_auc": str(args.output_dir / "best_success_auc.pt"),
            "best_reward_auc": str(args.output_dir / "best_reward_auc.pt"),
            "best_actor_loss": str(args.output_dir / "best_actor_loss.pt"),
            "last": str(args.output_dir / "last.pt"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth")
    parser.add_argument("--feature-index", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--demo-dataset", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--rollout-dataset", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--critic-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--actor-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--train-actor-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--critic-weight-decay", type=float, default=0.0)
    parser.add_argument("--actor-weight-decay", type=float, default=0.0)
    parser.add_argument("--critic-dropout", type=float, default=0.0)
    parser.add_argument("--actor-dropout", type=float, default=0.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--actor-target-tau", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--aux-next-pred-weight", type=float, default=0.0)
    parser.add_argument("--aux-next-pred-mode", choices=("delta", "next"), default="delta")
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--clip-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--actor-selection-weight", type=float, default=1.0)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    args = parser.parse_args()

    for key in ("checkpoint", "feature_index", "demo_dataset", "rollout_dataset", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.tensorboard_dir is not None:
        args.tensorboard_dir = args.tensorboard_dir.resolve()
    train(args)


if __name__ == "__main__":
    main()
