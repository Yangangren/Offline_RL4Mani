#!/usr/bin/env python3
"""Paper-faithful one-step IDQL baseline for Square RGB-DP data.

This script intentionally treats offline RL transitions as one-step MDP
transitions:

    (phi(o_t), a_t, r_t, phi(o_{t+1}), done_t)

The critic is standard IQL:

    L_V = E[L_2^tau(Q_target(s,a) - V(s))]
    L_Q = E[(r + gamma * (1-done) * V(s') - Q(s,a))^2]

The behavior actor is a one-step conditional diffusion model trained by the
usual DDPM behavior cloning objective. Policy improvement is not done by AWR;
instead, IDQL extraction samples K actions from the diffusion behavior actor and
selects / resamples using the learned critic at evaluation time.

The observation representation is the frozen RGB-DP encoder feature cache built
by build_square_rgb_dp_chunk_idql_features.py with --chunk-horizon 1 --stride 1.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # TensorBoard is optional for training.
    SummaryWriter = None


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        last = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(last, int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            last = int(hidden)
        layers.append(nn.Linear(last, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        last = int(input_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend([nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            last = hidden
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChunkIQLCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        chunk_horizon: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        aux_next_pred: bool = False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.aux_next_pred = bool(aux_next_pred)
        flat_action_dim = self.action_dim * self.chunk_horizon
        q_input = self.feature_dim + flat_action_dim
        if self.aux_next_pred:
            self.sa_trunk = MLPBackbone(q_input, hidden_dims, dropout=dropout)
            self.q1_head = nn.Linear(self.sa_trunk.output_dim, 1)
            self.q2_head = nn.Linear(self.sa_trunk.output_dim, 1)
            self.next_pred_head = nn.Linear(self.sa_trunk.output_dim, self.feature_dim)
        else:
            self.q1 = MLP(q_input, hidden_dims, dropout=dropout)
            self.q2 = MLP(q_input, hidden_dims, dropout=dropout)
        self.v = MLP(self.feature_dim, hidden_dims, dropout=dropout)

    def flatten_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(actions.shape[0], self.chunk_horizon * self.action_dim)

    def sa_features(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.aux_next_pred:
            raise RuntimeError("state-action trunk is only available when aux_next_pred=True")
        x = torch.cat([obs_features, self.flatten_action(actions)], dim=-1)
        return self.sa_trunk(x)

    def q_values(self, obs_features: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_features, self.flatten_action(actions)], dim=-1)
        if self.aux_next_pred:
            h = self.sa_trunk(x)
            return self.q1_head(h), self.q2_head(h)
        return self.q1(x), self.q2(x)

    def q_min(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.q_values(obs_features, actions)
        return torch.minimum(q1, q2)

    def value(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self.v(obs_features)

    def next_prediction(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.next_pred_head(self.sa_features(obs_features, actions))


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, np.ndarray], split: str, reward_scale: float, normalize_actions: bool):
        split_mask = data["split"].astype(str) == split
        self.indices = np.flatnonzero(split_mask)
        self.obs = data["obs_features"][self.indices].astype(np.float32)
        self.next_obs = data["next_obs_features"][self.indices].astype(np.float32)
        actions = data["action_chunks"][self.indices].astype(np.float32)
        self.action_mean = data["action_mean"].astype(np.float32)
        self.action_std = np.maximum(data["action_std"].astype(np.float32), 1e-6)
        if normalize_actions:
            actions = (actions - self.action_mean.reshape(1, 1, -1)) / self.action_std.reshape(1, 1, -1)
        self.actions = actions.astype(np.float32)
        self.raw_rewards = data["chunk_returns"][self.indices].astype(np.float32)
        self.rewards = self.raw_rewards * float(reward_scale)
        if "default_chunk_returns" in data:
            self.default_rewards = data["default_chunk_returns"][self.indices].astype(np.float32)
        else:
            self.default_rewards = self.raw_rewards.astype(np.float32)
        if "risk_cost" in data:
            self.risk_cost = data["risk_cost"][self.indices].astype(np.float32)
        else:
            self.risk_cost = np.zeros_like(self.raw_rewards, dtype=np.float32)
        if "risk_penalty" in data:
            self.risk_penalty = data["risk_penalty"][self.indices].astype(np.float32)
        else:
            self.risk_penalty = np.zeros_like(self.raw_rewards, dtype=np.float32)
        self.dones = data["dones"][self.indices].astype(np.float32)
        self.success = data["episode_success"][self.indices].astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "raw_rewards": torch.tensor(self.raw_rewards[idx], dtype=torch.float32),
            "default_rewards": torch.tensor(self.default_rewards[idx], dtype=torch.float32),
            "risk_cost": torch.tensor(self.risk_cost[idx], dtype=torch.float32),
            "risk_penalty": torch.tensor(self.risk_penalty[idx], dtype=torch.float32),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32),
            "success": torch.tensor(self.success[idx], dtype=torch.float32),
        }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, float(expectile), 1.0 - float(expectile))
    return weight * diff.square()


def q_regression_loss(q: torch.Tensor, target: torch.Tensor, use_huber: bool) -> torch.Tensor:
    if use_huber:
        return F.smooth_l1_loss(q, target)
    return F.mse_loss(q, target)


def next_prediction_target(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "delta":
        return batch["next_obs"] - batch["obs"]
    if mode == "next":
        return batch["next_obs"]
    raise ValueError(f"unknown aux_next_pred_mode={mode}")


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    num_pos = int(labels.sum())
    num_neg = int((~labels).sum())
    if num_pos == 0 or num_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    pos_rank_sum = float(ranks[labels].sum())
    return float((pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    num_pos = int(labels.sum())
    if num_pos == 0 or num_pos == len(labels):
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.float64)
    precision = tp / (np.arange(len(sorted_labels), dtype=np.float64) + 1.0)
    return float((precision * sorted_labels).sum() / num_pos)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "trained_models/square_rgb_dp_idql/default_reward_one_step_idql_actor_critic"
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -scale)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class OneStepDiffusionActor(nn.Module):
    """Conditional DDPM behavior actor for a single continuous action."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        time_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.time_dim = int(time_dim)
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        layers: list[nn.Module] = []
        last = feature_dim + action_dim + time_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.ndim == 3:
            noisy_actions = noisy_actions.squeeze(1)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(obs_features.shape[0])
        time_emb = self.time_encoder(timesteps.long())
        x = torch.cat([obs_features, noisy_actions, time_emb], dim=-1)
        return self.net(x)


class OneStepFeatureDataset(FeatureDataset):
    def __init__(self, data: dict[str, np.ndarray], split: str, reward_scale: float, normalize_actions: bool):
        super().__init__(data, split, reward_scale, normalize_actions)
        if self.actions.ndim != 3 or self.actions.shape[1] != 1:
            raise ValueError(
                "OneStepFeatureDataset requires action_chunks with shape [N, 1, A]; "
                f"got {self.actions.shape}"
            )


def make_scheduler(args_or_dict: Any) -> DDPMScheduler:
    getter = args_or_dict.get if isinstance(args_or_dict, dict) else lambda k, d=None: getattr(args_or_dict, k, d)
    return DDPMScheduler(
        num_train_timesteps=int(getter("num_diffusion_steps", 100)),
        beta_schedule=str(getter("beta_schedule", "squaredcos_cap_v2")),
        prediction_type="epsilon",
        clip_sample=bool(getter("clip_sample", True)),
    )


def actor_bc_loss(
    actor: OneStepDiffusionActor,
    scheduler: DDPMScheduler,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean = actions.squeeze(1)
    noise = torch.randn_like(clean)
    timesteps = torch.randint(
        low=0,
        high=int(scheduler.config.num_train_timesteps),
        size=(clean.shape[0],),
        device=clean.device,
    ).long()
    noisy = scheduler.add_noise(clean, noise, timesteps)
    pred = actor(obs, noisy, timesteps)
    loss = F.mse_loss(pred, noise)
    return loss, {"actor_bc_loss": loss.detach()}


def compute_iql_losses(
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    batch: dict[str, torch.Tensor],
    gamma: float,
    expectile: float,
    use_huber: bool,
    aux_next_pred_weight: float = 0.0,
    aux_next_pred_mode: str = "delta",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    actions = batch["actions"]
    rewards = batch["rewards"].unsqueeze(-1)
    dones = batch["dones"].unsqueeze(-1)

    with torch.no_grad():
        q_for_v = target_critic.q_min(obs, actions)
        next_v = critic.value(next_obs)
        q_backup = rewards + (1.0 - dones) * float(gamma) * next_v

    v = critic.value(obs)
    v_loss = expectile_loss(q_for_v - v, float(expectile)).mean()

    q1, q2 = critic.q_values(obs, actions)
    q1_loss = q_regression_loss(q1, q_backup, use_huber)
    q2_loss = q_regression_loss(q2, q_backup, use_huber)
    iql_loss = q1_loss + q2_loss + v_loss
    next_pred_loss = q1.new_tensor(0.0)
    next_pred_weighted_loss = q1.new_tensor(0.0)
    if float(aux_next_pred_weight) > 0.0:
        pred_next = critic.next_prediction(obs, actions)
        target_next = next_prediction_target(batch, aux_next_pred_mode).detach()
        next_pred_loss = F.mse_loss(pred_next, target_next)
        next_pred_weighted_loss = float(aux_next_pred_weight) * next_pred_loss
    loss = iql_loss + next_pred_weighted_loss

    with torch.no_grad():
        q_min = torch.minimum(q1, q2)
        adv = q_for_v - v
    info = {
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
    }
    return loss, info


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


def make_loader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
    )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu())
        else:
            value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def add_tensorboard_scalars(writer, prefix: str, scalars: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in scalars.items():
        value = finite_float(value)
        if value is None:
            continue
        writer.add_scalar(f"{prefix}/{key}", value, int(step))


def tensor_stats(array: np.ndarray) -> dict[str, float]:
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def reward_decomposition_stats(datasets: dict[str, torch.utils.data.Dataset]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for split, dataset in datasets.items():
        result[split] = {
            "reward": tensor_stats(dataset.raw_rewards),
            "reward_scaled": tensor_stats(dataset.rewards),
            "default_reward": tensor_stats(dataset.default_rewards),
            "risk_cost": tensor_stats(dataset.risk_cost),
            "risk_penalty": tensor_stats(dataset.risk_penalty),
            "success": tensor_stats(dataset.success),
        }
    return result


def flatten_stats(nested: dict, prefix: str = "") -> dict[str, float]:
    flat = {}
    for key, value in nested.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_stats(value, name))
        else:
            flat[name] = value
    return flat


def batch_reward_stats(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "reward_mean": batch["raw_rewards"].mean(),
        "reward_std": batch["raw_rewards"].std(unbiased=False),
        "reward_scaled_mean": batch["rewards"].mean(),
        "reward_scaled_std": batch["rewards"].std(unbiased=False),
        "default_reward_mean": batch["default_rewards"].mean(),
        "default_reward_std": batch["default_rewards"].std(unbiased=False),
        "risk_cost_mean": batch["risk_cost"].mean(),
        "risk_cost_std": batch["risk_cost"].std(unbiased=False),
        "risk_penalty_mean": batch["risk_penalty"].mean(),
        "risk_penalty_std": batch["risk_penalty"].std(unbiased=False),
    }


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
    critic: ChunkIQLCritic,
    actor: OneStepDiffusionActor,
    scheduler: DDPMScheduler,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
    gamma: float,
    expectile: float,
    use_huber: bool,
    aux_next_pred_weight: float = 0.0,
    aux_next_pred_mode: str = "delta",
) -> dict[str, float | None]:
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    critic.eval()
    actor.eval()
    q_values = []
    v_values = []
    rewards = []
    success = []
    dones = []
    critic_losses = []
    iql_losses = []
    next_pred_losses = []
    actor_losses = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        critic_loss, critic_info = compute_iql_losses(
            critic,
            critic,
            batch,
            gamma=gamma,
            expectile=expectile,
            use_huber=use_huber,
            aux_next_pred_weight=aux_next_pred_weight,
            aux_next_pred_mode=aux_next_pred_mode,
        )
        actor_loss, _ = actor_bc_loss(actor, scheduler, batch["obs"], batch["actions"])
        q = critic.q_min(batch["obs"], batch["actions"]).reshape(-1)
        v = critic.value(batch["obs"]).reshape(-1)
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
    reward_positive = reward_np > 0.0
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
        "reward_auc": binary_roc_auc(reward_positive, q_np),
        "reward_ap": binary_average_precision(reward_positive, q_np),
    }


def load_feature_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def save_checkpoint(
    path: Path,
    *,
    critic: ChunkIQLCritic,
    target_critic: ChunkIQLCritic,
    actor: OneStepDiffusionActor,
    target_actor: OneStepDiffusionActor | None = None,
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    step: int,
    history: list[dict],
    metrics: dict,
    critic_optimizer: torch.optim.Optimizer | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if target_actor is None:
        target_actor = actor
    checkpoint = {
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "actor": actor.state_dict(),
        "target_actor": target_actor.state_dict(),
        "args": vars(args),
        "feature_dim": int(data["obs_features"].shape[-1]),
        "action_dim": int(data["action_dim"]),
        "chunk_horizon": int(data["chunk_horizon"]),
        "gamma": float(data["gamma"]),
        "action_mean": data["action_mean"].astype(np.float32),
        "action_std": data["action_std"].astype(np.float32),
        "normalize_actions": bool(args.normalize_actions),
        "aux_next_pred_enabled": bool(float(args.aux_next_pred_weight) > 0.0),
        "aux_next_pred_mode": str(args.aux_next_pred_mode),
        "aux_next_pred_weight": float(args.aux_next_pred_weight),
        "features": str(args.features),
        "pretrained_dp_checkpoint": str(data["checkpoint"].astype(str).item()),
        "step": int(step),
        "history": history,
        "metrics": metrics,
    }
    if critic_optimizer is not None:
        checkpoint["critic_optimizer"] = critic_optimizer.state_dict()
    if actor_optimizer is not None:
        checkpoint["actor_optimizer"] = actor_optimizer.state_dict()
    torch.save(checkpoint, path)


def rollout_eval_enabled(args: argparse.Namespace, step: int) -> bool:
    return int(args.rollout_eval_every) > 0 and (step % int(args.rollout_eval_every) == 0 or step == args.total_steps)


def run_resilient_rollout_eval(args: argparse.Namespace, checkpoint_path: Path, step: int) -> dict:
    """Run chunked closed-loop rollout evaluation through the resilient grid helper."""
    eval_grid_script = ROOT / "scripts/run_square_rgb_dp_one_step_idql_eval_grid.py"
    output_dir = args.rollout_eval_output_dir / f"step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "resilient_rollout_eval.log"
    device = args.rollout_eval_device or args.device
    cmd = [
        sys.executable,
        "-B",
        str(eval_grid_script),
        "--idql-checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--device",
        str(device),
        "--actor-source",
        "idql_target_one_step_mlp",
        "--critic-source",
        "target",
        "--n-rollouts",
        str(args.rollout_eval_n_rollouts),
        "--horizon",
        str(args.rollout_eval_horizon),
        "--num-candidates",
        str(args.rollout_eval_num_candidates),
        "--seeds",
        *[str(int(seed)) for seed in args.rollout_eval_seeds],
        "--rollouts-per-chunk",
        str(args.rollout_eval_rollouts_per_chunk),
        "--max-retries",
        str(args.rollout_eval_retries),
        "--candidate-batch-size",
        str(args.rollout_eval_candidate_batch_size),
        "--num-inference-steps",
        str(args.rollout_eval_num_inference_steps),
        "--selection",
        str(args.rollout_eval_selection),
        "--softmax-temperature",
        str(args.rollout_eval_softmax_temperature),
    ]
    cmd.append("--accept-partial" if args.rollout_eval_accept_partial else "--no-accept-partial")
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    print(f"[resilient-rollout-eval step={step}] " + " ".join(cmd), flush=True)
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    summary_path = output_dir / "one_step_idql_eval_grid_summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        record = {
            "step": int(step),
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "summary_json": str(summary_path),
            "log": str(log_path),
            "num_candidates": int(args.rollout_eval_num_candidates),
            "num_inference_steps": int(args.rollout_eval_num_inference_steps),
            "selection": str(args.rollout_eval_selection),
            "seeds": [int(x) for x in args.rollout_eval_seeds],
            "requested_rollouts_per_seed": int(args.rollout_eval_n_rollouts),
            "expected_rollouts": int(args.rollout_eval_n_rollouts) * len(args.rollout_eval_seeds),
            "completed_rollouts": 0,
            "valid_for_checkpoint_selection": False,
            "num_success": 0.0,
            "success_rate": 0.0,
            "mean_return": 0.0,
            "mean_horizon": 0.0,
            "ok": False,
            "returncode": int(proc.returncode),
        }
        print(json.dumps({"resilient_rollout_eval_failed": record}, indent=2), flush=True)
        if args.rollout_eval_strict:
            raise RuntimeError(f"resilient rollout evaluation failed; see {log_path}")
        return record

    summary = json.loads(summary_path.read_text())
    rows = [
        row
        for row in summary.get("by_num_candidates", [])
        if int(row.get("num_candidates", -1)) == int(args.rollout_eval_num_candidates)
    ]
    row = rows[0] if rows else {}
    expected_rollouts = int(args.rollout_eval_n_rollouts) * len(args.rollout_eval_seeds)
    completed_rollouts = int(row.get("total_rollouts", 0) or 0)
    status = str(row.get("status", "missing"))
    aggregate = {
        "step": int(step),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "log": str(log_path),
        "resilient": True,
        "status": status,
        "num_candidates": int(args.rollout_eval_num_candidates),
        "num_inference_steps": int(args.rollout_eval_num_inference_steps),
        "selection": str(args.rollout_eval_selection),
        "seeds": [int(x) for x in args.rollout_eval_seeds],
        "requested_rollouts_per_seed": int(args.rollout_eval_n_rollouts),
        "expected_rollouts": int(expected_rollouts),
        "completed_rollouts": int(completed_rollouts),
        "valid_for_checkpoint_selection": bool(status == "complete" and completed_rollouts == expected_rollouts),
        "num_success": float(row.get("total_success", 0.0) or 0.0),
        "success_rate": float(row.get("success_rate", 0.0) or 0.0),
        "mean_return": float(row.get("mean_return", 0.0) or 0.0),
        "mean_horizon": float(row.get("mean_horizon", 0.0) or 0.0),
        "seed_success_rate_mean": finite_float(row.get("seed_success_rate_mean")),
        "seed_success_rate_std": finite_float(row.get("seed_success_rate_std")),
        "seed_results": row.get("seeds", []),
    }
    print(json.dumps({"resilient_rollout_eval_summary": aggregate}, indent=2), flush=True)
    return aggregate


def run_rollout_eval(args: argparse.Namespace, checkpoint_path: Path, step: int) -> dict:
    """Run closed-loop rollout evaluation in subprocesses.

    Robosuite / MuJoCo evaluation is deliberately launched out-of-process so
    that renderer or simulator failures do not corrupt the training process.
    """
    if args.rollout_eval_resilient:
        return run_resilient_rollout_eval(args, checkpoint_path, step)
    eval_script = ROOT / "scripts/eval_square_rgb_dp_one_step_idql.py"
    output_dir = args.rollout_eval_output_dir / f"step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.rollout_eval_device or args.device
    seed_records = []
    total_success = 0.0
    total_rollouts = 0
    weighted_return = 0.0
    weighted_horizon = 0.0

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    for seed in args.rollout_eval_seeds:
        seed = int(seed)
        log_path = output_dir / f"seed_{seed}.log"
        cmd = [
            sys.executable,
            "-B",
            str(eval_script),
            "--idql-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--device",
            str(device),
            "--n-rollouts",
            str(args.rollout_eval_n_rollouts),
            "--horizon",
            str(args.rollout_eval_horizon),
            "--seed",
            str(seed),
            "--num-candidates",
            str(args.rollout_eval_num_candidates),
            "--candidate-batch-size",
            str(args.rollout_eval_candidate_batch_size),
            "--num-inference-steps",
            str(args.rollout_eval_num_inference_steps),
            "--selection",
            str(args.rollout_eval_selection),
        ]
        json_path = output_dir / f"one_step_idql_N{args.rollout_eval_num_candidates}_seed{seed}.json"
        proc = None
        for attempt in range(1, int(args.rollout_eval_retries) + 2):
            attempt_log_path = output_dir / f"seed_{seed}_attempt_{attempt}.log"
            # Keep seed_X.log as the latest-attempt convenience path.
            print(
                f"[rollout-eval step={step} seed={seed} attempt={attempt}] " + " ".join(cmd),
                flush=True,
            )
            with attempt_log_path.open("w") as log_f:
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            log_path.write_text(attempt_log_path.read_text(errors="replace"), errors="replace")
            if proc.returncode == 0 and json_path.exists():
                break
            print(
                json.dumps(
                    {
                        "rollout_eval_attempt_failed": {
                            "seed": seed,
                            "attempt": attempt,
                            "returncode": int(proc.returncode),
                            "log": str(attempt_log_path),
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
        if proc is None or proc.returncode != 0 or not json_path.exists():
            record = {
                "seed": seed,
                "ok": False,
                "returncode": int(proc.returncode) if proc is not None else None,
                "log": str(log_path),
                "json": str(json_path),
            }
            seed_records.append(record)
            print(json.dumps({"rollout_eval_failed": record}, indent=2), flush=True)
            if args.rollout_eval_strict:
                raise RuntimeError(f"rollout evaluation failed for seed={seed}; see {log_path}")
            continue

        summary = json.loads(json_path.read_text())
        avg = summary["average_rollout_stats"]
        n = int(avg["Num_Rollouts"])
        succ = float(avg["Num_Success"])
        total_success += succ
        total_rollouts += n
        weighted_return += float(avg["Return"]) * n
        weighted_horizon += float(avg["Horizon"]) * n
        seed_record = {
            "seed": seed,
            "ok": True,
            "json": str(json_path),
            "log": str(log_path),
            "num_rollouts": n,
            "num_success": succ,
            "success_rate": float(avg["Success_Rate"]),
            "return": float(avg["Return"]),
            "horizon": float(avg["Horizon"]),
        }
        seed_records.append(seed_record)
        print(json.dumps({"rollout_eval_seed": seed_record}, sort_keys=True), flush=True)

    expected_rollouts = int(args.rollout_eval_n_rollouts) * len(args.rollout_eval_seeds)
    aggregate = {
        "step": int(step),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_candidates": int(args.rollout_eval_num_candidates),
        "num_inference_steps": int(args.rollout_eval_num_inference_steps),
        "selection": str(args.rollout_eval_selection),
        "seeds": [int(x) for x in args.rollout_eval_seeds],
        "requested_rollouts_per_seed": int(args.rollout_eval_n_rollouts),
        "expected_rollouts": int(expected_rollouts),
        "completed_rollouts": int(total_rollouts),
        "valid_for_checkpoint_selection": bool(total_rollouts == expected_rollouts),
        "num_success": float(total_success),
        "success_rate": float(total_success / max(total_rollouts, 1)),
        "mean_return": float(weighted_return / max(total_rollouts, 1)),
        "mean_horizon": float(weighted_horizon / max(total_rollouts, 1)),
        "seed_results": seed_records,
    }
    (output_dir / "rollout_eval_summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps({"rollout_eval_summary": aggregate}, indent=2), flush=True)
    return aggregate


def best_from_history(history: list[dict]) -> dict:
    best = {
        "val_loss": float("inf"),
        "val_success_auc": -float("inf"),
        "val_reward_auc": -float("inf"),
        "val_actor_loss": float("inf"),
        "rollout_success_rate": -float("inf"),
    }
    for record in history:
        val = record.get("val", {})
        if "critic_loss" in val and "actor_bc_loss" in val:
            # This uses actor_selection_weight=1.0 for old histories. The exact
            # threshold only affects whether we overwrite best_loss on resume.
            best["val_loss"] = min(best["val_loss"], float(val["critic_loss"]) + float(val["actor_bc_loss"]))
        if val.get("success_auc") is not None:
            best["val_success_auc"] = max(best["val_success_auc"], float(val["success_auc"]))
        if val.get("reward_auc") is not None:
            best["val_reward_auc"] = max(best["val_reward_auc"], float(val["reward_auc"]))
        if val.get("actor_bc_loss") is not None:
            best["val_actor_loss"] = min(best["val_actor_loss"], float(val["actor_bc_loss"]))
        rollout = record.get("rollout")
        if rollout and rollout.get("valid_for_checkpoint_selection", False):
            best["rollout_success_rate"] = max(best["rollout_success_rate"], float(rollout["success_rate"]))
    return best


def write_partial_summary(args: argparse.Namespace, data: dict[str, np.ndarray], datasets: dict, feature_dim: int, action_dim: int, gamma: float, best: dict, history: list[dict]) -> None:
    partial = {
        "features": str(args.features),
        "output_dir": str(args.output_dir),
        "pretrained_dp_checkpoint": str(data["checkpoint"].astype(str).item()),
        "num_train": len(datasets["train"]),
        "num_val": len(datasets["val"]),
        "num_test": len(datasets["test"]),
        "feature_dim": int(feature_dim),
        "action_dim": int(action_dim),
        "gamma": float(gamma),
        "aux_next_pred": {
            "enabled": bool(float(args.aux_next_pred_weight) > 0.0),
            "weight": float(args.aux_next_pred_weight),
            "mode": str(args.aux_next_pred_mode),
        },
        "best": best,
        "last_completed_eval_step": int(history[-1]["step"]) if history else None,
        "history": history,
        "checkpoints": {
            "latest": str(args.output_dir / "latest.pt"),
            "latest": str(args.output_dir / "latest.pt"),
            "best_loss": str(args.output_dir / "best_loss.pt"),
            "best_success_auc": str(args.output_dir / "best_success_auc.pt"),
            "best_reward_auc": str(args.output_dir / "best_reward_auc.pt"),
            "best_actor_loss": str(args.output_dir / "best_actor_loss.pt"),
            "best_rollout_success": str(args.output_dir / "best_rollout_success.pt"),
            "last": str(args.output_dir / "last.pt"),
        },
    }
    (args.output_dir / "partial_summary.json").write_text(json.dumps(partial, indent=2))


def train(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    data = load_feature_npz(args.features)
    if int(data["chunk_horizon"]) != 1:
        raise ValueError(
            f"This is the one-step IDQL trainer, but feature cache chunk_horizon={int(data['chunk_horizon'])}. "
            "Rebuild with --chunk-horizon 1 --stride 1."
        )
    feature_dim = int(data["obs_features"].shape[-1])
    action_dim = int(data["action_dim"])
    gamma = float(data["gamma"])

    datasets = {
        split: OneStepFeatureDataset(
            data, split=split, reward_scale=args.reward_scale, normalize_actions=args.normalize_actions
        )
        for split in ("train", "val", "test")
    }
    reward_stats = reward_decomposition_stats(datasets)
    (args.output_dir / "reward_decomposition_stats.json").write_text(json.dumps(reward_stats, indent=2))
    print(json.dumps({"reward_decomposition_stats": reward_stats}, indent=2), flush=True)

    writer = make_tensorboard_writer(args)
    if writer is not None:
        writer.add_text("config/features", str(args.features), 0)
        writer.add_text("config/output_dir", str(args.output_dir), 0)
        if "risk_reward_mode" in data:
            writer.add_text("reward/mode", str(data["risk_reward_mode"].astype(str).item()), 0)
        if "risk_reward_formula" in data:
            writer.add_text("reward/formula", str(data["risk_reward_formula"].astype(str).item()), 0)
        add_tensorboard_scalars(writer, "data", flatten_stats(reward_stats), 0)
        writer.flush()

    train_loader = make_loader(datasets["train"], args.batch_size, shuffle=True, seed=args.seed)
    iterator = cycle(train_loader)

    critic = ChunkIQLCritic(
        feature_dim=feature_dim,
        action_dim=action_dim,
        chunk_horizon=1,
        hidden_dims=tuple(args.critic_hidden_dims),
        dropout=args.critic_dropout,
        aux_next_pred=bool(float(args.aux_next_pred_weight) > 0.0),
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    target_critic.eval()
    target_critic.requires_grad_(False)
    actor = OneStepDiffusionActor(
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.actor_hidden_dims),
        time_dim=args.time_dim,
        dropout=args.actor_dropout,
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_actor.eval()
    target_actor.requires_grad_(False)
    scheduler = make_scheduler(args)

    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_weight_decay
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_weight_decay
    )

    history: list[dict] = []
    start_step = 0
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
        print(f"Resuming one-step IDQL from {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        critic.load_state_dict(checkpoint["critic"])
        if "target_critic" in checkpoint:
            target_critic.load_state_dict(checkpoint["target_critic"])
        else:
            target_critic.load_state_dict(checkpoint["critic"])
        actor.load_state_dict(checkpoint["actor"])
        if "target_actor" in checkpoint:
            target_actor.load_state_dict(checkpoint["target_actor"])
        else:
            target_actor.load_state_dict(checkpoint["actor"])
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "actor_optimizer" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resume start_step={start_step}, history_len={len(history)}", flush=True)

    best = best_from_history(history)
    # Recompute combined loss threshold with the current actor_selection_weight.
    for record in history:
        val = record.get("val", {})
        if "critic_loss" in val and "actor_bc_loss" in val:
            best["val_loss"] = min(
                best["val_loss"],
                float(val["critic_loss"]) + args.actor_selection_weight * float(val["actor_bc_loss"]),
            )

    for step in range(start_step + 1, args.total_steps + 1):
        critic.train()
        actor.train()
        batch = batch_to_device(next(iterator), device)

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss, critic_info = compute_iql_losses(
            critic,
            target_critic,
            batch,
            gamma=gamma,
            expectile=args.expectile,
            use_huber=args.use_huber,
            aux_next_pred_weight=args.aux_next_pred_weight,
            aux_next_pred_mode=args.aux_next_pred_mode,
        )
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
        critic_optimizer.step()
        soft_update(target_critic, critic, args.target_tau)

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss, actor_info = actor_bc_loss(actor, scheduler, batch["obs"], batch["actions"])
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
        actor_optimizer.step()
        soft_update(target_actor, actor, args.actor_target_tau)

        if step % args.log_every == 0:
            batch_info = batch_reward_stats(batch)
            log_payload = {
                "step": step,
                **{k: float(v.detach().cpu()) for k, v in critic_info.items()},
                **{k: float(v.detach().cpu()) for k, v in actor_info.items()},
                **{f"batch_{k}": float(v.detach().cpu()) for k, v in batch_info.items()},
                "critic_grad_norm": float(critic_grad),
                "actor_grad_norm": float(actor_grad),
                "critic_lr": float(critic_optimizer.param_groups[0]["lr"]),
                "actor_lr": float(actor_optimizer.param_groups[0]["lr"]),
            }
            print(json.dumps(log_payload, sort_keys=True), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "train", {k: v for k, v in log_payload.items() if k != "step"}, step)
                writer.flush()

        if step % args.eval_every == 0 or step == args.total_steps:
            val_metrics = evaluate_split(
                critic,
                actor,
                scheduler,
                datasets["val"],
                args.eval_batch_size,
                device,
                gamma=gamma,
                expectile=args.expectile,
                use_huber=args.use_huber,
                aux_next_pred_weight=args.aux_next_pred_weight,
                aux_next_pred_mode=args.aux_next_pred_mode,
            )
            test_metrics = evaluate_split(
                critic,
                actor,
                scheduler,
                datasets["test"],
                args.eval_batch_size,
                device,
                gamma=gamma,
                expectile=args.expectile,
                use_huber=args.use_huber,
                aux_next_pred_weight=args.aux_next_pred_weight,
                aux_next_pred_mode=args.aux_next_pred_mode,
            )
            record = {"step": step, "val": val_metrics, "test": test_metrics}
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)
            if writer is not None:
                add_tensorboard_scalars(writer, "val", val_metrics, step)
                add_tensorboard_scalars(writer, "test", test_metrics, step)
                writer.flush()

            metrics = {"val": val_metrics, "test": test_metrics}
            save_checkpoint(
                args.output_dir / "latest.pt",
                critic=critic,
                target_critic=target_critic,
                actor=actor,
                target_actor=target_actor,
                args=args,
                data=data,
                step=step,
                history=history,
                metrics=metrics,
                critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer,
            )
            write_partial_summary(args, data, datasets, feature_dim, action_dim, gamma, best, history)
            if rollout_eval_enabled(args, step):
                rollout_ckpt = args.output_dir / "rollout_eval_checkpoints" / f"step_{step:06d}.pt"
                save_checkpoint(
                    rollout_ckpt,
                    critic=critic,
                    target_critic=target_critic,
                    actor=actor,
                    target_actor=target_actor,
                    args=args,
                    data=data,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
                rollout_metrics = run_rollout_eval(args, rollout_ckpt, step)
                record["rollout"] = rollout_metrics
                metrics = {"val": val_metrics, "test": test_metrics, "rollout": rollout_metrics}
                rollout_success = float(rollout_metrics["success_rate"])
                if writer is not None:
                    add_tensorboard_scalars(writer, "rollout", rollout_metrics, step)
                    writer.flush()
                write_partial_summary(args, data, datasets, feature_dim, action_dim, gamma, best, history)
                if not rollout_metrics.get("valid_for_checkpoint_selection", False):
                    print(
                        json.dumps(
                            {
                                "rollout_eval_invalid_for_selection": {
                                    "step": step,
                                    "completed_rollouts": rollout_metrics.get("completed_rollouts"),
                                    "expected_rollouts": rollout_metrics.get("expected_rollouts"),
                                }
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                elif rollout_success > best["rollout_success_rate"]:
                    best["rollout_success_rate"] = rollout_success
                    save_checkpoint(
                        args.output_dir / "best_rollout_success.pt",
                        critic=critic,
                        target_critic=target_critic,
                        actor=actor,
                        target_actor=target_actor,
                        args=args,
                        data=data,
                        step=step,
                        history=history,
                        metrics=metrics,
                    )

            combined_val_loss = float(val_metrics["critic_loss"]) + args.actor_selection_weight * float(val_metrics["actor_bc_loss"])
            if combined_val_loss < best["val_loss"]:
                best["val_loss"] = combined_val_loss
                save_checkpoint(
                    args.output_dir / "best_loss.pt",
                    critic=critic,
                    target_critic=target_critic,
                    actor=actor,
                    target_actor=target_actor,
                    args=args,
                    data=data,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            val_success_auc = val_metrics.get("success_auc")
            if val_success_auc is not None and float(val_success_auc) > best["val_success_auc"]:
                best["val_success_auc"] = float(val_success_auc)
                save_checkpoint(
                    args.output_dir / "best_success_auc.pt",
                    critic=critic,
                    target_critic=target_critic,
                    actor=actor,
                    target_actor=target_actor,
                    args=args,
                    data=data,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            val_reward_auc = val_metrics.get("reward_auc")
            if val_reward_auc is not None and float(val_reward_auc) > best["val_reward_auc"]:
                best["val_reward_auc"] = float(val_reward_auc)
                save_checkpoint(
                    args.output_dir / "best_reward_auc.pt",
                    critic=critic,
                    target_critic=target_critic,
                    actor=actor,
                    target_actor=target_actor,
                    args=args,
                    data=data,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            if float(val_metrics["actor_bc_loss"]) < best["val_actor_loss"]:
                best["val_actor_loss"] = float(val_metrics["actor_bc_loss"])
                save_checkpoint(
                    args.output_dir / "best_actor_loss.pt",
                    critic=critic,
                    target_critic=target_critic,
                    actor=actor,
                    target_actor=target_actor,
                    args=args,
                    data=data,
                    step=step,
                    history=history,
                    metrics=metrics,
                )
            write_partial_summary(args, data, datasets, feature_dim, action_dim, gamma, best, history)

    final_metrics = {
        split: evaluate_split(
            critic,
            actor,
            scheduler,
            datasets[split],
            args.eval_batch_size,
            device,
            gamma=gamma,
            expectile=args.expectile,
            use_huber=args.use_huber,
            aux_next_pred_weight=args.aux_next_pred_weight,
            aux_next_pred_mode=args.aux_next_pred_mode,
        )
        for split in ("train", "val", "test")
    }
    save_checkpoint(
        args.output_dir / "last.pt",
        critic=critic,
        target_critic=target_critic,
        actor=actor,
        target_actor=target_actor,
        args=args,
        data=data,
        step=args.total_steps,
        history=history,
        metrics=final_metrics,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
    )
    if writer is not None:
        for split, metrics in final_metrics.items():
            add_tensorboard_scalars(writer, f"final/{split}", metrics, args.total_steps)
        writer.flush()
        writer.close()

    summary = {
        "features": str(args.features),
        "output_dir": str(args.output_dir),
        "pretrained_dp_checkpoint": str(data["checkpoint"].astype(str).item()),
        "num_train": len(datasets["train"]),
        "num_val": len(datasets["val"]),
        "num_test": len(datasets["test"]),
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "gamma": gamma,
        "aux_next_pred": {
            "enabled": bool(float(args.aux_next_pred_weight) > 0.0),
            "weight": float(args.aux_next_pred_weight),
            "mode": str(args.aux_next_pred_mode),
        },
        "best": best,
        "final_metrics": final_metrics,
        "history": history,
        "checkpoints": {
            "latest": str(args.output_dir / "latest.pt"),
            "best_loss": str(args.output_dir / "best_loss.pt"),
            "best_success_auc": str(args.output_dir / "best_success_auc.pt"),
            "best_reward_auc": str(args.output_dir / "best_reward_auc.pt"),
            "best_actor_loss": str(args.output_dir / "best_actor_loss.pt"),
            "best_rollout_success": str(args.output_dir / "best_rollout_success.pt"),
            "last": str(args.output_dir / "last.pt"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-weight-decay", type=float, default=0.0)
    parser.add_argument("--actor-weight-decay", type=float, default=0.0)
    parser.add_argument("--critic-dropout", type=float, default=0.0)
    parser.add_argument("--actor-dropout", type=float, default=0.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--actor-target-tau", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument(
        "--aux-next-pred-weight",
        type=float,
        default=0.0,
        help="Weight for optional critic next-latent prediction loss. 0 keeps the standard IDQL critic.",
    )
    parser.add_argument(
        "--aux-next-pred-mode",
        choices=("delta", "next"),
        default="delta",
        help="Auxiliary target: z_{t+1}-z_t or absolute z_{t+1}.",
    )
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--clip-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--actor-selection-weight", type=float, default=1.0)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--rollout-eval-every", type=int, default=0, help="0 disables closed-loop evaluation during training")
    parser.add_argument("--rollout-eval-output-dir", type=Path, default=None)
    parser.add_argument("--rollout-eval-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--rollout-eval-n-rollouts", type=int, default=50)
    parser.add_argument("--rollout-eval-horizon", type=int, default=400)
    parser.add_argument("--rollout-eval-num-candidates", type=int, default=16)
    parser.add_argument("--rollout-eval-candidate-batch-size", type=int, default=16)
    parser.add_argument("--rollout-eval-num-inference-steps", type=int, default=100)
    parser.add_argument("--rollout-eval-selection", choices=("argmax", "greedy", "softmax", "advantage_softmax"), default="argmax")
    parser.add_argument("--rollout-eval-softmax-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-eval-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--rollout-eval-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rollout-eval-retries", type=int, default=3)
    parser.add_argument("--rollout-eval-resilient", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-eval-rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--rollout-eval-accept-partial", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.features = args.features.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.tensorboard_dir is not None:
        args.tensorboard_dir = args.tensorboard_dir.resolve()
    if args.rollout_eval_output_dir is None:
        args.rollout_eval_output_dir = args.output_dir / "rollout_eval"
    else:
        args.rollout_eval_output_dir = args.rollout_eval_output_dir.resolve()
    train(args)


if __name__ == "__main__":
    main()
