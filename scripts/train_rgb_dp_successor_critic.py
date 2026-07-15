#!/usr/bin/env python3
"""Train a successor-based RGB-DP success critic.

The model has three coupled pieces:

* V(h_t): state success probability from the causal observation prefix;
* F(h_t, a_t): action-conditioned latent successor prediction;
* Q(h_t, a_t) = V(F(h_t, a_t)): action success value.

Logged transitions supervise F with the actual next latent context and supervise
Q with a sparse-reward Bellman target. Contrastive action ranking is disabled
by default and intentionally not needed for this first experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from robomimic.models.prefix_risk_nets import make_causal_prefix_model
from robomimic.utils.rgb_critic_utils import load_rgb_encoder_from_actor_checkpoint
from train_rgb_dp_hazard_mil import (
    OBS_KEYS,
    average_precision,
    balanced_episode_batch,
    decode,
    rank_auc,
    sorted_demo_keys,
    stratified_split,
)


ROOT = Path(__file__).resolve().parents[1]

STATE_MODULE_NAMES = (
    "obs_projection",
    "prefix_temporal_encoder",
    "prefix_encoder",
    "context_norm",
    "state_head",
)


def episode_indices(offsets: np.ndarray, episode: int) -> np.ndarray:
    return np.arange(offsets[episode], offsets[episode + 1], dtype=np.int64)


def compute_normalizers(
    features: np.ndarray,
    actions: np.ndarray,
    action_mask: np.ndarray,
    offsets: np.ndarray,
    train_episodes: np.ndarray,
) -> dict[str, np.ndarray]:
    indices = np.concatenate(
        [episode_indices(offsets, int(ep)) for ep in train_episodes]
    )
    train_features = features[indices]
    train_actions = actions[indices]
    train_action_mask = action_mask[indices]
    valid_actions = train_actions[train_action_mask]
    return {
        "feature_mean": train_features.mean(axis=0).astype(np.float32),
        "feature_std": np.maximum(train_features.std(axis=0), 1e-4).astype(np.float32),
        "action_mean": valid_actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(valid_actions.std(axis=0), 1e-4).astype(np.float32),
    }


def normalize_arrays(
    features: np.ndarray,
    next_features: np.ndarray,
    actions: np.ndarray,
    action_mask: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = (features - stats["feature_mean"]) / stats["feature_std"]
    next_features = (next_features - stats["feature_mean"]) / stats["feature_std"]
    actions = (actions - stats["action_mean"]) / stats["action_std"]
    actions = actions.astype(np.float32)
    actions[~action_mask] = 0.0
    return (
        features.astype(np.float32),
        next_features.astype(np.float32),
        actions,
    )


def build_raw_transition_table(
    rollout_path: Path,
    action_horizon: int,
    max_episodes: int | None = None,
) -> tuple[dict, dict]:
    """Index raw HDF5 rollouts without precomputing visual features."""
    episode_keys = []
    episode_labels = []
    episode_boundaries = []
    offsets = [0]
    all_actions = []
    all_action_masks = []
    all_rewards = []
    all_terminals = []
    action_dim = None
    with h5py.File(rollout_path, "r") as dataset:
        failures = set(decode(dataset["mask/failure"][:]))
        demo_keys = sorted_demo_keys(dataset)
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]
        for demo_key in demo_keys:
            group = dataset[f"data/{demo_key}"]
            length = int(group.attrs["num_samples"])
            if length <= 0:
                continue
            boundaries = np.arange(0, length, action_horizon, dtype=np.int64)
            action_dim = int(group["actions"].shape[-1])
            actions = np.zeros(
                (len(boundaries), action_horizon, action_dim), dtype=np.float32
            )
            action_mask = np.zeros(
                (len(boundaries), action_horizon), dtype=np.bool_
            )
            rewards = np.zeros(
                (len(boundaries), action_horizon), dtype=np.float32
            )
            terminals = np.zeros(len(boundaries), dtype=np.bool_)
            for row, boundary in enumerate(boundaries):
                end = min(int(boundary) + action_horizon, length)
                count = end - int(boundary)
                actions[row, :count] = group["actions"][boundary:end]
                rewards[row, :count] = group["rewards"][boundary:end]
                action_mask[row, :count] = True
            terminals[-1] = True
            label = float(demo_key not in failures)
            terminal_return = float(rewards[-1].sum())
            if label > 0.5 and terminal_return <= 0.0:
                raise ValueError(
                    f"successful rollout {demo_key} has no sparse terminal reward"
                )
            if label < 0.5 and terminal_return > 0.0:
                raise ValueError(
                    f"failed rollout {demo_key} has a positive terminal reward"
                )
            episode_keys.append(demo_key)
            episode_labels.append(label)
            episode_boundaries.append(boundaries)
            all_actions.append(actions)
            all_action_masks.append(action_mask)
            all_rewards.append(rewards)
            all_terminals.append(terminals)
            offsets.append(offsets[-1] + len(boundaries))
    if not episode_keys or action_dim is None:
        raise ValueError(f"no rollouts found in {rollout_path}")
    arrays = {
        "actions_raw": np.concatenate(all_actions, axis=0),
        "action_mask": np.concatenate(all_action_masks, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "terminals": np.concatenate(all_terminals, axis=0),
        "labels": np.asarray(episode_labels, dtype=np.float32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "episode_keys": episode_keys,
        "episode_boundaries": episode_boundaries,
    }
    metadata = {
        "rollout_path": str(rollout_path),
        "action_horizon": int(action_horizon),
        "q_action_horizon": int(action_horizon),
        "q_boundary_stride": 1,
        "num_episodes": len(episode_keys),
        "num_transitions": int(offsets[-1]),
        "action_dim": int(action_dim),
        "input_source": "raw_hdf5_rgb",
    }
    return arrays, metadata


def _read_ordered(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(indices, return_inverse=True)
    return dataset[unique][inverse]


def padded_rgb_batch(
    *,
    dataset: h5py.File,
    arrays: dict,
    episodes: np.ndarray,
    observation_shapes: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Read raw RGB/proprioception prefixes for complete sampled episodes."""
    import robomimic.utils.obs_utils as ObsUtils

    lengths = np.asarray(
        [arrays["offsets"][ep + 1] - arrays["offsets"][ep] for ep in episodes],
        dtype=np.int64,
    )
    max_length = int(lengths.max())
    batch_size = len(episodes)
    raw_observations = {}
    for key in observation_shapes:
        first_key = arrays["episode_keys"][int(episodes[0])]
        first_group = dataset[f"data/{first_key}"]
        raw_shape = tuple(first_group[f"obs/{key}"].shape[1:])
        raw_observations[key] = np.zeros(
            (batch_size, max_length, 2, *raw_shape),
            dtype=first_group[f"obs/{key}"].dtype,
        )

    action_horizon = arrays["actions"].shape[1]
    action_dim = arrays["actions"].shape[2]
    batch_actions = np.zeros(
        (batch_size, max_length, action_horizon, action_dim), dtype=np.float32
    )
    batch_action_mask = np.zeros(
        (batch_size, max_length, action_horizon), dtype=np.bool_
    )
    batch_rewards = np.zeros(
        (batch_size, max_length, action_horizon), dtype=np.float32
    )
    batch_terminals = np.zeros((batch_size, max_length), dtype=np.bool_)
    batch_mc_returns = np.zeros((batch_size, max_length), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    for row, episode_value in enumerate(episodes):
        episode = int(episode_value)
        start = int(arrays["offsets"][episode])
        end = int(arrays["offsets"][episode + 1])
        length = end - start
        group = dataset[f"data/{arrays['episode_keys'][episode]}"]
        boundaries = arrays["episode_boundaries"][episode]
        previous = np.maximum(boundaries - 1, 0)
        for key in observation_shapes:
            values = group[f"obs/{key}"]
            raw_observations[key][row, :length] = np.stack(
                [_read_ordered(values, previous), _read_ordered(values, boundaries)],
                axis=1,
            )
        batch_actions[row, :length] = arrays["actions"][start:end]
        batch_action_mask[row, :length] = arrays["action_mask"][start:end]
        batch_rewards[row, :length] = arrays["rewards"][start:end]
        batch_terminals[row, :length] = arrays["terminals"][start:end]
        batch_mc_returns[row, :length] = arrays["mc_returns"][start:end]
        batch_mask[row, :length] = True

    observations = {}
    for key, values in raw_observations.items():
        tensor = torch.as_tensor(values, device=device).float()
        if ObsUtils.key_is_obs_modality(key=key, obs_modality="rgb") or (
            ObsUtils.key_is_obs_modality(key=key, obs_modality="depth")
        ):
            tensor = ObsUtils.process_obs(tensor, obs_key=key)
        expected_shape = tuple(observation_shapes[key])
        if tuple(tensor.shape[-len(expected_shape) :]) != expected_shape:
            raise ValueError(
                f"processed {key} shape={tuple(tensor.shape)} does not end in "
                f"{expected_shape}"
            )
        observations[key] = tensor
    return {
        "observations": observations,
        "actions": torch.from_numpy(batch_actions).to(device),
        "action_mask": torch.from_numpy(batch_action_mask).to(device),
        "rewards": torch.from_numpy(batch_rewards).to(device),
        "terminals": torch.from_numpy(batch_terminals).to(device),
        "mc_returns": torch.from_numpy(batch_mc_returns).to(device),
        "mask": torch.from_numpy(batch_mask).to(device),
        "lengths": torch.from_numpy(lengths).to(device),
        "labels": torch.as_tensor(arrays["labels"][episodes], device=device).float(),
    }


@torch.no_grad()
def compute_raw_rgb_normalizers(
    *,
    dataset: h5py.File,
    arrays: dict,
    train_episodes: np.ndarray,
    rgb_encoder,
    observation_shapes: dict,
    device: torch.device,
    episode_batch_size: int,
) -> dict[str, np.ndarray]:
    """Compute fixed normalization from the actor-initialized critic encoder."""
    rgb_encoder.eval()
    feature_sum = None
    feature_square_sum = None
    feature_count = 0
    for start in range(0, len(train_episodes), episode_batch_size):
        episodes = train_episodes[start : start + episode_batch_size]
        batch = padded_rgb_batch(
            dataset=dataset,
            arrays=arrays,
            episodes=episodes,
            observation_shapes=observation_shapes,
            device=device,
        )
        observations = batch["observations"]
        first = observations[next(iter(observation_shapes))]
        batch_size, prefix_length, obs_horizon = first.shape[:3]
        flattened = {
            key: value.reshape(batch_size * prefix_length * obs_horizon, *shape)
            for (key, shape), value in zip(
                observation_shapes.items(),
                (observations[k] for k in observation_shapes),
            )
        }
        encoded = rgb_encoder(obs=flattened).reshape(
            batch_size, prefix_length, obs_horizon, -1
        ).flatten(start_dim=2)
        valid = batch["mask"]
        values = encoded[valid].double()
        current_sum = values.sum(dim=0)
        current_square_sum = (values * values).sum(dim=0)
        feature_sum = current_sum if feature_sum is None else feature_sum + current_sum
        feature_square_sum = (
            current_square_sum
            if feature_square_sum is None
            else feature_square_sum + current_square_sum
        )
        feature_count += int(values.shape[0])
    feature_mean = feature_sum / feature_count
    feature_variance = feature_square_sum / feature_count - feature_mean.square()
    train_rows = np.concatenate(
        [episode_indices(arrays["offsets"], int(ep)) for ep in train_episodes]
    )
    valid_actions = arrays["actions_raw"][train_rows][
        arrays["action_mask"][train_rows]
    ]
    return {
        "feature_mean": feature_mean.float().cpu().numpy(),
        "feature_std": torch.sqrt(feature_variance.clamp_min(1e-8))
        .clamp_min(1e-4)
        .float()
        .cpu()
        .numpy(),
        "action_mean": valid_actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(valid_actions.std(axis=0), 1e-4).astype(np.float32),
    }


def validate_cache(raw) -> dict:
    required = {
        "features",
        "next_features",
        "actions",
        "action_valid_mask",
        "chunk_rewards",
        "terminals",
        "steps",
        "episode_success_labels",
        "episode_offsets",
        "action_horizon",
        "q_action_horizon",
        "q_boundary_stride",
        "transition_complete",
        "dp_checkpoint",
        "rollout_path",
    }
    missing = sorted(required.difference(raw.files))
    if missing:
        raise ValueError(
            "successor feature cache is missing keys "
            f"{missing}; rebuild it with build_rgb_dp_successor_features.py"
        )
    if not bool(np.asarray(raw["transition_complete"]).item()):
        raise ValueError("feature cache is not transition-complete")
    action_horizon = int(np.asarray(raw["action_horizon"]).item())
    q_action_horizon = int(np.asarray(raw["q_action_horizon"]).item())
    q_boundary_stride = int(np.asarray(raw["q_boundary_stride"]).item())
    if q_boundary_stride != 1 or q_action_horizon != action_horizon:
        raise ValueError(
            "this first successor critic requires one executed policy chunk per "
            "transition: q_action_horizon must equal action_horizon"
        )
    return {
        "action_horizon": action_horizon,
        "q_action_horizon": q_action_horizon,
        "q_boundary_stride": q_boundary_stride,
        "dp_checkpoint": str(np.asarray(raw["dp_checkpoint"]).item()),
        "rollout_path": str(np.asarray(raw["rollout_path"]).item()),
    }


def validate_transition_rows(
    features: np.ndarray,
    next_features: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
) -> dict:
    alignment_errors = []
    terminal_count = 0
    reward_count = int(np.sum(rewards.sum(axis=1) > 0.0))
    for episode in range(len(labels)):
        start, end = int(offsets[episode]), int(offsets[episode + 1])
        episode_terminals = np.flatnonzero(terminals[start:end])
        if not np.array_equal(episode_terminals, np.asarray([end - start - 1])):
            raise ValueError(
                f"episode {episode} must have exactly one terminal final chunk; "
                f"got {episode_terminals.tolist()}"
            )
        terminal_count += len(episode_terminals)
        if end - start > 1:
            alignment_errors.append(
                np.abs(next_features[start : end - 1] - features[start + 1 : end])
            )
        terminal_return = float(rewards[end - 1].sum())
        if labels[episode] > 0.5 and terminal_return <= 0.0:
            raise ValueError(f"successful episode {episode} has no terminal reward")
        if labels[episode] < 0.5 and terminal_return > 0.0:
            raise ValueError(f"failed episode {episode} has a positive terminal reward")
    alignment = np.concatenate(alignment_errors) if alignment_errors else np.zeros(1)
    max_alignment_error = float(np.max(alignment))
    if max_alignment_error > 1e-4:
        raise ValueError(
            "next_features do not align with the next policy boundary; "
            f"max error={max_alignment_error}"
        )
    return {
        "terminal_rows": terminal_count,
        "positive_reward_rows": reward_count,
        "max_next_feature_alignment_error": max_alignment_error,
    }


def compute_discounted_mc_returns(
    rewards: np.ndarray,
    action_mask: np.ndarray,
    terminals: np.ndarray,
    offsets: np.ndarray,
    step_gamma: float,
) -> np.ndarray:
    """Compute behavior-policy Monte Carlo returns at chunk boundaries.

    ``step_gamma`` is the simulator-step discount. This handles a short final
    chunk correctly while making a complete chunk discount its successor by
    exactly the configured chunk-level gamma.
    """
    powers = np.power(
        np.float64(step_gamma),
        np.arange(rewards.shape[1], dtype=np.float64),
    )
    chunk_returns = (
        rewards.astype(np.float64)
        * action_mask.astype(np.float64)
        * powers[None, :]
    ).sum(axis=1)
    valid_lengths = action_mask.sum(axis=1).astype(np.float64)
    bootstrap_discounts = np.power(np.float64(step_gamma), valid_lengths)
    returns = np.zeros(len(rewards), dtype=np.float32)
    for episode in range(len(offsets) - 1):
        start, end = int(offsets[episode]), int(offsets[episode + 1])
        running = 0.0
        for index in range(end - 1, start - 1, -1):
            continuation = 0.0 if terminals[index] else bootstrap_discounts[index]
            running = chunk_returns[index] + continuation * running
            returns[index] = np.float32(np.clip(running, 0.0, 1.0))
    return returns


def padded_batch(
    *,
    features: np.ndarray,
    actions: np.ndarray,
    action_mask: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    mc_returns: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    lengths = np.asarray(
        [offsets[ep + 1] - offsets[ep] for ep in episodes],
        dtype=np.int64,
    )
    max_length = int(lengths.max())
    batch_size = len(episodes)
    batch_features = np.zeros(
        (batch_size, max_length, features.shape[-1]), dtype=np.float32
    )
    batch_actions = np.zeros(
        (batch_size, max_length, actions.shape[1], actions.shape[2]),
        dtype=np.float32,
    )
    batch_action_mask = np.zeros(
        (batch_size, max_length, actions.shape[1]), dtype=np.bool_
    )
    batch_rewards = np.zeros(
        (batch_size, max_length, rewards.shape[1]), dtype=np.float32
    )
    batch_terminals = np.zeros((batch_size, max_length), dtype=np.bool_)
    batch_mc_returns = np.zeros((batch_size, max_length), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    for row, episode in enumerate(episodes):
        start, end = int(offsets[episode]), int(offsets[episode + 1])
        length = end - start
        batch_features[row, :length] = features[start:end]
        batch_actions[row, :length] = actions[start:end]
        batch_action_mask[row, :length] = action_mask[start:end]
        batch_rewards[row, :length] = rewards[start:end]
        batch_terminals[row, :length] = terminals[start:end]
        batch_mc_returns[row, :length] = mc_returns[start:end]
        batch_mask[row, :length] = True
    return {
        "features": torch.from_numpy(batch_features).to(device),
        "actions": torch.from_numpy(batch_actions).to(device),
        "action_mask": torch.from_numpy(batch_action_mask).to(device),
        "rewards": torch.from_numpy(batch_rewards).to(device),
        "terminals": torch.from_numpy(batch_terminals).to(device),
        "mc_returns": torch.from_numpy(batch_mc_returns).to(device),
        "mask": torch.from_numpy(batch_mask).to(device),
        "lengths": torch.from_numpy(lengths).to(device),
        "labels": torch.as_tensor(labels[episodes], device=device).float(),
    }


def episode_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(values.dtype)
    reduce_dims = tuple(range(1, values.ndim))
    numerator = (values * weights).sum(dim=reduce_dims)
    denominator = weights.expand_as(values).sum(dim=reduce_dims)
    valid = denominator > 0
    if not torch.any(valid):
        return values.new_zeros(())
    return (numerator[valid] / denominator[valid].clamp_min(1.0)).mean()


def future_context(
    context: torch.Tensor,
    lengths: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, horizon, _ = context.shape
    positions = torch.arange(horizon, device=context.device)[None, :]
    future_positions = positions + int(stride)
    valid = future_positions < lengths[:, None]
    gather_positions = torch.minimum(
        future_positions,
        (lengths[:, None] - 1).clamp_min(0),
    ).long()
    gathered = torch.gather(
        context,
        dim=1,
        index=gather_positions[..., None].expand(batch_size, horizon, context.shape[-1]),
    )
    return gathered, valid


def discounted_chunk_return(
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    powers = torch.pow(
        rewards.new_tensor(float(gamma)),
        torch.arange(rewards.shape[-1], device=rewards.device, dtype=rewards.dtype),
    )
    returns = (rewards * action_mask.to(rewards.dtype) * powers).sum(dim=-1)
    valid_lengths = action_mask.sum(dim=-1).to(rewards.dtype)
    bootstrap_discount = torch.pow(rewards.new_tensor(float(gamma)), valid_lengths)
    return returns, bootstrap_discount


def smooth_probability_targets(
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if epsilon <= 0.0:
        return targets
    return targets * (1.0 - 2.0 * epsilon) + epsilon


def compute_losses(
    model,
    target_model,
    batch: dict[str, torch.Tensor],
    args,
    q_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "observations" in batch:
        # End-to-end path: every optimization step runs the critic-owned RGB
        # encoder. No actor features are read or injected here.
        features = model.encode_rgb_history(batch["observations"])
        with torch.no_grad():
            target_features = target_model.encode_rgb_history(batch["observations"])
    else:
        features = batch["features"]
        target_features = batch["features"]
    output = model(
        features,
        batch["actions"],
        batch["action_mask"],
    )
    with torch.no_grad():
        target_context = target_model.encode_prefix(target_features)
        next_context, has_future = future_context(
            target_context,
            batch["lengths"],
            args.q_boundary_stride,
        )
        next_logit = target_model.state_head(next_context).squeeze(-1)
        next_probability = torch.sigmoid(next_logit)

    state_targets = batch["mc_returns"]
    state_values = F.binary_cross_entropy_with_logits(
        output["state_logit"],
        smooth_probability_targets(state_targets, args.target_label_smoothing),
        reduction="none",
    )
    state_loss = episode_mean(state_values, batch["mask"])

    dynamics_mask = batch["mask"] & has_future & ~batch["terminals"]
    dynamics_values = F.smooth_l1_loss(
        output["predicted_next_context"],
        next_context.detach(),
        reduction="none",
        beta=args.dynamics_huber_beta,
    )
    dynamics_loss = episode_mean(dynamics_values, dynamics_mask)
    dynamics_cosine_values = 1.0 - F.cosine_similarity(
        output["predicted_next_context"],
        next_context.detach(),
        dim=-1,
    )
    dynamics_cosine_loss = episode_mean(dynamics_cosine_values, dynamics_mask)

    chunk_return, bootstrap_discount = discounted_chunk_return(
        batch["rewards"],
        batch["action_mask"],
        args.step_gamma,
    )
    q_target = chunk_return + (
        (~batch["terminals"]).to(chunk_return.dtype)
        * bootstrap_discount
        * next_probability
    )
    q_target = q_target.clamp(0.0, 1.0)
    q_values = F.binary_cross_entropy_with_logits(
        output["action_logit"],
        smooth_probability_targets(q_target, args.target_label_smoothing),
        reduction="none",
    )
    q_loss = episode_mean(q_values, batch["mask"])

    value_consistency_values = F.binary_cross_entropy_with_logits(
        output["state_logit"],
        torch.sigmoid(output["action_logit"].detach()),
        reduction="none",
    )
    value_consistency_loss = episode_mean(
        value_consistency_values,
        batch["mask"],
    )

    contrast_loss = output["state_logit"].new_zeros(())
    total = (
        args.state_weight * state_loss
        + args.dynamics_weight * dynamics_loss
        + args.dynamics_cosine_weight * dynamics_cosine_loss
        + float(q_scale) * args.q_weight * q_loss
        + float(q_scale)
        * args.value_consistency_weight
        * value_consistency_loss
        + args.contrast_weight * contrast_loss
        + args.encoder_reference_weight * encoder_reference_loss
    )
    tensors = {
        "total": total,
        "state_loss": state_loss,
        "dynamics_loss": dynamics_loss,
        "dynamics_cosine_loss": dynamics_cosine_loss,
        "q_loss": q_loss,
        "value_consistency_loss": value_consistency_loss,
        "contrast_loss": contrast_loss,
        "encoder_reference_loss": encoder_reference_loss,
        "q_target": q_target,
        "state_target": state_targets,
        "next_logit": next_logit,
        "next_context": next_context,
        "dynamics_mask": dynamics_mask,
        **output,
    }
    return total, tensors


@torch.no_grad()
def update_target(target_model, model, decay: float) -> None:
    for target, source in zip(target_model.parameters(), model.parameters()):
        # Frozen modules (especially the 22M-parameter RGB encoder) are
        # identical in model and target_model and never change. Skipping them
        # avoids a large, pointless device copy at every optimization step.
        if source.requires_grad:
            target.mul_(decay).add_(source, alpha=1.0 - decay)


def state_parameters(model):
    for module_name in STATE_MODULE_NAMES:
        yield from getattr(model, module_name).parameters()


def set_state_trainable(model, trainable: bool) -> None:
    for parameter in state_parameters(model):
        parameter.requires_grad_(trainable)


def set_frozen_state_eval(model) -> None:
    for module_name in STATE_MODULE_NAMES:
        getattr(model, module_name).eval()


def set_frozen_rgb_encoder_eval(model) -> None:
    """Keep actor-initialized visual features fixed and deterministic."""
    if hasattr(model, "rgb_encoder"):
        model.rgb_encoder.eval()


def restore_state_modules(model, checkpoint: dict) -> None:
    current = model.state_dict()
    source = checkpoint["model"]
    prefixes = tuple(f"{name}." for name in STATE_MODULE_NAMES)
    restored = 0
    for name in current:
        if name.startswith(prefixes):
            current[name] = source[name]
            restored += 1
    if restored == 0:
        raise RuntimeError("no state-value parameters were restored")
    model.load_state_dict(current)


def make_optimizer_and_scheduler(model, args, steps: int):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(steps), 1),
    )
    return optimizer, scheduler


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64) - float(np.mean(x))
    y = y.astype(np.float64) - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 1e-12 else float("nan")


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return {
        "roc_auc": rank_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "bce": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
        "brier": float(np.mean((scores - labels) ** 2)),
    }


def probability_target_metrics(targets: np.ndarray, scores: np.ndarray) -> dict:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return {
        "bce": float(
            -np.mean(
                targets * np.log(clipped)
                + (1.0 - targets) * np.log(1.0 - clipped)
            )
        ),
        "mse": float(np.mean((scores - targets) ** 2)),
        "pearson": safe_pearson(scores, targets),
    }


def finite_metric(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(float(value)) else float(fallback)


def action_selection_validation_score(metrics: dict) -> float:
    """Prefer progress fidelity and genuine action-conditioned dynamics."""
    progress = finite_metric(metrics["progress_logit_pearson"], -1.0)
    relative_gain = np.clip(
        finite_metric(metrics["relative_action_conditioning_gain"], -1.0),
        -1.0,
        1.0,
    )
    q_bce = finite_metric(metrics["q_target_bce"], 20.0)
    return float(progress + 0.25 * relative_gain - 0.05 * q_bce)


@torch.no_grad()
def evaluate(
    model,
    target_model,
    arrays: dict,
    episodes: np.ndarray,
    args,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[dict, dict | None]:
    model.eval()
    target_model.eval()
    state_scores = []
    q_scores = []
    q_targets = []
    state_targets = []
    outcome_labels = []
    state_logits = []
    q_logits = []
    target_progress = []
    predicted_progress = []
    dynamics_errors = []
    dynamics_cosines = []
    shuffled_differences = []
    logged_action_dynamics_errors = []
    shuffled_action_dynamics_errors = []
    row_indices = []

    for start in range(0, len(episodes), args.eval_episode_batch_size):
        selected = episodes[start : start + args.eval_episode_batch_size]
        batch = padded_batch(
            features=arrays["features"],
            actions=arrays["actions"],
            action_mask=arrays["action_mask"],
            rewards=arrays["rewards"],
            terminals=arrays["terminals"],
            mc_returns=arrays["mc_returns"],
            labels=arrays["labels"],
            offsets=arrays["offsets"],
            episodes=selected,
            device=device,
        )
        _, tensors = compute_losses(model, target_model, batch, args, q_scale=1.0)
        valid = batch["mask"]
        state_probability = torch.sigmoid(tensors["state_logit"])
        q_probability = torch.sigmoid(tensors["action_logit"])
        labels = batch["labels"][:, None].expand_as(state_probability)

        state_scores.append(state_probability[valid].cpu().numpy())
        q_scores.append(q_probability[valid].cpu().numpy())
        q_targets.append(tensors["q_target"][valid].cpu().numpy())
        state_targets.append(tensors["state_target"][valid].cpu().numpy())
        outcome_labels.append(labels[valid].cpu().numpy())
        state_logits.append(tensors["state_logit"][valid].cpu().numpy())
        q_logits.append(tensors["action_logit"][valid].cpu().numpy())

        dyn_mask = tensors["dynamics_mask"]
        if torch.any(dyn_mask):
            dyn_error = F.smooth_l1_loss(
                tensors["predicted_next_context"],
                tensors["next_context"],
                reduction="none",
                beta=args.dynamics_huber_beta,
            ).mean(dim=-1)
            dyn_cosine = F.cosine_similarity(
                tensors["predicted_next_context"],
                tensors["next_context"],
                dim=-1,
            )
            dynamics_errors.append(dyn_error[dyn_mask].cpu().numpy())
            dynamics_cosines.append(dyn_cosine[dyn_mask].cpu().numpy())
            target_progress.append(
                (tensors["next_logit"] - tensors["state_logit"])[dyn_mask]
                .cpu()
                .numpy()
            )
            predicted_progress.append(
                tensors["action_delta"][dyn_mask].cpu().numpy()
            )

        full_chunks = dyn_mask & batch["action_mask"].all(dim=-1)
        if int(full_chunks.sum()) > 1:
            flat_context = tensors["context"][full_chunks]
            flat_actions = batch["actions"][full_chunks]
            flat_action_mask = batch["action_mask"][full_chunks]
            shuffled_actions = torch.roll(flat_actions, shifts=1, dims=0)
            shuffled_action_mask = torch.roll(flat_action_mask, shifts=1, dims=0)
            shuffled_q = model.action_value_logit(
                flat_context[:, None, :],
                shuffled_actions[:, None, :, :],
                shuffled_action_mask[:, None, :],
            ).squeeze(1)
            logged_q = tensors["action_logit"][full_chunks]
            shuffled_differences.append(
                torch.abs(logged_q - shuffled_q).cpu().numpy()
            )
            target_next = tensors["next_context"][full_chunks]
            logged_next = tensors["predicted_next_context"][full_chunks]
            shuffled_next = model.predict_next_context(
                flat_context[:, None, :],
                shuffled_actions[:, None, :, :],
                shuffled_action_mask[:, None, :],
            ).squeeze(1)
            logged_error = F.smooth_l1_loss(
                logged_next,
                target_next,
                reduction="none",
                beta=args.dynamics_huber_beta,
            ).mean(dim=-1)
            shuffled_error = F.smooth_l1_loss(
                shuffled_next,
                target_next,
                reduction="none",
                beta=args.dynamics_huber_beta,
            ).mean(dim=-1)
            logged_action_dynamics_errors.append(logged_error.cpu().numpy())
            shuffled_action_dynamics_errors.append(shuffled_error.cpu().numpy())

        for episode in selected:
            row_indices.append(episode_indices(arrays["offsets"], int(episode)))

    state_scores_np = np.concatenate(state_scores)
    q_scores_np = np.concatenate(q_scores)
    q_targets_np = np.concatenate(q_targets)
    state_targets_np = np.concatenate(state_targets)
    labels_np = np.concatenate(outcome_labels)
    state_logits_np = np.concatenate(state_logits)
    q_logits_np = np.concatenate(q_logits)
    target_progress_np = (
        np.concatenate(target_progress) if target_progress else np.zeros(0)
    )
    predicted_progress_np = (
        np.concatenate(predicted_progress) if predicted_progress else np.zeros(0)
    )
    dynamics_errors_np = (
        np.concatenate(dynamics_errors) if dynamics_errors else np.zeros(0)
    )
    dynamics_cosines_np = (
        np.concatenate(dynamics_cosines) if dynamics_cosines else np.zeros(0)
    )
    shuffled_np = (
        np.concatenate(shuffled_differences) if shuffled_differences else np.zeros(0)
    )
    logged_action_dynamics_np = (
        np.concatenate(logged_action_dynamics_errors)
        if logged_action_dynamics_errors
        else np.zeros(0)
    )
    shuffled_action_dynamics_np = (
        np.concatenate(shuffled_action_dynamics_errors)
        if shuffled_action_dynamics_errors
        else np.zeros(0)
    )
    conditioning_gain = (
        float(np.mean(shuffled_action_dynamics_np - logged_action_dynamics_np))
        if len(logged_action_dynamics_np)
        else float("nan")
    )
    relative_conditioning_gain = (
        conditioning_gain / max(float(np.mean(logged_action_dynamics_np)), 1e-8)
        if len(logged_action_dynamics_np)
        else float("nan")
    )
    q_clipped = np.clip(q_scores_np, 1e-6, 1.0 - 1e-6)
    q_target_bce = float(
        -np.mean(
            q_targets_np * np.log(q_clipped)
            + (1.0 - q_targets_np) * np.log(1.0 - q_clipped)
        )
    )
    metrics = {
        "num_episodes": int(len(episodes)),
        "num_transitions": int(len(labels_np)),
        "state_outcome": binary_metrics(labels_np, state_scores_np),
        "state_mc_target": probability_target_metrics(
            state_targets_np,
            state_scores_np,
        ),
        "q_outcome": binary_metrics(labels_np, q_scores_np),
        "q_target_bce": q_target_bce,
        "q_target_mse": float(np.mean((q_scores_np - q_targets_np) ** 2)),
        "q_target_pearson": safe_pearson(q_scores_np, q_targets_np),
        "dynamics_huber": (
            float(np.mean(dynamics_errors_np)) if len(dynamics_errors_np) else float("nan")
        ),
        "dynamics_cosine": (
            float(np.mean(dynamics_cosines_np)) if len(dynamics_cosines_np) else float("nan")
        ),
        "progress_logit_pearson": safe_pearson(
            predicted_progress_np,
            target_progress_np,
        ),
        "progress_logit_mae": (
            float(np.mean(np.abs(predicted_progress_np - target_progress_np)))
            if len(target_progress_np)
            else float("nan")
        ),
        "action_delta_abs_mean": float(np.mean(np.abs(q_logits_np - state_logits_np))),
        "shuffled_action_q_abs_difference": (
            float(np.mean(shuffled_np)) if len(shuffled_np) else float("nan")
        ),
        "logged_action_dynamics_huber": (
            float(np.mean(logged_action_dynamics_np))
            if len(logged_action_dynamics_np)
            else float("nan")
        ),
        "shuffled_action_dynamics_huber": (
            float(np.mean(shuffled_action_dynamics_np))
            if len(shuffled_action_dynamics_np)
            else float("nan")
        ),
        "action_conditioning_gain": conditioning_gain,
        "relative_action_conditioning_gain": relative_conditioning_gain,
    }
    predictions = None
    if return_predictions:
        predictions = {
            "row_indices": np.concatenate(row_indices),
            "state_probability": state_scores_np,
            "q_probability": q_scores_np,
            "q_target": q_targets_np,
            "state_mc_target": state_targets_np,
            "state_logit": state_logits_np,
            "q_logit": q_logits_np,
        }
    return metrics, predictions


def save_checkpoint(
    path: Path,
    model,
    target_model,
    stats: dict,
    splits: dict,
    arrays: dict,
    args,
    step: int,
    metric_name: str,
    metric_value: float,
) -> None:
    args_dict = vars(args).copy()
    checkpoint = {
            "model": model.state_dict(),
            "target_model": target_model.state_dict(),
            "stats": stats,
            "splits": splits,
            "args": args_dict,
            "feature_dim": int(
                model.feature_mean.numel()
                if hasattr(model, "feature_mean")
                else arrays["features"].shape[-1]
            ),
            "prediction_horizon": int(arrays["actions"].shape[1]),
            "action_dim": int(arrays["actions"].shape[2]),
            "best_step": int(step),
            "best_quality": (float(metric_value),),
            "checkpoint_metric": metric_name,
            "scorer_semantics": "successor_bellman_success_probability",
            "critic_input": (
                "processed_rgb_history_and_raw_action_chunk"
                if args.model_arch == "v4"
                else "actor_feature_history_and_normalized_action_chunk"
            ),
            "self_contained_rgb_critic": bool(args.model_arch == "v4"),
        }
    if args.model_arch == "v4":
        checkpoint["rgb_encoder_spec"] = copy.deepcopy(model.rgb_encoder_spec)
    torch.save(checkpoint, path)


def train(args) -> dict:
    if args.contrast_weight != 0.0:
        raise ValueError(
            "contrastive ranking is intentionally disabled for this experiment; "
            "set --contrast-weight 0"
        )
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0, 1]")
    if not (0 < args.warmup_steps < args.total_steps):
        raise ValueError("--warmup-steps must be between 1 and total_steps - 1")
    if not (0.0 <= args.target_label_smoothing < 0.5):
        raise ValueError("--target-label-smoothing must be in [0, 0.5)")
    if args.encoder_unfreeze_step > 0 and not (
        args.warmup_steps < args.encoder_unfreeze_step <= args.total_steps
    ):
        raise ValueError(
            "--encoder-unfreeze-step must be after warmup and no later than total steps"
        )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = None
    rgb_encoder = None
    rgb_encoder_spec = None
    if args.model_arch == "v4":
        if args.rollouts is None:
            raise ValueError("end-to-end V4 training requires --rollouts")
        encoder_init_checkpoint = (
            args.encoder_init_checkpoint
            if args.encoder_init_checkpoint is not None
            else args.expected_dp_checkpoint
        )
        if encoder_init_checkpoint is None:
            raise ValueError(
                "V4 requires --encoder-init-checkpoint"
            )
        arrays, metadata = build_raw_transition_table(
            args.rollouts,
            args.action_horizon,
            args.max_episodes,
        )
        metadata["encoder_init_checkpoint"] = str(encoder_init_checkpoint)
        labels = arrays["labels"]
        offsets = arrays["offsets"]
        action_mask = arrays["action_mask"]
        rewards = arrays["rewards"]
        terminals = arrays["terminals"]
        args.step_gamma = float(
            args.gamma ** (1.0 / float(metadata["q_action_horizon"]))
        )
        arrays["mc_returns"] = compute_discounted_mc_returns(
            rewards,
            action_mask,
            terminals,
            offsets,
            args.step_gamma,
        )
        # Raw actions are temporarily exposed so the RGB normalizer pass can
        # reuse the ordinary episode batch loader.
        arrays["actions"] = arrays["actions_raw"]
        splits = stratified_split(labels, args.seed)
        rgb_encoder, rgb_encoder_spec = load_rgb_encoder_from_actor_checkpoint(
            encoder_init_checkpoint,
            device,
            root=ROOT,
        )
        rgb_encoder.requires_grad_(False)
        rgb_encoder.eval()
        dataset = h5py.File(args.rollouts, "r")
        stats = compute_raw_rgb_normalizers(
            dataset=dataset,
            arrays=arrays,
            train_episodes=splits["train"],
            rgb_encoder=rgb_encoder,
            observation_shapes=rgb_encoder_spec["observation_shapes"],
            device=device,
            episode_batch_size=args.normalizer_episode_batch_size,
        )
        actions = (
            arrays["actions_raw"] - stats["action_mean"][None, None, :]
        ) / stats["action_std"][None, None, :]
        actions = actions.astype(np.float32)
        actions[~action_mask] = 0.0
        arrays["actions"] = actions
        feature_dim = int(rgb_encoder.output_shape()[0]) * int(
            rgb_encoder_spec["observation_horizon"]
        )
        transition_audit = {
            "terminal_rows": int(terminals.sum()),
            "positive_reward_rows": int(np.sum(rewards.sum(axis=1) > 0.0)),
            "raw_rgb_forward_in_training": True,
            "chunk_gamma": float(args.gamma),
            "derived_step_gamma": float(args.step_gamma),
            "mc_return_min": float(arrays["mc_returns"].min()),
            "mc_return_mean": float(arrays["mc_returns"].mean()),
            "mc_return_max": float(arrays["mc_returns"].max()),
        }
        args.input_mode = "raw_rgb_hdf5"
    else:
        if args.features is None:
            raise ValueError("legacy V3 training requires --features")
        raw = np.load(args.features, allow_pickle=False)
        metadata = validate_cache(raw)
        features_raw = raw["features"].astype(np.float32)
        next_features_raw = raw["next_features"].astype(np.float32)
        actions_raw = raw["actions"].astype(np.float32)
        action_mask = raw["action_valid_mask"].astype(np.bool_)
        rewards = raw["chunk_rewards"].astype(np.float32)
        terminals = raw["terminals"].astype(np.bool_)
        labels = raw["episode_success_labels"].astype(np.float32)
        offsets = raw["episode_offsets"].astype(np.int64)
        args.step_gamma = float(
            args.gamma ** (1.0 / float(metadata["q_action_horizon"]))
        )
        mc_returns = compute_discounted_mc_returns(
            rewards, action_mask, terminals, offsets, args.step_gamma
        )
        transition_audit = validate_transition_rows(
            features_raw,
            next_features_raw,
            rewards,
            terminals,
            labels,
            offsets,
        )
        transition_audit.update(
            {
                "chunk_gamma": float(args.gamma),
                "derived_step_gamma": float(args.step_gamma),
                "mc_return_min": float(mc_returns.min()),
                "mc_return_mean": float(mc_returns.mean()),
                "mc_return_max": float(mc_returns.max()),
            }
        )
        splits = stratified_split(labels, args.seed)
        stats = compute_normalizers(
            features_raw, actions_raw, action_mask, offsets, splits["train"]
        )
        features, next_features, actions = normalize_arrays(
            features_raw,
            next_features_raw,
            actions_raw,
            action_mask,
            stats,
        )
        arrays = {
            "features": features,
            "next_features": next_features,
            "actions": actions,
            "action_mask": action_mask,
            "rewards": rewards,
            "terminals": terminals,
            "mc_returns": mc_returns,
            "labels": labels,
            "offsets": offsets,
        }
        feature_dim = int(features.shape[-1])
        args.input_mode = "cached_features_legacy"

    model = make_causal_prefix_model(
        model_arch=args.model_arch,
        feature_dim=feature_dim,
        prediction_horizon=arrays["actions"].shape[1],
        action_dim=arrays["actions"].shape[2],
        hidden_dim=args.hidden_dim,
        action_hidden_dim=args.action_hidden_dim,
        dropout=args.dropout,
        action_num_heads=args.action_num_heads,
        action_conv_layers=args.action_conv_layers,
        prefix_conv_layers=args.prefix_conv_layers,
        rgb_encoder=rgb_encoder,
        observation_shapes=(
            rgb_encoder_spec["observation_shapes"]
            if rgb_encoder_spec is not None
            else None
        ),
        observation_horizon=(
            rgb_encoder_spec["observation_horizon"]
            if rgb_encoder_spec is not None
            else None
        ),
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
    ).to(device)
    if args.model_arch == "v4":
        model.rgb_encoder_spec = rgb_encoder_spec
        model.rgb_encoder.requires_grad_(False)
        model.rgb_encoder.eval()
    with torch.no_grad():
        model.state_head[-1].bias.fill_(args.initial_state_logit_bias)
    target_model = copy.deepcopy(model).to(device).eval()
    target_model.requires_grad_(False)

    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        args,
        args.warmup_steps,
    )

    args.target_outcome = "success"
    args.q_boundary_stride = metadata["q_boundary_stride"]
    if args.expected_dp_checkpoint is None:
        args.expected_dp_checkpoint = Path(metadata["dp_checkpoint"]).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_q_bce = float("inf")
    best_dynamics = float("inf")
    best_progress = -float("inf")
    best_action_conditioning = -float("inf")
    best_selection_score = -float("inf")
    best_state_key = (-float("inf"), -float("inf"))
    best_steps = {}
    training_stage = "state_warmup"
    all_episodes = np.arange(len(labels), dtype=np.int64)

    print(
        json.dumps(
            {
                "event": "successor_critic_training_start",
                "features": str(args.features),
                "metadata": metadata,
                "transition_audit": transition_audit,
                "episodes": int(len(labels)),
                "success": int(labels.sum()),
                "failure": int(len(labels) - labels.sum()),
                "discount": {
                    "semantics": "gamma is the discount over one full action chunk",
                    "chunk_gamma": float(args.gamma),
                    "derived_step_gamma": float(args.step_gamma),
                },
                "target_label_smoothing": args.target_label_smoothing,
                "training_schedule": (
                    "warm up V and dynamics; restore best warmup V; freeze V/context; "
                    "then train only action-conditioned successor dynamics and Q"
                ),
                "contrast_weight": args.contrast_weight,
                "model_arch": args.model_arch,
                "self_contained_rgb_critic": bool(args.model_arch == "v4"),
                "rgb_encoder_training": (
                    "initialized from DP actor and frozen for all stages"
                    if args.model_arch == "v4"
                    else None
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    for step in range(1, args.total_steps + 1):
        if step == args.warmup_steps + 1:
            warmup_path = args.output_dir / "best_state_warmup.pt"
            if not warmup_path.exists():
                raise RuntimeError("state warmup completed without a checkpoint")
            warmup_checkpoint = torch.load(
                warmup_path,
                map_location=device,
                weights_only=False,
            )
            restore_state_modules(model, warmup_checkpoint)
            set_state_trainable(model, False)
            target_model = copy.deepcopy(model).to(device).eval()
            target_model.requires_grad_(False)
            optimizer, scheduler = make_optimizer_and_scheduler(
                model,
                args,
                args.total_steps - args.warmup_steps,
            )
            training_stage = "frozen_value_action_learning"
            print(
                json.dumps(
                    {
                        "event": "freeze_best_state_value",
                        "step": step,
                        "restored_checkpoint": str(warmup_path),
                        "restored_from_step": int(warmup_checkpoint["best_step"]),
                        "trainable_parameters": int(
                            sum(
                                parameter.numel()
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            )
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )
        if (
            args.model_arch == "v4"
            and args.encoder_unfreeze_step > 0
            and step == args.encoder_unfreeze_step
        ):
            set_state_trainable(model, True)
            model.rgb_encoder.requires_grad_(True)
            # Keep crop randomizers deterministic while still allowing
            # gradients through the visual encoder.
            model.rgb_encoder.eval()
            optimizer, scheduler = make_optimizer_and_scheduler(
                model,
                args,
                args.total_steps - step + 1,
            )
            training_stage = "joint_rgb_finetune"
            print(
                json.dumps(
                    {
                        "event": "unfreeze_critic_rgb_encoder",
                        "step": step,
                        "encoder_lr": args.lr * args.encoder_lr_scale,
                        "reference_weight": args.encoder_reference_weight,
                        "trainable_parameters": int(
                            sum(
                                parameter.numel()
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            )
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )
        model.train()
        set_frozen_rgb_encoder_eval(model)
        if training_stage == "frozen_value_action_learning":
            set_frozen_state_eval(model)
        episodes = balanced_episode_batch(
            labels,
            splits["train"],
            args.episode_batch_size,
            rng,
        )
        if dataset is not None:
            batch = padded_rgb_batch(
                dataset=dataset,
                arrays=arrays,
                episodes=episodes,
                observation_shapes=model.observation_shapes,
                device=device,
            )
        else:
            batch = padded_batch(
                features=features,
                actions=actions,
                action_mask=action_mask,
                rewards=rewards,
                terminals=terminals,
                mc_returns=mc_returns,
                labels=labels,
                offsets=offsets,
                episodes=episodes,
                device=device,
            )
        if training_stage == "state_warmup":
            q_scale = 0.0
        else:
            q_scale = min(
                1.0,
                (step - args.warmup_steps) / max(args.q_ramp_steps, 1),
            )
        optimizer.zero_grad(set_to_none=True)
        total, tensors = compute_losses(
            model,
            target_model,
            batch,
            args,
            q_scale=q_scale,
        )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        if training_stage == "state_warmup":
            update_target(target_model, model, args.target_ema_decay)
        scheduler.step()

        if (
            step % args.eval_every == 0
            or step == args.warmup_steps
            or step == args.total_steps
        ):
            validation, _ = evaluate(
                model,
                target_model,
                arrays,
                splits["val"],
                args,
                device,
            )
            record = {
                "step": step,
                "training_stage": training_stage,
                "q_scale": float(q_scale),
                "loss": float(total.detach()),
                "state_loss": float(tensors["state_loss"].detach()),
                "dynamics_loss": float(tensors["dynamics_loss"].detach()),
                "dynamics_cosine_loss": float(
                    tensors["dynamics_cosine_loss"].detach()
                ),
                "q_loss": float(tensors["q_loss"].detach()),
                "value_consistency_loss": float(
                    tensors["value_consistency_loss"].detach()
                ),
                "contrast_loss": 0.0,
                "grad_norm": float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "validation": validation,
            }
            if training_stage == "frozen_value_action_learning":
                record["action_selection_validation_score"] = (
                    action_selection_validation_score(validation)
                )
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)

            if training_stage == "state_warmup":
                state_auc = finite_metric(
                    validation["state_outcome"]["roc_auc"],
                    -1.0,
                )
                state_target_bce = finite_metric(
                    validation["state_mc_target"]["bce"],
                    20.0,
                )
                state_key = (state_auc, -state_target_bce)
                if state_key > best_state_key:
                    best_state_key = state_key
                    best_steps["state_warmup"] = step
                    save_checkpoint(
                        args.output_dir / "best_state_warmup.pt",
                        model,
                        target_model,
                        stats,
                        splits,
                        arrays,
                        args,
                        step,
                        "state_outcome_auc_then_mc_bce",
                        state_auc,
                    )
                    save_checkpoint(
                        args.output_dir / "best_state_auc.pt",
                        model,
                        target_model,
                        stats,
                        splits,
                        arrays,
                        args,
                        step,
                        "state_outcome_auc_then_mc_bce",
                        state_auc,
                    )
                continue

            selection_score = record["action_selection_validation_score"]
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_steps["action_selection"] = step
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    target_model,
                    stats,
                    splits,
                    arrays,
                    args,
                    step,
                    "action_selection_composite",
                    selection_score,
                )
            if validation["q_target_bce"] < best_q_bce:
                best_q_bce = validation["q_target_bce"]
                best_steps["q_bellman"] = step
                save_checkpoint(
                    args.output_dir / "best_q_bellman.pt",
                    model,
                    target_model,
                    stats,
                    splits,
                    arrays,
                    args,
                    step,
                    "q_target_bce",
                    -best_q_bce,
                )
            if validation["dynamics_huber"] < best_dynamics:
                best_dynamics = validation["dynamics_huber"]
                best_steps["dynamics"] = step
                save_checkpoint(
                    args.output_dir / "best_dynamics.pt",
                    model,
                    target_model,
                    stats,
                    splits,
                    arrays,
                    args,
                    step,
                    "dynamics_huber",
                    -best_dynamics,
                )
            progress = finite_metric(validation["progress_logit_pearson"], -1.0)
            if progress > best_progress:
                best_progress = progress
                best_steps["progress"] = step
                save_checkpoint(
                    args.output_dir / "best_progress.pt",
                    model,
                    target_model,
                    stats,
                    splits,
                    arrays,
                    args,
                    step,
                    "progress_logit_pearson",
                    best_progress,
                )
            conditioning = finite_metric(
                validation["relative_action_conditioning_gain"],
                -1.0,
            )
            if conditioning > best_action_conditioning:
                best_action_conditioning = conditioning
                best_steps["action_conditioning"] = step
                save_checkpoint(
                    args.output_dir / "best_action_conditioning.pt",
                    model,
                    target_model,
                    stats,
                    splits,
                    arrays,
                    args,
                    step,
                    "relative_action_conditioning_gain",
                    best_action_conditioning,
                )

    save_checkpoint(
        args.output_dir / "final.pt",
        model,
        target_model,
        stats,
        splits,
        arrays,
        args,
        args.total_steps,
        "final",
        float("nan"),
    )
    best_path = args.output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("no post-warmup best checkpoint was saved")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    target_model.load_state_dict(best["target_model"])
    test_metrics, predictions = evaluate(
        model,
        target_model,
        arrays,
        splits["test"],
        args,
        device,
        return_predictions=True,
    )
    all_metrics, all_predictions = evaluate(
        model,
        target_model,
        arrays,
        all_episodes,
        args,
        device,
        return_predictions=True,
    )

    order = np.argsort(all_predictions["row_indices"])
    np.savez_compressed(
        args.output_dir / "successor_critic_predictions.npz",
        row_indices=all_predictions["row_indices"][order],
        state_probability=all_predictions["state_probability"][order],
        q_probability=all_predictions["q_probability"][order],
        q_target=all_predictions["q_target"][order],
        state_mc_target=all_predictions["state_mc_target"][order],
        state_logit=all_predictions["state_logit"][order],
        q_logit=all_predictions["q_logit"][order],
        episode_offsets=offsets,
        episode_success_labels=labels,
    )

    args_json = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary = {
        "features": str(args.features),
        "output_dir": str(args.output_dir),
        "checkpoint": str(best_path),
        "model_arch": args.model_arch,
        "self_contained_rgb_critic": bool(args.model_arch == "v4"),
        "scorer_semantics": "Q=V(predicted_action_conditioned_successor)",
        "discount_semantics": {
            "gamma_scope": "one complete executed action chunk",
            "chunk_gamma": float(args.gamma),
            "derived_step_gamma": float(args.step_gamma),
        },
        "training_schedule": "best warmup V is frozen during action/Q training",
        "contrastive_loss_weight": float(args.contrast_weight),
        "metadata": metadata,
        "transition_audit": transition_audit,
        "num_episodes": int(len(labels)),
        "num_success_episodes": int(labels.sum()),
        "num_failure_episodes": int(len(labels) - labels.sum()),
        "best_steps": best_steps,
        "best_validation_action_selection_score": best_selection_score,
        "best_validation_q_target_bce": best_q_bce,
        "best_validation_progress_logit_pearson": best_progress,
        "best_validation_relative_action_conditioning_gain": (
            best_action_conditioning
        ),
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "args": args_json,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=None,
        help="raw rollout HDF5 used directly by end-to-end V4 training",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="legacy V3 feature cache; ignored by end-to-end V4",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dp-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--encoder-init-checkpoint",
        type=Path,
        default=None,
        help="actor used once to initialize the critic-owned RGB encoder",
    )
    parser.add_argument(
        "--model-arch",
        choices=("v3", "v4"),
        default="v4",
        help=(
            "v4 bundles an actor-initialized RGB encoder and normalization into "
            "the critic; v3 is retained only for legacy checkpoints"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-num-heads", type=int, default=4)
    parser.add_argument("--action-conv-layers", type=int, default=2)
    parser.add_argument("--prefix-conv-layers", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=750)
    parser.add_argument("--q-ramp-steps", type=int, default=500)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--eval-episode-batch-size", type=int, default=2)
    parser.add_argument("--normalizer-episode-batch-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--encoder-unfreeze-step",
        type=int,
        default=4000,
        help="set <=0 to keep the critic RGB encoder frozen for all training",
    )
    parser.add_argument("--encoder-lr-scale", type=float, default=0.05)
    parser.add_argument("--encoder-reference-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "discount over one complete executed action chunk; the trainer "
            "derives the matching per-simulator-step discount"
        ),
    )
    parser.add_argument("--target-label-smoothing", type=float, default=0.01)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-cosine-weight", type=float, default=0.1)
    parser.add_argument("--dynamics-huber-beta", type=float, default=0.1)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--value-consistency-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument("--target-ema-decay", type=float, default=0.995)
    parser.add_argument("--initial-state-logit-bias", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.features is not None:
        args.features = args.features.resolve()
    if args.rollouts is not None:
        args.rollouts = args.rollouts.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.expected_dp_checkpoint is not None:
        args.expected_dp_checkpoint = args.expected_dp_checkpoint.resolve()
    if args.encoder_init_checkpoint is not None:
        args.encoder_init_checkpoint = args.encoder_init_checkpoint.resolve()
    train(args)


if __name__ == "__main__":
    main()
