#!/usr/bin/env python3
"""Learn causal prefix outcome risk / advantage from rollout outcomes.

This experiment uses only rollout-level success / failure outcomes for
optimization. At policy boundary t, a causal GRU summarizes observation
features up to t. Two predictions are then made:

* V(h_t): target outcome probability from the observation prefix;
* Q(h_t, a_t): target outcome probability after adding the executed action chunk.

Q is parameterized as V + delta. With the default ``--target-outcome failure``,
positive delta is incremental action risk in log-odds units. With
``--target-outcome success``, positive delta is incremental action success
advantage. Privileged failed-grasp and safe-reach labels stored in the feature
cache are used only after training for simulator-side localization evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robomimic.models.prefix_risk_nets import CausalPrefixRisk, make_causal_prefix_model
from train_rgb_dp_hazard_mil import (
    average_precision,
    balanced_episode_batch,
    rank_auc,
    stratified_split,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "rollouts/rgb_dp/hazard_mil/chunk_features.npz"
DEFAULT_OUTPUT = ROOT / "trained_models/rgb_dp_causal_prefix_risk"


def episode_indices(offsets: np.ndarray, episode: int) -> np.ndarray:
    return np.arange(offsets[episode], offsets[episode + 1], dtype=np.int64)


def normalizers(
    features: np.ndarray,
    actions: np.ndarray,
    offsets: np.ndarray,
    train_episodes: np.ndarray,
) -> dict[str, np.ndarray]:
    indices = np.concatenate(
        [episode_indices(offsets, int(episode)) for episode in train_episodes]
    )
    feature_mean = features[indices].mean(axis=0)
    feature_std = np.maximum(features[indices].std(axis=0), 1e-4)
    action_mean = actions[indices].reshape(-1, actions.shape[-1]).mean(axis=0)
    action_std = np.maximum(
        actions[indices].reshape(-1, actions.shape[-1]).std(axis=0),
        1e-4,
    )
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def normalize(
    features: np.ndarray,
    actions: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    features = (features - stats["feature_mean"]) / stats["feature_std"]
    actions = (actions - stats["action_mean"]) / stats["action_std"]
    return features.astype(np.float32), actions.astype(np.float32)


def true_success_episode_indices(
    labels: np.ndarray,
    target_outcome: str,
) -> np.ndarray:
    if target_outcome == "failure":
        return np.flatnonzero(labels < 0.5)
    if target_outcome == "success":
        return np.flatnonzero(labels > 0.5)
    raise ValueError(f"unknown target_outcome={target_outcome}")


def true_success_rows(labels: torch.Tensor, target_outcome: str) -> torch.Tensor:
    if target_outcome == "failure":
        return labels < 0.5
    if target_outcome == "success":
        return labels > 0.5

def true_failure_rows(labels: torch.Tensor, target_outcome: str) -> torch.Tensor:
    return ~true_success_rows(labels, target_outcome)


def success_action_indices_by_step(
    steps: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    target_outcome: str,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grouped: dict[int, list[int]] = {}
    global_indices = []
    for episode in true_success_episode_indices(labels, target_outcome):
        sl = slice(offsets[episode], offsets[episode + 1])
        episode_indices_local = np.arange(sl.start, sl.stop, dtype=np.int64)
        global_indices.extend(episode_indices_local.tolist())
        for index in episode_indices_local:
            grouped.setdefault(int(steps[index]), []).append(int(index))
    by_step = {step: np.asarray(indices, dtype=np.int64) for step, indices in grouped.items()}
    return by_step, np.asarray(global_indices, dtype=np.int64)


def sample_success_reference_actions(
    *,
    actions: np.ndarray,
    steps: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    lengths: torch.Tensor,
    by_step: dict[int, np.ndarray],
    all_success_indices: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
    max_length = int(lengths_np.max())
    reference_actions = np.zeros(
        (len(episodes), max_length, num_negatives, actions.shape[1], actions.shape[2]),
        dtype=np.float32,
    )
    if len(all_success_indices) == 0:
        raise RuntimeError("cannot build action contrast references without success chunks")
    for row, episode in enumerate(episodes):
        start = int(offsets[int(episode)])
        for time_index in range(int(lengths_np[row])):
            chunk_index = start + time_index
            pool = by_step.get(int(steps[chunk_index]), all_success_indices)
            if len(pool) == 0:
                pool = all_success_indices
            sampled = rng.choice(pool, size=num_negatives, replace=True)
            reference_actions[row, time_index] = actions[sampled]
    return torch.from_numpy(reference_actions).to(device)


def build_matched_success_indices(
    *,
    features: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    target_outcome: str,
    top_k: int,
    step_window: int,
    device: torch.device,
) -> np.ndarray:
    success_indices = []
    for episode in true_success_episode_indices(labels, target_outcome):
        success_indices.extend(range(int(offsets[episode]), int(offsets[episode + 1])))
    success_indices = np.asarray(success_indices, dtype=np.int64)
    if len(success_indices) == 0:
        raise RuntimeError("cannot build matched action references without success chunks")

    result = np.empty((len(features), top_k), dtype=np.int64)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    success_steps = steps[success_indices]
    all_success_tensor = torch.as_tensor(success_indices, dtype=torch.long, device=device)
    unique_steps = np.unique(steps)
    for step_value in unique_steps:
        query_indices = np.flatnonzero(steps == step_value).astype(np.int64)
        candidate_mask = np.abs(success_steps - int(step_value)) <= step_window
        candidate_indices = success_indices[candidate_mask]
        if len(candidate_indices) == 0:
            candidate_indices = success_indices
        candidate_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=device)
        query_tensor = torch.as_tensor(query_indices, dtype=torch.long, device=device)
        query_features = F.normalize(feature_tensor[query_tensor], dim=-1)
        candidate_features = F.normalize(feature_tensor[candidate_tensor], dim=-1)
        similarity = query_features @ candidate_features.T
        selected_k = min(top_k, int(candidate_tensor.numel()))
        nearest = torch.topk(similarity, k=selected_k, dim=1).indices
        matched = candidate_tensor[nearest].detach().cpu().numpy()
        if selected_k < top_k:
            pad = np.repeat(matched[:, -1:], top_k - selected_k, axis=1)
            matched = np.concatenate([matched, pad], axis=1)
        result[query_indices] = matched
    return result


def sample_matched_reference_actions(
    *,
    actions: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    lengths: torch.Tensor,
    matched_indices: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
    max_length = int(lengths_np.max())
    reference_actions = np.zeros(
        (len(episodes), max_length, num_negatives, actions.shape[1], actions.shape[2]),
        dtype=np.float32,
    )
    for row, episode in enumerate(episodes):
        start = int(offsets[int(episode)])
        for time_index in range(int(lengths_np[row])):
            chunk_index = start + time_index
            pool = matched_indices[chunk_index]
            selected = rng.choice(pool, size=num_negatives, replace=True)
            reference_actions[row, time_index] = actions[selected]
    return torch.from_numpy(reference_actions).to(device)


def action_indices_by_outcome_and_step(
    steps: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    *,
    positive: bool,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return global chunk indices grouped by policy step and outcome label."""
    grouped: dict[int, list[int]] = {}
    global_indices: list[int] = []
    if positive:
        selected_episodes = np.flatnonzero(labels > 0.5)
    else:
        selected_episodes = np.flatnonzero(labels < 0.5)
    for episode in selected_episodes:
        sl = slice(int(offsets[episode]), int(offsets[episode + 1]))
        episode_indices_local = np.arange(sl.start, sl.stop, dtype=np.int64)
        global_indices.extend(episode_indices_local.tolist())
        for index in episode_indices_local:
            grouped.setdefault(int(steps[index]), []).append(int(index))
    by_step = {
        step: np.asarray(indices, dtype=np.int64)
        for step, indices in grouped.items()
    }
    return by_step, np.asarray(global_indices, dtype=np.int64)


def sample_outcome_pairwise_actions(
    *,
    actions: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    lengths: torch.Tensor,
    positive_by_step: dict[int, np.ndarray],
    negative_by_step: dict[int, np.ndarray],
    all_positive_indices: np.ndarray,
    all_negative_indices: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build fixed-context pairwise positive / negative action chunks.

    For a positive-outcome episode, the logged chunk is treated as the positive
    action and negatives are sampled from opposite-outcome chunks at the same
    policy step. For a negative-outcome episode, the logged chunk is treated as
    negative and positives are sampled from positive-outcome chunks at the same
    policy step. This creates a simple action-ranking signal under the current
    prefix context without requiring simulator branch labels.
    """
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
    max_length = int(lengths_np.max())
    shape = (
        len(episodes),
        max_length,
        num_negatives,
        actions.shape[1],
        actions.shape[2],
    )
    positive_actions = np.zeros(shape, dtype=np.float32)
    negative_actions = np.zeros(shape, dtype=np.float32)
    pairwise_mask = np.zeros(shape[:3], dtype=np.bool_)
    if len(all_positive_indices) == 0 or len(all_negative_indices) == 0:
        return {
            "pairwise_positive_actions": torch.from_numpy(positive_actions).to(device),
            "pairwise_negative_actions": torch.from_numpy(negative_actions).to(device),
            "pairwise_mask": torch.from_numpy(pairwise_mask).to(device),
        }
    for row, episode in enumerate(episodes):
        episode = int(episode)
        start = int(offsets[episode])
        episode_is_positive = bool(labels[episode] > 0.5)
        for time_index in range(int(lengths_np[row])):
            chunk_index = start + time_index
            step = int(steps[chunk_index])
            positive_pool = positive_by_step.get(step, all_positive_indices)
            negative_pool = negative_by_step.get(step, all_negative_indices)
            if len(positive_pool) == 0 or len(negative_pool) == 0:
                continue
            if episode_is_positive:
                sampled_negative = rng.choice(
                    negative_pool,
                    size=num_negatives,
                    replace=True,
                )
                positive_actions[row, time_index] = actions[chunk_index][None, :, :]
                negative_actions[row, time_index] = actions[sampled_negative]
            else:
                sampled_positive = rng.choice(
                    positive_pool,
                    size=num_negatives,
                    replace=True,
                )
                positive_actions[row, time_index] = actions[sampled_positive]
                negative_actions[row, time_index] = actions[chunk_index][None, :, :]
            pairwise_mask[row, time_index, :] = True
    return {
        "pairwise_positive_actions": torch.from_numpy(positive_actions).to(device),
        "pairwise_negative_actions": torch.from_numpy(negative_actions).to(device),
        "pairwise_mask": torch.from_numpy(pairwise_mask).to(device),
    }


def padded_episode_batch(
    *,
    features: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    lengths = np.asarray(
        [offsets[episode + 1] - offsets[episode] for episode in episodes],
        dtype=np.int64,
    )
    max_length = int(lengths.max())
    batch_features = np.zeros(
        (len(episodes), max_length, features.shape[-1]),
        dtype=np.float32,
    )
    batch_actions = np.zeros(
        (len(episodes), max_length, actions.shape[1], actions.shape[2]),
        dtype=np.float32,
    )
    mask = np.zeros((len(episodes), max_length), dtype=np.bool_)
    for row, episode in enumerate(episodes):
        sl = slice(offsets[episode], offsets[episode + 1])
        length = lengths[row]
        batch_features[row, :length] = features[sl]
        batch_actions[row, :length] = actions[sl]
        mask[row, :length] = True
    return {
        "features": torch.from_numpy(batch_features).to(device),
        "actions": torch.from_numpy(batch_actions).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "labels": torch.as_tensor(labels[episodes], device=device).float(),
        "lengths": torch.from_numpy(lengths).to(device),
    }


def episode_normalized_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    targets = labels[:, None].expand_as(logits)
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return ((elementwise * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def noisy_or_episode_probability(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(episodes), dtype=np.float32)
    for row, episode in enumerate(episodes):
        sl = slice(offsets[episode], offsets[episode + 1])
        chunk_probs = np.clip(probabilities[sl], 1e-6, 1.0 - 1e-6)
        result[row] = 1.0 - float(np.prod(1.0 - chunk_probs))
    return result


def noisy_and_episode_probability(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(episodes), dtype=np.float32)
    for row, episode in enumerate(episodes):
        sl = slice(offsets[episode], offsets[episode + 1])
        chunk_probs = np.clip(probabilities[sl], 1e-6, 1.0 - 1e-6)
        result[row] = float(np.prod(chunk_probs))
    return result


def mil_episode_probability(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    target_outcome: str,
) -> np.ndarray:
    if target_outcome == "failure":
        return noisy_or_episode_probability(probabilities, offsets, episodes)
    if target_outcome == "success":
        return noisy_and_episode_probability(probabilities, offsets, episodes)
    raise ValueError(f"unknown target_outcome={target_outcome}")


def noisy_or_episode_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Noisy-OR multiple-instance BCE for failure / hazard targets.

    A positive bag is positive if at least one valid chunk has high target
    probability. This matches failure-risk learning:

        P(failure episode) = 1 - prod_t (1 - P(bad chunk_t)).
    """
    valid_log_not_outcome = F.logsigmoid(-logits) * mask  # log(1-sigmoid(x)) = log(sigmoid(-x))
    log_not_outcome = valid_log_not_outcome.sum(dim=1)
    not_outcome_prob = torch.exp(log_not_outcome).clamp(1e-6, 1.0 - 1e-6)
    outcome_prob = 1.0 - not_outcome_prob
    loss = -(
        labels * torch.log(outcome_prob.clamp_min(1e-6))
        + (1.0 - labels) * torch.log(not_outcome_prob.clamp_min(1e-6))
    )
    return loss.mean()


def noisy_and_episode_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Noisy-AND multiple-instance BCE for success / goodness targets.

    A positive bag is positive only if all valid chunks have high target
    probability. This matches success learning:

        P(success episode) = prod_t P(good chunk_t).
    """
    valid_log_outcome = F.logsigmoid(logits) * mask
    log_outcome = valid_log_outcome.sum(dim=1)

    # Keep this loss in log-space. With many chunks and an initial bias such as
    # -5, prod_t sigmoid(logit_t) can be far below 1e-6. Computing the product
    # and clamping it would make positive success episodes receive zero
    # gradient exactly when they most need a strong gradient.
    safe_log_outcome = torch.minimum(
        log_outcome,
        log_outcome.new_full((), -torch.finfo(log_outcome.dtype).eps),
    )
    threshold = -math.log(2.0)
    log_not_outcome = torch.where(
        safe_log_outcome < threshold,
        torch.log1p(-torch.exp(safe_log_outcome)),
        torch.log(-torch.expm1(safe_log_outcome)),
    )
    loss = -(
        labels * log_outcome
        + (1.0 - labels) * log_not_outcome
    )
    return loss.mean()


def success_chunk_bce_failure_mil_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Asymmetric success-target episode loss.

    This is intended for ``target_outcome=success``.

    * Success episodes: every valid logged chunk is treated as success/good,
      but the loss is normalized by trajectory length.
    * Failure episodes: use a MIL complement saying the trajectory failed
      because not all chunks were good, i.e. at least one chunk was bad.

    This keeps the correct success/failure asymmetry while avoiding the
    excessive length-scaled pressure of full noisy-AND on successful episodes.
    """
    success_rows = labels > 0.5
    failure_rows = labels < 0.5
    losses = []

    if torch.any(success_rows):
        per_chunk_success_loss = F.softplus(-logits)
        success_episode_loss = (
            (per_chunk_success_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )
        losses.append(success_episode_loss[success_rows])

    if torch.any(failure_rows):
        valid_log_all_good = F.logsigmoid(logits) * mask
        log_all_good = valid_log_all_good.sum(dim=1)
        safe_log_all_good = torch.minimum(
            log_all_good,
            log_all_good.new_full((), -torch.finfo(log_all_good.dtype).eps),
        )
        threshold = -math.log(2.0)
        log_not_all_good = torch.where(
            safe_log_all_good < threshold,
            torch.log1p(-torch.exp(safe_log_all_good)),
            torch.log(-torch.expm1(safe_log_all_good)),
        )
        failure_episode_loss = -log_not_all_good
        losses.append(failure_episode_loss[failure_rows])

    if not losses:
        return logits.new_zeros(())
    return torch.cat(losses).mean()


def mil_episode_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    target_outcome: str,
    success_loss_mode: str = "noisy_and",
) -> torch.Tensor:
    """Target-aware multiple-instance episode BCE.

    ``labels`` are already converted to the configured target outcome before
    this function is called:

    * target_outcome=failure: label 1 means failed rollout.
    * target_outcome=success: label 1 means successful rollout.

    The per-chunk logits therefore always represent target-outcome evidence.
    Failure and success use different causal assumptions:

    * failure: at least one bad chunk can cause failure, so use noisy-OR.
    * success: all required chunks must be good, so use noisy-AND.
    """
    if target_outcome == "failure":
        return noisy_or_episode_bce(logits, labels, mask)
    if target_outcome == "success":
        if success_loss_mode == "noisy_and":
            return noisy_and_episode_bce(logits, labels, mask)
        if success_loss_mode == "chunk_bce_failure_mil":
            return success_chunk_bce_failure_mil_bce(logits, labels, mask)
        raise ValueError(f"unknown success_loss_mode={success_loss_mode}")
    raise ValueError(f"unknown target_outcome={target_outcome}")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = mask.float()
    if weights is not None:
        selected = selected * weights
    return (values * selected).sum() / selected.sum().clamp_min(1e-8)


def masked_corr_square(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.bool()
    if int(selected.sum().detach().cpu()) < 2:
        return first.new_zeros(())
    x = first[selected]
    y = second[selected]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-8))
    return ((x * y).sum() / denom).square()


def temporal_risk_weights(
    state_probability: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    stride: int,
    min_increase: float,
    normalize: bool,
) -> torch.Tensor:
    batch_size, horizon = state_probability.shape
    positions = torch.arange(horizon, device=state_probability.device)[None, :]
    last = (lengths.to(state_probability.device) - 1).clamp_min(0)[:, None]
    future_positions = torch.minimum(positions + int(stride), last).long()
    future = torch.gather(state_probability, dim=1, index=future_positions)
    increase = (future - state_probability - float(min_increase)).clamp_min(0.0)
    target_positive_mask = mask & (labels[:, None] > 0.5)
    weights = increase * target_positive_mask.float()
    if normalize:
        per_episode_sum = weights.sum(dim=1, keepdim=True)
        valid_positive = per_episode_sum > 1e-8
        normalized = weights / per_episode_sum.clamp_min(1e-8)
        weights = torch.where(valid_positive, normalized, weights)
    return weights


def temporal_future_difference(
    value: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return value[t + stride] - value[t] and a real-future mask.

    This is the direct supervision we need for Q(s, a) - V(s): an
    H-step action chunk should predict the change in learned prefix outcome
    potential after H logged decision points, not just satisfy an episode bag
    classifier.
    """
    batch_size, horizon = value.shape
    stride = max(int(stride), 1)
    positions = torch.arange(horizon, device=value.device)[None, :]
    future_positions = positions + stride
    valid_future = future_positions < lengths.to(value.device)[:, None]
    future_positions = torch.minimum(
        future_positions,
        (lengths.to(value.device) - 1).clamp_min(0)[:, None],
    ).long()
    future = torch.gather(value, dim=1, index=future_positions)
    return future - value, mask & valid_future


def compute_loss(
    model: CausalPrefixRisk,
    batch: dict[str, torch.Tensor],
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["features"], batch["actions"])
    mask = batch["mask"]
    labels = batch["labels"]
    zero = output["state_logit"].new_zeros(())
    stage = getattr(args, "training_stage", "joint")

    if args.objective == "per_chunk_bce":
        state_loss = episode_normalized_bce(
            output["state_logit"],
            labels,
            mask,
        )
        action_loss = episode_normalized_bce(
            output["action_logit"],
            labels,
            mask,
        )
    elif args.objective == "noisy_or_mil":
        state_loss = mil_episode_bce(
            output["state_logit"],
            labels,
            mask,
            args.target_outcome,
            args.success_loss_mode,
        )
        action_loss = mil_episode_bce(
            output["action_logit"],
            labels,
            mask,
            args.target_outcome,
            args.success_loss_mode,
        )
    elif args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor"):
        if stage == "state":
            state_loss = mil_episode_bce(
                output["state_logit"],
                labels,
                mask,
                args.target_outcome,
                args.success_loss_mode,
            )
            action_loss = zero
        elif stage == "residual":
            state_loss = zero
            action_loss = mil_episode_bce(
                output["action_logit"],
                labels,
                mask,
                args.target_outcome,
                args.success_loss_mode,
            )
        else:
            raise ValueError(f"unknown two-stage training stage: {stage}")
    else:
        raise ValueError(f"unknown objective: {args.objective}")

    residual_l1 = masked_mean(output["action_delta"].abs(), mask)

    permutation = torch.randperm(len(mask), device=mask.device)
    shuffled_mask = mask & mask[permutation]
    shuffled_delta = model.action_delta(
        output["context"].detach(),
        batch["actions"][permutation],
    )
    shuffled_residual = masked_mean(shuffled_delta.square(), shuffled_mask)

    probability = torch.sigmoid(output["action_logit"])
    adjacent_mask = mask[:, 1:] & mask[:, :-1]
    smoothness = masked_mean(
        (probability[:, 1:] - probability[:, :-1]).abs(),
        adjacent_mask,
    )
    success_mask = mask & true_success_rows(labels, args.target_outcome)[:, None]
    if args.target_outcome == "failure":
        success_residual_values = output["action_delta"].square()
    elif args.target_outcome == "success":
        # In success mode, positive delta is desirable success advantage.
        # Only penalize logged successful chunks that reduce success evidence.
        success_residual_values = F.relu(-output["action_delta"]).square()
    else:
        raise ValueError(f"unknown target_outcome={args.target_outcome}")
    success_residual = masked_mean(success_residual_values, success_mask)
    residual_decorrelation = masked_corr_square(
        output["action_delta"],
        output["state_logit"].detach(),
        mask,
    )
    contrast_loss = zero
    if (
        args.objective in ("two_stage_action_contrast", "two_stage_matched_action_contrast")
        and stage == "residual"
        and "reference_actions" in batch
    ):
        reference_actions = batch["reference_actions"]
        num_negatives = int(reference_actions.shape[2])
        batch_size, horizon, hidden_dim = output["context"].shape
        reference_context = (
            output["context"]
            .detach()
            .unsqueeze(2)
            .expand(batch_size, horizon, num_negatives, hidden_dim)
            .reshape(batch_size, horizon * num_negatives, hidden_dim)
        )
        flat_reference_actions = reference_actions.reshape(
            batch_size,
            horizon * num_negatives,
            reference_actions.shape[-2],
            reference_actions.shape[-1],
        )
        reference_delta = model.action_delta(
            reference_context,
            flat_reference_actions,
        ).reshape(batch_size, horizon, num_negatives)
        if args.target_outcome == "failure":
            contrast_margin = output["action_delta"].unsqueeze(-1) - reference_delta
        elif args.target_outcome == "success":
            contrast_margin = reference_delta - output["action_delta"].unsqueeze(-1)
        else:
            raise ValueError(f"unknown target_outcome={args.target_outcome}")
        margin_violation = F.relu(args.contrast_margin - contrast_margin).mean(dim=-1)
        failure_rows = true_failure_rows(labels, args.target_outcome)
        failure_mask = mask & failure_rows[:, None]
        if torch.any(failure_rows):
            if args.contrast_weighting == "uniform":
                weights = failure_mask.float() / failure_mask.float().sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
            elif args.contrast_weighting == "delta_softmax":
                logits_for_weights = output["action_delta"].detach() / args.contrast_temperature
                logits_for_weights = logits_for_weights.masked_fill(~failure_mask, -1e9)
                weights = torch.softmax(logits_for_weights, dim=1) * failure_rows[:, None].float()
            else:
                logits_for_weights = output["action_logit"].detach() / args.contrast_temperature
                logits_for_weights = logits_for_weights.masked_fill(~failure_mask, -1e9)
                weights = torch.softmax(logits_for_weights, dim=1) * failure_rows[:, None].float()
            contrast_loss = (margin_violation * weights).sum() / failure_rows.float().sum().clamp_min(1.0)

    pairwise_rank_loss = zero
    pairwise_rank_accuracy = zero
    pairwise_rank_active_fraction = zero
    if (
        args.pairwise_rank_weight > 0.0
        and stage != "state"
        and "pairwise_positive_actions" in batch
        and "pairwise_negative_actions" in batch
    ):
        positive_actions = batch["pairwise_positive_actions"]
        negative_actions = batch["pairwise_negative_actions"]
        pairwise_mask = batch["pairwise_mask"]
        num_pairs = int(positive_actions.shape[2])
        batch_size, horizon, hidden_dim = output["context"].shape
        pairwise_context = (
            output["context"]
            .detach()
            .unsqueeze(2)
            .expand(batch_size, horizon, num_pairs, hidden_dim)
            .reshape(batch_size, horizon * num_pairs, hidden_dim)
        )
        flat_positive_actions = positive_actions.reshape(
            batch_size,
            horizon * num_pairs,
            positive_actions.shape[-2],
            positive_actions.shape[-1],
        )
        flat_negative_actions = negative_actions.reshape(
            batch_size,
            horizon * num_pairs,
            negative_actions.shape[-2],
            negative_actions.shape[-1],
        )
        positive_delta = model.action_delta(
            pairwise_context,
            flat_positive_actions,
        ).reshape(batch_size, horizon, num_pairs)
        negative_delta = model.action_delta(
            pairwise_context,
            flat_negative_actions,
        ).reshape(batch_size, horizon, num_pairs)
        pairwise_margin = positive_delta - negative_delta
        pairwise_losses = F.softplus(args.pairwise_rank_margin - pairwise_margin)
        pairwise_rank_loss = (
            pairwise_losses * pairwise_mask.float()
        ).sum() / pairwise_mask.float().sum().clamp_min(1.0)
        pairwise_rank_accuracy = (
            ((pairwise_margin > 0.0) & pairwise_mask).float().sum()
            / pairwise_mask.float().sum().clamp_min(1.0)
        )
        pairwise_rank_active_fraction = (
            pairwise_mask.float().sum()
            / (batch["mask"].float().sum() * max(num_pairs, 1)).clamp_min(1.0)
        )

    progress_consistency_loss = zero
    progress_consistency_active_fraction = zero
    if args.progress_consistency_weight > 0.0 and stage != "state":
        adjacent_mask = mask[:, 1:] & mask[:, :-1]
        if args.progress_consistency_space == "prob":
            state_value = torch.sigmoid(output["state_logit"].detach())
            action_gain = torch.sigmoid(output["action_logit"]) - torch.sigmoid(
                output["state_logit"].detach()
            )
        else:
            state_value = output["state_logit"].detach()
            action_gain = output["action_delta"]
        progress_target = state_value[:, 1:] - state_value[:, :-1]
        progress_error = action_gain[:, :-1] - progress_target
        progress_consistency_loss = masked_mean(
            progress_error.square(),
            adjacent_mask,
        )
        progress_consistency_active_fraction = adjacent_mask.float().sum() / mask.float().sum().clamp_min(1.0)

    temporal_safe_anchor = zero
    temporal_risk_loss = zero
    temporal_weight_mean = zero
    temporal_weight_sum = zero
    temporal_active_fraction = zero
    delta_progress_loss = zero
    delta_progress_active_fraction = zero
    delta_progress_target_abs_mean = zero
    if args.objective == "two_stage_temporal_safe_anchor" and stage == "residual":
        success_safe_mask = mask & true_success_rows(labels, args.target_outcome)[:, None]
        if args.target_outcome == "failure":
            safe_anchor_violation = F.relu(
                output["action_delta"] - args.safe_anchor_epsilon
            )
        elif args.target_outcome == "success":
            # Success-mode safe anchor: successful logged actions should not
            # decrease success evidence. This aligns with temporal progress
            # chunks that require positive success delta.
            safe_anchor_violation = F.relu(
                args.safe_anchor_epsilon - output["action_delta"]
            )
        else:
            raise ValueError(f"unknown target_outcome={args.target_outcome}")
        temporal_safe_anchor = masked_mean(
            safe_anchor_violation.square(),
            success_safe_mask,
        )
        state_probability = torch.sigmoid(output["state_logit"].detach())
        risk_weights = temporal_risk_weights(
            state_probability=state_probability,
            lengths=batch["lengths"],
            mask=mask,
            labels=labels,
            stride=args.temporal_stride,
            min_increase=args.temporal_min_increase,
            normalize=args.temporal_normalize_weights,
        )
        temporal_weight_sum = risk_weights.sum()
        temporal_weight_mean = risk_weights.sum() / mask.sum().clamp_min(1)
        temporal_active_fraction = (risk_weights > 0).float().sum() / mask.sum().clamp_min(1)
        temporal_risk_loss = (
            risk_weights
            * F.relu(args.temporal_risk_margin - output["action_delta"]).square()
        ).sum() / risk_weights.sum().clamp_min(1e-8)

    if args.delta_progress_weight > 0.0 and stage != "state":
        stride = args.delta_progress_stride
        if stride <= 0:
            stride = batch["actions"].shape[-2]
        if args.delta_progress_space == "prob":
            state_value = torch.sigmoid(output["state_logit"].detach())
        else:
            state_value = output["state_logit"].detach()
        delta_target, delta_target_mask = temporal_future_difference(
            state_value,
            batch["lengths"],
            mask,
            stride,
        )
        if args.delta_progress_clip > 0.0:
            delta_target = delta_target.clamp(
                -args.delta_progress_clip,
                args.delta_progress_clip,
            )
        delta_error = output["action_delta"] - delta_target
        if args.delta_progress_huber_beta > 0.0:
            delta_loss_values = F.smooth_l1_loss(
                output["action_delta"],
                delta_target,
                reduction="none",
                beta=args.delta_progress_huber_beta,
            )
        else:
            delta_loss_values = delta_error.square()
        if args.delta_progress_abs_weight > 0.0:
            delta_weights = 1.0 + args.delta_progress_abs_weight * delta_target.abs()
        else:
            delta_weights = None
        delta_progress_loss = masked_weighted_mean(
            delta_loss_values,
            delta_target_mask,
            delta_weights,
        )
        delta_progress_active_fraction = (
            delta_target_mask.float().sum() / mask.float().sum().clamp_min(1.0)
        )
        delta_progress_target_abs_mean = masked_mean(
            delta_target.abs(),
            delta_target_mask,
        )

    if args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor") and stage == "state":
        total = args.state_weight * state_loss
    else:
        total = (
            args.state_weight * state_loss
            + args.action_weight * action_loss
            + args.residual_l1_weight * residual_l1
            + args.shuffled_residual_weight * shuffled_residual
            + args.smoothness_weight * smoothness
        )
        if args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor"):
            total = (
                total
                + args.success_residual_weight * success_residual
                + args.decorrelation_weight * residual_decorrelation
                + args.contrast_weight * contrast_loss
            )
            total = (
                total
                + args.pairwise_rank_weight * pairwise_rank_loss
                + args.progress_consistency_weight * progress_consistency_loss
                + args.delta_progress_weight * delta_progress_loss
            )
            if args.objective == "two_stage_temporal_safe_anchor":
                total = (
                    total
                    + args.safe_anchor_weight * temporal_safe_anchor
                    + args.temporal_risk_weight * temporal_risk_loss
                )
    return total, {
        "stage": stage,
        "state_loss": float(state_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "residual_l1": float(residual_l1.detach()),
        "shuffled_residual": float(shuffled_residual.detach()),
        "smoothness": float(smoothness.detach()),
        "success_residual": float(success_residual.detach()),
        "residual_decorrelation": float(residual_decorrelation.detach()),
        "contrast_loss": float(contrast_loss.detach()),
        "pairwise_rank_loss": float(pairwise_rank_loss.detach()),
        "pairwise_rank_accuracy": float(pairwise_rank_accuracy.detach()),
        "pairwise_rank_active_fraction": float(pairwise_rank_active_fraction.detach()),
        "progress_consistency_loss": float(progress_consistency_loss.detach()),
        "progress_consistency_active_fraction": float(progress_consistency_active_fraction.detach()),
        "temporal_safe_anchor": float(temporal_safe_anchor.detach()),
        "temporal_risk_loss": float(temporal_risk_loss.detach()),
        "temporal_weight_mean": float(temporal_weight_mean.detach()),
        "temporal_weight_sum": float(temporal_weight_sum.detach()),
        "temporal_active_fraction": float(temporal_active_fraction.detach()),
        "delta_progress_loss": float(delta_progress_loss.detach()),
        "delta_progress_active_fraction": float(delta_progress_active_fraction.detach()),
        "delta_progress_target_abs_mean": float(delta_progress_target_abs_mean.detach()),
    }


@torch.no_grad()
def predict(
    model: CausalPrefixRisk,
    features: np.ndarray,
    actions: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_scores = np.empty(len(features), dtype=np.float32)
    action_scores = np.empty(len(features), dtype=np.float32)
    action_deltas = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(episodes), batch_size):
        selected = episodes[start : start + batch_size]
        dummy_labels = np.zeros(len(offsets) - 1, dtype=np.float32)
        batch = padded_episode_batch(
            features=features,
            actions=actions,
            labels=dummy_labels,
            offsets=offsets,
            episodes=selected,
            device=device,
        )
        output = model(batch["features"], batch["actions"])
        for row, episode in enumerate(selected):
            sl = slice(offsets[episode], offsets[episode + 1])
            length = sl.stop - sl.start
            state_scores[sl] = torch.sigmoid(
                output["state_logit"][row, :length]
            ).cpu().numpy()
            action_scores[sl] = torch.sigmoid(
                output["action_logit"][row, :length]
            ).cpu().numpy()
            action_deltas[sl] = output["action_delta"][
                row, :length
            ].cpu().numpy()
    return state_scores, action_scores, action_deltas


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return {
        "roc_auc": rank_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "binary_cross_entropy": float(
            -np.mean(
                labels * np.log(clipped)
                + (1.0 - labels) * np.log(1.0 - clipped)
            )
        ),
        "accuracy_at_0.5": float(
            np.mean((scores >= 0.5) == (labels > 0.5))
        ),
        "positive_score_mean": float(np.mean(scores[labels > 0.5])),
        "negative_score_mean": float(np.mean(scores[labels < 0.5])),
    }


def logit_np(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def pearson_np(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return float("nan")
    x = first.astype(np.float64) - float(np.mean(first))
    y = second.astype(np.float64) - float(np.mean(second))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def delta_progress_metrics(
    state_scores: np.ndarray,
    action_deltas: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    stride: int,
    space: str,
    clip: float,
) -> dict:
    stride = max(int(stride), 1)
    current_indices = []
    future_indices = []
    for episode in episodes:
        start = int(offsets[episode])
        end = int(offsets[episode + 1])
        if end - start <= stride:
            continue
        current_indices.append(np.arange(start, end - stride, dtype=np.int64))
        future_indices.append(np.arange(start + stride, end, dtype=np.int64))
    if not current_indices:
        return {
            "num_pairs": 0,
            "pearson": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "sign_accuracy": float("nan"),
            "positive_progress_auc": float("nan"),
            "target_abs_mean": float("nan"),
            "delta_abs_mean": float("nan"),
        }
    current = np.concatenate(current_indices)
    future = np.concatenate(future_indices)
    if space == "prob":
        value = state_scores
    else:
        value = logit_np(state_scores)
    target = value[future] - value[current]
    if clip > 0.0:
        target = np.clip(target, -clip, clip)
    pred = action_deltas[current]
    error = pred - target
    sign_labels = target > 0.0
    if np.any(sign_labels) and np.any(~sign_labels):
        positive_progress_auc = rank_auc(sign_labels.astype(np.float32), pred)
    else:
        positive_progress_auc = float("nan")
    return {
        "num_pairs": int(len(current)),
        "pearson": pearson_np(pred, target),
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "sign_accuracy": float(np.mean((pred > 0.0) == sign_labels)),
        "positive_progress_auc": positive_progress_auc,
        "target_abs_mean": float(np.mean(np.abs(target))),
        "delta_abs_mean": float(np.mean(np.abs(pred))),
    }


def outcome_metrics(
    state_scores: np.ndarray,
    action_scores: np.ndarray,
    action_deltas: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    target_outcome: str,
) -> dict:
    indices = np.concatenate(
        [episode_indices(offsets, int(episode)) for episode in episodes]
    )
    chunk_labels = np.concatenate(
        [
            np.full(
                offsets[episode + 1] - offsets[episode],
                labels[episode],
                dtype=np.float32,
            )
            for episode in episodes
        ]
    )
    final_indices = np.asarray(
        [offsets[episode + 1] - 1 for episode in episodes],
        dtype=np.int64,
    )
    bag_state_scores = mil_episode_probability(
        state_scores,
        offsets,
        episodes,
        target_outcome,
    )
    bag_action_scores = mil_episode_probability(
        action_scores,
        offsets,
        episodes,
        target_outcome,
    )
    bag_state_metrics = binary_metrics(labels[episodes], bag_state_scores)
    bag_action_metrics = binary_metrics(labels[episodes], bag_action_scores)
    positive_count = int(np.sum(labels[episodes] > 0.5))
    return {
        "num_episodes": int(len(episodes)),
        "num_positive_episodes": positive_count,
        "num_negative_episodes": int(len(episodes) - positive_count),
        "bag_aggregation": "noisy_or" if target_outcome == "failure" else "noisy_and",
        "bag_state_mil": bag_state_metrics,
        "bag_action_mil": bag_action_metrics,
        # Backward-compatible aliases used by older checkpoint-selection code.
        # For target_outcome=success these are noisy-AND values despite the old key name.
        "bag_state_noisy_or": bag_state_metrics,
        "bag_action_noisy_or": bag_action_metrics,
        "prefix_state": binary_metrics(chunk_labels, state_scores[indices]),
        "prefix_action_conditioned": binary_metrics(
            chunk_labels,
            action_scores[indices],
        ),
        "action_delta_conditioned": binary_metrics(
            chunk_labels,
            1.0 / (1.0 + np.exp(-np.clip(action_deltas[indices], -60.0, 60.0))),
        ),
        "final_state": binary_metrics(
            labels[episodes],
            state_scores[final_indices],
        ),
        "final_action_conditioned": binary_metrics(
            labels[episodes],
            action_scores[final_indices],
        ),
    }


def checkpoint_quality(metric_name: str, validation: dict) -> tuple[float, float]:
    """Return a lexicographic quality tuple for checkpoint selection.

    Bag-level metrics are useful for verifying the trajectory classifier, but
    closed-loop candidate extraction depends on local action ranking. This helper
    makes that distinction explicit and lets us save several best checkpoints
    from the same training run.
    """

    if metric_name == "bag_action_mil":
        selected = validation["bag_action_mil"]
        return (
            selected["roc_auc"],
            -selected["binary_cross_entropy"],
        )
    if metric_name == "prefix_action_conditioned":
        selected = validation["prefix_action_conditioned"]
        return (
            selected["roc_auc"],
            -selected["binary_cross_entropy"],
        )
    if metric_name == "final_action_conditioned":
        selected = validation["final_action_conditioned"]
        return (
            selected["roc_auc"],
            -selected["binary_cross_entropy"],
        )
    if metric_name == "action_delta_auc":
        selected = validation["action_delta_conditioned"]
        return (
            selected["roc_auc"],
            -selected["binary_cross_entropy"],
        )
    if metric_name == "delta_progress_pearson":
        selected = validation["delta_progress"]
        return (
            selected["pearson"],
            -selected["mse"],
        )
    if metric_name == "mixed_action":
        prefix_validation = validation["prefix_action_conditioned"]
        delta_validation = validation["action_delta_conditioned"]
        progress_validation = validation["delta_progress"]
        progress_score = progress_validation["pearson"]
        if not math.isfinite(progress_score):
            progress_score = -1.0
        return (
            prefix_validation["roc_auc"]
            + 0.25 * delta_validation["roc_auc"]
            + 0.25 * progress_score,
            -prefix_validation["binary_cross_entropy"],
        )
    raise ValueError(f"unknown checkpoint metric {metric_name}")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    stats: dict,
    splits: dict,
    args: argparse.Namespace,
    features: np.ndarray,
    actions: np.ndarray,
    step: int,
    quality: tuple[float, float],
    checkpoint_metric: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "stats": stats,
            "splits": splits,
            "args": vars(args),
            "feature_dim": int(features.shape[-1]),
            "prediction_horizon": int(actions.shape[1]),
            "action_dim": int(actions.shape[2]),
            "best_step": int(step),
            "best_quality": tuple(float(x) for x in quality),
            "checkpoint_metric": checkpoint_metric,
        },
        path,
    )


def calibrated_thresholds(
    action_scores: np.ndarray,
    action_deltas: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    validation_episodes: np.ndarray,
    quantile: float,
    target_outcome: str,
) -> dict:
    success_episodes = np.intersect1d(
        validation_episodes,
        true_success_episode_indices(labels, target_outcome),
        assume_unique=False,
    )
    indices = np.concatenate(
        [episode_indices(offsets, int(episode)) for episode in success_episodes]
    )
    positive_target_delta = np.maximum(action_deltas[indices], 0.0)
    return {
        "quantile": quantile,
        "action_probability": float(np.quantile(action_scores[indices], quantile)),
        "positive_action_logodds": float(np.quantile(positive_target_delta, quantile)),
        "num_validation_success_chunks": int(len(indices)),
    }


def persistent_onset(signal: np.ndarray, persistence: int) -> int | None:
    if persistence <= 1:
        locations = np.flatnonzero(signal)
    elif len(signal) < persistence:
        return None
    else:
        locations = np.flatnonzero(
            np.convolve(signal.astype(np.int32), np.ones(persistence, dtype=np.int32), mode="valid")
            == persistence
        )
    return int(locations[0]) if len(locations) else None


def set_state_modules_trainable(model: CausalPrefixRisk, trainable: bool) -> None:
    module_names = (
        "obs_projection",
        "prefix_temporal_encoder",
        "prefix_encoder",
        "context_norm",
        "state_head",
    )
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def set_action_modules_trainable(model: CausalPrefixRisk, trainable: bool) -> None:
    for module in (model.action_encoder, model.action_residual):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def make_optimizer_and_scheduler(
    model: CausalPrefixRisk,
    args,
    stage_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters for current stage")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(stage_steps), 1),
    )
    return optimizer, scheduler


def score_separation(
    scores: np.ndarray,
    critical_mask: np.ndarray,
    safe_mask: np.ndarray,
) -> dict:
    critical_scores = scores[critical_mask]
    safe_scores = scores[safe_mask]
    labels = np.concatenate(
        [np.ones(len(critical_scores)), np.zeros(len(safe_scores))]
    )
    selected_scores = np.concatenate([critical_scores, safe_scores])
    return {
        "num_critical": int(len(critical_scores)),
        "num_safe": int(len(safe_scores)),
        "critical_mean": (
            float(np.mean(critical_scores)) if len(critical_scores) else None
        ),
        "safe_mean": float(np.mean(safe_scores)) if len(safe_scores) else None,
        "critical_vs_safe_roc_auc": (
            rank_auc(labels, selected_scores)
            if len(critical_scores) and len(safe_scores)
            else None
        ),
    }


def onset_metrics(
    *,
    action_scores: np.ndarray,
    action_deltas: np.ndarray,
    steps: np.ndarray,
    critical: np.ndarray,
    safe: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
    probability_threshold: float | None,
    action_risk_threshold: float,
    action_horizon: int,
    persistence: int,
) -> dict:
    selected = np.zeros(len(action_scores), dtype=bool)
    success_selected = np.zeros(len(action_scores), dtype=bool)
    for episode in episodes:
        sl = slice(offsets[episode], offsets[episode + 1])
        selected[sl] = True
        if labels[episode] < 0.5:
            success_selected[sl] = True

    action_risk = np.maximum(action_deltas, 0.0)
    signal = action_risk > action_risk_threshold
    if probability_threshold is not None:
        signal &= action_scores >= probability_threshold
    localized = 0
    detected = 0
    exact = 0
    near = 0
    before = 0
    after = 0
    signed_lags = []
    success_episode_false_positives = []
    for episode in episodes:
        sl = slice(offsets[episode], offsets[episode + 1])
        local_signal = signal[sl]
        onset = persistent_onset(local_signal, persistence)
        if labels[episode] < 0.5:
            success_episode_false_positives.append(onset is not None)
            continue
        local_critical = np.flatnonzero(critical[sl])
        if not len(local_critical):
            continue
        localized += 1
        if onset is None:
            continue
        detected += 1
        local_steps = steps[sl]
        onset_step = int(local_steps[onset])
        critical_steps = local_steps[local_critical]
        nearest = int(
            critical_steps[np.argmin(np.abs(critical_steps - onset_step))]
        )
        lag = onset_step - nearest
        signed_lags.append(lag)
        exact += int(onset in set(local_critical.tolist()))
        near += int(abs(lag) <= action_horizon)
        before += int(onset_step < int(critical_steps.min()))
        after += int(onset_step > int(critical_steps.max()))

    critical_selected = critical & selected
    safe_selected = safe & selected
    return {
        "persistence_boundaries": persistence,
        "probability_threshold": probability_threshold,
        "positive_action_logodds_threshold": action_risk_threshold,
        "episodes_with_privileged_critical_label": localized,
        "episodes_with_detected_onset": detected,
        "onset_detection_rate": detected / localized if localized else None,
        "exact_onset_hit_rate": exact / localized if localized else None,
        "within_one_boundary_onset_hit_rate": (
            near / localized if localized else None
        ),
        "onset_before_critical_rate": before / localized if localized else None,
        "onset_after_critical_rate": after / localized if localized else None,
        "detected_onset_signed_lag_steps": {
            "median": float(np.median(signed_lags)) if signed_lags else None,
            "q25": float(np.quantile(signed_lags, 0.25)) if signed_lags else None,
            "q75": float(np.quantile(signed_lags, 0.75)) if signed_lags else None,
        },
        "critical_chunk_signal_recall": (
            float(np.mean(signal[critical_selected]))
            if np.any(critical_selected)
            else None
        ),
        "safe_chunk_signal_false_positive_rate": (
            float(np.mean(signal[safe_selected]))
            if np.any(safe_selected)
            else None
        ),
        "success_chunk_signal_false_positive_rate": (
            float(np.mean(signal[success_selected]))
            if np.any(success_selected)
            else None
        ),
        "success_episode_onset_false_positive_rate": (
            float(np.mean(success_episode_false_positives))
            if success_episode_false_positives
            else None
        ),
        "action_probability_separation": score_separation(
            action_scores,
            critical_selected,
            safe_selected,
        ),
        "positive_action_risk_separation": score_separation(
            action_risk,
            critical_selected,
            safe_selected,
        ),
    }


def train(args) -> dict:
    raw = np.load(args.features)
    features_raw = raw["features"].astype(np.float32)
    actions_raw = raw["actions"].astype(np.float32)
    raw_failure_labels = raw["episode_labels"].astype(np.float32)
    if args.target_outcome == "failure":
        labels = raw_failure_labels
    elif args.target_outcome == "success":
        labels = 1.0 - raw_failure_labels
    else:
        raise ValueError(f"unknown target_outcome={args.target_outcome}")
    offsets = raw["episode_offsets"].astype(np.int64)
    steps = raw["steps"].astype(np.int64)
    critical = raw["critical_labels"].astype(bool)
    safe = raw["safe_labels"].astype(bool)
    splits = stratified_split(labels, args.seed)
    stats = normalizers(
        features_raw,
        actions_raw,
        offsets,
        splits["train"],
    )
    features, actions = normalize(features_raw, actions_raw, stats)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_causal_prefix_model(
        model_arch=args.model_arch,
        feature_dim=features.shape[-1],
        prediction_horizon=actions.shape[1],
        action_dim=actions.shape[2],
        hidden_dim=args.hidden_dim,
        action_hidden_dim=args.action_hidden_dim,
        dropout=args.dropout,
        action_num_heads=args.action_num_heads,
        action_conv_layers=args.action_conv_layers,
        prefix_conv_layers=args.prefix_conv_layers,
    ).to(device)
    if args.objective in ("noisy_or_mil", "two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor"):
        with torch.no_grad():
            model.state_head[-1].bias.fill_(args.initial_state_logit_bias)
            model.action_residual[-1].bias.fill_(
                args.initial_action_delta_bias
            )

    if args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor"):
        args.stage1_steps = min(max(int(args.stage1_steps), 1), args.total_steps - 1)
        current_stage = "state"
        args.training_stage = current_stage
        set_state_modules_trainable(model, True)
        set_action_modules_trainable(model, False)
        optimizer, scheduler = make_optimizer_and_scheduler(
            model,
            args,
            args.stage1_steps,
        )
    else:
        current_stage = "joint"
        args.training_stage = current_stage
        set_state_modules_trainable(model, True)
        set_action_modules_trainable(model, True)
        optimizer, scheduler = make_optimizer_and_scheduler(
            model,
            args,
            args.total_steps,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    checkpoint_metric_paths = {
        "bag_action_mil": args.output_dir / "best_bag.pt",
        "prefix_action_conditioned": args.output_dir / "best_prefix_action_auc.pt",
        "action_delta_auc": args.output_dir / "best_action_delta_auc.pt",
        "mixed_action": args.output_dir / "best_mixed_action_score.pt",
    }
    checkpoint_metric_qualities = {
        name: (-float("inf"), -float("inf"))
        for name in checkpoint_metric_paths
    }
    checkpoint_metric_steps = {
        name: None
        for name in checkpoint_metric_paths
    }
    best_quality = (-float("inf"), -float("inf"))
    history = []
    all_episodes = np.arange(len(labels), dtype=np.int64)
    success_by_step, all_success_action_indices = success_action_indices_by_step(
        steps,
        labels,
        offsets,
        args.target_outcome,
    )
    positive_by_step, all_positive_action_indices = action_indices_by_outcome_and_step(
        steps,
        labels,
        offsets,
        positive=True,
    )
    negative_by_step, all_negative_action_indices = action_indices_by_outcome_and_step(
        steps,
        labels,
        offsets,
        positive=False,
    )
    matched_success_indices = None
    if args.objective == "two_stage_matched_action_contrast":
        print(
            json.dumps(
                {
                    "event": "build_matched_success_indices",
                    "top_k": args.match_topk,
                    "step_window": args.match_step_window,
                },
                indent=2,
            ),
            flush=True,
        )
        matched_success_indices = build_matched_success_indices(
            features=features,
            steps=steps,
            labels=labels,
            offsets=offsets,
            target_outcome=args.target_outcome,
            top_k=args.match_topk,
            step_window=args.match_step_window,
            device=device,
        )
    for step in range(1, args.total_steps + 1):
        if (
            args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor")
            and current_stage == "state"
            and step > args.stage1_steps
        ):
            current_stage = "residual"
            args.training_stage = current_stage
            set_state_modules_trainable(model, False)
            set_action_modules_trainable(model, True)
            optimizer, scheduler = make_optimizer_and_scheduler(
                model,
                args,
                args.total_steps - args.stage1_steps,
            )
            print(
                json.dumps(
                    {
                        "event": "switch_to_residual_stage",
                        "step": step,
                        "frozen_state_modules": True,
                    },
                    indent=2,
                ),
                flush=True,
            )
        args.training_stage = current_stage
        model.train()
        episodes = balanced_episode_batch(
            labels,
            splits["train"],
            args.episode_batch_size,
            rng,
        )
        batch = padded_episode_batch(
            features=features,
            actions=actions,
            labels=labels,
            offsets=offsets,
            episodes=episodes,
            device=device,
        )
        if current_stage == "residual" and args.objective in (
            "two_stage_action_contrast",
            "two_stage_matched_action_contrast",
        ):
            if args.objective == "two_stage_matched_action_contrast":
                batch["reference_actions"] = sample_matched_reference_actions(
                    actions=actions,
                    offsets=offsets,
                    episodes=episodes,
                    lengths=batch["lengths"],
                    matched_indices=matched_success_indices,
                    num_negatives=args.contrast_negatives,
                    rng=rng,
                    device=device,
                )
            else:
                batch["reference_actions"] = sample_success_reference_actions(
                    actions=actions,
                    steps=steps,
                    offsets=offsets,
                    episodes=episodes,
                    lengths=batch["lengths"],
                    by_step=success_by_step,
                    all_success_indices=all_success_action_indices,
                    num_negatives=args.contrast_negatives,
                    rng=rng,
                    device=device,
                )
        if current_stage != "state" and args.pairwise_rank_weight > 0.0:
            batch.update(
                sample_outcome_pairwise_actions(
                    actions=actions,
                    steps=steps,
                    labels=labels,
                    offsets=offsets,
                    episodes=episodes,
                    lengths=batch["lengths"],
                    positive_by_step=positive_by_step,
                    negative_by_step=negative_by_step,
                    all_positive_indices=all_positive_action_indices,
                    all_negative_indices=all_negative_action_indices,
                    num_negatives=args.pairwise_rank_negatives,
                    rng=rng,
                    device=device,
                )
            )
        optimizer.zero_grad(set_to_none=True)
        loss, components = compute_loss(model, batch, args)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.total_steps:
            model.eval()
            state_scores, action_scores, action_deltas = predict(
                model,
                features,
                actions,
                offsets,
                all_episodes,
                device,
                args.eval_batch_size,
            )
            validation = outcome_metrics(
                state_scores,
                action_scores,
                action_deltas,
                labels,
                offsets,
                splits["val"],
                args.target_outcome,
            )
            delta_progress_stride = args.delta_progress_stride
            if delta_progress_stride <= 0:
                delta_progress_stride = actions.shape[1]
            validation["delta_progress"] = delta_progress_metrics(
                state_scores,
                action_deltas,
                offsets,
                splits["val"],
                delta_progress_stride,
                args.delta_progress_space,
                args.delta_progress_clip,
            )
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "training_stage": current_stage,
                **components,
                "validation": validation,
            }
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)
            quality = checkpoint_quality(args.checkpoint_metric, validation)
            allow_checkpoint = not (
                args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor")
                and current_stage != "residual"
            )
            if allow_checkpoint:
                for metric_name, path in checkpoint_metric_paths.items():
                    metric_quality = checkpoint_quality(metric_name, validation)
                    if (
                        math.isfinite(metric_quality[0])
                        and metric_quality > checkpoint_metric_qualities[metric_name]
                    ):
                        checkpoint_metric_qualities[metric_name] = metric_quality
                        checkpoint_metric_steps[metric_name] = int(step)
                        save_checkpoint(
                            path,
                            model=model,
                            stats=stats,
                            splits=splits,
                            args=args,
                            features=features,
                            actions=actions,
                            step=step,
                            quality=metric_quality,
                            checkpoint_metric=metric_name,
                        )
                if math.isfinite(quality[0]) and quality > best_quality:
                    best_quality = quality
                    save_checkpoint(
                        best_path,
                        model=model,
                        stats=stats,
                        splits=splits,
                        args=args,
                        features=features,
                        actions=actions,
                        step=step,
                        quality=quality,
                        checkpoint_metric=args.checkpoint_metric,
                    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    state_scores, action_scores, action_deltas = predict(
        model,
        features,
        actions,
        offsets,
        all_episodes,
        device,
        args.eval_batch_size,
    )
    thresholds = calibrated_thresholds(
        action_scores,
        action_deltas,
        labels,
        offsets,
        splits["val"],
        args.success_quantile,
        args.target_outcome,
    )
    metrics = {
        split: outcome_metrics(
            state_scores,
            action_scores,
            action_deltas,
            labels,
            offsets,
            episodes,
            args.target_outcome,
        )
        for split, episodes in splits.items()
    }
    delta_progress_stride = args.delta_progress_stride
    if delta_progress_stride <= 0:
        delta_progress_stride = actions.shape[1]
    for split, episodes in splits.items():
        metrics[split]["delta_progress"] = delta_progress_metrics(
            state_scores,
            action_deltas,
            offsets,
            episodes,
            delta_progress_stride,
            args.delta_progress_space,
            args.delta_progress_clip,
        )
    if args.target_outcome == "failure":
        for persistence in (1, 2):
            metrics[
                f"privileged_action_risk_onset_test_persistence_{persistence}"
            ] = onset_metrics(
                action_scores=action_scores,
                action_deltas=action_deltas,
                steps=steps,
                critical=critical,
                safe=safe,
                labels=labels,
                offsets=offsets,
                episodes=splits["test"],
                probability_threshold=None,
                action_risk_threshold=thresholds["positive_action_logodds"],
                action_horizon=args.action_horizon,
                persistence=persistence,
            )
            metrics[
                f"privileged_action_risk_onset_all_persistence_{persistence}"
            ] = onset_metrics(
                action_scores=action_scores,
                action_deltas=action_deltas,
                steps=steps,
                critical=critical,
                safe=safe,
                labels=labels,
                offsets=offsets,
                episodes=all_episodes,
                probability_threshold=None,
                action_risk_threshold=thresholds["positive_action_logodds"],
                action_horizon=args.action_horizon,
                persistence=persistence,
            )
            metrics[f"privileged_joint_onset_test_persistence_{persistence}"] = (
                onset_metrics(
                    action_scores=action_scores,
                    action_deltas=action_deltas,
                    steps=steps,
                    critical=critical,
                    safe=safe,
                    labels=labels,
                    offsets=offsets,
                    episodes=splits["test"],
                    probability_threshold=thresholds["action_probability"],
                    action_risk_threshold=thresholds["positive_action_logodds"],
                    action_horizon=args.action_horizon,
                    persistence=persistence,
                )
            )
    else:
        metrics["privileged_onset_note"] = (
            "Skipped privileged failure-onset localization because target_outcome="
            f"{args.target_outcome}; critical/safe labels are failure-risk diagnostics."
        )

    split_labels = np.full(len(labels), "unused", dtype="<U5")
    for split, episodes in splits.items():
        split_labels[episodes] = split
    predictions_path = args.output_dir / "prefix_predictions.npz"
    positive_target_delta = np.maximum(action_deltas, 0.0)
    negative_target_delta = np.maximum(-action_deltas, 0.0)
    if args.target_outcome == "failure":
        positive_action_risk = positive_target_delta
        positive_action_success_advantage = negative_target_delta
    elif args.target_outcome == "success":
        positive_action_risk = negative_target_delta
        positive_action_success_advantage = positive_target_delta
    else:
        raise ValueError(f"unknown target_outcome={args.target_outcome}")
    np.savez_compressed(
        predictions_path,
        state_scores=state_scores,
        action_scores=action_scores,
        action_deltas=action_deltas,
        positive_action_risk=positive_action_risk,
        positive_action_success_advantage=positive_action_success_advantage,
        positive_target_action_delta=positive_target_delta,
        negative_target_action_delta=negative_target_delta,
        steps=steps,
        episode_indices=raw["episode_indices"],
        episode_keys=raw["episode_keys"],
        episode_labels=labels,
        raw_failure_episode_labels=raw_failure_labels,
        target_outcome=np.asarray(args.target_outcome.encode("utf-8")),
        episode_offsets=offsets,
        episode_splits=split_labels,
        critical_labels=critical,
        safe_labels=safe,
        action_probability_threshold=np.asarray(
            thresholds["action_probability"]
        ),
        positive_action_logodds_threshold=np.asarray(
            thresholds["positive_action_logodds"]
        ),
        # Backward-compatible alias: positive delta for the configured target
        # outcome. For target_outcome=success this is success advantage; for
        # target_outcome=failure this is risk increase.
        positive_action_delta=positive_target_delta,
    )
    summary = {
        "features": str(args.features),
        "checkpoint": str(best_path),
        "predictions": str(predictions_path),
        "target_outcome": args.target_outcome,
        "num_episodes": int(len(labels)),
        "num_positive_episodes": int(np.sum(labels)),
        "num_failure_episodes": int(np.sum(raw_failure_labels)),
        "num_success_episodes": int(len(raw_failure_labels) - np.sum(raw_failure_labels)),
        "num_chunks": int(len(features)),
        "objective": args.objective,
        "success_loss_mode": args.success_loss_mode,
        "delta_progress_weight": args.delta_progress_weight,
        "delta_progress_stride": delta_progress_stride,
        "delta_progress_space": args.delta_progress_space,
        "delta_progress_clip": args.delta_progress_clip,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_metric_paths": {
            name: str(path)
            for name, path in checkpoint_metric_paths.items()
            if path.exists()
        },
        "checkpoint_metric_best_steps": {
            name: step
            for name, step in checkpoint_metric_steps.items()
            if step is not None
        },
        "checkpoint_metric_best_qualities": {
            name: [float(x) for x in quality]
            for name, quality in checkpoint_metric_qualities.items()
            if math.isfinite(quality[0])
        },
        "stage1_steps": int(args.stage1_steps) if args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor") else None,
        "best_step": int(checkpoint["best_step"]),
        "thresholds": thresholds,
        "metrics": metrics,
        "history": history,
        "privileged_labels_used_for_training": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "history"},
            indent=2,
        ),
        flush=True,
    )
    print(f"Wrote {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model-arch",
        choices=("v1", "v2", "flat", "cross_attn_v2"),
        default="v1",
        help="Prefix outcome model architecture. v1 is the legacy flat-action model; v2 uses sequential action conv + cross attention.",
    )
    parser.add_argument("--action-num-heads", type=int, default=4)
    parser.add_argument("--action-conv-layers", type=int, default=2)
    parser.add_argument("--prefix-conv-layers", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-outcome",
        choices=("failure", "success"),
        default="failure",
        help=(
            "Outcome treated as the positive class. failure reproduces the "
            "original risk model; success trains a turned-over success "
            "advantage model with the same architecture and objective."
        ),
    )
    parser.add_argument(
        "--success-loss-mode",
        choices=("noisy_and", "chunk_bce_failure_mil"),
        default="chunk_bce_failure_mil",
        help=(
            "MIL aggregation used when --target-outcome success. noisy_and "
            "uses prod_t P(good_t) for both success and failure episodes. "
            "chunk_bce_failure_mil uses length-normalized positive BCE on "
            "successful episodes and a not-all-good MIL term on failed episodes."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=(
            "per_chunk_bce",
            "noisy_or_mil",
            "two_stage_residual",
            "two_stage_action_contrast",
            "two_stage_matched_action_contrast",
            "two_stage_temporal_safe_anchor",
        ),
        default="per_chunk_bce",
    )
    parser.add_argument("--stage1-steps", type=int, default=500)
    parser.add_argument("--initial-state-logit-bias", type=float, default=-5.0)
    parser.add_argument("--initial-action-delta-bias", type=float, default=0.0)
    parser.add_argument("--residual-l1-weight", type=float, default=0.01)
    parser.add_argument("--shuffled-residual-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--success-residual-weight", type=float, default=0.1)
    parser.add_argument("--decorrelation-weight", type=float, default=0.1)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument("--contrast-margin", type=float, default=0.05)
    parser.add_argument("--contrast-negatives", type=int, default=4)
    parser.add_argument("--contrast-temperature", type=float, default=0.5)
    parser.add_argument("--pairwise-rank-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-rank-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-rank-negatives", type=int, default=4)
    parser.add_argument("--progress-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--progress-consistency-space",
        choices=("logit", "prob"),
        default="logit",
        help="Space used for matching action gain to next-prefix progress.",
    )
    parser.add_argument(
        "--delta-progress-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for direct signed progress regression: action_delta should "
            "match V(prefix_{t+H}) - V(prefix_t). This is the main Q-V "
            "sensitivity loss for action scoring."
        ),
    )
    parser.add_argument(
        "--delta-progress-stride",
        type=int,
        default=-1,
        help="Future prefix stride for delta progress target. <=0 uses the action chunk horizon.",
    )
    parser.add_argument(
        "--delta-progress-space",
        choices=("logit", "prob"),
        default="logit",
        help="Use state logits or probabilities for V(prefix_{t+H}) - V(prefix_t).",
    )
    parser.add_argument("--delta-progress-clip", type=float, default=0.5)
    parser.add_argument("--delta-progress-huber-beta", type=float, default=0.05)
    parser.add_argument("--delta-progress-abs-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-metric",
        choices=(
            "bag_action_mil",
            "prefix_action_conditioned",
            "final_action_conditioned",
            "action_delta_auc",
            "delta_progress_pearson",
            "mixed_action",
        ),
        default="prefix_action_conditioned",
        help="Metric used to keep best.pt. Avoid bag_action_mil for action-scoring models because it saturates early.",
    )
    parser.add_argument(
        "--contrast-weighting",
        choices=("action_softmax", "delta_softmax", "uniform"),
        default="action_softmax",
    )
    parser.add_argument("--match-topk", type=int, default=16)
    parser.add_argument("--match-step-window", type=int, default=16)
    parser.add_argument("--safe-anchor-weight", type=float, default=0.1)
    parser.add_argument("--safe-anchor-epsilon", type=float, default=0.0)
    parser.add_argument("--temporal-risk-weight", type=float, default=0.5)
    parser.add_argument("--temporal-risk-margin", type=float, default=0.05)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--temporal-min-increase", type=float, default=0.0)
    parser.add_argument(
        "--temporal-normalize-weights",
        action="store_true",
        help="Normalize temporal risk weights within each failed episode.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--success-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    args.features = args.features.resolve()
    args.output_dir = args.output_dir.resolve()
    train(args)


if __name__ == "__main__":
    main()
