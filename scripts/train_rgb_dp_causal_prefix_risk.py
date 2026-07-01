#!/usr/bin/env python3
"""Learn causal prefix failure risk and incremental action risk.

This experiment uses only rollout-level success / failure outcomes for
optimization. At policy boundary t, a causal GRU summarizes observation
features up to t. Two predictions are then made:

* V(h_t): failure risk from the observation prefix (state difficulty);
* Q(h_t, a_t): failure risk after adding the executed 16-action chunk.

Q is parameterized as V + delta. Positive delta is therefore incremental
action risk in log-odds units. Privileged failed-grasp and safe-reach labels
stored in the feature cache are used only after training for simulator-side
localization evaluation.
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

from robomimic.models.prefix_risk_nets import CausalPrefixRisk
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


def success_action_indices_by_step(
    steps: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grouped: dict[int, list[int]] = {}
    global_indices = []
    for episode in np.flatnonzero(labels < 0.5):
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
    top_k: int,
    step_window: int,
    device: torch.device,
) -> np.ndarray:
    success_indices = []
    for episode in np.flatnonzero(labels < 0.5):
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


def noisy_or_episode_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Multiple-instance episode BCE.

    A successful rollout is a negative bag, so every valid chunk should have
    low failure probability. A failed rollout is a positive bag, so at least
    one chunk should have high failure probability. This avoids assigning the
    failure label to every prefix-action chunk in a failed trajectory.
    """
    valid_log_not_fail = F.logsigmoid(-logits) * mask
    log_not_fail = valid_log_not_fail.sum(dim=1)
    not_fail_prob = torch.exp(log_not_fail).clamp(1e-6, 1.0 - 1e-6)
    fail_prob = 1.0 - not_fail_prob
    loss = -(
        labels * torch.log(fail_prob.clamp_min(1e-6))
        + (1.0 - labels) * torch.log(not_fail_prob.clamp_min(1e-6))
    )
    return loss.mean()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


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
    failure_mask = mask & (labels[:, None] > 0.5)
    weights = increase * failure_mask.float()
    if normalize:
        per_episode_sum = weights.sum(dim=1, keepdim=True)
        valid_failure = per_episode_sum > 1e-8
        normalized = weights / per_episode_sum.clamp_min(1e-8)
        weights = torch.where(valid_failure, normalized, weights)
    return weights


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
        state_loss = noisy_or_episode_bce(
            output["state_logit"],
            labels,
            mask,
        )
        action_loss = noisy_or_episode_bce(
            output["action_logit"],
            labels,
            mask,
        )
    elif args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor"):
        if stage == "state":
            state_loss = noisy_or_episode_bce(
                output["state_logit"],
                labels,
                mask,
            )
            action_loss = zero
        elif stage == "residual":
            state_loss = zero
            action_loss = noisy_or_episode_bce(
                output["action_logit"],
                labels,
                mask,
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
    success_mask = mask & (labels[:, None] < 0.5)
    success_residual = masked_mean(output["action_delta"].square(), success_mask)
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
        margin_violation = F.relu(
            args.contrast_margin
            - (output["action_delta"].unsqueeze(-1) - reference_delta)
        ).mean(dim=-1)
        failure_rows = labels > 0.5
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

    temporal_safe_anchor = zero
    temporal_risk_loss = zero
    temporal_weight_mean = zero
    temporal_weight_sum = zero
    temporal_active_fraction = zero
    if args.objective == "two_stage_temporal_safe_anchor" and stage == "residual":
        success_safe_mask = mask & (labels[:, None] < 0.5)
        temporal_safe_anchor = masked_mean(
            F.relu(output["action_delta"] - args.safe_anchor_epsilon).square(),
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
        "temporal_safe_anchor": float(temporal_safe_anchor.detach()),
        "temporal_risk_loss": float(temporal_risk_loss.detach()),
        "temporal_weight_mean": float(temporal_weight_mean.detach()),
        "temporal_weight_sum": float(temporal_weight_sum.detach()),
        "temporal_active_fraction": float(temporal_active_fraction.detach()),
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


def outcome_metrics(
    state_scores: np.ndarray,
    action_scores: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    episodes: np.ndarray,
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
    bag_state_scores = noisy_or_episode_probability(
        state_scores,
        offsets,
        episodes,
    )
    bag_action_scores = noisy_or_episode_probability(
        action_scores,
        offsets,
        episodes,
    )
    return {
        "num_episodes": int(len(episodes)),
        "num_failure_episodes": int(np.sum(labels[episodes])),
        "bag_state_noisy_or": binary_metrics(labels[episodes], bag_state_scores),
        "bag_action_noisy_or": binary_metrics(labels[episodes], bag_action_scores),
        "prefix_state": binary_metrics(chunk_labels, state_scores[indices]),
        "prefix_action_conditioned": binary_metrics(
            chunk_labels,
            action_scores[indices],
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


def calibrated_thresholds(
    action_scores: np.ndarray,
    action_deltas: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    validation_episodes: np.ndarray,
    quantile: float,
) -> dict:
    success_episodes = validation_episodes[labels[validation_episodes] < 0.5]
    indices = np.concatenate(
        [episode_indices(offsets, int(episode)) for episode in success_episodes]
    )
    positive_risk = np.maximum(action_deltas[indices], 0.0)
    return {
        "quantile": quantile,
        "action_probability": float(np.quantile(action_scores[indices], quantile)),
        "positive_action_logodds": float(np.quantile(positive_risk, quantile)),
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
    for module in (model.obs_projection, model.prefix_encoder, model.state_head):
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
    labels = raw["episode_labels"].astype(np.float32)
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
    model = CausalPrefixRisk(
        feature_dim=features.shape[-1],
        prediction_horizon=actions.shape[1],
        action_dim=actions.shape[2],
        hidden_dim=args.hidden_dim,
        action_hidden_dim=args.action_hidden_dim,
        dropout=args.dropout,
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
    best_quality = (-float("inf"), -float("inf"))
    history = []
    all_episodes = np.arange(len(labels), dtype=np.int64)
    success_by_step, all_success_action_indices = success_action_indices_by_step(
        steps,
        labels,
        offsets,
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
                labels,
                offsets,
                splits["val"],
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
            action_validation = validation["prefix_action_conditioned"]
            if args.objective in ("noisy_or_mil", "two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor"):
                action_validation = validation["bag_action_noisy_or"]
            quality = (
                action_validation["roc_auc"],
                -action_validation["binary_cross_entropy"],
            )
            allow_checkpoint = not (
                args.objective in ("two_stage_residual", "two_stage_action_contrast", "two_stage_matched_action_contrast", "two_stage_temporal_safe_anchor")
                and current_stage != "residual"
            )
            if allow_checkpoint and math.isfinite(quality[0]) and quality > best_quality:
                best_quality = quality
                torch.save(
                    {
                        "model": model.state_dict(),
                        "stats": stats,
                        "splits": splits,
                        "args": vars(args),
                        "feature_dim": int(features.shape[-1]),
                        "prediction_horizon": int(actions.shape[1]),
                        "action_dim": int(actions.shape[2]),
                        "best_step": step,
                        "best_quality": quality,
                    },
                    best_path,
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
    )
    metrics = {
        split: outcome_metrics(
            state_scores,
            action_scores,
            labels,
            offsets,
            episodes,
        )
        for split, episodes in splits.items()
    }
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

    split_labels = np.full(len(labels), "unused", dtype="<U5")
    for split, episodes in splits.items():
        split_labels[episodes] = split
    predictions_path = args.output_dir / "prefix_predictions.npz"
    np.savez_compressed(
        predictions_path,
        state_scores=state_scores,
        action_scores=action_scores,
        action_deltas=action_deltas,
        positive_action_risk=np.maximum(action_deltas, 0.0),
        steps=steps,
        episode_indices=raw["episode_indices"],
        episode_keys=raw["episode_keys"],
        episode_labels=labels,
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
    )
    summary = {
        "features": str(args.features),
        "checkpoint": str(best_path),
        "predictions": str(predictions_path),
        "num_episodes": int(len(labels)),
        "num_failure_episodes": int(np.sum(labels)),
        "num_chunks": int(len(features)),
        "objective": args.objective,
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
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
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
