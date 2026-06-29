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


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_loss(
    model: CausalPrefixRisk,
    batch: dict[str, torch.Tensor],
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["features"], batch["actions"])
    mask = batch["mask"]
    state_loss = episode_normalized_bce(
        output["state_logit"],
        batch["labels"],
        mask,
    )
    action_loss = episode_normalized_bce(
        output["action_logit"],
        batch["labels"],
        mask,
    )
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
    total = (
        args.state_weight * state_loss
        + args.action_weight * action_loss
        + args.residual_l1_weight * residual_l1
        + args.shuffled_residual_weight * shuffled_residual
        + args.smoothness_weight * smoothness
    )
    return total, {
        "state_loss": float(state_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "residual_l1": float(residual_l1.detach()),
        "shuffled_residual": float(shuffled_residual.detach()),
        "smoothness": float(smoothness.detach()),
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
    return {
        "num_episodes": int(len(episodes)),
        "num_failure_episodes": int(np.sum(labels[episodes])),
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.total_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    best_quality = (-float("inf"), -float("inf"))
    history = []
    all_episodes = np.arange(len(labels), dtype=np.int64)
    for step in range(1, args.total_steps + 1):
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
                **components,
                "validation": validation,
            }
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)
            action_validation = validation["prefix_action_conditioned"]
            quality = (
                action_validation["roc_auc"],
                -action_validation["binary_cross_entropy"],
            )
            if math.isfinite(quality[0]) and quality > best_quality:
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
    parser.add_argument("--residual-l1-weight", type=float, default=0.01)
    parser.add_argument("--shuffled-residual-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
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
