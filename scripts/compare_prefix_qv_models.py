#!/usr/bin/env python3
"""Compare causal prefix outcome models with Q/V-style offline metrics.

This script is the reward-model analogue of the earlier risk-model table. It
loads one or more trained prefix outcome checkpoints, evaluates them on the
same cached rollout features, and writes a markdown table that focuses on the
part that matters for action selection: whether Q-V contains action-specific
signal beyond state difficulty V.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from robomimic.models.prefix_risk_nets import make_causal_prefix_model
from train_rgb_dp_causal_prefix_risk import (
    binary_metrics,
    episode_indices,
    noisy_or_episode_probability,
    predict,
    rank_auc,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return path.parent.name, path


def corr(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(valid.sum()) < 2:
        return None
    x = first[valid] - first[valid].mean()
    y = second[valid] - second[valid].mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return None
    return float(np.sum(x * y) / denom)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"{path} is not a prefix outcome checkpoint")
    return checkpoint


def load_features(path: Path, checkpoint: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(path)
    features_raw = raw["features"].astype(np.float32)
    actions_raw = raw["actions"].astype(np.float32)
    raw_failure_labels = raw["episode_labels"].astype(np.float32)
    offsets = raw["episode_offsets"].astype(np.int64)
    stats = checkpoint["stats"]
    features = ((features_raw - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
    actions = ((actions_raw - stats["action_mean"]) / stats["action_std"]).astype(np.float32)
    return features, actions, raw_failure_labels, offsets


def chunk_labels(labels: np.ndarray, offsets: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.concatenate([episode_indices(offsets, int(e)) for e in episodes])
    repeated = np.concatenate(
        [
            np.full(offsets[e + 1] - offsets[e], labels[e], dtype=np.float32)
            for e in episodes
        ]
    )
    return indices, repeated


def evaluate_model(
    *,
    name: str,
    checkpoint_path: Path,
    features_path: Path | None,
    split: str,
    device: torch.device,
    batch_size: int,
    positive_delta_quantile: float,
) -> dict:
    checkpoint = load_checkpoint(checkpoint_path, device)
    args = checkpoint.get("args", {})
    resolved_features = features_path
    if resolved_features is None:
        resolved_features = Path(args["features"]).expanduser().resolve()
    features, actions, raw_failure_labels, offsets = load_features(resolved_features, checkpoint)
    target_outcome = str(args.get("target_outcome", "failure"))
    if target_outcome == "failure":
        labels = raw_failure_labels
    elif target_outcome == "success":
        labels = 1.0 - raw_failure_labels
    else:
        raise ValueError(f"unknown target_outcome={target_outcome}")
    splits = checkpoint.get("splits")
    if splits is None:
        episodes = np.arange(len(labels), dtype=np.int64)
    else:
        episodes = np.asarray(splits[split], dtype=np.int64)

    model = make_causal_prefix_model(
        model_arch=str(args.get("model_arch", "v1")),
        feature_dim=int(checkpoint["feature_dim"]),
        prediction_horizon=int(checkpoint["prediction_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        action_hidden_dim=int(args["action_hidden_dim"]),
        dropout=float(args["dropout"]),
        action_num_heads=int(args.get("action_num_heads", 4)),
        action_conv_layers=int(args.get("action_conv_layers", 2)),
        prefix_conv_layers=int(args.get("prefix_conv_layers", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    state_scores, action_scores, action_deltas = predict(
        model,
        features,
        actions,
        offsets,
        np.arange(len(labels), dtype=np.int64),
        device,
        batch_size,
    )

    selected_indices, selected_chunk_labels = chunk_labels(labels, offsets, episodes)
    v_prob = np.clip(state_scores[selected_indices], 1e-6, 1.0 - 1e-6)
    q_prob = np.clip(action_scores[selected_indices], 1e-6, 1.0 - 1e-6)
    delta = action_deltas[selected_indices]
    positive_delta = np.maximum(delta, 0.0)
    v_logit = np.log(v_prob / (1.0 - v_prob))
    q_logit = np.log(q_prob / (1.0 - q_prob))
    bag_q = noisy_or_episode_probability(action_scores, offsets, episodes)
    bag_v = noisy_or_episode_probability(state_scores, offsets, episodes)
    bag_q_metrics = binary_metrics(labels[episodes], bag_q)
    bag_v_metrics = binary_metrics(labels[episodes], bag_v)
    q_metrics = binary_metrics(selected_chunk_labels, q_prob)
    v_metrics = binary_metrics(selected_chunk_labels, v_prob)

    threshold = float(np.quantile(positive_delta, positive_delta_quantile))
    if target_outcome == "success":
        high_positive_mask = positive_delta >= threshold
        success_fpr = float(np.mean(high_positive_mask[selected_chunk_labels < 0.5]))
        positive_coverage = float(np.mean(high_positive_mask[selected_chunk_labels > 0.5]))
    else:
        high_positive_mask = positive_delta >= threshold
        success_fpr = float(np.mean(high_positive_mask[selected_chunk_labels < 0.5]))
        positive_coverage = float(np.mean(high_positive_mask[selected_chunk_labels > 0.5]))

    result = {
        "experiment": name,
        "checkpoint": str(checkpoint_path),
        "features": str(resolved_features),
        "target_outcome": target_outcome,
        "model_arch": str(args.get("model_arch", "v1")),
        "best_step": int(checkpoint.get("best_step", -1)),
        "split": split,
        "num_episodes": int(len(episodes)),
        "num_positive_episodes": int(np.sum(labels[episodes] > 0.5)),
        "bag_q_auc": bag_q_metrics["roc_auc"],
        "bag_q_ap": bag_q_metrics["average_precision"],
        "bag_q_bce": bag_q_metrics["binary_cross_entropy"],
        "bag_v_auc": bag_v_metrics["roc_auc"],
        "chunk_q_auc": q_metrics["roc_auc"],
        "chunk_q_ap": q_metrics["average_precision"],
        "chunk_v_auc": v_metrics["roc_auc"],
        "q_minus_v_auc": rank_auc(selected_chunk_labels, delta),
        "positive_q_minus_v_auc": rank_auc(selected_chunk_labels, positive_delta),
        "corr_v_q_prob": corr(v_prob, q_prob),
        "corr_v_delta": corr(v_prob, delta),
        "corr_vlogit_delta": corr(v_logit, delta),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "positive_delta_quantile": positive_delta_quantile,
        "positive_delta_threshold": threshold,
        "high_positive_chunks": int(np.sum(high_positive_mask)),
        "negative_fpr": success_fpr,
        "positive_coverage": positive_coverage,
        "q_logit_minus_v_logit_abs_err": float(np.mean(np.abs((q_logit - v_logit) - delta))),
    }
    return result


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def verdict(row: dict) -> str:
    parts = []
    if row["q_minus_v_auc"] is not None and row["q_minus_v_auc"] >= 0.75:
        parts.append("strong residual ranking")
    elif row["q_minus_v_auc"] is not None and row["q_minus_v_auc"] >= 0.6:
        parts.append("moderate residual ranking")
    else:
        parts.append("weak residual ranking")
    if row["delta_std"] < 0.05:
        parts.append("delta near collapse")
    if row["corr_v_delta"] is not None and abs(row["corr_v_delta"]) > 0.5:
        parts.append("residual still state-like")
    if row["bag_q_auc"] is not None and row["bag_q_auc"] > 0.9:
        parts.append("good bag predictor")
    return "; ".join(parts) + "."


def write_markdown(rows: list[dict], output_path: Path) -> None:
    headers = [
        "experiment",
        "arch",
        "best step",
        "bag Q AUC",
        "bag Q BCE",
        "chunk Q AUROC",
        "Q-V AUROC",
        "pos Q-V AUROC",
        "corr(V,Q)",
        "corr(V,Q-V)",
        "delta std",
        "high-pos chunks",
        "neg FPR",
        "pos coverage",
        "verdict",
    ]
    lines = ["# Prefix Q/V Reward Model Metric Table", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row["experiment"],
            row["model_arch"],
            row["best_step"],
            fmt(row["bag_q_auc"]),
            fmt(row["bag_q_bce"]),
            fmt(row["chunk_q_auc"]),
            fmt(row["q_minus_v_auc"]),
            fmt(row["positive_q_minus_v_auc"]),
            fmt(row["corr_v_q_prob"], 4),
            fmt(row["corr_v_delta"], 4),
            fmt(row["delta_std"]),
            row["high_positive_chunks"],
            fmt(row["negative_fpr"]),
            fmt(row["positive_coverage"]),
            verdict(row),
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines.extend([
        "",
        "Notes:",
        "",
        "- `Q` is the action-conditioned outcome probability, `V` is the prefix-only outcome probability.",
        "- `Q-V AUROC` uses the learned action residual / delta log-odds as the score.",
        "- For success-target models, positive coverage means retained successful chunks; negative FPR means failed chunks above the high-positive threshold.",
        "- For failure-target models, positive coverage means retained failure-risk chunks; negative FPR means successful chunks above the high-risk threshold.",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/best.pt or /path/to/best.pt. Can be repeated.")
    parser.add_argument("--features", type=Path, default=None, help="Override feature cache for all models. By default each checkpoint args['features'] is used.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--positive-delta-quantile", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.model:
        name, path = parse_model_spec(spec)
        rows.append(
            evaluate_model(
                name=name,
                checkpoint_path=path,
                features_path=args.features.expanduser().resolve() if args.features else None,
                split=args.split,
                device=device,
                batch_size=args.batch_size,
                positive_delta_quantile=args.positive_delta_quantile,
            )
        )
    json_path = args.output_dir / "prefix_qv_metrics.json"
    md_path = args.output_dir / "prefix_qv_metrics.md"
    json_path.write_text(json.dumps(rows, indent=2))
    write_markdown(rows, md_path)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
