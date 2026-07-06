#!/usr/bin/env python3
"""Relabel one-step Square RGB-DP IDQL features with success-potential progress.

Stage 1 learned a success-potential model

    p_theta(h_t) = P(eventual success | current observation/history feature).

This script turns it into a dense one-step reward while preserving the sparse
task-completion reward as an anchor:

    r'_t = r_sparse_t + alpha * clip(p_theta(h_{t+1}) - p_theta(h_t), -c, c)

The output NPZ keeps the same schema as the default one-step IDQL feature file,
so it can be passed directly to train_square_rgb_dp_one_step_idql.py. Extra
arrays are stored for audit/comparison:

    sparse_chunk_returns
    success_potential_p
    success_potential_next_p
    success_potential_progress_raw
    success_potential_progress
    success_potential_progress_reward
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_square_success_potential_stage1 import MLP, binary_average_precision, binary_roc_auc, stable_sigmoid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_FEATURES = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz"
)
DEFAULT_POTENTIAL_CHECKPOINT = (
    ROOT / "trained_models/square_success_potential/stage1_default_features/best.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/"
    "success_potential_sparse_plus_progress_alpha0p5_clip0p1_one_step_features.npz"
)


def decode_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return array.astype(str)


def encode_string(value: str) -> np.ndarray:
    return np.asarray(value.encode("utf-8"))


def safe_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "positive_fraction": float(np.mean(values > 0.0)),
        "negative_fraction": float(np.mean(values < 0.0)),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def instantiate_potential_model(checkpoint: dict[str, Any], feature_dim: int, device: torch.device) -> MLP:
    ckpt_args = checkpoint["args"]
    include_step_feature = bool(ckpt_args.get("include_step_feature", False))
    input_dim = int(feature_dim + (1 if include_step_feature else 0))
    model = MLP(
        input_dim=input_dim,
        hidden_dims=tuple(int(x) for x in ckpt_args.get("hidden_dims", (512, 512, 256))),
        dropout=float(ckpt_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def build_model_input(
    *,
    features: np.ndarray,
    steps: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    include_step_feature: bool,
    max_step: float,
) -> np.ndarray:
    x = (features.astype(np.float32) - feature_mean.astype(np.float32)) / feature_std.astype(np.float32)
    if include_step_feature:
        step_feature = steps.astype(np.float32) / max(max_step, 1.0)
        x = np.concatenate([x, step_feature[:, None]], axis=1)
    return x.astype(np.float32)


@torch.no_grad()
def predict_potential(
    *,
    model: MLP,
    features: np.ndarray,
    steps: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    include_step_feature: bool,
    max_step: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = build_model_input(
        features=features,
        steps=steps,
        feature_mean=feature_mean,
        feature_std=feature_std,
        include_step_feature=include_step_feature,
        max_step=max_step,
    )
    logits: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        out = model(batch).detach().cpu().numpy().astype(np.float32)
        logits.append(out)
    logits_np = np.concatenate(logits, axis=0).astype(np.float32)
    probs_np = stable_sigmoid(logits_np).astype(np.float32)
    return logits_np, probs_np


def group_episode_indices(source: np.ndarray, demo: np.ndarray, steps: np.ndarray) -> OrderedDict[str, np.ndarray]:
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, (source_name, demo_name) in enumerate(zip(source, demo, strict=True)):
        groups.setdefault(f"{source_name}:{demo_name}", []).append(index)
    ordered: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for episode_id, indices in groups.items():
        ordered[episode_id] = np.asarray(sorted(indices, key=lambda i: int(steps[i])), dtype=np.int64)
    return ordered


def episode_summary(
    *,
    source: np.ndarray,
    demo: np.ndarray,
    steps: np.ndarray,
    success: np.ndarray,
    sparse_rewards: np.ndarray,
    progress_rewards: np.ndarray,
    final_rewards: np.ndarray,
    p: np.ndarray,
) -> dict[str, Any]:
    groups = group_episode_indices(source, demo, steps)
    rows = []
    for episode_id, indices in groups.items():
        rows.append(
            {
                "episode_id": episode_id,
                "source": str(source[indices[0]]),
                "success": float(success[indices[0]]),
                "num_steps": int(len(indices)),
                "p_start": float(p[indices[0]]),
                "p_end": float(p[indices[-1]]),
                "p_mean": float(np.mean(p[indices])),
                "p_max": float(np.max(p[indices])),
                "p_delta": float(p[indices[-1]] - p[indices[0]]),
                "sparse_return": float(np.sum(sparse_rewards[indices])),
                "progress_return": float(np.sum(progress_rewards[indices])),
                "final_return": float(np.sum(final_rewards[indices])),
            }
        )

    labels = np.asarray([row["success"] for row in rows], dtype=np.float32)
    out: dict[str, Any] = {"num_episodes": len(rows)}
    for key in ("p_start", "p_end", "p_mean", "p_max", "p_delta", "sparse_return", "progress_return", "final_return"):
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        out[f"{key}_auc"] = binary_roc_auc(labels > 0.5, values)
        out[f"{key}_ap"] = binary_average_precision(labels > 0.5, values)
        out[f"{key}_stats"] = safe_stats(values)

    for subset_name, mask in {
        "success": labels > 0.5,
        "failure": labels <= 0.5,
        "all": np.ones_like(labels, dtype=bool),
    }.items():
        if not mask.any():
            continue
        subset = [row for row, keep in zip(rows, mask, strict=True) if keep]
        out[subset_name] = {
            "num_episodes": int(mask.sum()),
            "p_start_mean": float(np.mean([row["p_start"] for row in subset])),
            "p_end_mean": float(np.mean([row["p_end"] for row in subset])),
            "p_delta_mean": float(np.mean([row["p_delta"] for row in subset])),
            "sparse_return_mean": float(np.mean([row["sparse_return"] for row in subset])),
            "progress_return_mean": float(np.mean([row["progress_return"] for row in subset])),
            "final_return_mean": float(np.mean([row["final_return"] for row in subset])),
        }

    by_source = {}
    for source_name in sorted(np.unique(source).tolist()):
        mask = np.asarray([row["source"] == source_name for row in rows], dtype=bool)
        if not mask.any():
            continue
        source_rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
        by_source[source_name] = {
            "num_episodes": int(mask.sum()),
            "success_fraction": float(np.mean([row["success"] for row in source_rows])),
            "p_start_mean": float(np.mean([row["p_start"] for row in source_rows])),
            "p_end_mean": float(np.mean([row["p_end"] for row in source_rows])),
            "p_delta_mean": float(np.mean([row["p_delta"] for row in source_rows])),
            "sparse_return_mean": float(np.mean([row["sparse_return"] for row in source_rows])),
            "progress_return_mean": float(np.mean([row["progress_return"] for row in source_rows])),
            "final_return_mean": float(np.mean([row["final_return"] for row in source_rows])),
        }
    out["by_source"] = by_source
    return out


def sample_summary(
    *,
    source: np.ndarray,
    split: np.ndarray,
    success: np.ndarray,
    p: np.ndarray,
    next_p: np.ndarray,
    sparse_rewards: np.ndarray,
    progress: np.ndarray,
    progress_rewards: np.ndarray,
    final_rewards: np.ndarray,
) -> dict[str, Any]:
    labels = success > 0.5
    out: dict[str, Any] = {
        "num_samples": int(len(success)),
        "success_fraction": float(success.mean()),
        "p_auc": binary_roc_auc(labels, p),
        "p_ap": binary_average_precision(labels, p),
        "next_p_auc": binary_roc_auc(labels, next_p),
        "next_p_ap": binary_average_precision(labels, next_p),
        "sparse_reward_stats": safe_stats(sparse_rewards),
        "progress_stats": safe_stats(progress),
        "progress_reward_stats": safe_stats(progress_rewards),
        "final_reward_stats": safe_stats(final_rewards),
    }
    by_source = {}
    for source_name in sorted(np.unique(source).tolist()):
        mask = source == source_name
        by_source[source_name] = {
            "count": int(mask.sum()),
            "success_fraction": float(success[mask].mean()),
            "p_mean": float(p[mask].mean()),
            "next_p_mean": float(next_p[mask].mean()),
            "sparse_reward_mean": float(sparse_rewards[mask].mean()),
            "sparse_nonzero_fraction": float(np.mean(sparse_rewards[mask] > 0.0)),
            "progress_mean": float(progress[mask].mean()),
            "progress_positive_fraction": float(np.mean(progress[mask] > 0.0)),
            "progress_negative_fraction": float(np.mean(progress[mask] < 0.0)),
            "progress_reward_mean": float(progress_rewards[mask].mean()),
            "final_reward_mean": float(final_rewards[mask].mean()),
        }
    out["by_source"] = by_source
    by_split = {}
    for split_name in sorted(np.unique(split).tolist()):
        mask = split == split_name
        by_split[split_name] = {
            "count": int(mask.sum()),
            "success_fraction": float(success[mask].mean()),
            "p_auc": binary_roc_auc(success[mask] > 0.5, p[mask]),
            "p_ap": binary_average_precision(success[mask] > 0.5, p[mask]),
            "sparse_reward_mean": float(sparse_rewards[mask].mean()),
            "progress_reward_mean": float(progress_rewards[mask].mean()),
            "final_reward_mean": float(final_rewards[mask].mean()),
        }
    out["by_split"] = by_split
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--potential-checkpoint", type=Path, default=DEFAULT_POTENTIAL_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--progress-clip", type=float, default=0.1)
    parser.add_argument(
        "--reward-mode",
        choices=("sparse_plus_progress", "progress_only", "sparse_plus_positive_progress"),
        default="sparse_plus_progress",
    )
    parser.add_argument(
        "--terminal-next-mode",
        choices=("feature", "outcome", "current"),
        default="feature",
        help=(
            "How to handle next potential on done transitions. "
            "feature uses cached next_obs_features; outcome uses the terminal success label; "
            "current makes terminal progress zero."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_features = args.base_features.resolve()
    args.potential_checkpoint = args.potential_checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    base_npz = np.load(args.base_features, allow_pickle=True)
    base = {key: base_npz[key] for key in base_npz.files}
    num_total = int(base["obs_features"].shape[0])
    num_samples = num_total if args.max_samples is None else min(int(args.max_samples), num_total)
    if int(base["chunk_horizon"]) != 1:
        raise ValueError(f"expected one-step features, got chunk_horizon={base['chunk_horizon']}")

    checkpoint = torch.load(args.potential_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint["args"]
    feature_mean = checkpoint["feature_mean"].astype(np.float32)
    feature_std = np.maximum(checkpoint["feature_std"].astype(np.float32), 1e-6)
    feature_dim = int(base["obs_features"].shape[-1])
    if feature_mean.shape[0] != feature_dim:
        raise ValueError(
            f"potential feature_dim={feature_mean.shape[0]} but base feature_dim={feature_dim}"
        )

    include_step_feature = bool(ckpt_args.get("include_step_feature", False))
    max_step = float(np.max(base["steps"][:num_samples]))
    model = instantiate_potential_model(checkpoint, feature_dim, device)

    obs_features = base["obs_features"][:num_samples].astype(np.float32)
    next_obs_features = base["next_obs_features"][:num_samples].astype(np.float32)
    steps = base["steps"][:num_samples].astype(np.int64)
    next_steps = base["next_steps"][:num_samples].astype(np.int64)
    dones = base["dones"][:num_samples].astype(np.float32)
    success = base["episode_success"][:num_samples].astype(np.float32)
    source = decode_array(base["source"][:num_samples])
    demo = decode_array(base["demo"][:num_samples])
    split = decode_array(base["split"][:num_samples])

    print(f"Scoring p(h_t) for {num_samples} one-step transitions on {device}", flush=True)
    logits, p = predict_potential(
        model=model,
        features=obs_features,
        steps=steps,
        feature_mean=feature_mean,
        feature_std=feature_std,
        include_step_feature=include_step_feature,
        max_step=max_step,
        batch_size=int(args.batch_size),
        device=device,
    )
    next_logits, next_p = predict_potential(
        model=model,
        features=next_obs_features,
        steps=next_steps,
        feature_mean=feature_mean,
        feature_std=feature_std,
        include_step_feature=include_step_feature,
        max_step=max_step,
        batch_size=int(args.batch_size),
        device=device,
    )

    done_mask = dones > 0.5
    if args.terminal_next_mode == "outcome":
        next_p = next_p.copy()
        next_logits = next_logits.copy()
        terminal_p = success[done_mask].astype(np.float32)
        next_p[done_mask] = terminal_p
        eps = 1e-6
        next_logits[done_mask] = np.log(np.clip(terminal_p, eps, 1.0 - eps) / np.clip(1.0 - terminal_p, eps, 1.0 - eps))
    elif args.terminal_next_mode == "current":
        next_p = next_p.copy()
        next_logits = next_logits.copy()
        next_p[done_mask] = p[done_mask]
        next_logits[done_mask] = logits[done_mask]
    elif args.terminal_next_mode != "feature":
        raise ValueError(f"unknown terminal next mode: {args.terminal_next_mode}")

    progress_raw = (next_p - p).astype(np.float32)
    progress = np.clip(progress_raw, -float(args.progress_clip), float(args.progress_clip)).astype(np.float32)
    progress_reward = (float(args.alpha) * progress).astype(np.float32)
    sparse_rewards = base["chunk_returns"][:num_samples].astype(np.float32)

    if args.reward_mode == "sparse_plus_progress":
        final_rewards = (sparse_rewards + progress_reward).astype(np.float32)
        formula = "r'_t = r_sparse_t + alpha * clip(p(h_{t+1}) - p(h_t), -c, c)"
    elif args.reward_mode == "progress_only":
        final_rewards = progress_reward.astype(np.float32)
        formula = "r'_t = alpha * clip(p(h_{t+1}) - p(h_t), -c, c)"
    elif args.reward_mode == "sparse_plus_positive_progress":
        progress_reward = (float(args.alpha) * np.maximum(progress, 0.0)).astype(np.float32)
        final_rewards = (sparse_rewards + progress_reward).astype(np.float32)
        formula = "r'_t = r_sparse_t + alpha * max(clip(p(h_{t+1}) - p(h_t), -c, c), 0)"
    else:
        raise ValueError(f"unknown reward mode: {args.reward_mode}")

    output: dict[str, np.ndarray] = {}
    for key, value in base.items():
        if value.shape == ():
            output[key] = value
        elif value.shape[0] == num_total:
            output[key] = value[:num_samples]
        else:
            output[key] = value

    output["default_chunk_returns"] = sparse_rewards
    output["sparse_chunk_returns"] = sparse_rewards
    output["chunk_returns"] = final_rewards
    output["reward_mean"] = np.asarray(float(final_rewards.mean()), dtype=np.float32)
    output["reward_std"] = np.asarray(max(float(final_rewards.std()), 1e-6), dtype=np.float32)
    output["success_potential_logits"] = logits.astype(np.float32)
    output["success_potential_next_logits"] = next_logits.astype(np.float32)
    output["success_potential_p"] = p.astype(np.float32)
    output["success_potential_next_p"] = next_p.astype(np.float32)
    output["success_potential_progress_raw"] = progress_raw.astype(np.float32)
    output["success_potential_progress"] = progress.astype(np.float32)
    output["success_potential_progress_reward"] = progress_reward.astype(np.float32)
    output["success_potential_alpha"] = np.asarray(float(args.alpha), dtype=np.float32)
    output["success_potential_progress_clip"] = np.asarray(float(args.progress_clip), dtype=np.float32)
    output["success_potential_reward_mode"] = encode_string(args.reward_mode)
    output["success_potential_reward_formula"] = encode_string(formula)
    output["success_potential_terminal_next_mode"] = encode_string(args.terminal_next_mode)
    output["success_potential_checkpoint"] = encode_string(str(args.potential_checkpoint))
    output["base_features"] = encode_string(str(args.base_features))
    # Compatibility with the current IDQL logging hook.
    output["risk_reward_mode"] = encode_string(f"success_potential_{args.reward_mode}")
    output["risk_reward_formula"] = encode_string(formula)

    np.savez_compressed(args.output, **output)

    sample_stats = sample_summary(
        source=source,
        split=split,
        success=success,
        p=p,
        next_p=next_p,
        sparse_rewards=sparse_rewards,
        progress=progress,
        progress_rewards=progress_reward,
        final_rewards=final_rewards,
    )
    ep_stats = episode_summary(
        source=source,
        demo=demo,
        steps=steps,
        success=success,
        sparse_rewards=sparse_rewards,
        progress_rewards=progress_reward,
        final_rewards=final_rewards,
        p=p,
    )
    summary = {
        "base_features": str(args.base_features),
        "potential_checkpoint": str(args.potential_checkpoint),
        "output": str(args.output),
        "num_samples": int(num_samples),
        "num_total_base_samples": int(num_total),
        "feature_dim": int(feature_dim),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "include_step_feature": bool(include_step_feature),
        "reward_mode": args.reward_mode,
        "reward_formula": formula,
        "alpha": float(args.alpha),
        "progress_clip": float(args.progress_clip),
        "terminal_next_mode": args.terminal_next_mode,
        "sample": sample_stats,
        "episode": ep_stats,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
