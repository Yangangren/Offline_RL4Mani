#!/usr/bin/env python3
"""Stage-1 success-potential model for Square RGB-DP rollouts.

This is the first, deliberately simple, part of the progress-reward reset:

    p_theta(h_t) = probability that the current trajectory will eventually
                   succeed, given the current observation/history feature.

The supervision is only trajectory-level success / failure. Every transition in
an episode inherits the same terminal label. This script focuses on validation:
besides BCE and AUC, it reports episode-level start/end/mean/max separation and
whether p_theta tends to increase along successful trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz"
)
DEFAULT_OUTPUT = ROOT / "trained_models/square_success_potential/stage1_default_features"


def decode_array(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind == "S":
        return x.astype(str)
    return x


def load_feature_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg
        start = end
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    if pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.float64)
    precision = tp / (np.arange(len(labels), dtype=np.float64) + 1.0)
    return float((precision * sorted_labels).sum() / pos)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.float64)
    probs = probs.astype(np.float64)
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    labels = labels.astype(np.float64)
    probs = probs.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = max(len(labels), 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += float(mask.sum()) / n * abs(conf - acc)
    return float(ece)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PotentialDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        split: str,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        include_step_feature: bool,
    ):
        split_array = decode_array(data["split"]).astype(str)
        self.indices = np.flatnonzero(split_array == split)
        features = data["obs_features"][self.indices].astype(np.float32)
        features = (features - feature_mean.astype(np.float32)) / feature_std.astype(np.float32)
        if include_step_feature:
            steps = data["steps"][self.indices].astype(np.float32)
            # Per-sample feature in [0, 1]-ish. This is off by default because
            # we first want state/history success potential, not a time shortcut.
            steps = steps / max(float(np.max(data["steps"])), 1.0)
            features = np.concatenate([features, steps[:, None]], axis=1)
        self.features = features.astype(np.float32)
        self.labels = data["episode_success"][self.indices].astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def make_loader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def class_balanced_bce(logits: torch.Tensor, labels: torch.Tensor, pos_frac: float) -> torch.Tensor:
    # Equal total mass for positive and negative samples.
    pos_w = 0.5 / max(pos_frac, 1e-6)
    neg_w = 0.5 / max(1.0 - pos_frac, 1e-6)
    weights = torch.where(labels > 0.5, torch.as_tensor(pos_w, device=labels.device), torch.as_tensor(neg_w, device=labels.device))
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * weights).mean()


@torch.no_grad()
def predict_split(
    model: nn.Module,
    dataset: PotentialDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch["features"])
        logits.append(out.detach().cpu().numpy().astype(np.float32))
        labels.append(batch["labels"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits), np.concatenate(labels)


def scalar_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probs = stable_sigmoid(logits)
    labels_bool = labels > 0.5
    eps = 1e-7
    nll = -np.mean(labels * np.log(np.clip(probs, eps, 1.0)) + (1.0 - labels) * np.log(np.clip(1.0 - probs, eps, 1.0)))
    pred = probs >= 0.5
    return {
        "num_samples": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "binary_cross_entropy": float(nll),
        "accuracy_at_0p5": float(np.mean(pred == labels_bool)),
        "auc": binary_roc_auc(labels_bool, probs),
        "average_precision": binary_average_precision(labels_bool, probs),
        "brier": brier_score(labels, probs),
        "ece_10bin": expected_calibration_error(labels, probs, bins=10),
        "prob_mean": float(probs.mean()),
        "prob_positive_mean": float(probs[labels_bool].mean()) if labels_bool.any() else None,
        "prob_negative_mean": float(probs[~labels_bool].mean()) if (~labels_bool).any() else None,
    }


@dataclass
class EpisodeRecord:
    episode_id: str
    split: str
    source: str
    success: float
    num_steps: int
    p_start: float
    p_end: float
    p_mean: float
    p_max: float
    p_min: float
    p_delta: float
    monotonic_fraction: float
    negative_step_fraction: float
    mean_abs_diff: float


def episode_records(
    data: dict[str, np.ndarray],
    sample_probs: np.ndarray,
    sample_indices: np.ndarray,
    monotonic_eps: float,
) -> list[EpisodeRecord]:
    source = decode_array(data["source"]).astype(str)
    demo = decode_array(data["demo"]).astype(str)
    split = decode_array(data["split"]).astype(str)
    ids = np.char.add(np.char.add(source, ":"), demo)
    by_episode: dict[str, list[int]] = {}
    for local_i, global_i in enumerate(sample_indices):
        by_episode.setdefault(str(ids[global_i]), []).append(local_i)
    records: list[EpisodeRecord] = []
    for episode_id, local_indices in by_episode.items():
        global_indices = sample_indices[np.asarray(local_indices, dtype=np.int64)]
        order = np.argsort(data["steps"][global_indices], kind="mergesort")
        local = np.asarray(local_indices, dtype=np.int64)[order]
        global_sorted = global_indices[order]
        probs = sample_probs[local].astype(np.float64)
        diffs = np.diff(probs)
        if len(diffs) == 0:
            monotonic_fraction = 1.0
            negative_step_fraction = 0.0
            mean_abs_diff = 0.0
        else:
            monotonic_fraction = float(np.mean(diffs >= -monotonic_eps))
            negative_step_fraction = float(np.mean(diffs < -monotonic_eps))
            mean_abs_diff = float(np.mean(np.abs(diffs)))
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                split=str(split[global_sorted[0]]),
                source=str(source[global_sorted[0]]),
                success=float(data["episode_success"][global_sorted[0]]),
                num_steps=int(len(global_sorted)),
                p_start=float(probs[0]),
                p_end=float(probs[-1]),
                p_mean=float(probs.mean()),
                p_max=float(probs.max()),
                p_min=float(probs.min()),
                p_delta=float(probs[-1] - probs[0]),
                monotonic_fraction=monotonic_fraction,
                negative_step_fraction=negative_step_fraction,
                mean_abs_diff=mean_abs_diff,
            )
        )
    return records


def summarize_episode_subset(records: list[EpisodeRecord]) -> dict[str, Any]:
    if not records:
        return {"num_episodes": 0}
    out: dict[str, Any] = {"num_episodes": len(records)}
    labels = np.asarray([r.success for r in records], dtype=np.float32)
    for score_name in ("p_start", "p_end", "p_mean", "p_max", "p_delta"):
        scores = np.asarray([getattr(r, score_name) for r in records], dtype=np.float32)
        out[f"{score_name}_auc"] = binary_roc_auc(labels > 0.5, scores)
        out[f"{score_name}_ap"] = binary_average_precision(labels > 0.5, scores)
    for subset_name, mask in {
        "success": labels > 0.5,
        "failure": labels <= 0.5,
        "all": np.ones_like(labels, dtype=bool),
    }.items():
        if not mask.any():
            continue
        subset = [r for r, keep in zip(records, mask) if keep]
        out[subset_name] = {
            "num_episodes": int(mask.sum()),
            "p_start_mean": float(np.mean([r.p_start for r in subset])),
            "p_end_mean": float(np.mean([r.p_end for r in subset])),
            "p_mean_mean": float(np.mean([r.p_mean for r in subset])),
            "p_max_mean": float(np.mean([r.p_max for r in subset])),
            "p_delta_mean": float(np.mean([r.p_delta for r in subset])),
            "monotonic_fraction_mean": float(np.mean([r.monotonic_fraction for r in subset])),
            "negative_step_fraction_mean": float(np.mean([r.negative_step_fraction for r in subset])),
            "mean_abs_diff": float(np.mean([r.mean_abs_diff for r in subset])),
        }
    return out


def summarize_episode_records(records: list[EpisodeRecord]) -> dict[str, Any]:
    out = summarize_episode_subset(records)
    if not records:
        return out
    by_source: dict[str, list[EpisodeRecord]] = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)
    out["by_source"] = {
        source: summarize_episode_subset(source_records)
        for source, source_records in sorted(by_source.items())
        if source_records
    }
    return out


def build_time_bin_curves(
    data: dict[str, np.ndarray],
    sample_probs: np.ndarray,
    sample_indices: np.ndarray,
    num_bins: int,
) -> dict[str, list[float | None]]:
    source = decode_array(data["source"]).astype(str)
    demo = decode_array(data["demo"]).astype(str)
    ids = np.char.add(np.char.add(source, ":"), demo)
    by_episode: dict[str, list[int]] = {}
    for local_i, global_i in enumerate(sample_indices):
        by_episode.setdefault(str(ids[global_i]), []).append(local_i)
    bins = {
        "success": [[] for _ in range(num_bins)],
        "failure": [[] for _ in range(num_bins)],
    }
    for _, local_indices in by_episode.items():
        global_indices = sample_indices[np.asarray(local_indices, dtype=np.int64)]
        order = np.argsort(data["steps"][global_indices], kind="mergesort")
        local = np.asarray(local_indices, dtype=np.int64)[order]
        global_sorted = global_indices[order]
        probs = sample_probs[local].astype(np.float64)
        target = "success" if float(data["episode_success"][global_sorted[0]]) > 0.5 else "failure"
        denom = max(len(probs) - 1, 1)
        for i, p in enumerate(probs):
            b = min(int((i / denom) * num_bins), num_bins - 1)
            bins[target][b].append(float(p))
    return {
        key: [float(np.mean(values)) if values else None for values in value_bins]
        for key, value_bins in bins.items()
    }


def evaluate_all(
    model: nn.Module,
    datasets: dict[str, PotentialDataset],
    data: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    monotonic_eps: float,
    num_time_bins: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    for split, dataset in datasets.items():
        logits, labels = predict_split(model, dataset, batch_size, device)
        probs = stable_sigmoid(logits).astype(np.float32)
        split_metrics = scalar_metrics(labels, logits)
        records = episode_records(data, probs, dataset.indices, monotonic_eps)
        split_metrics["episode"] = summarize_episode_records(records)
        split_metrics["time_bin_curves"] = build_time_bin_curves(data, probs, dataset.indices, num_time_bins)
        metrics[split] = split_metrics
        prediction_arrays[f"{split}_indices"] = dataset.indices.astype(np.int64)
        prediction_arrays[f"{split}_logits"] = logits.astype(np.float32)
        prediction_arrays[f"{split}_probs"] = probs.astype(np.float32)
        prediction_arrays[f"{split}_labels"] = labels.astype(np.float32)
    return metrics, prediction_arrays


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    args: argparse.Namespace,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    step: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "step": int(step),
            "metrics": metrics,
            "history": history,
        },
        path,
    )


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    feature_dim: int,
    best: dict[str, Any],
    history: list[dict[str, Any]],
    final_metrics: dict[str, Any] | None = None,
) -> None:
    summary = {
        "features": str(args.features),
        "output_dir": str(args.output_dir),
        "feature_dim": int(feature_dim),
        "input_dim": int(feature_dim + (1 if args.include_step_feature else 0)),
        "include_step_feature": bool(args.include_step_feature),
        "num_samples": int(data["obs_features"].shape[0]),
        "num_train": int((decode_array(data["split"]).astype(str) == "train").sum()),
        "num_val": int((decode_array(data["split"]).astype(str) == "val").sum()),
        "num_test": int((decode_array(data["split"]).astype(str) == "test").sum()),
        "best": best,
        "history": history,
        "final_metrics": final_metrics,
        "checkpoints": {
            "best": str(args.output_dir / "best.pt"),
            "latest": str(args.output_dir / "latest.pt"),
            "last": str(args.output_dir / "last.pt"),
            "predictions": str(args.output_dir / "predictions.npz"),
        },
    }
    path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--balanced-bce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step-feature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--monotonic-eps", type=float, default=0.01)
    parser.add_argument("--num-time-bins", type=int, default=10)
    args = parser.parse_args()

    args.features = args.features.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data = load_feature_npz(args.features)
    split_array = decode_array(data["split"]).astype(str)
    train_indices = np.flatnonzero(split_array == "train")
    raw_train_features = data["obs_features"][train_indices].astype(np.float32)
    if args.normalize_features:
        feature_mean = raw_train_features.mean(axis=0).astype(np.float32)
        feature_std = raw_train_features.std(axis=0).astype(np.float32)
        feature_std = np.maximum(feature_std, 1e-6).astype(np.float32)
    else:
        feature_mean = np.zeros(raw_train_features.shape[-1], dtype=np.float32)
        feature_std = np.ones(raw_train_features.shape[-1], dtype=np.float32)

    datasets = {
        split: PotentialDataset(data, split, feature_mean, feature_std, args.include_step_feature)
        for split in ("train", "val", "test")
    }
    input_dim = int(datasets["train"].features.shape[-1])
    model = MLP(input_dim, tuple(int(x) for x in args.hidden_dims), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(datasets["train"], args.batch_size, shuffle=True, seed=args.seed)
    iterator = cycle(loader)
    train_pos_frac = float(datasets["train"].labels.mean())

    best = {
        "val_auc": -float("inf"),
        "val_episode_end_auc": -float("inf"),
        "val_bce": float("inf"),
        "step": None,
    }
    history: list[dict[str, Any]] = []

    for step in range(1, args.total_steps + 1):
        model.train()
        batch = batch_to_device(next(iterator), device)
        logits = model(batch["features"])
        labels = batch["labels"]
        if args.balanced_bce:
            loss = class_balanced_bce(logits, labels, train_pos_frac)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "grad_norm": float(grad_norm),
                    }
                ),
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.total_steps:
            metrics, _ = evaluate_all(
                model,
                datasets,
                data,
                args.eval_batch_size,
                device,
                args.monotonic_eps,
                args.num_time_bins,
            )
            record = {"step": step, **metrics}
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)
            save_checkpoint(
                args.output_dir / "latest.pt",
                model=model,
                args=args,
                feature_mean=feature_mean,
                feature_std=feature_std,
                step=step,
                metrics=metrics,
                history=history,
            )
            val_auc = metrics["val"]["auc"]
            val_episode_end_auc = metrics["val"]["episode"].get("p_end_auc")
            val_bce = float(metrics["val"]["binary_cross_entropy"])
            # Primary checkpoint: sample-level validation AUC. We still log
            # episode-end AUC because it is often more physically meaningful.
            is_better = val_auc is not None and float(val_auc) > float(best["val_auc"])
            if is_better:
                best = {
                    "val_auc": float(val_auc),
                    "val_episode_end_auc": None if val_episode_end_auc is None else float(val_episode_end_auc),
                    "val_bce": val_bce,
                    "step": int(step),
                }
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model=model,
                    args=args,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    step=step,
                    metrics=metrics,
                    history=history,
                )
            write_summary(
                args.output_dir / "partial_summary.json",
                args=args,
                data=data,
                feature_dim=data["obs_features"].shape[-1],
                best=best,
                history=history,
            )

    final_metrics, prediction_arrays = evaluate_all(
        model,
        datasets,
        data,
        args.eval_batch_size,
        device,
        args.monotonic_eps,
        args.num_time_bins,
    )
    save_checkpoint(
        args.output_dir / "last.pt",
        model=model,
        args=args,
        feature_mean=feature_mean,
        feature_std=feature_std,
        step=args.total_steps,
        metrics=final_metrics,
        history=history,
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        **prediction_arrays,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        features=np.asarray(str(args.features)),
        best_checkpoint=np.asarray(str(args.output_dir / "best.pt")),
    )
    write_summary(
        args.output_dir / "summary.json",
        args=args,
        data=data,
        feature_dim=data["obs_features"].shape[-1],
        best=best,
        history=history,
        final_metrics=final_metrics,
    )
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
