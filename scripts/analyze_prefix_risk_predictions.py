#!/usr/bin/env python3
"""Analyze causal prefix-risk predictions.

This script is intentionally task-agnostic. It loads:

* a rollout HDF5 with ``mask/success`` and ``mask/failure``;
* a ``prefix_predictions.npz`` produced by ``train_rgb_dp_causal_prefix_risk.py``.

It writes compact JSON files that answer the first verification questions:

* Can state risk / action risk separate success and failure chunks?
* Is the incremental action risk ``Q - V`` sparse?
* How many failure chunks would be retained as low-risk under calibrated
  thresholds?
* Which failure chunks are highest-risk / lowest-risk for visual inspection?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def as_str_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]
    )


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q10": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(np.max(values)),
    }


def episode_indices(offsets: np.ndarray, episode: int) -> np.ndarray:
    return np.arange(offsets[episode], offsets[episode + 1], dtype=np.int64)


def record_for_chunk(
    *,
    chunk_index: int,
    episode_keys: np.ndarray,
    episode_labels: np.ndarray,
    episode_indices_array: np.ndarray,
    steps: np.ndarray,
    state_scores: np.ndarray,
    action_scores: np.ndarray,
    action_deltas: np.ndarray,
    rollouts: h5py.File,
) -> dict:
    episode = int(episode_indices_array[chunk_index])
    demo_key = str(episode_keys[episode])
    group = rollouts[f"data/{demo_key}"]
    episode_return = float(np.sum(group["rewards"][:]))
    return {
        "chunk_index": int(chunk_index),
        "episode_index": episode,
        "demo_key": demo_key,
        "episode_label": "failure" if episode_labels[episode] > 0.5 else "success",
        "episode_return": episode_return,
        "episode_horizon": int(group.attrs["num_samples"]),
        "decision_step": int(steps[chunk_index]),
        "state_risk": float(state_scores[chunk_index]),
        "action_risk": float(action_scores[chunk_index]),
        "action_delta_logodds": float(action_deltas[chunk_index]),
        "positive_action_risk": float(max(action_deltas[chunk_index], 0.0)),
    }


def threshold_report(
    *,
    name: str,
    threshold: float,
    positive_risk: np.ndarray,
    failure_chunks: np.ndarray,
    success_chunks: np.ndarray,
    episode_indices_array: np.ndarray,
    episode_labels: np.ndarray,
) -> dict:
    high = positive_risk > threshold
    failure_high = failure_chunks & high
    failure_low = failure_chunks & ~high
    success_high = success_chunks & high
    failure_episodes = np.flatnonzero(episode_labels > 0.5)
    failure_episodes_with_high = {
        int(episode_indices_array[index]) for index in np.flatnonzero(failure_high)
    }
    return {
        "name": name,
        "threshold": float(threshold),
        "failure_chunks": int(np.sum(failure_chunks)),
        "low_risk_failure_chunks": int(np.sum(failure_low)),
        "high_risk_failure_chunks": int(np.sum(failure_high)),
        "low_risk_failure_chunk_fraction": (
            float(np.mean(~high[failure_chunks])) if np.any(failure_chunks) else None
        ),
        "high_risk_failure_chunk_fraction": (
            float(np.mean(high[failure_chunks])) if np.any(failure_chunks) else None
        ),
        "success_chunks": int(np.sum(success_chunks)),
        "success_chunk_false_positive_fraction": (
            float(np.mean(high[success_chunks])) if np.any(success_chunks) else None
        ),
        "failure_episodes": int(len(failure_episodes)),
        "failure_episodes_with_any_high_risk_chunk": int(
            len(failure_episodes_with_high)
        ),
        "failure_episode_high_risk_coverage": (
            len(failure_episodes_with_high) / len(failure_episodes)
            if len(failure_episodes)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--success-quantiles",
        type=float,
        nargs="*",
        default=[0.50, 0.75, 0.90, 0.95],
        help="Positive Q-V quantiles on success chunks used as risk thresholds.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.predictions, allow_pickle=True)
    state_scores = raw["state_scores"].astype(np.float32)
    action_scores = raw["action_scores"].astype(np.float32)
    action_deltas = raw["action_deltas"].astype(np.float32)
    positive_risk = np.maximum(action_deltas, 0.0)
    steps = raw["steps"].astype(np.int64)
    episode_indices_array = raw["episode_indices"].astype(np.int64)
    episode_keys = as_str_array(raw["episode_keys"])
    episode_labels = raw["episode_labels"].astype(np.float32)
    offsets = raw["episode_offsets"].astype(np.int64)
    split_labels = (
        as_str_array(raw["episode_splits"])
        if "episode_splits" in raw
        else np.full(len(episode_labels), "unknown")
    )

    chunk_labels = episode_labels[episode_indices_array]
    failure_chunks = chunk_labels > 0.5
    success_chunks = chunk_labels < 0.5
    final_indices = np.asarray(
        [offsets[episode + 1] - 1 for episode in range(len(episode_labels))],
        dtype=np.int64,
    )

    success_positive_risk = positive_risk[success_chunks]
    threshold_reports = []
    for quantile in args.success_quantiles:
        if len(success_positive_risk):
            threshold = float(np.quantile(success_positive_risk, quantile))
            threshold_reports.append(
                threshold_report(
                    name=f"success_positive_risk_q{quantile:g}",
                    threshold=threshold,
                    positive_risk=positive_risk,
                    failure_chunks=failure_chunks,
                    success_chunks=success_chunks,
                    episode_indices_array=episode_indices_array,
                    episode_labels=episode_labels,
                )
            )
    if "positive_action_logodds_threshold" in raw:
        threshold_reports.append(
            threshold_report(
                name="stored_positive_action_logodds_threshold",
                threshold=float(np.asarray(raw["positive_action_logodds_threshold"])),
                positive_risk=positive_risk,
                failure_chunks=failure_chunks,
                success_chunks=success_chunks,
                episode_indices_array=episode_indices_array,
                episode_labels=episode_labels,
            )
        )

    metric_scores = {
        "state_risk": state_scores,
        "action_risk": action_scores,
        "action_delta_logodds": action_deltas,
        "positive_action_risk": positive_risk,
    }
    chunk_separation = {
        name: {
            "roc_auc_failure_vs_success": rank_auc(chunk_labels, values),
            "average_precision_failure": average_precision(chunk_labels, values),
            "success": stats(values[success_chunks]),
            "failure": stats(values[failure_chunks]),
        }
        for name, values in metric_scores.items()
    }
    final_separation = {
        name: {
            "roc_auc_failure_vs_success": rank_auc(
                episode_labels,
                values[final_indices],
            ),
            "average_precision_failure": average_precision(
                episode_labels,
                values[final_indices],
            ),
            "success": stats(values[final_indices][episode_labels < 0.5]),
            "failure": stats(values[final_indices][episode_labels > 0.5]),
        }
        for name, values in metric_scores.items()
    }

    with h5py.File(args.rollouts, "r") as rollouts:
        high_order = np.flatnonzero(failure_chunks)[
            np.argsort(-positive_risk[failure_chunks])
        ][: args.top_k]
        low_order = np.flatnonzero(failure_chunks)[
            np.argsort(positive_risk[failure_chunks])
        ][: args.top_k]
        high_risk_failure_chunks = [
            record_for_chunk(
                chunk_index=int(index),
                episode_keys=episode_keys,
                episode_labels=episode_labels,
                episode_indices_array=episode_indices_array,
                steps=steps,
                state_scores=state_scores,
                action_scores=action_scores,
                action_deltas=action_deltas,
                rollouts=rollouts,
            )
            for index in high_order
        ]
        low_risk_failure_chunks = [
            record_for_chunk(
                chunk_index=int(index),
                episode_keys=episode_keys,
                episode_labels=episode_labels,
                episode_indices_array=episode_indices_array,
                steps=steps,
                state_scores=state_scores,
                action_scores=action_scores,
                action_deltas=action_deltas,
                rollouts=rollouts,
            )
            for index in low_order
        ]

    split_summary = {}
    for split in sorted(set(split_labels.tolist())):
        episodes = split_labels == split
        split_summary[split] = {
            "episodes": int(np.sum(episodes)),
            "failures": int(np.sum(episode_labels[episodes] > 0.5)),
            "successes": int(np.sum(episode_labels[episodes] < 0.5)),
        }

    summary = {
        "rollouts": str(args.rollouts),
        "predictions": str(args.predictions),
        "num_episodes": int(len(episode_labels)),
        "num_success_episodes": int(np.sum(episode_labels < 0.5)),
        "num_failure_episodes": int(np.sum(episode_labels > 0.5)),
        "num_chunks": int(len(chunk_labels)),
        "num_success_chunks": int(np.sum(success_chunks)),
        "num_failure_chunks": int(np.sum(failure_chunks)),
        "episode_split_summary": split_summary,
        "chunk_score_separation": chunk_separation,
        "final_chunk_score_separation": final_separation,
        "threshold_reports": threshold_reports,
        "top_k": args.top_k,
        "high_risk_failure_chunks_path": str(
            args.output_dir / "high_risk_failure_chunks.json"
        ),
        "low_risk_failure_chunks_path": str(
            args.output_dir / "low_risk_failure_chunks.json"
        ),
    }
    (args.output_dir / "risk_analysis_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (args.output_dir / "high_risk_failure_chunks.json").write_text(
        json.dumps(high_risk_failure_chunks, indent=2)
    )
    (args.output_dir / "low_risk_failure_chunks.json").write_text(
        json.dumps(low_risk_failure_chunks, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
