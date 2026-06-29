#!/usr/bin/env python3
"""Build successful-rollout contexts for hazard regularization only.

This dataset is an ablation counterpart to high-risk failure constraints. It
contains policy decision boundaries from successful deployment rollouts, with
the frozen prefix-risk context attached. During DP post-training these chunks
are marked ``hazard_failure=True`` so their logged actions are *not* imitated;
they only provide states where sampled DP actions are regularized to have low
frozen hazard score relative to the frozen reference policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from build_rgb_dp_mil_filtered_chunks import copy_fixed_window, decode
from build_rgb_dp_high_risk_constraints import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FEATURES,
    DEFAULT_PREDICTIONS,
    DEFAULT_SOURCE,
    load_model,
    normalize_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_success_hazard_context_chunks.hdf5"
)


def select_success_indices(
    steps: np.ndarray,
    *,
    max_chunks: int,
    minimum_spacing: int,
) -> list[int]:
    selected: list[int] = []
    for index, step in enumerate(steps):
        step = int(step)
        if all(abs(step - int(steps[prev])) >= minimum_spacing for prev in selected):
            selected.append(index)
        if max_chunks > 0 and len(selected) >= max_chunks:
            break
    return selected


@torch.no_grad()
def build(args) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    features_cache = np.load(args.features)
    predictions = np.load(args.predictions)

    features = features_cache["features"].astype(np.float32)
    feature_episode_keys = decode(features_cache["episode_keys"])
    prediction_episode_keys = decode(predictions["episode_keys"])
    if feature_episode_keys != prediction_episode_keys:
        raise RuntimeError("feature and prediction episode orders differ")

    labels = predictions["episode_labels"].astype(np.float32)
    offsets = predictions["episode_offsets"].astype(np.int64)
    steps = predictions["steps"].astype(np.int64)
    state_scores = predictions["state_scores"].astype(np.float32)
    action_deltas = predictions["action_deltas"].astype(np.float32)
    positive_risk = predictions["positive_action_risk"].astype(np.float32)
    threshold = float(predictions["positive_action_logodds_threshold"])
    normalized_features = normalize_features(features, checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    success_rollouts_seen = 0
    success_rollouts_retained = 0

    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        source_successes = set(decode(source["mask/success"][:]))
        output_data = output.create_group("data")
        output_keys = []

        for episode, source_key in enumerate(prediction_episode_keys):
            if labels[episode] > 0.5:
                continue
            success_rollouts_seen += 1
            if source_key not in source_successes:
                raise RuntimeError(f"{source_key} is absent from the success mask")

            sl = slice(offsets[episode], offsets[episode + 1])
            selected = select_success_indices(
                steps[sl],
                max_chunks=args.max_chunks_per_success,
                minimum_spacing=args.minimum_spacing,
            )
            if not selected:
                continue
            success_rollouts_retained += 1

            episode_features = torch.from_numpy(
                normalized_features[sl][None]
            ).to(device)
            contexts = model.encode_prefix(episode_features)[0]
            source_group = source[f"data/{source_key}"]
            for local_index in selected:
                global_index = offsets[episode] + local_index
                boundary = int(steps[global_index])
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    boundary,
                    args.prediction_horizon,
                )
                context_np = contexts[local_index].cpu().numpy().astype(np.float32)
                output_group.create_dataset(
                    "hazard_context",
                    data=np.repeat(
                        context_np[None],
                        args.prediction_horizon + 1,
                        axis=0,
                    ),
                )
                for key, value in source_group.attrs.items():
                    if key != "num_samples":
                        output_group.attrs[key] = value
                output_group.attrs["num_samples"] = args.prediction_horizon + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = boundary
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = "success_hazard_regularization"
                output_group.attrs["state_failure_score"] = float(
                    state_scores[global_index]
                )
                output_group.attrs["positive_action_logodds"] = float(
                    positive_risk[global_index]
                )
                output_group.attrs["action_delta"] = float(
                    action_deltas[global_index]
                )
                output_group.attrs["action_risk_threshold"] = threshold
                output_keys.append(output_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": boundary,
                        "target_end_exclusive": boundary
                        + args.prediction_horizon,
                        "state_failure_score": float(state_scores[global_index]),
                        "positive_action_logodds": float(
                            positive_risk[global_index]
                        ),
                        "action_delta": float(action_deltas[global_index]),
                    }
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (
            args.prediction_horizon + 1
        )
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    risks = np.asarray(
        [record["positive_action_logodds"] for record in records],
        dtype=np.float32,
    )
    result = {
        "source": str(args.source),
        "features": str(args.features),
        "hazard_checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "retained_chunks": len(records),
        "retained_source_rollouts": success_rollouts_retained,
        "success_rollouts_seen": success_rollouts_seen,
        "prediction_horizon": args.prediction_horizon,
        "hazard_context_dim": int(checkpoint["args"]["hidden_dim"]),
        "selection": {
            "source": "successful deployment rollouts only",
            "max_chunks_per_success": args.max_chunks_per_success,
            "minimum_spacing": args.minimum_spacing,
            "all_success_boundaries_when_max_chunks_is_zero": (
                args.max_chunks_per_success == 0
            ),
            "logged_actions_are_not_imitation_targets": True,
        },
        "risk_statistics": {
            "mean_positive_action_logodds": float(risks.mean()) if len(risks) else None,
            "q50_positive_action_logodds": float(np.quantile(risks, 0.5)) if len(risks) else None,
            "q90_positive_action_logodds": float(np.quantile(risks, 0.9)) if len(risks) else None,
            "fraction_above_threshold": float(np.mean(risks > threshold)) if len(risks) else None,
            "threshold": threshold,
        },
        "chunks": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2))
    print(
        f"Wrote {args.output}: {len(records)} success hazard contexts from "
        f"{success_rollouts_retained}/{success_rollouts_seen} successful rollouts"
    )
    print(f"Wrote {summary_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument(
        "--max-chunks-per-success",
        type=int,
        default=0,
        help="0 means retain every policy decision boundary from each success.",
    )
    parser.add_argument("--minimum-spacing", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in ("source", "features", "predictions", "checkpoint", "output"):
        setattr(args, key, getattr(args, key).resolve())
    build(args)


if __name__ == "__main__":
    main()
