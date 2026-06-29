#!/usr/bin/env python3
"""Build high-risk failure states for frozen-hazard policy regularization.

The output actions are retained only to preserve a valid robomimic sequence;
they are never imitation targets. Each failed rollout contributes its earliest
detected high incremental-action-risk boundary by default. The frozen causal
prefix context is stored alongside the RGB observation history so post-training
can score actions sampled from the current and initialization policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from build_rgb_dp_mil_filtered_chunks import copy_fixed_window, decode
from build_rgb_dp_prefix_risk_filtered_chunks import (
    evaluate_gate,
    privileged_overlap_audit,
)
from robomimic.models.prefix_risk_nets import CausalPrefixRisk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_FEATURES = ROOT / "rollouts/rgb_dp/hazard_mil/chunk_features.npz"
DEFAULT_CHECKPOINT = ROOT / "trained_models/rgb_dp_causal_prefix_risk/best.pt"
DEFAULT_PREDICTIONS = (
    ROOT / "trained_models/rgb_dp_causal_prefix_risk/prefix_predictions.npz"
)
DEFAULT_SUMMARY = ROOT / "trained_models/rgb_dp_causal_prefix_risk/summary.json"
DEFAULT_CRITICAL_SUMMARY = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_critical_failure_chunks.summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_high_risk_constraint_chunks.hdf5"
)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    model = CausalPrefixRisk(
        feature_dim=int(checkpoint["feature_dim"]),
        prediction_horizon=int(checkpoint["prediction_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(train_args["hidden_dim"]),
        action_hidden_dim=int(train_args["action_hidden_dim"]),
        dropout=float(train_args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def normalize_features(features: np.ndarray, checkpoint: dict) -> np.ndarray:
    stats = checkpoint["stats"]
    return (
        (features - stats["feature_mean"]) / stats["feature_std"]
    ).astype(np.float32)


def normalize_actions(actions: np.ndarray, checkpoint: dict) -> np.ndarray:
    stats = checkpoint["stats"]
    return (
        (actions - stats["action_mean"]) / stats["action_std"]
    ).astype(np.float32)


def choose_high_risk_indices(
    steps: np.ndarray,
    positive_risk: np.ndarray,
    threshold: float,
    max_chunks: int,
    minimum_spacing: int,
) -> list[int]:
    candidates = np.flatnonzero(positive_risk > threshold)
    selected = []
    for index in candidates:
        step = int(steps[index])
        if all(
            abs(step - int(steps[previous])) >= minimum_spacing
            for previous in selected
        ):
            selected.append(int(index))
        if len(selected) >= max_chunks:
            break
    return selected


@torch.no_grad()
def build(args, gate: dict) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    features_cache = np.load(args.features)
    predictions = np.load(args.predictions)

    features = features_cache["features"].astype(np.float32)
    actions = features_cache["actions"].astype(np.float32)
    feature_episode_keys = decode(features_cache["episode_keys"])
    prediction_episode_keys = decode(predictions["episode_keys"])
    if feature_episode_keys != prediction_episode_keys:
        raise RuntimeError("feature and prediction episode orders differ")

    labels = predictions["episode_labels"].astype(np.float32)
    offsets = predictions["episode_offsets"].astype(np.int64)
    steps = predictions["steps"].astype(np.int64)
    action_deltas = predictions["action_deltas"].astype(np.float32)
    positive_risk = predictions["positive_action_risk"].astype(np.float32)
    threshold = float(predictions["positive_action_logodds_threshold"])
    normalized_features = normalize_features(features, checkpoint)
    normalized_actions = normalize_actions(actions, checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    detected_failures = 0
    split_labels = decode(predictions["episode_splits"])
    with h5py.File(args.source, "r") as source, h5py.File(
        args.output, "w"
    ) as output:
        source_failures = set(decode(source["mask/failure"][:]))
        output_data = output.create_group("data")
        output_keys = []

        for episode, source_key in enumerate(prediction_episode_keys):
            if labels[episode] < 0.5:
                continue
            if source_key not in source_failures:
                raise RuntimeError(f"{source_key} is absent from the failure mask")
            sl = slice(offsets[episode], offsets[episode + 1])
            selected = choose_high_risk_indices(
                steps[sl],
                positive_risk[sl],
                threshold,
                args.max_chunks_per_failure,
                args.minimum_spacing,
            )
            if selected:
                detected_failures += 1

            episode_features = torch.from_numpy(
                normalized_features[sl][None]
            ).to(device)
            contexts = model.encode_prefix(episode_features)[0]
            source_group = source[f"data/{source_key}"]
            for local_index in selected:
                global_index = offsets[episode] + local_index
                boundary = int(steps[global_index])
                context = contexts[local_index]
                scored_delta = model.action_delta(
                    context[None, None],
                    torch.from_numpy(
                        normalized_actions[global_index][None, None]
                    ).to(device),
                ).item()
                if not np.isclose(
                    scored_delta,
                    action_deltas[global_index],
                    atol=1e-4,
                    rtol=1e-4,
                ):
                    raise RuntimeError(
                        f"hazard score mismatch for {source_key} step {boundary}"
                    )

                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    boundary,
                    args.prediction_horizon,
                )
                context_np = context.cpu().numpy().astype(np.float32)
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
                output_group.attrs["segment_type"] = "prefix_action_risk_constraint"
                output_group.attrs["positive_action_logodds"] = float(
                    positive_risk[global_index]
                )
                output_group.attrs["action_delta"] = float(
                    action_deltas[global_index]
                )
                output_group.attrs["action_risk_threshold"] = threshold
                output_group.attrs["hazard_split"] = split_labels[episode]
                output_keys.append(output_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": boundary,
                        "target_end_exclusive": boundary + args.prediction_horizon,
                        "positive_action_logodds": float(
                            positive_risk[global_index]
                        ),
                        "action_delta": float(action_deltas[global_index]),
                        "hazard_split": split_labels[episode],
                    }
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (
            args.prediction_horizon + 1
        )
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    risks = [record["positive_action_logodds"] for record in records]
    result = {
        "source": str(args.source),
        "features": str(args.features),
        "hazard_checkpoint": str(args.checkpoint),
        "predictions": str(args.predictions),
        "output": str(args.output),
        "quality_gate": gate,
        "quality_gate_overridden": bool(args.allow_failed_gate and not gate["passed"]),
        "source_failure_rollouts": int(np.sum(labels)),
        "failure_rollouts_with_detected_transition": detected_failures,
        "retained_chunks": len(records),
        "retained_source_rollouts": len(
            {record["source_demo"] for record in records}
        ),
        "hazard_context_dim": int(checkpoint["args"]["hidden_dim"]),
        "selection": {
            "positive_action_logodds_threshold": threshold,
            "max_chunks_per_failure": args.max_chunks_per_failure,
            "minimum_spacing": args.minimum_spacing,
            "earliest_high_risk_transition_is_preferred": True,
            "logged_actions_are_not_imitation_targets": True,
        },
        "positive_action_logodds": {
            "minimum": float(np.min(risks)) if risks else None,
            "mean": float(np.mean(risks)) if risks else None,
            "maximum": float(np.max(risks)) if risks else None,
        },
        "split_counts": {
            split: sum(record["hazard_split"] == split for record in records)
            for split in ("train", "val", "test")
        },
        "privileged_overlap_audit": privileged_overlap_audit(
            records,
            args.critical_summary,
        ),
        "chunks": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "chunks"},
            indent=2,
        )
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--prefix-risk-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--critical-summary",
        type=Path,
        default=DEFAULT_CRITICAL_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--max-chunks-per-failure", type=int, default=1)
    parser.add_argument("--minimum-spacing", type=int, default=16)
    parser.add_argument("--min-test-detection", type=float, default=0.50)
    parser.add_argument("--min-test-near-hit", type=float, default=0.50)
    parser.add_argument("--min-test-critical-recall", type=float, default=0.50)
    parser.add_argument("--max-test-safe-fpr", type=float, default=0.10)
    parser.add_argument("--max-test-success-fpr", type=float, default=0.10)
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for field in (
        "source",
        "features",
        "checkpoint",
        "predictions",
        "prefix_risk_summary",
        "critical_summary",
        "output",
    ):
        setattr(args, field, getattr(args, field).resolve())

    summary = json.loads(args.prefix_risk_summary.read_text())
    gate = evaluate_gate(args, summary)
    gate_path = args.output.with_suffix(".gate.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    print(f"Wrote {gate_path}")
    if not gate["passed"] and not args.allow_failed_gate:
        raise RuntimeError(
            "causal prefix-risk localization gate failed; refusing to build "
            "negative policy constraints"
        )
    build(args, gate)


if __name__ == "__main__":
    main()
