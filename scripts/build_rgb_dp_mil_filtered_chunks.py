#!/usr/bin/env python3
"""Build positive DP chunks selected by a validated MIL hazard model.

The hazard model is trained only from rollout-level success / failure labels.
Privileged critical-failure labels are used here solely as a simulator-side
feasibility gate. By default, no dataset is written unless the model localizes
held-out critical chunks well enough to support causal segment filtering.

For a validated model, a failed rollout contributes only low-hazard chunks
whose complete 16-action target ends before the first detected hazardous
policy boundary. This avoids imitating either the detected failure or its
downstream consequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_PREDICTIONS = (
    ROOT / "trained_models/rgb_dp_hazard_mil/chunk_predictions.npz"
)
DEFAULT_MIL_SUMMARY = ROOT / "trained_models/rgb_dp_hazard_mil/summary.json"
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_mil_low_hazard_chunks.hdf5"
)


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def copy_fixed_window(
    source: h5py.Group,
    destination: h5py.Group,
    boundary: int,
    prediction_horizon: int,
) -> None:
    """Copy one history frame plus a real, unpadded action target."""
    indices = np.asarray(
        [max(0, boundary - 1)]
        + list(range(boundary, boundary + prediction_horizon)),
        dtype=np.int64,
    )
    for key, item in source.items():
        if isinstance(item, h5py.Group):
            child = destination.create_group(key)
            copy_fixed_window(item, child, boundary, prediction_horizon)
        else:
            destination.create_dataset(key, data=np.asarray(item)[indices])


def chunk_mask_for_episodes(
    episode_indices: np.ndarray,
    episodes: np.ndarray,
) -> np.ndarray:
    return np.isin(episode_indices, episodes)


def safe_rate(mask: np.ndarray) -> float | None:
    return float(np.mean(mask)) if len(mask) else None


def load_test_episodes(summary: dict) -> np.ndarray:
    checkpoint_path = Path(summary["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    return np.asarray(checkpoint["splits"]["test"], dtype=np.int64)


def evaluate_gate(args, summary: dict, predictions) -> dict:
    scores = predictions["scores"].astype(np.float64)
    episode_indices = predictions["episode_indices"].astype(np.int64)
    episode_labels = predictions["episode_labels"].astype(np.float32)
    critical = predictions["critical_labels"].astype(bool)
    threshold = float(predictions["low_hazard_threshold"])
    test_episodes = load_test_episodes(summary)
    test_chunks = chunk_mask_for_episodes(episode_indices, test_episodes)
    test_success_episodes = test_episodes[episode_labels[test_episodes] < 0.5]
    test_success_chunks = chunk_mask_for_episodes(
        episode_indices,
        test_success_episodes,
    )
    test_critical = critical & test_chunks

    localization = summary["metrics"]["privileged_localization_test"]
    top3 = localization["top3_hit_rate"]
    near_top1 = localization["within_one_boundary_top1_hit_rate"]
    critical_recall = safe_rate(scores[test_critical] >= threshold)
    success_fpr = safe_rate(scores[test_success_chunks] >= threshold)

    checks = {
        "heldout_top3_localization": {
            "value": top3,
            "required": args.min_test_top3,
            "passed": top3 is not None and top3 >= args.min_test_top3,
        },
        "heldout_near_top1_localization": {
            "value": near_top1,
            "required": args.min_test_near_top1,
            "passed": near_top1 is not None
            and near_top1 >= args.min_test_near_top1,
        },
        "heldout_critical_recall": {
            "value": critical_recall,
            "required": args.min_test_critical_recall,
            "threshold": threshold,
            "num_critical_chunks": int(np.sum(test_critical)),
            "passed": critical_recall is not None
            and critical_recall >= args.min_test_critical_recall,
        },
        "heldout_success_false_positive_rate": {
            "value": success_fpr,
            "maximum": args.max_test_success_fpr,
            "threshold": threshold,
            "num_success_chunks": int(np.sum(test_success_chunks)),
            "passed": success_fpr is not None
            and success_fpr <= args.max_test_success_fpr,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "test_episodes": int(len(test_episodes)),
        "hazard_threshold": threshold,
        "interpretation": (
            "A high rollout-level AUC is insufficient. The model must also "
            "localize privileged critical chunks on held-out failures while "
            "rarely flagging chunks from held-out successful rollouts."
        ),
    }


def choose_episode_chunks(
    *,
    local_steps: np.ndarray,
    local_scores: np.ndarray,
    threshold: float,
    prediction_horizon: int,
    max_chunks: int,
    minimum_spacing: int,
) -> tuple[list[int], int | None]:
    hazardous = np.flatnonzero(local_scores >= threshold)
    first_hazard_step = (
        int(local_steps[hazardous[0]]) if len(hazardous) else None
    )
    eligible = np.flatnonzero(local_scores < threshold)
    if first_hazard_step is not None:
        eligible = eligible[
            local_steps[eligible] + prediction_horizon <= first_hazard_step
        ]

    # Prefer the latest pre-hazard chunks because early reaching behavior is
    # already abundant in the original demonstrations. Enforce spacing so two
    # selected targets are not near-duplicates.
    selected = []
    for index in eligible[::-1]:
        step = int(local_steps[index])
        if all(abs(step - previous) >= minimum_spacing for previous in selected):
            selected.append(step)
        if len(selected) >= max_chunks:
            break
    return sorted(selected), first_hazard_step


def build(args, gate: dict, summary: dict, predictions) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    scores = predictions["scores"].astype(np.float64)
    steps = predictions["steps"].astype(np.int64)
    episode_keys = decode(predictions["episode_keys"])
    episode_labels = predictions["episode_labels"].astype(np.float32)
    offsets = predictions["episode_offsets"].astype(np.int64)
    threshold = float(predictions["low_hazard_threshold"])
    prediction_horizon = int(args.prediction_horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        source_failures = set(decode(source["mask/failure"][:]))
        output_data = output.create_group("data")
        output_keys = []

        for episode, source_key in enumerate(episode_keys):
            if episode_labels[episode] < 0.5:
                continue
            if source_key not in source_failures:
                raise RuntimeError(
                    f"prediction labels {source_key} as failure but source mask does not"
                )
            sl = slice(offsets[episode], offsets[episode + 1])
            selected, first_hazard_step = choose_episode_chunks(
                local_steps=steps[sl],
                local_scores=scores[sl],
                threshold=threshold,
                prediction_horizon=prediction_horizon,
                max_chunks=args.max_chunks_per_failure,
                minimum_spacing=args.minimum_spacing,
            )
            score_by_step = {
                int(step): float(score)
                for step, score in zip(steps[sl], scores[sl])
            }
            source_group = source[f"data/{source_key}"]
            for boundary in selected:
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    boundary,
                    prediction_horizon,
                )
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = prediction_horizon + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = boundary
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = "mil_low_hazard_prefix"
                output_group.attrs["hazard_score"] = score_by_step[boundary]
                output_group.attrs["hazard_threshold"] = threshold
                output_group.attrs["first_detected_hazard_step"] = (
                    -1 if first_hazard_step is None else first_hazard_step
                )
                output_keys.append(output_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": boundary,
                        "target_end_exclusive": boundary + prediction_horizon,
                        "hazard_score": score_by_step[boundary],
                        "first_detected_hazard_step": first_hazard_step,
                    }
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (prediction_horizon + 1)
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    result = {
        "source": str(args.source),
        "predictions": str(args.predictions),
        "mil_summary": str(args.mil_summary),
        "output": str(args.output),
        "quality_gate": gate,
        "quality_gate_overridden": bool(args.allow_failed_gate and not gate["passed"]),
        "source_failure_rollouts": int(np.sum(episode_labels)),
        "retained_chunks": len(records),
        "retained_source_rollouts": len({record["source_demo"] for record in records}),
        "hazard_threshold": threshold,
        "selection": {
            "prediction_horizon": prediction_horizon,
            "max_chunks_per_failure": args.max_chunks_per_failure,
            "minimum_spacing": args.minimum_spacing,
            "requires_complete_target_before_first_detected_hazard": True,
            "contains_padded_target_actions": False,
        },
        "chunks": records,
    }
    result_path = args.output.with_suffix(".summary.json")
    result_path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "chunks"},
            indent=2,
        )
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--mil-summary", type=Path, default=DEFAULT_MIL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--max-chunks-per-failure", type=int, default=2)
    parser.add_argument("--minimum-spacing", type=int, default=16)
    parser.add_argument("--min-test-top3", type=float, default=0.25)
    parser.add_argument("--min-test-near-top1", type=float, default=0.25)
    parser.add_argument("--min-test-critical-recall", type=float, default=0.50)
    parser.add_argument("--max-test-success-fpr", type=float, default=0.10)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="Diagnostic escape hatch; never use this dataset as validated data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for field in ("source", "predictions", "mil_summary", "output"):
        setattr(args, field, getattr(args, field).resolve())
    summary = json.loads(args.mil_summary.read_text())
    predictions = np.load(args.predictions)
    gate = evaluate_gate(args, summary, predictions)
    gate_path = args.output.with_suffix(".gate.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    print(f"Wrote {gate_path}")

    if args.report_only:
        return
    if not gate["passed"] and not args.allow_failed_gate:
        raise RuntimeError(
            "MIL localization gate failed; refusing to create policy-training "
            "data. Inspect the gate report. --allow-failed-gate is diagnostic only."
        )
    build(args, gate, summary, predictions)


if __name__ == "__main__":
    main()
