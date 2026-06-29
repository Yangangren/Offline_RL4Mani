#!/usr/bin/env python3
"""Build exact positive chunks before a validated action-risk transition.

The causal prefix model is trained from rollout outcomes only. Simulator
privileged labels are used here only as a feasibility gate. For each failed
rollout with a detected transition, this builder retains at most two complete
16-action chunks ending before the first high incremental action-risk boundary.
The transition chunk and every later consequence are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from build_rgb_dp_mil_filtered_chunks import copy_fixed_window, decode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
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
    / "lift_rgb_dp_prefix_risk_good_chunks.hdf5"
)


def gate_check(value, *, minimum=None, maximum=None) -> dict:
    passed = value is not None
    result = {"value": value}
    if minimum is not None:
        result["minimum"] = minimum
        passed = passed and value >= minimum
    if maximum is not None:
        result["maximum"] = maximum
        passed = passed and value <= maximum
    result["passed"] = bool(passed)
    return result


def evaluate_gate(args, summary: dict) -> dict:
    metrics = summary["metrics"][
        "privileged_action_risk_onset_test_persistence_1"
    ]
    checks = {
        "heldout_onset_detection_rate": gate_check(
            metrics["onset_detection_rate"],
            minimum=args.min_test_detection,
        ),
        "heldout_within_one_boundary_rate": gate_check(
            metrics["within_one_boundary_onset_hit_rate"],
            minimum=args.min_test_near_hit,
        ),
        "heldout_critical_chunk_recall": gate_check(
            metrics["critical_chunk_signal_recall"],
            minimum=args.min_test_critical_recall,
        ),
        "heldout_safe_chunk_false_positive_rate": gate_check(
            metrics["safe_chunk_signal_false_positive_rate"],
            maximum=args.max_test_safe_fpr,
        ),
        "heldout_success_chunk_false_positive_rate": gate_check(
            metrics["success_chunk_signal_false_positive_rate"],
            maximum=args.max_test_success_fpr,
        ),
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "heldout_privileged_episodes": metrics[
            "episodes_with_privileged_critical_label"
        ],
        "threshold_source": (
            "95th percentile of positive action log-odds on successful "
            "validation-rollout chunks"
        ),
        "privileged_labels_used_for_training": summary[
            "privileged_labels_used_for_training"
        ],
    }


def select_pretransition_boundaries(
    *,
    steps: np.ndarray,
    action_risk: np.ndarray,
    threshold: float,
    prediction_horizon: int,
    safety_margin: int,
    max_chunks: int,
    minimum_spacing: int,
) -> tuple[list[int], int | None]:
    detected = np.flatnonzero(action_risk > threshold)
    if not len(detected):
        return [], None
    onset_step = int(steps[detected[0]])
    eligible = steps[
        steps + prediction_horizon <= onset_step - safety_margin
    ]
    selected = []
    for boundary in eligible[::-1]:
        boundary = int(boundary)
        if all(
            abs(boundary - existing) >= minimum_spacing
            for existing in selected
        ):
            selected.append(boundary)
        if len(selected) >= max_chunks:
            break
    return sorted(selected), onset_step


def privileged_overlap_audit(records: list[dict], path: Path) -> dict:
    critical_by_episode = {}
    if path.exists():
        summary = json.loads(path.read_text())
        for record in summary.get("chunks", []):
            critical_by_episode.setdefault(record["source_demo"], []).append(
                (
                    int(record["decision_boundary"]),
                    int(record["target_end_exclusive"]),
                )
            )
    labeled = 0
    overlap = 0
    before = 0
    after = 0
    exact = 0
    for record in records:
        windows = critical_by_episode.get(record["source_demo"], [])
        if not windows:
            continue
        labeled += 1
        start = int(record["decision_boundary"])
        end = int(record["target_end_exclusive"])
        exact += int(any(start == left for left, _ in windows))
        overlap += int(
            any(max(start, left) < min(end, right) for left, right in windows)
        )
        before += int(end <= min(left for left, _ in windows))
        after += int(start >= max(right for _, right in windows))
    return {
        "critical_summary": str(path),
        "chunks_from_privileged_labeled_failures": labeled,
        "exact_critical_boundary": exact,
        "target_overlaps_any_critical_window": overlap,
        "target_ends_before_first_critical_window": before,
        "target_starts_after_last_critical_window": after,
        "privileged_labels_used_for_selection": False,
    }


def build(args, gate: dict, predictions) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    episode_keys = decode(predictions["episode_keys"])
    labels = predictions["episode_labels"].astype(np.float32)
    offsets = predictions["episode_offsets"].astype(np.int64)
    steps = predictions["steps"].astype(np.int64)
    state_scores = predictions["state_scores"].astype(np.float64)
    action_scores = predictions["action_scores"].astype(np.float64)
    action_risk = predictions["positive_action_risk"].astype(np.float64)
    threshold = float(predictions["positive_action_logodds_threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    detected_failure_rollouts = 0
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        source_failures = set(decode(source["mask/failure"][:]))
        output_data = output.create_group("data")
        output_keys = []
        for episode, source_key in enumerate(episode_keys):
            if labels[episode] < 0.5:
                continue
            if source_key not in source_failures:
                raise RuntimeError(f"{source_key} is missing from source failure mask")
            sl = slice(offsets[episode], offsets[episode + 1])
            selected, onset_step = select_pretransition_boundaries(
                steps=steps[sl],
                action_risk=action_risk[sl],
                threshold=threshold,
                prediction_horizon=args.prediction_horizon,
                safety_margin=args.safety_margin,
                max_chunks=args.max_chunks_per_failure,
                minimum_spacing=args.minimum_spacing,
            )
            if onset_step is not None:
                detected_failure_rollouts += 1
            local_index = {
                int(step): offsets[episode] + index
                for index, step in enumerate(steps[sl])
            }
            source_group = source[f"data/{source_key}"]
            for boundary in selected:
                prediction_index = local_index[boundary]
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    boundary,
                    args.prediction_horizon,
                )
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = args.prediction_horizon + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = boundary
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = "prefix_before_action_risk"
                output_group.attrs["detected_risk_step"] = onset_step
                output_group.attrs["state_failure_score"] = state_scores[
                    prediction_index
                ]
                output_group.attrs["action_failure_score"] = action_scores[
                    prediction_index
                ]
                output_group.attrs["positive_action_logodds"] = action_risk[
                    prediction_index
                ]
                output_group.attrs["positive_action_logodds_threshold"] = threshold
                output_keys.append(output_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": boundary,
                        "target_end_exclusive": (
                            boundary + args.prediction_horizon
                        ),
                        "detected_risk_step": onset_step,
                        "state_failure_score": float(
                            state_scores[prediction_index]
                        ),
                        "action_failure_score": float(
                            action_scores[prediction_index]
                        ),
                        "positive_action_logodds": float(
                            action_risk[prediction_index]
                        ),
                    }
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (
            args.prediction_horizon + 1
        )
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    result = {
        "source": str(args.source),
        "predictions": str(args.predictions),
        "prefix_risk_summary": str(args.prefix_risk_summary),
        "output": str(args.output),
        "quality_gate": gate,
        "quality_gate_overridden": bool(args.allow_failed_gate and not gate["passed"]),
        "source_failure_rollouts": int(np.sum(labels)),
        "failure_rollouts_with_detected_transition": detected_failure_rollouts,
        "retained_chunks": len(records),
        "retained_source_rollouts": len(
            {record["source_demo"] for record in records}
        ),
        "selection": {
            "action_risk_threshold": threshold,
            "prediction_horizon": args.prediction_horizon,
            "pretransition_safety_margin": args.safety_margin,
            "max_chunks_per_failure": args.max_chunks_per_failure,
            "minimum_spacing": args.minimum_spacing,
            "complete_target_must_end_before_transition_minus_safety_margin": True,
            "contains_padded_target_actions": False,
            "failures_without_detected_transition_are_skipped": True,
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
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--prefix-risk-summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--critical-summary",
        type=Path,
        default=DEFAULT_CRITICAL_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--safety-margin", type=int, default=16)
    parser.add_argument("--max-chunks-per-failure", type=int, default=2)
    parser.add_argument("--minimum-spacing", type=int, default=16)
    parser.add_argument("--min-test-detection", type=float, default=0.50)
    parser.add_argument("--min-test-near-hit", type=float, default=0.50)
    parser.add_argument("--min-test-critical-recall", type=float, default=0.50)
    parser.add_argument("--max-test-safe-fpr", type=float, default=0.10)
    parser.add_argument("--max-test-success-fpr", type=float, default=0.10)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for field in (
        "source",
        "predictions",
        "prefix_risk_summary",
        "critical_summary",
        "output",
    ):
        setattr(args, field, getattr(args, field).resolve())

    summary = json.loads(args.prefix_risk_summary.read_text())
    predictions = np.load(args.predictions)
    gate = evaluate_gate(args, summary)
    gate_path = args.output.with_suffix(".gate.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    print(f"Wrote {gate_path}")
    if args.report_only:
        return
    if not gate["passed"] and not args.allow_failed_gate:
        raise RuntimeError(
            "causal prefix-risk localization gate failed; refusing to create "
            "policy-training data"
        )
    build(args, gate, predictions)


if __name__ == "__main__":
    main()
