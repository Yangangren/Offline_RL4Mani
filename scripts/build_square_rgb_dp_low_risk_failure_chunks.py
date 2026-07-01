#!/usr/bin/env python3
"""Build Square RGB-DP low-risk chunks from failed policy rollouts.

The causal prefix-risk model is trained from rollout-level success / failure
labels. This script uses its incremental action-risk score, ``positive(Q-V)``,
to select failure-rollout chunks that look low risk. The selected chunks are
used as positive imitation targets, alongside original demos and successful
policy rollouts.

Each output demo contains exactly:

* one context observation at ``max(0, boundary - 1)``;
* sixteen real future actions from ``boundary : boundary + prediction_horizon``.

Configure robomimic's SequenceDataset with ``demo_start_only=True`` and
``sample_start_offset=1`` to recover the original two-frame observation history
without training on padded target actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_PREDICTIONS = (
    ROOT / "trained_models/square_rgb_dp_causal_prefix_risk/epoch190/prefix_predictions.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection"
    / "square_rgb_dp_low_risk_failure_chunks.hdf5"
)
DEFAULT_TARGET_PEG_XY = (0.23, 0.10)
SQUARE_NUT_POS_SLICE = slice(7, 10)


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def copy_fixed_window(
    source: h5py.Group,
    destination: h5py.Group,
    boundary: int,
    prediction_horizon: int,
) -> None:
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


def clip_action_dataset(group: h5py.Group) -> None:
    """Remove tiny float overshoot introduced by rollout serialization."""
    if "actions" not in group:
        return
    actions = np.asarray(group["actions"][:], dtype=np.float32)
    group["actions"][...] = np.clip(actions, -1.0, 1.0)


def threshold_from_predictions(args, predictions) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.use_stored_threshold and "positive_action_logodds_threshold" in predictions:
        return float(np.asarray(predictions["positive_action_logodds_threshold"]))
    positive_risk = predictions["positive_action_risk"].astype(np.float64)
    labels = predictions["episode_labels"].astype(np.float32)
    episode_indices = predictions["episode_indices"].astype(np.int64)
    success_chunks = labels[episode_indices] < 0.5
    if not np.any(success_chunks):
        raise RuntimeError("cannot compute success quantile: no successful chunks")
    return float(np.quantile(positive_risk[success_chunks], args.success_quantile))


def progress_metrics_for_episode(
    source_group: h5py.Group,
    steps: np.ndarray,
    prediction_horizon: int,
    target_peg_xy: np.ndarray,
    args,
) -> dict[str, np.ndarray]:
    """Compute privileged Square task-progress metrics for candidate chunks.

    The risk model tells us which chunks are probably not hazardous. For positive
    imitation, however, low risk is not enough: a safe no-op should not become a
    target. These metrics use simulator ground-truth observations available in
    robomimic datasets to prefer chunks that actually move the square nut toward
    the target peg.
    """

    obj = source_group["obs/object"]
    eef = source_group["obs/robot0_eef_pos"]
    horizon = int(prediction_horizon)
    num_steps = int(source_group["actions"].shape[0])

    peg_xy_progress = np.zeros(len(steps), dtype=np.float64)
    nut_disp_end = np.zeros(len(steps), dtype=np.float64)
    nut_z_gain = np.zeros(len(steps), dtype=np.float64)
    eef_approach_gain = np.zeros(len(steps), dtype=np.float64)
    eef_dist_start = np.zeros(len(steps), dtype=np.float64)
    eef_dist_end = np.zeros(len(steps), dtype=np.float64)

    for i, step in enumerate(steps):
        step = int(step)
        end = min(step + horizon - 1, num_steps - 1)
        nut0 = np.asarray(obj[step, SQUARE_NUT_POS_SLICE], dtype=np.float64)
        nut1 = np.asarray(obj[end, SQUARE_NUT_POS_SLICE], dtype=np.float64)
        eef0 = np.asarray(eef[step], dtype=np.float64)
        eef1 = np.asarray(eef[end], dtype=np.float64)
        peg_start = float(np.linalg.norm(nut0[:2] - target_peg_xy))
        peg_end = float(np.linalg.norm(nut1[:2] - target_peg_xy))
        eef_d0 = float(np.linalg.norm(eef0 - nut0))
        eef_d1 = float(np.linalg.norm(eef1 - nut1))
        peg_xy_progress[i] = peg_start - peg_end
        nut_disp_end[i] = float(np.linalg.norm(nut1 - nut0))
        nut_z_gain[i] = float(nut1[2] - nut0[2])
        eef_approach_gain[i] = eef_d0 - eef_d1
        eef_dist_start[i] = eef_d0
        eef_dist_end[i] = eef_d1

    progress_mask = (
        (peg_xy_progress >= args.min_peg_xy_progress)
        & (nut_disp_end >= args.min_nut_displacement)
        & (nut_z_gain >= args.min_nut_z_gain)
    )
    progress_score = (
        args.peg_progress_weight * peg_xy_progress
        + args.z_gain_weight * np.maximum(nut_z_gain, 0.0)
        + args.nut_displacement_weight * nut_disp_end
        + args.eef_approach_weight * np.maximum(eef_approach_gain, 0.0)
    )
    return {
        "peg_xy_progress": peg_xy_progress,
        "nut_disp_end": nut_disp_end,
        "nut_z_gain": nut_z_gain,
        "eef_approach_gain": eef_approach_gain,
        "eef_dist_start": eef_dist_start,
        "eef_dist_end": eef_dist_end,
        "progress_mask": progress_mask,
        "progress_score": progress_score,
    }


def choose_low_risk_chunks(
    *,
    steps: np.ndarray,
    positive_risk: np.ndarray,
    threshold: float,
    max_chunks: int,
    minimum_spacing: int,
    prefer: str,
    progress_metrics: dict[str, np.ndarray] | None = None,
    progress_risk_penalty: float = 0.0,
) -> list[int]:
    eligible_mask = positive_risk <= threshold
    if prefer == "progress_aware":
        if progress_metrics is None:
            raise ValueError("progress_aware selection requires progress metrics")
        eligible_mask = eligible_mask & progress_metrics["progress_mask"]

    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) == 0:
        return []
    if prefer == "lowest":
        order = eligible[np.argsort(positive_risk[eligible])]
    elif prefer == "latest":
        order = eligible[np.argsort(-steps[eligible])]
    elif prefer == "earliest":
        order = eligible[np.argsort(steps[eligible])]
    elif prefer == "progress_aware":
        progress_score = progress_metrics["progress_score"] - progress_risk_penalty * positive_risk
        # Stable lexicographic ordering: maximize progress-aware score, then prefer
        # lower risk, then earlier chunks for deterministic tie-breaking.
        order = eligible[
            np.lexsort((steps[eligible], positive_risk[eligible], -progress_score[eligible]))
        ]
    else:
        raise ValueError(f"unknown prefer mode: {prefer}")

    selected: list[int] = []
    for index in order:
        step = int(steps[index])
        if all(abs(step - existing) >= minimum_spacing for existing in selected):
            selected.append(step)
        if len(selected) >= max_chunks:
            break
    return sorted(selected)


def audit_source_match(
    source: h5py.File,
    output: h5py.File,
    records: list[dict],
    prediction_horizon: int,
) -> None:
    for record in records:
        out = output[f"data/{record['output_demo']}"]
        src = source[f"data/{record['source_demo']}"]
        boundary = int(record["decision_boundary"])
        indices = np.asarray(
            [max(0, boundary - 1)] + list(range(boundary, boundary + prediction_horizon))
        )
        if int(out.attrs["num_samples"]) != prediction_horizon + 1:
            raise RuntimeError(f"{record['output_demo']} has wrong num_samples")
        if int(out.attrs["sample_start_offset"]) != 1:
            raise RuntimeError(f"{record['output_demo']} has wrong sample_start_offset")
        for key in (
            "actions",
            "obs/agentview_image",
            "obs/robot0_eye_in_hand_image",
            "obs/robot0_eef_pos",
            "obs/robot0_eef_quat",
            "obs/robot0_gripper_qpos",
        ):
            if key not in src:
                continue
            expected = src[key][:][indices]
            if key == "actions":
                expected = np.clip(expected.astype(np.float32), -1.0, 1.0)
            if not np.array_equal(out[key][:], expected):
                raise RuntimeError(
                    f"source mismatch for {record['output_demo']} key={key}"
                )


def build(args) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    predictions = np.load(args.predictions, allow_pickle=True)
    episode_keys = decode(predictions["episode_keys"])
    labels = predictions["episode_labels"].astype(np.float32)
    offsets = predictions["episode_offsets"].astype(np.int64)
    steps = predictions["steps"].astype(np.int64)
    state_scores = predictions["state_scores"].astype(np.float64)
    action_scores = predictions["action_scores"].astype(np.float64)
    action_deltas = predictions["action_deltas"].astype(np.float64)
    positive_risk = predictions["positive_action_risk"].astype(np.float64)
    threshold = threshold_from_predictions(args, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    source_failure_steps = 0
    failure_rollouts_with_retained_chunks = set()
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        source_failures = set(decode(source["mask/failure"][:]))
        data = output.create_group("data")
        output_keys = []
        for episode, source_key in enumerate(episode_keys):
            if labels[episode] < 0.5:
                continue
            if source_key not in source_failures:
                raise RuntimeError(f"{source_key} is missing from source failure mask")
            source_group = source[f"data/{source_key}"]
            source_failure_steps += int(source_group.attrs["num_samples"])
            sl = slice(offsets[episode], offsets[episode + 1])
            local_steps = steps[sl]
            progress_metrics = progress_metrics_for_episode(
                source_group=source_group,
                steps=local_steps,
                prediction_horizon=args.prediction_horizon,
                target_peg_xy=np.asarray(args.target_peg_xy, dtype=np.float64),
                args=args,
            )
            selected = choose_low_risk_chunks(
                steps=local_steps,
                positive_risk=positive_risk[sl],
                threshold=threshold,
                max_chunks=args.max_chunks_per_failure,
                minimum_spacing=args.minimum_spacing,
                prefer=args.prefer,
                progress_metrics=progress_metrics,
                progress_risk_penalty=args.progress_risk_penalty,
            )
            index_by_step = {
                int(step): offsets[episode] + index
                for index, step in enumerate(steps[sl])
            }
            for boundary in selected:
                prediction_index = index_by_step[boundary]
                output_key = f"demo_{len(output_keys)}"
                output_group = data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    boundary,
                    args.prediction_horizon,
                )
                clip_action_dataset(output_group)
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = args.prediction_horizon + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = boundary
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = (
                    "square_prefix_risk_progress_low_failure_chunk"
                    if args.prefer == "progress_aware"
                    else "square_prefix_risk_low_failure_chunk"
                )
                output_group.attrs["state_failure_score"] = state_scores[prediction_index]
                output_group.attrs["action_failure_score"] = action_scores[prediction_index]
                output_group.attrs["action_delta_logodds"] = action_deltas[prediction_index]
                local_prediction_index = prediction_index - offsets[episode]
                output_group.attrs["positive_action_logodds"] = positive_risk[
                    prediction_index
                ]
                output_group.attrs["positive_action_logodds_threshold"] = threshold
                output_group.attrs["privileged_peg_xy_progress"] = progress_metrics[
                    "peg_xy_progress"
                ][local_prediction_index]
                output_group.attrs["privileged_nut_displacement"] = progress_metrics[
                    "nut_disp_end"
                ][local_prediction_index]
                output_group.attrs["privileged_nut_z_gain"] = progress_metrics[
                    "nut_z_gain"
                ][local_prediction_index]
                output_group.attrs["privileged_eef_approach_gain"] = progress_metrics[
                    "eef_approach_gain"
                ][local_prediction_index]
                output_group.attrs["privileged_progress_score"] = progress_metrics[
                    "progress_score"
                ][local_prediction_index]
                output_group.attrs["privileged_selection_score"] = (
                    progress_metrics["progress_score"][local_prediction_index]
                    - args.progress_risk_penalty * positive_risk[prediction_index]
                )
                output_keys.append(output_key)
                failure_rollouts_with_retained_chunks.add(source_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": int(boundary),
                        "target_end_exclusive": int(boundary + args.prediction_horizon),
                        "state_failure_score": float(state_scores[prediction_index]),
                        "action_failure_score": float(action_scores[prediction_index]),
                        "action_delta_logodds": float(action_deltas[prediction_index]),
                        "positive_action_logodds": float(positive_risk[prediction_index]),
                        "privileged_peg_xy_progress": float(
                            progress_metrics["peg_xy_progress"][local_prediction_index]
                        ),
                        "privileged_nut_displacement": float(
                            progress_metrics["nut_disp_end"][local_prediction_index]
                        ),
                        "privileged_nut_z_gain": float(
                            progress_metrics["nut_z_gain"][local_prediction_index]
                        ),
                        "privileged_eef_approach_gain": float(
                            progress_metrics["eef_approach_gain"][local_prediction_index]
                        ),
                        "privileged_progress_score": float(
                            progress_metrics["progress_score"][local_prediction_index]
                        ),
                        "privileged_selection_score": float(
                            progress_metrics["progress_score"][local_prediction_index]
                            - args.progress_risk_penalty * positive_risk[prediction_index]
                        ),
                    }
                )

        data.attrs["env_args"] = source["data"].attrs["env_args"]
        data.attrs["total"] = len(output_keys) * (args.prediction_horizon + 1)
        mask = output.create_group("mask")
        mask["all"] = np.asarray(output_keys, dtype="S")
        audit_source_match(source, output, records, args.prediction_horizon)

    risks = np.asarray([record["positive_action_logodds"] for record in records])
    peg_progress = np.asarray(
        [record["privileged_peg_xy_progress"] for record in records], dtype=np.float64
    )
    nut_disp = np.asarray(
        [record["privileged_nut_displacement"] for record in records], dtype=np.float64
    )
    nut_z = np.asarray(
        [record["privileged_nut_z_gain"] for record in records], dtype=np.float64
    )
    progress_score = np.asarray(
        [record["privileged_progress_score"] for record in records], dtype=np.float64
    )

    def stats(values: np.ndarray) -> dict:
        if len(values) == 0:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "source": str(args.source),
        "predictions": str(args.predictions),
        "output": str(args.output),
        "source_failure_rollouts": int(np.sum(labels > 0.5)),
        "source_failure_steps": int(source_failure_steps),
        "retained_chunks": len(records),
        "retained_source_rollouts": len(failure_rollouts_with_retained_chunks),
        "target_actions_per_chunk": args.prediction_horizon,
        "stored_frames_per_chunk": args.prediction_horizon + 1,
        "contains_padded_target_actions": False,
        "selection": {
            "score": (
                "privileged task progress - risk penalty, gated by positive(Q - V)"
                if args.prefer == "progress_aware"
                else "positive(Q - V) action log-odds"
            ),
            "threshold": threshold,
            "threshold_source": (
                "explicit"
                if args.threshold is not None
                else (
                    "stored prediction threshold"
                    if args.use_stored_threshold
                    else f"success chunk q{args.success_quantile:g}"
                )
            ),
            "max_chunks_per_failure": args.max_chunks_per_failure,
            "minimum_spacing": args.minimum_spacing,
            "prefer": args.prefer,
            "prediction_horizon": args.prediction_horizon,
            "target_peg_xy": list(args.target_peg_xy),
            "min_peg_xy_progress": args.min_peg_xy_progress,
            "min_nut_displacement": args.min_nut_displacement,
            "min_nut_z_gain": args.min_nut_z_gain,
            "peg_progress_weight": args.peg_progress_weight,
            "z_gain_weight": args.z_gain_weight,
            "nut_displacement_weight": args.nut_displacement_weight,
            "eef_approach_weight": args.eef_approach_weight,
            "progress_risk_penalty": args.progress_risk_penalty,
        },
        "risk_stats": stats(risks),
        "privileged_progress_stats": {
            "peg_xy_progress": stats(peg_progress),
            "nut_displacement": stats(nut_disp),
            "nut_z_gain": stats(nut_z),
            "progress_score": stats(progress_score),
        },
        "chunks": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "chunks"}, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--max-chunks-per-failure", type=int, default=2)
    parser.add_argument("--minimum-spacing", type=int, default=16)
    parser.add_argument(
        "--prefer",
        choices=("lowest", "latest", "earliest", "progress_aware"),
        default="lowest",
        help=(
            "How to choose among low-risk chunks within each failed rollout. "
            "progress_aware additionally requires privileged Square task progress."
        ),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--success-quantile", type=float, default=0.95)
    parser.add_argument("--target-peg-xy", type=float, nargs=2, default=DEFAULT_TARGET_PEG_XY)
    parser.add_argument("--min-peg-xy-progress", type=float, default=0.02)
    parser.add_argument("--min-nut-displacement", type=float, default=0.02)
    parser.add_argument("--min-nut-z-gain", type=float, default=-0.01)
    parser.add_argument("--peg-progress-weight", type=float, default=2.0)
    parser.add_argument("--z-gain-weight", type=float, default=1.0)
    parser.add_argument("--nut-displacement-weight", type=float, default=0.25)
    parser.add_argument("--eef-approach-weight", type=float, default=0.0)
    parser.add_argument("--progress-risk-penalty", type=float, default=0.5)
    parser.add_argument(
        "--use-stored-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in ("source", "predictions", "output"):
        setattr(args, key, getattr(args, key).resolve())
    build(args)


if __name__ == "__main__":
    main()
