#!/usr/bin/env python3
"""Build privileged-GT good action chunks from failed RGB-DP rollouts.

For each task, a compact task-state feature is extracted from the privileged
``obs/object`` vector. Successful rollout endpoints define the task goal in
that feature space. A failure chunk is retained when it moves toward that goal
by a configurable margin and changes the task state by a minimum amount.

The output layout matches the fixed-window failure datasets consumed by
``train_rgb_dp_mixed_imitation.py``: one context frame followed by a complete
Diffusion Policy prediction horizon, trained with ``demo_start_only=True`` and
``sample_start_offset=1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from build_square_rgb_dp_low_risk_failure_chunks import (
    audit_source_match,
    clip_action_dataset,
    copy_fixed_window,
    decode,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection"
    / "square_rgb_dp_gt_good_failure_chunks.hdf5"
)
TASKS = ("square", "can", "transport", "tool_hang")


def demo_sort_key(key: str) -> int:
    return int(key.rsplit("_", 1)[-1])


def read_mask(handle: h5py.File, key: str) -> list[str]:
    path = f"mask/{key}"
    if path not in handle:
        available = sorted(handle.get("mask", {}).keys())
        raise KeyError(f"{handle.filename} has no {path}; available={available}")
    return sorted(decode(handle[path][:]), key=demo_sort_key)


def task_feature(task: str, object_state: np.ndarray) -> np.ndarray:
    """Extract task-relevant privileged coordinates from one or more states."""
    values = np.asarray(object_state, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"object state must be rank 1 or 2, got {values.shape}")

    expected_dims = {
        "square": 14,
        "can": 14,
        "transport": 41,
        "tool_hang": 44,
    }
    expected = expected_dims[task]
    if values.shape[1] != expected:
        raise ValueError(
            f"task={task} expects object-state dim {expected}, got {values.shape[1]}"
        )

    if task in ("square", "can"):
        # Both robosuite tasks concatenate gripper-relative object pose first,
        # followed by absolute object position at indices [7:10].
        feature = values[:, 7:10]
    elif task == "transport":
        # Use goal-relative payload / trash positions and the two privileged
        # completion-stage bits. Goal-relative features handle randomized bins.
        payload_to_target = values[:, 0:3] - values[:, 21:24]
        trash_to_bin = values[:, 7:10] - values[:, 24:27]
        stages = values[:, 27:29]
        feature = np.concatenate((payload_to_target, trash_to_bin, stages), axis=1)
    elif task == "tool_hang":
        # Absolute base, frame, and tool positions appear after their respective
        # gripper-relative poses; the final two values are assembly stage bits.
        frame_to_base = values[:, 21:24] - values[:, 7:10]
        tool_to_frame = values[:, 35:38] - values[:, 21:24]
        stages = values[:, 42:44]
        feature = np.concatenate((frame_to_base, tool_to_frame, stages), axis=1)
    else:
        raise ValueError(f"unsupported task={task}")
    return feature[0] if squeeze else feature


def feature_scale_floor(task: str, position_floor: float) -> np.ndarray:
    if task in ("square", "can"):
        return np.full(3, float(position_floor), dtype=np.float64)
    # Six geometric coordinates followed by two binary task-stage features.
    return np.concatenate(
        (
            np.full(6, float(position_floor), dtype=np.float64),
            np.ones(2, dtype=np.float64),
        )
    )


def successful_endpoint_features(
    source: h5py.File,
    task: str,
    success_keys: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    endpoints = []
    records = []
    for key in success_keys:
        group = source[f"data/{key}"]
        rewards = np.asarray(group["rewards"][:], dtype=np.float64).reshape(-1)
        successes = np.flatnonzero(rewards > 0.0)
        endpoint = int(successes[0]) if len(successes) else int(len(rewards) - 1)
        feature = task_feature(task, group["obs/object"][endpoint])
        endpoints.append(feature)
        records.append(
            {
                "demo": key,
                "endpoint": endpoint,
                "sparse_success_observed": bool(len(successes)),
            }
        )
    if not endpoints:
        raise RuntimeError("success mask is empty; cannot define the GT task goal")
    return np.stack(endpoints), records


def goal_statistics(
    task: str,
    endpoint_features: np.ndarray,
    position_scale_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.median(endpoint_features, axis=0)
    median_absolute_deviation = np.median(
        np.abs(endpoint_features - reference[None]),
        axis=0,
    )
    robust_scale = 1.4826 * median_absolute_deviation
    scale = np.maximum(
        robust_scale,
        feature_scale_floor(task, position_scale_floor),
    )
    return reference, scale


def normalized_distance(
    features: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (features - reference[None]) / scale[None]
    return np.sqrt(np.mean(np.square(standardized), axis=1))


def candidate_steps(num_samples: int, args: argparse.Namespace) -> np.ndarray:
    latest = int(num_samples) - int(args.prediction_horizon)
    if latest < 0:
        return np.zeros(0, dtype=np.int64)
    lower = max(0, int(args.min_start_step))
    upper = latest
    if args.max_start_step is not None:
        upper = min(upper, int(args.max_start_step))
    if args.max_start_fraction is not None:
        upper = min(
            upper,
            int(np.floor(float(args.max_start_fraction) * num_samples)),
        )
    if upper < lower:
        return np.zeros(0, dtype=np.int64)
    return np.arange(lower, upper + 1, int(args.stride), dtype=np.int64)


def episode_progress(
    *,
    task: str,
    group: h5py.Group,
    steps: np.ndarray,
    prediction_horizon: int,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    end_steps = np.minimum(
        steps + int(prediction_horizon) - 1,
        int(group["actions"].shape[0]) - 1,
    )
    object_states = group["obs/object"]
    start_features = task_feature(task, np.asarray(object_states[steps]))
    end_features = task_feature(task, np.asarray(object_states[end_steps]))
    start_distance = normalized_distance(start_features, reference, scale)
    end_distance = normalized_distance(end_features, reference, scale)
    normalized_displacement = np.sqrt(
        np.mean(
            np.square((end_features - start_features) / scale[None]),
            axis=1,
        )
    )
    return {
        "end_steps": end_steps,
        "start_distance": start_distance,
        "end_distance": end_distance,
        "goal_progress": start_distance - end_distance,
        "normalized_displacement": normalized_displacement,
    }


def choose_chunks(
    *,
    steps: np.ndarray,
    metrics: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[int]:
    eligible = np.flatnonzero(
        (metrics["goal_progress"] >= float(args.min_goal_progress))
        & (
            metrics["normalized_displacement"]
            >= float(args.min_normalized_displacement)
        )
    )
    if len(eligible) == 0:
        return []
    score = (
        float(args.goal_progress_weight) * metrics["goal_progress"]
        + float(args.displacement_weight) * metrics["normalized_displacement"]
    )
    if args.prefer == "progress":
        order = eligible[np.lexsort((steps[eligible], -score[eligible]))]
    elif args.prefer == "earliest":
        order = eligible[np.argsort(steps[eligible])]
    elif args.prefer == "latest":
        order = eligible[np.argsort(-steps[eligible])]
    else:
        raise ValueError(f"unknown prefer={args.prefer}")

    selected: list[int] = []
    for index in order:
        step = int(steps[index])
        if all(
            abs(step - previous) >= int(args.minimum_spacing)
            for previous in selected
        ):
            selected.append(step)
        if len(selected) >= int(args.max_chunks_per_failure):
            break
    return sorted(selected)


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    retained_source_rollouts: set[str] = set()
    source_failure_samples = 0
    endpoint_records: list[dict[str, Any]]
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        success_keys = read_mask(source, args.success_mask)
        failure_keys = read_mask(source, args.failure_mask)
        overlap = sorted(set(success_keys).intersection(failure_keys), key=demo_sort_key)
        if overlap:
            raise ValueError(f"success and failure masks overlap: {overlap[:10]}")

        endpoint_features, endpoint_records = successful_endpoint_features(
            source,
            args.task,
            success_keys,
        )
        reference, scale = goal_statistics(
            args.task,
            endpoint_features,
            args.position_scale_floor,
        )

        output_data = output.create_group("data")
        output_keys: list[str] = []
        for failure_index, source_key in enumerate(failure_keys):
            source_group = source[f"data/{source_key}"]
            num_samples = int(source_group.attrs["num_samples"])
            source_failure_samples += num_samples
            steps = candidate_steps(num_samples, args)
            if len(steps) == 0:
                continue
            metrics = episode_progress(
                task=args.task,
                group=source_group,
                steps=steps,
                prediction_horizon=args.prediction_horizon,
                reference=reference,
                scale=scale,
            )
            selected = choose_chunks(steps=steps, metrics=metrics, args=args)
            step_to_index = {int(step): index for index, step in enumerate(steps)}
            for boundary in selected:
                local_index = step_to_index[int(boundary)]
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    int(boundary),
                    int(args.prediction_horizon),
                )
                clip_action_dataset(output_group)
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = int(args.prediction_horizon) + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = int(boundary)
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = f"{args.task}_gt_good_failure_chunk"
                output_group.attrs["privileged_goal_progress"] = float(
                    metrics["goal_progress"][local_index]
                )
                output_group.attrs["privileged_normalized_displacement"] = float(
                    metrics["normalized_displacement"][local_index]
                )
                output_group.attrs["privileged_goal_distance_start"] = float(
                    metrics["start_distance"][local_index]
                )
                output_group.attrs["privileged_goal_distance_end"] = float(
                    metrics["end_distance"][local_index]
                )

                output_keys.append(output_key)
                retained_source_rollouts.add(source_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": int(boundary),
                        "target_end_exclusive": int(
                            boundary + args.prediction_horizon
                        ),
                        "source_num_samples": num_samples,
                        "privileged_goal_progress": float(
                            metrics["goal_progress"][local_index]
                        ),
                        "privileged_normalized_displacement": float(
                            metrics["normalized_displacement"][local_index]
                        ),
                        "privileged_goal_distance_start": float(
                            metrics["start_distance"][local_index]
                        ),
                        "privileged_goal_distance_end": float(
                            metrics["end_distance"][local_index]
                        ),
                    }
                )
            if (failure_index + 1) % 25 == 0:
                print(
                    f"processed {failure_index + 1}/{len(failure_keys)} failures; "
                    f"retained {len(records)} chunks",
                    flush=True,
                )

        if not records:
            raise RuntimeError(
                "GT-good failure filter retained no chunks; lower "
                "--min-goal-progress or --min-normalized-displacement"
            )
        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (
            int(args.prediction_horizon) + 1
        )
        mask = output.create_group("mask")
        encoded_keys = np.asarray(output_keys, dtype="S")
        mask["all"] = encoded_keys
        mask["gt_good_failure"] = encoded_keys
        output.attrs["task"] = args.task
        output.attrs["selection_definition"] = (
            "privileged object-state movement toward successful terminal task states"
        )
        output.attrs["source_success_mask"] = args.success_mask
        output.attrs["source_failure_mask"] = args.failure_mask
        audit_source_match(source, output, records, int(args.prediction_horizon))

    progress = np.asarray(
        [record["privileged_goal_progress"] for record in records],
        dtype=np.float64,
    )
    displacement = np.asarray(
        [record["privileged_normalized_displacement"] for record in records],
        dtype=np.float64,
    )
    starts = np.asarray(
        [record["decision_boundary"] for record in records],
        dtype=np.int64,
    )
    summary = {
        "task": args.task,
        "source": str(args.source),
        "output": str(args.output),
        "success_mask": args.success_mask,
        "success_reference_rollouts": len(endpoint_records),
        "success_reference_endpoints": endpoint_records,
        "failure_mask": args.failure_mask,
        "source_failure_rollouts": len(failure_keys),
        "source_failure_samples": source_failure_samples,
        "retained_chunks": len(records),
        "retained_source_rollouts": len(retained_source_rollouts),
        "target_actions_per_chunk": int(args.prediction_horizon),
        "stored_frames_per_chunk": int(args.prediction_horizon) + 1,
        "contains_padded_target_actions": False,
        "training_dataset_hints": {
            "filter_key": "gt_good_failure",
            "demo_start_only": True,
            "sample_start_offset": 1,
            "treat_as_positive_imitation": True,
        },
        "privileged_feature": {
            "definition": (
                "task-specific object position / goal-relative coordinates and stage bits"
            ),
            "successful_endpoint_reference": reference.tolist(),
            "robust_scale": scale.tolist(),
            "position_scale_floor": float(args.position_scale_floor),
        },
        "selection": {
            "prefer": args.prefer,
            "stride": int(args.stride),
            "max_chunks_per_failure": int(args.max_chunks_per_failure),
            "minimum_spacing": int(args.minimum_spacing),
            "prediction_horizon": int(args.prediction_horizon),
            "min_start_step": int(args.min_start_step),
            "max_start_step": args.max_start_step,
            "max_start_fraction": args.max_start_fraction,
            "min_goal_progress": float(args.min_goal_progress),
            "min_normalized_displacement": float(
                args.min_normalized_displacement
            ),
            "goal_progress_weight": float(args.goal_progress_weight),
            "displacement_weight": float(args.displacement_weight),
        },
        "retained_chunk_stats": {
            "start_step": stats(starts),
            "goal_progress": stats(progress),
            "normalized_displacement": stats(displacement),
        },
        "chunks": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"chunks", "success_reference_endpoints"}
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="square")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--success-mask", default="success_100")
    parser.add_argument("--failure-mask", default="failure_50")
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-start-step", type=int, default=0)
    parser.add_argument("--max-start-step", type=int, default=None)
    parser.add_argument("--max-start-fraction", type=float, default=None)
    parser.add_argument("--max-chunks-per-failure", type=int, default=4)
    parser.add_argument("--minimum-spacing", type=int, default=8)
    parser.add_argument("--prefer", choices=("progress", "earliest", "latest"), default="progress")
    parser.add_argument("--min-goal-progress", type=float, default=0.05)
    parser.add_argument("--min-normalized-displacement", type=float, default=0.05)
    parser.add_argument("--goal-progress-weight", type=float, default=1.0)
    parser.add_argument("--displacement-weight", type=float, default=0.1)
    parser.add_argument("--position-scale-floor", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in ("source", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.prediction_horizon <= 0:
        parser.error("prediction-horizon must be positive")
    if args.stride <= 0:
        parser.error("stride must be positive")
    if args.max_chunks_per_failure <= 0:
        parser.error("max-chunks-per-failure must be positive")
    if args.minimum_spacing <= 0:
        parser.error("minimum-spacing must be positive")
    if args.position_scale_floor <= 0.0:
        parser.error("position-scale-floor must be positive")
    if args.max_start_fraction is not None and not 0.0 <= args.max_start_fraction <= 1.0:
        parser.error("max-start-fraction must be in [0, 1]")
    return args


if __name__ == "__main__":
    build(parse_args())
