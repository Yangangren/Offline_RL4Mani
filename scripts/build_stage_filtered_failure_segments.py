#!/usr/bin/env python3
"""Build high-confidence segments from failed robomimic Lift rollouts.

The source failure trajectories are weakly labeled only at episode level. This
builder uses privileged simulator state to retain two physically meaningful
segment types:

1. ``safe_reach``: a monotonic approach that ends before close-contact actions.
2. ``grasp_lift``: a persistent simulator-confirmed grasp that raises the cube.

Everything after an unsuccessful contact, drop, retreat, or prolonged wandering
is excluded. The output is a regular robomimic HDF5 where every retained
contiguous segment is represented as a short demonstration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "rollouts/lift_bc_epoch50_rollouts_500_lowdim.hdf5"
DEFAULT_OUTPUT = ROOT / "rollouts/lift_bc_epoch50_failure_segments_stage_filtered.hdf5"


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[start, end)`` runs of true values."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def copy_time_slice(
    source: h5py.Group,
    destination: h5py.Group,
    start: int,
    end: int,
) -> None:
    for key, item in source.items():
        if isinstance(item, h5py.Group):
            child = destination.create_group(key)
            copy_time_slice(item, child, start, end)
        else:
            destination.create_dataset(key, data=item[start:end])


def exact_grasp_mask(env, model_file: str, states: np.ndarray) -> np.ndarray:
    """Query robosuite contact-based grasp state without recomputing observations."""
    env.reset_to({"model": model_file})
    base_env = env.env
    grasped = np.zeros(len(states), dtype=bool)
    for index, state in enumerate(states):
        base_env.sim.set_state_from_flattened(state)
        base_env.sim.forward()
        grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[0].gripper,
                object_geoms=base_env.cube,
            )
        )
    return grasped


def select_segments(
    distance: np.ndarray,
    cube_z: np.ndarray,
    grasped: np.ndarray,
    close_distance: float,
    reach_context: int,
    min_segment_len: int,
    min_reach_gain: float,
    min_progress_fraction: float,
    max_reach_regression: float,
    grasp_context: int,
    min_grasp_frames: int,
    min_lift_gain: float,
) -> list[dict]:
    segments = []
    trajectory_length = len(distance)

    # Keep actions that produce a clean approach, but stop at the first state
    # inside the contact-risk region. Thus the action at that close state is
    # deliberately not copied.
    close_indices = np.flatnonzero(distance <= close_distance)
    if len(close_indices):
        end = int(close_indices[0])
        start = max(0, end - reach_context)
        deltas = np.diff(distance[start : end + 1])
        reach_gain = float(distance[start] - distance[end])
        progress_fraction = float(np.mean(deltas <= 0.0)) if len(deltas) else 0.0
        max_regression = float(np.max(deltas)) if len(deltas) else float("inf")
        if (
            end - start >= min_segment_len
            and reach_gain >= min_reach_gain
            and progress_fraction >= min_progress_fraction
            and max_regression <= max_reach_regression
        ):
            segments.append(
                {
                    "type": "safe_reach",
                    "start": start,
                    "end": end,
                    "reach_gain": reach_gain,
                    "progress_fraction": progress_fraction,
                    "max_reach_regression": max_regression,
                    "end_distance": float(distance[end]),
                }
            )

    # A brief collision can move the cube upward without constituting a useful
    # manipulation action. Require exact bilateral grasp contact for several
    # consecutive states and measurable lift while that grasp persists.
    for grasp_start, grasp_end in true_runs(grasped):
        if grasp_end - grasp_start < min_grasp_frames:
            continue
        local_peak = grasp_start + int(np.argmax(cube_z[grasp_start:grasp_end]))
        lift_gain = float(cube_z[local_peak] - cube_z[grasp_start])
        if lift_gain < min_lift_gain:
            continue
        start = max(0, grasp_start - grasp_context)
        end = min(trajectory_length, local_peak + 1)
        if end - start < min_segment_len:
            continue
        segments.append(
            {
                "type": "grasp_lift",
                "start": start,
                "end": end,
                "grasp_start": grasp_start,
                "grasp_end": grasp_end,
                "persistent_grasp_frames": grasp_end - grasp_start,
                "lift_gain": lift_gain,
                "peak_cube_z": float(cube_z[local_peak]),
            }
        )
    return segments


def build(args) -> dict:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(args.source))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
    )

    records = []
    source_failure_steps = 0
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        output_data = output.create_group("data")
        output_keys = []
        failure_keys = decode(source["mask/failure"][:])

        for source_index, source_key in enumerate(failure_keys):
            source_group = source[f"data/{source_key}"]
            states = source_group["states"][:]
            objects = source_group["obs/object"][:]
            distance = np.linalg.norm(objects[:, -3:], axis=1)
            cube_z = objects[:, 2]
            grasped = exact_grasp_mask(
                env=env,
                model_file=source_group.attrs["model_file"],
                states=states,
            )
            source_failure_steps += len(states)
            segments = select_segments(
                distance=distance,
                cube_z=cube_z,
                grasped=grasped,
                close_distance=args.close_distance,
                reach_context=args.reach_context,
                min_segment_len=args.min_segment_len,
                min_reach_gain=args.min_reach_gain,
                min_progress_fraction=args.min_progress_fraction,
                max_reach_regression=args.max_reach_regression,
                grasp_context=args.grasp_context,
                min_grasp_frames=args.min_grasp_frames,
                min_lift_gain=args.min_lift_gain,
            )

            for segment in segments:
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                start, end = segment["start"], segment["end"]
                copy_time_slice(source_group, output_group, start, end)
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = end - start
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_start"] = start
                output_group.attrs["source_end_exclusive"] = end
                output_group.attrs["segment_type"] = segment["type"]
                output_group.attrs["segment_metrics_json"] = json.dumps(segment)
                output_keys.append(output_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        **segment,
                        "num_samples": end - start,
                    }
                )

            if (source_index + 1) % 25 == 0:
                print(
                    f"processed {source_index + 1}/{len(failure_keys)} "
                    f"failure rollouts; retained {len(records)} segments",
                    flush=True,
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = sum(record["num_samples"] for record in records)
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    type_counts = {
        segment_type: sum(record["type"] == segment_type for record in records)
        for segment_type in ("safe_reach", "grasp_lift")
    }
    type_samples = {
        segment_type: sum(
            record["num_samples"]
            for record in records
            if record["type"] == segment_type
        )
        for segment_type in ("safe_reach", "grasp_lift")
    }
    retained_samples = sum(record["num_samples"] for record in records)
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "source_failure_rollouts": len(failure_keys),
        "source_failure_samples": source_failure_steps,
        "retained_segments": len(records),
        "retained_source_rollouts": len({record["source_demo"] for record in records}),
        "retained_samples": retained_samples,
        "retained_fraction_of_failure_samples": retained_samples
        / max(1, source_failure_steps),
        "segment_counts": type_counts,
        "segment_samples": type_samples,
        "thresholds": {
            "close_distance": args.close_distance,
            "reach_context": args.reach_context,
            "min_segment_len": args.min_segment_len,
            "min_reach_gain": args.min_reach_gain,
            "min_progress_fraction": args.min_progress_fraction,
            "max_reach_regression": args.max_reach_regression,
            "grasp_context": args.grasp_context,
            "min_grasp_frames": args.min_grasp_frames,
            "min_lift_gain": args.min_lift_gain,
        },
        "segments": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=4))
    print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=4))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--close-distance", type=float, default=0.06)
    parser.add_argument("--reach-context", type=int, default=24)
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--min-reach-gain", type=float, default=0.05)
    parser.add_argument("--min-progress-fraction", type=float, default=0.8)
    parser.add_argument("--max-reach-regression", type=float, default=0.01)
    parser.add_argument("--grasp-context", type=int, default=8)
    parser.add_argument("--min-grasp-frames", type=int, default=5)
    parser.add_argument("--min-lift-gain", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    build(args)


if __name__ == "__main__":
    main()
