#!/usr/bin/env python3
"""Build exact, privileged-filtered positive DP chunks from failed rollouts.

Unlike the earlier segment dataset, every output demo represents exactly one
policy-boundary training example:

* one preceding observation for the two-frame RGB history;
* sixteen real (never padded) future actions;
* privileged validation over the full sixteen-action target.

This prevents short-segment tail padding from turning the final action into a
large fraction of the diffusion target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from build_rgb_dp_critical_failure_chunks import (
    decode,
    exact_grasp_mask,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_good_chunks_fixed_window.hdf5"
)


def copy_fixed_window(
    source: h5py.Group,
    destination: h5py.Group,
    boundary: int,
    prediction_horizon: int,
) -> None:
    """Copy context plus target; duplicate reset observation at boundary zero."""
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


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return list(
        zip(
            np.flatnonzero(changes == 1).tolist(),
            np.flatnonzero(changes == -1).tolist(),
        )
    )


def evaluate_boundary(
    *,
    boundary: int,
    prediction_horizon: int,
    actions: np.ndarray,
    object_obs: np.ndarray,
    grasped: np.ndarray,
    close_command: float,
    safe_min_start_distance: float,
    safe_min_end_distance: float,
    safe_min_reach_gain: float,
    safe_min_progress_fraction: float,
    safe_max_regression: float,
    safe_max_cube_displacement: float,
    grasp_min_frames: int,
    grasp_min_lift_gain: float,
    grasp_max_drop: float,
) -> dict | None:
    end = boundary + prediction_horizon
    # State observations are pre-action. State[end] validates the result of
    # action end-1, so do not use a chunk ending at the final stored action.
    if end >= len(actions):
        return None

    states = slice(boundary, end + 1)
    target = slice(boundary, end)
    distance = np.linalg.norm(object_obs[states, -3:], axis=1)
    cube_pos = object_obs[states, :3]
    cube_z = cube_pos[:, 2]
    local_grasp = grasped[states]
    reach_deltas = np.diff(distance)
    close_steps = int(np.count_nonzero(actions[target, -1] >= close_command))

    reach_gain = float(distance[0] - distance[-1])
    progress_fraction = float(np.mean(reach_deltas <= 0.0))
    max_regression = float(np.max(reach_deltas))
    cube_displacement = float(
        np.max(np.linalg.norm(cube_pos - cube_pos[0], axis=1))
    )

    # A clean approach must remain outside contact risk for all sixteen actions.
    if (
        float(distance[0]) >= safe_min_start_distance
        and float(np.min(distance)) >= safe_min_end_distance
        and reach_gain >= safe_min_reach_gain
        and progress_fraction >= safe_min_progress_fraction
        and max_regression <= safe_max_regression
        and cube_displacement <= safe_max_cube_displacement
        and close_steps == 0
        and not np.any(local_grasp)
    ):
        return {
            "type": "safe_reach",
            "decision_boundary": boundary,
            "target_end_exclusive": end,
            "start_distance": float(distance[0]),
            "end_distance": float(distance[-1]),
            "min_distance": float(np.min(distance)),
            "reach_gain": reach_gain,
            "progress_fraction": progress_fraction,
            "max_regression": max_regression,
            "cube_displacement": cube_displacement,
            "close_steps": close_steps,
            "score": reach_gain,
        }

    # A useful manipulation prefix may occur in an episode that eventually
    # fails. Require the grasp to persist through the end of this target and
    # measurable lift without a drop inside the chunk.
    grasp_runs = true_runs(local_grasp)
    final_run = next((run for run in grasp_runs if run[1] == len(local_grasp)), None)
    if final_run is not None:
        grasp_start, grasp_end = final_run
        grasp_frames = grasp_end - grasp_start
        lift_gain = float(cube_z[-1] - cube_z[grasp_start])
        drop = float(np.max(cube_z[grasp_start:]) - cube_z[-1])
        if (
            grasp_frames >= grasp_min_frames
            and lift_gain >= grasp_min_lift_gain
            and drop <= grasp_max_drop
        ):
            return {
                "type": "grasp_lift",
                "decision_boundary": boundary,
                "target_end_exclusive": end,
                "grasp_start_in_window": grasp_start,
                "persistent_grasp_frames": grasp_frames,
                "lift_gain": lift_gain,
                "drop": drop,
                "start_distance": float(distance[0]),
                "end_distance": float(distance[-1]),
                "score": lift_gain,
            }
    return None


def select_chunks(args, actions, object_obs, grasped) -> list[dict]:
    candidates = []
    full_distance = np.linalg.norm(object_obs[:, -3:], axis=1)
    contact_risk = (
        (full_distance <= args.first_risk_distance)
        | grasped
        | (
            (actions[:, -1] >= args.close_command)
            & (full_distance <= args.close_near_distance)
        )
    )
    risk_indices = np.flatnonzero(contact_risk)
    first_risk_step = int(risk_indices[0]) if len(risk_indices) else len(actions)
    latest_boundary = len(actions) - args.prediction_horizon - 1
    for boundary in range(0, latest_boundary + 1, args.action_horizon):
        candidate = evaluate_boundary(
            boundary=boundary,
            prediction_horizon=args.prediction_horizon,
            actions=actions,
            object_obs=object_obs,
            grasped=grasped,
            close_command=args.close_command,
            safe_min_start_distance=args.safe_min_start_distance,
            safe_min_end_distance=args.safe_min_end_distance,
            safe_min_reach_gain=args.safe_min_reach_gain,
            safe_min_progress_fraction=args.safe_min_progress_fraction,
            safe_max_regression=args.safe_max_regression,
            safe_max_cube_displacement=args.safe_max_cube_displacement,
            grasp_min_frames=args.grasp_min_frames,
            grasp_min_lift_gain=args.grasp_min_lift_gain,
            grasp_max_drop=args.grasp_max_drop,
        )
        if candidate is not None:
            if (
                candidate["type"] == "safe_reach"
                and candidate["target_end_exclusive"] > first_risk_step
            ):
                continue
            candidate["first_risk_step"] = first_risk_step
            candidates.append(candidate)

    selected = []
    for stage in ("safe_reach", "grasp_lift"):
        stage_candidates = [x for x in candidates if x["type"] == stage]
        stage_candidates.sort(key=lambda x: x["score"], reverse=True)
        selected.extend(stage_candidates[: args.max_chunks_per_stage_per_rollout])
    selected.sort(key=lambda x: x["decision_boundary"])
    return selected


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
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        output_data = output.create_group("data")
        output_keys = []
        failure_keys = decode(source["mask/failure"][:])

        for source_index, source_key in enumerate(failure_keys):
            source_group = source[f"data/{source_key}"]
            states = source_group["states"][:]
            grasped = exact_grasp_mask(
                env=env,
                model_file=source_group.attrs["model_file"],
                states=states,
            )
            chunks = select_chunks(
                args,
                actions=source_group["actions"][:],
                object_obs=source_group["obs/object"][:],
                grasped=grasped,
            )
            for chunk in chunks:
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                copy_fixed_window(
                    source=source_group,
                    destination=output_group,
                    boundary=chunk["decision_boundary"],
                    prediction_horizon=args.prediction_horizon,
                )
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = args.prediction_horizon + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = chunk[
                    "decision_boundary"
                ]
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = chunk["type"]
                output_group.attrs["segment_metrics_json"] = json.dumps(chunk)
                output_keys.append(output_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        **chunk,
                    }
                )
            if (source_index + 1) % 10 == 0:
                print(
                    f"processed {source_index + 1}/{len(failure_keys)} failures; "
                    f"retained {len(records)} fixed chunks",
                    flush=True,
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (
            args.prediction_horizon + 1
        )
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    stage_counts = {
        stage: sum(record["type"] == stage for record in records)
        for stage in ("safe_reach", "grasp_lift")
    }
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "source_failure_rollouts": len(failure_keys),
        "retained_chunks": len(records),
        "retained_source_rollouts": len({x["source_demo"] for x in records}),
        "stage_counts": stage_counts,
        "stored_frames_per_chunk": args.prediction_horizon + 1,
        "target_actions_per_chunk": args.prediction_horizon,
        "contains_padded_target_actions": False,
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if key not in ("source", "output", "overwrite")
        },
        "chunks": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=4))
    print(json.dumps({k: v for k, v in summary.items() if k != "chunks"}, indent=4))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--close-command", type=float, default=0.25)
    parser.add_argument("--first-risk-distance", type=float, default=0.06)
    parser.add_argument("--close-near-distance", type=float, default=0.08)
    parser.add_argument("--safe-min-start-distance", type=float, default=0.08)
    parser.add_argument("--safe-min-end-distance", type=float, default=0.05)
    parser.add_argument("--safe-min-reach-gain", type=float, default=0.04)
    parser.add_argument("--safe-min-progress-fraction", type=float, default=0.8)
    parser.add_argument("--safe-max-regression", type=float, default=0.01)
    parser.add_argument("--safe-max-cube-displacement", type=float, default=0.004)
    parser.add_argument("--grasp-min-frames", type=int, default=5)
    parser.add_argument("--grasp-min-lift-gain", type=float, default=0.005)
    parser.add_argument("--grasp-max-drop", type=float, default=0.003)
    parser.add_argument("--max-chunks-per-stage-per-rollout", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    build(args)


if __name__ == "__main__":
    main()
