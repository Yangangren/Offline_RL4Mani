#!/usr/bin/env python3
"""Extract high-confidence failed manipulation chunks from RGB-DP rollouts.

The rollout has only an episode-level failure label. For this simulator-side
feasibility test, privileged state is used to find policy decision boundaries
whose 16-step action target closes the gripper near the cube but never obtains
an exact robosuite grasp. Safe approach prefixes are deliberately excluded.

Each output demonstration contains one preceding context frame followed by the
16-step negative target. Configure SequenceDataset with ``demo_start_only`` and
``sample_start_offset=1`` to recover the original two-frame observation history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_critical_failure_chunks.hdf5"
)


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


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


def true_run_starts(mask: np.ndarray) -> np.ndarray:
    previous = np.pad(mask[:-1], (1, 0), constant_values=False)
    return np.flatnonzero(mask & ~previous)


def select_failed_grasp_chunks(
    *,
    actions: np.ndarray,
    object_obs: np.ndarray,
    gripper_qpos: np.ndarray,
    grasped: np.ndarray,
    action_horizon: int,
    prediction_horizon: int,
    close_command: float,
    near_distance: float,
    required_near_distance: float,
    min_close_steps: int,
    min_aperture_reduction: float,
    max_lift_gain: float,
    max_chunks_per_rollout: int,
) -> list[dict]:
    distance = np.linalg.norm(object_obs[:, -3:], axis=1)
    cube_z = object_obs[:, 2]
    aperture = np.abs(gripper_qpos[:, 0] - gripper_qpos[:, 1])
    close_near = (actions[:, -1] >= close_command) & (distance <= near_distance)

    selected = []
    used_boundaries = set()
    for event in true_run_starts(close_near):
        boundary = int(event // action_horizon * action_horizon)
        end = boundary + prediction_horizon
        # One previous observation is copied to preserve the original history.
        if boundary < 1 or end > len(actions) or boundary in used_boundaries:
            continue

        chunk_close = actions[boundary:end, -1] >= close_command
        chunk_distance = distance[boundary:end]
        state_end = min(len(grasped), end + 1)
        chunk_grasp = grasped[boundary:state_end]
        aperture_reduction = float(
            aperture[boundary] - np.min(aperture[boundary:state_end])
        )
        lift_gain = float(
            np.max(cube_z[boundary:state_end]) - cube_z[boundary]
        )
        if np.count_nonzero(chunk_close) < min_close_steps:
            continue
        if float(np.min(chunk_distance)) > required_near_distance:
            continue
        if np.any(chunk_grasp):
            continue
        if aperture_reduction < min_aperture_reduction:
            continue
        if lift_gain > max_lift_gain:
            continue

        selected.append(
            {
                "type": "failed_grasp",
                "event_step": int(event),
                "decision_boundary": boundary,
                "target_end_exclusive": end,
                "min_distance": float(np.min(chunk_distance)),
                "close_steps": int(np.count_nonzero(chunk_close)),
                "aperture_reduction": aperture_reduction,
                "lift_gain": lift_gain,
                "grasp_frames": int(np.count_nonzero(chunk_grasp)),
            }
        )
        used_boundaries.add(boundary)
        if len(selected) >= max_chunks_per_rollout:
            break
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
    source_failure_steps = 0
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        output_data = output.create_group("data")
        output_keys = []
        failure_keys = decode(source["mask/failure"][:])

        for source_index, source_key in enumerate(failure_keys):
            source_group = source[f"data/{source_key}"]
            states = source_group["states"][:]
            actions = source_group["actions"][:]
            object_obs = source_group["obs/object"][:]
            gripper_qpos = source_group["obs/robot0_gripper_qpos"][:]
            grasped = exact_grasp_mask(
                env=env,
                model_file=source_group.attrs["model_file"],
                states=states,
            )
            source_failure_steps += len(actions)
            chunks = select_failed_grasp_chunks(
                actions=actions,
                object_obs=object_obs,
                gripper_qpos=gripper_qpos,
                grasped=grasped,
                action_horizon=args.action_horizon,
                prediction_horizon=args.prediction_horizon,
                close_command=args.close_command,
                near_distance=args.near_distance,
                required_near_distance=args.required_near_distance,
                min_close_steps=args.min_close_steps,
                min_aperture_reduction=args.min_aperture_reduction,
                max_lift_gain=args.max_lift_gain,
                max_chunks_per_rollout=args.max_chunks_per_rollout,
            )

            for chunk in chunks:
                output_key = f"demo_{len(output_keys)}"
                output_group = output_data.create_group(output_key)
                target_start = chunk["decision_boundary"]
                copy_start = target_start - 1
                copy_end = chunk["target_end_exclusive"]
                copy_time_slice(source_group, output_group, copy_start, copy_end)
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = copy_end - copy_start
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = target_start
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
                    f"retained {len(records)} chunks",
                    flush=True,
                )

        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = sum(
            int(output_data[key].attrs["num_samples"]) for key in output_keys
        )
        output_mask = output.create_group("mask")
        output_mask["all"] = np.asarray(output_keys, dtype="S")

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "source_failure_rollouts": len(failure_keys),
        "source_failure_samples": source_failure_steps,
        "retained_chunks": len(records),
        "retained_source_rollouts": len({record["source_demo"] for record in records}),
        "target_actions_per_chunk": args.prediction_horizon,
        "stored_frames_per_chunk": args.prediction_horizon + 1,
        "thresholds": {
            "action_horizon": args.action_horizon,
            "prediction_horizon": args.prediction_horizon,
            "close_command": args.close_command,
            "near_distance": args.near_distance,
            "required_near_distance": args.required_near_distance,
            "min_close_steps": args.min_close_steps,
            "min_aperture_reduction": args.min_aperture_reduction,
            "max_lift_gain": args.max_lift_gain,
            "max_chunks_per_rollout": args.max_chunks_per_rollout,
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
    parser.add_argument("--near-distance", type=float, default=0.05)
    parser.add_argument("--required-near-distance", type=float, default=0.035)
    parser.add_argument("--min-close-steps", type=int, default=2)
    parser.add_argument("--min-aperture-reduction", type=float, default=0.01)
    parser.add_argument("--max-lift-gain", type=float, default=0.008)
    parser.add_argument("--max-chunks-per-rollout", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    build(args)


if __name__ == "__main__":
    main()
