#!/usr/bin/env python3
"""Build privileged-GT good action chunks from failed RGB-DP rollouts.

For each task, a compact task-state feature is extracted from the privileged
``obs/object`` vector. Successful rollout endpoints define the task goal in
that feature space. A failure chunk is retained when it moves toward that goal
by a configurable margin and changes the task state by a minimum amount.

The output layout matches the fixed-window failure datasets consumed by
``train_rgb_dp_mixed_imitation.py``: one preceding context row followed by a
complete Diffusion Policy prediction horizon. With ``demo_start_only=True`` and
``sample_start_offset=1``, robomimic returns the context row for frame stacking;
Diffusion Policy trains on stored action rows ``[0:prediction_horizon]`` and
executes from model index ``observation_horizon - 1``. Thus stored row 0 is the
previous-action alignment target and stored rows 1 onward are current/future
actions, matching the pretrained policy convention without padded loss targets.
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
GENERIC_FILTER_VERSION = "goal_endpoint_progress_v1"
CAN_FILTER_VERSION = "can_privileged_stage_v1"
CAN_STAGE_TYPES = ("safe_reach", "grasp_lift", "transport")
TRANSPORT_FILTER_VERSION = "transport_privileged_stage_v1"
TRANSPORT_STAGE_TYPES = (
    "coordinated_safe_reach",
    "lid_lift_clear",
    "trash_grasp_lift",
    "trash_transport",
    "trash_place",
    "payload_safe_reach",
    "payload_grasp_lift",
    "payload_transport",
)
TOOL_HANG_FILTER_VERSION = "tool_hang_privileged_stage_v1"
TOOL_HANG_STAGE_TYPES = (
    "frame_safe_reach",
    "frame_grasp_lift",
    "frame_transport",
    "frame_insert",
    "tool_safe_reach",
    "tool_grasp_lift",
    "tool_transport",
    "tool_hook_align",
)


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


def create_state_replay_env(source_path: Path):
    """Create a low-dimensional robosuite environment for privileged replay."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(source_path))
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
    )


def replay_can_privileged_signals(env, group: h5py.Group) -> dict[str, np.ndarray]:
    """Replay exact Can task stages for every stored pre-action state."""
    states = np.asarray(group["states"][:])
    env.reset_to({"model": group.attrs["model_file"]})
    base_env = env.env
    if base_env.__class__.__name__ != "PickPlaceCan":
        raise ValueError(
            "Can privileged replay expects PickPlaceCan, got "
            f"{base_env.__class__.__name__}"
        )

    can = base_env.objects[base_env.object_id]
    can_body_id = base_env.obj_body_id[can.name]
    target_xy = np.asarray(
        base_env.target_bin_placements[base_env.object_id, :2],
        dtype=np.float64,
    )

    count = len(states)
    stage_components = np.zeros((count, 4), dtype=np.float64)
    grasped = np.zeros(count, dtype=np.bool_)
    placed = np.zeros(count, dtype=np.bool_)
    eef_can_distance = np.zeros(count, dtype=np.float64)
    can_position = np.zeros((count, 3), dtype=np.float64)
    target_xy_distance = np.zeros(count, dtype=np.float64)

    for index, state in enumerate(states):
        base_env.sim.set_state_from_flattened(state)
        base_env.sim.forward()
        # Both methods use this mutable cache. Clear it at every independently
        # restored state so replay order cannot leak a previous placement label.
        base_env.objects_in_bins[:] = 0
        placed[index] = bool(base_env._check_success())
        stage_components[index] = np.asarray(
            base_env.staged_rewards(),
            dtype=np.float64,
        )
        grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[0].gripper,
                object_geoms=can.contact_geoms,
            )
        )
        position = np.asarray(
            base_env.sim.data.body_xpos[can_body_id],
            dtype=np.float64,
        )
        can_position[index] = position
        target_xy_distance[index] = np.linalg.norm(position[:2] - target_xy)
        eef_can_distance[index] = min(
            np.linalg.norm(
                np.asarray(
                    base_env.sim.data.site_xpos[
                        base_env.robots[0].eef_site_id[arm]
                    ],
                    dtype=np.float64,
                )
                - position
            )
            for arm in base_env.robots[0].arms
        )

    return {
        "stage_components": stage_components,
        "stage_progress": np.max(stage_components, axis=1),
        "grasped": grasped,
        "placed": placed,
        "eef_can_distance": eef_can_distance,
        "can_position": can_position,
        "can_z": can_position[:, 2],
        "target_xy_distance": target_xy_distance,
        "target_xy": target_xy,
        "initial_can_z": np.asarray(can_position[0, 2]),
    }


def replay_transport_privileged_signals(
    env,
    group: h5py.Group,
) -> dict[str, np.ndarray]:
    """Replay exact two-arm Transport contacts, grasps, and object poses."""
    states = np.asarray(group["states"][:])
    env.reset_to({"model": group.attrs["model_file"]})
    base_env = env.env
    if base_env.__class__.__name__ != "TwoArmTransport":
        raise ValueError(
            "Transport privileged replay expects TwoArmTransport, got "
            f"{base_env.__class__.__name__}"
        )

    transport = base_env.transport
    count = len(states)
    payload_position = np.zeros((count, 3), dtype=np.float64)
    trash_position = np.zeros((count, 3), dtype=np.float64)
    lid_position = np.zeros((count, 3), dtype=np.float64)
    target_bin_position = np.zeros((count, 3), dtype=np.float64)
    trash_bin_position = np.zeros((count, 3), dtype=np.float64)
    robot0_eef_position = np.zeros((count, 3), dtype=np.float64)
    robot1_eef_position = np.zeros((count, 3), dtype=np.float64)
    robot0_lid_grasped = np.zeros(count, dtype=np.bool_)
    robot0_payload_grasped = np.zeros(count, dtype=np.bool_)
    robot1_trash_grasped = np.zeros(count, dtype=np.bool_)
    payload_placed = np.zeros(count, dtype=np.bool_)
    trash_placed = np.zeros(count, dtype=np.bool_)

    for index, state in enumerate(states):
        base_env.sim.set_state_from_flattened(state)
        base_env.sim.forward()
        payload_position[index] = transport.payload_pos
        trash_position[index] = transport.trash_pos
        lid_position[index] = transport.lid_handle_pos
        target_bin_position[index] = transport.target_bin_pos
        trash_bin_position[index] = transport.trash_bin_pos
        robot0_eef_position[index] = base_env.sim.data.site_xpos[
            base_env.robots[0].eef_site_id[base_env.robots[0].arms[0]]
        ]
        robot1_eef_position[index] = base_env.sim.data.site_xpos[
            base_env.robots[1].eef_site_id[base_env.robots[1].arms[0]]
        ]
        robot0_lid_grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[0].gripper,
                object_geoms=transport.lid.handle_geoms,
            )
        )
        robot0_payload_grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[0].gripper,
                object_geoms=transport.payload.contact_geoms,
            )
        )
        robot1_trash_grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[1].gripper,
                object_geoms=transport.trash.contact_geoms,
            )
        )
        payload_placed[index] = bool(transport.payload_in_target_bin)
        trash_placed[index] = bool(transport.trash_in_trash_bin)

    payload_target_xy_distance = np.linalg.norm(
        payload_position[:, :2] - target_bin_position[:, :2],
        axis=1,
    )
    trash_target_xy_distance = np.linalg.norm(
        trash_position[:, :2] - trash_bin_position[:, :2],
        axis=1,
    )
    return {
        "payload_position": payload_position,
        "payload_z": payload_position[:, 2],
        "trash_position": trash_position,
        "trash_z": trash_position[:, 2],
        "lid_position": lid_position,
        "lid_z": lid_position[:, 2],
        "target_bin_position": target_bin_position,
        "trash_bin_position": trash_bin_position,
        "robot0_lid_distance": np.linalg.norm(
            robot0_eef_position - lid_position,
            axis=1,
        ),
        "robot0_payload_distance": np.linalg.norm(
            robot0_eef_position - payload_position,
            axis=1,
        ),
        "robot1_trash_distance": np.linalg.norm(
            robot1_eef_position - trash_position,
            axis=1,
        ),
        "robot0_lid_grasped": robot0_lid_grasped,
        "robot0_payload_grasped": robot0_payload_grasped,
        "robot1_trash_grasped": robot1_trash_grasped,
        "payload_placed": payload_placed,
        "trash_placed": trash_placed,
        "payload_target_xy_distance": payload_target_xy_distance,
        "trash_target_xy_distance": trash_target_xy_distance,
        "initial_payload_position": payload_position[0].copy(),
        "initial_trash_position": trash_position[0].copy(),
        "initial_lid_position": lid_position[0].copy(),
        "initial_payload_z": np.asarray(payload_position[0, 2]),
        "initial_trash_z": np.asarray(trash_position[0, 2]),
    }



def replay_tool_hang_privileged_signals(
    env,
    group: h5py.Group,
) -> dict[str, np.ndarray]:
    """Replay exact Tool Hang grasps, assembly predicates, and hook geometry."""
    states = np.asarray(group["states"][:])
    env.reset_to({"model": group.attrs["model_file"]})
    base_env = env.env
    if base_env.__class__.__name__ != "ToolHang":
        raise ValueError(
            "Tool Hang privileged replay expects ToolHang, got "
            f"{base_env.__class__.__name__}"
        )

    from robosuite.utils.sim_utils import check_contact

    count = len(states)
    frame_position = np.zeros((count, 3), dtype=np.float64)
    tool_position = np.zeros((count, 3), dtype=np.float64)
    eef_position = np.zeros((count, 3), dtype=np.float64)
    frame_mount_position = np.zeros((count, 3), dtype=np.float64)
    frame_tip_position = np.zeros((count, 3), dtype=np.float64)
    base_position = np.zeros((count, 3), dtype=np.float64)
    frame_hang_position = np.zeros((count, 3), dtype=np.float64)
    frame_intersection_position = np.zeros((count, 3), dtype=np.float64)
    tool_hole_position = np.zeros((count, 3), dtype=np.float64)
    frame_grasped = np.zeros(count, dtype=np.bool_)
    tool_grasped = np.zeros(count, dtype=np.bool_)
    frame_assembled = np.zeros(count, dtype=np.bool_)
    tool_on_frame = np.zeros(count, dtype=np.bool_)
    frame_between_walls = np.zeros(count, dtype=np.bool_)
    tool_between_hook = np.zeros(count, dtype=np.bool_)
    tool_hook_contact = np.zeros(count, dtype=np.bool_)
    tool_hook_orthogonal_distance = np.zeros(count, dtype=np.float64)
    tool_hook_normalized_axial_position = np.zeros(count, dtype=np.float64)
    hook_length = np.zeros(count, dtype=np.float64)

    frame_tip_site = base_env.obj_site_id.get(
        "frame_tip_site",
        base_env.obj_site_id["frame_mount_site"],
    )
    hole_geoms = [
        f"tool_hole1_hc_{index}"
        for index in range(base_env.tool_args["ngeoms"])
    ]
    opposite_hole_geom = base_env.tool_args["ngeoms"] // 2
    for index, state in enumerate(states):
        base_env.sim.set_state_from_flattened(state)
        base_env.sim.forward()
        frame_position[index] = base_env.sim.data.body_xpos[
            base_env.obj_body_id["frame"]
        ]
        tool_position[index] = base_env.sim.data.body_xpos[
            base_env.obj_body_id["tool"]
        ]
        eef_position[index] = base_env.sim.data.site_xpos[
            base_env.robots[0].eef_site_id[base_env.robots[0].arms[0]]
        ]
        frame_mount_position[index] = base_env.sim.data.site_xpos[
            base_env.obj_site_id["frame_mount_site"]
        ]
        frame_tip_position[index] = base_env.sim.data.site_xpos[frame_tip_site]
        base_position[index] = base_env.sim.data.geom_xpos[
            base_env.obj_geom_id["stand_base"]
        ]
        frame_hang_position[index] = base_env.sim.data.site_xpos[
            base_env.obj_site_id["frame_hang_site"]
        ]
        frame_intersection_position[index] = base_env.sim.data.site_xpos[
            base_env.obj_site_id["frame_intersection_site"]
        ]
        tool_hole_position[index] = base_env.sim.data.site_xpos[
            base_env.obj_site_id["tool_hole1_center"]
        ]
        frame_grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[0].gripper,
                object_geoms=["frame_grip_frame"],
            )
        )
        tool_grasped[index] = bool(
            base_env._check_grasp(
                gripper=base_env.robots[0].gripper,
                object_geoms=["tool_grip_g0"],
            )
        )
        frame_assembled[index] = bool(base_env._check_frame_assembled())
        tool_on_frame[index] = bool(base_env._check_tool_on_frame())

        hook_endpoint = frame_hang_position[index]
        hook_vector = frame_intersection_position[index] - hook_endpoint
        hook_length[index] = np.linalg.norm(hook_vector)
        hook_unit = hook_vector / hook_length[index]
        hole_vector = tool_hole_position[index] - hook_endpoint
        axial_distance = float(np.dot(hole_vector, hook_unit))
        orthogonal_vector = hole_vector - axial_distance * hook_unit
        tool_hook_orthogonal_distance[index] = np.linalg.norm(
            orthogonal_vector
        )
        tool_hook_normalized_axial_position[index] = (
            axial_distance / hook_length[index]
        )
        tool_hook_contact[index] = bool(
            check_contact(
                base_env.sim,
                hole_geoms,
                "frame_horizontal_frame",
            )
        )

        first_hole_position = base_env.sim.data.geom_xpos[
            base_env.obj_geom_id["tool_hole1_hc_0"]
        ]
        opposite_hole_position = base_env.sim.data.geom_xpos[
            base_env.obj_geom_id[
                f"tool_hole1_hc_{opposite_hole_geom}"
            ]
        ]
        tool_between_hook[index] = bool(
            np.dot(
                np.cross(first_hole_position - hook_endpoint, hook_unit),
                np.cross(opposite_hole_position - hook_endpoint, hook_unit),
            )
            < 0.0
        )

        frame_hook_vector = (
            frame_intersection_position[index]
            - frame_mount_position[index]
        )
        wall_positions = [
            base_env.sim.data.geom_xpos[
                base_env.obj_geom_id[f"stand_wall_{wall_index}"]
            ]
            - frame_mount_position[index]
            for wall_index in range(4)
        ]
        frame_between_walls[index] = bool(
            np.dot(
                np.cross(wall_positions[0], frame_hook_vector),
                np.cross(wall_positions[2], frame_hook_vector),
            )
            < 0.0
            and np.dot(
                np.cross(wall_positions[1], frame_hook_vector),
                np.cross(wall_positions[3], frame_hook_vector),
            )
            < 0.0
        )

    frame_insertion_distance = np.linalg.norm(
        frame_tip_position - base_position,
        axis=1,
    )
    tool_hook_endpoint_distance = np.linalg.norm(
        tool_hole_position - frame_hang_position,
        axis=1,
    )
    axial = tool_hook_normalized_axial_position
    axial_penalty = np.maximum(0.05 - axial, 0.0) + np.maximum(
        axial - 1.0,
        0.0,
    )
    tool_hook_error = np.sqrt(
        np.square(tool_hook_orthogonal_distance)
        + np.square(axial_penalty * hook_length)
    )
    return {
        "frame_position": frame_position,
        "frame_z": frame_position[:, 2],
        "tool_position": tool_position,
        "tool_z": tool_position[:, 2],
        "eef_frame_distance": np.linalg.norm(
            eef_position - frame_position,
            axis=1,
        ),
        "eef_tool_distance": np.linalg.norm(
            eef_position - tool_position,
            axis=1,
        ),
        "frame_grasped": frame_grasped,
        "tool_grasped": tool_grasped,
        "frame_assembled": frame_assembled,
        "tool_on_frame": tool_on_frame,
        "frame_between_walls": frame_between_walls,
        "frame_insertion_distance": frame_insertion_distance,
        "tool_hook_endpoint_distance": tool_hook_endpoint_distance,
        "tool_hook_orthogonal_distance": tool_hook_orthogonal_distance,
        "tool_hook_normalized_axial_position": (
            tool_hook_normalized_axial_position
        ),
        "tool_hook_error": tool_hook_error,
        "tool_hook_contact": tool_hook_contact,
        "tool_between_hook": tool_between_hook,
        "initial_frame_position": frame_position[0].copy(),
        "initial_tool_position": tool_position[0].copy(),
        "initial_frame_z": np.asarray(frame_position[0, 2]),
        "initial_tool_z": np.asarray(tool_position[0, 2]),
    }


def final_true_run_length(values: np.ndarray) -> int:
    count = 0
    for value in values[::-1]:
        if not bool(value):
            break
        count += 1
    return count


def evaluate_can_candidate(
    *,
    boundary: int,
    prediction_horizon: int,
    signals: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Validate all actions through the successor state at ``t + horizon``."""
    end = int(boundary) + int(prediction_horizon)
    if end >= len(signals["can_z"]):
        return None
    interval = slice(int(boundary), end + 1)
    eef_distance = signals["eef_can_distance"][interval]
    can_position = signals["can_position"][interval]
    can_z = signals["can_z"][interval]
    bin_distance = signals["target_xy_distance"][interval]
    grasped = signals["grasped"][interval]
    placed = signals["placed"][interval]
    stage_progress = signals["stage_progress"][interval]

    eef_deltas = np.diff(eef_distance)
    bin_deltas = np.diff(bin_distance)
    eef_gain = float(eef_distance[0] - eef_distance[-1])
    bin_gain = float(bin_distance[0] - bin_distance[-1])
    max_can_displacement = float(
        np.max(np.linalg.norm(can_position - can_position[0], axis=1))
    )
    z_drop = float(np.max(can_z) - can_z[-1])
    final_grasp_frames = final_true_run_length(grasped)
    grasp_fraction = float(np.mean(grasped))
    stage_gain = float(stage_progress[-1] - stage_progress[0])
    common = {
        "decision_boundary": int(boundary),
        "target_end_exclusive": int(end),
        "successor_state_step": int(end),
        "eef_distance_start": float(eef_distance[0]),
        "eef_distance_end": float(eef_distance[-1]),
        "eef_distance_gain": eef_gain,
        "bin_xy_distance_start": float(bin_distance[0]),
        "bin_xy_distance_end": float(bin_distance[-1]),
        "bin_xy_progress": bin_gain,
        "can_z_start": float(can_z[0]),
        "can_z_end": float(can_z[-1]),
        "can_z_peak": float(np.max(can_z)),
        "can_z_drop": z_drop,
        "max_can_displacement": max_can_displacement,
        "grasp_fraction": grasp_fraction,
        "final_grasp_frames": int(final_grasp_frames),
        "stage_progress_start": float(stage_progress[0]),
        "stage_progress_end": float(stage_progress[-1]),
        "stage_progress_gain": stage_gain,
        "placed_states": int(np.count_nonzero(placed)),
    }

    # Preserve clean pre-contact reaching, but reject pushes and windows that
    # enter the contact-risk region without establishing a real grasp.
    eef_progress_fraction = float(
        np.mean(eef_deltas <= float(args.can_safe_progress_tolerance))
    )
    if (
        not np.any(grasped)
        and not np.any(placed)
        and float(eef_distance[0]) >= float(args.can_safe_min_start_distance)
        and float(np.min(eef_distance)) >= float(args.can_safe_min_distance)
        and eef_gain >= float(args.can_safe_min_reach_gain)
        and eef_progress_fraction >= float(args.can_safe_min_progress_fraction)
        and float(np.max(eef_deltas)) <= float(args.can_safe_max_regression)
        and max_can_displacement <= float(args.can_safe_max_can_displacement)
    ):
        return {
            **common,
            "type": "safe_reach",
            "eef_progress_fraction": eef_progress_fraction,
            "eef_max_regression": float(np.max(eef_deltas)),
            "score": eef_gain,
        }

    # Once already lifted at the start of a window, retain only sustained-grasp
    # transport toward the correct bin. Requiring the starting lift state keeps
    # combined grasp-and-lift windows in the earlier grasp_lift stage.
    bin_progress_fraction = float(
        np.mean(bin_deltas <= float(args.can_transport_progress_tolerance))
    )
    lifted_height_start = float(can_z[0] - float(signals["initial_can_z"]))
    lifted_height_end = float(can_z[-1] - float(signals["initial_can_z"]))
    if (
        bool(grasped[-1])
        and grasp_fraction >= float(args.can_transport_min_grasp_fraction)
        and lifted_height_start >= float(args.can_transport_min_lift_height)
        and lifted_height_end >= float(args.can_transport_min_lift_height)
        and bin_gain >= float(args.can_transport_min_bin_progress)
        and bin_progress_fraction >= float(args.can_transport_min_progress_fraction)
        and float(np.max(bin_deltas)) <= float(args.can_transport_max_regression)
        and z_drop <= float(args.can_transport_max_drop)
        and not np.any(placed)
    ):
        return {
            **common,
            "type": "transport",
            "lifted_height_start_over_initial": lifted_height_start,
            "lifted_height_end_over_initial": lifted_height_end,
            "bin_progress_fraction": bin_progress_fraction,
            "bin_max_regression": float(np.max(bin_deltas)),
            "score": bin_gain + 0.25 * max(stage_gain, 0.0),
        }

    # A useful pick chunk must end in a sustained exact grasp and lift the can
    # without dropping it again before the target ends.
    if final_grasp_frames >= int(args.can_grasp_min_frames):
        final_run_start = len(grasped) - final_grasp_frames
        lift_gain = float(can_z[-1] - can_z[final_run_start])
        if (
            lift_gain >= float(args.can_grasp_min_lift_gain)
            and z_drop <= float(args.can_grasp_max_drop)
            and not np.any(placed)
        ):
            return {
                **common,
                "type": "grasp_lift",
                "grasp_start_in_window": int(final_run_start),
                "lift_gain_from_grasp": lift_gain,
                "score": lift_gain + 0.25 * max(stage_gain, 0.0),
            }
    return None


def select_can_chunks(
    *,
    signals: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Select high-confidence, stage-balanced Can windows."""
    latest = len(signals["can_z"]) - int(args.prediction_horizon) - 1
    if latest < int(args.min_start_step):
        return []
    upper = latest
    if args.max_start_step is not None:
        upper = min(upper, int(args.max_start_step))
    if args.max_start_fraction is not None:
        upper = min(
            upper,
            int(np.floor(float(args.max_start_fraction) * len(signals["can_z"]))),
        )
    candidates: list[dict[str, Any]] = []
    for boundary in range(int(args.min_start_step), upper + 1, int(args.stride)):
        candidate = evaluate_can_candidate(
            boundary=boundary,
            prediction_horizon=int(args.prediction_horizon),
            signals=signals,
            args=args,
        )
        if candidate is not None:
            candidates.append(candidate)

    # Later manipulation stages are rarer and more valuable than reaching, so
    # allocate their quota first while retaining deterministic score ordering.
    selected: list[dict[str, Any]] = []
    for stage_type in ("transport", "grasp_lift", "safe_reach"):
        stage_candidates = sorted(
            (item for item in candidates if item["type"] == stage_type),
            key=lambda item: (-float(item["score"]), int(item["decision_boundary"])),
        )
        kept_for_stage = 0
        for candidate in stage_candidates:
            boundary = int(candidate["decision_boundary"])
            if any(
                abs(boundary - int(previous["decision_boundary"]))
                < int(args.minimum_spacing)
                for previous in selected
            ):
                continue
            selected.append(candidate)
            kept_for_stage += 1
            if kept_for_stage >= int(args.can_max_chunks_per_stage_per_failure):
                break
            if len(selected) >= int(args.max_chunks_per_failure):
                break
        if len(selected) >= int(args.max_chunks_per_failure):
            break
    return sorted(selected, key=lambda item: int(item["decision_boundary"]))


def evaluate_transport_candidate(
    *,
    boundary: int,
    prediction_horizon: int,
    signals: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Validate a joint two-arm action window through its successor state."""
    end = int(boundary) + int(prediction_horizon)
    if end >= len(signals["payload_z"]):
        return None
    interval = slice(int(boundary), end + 1)

    payload_position = signals["payload_position"][interval]
    trash_position = signals["trash_position"][interval]
    lid_position = signals["lid_position"][interval]
    payload_z = signals["payload_z"][interval]
    trash_z = signals["trash_z"][interval]
    lid_z = signals["lid_z"][interval]
    lid_distance = signals["robot0_lid_distance"][interval]
    payload_eef_distance = signals["robot0_payload_distance"][interval]
    trash_eef_distance = signals["robot1_trash_distance"][interval]
    lid_grasped = signals["robot0_lid_grasped"][interval]
    payload_grasped = signals["robot0_payload_grasped"][interval]
    trash_grasped = signals["robot1_trash_grasped"][interval]
    payload_placed = signals["payload_placed"][interval]
    trash_placed = signals["trash_placed"][interval]
    payload_bin_distance = signals["payload_target_xy_distance"][interval]
    trash_bin_distance = signals["trash_target_xy_distance"][interval]
    lid_clearance = np.linalg.norm(
        lid_position - signals["initial_lid_position"][None],
        axis=1,
    )

    lid_eef_deltas = np.diff(lid_distance)
    payload_eef_deltas = np.diff(payload_eef_distance)
    trash_eef_deltas = np.diff(trash_eef_distance)
    payload_bin_deltas = np.diff(payload_bin_distance)
    trash_bin_deltas = np.diff(trash_bin_distance)
    lid_reach_gain = float(lid_distance[0] - lid_distance[-1])
    payload_reach_gain = float(
        payload_eef_distance[0] - payload_eef_distance[-1]
    )
    trash_reach_gain = float(trash_eef_distance[0] - trash_eef_distance[-1])
    payload_bin_gain = float(payload_bin_distance[0] - payload_bin_distance[-1])
    trash_bin_gain = float(trash_bin_distance[0] - trash_bin_distance[-1])
    lid_displacement = np.linalg.norm(lid_position - lid_position[0], axis=1)
    payload_displacement = np.linalg.norm(
        payload_position - payload_position[0],
        axis=1,
    )
    trash_displacement = np.linalg.norm(
        trash_position - trash_position[0],
        axis=1,
    )
    lid_z_drop = float(np.max(lid_z) - lid_z[-1])
    payload_z_drop = float(np.max(payload_z) - payload_z[-1])
    trash_z_drop = float(np.max(trash_z) - trash_z[-1])
    lid_grasp_fraction = float(np.mean(lid_grasped))
    payload_grasp_fraction = float(np.mean(payload_grasped))
    trash_grasp_fraction = float(np.mean(trash_grasped))
    final_lid_grasp_frames = final_true_run_length(lid_grasped)
    final_payload_grasp_frames = final_true_run_length(payload_grasped)
    final_trash_grasp_frames = final_true_run_length(trash_grasped)

    def placement_preserved(values: np.ndarray) -> bool:
        placed_indices = np.flatnonzero(values)
        if not len(placed_indices):
            return True
        return bool(np.all(values[int(placed_indices[0]) :]))

    if not placement_preserved(payload_placed) or not placement_preserved(
        trash_placed
    ):
        return None

    static_limit = float(args.transport_secondary_max_static_displacement)
    min_moving_grasp = float(args.transport_secondary_min_grasp_fraction)
    max_drop = float(args.transport_target_max_drop)
    max_regression = float(args.transport_target_max_regression)
    lid_min_clearance = float(args.transport_lid_min_clearance)
    lid_clearance_regression = float(
        args.transport_lid_max_clearance_regression
    )

    def trash_branch_safe() -> bool:
        if bool(trash_placed[0]):
            return bool(np.all(trash_placed))
        if float(np.max(trash_displacement)) <= static_limit:
            return True
        if bool(trash_placed[-1]):
            return trash_bin_gain >= -max_regression
        return bool(
            trash_grasp_fraction >= min_moving_grasp
            and bool(trash_grasped[-1])
            and trash_z_drop <= max_drop
            and trash_bin_gain >= -max_regression
        )

    def lid_and_payload_branch_safe() -> bool:
        if float(np.max(payload_displacement)) > static_limit:
            return False
        if float(np.max(lid_displacement)) <= static_limit:
            return True
        grasped_motion = bool(
            lid_grasp_fraction >= min_moving_grasp
            and (bool(lid_grasped[-1]) or lid_clearance[-1] >= lid_min_clearance)
            and lid_z_drop <= float(args.transport_lid_max_drop)
        )
        already_clear = bool(
            lid_clearance[0] >= lid_min_clearance
            and lid_clearance[-1]
            >= lid_clearance[0] - lid_clearance_regression
        )
        return grasped_motion or already_clear

    common = {
        "decision_boundary": int(boundary),
        "target_end_exclusive": int(end),
        "successor_state_step": int(end),
        "lid_reach_gain": lid_reach_gain,
        "payload_reach_gain": payload_reach_gain,
        "trash_reach_gain": trash_reach_gain,
        "lid_clearance_start": float(lid_clearance[0]),
        "lid_clearance_end": float(lid_clearance[-1]),
        "lid_clearance_peak": float(np.max(lid_clearance)),
        "lid_max_displacement": float(np.max(lid_displacement)),
        "payload_max_displacement": float(np.max(payload_displacement)),
        "trash_max_displacement": float(np.max(trash_displacement)),
        "payload_bin_xy_distance_start": float(payload_bin_distance[0]),
        "payload_bin_xy_distance_end": float(payload_bin_distance[-1]),
        "payload_bin_xy_progress": payload_bin_gain,
        "trash_bin_xy_distance_start": float(trash_bin_distance[0]),
        "trash_bin_xy_distance_end": float(trash_bin_distance[-1]),
        "trash_bin_xy_progress": trash_bin_gain,
        "payload_lifted_height_start": float(
            payload_z[0] - float(signals["initial_payload_z"])
        ),
        "payload_lifted_height_end": float(
            payload_z[-1] - float(signals["initial_payload_z"])
        ),
        "trash_lifted_height_start": float(
            trash_z[0] - float(signals["initial_trash_z"])
        ),
        "trash_lifted_height_end": float(
            trash_z[-1] - float(signals["initial_trash_z"])
        ),
        "lid_z_drop": lid_z_drop,
        "payload_z_drop": payload_z_drop,
        "trash_z_drop": trash_z_drop,
        "lid_grasp_fraction": lid_grasp_fraction,
        "payload_grasp_fraction": payload_grasp_fraction,
        "trash_grasp_fraction": trash_grasp_fraction,
        "final_lid_grasp_frames": int(final_lid_grasp_frames),
        "final_payload_grasp_frames": int(final_payload_grasp_frames),
        "final_trash_grasp_frames": int(final_trash_grasp_frames),
        "payload_placed_states": int(np.count_nonzero(payload_placed)),
        "trash_placed_states": int(np.count_nonzero(trash_placed)),
    }

    payload_progress_fraction = float(
        np.mean(
            payload_bin_deltas
            <= float(args.transport_target_progress_tolerance)
        )
    )
    trash_progress_fraction = float(
        np.mean(
            trash_bin_deltas <= float(args.transport_target_progress_tolerance)
        )
    )
    trash_stably_complete = bool(np.all(trash_placed))
    lid_stably_clear = bool(np.min(lid_clearance) >= lid_min_clearance)

    # Late payload windows are only positive joint-action examples after the
    # other arm has stably completed trash disposal and the lid remains clear.
    if trash_stably_complete and lid_stably_clear:
        if (
            not np.any(payload_placed)
            and bool(payload_grasped[-1])
            and payload_grasp_fraction
            >= float(args.transport_target_min_grasp_fraction)
            and common["payload_lifted_height_start"]
            >= float(args.transport_target_min_lift_height)
            and common["payload_lifted_height_end"]
            >= float(args.transport_target_min_lift_height)
            and payload_bin_gain >= float(args.transport_target_min_bin_progress)
            and payload_progress_fraction
            >= float(args.transport_target_min_progress_fraction)
            and float(np.max(payload_bin_deltas)) <= max_regression
            and payload_z_drop <= max_drop
        ):
            return {
                **common,
                "type": "payload_transport",
                "payload_progress_fraction": payload_progress_fraction,
                "payload_bin_max_regression": float(
                    np.max(payload_bin_deltas)
                ),
                "score": payload_bin_gain,
            }

        if final_payload_grasp_frames >= int(args.transport_grasp_min_frames):
            final_run_start = len(payload_grasped) - final_payload_grasp_frames
            lift_gain = float(payload_z[-1] - payload_z[final_run_start])
            if (
                lift_gain >= float(args.transport_grasp_min_lift_gain)
                and payload_z_drop <= float(args.transport_grasp_max_drop)
                and not np.any(payload_placed)
            ):
                return {
                    **common,
                    "type": "payload_grasp_lift",
                    "payload_grasp_start_in_window": int(final_run_start),
                    "payload_lift_gain_from_grasp": lift_gain,
                    "score": lift_gain,
                }

        payload_reach_fraction = float(
            np.mean(
                payload_eef_deltas
                <= float(args.transport_safe_progress_tolerance)
            )
        )
        if (
            not np.any(payload_grasped)
            and not np.any(payload_placed)
            and float(payload_eef_distance[0])
            >= float(args.transport_safe_min_start_distance)
            and float(np.min(payload_eef_distance))
            >= float(args.transport_safe_min_distance)
            and payload_reach_gain >= float(args.transport_safe_min_reach_gain)
            and payload_reach_fraction
            >= float(args.transport_safe_min_progress_fraction)
            and float(np.max(payload_eef_deltas))
            <= float(args.transport_safe_max_regression)
            and float(np.max(payload_displacement)) <= static_limit
        ):
            return {
                **common,
                "type": "payload_safe_reach",
                "payload_reach_progress_fraction": payload_reach_fraction,
                "payload_reach_max_regression": float(
                    np.max(payload_eef_deltas)
                ),
                "score": payload_reach_gain,
            }

    # Trash placement is a useful exact completion transition. The first arm
    # must simultaneously keep the unopened payload fixed and the lid safe.
    if (
        not bool(trash_placed[0])
        and bool(trash_placed[-1])
        and trash_grasp_fraction >= min_moving_grasp
        and trash_bin_gain >= float(args.transport_place_min_bin_progress)
        and lid_and_payload_branch_safe()
    ):
        return {
            **common,
            "type": "trash_place",
            "trash_progress_fraction": trash_progress_fraction,
            "score": trash_bin_gain + float(args.transport_place_score_bonus),
        }

    if (
        not np.any(trash_placed)
        and bool(trash_grasped[-1])
        and trash_grasp_fraction
        >= float(args.transport_target_min_grasp_fraction)
        and common["trash_lifted_height_start"]
        >= float(args.transport_target_min_lift_height)
        and common["trash_lifted_height_end"]
        >= float(args.transport_target_min_lift_height)
        and trash_bin_gain >= float(args.transport_target_min_bin_progress)
        and trash_progress_fraction
        >= float(args.transport_target_min_progress_fraction)
        and float(np.max(trash_bin_deltas)) <= max_regression
        and trash_z_drop <= max_drop
        and lid_and_payload_branch_safe()
    ):
        return {
            **common,
            "type": "trash_transport",
            "trash_progress_fraction": trash_progress_fraction,
            "trash_bin_max_regression": float(np.max(trash_bin_deltas)),
            "score": trash_bin_gain,
        }

    if final_trash_grasp_frames >= int(args.transport_grasp_min_frames):
        final_run_start = len(trash_grasped) - final_trash_grasp_frames
        lift_gain = float(trash_z[-1] - trash_z[final_run_start])
        if (
            lift_gain >= float(args.transport_grasp_min_lift_gain)
            and trash_z_drop <= float(args.transport_grasp_max_drop)
            and not np.any(trash_placed)
            and lid_and_payload_branch_safe()
        ):
            return {
                **common,
                "type": "trash_grasp_lift",
                "trash_grasp_start_in_window": int(final_run_start),
                "trash_lift_gain_from_grasp": lift_gain,
                "score": lift_gain,
            }

    if final_lid_grasp_frames >= int(args.transport_grasp_min_frames):
        final_run_start = len(lid_grasped) - final_lid_grasp_frames
        clearance_gain = float(
            lid_clearance[-1] - lid_clearance[final_run_start]
        )
        if (
            clearance_gain >= float(args.transport_lid_min_clearance_gain)
            and lid_clearance[-1] >= lid_min_clearance
            and lid_z_drop <= float(args.transport_lid_max_drop)
            and float(np.max(payload_displacement)) <= static_limit
            and trash_branch_safe()
        ):
            return {
                **common,
                "type": "lid_lift_clear",
                "lid_grasp_start_in_window": int(final_run_start),
                "lid_clearance_gain_from_grasp": clearance_gain,
                "score": clearance_gain + 0.25 * max(trash_bin_gain, 0.0),
            }

    # Before any contact, retain coordinated reaching only when both arms keep
    # all three objects stationary and neither arm substantially regresses.
    if (
        not np.any(lid_grasped)
        and not np.any(payload_grasped)
        and not np.any(trash_grasped)
        and not np.any(payload_placed)
        and not np.any(trash_placed)
        and float(np.max(lid_displacement)) <= static_limit
        and float(np.max(payload_displacement)) <= static_limit
        and float(np.max(trash_displacement)) <= static_limit
    ):
        reach_options = (
            (
                "lid",
                lid_distance,
                lid_eef_deltas,
                lid_reach_gain,
            ),
            (
                "trash",
                trash_eef_distance,
                trash_eef_deltas,
                trash_reach_gain,
            ),
        )
        primary_name, primary_distance, primary_deltas, primary_gain = max(
            reach_options,
            key=lambda item: item[3],
        )
        secondary_gain = (
            trash_reach_gain if primary_name == "lid" else lid_reach_gain
        )
        progress_fraction = float(
            np.mean(
                primary_deltas
                <= float(args.transport_safe_progress_tolerance)
            )
        )
        if (
            float(primary_distance[0])
            >= float(args.transport_safe_min_start_distance)
            and float(np.min(primary_distance))
            >= float(args.transport_safe_min_distance)
            and primary_gain >= float(args.transport_safe_min_reach_gain)
            and progress_fraction
            >= float(args.transport_safe_min_progress_fraction)
            and float(np.max(primary_deltas))
            <= float(args.transport_safe_max_regression)
            and secondary_gain >= -float(args.transport_safe_max_regression)
        ):
            return {
                **common,
                "type": "coordinated_safe_reach",
                "primary_reach_object": primary_name,
                "primary_reach_progress_fraction": progress_fraction,
                "primary_reach_max_regression": float(np.max(primary_deltas)),
                "score": primary_gain + 0.25 * max(secondary_gain, 0.0),
            }
    return None


def select_transport_chunks(
    *,
    signals: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Select high-confidence, late-stage-first Transport windows."""
    latest = len(signals["payload_z"]) - int(args.prediction_horizon) - 1
    if latest < int(args.min_start_step):
        return []
    upper = latest
    if args.max_start_step is not None:
        upper = min(upper, int(args.max_start_step))
    if args.max_start_fraction is not None:
        upper = min(
            upper,
            int(
                np.floor(
                    float(args.max_start_fraction)
                    * len(signals["payload_z"])
                )
            ),
        )

    candidates: list[dict[str, Any]] = []
    for boundary in range(int(args.min_start_step), upper + 1, int(args.stride)):
        candidate = evaluate_transport_candidate(
            boundary=boundary,
            prediction_horizon=int(args.prediction_horizon),
            signals=signals,
            args=args,
        )
        if candidate is not None:
            candidates.append(candidate)

    priority = (
        "payload_transport",
        "payload_grasp_lift",
        "payload_safe_reach",
        "trash_place",
        "trash_transport",
        "trash_grasp_lift",
        "lid_lift_clear",
        "coordinated_safe_reach",
    )
    selected: list[dict[str, Any]] = []
    for stage_type in priority:
        stage_candidates = sorted(
            (item for item in candidates if item["type"] == stage_type),
            key=lambda item: (
                -float(item["score"]),
                int(item["decision_boundary"]),
            ),
        )
        kept_for_stage = 0
        for candidate in stage_candidates:
            boundary = int(candidate["decision_boundary"])
            if any(
                abs(boundary - int(previous["decision_boundary"]))
                < int(args.minimum_spacing)
                for previous in selected
            ):
                continue
            selected.append(candidate)
            kept_for_stage += 1
            if kept_for_stage >= int(
                args.transport_max_chunks_per_stage_per_failure
            ):
                break
            if len(selected) >= int(args.max_chunks_per_failure):
                break
        if len(selected) >= int(args.max_chunks_per_failure):
            break
    return sorted(selected, key=lambda item: int(item["decision_boundary"]))



def evaluate_tool_hang_candidate(
    *,
    boundary: int,
    prediction_horizon: int,
    signals: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Validate a Tool Hang stage through the final action successor state."""
    end = int(boundary) + int(prediction_horizon)
    if end >= len(signals["frame_z"]):
        return None
    interval = slice(int(boundary), end + 1)

    frame_position = signals["frame_position"][interval]
    tool_position = signals["tool_position"][interval]
    frame_z = signals["frame_z"][interval]
    tool_z = signals["tool_z"][interval]
    frame_eef_distance = signals["eef_frame_distance"][interval]
    tool_eef_distance = signals["eef_tool_distance"][interval]
    frame_grasped = signals["frame_grasped"][interval]
    tool_grasped = signals["tool_grasped"][interval]
    frame_assembled = signals["frame_assembled"][interval]
    tool_on_frame = signals["tool_on_frame"][interval]
    frame_between_walls = signals["frame_between_walls"][interval]
    frame_insertion_distance = signals["frame_insertion_distance"][interval]
    tool_hook_endpoint_distance = signals[
        "tool_hook_endpoint_distance"
    ][interval]
    tool_hook_orthogonal_distance = signals[
        "tool_hook_orthogonal_distance"
    ][interval]
    tool_hook_axial_position = signals[
        "tool_hook_normalized_axial_position"
    ][interval]
    tool_hook_error = signals["tool_hook_error"][interval]
    tool_hook_contact = signals["tool_hook_contact"][interval]
    tool_between_hook = signals["tool_between_hook"][interval]

    frame_eef_deltas = np.diff(frame_eef_distance)
    tool_eef_deltas = np.diff(tool_eef_distance)
    frame_insertion_deltas = np.diff(frame_insertion_distance)
    tool_endpoint_deltas = np.diff(tool_hook_endpoint_distance)
    tool_error_deltas = np.diff(tool_hook_error)
    frame_reach_gain = float(
        frame_eef_distance[0] - frame_eef_distance[-1]
    )
    tool_reach_gain = float(tool_eef_distance[0] - tool_eef_distance[-1])
    frame_insertion_gain = float(
        frame_insertion_distance[0] - frame_insertion_distance[-1]
    )
    tool_endpoint_gain = float(
        tool_hook_endpoint_distance[0] - tool_hook_endpoint_distance[-1]
    )
    tool_hook_error_gain = float(tool_hook_error[0] - tool_hook_error[-1])
    frame_displacement = np.linalg.norm(
        frame_position - frame_position[0],
        axis=1,
    )
    tool_displacement = np.linalg.norm(
        tool_position - tool_position[0],
        axis=1,
    )
    frame_z_drop = float(np.max(frame_z) - frame_z[-1])
    tool_z_drop = float(np.max(tool_z) - tool_z[-1])
    frame_grasp_fraction = float(np.mean(frame_grasped))
    tool_grasp_fraction = float(np.mean(tool_grasped))
    final_frame_grasp_frames = final_true_run_length(frame_grasped)
    final_tool_grasp_frames = final_true_run_length(tool_grasped)

    def predicate_preserved(values: np.ndarray) -> bool:
        true_indices = np.flatnonzero(values)
        if not len(true_indices):
            return True
        return bool(np.all(values[int(true_indices[0]) :]))

    if not predicate_preserved(frame_assembled) or not predicate_preserved(
        tool_on_frame
    ):
        return None

    static_limit = float(args.tool_hang_max_static_object_displacement)
    common = {
        "decision_boundary": int(boundary),
        "target_end_exclusive": int(end),
        "successor_state_step": int(end),
        "frame_reach_gain": frame_reach_gain,
        "tool_reach_gain": tool_reach_gain,
        "frame_insertion_distance_start": float(frame_insertion_distance[0]),
        "frame_insertion_distance_end": float(frame_insertion_distance[-1]),
        "frame_insertion_progress": frame_insertion_gain,
        "tool_hook_endpoint_distance_start": float(
            tool_hook_endpoint_distance[0]
        ),
        "tool_hook_endpoint_distance_end": float(
            tool_hook_endpoint_distance[-1]
        ),
        "tool_hook_endpoint_progress": tool_endpoint_gain,
        "tool_hook_error_start": float(tool_hook_error[0]),
        "tool_hook_error_end": float(tool_hook_error[-1]),
        "tool_hook_error_progress": tool_hook_error_gain,
        "tool_hook_orthogonal_distance_end": float(
            tool_hook_orthogonal_distance[-1]
        ),
        "tool_hook_axial_position_end": float(tool_hook_axial_position[-1]),
        "frame_max_displacement": float(np.max(frame_displacement)),
        "tool_max_displacement": float(np.max(tool_displacement)),
        "frame_lifted_height_start": float(
            frame_z[0] - float(signals["initial_frame_z"])
        ),
        "frame_lifted_height_end": float(
            frame_z[-1] - float(signals["initial_frame_z"])
        ),
        "tool_lifted_height_start": float(
            tool_z[0] - float(signals["initial_tool_z"])
        ),
        "tool_lifted_height_end": float(
            tool_z[-1] - float(signals["initial_tool_z"])
        ),
        "frame_z_drop": frame_z_drop,
        "tool_z_drop": tool_z_drop,
        "frame_grasp_fraction": frame_grasp_fraction,
        "tool_grasp_fraction": tool_grasp_fraction,
        "final_frame_grasp_frames": int(final_frame_grasp_frames),
        "final_tool_grasp_frames": int(final_tool_grasp_frames),
        "frame_assembled_states": int(np.count_nonzero(frame_assembled)),
        "tool_on_frame_states": int(np.count_nonzero(tool_on_frame)),
        "frame_between_walls_states": int(
            np.count_nonzero(frame_between_walls)
        ),
        "tool_hook_contact_states": int(np.count_nonzero(tool_hook_contact)),
        "tool_between_hook_states": int(np.count_nonzero(tool_between_hook)),
    }

    frame_stably_assembled = bool(
        np.all(frame_assembled)
        and float(np.max(frame_displacement)) <= static_limit
    )
    if frame_stably_assembled:
        tool_error_fraction = float(
            np.mean(
                tool_error_deltas
                <= float(args.tool_hang_progress_tolerance)
            )
        )
        if (
            not np.any(tool_on_frame)
            and bool(tool_grasped[-1])
            and tool_grasp_fraction
            >= float(args.tool_hang_transport_min_grasp_fraction)
            and common["tool_lifted_height_start"]
            >= float(args.tool_hang_transport_min_lift_height)
            and common["tool_lifted_height_end"]
            >= float(args.tool_hang_transport_min_lift_height)
            and tool_hook_endpoint_distance[-1]
            <= float(args.tool_hang_align_max_endpoint_distance)
            and tool_hook_error_gain
            >= float(args.tool_hang_align_min_error_progress)
            and tool_error_fraction
            >= float(args.tool_hang_align_min_progress_fraction)
            and float(np.max(tool_error_deltas))
            <= float(args.tool_hang_align_max_regression)
            and tool_z_drop <= float(args.tool_hang_transport_max_drop)
        ):
            return {
                **common,
                "type": "tool_hook_align",
                "tool_hook_error_progress_fraction": tool_error_fraction,
                "tool_hook_error_max_regression": float(
                    np.max(tool_error_deltas)
                ),
                "score": tool_hook_error_gain
                + float(args.tool_hang_hook_contact_score_bonus)
                * float(bool(tool_hook_contact[-1])),
            }

        tool_endpoint_fraction = float(
            np.mean(
                tool_endpoint_deltas
                <= float(args.tool_hang_progress_tolerance)
            )
        )
        if (
            not np.any(tool_on_frame)
            and bool(tool_grasped[-1])
            and tool_grasp_fraction
            >= float(args.tool_hang_transport_min_grasp_fraction)
            and common["tool_lifted_height_start"]
            >= float(args.tool_hang_transport_min_lift_height)
            and common["tool_lifted_height_end"]
            >= float(args.tool_hang_transport_min_lift_height)
            and tool_endpoint_gain
            >= float(args.tool_hang_transport_min_progress)
            and tool_endpoint_fraction
            >= float(args.tool_hang_transport_min_progress_fraction)
            and float(np.max(tool_endpoint_deltas))
            <= float(args.tool_hang_transport_max_regression)
            and tool_z_drop <= float(args.tool_hang_transport_max_drop)
        ):
            return {
                **common,
                "type": "tool_transport",
                "tool_endpoint_progress_fraction": tool_endpoint_fraction,
                "tool_endpoint_max_regression": float(
                    np.max(tool_endpoint_deltas)
                ),
                "score": tool_endpoint_gain,
            }

        if final_tool_grasp_frames >= int(args.tool_hang_grasp_min_frames):
            final_run_start = len(tool_grasped) - final_tool_grasp_frames
            lift_gain = float(tool_z[-1] - tool_z[final_run_start])
            if (
                lift_gain >= float(args.tool_hang_grasp_min_lift_gain)
                and tool_z_drop <= float(args.tool_hang_grasp_max_drop)
                and not np.any(tool_on_frame)
            ):
                return {
                    **common,
                    "type": "tool_grasp_lift",
                    "tool_grasp_start_in_window": int(final_run_start),
                    "tool_lift_gain_from_grasp": lift_gain,
                    "score": lift_gain,
                }

        tool_reach_fraction = float(
            np.mean(
                tool_eef_deltas
                <= float(args.tool_hang_progress_tolerance)
            )
        )
        if (
            not np.any(tool_grasped)
            and not np.any(tool_on_frame)
            and float(tool_eef_distance[0])
            >= float(args.tool_hang_safe_min_start_distance)
            and float(np.min(tool_eef_distance))
            >= float(args.tool_hang_safe_min_distance)
            and tool_reach_gain >= float(args.tool_hang_safe_min_reach_gain)
            and tool_reach_fraction
            >= float(args.tool_hang_safe_min_progress_fraction)
            and float(np.max(tool_eef_deltas))
            <= float(args.tool_hang_safe_max_regression)
            and float(np.max(tool_displacement)) <= static_limit
        ):
            return {
                **common,
                "type": "tool_safe_reach",
                "tool_reach_progress_fraction": tool_reach_fraction,
                "tool_reach_max_regression": float(
                    np.max(tool_eef_deltas)
                ),
                "score": tool_reach_gain,
            }

    # Exact transition to a stable assembly is retained even if the gripper has
    # already loosened: successful insertions often settle after release.
    if (
        not bool(frame_assembled[0])
        and bool(frame_assembled[-1])
        and frame_insertion_distance[-1]
        <= float(args.tool_hang_frame_insert_max_distance)
        and float(np.max(tool_displacement)) <= static_limit
    ):
        return {
            **common,
            "type": "frame_insert",
            "score": frame_insertion_gain
            + float(args.tool_hang_frame_insert_score_bonus),
        }

    frame_insertion_fraction = float(
        np.mean(
            frame_insertion_deltas
            <= float(args.tool_hang_progress_tolerance)
        )
    )
    if (
        not np.any(frame_assembled)
        and bool(frame_grasped[-1])
        and frame_grasp_fraction
        >= float(args.tool_hang_transport_min_grasp_fraction)
        and common["frame_lifted_height_start"]
        >= float(args.tool_hang_transport_min_lift_height)
        and common["frame_lifted_height_end"]
        >= float(args.tool_hang_transport_min_lift_height)
        and frame_insertion_gain
        >= float(args.tool_hang_transport_min_progress)
        and frame_insertion_fraction
        >= float(args.tool_hang_transport_min_progress_fraction)
        and float(np.max(frame_insertion_deltas))
        <= float(args.tool_hang_transport_max_regression)
        and frame_z_drop <= float(args.tool_hang_transport_max_drop)
        and float(np.max(tool_displacement)) <= static_limit
    ):
        return {
            **common,
            "type": "frame_transport",
            "frame_insertion_progress_fraction": frame_insertion_fraction,
            "frame_insertion_max_regression": float(
                np.max(frame_insertion_deltas)
            ),
            "score": frame_insertion_gain,
        }

    if final_frame_grasp_frames >= int(args.tool_hang_grasp_min_frames):
        final_run_start = len(frame_grasped) - final_frame_grasp_frames
        lift_gain = float(frame_z[-1] - frame_z[final_run_start])
        if (
            lift_gain >= float(args.tool_hang_grasp_min_lift_gain)
            and frame_z_drop <= float(args.tool_hang_grasp_max_drop)
            and not np.any(frame_assembled)
            and float(np.max(tool_displacement)) <= static_limit
        ):
            return {
                **common,
                "type": "frame_grasp_lift",
                "frame_grasp_start_in_window": int(final_run_start),
                "frame_lift_gain_from_grasp": lift_gain,
                "score": lift_gain,
            }

    frame_reach_fraction = float(
        np.mean(
            frame_eef_deltas <= float(args.tool_hang_progress_tolerance)
        )
    )
    if (
        not np.any(frame_grasped)
        and not np.any(frame_assembled)
        and float(frame_eef_distance[0])
        >= float(args.tool_hang_safe_min_start_distance)
        and float(np.min(frame_eef_distance))
        >= float(args.tool_hang_safe_min_distance)
        and frame_reach_gain >= float(args.tool_hang_safe_min_reach_gain)
        and frame_reach_fraction
        >= float(args.tool_hang_safe_min_progress_fraction)
        and float(np.max(frame_eef_deltas))
        <= float(args.tool_hang_safe_max_regression)
        and float(np.max(frame_displacement)) <= static_limit
        and float(np.max(tool_displacement)) <= static_limit
    ):
        return {
            **common,
            "type": "frame_safe_reach",
            "frame_reach_progress_fraction": frame_reach_fraction,
            "frame_reach_max_regression": float(np.max(frame_eef_deltas)),
            "score": frame_reach_gain,
        }
    return None


def select_tool_hang_chunks(
    *,
    signals: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Select high-confidence Tool Hang windows, allocating late stages first."""
    latest = len(signals["frame_z"]) - int(args.prediction_horizon) - 1
    if latest < int(args.min_start_step):
        return []
    upper = latest
    if args.max_start_step is not None:
        upper = min(upper, int(args.max_start_step))
    if args.max_start_fraction is not None:
        upper = min(
            upper,
            int(
                np.floor(
                    float(args.max_start_fraction) * len(signals["frame_z"])
                )
            ),
        )

    candidates: list[dict[str, Any]] = []
    for boundary in range(int(args.min_start_step), upper + 1, int(args.stride)):
        candidate = evaluate_tool_hang_candidate(
            boundary=boundary,
            prediction_horizon=int(args.prediction_horizon),
            signals=signals,
            args=args,
        )
        if candidate is not None:
            candidates.append(candidate)

    priority = (
        "tool_hook_align",
        "tool_transport",
        "tool_grasp_lift",
        "frame_insert",
        "tool_safe_reach",
        "frame_transport",
        "frame_grasp_lift",
        "frame_safe_reach",
    )
    selected: list[dict[str, Any]] = []
    for stage_type in priority:
        stage_candidates = sorted(
            (item for item in candidates if item["type"] == stage_type),
            key=lambda item: (
                -float(item["score"]),
                int(item["decision_boundary"]),
            ),
        )
        kept_for_stage = 0
        for candidate in stage_candidates:
            boundary = int(candidate["decision_boundary"])
            if any(
                abs(boundary - int(previous["decision_boundary"]))
                < int(args.minimum_spacing)
                for previous in selected
            ):
                continue
            selected.append(candidate)
            kept_for_stage += 1
            if kept_for_stage >= int(
                args.tool_hang_max_chunks_per_stage_per_failure
            ):
                break
            if len(selected) >= int(args.max_chunks_per_failure):
                break
        if len(selected) >= int(args.max_chunks_per_failure):
            break
    return sorted(selected, key=lambda item: int(item["decision_boundary"]))


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


def can_record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = {
        stage: sum(record.get("type") == stage for record in records)
        for stage in CAN_STAGE_TYPES
    }


def transport_record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = {
        stage: sum(record.get("type") == stage for record in records)
        for stage in TRANSPORT_STAGE_TYPES
    }
    stage_source_rollouts = {
        stage: len(
            {
                record["source_demo"]
                for record in records
                if record.get("type") == stage
            }
        )
        for stage in TRANSPORT_STAGE_TYPES
    }
    return {
        "chunks": len(records),
        "source_rollouts": len({record["source_demo"] for record in records}),
        "stage_counts": stage_counts,
        "stage_source_rollouts": stage_source_rollouts,
        "boundary": stats(
            np.asarray(
                [record["decision_boundary"] for record in records],
                dtype=np.int64,
            )
        ),
        "lid_clearance_end": stats(
            np.asarray(
                [record["lid_clearance_end"] for record in records],
                dtype=np.float64,
            )
        ),
        "trash_bin_xy_progress": stats(
            np.asarray(
                [record["trash_bin_xy_progress"] for record in records],
                dtype=np.float64,
            )
        ),
        "payload_bin_xy_progress": stats(
            np.asarray(
                [record["payload_bin_xy_progress"] for record in records],
                dtype=np.float64,
            )
        ),
        "trash_placed_states": stats(
            np.asarray(
                [record["trash_placed_states"] for record in records],
                dtype=np.int64,
            )
        ),
    }
    stage_source_rollouts = {
        stage: len(
            {
                record["source_demo"]
                for record in records
                if record.get("type") == stage
            }
        )
        for stage in CAN_STAGE_TYPES
    }
    return {
        "chunks": len(records),
        "source_rollouts": len({record["source_demo"] for record in records}),
        "stage_counts": stage_counts,
        "stage_source_rollouts": stage_source_rollouts,
        "boundary": stats(
            np.asarray(
                [record["decision_boundary"] for record in records],
                dtype=np.int64,
            )
        ),
        "eef_distance_gain": stats(
            np.asarray(
                [record["eef_distance_gain"] for record in records],
                dtype=np.float64,
            )
        ),
        "bin_xy_progress": stats(
            np.asarray(
                [record["bin_xy_progress"] for record in records],
                dtype=np.float64,
            )
        ),
        "can_z_end": stats(
            np.asarray(
                [record["can_z_end"] for record in records],
                dtype=np.float64,
            )
        ),
    }



def tool_hang_record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = {
        stage: sum(record.get("type") == stage for record in records)
        for stage in TOOL_HANG_STAGE_TYPES
    }
    stage_source_rollouts = {
        stage: len(
            {
                record["source_demo"]
                for record in records
                if record.get("type") == stage
            }
        )
        for stage in TOOL_HANG_STAGE_TYPES
    }
    return {
        "chunks": len(records),
        "source_rollouts": len({record["source_demo"] for record in records}),
        "stage_counts": stage_counts,
        "stage_source_rollouts": stage_source_rollouts,
        "boundary": stats(
            np.asarray(
                [record["decision_boundary"] for record in records],
                dtype=np.int64,
            )
        ),
        "frame_insertion_progress": stats(
            np.asarray(
                [record["frame_insertion_progress"] for record in records],
                dtype=np.float64,
            )
        ),
        "tool_hook_endpoint_progress": stats(
            np.asarray(
                [record["tool_hook_endpoint_progress"] for record in records],
                dtype=np.float64,
            )
        ),
        "tool_hook_error_progress": stats(
            np.asarray(
                [record["tool_hook_error_progress"] for record in records],
                dtype=np.float64,
            )
        ),
        "frame_assembled_states": stats(
            np.asarray(
                [record["frame_assembled_states"] for record in records],
                dtype=np.int64,
            )
        ),
        "tool_hook_contact_states": stats(
            np.asarray(
                [record["tool_hook_contact_states"] for record in records],
                dtype=np.int64,
            )
        ),
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
    endpoint_records: list[dict[str, Any]] = []
    success_calibration_records: list[dict[str, Any]] = []
    filter_version = {
        "can": CAN_FILTER_VERSION,
        "transport": TRANSPORT_FILTER_VERSION,
        "tool_hang": TOOL_HANG_FILTER_VERSION,
    }.get(args.task, GENERIC_FILTER_VERSION)
    replay_env = (
        create_state_replay_env(args.source)
        if args.task in ("can", "transport", "tool_hang")
        else None
    )
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        success_keys = read_mask(source, args.success_mask)
        failure_keys = read_mask(source, args.failure_mask)
        overlap = sorted(set(success_keys).intersection(failure_keys), key=demo_sort_key)
        if overlap:
            raise ValueError(f"success and failure masks overlap: {overlap[:10]}")

        reference = None
        scale = None
        if args.task == "can":
            calibration_keys = success_keys[: int(args.can_success_calibration_limit)]
            for calibration_index, source_key in enumerate(calibration_keys):
                signals = replay_can_privileged_signals(
                    replay_env,
                    source[f"data/{source_key}"],
                )
                for candidate in select_can_chunks(signals=signals, args=args):
                    success_calibration_records.append(
                        {"source_demo": source_key, **candidate}
                    )
                if (calibration_index + 1) % 25 == 0:
                    print(
                        "calibrated Can filter on "
                        f"{calibration_index + 1}/{len(calibration_keys)} successes",
                        flush=True,
                    )
        elif args.task == "transport":
            calibration_keys = success_keys[
                : int(args.transport_success_calibration_limit)
            ]
            for calibration_index, source_key in enumerate(calibration_keys):
                signals = replay_transport_privileged_signals(
                    replay_env,
                    source[f"data/{source_key}"],
                )
                for candidate in select_transport_chunks(
                    signals=signals,
                    args=args,
                ):
                    success_calibration_records.append(
                        {"source_demo": source_key, **candidate}
                    )
                if (calibration_index + 1) % 25 == 0:
                    print(
                        "calibrated Transport filter on "
                        f"{calibration_index + 1}/{len(calibration_keys)} successes",
                        flush=True,
                    )
        elif args.task == "tool_hang":
            calibration_keys = success_keys[
                : int(args.tool_hang_success_calibration_limit)
            ]
            for calibration_index, source_key in enumerate(calibration_keys):
                signals = replay_tool_hang_privileged_signals(
                    replay_env,
                    source[f"data/{source_key}"],
                )
                for candidate in select_tool_hang_chunks(
                    signals=signals,
                    args=args,
                ):
                    success_calibration_records.append(
                        {"source_demo": source_key, **candidate}
                    )
                if (calibration_index + 1) % 25 == 0:
                    print(
                        "calibrated Tool Hang filter on "
                        f"{calibration_index + 1}/{len(calibration_keys)} successes",
                        flush=True,
                    )
        else:
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
            if args.task == "can":
                signals = replay_can_privileged_signals(replay_env, source_group)
                selected_records = select_can_chunks(signals=signals, args=args)
            elif args.task == "transport":
                signals = replay_transport_privileged_signals(
                    replay_env,
                    source_group,
                )
                selected_records = select_transport_chunks(
                    signals=signals,
                    args=args,
                )
            elif args.task == "tool_hang":
                signals = replay_tool_hang_privileged_signals(
                    replay_env,
                    source_group,
                )
                selected_records = select_tool_hang_chunks(
                    signals=signals,
                    args=args,
                )
            else:
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
                step_to_index = {
                    int(step): index for index, step in enumerate(steps)
                }
                selected_records = []
                for boundary in selected:
                    local_index = step_to_index[int(boundary)]
                    selected_records.append(
                        {
                            "decision_boundary": int(boundary),
                            "target_end_exclusive": int(
                                boundary + args.prediction_horizon
                            ),
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

            for selected_record in selected_records:
                boundary = int(selected_record["decision_boundary"])
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
                if args.task == "can":
                    output_group.attrs["segment_type"] = (
                        f"can_gt_good_{selected_record['type']}"
                    )
                    output_group.attrs["segment_metrics_json"] = json.dumps(
                        selected_record,
                        sort_keys=True,
                    )
                elif args.task == "transport":
                    output_group.attrs["segment_type"] = (
                        f"transport_gt_good_{selected_record['type']}"
                    )
                    output_group.attrs["segment_metrics_json"] = json.dumps(
                        selected_record,
                        sort_keys=True,
                    )
                elif args.task == "tool_hang":
                    output_group.attrs["segment_type"] = (
                        f"tool_hang_gt_good_{selected_record['type']}"
                    )
                    output_group.attrs["segment_metrics_json"] = json.dumps(
                        selected_record,
                        sort_keys=True,
                    )
                else:
                    output_group.attrs["segment_type"] = (
                        f"{args.task}_gt_good_failure_chunk"
                    )
                    for metric_key in (
                        "privileged_goal_progress",
                        "privileged_normalized_displacement",
                        "privileged_goal_distance_start",
                        "privileged_goal_distance_end",
                    ):
                        output_group.attrs[metric_key] = selected_record[metric_key]

                output_keys.append(output_key)
                retained_source_rollouts.add(source_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "source_num_samples": num_samples,
                        **selected_record,
                    }
                )
            if (failure_index + 1) % 25 == 0:
                print(
                    f"processed {failure_index + 1}/{len(failure_keys)} failures; "
                    f"retained {len(records)} chunks",
                    flush=True,
                )

        if not records:
            raise RuntimeError("privileged good-failure filter retained no chunks")
        output_data.attrs["env_args"] = source["data"].attrs["env_args"]
        output_data.attrs["total"] = len(output_keys) * (
            int(args.prediction_horizon) + 1
        )
        mask = output.create_group("mask")
        encoded_keys = np.asarray(output_keys, dtype="S")
        mask["all"] = encoded_keys
        mask["gt_good_failure"] = encoded_keys
        output.attrs["task"] = args.task
        output.attrs["filter_version"] = filter_version
        output.attrs["source_path"] = str(args.source)
        output.attrs["prediction_horizon"] = int(args.prediction_horizon)
        selection_definitions = {
            "can": (
                "exact simulator-replayed Can reach, grasp, lift, and "
                "transport stages"
            ),
            "transport": (
                "exact simulator-replayed two-arm lid, trash, and payload "
                "stages with concurrent-branch safety"
            ),
            "tool_hang": (
                "exact simulator-replayed frame assembly and tool-hook "
                "stages with prerequisite preservation"
            ),
        }
        output.attrs["selection_definition"] = selection_definitions.get(
            args.task,
            "privileged object-state movement toward successful terminal task states",
        )
        output.attrs["source_success_mask"] = args.success_mask
        output.attrs["source_failure_mask"] = args.failure_mask
        audit_source_match(source, output, records, int(args.prediction_horizon))

    starts = np.asarray(
        [record["decision_boundary"] for record in records],
        dtype=np.int64,
    )
    summary = {
        "task": args.task,
        "filter_version": filter_version,
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
            "raw_sequence_rows": int(args.prediction_horizon) + 1,
            "dp_loss_action_rows": [0, int(args.prediction_horizon)],
            "execution_aligned_start_row": 1,
        },
        "chunks": records,
    }
    if args.task == "can":
        summary["privileged_feature"] = {
            "definition": (
                "simulator-replayed staged rewards, exact grasp, object pose, "
                "gripper distance, correct-bin distance, and placement predicate"
            ),
            "state_action_alignment": (
                "core actions t:t+16 are validated with pre-action states "
                "t:t+17; stored row t-1 supplies the pretrained DP alignment "
                "target at model index 0, and executed targets begin at row t"
            ),
        }
        summary["selection"] = {
            "stride": int(args.stride),
            "max_chunks_per_failure": int(args.max_chunks_per_failure),
            "max_chunks_per_stage_per_failure": int(
                args.can_max_chunks_per_stage_per_failure
            ),
            "minimum_spacing": int(args.minimum_spacing),
            "prediction_horizon": int(args.prediction_horizon),
            "requires_successor_state_after_final_action": True,
            "safe_reach": {
                "min_start_distance": float(args.can_safe_min_start_distance),
                "min_distance": float(args.can_safe_min_distance),
                "min_reach_gain": float(args.can_safe_min_reach_gain),
                "min_progress_fraction": float(
                    args.can_safe_min_progress_fraction
                ),
                "progress_tolerance": float(args.can_safe_progress_tolerance),
                "max_regression": float(args.can_safe_max_regression),
                "max_can_displacement": float(
                    args.can_safe_max_can_displacement
                ),
            },
            "grasp_lift": {
                "min_final_grasp_frames": int(args.can_grasp_min_frames),
                "min_lift_gain": float(args.can_grasp_min_lift_gain),
                "max_drop": float(args.can_grasp_max_drop),
            },
            "transport": {
                "min_grasp_fraction": float(
                    args.can_transport_min_grasp_fraction
                ),
                "min_lift_height": float(args.can_transport_min_lift_height),
                "min_bin_progress": float(args.can_transport_min_bin_progress),
                "min_progress_fraction": float(
                    args.can_transport_min_progress_fraction
                ),
                "progress_tolerance": float(
                    args.can_transport_progress_tolerance
                ),
                "max_regression": float(args.can_transport_max_regression),
                "max_drop": float(args.can_transport_max_drop),
            },
        }
        summary["retained_chunk_stats"] = can_record_summary(records)
        summary["success_calibration"] = {
            "rollouts_checked": min(
                len(success_keys),
                int(args.can_success_calibration_limit),
            ),
            **can_record_summary(success_calibration_records),
        }
    elif args.task == "transport":
        summary["privileged_feature"] = {
            "definition": (
                "simulator-replayed exact robot0-lid, robot0-payload, and "
                "robot1-trash grasps; object and bin poses; target-bin "
                "contacts; per-arm gripper distances"
            ),
            "joint_action_safety": (
                "every retained 14-D two-arm action window validates the "
                "concurrent non-primary branch; payload stages additionally "
                "require stable trash completion and lid clearance"
            ),
            "state_action_alignment": (
                "core actions t:t+16 are validated with pre-action states "
                "t:t+17; stored row t-1 supplies the pretrained DP alignment "
                "target at model index 0, and executed targets begin at row t"
            ),
        }
        summary["selection"] = {
            "stride": int(args.stride),
            "max_chunks_per_failure": int(args.max_chunks_per_failure),
            "max_chunks_per_stage_per_failure": int(
                args.transport_max_chunks_per_stage_per_failure
            ),
            "minimum_spacing": int(args.minimum_spacing),
            "prediction_horizon": int(args.prediction_horizon),
            "requires_successor_state_after_final_action": True,
            "payload_requires_stable_trash_completion": True,
            "payload_requires_stable_lid_clearance": True,
            "safe_reach": {
                "min_start_distance": float(
                    args.transport_safe_min_start_distance
                ),
                "min_distance": float(args.transport_safe_min_distance),
                "min_reach_gain": float(
                    args.transport_safe_min_reach_gain
                ),
                "min_progress_fraction": float(
                    args.transport_safe_min_progress_fraction
                ),
                "progress_tolerance": float(
                    args.transport_safe_progress_tolerance
                ),
                "max_regression": float(
                    args.transport_safe_max_regression
                ),
            },
            "lid": {
                "min_clearance": float(args.transport_lid_min_clearance),
                "min_clearance_gain": float(
                    args.transport_lid_min_clearance_gain
                ),
                "max_clearance_regression": float(
                    args.transport_lid_max_clearance_regression
                ),
                "max_drop": float(args.transport_lid_max_drop),
            },
            "grasp_lift": {
                "min_final_grasp_frames": int(
                    args.transport_grasp_min_frames
                ),
                "min_lift_gain": float(
                    args.transport_grasp_min_lift_gain
                ),
                "max_drop": float(args.transport_grasp_max_drop),
            },
            "target_transport": {
                "min_grasp_fraction": float(
                    args.transport_target_min_grasp_fraction
                ),
                "min_lift_height": float(
                    args.transport_target_min_lift_height
                ),
                "min_bin_progress": float(
                    args.transport_target_min_bin_progress
                ),
                "min_progress_fraction": float(
                    args.transport_target_min_progress_fraction
                ),
                "progress_tolerance": float(
                    args.transport_target_progress_tolerance
                ),
                "max_regression": float(
                    args.transport_target_max_regression
                ),
                "max_drop": float(args.transport_target_max_drop),
            },
            "concurrent_branch": {
                "max_static_object_displacement": float(
                    args.transport_secondary_max_static_displacement
                ),
                "min_grasp_fraction_for_moving_object": float(
                    args.transport_secondary_min_grasp_fraction
                ),
            },
        }
        summary["retained_chunk_stats"] = transport_record_summary(records)
        summary["success_calibration"] = {
            "rollouts_checked": min(
                len(success_keys),
                int(args.transport_success_calibration_limit),
            ),
            **transport_record_summary(success_calibration_records),
        }
    elif args.task == "tool_hang":
        summary["privileged_feature"] = {
            "definition": (
                "simulator-replayed exact frame-grip and tool-grip grasps, "
                "frame assembly predicate components, tool-hole / hook-line "
                "geometry, contact, insertion, object poses, and gripper distance"
            ),
            "stage_prerequisites": (
                "all tool-stage windows require the frame assembly predicate "
                "for all 17 successor-validated states and a stationary frame"
            ),
            "state_action_alignment": (
                "core actions t:t+16 are validated with pre-action states "
                "t:t+17; stored row t-1 supplies the pretrained DP alignment "
                "target at model index 0, and executed targets begin at row t"
            ),
        }
        summary["selection"] = {
            "stride": int(args.stride),
            "max_chunks_per_failure": int(args.max_chunks_per_failure),
            "max_chunks_per_stage_per_failure": int(
                args.tool_hang_max_chunks_per_stage_per_failure
            ),
            "minimum_spacing": int(args.minimum_spacing),
            "prediction_horizon": int(args.prediction_horizon),
            "requires_successor_state_after_final_action": True,
            "tool_stages_require_stable_frame_assembly": True,
            "safe_reach": {
                "min_start_distance": float(
                    args.tool_hang_safe_min_start_distance
                ),
                "min_distance": float(args.tool_hang_safe_min_distance),
                "min_reach_gain": float(args.tool_hang_safe_min_reach_gain),
                "min_progress_fraction": float(
                    args.tool_hang_safe_min_progress_fraction
                ),
                "max_regression": float(
                    args.tool_hang_safe_max_regression
                ),
            },
            "grasp_lift": {
                "min_final_grasp_frames": int(
                    args.tool_hang_grasp_min_frames
                ),
                "min_lift_gain": float(
                    args.tool_hang_grasp_min_lift_gain
                ),
                "max_drop": float(args.tool_hang_grasp_max_drop),
            },
            "transport": {
                "min_grasp_fraction": float(
                    args.tool_hang_transport_min_grasp_fraction
                ),
                "min_lift_height": float(
                    args.tool_hang_transport_min_lift_height
                ),
                "min_progress": float(
                    args.tool_hang_transport_min_progress
                ),
                "min_progress_fraction": float(
                    args.tool_hang_transport_min_progress_fraction
                ),
                "progress_tolerance": float(
                    args.tool_hang_progress_tolerance
                ),
                "max_regression": float(
                    args.tool_hang_transport_max_regression
                ),
                "max_drop": float(args.tool_hang_transport_max_drop),
            },
            "frame_insert": {
                "max_insertion_distance": float(
                    args.tool_hang_frame_insert_max_distance
                ),
                "score_bonus": float(
                    args.tool_hang_frame_insert_score_bonus
                ),
            },
            "tool_hook_align": {
                "max_endpoint_distance": float(
                    args.tool_hang_align_max_endpoint_distance
                ),
                "min_hook_error_progress": float(
                    args.tool_hang_align_min_error_progress
                ),
                "min_progress_fraction": float(
                    args.tool_hang_align_min_progress_fraction
                ),
                "max_regression": float(
                    args.tool_hang_align_max_regression
                ),
                "hook_contact_score_bonus": float(
                    args.tool_hang_hook_contact_score_bonus
                ),
            },
            "max_static_object_displacement": float(
                args.tool_hang_max_static_object_displacement
            ),
        }
        summary["retained_chunk_stats"] = tool_hang_record_summary(records)
        summary["success_calibration"] = {
            "rollouts_checked": min(
                len(success_keys),
                int(args.tool_hang_success_calibration_limit),
            ),
            **tool_hang_record_summary(success_calibration_records),
        }
    else:
        progress = np.asarray(
            [record["privileged_goal_progress"] for record in records],
            dtype=np.float64,
        )
        displacement = np.asarray(
            [record["privileged_normalized_displacement"] for record in records],
            dtype=np.float64,
        )
        summary["privileged_feature"] = {
            "definition": (
                "task-specific object position / goal-relative coordinates and stage bits"
            ),
            "successful_endpoint_reference": reference.tolist(),
            "robust_scale": scale.tolist(),
            "position_scale_floor": float(args.position_scale_floor),
        }
        summary["selection"] = {
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
        }
        summary["retained_chunk_stats"] = {
            "start_step": stats(starts),
            "goal_progress": stats(progress),
            "normalized_displacement": stats(displacement),
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
    parser.add_argument("--can-success-calibration-limit", type=int, default=100)
    parser.add_argument(
        "--can-max-chunks-per-stage-per-failure",
        type=int,
        default=2,
    )
    parser.add_argument("--can-safe-min-start-distance", type=float, default=0.10)
    parser.add_argument("--can-safe-min-distance", type=float, default=0.05)
    parser.add_argument("--can-safe-min-reach-gain", type=float, default=0.04)
    parser.add_argument(
        "--can-safe-min-progress-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--can-safe-progress-tolerance",
        type=float,
        default=0.001,
    )
    parser.add_argument("--can-safe-max-regression", type=float, default=0.015)
    parser.add_argument(
        "--can-safe-max-can-displacement",
        type=float,
        default=0.01,
    )
    parser.add_argument("--can-grasp-min-frames", type=int, default=6)
    parser.add_argument("--can-grasp-min-lift-gain", type=float, default=0.025)
    parser.add_argument("--can-grasp-max-drop", type=float, default=0.015)
    parser.add_argument(
        "--can-transport-min-grasp-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument("--can-transport-min-lift-height", type=float, default=0.04)
    parser.add_argument("--can-transport-min-bin-progress", type=float, default=0.04)
    parser.add_argument(
        "--can-transport-min-progress-fraction",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--can-transport-progress-tolerance",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--can-transport-max-regression",
        type=float,
        default=0.025,
    )
    parser.add_argument("--can-transport-max-drop", type=float, default=0.025)
    parser.add_argument(
        "--transport-success-calibration-limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--transport-max-chunks-per-stage-per-failure",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--transport-safe-min-start-distance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--transport-safe-min-distance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--transport-safe-min-reach-gain",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--transport-safe-min-progress-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--transport-safe-progress-tolerance",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--transport-safe-max-regression",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--transport-lid-min-clearance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--transport-lid-min-clearance-gain",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--transport-lid-max-clearance-regression",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--transport-lid-max-drop",
        type=float,
        default=0.04,
    )
    parser.add_argument("--transport-grasp-min-frames", type=int, default=6)
    parser.add_argument(
        "--transport-grasp-min-lift-gain",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--transport-grasp-max-drop",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--transport-target-min-grasp-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--transport-target-min-lift-height",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--transport-target-min-bin-progress",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--transport-target-min-progress-fraction",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--transport-target-progress-tolerance",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--transport-target-max-regression",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--transport-target-max-drop",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--transport-place-min-bin-progress",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--transport-place-score-bonus",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--transport-secondary-max-static-displacement",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--transport-secondary-min-grasp-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--tool-hang-success-calibration-limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--tool-hang-max-chunks-per-stage-per-failure",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tool-hang-safe-min-start-distance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--tool-hang-safe-min-distance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--tool-hang-safe-min-reach-gain",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--tool-hang-safe-min-progress-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--tool-hang-safe-max-regression",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--tool-hang-progress-tolerance",
        type=float,
        default=0.001,
    )
    parser.add_argument("--tool-hang-grasp-min-frames", type=int, default=6)
    parser.add_argument(
        "--tool-hang-grasp-min-lift-gain",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--tool-hang-grasp-max-drop",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--tool-hang-transport-min-grasp-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--tool-hang-transport-min-lift-height",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--tool-hang-transport-min-progress",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--tool-hang-transport-min-progress-fraction",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--tool-hang-transport-max-regression",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--tool-hang-transport-max-drop",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--tool-hang-frame-insert-max-distance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--tool-hang-frame-insert-score-bonus",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--tool-hang-align-max-endpoint-distance",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--tool-hang-align-min-error-progress",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--tool-hang-align-min-progress-fraction",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--tool-hang-align-max-regression",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--tool-hang-hook-contact-score-bonus",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--tool-hang-max-static-object-displacement",
        type=float,
        default=0.01,
    )
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
    if args.can_success_calibration_limit < 0:
        parser.error("can-success-calibration-limit must be non-negative")
    if args.can_max_chunks_per_stage_per_failure <= 0:
        parser.error("can-max-chunks-per-stage-per-failure must be positive")
    if args.transport_success_calibration_limit < 0:
        parser.error("transport-success-calibration-limit must be non-negative")
    if args.transport_max_chunks_per_stage_per_failure <= 0:
        parser.error(
            "transport-max-chunks-per-stage-per-failure must be positive"
        )
    if args.transport_grasp_min_frames <= 0:
        parser.error("transport-grasp-min-frames must be positive")
    if args.tool_hang_success_calibration_limit < 0:
        parser.error("tool-hang-success-calibration-limit must be non-negative")
    if args.tool_hang_max_chunks_per_stage_per_failure <= 0:
        parser.error(
            "tool-hang-max-chunks-per-stage-per-failure must be positive"
        )
    if args.tool_hang_grasp_min_frames <= 0:
        parser.error("tool-hang-grasp-min-frames must be positive")
    for key in (
        "can_safe_min_progress_fraction",
        "can_transport_min_grasp_fraction",
        "can_transport_min_progress_fraction",
        "transport_safe_min_progress_fraction",
        "transport_target_min_grasp_fraction",
        "transport_target_min_progress_fraction",
        "transport_secondary_min_grasp_fraction",
        "tool_hang_safe_min_progress_fraction",
        "tool_hang_transport_min_grasp_fraction",
        "tool_hang_transport_min_progress_fraction",
        "tool_hang_align_min_progress_fraction",
    ):
        if not 0.0 <= float(getattr(args, key)) <= 1.0:
            parser.error(f"{key.replace('_', '-')} must be in [0, 1]")
    if args.max_start_fraction is not None and not 0.0 <= args.max_start_fraction <= 1.0:
        parser.error("max-start-fraction must be in [0, 1]")
    return args


if __name__ == "__main__":
    build(parse_args())
