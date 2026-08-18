#!/usr/bin/env python3
"""Build the QA-eligible, round-1 pick-cup dataset on a true 5 Hz grid.

The source contains normalized commands and robot state at 20 Hz and paired RGB
at 5 Hz. Each output row starts at source action index ``2 + 4k`` and represents
the complete four-command interval beginning there. Motion command components
are averaged, while their physical decode scales are multiplied by four. The
gripper target is the dense logical state after the final command in the bin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot.build_pick_cup_dataset import (  # noqa: E402
    EpisodePayload,
    _compression_kwargs,
    _load_resized_rgb,
    load_episode_payload,
)
from scripts.real_robot.pick_cup_common import (  # noqa: E402
    CONVERSION_MANIFEST_ATTR,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MAX_IMAGE_AGE_SEC,
    DEFAULT_SOURCE,
    LOW_DIM_KEYS,
    RGB_KEYS,
    EpisodeIndexRow,
    atomic_write_json,
    decode_hdf5_strings,
    eligible_rows,
    load_json,
    resolve_episode_dir,
    select_spatial_validation_rows,
    source_identity,
)


DEFAULT_5HZ_DATASET_DIR = ROOT / "datasets/real_robot/pick_cup_5hz_round1"
OUTPUT_FILENAME = "round1_5hz_rgb.hdf5"
SUMMARY_FILENAME = "conversion_summary.json"
COMMIT_FILENAME = "dataset_commit.json"

SCHEMA_VERSION = 1
CONVERSION_VERSION = "pick_cup_rgb_dp_round1_5hz_v1"
SOURCE_HZ = 20.0
CONTROL_HZ = 5.0
SOURCE_STRIDE = 4
SOURCE_PHASE = 2
SOURCE_TRANSLATION_SCALE_M = 0.012
SOURCE_ROTATION_SCALE_RAD = 0.036
TRANSLATION_SCALE_M = SOURCE_STRIDE * SOURCE_TRANSLATION_SCALE_M
ROTATION_SCALE_RAD = SOURCE_STRIDE * SOURCE_ROTATION_SCALE_RAD
ROTATION_AGGREGATION = "fixed_axis_component_mean"


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_5HZ_DATASET_DIR
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    validation_count: int = 5
    split_seed: int = 1
    max_image_age_sec: float = DEFAULT_MAX_IMAGE_AGE_SEC
    compression: str = "gzip"
    overwrite: bool = False
    validate_only: bool = False
    require_controller_manifest: bool = True


@dataclass
class MacroEpisode:
    row: EpisodeIndexRow
    source: EpisodePayload
    starts_local: np.ndarray
    starts_raw: np.ndarray
    actions: np.ndarray
    eef_poses: np.ndarray
    gripper_observations: np.ndarray
    selected_frame_positions: np.ndarray
    selected_frame_indices: np.ndarray
    selected_frame_nominal_times: np.ndarray
    selected_main_capture_times: np.ndarray
    selected_wrist_capture_times: np.ndarray
    image_ages: np.ndarray
    source_actions_20hz: np.ndarray
    source_dense_actions_20hz: np.ndarray
    source_gripper_observations: np.ndarray
    source_before_poses: np.ndarray
    source_raw_gripper_events: np.ndarray
    source_action_target_times: np.ndarray
    source_action_source_times: np.ndarray
    source_action_indices: np.ndarray
    source_action_steps: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.actions.shape[0])


def dataset_path(output_dir: Path) -> Path:
    return Path(output_dir).expanduser().resolve() / OUTPUT_FILENAME


def _json_attr(attrs: h5py.AttributeManager, key: str, *, location: str) -> Any:
    raw = attrs.get(key)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"{location} is missing JSON attribute {key}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} has malformed JSON attribute {key}") from exc


def _verify_controller_manifest(
    source_root: Path,
    row: EpisodeIndexRow,
    *,
    required: bool,
) -> None:
    manifest_path = (
        resolve_episode_dir(source_root, row)
        / "snapshots"
        / "collector_manifest.json"
    )
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError(
                f"episode {row.episode_number} lacks controller manifest: "
                f"{manifest_path}"
            )
        return
    manifest = load_json(manifest_path)
    expected_scale = np.asarray(
        [SOURCE_TRANSLATION_SCALE_M, SOURCE_ROTATION_SCALE_RAD, 1.0]
    )
    actual_scale = np.asarray(manifest.get("action_scale", ()), dtype=np.float64)
    if actual_scale.shape != (3,) or not np.allclose(
        actual_scale, expected_scale, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"episode {row.episode_number} action scale is incompatible: "
            f"{actual_scale.tolist()}"
        )
    if not np.isclose(float(manifest.get("hz", np.nan)), SOURCE_HZ):
        raise ValueError(f"episode {row.episode_number} is not a 20 Hz source")
    axis_map = manifest.get("axis_map", {})
    expected_axes = [0, 1, 2]
    expected_signs = [1.0, 1.0, 1.0]
    for prefix in ("pos", "rot"):
        signs = np.asarray(axis_map.get(f"{prefix}_signs", ()), dtype=np.float64)
        if (
            axis_map.get(f"{prefix}_axes") != expected_axes
            or signs.shape != (3,)
            or not np.allclose(
                signs,
                expected_signs,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"episode {row.episode_number} has incompatible {prefix} axis map"
            )
    if axis_map.get("status") != "PASS":
        raise ValueError(
            f"episode {row.episode_number} controller axis map is not verified"
        )


def _source_block(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.stack([values[start : start + SOURCE_STRIDE] for start in starts])


def _macro_episode(
    source_root: Path,
    row: EpisodeIndexRow,
    *,
    max_image_age_sec: float,
    require_controller_manifest: bool,
) -> MacroEpisode:
    _verify_controller_manifest(
        source_root,
        row,
        required=require_controller_manifest,
    )
    source = load_episode_payload(
        source_root,
        row,
        max_image_age_sec=max_image_age_sec,
    )
    raw_length = source.dropped_prefix_actions + source.num_samples
    starts_raw = np.arange(
        SOURCE_PHASE,
        raw_length - SOURCE_STRIDE + 1,
        SOURCE_STRIDE,
        dtype=np.int64,
    )
    starts_raw = starts_raw[starts_raw >= source.dropped_prefix_actions]
    starts_local = starts_raw - source.dropped_prefix_actions
    if starts_local.size == 0:
        raise ValueError(
            f"episode {row.episode_number} has no complete causal 5 Hz action bin"
        )
    if np.any(starts_local < 0) or np.any(
        starts_local + SOURCE_STRIDE > source.num_samples
    ):
        raise ValueError(f"episode {row.episode_number} has an invalid source bin")

    dense_blocks = _source_block(source.actions, starts_local)
    motion = np.mean(dense_blocks[:, :, :6], axis=1, dtype=np.float64).astype(
        np.float32
    )
    if np.any(motion < -1.000001) or np.any(motion > 1.000001):
        raise ValueError(f"episode {row.episode_number} macro motion exceeds [-1,1]")
    actions = np.concatenate(
        [motion, dense_blocks[:, -1, 6:7].astype(np.float32)],
        axis=1,
    )
    raw_blocks = _source_block(source.raw_gripper_events, starts_local)
    raw_action_blocks = dense_blocks.copy()
    raw_action_blocks[:, :, 6] = raw_blocks
    gripper_blocks = _source_block(source.gripper_observations, starts_local)
    pose_blocks = _source_block(source.eef_poses, starts_local)
    target_time_blocks = _source_block(source.action_target_times, starts_local)
    source_time_blocks = _source_block(source.action_source_times, starts_local)
    index_blocks = _source_block(source.action_source_indices, starts_local)
    step_blocks = _source_block(source.action_steps, starts_local)

    selected = source.selected_frame_positions[starts_local]
    main_times = source.selected_main_capture_times[starts_local]
    wrist_times = source.selected_wrist_capture_times[starts_local]
    ages = target_time_blocks[:, 0] - np.maximum(main_times, wrist_times)
    if np.any(ages < -1e-6):
        raise ValueError(f"episode {row.episode_number} selected a future RGB pair")
    if float(np.max(ages)) > max_image_age_sec + 1e-6:
        raise ValueError(f"episode {row.episode_number} exceeds image-age limit")

    return MacroEpisode(
        row=row,
        source=source,
        starts_local=starts_local,
        starts_raw=starts_raw,
        actions=actions,
        eef_poses=pose_blocks[:, 0].astype(np.float32),
        gripper_observations=gripper_blocks[:, 0].astype(np.float32),
        selected_frame_positions=selected.astype(np.int64),
        selected_frame_indices=source.selected_frame_indices[starts_local].astype(
            np.int64
        ),
        selected_frame_nominal_times=source.selected_frame_nominal_times[
            starts_local
        ],
        selected_main_capture_times=main_times,
        selected_wrist_capture_times=wrist_times,
        image_ages=ages,
        source_actions_20hz=raw_action_blocks.astype(np.float32),
        source_dense_actions_20hz=dense_blocks.astype(np.float32),
        source_gripper_observations=gripper_blocks.astype(np.float32),
        source_before_poses=pose_blocks.astype(np.float32),
        source_raw_gripper_events=raw_blocks.astype(np.float32),
        source_action_target_times=target_time_blocks,
        source_action_source_times=source_time_blocks,
        source_action_indices=index_blocks.astype(np.int64),
        source_action_steps=step_blocks.astype(np.int64),
    )


def _write_images(
    obs: h5py.Group,
    payload: MacroEpisode,
    *,
    image_height: int,
    image_width: int,
    compression: str,
) -> None:
    count = payload.num_samples
    chunks = (min(8, count), image_height, image_width, 3)
    kwargs = _compression_kwargs(compression)
    outputs = {
        "main_image": obs.create_dataset(
            "main_image",
            shape=(count, image_height, image_width, 3),
            dtype=np.uint8,
            chunks=chunks,
            **kwargs,
        ),
        "wrist_image": obs.create_dataset(
            "wrist_image",
            shape=(count, image_height, image_width, 3),
            dtype=np.uint8,
            chunks=chunks,
            **kwargs,
        ),
    }
    used = sorted({int(value) for value in payload.selected_frame_positions})
    caches: dict[str, dict[int, np.ndarray]] = {
        "main_image": {},
        "wrist_image": {},
    }
    for position in used:
        expected_size = payload.source.frame_source_sizes[position]
        caches["main_image"][position] = _load_resized_rgb(
            payload.source.main_paths[position],
            expected_size=expected_size,
            output_size=(image_width, image_height),
        )
        caches["wrist_image"][position] = _load_resized_rgb(
            payload.source.wrist_paths[position],
            expected_size=expected_size,
            output_size=(image_width, image_height),
        )
    for start in range(0, count, 128):
        end = min(count, start + 128)
        positions = payload.selected_frame_positions[start:end]
        for key, output in outputs.items():
            output[start:end] = np.stack(
                [caches[key][int(position)] for position in positions]
            )


def _manifest(
    options: BuildOptions,
    payloads: list[MacroEpisode],
    validation_numbers: set[int],
    generation_id: str,
) -> dict[str, Any]:
    train_keys = [
        payload.row.demo_key
        for payload in payloads
        if payload.row.episode_number not in validation_numbers
    ]
    valid_keys = [
        payload.row.demo_key
        for payload in payloads
        if payload.row.episode_number in validation_numbers
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "conversion_version": CONVERSION_VERSION,
        "generation_id": generation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity(options.source_root),
        "subset": {
            "collection_round": 1,
            "eligibility": "training_eligible_43qa=true",
            "episode_numbers": [payload.row.episode_number for payload in payloads],
        },
        "controller_contract": {
            "collector_manifests_required": options.require_controller_manifest,
            "verified_episode_count": (
                len(payloads) if options.require_controller_manifest else 0
            ),
            "source_translation_scale_m": SOURCE_TRANSLATION_SCALE_M,
            "source_rotation_scale_rad": SOURCE_ROTATION_SCALE_RAD,
            "axis_order": [0, 1, 2],
            "axis_signs": [1.0, 1.0, 1.0],
        },
        "downsampling": {
            "source_hz": SOURCE_HZ,
            "target_hz": CONTROL_HZ,
            "stride_20hz_rows": SOURCE_STRIDE,
            "phase_20hz_row": SOURCE_PHASE,
            "complete_bins_only": True,
            "observation": "source state at first row of each four-row bin",
        },
        "alignment": {
            "master_clock": "actions.samples.target_time",
            "image_availability": (
                "max(main_rgb.header_stamp_ns,wrist_rgb.header_stamp_ns)"
            ),
            "selection": "latest paired capture not later than bin start",
            "max_image_age_sec": options.max_image_age_sec,
        },
        "action": {
            "hz": CONTROL_HZ,
            "source_hz": SOURCE_HZ,
            "translation_scale_m": TRANSLATION_SCALE_M,
            "rotation_scale_rad": ROTATION_SCALE_RAD,
            "motion_aggregation": "componentwise mean of four normalized commands",
            "rotation_aggregation": ROTATION_AGGREGATION,
            "normalization": "motion and dense gripper in [-1,1]",
        },
        "gripper": {
            "raw": "sparse event: -1 close, 0 hold, +1 open",
            "observation": "logical state before first command in bin",
            "action": "dense logical target after fourth command in bin",
            "initial_state": "+1 open",
        },
        "image": {
            "source_hz": CONTROL_HZ,
            "height": options.image_height,
            "width": options.image_width,
            "layout": "HWC RGB uint8",
            "resize": "Pillow Lanczos",
            "compression": options.compression,
        },
        "split": {
            "method": "seeded max-min coverage over first-close EEF xy",
            "seed": options.split_seed,
            "requested_validation_count": options.validation_count,
            "train": train_keys,
            "valid": valid_keys,
        },
        "episodes": [
            {
                "demo_key": payload.row.demo_key,
                "episode_number": payload.row.episode_number,
                "num_samples": payload.num_samples,
                "split": (
                    "valid"
                    if payload.row.episode_number in validation_numbers
                    else "train"
                ),
                "max_image_age_sec": float(np.max(payload.image_ages)),
                "adjacent_rgb_repeat_count": int(
                    np.sum(
                        payload.selected_frame_positions[1:]
                        == payload.selected_frame_positions[:-1]
                    )
                ),
            }
            for payload in payloads
        ],
    }


def _write_hdf5(
    path: Path,
    options: BuildOptions,
    payloads: list[MacroEpisode],
    validation_numbers: set[int],
    generation_id: str,
) -> dict[str, Any]:
    manifest = _manifest(options, payloads, validation_numbers, generation_id)
    train_keys = manifest["split"]["train"]
    valid_keys = manifest["split"]["valid"]
    env_args = {
        "env_name": "PickCupReal5Hz-v0",
        "env_version": CONVERSION_VERSION,
        "type": 2,
        "env_kwargs": {
            "real_robot": True,
            "control_freq": int(CONTROL_HZ),
            "camera_names": ["main", "wrist"],
            "camera_height": options.image_height,
            "camera_width": options.image_width,
            "task": "pick_cup_place_on_plate",
            "translation_scale_m": TRANSLATION_SCALE_M,
            "rotation_scale_rad": ROTATION_SCALE_RAD,
        },
    }
    with h5py.File(path, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["conversion_version"] = CONVERSION_VERSION
        output.attrs["generation_id"] = generation_id
        output.attrs[CONVERSION_MANIFEST_ATTR] = json.dumps(manifest, sort_keys=True)
        data = output.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args, sort_keys=True)
        total = 0
        for payload in payloads:
            demo = data.create_group(payload.row.demo_key)
            demo.attrs["num_samples"] = payload.num_samples
            demo.attrs["source_episode_number"] = payload.row.episode_number
            demo.attrs["source_run_id"] = payload.row.run_id
            demo.attrs["source_directory"] = payload.row.directory
            demo.attrs["collection_round"] = 1
            demo.attrs["qa_status"] = payload.row.qa_status
            demo.attrs["source_phase_20hz_row"] = SOURCE_PHASE
            demo.attrs["source_stride_20hz_rows"] = SOURCE_STRIDE
            demo.attrs["close_position_xyz"] = payload.source.close_position
            demo.attrs["release_position_xyz"] = payload.source.release_position

            demo.create_dataset("actions", data=payload.actions.astype(np.float32))
            rewards = np.zeros(payload.num_samples, dtype=np.float32)
            rewards[-1] = 1.0
            dones = np.zeros(payload.num_samples, dtype=np.uint8)
            dones[-1] = 1
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)

            obs = demo.create_group("obs")
            obs.create_dataset(
                LOW_DIM_KEYS[0], data=payload.eef_poses[:, :3].astype(np.float32)
            )
            obs.create_dataset(
                LOW_DIM_KEYS[1], data=payload.eef_poses[:, 3:7].astype(np.float32)
            )
            obs.create_dataset(
                LOW_DIM_KEYS[2],
                data=payload.gripper_observations[:, None].astype(np.float32),
            )
            _write_images(
                obs,
                payload,
                image_height=options.image_height,
                image_width=options.image_width,
                compression=options.compression,
            )

            provenance = demo.create_group("provenance")
            provenance.create_dataset(
                "bin_start_action_array_index", data=payload.starts_raw
            )
            provenance.create_dataset(
                "bin_end_action_array_index",
                data=payload.starts_raw + SOURCE_STRIDE - 1,
            )
            provenance.create_dataset(
                "source_actions_20hz", data=payload.source_actions_20hz
            )
            provenance.create_dataset(
                "source_dense_actions_20hz",
                data=payload.source_dense_actions_20hz,
            )
            provenance.create_dataset(
                "source_gripper_observations",
                data=payload.source_gripper_observations,
            )
            provenance.create_dataset(
                "source_before_poses", data=payload.source_before_poses
            )
            provenance.create_dataset(
                "source_raw_gripper_events",
                data=payload.source_raw_gripper_events,
            )
            provenance.create_dataset(
                "source_action_target_times",
                data=payload.source_action_target_times,
            )
            provenance.create_dataset(
                "source_action_source_times",
                data=payload.source_action_source_times,
            )
            provenance.create_dataset(
                "source_action_indices", data=payload.source_action_indices
            )
            provenance.create_dataset(
                "source_action_steps", data=payload.source_action_steps
            )
            provenance.create_dataset(
                "selected_frame_position",
                data=payload.selected_frame_positions,
            )
            provenance.create_dataset(
                "selected_frame_index", data=payload.selected_frame_indices
            )
            provenance.create_dataset(
                "selected_frame_nominal_time",
                data=payload.selected_frame_nominal_times,
            )
            provenance.create_dataset(
                "selected_main_capture_time",
                data=payload.selected_main_capture_times,
            )
            provenance.create_dataset(
                "selected_wrist_capture_time",
                data=payload.selected_wrist_capture_times,
            )
            provenance.create_dataset("image_age_sec", data=payload.image_ages)
            total += payload.num_samples
        data.attrs["total"] = total
        mask = output.create_group("mask")
        all_keys = train_keys + valid_keys
        mask.create_dataset("all", data=np.asarray(all_keys, dtype="S"))
        mask.create_dataset("train", data=np.asarray(train_keys, dtype="S"))
        mask.create_dataset("valid", data=np.asarray(valid_keys, dtype="S"))
        output.flush()
    return manifest


def _validate_demo(
    demo: h5py.Group,
    *,
    image_height: int,
    image_width: int,
    max_image_age_sec: float,
) -> dict[str, Any]:
    count = int(demo.attrs.get("num_samples", -1))
    if count < 1:
        raise ValueError(f"{demo.name} has no samples")
    actions = np.asarray(demo["actions"][:], dtype=np.float32)
    if actions.shape != (count, 7) or not np.all(np.isfinite(actions)):
        raise ValueError(f"{demo.name} has invalid actions")
    if np.any(actions < -1.000001) or np.any(actions > 1.000001):
        raise ValueError(f"{demo.name} actions exceed [-1,1]")
    if not np.all(np.isin(actions[:, 6], (-1.0, 1.0))):
        raise ValueError(f"{demo.name} gripper targets are not dense signs")
    rewards = np.asarray(demo["rewards"][:])
    dones = np.asarray(demo["dones"][:])
    if rewards.shape != (count,) or dones.shape != (count,):
        raise ValueError(f"{demo.name} has invalid reward or done shape")
    if (
        not np.all(rewards[:-1] == 0.0)
        or float(rewards[-1]) != 1.0
        or not np.all(dones[:-1] == 0)
        or int(dones[-1]) != 1
    ):
        raise ValueError(f"{demo.name} has invalid terminal markers")
    for key in RGB_KEYS:
        dataset = demo[f"obs/{key}"]
        if dataset.shape != (count, image_height, image_width, 3):
            raise ValueError(f"{dataset.name} has invalid shape")
        if dataset.dtype != np.uint8:
            raise ValueError(f"{dataset.name} must be uint8")
    expected_low_dim = {
        LOW_DIM_KEYS[0]: (count, 3),
        LOW_DIM_KEYS[1]: (count, 4),
        LOW_DIM_KEYS[2]: (count, 1),
    }
    for key, shape in expected_low_dim.items():
        values = np.asarray(demo[f"obs/{key}"][:])
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{demo.name}/obs/{key} is invalid")
    quaternion = np.asarray(demo[f"obs/{LOW_DIM_KEYS[1]}"][:])
    if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=5e-3):
        raise ValueError(f"{demo.name} has non-unit EEF quaternion")

    provenance = demo["provenance"]
    starts = np.asarray(provenance["bin_start_action_array_index"][:])
    ends = np.asarray(provenance["bin_end_action_array_index"][:])
    if (
        starts.shape != (count,)
        or int(starts[0]) < SOURCE_PHASE
        or np.any((starts - SOURCE_PHASE) % SOURCE_STRIDE != 0)
        or np.any(np.diff(starts) != SOURCE_STRIDE)
    ):
        raise ValueError(f"{demo.name} has invalid fixed-rate bin starts")
    if not np.array_equal(ends, starts + SOURCE_STRIDE - 1):
        raise ValueError(f"{demo.name} has invalid bin ends")
    dense = np.asarray(provenance["source_dense_actions_20hz"][:])
    raw = np.asarray(provenance["source_actions_20hz"][:])
    raw_gripper = np.asarray(provenance["source_raw_gripper_events"][:])
    gripper_obs = np.asarray(provenance["source_gripper_observations"][:])
    poses = np.asarray(provenance["source_before_poses"][:])
    target_times = np.asarray(provenance["source_action_target_times"][:])
    expected_block_shape = (count, SOURCE_STRIDE)
    if dense.shape != (count, SOURCE_STRIDE, 7):
        raise ValueError(f"{demo.name} source dense actions have invalid shape")
    if raw.shape != (count, SOURCE_STRIDE, 7):
        raise ValueError(f"{demo.name} source actions have invalid shape")
    if raw_gripper.shape != expected_block_shape or not np.array_equal(
        raw[:, :, 6], raw_gripper
    ):
        raise ValueError(f"{demo.name} raw gripper provenance is invalid")
    if gripper_obs.shape != expected_block_shape:
        raise ValueError(f"{demo.name} source gripper observations are invalid")
    if poses.shape != (count, SOURCE_STRIDE, 7):
        raise ValueError(f"{demo.name} source poses are invalid")
    if target_times.shape != expected_block_shape:
        raise ValueError(f"{demo.name} source target times are invalid")
    if not np.allclose(np.diff(target_times, axis=1), 1.0 / SOURCE_HZ, atol=1e-6):
        raise ValueError(f"{demo.name} source bins are not on the 20 Hz grid")
    if count > 1 and not np.allclose(
        np.diff(target_times[:, 0]), 1.0 / CONTROL_HZ, atol=1e-6
    ):
        raise ValueError(f"{demo.name} macro rows are not on the 5 Hz grid")
    if not np.allclose(
        actions[:, :6], np.mean(dense[:, :, :6], axis=1), atol=1e-7
    ):
        raise ValueError(f"{demo.name} macro motion is not a four-row mean")
    if not np.array_equal(actions[:, 6], dense[:, -1, 6]):
        raise ValueError(f"{demo.name} macro gripper is not the final dense target")
    state = float(gripper_obs[0, 0])
    if state not in (-1.0, 1.0):
        raise ValueError(f"{demo.name} has an invalid initial gripper state")
    for block_index in range(count):
        for offset in range(SOURCE_STRIDE):
            if float(gripper_obs[block_index, offset]) != state:
                raise ValueError(f"{demo.name} gripper state provenance is inconsistent")
            event = float(raw_gripper[block_index, offset])
            if event < 0.0:
                state = -1.0
            elif event > 0.0:
                state = 1.0
            if float(dense[block_index, offset, 6]) != state:
                raise ValueError(f"{demo.name} dense gripper provenance is inconsistent")
    if not np.array_equal(
        demo[f"obs/{LOW_DIM_KEYS[2]}"][:, 0], gripper_obs[:, 0]
    ):
        raise ValueError(f"{demo.name} gripper observation is misaligned")
    if not np.allclose(demo[f"obs/{LOW_DIM_KEYS[0]}"][:], poses[:, 0, :3]):
        raise ValueError(f"{demo.name} EEF position is misaligned")
    if not np.allclose(demo[f"obs/{LOW_DIM_KEYS[1]}"][:], poses[:, 0, 3:7]):
        raise ValueError(f"{demo.name} EEF quaternion is misaligned")

    main_times = np.asarray(provenance["selected_main_capture_time"][:])
    wrist_times = np.asarray(provenance["selected_wrist_capture_time"][:])
    ages = np.asarray(provenance["image_age_sec"][:])
    control_times = target_times[:, 0]
    if np.any(main_times > control_times + 1e-6) or np.any(
        wrist_times > control_times + 1e-6
    ):
        raise ValueError(f"{demo.name} uses a future camera capture")
    expected_ages = control_times - np.maximum(main_times, wrist_times)
    if not np.allclose(ages, expected_ages, atol=1e-7):
        raise ValueError(f"{demo.name} image ages do not match timestamps")
    if float(np.min(ages)) < -1e-6 or float(np.max(ages)) > max_image_age_sec + 1e-6:
        raise ValueError(f"{demo.name} image age is outside configured bounds")
    frame_positions = np.asarray(provenance["selected_frame_position"][:])
    if frame_positions.shape != (count,) or np.any(np.diff(frame_positions) < 0):
        raise ValueError(f"{demo.name} frame mapping is invalid")
    terminal_motion_zero = bool(np.allclose(actions[-1, :6], 0.0, atol=1e-8))
    return {
        "demo_key": demo.name.rsplit("/", 1)[-1],
        "samples": count,
        "max_image_age_sec": float(np.max(ages)),
        "adjacent_rgb_repeats": int(
            np.sum(frame_positions[1:] == frame_positions[:-1])
        ),
        "terminal_motion_zero": terminal_motion_zero,
    }


def validate_5hz_dataset(
    path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"5 Hz dataset does not exist: {path}")
    with h5py.File(path, "r") as dataset:
        manifest = _json_attr(
            dataset.attrs,
            CONVERSION_MANIFEST_ATTR,
            location=str(path),
        )
        if manifest.get("conversion_version") != CONVERSION_VERSION:
            raise ValueError(f"{path} has incompatible conversion version")
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"{path} has incompatible schema version")
        if source_root is not None and manifest.get("source_identity") != source_identity(
            Path(source_root).expanduser().resolve()
        ):
            raise ValueError(f"{path} source identity does not match")
        image = manifest.get("image", {})
        action = manifest.get("action", {})
        downsampling = manifest.get("downsampling", {})
        if not np.isclose(float(action.get("hz", np.nan)), CONTROL_HZ):
            raise ValueError(f"{path} is not a 5 Hz action dataset")
        if not np.isclose(
            float(action.get("translation_scale_m", np.nan)),
            TRANSLATION_SCALE_M,
        ) or not np.isclose(
            float(action.get("rotation_scale_rad", np.nan)), ROTATION_SCALE_RAD
        ):
            raise ValueError(f"{path} has incorrect 5 Hz action scales")
        if action.get("rotation_aggregation") != ROTATION_AGGREGATION:
            raise ValueError(f"{path} has an unknown rotation aggregation contract")
        controller_contract = manifest.get("controller_contract", {})
        if controller_contract.get("collector_manifests_required") not in (
            True,
            False,
        ):
            raise ValueError(f"{path} lacks its controller verification contract")
        if int(downsampling.get("stride_20hz_rows", -1)) != SOURCE_STRIDE or int(
            downsampling.get("phase_20hz_row", -1)
        ) != SOURCE_PHASE:
            raise ValueError(f"{path} has an invalid downsampling grid")
        if "data" not in dataset or "mask" not in dataset:
            raise ValueError(f"{path} is missing data or masks")
        all_keys = decode_hdf5_strings(dataset["mask/all"][:])
        train_keys = decode_hdf5_strings(dataset["mask/train"][:])
        valid_keys = decode_hdf5_strings(dataset["mask/valid"][:])
        if set(train_keys).intersection(valid_keys):
            raise ValueError(f"{path} train and valid masks overlap")
        if set(all_keys) != set(train_keys).union(valid_keys):
            raise ValueError(f"{path} all mask does not cover train and valid")
        if set(all_keys) != set(dataset["data"].keys()):
            raise ValueError(f"{path} masks do not match demonstrations")
        if train_keys != list(manifest.get("split", {}).get("train", ())) or (
            valid_keys != list(manifest.get("split", {}).get("valid", ()))
        ):
            raise ValueError(f"{path} masks do not match the split manifest")
        manifest_numbers = manifest.get("subset", {}).get("episode_numbers", [])
        actual_numbers = sorted(
            int(dataset[f"data/{key}"].attrs["source_episode_number"])
            for key in all_keys
        )
        if actual_numbers != sorted(int(value) for value in manifest_numbers):
            raise ValueError(f"{path} subset manifest does not match demonstrations")
        if any(number < 1 or number > 50 for number in actual_numbers):
            raise ValueError(f"{path} contains a non-round-1 episode")
        if source_root is not None:
            source_root = Path(source_root).expanduser().resolve()
            expected_rows = [
                row
                for row in eligible_rows(source_root)
                if row.collection_round == 1
            ]
            if actual_numbers != [row.episode_number for row in expected_rows]:
                raise ValueError(f"{path} does not contain the eligible round-1 subset")
            if controller_contract["collector_manifests_required"]:
                for row in expected_rows:
                    _verify_controller_manifest(source_root, row, required=True)
        reports = [
            _validate_demo(
                dataset[f"data/{key}"],
                image_height=int(image["height"]),
                image_width=int(image["width"]),
                max_image_age_sec=float(
                    manifest["alignment"]["max_image_age_sec"]
                ),
            )
            for key in all_keys
        ]
        total = sum(report["samples"] for report in reports)
        if int(dataset["data"].attrs.get("total", -1)) != total:
            raise ValueError(f"{path} total sample count is inconsistent")

    transitions = sum(max(0, report["samples"] - 1) for report in reports)
    repeats = sum(report["adjacent_rgb_repeats"] for report in reports)
    return {
        "validated": True,
        "path": str(path),
        "generation_id": manifest["generation_id"],
        "episodes": len(reports),
        "train_episodes": len(train_keys),
        "valid_episodes": len(valid_keys),
        "samples": total,
        "max_image_age_sec": max(
            report["max_image_age_sec"] for report in reports
        ),
        "adjacent_rgb_repeat_count": repeats,
        "adjacent_rgb_repeat_fraction": (
            float(repeats / transitions) if transitions else 0.0
        ),
        "terminal_zero_motion_episodes": sum(
            report["terminal_motion_zero"] for report in reports
        ),
        "schema_signature": {
            "action_shape": [7],
            "image_shape": [int(image["height"]), int(image["width"]), 3],
            "low_dim_shapes": {
                LOW_DIM_KEYS[0]: [3],
                LOW_DIM_KEYS[1]: [4],
                LOW_DIM_KEYS[2]: [1],
            },
        },
    }


def _requested_settings(options: BuildOptions) -> dict[str, Any]:
    return {
        "source_identity": source_identity(options.source_root),
        "image": {
            "height": options.image_height,
            "width": options.image_width,
            "compression": options.compression,
        },
        "alignment": {"max_image_age_sec": options.max_image_age_sec},
        "split": {
            "seed": options.split_seed,
            "requested_validation_count": options.validation_count,
        },
        "downsampling": {
            "stride_20hz_rows": SOURCE_STRIDE,
            "phase_20hz_row": SOURCE_PHASE,
        },
        "controller_contract": {
            "collector_manifests_required": options.require_controller_manifest,
        },
    }


def _existing_matches(path: Path, options: BuildOptions) -> bool:
    with h5py.File(path, "r") as dataset:
        manifest = _json_attr(
            dataset.attrs,
            CONVERSION_MANIFEST_ATTR,
            location=str(path),
        )
    requested = _requested_settings(options)
    for section, expected in requested.items():
        actual = manifest.get(section, {})
        if section == "source_identity":
            if actual != expected:
                return False
            continue
        for key, value in expected.items():
            if actual.get(key) != value:
                return False
    return True


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    source_root = Path(options.source_root).expanduser().resolve()
    output_dir = Path(options.output_dir).expanduser().resolve()
    options = BuildOptions(
        **{
            **vars(options),
            "source_root": source_root,
            "output_dir": output_dir,
        }
    )
    if options.image_height < 2 or options.image_width < 2:
        raise ValueError("image dimensions must be at least 2")
    if options.validation_count < 1:
        raise ValueError("validation_count must be positive")
    if options.max_image_age_sec <= 0.0:
        raise ValueError("max_image_age_sec must be positive")
    _compression_kwargs(options.compression)
    path = dataset_path(output_dir)
    if path.exists() and not options.overwrite:
        if not _existing_matches(path, options):
            raise ValueError(
                f"existing 5 Hz dataset settings do not match request: {path}"
            )
        report = validate_5hz_dataset(path, source_root=source_root)
        report["reused_existing"] = True
        return report
    if options.validate_only:
        report = validate_5hz_dataset(path, source_root=source_root)
        report["validated_only"] = True
        return report

    rows = [row for row in eligible_rows(source_root) if row.collection_round == 1]
    if len(rows) < 2:
        raise ValueError("round 1 needs at least two QA-eligible demonstrations")
    payloads = [
        _macro_episode(
            source_root,
            row,
            max_image_age_sec=options.max_image_age_sec,
            require_controller_manifest=options.require_controller_manifest,
        )
        for row in rows
    ]
    close_xy = {
        payload.row.episode_number: payload.source.close_position[:2]
        for payload in payloads
    }
    validation_numbers = select_spatial_validation_rows(
        rows,
        close_xy,
        count=options.validation_count,
        seed=options.split_seed,
    )
    generation_id = uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT_FILENAME}.", suffix=".partial", dir=output_dir
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    try:
        _write_hdf5(
            temporary,
            options,
            payloads,
            validation_numbers,
            generation_id,
        )
        validate_5hz_dataset(temporary, source_root=source_root)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    report = validate_5hz_dataset(path, source_root=source_root)
    report["reused_existing"] = False
    report["source"] = str(source_root)
    atomic_write_json(output_dir / SUMMARY_FILENAME, report)
    stat = path.stat()
    atomic_write_json(
        output_dir / COMMIT_FILENAME,
        {
            "conversion_version": CONVERSION_VERSION,
            "generation_id": generation_id,
            "dataset": OUTPUT_FILENAME,
            "size_bytes": int(stat.st_size),
            "source_identity": source_identity(source_root),
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_5HZ_DATASET_DIR)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--validation-count", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--max-image-age-sec", type=float, default=DEFAULT_MAX_IMAGE_AGE_SEC
    )
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-missing-controller-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = build_dataset(
        BuildOptions(
            source_root=args.source,
            output_dir=args.output_dir,
            image_height=args.image_height,
            image_width=args.image_width,
            validation_count=args.validation_count,
            split_seed=args.split_seed,
            max_image_age_sec=args.max_image_age_sec,
            compression=args.compression,
            overwrite=args.overwrite,
            validate_only=args.validate_only,
            require_controller_manifest=not args.allow_missing_controller_manifest,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
