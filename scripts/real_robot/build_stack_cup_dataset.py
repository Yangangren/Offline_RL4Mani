#!/usr/bin/env python3
"""Build the real-robot stack-cup RGB Diffusion Policy dataset.

The handoff records actions and robot state on an exact 20 Hz target clock and
paired RGB at 5 Hz. This converter preserves the normalized controller command,
densifies sparse gripper events, and performs causal alignment using the actual
camera header stamps. Episode 007 is deliberately and permanently excluded by
the dataset contract because its camera gap violates the 0.5 second age bound.
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
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot.stack_cup_common import (  # noqa: E402
    ACTION_HZ,
    CONVERSION_MANIFEST_ATTR,
    CONVERSION_SUMMARY_FILENAME,
    CONVERSION_VERSION,
    DATASET_COMMIT_FILENAME,
    DATASET_FILENAME,
    DEFAULT_DATASET_DIR,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MAX_IMAGE_AGE_SEC,
    DEFAULT_SOURCE,
    EXCLUDED_EPISODES,
    IMAGE_HZ,
    LOW_DIM_KEYS,
    ROTATION_SCALE_RAD,
    RGB_KEYS,
    SCHEMA_VERSION,
    TRANSLATION_SCALE_M,
    VALIDATION_EPISODE_NUMBERS,
    StackCupEpisodeRow,
    as_float_array,
    atomic_write_json,
    dataset_commit_path,
    dataset_path,
    densify_gripper_events,
    included_rows,
    load_json,
    read_episode_rows,
    resolve_episode_dir,
    source_identity,
    strictly_increasing,
)


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_DATASET_DIR
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    max_image_age_sec: float = DEFAULT_MAX_IMAGE_AGE_SEC
    compression: str = "gzip"
    overwrite: bool = False
    validate_only: bool = False


@dataclass
class EpisodePayload:
    row: StackCupEpisodeRow
    episode_dir: Path
    actions: np.ndarray
    raw_gripper_events: np.ndarray
    gripper_observations: np.ndarray
    source_action_array_indices: np.ndarray
    action_target_times: np.ndarray
    action_source_times: np.ndarray
    action_source_indices: np.ndarray
    action_steps: np.ndarray
    action_offsets_ms: np.ndarray
    eef_poses: np.ndarray
    selected_frame_positions: np.ndarray
    selected_frame_indices: np.ndarray
    selected_frame_nominal_times: np.ndarray
    selected_main_capture_times: np.ndarray
    selected_wrist_capture_times: np.ndarray
    selected_pair_capture_times: np.ndarray
    image_ages: np.ndarray
    main_paths: list[Path]
    wrist_paths: list[Path]
    frame_source_sizes: list[tuple[int, int]]
    dropped_prefix_actions: int
    initial_gripper_state: float
    close_position: np.ndarray
    release_position: np.ndarray
    simultaneous_button_rows: int

    @property
    def num_samples(self) -> int:
        return int(self.actions.shape[0])


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_nonempty_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def _resolve_stream_path(episode_dir: Path, relative: Any, *, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name}.path must be a non-empty string")
    resolved = (episode_dir / relative).resolve()
    try:
        resolved.relative_to(episode_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{name}.path escapes episode directory: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {name} image: {resolved}")
    return resolved


def _first_event_pose(
    samples: Sequence[dict[str, Any]],
    *,
    event_sign: int,
    start: int = 0,
    name: str,
) -> tuple[int, np.ndarray]:
    for index in range(start, len(samples)):
        action = as_float_array(
            samples[index].get("action"),
            shape=(7,),
            name=f"{name}.samples[{index}].action",
        )
        if (event_sign < 0 and action[6] < 0.0) or (
            event_sign > 0 and action[6] > 0.0
        ):
            pose = as_float_array(
                samples[index].get("before_pose"),
                shape=(7,),
                name=f"{name}.samples[{index}].before_pose",
            )
            return index, pose[:3].astype(np.float32)
    direction = "close" if event_sign < 0 else "open"
    raise ValueError(f"{name} contains no {direction} gripper event")


def _validate_collection_contract(episode_dir: Path, row: StackCupEpisodeRow) -> None:
    contract_path = episode_dir / "contract.json"
    collector_path = episode_dir / "snapshots/collector_manifest.json"
    contract = _require_object(load_json(contract_path), name=str(contract_path))
    collector = _require_object(load_json(collector_path), name=str(collector_path))

    actions = _require_object(contract.get("actions"), name=f"{contract_path}.actions")
    video = _require_object(contract.get("video"), name=f"{contract_path}.video")
    if not np.isclose(float(actions.get("hz", -1.0)), ACTION_HZ, atol=1e-9):
        raise ValueError(f"{contract_path} action rate is not {ACTION_HZ} Hz")
    if int(actions.get("num_actions", -1)) != row.manifest_actions:
        raise ValueError(f"{contract_path} action count disagrees with manifest")
    if not np.isclose(float(video.get("hz", -1.0)), IMAGE_HZ, atol=1e-9):
        raise ValueError(f"{contract_path} image rate is not {IMAGE_HZ} Hz")
    if int(video.get("num_frames", -1)) != row.manifest_frames:
        raise ValueError(f"{contract_path} frame count disagrees with manifest")
    if int(video.get("dropped_num_frames", -1)) < 0:
        raise ValueError(f"{contract_path} has invalid dropped_num_frames")

    expected_scale = np.asarray(
        [TRANSLATION_SCALE_M, ROTATION_SCALE_RAD, 1.0], dtype=np.float64
    )
    actual_scale = as_float_array(
        collector.get("action_scale"),
        shape=(3,),
        name=f"{collector_path}.action_scale",
    )
    if not np.allclose(actual_scale, expected_scale, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"{collector_path} action_scale {actual_scale.tolist()} does not match "
            f"{expected_scale.tolist()}"
        )
    if not np.isclose(float(collector.get("hz", -1.0)), ACTION_HZ, atol=1e-9):
        raise ValueError(f"{collector_path} collection rate is not {ACTION_HZ} Hz")
    telemetry = _require_object(
        collector.get("ros_telemetry"), name=f"{collector_path}.ros_telemetry"
    )
    if telemetry.get("base_frame") != "panda_link0":
        raise ValueError(f"{collector_path} base frame must be panda_link0")


def load_episode_payload(
    source_root: Path,
    row: StackCupEpisodeRow,
    *,
    max_image_age_sec: float,
) -> EpisodePayload:
    """Validate and load one included episode without scaling its actions."""

    if row.excluded:
        raise ValueError(
            f"episode {row.episode_number:03d} is hard-excluded: "
            f"{EXCLUDED_EPISODES[row.episode_number]}"
        )
    episode_dir = resolve_episode_dir(source_root, row)
    _validate_collection_contract(episode_dir, row)
    actions_path = episode_dir / "actions.json"
    frames_path = episode_dir / "frames.json"
    actions_document = _require_object(load_json(actions_path), name=str(actions_path))
    frames_document = _require_object(load_json(frames_path), name=str(frames_path))
    samples = _require_nonempty_list(
        actions_document.get("samples"), name=f"{actions_path}.samples"
    )
    frames = _require_nonempty_list(
        frames_document.get("frames"), name=f"{frames_path}.frames"
    )
    if len(samples) != row.manifest_actions:
        raise ValueError(
            f"{actions_path} has {len(samples)} rows; manifest says {row.manifest_actions}"
        )
    if len(frames) != row.manifest_frames:
        raise ValueError(
            f"{frames_path} has {len(frames)} frames; manifest says {row.manifest_frames}"
        )

    raw_actions: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    action_target_times: list[float] = []
    action_source_times: list[float] = []
    action_source_indices: list[int] = []
    action_steps: list[int] = []
    action_offsets_ms: list[float] = []
    simultaneous_button_rows = 0
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"{actions_path}.samples[{index}] must be an object")
        action = as_float_array(
            raw_sample.get("action"),
            shape=(7,),
            name=f"{actions_path}.samples[{index}].action",
        )
        if np.any(action < -1.000001) or np.any(action > 1.000001):
            raise ValueError(
                f"{actions_path}.samples[{index}].action is outside normalized [-1,1]"
            )
        if action[6] not in (-1.0, 0.0, 1.0):
            raise ValueError(
                f"{actions_path}.samples[{index}] has invalid gripper event {action[6]}"
            )
        raw_motion = as_float_array(
            raw_sample.get("raw_action"),
            shape=(6,),
            name=f"{actions_path}.samples[{index}].raw_action",
        )
        if not np.allclose(action[:6], raw_motion, atol=5e-7, rtol=0.0):
            raise ValueError(
                f"{actions_path}.samples[{index}] action motion differs from raw_action"
            )
        clip = raw_sample.get("clip")
        if clip not in (None, []):
            raise ValueError(f"{actions_path}.samples[{index}] reports clipping: {clip}")

        pose = as_float_array(
            raw_sample.get("before_pose"),
            shape=(7,),
            name=f"{actions_path}.samples[{index}].before_pose",
        )
        after_pose = as_float_array(
            raw_sample.get("after_pose"),
            shape=(7,),
            name=f"{actions_path}.samples[{index}].after_pose",
        )
        for label, quaternion in (
            ("before", pose[3:7]),
            ("after", after_pose[3:7]),
        ):
            norm = float(np.linalg.norm(quaternion))
            if not np.isclose(norm, 1.0, atol=5e-3):
                raise ValueError(
                    f"{actions_path}.samples[{index}] {label} quaternion norm "
                    f"is {norm:.6f}, expected 1"
                )
        try:
            target_time = float(raw_sample["target_time"])
            source_time = float(raw_sample.get("source_time", target_time))
            source_index = int(raw_sample.get("source_index", index))
            step = int(raw_sample.get("step", index))
            offset_ms = float(
                raw_sample.get("offset_ms", (source_time - target_time) * 1000.0)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid timing/index fields in {actions_path}.samples[{index}]"
            ) from exc
        if not all(np.isfinite(value) for value in (target_time, source_time, offset_ms)):
            raise ValueError(f"non-finite time in {actions_path}.samples[{index}]")
        if step != index:
            raise ValueError(
                f"{actions_path}.samples[{index}].step={step}, expected {index}"
            )
        buttons = raw_sample.get("buttons", [])
        if isinstance(buttons, list) and len(buttons) >= 2 and buttons[0] and buttons[1]:
            simultaneous_button_rows += 1
        raw_actions.append(action.astype(np.float32))
        poses.append(pose.astype(np.float32))
        action_target_times.append(target_time)
        action_source_times.append(source_time)
        action_source_indices.append(source_index)
        action_steps.append(step)
        action_offsets_ms.append(offset_ms)

    raw_actions_array = np.stack(raw_actions)
    poses_array = np.stack(poses)
    target_times = np.asarray(action_target_times, dtype=np.float64)
    strictly_increasing(target_times, name=f"{actions_path} target times")
    if target_times.size > 1 and not np.allclose(
        np.diff(target_times), 1.0 / ACTION_HZ, atol=5e-6, rtol=0.0
    ):
        raise ValueError(f"{actions_path} target clock is not exact {ACTION_HZ} Hz")

    gripper_observations, dense_gripper_targets = densify_gripper_events(
        raw_actions_array[:, 6]
    )
    dense_actions = raw_actions_array.copy()
    dense_actions[:, 6] = dense_gripper_targets
    close_index, close_position = _first_event_pose(
        samples, event_sign=-1, name=str(actions_path)
    )
    _, release_position = _first_event_pose(
        samples, event_sign=1, start=close_index + 1, name=str(actions_path)
    )

    frame_indices: list[int] = []
    frame_nominal_times: list[float] = []
    main_capture_times: list[float] = []
    wrist_capture_times: list[float] = []
    main_paths: list[Path] = []
    wrist_paths: list[Path] = []
    source_sizes: list[tuple[int, int]] = []
    for position, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, dict):
            raise ValueError(f"{frames_path}.frames[{position}] must be an object")
        streams = raw_frame.get("streams")
        if not isinstance(streams, dict):
            raise ValueError(f"{frames_path}.frames[{position}].streams must be an object")
        rgb_streams: list[tuple[float, int, int, Path]] = []
        for stream_name in ("main_rgb", "wrist_rgb"):
            stream = streams.get(stream_name)
            if not isinstance(stream, dict):
                raise ValueError(
                    f"{frames_path}.frames[{position}] is missing {stream_name}"
                )
            if stream.get("encoding") != "rgb8":
                raise ValueError(
                    f"{frames_path}.frames[{position}].{stream_name} encoding "
                    f"must be rgb8, got {stream.get('encoding')!r}"
                )
            try:
                capture_time = int(stream["header_stamp_ns"]) / 1e9
                width = int(stream["width"])
                height = int(stream["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid dimensions/header stamp for {stream_name} in "
                    f"{frames_path}.frames[{position}]"
                ) from exc
            if min(width, height) <= 1:
                raise ValueError(f"invalid image dimensions in {frames_path}")
            path = _resolve_stream_path(
                episode_dir,
                stream.get("path"),
                name=f"frames[{position}].{stream_name}",
            )
            rgb_streams.append((capture_time, width, height, path))
        main, wrist = rgb_streams
        if (main[1], main[2]) != (wrist[1], wrist[2]):
            raise ValueError(
                f"camera dimensions differ in {frames_path}.frames[{position}]"
            )
        if abs(main[0] - wrist[0]) > 0.050001:
            raise ValueError(
                f"RGB pair skew exceeds 50 ms in {frames_path}.frames[{position}]"
            )
        try:
            frame_index = int(raw_frame.get("index", position))
            nominal_time = float(raw_frame["target_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid frame index/time in {frames_path}.frames[{position}]"
            ) from exc
        if frame_index != position:
            raise ValueError(
                f"{frames_path}.frames[{position}].index={frame_index}, expected {position}"
            )
        if not np.isfinite(nominal_time):
            raise ValueError(f"non-finite nominal frame time in {frames_path}")
        frame_indices.append(frame_index)
        frame_nominal_times.append(nominal_time)
        main_capture_times.append(main[0])
        wrist_capture_times.append(wrist[0])
        main_paths.append(main[3])
        wrist_paths.append(wrist[3])
        source_sizes.append((main[1], main[2]))

    nominal_times = np.asarray(frame_nominal_times, dtype=np.float64)
    strictly_increasing(nominal_times, name=f"{frames_path} nominal times")
    paired_capture_times = np.maximum(
        np.asarray(main_capture_times, dtype=np.float64),
        np.asarray(wrist_capture_times, dtype=np.float64),
    )
    strictly_increasing(paired_capture_times, name=f"{frames_path} paired capture times")
    mapping = np.searchsorted(paired_capture_times, target_times, side="right") - 1
    causal = mapping >= 0
    if not np.any(causal):
        raise ValueError(f"{actions_path} has no action after a causal RGB pair")
    first_causal = int(np.flatnonzero(causal)[0])
    if not np.all(causal[first_causal:]):
        raise ValueError(f"{actions_path} causal alignment has an internal negative index")
    keep = slice(first_causal, None)
    selected_positions = mapping[keep].astype(np.int64)
    selected_pair_times = paired_capture_times[selected_positions]
    kept_target_times = target_times[keep]
    image_ages = kept_target_times - selected_pair_times
    if np.any(image_ages < -1e-6):
        raise ValueError(f"{actions_path} selected a future RGB frame")
    maximum_age = float(np.max(image_ages))
    if maximum_age > max_image_age_sec + 1e-6:
        raise ValueError(
            f"{actions_path} maximum causal image age {maximum_age:.6f}s exceeds "
            f"limit {max_image_age_sec:.6f}s"
        )

    selected_indices = np.asarray(frame_indices, dtype=np.int64)[selected_positions]
    return EpisodePayload(
        row=row,
        episode_dir=episode_dir,
        actions=dense_actions[keep],
        raw_gripper_events=raw_actions_array[keep, 6],
        gripper_observations=gripper_observations[keep],
        source_action_array_indices=np.arange(len(samples), dtype=np.int64)[keep],
        action_target_times=kept_target_times,
        action_source_times=np.asarray(action_source_times, dtype=np.float64)[keep],
        action_source_indices=np.asarray(action_source_indices, dtype=np.int64)[keep],
        action_steps=np.asarray(action_steps, dtype=np.int64)[keep],
        action_offsets_ms=np.asarray(action_offsets_ms, dtype=np.float64)[keep],
        eef_poses=poses_array[keep],
        selected_frame_positions=selected_positions,
        selected_frame_indices=selected_indices,
        selected_frame_nominal_times=nominal_times[selected_positions],
        selected_main_capture_times=np.asarray(main_capture_times, dtype=np.float64)[
            selected_positions
        ],
        selected_wrist_capture_times=np.asarray(wrist_capture_times, dtype=np.float64)[
            selected_positions
        ],
        selected_pair_capture_times=selected_pair_times,
        image_ages=image_ages,
        main_paths=main_paths,
        wrist_paths=wrist_paths,
        frame_source_sizes=source_sizes,
        dropped_prefix_actions=first_causal,
        initial_gripper_state=float(gripper_observations[first_causal]),
        close_position=close_position,
        release_position=release_position,
        simultaneous_button_rows=simultaneous_button_rows,
    )


def _load_resized_rgb(
    path: Path,
    *,
    expected_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(
                f"{path} has size {image.size}, metadata says {expected_size}"
            )
        if image.mode != "RGB":
            raise ValueError(f"{path} has mode {image.mode}, expected RGB")
        image = image.resize(output_size, resample=Image.Resampling.LANCZOS)
        result = np.asarray(image, dtype=np.uint8)
    expected_shape = (output_size[1], output_size[0], 3)
    if result.shape != expected_shape:
        raise ValueError(f"resized {path} has shape {result.shape}, expected {expected_shape}")
    return result


def _compression_kwargs(compression: str) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf"}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    raise ValueError(f"unsupported compression: {compression}")


def _write_images(
    obs: h5py.Group,
    payload: EpisodePayload,
    *,
    image_height: int,
    image_width: int,
    compression: str,
) -> None:
    count = payload.num_samples
    chunks = (min(8, count), image_height, image_width, 3)
    kwargs = _compression_kwargs(compression)
    datasets = {
        RGB_KEYS[0]: obs.create_dataset(
            RGB_KEYS[0],
            shape=(count, image_height, image_width, 3),
            dtype=np.uint8,
            chunks=chunks,
            **kwargs,
        ),
        RGB_KEYS[1]: obs.create_dataset(
            RGB_KEYS[1],
            shape=(count, image_height, image_width, 3),
            dtype=np.uint8,
            chunks=chunks,
            **kwargs,
        ),
    }
    used_positions = sorted({int(value) for value in payload.selected_frame_positions})
    caches: dict[str, dict[int, np.ndarray]] = {key: {} for key in RGB_KEYS}
    for position in used_positions:
        expected_size = payload.frame_source_sizes[position]
        caches[RGB_KEYS[0]][position] = _load_resized_rgb(
            payload.main_paths[position],
            expected_size=expected_size,
            output_size=(image_width, image_height),
        )
        caches[RGB_KEYS[1]][position] = _load_resized_rgb(
            payload.wrist_paths[position],
            expected_size=expected_size,
            output_size=(image_width, image_height),
        )

    block_size = 128
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        positions = payload.selected_frame_positions[start:end]
        for key, dataset in datasets.items():
            dataset[start:end] = np.stack(
                [caches[key][int(position)] for position in positions]
            )


def _mask_keys(
    payloads: Sequence[EpisodePayload],
) -> dict[str, list[str]]:
    all_keys = [payload.row.demo_key for payload in payloads]
    valid_keys = [
        payload.row.demo_key
        for payload in payloads
        if payload.row.episode_number in VALIDATION_EPISODE_NUMBERS
    ]
    valid_set = set(valid_keys)
    train_keys = [key for key in all_keys if key not in valid_set]
    qa_pass_keys = [
        payload.row.demo_key for payload in payloads if payload.row.qa_status == "PASS"
    ]
    clean_keys = [payload.row.demo_key for payload in payloads if payload.row.clean]
    train_clean_keys = [
        payload.row.demo_key
        for payload in payloads
        if payload.row.demo_key not in valid_set and payload.row.clean
    ]
    masks = {
        "all": all_keys,
        "train": train_keys,
        "valid": valid_keys,
        "qa_pass": qa_pass_keys,
        "clean": clean_keys,
        "train_clean": train_clean_keys,
    }
    if set(valid_keys) != {f"demo_{number:03d}" for number in VALIDATION_EPISODE_NUMBERS}:
        raise ValueError("fixed validation set is incomplete")
    if any(not values for values in masks.values()):
        raise ValueError(f"all dataset masks must be non-empty: {masks}")
    return masks


def _episode_summary(payload: EpisodePayload, masks: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "demo_key": payload.row.demo_key,
        "episode_number": payload.row.episode_number,
        "run_id": payload.row.run_id,
        "source_directory": payload.row.directory,
        "qa_status": payload.row.qa_status,
        "model_window_ready": payload.row.model_window_ready,
        "invalid_windows": payload.row.invalid_windows,
        "mask_membership": [name for name, keys in masks.items() if payload.row.demo_key in keys],
        "num_samples": payload.num_samples,
        "source_num_actions": payload.row.manifest_actions,
        "source_num_frames": payload.row.manifest_frames,
        "dropped_prefix_actions": payload.dropped_prefix_actions,
        "max_image_age_sec": float(np.max(payload.image_ages)),
        "close_position_xyz": payload.close_position.tolist(),
        "release_position_xyz": payload.release_position.tolist(),
        "simultaneous_button_rows": payload.simultaneous_button_rows,
    }


def write_dataset(
    path: Path,
    payloads: Sequence[EpisodePayload],
    *,
    generation_id: str,
    options: BuildOptions,
    identity: dict[str, Any],
    all_source_rows: Sequence[StackCupEpisodeRow],
) -> dict[str, Any]:
    if len(payloads) != 49:
        raise ValueError(f"expected 49 included payloads, got {len(payloads)}")
    masks = _mask_keys(payloads)
    excluded_rows = [row for row in all_source_rows if row.excluded]
    excluded = [
        {
            "episode_number": row.episode_number,
            "demo_key": row.demo_key,
            "run_id": row.run_id,
            "source_directory": row.directory,
            "reason": EXCLUDED_EPISODES[row.episode_number],
        }
        for row in excluded_rows
    ]
    if len(excluded) != 1 or excluded[0]["episode_number"] != 7:
        raise ValueError("the source audit did not resolve hard-excluded episode 007")

    episode_summaries = [_episode_summary(payload, masks) for payload in payloads]
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "conversion_version": CONVERSION_VERSION,
        "generation_id": generation_id,
        "created_at": created_at,
        "source_identity": identity,
        "excluded_episodes": excluded,
        "alignment": {
            "master_clock": "actions.samples.target_time",
            "image_availability": "max(main_rgb.header_stamp_ns,wrist_rgb.header_stamp_ns)",
            "selection": "latest paired capture not later than action target",
            "startup_policy": "drop action rows before the first causal RGB pair",
            "max_cross_camera_skew_sec": 0.05,
            "max_image_age_sec": options.max_image_age_sec,
        },
        "gripper": {
            "raw": "sparse event: -1 close, 0 hold, +1 open",
            "observation": "logical state before current event",
            "action": "dense logical target after current event",
            "initial_state": "+1 open",
        },
        "action": {
            "hz": ACTION_HZ,
            "dimensions": 7,
            "controller_input": "normalized, unscaled",
            "translation_scale_m": TRANSLATION_SCALE_M,
            "rotation_scale_rad": ROTATION_SCALE_RAD,
            "normalization": "motion and dense gripper in [-1,1]",
        },
        "outcome": {
            "source_machine_label": False,
            "manual_review": (
                "all included terminal main-RGB frames were reviewed and show the "
                "pink cup nested in the white cup"
            ),
            "robomimic_compatibility_label": (
                "reward=1 and done=1 only on the final retained row; earlier rows are 0"
            ),
            "diffusion_policy_uses_reward": False,
        },
        "image": {
            "source_hz": IMAGE_HZ,
            "height": options.image_height,
            "width": options.image_width,
            "layout": "HWC RGB uint8",
            "resize": "Pillow Lanczos",
            "compression": options.compression,
        },
        "split": {
            "method": "fixed episode split chosen before training for clean spatial coverage",
            "validation_episode_numbers": sorted(VALIDATION_EPISODE_NUMBERS),
            **masks,
            "qa_pass_definition": "source qa.status == PASS",
            "clean_definition": (
                "qa_pass intersection windows.invalid_windows == 0"
            ),
            "train_clean_definition": (
                "train intersection qa_pass with windows.invalid_windows == 0"
            ),
        },
        "episodes": episode_summaries,
    }
    env_args = {
        "env_name": "StackCupReal-v0",
        "env_version": CONVERSION_VERSION,
        "type": 2,
        "env_kwargs": {
            "real_robot": True,
            "control_freq": int(ACTION_HZ),
            "camera_names": ["main", "wrist"],
            "camera_height": options.image_height,
            "camera_width": options.image_width,
            "task": "stack_cup",
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
            demo.attrs["qa_status"] = payload.row.qa_status
            demo.attrs["model_window_ready"] = np.bool_(
                payload.row.model_window_ready
            )
            demo.attrs["invalid_windows"] = payload.row.invalid_windows
            demo.attrs["dropped_prefix_actions"] = payload.dropped_prefix_actions
            demo.attrs["initial_gripper_state"] = payload.initial_gripper_state
            demo.attrs["close_position_xyz"] = payload.close_position
            demo.attrs["release_position_xyz"] = payload.release_position
            demo.attrs["simultaneous_button_rows"] = payload.simultaneous_button_rows
            demo.attrs["max_image_age_sec"] = float(np.max(payload.image_ages))

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
                "raw_gripper_event", data=payload.raw_gripper_events.astype(np.float32)
            )
            provenance.create_dataset(
                "source_action_array_index", data=payload.source_action_array_indices
            )
            provenance.create_dataset("source_action_index", data=payload.action_source_indices)
            provenance.create_dataset("source_action_step", data=payload.action_steps)
            provenance.create_dataset(
                "source_action_offset_ms", data=payload.action_offsets_ms
            )
            provenance.create_dataset("action_target_time", data=payload.action_target_times)
            provenance.create_dataset("action_source_time", data=payload.action_source_times)
            provenance.create_dataset(
                "selected_frame_position", data=payload.selected_frame_positions
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
            provenance.create_dataset(
                "selected_pair_capture_time",
                data=payload.selected_pair_capture_times,
            )
            provenance.create_dataset("image_age_sec", data=payload.image_ages)
            total += payload.num_samples

        data.attrs["total"] = total
        mask = output.create_group("mask")
        for name, keys in masks.items():
            mask.create_dataset(name, data=np.asarray(keys, dtype="S"))
        output.flush()

    return {
        "path": str(path),
        "episodes": len(payloads),
        "excluded_episodes": excluded,
        "samples": sum(payload.num_samples for payload in payloads),
        "dropped_prefix_actions": sum(
            payload.dropped_prefix_actions for payload in payloads
        ),
        "max_image_age_sec": max(float(np.max(payload.image_ages)) for payload in payloads),
        "mask_counts": {name: len(keys) for name, keys in masks.items()},
    }


def _temporary_path(directory: Path, filename: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".partial", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _assert_existing_options(path: Path, options: BuildOptions) -> None:
    expected = {
        "height": int(options.image_height),
        "width": int(options.image_width),
        "max_image_age_sec": float(options.max_image_age_sec),
        "compression": options.compression,
    }
    with h5py.File(path, "r") as dataset:
        raw_manifest = dataset.attrs.get(CONVERSION_MANIFEST_ATTR)
        if isinstance(raw_manifest, bytes):
            raw_manifest = raw_manifest.decode("utf-8")
        try:
            manifest = json.loads(raw_manifest)
            actual = {
                "height": int(manifest["image"]["height"]),
                "width": int(manifest["image"]["width"]),
                "max_image_age_sec": float(manifest["alignment"]["max_image_age_sec"]),
                "compression": str(manifest["image"]["compression"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path} does not record reusable conversion settings; rebuild with --overwrite"
            ) from exc
    mismatches = {
        key: {"requested": expected[key], "existing": actual[key]}
        for key in expected
        if (
            not np.isclose(expected[key], actual[key], atol=1e-12, rtol=0.0)
            if key == "max_image_age_sec"
            else expected[key] != actual[key]
        )
    }
    if mismatches:
        raise ValueError(
            f"{path} conversion settings do not match this request: {mismatches}; "
            "rebuild with --overwrite"
        )


def validate_published_dataset(
    output_dir: Path = DEFAULT_DATASET_DIR,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the committed dataset through the independent contract checker."""

    from scripts.real_robot.validate_stack_cup_dataset import (
        validate_published_dataset as validate,
    )

    return validate(output_dir, source_root=source_root)


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    source_root = options.source_root.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    final_path = dataset_path(output_dir)
    commit_path = dataset_commit_path(output_dir)
    if options.image_height <= 1 or options.image_width <= 1:
        raise ValueError("image dimensions must both be greater than one")
    if options.max_image_age_sec <= 0.0:
        raise ValueError("max_image_age_sec must be positive")
    if not np.isclose(
        options.max_image_age_sec,
        DEFAULT_MAX_IMAGE_AGE_SEC,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "stack-cup conversion uses the fixed 0.5 s image-age contract; "
            f"got {options.max_image_age_sec}"
        )
    _compression_kwargs(options.compression)

    if options.validate_only:
        if not final_path.is_file():
            raise FileNotFoundError(f"cannot validate missing dataset: {final_path}")
        return validate_published_dataset(output_dir, source_root=source_root)
    if final_path.exists() and not options.overwrite:
        _assert_existing_options(final_path, options)
        report = validate_published_dataset(output_dir, source_root=source_root)
        report["reused_existing"] = True
        return report

    all_source_rows = read_episode_rows(source_root)
    rows = included_rows(source_root)
    payloads: list[EpisodePayload] = []
    for row in rows:
        print(
            f"[stack_cup dataset] auditing episode {row.episode_number:03d} "
            f"({row.run_id})",
            flush=True,
        )
        payloads.append(
            load_episode_payload(
                source_root, row, max_image_age_sec=options.max_image_age_sec
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = source_identity(source_root)
    generation_id = uuid.uuid4().hex
    temporary = _temporary_path(output_dir, DATASET_FILENAME)
    backup = _temporary_path(output_dir, f"{DATASET_FILENAME}.backup")
    commit_backup = _temporary_path(output_dir, f"{DATASET_COMMIT_FILENAME}.backup")
    try:
        summary = write_dataset(
            temporary,
            payloads,
            generation_id=generation_id,
            options=options,
            identity=identity,
            all_source_rows=all_source_rows,
        )
        from scripts.real_robot.validate_stack_cup_dataset import validate_dataset

        validate_dataset(temporary, source_root=source_root)
        final_was_backed_up = False
        commit_was_backed_up = False
        published = False
        try:
            if final_path.exists():
                os.replace(final_path, backup)
                final_was_backed_up = True
            if commit_path.exists():
                os.replace(commit_path, commit_backup)
                commit_was_backed_up = True
            os.replace(temporary, final_path)
            published = True
            validate_dataset(final_path, source_root=source_root)
            commit = {
                "schema_version": SCHEMA_VERSION,
                "conversion_version": CONVERSION_VERSION,
                "generation_id": generation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_identity": identity,
                "shards": [
                    {
                        "filename": DATASET_FILENAME,
                        "size_bytes": int(final_path.stat().st_size),
                    }
                ],
            }
            atomic_write_json(commit_path, commit)
            validation = validate_published_dataset(
                output_dir, source_root=source_root
            )
        except Exception:
            if published:
                final_path.unlink(missing_ok=True)
            commit_path.unlink(missing_ok=True)
            if final_was_backed_up:
                os.replace(backup, final_path)
            if commit_was_backed_up:
                os.replace(commit_backup, commit_path)
            raise
        else:
            backup.unlink(missing_ok=True)
            commit_backup.unlink(missing_ok=True)

        report = {
            "conversion_version": CONVERSION_VERSION,
            "source": str(source_root),
            "output_dir": str(output_dir),
            "dataset": {**summary, "path": str(final_path)},
            "validation": validation,
            "reused_existing": False,
        }
        atomic_write_json(output_dir / CONVERSION_SUMMARY_FILENAME, report)
        return report
    finally:
        temporary.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
        commit_backup.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument(
        "--max-image-age-sec", type=float, default=DEFAULT_MAX_IMAGE_AGE_SEC
    )
    parser.add_argument(
        "--compression", choices=("gzip", "lzf", "none"), default="gzip"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = build_dataset(
        BuildOptions(
            source_root=args.source,
            output_dir=args.output_dir,
            image_height=args.image_height,
            image_width=args.image_width,
            max_image_age_sec=args.max_image_age_sec,
            compression=args.compression,
            overwrite=args.overwrite,
            validate_only=args.validate_only,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
