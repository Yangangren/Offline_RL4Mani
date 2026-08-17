#!/usr/bin/env python3
"""Convert the Real4D pick-cup handoff into robomimic RGB HDF5 shards.

The raw package uses a 20 Hz action/state clock and paired 5 Hz cameras. This
converter makes all temporal decisions once, records them as provenance, and
leaves training to robomimic's standard ``SequenceDataset`` / ``MetaDataset``.
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

from scripts.real_robot.pick_cup_common import (  # noqa: E402
    ACTION_HZ,
    CONVERSION_MANIFEST_ATTR,
    CONVERSION_VERSION,
    DEFAULT_DATASET_DIR,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MAX_IMAGE_AGE_SEC,
    DEFAULT_SOURCE,
    IMAGE_HZ,
    ROTATION_SCALE_RAD,
    ROUND_FILENAMES,
    SCHEMA_VERSION,
    TRANSLATION_SCALE_M,
    EpisodeIndexRow,
    as_float_array,
    atomic_write_json,
    densify_gripper_events,
    dataset_commit_path,
    eligible_rows,
    load_json,
    resolve_episode_dir,
    round_paths,
    select_spatial_validation_rows,
    source_identity,
    strictly_increasing,
)


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_DATASET_DIR
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    validation_count_per_round: int = 5
    split_seed: int = 1
    max_image_age_sec: float = DEFAULT_MAX_IMAGE_AGE_SEC
    compression: str = "gzip"
    overwrite: bool = False
    validate_only: bool = False


@dataclass
class EpisodePayload:
    row: EpisodeIndexRow
    episode_dir: Path
    actions: np.ndarray
    raw_gripper_events: np.ndarray
    gripper_observations: np.ndarray
    action_target_times: np.ndarray
    action_source_times: np.ndarray
    action_source_indices: np.ndarray
    action_steps: np.ndarray
    eef_poses: np.ndarray
    selected_frame_positions: np.ndarray
    selected_frame_indices: np.ndarray
    selected_frame_nominal_times: np.ndarray
    selected_main_capture_times: np.ndarray
    selected_wrist_capture_times: np.ndarray
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


def _require_object(value: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _require_list(value: Any, *, name: str) -> list[Any]:
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


def load_episode_payload(
    source_root: Path,
    row: EpisodeIndexRow,
    *,
    max_image_age_sec: float,
) -> EpisodePayload:
    """Load one episode and perform all semantic and temporal validation."""

    episode_dir = resolve_episode_dir(source_root, row)
    actions_path = episode_dir / "actions.json"
    frames_path = episode_dir / "frames.json"
    actions_document = _require_object(load_json(actions_path), path=actions_path)
    frames_document = _require_object(load_json(frames_path), path=frames_path)
    samples = _require_list(actions_document.get("samples"), name=f"{actions_path}.samples")
    frames = _require_list(frames_document.get("frames"), name=f"{frames_path}.frames")

    raw_actions: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    action_target_times: list[float] = []
    action_source_times: list[float] = []
    action_source_indices: list[int] = []
    action_steps: list[int] = []
    simultaneous_button_rows = 0
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"{actions_path}.samples[{index}] must be an object")
        action = as_float_array(
            raw_sample.get("action"),
            shape=(7,),
            name=f"{actions_path}.samples[{index}].action",
        )
        if np.any(action[:6] < -1.001) or np.any(action[:6] > 1.001):
            raise ValueError(
                f"{actions_path}.samples[{index}].action motion is outside [-1, 1]"
            )
        pose = as_float_array(
            raw_sample.get("before_pose"),
            shape=(7,),
            name=f"{actions_path}.samples[{index}].before_pose",
        )
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        if not np.isclose(quaternion_norm, 1.0, atol=5e-3):
            raise ValueError(
                f"{actions_path}.samples[{index}] quaternion norm "
                f"is {quaternion_norm:.6f}, expected 1"
            )
        try:
            target_time = float(raw_sample["target_time"])
            source_time = float(raw_sample.get("source_time", target_time))
            source_index = int(raw_sample.get("source_index", index))
            step = int(raw_sample.get("step", index))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid timing/index fields in {actions_path}.samples[{index}]") from exc
        if not np.isfinite(target_time) or not np.isfinite(source_time):
            raise ValueError(f"non-finite time in {actions_path}.samples[{index}]")
        buttons = raw_sample.get("buttons", [])
        if isinstance(buttons, list) and len(buttons) >= 2 and buttons[0] and buttons[1]:
            simultaneous_button_rows += 1
        raw_actions.append(action.astype(np.float32))
        poses.append(pose.astype(np.float32))
        action_target_times.append(target_time)
        action_source_times.append(source_time)
        action_source_indices.append(source_index)
        action_steps.append(step)

    raw_actions_array = np.stack(raw_actions)
    poses_array = np.stack(poses)
    action_target_times_array = np.asarray(action_target_times, dtype=np.float64)
    strictly_increasing(action_target_times_array, name=f"{actions_path} target times")
    gripper_observations, dense_gripper_targets = densify_gripper_events(
        raw_actions_array[:, 6]
    )
    dense_actions = raw_actions_array.copy()
    dense_actions[:, 6] = dense_gripper_targets
    dense_actions[:, :6] = np.clip(dense_actions[:, :6], -1.0, 1.0)

    close_index, close_position = _first_event_pose(
        samples,
        event_sign=-1,
        name=str(actions_path),
    )
    _, release_position = _first_event_pose(
        samples,
        event_sign=1,
        start=close_index + 1,
        name=str(actions_path),
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
        rgb_streams = []
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
        try:
            frame_index = int(raw_frame.get("index", position))
            nominal_time = float(raw_frame["target_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frame index/time in {frames_path}.frames[{position}]") from exc
        if not np.isfinite(nominal_time):
            raise ValueError(f"non-finite nominal frame time in {frames_path}.frames[{position}]")
        frame_indices.append(frame_index)
        frame_nominal_times.append(nominal_time)
        main_capture_times.append(main[0])
        wrist_capture_times.append(wrist[0])
        main_paths.append(main[3])
        wrist_paths.append(wrist[3])
        source_sizes.append((main[1], main[2]))

    paired_capture_times = np.maximum(
        np.asarray(main_capture_times, dtype=np.float64),
        np.asarray(wrist_capture_times, dtype=np.float64),
    )
    strictly_increasing(paired_capture_times, name=f"{frames_path} paired capture times")
    mapping = np.searchsorted(
        paired_capture_times,
        action_target_times_array,
        side="right",
    ) - 1
    causal = mapping >= 0
    if not np.any(causal):
        raise ValueError(f"{actions_path} has no action after a causal RGB pair")
    first_causal = int(np.flatnonzero(causal)[0])
    if not np.all(causal[first_causal:]):
        raise ValueError(f"{actions_path} causal alignment has an internal negative index")
    keep = slice(first_causal, None)
    selected_positions = mapping[keep].astype(np.int64)
    selected_pair_times = paired_capture_times[selected_positions]
    kept_action_times = action_target_times_array[keep]
    image_ages = kept_action_times - selected_pair_times
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
        action_target_times=kept_action_times,
        action_source_times=np.asarray(action_source_times, dtype=np.float64)[keep],
        action_source_indices=np.asarray(action_source_indices, dtype=np.int64)[keep],
        action_steps=np.asarray(action_steps, dtype=np.int64)[keep],
        eef_poses=poses_array[keep],
        selected_frame_positions=selected_positions,
        selected_frame_indices=selected_indices,
        selected_frame_nominal_times=np.asarray(frame_nominal_times, dtype=np.float64)[
            selected_positions
        ],
        selected_main_capture_times=np.asarray(main_capture_times, dtype=np.float64)[
            selected_positions
        ],
        selected_wrist_capture_times=np.asarray(wrist_capture_times, dtype=np.float64)[
            selected_positions
        ],
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
            image = image.convert("RGB")
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
    used_positions = sorted({int(value) for value in payload.selected_frame_positions})
    caches: dict[str, dict[int, np.ndarray]] = {"main_image": {}, "wrist_image": {}}
    for position in used_positions:
        expected_size = payload.frame_source_sizes[position]
        caches["main_image"][position] = _load_resized_rgb(
            payload.main_paths[position],
            expected_size=expected_size,
            output_size=(image_width, image_height),
        )
        caches["wrist_image"][position] = _load_resized_rgb(
            payload.wrist_paths[position],
            expected_size=expected_size,
            output_size=(image_width, image_height),
        )

    block_size = 128
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        positions = payload.selected_frame_positions[start:end]
        for key, dataset in datasets.items():
            block = np.stack([caches[key][int(position)] for position in positions])
            dataset[start:end] = block


def _episode_summary(payload: EpisodePayload, *, split: str) -> dict[str, Any]:
    return {
        "demo_key": payload.row.demo_key,
        "episode_number": payload.row.episode_number,
        "run_id": payload.row.run_id,
        "collection_round": payload.row.collection_round,
        "qa_status": payload.row.qa_status,
        "split": split,
        "num_samples": payload.num_samples,
        "dropped_prefix_actions": payload.dropped_prefix_actions,
        "max_image_age_sec": float(np.max(payload.image_ages)),
        "close_position_xyz": payload.close_position.tolist(),
        "release_position_xyz": payload.release_position.tolist(),
        "simultaneous_button_rows": payload.simultaneous_button_rows,
    }


def write_round_shard(
    path: Path,
    payloads: Sequence[EpisodePayload],
    *,
    generation_id: str,
    validation_episode_numbers: set[int],
    options: BuildOptions,
    identity: dict[str, Any],
) -> dict[str, Any]:
    round_ids = {payload.row.collection_round for payload in payloads}
    if len(round_ids) != 1:
        raise ValueError(f"one shard received multiple collection rounds: {round_ids}")
    collection_round = next(iter(round_ids))
    if not payloads:
        raise ValueError(f"round {collection_round} has no payloads")

    all_keys = [payload.row.demo_key for payload in payloads]
    valid_keys = [
        payload.row.demo_key
        for payload in payloads
        if payload.row.episode_number in validation_episode_numbers
    ]
    train_keys = [key for key in all_keys if key not in set(valid_keys)]
    if not train_keys or not valid_keys:
        raise ValueError(
            f"round {collection_round} needs non-empty train and valid splits"
        )

    episode_summaries = [
        _episode_summary(
            payload,
            split=(
                "valid"
                if payload.row.episode_number in validation_episode_numbers
                else "train"
            ),
        )
        for payload in payloads
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "conversion_version": CONVERSION_VERSION,
        "generation_id": generation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": identity,
        "collection_round": collection_round,
        "alignment": {
            "master_clock": "actions.samples.target_time",
            "image_availability": "max(main_rgb.header_stamp_ns,wrist_rgb.header_stamp_ns)",
            "selection": "latest paired capture not later than action target",
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
            "translation_scale_m": TRANSLATION_SCALE_M,
            "rotation_scale_rad": ROTATION_SCALE_RAD,
            "normalization": "motion and dense gripper in [-1,1]",
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
            "method": "seeded max-min coverage over first-close EEF xy",
            "seed": options.split_seed,
            "requested_validation_count_per_round": (
                options.validation_count_per_round
            ),
            "train": train_keys,
            "valid": valid_keys,
        },
        "episodes": episode_summaries,
    }
    env_args = {
        "env_name": "PickCupReal-v0",
        "env_version": CONVERSION_VERSION,
        # Rollouts are disabled. A supported type keeps standard metadata readers happy.
        "type": 2,
        "env_kwargs": {
            "real_robot": True,
            "control_freq": int(ACTION_HZ),
            "camera_names": ["main", "wrist"],
            "camera_height": options.image_height,
            "camera_width": options.image_width,
            "task": "pick_cup_place_on_plate",
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
            demo.attrs["collection_round"] = collection_round
            demo.attrs["qa_status"] = payload.row.qa_status
            demo.attrs["dropped_prefix_actions"] = payload.dropped_prefix_actions
            demo.attrs["initial_gripper_state"] = payload.initial_gripper_state
            demo.attrs["close_position_xyz"] = payload.close_position
            demo.attrs["release_position_xyz"] = payload.release_position
            demo.attrs["simultaneous_button_rows"] = payload.simultaneous_button_rows

            demo.create_dataset("actions", data=payload.actions.astype(np.float32))
            rewards = np.zeros(payload.num_samples, dtype=np.float32)
            rewards[-1] = 1.0
            dones = np.zeros(payload.num_samples, dtype=np.uint8)
            dones[-1] = 1
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)

            obs = demo.create_group("obs")
            obs.create_dataset(
                "robot0_eef_pos",
                data=payload.eef_poses[:, :3].astype(np.float32),
            )
            obs.create_dataset(
                "robot0_eef_quat",
                data=payload.eef_poses[:, 3:7].astype(np.float32),
            )
            obs.create_dataset(
                "robot0_gripper_state",
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
                "raw_gripper_event",
                data=payload.raw_gripper_events.astype(np.float32),
            )
            provenance.create_dataset(
                "source_action_array_index",
                data=np.arange(
                    payload.dropped_prefix_actions,
                    payload.dropped_prefix_actions + payload.num_samples,
                    dtype=np.int64,
                ),
            )
            provenance.create_dataset(
                "source_action_index",
                data=payload.action_source_indices,
            )
            provenance.create_dataset("source_action_step", data=payload.action_steps)
            provenance.create_dataset(
                "action_target_time",
                data=payload.action_target_times,
            )
            provenance.create_dataset(
                "action_source_time",
                data=payload.action_source_times,
            )
            provenance.create_dataset(
                "selected_frame_position",
                data=payload.selected_frame_positions,
            )
            provenance.create_dataset(
                "selected_frame_index",
                data=payload.selected_frame_indices,
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
        mask.create_dataset("all", data=np.asarray(all_keys, dtype="S"))
        mask.create_dataset("train", data=np.asarray(train_keys, dtype="S"))
        mask.create_dataset("valid", data=np.asarray(valid_keys, dtype="S"))
        output.flush()

    return {
        "path": str(path),
        "collection_round": collection_round,
        "episodes": len(payloads),
        "train_episodes": len(train_keys),
        "valid_episodes": len(valid_keys),
        "samples": sum(payload.num_samples for payload in payloads),
        "dropped_prefix_actions": sum(
            payload.dropped_prefix_actions for payload in payloads
        ),
        "max_image_age_sec": max(
            float(np.max(payload.image_ages)) for payload in payloads
        ),
    }


def _temporary_hdf5(output_dir: Path, filename: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.",
        suffix=".partial",
        dir=output_dir,
    )
    os.close(descriptor)
    path = Path(temporary_name)
    path.unlink()
    return path


def _assert_existing_options(
    paths: Sequence[Path],
    options: BuildOptions,
) -> None:
    """Refuse to silently reuse shards produced with different settings."""

    expected = {
        "image_height": int(options.image_height),
        "image_width": int(options.image_width),
        "validation_count_per_round": int(options.validation_count_per_round),
        "split_seed": int(options.split_seed),
        "max_image_age_sec": float(options.max_image_age_sec),
        "compression": options.compression,
    }
    for path in paths:
        with h5py.File(path, "r") as dataset:
            raw_manifest = dataset.attrs.get(CONVERSION_MANIFEST_ATTR)
            if isinstance(raw_manifest, bytes):
                raw_manifest = raw_manifest.decode("utf-8")
            try:
                manifest = json.loads(raw_manifest)
                actual = {
                    "image_height": int(manifest["image"]["height"]),
                    "image_width": int(manifest["image"]["width"]),
                    "validation_count_per_round": int(
                        manifest["split"]["requested_validation_count_per_round"]
                    ),
                    "split_seed": int(manifest["split"]["seed"]),
                    "max_image_age_sec": float(
                        manifest["alignment"]["max_image_age_sec"]
                    ),
                    "compression": str(manifest["image"]["compression"]),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{path} does not record reusable conversion settings; "
                    "rebuild with --overwrite"
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
                f"{path} conversion settings do not match this request: "
                f"{mismatches}; rebuild with --overwrite"
            )


def build_datasets(options: BuildOptions) -> dict[str, Any]:
    source_root = options.source_root.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if options.image_height <= 1 or options.image_width <= 1:
        raise ValueError("image dimensions must both be greater than one")
    if options.validation_count_per_round < 1:
        raise ValueError("validation_count_per_round must be positive")
    if options.max_image_age_sec <= 0.0:
        raise ValueError("max_image_age_sec must be positive")
    _compression_kwargs(options.compression)

    final_paths = round_paths(output_dir)
    existing = [path.exists() for path in final_paths]
    if options.validate_only:
        if not all(existing):
            missing = [str(path) for path, present in zip(final_paths, existing) if not present]
            raise FileNotFoundError(f"cannot validate missing dataset shards: {missing}")
        from scripts.real_robot.validate_pick_cup_dataset import (
            validate_published_datasets,
        )

        return validate_published_datasets(output_dir, source_root=source_root)
    if any(existing) and not options.overwrite:
        if not all(existing):
            raise FileExistsError(
                "only some output shards exist; remove them or rebuild with --overwrite"
            )
        _assert_existing_options(final_paths, options)
        from scripts.real_robot.validate_pick_cup_dataset import (
            validate_published_datasets,
        )

        report = validate_published_datasets(output_dir, source_root=source_root)
        report["reused_existing"] = True
        return report

    rows = eligible_rows(source_root)
    by_round = {
        round_id: [row for row in rows if row.collection_round == round_id]
        for round_id in (1, 2)
    }
    for round_id, round_rows in by_round.items():
        if len(round_rows) < 2:
            raise ValueError(
                f"collection round {round_id} needs at least two eligible episodes"
            )

    payloads: dict[int, EpisodePayload] = {}
    for row in rows:
        print(
            f"[pick_cup dataset] auditing episode {row.episode_number:03d} "
            f"({row.run_id})",
            flush=True,
        )
        payloads[row.episode_number] = load_episode_payload(
            source_root,
            row,
            max_image_age_sec=options.max_image_age_sec,
        )

    validation_numbers: dict[int, set[int]] = {}
    for round_id, round_rows in by_round.items():
        close_xy = {
            row.episode_number: payloads[row.episode_number].close_position[:2]
            for row in round_rows
        }
        validation_numbers[round_id] = select_spatial_validation_rows(
            round_rows,
            close_xy,
            count=options.validation_count_per_round,
            seed=options.split_seed + round_id,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = source_identity(source_root)
    generation_id = uuid.uuid4().hex
    temporary_paths = tuple(
        _temporary_hdf5(output_dir, ROUND_FILENAMES[round_id])
        for round_id in (1, 2)
    )
    summaries: list[dict[str, Any]] = []
    try:
        for round_id, temporary_path in zip((1, 2), temporary_paths):
            print(
                f"[pick_cup dataset] writing round {round_id}: {temporary_path}",
                flush=True,
            )
            summaries.append(
                write_round_shard(
                    temporary_path,
                    [payloads[row.episode_number] for row in by_round[round_id]],
                    generation_id=generation_id,
                    validation_episode_numbers=validation_numbers[round_id],
                    options=options,
                    identity=identity,
                )
            )

        from scripts.real_robot.validate_pick_cup_dataset import (
            validate_datasets,
            validate_published_datasets,
        )

        # Validate the complete temporary pair before publishing either shard.
        validate_datasets(temporary_paths, source_root=source_root)

        # A directory cannot atomically rename two files at once. Preserve old
        # shards for exception rollback and use a generation-bound commit marker
        # as the publication point. Readers reject missing or stale markers.
        backups: dict[Path, Path] = {}
        published: list[Path] = []
        try:
            for final_path in final_paths:
                if final_path.exists():
                    backup = _temporary_hdf5(
                        output_dir,
                        f"{final_path.name}.backup",
                    )
                    os.replace(final_path, backup)
                    backups[final_path] = backup
            for temporary_path, final_path in zip(temporary_paths, final_paths):
                os.replace(temporary_path, final_path)
                published.append(final_path)
            validate_datasets(final_paths, source_root=source_root)
            commit = {
                "schema_version": SCHEMA_VERSION,
                "conversion_version": CONVERSION_VERSION,
                "generation_id": generation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_identity": identity,
                "shards": [
                    {
                        "filename": path.name,
                        "size_bytes": int(path.stat().st_size),
                    }
                    for path in final_paths
                ],
            }
            atomic_write_json(dataset_commit_path(output_dir), commit)
        except Exception:
            for final_path in published:
                final_path.unlink(missing_ok=True)
            for final_path, backup in backups.items():
                os.replace(backup, final_path)
            raise
        else:
            for backup in backups.values():
                backup.unlink(missing_ok=True)

        validation = validate_published_datasets(
            output_dir,
            source_root=source_root,
        )
        final_summaries = []
        for summary, final_path in zip(summaries, final_paths):
            summary = dict(summary)
            summary["path"] = str(final_path)
            final_summaries.append(summary)
        report = {
            "conversion_version": CONVERSION_VERSION,
            "source": str(source_root),
            "output_dir": str(output_dir),
            "shards": final_summaries,
            "validation": validation,
            "reused_existing": False,
        }
        atomic_write_json(output_dir / "conversion_summary.json", report)
        return report
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--validation-count-per-round", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--max-image-age-sec",
        type=float,
        default=DEFAULT_MAX_IMAGE_AGE_SEC,
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = build_datasets(
        BuildOptions(
            source_root=args.source,
            output_dir=args.output_dir,
            image_height=args.image_height,
            image_width=args.image_width,
            validation_count_per_round=args.validation_count_per_round,
            split_seed=args.split_seed,
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
