#!/usr/bin/env python3
"""Shared, fail-closed contracts for the real-robot stack-cup dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/home/ryan/datasets/stack_cup/human_demo")
DEFAULT_DATASET_DIR = ROOT / "datasets/real_robot/stack_cup"
DATASET_FILENAME = "stack_cup_rgb.hdf5"
DATASET_COMMIT_FILENAME = "dataset_commit.json"
CONVERSION_SUMMARY_FILENAME = "conversion_summary.json"

SCHEMA_VERSION = 1
CONVERSION_VERSION = "stack_cup_rgb_dp_v1"
CONVERSION_MANIFEST_ATTR = "real_robot_conversion_manifest"

ACTION_HZ = 20.0
IMAGE_HZ = 5.0
TRANSLATION_SCALE_M = 0.012
ROTATION_SCALE_RAD = 0.036
DEFAULT_IMAGE_HEIGHT = 96
DEFAULT_IMAGE_WIDTH = 128
DEFAULT_CROP_HEIGHT = 84
DEFAULT_CROP_WIDTH = 112
DEFAULT_MAX_IMAGE_AGE_SEC = 0.5

RGB_KEYS = ("main_image", "wrist_image")
LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
)
OBS_KEYS = (*RGB_KEYS, *LOW_DIM_KEYS)

EXPECTED_EPISODE_NUMBERS = frozenset(range(1, 51))
EXCLUDED_EPISODES = {
    7: (
        "two consecutive 5 Hz RGB targets were dropped; strict causal "
        "alignment reaches 0.617498 s and violates the 0.5 s image-age contract"
    ),
}
VALIDATION_EPISODE_NUMBERS = frozenset({4, 24, 27, 40, 46})

_EPISODE_PATTERN = re.compile(r"^episode_(\d{3})__(.+)$")
_SOURCE_METADATA_FILENAMES = (
    "actions.json",
    "frames.json",
    "contract.json",
    "qa.json",
    "windows.json",
    "synchronization.json",
    "motion_trim.json",
    "snapshots/collector_manifest.json",
)


@dataclass(frozen=True)
class StackCupEpisodeRow:
    """Authoritative metadata for one source episode."""

    episode_number: int
    run_id: str
    directory: str
    qa_status: str
    model_window_ready: bool
    invalid_windows: int
    manifest_actions: int
    manifest_frames: int
    removed_actions: int
    removed_frames: int

    @property
    def demo_key(self) -> str:
        return f"demo_{self.episode_number:03d}"

    @property
    def excluded(self) -> bool:
        return self.episode_number in EXCLUDED_EPISODES

    @property
    def clean(self) -> bool:
        return self.qa_status == "PASS" and self.invalid_windows == 0


def load_json(path: Path) -> Any:
    try:
        with path.open() as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


def as_float_array(value: Any, *, shape: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def strictly_increasing(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def densify_gripper_events(
    raw_events: Iterable[float],
    *,
    initial_state: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pre-event observations and dense post-event action targets."""

    events = np.asarray(list(raw_events), dtype=np.float32)
    if events.ndim != 1:
        raise ValueError(f"gripper events must be 1-D, got {events.shape}")
    if not np.all(np.isfinite(events)):
        raise ValueError("gripper events contain non-finite values")
    valid = np.isin(events, np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))
    if not np.all(valid):
        bad = sorted({float(value) for value in events[~valid]})
        raise ValueError(f"gripper events must be in {{-1, 0, 1}}, got {bad}")
    if initial_state not in (-1.0, 1.0):
        raise ValueError(f"initial gripper state must be -1 or 1, got {initial_state}")

    observations = np.empty(events.shape, dtype=np.float32)
    targets = np.empty(events.shape, dtype=np.float32)
    state = float(initial_state)
    for index, event in enumerate(events):
        observations[index] = state
        if event < 0.0:
            state = -1.0
        elif event > 0.0:
            state = 1.0
        targets[index] = state
    return observations, targets


def resolve_episode_dir(source_root: Path, row: StackCupEpisodeRow) -> Path:
    source_resolved = source_root.resolve()
    directory = (source_resolved / row.directory).resolve()
    try:
        directory.relative_to(source_resolved)
    except ValueError as exc:
        raise ValueError(
            f"episode {row.episode_number:03d} escapes source root: {row.directory}"
        ) from exc
    if not directory.is_dir():
        raise FileNotFoundError(
            f"episode {row.episode_number:03d} directory does not exist: {directory}"
        )
    return directory


def _required_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def read_episode_rows(source_root: Path) -> list[StackCupEpisodeRow]:
    """Read and cross-check the 50-episode handoff manifest."""

    source_root = source_root.expanduser().resolve()
    manifest_path = source_root / "manifest.json"
    manifest = _required_object(load_json(manifest_path), name=str(manifest_path))
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{manifest_path}.episodes must be a non-empty list")
    if int(manifest.get("episode_count", -1)) != len(raw_episodes):
        raise ValueError(
            f"{manifest_path} episode_count does not match episodes list"
        )

    episodes_root = source_root / "episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError(episodes_root)
    listed_directories = {
        raw.get("episode")
        for raw in raw_episodes
        if isinstance(raw, dict) and isinstance(raw.get("episode"), str)
    }
    actual_directories = {
        path.name
        for path in episodes_root.iterdir()
        if path.is_dir() and path.name.startswith("episode_")
    }
    if listed_directories != actual_directories:
        raise ValueError(
            "manifest episode inventory differs from filesystem; "
            f"missing={sorted(listed_directories - actual_directories)}, "
            f"unlisted={sorted(actual_directories - listed_directories)}"
        )

    rows: list[StackCupEpisodeRow] = []
    seen: set[int] = set()
    for position, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict):
            raise ValueError(f"{manifest_path}.episodes[{position}] must be an object")
        directory_name = raw.get("episode")
        if not isinstance(directory_name, str):
            raise ValueError(
                f"{manifest_path}.episodes[{position}].episode must be a string"
            )
        match = _EPISODE_PATTERN.fullmatch(directory_name)
        if match is None:
            raise ValueError(f"invalid episode directory name: {directory_name!r}")
        number = int(match.group(1))
        run_id_from_name = match.group(2)
        if number in seen:
            raise ValueError(f"duplicate episode number {number:03d} in {manifest_path}")
        seen.add(number)
        directory = f"episodes/{directory_name}"
        episode_dir = (source_root / directory).resolve()
        try:
            episode_dir.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"episode path escapes source root: {directory}") from exc
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"episode directory does not exist: {episode_dir}")

        contract_path = episode_dir / "contract.json"
        qa_path = episode_dir / "qa.json"
        windows_path = episode_dir / "windows.json"
        contract = _required_object(load_json(contract_path), name=str(contract_path))
        qa = _required_object(load_json(qa_path), name=str(qa_path))
        windows = _required_object(load_json(windows_path), name=str(windows_path))
        run_id = str(contract.get("run_id", ""))
        if run_id != run_id_from_name:
            raise ValueError(
                f"{contract_path} run_id {run_id!r} does not match {run_id_from_name!r}"
            )
        if qa.get("run_id") != run_id:
            raise ValueError(f"{qa_path} run_id does not match {run_id!r}")
        qa_status = str(qa.get("status", ""))
        if qa_status not in {"PASS", "WARN"}:
            raise ValueError(f"{qa_path} has unsupported status {qa_status!r}")
        invalid_windows = int(windows.get("invalid_windows", -1))
        valid_windows = int(windows.get("valid_windows", -1))
        num_windows = int(windows.get("num_windows", -1))
        if min(invalid_windows, valid_windows) < 0 or (
            invalid_windows + valid_windows != num_windows
        ):
            raise ValueError(f"inconsistent window counts in {windows_path}")

        model_window_ready = (episode_dir / "MODEL_WINDOW_READY").is_file()
        if model_window_ready != (qa_status == "PASS"):
            raise ValueError(
                f"{episode_dir} MODEL_WINDOW_READY marker disagrees with "
                f"qa.status={qa_status}"
            )
        manifest_actions = int(raw.get("actions", -1))
        manifest_frames = int(raw.get("frames", -1))
        removed_actions = int(raw.get("removed_actions", -1))
        removed_frames = int(raw.get("removed_frames", -1))
        if min(manifest_actions, manifest_frames) <= 0:
            raise ValueError(f"invalid action/frame count for episode {number:03d}")
        if min(removed_actions, removed_frames) < 0:
            raise ValueError(f"invalid removed count for episode {number:03d}")

        rows.append(
            StackCupEpisodeRow(
                episode_number=number,
                run_id=run_id,
                directory=directory,
                qa_status=qa_status,
                model_window_ready=model_window_ready,
                invalid_windows=invalid_windows,
                manifest_actions=manifest_actions,
                manifest_frames=manifest_frames,
                removed_actions=removed_actions,
                removed_frames=removed_frames,
            )
        )

    if seen != EXPECTED_EPISODE_NUMBERS:
        raise ValueError(
            "stack-cup handoff must contain exactly episodes 001..050; "
            f"missing={sorted(EXPECTED_EPISODE_NUMBERS - seen)}, "
            f"unexpected={sorted(seen - EXPECTED_EPISODE_NUMBERS)}"
        )
    if not VALIDATION_EPISODE_NUMBERS.issubset(seen - set(EXCLUDED_EPISODES)):
        raise ValueError("fixed validation episodes are absent or excluded")
    return sorted(rows, key=lambda row: row.episode_number)


def included_rows(source_root: Path) -> list[StackCupEpisodeRow]:
    rows = [row for row in read_episode_rows(source_root) if not row.excluded]
    if len(rows) != 49:
        raise ValueError(f"expected 49 included stack-cup episodes, got {len(rows)}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": int(stat.st_size), "sha256": file_sha256(path)}


def source_identity(source_root: Path) -> dict[str, Any]:
    """Fingerprint authoritative metadata for all included and excluded episodes."""

    source_root = source_root.expanduser().resolve()
    rows = read_episode_rows(source_root)
    files: dict[str, dict[str, Any]] = {}
    for filename in ("manifest.json", "PROCESSING_COMPLETE"):
        path = source_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        files[filename] = _file_identity(path)

    episode_files: dict[str, dict[str, Any]] = {}
    rgb_inventory = hashlib.sha256()
    rgb_file_count = 0
    rgb_total_bytes = 0
    for row in rows:
        episode_dir = resolve_episode_dir(source_root, row)
        identities: dict[str, Any] = {}
        for filename in _SOURCE_METADATA_FILENAMES:
            path = episode_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            identities[filename] = _file_identity(path)
        marker = episode_dir / "MODEL_WINDOW_READY"
        if marker.is_file():
            identities[marker.name] = _file_identity(marker)
        episode_files[row.demo_key] = {
            "directory": row.directory,
            "files": identities,
        }

        frames_document = _required_object(
            load_json(episode_dir / "frames.json"),
            name=str(episode_dir / "frames.json"),
        )
        frames = frames_document.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"{episode_dir / 'frames.json'}.frames must be a list")
        for frame in frames:
            streams = frame.get("streams") if isinstance(frame, dict) else None
            if not isinstance(streams, dict):
                raise ValueError(f"invalid stream metadata in {episode_dir / 'frames.json'}")
            for stream_name in ("main_rgb", "wrist_rgb"):
                stream = streams.get(stream_name)
                relative = stream.get("path") if isinstance(stream, dict) else None
                if not isinstance(relative, str) or not relative:
                    raise ValueError(
                        f"invalid {stream_name} path in {episode_dir / 'frames.json'}"
                    )
                image_path = (episode_dir / relative).resolve()
                try:
                    image_path.relative_to(episode_dir.resolve())
                except ValueError as exc:
                    raise ValueError(f"RGB path escapes episode: {relative}") from exc
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                stat = image_path.stat()
                record = (
                    f"{row.demo_key}/{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
                )
                rgb_inventory.update(record.encode("utf-8"))
                rgb_file_count += 1
                rgb_total_bytes += int(stat.st_size)

    return {
        "root": str(source_root),
        "files": files,
        "episodes": episode_files,
        "rgb_inventory": {
            "file_count": rgb_file_count,
            "total_bytes": rgb_total_bytes,
            "path_size_mtime_sha256": rgb_inventory.hexdigest(),
        },
    }


def dataset_path(output_dir: Path) -> Path:
    return output_dir / DATASET_FILENAME


def dataset_commit_path(output_dir: Path) -> Path:
    return output_dir / DATASET_COMMIT_FILENAME


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def decode_hdf5_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]
