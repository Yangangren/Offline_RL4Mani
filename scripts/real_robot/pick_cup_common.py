#!/usr/bin/env python3
"""Shared contracts for the Real4D pick-cup RGB Diffusion Policy baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/home/ryan/datasets/pick_cup")
DEFAULT_DATASET_DIR = ROOT / "datasets/real_robot/pick_cup"
ROUND_FILENAMES = {1: "round1_rgb.hdf5", 2: "round2_rgb.hdf5"}

SCHEMA_VERSION = 1
CONVERSION_VERSION = "pick_cup_rgb_dp_v1"
CONVERSION_MANIFEST_ATTR = "real_robot_conversion_manifest"
DATASET_COMMIT_FILENAME = "dataset_commit.json"

RGB_KEYS = ("main_image", "wrist_image")
LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
)
OBS_KEYS = (*RGB_KEYS, *LOW_DIM_KEYS)

ACTION_HZ = 20.0
IMAGE_HZ = 5.0
TRANSLATION_SCALE_M = 0.012
ROTATION_SCALE_RAD = 0.036
DEFAULT_IMAGE_HEIGHT = 96
DEFAULT_IMAGE_WIDTH = 128
DEFAULT_CROP_HEIGHT = 84
DEFAULT_CROP_WIDTH = 112
DEFAULT_MAX_IMAGE_AGE_SEC = 0.5


@dataclass(frozen=True)
class EpisodeIndexRow:
    """One authoritative row from ``episode_index.csv``."""

    episode_number: int
    run_id: str
    directory: str
    qa_status: str
    training_eligible: bool
    review_reason: str

    @property
    def collection_round(self) -> int:
        return 1 if self.episode_number <= 50 else 2

    @property
    def demo_key(self) -> str:
        return f"demo_{self.episode_number:03d}"


def parse_bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean for {field}: {value!r}")


def read_episode_index(source_root: Path) -> list[EpisodeIndexRow]:
    """Read and validate the dataset's eligibility index."""

    index_path = source_root / "episode_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"episode index does not exist: {index_path}")

    rows: list[EpisodeIndexRow] = []
    seen_numbers: set[int] = set()
    with index_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "episode_number",
            "run_id",
            "qa_status",
            "training_eligible_43qa",
            "review_reason",
            "directory",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{index_path} is missing required columns: {sorted(missing)}"
            )
        for raw in reader:
            number = int(raw["episode_number"])
            if number in seen_numbers:
                raise ValueError(f"duplicate episode_number={number} in {index_path}")
            seen_numbers.add(number)
            row = EpisodeIndexRow(
                episode_number=number,
                run_id=raw["run_id"].strip(),
                directory=raw["directory"].strip(),
                qa_status=raw["qa_status"].strip(),
                training_eligible=parse_bool(
                    raw["training_eligible_43qa"],
                    field="training_eligible_43qa",
                ),
                review_reason=raw["review_reason"].strip(),
            )
            if not row.run_id or not row.directory:
                raise ValueError(f"episode {number} has an empty run_id or directory")
            rows.append(row)

    if not rows:
        raise ValueError(f"no episodes found in {index_path}")
    return sorted(rows, key=lambda row: row.episode_number)


def eligible_rows(source_root: Path) -> list[EpisodeIndexRow]:
    rows = [row for row in read_episode_index(source_root) if row.training_eligible]
    if not rows:
        raise ValueError(f"no training-eligible episodes in {source_root}")
    return rows


def resolve_episode_dir(source_root: Path, row: EpisodeIndexRow) -> Path:
    directory = (source_root / row.directory).resolve()
    source_resolved = source_root.resolve()
    try:
        directory.relative_to(source_resolved)
    except ValueError as exc:
        raise ValueError(
            f"episode {row.episode_number} escapes source root: {row.directory}"
        ) from exc
    if not directory.is_dir():
        raise FileNotFoundError(
            f"episode {row.episode_number} directory does not exist: {directory}"
        )
    return directory


def load_json(path: Path) -> Any:
    try:
        with path.open() as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source_root: Path) -> dict[str, Any]:
    """Fingerprint small authoritative metadata without hashing 5 GB of images."""

    filenames = (
        "episode_index.csv",
        "manifest.json",
        "metadata_checksums.sha256",
        "FULL_SEQUENCE_PACKAGE.txt",
    )
    files: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        path = source_root / filename
        if path.is_file():
            stat = path.stat()
            files[filename] = {
                "size": int(stat.st_size),
                "sha256": file_sha256(path),
            }
    if "episode_index.csv" not in files:
        raise FileNotFoundError(source_root / "episode_index.csv")
    return {
        "root": str(source_root.resolve()),
        "files": files,
    }


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
    """Return pre-command observations and dense post-command targets.

    Real4D stores sparse events despite its handoff document calling the channel
    absolute. ``-1`` closes, ``+1`` opens, and ``0`` holds the prior command.
    The observation at index ``t`` is deliberately recorded before applying the
    event at ``t`` so the gripper target is not leaked into the policy input.
    """

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


def select_spatial_validation_rows(
    rows: Sequence[EpisodeIndexRow],
    close_xy: dict[int, np.ndarray],
    *,
    count: int,
    seed: int,
) -> set[int]:
    """Select a reproducible max-min coverage subset for validation."""

    if len(rows) < 2:
        raise ValueError("each collection round needs at least two eligible episodes")
    count = min(max(1, int(count)), len(rows) - 1)
    points = np.stack(
        [as_float_array(close_xy[row.episode_number], shape=(2,), name="close_xy") for row in rows]
    )
    span = np.ptp(points, axis=0)
    span[span < 1e-9] = 1.0
    normalized = (points - np.min(points, axis=0)) / span

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, len(rows)))]
    while len(selected) < count:
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :],
            axis=-1,
        )
        min_distance = np.min(distances, axis=1)
        min_distance[np.asarray(selected)] = -1.0
        # Seeded jitter makes exact geometric ties deterministic but seed-sensitive.
        min_distance += rng.uniform(0.0, 1e-12, size=min_distance.shape)
        selected.append(int(np.argmax(min_distance)))
    return {rows[index].episode_number for index in selected}


def decode_hdf5_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def round_paths(dataset_dir: Path) -> tuple[Path, Path]:
    return tuple(dataset_dir / ROUND_FILENAMES[round_id] for round_id in (1, 2))


def dataset_commit_path(dataset_dir: Path) -> Path:
    return dataset_dir / DATASET_COMMIT_FILENAME
