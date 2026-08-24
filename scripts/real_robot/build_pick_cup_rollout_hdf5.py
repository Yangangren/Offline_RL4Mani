#!/usr/bin/env python3
"""Convert the successful and failed pick-cup policy rollouts to HDF5.

The rollout logger saved the action actually proposed by Diffusion Policy, the
pose immediately before dispatch, exact observation timestamps, golden NPZ
inputs at every replan boundary, and a ROS 2 SQLite bag.  This converter uses
those records without importing ROS:

* ``raw_action`` is the normalized seven-dimensional training action;
* ``pose_before`` and the pre-action logical gripper state are observations;
* standard little-endian CDR ``sensor_msgs/msg/Image`` messages are decoded
  directly from SQLite;
* synchronized raw RealSense captures are sampled on a wall-clock 5 Hz grid
  anchored at the first action state, then causally held on 20 Hz rows; and
* every saved golden input, timestamp, and digest must match before an episode
  is admitted.

The script deliberately fails instead of guessing when the logged temporal
semantics cannot be reconstructed exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot.pick_cup_common import (  # noqa: E402
    CONVERSION_MANIFEST_ATTR,
)


SCHEMA = "real4d.robomimic.pick_cup_rollouts.v1"
CONVERSION_VERSION = "pick_cup_epoch200_20hz_rollouts_v2"
DEFAULT_SOURCE = Path("/home/ryan/datasets/pick_cup/rollout")
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/pick_cup/idql/pick_cup_epoch200_20hz_rollouts.hdf5"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "0d37bc1e57987d603ef46c4808f87e3b8ae281b673b6cb4e3bf07b9666b87742"
)
EXPECTED_CONTRACT_SHA256 = (
    "3d9b7a33ded0594a97ff1dd54d9f7240d0fe27a3738dd14a7506653c5f724a08"
)
MAIN_TOPIC = "/main/main_camera/color/image_raw"
WRIST_TOPIC = "/wrist/wrist_camera/color/image_raw"
OBS_KEYS = (
    "main_image",
    "wrist_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
)
ACTION_SCALE = np.asarray(
    [0.012, 0.012, 0.012, 0.036, 0.036, 0.036, 1.0], dtype=np.float32
)
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128
RGB_ALIGNMENT_DESCRIPTION = (
    "wall-clock 5 Hz grid anchored at first precommand state; latest "
    "synchronized causal ROS-header pair per tick; held only while "
    "rows share a tick; every row image age <=0.5s"
)
EPISODE_RE = re.compile(
    r"^epoch200_20hz_real_(?P<day>\d{8})_(?P<clock>\d{6})$"
)
ENV_ARGS = {
    "env_name": "PickCupReal-v0",
    "env_version": "pick_cup_rgb_dp_v1",
    "type": 2,
    "env_kwargs": {
        "real_robot": True,
        "task": "pick_cup_place_on_plate",
        "control_freq": 20,
        "camera_names": ["main", "wrist"],
        "camera_height": IMAGE_HEIGHT,
        "camera_width": IMAGE_WIDTH,
    },
}
FULL_DEFAULT_EXPECTED_ACTIONS = 400
FULL_DEFAULT_SUCCESS_VALID_COUNT = 6
FULL_DEFAULT_FAILURE_VALID_COUNT = 3
EXPECTED_FULL_OUTCOME_COUNTS = {"success": 29, "failure": 14}
EXPECTED_FULL_SAMPLES: int | None = 17_193
EXPECTED_FULL_DROPPED_PREFIX_ACTIONS: int | None = 7
# Audited task profiles may allow an exact, finite set of adjacent equal
# precommand state timestamps. Backward timestamps and every unlisted repeat
# remain fatal. Values are zero-based edge indices (row i -> row i + 1).
ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES: dict[str, tuple[int, ...]] = {}
STARTUP_PREFIX_TRIM_ACTION_LIMIT = 8


class RolloutConversionError(ValueError):
    """A rollout is incomplete, unsafe, or cannot be reconstructed exactly."""


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output: Path = DEFAULT_OUTPUT
    compression: str = "gzip"
    split_seed: int = 1
    success_valid_count: int = 6
    failure_valid_count: int = 3
    expected_actions: int = 400
    action_horizon: int = 8
    episode_limit: int | None = None
    overwrite: bool = False
    validate_only: bool = False


@dataclass(frozen=True)
class EpisodeSource:
    path: Path
    episode_id: str
    outcome: str
    day: str


@dataclass(frozen=True)
class ImageRecord:
    message_id: int
    bag_timestamp_ns: int
    header_timestamp: float
    height: int
    width: int
    encoding: str
    step: int


@dataclass
class EpisodePayload:
    source: EpisodeSource
    actions: np.ndarray
    physical_actions: np.ndarray
    commands: np.ndarray
    gripper_predictions: np.ndarray
    gripper_commands: np.ndarray
    gripper_decisions: np.ndarray
    gripper_intervened: np.ndarray
    gripper_requested: np.ndarray
    gripper_success: np.ndarray
    poses: np.ndarray
    gripper_observations: np.ndarray
    action_state_times: np.ndarray
    action_unix_times: np.ndarray
    chunks: np.ndarray
    substeps: np.ndarray
    main_records: list[ImageRecord]
    wrist_records: list[ImageRecord]
    main_row_positions: np.ndarray
    wrist_row_positions: np.ndarray
    logical_tick_indices: np.ndarray
    logical_tick_times: np.ndarray
    logical_tick_origin_time: float
    dropped_prefix_actions: int
    prefix_trim_reason: str | None
    main_images: dict[int, np.ndarray]
    wrist_images: dict[int, np.ndarray]
    checkpoint_sha256: str
    contract_sha256: str
    runner_sha256: str
    capture_recoveries: int
    golden_chunks: int
    golden_input_recoveries: list[dict[str, Any]]
    golden_npz_identity: dict[str, Any]
    bag_identity: dict[str, Any]
    source_file_sha256: dict[str, str]

    @property
    def num_samples(self) -> int:
        return int(self.actions.shape[0])


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise RolloutConversionError(f"malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RolloutConversionError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise RolloutConversionError(
            f"{name} has shape {array.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise RolloutConversionError(f"{name} contains NaN or Inf")
    return array


def discover_episodes(source_root: Path) -> list[EpisodeSource]:
    source_root = source_root.expanduser().resolve()
    episodes: list[EpisodeSource] = []
    for outcome in ("success", "failure"):
        outcome_root = source_root / outcome
        if not outcome_root.is_dir():
            raise FileNotFoundError(f"missing rollout outcome directory: {outcome_root}")
        for path in sorted(outcome_root.iterdir()):
            if not path.is_dir():
                continue
            match = EPISODE_RE.fullmatch(path.name)
            if match is None:
                raise RolloutConversionError(
                    f"unexpected rollout directory name under {outcome_root}: {path.name}"
                )
            episodes.append(
                EpisodeSource(
                    path=path.resolve(),
                    episode_id=path.name,
                    outcome=outcome,
                    day=match.group("day"),
                )
            )
    if not episodes:
        raise RolloutConversionError(f"no rollout episodes found in {source_root}")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise RolloutConversionError("rollout episode IDs are not unique")
    return sorted(episodes, key=lambda episode: episode.episode_id)


def _read_steps(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RolloutConversionError(
                    f"malformed JSONL {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RolloutConversionError(
                    f"expected object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


class _CdrReader:
    """Small CDR1 reader for the fields in ``sensor_msgs/msg/Image``."""

    def __init__(self, value: bytes | memoryview):
        self.value = memoryview(value)
        if len(self.value) < 4:
            raise RolloutConversionError("truncated CDR encapsulation")
        representation = bytes(self.value[:2])
        if representation != b"\x00\x01":
            raise RolloutConversionError(
                "only standard little-endian CDR sensor_msgs/Image is supported; "
                f"encapsulation={representation.hex()}"
            )
        self.offset = 4

    def align(self, size: int) -> None:
        self.offset += (-self.offset) % size

    def unpack(self, format_: str, alignment: int) -> int:
        self.align(alignment)
        size = struct.calcsize(format_)
        if self.offset + size > len(self.value):
            raise RolloutConversionError("truncated CDR primitive")
        result = struct.unpack_from("<" + format_, self.value, self.offset)[0]
        self.offset += size
        return int(result)

    def string(self) -> str:
        size = self.unpack("I", 4)
        if size < 1 or self.offset + size > len(self.value):
            raise RolloutConversionError("invalid or truncated CDR string")
        encoded = bytes(self.value[self.offset : self.offset + size])
        self.offset += size
        if encoded[-1:] != b"\0":
            raise RolloutConversionError("CDR string lacks a null terminator")
        try:
            return encoded[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RolloutConversionError("CDR string is not UTF-8") from exc


def decode_sensor_image_cdr(
    value: bytes | memoryview,
    *,
    metadata_only: bool = False,
) -> tuple[float, int, int, str, int, np.ndarray | None]:
    """Decode a little-endian CDR ``sensor_msgs/msg/Image`` without ROS."""

    reader = _CdrReader(value)
    seconds = reader.unpack("i", 4)
    nanoseconds = reader.unpack("I", 4)
    if nanoseconds >= 1_000_000_000:
        raise RolloutConversionError(f"invalid ROS timestamp nanoseconds={nanoseconds}")
    reader.string()  # header.frame_id
    height = reader.unpack("I", 4)
    width = reader.unpack("I", 4)
    encoding = reader.string()
    is_bigendian = reader.unpack("B", 1)
    step = reader.unpack("I", 4)
    data_size = reader.unpack("I", 4)
    header_timestamp = float(seconds) + float(nanoseconds) * 1e-9
    if encoding != "rgb8":
        raise RolloutConversionError(f"expected rgb8 image, got {encoding!r}")
    if is_bigendian != 0:
        raise RolloutConversionError("big-endian rgb8 image is unsupported")
    if height <= 0 or width <= 0 or step != width * 3:
        raise RolloutConversionError(
            f"invalid packed rgb8 geometry: height={height}, width={width}, step={step}"
        )
    expected_size = height * step
    if data_size != expected_size:
        raise RolloutConversionError(
            f"rgb8 data has {data_size} bytes, expected {expected_size}"
        )
    if metadata_only:
        return header_timestamp, height, width, encoding, step, None
    if reader.offset + data_size != len(reader.value):
        raise RolloutConversionError(
            "CDR image payload length does not match its sequence length"
        )
    image = np.frombuffer(
        reader.value[reader.offset : reader.offset + data_size], dtype=np.uint8
    ).reshape(height, width, 3)
    return header_timestamp, height, width, encoding, step, image


def _open_bag(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"missing ROS bag SQLite database: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _topic_id(connection: sqlite3.Connection, topic: str) -> int:
    rows = connection.execute(
        "SELECT id, type, serialization_format FROM topics WHERE name = ?", (topic,)
    ).fetchall()
    if len(rows) != 1:
        raise RolloutConversionError(f"expected exactly one topic {topic!r}, got {rows}")
    topic_id, message_type, serialization = rows[0]
    if message_type != "sensor_msgs/msg/Image" or serialization != "cdr":
        raise RolloutConversionError(
            f"{topic} has type/serialization {message_type}/{serialization}"
        )
    return int(topic_id)


def _image_records(
    connection: sqlite3.Connection,
    topic: str,
) -> list[ImageRecord]:
    topic_id = _topic_id(connection, topic)
    # The Image header and geometry fit in this prefix. This avoids reading all
    # 0.9 MB image payloads merely to select the causal rows.
    rows = connection.execute(
        "SELECT id, timestamp, substr(data, 1, 512) "
        "FROM messages WHERE topic_id = ? ORDER BY timestamp, id",
        (topic_id,),
    ).fetchall()
    records: list[ImageRecord] = []
    for message_id, bag_timestamp_ns, prefix in rows:
        stamp, height, width, encoding, step, _ = decode_sensor_image_cdr(
            prefix, metadata_only=True
        )
        records.append(
            ImageRecord(
                message_id=int(message_id),
                bag_timestamp_ns=int(bag_timestamp_ns),
                header_timestamp=stamp,
                height=height,
                width=width,
                encoding=encoding,
                step=step,
            )
        )
    if not records:
        raise RolloutConversionError(f"ROS bag contains no messages for {topic}")
    stamps = np.asarray([record.header_timestamp for record in records])
    if np.any(np.diff(stamps) <= 0.0):
        raise RolloutConversionError(f"{topic} header timestamps are not increasing")
    return records


def _fetch_resized_image(
    connection: sqlite3.Connection,
    record: ImageRecord,
) -> np.ndarray:
    row = connection.execute(
        "SELECT data FROM messages WHERE id = ?", (record.message_id,)
    ).fetchone()
    if row is None:
        raise RolloutConversionError(f"missing ROS bag message id={record.message_id}")
    stamp, height, width, encoding, step, image = decode_sensor_image_cdr(row[0])
    if image is None:
        raise AssertionError("full CDR decode did not return pixels")
    actual = (stamp, height, width, encoding, step)
    expected = (
        record.header_timestamp,
        record.height,
        record.width,
        record.encoding,
        record.step,
    )
    if actual != expected:
        raise RolloutConversionError(
            f"ROS image metadata changed between prefix/full reads: {actual} != {expected}"
        )
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT), resample=Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )
    if resized.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise RolloutConversionError(f"unexpected resized image shape {resized.shape}")
    return np.ascontiguousarray(resized)


def _find_stamp_index(
    records: Sequence[ImageRecord],
    target: float,
    *,
    name: str,
    tolerance: float = 2e-6,
) -> int:
    stamps = np.asarray([record.header_timestamp for record in records])
    position = int(np.searchsorted(stamps, target))
    candidates = [index for index in (position - 1, position) if 0 <= index < len(records)]
    if not candidates:
        raise RolloutConversionError(f"cannot find {name} timestamp {target:.9f}")
    best = min(candidates, key=lambda index: abs(stamps[index] - target))
    error = abs(float(stamps[best]) - float(target))
    if error > tolerance:
        raise RolloutConversionError(
            f"{name} timestamp {target:.9f} is absent from ROS bag; nearest error={error:.9g}s"
        )
    return best


def _select_synchronized_camera_rows(
    main: Sequence[ImageRecord],
    wrist: Sequence[ImageRecord],
    target_times: np.ndarray,
    *,
    max_skew_sec: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the latest causal synchronized raw pair for each logical row.

    The raw RealSense topics are nominally 15 Hz but can independently drop or
    restart, so record-index phases are not stable. The logical 5 Hz cadence is
    defined by the caller's discrete targets (one per four action rows). At
    each target we walk back the newer camera until the pair is causal and
    within the deployment skew bound.
    """

    targets = np.asarray(target_times, dtype=np.float64)
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ValueError("camera target times must be a finite vector")
    main_stamps = np.asarray([record.header_timestamp for record in main])
    wrist_stamps = np.asarray([record.header_timestamp for record in wrist])
    selected_main: list[int] = []
    selected_wrist: list[int] = []
    for row, target in enumerate(targets):
        main_index = int(np.searchsorted(main_stamps, target, side="right") - 1)
        wrist_index = int(np.searchsorted(wrist_stamps, target, side="right") - 1)
        while main_index >= 0 and wrist_index >= 0:
            main_stamp = main_stamps[main_index]
            wrist_stamp = wrist_stamps[wrist_index]
            if abs(main_stamp - wrist_stamp) <= max_skew_sec + 1e-6:
                break
            if main_stamp > wrist_stamp:
                main_index -= 1
            else:
                wrist_index -= 1
        if main_index < 0 or wrist_index < 0:
            raise RolloutConversionError(
                f"logical camera row {row} at {target:.9f} has no causal synchronized pair"
            )
        selected_main.append(main_index)
        selected_wrist.append(wrist_index)
    return np.asarray(selected_main, dtype=np.int32), np.asarray(selected_wrist, dtype=np.int32)


def _validate_run_header(
    source: EpisodeSource,
    run: Mapping[str, Any],
    outcome: Mapping[str, Any],
    manifest: Mapping[str, Any],
    options: BuildOptions,
) -> tuple[str, str, str]:
    prefix = source.episode_id
    if run.get("episode_id") != prefix or outcome.get("episode_id") != prefix:
        raise RolloutConversionError(f"{prefix}: episode IDs disagree")
    if outcome.get("task_outcome") != source.outcome:
        raise RolloutConversionError(
            f"{prefix}: directory outcome={source.outcome}, annotation={outcome.get('task_outcome')}"
        )
    if outcome.get("discarded") is not False:
        raise RolloutConversionError(f"{prefix}: discarded rollout is not training data")
    if outcome.get("runner_error") is not None or outcome.get("runner_exit_code") != 0:
        raise RolloutConversionError(f"{prefix}: rollout has a runner/system error")
    if outcome.get("termination_class") != "policy_rollout_completed":
        raise RolloutConversionError(f"{prefix}: rollout did not complete as a policy rollout")
    if run.get("completion_reason") != "max_actions":
        raise RolloutConversionError(f"{prefix}: unexpected completion reason")
    if int(run.get("actions_completed", -1)) != options.expected_actions:
        raise RolloutConversionError(
            f"{prefix}: expected {options.expected_actions} actions, got {run.get('actions_completed')}"
        )
    arguments = run.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RolloutConversionError(f"{prefix}: run arguments are absent")
    exact_arguments = {
        "execute": True,
        "max_actions": options.expected_actions,
        "clamp_mode": "reject",
    }
    for key, expected in exact_arguments.items():
        if arguments.get(key) != expected:
            raise RolloutConversionError(
                f"{prefix}: argument {key}={arguments.get(key)!r}, expected {expected!r}"
            )
    numeric_arguments = {
        "pos_scale": 0.012,
        "rot_scale": 0.036,
    }
    for key, expected in numeric_arguments.items():
        try:
            actual = float(arguments[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutConversionError(f"{prefix}: invalid argument {key}") from exc
        if not np.isclose(actual, expected, atol=1e-12, rtol=0.0):
            raise RolloutConversionError(
                f"{prefix}: argument {key}={actual}, expected {expected}"
            )
    checkpoint_sha = str(arguments.get("expected_checkpoint_sha256", ""))
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RolloutConversionError(
            f"{prefix}: checkpoint SHA256 is {checkpoint_sha!r}, expected epoch-200 {EXPECTED_CHECKPOINT_SHA256}"
        )
    identity_checkpoint = (
        run.get("server_health", {})
        .get("identity", {})
        .get("checkpoint", {})
        .get("sha256")
    )
    if identity_checkpoint != checkpoint_sha:
        raise RolloutConversionError(f"{prefix}: server and requested checkpoint hashes differ")
    contract_sha = str(manifest.get("contract_sha256", ""))
    if contract_sha != EXPECTED_CONTRACT_SHA256:
        raise RolloutConversionError(
            f"{prefix}: unexpected collection contract SHA256 {contract_sha!r}"
        )
    if manifest.get("episode_id") != prefix:
        raise RolloutConversionError(f"{prefix}: collection manifest ID differs")
    runner_sha = str(manifest.get("runner_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", runner_sha):
        raise RolloutConversionError(f"{prefix}: invalid runner SHA256")
    return checkpoint_sha, contract_sha, runner_sha


def _validate_steps(
    source: EpisodeSource,
    steps: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    options: BuildOptions,
) -> tuple[np.ndarray, ...]:
    prefix = source.episode_id
    if len(steps) != options.expected_actions:
        raise RolloutConversionError(
            f"{prefix}: steps.jsonl has {len(steps)} rows, expected {options.expected_actions}"
        )
    expected_chunks = options.expected_actions // options.action_horizon
    if options.expected_actions % options.action_horizon or len(chunks) != expected_chunks:
        raise RolloutConversionError(
            f"{prefix}: expected {expected_chunks} complete chunks, got {len(chunks)}"
        )
    actions: list[np.ndarray] = []
    physical: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    gripper_predictions: list[float] = []
    gripper_commands: list[float] = []
    gripper_decisions: list[int] = []
    gripper_intervened: list[bool] = []
    gripper_requested: list[bool] = []
    gripper_success: list[int] = []
    poses: list[np.ndarray] = []
    grippers: list[float] = []
    state_times: list[float] = []
    unix_times: list[float] = []
    chunk_values: list[int] = []
    substeps: list[int] = []
    for index, step in enumerate(steps):
        if int(step.get("action_index", -1)) != index:
            raise RolloutConversionError(f"{prefix}: non-contiguous action index at {index}")
        expected_chunk, expected_substep = divmod(index, options.action_horizon)
        if int(step.get("chunk", -1)) != expected_chunk or int(step.get("substep", -1)) != expected_substep:
            raise RolloutConversionError(f"{prefix}: invalid chunk/substep at action {index}")
        if not np.isclose(float(step.get("scheduled_period_sec", -1.0)), 0.05, atol=1e-9):
            raise RolloutConversionError(f"{prefix}: action {index} is not scheduled at 20 Hz")
        action = _as_array(step.get("raw_action"), shape=(7,), name=f"{prefix}.raw_action[{index}]")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise RolloutConversionError(f"{prefix}: normalized action {index} is outside [-1,1]")
        physical_action = _as_array(
            step.get("physical_action"), shape=(7,), name=f"{prefix}.physical_action[{index}]"
        )
        if not np.allclose(physical_action, action * ACTION_SCALE, atol=2e-8, rtol=1e-6):
            raise RolloutConversionError(f"{prefix}: physical action scale differs at {index}")
        command = _as_array(step.get("command"), shape=(7,), name=f"{prefix}.command[{index}]")
        if not np.allclose(command[:6], action[:6], atol=1e-9, rtol=0.0):
            raise RolloutConversionError(f"{prefix}: executed motion command differs at {index}")
        if command[6] != 0.0:
            raise RolloutConversionError(
                f"{prefix}: motion-controller gripper placeholder is nonzero at {index}"
            )
        clamp = step.get("clamp")
        if not isinstance(clamp, Mapping) or clamp.get("intervened") is not False or clamp.get("rejected") is not False:
            raise RolloutConversionError(f"{prefix}: action {index} was clamped or rejected")
        dispatch = step.get("dispatch")
        if not isinstance(dispatch, Mapping) or dispatch.get("status") != "PASS":
            raise RolloutConversionError(f"{prefix}: dispatch failed at action {index}")
        if dispatch.get("may_have_been_sent") is not True:
            raise RolloutConversionError(f"{prefix}: dispatch certainty is invalid at action {index}")
        gate = dispatch.get("final_dispatch_gate")
        if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
            raise RolloutConversionError(f"{prefix}: final safety gate failed at action {index}")
        environment_sync = dispatch.get("precommand_environment_sync")
        if not isinstance(environment_sync, Mapping) or environment_sync.get(
            "status"
        ) != "PASS":
            raise RolloutConversionError(
                f"{prefix}: precommand environment sync failed at action {index}"
            )
        pose_parity = step.get("pose_parity")
        if not isinstance(pose_parity, Mapping) or pose_parity.get("status") not in {
            "PASS",
            "NOT_REPEATED",
        }:
            raise RolloutConversionError(
                f"{prefix}: pose parity provenance differs at action {index}"
            )
        gripper = step.get("gripper")
        if not isinstance(gripper, Mapping):
            raise RolloutConversionError(f"{prefix}: gripper provenance absent at action {index}")
        logical = float(gripper.get("logical_state", np.nan))
        if logical not in (-1.0, 1.0):
            raise RolloutConversionError(f"{prefix}: invalid logical gripper state at {index}")
        try:
            prediction = float(gripper["prediction"])
            gripper_command = float(gripper["command"])
            decision = str(gripper["decision"])
            intervened_value = gripper["intervened"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutConversionError(
                f"{prefix}: invalid gripper decision provenance at action {index}"
            ) from exc
        if not isinstance(intervened_value, bool):
            raise RolloutConversionError(
                f"{prefix}: gripper.intervened is not boolean at action {index}"
            )
        intervened = intervened_value
        if not np.isclose(prediction, float(action[6]), atol=1e-9, rtol=0.0):
            raise RolloutConversionError(
                f"{prefix}: gripper prediction differs from raw_action[6] at {index}"
            )
        decision_codes = {"close": -1, "hold": 0, "open": 1}
        if decision not in decision_codes or gripper_command != float(
            decision_codes[decision]
        ):
            raise RolloutConversionError(
                f"{prefix}: inconsistent gripper command/decision at action {index}"
            )
        execution = step.get("gripper_execution")
        if not isinstance(execution, Mapping) or not isinstance(
            execution.get("requested"), bool
        ):
            raise RolloutConversionError(
                f"{prefix}: gripper execution provenance absent at action {index}"
            )
        requested = bool(execution["requested"])
        success_value = execution.get("success")
        if requested and success_value is not True:
            raise RolloutConversionError(f"{prefix}: gripper execution failed at action {index}")
        if not requested and success_value not in (None, False):
            raise RolloutConversionError(
                f"{prefix}: unrequested gripper command reports success at action {index}"
            )
        pose = _as_array(step.get("pose_before"), shape=(7,), name=f"{prefix}.pose_before[{index}]")
        synchronized_pose = _as_array(
            environment_sync.get("pose"),
            shape=(7,),
            name=f"{prefix}.precommand_environment_sync.pose[{index}]",
        )
        if not np.allclose(synchronized_pose, pose, atol=1e-9, rtol=0.0):
            raise RolloutConversionError(
                f"{prefix}: pose_before differs from fresh HTTP pose at action {index}"
            )
        norm = float(np.linalg.norm(pose[3:]))
        if not np.isclose(norm, 1.0, atol=5e-3):
            raise RolloutConversionError(f"{prefix}: quaternion norm={norm:.6f} at action {index}")
        join = step.get("rosbag_join")
        try:
            state_time = float(join["precommand_robot_state_stamp"])
            unix_time = float(step["unix_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutConversionError(f"{prefix}: invalid action timestamp at {index}") from exc
        if not np.isfinite(state_time) or not np.isfinite(unix_time):
            raise RolloutConversionError(f"{prefix}: non-finite action timestamp at {index}")
        actions.append(action.astype(np.float32))
        physical.append(physical_action.astype(np.float32))
        commands.append(command.astype(np.float32))
        gripper_predictions.append(prediction)
        gripper_commands.append(gripper_command)
        gripper_decisions.append(decision_codes[decision])
        gripper_intervened.append(intervened)
        gripper_requested.append(requested)
        gripper_success.append(
            1 if success_value is True else (0 if success_value is False else -1)
        )
        poses.append(pose.astype(np.float32))
        grippers.append(logical)
        state_times.append(state_time)
        unix_times.append(unix_time)
        chunk_values.append(expected_chunk)
        substeps.append(expected_substep)

    state_deltas = np.diff(state_times)
    if np.any(state_deltas < 0.0):
        raise RolloutConversionError(
            f"{prefix}: precommand robot state timestamps move backward"
        )
    repeated_edges = tuple(int(index) for index in np.flatnonzero(state_deltas == 0.0))
    expected_repeated_edges = ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES.get(prefix, ())
    if repeated_edges != expected_repeated_edges:
        raise RolloutConversionError(
            f"{prefix}: repeated precommand robot state timestamp edges "
            f"{repeated_edges} differ from audited edges {expected_repeated_edges}"
        )
    for edge in repeated_edges:
        try:
            left_sequence = int(
                steps[edge]["rosbag_join"]["precommand_robot_state_sequence"]
            )
            right_sequence = int(
                steps[edge + 1]["rosbag_join"]["precommand_robot_state_sequence"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutConversionError(
                f"{prefix}: repeated state timestamp edge {edge} lacks sequence provenance"
            ) from exc
        if left_sequence != right_sequence:
            raise RolloutConversionError(
                f"{prefix}: repeated timestamp edge {edge} has different state sequences"
            )
    for index in range(len(steps) - 1):
        if gripper_requested[index] and gripper_success[index] == 1:
            expected_logical = float(gripper_decisions[index])
            if expected_logical not in (-1.0, 1.0) or grippers[index + 1] != expected_logical:
                raise RolloutConversionError(
                    f"{prefix}: logical gripper state does not reflect successful "
                    f"command on row {index}"
                )

    stacked_actions = np.stack(actions)
    for chunk_index, chunk in enumerate(chunks):
        start = chunk_index * options.action_horizon
        end = start + options.action_horizon
        if int(chunk.get("chunk", -1)) != chunk_index or int(chunk.get("start_action_index", -1)) != start or int(chunk.get("end_action_index", -1)) != end:
            raise RolloutConversionError(f"{prefix}: invalid chunk range at {chunk_index}")
        chunk_actions = _as_array(
            chunk.get("raw_actions"),
            shape=(options.action_horizon, 7),
            name=f"{prefix}.chunks[{chunk_index}].raw_actions",
        )
        if not np.array_equal(chunk_actions.astype(np.float32), stacked_actions[start:end]):
            raise RolloutConversionError(f"{prefix}: run/step raw actions differ in chunk {chunk_index}")
    return (
        stacked_actions,
        np.stack(physical),
        np.stack(commands),
        np.asarray(gripper_predictions, dtype=np.float32),
        np.asarray(gripper_commands, dtype=np.float32),
        np.asarray(gripper_decisions, dtype=np.int8),
        np.asarray(gripper_intervened, dtype=np.uint8),
        np.asarray(gripper_requested, dtype=np.uint8),
        np.asarray(gripper_success, dtype=np.int8),
        np.stack(poses),
        np.asarray(grippers, dtype=np.float32),
        np.asarray(state_times, dtype=np.float64),
        np.asarray(unix_times, dtype=np.float64),
        np.asarray(chunk_values, dtype=np.int16),
        np.asarray(substeps, dtype=np.int8),
    )


def _validate_golden_chunks(
    source: EpisodeSource,
    run: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    main_records: list[ImageRecord],
    wrist_records: list[ImageRecord],
    connection: sqlite3.Connection,
    golden_npz_identity: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict[str, Any]]]:
    prefix = source.episode_id
    main_cache: dict[int, np.ndarray] = {}
    wrist_cache: dict[int, np.ndarray] = {}
    recovery_by_key: dict[tuple[str, int, float], dict[str, Any]] = {}
    recovered_record_by_stamp: dict[tuple[str, float], ImageRecord] = {}

    def image_for(
        records: list[ImageRecord],
        cache: dict[int, np.ndarray],
        timestamp: float,
        name: str,
        golden_frame: np.ndarray,
        *,
        chunk_index: int,
        history_index: int,
        npz_name: str,
    ) -> np.ndarray:
        matching = [
            record
            for record in records
            if abs(record.header_timestamp - timestamp) <= 5e-7
        ]
        if len(matching) > 1:
            raise RolloutConversionError(
                f"{prefix}: golden {name} timestamp {timestamp:.9f} is ambiguous in ROS bag"
            )
        if matching:
            record = matching[0]
        else:
            # A handful of logger inputs survived in the digest-verified NPZ
            # while the corresponding camera message was dropped by rosbag.
            # Recover only that exact timestamp and exact saved pixel array;
            # interpolation or nearest-frame substitution remains forbidden.
            recovered_key = (name, timestamp)
            record = recovered_record_by_stamp.get(recovered_key)
            if record is None:
                record = ImageRecord(
                    message_id=-(len(recovered_record_by_stamp) + 1),
                    bag_timestamp_ns=-1,
                    header_timestamp=timestamp,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    encoding="rgb8",
                    step=IMAGE_WIDTH * 3,
                )
                recovered_record_by_stamp[recovered_key] = record
                records.append(record)
                cache[record.message_id] = np.ascontiguousarray(golden_frame)
            elif not np.array_equal(cache[record.message_id], golden_frame):
                raise RolloutConversionError(
                    f"{prefix}: conflicting golden pixels for recovered {name} "
                    f"timestamp {timestamp:.9f}"
                )
        if record.message_id not in cache:
            cache[record.message_id] = _fetch_resized_image(connection, record)
        if record.message_id < 0:
            if not np.array_equal(cache[record.message_id], golden_frame):
                raise RolloutConversionError(
                    f"{prefix}: recovered {name} frame conflicts with {npz_name}"
                )
            detail_key = (name, chunk_index, timestamp)
            detail = recovery_by_key.get(detail_key)
            if detail is None:
                file_identity = golden_npz_identity["files"][npz_name]
                detail = {
                    "camera": name,
                    "chunk": chunk_index,
                    "header_timestamp": timestamp,
                    "history_indices": [],
                    "reason": "exact_header_timestamp_absent_from_rosbag",
                    "source": "digest_verified_golden_npz",
                    "npz_file": npz_name,
                    "npz_file_sha256": file_identity["sha256"],
                    "frame_sha256": _array_sha256(golden_frame),
                }
                recovery_by_key[detail_key] = detail
            if history_index not in detail["history_indices"]:
                detail["history_indices"].append(history_index)
        return cache[record.message_id]

    contract_schema = run.get("runtime_contract", {}).get("schema")
    if not isinstance(contract_schema, str) or not contract_schema:
        raise RolloutConversionError(f"{prefix}: runtime contract schema absent")
    for chunk_index, chunk in enumerate(chunks):
        npz_path = source.path / f"chunk_{chunk_index:04d}_input.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"{prefix}: missing golden input {npz_path.name}")
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                if set(archive.files) != set(OBS_KEYS):
                    raise RolloutConversionError(
                        f"{prefix}: {npz_path.name} keys differ: {sorted(archive.files)}"
                    )
                golden = {key: np.ascontiguousarray(archive[key]) for key in OBS_KEYS}
        except (OSError, ValueError) as exc:
            raise RolloutConversionError(f"{prefix}: invalid golden NPZ {npz_path}") from exc
        expected_shapes = {
            "main_image": (2, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "wrist_image": (2, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "robot0_eef_pos": (2, 3),
            "robot0_eef_quat": (2, 4),
            "robot0_gripper_state": (2, 1),
        }
        for key in OBS_KEYS:
            expected_dtype = np.dtype(np.uint8 if key.endswith("image") else np.float32)
            if golden[key].shape != expected_shapes[key] or golden[key].dtype != expected_dtype:
                raise RolloutConversionError(
                    f"{prefix}: {npz_path.name}:{key} has {golden[key].shape}/{golden[key].dtype}, "
                    f"expected {expected_shapes[key]}/{expected_dtype}"
                )
        digest = chunk.get("input_digest")
        if not isinstance(digest, Mapping) or not isinstance(digest.get("arrays"), Mapping):
            raise RolloutConversionError(f"{prefix}: chunk {chunk_index} lacks input digest")
        per_key = {key: _array_sha256(golden[key]) for key in OBS_KEYS}
        if per_key != dict(digest["arrays"]):
            raise RolloutConversionError(f"{prefix}: golden array digest mismatch in chunk {chunk_index}")
        aggregate = _canonical_sha256({"schema": contract_schema, "arrays": per_key})
        if aggregate != digest.get("aggregate"):
            raise RolloutConversionError(f"{prefix}: golden aggregate digest mismatch in chunk {chunk_index}")
        timestamps = chunk["timestamps"]
        reconstructed_main = np.stack(
            [
                image_for(
                    main_records,
                    main_cache,
                    float(stamp),
                    "main",
                    golden["main_image"][history_index],
                    chunk_index=chunk_index,
                    history_index=history_index,
                    npz_name=npz_path.name,
                )
                for history_index, stamp in enumerate(timestamps["main"])
            ]
        )
        reconstructed_wrist = np.stack(
            [
                image_for(
                    wrist_records,
                    wrist_cache,
                    float(stamp),
                    "wrist",
                    golden["wrist_image"][history_index],
                    chunk_index=chunk_index,
                    history_index=history_index,
                    npz_name=npz_path.name,
                )
                for history_index, stamp in enumerate(timestamps["wrist"])
            ]
        )
        if not np.array_equal(reconstructed_main, golden["main_image"]):
            raise RolloutConversionError(f"{prefix}: golden main pixels differ in chunk {chunk_index}")
        if not np.array_equal(reconstructed_wrist, golden["wrist_image"]):
            raise RolloutConversionError(f"{prefix}: golden wrist pixels differ in chunk {chunk_index}")
    return main_cache, wrist_cache, [
        recovery_by_key[key] for key in sorted(recovery_by_key)
    ]


def _bag_path(episode_path: Path) -> Path:
    bag_root = episode_path / "episode_bag"
    databases = sorted(bag_root.glob("*.db3"))
    if len(databases) != 1:
        raise RolloutConversionError(
            f"expected exactly one ROS bag database under {bag_root}, got {databases}"
        )
    return databases[0]


def _bag_identity(episode_path: Path, bag_path: Path) -> dict[str, Any]:
    stat = bag_path.stat()
    return {
        "relative_path": str(bag_path.relative_to(episode_path)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _golden_npz_identity(
    source: EpisodeSource,
    *,
    expected_chunks: int,
) -> dict[str, Any]:
    paths = sorted(source.path.glob("chunk_*_input.npz"))
    expected_names = [f"chunk_{index:04d}_input.npz" for index in range(expected_chunks)]
    if [path.name for path in paths] != expected_names:
        raise RolloutConversionError(
            f"{source.episode_id}: golden NPZ file set is incomplete or has extras"
        )
    files = {
        path.name: {"size": int(path.stat().st_size), "sha256": _file_sha256(path)}
        for path in paths
    }
    return {
        "file_count": len(files),
        "total_size": sum(item["size"] for item in files.values()),
        "aggregate_sha256": _canonical_sha256(files),
        # Per-file hashes make a recovery record independently auditable.
        "files": files,
    }


def _startup_prefix_trim(
    image_ages: np.ndarray,
    golden_input_recoveries: Sequence[Mapping[str, Any]],
    *,
    startup_action_limit: int,
) -> tuple[int, str | None]:
    """Return an auditable startup trim, rejecting any internal stale gap."""

    stale_rows = np.flatnonzero(np.asarray(image_ages) > 0.5 + 1e-6)
    if not stale_rows.size:
        return 0, None
    last_stale = int(stale_rows[-1])
    startup_recovery = any(
        int(detail["chunk"]) <= 1
        and detail["reason"] == "exact_header_timestamp_absent_from_rosbag"
        for detail in golden_input_recoveries
    )
    if (
        last_stale >= startup_action_limit
        or not startup_recovery
        or last_stale + 1 >= len(image_ages)
        or np.any(image_ages[last_stale + 1 :] > 0.5 + 1e-6)
    ):
        raise RolloutConversionError(
            "internal per-row causal image age outside [0,0.5]s: "
            f"[{float(np.min(image_ages)):.6f},{float(np.max(image_ages)):.6f}]"
        )
    return (
        last_stale + 1,
        "startup_rosbag_camera_gap_before_first_subsequent_valid_row",
    )


def load_episode(source: EpisodeSource, options: BuildOptions) -> EpisodePayload:
    run_path = source.path / "run.json"
    outcome_path = source.path / "outcome.json"
    manifest_path = source.path / "collection_manifest.json"
    steps_path = source.path / "steps.jsonl"
    for path in (run_path, outcome_path, manifest_path, steps_path):
        if not path.is_file():
            raise FileNotFoundError(f"{source.episode_id}: missing {path.name}")
    run = _load_json(run_path)
    outcome = _load_json(outcome_path)
    manifest = _load_json(manifest_path)
    checkpoint_sha, contract_sha, runner_sha = _validate_run_header(
        source, run, outcome, manifest, options
    )
    chunks = run.get("chunks")
    if not isinstance(chunks, list):
        raise RolloutConversionError(f"{source.episode_id}: run chunks are absent")
    steps = _read_steps(steps_path)
    arrays = _validate_steps(source, steps, chunks, options)
    (
        actions,
        physical,
        commands,
        gripper_predictions,
        gripper_commands,
        gripper_decisions,
        gripper_intervened,
        gripper_requested,
        gripper_success,
        poses,
        grippers,
        state_times,
        unix_times,
        chunk_values,
        substeps,
    ) = arrays

    bag_path = _bag_path(source.path)
    bag_identity = _bag_identity(source.path, bag_path)
    golden_npz_identity = _golden_npz_identity(
        source, expected_chunks=len(chunks)
    )
    with _open_bag(bag_path) as connection:
        main_raw = _image_records(connection, MAIN_TOPIC)
        wrist_raw = _image_records(connection, WRIST_TOPIC)
        main_cache, wrist_cache, golden_input_recoveries = _validate_golden_chunks(
            source,
            run,
            chunks,
            main_raw,
            wrist_raw,
            connection,
            golden_npz_identity,
        )
        main_raw.sort(key=lambda record: (record.header_timestamp, record.message_id))
        wrist_raw.sort(key=lambda record: (record.header_timestamp, record.message_id))
        # Define a wall-clock logical 5 Hz grid on the actual pre-command state
        # clock. Normal 20 Hz execution produces four held rows per tick. A
        # gripper or inference pause advances the tick immediately on the next
        # action instead of attaching a seconds-old image to a fresh pose.
        logical_origin = float(state_times[0])
        logical_tick_indices = np.floor(
            (state_times - logical_origin) / 0.2 + 1e-9
        ).astype(np.int64)
        if np.any(logical_tick_indices < 0) or np.any(
            np.diff(logical_tick_indices) < 0
        ):
            raise RolloutConversionError(
                f"{source.episode_id}: logical camera tick indices are invalid"
            )
        unique_ticks, inverse_ticks = np.unique(
            logical_tick_indices, return_inverse=True
        )
        unique_tick_times = logical_origin + unique_ticks.astype(np.float64) * 0.2
        selected_main_ticks, selected_wrist_ticks = _select_synchronized_camera_rows(
            main_raw,
            wrist_raw,
            unique_tick_times,
        )
        selected_main = selected_main_ticks[inverse_ticks]
        selected_wrist = selected_wrist_ticks[inverse_ticks]
        logical_tick_times = logical_origin + logical_tick_indices.astype(np.float64) * 0.2
        availability = np.maximum(
            np.asarray(
                [main_raw[int(index)].header_timestamp for index in selected_main]
            ),
            np.asarray(
                [wrist_raw[int(index)].header_timestamp for index in selected_wrist]
            ),
        )
        image_ages = state_times - availability
        if np.any(availability > logical_tick_times + 1e-6):
            raise RolloutConversionError(
                f"{source.episode_id}: RGB capture is later than its logical 5 Hz tick"
            )
        if np.any(image_ages < -1e-6):
            raise RolloutConversionError(
                f"{source.episode_id}: per-row causal image is in the future: "
                f"[{float(np.min(image_ages)):.6f},{float(np.max(image_ages)):.6f}]"
            )
        dropped_prefix_actions = 0
        prefix_trim_reason: str | None = None
        # A startup recording gap can leave an initially useful but soon stale
        # held pair before the first later camera pair exists. Trim the entire
        # disconnected startup prefix. Any stale row after the first deployed
        # task-audited startup window, or without exact NPZ evidence of a bag drop, is
        # an internal data failure and remains fatal.
        try:
            dropped_prefix_actions, prefix_trim_reason = _startup_prefix_trim(
                image_ages,
                golden_input_recoveries,
                startup_action_limit=max(
                    options.action_horizon, STARTUP_PREFIX_TRIM_ACTION_LIMIT
                ),
            )
        except RolloutConversionError as exc:
            raise RolloutConversionError(f"{source.episode_id}: {exc}") from exc

        keep = slice(dropped_prefix_actions, None)
        actions = actions[keep]
        physical = physical[keep]
        commands = commands[keep]
        gripper_predictions = gripper_predictions[keep]
        gripper_commands = gripper_commands[keep]
        gripper_decisions = gripper_decisions[keep]
        gripper_intervened = gripper_intervened[keep]
        gripper_requested = gripper_requested[keep]
        gripper_success = gripper_success[keep]
        poses = poses[keep]
        grippers = grippers[keep]
        state_times = state_times[keep]
        unix_times = unix_times[keep]
        chunk_values = chunk_values[keep]
        substeps = substeps[keep]
        selected_main = selected_main[keep]
        selected_wrist = selected_wrist[keep]
        logical_tick_indices = logical_tick_indices[keep]
        logical_tick_times = logical_tick_times[keep]
        image_ages = image_ages[keep]
        if np.any(image_ages > 0.5 + 1e-6):
            raise AssertionError("prefix trim retained a stale RGB row")
        needed_main = {main_raw[int(index)].message_id for index in selected_main}
        needed_wrist = {wrist_raw[int(index)].message_id for index in selected_wrist}
        for record in main_raw:
            if record.message_id in needed_main and record.message_id not in main_cache:
                main_cache[record.message_id] = _fetch_resized_image(connection, record)
        for record in wrist_raw:
            if record.message_id in needed_wrist and record.message_id not in wrist_cache:
                wrist_cache[record.message_id] = _fetch_resized_image(connection, record)

    recoveries = run.get("capture_recoveries", [])
    if not isinstance(recoveries, list) or any(
        not isinstance(item, Mapping) or item.get("status") != "PASS"
        for item in recoveries
    ):
        raise RolloutConversionError(f"{source.episode_id}: unresolved capture recovery")
    return EpisodePayload(
        source=source,
        actions=actions,
        physical_actions=physical,
        commands=commands,
        gripper_predictions=gripper_predictions,
        gripper_commands=gripper_commands,
        gripper_decisions=gripper_decisions,
        gripper_intervened=gripper_intervened,
        gripper_requested=gripper_requested,
        gripper_success=gripper_success,
        poses=poses,
        gripper_observations=grippers,
        action_state_times=state_times,
        action_unix_times=unix_times,
        chunks=chunk_values,
        substeps=substeps,
        main_records=main_raw,
        wrist_records=wrist_raw,
        main_row_positions=selected_main.astype(np.int32),
        wrist_row_positions=selected_wrist.astype(np.int32),
        logical_tick_indices=logical_tick_indices,
        logical_tick_times=logical_tick_times,
        logical_tick_origin_time=logical_origin,
        dropped_prefix_actions=dropped_prefix_actions,
        prefix_trim_reason=prefix_trim_reason,
        main_images=main_cache,
        wrist_images=wrist_cache,
        checkpoint_sha256=checkpoint_sha,
        contract_sha256=contract_sha,
        runner_sha256=runner_sha,
        capture_recoveries=len(recoveries),
        golden_chunks=len(chunks),
        golden_input_recoveries=golden_input_recoveries,
        golden_npz_identity=golden_npz_identity,
        bag_identity=bag_identity,
        source_file_sha256={
            "run.json": _file_sha256(run_path),
            "outcome.json": _file_sha256(outcome_path),
            "collection_manifest.json": _file_sha256(manifest_path),
            "steps.jsonl": _file_sha256(steps_path),
        },
    )


def _stratified_valid_ids(
    episodes: Sequence[EpisodeSource],
    *,
    outcome: str,
    count: int,
    seed: int,
) -> set[str]:
    candidates = [episode for episode in episodes if episode.outcome == outcome]
    if count < 0 or count > len(candidates):
        raise RolloutConversionError(
            f"requested {count} {outcome} validation episodes from {len(candidates)}"
        )
    if count == 0:
        return set()
    by_day: dict[str, list[EpisodeSource]] = {}
    for episode in candidates:
        by_day.setdefault(episode.day, []).append(episode)
    total = len(candidates)
    quotas = {day: count * len(group) / total for day, group in by_day.items()}
    allocation = {day: int(np.floor(quota)) for day, quota in quotas.items()}
    remaining = count - sum(allocation.values())
    day_order = sorted(
        by_day,
        key=lambda day: (-(quotas[day] - allocation[day]), day),
    )
    for day in day_order[:remaining]:
        allocation[day] += 1

    selected: set[str] = set()
    for day, group in sorted(by_day.items()):
        ranked = sorted(
            group,
            key=lambda episode: hashlib.sha256(
                f"{seed}:{outcome}:{day}:{episode.episode_id}".encode("utf-8")
            ).hexdigest(),
        )
        selected.update(episode.episode_id for episode in ranked[: allocation[day]])
    if len(selected) != count:
        raise AssertionError("stratified split selected the wrong number of episodes")
    return selected


def split_episodes(
    episodes: Sequence[EpisodeSource], options: BuildOptions
) -> dict[str, list[str]]:
    success_valid = _stratified_valid_ids(
        episodes,
        outcome="success",
        count=options.success_valid_count,
        seed=options.split_seed,
    )
    failure_valid = _stratified_valid_ids(
        episodes,
        outcome="failure",
        count=options.failure_valid_count,
        seed=options.split_seed,
    )
    all_ids = [episode.episode_id for episode in episodes]
    success_ids = [episode.episode_id for episode in episodes if episode.outcome == "success"]
    failure_ids = [episode.episode_id for episode in episodes if episode.outcome == "failure"]
    masks = {
        "all": all_ids,
        "success": success_ids,
        "failure": failure_ids,
        "success_train": [episode_id for episode_id in success_ids if episode_id not in success_valid],
        "success_valid": [episode_id for episode_id in success_ids if episode_id in success_valid],
        "failure_train": [episode_id for episode_id in failure_ids if episode_id not in failure_valid],
        "failure_valid": [episode_id for episode_id in failure_ids if episode_id in failure_valid],
    }
    masks["train"] = sorted(masks["success_train"] + masks["failure_train"])
    masks["valid"] = sorted(masks["success_valid"] + masks["failure_valid"])
    return masks


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
    compression: str,
) -> None:
    kwargs = _compression_kwargs(compression)
    chunks = (min(8, payload.num_samples), IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    main_dataset = obs.create_dataset(
        "main_image",
        shape=(payload.num_samples, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8,
        chunks=chunks,
        **kwargs,
    )
    wrist_dataset = obs.create_dataset(
        "wrist_image",
        shape=(payload.num_samples, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8,
        chunks=chunks,
        **kwargs,
    )
    for start in range(0, payload.num_samples, 128):
        end = min(payload.num_samples, start + 128)
        main_dataset[start:end] = np.stack(
            [
                payload.main_images[
                    payload.main_records[int(position)].message_id
                ]
                for position in payload.main_row_positions[start:end]
            ]
        )
        wrist_dataset[start:end] = np.stack(
            [
                payload.wrist_images[
                    payload.wrist_records[int(position)].message_id
                ]
                for position in payload.wrist_row_positions[start:end]
            ]
        )


def _manifest(
    payloads: Sequence[EpisodePayload],
    masks: Mapping[str, Sequence[str]],
    options: BuildOptions,
) -> dict[str, Any]:
    split_by_id = {
        episode_id: ("valid" if episode_id in set(masks["valid"]) else "train")
        for episode_id in masks["all"]
    }
    return {
        "schema": SCHEMA,
        "conversion_version": CONVERSION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(options.source_root.expanduser().resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "collection_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "action": {
            "space": "normalized Diffusion Policy proposal raw_action",
            "dimension": 7,
            "control_hz": 20.0,
            "physical_scale": ACTION_SCALE.tolist(),
            "gripper": (
                "actions[:,6] is the continuous policy prediction; thresholded "
                "command/decision/request/success are separate provenance"
            ),
        },
        "observation": {
            "state": "steps.jsonl pose_before (xyzw) and logical gripper before action",
            "rgb": RGB_ALIGNMENT_DESCRIPTION,
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "layout": "HWC RGB uint8",
            "resize": "Pillow Lanczos",
            "startup_prefix_policy": (
                "trim through last stale row only when stale rows are confined "
                f"to the first {STARTUP_PREFIX_TRIM_ACTION_LIMIT} actions and "
                "exact golden NPZ proves a startup "
                "rosbag camera drop; internal stale rows are fatal"
            ),
        },
        "reward": {
            "success": "1 on final transition only",
            "failure": "all zero",
            "done": "1 on final transition for both outcomes",
        },
        "validation": {
            "golden": "all chunk NPZ arrays, timestamps, hashes, and ROS pixels",
            "golden_npz_recovery": (
                "exact digest-verified NPZ frame only when its logged header "
                "timestamp is absent from rosbag; no interpolation"
            ),
            "golden_input_recovery_count": sum(
                len(payload.golden_input_recoveries) for payload in payloads
            ),
            "dropped_startup_prefix_actions": sum(
                payload.dropped_prefix_actions for payload in payloads
            ),
            "no_action_clamps": True,
            "checkpoint_and_scales": "strict",
            "allowed_repeated_state_timestamp_edges": {
                episode_id: list(edges)
                for episode_id, edges in ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES.items()
            },
        },
        "split": {
            "seed": options.split_seed,
            "method": "episode outcome/day-stratified deterministic SHA256 ranking",
            **{key: list(values) for key, values in masks.items()},
        },
        "episodes": [
            {
                "demo_key": f"demo_{index:04d}",
                "episode_id": payload.source.episode_id,
                "outcome": payload.source.outcome,
                "collection_day": payload.source.day,
                "split": split_by_id[payload.source.episode_id],
                "samples": payload.num_samples,
                "source_num_actions": payload.num_samples
                + payload.dropped_prefix_actions,
                "dropped_prefix_actions": payload.dropped_prefix_actions,
                "prefix_trim_reason": payload.prefix_trim_reason,
                "logical_camera_tick_origin_time": payload.logical_tick_origin_time,
                "golden_chunks": payload.golden_chunks,
                "capture_recoveries": payload.capture_recoveries,
                "golden_input_recovery_count": len(
                    payload.golden_input_recoveries
                ),
                "golden_input_recoveries": payload.golden_input_recoveries,
                "golden_npz_identity": payload.golden_npz_identity,
                "bag_identity": payload.bag_identity,
                "runner_sha256": payload.runner_sha256,
                "source_file_sha256": payload.source_file_sha256,
                "repeated_state_timestamp_edges": list(
                    ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES.get(
                        payload.source.episode_id, ()
                    )
                ),
            }
            for index, payload in enumerate(payloads)
        ],
    }


def write_dataset(
    path: Path,
    payloads: Sequence[EpisodePayload],
    masks: Mapping[str, Sequence[str]],
    options: BuildOptions,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(payloads, masks, options)
    id_to_demo = {
        payload.source.episode_id: f"demo_{index:04d}"
        for index, payload in enumerate(payloads)
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["schema"] = SCHEMA
            output.attrs["conversion_version"] = CONVERSION_VERSION
            output.attrs[CONVERSION_MANIFEST_ATTR] = json.dumps(manifest, sort_keys=True)
            data = output.create_group("data")
            data.attrs["env_args"] = json.dumps(ENV_ARGS, sort_keys=True)
            total = 0
            valid_ids = set(masks["valid"])
            for index, payload in enumerate(payloads):
                key = f"demo_{index:04d}"
                demo = data.create_group(key)
                success = payload.source.outcome == "success"
                demo.attrs["num_samples"] = payload.num_samples
                demo.attrs["source_num_actions"] = (
                    payload.num_samples + payload.dropped_prefix_actions
                )
                demo.attrs["dropped_prefix_actions"] = payload.dropped_prefix_actions
                demo.attrs["prefix_trim_reason"] = payload.prefix_trim_reason or ""
                demo.attrs["logical_camera_tick_origin_time"] = (
                    payload.logical_tick_origin_time
                )
                demo.attrs["source_episode_id"] = payload.source.episode_id
                demo.attrs["repeated_state_timestamp_edges"] = json.dumps(
                    list(
                        ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES.get(
                            payload.source.episode_id, ()
                        )
                    )
                )
                demo.attrs["source_directory"] = str(payload.source.path)
                demo.attrs["task_outcome"] = payload.source.outcome
                demo.attrs["outcome"] = int(success)
                demo.attrs["collection_day"] = payload.source.day
                demo.attrs["split"] = (
                    "valid" if payload.source.episode_id in valid_ids else "train"
                )
                demo.attrs["checkpoint_sha256"] = payload.checkpoint_sha256
                demo.attrs["contract_sha256"] = payload.contract_sha256
                demo.attrs["runner_sha256"] = payload.runner_sha256
                demo.attrs["capture_recoveries"] = payload.capture_recoveries
                demo.attrs["golden_chunks_validated"] = payload.golden_chunks
                demo.attrs["golden_input_recovery_count"] = len(
                    payload.golden_input_recoveries
                )
                demo.attrs["golden_input_recoveries"] = json.dumps(
                    payload.golden_input_recoveries, sort_keys=True
                )
                demo.attrs["golden_npz_identity_sha256"] = payload.golden_npz_identity[
                    "aggregate_sha256"
                ]
                demo.attrs["bag_identity"] = json.dumps(
                    payload.bag_identity, sort_keys=True
                )
                demo.create_dataset("actions", data=payload.actions.astype(np.float32))
                rewards = np.zeros(payload.num_samples, dtype=np.float32)
                if success:
                    rewards[-1] = 1.0
                dones = np.zeros(payload.num_samples, dtype=np.uint8)
                dones[-1] = 1
                demo.create_dataset("rewards", data=rewards)
                demo.create_dataset("dones", data=dones)
                obs = demo.create_group("obs")
                obs.create_dataset("robot0_eef_pos", data=payload.poses[:, :3])
                obs.create_dataset("robot0_eef_quat", data=payload.poses[:, 3:])
                obs.create_dataset(
                    "robot0_gripper_state",
                    data=payload.gripper_observations[:, None],
                )
                _write_images(obs, payload, compression=options.compression)

                provenance = demo.create_group("provenance")
                provenance.create_dataset(
                    "source_action_index",
                    data=np.arange(
                        payload.dropped_prefix_actions,
                        payload.dropped_prefix_actions + payload.num_samples,
                        dtype=np.int32,
                    ),
                )
                provenance.create_dataset("chunk", data=payload.chunks)
                provenance.create_dataset("substep", data=payload.substeps)
                provenance.create_dataset(
                    "physical_action", data=payload.physical_actions
                )
                provenance.create_dataset(
                    "motion_controller_command", data=payload.commands
                )
                provenance["motion_controller_command"].attrs["gripper_channel"] = (
                    "placeholder zero; real gripper execution is recorded separately"
                )
                provenance.create_dataset(
                    "gripper_prediction", data=payload.gripper_predictions
                )
                provenance.create_dataset(
                    "gripper_command", data=payload.gripper_commands
                )
                provenance.create_dataset(
                    "gripper_decision_code", data=payload.gripper_decisions
                )
                provenance["gripper_decision_code"].attrs["mapping"] = json.dumps(
                    {"close": -1, "hold": 0, "open": 1}, sort_keys=True
                )
                provenance.create_dataset(
                    "gripper_intervened", data=payload.gripper_intervened
                )
                provenance.create_dataset(
                    "gripper_execution_requested", data=payload.gripper_requested
                )
                provenance.create_dataset(
                    "gripper_execution_success", data=payload.gripper_success
                )
                provenance["gripper_execution_success"].attrs["mapping"] = json.dumps(
                    {"not_reported": -1, "failure": 0, "success": 1},
                    sort_keys=True,
                )
                provenance.create_dataset(
                    "precommand_robot_state_stamp", data=payload.action_state_times
                )
                provenance.create_dataset(
                    "logical_camera_tick_index", data=payload.logical_tick_indices
                )
                provenance.create_dataset(
                    "logical_camera_tick_time", data=payload.logical_tick_times
                )
                provenance.create_dataset(
                    "step_unix_time", data=payload.action_unix_times
                )
                provenance.create_dataset(
                    "selected_main_frame_position", data=payload.main_row_positions
                )
                provenance.create_dataset(
                    "selected_wrist_frame_position", data=payload.wrist_row_positions
                )
                main_records = [
                    payload.main_records[int(position)]
                    for position in payload.main_row_positions
                ]
                wrist_records = [
                    payload.wrist_records[int(position)]
                    for position in payload.wrist_row_positions
                ]
                main_stamps = np.asarray(
                    [record.header_timestamp for record in main_records]
                )
                wrist_stamps = np.asarray(
                    [record.header_timestamp for record in wrist_records]
                )
                provenance.create_dataset("selected_main_header_stamp", data=main_stamps)
                provenance.create_dataset("selected_wrist_header_stamp", data=wrist_stamps)
                provenance.create_dataset(
                    "selected_main_bag_timestamp_ns",
                    data=np.asarray([record.bag_timestamp_ns for record in main_records]),
                )
                provenance.create_dataset(
                    "selected_wrist_bag_timestamp_ns",
                    data=np.asarray([record.bag_timestamp_ns for record in wrist_records]),
                )
                provenance.create_dataset(
                    "selected_main_golden_recovery",
                    data=np.asarray(
                        [record.message_id < 0 for record in main_records],
                        dtype=np.uint8,
                    ),
                )
                provenance.create_dataset(
                    "selected_wrist_golden_recovery",
                    data=np.asarray(
                        [record.message_id < 0 for record in wrist_records],
                        dtype=np.uint8,
                    ),
                )
                provenance.create_dataset(
                    "image_age_sec",
                    data=payload.action_state_times - np.maximum(main_stamps, wrist_stamps),
                )
                total += payload.num_samples
            data.attrs["total"] = total
            mask_group = output.create_group("mask")
            for name, episode_ids in masks.items():
                demo_keys = [id_to_demo[episode_id] for episode_id in episode_ids]
                mask_group.create_dataset(name, data=np.asarray(demo_keys, dtype="S"))
            output.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_dataset(
    path: Path,
    *,
    expected_sources: Sequence[EpisodeSource] | None = None,
    expected_masks: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rollout HDF5 does not exist: {path}")
    with h5py.File(path, "r") as dataset:
        if dataset.attrs.get("schema") != SCHEMA:
            raise RolloutConversionError(f"{path}: unexpected schema")
        if dataset.attrs.get("conversion_version") != CONVERSION_VERSION:
            raise RolloutConversionError(
                f"{path}: conversion version is not the wall-clock RGB v2 contract"
            )
        raw_manifest = dataset.attrs.get(CONVERSION_MANIFEST_ATTR)
        if isinstance(raw_manifest, bytes):
            raw_manifest = raw_manifest.decode("utf-8")
        try:
            manifest = json.loads(raw_manifest)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RolloutConversionError(f"{path}: invalid conversion manifest") from exc
        if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            raise RolloutConversionError(f"{path}: checkpoint identity differs")
        if manifest.get("conversion_version") != CONVERSION_VERSION or (
            manifest.get("observation", {}).get("rgb")
            != RGB_ALIGNMENT_DESCRIPTION
        ):
            raise RolloutConversionError(
                f"{path}: wall-clock RGB alignment contract differs"
            )
        manifest_episodes = manifest.get("episodes")
        if not isinstance(manifest_episodes, list):
            raise RolloutConversionError(f"{path}: manifest episode provenance is absent")
        manifest_by_id = {
            item.get("episode_id"): item
            for item in manifest_episodes
            if isinstance(item, Mapping)
        }
        if len(manifest_by_id) != len(manifest_episodes):
            raise RolloutConversionError(f"{path}: manifest episode IDs are invalid")
        if expected_sources is not None:
            expected_source_root = str(
                expected_sources[0].path.parents[1]
                if expected_sources
                else Path(manifest.get("source_root", ""))
            )
            if str(Path(manifest.get("source_root", "")).resolve()) != str(
                Path(expected_source_root).resolve()
            ):
                raise RolloutConversionError(f"{path}: raw rollout source root differs")
            expected_ids = {source.episode_id for source in expected_sources}
            if set(manifest_by_id) != expected_ids:
                raise RolloutConversionError(f"{path}: manifest source episode set differs")
            for source in expected_sources:
                item = manifest_by_id[source.episode_id]
                if item.get("outcome") != source.outcome or item.get("collection_day") != source.day:
                    raise RolloutConversionError(
                        f"{path}: outcome/day provenance differs for {source.episode_id}"
                    )
                recorded_hashes = item.get("source_file_sha256")
                if not isinstance(recorded_hashes, Mapping):
                    raise RolloutConversionError(
                        f"{path}: source hashes absent for {source.episode_id}"
                    )
                for filename in (
                    "run.json",
                    "outcome.json",
                    "collection_manifest.json",
                    "steps.jsonl",
                ):
                    source_path = source.path / filename
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"{source.episode_id}: source file disappeared: {filename}"
                        )
                    actual_hash = _file_sha256(source_path)
                    if recorded_hashes.get(filename) != actual_hash:
                        raise RolloutConversionError(
                            f"{path}: raw source hash changed for {source.episode_id}/{filename}"
                        )
                recorded_npz = item.get("golden_npz_identity")
                if not isinstance(recorded_npz, Mapping):
                    raise RolloutConversionError(
                        f"{path}: golden NPZ identity absent for {source.episode_id}"
                    )
                actual_npz = _golden_npz_identity(
                    source, expected_chunks=int(item.get("golden_chunks", -1))
                )
                if actual_npz != dict(recorded_npz):
                    raise RolloutConversionError(
                        f"{path}: golden NPZ identity changed for {source.episode_id}"
                    )
                recorded_bag = item.get("bag_identity")
                if not isinstance(recorded_bag, Mapping):
                    raise RolloutConversionError(
                        f"{path}: bag identity absent for {source.episode_id}"
                    )
                actual_bag_path = _bag_path(source.path)
                actual_bag = _bag_identity(source.path, actual_bag_path)
                if actual_bag != dict(recorded_bag):
                    raise RolloutConversionError(
                        f"{path}: ROS bag stat identity changed for {source.episode_id}"
                    )
                try:
                    dropped_prefix = int(item["dropped_prefix_actions"])
                    source_num_actions = int(item["source_num_actions"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RolloutConversionError(
                        f"{path}: prefix trim provenance absent for {source.episode_id}"
                    ) from exc
                raw_actions_completed = int(
                    _load_json(source.path / "run.json").get("actions_completed", -1)
                )
                if source_num_actions != raw_actions_completed or not (
                    0 <= dropped_prefix < source_num_actions
                ):
                    raise RolloutConversionError(
                        f"{path}: prefix trim size differs for {source.episode_id}"
                    )
                expected_reason = (
                    "startup_rosbag_camera_gap_before_first_subsequent_valid_row"
                    if dropped_prefix
                    else None
                )
                if item.get("prefix_trim_reason") != expected_reason:
                    raise RolloutConversionError(
                        f"{path}: prefix trim reason differs for {source.episode_id}"
                    )
        data = dataset.get("data")
        mask = dataset.get("mask")
        if not isinstance(data, h5py.Group) or not isinstance(mask, h5py.Group):
            raise RolloutConversionError(f"{path}: data/mask groups are absent")
        required_masks = {
            "all",
            "success",
            "failure",
            "success_train",
            "success_valid",
            "failure_train",
            "failure_valid",
            "train",
            "valid",
        }
        if set(mask) != required_masks:
            raise RolloutConversionError(f"{path}: masks differ: {sorted(mask)}")
        masks = {
            name: {
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in mask[name][:]
            }
            for name in required_masks
        }
        demos = set(data.keys())
        if masks["all"] != demos:
            raise RolloutConversionError(f"{path}: all mask differs from data demos")
        if masks["success"] & masks["failure"] or masks["success"] | masks["failure"] != demos:
            raise RolloutConversionError(f"{path}: outcome masks are not a partition")
        if masks["train"] & masks["valid"] or masks["train"] | masks["valid"] != demos:
            raise RolloutConversionError(f"{path}: train/valid masks are not a partition")
        episode_ids: set[str] = set()
        demo_to_episode: dict[str, str] = {}
        samples = 0
        successes = 0
        failures = 0
        for key in sorted(demos):
            demo = data[key]
            count = int(demo.attrs.get("num_samples", -1))
            if count <= 0:
                raise RolloutConversionError(f"{path}:{key}: invalid num_samples")
            outcome = str(demo.attrs.get("task_outcome", ""))
            if outcome not in {"success", "failure"}:
                raise RolloutConversionError(f"{path}:{key}: invalid outcome")
            episode_id = str(demo.attrs.get("source_episode_id", ""))
            if not EPISODE_RE.fullmatch(episode_id) or episode_id in episode_ids:
                raise RolloutConversionError(f"{path}:{key}: invalid/duplicate source ID")
            episode_ids.add(episode_id)
            demo_to_episode[key] = episode_id
            manifest_item = manifest_by_id.get(episode_id)
            if not isinstance(manifest_item, Mapping):
                raise RolloutConversionError(f"{path}:{key}: manifest provenance absent")
            expected_repeated_edges = list(
                ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES.get(episode_id, ())
            )
            try:
                attr_repeated_edges = json.loads(
                    str(demo.attrs.get("repeated_state_timestamp_edges", "[]"))
                )
            except json.JSONDecodeError as exc:
                raise RolloutConversionError(
                    f"{path}:{key}: repeated timestamp provenance is invalid"
                ) from exc
            if (
                manifest_item.get("repeated_state_timestamp_edges", [])
                != expected_repeated_edges
                or attr_repeated_edges != expected_repeated_edges
            ):
                raise RolloutConversionError(
                    f"{path}:{key}: repeated timestamp provenance differs"
                )
            expected_recoveries = manifest_item.get("golden_input_recoveries")
            if not isinstance(expected_recoveries, list):
                raise RolloutConversionError(f"{path}:{key}: golden recovery details absent")
            if int(demo.attrs.get("golden_input_recovery_count", -1)) != len(
                expected_recoveries
            ):
                raise RolloutConversionError(f"{path}:{key}: golden recovery count differs")
            dropped_prefix = int(manifest_item.get("dropped_prefix_actions", -1))
            if (
                int(demo.attrs.get("dropped_prefix_actions", -1))
                != dropped_prefix
                or int(demo.attrs.get("source_num_actions", -1))
                != count + dropped_prefix
                or str(demo.attrs.get("prefix_trim_reason", ""))
                != str(manifest_item.get("prefix_trim_reason") or "")
            ):
                raise RolloutConversionError(f"{path}:{key}: prefix trim attrs differ")
            expected_shapes = {
                "actions": (count, 7),
                "rewards": (count,),
                "dones": (count,),
                "obs/main_image": (count, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "obs/wrist_image": (count, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "obs/robot0_eef_pos": (count, 3),
                "obs/robot0_eef_quat": (count, 4),
                "obs/robot0_gripper_state": (count, 1),
            }
            for name, shape in expected_shapes.items():
                if name not in demo or demo[name].shape != shape:
                    raise RolloutConversionError(f"{path}:{key}:{name} shape differs")
            actions = demo["actions"][:]
            if actions.dtype != np.float32 or not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.0):
                raise RolloutConversionError(f"{path}:{key}: invalid normalized actions")
            rewards = demo["rewards"][:]
            dones = demo["dones"][:]
            expected_rewards = np.zeros(count, dtype=np.float32)
            if outcome == "success":
                expected_rewards[-1] = 1.0
                successes += 1
            else:
                failures += 1
            expected_dones = np.zeros(count, dtype=np.uint8)
            expected_dones[-1] = 1
            if not np.array_equal(rewards, expected_rewards) or not np.array_equal(dones, expected_dones):
                raise RolloutConversionError(f"{path}:{key}: reward/done semantics differ")
            gripper = demo["obs/robot0_gripper_state"][:]
            if not np.all(np.isin(gripper, (-1.0, 1.0))):
                raise RolloutConversionError(f"{path}:{key}: invalid gripper observation")
            quaternion = demo["obs/robot0_eef_quat"][:]
            if np.any(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0) > 5e-3):
                raise RolloutConversionError(f"{path}:{key}: invalid quaternion")
            provenance = demo.get("provenance")
            if not isinstance(provenance, h5py.Group):
                raise RolloutConversionError(f"{path}:{key}: provenance is absent")
            gripper_names = (
                "motion_controller_command",
                "gripper_prediction",
                "gripper_command",
                "gripper_decision_code",
                "gripper_intervened",
                "gripper_execution_requested",
                "gripper_execution_success",
            )
            if any(name not in provenance for name in gripper_names):
                raise RolloutConversionError(f"{path}:{key}: gripper provenance differs")
            motion_command = provenance["motion_controller_command"][:]
            prediction = provenance["gripper_prediction"][:]
            gripper_command = provenance["gripper_command"][:]
            decision = provenance["gripper_decision_code"][:]
            requested = provenance["gripper_execution_requested"][:]
            execution_success = provenance["gripper_execution_success"][:]
            if motion_command.shape != (count, 7) or np.any(
                motion_command[:, 6] != 0.0
            ):
                raise RolloutConversionError(
                    f"{path}:{key}: motion-controller gripper placeholder differs"
                )
            if not np.array_equal(prediction, actions[:, 6]) or not np.array_equal(
                gripper_command, decision.astype(gripper_command.dtype)
            ):
                raise RolloutConversionError(f"{path}:{key}: gripper proposal/decision differs")
            if np.any(~np.isin(decision, (-1, 0, 1))) or np.any(
                ~np.isin(execution_success, (-1, 0, 1))
            ):
                raise RolloutConversionError(f"{path}:{key}: gripper codes differ")
            logical_state = demo["obs/robot0_gripper_state"][:, 0]
            for row in range(count - 1):
                if requested[row] and execution_success[row] == 1:
                    if decision[row] not in (-1, 1) or logical_state[row + 1] != decision[row]:
                        raise RolloutConversionError(
                            f"{path}:{key}: gripper state transition differs at {row}"
                        )
            timing_names = (
                "source_action_index",
                "precommand_robot_state_stamp",
                "logical_camera_tick_index",
                "logical_camera_tick_time",
                "selected_main_header_stamp",
                "selected_wrist_header_stamp",
                "image_age_sec",
            )
            if any(
                name not in provenance or provenance[name].shape != (count,)
                for name in timing_names
            ):
                raise RolloutConversionError(f"{path}:{key}: RGB timing provenance differs")
            state_time = provenance["precommand_robot_state_stamp"][:]
            source_action_index = provenance["source_action_index"][:]
            tick_index = provenance["logical_camera_tick_index"][:]
            tick_time = provenance["logical_camera_tick_time"][:]
            main_stamp = provenance["selected_main_header_stamp"][:]
            wrist_stamp = provenance["selected_wrist_header_stamp"][:]
            image_age = provenance["image_age_sec"][:]
            if not all(
                np.all(np.isfinite(value))
                for value in (
                    state_time,
                    tick_index,
                    tick_time,
                    main_stamp,
                    wrist_stamp,
                    image_age,
                )
            ) or np.any(np.diff(state_time) < 0.0) or np.any(
                np.diff(tick_index) < 0
            ):
                raise RolloutConversionError(
                    f"{path}:{key}: RGB timing is non-finite or non-monotone"
                )
            actual_repeated_edges = [
                int(index)
                for index in np.flatnonzero(np.diff(state_time) == 0.0)
            ]
            # Prefix trimming changes row indices in the materialized dataset.
            materialized_expected_repeated_edges = [
                edge - dropped_prefix
                for edge in expected_repeated_edges
                if edge >= dropped_prefix
            ]
            if actual_repeated_edges != materialized_expected_repeated_edges:
                raise RolloutConversionError(
                    f"{path}:{key}: actual repeated timestamp edges differ"
                )
            if not np.array_equal(
                source_action_index,
                np.arange(dropped_prefix, dropped_prefix + count),
            ):
                raise RolloutConversionError(f"{path}:{key}: source action index differs")
            logical_origin = float(
                demo.attrs.get("logical_camera_tick_origin_time", np.nan)
            )
            if not np.isfinite(logical_origin) or not np.isclose(
                logical_origin,
                float(manifest_item.get("logical_camera_tick_origin_time", np.nan)),
                atol=1e-9,
                rtol=0.0,
            ):
                raise RolloutConversionError(f"{path}:{key}: logical grid origin differs")
            expected_tick = np.floor(
                (state_time - logical_origin) / 0.2 + 1e-9
            ).astype(np.int64)
            expected_tick_time = logical_origin + expected_tick * 0.2
            if not np.array_equal(tick_index, expected_tick) or not np.allclose(
                tick_time, expected_tick_time, atol=1e-9, rtol=0.0
            ):
                raise RolloutConversionError(f"{path}:{key}: logical 5 Hz grid differs")
            if np.any(np.maximum(main_stamp, wrist_stamp) > tick_time + 1e-6):
                raise RolloutConversionError(f"{path}:{key}: RGB is later than logical tick")
            expected_age = state_time - np.maximum(main_stamp, wrist_stamp)
            if not np.allclose(image_age, expected_age, atol=1e-9, rtol=0.0) or np.any(
                image_age < -1e-6
            ) or np.any(image_age > 0.5 + 1e-6):
                raise RolloutConversionError(f"{path}:{key}: per-row RGB image age differs")
            samples += count
        if expected_sources is not None and episode_ids != {
            source.episode_id for source in expected_sources
        }:
            raise RolloutConversionError(f"{path}: source episode identity differs")
        if expected_masks is not None:
            for name, expected_ids in expected_masks.items():
                actual_ids = {demo_to_episode[key] for key in masks[name]}
                if actual_ids != set(expected_ids):
                    raise RolloutConversionError(
                        f"{path}: deterministic {name} split differs from current source"
                    )
        if int(data.attrs.get("total", -1)) != samples:
            raise RolloutConversionError(f"{path}: data.total differs")
        if {
            "success": successes,
            "failure": failures,
        } == EXPECTED_FULL_OUTCOME_COUNTS:
            manifest_dropped = int(
                manifest.get("validation", {}).get(
                    "dropped_startup_prefix_actions", -1
                )
            )
            samples_differ = (
                EXPECTED_FULL_SAMPLES is not None
                and samples != EXPECTED_FULL_SAMPLES
            )
            dropped_differ = (
                EXPECTED_FULL_DROPPED_PREFIX_ACTIONS is not None
                and manifest_dropped != EXPECTED_FULL_DROPPED_PREFIX_ACTIONS
            )
            if samples_differ or dropped_differ:
                raise RolloutConversionError(
                    f"{path}: full rollout sample/prefix-trim totals differ"
                )
    return {
        "path": str(path),
        "validated": True,
        "episodes": successes + failures,
        "successes": successes,
        "failures": failures,
        "samples": samples,
        "success_train": len(masks["success_train"]),
        "success_valid": len(masks["success_valid"]),
        "failure_train": len(masks["failure_train"]),
        "failure_valid": len(masks["failure_valid"]),
    }


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    source_root = options.source_root.expanduser().resolve()
    output = options.output.expanduser().resolve()
    if options.expected_actions <= 0 or options.action_horizon <= 0:
        raise ValueError("expected actions and action horizon must be positive")
    if options.episode_limit is not None and options.episode_limit <= 0:
        raise ValueError("episode limit must be positive")
    _compression_kwargs(options.compression)
    episodes = discover_episodes(source_root)
    full_default = (
        options.episode_limit is None
        and options.expected_actions == FULL_DEFAULT_EXPECTED_ACTIONS
        and options.action_horizon == 8
        and options.success_valid_count == FULL_DEFAULT_SUCCESS_VALID_COUNT
        and options.failure_valid_count == FULL_DEFAULT_FAILURE_VALID_COUNT
    )
    if full_default:
        outcome_counts = {
            outcome: sum(episode.outcome == outcome for episode in episodes)
            for outcome in ("success", "failure")
        }
        if outcome_counts != EXPECTED_FULL_OUTCOME_COUNTS:
            raise RolloutConversionError(
                "full rollout source outcome counts differ: expected "
                f"{EXPECTED_FULL_OUTCOME_COUNTS}, got {outcome_counts}"
            )
    # Split the full source set before applying the smoke-test limit so the
    # identity of an episode never changes with --episode-limit.
    masks_full = split_episodes(episodes, options)
    if options.episode_limit is not None:
        episodes = episodes[: options.episode_limit]
    selected_ids = {episode.episode_id for episode in episodes}
    masks = {
        name: [episode_id for episode_id in values if episode_id in selected_ids]
        for name, values in masks_full.items()
    }
    if options.validate_only:
        return validate_dataset(
            output,
            expected_sources=episodes,
            expected_masks=masks,
        )
    if output.exists() and not options.overwrite:
        report = validate_dataset(
            output,
            expected_sources=episodes,
            expected_masks=masks,
        )
        report["reused_existing"] = True
        return report
    payloads = [load_episode(episode, options) for episode in episodes]
    if full_default:
        total_samples = sum(payload.num_samples for payload in payloads)
        total_dropped = sum(payload.dropped_prefix_actions for payload in payloads)
        samples_differ = (
            EXPECTED_FULL_SAMPLES is not None
            and total_samples != EXPECTED_FULL_SAMPLES
        )
        dropped_differ = (
            EXPECTED_FULL_DROPPED_PREFIX_ACTIONS is not None
            and total_dropped != EXPECTED_FULL_DROPPED_PREFIX_ACTIONS
        )
        if samples_differ or dropped_differ:
            raise RolloutConversionError(
                "full rollout startup-trim contract differs: expected "
                f"samples={EXPECTED_FULL_SAMPLES}, dropped_prefix_actions="
                f"{EXPECTED_FULL_DROPPED_PREFIX_ACTIONS}; got {total_samples} "
                f"and {total_dropped}"
            )
    write_dataset(output, payloads, masks, options)
    report = validate_dataset(
        output,
        expected_sources=episodes,
        expected_masks=masks,
    )
    report.update(
        {
            "reused_existing": False,
            "success_train": len(masks["success_train"]),
            "success_valid": len(masks["success_valid"]),
            "failure_train": len(masks["failure_train"]),
            "failure_valid": len(masks["failure_valid"]),
            "golden_chunks": sum(payload.golden_chunks for payload in payloads),
            "capture_recoveries": sum(payload.capture_recoveries for payload in payloads),
            "golden_input_recoveries": sum(
                len(payload.golden_input_recoveries) for payload in payloads
            ),
            "dropped_startup_prefix_actions": sum(
                payload.dropped_prefix_actions for payload in payloads
            ),
        }
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> BuildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    return BuildOptions(
        source_root=args.source_root,
        output=args.output,
        compression=args.compression,
        split_seed=args.split_seed,
        episode_limit=args.episode_limit,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    report = build_dataset(options)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
