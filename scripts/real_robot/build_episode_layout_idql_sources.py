#!/usr/bin/env python3
"""Build request-aligned real-robot HDF5 sources from episode-layout exports.

Human demonstrations and policy rollouts are supplied as separate immutable
packages with the same ``stack_cup_proposal_episode_layout.v1`` schema. Policy
rollouts retain one exact two-frame ``request_obs`` entry per executed H8
proposal. Human observations are causally aligned onto their exact 20 Hz target
grid and keep the pre-command pose and logical gripper state used by deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCHEMA = "stack_cup_proposal_episode_layout.v1"
CONVERSION_VERSION = "real_robot_episode_layout_idql_sources_v1"
MANIFEST_ATTR = "real_robot_conversion_manifest"
OBS_KEYS = (
    "main_image",
    "wrist_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
)
RGB_KEYS = OBS_KEYS[:2]
LOW_DIM_KEYS = OBS_KEYS[2:]
OBS_SHAPES = {
    "main_image": (96, 128, 3),
    "wrist_image": (96, 128, 3),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_state": (1,),
}
ACTION_HORIZON = 8
EXPECTED_CONTRACT_SHA256 = (
    "ffbda027f310ea9b2329dddcc23473fcfc90ca34693190f521534d0baa6cef75"
)
RUNTIME_SCHEMA = "real4d.robomimic.pick_cup_20hz_held_rgb.v1"
HUMAN_VALIDATION_EPISODES = {
    "pick_cup": frozenset({2, 5, 43, 48, 49}),
    "stack_cup": frozenset({4, 24, 27, 40, 46}),
    "move_spoon": frozenset({5, 15, 27, 35, 45}),
}
ROLLOUT_VALIDATION_EPISODES = {
    "pick_cup": frozenset({1, 8, 11, 13, 23, 24, 26, 32, 34}),
    "stack_cup": frozenset({1, 4, 6, 11, 15, 16, 22, 23, 32, 36}),
    "move_spoon": frozenset({2, 12, 14, 20, 23, 26, 28, 30, 39}),
}


@dataclass(frozen=True)
class TaskProfile:
    task: str
    env_name: str
    human_source_root: Path
    rollout_source_root: Path
    baseline_human_dataset: Path
    human_output: Path
    rollout_output: Path
    expected_humans: int
    expected_rollouts: int
    expected_rollout_actions: int
    expected_requests_per_rollout: int
    expected_outcomes: Mapping[str, int]
    expected_checkpoint_sha256: str
    expected_human_transitions: int
    expected_experiment_name: str
    rollout_identity_layout: str
    expected_runtime_inference_steps: int
    max_human_image_age_sec: float

    @property
    def expected_requests(self) -> int:
        return self.expected_rollouts * self.expected_requests_per_rollout


TASK_PROFILES = {
    "pick_cup": TaskProfile(
        task="pick_cup",
        env_name="PickCupReal-v0",
        human_source_root=Path("/home/ryan/datasets_new/pick_cup/human"),
        rollout_source_root=Path("/home/ryan/datasets_new/pick_cup/rollout"),
        baseline_human_dataset=ROOT / "datasets/real_robot/pick_cup/pick_cup_rgb.hdf5",
        human_output=ROOT / "datasets/real_robot/pick_cup/pick_cup_rgb.hdf5",
        rollout_output=ROOT / "datasets/real_robot/pick_cup/idql/pick_cup_episode_layout_v1_request_rollouts.hdf5",
        expected_humans=50,
        expected_rollouts=43,
        expected_rollout_actions=400,
        expected_requests_per_rollout=50,
        expected_outcomes={"success": 29, "failure": 14},
        expected_checkpoint_sha256="0d37bc1e57987d603ef46c4808f87e3b8ae281b673b6cb4e3bf07b9666b87742",
        expected_human_transitions=18_739,
        expected_experiment_name="pick_cup_rgb_dp_ddim_s1",
        rollout_identity_layout="legacy_checkpoint_contract",
        expected_runtime_inference_steps=10,
        max_human_image_age_sec=1.0,
    ),
    "stack_cup": TaskProfile(
        task="stack_cup",
        env_name="StackCupReal-v0",
        human_source_root=Path("/home/ryan/datasets_new/stack_cup/human"),
        rollout_source_root=Path("/home/ryan/datasets_new/stack_cup/rollout"),
        baseline_human_dataset=ROOT / "datasets/real_robot/stack_cup/stack_cup_rgb.hdf5",
        human_output=ROOT / "datasets/real_robot/stack_cup/idql/stack_cup_episode_layout_v1_human.hdf5",
        rollout_output=ROOT / "datasets/real_robot/stack_cup/idql/stack_cup_episode_layout_v1_request_rollouts.hdf5",
        expected_humans=50,
        expected_rollouts=40,
        expected_rollout_actions=600,
        expected_requests_per_rollout=75,
        expected_outcomes={"success": 26, "failure": 14},
        expected_checkpoint_sha256="b1bbe2f6be8eeb1317ba270777c0b17e265464c33906134164f4d62d7b4bfa6d",
        expected_human_transitions=21_166,
        expected_experiment_name="stack_cup_rgb_dp_ddim_s1",
        rollout_identity_layout="runtime_sampler_override",
        expected_runtime_inference_steps=100,
        max_human_image_age_sec=0.5,
    ),
    "move_spoon": TaskProfile(
        task="move_spoon",
        env_name="MoveSpoonReal-v0",
        human_source_root=Path("/home/ryan/datasets_new/move_spoon/human"),
        rollout_source_root=Path("/home/ryan/datasets_new/move_spoon/rollout"),
        baseline_human_dataset=ROOT / "datasets/real_robot/move_spoon/move_spoon_rgb.hdf5",
        human_output=ROOT / "datasets/real_robot/move_spoon/idql/move_spoon_episode_layout_v1_human.hdf5",
        rollout_output=ROOT / "datasets/real_robot/move_spoon/idql/move_spoon_episode_layout_v1_request_rollouts.hdf5",
        expected_humans=50,
        expected_rollouts=40,
        expected_rollout_actions=600,
        expected_requests_per_rollout=75,
        expected_outcomes={"success": 25, "failure": 15},
        expected_checkpoint_sha256="c20a6497c82ffedc6dd8849a4bbaad9f29612bff70bb02015b8e2fa59f65ecdf",
        expected_human_transitions=20_581,
        expected_experiment_name="move_spoon_rgb_dp_ddim_s1",
        rollout_identity_layout="runtime_sampler_override",
        expected_runtime_inference_steps=100,
        max_human_image_age_sec=0.5,
    ),
}


class ProposalConversionError(ValueError):
    """The proposal export violates the immutable training contract."""


@dataclass(frozen=True)
class BuildOptions:
    task: str = "stack_cup"
    rollout_source_root: Path | None = None
    human_source_root: Path | None = None
    output: Path | None = None
    human_output: Path | None = None
    compression: str = "gzip"
    overwrite: bool = False
    validate_only: bool = False
    validate_output_only: bool = False
    source_kind: str = "both"


@dataclass(frozen=True)
class EpisodeRecord:
    episode_number: int
    package_name: str
    run_id: str
    split: str
    source_group: str
    proposal_exact: bool
    actions: int
    frames: int
    windows: int

    @property
    def kind(self) -> str:
        return "rollout" if self.proposal_exact else "human"


def _load_json(path: Path) -> Any:
    try:
        with path.open() as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ProposalConversionError(f"malformed JSON in {path}: {exc}") from exc


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


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
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _verify_checksums(source_root: Path) -> dict[str, Any]:
    checksum_path = source_root / "metadata_checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    count = 0
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), 1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ProposalConversionError(
                f"{checksum_path}:{line_number} is malformed"
            ) from exc
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ProposalConversionError(
                f"{checksum_path}:{line_number} has an invalid digest"
            )
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ProposalConversionError(
                f"{checksum_path}:{line_number} escapes the source root"
            ) from exc
        if not path.is_file() or _file_sha256(path) != expected:
            raise ProposalConversionError(f"published checksum differs: {relative}")
        count += 1
    return {
        "entries": count,
        "checksum_file_sha256": _file_sha256(checksum_path),
        "manifest_sha256": _file_sha256(source_root / "manifest.json"),
        "verification_sha256": _file_sha256(source_root / "verification.json"),
    }


def _source_identity(
    source_root: Path,
    checksums: Mapping[str, Any],
    *,
    profile: TaskProfile,
    kind: str,
) -> dict[str, Any]:
    return {
        "root": str(source_root),
        "task": profile.task,
        "kind": kind,
        "schema": SOURCE_SCHEMA,
        "checksums": dict(checksums),
        "dataset_ready_sha256": _file_sha256(source_root / "DATASET_READY"),
        "processing_complete_sha256": _file_sha256(
            source_root / "PROCESSING_COMPLETE"
        ),
    }


def discover_episodes(
    source_root: Path,
    *,
    kind: str,
    profile: TaskProfile,
) -> tuple[list[EpisodeRecord], dict[str, Any]]:
    if kind not in {"human", "rollout"}:
        raise ValueError(f"unsupported source kind {kind!r}")
    source_root = source_root.expanduser().resolve()
    checksums = _verify_checksums(source_root)
    manifest = _load_json(source_root / "manifest.json")
    verification = _load_json(source_root / "verification.json")
    ready = _load_json(source_root / "DATASET_READY")
    complete = _load_json(source_root / "PROCESSING_COMPLETE")
    if (
        manifest.get("version") != SOURCE_SCHEMA
        or verification.get("version") not in (None, SOURCE_SCHEMA)
        or verification.get("status") != "PASS"
        or ready.get("status") != "PASS"
        or complete.get("version") not in (None, SOURCE_SCHEMA)
        or complete.get("status") not in {
            "PASS",
            "export_and_validation_complete",
        }
    ):
        raise ProposalConversionError("proposal source completion contract differs")
    if ready.get("schema") not in (None, SOURCE_SCHEMA):
        raise ProposalConversionError("DATASET_READY schema differs")
    rows = manifest.get("episodes")
    verified = verification.get("episodes")
    expected_episodes = (
        profile.expected_rollouts if kind == "rollout" else profile.expected_humans
    )
    if (
        not isinstance(rows, list)
        or len(rows) != expected_episodes
        or int(manifest.get("episode_count", -1)) != expected_episodes
        or not isinstance(verified, list)
        or len(verified) != expected_episodes
    ):
        raise ProposalConversionError("proposal source episode inventory differs")
    verified_by_name = {
        str(row.get("episode")): row for row in verified if isinstance(row, dict)
    }
    episodes_root = source_root / "episodes"
    actual = {path.name for path in episodes_root.iterdir() if path.is_dir()}
    listed = {str(row.get("episode")) for row in rows if isinstance(row, dict)}
    if actual != listed:
        raise ProposalConversionError(
            f"episode inventory differs: missing={sorted(listed-actual)}, "
            f"unlisted={sorted(actual-listed)}"
        )

    records: list[EpisodeRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ProposalConversionError("manifest episode entry must be an object")
        name = str(row.get("episode", ""))
        match = re.match(r"^episode_(\d{3})__", name)
        if match is None:
            raise ProposalConversionError(f"invalid episode directory {name!r}")
        episode_number = int(match.group(1))
        verified_row = verified_by_name.get(name)
        accepted_statuses = {"PASS"} if kind == "rollout" else {"PASS", "WARN"}
        if verified_row is None or verified_row.get("status") not in accepted_statuses:
            raise ProposalConversionError(f"{name} was not independently verified")
        split = row.get("split")
        if split not in {"train", "valid"}:
            validation_numbers = (
                ROLLOUT_VALIDATION_EPISODES[profile.task]
                if kind == "rollout"
                else HUMAN_VALIDATION_EPISODES[profile.task]
            )
            split = "valid" if episode_number in validation_numbers else "train"
        record = EpisodeRecord(
            episode_number=episode_number,
            package_name=name,
            run_id=str(row.get("run_id", "")),
            split=str(split),
            source_group=str(row.get("source_group", "")),
            proposal_exact=bool(row.get("proposal_exact")),
            actions=int(row.get("actions", -1)),
            frames=int(row.get("frames", -1)),
            windows=int(row.get("windows", -1)),
        )
        if (
            record.split not in {"train", "valid"}
            or not record.run_id
            or record.proposal_exact != (kind == "rollout")
        ):
            raise ProposalConversionError(f"{name} has invalid split or run ID")
        episode_dir = episodes_root / name
        required = [
            "actions.json",
            "frames.json",
            "windows.json",
            "contract.json",
            "qa.json",
        ]
        if kind == "rollout":
            required.extend(
                [
                    "MODEL_WINDOW_READY",
                    "snapshots/run.json",
                    "snapshots/steps.jsonl",
                    "snapshots/outcome.json",
                ]
            )
        elif not any(
            (episode_dir / marker).is_file()
            for marker in ("MODEL_WINDOW_READY", "MODEL_WINDOW_WARN")
        ):
            raise FileNotFoundError(f"{episode_dir}: missing model-window marker")
        for relative in required:
            if not (episode_dir / relative).is_file():
                raise FileNotFoundError(episode_dir / relative)
        contract = _load_json(episode_dir / "contract.json")
        qa = _load_json(episode_dir / "qa.json")
        if (
            contract.get("version") != SOURCE_SCHEMA
            or contract.get("run_id") != record.run_id
            or bool(contract.get("proposal_exact")) != record.proposal_exact
            or qa.get("status") not in accepted_statuses
            or bool(qa.get("proposal_exact")) != record.proposal_exact
        ):
            raise ProposalConversionError(f"{name} contract or QA differs")
        records.append(record)

    if kind == "rollout" and int(ready.get("exact_requests", -1)) != profile.expected_requests:
        raise ProposalConversionError("DATASET_READY exact request count differs")
    expected_split = ROLLOUT_VALIDATION_EPISODES[profile.task]
    if kind == "rollout" and {
        record.episode_number for record in records if record.split == "valid"
    } != expected_split:
        raise ProposalConversionError("rollout validation split differs")
    expected_human_split = HUMAN_VALIDATION_EPISODES[profile.task]
    if kind == "human" and {
        record.episode_number for record in records if record.split == "valid"
    } != expected_human_split:
        raise ProposalConversionError("human validation split differs")
    return records, _source_identity(
        source_root,
        checksums,
        profile=profile,
        kind=kind,
    )


def _read_actions(episode_dir: Path, expected: int) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    document = _load_json(episode_dir / "actions.json")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != expected:
        raise ProposalConversionError(f"{episode_dir}: action count differs")
    actions = np.asarray([sample.get("action") for sample in samples], dtype=np.float32)
    if (
        actions.shape != (expected, 7)
        or not np.isfinite(actions).all()
        or np.any(np.abs(actions) > 1.000001)
    ):
        raise ProposalConversionError(f"{episode_dir}: actions are invalid")
    if not np.array_equal(
        actions,
        np.asarray([sample.get("raw_action") for sample in samples], dtype=np.float32),
    ):
        raise ProposalConversionError(f"{episode_dir}: action/raw_action differ")
    if [int(sample.get("step", -1)) for sample in samples] != list(range(expected)):
        raise ProposalConversionError(f"{episode_dir}: action step indices differ")
    return document, samples, actions


def _read_windows(
    episode_dir: Path,
    expected: int,
    *,
    require_all_valid: bool = True,
) -> list[dict[str, Any]]:
    document = _load_json(episode_dir / "windows.json")
    windows = document.get("windows")
    if (
        document.get("version") != SOURCE_SCHEMA
        or not isinstance(windows, list)
        or len(windows) != expected
    ):
        raise ProposalConversionError(f"{episode_dir}: window inventory differs")
    valid = sum(bool(window.get("action_valid")) for window in windows)
    invalid = len(windows) - valid
    if (
        int(document.get("valid_windows", -1)) != valid
        or int(document.get("invalid_windows", -1)) != invalid
        or (require_all_valid and invalid != 0)
    ):
        raise ProposalConversionError(f"{episode_dir}: window validity differs")
    return windows


def _frame_state(frame: Mapping[str, Any], key: str) -> np.ndarray:
    value = np.asarray(frame.get("robot_state", {}).get(key), dtype=np.float32)
    if value.shape != OBS_SHAPES[key] or not np.isfinite(value).all():
        raise ProposalConversionError(f"invalid frame state {key}: {value.shape}")
    return value


def _load_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if value.shape != OBS_SHAPES["main_image"]:
        raise ProposalConversionError(f"{path}: RGB shape={value.shape}")
    return value


def _episode_observations(
    episode_dir: Path,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, stream in (("main_image", "main_rgb"), ("wrist_image", "wrist_rgb")):
        values = []
        cache: dict[str, np.ndarray] = {}
        for index, frame in enumerate(frames):
            if int(frame.get("index", -1)) != index:
                raise ProposalConversionError(f"{episode_dir}: frame index differs")
            relative = str(frame.get("streams", {}).get(stream, {}).get("path", ""))
            source = (episode_dir / relative).resolve()
            try:
                source.relative_to(episode_dir)
            except ValueError as exc:
                raise ProposalConversionError(f"{episode_dir}: image path escapes") from exc
            if relative not in cache:
                cache[relative] = _load_png(source)
            values.append(cache[relative])
        result[key] = np.stack(values)
    for key in LOW_DIM_KEYS:
        result[key] = np.stack([_frame_state(frame, key) for frame in frames])
    return result


def _validate_policy_identity(
    episode_dir: Path,
    record: EpisodeRecord,
    profile: TaskProfile,
) -> tuple[dict[str, Any], str]:
    run = _load_json(episode_dir / "snapshots/run.json")
    outcome = _load_json(episode_dir / "snapshots/outcome.json")
    if (
        run.get("episode_id") != record.run_id
        or run.get("status") != "PASS"
        or run.get("completion_reason") != "max_actions"
        or int(run.get("actions_completed", -1)) != profile.expected_rollout_actions
        or outcome.get("episode_id") != record.run_id
        or outcome.get("task_outcome") not in {"success", "failure"}
        or outcome.get("discarded") is not False
    ):
        raise ProposalConversionError(f"{episode_dir}: rollout identity differs")
    identity = run.get("server_health", {}).get("identity", {})
    if (
        identity.get("checkpoint", {}).get("sha256")
        != profile.expected_checkpoint_sha256
    ):
        raise ProposalConversionError(f"{episode_dir}: checkpoint identity differs")
    checkpoint_contract = identity.get("checkpoint_contract", {})
    if (
        checkpoint_contract.get("schema") != RUNTIME_SCHEMA
        or checkpoint_contract.get("runtime_contract_sha256")
        != EXPECTED_CONTRACT_SHA256
        or checkpoint_contract.get("experiment_name")
        != profile.expected_experiment_name
        or int(checkpoint_contract.get("epoch", -1)) != 200
        or int(checkpoint_contract.get("action_horizon", -1)) != ACTION_HORIZON
        or int(checkpoint_contract.get("observation_horizon", -1)) != 2
    ):
        raise ProposalConversionError(f"{episode_dir}: checkpoint contract differs")
    if profile.rollout_identity_layout == "runtime_sampler_override":
        sampler = identity.get("sampler", {})
        if (
            sampler.get("kind") != "ddim"
            or int(sampler.get("checkpoint_num_inference_steps", -1)) != 10
            or int(sampler.get("num_inference_steps", -1))
            != profile.expected_runtime_inference_steps
            or identity.get("task") != profile.task
        ):
            raise ProposalConversionError(
                f"{episode_dir}: DDIM-{profile.expected_runtime_inference_steps} "
                "runtime override differs"
            )
    elif profile.rollout_identity_layout == "legacy_checkpoint_contract":
        sampler = checkpoint_contract.get("diffusion_sampler", {})
        if (
            sampler.get("type") != "ddim"
            or int(sampler.get("num_inference_timesteps", -1))
            != profile.expected_runtime_inference_steps
            or int(sampler.get("num_train_timesteps", -1)) != 100
        ):
            raise ProposalConversionError(
                f"{episode_dir}: legacy DDIM-{profile.expected_runtime_inference_steps} "
                "checkpoint sampler differs"
            )
    else:
        raise ProposalConversionError(
            f"{episode_dir}: unsupported rollout identity layout "
            f"{profile.rollout_identity_layout!r}"
        )
    runtime = run.get("runtime_contract", {})
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("sha256") != EXPECTED_CONTRACT_SHA256
        or runtime.get("horizons", {}).get("action") != ACTION_HORIZON
        or runtime.get("horizons", {}).get("observation") != 2
    ):
        raise ProposalConversionError(f"{episode_dir}: runtime contract differs")
    return run, str(outcome["task_outcome"])


def _policy_payload(
    episode_dir: Path,
    record: EpisodeRecord,
    profile: TaskProfile,
) -> dict[str, Any]:
    if (
        record.actions != profile.expected_rollout_actions
        or record.frames != 2 * profile.expected_requests_per_rollout
        or record.windows != profile.expected_requests_per_rollout
    ):
        raise ProposalConversionError(f"{episode_dir}: rollout dimensions differ")
    _, samples, actions = _read_actions(episode_dir, record.actions)
    windows = _read_windows(episode_dir, record.windows)
    frames_doc = _load_json(episode_dir / "frames.json")
    frames = frames_doc.get("frames")
    if not isinstance(frames, list) or len(frames) != record.frames:
        raise ProposalConversionError(f"{episode_dir}: frame inventory differs")
    run, outcome = _validate_policy_identity(episode_dir, record, profile)
    chunks = run.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != record.windows:
        raise ProposalConversionError(f"{episode_dir}: run chunks differ")

    request_values = {key: [] for key in OBS_KEYS}
    request_state_times = []
    request_main_times = []
    request_wrist_times = []
    request_files = []
    for request_index, (window, chunk) in enumerate(zip(windows, chunks)):
        expected_indices = list(
            range(request_index * ACTION_HORIZON, (request_index + 1) * ACTION_HORIZON)
        )
        if (
            int(window.get("window_id", -1)) != request_index
            or window.get("kind") != "exact_rollout_request"
            or window.get("proposal_exact") is not True
            or window.get("action_indices") != expected_indices
            or window.get("frame_indices") != [2 * request_index, 2 * request_index + 1]
            or int(chunk.get("chunk", -1)) != request_index
        ):
            raise ProposalConversionError(
                f"{episode_dir}: request window {request_index} differs"
            )
        raw_actions = np.asarray(chunk.get("raw_actions"), dtype=np.float32)
        if not np.array_equal(raw_actions, actions[expected_indices]):
            raise ProposalConversionError(
                f"{episode_dir}: request {request_index} actions differ"
            )
        npz_path = (episode_dir / str(window.get("input_npz", ""))).resolve()
        try:
            npz_path.relative_to(episode_dir)
        except ValueError as exc:
            raise ProposalConversionError(f"{episode_dir}: NPZ path escapes") from exc
        if _file_sha256(npz_path) != window.get("input_npz_sha256"):
            raise ProposalConversionError(
                f"{episode_dir}: request {request_index} file digest differs"
            )
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(OBS_KEYS):
                raise ProposalConversionError(
                    f"{episode_dir}: request {request_index} keys differ"
                )
            request = {
                key: np.ascontiguousarray(archive[key]) for key in OBS_KEYS
            }
        for key in OBS_KEYS:
            expected_dtype = np.dtype(np.uint8 if key in RGB_KEYS else np.float32)
            if request[key].shape != (2, *OBS_SHAPES[key]) or request[key].dtype != expected_dtype:
                raise ProposalConversionError(
                    f"{episode_dir}: request {request_index}/{key} has "
                    f"{request[key].shape}/{request[key].dtype}"
                )
        hashes = {key: _array_sha256(request[key]) for key in OBS_KEYS}
        digest = chunk.get("input_digest", {})
        if hashes != digest.get("arrays"):
            raise ProposalConversionError(
                f"{episode_dir}: request {request_index} array digest differs"
            )
        aggregate = _canonical_sha256(
            {"schema": run["runtime_contract"]["schema"], "arrays": hashes}
        )
        if aggregate != digest.get("aggregate"):
            raise ProposalConversionError(
                f"{episode_dir}: request {request_index} aggregate differs"
            )
        timestamps = chunk.get("timestamps", {})
        state_times = np.asarray(timestamps.get("state"), dtype=np.float64)
        main_times = np.asarray(timestamps.get("main"), dtype=np.float64)
        wrist_times = np.asarray(timestamps.get("wrist"), dtype=np.float64)
        if any(value.shape != (2,) for value in (state_times, main_times, wrist_times)):
            raise ProposalConversionError(
                f"{episode_dir}: request {request_index} timestamps differ"
            )
        state_delta = float(state_times[1] - state_times[0])
        if not 0.03 - 1e-9 <= state_delta <= 0.07 + 1e-9:
            raise ProposalConversionError(
                f"{episode_dir}: request {request_index} state dt={state_delta:.6f}"
            )
        for history_index, frame_index in enumerate(window["frame_indices"]):
            frame = frames[int(frame_index)]
            for key in LOW_DIM_KEYS:
                if not np.array_equal(_frame_state(frame, key), request[key][history_index]):
                    raise ProposalConversionError(
                        f"{episode_dir}: request frame state {key} differs"
                    )
        for key in OBS_KEYS:
            request_values[key].append(request[key])
        request_state_times.append(state_times)
        request_main_times.append(main_times)
        request_wrist_times.append(wrist_times)
        request_files.append(str(window["input_npz"]))
        if request_index + 1 < len(windows):
            successor = windows[request_index + 1]
            if (
                window.get("next_input_npz") != successor.get("input_npz")
                or window.get("next_frame_indices") != successor.get("frame_indices")
                or window.get("bootstrap_valid") is not True
            ):
                raise ProposalConversionError(
                    f"{episode_dir}: request {request_index} successor differs"
                )
        elif (
            window.get("next_input_npz") is not None
            or window.get("next_frame_indices") is not None
            or window.get("bootstrap_valid") is not False
            or window.get("dataset_terminal") is not True
        ):
            raise ProposalConversionError(f"{episode_dir}: final request differs")

    request_obs = {
        key: np.stack(values) for key, values in request_values.items()
    }
    # Compatibility rows keep SequenceDataset's action and conditioning
    # machinery intact. The request-aware sparse loader replaces these repeated
    # current observations with the exact two-frame request tensors.
    obs = {
        key: np.repeat(request_obs[key][:, 1], ACTION_HORIZON, axis=0)
        for key in OBS_KEYS
    }
    rewards = np.asarray([sample.get("reward") for sample in samples], dtype=np.float32)
    dones = np.asarray([sample.get("done") for sample in samples], dtype=np.uint8)
    expected_rewards = np.zeros(record.actions, dtype=np.float32)
    if outcome == "success":
        expected_rewards[-1] = 1.0
    expected_dones = np.zeros(record.actions, dtype=np.uint8)
    expected_dones[-1] = 1
    if not np.array_equal(rewards, expected_rewards) or not np.array_equal(dones, expected_dones):
        raise ProposalConversionError(f"{episode_dir}: terminal labels differ")
    before_pose = np.asarray([sample.get("before_pose") for sample in samples], dtype=np.float64)
    if before_pose.shape != (record.actions, 7) or not np.isfinite(before_pose).all():
        raise ProposalConversionError(f"{episode_dir}: before_pose differs")
    source_indices = np.asarray([sample.get("source_index") for sample in samples], dtype=np.int64)
    if not np.array_equal(source_indices, np.arange(record.actions)):
        raise ProposalConversionError(f"{episode_dir}: source indices differ")
    return {
        "actions": actions,
        "obs": obs,
        "request_obs": request_obs,
        "rewards": rewards,
        "dones": dones,
        "outcome": outcome,
        "chunk_critic_valid": (source_indices % ACTION_HORIZON == 0).astype(np.uint8),
        # This source is request-level H8 data, not a true one-step transition
        # corpus. Keeping this mask empty makes accidental one-step use fail.
        "one_step_critic_valid": np.zeros(record.actions, dtype=np.uint8),
        "provenance": {
            "source_action_index": source_indices,
            "source_action_time": np.asarray(
                [sample.get("source_time") for sample in samples], dtype=np.float64
            ),
            "action_dt_sec": np.asarray(
                [sample.get("dt_to_next_action_sec") for sample in samples],
                dtype=np.float64,
            ),
            "timing_boundary": np.asarray(
                [sample.get("timing_boundary") for sample in samples], dtype=np.uint8
            ),
            "policy_chunk_index": source_indices // ACTION_HORIZON,
            "policy_chunk_offset": (source_indices % ACTION_HORIZON).astype(np.uint8),
            "action_before_pose": before_pose,
            "request_state_time": np.stack(request_state_times),
            "request_main_time": np.stack(request_main_times),
            "request_wrist_time": np.stack(request_wrist_times),
            "request_action_start": np.arange(
                0, record.actions, ACTION_HORIZON, dtype=np.int64
            ),
            "request_bootstrap_valid": np.asarray(
                [window.get("bootstrap_valid") for window in windows], dtype=np.uint8
            ),
        },
        "request_files": request_files,
    }


def _densify_gripper(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if events.ndim != 1 or np.any(~np.isin(events, (-1.0, 0.0, 1.0))):
        raise ProposalConversionError("human gripper events must be in {-1, 0, 1}")
    observations = np.empty(events.shape, dtype=np.float32)
    targets = np.empty(events.shape, dtype=np.float32)
    state = 1.0
    for index, event in enumerate(events):
        observations[index] = state
        if event < 0.0:
            state = -1.0
        elif event > 0.0:
            state = 1.0
        targets[index] = state
    return observations, targets


def _human_payload(
    episode_dir: Path,
    record: EpisodeRecord,
    profile: TaskProfile,
) -> dict[str, Any]:
    _, samples, actions = _read_actions(episode_dir, record.actions)
    windows = _read_windows(
        episode_dir,
        record.windows,
        require_all_valid=False,
    )
    frames = _load_json(episode_dir / "frames.json").get("frames")
    if not isinstance(frames, list) or len(frames) != record.frames:
        raise ProposalConversionError(f"{episode_dir}: human frame inventory differs")

    coverage = np.zeros(record.actions, dtype=np.uint8)
    window_index = np.full(record.actions, -1, dtype=np.int64)
    window_offset = np.zeros(record.actions, dtype=np.uint8)
    for index, window in enumerate(windows):
        action_indices = np.asarray(window.get("action_indices"), dtype=np.int64)
        if (
            int(window.get("window_id", -1)) != index
            or window.get("kind") != "derived_causal_window_not_request"
            or window.get("proposal_exact") is not False
            or action_indices.ndim != 1
            or not 1 <= action_indices.size <= ACTION_HORIZON
            or not np.array_equal(
                action_indices,
                np.arange(action_indices[0], action_indices[0] + action_indices.size),
            )
        ):
            raise ProposalConversionError(f"{episode_dir}: human window {index} differs")
        coverage[action_indices] += 1
        window_index[action_indices] = index
        window_offset[action_indices] = np.arange(action_indices.size, dtype=np.uint8)
    if not np.all(coverage == 1):
        raise ProposalConversionError(
            f"{episode_dir}: human windows do not partition actions"
        )

    target_times = np.asarray(
        [sample.get("target_time") for sample in samples], dtype=np.float64
    )
    source_times = np.asarray(
        [sample.get("source_time") for sample in samples], dtype=np.float64
    )
    source_indices = np.asarray(
        [sample.get("source_index") for sample in samples], dtype=np.int64
    )
    if (
        not np.isfinite(target_times).all()
        or not np.isfinite(source_times).all()
        or not np.allclose(np.diff(target_times), 0.05, atol=5e-6, rtol=0.0)
        or np.any(np.diff(source_times) < 0.0)
        or np.any(np.diff(source_indices) < 0)
    ):
        raise ProposalConversionError(f"{episode_dir}: human timing grid differs")
    before_pose = np.asarray(
        [sample.get("before_pose") for sample in samples], dtype=np.float32
    )
    after_pose = np.asarray(
        [sample.get("after_pose") for sample in samples], dtype=np.float32
    )
    if (
        before_pose.shape != (record.actions, 7)
        or after_pose.shape != (record.actions, 7)
        or not np.isfinite(before_pose).all()
        or not np.isfinite(after_pose).all()
        or not np.allclose(
            np.linalg.norm(before_pose[:, 3:], axis=1), 1.0, atol=5e-3
        )
        or not np.allclose(
            np.linalg.norm(after_pose[:, 3:], axis=1), 1.0, atol=5e-3
        )
    ):
        raise ProposalConversionError(f"{episode_dir}: human poses differ")

    paired_times = []
    main_paths = []
    wrist_paths = []
    for frame_index, frame in enumerate(frames):
        if int(frame.get("index", -1)) != frame_index:
            raise ProposalConversionError(f"{episode_dir}: frame index differs")
        streams = frame.get("streams", {})
        times = []
        paths = []
        for stream_name in ("main_rgb", "wrist_rgb"):
            stream = streams.get(stream_name, {})
            if (
                stream.get("encoding") != "rgb8"
                or int(stream.get("width", -1)) != 128
                or int(stream.get("height", -1)) != 96
            ):
                raise ProposalConversionError(
                    f"{episode_dir}: {stream_name} contract differs"
                )
            times.append(float(stream.get("header_time_sec")))
            path = (episode_dir / str(stream.get("path", ""))).resolve()
            try:
                path.relative_to(episode_dir.resolve())
            except ValueError as exc:
                raise ProposalConversionError(
                    f"{episode_dir}: image path escapes"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        if abs(times[0] - times[1]) > 0.050001:
            raise ProposalConversionError(f"{episode_dir}: camera skew exceeds 50 ms")
        paired_times.append(max(times))
        main_paths.append(paths[0])
        wrist_paths.append(paths[1])
    paired_times_array = np.asarray(paired_times, dtype=np.float64)
    if not np.all(np.diff(paired_times_array) > 0.0):
        raise ProposalConversionError(f"{episode_dir}: frame times are not increasing")
    selected = np.searchsorted(paired_times_array, target_times, side="right") - 1
    causal = np.flatnonzero(selected >= 0)
    if causal.size == 0 or not np.array_equal(
        causal, np.arange(causal[0], record.actions)
    ):
        raise ProposalConversionError(f"{episode_dir}: causal RGB coverage differs")
    first = int(causal[0])
    selected = selected[first:].astype(np.int64)
    image_age = target_times[first:] - paired_times_array[selected]
    if (
        np.any(image_age < -1e-6)
        or np.max(image_age) > profile.max_human_image_age_sec + 1e-6
    ):
        raise ProposalConversionError(
            f"{episode_dir}: maximum causal RGB age {np.max(image_age):.6f}s "
            f"exceeds the {profile.max_human_image_age_sec:.6f}s "
            f"{profile.task} limit"
        )

    caches: dict[str, dict[int, np.ndarray]] = {
        "main_image": {},
        "wrist_image": {},
    }
    for position in sorted(set(selected.tolist())):
        caches["main_image"][position] = _load_png(main_paths[position])
        caches["wrist_image"][position] = _load_png(wrist_paths[position])
    obs = {
        key: np.stack([caches[key][int(position)] for position in selected])
        for key in RGB_KEYS
    }
    gripper_obs, gripper_targets = _densify_gripper(actions[:, 6])
    dense_actions = actions.copy()
    dense_actions[:, 6] = gripper_targets
    obs.update(
        {
            "robot0_eef_pos": before_pose[first:, :3],
            "robot0_eef_quat": before_pose[first:, 3:],
            "robot0_gripper_state": gripper_obs[first:, None],
        }
    )
    count = record.actions - first
    rewards = np.zeros(count, dtype=np.float32)
    rewards[-1] = 1.0
    dones = np.zeros(count, dtype=np.uint8)
    dones[-1] = 1
    valid = np.ones(count, dtype=np.uint8)
    boundary = np.asarray(
        [sample.get("timing_boundary") for sample in samples], dtype=np.uint8
    )
    return {
        "actions": dense_actions[first:],
        "obs": obs,
        "rewards": rewards,
        "dones": dones,
        "outcome": "success",
        "chunk_critic_valid": valid.copy(),
        "one_step_critic_valid": valid.copy(),
        "provenance": {
            "source_action_array_index": np.arange(
                first, record.actions, dtype=np.int64
            ),
            "source_action_index": source_indices[first:],
            "action_target_time": target_times[first:],
            "source_action_time": source_times[first:],
            "action_dt_sec": np.asarray(
                [sample.get("dt_to_next_action_sec") for sample in samples],
                dtype=np.float64,
            )[first:],
            "source_derived_window_boundary": boundary[first:],
            "human_window_index": window_index[first:],
            "human_window_offset": window_offset[first:],
            "action_before_pose": before_pose[first:],
            "selected_frame_position": selected,
            "selected_frame_time": paired_times_array[selected],
            "image_age_sec": image_age,
        },
        "request_files": [],
        "dropped_precausal_actions": first,
    }


def _compression_kwargs(compression: str) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf"}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    raise ValueError(f"unsupported compression {compression!r}")


def _write_array(group: h5py.Group, key: str, value: np.ndarray, compression: str) -> h5py.Dataset:
    kwargs = _compression_kwargs(compression) if value.ndim >= 3 else {}
    if value.ndim >= 3:
        kwargs["chunks"] = (min(8, int(value.shape[0])), *value.shape[1:])
    return group.create_dataset(key, data=value, **kwargs)


def _env_args(profile: TaskProfile) -> str:
    if profile.baseline_human_dataset.is_file():
        with h5py.File(profile.baseline_human_dataset, "r") as source:
            value = _text(source["data"].attrs["env_args"])
        parsed = json.loads(value)
        if parsed.get("env_name") != profile.env_name:
            raise ProposalConversionError(
                f"baseline {profile.task} env_args differ"
            )
    else:
        # The first baseline build cannot copy metadata from itself. Keep the
        # same real-robot environment contract used by the existing StackCup
        # and MoveSpoon baseline datasets.
        parsed = {
            "env_name": profile.env_name,
            "env_version": f"{profile.task}_rgb_dp_v1",
            "type": 2,
            "env_kwargs": {
                "real_robot": True,
                "control_freq": 20,
                "camera_names": ["main", "wrist"],
                "camera_height": 96,
                "camera_width": 128,
                "task": profile.task,
            },
        }
    return json.dumps(parsed, sort_keys=True)


def _atomic_target(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    return descriptor, Path(name)


def _write_source(
    *,
    source_root: Path,
    records: Sequence[EpisodeRecord],
    source_identity: Mapping[str, Any],
    profile: TaskProfile,
    output: Path,
    kind: str,
    compression: str,
) -> dict[str, Any]:
    descriptor, temporary = _atomic_target(output)
    os.close(descriptor)
    key_by_name = {
        record.package_name: f"demo_{index:03d}"
        for index, record in enumerate(records, 1)
    }
    outcomes: dict[str, str] = {}
    written_counts: dict[str, int] = {}
    total = 0
    request_total = 0
    full_chunk_total = 0
    generation_id = uuid.uuid4().hex
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["conversion_version"] = CONVERSION_VERSION
            target.attrs["task"] = profile.task
            target.attrs["generation_id"] = generation_id
            target.attrs["source_kind"] = kind
            data = target.create_group("data")
            data.attrs["env_args"] = _env_args(profile)
            for record in records:
                episode_dir = source_root / "episodes" / record.package_name
                payload = (
                    _policy_payload(episode_dir, record, profile)
                    if kind == "rollout"
                    else _human_payload(episode_dir, record, profile)
                )
                key = key_by_name[record.package_name]
                demo = data.create_group(key)
                count = int(payload["actions"].shape[0])
                demo.attrs["num_samples"] = count
                demo.attrs["source_package_episode"] = record.package_name
                demo.attrs["source_run_id"] = record.run_id
                demo.attrs["source_group"] = record.source_group
                demo.attrs["source_split"] = record.split
                demo.attrs["task_outcome"] = payload["outcome"]
                demo.attrs["proposal_exact"] = int(record.proposal_exact)
                demo.attrs["request_aligned"] = int(record.proposal_exact)
                demo.attrs["one_step_aligned"] = 0
                if not record.proposal_exact:
                    demo.attrs["dropped_precausal_actions"] = int(
                        payload["dropped_precausal_actions"]
                    )
                    demo.attrs["max_image_age_sec"] = float(
                        np.max(payload["provenance"]["image_age_sec"])
                    )
                demo.create_dataset("actions", data=payload["actions"])
                demo.create_dataset("rewards", data=payload["rewards"])
                demo.create_dataset("dones", data=payload["dones"])
                demo.create_dataset(
                    "chunk_critic_valid", data=payload["chunk_critic_valid"]
                )
                demo.create_dataset(
                    "one_step_critic_valid", data=payload["one_step_critic_valid"]
                )
                obs = demo.create_group("obs")
                for obs_key, values in payload["obs"].items():
                    _write_array(obs, obs_key, values, compression)
                if record.proposal_exact:
                    request_obs = demo.create_group("request_obs")
                    for obs_key, values in payload["request_obs"].items():
                        _write_array(request_obs, obs_key, values, compression)
                    request_obs.attrs["history_order"] = json.dumps(["t-1", "t"])
                    request_obs.attrs["source"] = "digest-verified chunk_XXXX_input.npz"
                    request_obs.attrs["files"] = json.dumps(payload["request_files"])
                    request_total += int(next(iter(payload["request_obs"].values())).shape[0])
                provenance = demo.create_group("provenance")
                for name, values in payload["provenance"].items():
                    provenance.create_dataset(name, data=values)
                full_chunk_total += int(np.sum(payload["chunk_critic_valid"]))
                total += count
                outcomes[record.package_name] = str(payload["outcome"])
                written_counts[record.package_name] = count
            data.attrs["total"] = total
            masks = target.create_group("mask")
            mask_values = {
                "all": [key_by_name[r.package_name] for r in records],
                "train": [key_by_name[r.package_name] for r in records if r.split == "train"],
                "valid": [key_by_name[r.package_name] for r in records if r.split == "valid"],
            }
            if kind == "rollout":
                for outcome in ("success", "failure"):
                    mask_values[outcome] = [
                        key_by_name[r.package_name]
                        for r in records
                        if outcomes[r.package_name] == outcome
                    ]
                    for split in ("train", "valid"):
                        mask_values[f"{outcome}_{split}"] = [
                            key_by_name[r.package_name]
                            for r in records
                            if outcomes[r.package_name] == outcome and r.split == split
                        ]
            for name, values in mask_values.items():
                masks.create_dataset(name, data=np.asarray(values, dtype="S"))
            manifest = {
                "conversion_version": CONVERSION_VERSION,
                "task": profile.task,
                "rollout_policy_provenance": (
                    {
                        "identity_layout": profile.rollout_identity_layout,
                        "experiment_name": profile.expected_experiment_name,
                        "checkpoint_sha256": profile.expected_checkpoint_sha256,
                        "sampler": "ddim",
                        "runtime_num_inference_steps": (
                            profile.expected_runtime_inference_steps
                        ),
                    }
                    if kind == "rollout"
                    else None
                ),
                "generation_id": generation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "source_identity": dict(source_identity),
                "episode_count": len(records),
                "transition_count": total,
                "request_count": request_total,
                "full_h8_window_count": full_chunk_total,
                "observation": {
                    "request_obs": (
                        "exact two-frame model request indexed by policy_chunk_index"
                        if kind == "rollout"
                        else None
                    ),
                    "obs": (
                        "request-current compatibility row repeated across H8; "
                        "request-aware sparse loader must replace it"
                        if kind == "rollout"
                        else "causal human action-time observation"
                    ),
                    "max_human_image_age_sec": (
                        profile.max_human_image_age_sec
                        if kind == "human"
                        else None
                    ),
                },
                "one_step_policy_rollouts_supported": False,
                "masks": mask_values,
                "episodes": [
                    {
                        "demo_key": key_by_name[r.package_name],
                        "source_package_episode": r.package_name,
                        "source_run_id": r.run_id,
                        "source_group": r.source_group,
                        "split": r.split,
                        "outcome": outcomes[r.package_name],
                        "num_samples": written_counts[r.package_name],
                        "proposal_exact": r.proposal_exact,
                    }
                    for r in records
                ],
            }
            target.attrs[MANIFEST_ATTR] = json.dumps(manifest, sort_keys=True)
            target.flush()
        mode = output.stat().st_mode & 0o777 if output.exists() else 0o664
        os.chmod(temporary, mode)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "kind": kind,
        "episodes": len(records),
        "transitions": total,
        "requests": request_total,
        "full_h8_windows": full_chunk_total,
    }


def _decode(values: Iterable[Any]) -> list[str]:
    return [_text(value) for value in values]


def validate_output(
    path: Path,
    *,
    kind: str,
    profile: TaskProfile,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_episodes = (
        profile.expected_rollouts if kind == "rollout" else profile.expected_humans
    )
    expected_transitions = (
        profile.expected_rollouts * profile.expected_rollout_actions
        if kind == "rollout"
        else profile.expected_human_transitions
    )
    expected_requests = profile.expected_requests if kind == "rollout" else 0
    errors = []
    with h5py.File(path, "r") as source:
        if _text(source.attrs.get("conversion_version", "")) != CONVERSION_VERSION:
            errors.append("conversion version differs")
        if _text(source.attrs.get("task", "")) != profile.task:
            errors.append("task differs")
        if _text(source.attrs.get("source_kind", "")) != kind:
            errors.append("source kind differs")
        try:
            manifest = json.loads(_text(source.attrs[MANIFEST_ATTR]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProposalConversionError(f"{path}: invalid conversion manifest") from exc
        data = source.get("data")
        masks = source.get("mask")
        if not isinstance(data, h5py.Group) or not isinstance(masks, h5py.Group):
            raise ProposalConversionError(f"{path}: data or mask group is absent")
        if len(data) != expected_episodes:
            errors.append(f"episode count={len(data)}, expected={expected_episodes}")
        total = 0
        requests = 0
        full_chunks = 0
        for key, demo in data.items():
            count = int(demo.attrs.get("num_samples", -1))
            total += count
            if demo["actions"].shape != (count, 7):
                errors.append(f"{key}/actions shape differs")
                continue
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
            if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.000001):
                errors.append(f"{key}/actions are not finite normalized actions")
            outcome = _text(demo.attrs.get("task_outcome", ""))
            expected_rewards = np.zeros(count, dtype=np.float32)
            if kind == "human" or outcome == "success":
                expected_rewards[-1] = 1.0
            expected_dones = np.zeros(count, dtype=np.uint8)
            expected_dones[-1] = 1
            if not np.array_equal(demo["rewards"][:], expected_rewards):
                errors.append(f"{key}/rewards differ")
            if not np.array_equal(demo["dones"][:], expected_dones):
                errors.append(f"{key}/dones differ")
            for obs_key, trailing in OBS_SHAPES.items():
                obs_dataset = demo[f"obs/{obs_key}"]
                if obs_dataset.shape != (count, *trailing):
                    errors.append(f"{key}/obs/{obs_key} shape differs")
                expected_dtype = np.dtype(
                    np.uint8 if obs_key in RGB_KEYS else np.float32
                )
                if obs_dataset.dtype != expected_dtype:
                    errors.append(f"{key}/obs/{obs_key} dtype differs")
            values = np.asarray(demo["chunk_critic_valid"][:], dtype=np.uint8)
            if values.shape != (count,) or np.any(~np.isin(values, (0, 1))):
                errors.append(f"{key}/chunk_critic_valid differs")
            full_chunks += int(values.sum())
            one_step = np.asarray(demo["one_step_critic_valid"][:], dtype=np.uint8)
            if one_step.shape != (count,) or np.any(~np.isin(one_step, (0, 1))):
                errors.append(f"{key}/one_step_critic_valid differs")
            request_aligned = bool(demo.attrs.get("request_aligned", 0))
            if request_aligned != (kind == "rollout"):
                errors.append(f"{key} request_aligned differs")
            if kind == "rollout":
                for obs_key, trailing in OBS_SHAPES.items():
                    request_dataset = demo[f"request_obs/{obs_key}"]
                    if request_dataset.shape != (
                        profile.expected_requests_per_rollout,
                        2,
                        *trailing,
                    ):
                        errors.append(f"{key}/request_obs/{obs_key} shape differs")
                    expected_dtype = np.dtype(
                        np.uint8 if obs_key in RGB_KEYS else np.float32
                    )
                    if request_dataset.dtype != expected_dtype:
                        errors.append(f"{key}/request_obs/{obs_key} dtype differs")
                starts = np.asarray(
                    demo["provenance/request_action_start"][:], dtype=np.int64
                )
                if not np.array_equal(
                    starts,
                    np.arange(
                        0,
                        profile.expected_rollout_actions,
                        ACTION_HORIZON,
                    ),
                ):
                    errors.append(f"{key} request starts differ")
                expected_valid = np.zeros(count, dtype=np.uint8)
                expected_valid[starts] = 1
                if not np.array_equal(values, expected_valid):
                    errors.append(f"{key} proposal validity differs")
                offsets = np.asarray(
                    demo["provenance/policy_chunk_offset"][:], dtype=np.uint8
                )
                chunk_indices = np.asarray(
                    demo["provenance/policy_chunk_index"][:], dtype=np.int64
                )
                bootstrap = np.asarray(
                    demo["provenance/request_bootstrap_valid"][:], dtype=np.uint8
                )
                if not np.array_equal(
                    offsets,
                    np.tile(np.arange(ACTION_HORIZON, dtype=np.uint8), len(starts)),
                ):
                    errors.append(f"{key} proposal offsets differ")
                if not np.array_equal(
                    chunk_indices,
                    np.repeat(np.arange(len(starts), dtype=np.int64), ACTION_HORIZON),
                ):
                    errors.append(f"{key} proposal indices differ")
                expected_bootstrap = np.ones(len(starts), dtype=np.uint8)
                expected_bootstrap[-1] = 0
                if not np.array_equal(bootstrap, expected_bootstrap):
                    errors.append(f"{key} request bootstrap validity differs")
                if np.any(one_step != 0):
                    errors.append(f"{key} incorrectly enables one-step rollout rows")
                for obs_key in OBS_KEYS:
                    if not np.array_equal(
                        demo[f"obs/{obs_key}"][starts],
                        demo[f"request_obs/{obs_key}"][:, -1],
                    ):
                        errors.append(f"{key}/{obs_key} compatibility rows differ")
                requests += len(starts)
            elif (
                np.any(values != 1)
                or np.any(one_step != 1)
                or bool(demo.attrs.get("one_step_aligned", 0))
            ):
                errors.append(f"{key} human stride-one validity differs")
        if total != expected_transitions or int(data.attrs.get("total", -1)) != total:
            errors.append(f"transition total={total}, expected={expected_transitions}")
        if requests != expected_requests:
            errors.append(f"request total={requests}, expected={expected_requests}")
        expected_full_chunks = (
            profile.expected_requests
            if kind == "rollout"
            else profile.expected_human_transitions
        )
        if full_chunks != expected_full_chunks:
            errors.append(
                f"full H8 windows={full_chunks}, expected={expected_full_chunks}"
            )
        if (
            manifest.get("kind") != kind
            or int(manifest.get("transition_count", -1)) != total
            or int(manifest.get("request_count", -1)) != requests
            or int(manifest.get("full_h8_window_count", -1)) != full_chunks
        ):
            errors.append("manifest totals differ")
        all_keys = set(_decode(masks["all"][:])) if "all" in masks else set()
        train_keys = set(_decode(masks["train"][:])) if "train" in masks else set()
        valid_keys = set(_decode(masks["valid"][:])) if "valid" in masks else set()
        if all_keys != set(data.keys()) or train_keys & valid_keys or train_keys | valid_keys != all_keys:
            errors.append("train/valid masks do not form an exact episode partition")
        expected_valid_count = len(
            ROLLOUT_VALIDATION_EPISODES[profile.task]
            if kind == "rollout"
            else HUMAN_VALIDATION_EPISODES[profile.task]
        )
        if len(valid_keys) != expected_valid_count:
            errors.append("validation mask count differs")
        if kind == "rollout":
            expected_sets: dict[str, set[str]] = {
                "success": set(),
                "failure": set(),
                "success_train": set(),
                "success_valid": set(),
                "failure_train": set(),
                "failure_valid": set(),
            }
            for key, demo in data.items():
                outcome = _text(demo.attrs.get("task_outcome", ""))
                split = _text(demo.attrs.get("source_split", ""))
                if outcome not in {"success", "failure"} or split not in {
                    "train",
                    "valid",
                }:
                    errors.append(f"{key} outcome or split differs")
                    continue
                expected_sets[outcome].add(key)
                expected_sets[f"{outcome}_{split}"].add(key)
            for name, expected_values in expected_sets.items():
                actual = set(_decode(masks[name][:])) if name in masks else set()
                if actual != expected_values:
                    errors.append(f"mask/{name} membership differs")
            for outcome, count in profile.expected_outcomes.items():
                if len(expected_sets[outcome]) != count:
                    errors.append(f"{outcome} outcome count differs")
    if errors:
        raise ProposalConversionError(
            f"{path} validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "path": str(path),
        "kind": kind,
        "episodes": expected_episodes,
        "transitions": total,
        "requests": expected_requests,
        "full_h8_windows": full_chunks,
    }


def _normalized_options(options: BuildOptions) -> tuple[BuildOptions, TaskProfile]:
    try:
        profile = TASK_PROFILES[options.task]
    except KeyError as exc:
        raise ValueError(f"unsupported task {options.task!r}") from exc
    if options.source_kind not in {"both", "human", "rollout"}:
        raise ValueError(f"unsupported source kind {options.source_kind!r}")
    normalized = BuildOptions(
        task=profile.task,
        rollout_source_root=(
            options.rollout_source_root or profile.rollout_source_root
        ).expanduser().resolve(),
        human_source_root=(
            options.human_source_root or profile.human_source_root
        ).expanduser().resolve(),
        output=(options.output or profile.rollout_output).expanduser().resolve(),
        human_output=(
            options.human_output or profile.human_output
        ).expanduser().resolve(),
        compression=options.compression,
        overwrite=options.overwrite,
        validate_only=options.validate_only,
        validate_output_only=options.validate_output_only,
        source_kind=options.source_kind,
    )
    return normalized, profile


def validate_source_backed(
    options: BuildOptions,
    profile: TaskProfile,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    if options.source_kind in {"both", "rollout"}:
        rollout_records, rollout_identity = discover_episodes(
            options.rollout_source_root,
            kind="rollout",
            profile=profile,
        )
        reports["rollout"] = validate_output(
            options.output,
            kind="rollout",
            profile=profile,
        )
        with h5py.File(options.output, "r") as source:
            manifest = json.loads(_text(source.attrs[MANIFEST_ATTR]))
        if manifest.get("source_identity") != rollout_identity:
            raise ProposalConversionError(f"{options.output}: source identity changed")
        if len(rollout_records) != profile.expected_rollouts:
            raise AssertionError("source-backed validation lost rollout episodes")
    if options.source_kind in {"both", "human"}:
        human_records, human_identity = discover_episodes(
            options.human_source_root,
            kind="human",
            profile=profile,
        )
        reports["human"] = validate_output(
            options.human_output,
            kind="human",
            profile=profile,
        )
        with h5py.File(options.human_output, "r") as source:
            manifest = json.loads(_text(source.attrs[MANIFEST_ATTR]))
        if manifest.get("source_identity") != human_identity:
            raise ProposalConversionError(
                f"{options.human_output}: source identity changed"
            )
        if len(human_records) != profile.expected_humans:
            raise AssertionError("source-backed validation lost human episodes")
    return {"validated": True, **reports}


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    options, profile = _normalized_options(options)
    if options.source_kind == "both" and options.output == options.human_output:
        raise ValueError("rollout and human outputs must differ")
    if options.validate_output_only:
        reports: dict[str, Any] = {"validated": True}
        if options.source_kind in {"both", "rollout"}:
            reports["rollout"] = validate_output(
                options.output,
                kind="rollout",
                profile=profile,
            )
        if options.source_kind in {"both", "human"}:
            reports["human"] = validate_output(
                options.human_output,
                kind="human",
                profile=profile,
            )
        return reports
    if options.validate_only:
        return validate_source_backed(options, profile)
    reports: dict[str, Any] = {}
    if options.source_kind in {"both", "rollout"}:
        rollout_records, rollout_identity = discover_episodes(
            options.rollout_source_root,
            kind="rollout",
            profile=profile,
        )
        if options.output.exists() and not options.overwrite:
            reports["rollout"] = validate_output(
                options.output,
                kind="rollout",
                profile=profile,
            )
        else:
            reports["rollout"] = _write_source(
                source_root=options.rollout_source_root,
                records=rollout_records,
                source_identity=rollout_identity,
                profile=profile,
                output=options.output,
                kind="rollout",
                compression=options.compression,
            )
    if options.source_kind in {"both", "human"}:
        human_records, human_identity = discover_episodes(
            options.human_source_root,
            kind="human",
            profile=profile,
        )
        if options.human_output.exists() and not options.overwrite:
            reports["human"] = validate_output(
                options.human_output,
                kind="human",
                profile=profile,
            )
        else:
            reports["human"] = _write_source(
                source_root=options.human_source_root,
                records=human_records,
                source_identity=human_identity,
                profile=profile,
                output=options.human_output,
                kind="human",
                compression=options.compression,
            )
    validation = validate_source_backed(options, profile)
    return {"built": True, **reports, "validation": validation}


def parse_args(argv: Sequence[str] | None = None) -> BuildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_PROFILES), default="stack_cup")
    parser.add_argument(
        "--source-root",
        "--rollout-source-root",
        dest="rollout_source_root",
        type=Path,
    )
    parser.add_argument("--human-source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--human-output", type=Path)
    parser.add_argument(
        "--source-kind",
        choices=("both", "human", "rollout"),
        default="both",
        help="build or validate both sources, or only the selected source kind",
    )
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output-only", action="store_true")
    # Accepted for launcher compatibility. Request-level dynamics uses only the
    # exact next request and is not filtered by action-time gaps.
    parser.add_argument("--max-dynamics-gap-sec", type=float)
    args = parser.parse_args(argv)
    if args.validate_only and args.validate_output_only:
        parser.error("--validate-only and --validate-output-only are mutually exclusive")
    return BuildOptions(
        task=args.task,
        rollout_source_root=args.rollout_source_root,
        human_source_root=args.human_source_root,
        output=args.output,
        human_output=args.human_output,
        compression=args.compression,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        validate_output_only=args.validate_output_only,
        source_kind=args.source_kind,
    )


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build_dataset(parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
