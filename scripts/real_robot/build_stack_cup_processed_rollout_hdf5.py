#!/usr/bin/env python3
"""Convert the audited StackCup DDIM-100 rollout handoff to robomimic HDF5.

The published handoff is on a 20 Hz wall-clock grid. Diffusion inference gaps
cause many grid rows to point at the same executed action. This converter
deduplicates immutable source indices, verifies every retained proposal against
the original per-chunk run record, and causally realigns RGB at source action
timestamps. It never trains on the synthetic repeated rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import h5py
import numpy as np

from scripts.real_robot import build_stack_cup_dataset as shared
from scripts.real_robot.stack_cup_common import (
    CONVERSION_MANIFEST_ATTR,
    StackCupEpisodeRow,
    file_sha256,
    load_json,
)


DEFAULT_SOURCE = Path("/home/ryan/datasets/stack_cup/rollout")
DEFAULT_HUMAN_DATASET = ROOT / "datasets/real_robot/stack_cup/stack_cup_rgb.hdf5"
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/stack_cup/idql/"
    "stack_cup_epoch200_ddim100_20hz_rollouts.hdf5"
)
CONVERSION_VERSION = "stack_cup_epoch200_ddim100_executed_actions_v1"
EXPECTED_EPISODES = frozenset(range(1, 41))
EXPECTED_OUTCOMES = {"success": 26, "failure": 14}
EXPECTED_EXECUTED_ACTIONS = 600
SUCCESS_VALID_COUNT = 6
FAILURE_VALID_COUNT = 4
EXPECTED_CHECKPOINT_SHA256 = (
    "b1bbe2f6be8eeb1317ba270777c0b17e265464c33906134164f4d62d7b4bfa6d"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "98808969729ec13c2f385d5e7e9da680dbc245e39f7514dc58a9d8b86ea6b413"
)
EXPECTED_COLLECTION_CONTRACT_SHA256 = (
    "3d9b7a33ded0594a97ff1dd54d9f7240d0fe27a3738dd14a7506653c5f724a08"
)
EXPECTED_SERVER_IDENTITY_SHA256 = (
    "a75fcfd8f9f8e32226a1d0c7c2a9621c74136594e0d6aa2e4396dfa327c6a85e"
)
EPISODE_RE = re.compile(
    r"^episode_(?P<number>\d{3})__"
    r"(?P<run>stack_cup_dp_epoch200_ddim100_20hz_real_"
    r"(?P<day>\d{8})_(?P<clock>\d{6}))$"
)


class RolloutConversionError(ValueError):
    """The rollout handoff or converted dataset violates its fixed contract."""


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output: Path = DEFAULT_OUTPUT
    compression: str = "gzip"
    split_seed: int = 1
    image_height: int = 96
    image_width: int = 128
    max_image_age_sec: float = 0.5
    episode_limit: int | None = None
    overwrite: bool = False
    validate_only: bool = False
    validate_output_only: bool = False


@dataclass(frozen=True)
class RolloutEpisodeRow(StackCupEpisodeRow):
    @property
    def excluded(self) -> bool:
        return False


@dataclass(frozen=True)
class SourceEpisode:
    row: StackCupEpisodeRow
    outcome: str
    day: str
    checkpoint_sha256: str
    effective_ddim_steps: int

    @property
    def demo_key(self) -> str:
        return self.row.demo_key


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RolloutConversionError(f"{name} must be a JSON object")
    return value


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode(values: np.ndarray) -> list[str]:
    return [_text(value) for value in values.tolist()]


def _verify_published_checksums(source_root: Path) -> dict[str, Any]:
    checksum_path = source_root / "metadata_checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checked = 0
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RolloutConversionError(
                f"{checksum_path}:{line_number} is malformed"
            ) from exc
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RolloutConversionError(
                f"{checksum_path}:{line_number} has an invalid digest"
            )
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise RolloutConversionError(
                f"{checksum_path}:{line_number} escapes the source root"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != expected:
            raise RolloutConversionError(f"published checksum differs: {relative}")
        checked += 1
    return {
        "entries": checked,
        "manifest_sha256": file_sha256(source_root / "manifest.json"),
        "verification_sha256": file_sha256(source_root / "verification.json"),
        "checksum_file_sha256": file_sha256(checksum_path),
    }


def _runtime_value(run: Mapping[str, Any], *path: str) -> Any:
    value: Any = run
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise RolloutConversionError(
                "missing run provenance: " + ".".join(path)
            )
        value = value[key]
    return value


def _validate_policy_identity(
    episode_dir: Path,
    run_id: str,
    outcome: str,
) -> tuple[str, int]:
    run_path = episode_dir / "snapshots/run.json"
    outcome_path = episode_dir / "snapshots/outcome.json"
    collection_path = episode_dir / "snapshots/collection_manifest.json"
    run = _object(load_json(run_path), name=str(run_path))
    annotation = _object(load_json(outcome_path), name=str(outcome_path))
    collection = _object(load_json(collection_path), name=str(collection_path))

    for source, name in ((run, run_path), (annotation, outcome_path), (collection, collection_path)):
        if source.get("episode_id") != run_id:
            raise RolloutConversionError(f"{name} episode_id differs from {run_id}")
    if (
        annotation.get("task_outcome") != outcome
        or annotation.get("discarded") is not False
        or annotation.get("runner_error") is not None
        or int(annotation.get("runner_exit_code", -1)) != 0
        or annotation.get("termination_class") != "policy_rollout_completed"
    ):
        raise RolloutConversionError(f"{outcome_path} is not a usable {outcome} rollout")
    if (
        run.get("status") != "PASS"
        or run.get("completion_reason") != "max_actions"
        or int(run.get("actions_completed", -1)) != EXPECTED_EXECUTED_ACTIONS
    ):
        raise RolloutConversionError(f"{run_path} did not complete 600 actions")

    arguments = _object(run.get("arguments"), name=f"{run_path}.arguments")
    exact_arguments = {
        "execute": True,
        "max_actions": EXPECTED_EXECUTED_ACTIONS,
        "clamp_mode": "reject",
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    for key, expected in exact_arguments.items():
        if arguments.get(key) != expected:
            raise RolloutConversionError(
                f"{run_path} argument {key}={arguments.get(key)!r}, expected {expected!r}"
            )
    for key, expected in (("pos_scale", 0.012), ("rot_scale", 0.036)):
        if not np.isclose(float(arguments.get(key, np.nan)), expected, atol=1e-12):
            raise RolloutConversionError(f"{run_path} argument {key} differs")
    if arguments.get("expected_dataset_manifests") != [
        EXPECTED_DATASET_MANIFEST_SHA256
    ]:
        raise RolloutConversionError(f"{run_path} dataset identity differs")

    if (
        collection.get("schema") != "real4d.dp_rollout_collection.v1"
        or collection.get("policy")
        != "robomimic_stack_cup_dp_epoch200_ddim100_20hz"
        or collection.get("contract_sha256")
        != EXPECTED_COLLECTION_CONTRACT_SHA256
    ):
        raise RolloutConversionError(f"{collection_path} policy contract differs")
    checkpoint_sha = _runtime_value(
        run, "server_health", "identity", "checkpoint", "sha256"
    )
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RolloutConversionError(f"{run_path} loaded checkpoint differs")
    if run.get("server_identity_sha256") != EXPECTED_SERVER_IDENTITY_SHA256:
        raise RolloutConversionError(f"{run_path} server identity differs")

    checkpoint_contract = _runtime_value(
        run, "server_health", "identity", "checkpoint_contract"
    )
    if checkpoint_contract.get("dataset_manifest_sha256") != [
        EXPECTED_DATASET_MANIFEST_SHA256
    ]:
        raise RolloutConversionError(f"{run_path} checkpoint dataset differs")
    sampler = _runtime_value(run, "server_health", "identity", "sampler")
    if (
        sampler.get("kind") != "ddim"
        or int(sampler.get("checkpoint_num_inference_steps", -1)) != 10
        or int(sampler.get("num_inference_steps", -1)) != 100
    ):
        raise RolloutConversionError(f"{run_path} effective DDIM override differs")
    runtime = _object(run.get("runtime_contract"), name=f"{run_path}.runtime_contract")
    if (
        not np.isclose(float(runtime.get("control_hz", -1)), 20.0)
        or not np.isclose(float(runtime.get("logical_camera_hz", -1)), 5.0)
        or runtime.get("horizons", {}).get("action") != 8
        or runtime.get("normalization", {}).get("actions") != "identity"
    ):
        raise RolloutConversionError(f"{run_path} model runtime contract differs")
    return str(checkpoint_sha), int(sampler["num_inference_steps"])


def discover_episodes(source_root: Path) -> list[SourceEpisode]:
    source_root = source_root.expanduser().resolve()
    manifest_path = source_root / "manifest.json"
    verification_path = source_root / "verification.json"
    manifest = _object(load_json(manifest_path), name=str(manifest_path))
    verification = _object(load_json(verification_path), name=str(verification_path))
    raw_episodes = manifest.get("episodes")
    verified = verification.get("episodes")
    if (
        not isinstance(raw_episodes, list)
        or int(manifest.get("episode_count", -1)) != len(raw_episodes)
        or not isinstance(verified, list)
        or int(verification.get("episode_count", -1)) != len(verified)
    ):
        raise RolloutConversionError("source manifest or verification inventory differs")
    verification_by_number = {
        int(item["episode_number"]): item
        for item in verified
        if isinstance(item, dict)
    }

    episodes_root = source_root / "episodes"
    actual = {path.name for path in episodes_root.iterdir() if path.is_dir()}
    listed = {
        str(item.get("episode"))
        for item in raw_episodes
        if isinstance(item, dict)
    }
    if actual != listed:
        raise RolloutConversionError(
            f"episode inventory differs: missing={sorted(listed-actual)}, "
            f"unlisted={sorted(actual-listed)}"
        )

    result: list[SourceEpisode] = []
    seen: set[int] = set()
    for item in raw_episodes:
        if not isinstance(item, dict):
            raise RolloutConversionError("manifest episode row must be an object")
        directory_name = str(item.get("episode", ""))
        match = EPISODE_RE.fullmatch(directory_name)
        if match is None:
            raise RolloutConversionError(f"unexpected episode directory: {directory_name}")
        number = int(match.group("number"))
        if number in seen:
            raise RolloutConversionError(f"duplicate episode number {number}")
        seen.add(number)
        episode_dir = episodes_root / directory_name
        run_id = match.group("run")
        contract = _object(load_json(episode_dir / "contract.json"), name="contract")
        qa = _object(load_json(episode_dir / "qa.json"), name="qa")
        windows = _object(load_json(episode_dir / "windows.json"), name="windows")
        annotation = _object(
            load_json(episode_dir / "snapshots/outcome.json"), name="outcome"
        )
        outcome = str(annotation.get("task_outcome", ""))
        if outcome not in {"success", "failure"}:
            raise RolloutConversionError(f"{episode_dir} has invalid outcome")
        if contract.get("run_id") != run_id or qa.get("run_id") != run_id:
            raise RolloutConversionError(f"{episode_dir} run IDs differ")
        if (
            not np.isclose(float(contract.get("actions", {}).get("hz", -1)), 20.0)
            or int(contract.get("actions", {}).get("num_actions", -1))
            != int(item.get("actions", -1))
            or not np.isclose(float(contract.get("video", {}).get("hz", -1)), 5.0)
            or int(contract.get("video", {}).get("num_frames", -1))
            != int(item.get("frames", -1))
        ):
            raise RolloutConversionError(f"{episode_dir} processed rate/count differs")
        if (
            qa.get("status") != "WARN"
            or not (episode_dir / "MODEL_WINDOW_WARN").is_file()
            or (episode_dir / "MODEL_WINDOW_READY").exists()
            or int(windows.get("valid_windows", -1)) != 0
        ):
            raise RolloutConversionError(
                f"{episode_dir} no longer matches the audited density-warning profile"
            )
        verification_row = verification_by_number.get(number)
        if (
            verification_row is None
            or verification_row.get("status") != "PASS"
            or verification_row.get("episode") != directory_name
        ):
            raise RolloutConversionError(f"{episode_dir} source verification differs")
        checkpoint_sha, effective_steps = _validate_policy_identity(
            episode_dir, run_id, outcome
        )
        row = RolloutEpisodeRow(
            episode_number=number,
            run_id=run_id,
            directory=f"episodes/{directory_name}",
            qa_status="WARN",
            model_window_ready=False,
            invalid_windows=int(windows.get("invalid_windows", -1)),
            manifest_actions=int(item["actions"]),
            manifest_frames=int(item["frames"]),
            removed_actions=int(item.get("removed_actions", 0)),
            removed_frames=int(item.get("removed_frames", 0)),
        )
        result.append(
            SourceEpisode(
                row=row,
                outcome=outcome,
                day=match.group("day"),
                checkpoint_sha256=checkpoint_sha,
                effective_ddim_steps=effective_steps,
            )
        )
    if seen != EXPECTED_EPISODES:
        raise RolloutConversionError(
            f"expected episodes 1..40, got {sorted(seen)}"
        )
    counts = {
        outcome: sum(episode.outcome == outcome for episode in result)
        for outcome in ("success", "failure")
    }
    if counts != EXPECTED_OUTCOMES:
        raise RolloutConversionError(
            f"outcome counts differ: expected {EXPECTED_OUTCOMES}, got {counts}"
        )
    return sorted(result, key=lambda episode: episode.row.episode_number)


def _ranked_validation(
    episodes: Sequence[SourceEpisode],
    *,
    outcome: str,
    count: int,
    seed: int,
) -> set[str]:
    candidates = [episode for episode in episodes if episode.outcome == outcome]
    ranked = sorted(
        candidates,
        key=lambda episode: hashlib.sha256(
            f"{seed}:{outcome}:{episode.day}:{episode.row.run_id}".encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise RolloutConversionError(f"not enough {outcome} validation episodes")
    return {episode.demo_key for episode in ranked[:count]}


def split_masks(
    episodes: Sequence[SourceEpisode], seed: int
) -> dict[str, list[str]]:
    success_valid = _ranked_validation(
        episodes, outcome="success", count=SUCCESS_VALID_COUNT, seed=seed
    )
    failure_valid = _ranked_validation(
        episodes, outcome="failure", count=FAILURE_VALID_COUNT, seed=seed
    )
    success = [episode.demo_key for episode in episodes if episode.outcome == "success"]
    failure = [episode.demo_key for episode in episodes if episode.outcome == "failure"]
    masks = {
        "all": [episode.demo_key for episode in episodes],
        "success": success,
        "failure": failure,
        "success_train": [key for key in success if key not in success_valid],
        "success_valid": [key for key in success if key in success_valid],
        "failure_train": [key for key in failure if key not in failure_valid],
        "failure_valid": [key for key in failure if key in failure_valid],
    }
    masks["train"] = sorted(masks["success_train"] + masks["failure_train"])
    masks["valid"] = sorted(masks["success_valid"] + masks["failure_valid"])
    return masks


def _env_args() -> str:
    if not DEFAULT_HUMAN_DATASET.is_file():
        raise FileNotFoundError(DEFAULT_HUMAN_DATASET)
    with h5py.File(DEFAULT_HUMAN_DATASET, "r") as dataset:
        value = _text(dataset["data"].attrs["env_args"])
    parsed = json.loads(value)
    if parsed.get("env_name") != "StackCupReal-v0":
        raise RolloutConversionError("human dataset environment identity differs")
    return json.dumps(parsed, sort_keys=True)


def _load_payloads(
    source_root: Path,
    episodes: Sequence[SourceEpisode],
    *,
    max_image_age_sec: float,
) -> list[shared.EpisodePayload]:
    return [
        shared.load_episode_payload(
            source_root,
            episode.row,
            max_image_age_sec=max_image_age_sec,
            policy_rollout=True,
        )
        for episode in episodes
    ]


def _source_identity(
    source_root: Path,
    checksum_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "root": str(source_root),
        "published_checksum_validation": dict(checksum_report),
        "processing_complete_sha256": file_sha256(source_root / "PROCESSING_COMPLETE"),
        "dataset_ready_sha256": file_sha256(source_root / "DATASET_READY"),
    }


def _manifest(
    options: BuildOptions,
    episodes: Sequence[SourceEpisode],
    payloads: Sequence[shared.EpisodePayload],
    masks: Mapping[str, list[str]],
    source_identity: Mapping[str, Any],
    generation_id: str,
) -> dict[str, Any]:
    details = []
    for episode, payload in zip(episodes, payloads):
        source_indices = payload.action_source_indices
        missing = int(
            np.sum(np.maximum(np.diff(source_indices) - 1, 0))
            if source_indices.size > 1
            else 0
        )
        details.append(
            {
                "demo_key": episode.demo_key,
                "episode_number": episode.row.episode_number,
                "run_id": episode.row.run_id,
                "outcome": episode.outcome,
                "processed_wall_clock_rows": episode.row.manifest_actions,
                "retained_executed_actions": payload.num_samples,
                "dropped_precausal_actions": payload.dropped_prefix_actions,
                "missing_internal_source_indices": missing,
                "max_image_age_sec": float(np.max(payload.image_ages)),
            }
        )
    return {
        "conversion_version": CONVERSION_VERSION,
        "generation_id": generation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": dict(source_identity),
        "policy": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_ddim_steps": 10,
            "effective_rollout_ddim_steps": 100,
            "executed_actions_per_source_rollout": EXPECTED_EXECUTED_ACTIONS,
        },
        "action": {
            "stored": "normalized continuous Diffusion Policy proposal in [-1,1]",
            "selection": (
                "one row per immutable source_index, selecting the processed row "
                "with minimum absolute source-time offset"
            ),
            "wall_clock_repeat_rows_are_training_data": False,
            "motion_scale": [0.012, 0.012, 0.012, 0.036, 0.036, 0.036],
            "gripper": "continuous absolute policy prediction",
        },
        "observation": {
            "eef_pose": "source precommand pose",
            "gripper": (
                "logical pre-action state reconstructed from the exact run action "
                "sequence and recorded close/open thresholds"
            ),
            "rgb": (
                "latest paired main/wrist capture not later than the original "
                "source action timestamp"
            ),
            "max_image_age_sec": options.max_image_age_sec,
            "shape": [options.image_height, options.image_width, 3],
        },
        "reward": {
            "success": "one terminal 1; all earlier transitions 0",
            "failure": "all transitions 0",
            "done": "one final 1 for both outcomes",
        },
        "split": {
            "seed": options.split_seed,
            "method": "outcome-stratified deterministic SHA256 ranking",
            **masks,
        },
        "episodes": details,
    }


def _write_dataset(
    options: BuildOptions,
    episodes: Sequence[SourceEpisode],
    payloads: Sequence[shared.EpisodePayload],
    masks: Mapping[str, list[str]],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    output = options.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    generation_id = uuid.uuid4().hex
    manifest = _manifest(
        options, episodes, payloads, masks, source_identity, generation_id
    )
    compression = options.compression
    try:
        with h5py.File(temporary, "w") as dataset:
            dataset.attrs["conversion_version"] = CONVERSION_VERSION
            dataset.attrs["generation_id"] = generation_id
            dataset.attrs[CONVERSION_MANIFEST_ATTR] = json.dumps(
                manifest, sort_keys=True
            )
            data = dataset.create_group("data")
            data.attrs["env_args"] = _env_args()
            total = 0
            for episode, payload in zip(episodes, payloads):
                demo = data.create_group(episode.demo_key)
                demo.attrs["num_samples"] = payload.num_samples
                demo.attrs["source_episode_number"] = episode.row.episode_number
                demo.attrs["source_run_id"] = episode.row.run_id
                demo.attrs["source_directory"] = episode.row.directory
                demo.attrs["task_outcome"] = episode.outcome
                demo.attrs["outcome"] = int(episode.outcome == "success")
                demo.attrs["checkpoint_sha256"] = episode.checkpoint_sha256
                demo.attrs["effective_ddim_steps"] = episode.effective_ddim_steps
                demo.attrs["processed_wall_clock_rows"] = episode.row.manifest_actions
                source_gaps = int(
                    np.sum(np.maximum(np.diff(payload.action_source_indices) - 1, 0))
                    if payload.num_samples > 1
                    else 0
                )
                demo.attrs["missing_internal_source_indices"] = source_gaps
                demo.attrs["dropped_prefix_actions"] = payload.dropped_prefix_actions
                demo.attrs["max_image_age_sec"] = float(np.max(payload.image_ages))
                demo.create_dataset("actions", data=payload.actions.astype(np.float32))
                rewards = np.zeros(payload.num_samples, dtype=np.float32)
                if episode.outcome == "success":
                    rewards[-1] = 1.0
                dones = np.zeros(payload.num_samples, dtype=np.uint8)
                dones[-1] = 1
                demo.create_dataset("rewards", data=rewards)
                demo.create_dataset("dones", data=dones)

                obs = demo.create_group("obs")
                obs.create_dataset(
                    "robot0_eef_pos", data=payload.eef_poses[:, :3].astype(np.float32)
                )
                obs.create_dataset(
                    "robot0_eef_quat", data=payload.eef_poses[:, 3:7].astype(np.float32)
                )
                obs.create_dataset(
                    "robot0_gripper_state",
                    data=payload.gripper_observations[:, None].astype(np.float32),
                )
                shared._write_images(
                    obs,
                    payload,
                    image_height=options.image_height,
                    image_width=options.image_width,
                    compression=compression,
                )

                provenance = demo.create_group("provenance")
                provenance.create_dataset(
                    "policy_gripper_prediction",
                    data=payload.raw_gripper_events.astype(np.float32),
                )
                provenance.create_dataset(
                    "processed_action_array_index",
                    data=payload.source_action_array_indices,
                )
                provenance.create_dataset(
                    "source_action_index", data=payload.action_source_indices
                )
                provenance.create_dataset(
                    "processed_target_time", data=payload.action_target_times
                )
                provenance.create_dataset(
                    "source_action_time", data=payload.action_source_times
                )
                provenance.create_dataset(
                    "processed_source_offset_ms", data=payload.action_offsets_ms
                )
                provenance.create_dataset(
                    "selected_frame_position", data=payload.selected_frame_positions
                )
                provenance.create_dataset(
                    "selected_frame_index", data=payload.selected_frame_indices
                )
                provenance.create_dataset(
                    "selected_pair_capture_time",
                    data=payload.selected_pair_capture_times,
                )
                provenance.create_dataset("image_age_sec", data=payload.image_ages)
                total += payload.num_samples
            data.attrs["total"] = total
            mask = dataset.create_group("mask")
            for name, keys in masks.items():
                mask.create_dataset(name, data=np.asarray(keys, dtype="S"))
            dataset.flush()
        mode = output.stat().st_mode & 0o777 if output.exists() else 0o664
        os.chmod(temporary, mode)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "episodes": len(episodes),
        "samples": sum(payload.num_samples for payload in payloads),
        "processed_wall_clock_rows": sum(
            episode.row.manifest_actions for episode in episodes
        ),
        "dropped_precausal_actions": sum(
            payload.dropped_prefix_actions for payload in payloads
        ),
        "missing_internal_source_indices": sum(
            int(np.sum(np.maximum(np.diff(payload.action_source_indices) - 1, 0)))
            for payload in payloads
        ),
        "max_image_age_sec": max(
            float(np.max(payload.image_ages)) for payload in payloads
        ),
        "mask_counts": {name: len(keys) for name, keys in masks.items()},
    }


def _validate_internal(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    outcomes = {"success": 0, "failure": 0}
    samples = 0
    with h5py.File(path, "r") as dataset:
        if _text(dataset.attrs.get("conversion_version", "")) != CONVERSION_VERSION:
            errors.append("conversion version differs")
        try:
            manifest = json.loads(
                _text(dataset.attrs[CONVERSION_MANIFEST_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RolloutConversionError(f"{path} has invalid manifest: {exc}") from exc
        if manifest.get("policy", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            errors.append("checkpoint identity differs")
        data = dataset.get("data")
        masks = dataset.get("mask")
        if not isinstance(data, h5py.Group) or not isinstance(masks, h5py.Group):
            raise RolloutConversionError(f"{path} is missing data or mask")
        keys = sorted(data.keys())
        if len(keys) != 40:
            errors.append(f"expected 40 demos, got {len(keys)}")
        for key in keys:
            demo = data[key]
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
            count = int(actions.shape[0])
            if (
                actions.ndim != 2
                or actions.shape[1] != 7
                or count < 1
                or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1.000001)
            ):
                errors.append(f"{key} actions are invalid")
                continue
            if int(demo.attrs.get("num_samples", -1)) != count:
                errors.append(f"{key} num_samples differs")
            outcome = _text(demo.attrs.get("task_outcome", ""))
            if outcome not in outcomes:
                errors.append(f"{key} outcome is invalid")
                continue
            outcomes[outcome] += 1
            rewards = np.asarray(demo["rewards"][:], dtype=np.float32)
            expected_rewards = np.zeros(count, dtype=np.float32)
            if outcome == "success":
                expected_rewards[-1] = 1.0
            expected_dones = np.zeros(count, dtype=np.uint8)
            expected_dones[-1] = 1
            if not np.array_equal(rewards, expected_rewards):
                errors.append(f"{key} rewards differ")
            if not np.array_equal(demo["dones"][:], expected_dones):
                errors.append(f"{key} dones differ")
            for obs_key, trailing in (
                ("main_image", (96, 128, 3)),
                ("wrist_image", (96, 128, 3)),
                ("robot0_eef_pos", (3,)),
                ("robot0_eef_quat", (4,)),
                ("robot0_gripper_state", (1,)),
            ):
                child = demo[f"obs/{obs_key}"]
                if child.shape != (count, *trailing):
                    errors.append(f"{key}/obs/{obs_key} shape differs")
            if np.any(~np.isin(demo["obs/robot0_gripper_state"][:], [-1.0, 1.0])):
                errors.append(f"{key} logical gripper observation differs")
            source_indices = np.asarray(
                demo["provenance/source_action_index"][:], dtype=np.int64
            )
            if source_indices.shape != (count,) or np.any(np.diff(source_indices) <= 0):
                errors.append(f"{key} source action indices are not increasing")
            if np.any(demo["provenance/image_age_sec"][:] < -1e-6) or np.any(
                demo["provenance/image_age_sec"][:] > 0.500001
            ):
                errors.append(f"{key} causal RGB ages differ")
            samples += count
        if outcomes != EXPECTED_OUTCOMES:
            errors.append(f"outcomes={outcomes}, expected={EXPECTED_OUTCOMES}")
        if int(data.attrs.get("total", -1)) != samples:
            errors.append("data total differs")
        split = manifest.get("split", {})
        for name in (
            "all",
            "train",
            "valid",
            "success",
            "failure",
            "success_train",
            "success_valid",
            "failure_train",
            "failure_valid",
        ):
            if name not in masks or _decode(masks[name][:]) != split.get(name):
                errors.append(f"mask/{name} differs from manifest")
    if errors:
        raise RolloutConversionError(
            f"{path} validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "validated": True,
        "output": str(path),
        "episodes": sum(outcomes.values()),
        "samples": samples,
        "outcomes": outcomes,
    }


def validate_source_backed(options: BuildOptions) -> dict[str, Any]:
    source_root = options.source_root.expanduser().resolve()
    output = options.output.expanduser().resolve()
    internal = _validate_internal(output)
    checksum_report = _verify_published_checksums(source_root)
    episodes = discover_episodes(source_root)
    payloads = _load_payloads(
        source_root, episodes, max_image_age_sec=options.max_image_age_sec
    )
    masks = split_masks(episodes, options.split_seed)
    with h5py.File(output, "r") as dataset:
        manifest = json.loads(_text(dataset.attrs[CONVERSION_MANIFEST_ATTR]))
        if manifest.get("source_identity") != _source_identity(
            source_root, checksum_report
        ):
            raise RolloutConversionError(f"{output} source identity differs")
        if manifest.get("split", {}).get("seed") != options.split_seed:
            raise RolloutConversionError(f"{output} split seed differs")
        for episode, payload in zip(episodes, payloads):
            demo = dataset[f"data/{episode.demo_key}"]
            expected_rewards = np.zeros(payload.num_samples, dtype=np.float32)
            if episode.outcome == "success":
                expected_rewards[-1] = 1.0
            exact = {
                "actions": payload.actions.astype(np.float32),
                "rewards": expected_rewards,
                "obs/robot0_eef_pos": payload.eef_poses[:, :3].astype(np.float32),
                "obs/robot0_eef_quat": payload.eef_poses[:, 3:7].astype(np.float32),
                "obs/robot0_gripper_state": payload.gripper_observations[
                    :, None
                ].astype(np.float32),
                "provenance/source_action_index": payload.action_source_indices,
                "provenance/processed_action_array_index": (
                    payload.source_action_array_indices
                ),
                "provenance/source_action_time": payload.action_source_times,
                "provenance/selected_frame_position": (
                    payload.selected_frame_positions
                ),
                "provenance/image_age_sec": payload.image_ages,
            }
            for name, expected in exact.items():
                if not np.array_equal(demo[name][:], expected):
                    raise RolloutConversionError(
                        f"{output}:data/{episode.demo_key}/{name} differs from source"
                    )
            for row_index in sorted({0, payload.num_samples // 2, payload.num_samples - 1}):
                frame_position = int(payload.selected_frame_positions[row_index])
                expected_size = payload.frame_source_sizes[frame_position]
                for obs_key, source_path in (
                    ("main_image", payload.main_paths[frame_position]),
                    ("wrist_image", payload.wrist_paths[frame_position]),
                ):
                    expected_image = shared._load_resized_rgb(
                        source_path,
                        expected_size=expected_size,
                        output_size=(options.image_width, options.image_height),
                    )
                    if not np.array_equal(
                        demo[f"obs/{obs_key}"][row_index], expected_image
                    ):
                        raise RolloutConversionError(
                            f"{output}:data/{episode.demo_key}/{obs_key} differs"
                        )
        for name, expected in masks.items():
            if _decode(dataset[f"mask/{name}"][:]) != expected:
                raise RolloutConversionError(f"{output}:mask/{name} differs")
    return {
        **internal,
        "source_checked": True,
        "published_checksums": checksum_report,
        "processed_wall_clock_rows": sum(
            episode.row.manifest_actions for episode in episodes
        ),
        "dropped_precausal_actions": sum(
            payload.dropped_prefix_actions for payload in payloads
        ),
        "missing_internal_source_indices": sum(
            int(np.sum(np.maximum(np.diff(payload.action_source_indices) - 1, 0)))
            for payload in payloads
        ),
        "max_image_age_sec": max(
            float(np.max(payload.image_ages)) for payload in payloads
        ),
        "mask_counts": {name: len(values) for name, values in masks.items()},
    }


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    source_root = options.source_root.expanduser().resolve()
    output = options.output.expanduser().resolve()
    if options.validate_output_only:
        return _validate_internal(output)
    if options.max_image_age_sec <= 0.0 or options.max_image_age_sec > 0.5:
        raise RolloutConversionError("max image age must be in (0, 0.5]")
    if options.validate_only:
        return validate_source_backed(options)
    if output.exists() and not options.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    checksum_report = _verify_published_checksums(source_root)
    episodes = discover_episodes(source_root)
    if options.episode_limit is not None:
        if options.episode_limit <= 0:
            raise RolloutConversionError("episode limit must be positive")
        episodes = episodes[: options.episode_limit]
    masks = split_masks(episodes, options.split_seed) if len(episodes) == 40 else {
        "all": [episode.demo_key for episode in episodes],
        "train": [episode.demo_key for episode in episodes],
        "valid": [],
        "success": [
            episode.demo_key for episode in episodes if episode.outcome == "success"
        ],
        "failure": [
            episode.demo_key for episode in episodes if episode.outcome == "failure"
        ],
        "success_train": [
            episode.demo_key for episode in episodes if episode.outcome == "success"
        ],
        "success_valid": [],
        "failure_train": [
            episode.demo_key for episode in episodes if episode.outcome == "failure"
        ],
        "failure_valid": [],
    }
    payloads = _load_payloads(
        source_root, episodes, max_image_age_sec=options.max_image_age_sec
    )
    report = _write_dataset(
        options,
        episodes,
        payloads,
        masks,
        _source_identity(source_root, checksum_report),
    )
    if len(episodes) == 40:
        report["validation"] = validate_source_backed(options)
    return report


def parse_args(argv: Sequence[str] | None = None) -> BuildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--max-image-age-sec", type=float, default=0.5)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only and args.validate_output_only:
        parser.error("choose at most one validation mode")
    return BuildOptions(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = build_dataset(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

