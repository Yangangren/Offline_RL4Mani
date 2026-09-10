#!/usr/bin/env python3
"""Build action-state rollout sources used by real-robot one-step IDQL.

Policy rollouts contain one exact two-frame camera request per executed H8
proposal and an exact pre-command robot pose / logical gripper state for every
20 Hz action.  This converter therefore keeps the request camera pair fixed
inside each proposal while replacing the low-dimensional history after
substep zero with consecutive pre-command states.  It deliberately excludes
the transition from substep seven to the next proposal because that interval
contains the variable DDIM inference pause.

The result is intentionally separate from the request-aligned chunk-IDQL
source.  ``one_step_obs[i]`` is the complete two-frame actor / critic input for
row ``i`` and is consumed directly by ``SparseOneStepSequenceDataset``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import build_episode_layout_idql_sources as core


CONVERSION_VERSION = "real_robot_episode_layout_one_step_hdf5_v2_variable_requests"


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


def _read_steps(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise core.ProposalConversionError(
                    f"{path}:{line_number} is malformed: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise core.ProposalConversionError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    if len(rows) != expected:
        raise core.ProposalConversionError(
            f"{path}: expected {expected} action records, got {len(rows)}"
        )
    return rows


def _one_step_payload(
    episode_dir: Path,
    record: core.EpisodeRecord,
    profile: core.TaskProfile,
) -> dict[str, Any]:
    # This performs the complete request-NPZ, action, label, checkpoint,
    # contract, timestamp, and digest validation before we derive any rows.
    proposal = core._policy_payload(episode_dir, record, profile)
    _, samples, actions = core._read_actions(episode_dir, record.actions)
    steps = _read_steps(
        episode_dir / "snapshots" / "steps.jsonl", record.actions
    )

    action_indices = np.asarray(
        [step.get("action_index", -1) for step in steps], dtype=np.int64
    )
    chunk_indices = np.asarray(
        [step.get("chunk", -1) for step in steps], dtype=np.int64
    )
    chunk_offsets = np.asarray(
        [step.get("substep", -1) for step in steps], dtype=np.int64
    )
    expected_indices = np.arange(record.actions, dtype=np.int64)
    expected_chunk_indices = np.asarray(
        proposal["provenance"]["policy_chunk_index"], dtype=np.int64
    )
    expected_chunk_offsets = np.asarray(
        proposal["provenance"]["policy_chunk_offset"], dtype=np.int64
    )
    if (
        not np.array_equal(action_indices, expected_indices)
        or not np.array_equal(chunk_indices, expected_chunk_indices)
        or not np.array_equal(chunk_offsets, expected_chunk_offsets)
    ):
        raise core.ProposalConversionError(
            f"{episode_dir}: action-state chunk indices differ"
        )

    step_actions = np.asarray(
        [step.get("raw_action") for step in steps], dtype=np.float32
    )
    action_pose = np.asarray(
        [sample.get("before_pose") for sample in samples], dtype=np.float64
    )
    step_pose = np.asarray(
        [step.get("pose_before") for step in steps], dtype=np.float64
    )
    logical_gripper = np.asarray(
        [step.get("gripper", {}).get("logical_state") for step in steps],
        dtype=np.float32,
    ).reshape(-1, 1)
    if (
        step_actions.shape != actions.shape
        or not np.array_equal(step_actions, actions)
        or step_pose.shape != (record.actions, 7)
        or not np.array_equal(step_pose, action_pose)
        or logical_gripper.shape != (record.actions, 1)
        or not np.isfinite(logical_gripper).all()
        or np.any(~np.isin(logical_gripper, (-1.0, 1.0)))
    ):
        raise core.ProposalConversionError(
            f"{episode_dir}: exact per-action state does not match action logs"
        )

    request_obs = proposal["request_obs"]
    request_action_counts = np.asarray(
        proposal["provenance"]["request_action_count"], dtype=np.int64
    )
    one_step_obs = {
        key: np.repeat(values, request_action_counts, axis=0)
        for key, values in request_obs.items()
    }
    pos = action_pose[:, :3].astype(np.float32)
    quat = action_pose[:, 3:].astype(np.float32)
    for index in range(record.actions):
        if chunk_offsets[index] == 0:
            # The deployed actor and critic both saw this exact digest-verified
            # request pair when the proposal was generated.
            continue
        one_step_obs["robot0_eef_pos"][index] = np.stack(
            (pos[index - 1], pos[index])
        )
        one_step_obs["robot0_eef_quat"][index] = np.stack(
            (quat[index - 1], quat[index])
        )
        one_step_obs["robot0_gripper_state"][index] = np.stack(
            (logical_gripper[index - 1], logical_gripper[index])
        )

    for key, trailing_shape in core.OBS_SHAPES.items():
        expected_dtype = np.dtype(
            np.uint8 if key in core.RGB_KEYS else np.float32
        )
        value = one_step_obs[key]
        if value.shape != (record.actions, 2, *trailing_shape):
            raise core.ProposalConversionError(
                f"{episode_dir}: one_step_obs/{key} shape differs"
            )
        if value.dtype != expected_dtype or not np.isfinite(value).all():
            raise core.ProposalConversionError(
                f"{episode_dir}: one_step_obs/{key} dtype or values differ"
            )

    boundary = np.asarray(
        [sample.get("timing_boundary") for sample in samples], dtype=np.uint8
    )
    expected_boundary = np.asarray(
        proposal["provenance"]["timing_boundary"], dtype=np.uint8
    )
    if not np.array_equal(boundary, expected_boundary):
        raise core.ProposalConversionError(
            f"{episode_dir}: timing boundaries differ from request ranges"
        )
    one_step_valid = (boundary == 0).astype(np.uint8)
    # Keep the final transition so successful rollouts retain their positive
    # terminal reward and failures retain their terminal zero reward. Its
    # bootstrap is masked by done=1, so no cross-episode state is introduced.
    one_step_valid[-1] = 1
    expected_valid = (
        record.actions - int(np.sum(request_action_counts > 0)) + 1
    )
    if int(one_step_valid.sum()) != expected_valid:
        raise core.ProposalConversionError(
            f"{episode_dir}: expected {expected_valid} valid "
            f"one-step rows, got {int(one_step_valid.sum())}"
        )

    result = dict(proposal)
    result["obs"] = {
        key: np.ascontiguousarray(values[:, -1])
        for key, values in one_step_obs.items()
    }
    result["one_step_obs"] = one_step_obs
    result["one_step_critic_valid"] = one_step_valid
    result["provenance"] = dict(proposal["provenance"])
    result["provenance"].update(
        {
            "action_state_pose_before": step_pose,
            "action_state_logical_gripper": logical_gripper,
        }
    )
    return result


def _mask_values(
    records: Sequence[core.EpisodeRecord],
    key_by_name: Mapping[str, str],
    outcomes: Mapping[str, str],
) -> dict[str, list[str]]:
    result = {
        "all": [key_by_name[record.package_name] for record in records],
        "train": [
            key_by_name[record.package_name]
            for record in records
            if record.split == "train"
        ],
        "valid": [
            key_by_name[record.package_name]
            for record in records
            if record.split == "valid"
        ],
    }
    for outcome in ("success", "failure"):
        result[outcome] = [
            key_by_name[record.package_name]
            for record in records
            if outcomes[record.package_name] == outcome
        ]
        for split in ("train", "valid"):
            result[f"{outcome}_{split}"] = [
                key_by_name[record.package_name]
                for record in records
                if outcomes[record.package_name] == outcome
                and record.split == split
            ]
    return result


def _write_rollouts(
    *,
    source_root: Path,
    records: Sequence[core.EpisodeRecord],
    source_identity: Mapping[str, Any],
    profile: core.TaskProfile,
    output: Path,
    compression: str,
) -> dict[str, Any]:
    descriptor, temporary = core._atomic_target(output)
    os.close(descriptor)
    generation_id = uuid.uuid4().hex
    key_by_name = {
        record.package_name: f"demo_{index:03d}"
        for index, record in enumerate(records, 1)
    }
    outcomes: dict[str, str] = {}
    written_counts: dict[str, int] = {}
    written_valid_counts: dict[str, int] = {}
    total = 0
    valid_total = 0
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["conversion_version"] = CONVERSION_VERSION
            target.attrs["task"] = profile.task
            target.attrs["generation_id"] = generation_id
            target.attrs["source_kind"] = "rollout_one_step_action_state"
            data = target.create_group("data")
            data.attrs["env_args"] = core._env_args(profile)
            for record in records:
                payload = _one_step_payload(
                    source_root / "episodes" / record.package_name,
                    record,
                    profile,
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
                demo.attrs["proposal_exact"] = 1
                demo.attrs["request_aligned"] = 0
                demo.attrs["one_step_aligned"] = 1
                demo.attrs["one_step_observation_semantics"] = (
                    "request camera pair held within H8; exact action-time "
                    "pre-command low-dimensional state"
                )
                for name in (
                    "actions",
                    "rewards",
                    "dones",
                    "chunk_critic_valid",
                    "one_step_critic_valid",
                ):
                    demo.create_dataset(name, data=payload[name])
                obs = demo.create_group("obs")
                aligned = demo.create_group("one_step_obs")
                for obs_key in core.OBS_KEYS:
                    core._write_array(
                        obs, obs_key, payload["obs"][obs_key], compression
                    )
                    core._write_array(
                        aligned,
                        obs_key,
                        payload["one_step_obs"][obs_key],
                        compression,
                    )
                aligned.attrs["history_order"] = json.dumps(["t-1", "t"])
                aligned.attrs["camera_source"] = (
                    "digest-verified request NPZ pair repeated within its H8 proposal"
                )
                aligned.attrs["low_dim_source"] = (
                    "substep0 exact request pair; substeps1-7 consecutive "
                    "pre-command action states"
                )
                provenance = demo.create_group("provenance")
                for name, values in payload["provenance"].items():
                    provenance.create_dataset(name, data=values)
                provenance.attrs["request_files"] = json.dumps(
                    payload["request_files"]
                )
                outcomes[record.package_name] = str(payload["outcome"])
                total += count
                episode_valid = int(payload["one_step_critic_valid"].sum())
                valid_total += episode_valid
                written_counts[record.package_name] = count
                written_valid_counts[record.package_name] = episode_valid
            data.attrs["total"] = total
            masks = target.create_group("mask")
            mask_values = _mask_values(records, key_by_name, outcomes)
            for name, values in mask_values.items():
                masks.create_dataset(name, data=np.asarray(values, dtype="S"))
            manifest = {
                "conversion_version": CONVERSION_VERSION,
                "task": profile.task,
                "generation_id": generation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "kind": "rollout_one_step_action_state",
                "source_identity": dict(source_identity),
                "episode_count": len(records),
                "transition_count": total,
                "one_step_valid_count": valid_total,
                "one_step_policy_rollouts_supported": True,
                "observation": {
                    "one_step_obs": (
                        "two-frame request cameras held within each H8 proposal; "
                        "exact per-action pre-command low-dimensional state"
                    ),
                    "excluded_transition": (
                        "last executed action to next proposal due variable "
                        "inference pause; final terminal action retained"
                    ),
                    "warning": (
                        "substeps1-7 are composite training states, not new "
                        "actor request captures"
                    ),
                },
                "masks": mask_values,
                "episodes": [
                    {
                        "demo_key": key_by_name[record.package_name],
                        "source_package_episode": record.package_name,
                        "source_run_id": record.run_id,
                        "source_group": record.source_group,
                        "split": record.split,
                        "outcome": outcomes[record.package_name],
                        "num_samples": written_counts[record.package_name],
                        "one_step_valid_count": written_valid_counts[
                            record.package_name
                        ],
                    }
                    for record in records
                ],
            }
            target.attrs[core.MANIFEST_ATTR] = json.dumps(
                manifest, sort_keys=True
            )
            target.flush()
        mode = output.stat().st_mode & 0o777 if output.exists() else 0o664
        os.chmod(temporary, mode)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "episodes": len(records),
        "transitions": total,
        "one_step_valid": valid_total,
    }


def validate_output(
    path: Path,
    *,
    profile: core.TaskProfile,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    errors = []
    total = 0
    valid_total = 0
    with h5py.File(path, "r") as source:
        if core._text(source.attrs.get("conversion_version", "")) != CONVERSION_VERSION:
            errors.append("conversion version differs")
        if core._text(source.attrs.get("task", "")) != profile.task:
            errors.append("task differs")
        if core._text(source.attrs.get("source_kind", "")) != "rollout_one_step_action_state":
            errors.append("source kind differs")
        try:
            manifest = json.loads(core._text(source.attrs[core.MANIFEST_ATTR]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise core.ProposalConversionError(
                f"{path}: invalid conversion manifest"
            ) from exc
        data = source.get("data")
        masks = source.get("mask")
        if not isinstance(data, h5py.Group) or not isinstance(masks, h5py.Group):
            raise core.ProposalConversionError(
                f"{path}: data or mask group is absent"
            )
        if len(data) != profile.expected_rollouts:
            errors.append(
                f"episode count={len(data)}, expected={profile.expected_rollouts}"
            )
        for key, demo in data.items():
            count = int(demo.attrs.get("num_samples", -1))
            total += count
            if (
                count < 1
                or (
                    profile.expected_rollout_actions is not None
                    and count != profile.expected_rollout_actions
                )
            ):
                errors.append(f"{key} action count differs")
                continue
            if bool(demo.attrs.get("request_aligned", 0)):
                errors.append(f"{key} must not be request_aligned")
            if not bool(demo.attrs.get("one_step_aligned", 0)):
                errors.append(f"{key} must be one_step_aligned")
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
            if (
                actions.shape != (count, 7)
                or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1.000001)
            ):
                errors.append(f"{key}/actions differ")
            outcome = core._text(demo.attrs.get("task_outcome", ""))
            rewards = np.zeros(count, dtype=np.float32)
            if outcome == "success":
                rewards[-1] = 1.0
            dones = np.zeros(count, dtype=np.uint8)
            dones[-1] = 1
            if not np.array_equal(demo["rewards"][:], rewards):
                errors.append(f"{key}/rewards differ")
            if not np.array_equal(demo["dones"][:], dones):
                errors.append(f"{key}/dones differ")
            boundary = np.asarray(
                demo["provenance/timing_boundary"][:], dtype=np.uint8
            )
            request_starts = np.asarray(
                demo["provenance/request_action_start"][:], dtype=np.int64
            )
            request_counts = np.asarray(
                demo["provenance/request_action_count"][:], dtype=np.uint8
            )
            expected_boundary = np.zeros(count, dtype=np.uint8)
            nonempty = request_counts > 0
            expected_boundary[
                request_starts[nonempty] + request_counts[nonempty] - 1
            ] = 1
            valid = np.asarray(
                demo["one_step_critic_valid"][:], dtype=np.uint8
            )
            expected_valid = (expected_boundary == 0).astype(np.uint8)
            expected_valid[-1] = 1
            if not np.array_equal(boundary, expected_boundary):
                errors.append(f"{key}/timing_boundary differs")
            if not np.array_equal(valid, expected_valid):
                errors.append(f"{key}/one_step_critic_valid differs")
            valid_total += int(valid.sum())
            for obs_key, trailing_shape in core.OBS_SHAPES.items():
                expected_dtype = np.dtype(
                    np.uint8 if obs_key in core.RGB_KEYS else np.float32
                )
                obs = demo[f"obs/{obs_key}"]
                aligned = demo[f"one_step_obs/{obs_key}"]
                if obs.shape != (count, *trailing_shape):
                    errors.append(f"{key}/obs/{obs_key} shape differs")
                if aligned.shape != (count, 2, *trailing_shape):
                    errors.append(f"{key}/one_step_obs/{obs_key} shape differs")
                if obs.dtype != expected_dtype or aligned.dtype != expected_dtype:
                    errors.append(f"{key}/{obs_key} dtype differs")
                for index in (0, count // 2, count - 1):
                    if not np.array_equal(obs[index], aligned[index, -1]):
                        errors.append(
                            f"{key}/{obs_key} compatibility row {index} differs"
                        )
        all_keys = set(core._decode(masks["all"][:])) if "all" in masks else set()
        train_keys = set(core._decode(masks["train"][:])) if "train" in masks else set()
        valid_keys = set(core._decode(masks["valid"][:])) if "valid" in masks else set()
        if (
            all_keys != set(data.keys())
            or train_keys & valid_keys
            or train_keys | valid_keys != all_keys
        ):
            errors.append("train/valid masks do not form an exact partition")
        expected_valid_episodes = len(
            core.ROLLOUT_VALIDATION_EPISODES[profile.task]
        )
        if len(valid_keys) != expected_valid_episodes:
            errors.append("validation mask count differs")
        expected_sets: dict[str, set[str]] = {
            "success": set(),
            "failure": set(),
            "success_train": set(),
            "success_valid": set(),
            "failure_train": set(),
            "failure_valid": set(),
        }
        for key, demo in data.items():
            outcome = core._text(demo.attrs.get("task_outcome", ""))
            split = core._text(demo.attrs.get("source_split", ""))
            if outcome not in {"success", "failure"} or split not in {
                "train",
                "valid",
            }:
                errors.append(f"{key} outcome or split differs")
                continue
            expected_sets[outcome].add(key)
            expected_sets[f"{outcome}_{split}"].add(key)
        for name, expected_values in expected_sets.items():
            actual = set(core._decode(masks[name][:])) if name in masks else set()
            if actual != expected_values:
                errors.append(f"mask/{name} membership differs")
        for outcome, count in profile.expected_outcomes.items():
            if len(expected_sets[outcome]) != count:
                errors.append(f"{outcome} outcome count differs")
        expected_total = profile.expected_rollout_transition_total
        if total != expected_total or int(data.attrs.get("total", -1)) != total:
            errors.append(
                f"transition total={total}, expected={expected_total}"
            )
        expected_valid_total = (
            profile.expected_rollout_transition_total
            - profile.expected_action_valid_request_total
            + profile.expected_rollouts
        )
        if valid_total != expected_valid_total:
            errors.append(
                f"one-step valid total={valid_total}, "
                f"expected={expected_valid_total}"
            )
        if (
            manifest.get("kind") != "rollout_one_step_action_state"
            or int(manifest.get("transition_count", -1)) != total
            or int(manifest.get("one_step_valid_count", -1)) != valid_total
            or manifest.get("one_step_policy_rollouts_supported") is not True
        ):
            errors.append("manifest totals or support flag differ")
    if errors:
        raise core.ProposalConversionError(
            f"{path} validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "path": str(path),
        "episodes": profile.expected_rollouts,
        "transitions": total,
        "one_step_valid": valid_total,
    }


def _validate_human(
    path: Path,
    profile: core.TaskProfile,
) -> dict[str, Any]:
    return core.validate_output(path, kind="human", profile=profile)


def _validate_source_identity(
    paths: Sequence[Path], identity: Mapping[str, Any]
) -> None:
    for path in paths:
        with h5py.File(path, "r") as source:
            manifest = json.loads(core._text(source.attrs[core.MANIFEST_ATTR]))
        if manifest.get("source_identity") != identity:
            raise core.ProposalConversionError(f"{path}: source identity changed")


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    try:
        profile = core.TASK_PROFILES[options.task]
    except KeyError as exc:
        raise ValueError(f"unsupported task {options.task!r}") from exc
    default_output = (
        ROOT
        / f"datasets/real_robot/{profile.task}/idql/"
        f"{profile.task}_episode_layout_v1_one_step_rollouts.hdf5"
    )
    options = BuildOptions(
        task=profile.task,
        rollout_source_root=(
            options.rollout_source_root or profile.rollout_source_root
        ).expanduser().resolve(),
        human_source_root=(
            options.human_source_root or profile.human_source_root
        ).expanduser().resolve(),
        output=(options.output or default_output).expanduser().resolve(),
        human_output=(
            options.human_output or profile.human_output
        ).expanduser().resolve(),
        compression=options.compression,
        overwrite=options.overwrite,
        validate_only=options.validate_only,
        validate_output_only=options.validate_output_only,
    )
    if options.output == options.human_output:
        raise ValueError("rollout and human outputs must differ")
    if options.validate_output_only:
        return {
            "validated": True,
            "rollout": validate_output(options.output, profile=profile),
            "human": _validate_human(options.human_output, profile),
        }

    rollout_records, rollout_identity = core.discover_episodes(
        options.rollout_source_root,
        kind="rollout",
        profile=profile,
    )
    human_records, human_identity = core.discover_episodes(
        options.human_source_root,
        kind="human",
        profile=profile,
    )
    if options.validate_only:
        reports = {
            "rollout": validate_output(options.output, profile=profile),
            "human": _validate_human(options.human_output, profile),
        }
        _validate_source_identity((options.output,), rollout_identity)
        _validate_source_identity((options.human_output,), human_identity)
        return {"validated": True, **reports}

    if options.output.exists() and not options.overwrite:
        rollout_report = validate_output(options.output, profile=profile)
    else:
        rollout_report = _write_rollouts(
            source_root=options.rollout_source_root,
            records=rollout_records,
            source_identity=rollout_identity,
            profile=profile,
            output=options.output,
            compression=options.compression,
        )
    # ``--overwrite`` is intended to refresh the derived rollout alignment.
    # The canonical human source is shared by one-step and chunk training and
    # does not depend on rollout request metadata, so never rewrite a valid
    # copy as a side effect of refreshing rollout data.
    if options.human_output.exists():
        human_report = _validate_human(options.human_output, profile)
    else:
        human_report = core._write_source(
            source_root=options.human_source_root,
            records=human_records,
            source_identity=human_identity,
            profile=profile,
            output=options.human_output,
            kind="human",
            compression=options.compression,
        )
    _validate_source_identity((options.output,), rollout_identity)
    _validate_source_identity((options.human_output,), human_identity)
    validation = {
        "rollout": validate_output(options.output, profile=profile),
        "human": _validate_human(options.human_output, profile),
    }
    return {
        "built": True,
        "rollout": rollout_report,
        "human": human_report,
        "validation": validation,
    }


def parse_args(argv: Sequence[str] | None = None) -> BuildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(core.TASK_PROFILES), default="stack_cup")
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
        "--compression", choices=("gzip", "lzf", "none"), default="gzip"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output-only", action="store_true")
    # Accepted for the shared launcher. One-step validity is defined by the
    # recorded proposal-boundary marker, not by a tunable gap threshold.
    parser.add_argument("--max-dynamics-gap-sec", type=float)
    args = parser.parse_args(argv)
    if args.validate_only and args.validate_output_only:
        parser.error(
            "--validate-only and --validate-output-only are mutually exclusive"
        )
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
    )


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build_dataset(parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
