#!/usr/bin/env python3
"""Build the 20 Hz pick-cup mixed dataset used by RGB chunk IDQL.

The human demonstrations already have an episode-level train / validation
split, and the rollout converter creates outcome-stratified train / validation
masks. This builder emits either a fitting or a held-out mixed dataset, while
auditing that the requested source masks do not overlap the opposite split. It
does not infer an earlier success time from robot state or gripper events: every
successful source episode must already contain exactly one positive task reward
at its final recorded transition, while failed rollouts must contain none.

Large source arrays remain zero-copy HDF5 external links.  ``next_obs`` is a
virtual one-step shift of ``obs`` with the final observation repeated.  Source
identities include every input shard and hashes of all known conversion
manifest attributes, so ``--validate-only`` detects stale in-place sources.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_DATASETS = (
    ROOT / "datasets/real_robot/pick_cup/round1_rgb.hdf5",
    ROOT / "datasets/real_robot/pick_cup/round2_rgb.hdf5",
)
DEFAULT_ROLLOUT_DATASET = (
    ROOT
    / "datasets/real_robot/pick_cup/idql/pick_cup_epoch200_20hz_rollouts.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/pick_cup/idql/"
    "pick_cup_chunk_idql_65demo_23success_11failure_terminal_success.hdf5"
)

TASK = "pick_cup"
REWARD_MODE = "terminal_success"
# Keep this byte-for-byte compatible with train_rgb_dp_{,chunk_}idql.py.
REWARD_DEFINITION = (
    "successful_episode: truncate_at_first_source_task_reward>0.5, "
    "reward=1_and_done=1_there; failed_episode: reward=0, "
    "done=1_at_source_end"
)
ACTOR_CONDITION_DEFINITIONS = {
    "human_only": "human_demo=1; success_rollout=0; failure_rollout=0",
    "human_success": "human_demo=1; success_rollout=1; failure_rollout=0",
}
SOURCE_LABELS = (
    "expert",
    "non_expert_success",
    "non_expert_failure",
)
SUCCESS_LABELS = frozenset(("expert", "non_expert_success"))
MANIFEST_ATTRS = (
    "real_robot_conversion_manifest",
    "_robomimic_conversion_manifest",
    "conversion_manifest",
)
SOURCE_IDENTITY_VERSION = 1
BUILDER_VERSION = "pick_cup_chunk_idql_mixed_v1"
DEFAULT_HUMAN_COUNT = 65
DEFAULT_EXPECTED_HUMAN_TRANSITIONS = 27_499
DEFAULT_SUCCESS_COUNT = 23
DEFAULT_FAILURE_COUNT = 11
DEFAULT_HUMAN_DATASETS_HELP = (
    "Repeat for each human HDF5 shard (defaults to round 1 and round 2)."
)
REQUIRED_OBS_SHAPES = {
    "main_image": (None, None, 3),
    "wrist_image": (None, None, 3),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_state": (1,),
}
RESERVED_EPISODE_KEYS = {
    "actions",
    "obs",
    "next_obs",
    "rewards",
    "task_rewards",
    "dones",
    "source_is_expert",
    "actor_condition",
}


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode(values: Iterable[Any]) -> list[str]:
    return [_text(value) for value in values]


def _demo_sort_key(key: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", key)
    return (int(match.group(1)) if match else 2**63 - 1, key)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _manifest_identity(value: Any) -> dict[str, Any]:
    encoded = _text(value).encode("utf-8")
    return {
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def source_file_identity(path: Path) -> dict[str, Any]:
    """Fingerprint one complete HDF5 source without hashing its RGB payload."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    manifests: dict[str, dict[str, Any]] = {}
    with h5py.File(resolved, "r") as source:
        for name in MANIFEST_ATTRS:
            if name in source.attrs:
                manifests[name] = _manifest_identity(source.attrs[name])
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"source changed while being identified: {resolved}")
    return {
        "path": str(resolved),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "conversion_manifests": manifests,
    }


def _read_mask(handle: h5py.File, name: str) -> list[str]:
    path = f"mask/{name}"
    if path not in handle:
        available = sorted(handle.get("mask", {}).keys())
        raise KeyError(
            f"{Path(handle.filename).resolve()} has no {path}; available={available}"
        )
    keys = _decode(handle[path][:])
    if len(keys) != len(set(keys)):
        raise ValueError(f"{handle.filename}:{path} contains duplicate episodes")
    missing = sorted(set(keys).difference(handle["data"].keys()), key=_demo_sort_key)
    if missing:
        raise ValueError(f"{handle.filename}:{path} references missing demos {missing}")
    return sorted(keys, key=_demo_sort_key)


def _select(
    candidates: list[tuple[Path, str]],
    count: int,
    rng: np.random.Generator,
    *,
    description: str,
) -> list[tuple[Path, str]]:
    if count < 0:
        return candidates
    if count > len(candidates):
        raise ValueError(
            f"requested {count} {description} episodes from only {len(candidates)}"
        )
    if count == len(candidates):
        return candidates
    indices = sorted(int(index) for index in rng.choice(len(candidates), count, replace=False))
    return [candidates[index] for index in indices]


def _normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize CLI and small programmatic Namespaces in one place."""

    values = vars(args).copy()
    human = values.get("human_datasets", values.get("human_dataset"))
    if human is None:
        human = DEFAULT_HUMAN_DATASETS
    if isinstance(human, (str, Path)):
        human = [human]
    values["human_datasets"] = tuple(
        Path(path).expanduser().resolve() for path in human
    )
    values["rollout_dataset"] = Path(
        values.get("rollout_dataset", DEFAULT_ROLLOUT_DATASET)
    ).expanduser().resolve()
    values["output"] = Path(values.get("output", DEFAULT_OUTPUT)).expanduser().resolve()
    defaults = {
        "task": TASK,
        "reward_mode": REWARD_MODE,
        "selection_role": "train",
        "human_mask": "train",
        "human_count": DEFAULT_HUMAN_COUNT,
        "expected_human_transitions": DEFAULT_EXPECTED_HUMAN_TRANSITIONS,
        "success_mask": "success_train",
        "success_count": DEFAULT_SUCCESS_COUNT,
        "failure_mask": "failure_train",
        "failure_count": DEFAULT_FAILURE_COUNT,
        "actor_condition_mode": "human_only",
        "seed": 0,
        "overwrite": False,
        "validate_only": False,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    normalized = argparse.Namespace(**values)
    if not normalized.human_datasets:
        raise ValueError("at least one --human-dataset is required")
    if len(set(normalized.human_datasets)) != len(normalized.human_datasets):
        raise ValueError("human dataset paths must be unique")
    if normalized.rollout_dataset in normalized.human_datasets:
        raise ValueError("rollout dataset must differ from every human dataset")
    if str(normalized.task) != TASK:
        raise ValueError(f"this builder only supports task={TASK!r}")
    if str(normalized.reward_mode) != REWARD_MODE:
        raise ValueError(f"this builder only supports reward_mode={REWARD_MODE!r}")
    if normalized.actor_condition_mode not in ACTOR_CONDITION_DEFINITIONS:
        raise ValueError(
            f"unsupported actor condition mode {normalized.actor_condition_mode!r}"
        )
    return normalized


def resolve_selection(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, Path, str]], str]:
    """Return canonical records and the shared robomimic ``env_args``."""

    args = _normalized_args(args)
    rng = np.random.default_rng(int(args.seed))
    human_candidates: list[tuple[Path, str]] = []
    env_args: str | None = None
    for path in args.human_datasets:
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as source:
            selected = _read_mask(source, str(args.human_mask))
            opposite_mask = (
                "valid" if args.selection_role == "train" else "train"
            )
            opposite_path = f"mask/{opposite_mask}"
            if opposite_path not in source:
                raise KeyError(
                    f"{path} is missing {opposite_path}; the real-robot "
                    f"{args.selection_role} split must be auditable"
                )
            opposite = set(_read_mask(source, opposite_mask))
            overlap = sorted(
                set(selected).intersection(opposite), key=_demo_sort_key
            )
            if overlap:
                raise ValueError(
                    f"{path} requested human {args.selection_role} mask "
                    f"overlaps mask/{opposite_mask}: {overlap}"
                )
            human_candidates.extend((path, key) for key in selected)
            candidate_env = _text(source["data"].attrs.get("env_args", ""))
            if not candidate_env:
                raise ValueError(f"{path}:data is missing env_args")
            if env_args is None:
                env_args = candidate_env
            elif json.loads(candidate_env) != json.loads(env_args):
                raise ValueError(f"human source env_args mismatch: {path}")

    humans = _select(
        human_candidates,
        int(args.human_count),
        rng,
        description="human",
    )
    with h5py.File(args.rollout_dataset, "r") as source:
        successes = [(args.rollout_dataset, key) for key in _read_mask(source, args.success_mask)]
        failures = [(args.rollout_dataset, key) for key in _read_mask(source, args.failure_mask)]
        opposite_suffix = (
            "valid" if args.selection_role == "train" else "train"
        )
        opposite_keys: set[str] = set()
        for opposite_mask in (
            f"success_{opposite_suffix}",
            f"failure_{opposite_suffix}",
        ):
            if f"mask/{opposite_mask}" not in source:
                raise KeyError(
                    f"{args.rollout_dataset} is missing mask/{opposite_mask}; "
                    f"the real-robot {args.selection_role} split must be auditable"
                )
            opposite_keys.update(_read_mask(source, opposite_mask))
        overlap = sorted(
            {key for _, key in successes}.intersection(key for _, key in failures),
            key=_demo_sort_key,
        )
        if overlap:
            raise ValueError(f"rollout success and failure masks overlap: {overlap}")
        leaked = sorted(
            ({key for _, key in successes} | {key for _, key in failures})
            & opposite_keys,
            key=_demo_sort_key,
        )
        if leaked:
            raise ValueError(
                f"rollout {args.selection_role} masks overlap the "
                f"{opposite_suffix} split: {leaked}"
            )
        rollout_env = _text(source["data"].attrs.get("env_args", ""))
        if not rollout_env:
            raise ValueError(f"{args.rollout_dataset}:data is missing env_args")
        if json.loads(rollout_env) != json.loads(env_args or "{}"):
            raise ValueError("rollout and human source env_args do not match")
    successes = _select(
        successes,
        int(args.success_count),
        rng,
        description="successful rollout",
    )
    failures = _select(
        failures,
        int(args.failure_count),
        rng,
        description="failed rollout",
    )
    records = (
        [("expert", path, key) for path, key in humans]
        + [("non_expert_success", path, key) for path, key in successes]
        + [("non_expert_failure", path, key) for path, key in failures]
    )
    return records, env_args or "{}"


def actor_condition_value(source_label: str, mode: str) -> int:
    if mode == "human_only":
        return int(source_label == "expert")
    if mode == "human_success":
        return int(source_label in SUCCESS_LABELS)
    raise ValueError(f"unsupported actor condition mode: {mode}")


def _validate_source_episode(
    source_path: Path,
    source_key: str,
    source_label: str,
) -> tuple[int, np.ndarray]:
    location = f"{source_path}:data/{source_key}"
    with h5py.File(source_path, "r") as source:
        if f"data/{source_key}" not in source:
            raise KeyError(location)
        episode = source[f"data/{source_key}"]
        if "actions" not in episode or not isinstance(episode["actions"], h5py.Dataset):
            raise KeyError(f"{location} is missing actions")
        actions = np.asarray(episode["actions"][:], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
            raise ValueError(f"{location}/actions shape={actions.shape}, expected (N, 7)")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.001):
            raise ValueError(f"{location}/actions are not finite normalized actions")
        count = int(actions.shape[0])
        if int(episode.attrs.get("num_samples", count)) != count:
            raise ValueError(f"{location} num_samples does not match actions")
        if "obs" not in episode or not isinstance(episode["obs"], h5py.Group):
            raise KeyError(f"{location} is missing obs")
        obs = episode["obs"]
        for key, trailing_shape in REQUIRED_OBS_SHAPES.items():
            if key not in obs or not isinstance(obs[key], h5py.Dataset):
                raise KeyError(f"{location}/obs is missing {key}")
            dataset = obs[key]
            if int(dataset.shape[0]) != count:
                raise ValueError(f"{dataset.name} length does not match actions")
            if len(dataset.shape) != len(trailing_shape) + 1:
                raise ValueError(f"{dataset.name} has invalid shape {dataset.shape}")
            for actual, expected in zip(dataset.shape[1:], trailing_shape):
                if expected is not None and int(actual) != int(expected):
                    raise ValueError(f"{dataset.name} has invalid shape {dataset.shape}")
        if obs["main_image"].dtype != np.dtype(np.uint8) or obs["wrist_image"].dtype != np.dtype(np.uint8):
            raise ValueError(f"{location} RGB observations must be uint8")
        if "rewards" not in episode:
            raise KeyError(f"{location} is missing source task rewards")
        source_rewards = np.asarray(episode["rewards"][:], dtype=np.float32).reshape(-1)
        if source_rewards.shape != (count,) or not np.isfinite(source_rewards).all():
            raise ValueError(f"{location}/rewards must be finite with shape ({count},)")
        expected = np.zeros(count, dtype=np.float32)
        if source_label in SUCCESS_LABELS:
            expected[-1] = 1.0
        if not np.array_equal(source_rewards, expected):
            outcome = "successful" if source_label in SUCCESS_LABELS else "failed"
            raise ValueError(
                f"{location} {outcome} source rewards must be zero except for "
                + ("one final 1" if source_label in SUCCESS_LABELS else "no positive reward")
                + "; no early-success truncation is inferred"
            )
        if "dones" in episode:
            source_dones = np.asarray(episode["dones"][:], dtype=np.float32).reshape(-1)
            expected_dones = np.zeros(count, dtype=np.float32)
            expected_dones[-1] = 1.0
            if not np.array_equal(source_dones, expected_dones):
                raise ValueError(f"{location}/dones must mark only the final transition")
    return count, expected


def _copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def _create_shifted_next_obs(
    target: h5py.Group,
    source_path: Path,
    source_obs: h5py.Group,
    count: int,
) -> None:
    _copy_attrs(source_obs.attrs, target.attrs)
    for key, child in source_obs.items():
        if isinstance(child, h5py.Group):
            _create_shifted_next_obs(
                target.create_group(key), source_path, child, count
            )
            continue
        if not isinstance(child, h5py.Dataset):
            raise TypeError(f"unsupported HDF5 object {child.name}")
        if child.ndim < 1 or int(child.shape[0]) != count:
            raise ValueError(f"{child.name} must have leading length {count}")
        source = h5py.VirtualSource(str(source_path), child.name, shape=child.shape)
        layout = h5py.VirtualLayout(shape=child.shape, dtype=child.dtype)
        if count > 1:
            layout[:-1] = source[1:]
            layout[-1:] = source[-1:]
        else:
            layout[...] = source[:]
        shifted = target.create_virtual_dataset(key, layout)
        _copy_attrs(child.attrs, shifted.attrs)


def _add_episode(
    output_data: h5py.Group,
    output_key: str,
    source_path: Path,
    source_key: str,
    source_label: str,
    actor_condition_mode: str,
) -> int:
    count, canonical_rewards = _validate_source_episode(
        source_path, source_key, source_label
    )
    with h5py.File(source_path, "r") as source:
        source_episode = source[f"data/{source_key}"]
        target = output_data.create_group(output_key)
        _copy_attrs(source_episode.attrs, target.attrs)
        target.attrs["num_samples"] = count
        target.attrs["source_num_samples"] = count
        target.attrs["truncated_transition_count"] = 0
        target.attrs["terminal_success_index"] = (
            count - 1 if source_label in SUCCESS_LABELS else -1
        )
        target.attrs["rise_source"] = source_label
        target.attrs["rise_source_demo"] = source_key
        target.attrs["rise_source_file"] = str(source_path)

        source_base = f"/data/{source_key}"
        target["actions"] = h5py.ExternalLink(str(source_path), f"{source_base}/actions")
        target["obs"] = h5py.ExternalLink(str(source_path), f"{source_base}/obs")
        target["task_rewards"] = h5py.ExternalLink(
            str(source_path), f"{source_base}/rewards"
        )
        for key in source_episode.keys():
            if key in RESERVED_EPISODE_KEYS:
                continue
            # Linking the top-level object recursively preserves arbitrary
            # provenance subgroups and their attributes without copying RGB.
            target[key] = h5py.ExternalLink(str(source_path), f"{source_base}/{key}")

        target.create_dataset("rewards", data=canonical_rewards)
        dones = np.zeros(count, dtype=np.float32)
        dones[-1] = 1.0
        target.create_dataset("dones", data=dones)
        target.create_dataset(
            "source_is_expert",
            data=np.full(count, source_label == "expert", dtype=np.uint8),
        )
        target.create_dataset(
            "actor_condition",
            data=np.full(
                count,
                actor_condition_value(source_label, actor_condition_mode),
                dtype=np.uint8,
            ),
        )
        _create_shifted_next_obs(
            target.create_group("next_obs"),
            source_path,
            source_episode["obs"],
            count,
        )
    return count


def _write_mask(group: h5py.Group, key: str, values: list[str]) -> None:
    group.create_dataset(key, data=np.asarray(values, dtype="S"))


def _expected_masks(
    records: list[tuple[str, Path, str]],
    actor_condition_mode: str,
    selection_role: str,
) -> dict[str, list[str]]:
    all_keys = [f"demo_{index}" for index in range(len(records))]
    labels = [record[0] for record in records]
    human = [key for key, label in zip(all_keys, labels) if label == "expert"]
    success = [
        key for key, label in zip(all_keys, labels) if label == "non_expert_success"
    ]
    failure = [
        key for key, label in zip(all_keys, labels) if label == "non_expert_failure"
    ]
    rollout = success + failure
    positive = [
        key
        for key, label in zip(all_keys, labels)
        if actor_condition_value(label, actor_condition_mode)
    ]
    negative = [key for key in all_keys if key not in set(positive)]
    return {
        "all": all_keys,
        selection_role if selection_role == "train" else "valid": all_keys,
        "expert": human,
        "human": human,
        "non_expert": rollout,
        "non_expert_success": success,
        "success_rollout": success,
        "non_expert_failure": failure,
        "failure_rollout": failure,
        "actor_condition_positive": positive,
        "actor_condition_negative": negative,
    }


def _source_identities(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "human": [source_file_identity(path) for path in args.human_datasets],
        "rollout": source_file_identity(args.rollout_dataset),
    }


def _default_output_mode() -> int:
    current_umask = os.umask(0)
    try:
        return 0o666 & ~current_umask
    finally:
        os.umask(current_umask)


@contextmanager
def _atomic_output(args: argparse.Namespace):
    descriptor, name = tempfile.mkstemp(
        prefix=f".{args.output.name}.",
        suffix=".partial",
        dir=str(args.output.parent),
    )
    os.close(descriptor)
    staged = Path(name)
    try:
        yield staged
        staged_args = argparse.Namespace(**vars(args))
        staged_args.output = staged
        validate_existing(staged_args)
        mode = (
            args.output.stat().st_mode & 0o777
            if args.output.exists()
            else _default_output_mode()
        )
        os.chmod(staged, mode)
        os.replace(staged, args.output)
    finally:
        staged.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _resolved_link_path(link: h5py.ExternalLink, output: Path) -> Path:
    path = Path(link.filename).expanduser()
    if not path.is_absolute():
        path = output.parent / path
    return path.resolve()


def _check_external_link(
    episode: h5py.Group,
    key: str,
    output: Path,
    source_path: Path,
    object_path: str,
    errors: list[str],
) -> None:
    location = f"{episode.name}/{key}"
    try:
        link = episode.get(key, getlink=True)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"{location} link is inaccessible: {exc}")
        return
    if not isinstance(link, h5py.ExternalLink):
        errors.append(f"{location} is not an external link")
        return
    if _resolved_link_path(link, output) != source_path.resolve() or link.path != object_path:
        errors.append(
            f"{location} points to ({link.filename!r}, {link.path!r}), expected "
            f"({str(source_path)!r}, {object_path!r})"
        )


def _walk_datasets(group: h5py.Group, prefix: str = "") -> Iterable[tuple[str, h5py.Dataset]]:
    for key, child in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(child, h5py.Group):
            yield from _walk_datasets(child, path)
        elif isinstance(child, h5py.Dataset):
            yield path, child


def validate_existing(raw_args: argparse.Namespace) -> dict[str, Any]:
    args = _normalized_args(raw_args)
    if not args.output.is_file():
        raise FileNotFoundError(args.output)
    records, expected_env_args = resolve_selection(args)
    identities = _source_identities(args)
    masks = _expected_masks(
        records, args.actor_condition_mode, args.selection_role
    )
    expected_keys = [f"demo_{index}" for index in range(len(records))]
    errors: list[str] = []
    transition_counts = {label: 0 for label in SOURCE_LABELS}

    with h5py.File(args.output, "r") as output:
        expected_attrs = {
            "task": TASK,
            "reward_mode": REWARD_MODE,
            "reward_definition": REWARD_DEFINITION,
            "actor_condition_mode": str(args.actor_condition_mode),
            "actor_condition_definition": ACTOR_CONDITION_DEFINITIONS[
                args.actor_condition_mode
            ],
            "builder_version": BUILDER_VERSION,
        }
        for key, expected in expected_attrs.items():
            if key not in output.attrs or _text(output.attrs[key]) != expected:
                errors.append(
                    f"root attribute {key!r}={output.attrs.get(key)!r}, expected {expected!r}"
                )
        stored_selection_role = output.attrs.get("selection_role")
        if stored_selection_role is None:
            if args.selection_role != "train":
                errors.append("root is missing validation selection_role")
        elif _text(stored_selection_role) != str(args.selection_role):
            errors.append("root selection_role does not match")
        if int(output.attrs.get("selection_seed", -1)) != int(args.seed):
            errors.append("root selection_seed does not match")
        if int(output.attrs.get("source_identity_version", -1)) != SOURCE_IDENTITY_VERSION:
            errors.append("root source_identity_version does not match")
        try:
            stored_identities = json.loads(_text(output.attrs["source_identities"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            stored_identities = None
            errors.append(f"root source_identities is invalid: {exc}")
        if stored_identities != identities:
            errors.append(
                "root source_identities do not match current source files "
                "(path, size, mtime_ns, or conversion manifest changed)"
            )
        if "data" not in output or "mask" not in output:
            errors.append("output must contain data and mask groups")
        else:
            data = output["data"]
            actual_keys = sorted(data.keys(), key=_demo_sort_key)
            if actual_keys != expected_keys:
                errors.append(f"data keys={actual_keys}, expected={expected_keys}")
            try:
                if json.loads(_text(data.attrs.get("env_args", ""))) != json.loads(
                    expected_env_args
                ):
                    errors.append("data env_args does not match sources")
            except json.JSONDecodeError:
                errors.append("data env_args is invalid JSON")

            for index, (label, source_path, source_key) in enumerate(records):
                output_key = f"demo_{index}"
                count, canonical = _validate_source_episode(
                    source_path, source_key, label
                )
                transition_counts[label] += count
                if output_key not in data:
                    continue
                episode = data[output_key]
                location = f"data/{output_key}"
                for attr, expected in (
                    ("rise_source", label),
                    ("rise_source_demo", source_key),
                    ("rise_source_file", str(source_path)),
                ):
                    if _text(episode.attrs.get(attr, "")) != expected:
                        errors.append(f"{location} {attr} does not match source")
                for attr, expected in (
                    ("num_samples", count),
                    ("source_num_samples", count),
                    ("truncated_transition_count", 0),
                    (
                        "terminal_success_index",
                        count - 1 if label in SUCCESS_LABELS else -1,
                    ),
                ):
                    if int(episode.attrs.get(attr, -2)) != expected:
                        errors.append(f"{location} {attr} does not match")
                base = f"/data/{source_key}"
                for key, object_path in (
                    ("actions", f"{base}/actions"),
                    ("obs", f"{base}/obs"),
                    ("task_rewards", f"{base}/rewards"),
                ):
                    _check_external_link(
                        episode,
                        key,
                        args.output,
                        source_path,
                        object_path,
                        errors,
                    )
                with h5py.File(source_path, "r") as source:
                    source_episode = source[f"data/{source_key}"]
                    for key in source_episode.keys():
                        if key in RESERVED_EPISODE_KEYS:
                            continue
                        _check_external_link(
                            episode,
                            key,
                            args.output,
                            source_path,
                            f"{base}/{key}",
                            errors,
                        )
                    if "next_obs" not in episode:
                        errors.append(f"{location} is missing next_obs")
                    else:
                        source_leaves = dict(_walk_datasets(source_episode["obs"]))
                        next_leaves = dict(_walk_datasets(episode["next_obs"]))
                        if set(source_leaves) != set(next_leaves):
                            errors.append(f"{location}/next_obs keys do not match obs")
                        for leaf, source_dataset in source_leaves.items():
                            if leaf not in next_leaves:
                                continue
                            target_dataset = next_leaves[leaf]
                            if not target_dataset.is_virtual:
                                errors.append(f"{location}/next_obs/{leaf} is not virtual")
                                continue
                            if target_dataset.shape != source_dataset.shape or target_dataset.dtype != source_dataset.dtype:
                                errors.append(f"{location}/next_obs/{leaf} shape/dtype mismatch")
                                continue
                            sample_indices = sorted({0, count // 2, count - 1})
                            for source_index in sample_indices:
                                expected_index = min(source_index + 1, count - 1)
                                if not np.array_equal(
                                    target_dataset[source_index],
                                    source_dataset[expected_index],
                                ):
                                    errors.append(
                                        f"{location}/next_obs/{leaf} has incorrect shift"
                                    )
                                    break
                for key, expected in (
                    ("rewards", canonical),
                    ("task_rewards", canonical),
                    (
                        "dones",
                        np.concatenate(
                            (np.zeros(count - 1, dtype=np.float32), np.ones(1, dtype=np.float32))
                        ),
                    ),
                    (
                        "source_is_expert",
                        np.full(count, label == "expert", dtype=np.uint8),
                    ),
                    (
                        "actor_condition",
                        np.full(
                            count,
                            actor_condition_value(label, args.actor_condition_mode),
                            dtype=np.uint8,
                        ),
                    ),
                ):
                    if key not in episode or not np.array_equal(episode[key][:], expected):
                        errors.append(f"{location}/{key} has incorrect values")

            expected_total = sum(transition_counts.values())
            if int(data.attrs.get("total", -1)) != expected_total:
                errors.append("data total does not match selected transitions")
            mask = output["mask"]
            if set(mask.keys()) != set(masks):
                errors.append(
                    f"mask keys={sorted(mask.keys())}, expected={sorted(masks)}"
                )
            for key, expected in masks.items():
                if key not in mask or _decode(mask[key][:]) != expected:
                    errors.append(f"mask/{key} does not match selection")

    human_transitions = transition_counts["expert"]
    if int(args.expected_human_transitions) >= 0 and human_transitions != int(
        args.expected_human_transitions
    ):
        errors.append(
            f"selected human transitions={human_transitions}, expected "
            f"{int(args.expected_human_transitions)}"
        )
    if errors:
        shown = "\n".join(f"- {error}" for error in errors[:50])
        if len(errors) > 50:
            shown += f"\n- ... {len(errors) - 50} more errors"
        raise ValueError(f"{TASK} mixed dataset validation failed:\n{shown}")
    positive_transition_count = transition_counts["expert"]
    if args.actor_condition_mode == "human_success":
        positive_transition_count += transition_counts[
            "non_expert_success"
        ]
    total_transition_count = sum(transition_counts.values())
    return {
        "validated": True,
        "output": str(args.output),
        "task": TASK,
        "selection_role": str(args.selection_role),
        "actor_condition": {
            "mode": str(args.actor_condition_mode),
            "definition": ACTOR_CONDITION_DEFINITIONS[
                args.actor_condition_mode
            ],
            "positive_episodes": len(
                masks["actor_condition_positive"]
            ),
            "negative_episodes": len(
                masks["actor_condition_negative"]
            ),
            "positive_transitions": int(positive_transition_count),
            "negative_transitions": int(
                total_transition_count - positive_transition_count
            ),
        },
        "episodes": {
            "human": sum(label == "expert" for label, _, _ in records),
            "success_rollout": sum(
                label == "non_expert_success" for label, _, _ in records
            ),
            "failure_rollout": sum(
                label == "non_expert_failure" for label, _, _ in records
            ),
            "total": len(records),
        },
        "transitions": {
            "human": transition_counts["expert"],
            "success_rollout": transition_counts["non_expert_success"],
            "failure_rollout": transition_counts["non_expert_failure"],
            "total": sum(transition_counts.values()),
        },
        "source_identities": identities,
    }


def build(raw_args: argparse.Namespace) -> dict[str, Any]:
    args = _normalized_args(raw_args)
    records, env_args = resolve_selection(args)
    identities = _source_identities(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    masks = _expected_masks(
        records, args.actor_condition_mode, args.selection_role
    )
    transition_counts = {label: 0 for label in SOURCE_LABELS}
    with _atomic_output(args) as staged:
        with h5py.File(staged, "w", libver="latest") as output:
            data = output.create_group("data")
            mask = output.create_group("mask")
            for index, (label, source_path, source_key) in enumerate(records):
                transition_counts[label] += _add_episode(
                    data,
                    f"demo_{index}",
                    source_path,
                    source_key,
                    label,
                    args.actor_condition_mode,
                )
            for key, values in masks.items():
                _write_mask(mask, key, values)
            data.attrs["total"] = sum(transition_counts.values())
            data.attrs["env_args"] = env_args
            output.attrs["builder_version"] = BUILDER_VERSION
            output.attrs["selection_role"] = str(args.selection_role)
            output.attrs["task"] = TASK
            output.attrs["reward_mode"] = REWARD_MODE
            output.attrs["reward_definition"] = REWARD_DEFINITION
            output.attrs["critic_reward_key"] = "rewards"
            output.attrs["preserved_source_task_reward_key"] = "task_rewards"
            output.attrs["terminal_policy"] = (
                "source_episode_end_verified_first_positive_is_final; "
                "no_inferred_early_truncation"
            )
            output.attrs["source_label_definition"] = (
                "source_is_expert=1 for human demo; 0 for deployment rollout"
            )
            output.attrs["actor_condition_mode"] = str(args.actor_condition_mode)
            output.attrs["actor_condition_definition"] = (
                ACTOR_CONDITION_DEFINITIONS[args.actor_condition_mode]
            )
            output.attrs["sampling_definition"] = (
                f"one concatenated {args.selection_role} dataset; "
                "uniform over SequenceDataset indices"
            )
            output.attrs["selection_seed"] = int(args.seed)
            output.attrs["source_identity_version"] = SOURCE_IDENTITY_VERSION
            output.attrs["source_identities"] = _json(identities)
            output.attrs["human_sources"] = _json(
                [str(path) for path in args.human_datasets]
            )
            output.attrs["rollout_source"] = str(args.rollout_dataset)
            output.flush()

    report = validate_existing(args)
    total = report["transitions"]["total"]
    summary = {
        **report,
        "validated": True,
        "reward_mode": REWARD_MODE,
        "reward_definition": REWARD_DEFINITION,
        "terminal_policy": (
            "source episode end; successful source reward must already be its "
            "sole final positive reward; no inferred early truncation"
        ),
        "actor_condition_mode": str(args.actor_condition_mode),
        "selection_role": str(args.selection_role),
        "human_mask": str(args.human_mask),
        "rollout_masks": {
            "success": str(args.success_mask),
            "failure": str(args.failure_mask),
        },
        "transition_fractions": {
            key: report["transitions"][key] / max(total, 1)
            for key in ("human", "success_rollout", "failure_rollout")
        },
        "storage": "external source links plus virtual shifted next_obs",
    }
    _atomic_write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-dataset",
        dest="human_datasets",
        action="append",
        type=Path,
        default=None,
        help=DEFAULT_HUMAN_DATASETS_HELP,
    )
    parser.add_argument("--rollout-dataset", type=Path, default=DEFAULT_ROLLOUT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=(TASK,), default=TASK)
    parser.add_argument("--reward-mode", choices=(REWARD_MODE,), default=REWARD_MODE)
    parser.add_argument(
        "--selection-role",
        choices=("train", "validation"),
        default="train",
        help=(
            "Whether the selected masks form the fitting dataset or the "
            "strictly held-out validation dataset."
        ),
    )
    parser.add_argument("--human-mask", default="train")
    parser.add_argument("--human-count", type=int, default=DEFAULT_HUMAN_COUNT)
    parser.add_argument(
        "--expected-human-transitions",
        type=int,
        default=DEFAULT_EXPECTED_HUMAN_TRANSITIONS,
        help="Fail if selected human rows drift; use -1 to disable this audit.",
    )
    parser.add_argument("--success-mask", default="success_train")
    parser.add_argument("--success-count", type=int, default=DEFAULT_SUCCESS_COUNT)
    parser.add_argument("--failure-mask", default="failure_train")
    parser.add_argument("--failure-count", type=int, default=DEFAULT_FAILURE_COUNT)
    parser.add_argument(
        "--actor-condition-mode",
        choices=tuple(ACTOR_CONDITION_DEFINITIONS),
        default="human_only",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.human_datasets is None:
        parsed.human_datasets = list(DEFAULT_HUMAN_DATASETS)
    return _normalized_args(parsed)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.validate_only:
        if args.overwrite:
            raise ValueError("--validate-only cannot be combined with --overwrite")
        report = validate_existing(args)
        print(json.dumps(report, indent=2), flush=True)
        return report
    return build(args)


if __name__ == "__main__":
    main()
