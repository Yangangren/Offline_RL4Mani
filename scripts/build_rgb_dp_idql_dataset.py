#!/usr/bin/env python3
"""Build one RGB IDQL dataset from expert and deployment trajectories.

The default ``task`` reward mode keeps each source trajectory's environment
reward. The optional ``rise`` mode reproduces the prior binary imitation reward
(human transition 1, rollout transition 0). Source identity and the actor's
condition are stored separately from critic rewards. Large arrays stay in their
source HDF5 files through external links; shifted ``next_obs`` arrays are HDF5
virtual datasets.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERT = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_ROLLOUTS = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/square/idql/square_rgb_dp_idql_200demo_100success_94failure_task_reward.hdf5"
)
REWARD_DEFINITIONS = {
    "task": "source_task_reward",
    "rise": "expert_transition=1; non_expert_transition=0",
}
ACTOR_CONDITION_DEFINITIONS = {
    "human_only": "human_demo=1; success_rollout=0; failure_rollout=0",
    "human_success": "human_demo=1; success_rollout=1; failure_rollout=0",
}
CONVERSION_MANIFEST_ATTR = "_robomimic_conversion_manifest"
SOURCE_IDENTITY_VERSION = 1
SOURCE_IDENTITY_ATTRS = {
    "expert": "expert_source_identity",
    "non_expert": "non_expert_source_identity",
}


def actor_condition_value(source_label: str, mode: str) -> bool:
    if mode == "human_only":
        return source_label == "expert"
    if mode == "human_success":
        return source_label != "non_expert_failure"
    raise ValueError(f"unsupported actor condition mode: {mode}")


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def demo_sort_key(key: str) -> int:
    return int(key.rsplit("_", 1)[-1])


def read_mask(handle: h5py.File, key: str) -> list[str]:
    path = f"mask/{key}"
    if path not in handle:
        available = sorted(handle.get("mask", {}).keys())
        raise KeyError(f"{handle.filename} has no {path}; available={available}")
    return sorted(decode(handle[path][:]), key=demo_sort_key)


def select_keys(
    keys: list[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    keys = sorted(keys, key=demo_sort_key)
    if count < 0:
        return keys
    if count > len(keys):
        raise ValueError(f"requested {count} episodes from only {len(keys)}")
    if count == len(keys):
        return keys
    indices = rng.choice(len(keys), size=count, replace=False)
    return sorted((keys[int(index)] for index in indices), key=demo_sort_key)


def resolve_selection(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str], Any]:
    """Resolve the exact ordered source episodes selected by the CLI inputs."""
    for path in (args.expert_dataset, args.rollout_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.expert_dataset, "r") as expert_file:
        if args.expert_mask:
            expert_candidates = read_mask(expert_file, args.expert_mask)
        else:
            expert_candidates = sorted(
                expert_file["data"].keys(), key=demo_sort_key
            )
        expert_keys = select_keys(expert_candidates, args.expert_count, rng)
        env_args = expert_file["data"].attrs["env_args"]

    with h5py.File(args.rollout_dataset, "r") as rollout_file:
        success_keys = select_keys(
            read_mask(rollout_file, args.success_mask),
            args.success_count,
            rng,
        )
        failure_keys = select_keys(
            read_mask(rollout_file, args.failure_mask),
            args.failure_count,
            rng,
        )
    overlap = sorted(
        set(success_keys).intersection(failure_keys), key=demo_sort_key
    )
    if overlap:
        raise ValueError(f"success and failure masks overlap: {overlap[:10]}")
    return expert_keys, success_keys, failure_keys, env_args


def selected_records(
    args: argparse.Namespace,
    expert_keys: list[str],
    success_keys: list[str],
    failure_keys: list[str],
) -> list[tuple[str, Path, str]]:
    """Return output episode provenance in the builder's canonical order."""
    return (
        [("expert", args.expert_dataset, key) for key in expert_keys]
        + [
            ("non_expert_success", args.rollout_dataset, key)
            for key in success_keys
        ]
        + [
            ("non_expert_failure", args.rollout_dataset, key)
            for key in failure_keys
        ]
    )


def attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def source_file_identity(path: Path) -> dict[str, Any]:
    """Return the source identity embedded in newly built mixed datasets."""
    resolved = path.expanduser().resolve()
    with h5py.File(resolved, "r") as source:
        conversion_manifest = source.attrs.get(CONVERSION_MANIFEST_ATTR)
    source_stat = resolved.stat()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
    }
    if conversion_manifest is not None:
        identity[CONVERSION_MANIFEST_ATTR] = attribute_text(
            conversion_manifest
        )
    return identity


def source_identity_mismatches(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    return sorted(
        key
        for key in set(actual).union(expected)
        if actual.get(key) != expected.get(key)
    )


def external_link_matches(
    link: h5py.ExternalLink,
    output_path: Path,
    source_path: Path,
    source_object: str,
) -> bool:
    linked_file = Path(link.filename).expanduser()
    if not linked_file.is_absolute():
        linked_file = output_path.parent / linked_file
    try:
        linked_file = linked_file.resolve()
    except (OSError, RuntimeError):
        return False
    return (
        linked_file == source_path.expanduser().resolve()
        and str(link.path) == source_object
    )


def paths_match(stored: Any, expected: Path) -> bool:
    try:
        stored_path = Path(attribute_text(stored)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return stored_path == expected.expanduser().resolve()


def validate_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Validate an existing mixed dataset without opening it for writing.

    Selection is recomputed from the requested source masks, counts, and seed.
    The validation deliberately relies on HDF5 provenance rather than the
    companion JSON summary, so deleting or moving that summary cannot make a
    stale mixed dataset appear current.
    """
    if not args.output.is_file():
        raise FileNotFoundError(args.output)

    expert_keys, success_keys, failure_keys, _ = resolve_selection(args)
    records = selected_records(
        args,
        expert_keys,
        success_keys,
        failure_keys,
    )
    expected_source_identities = {
        "expert": source_file_identity(args.expert_dataset),
        "non_expert": source_file_identity(args.rollout_dataset),
    }
    output_keys = [f"demo_{index}" for index in range(len(records))]
    expected_masks = {
        "all": output_keys,
        "expert": output_keys[: len(expert_keys)],
        "non_expert": output_keys[len(expert_keys) :],
        "non_expert_success": output_keys[
            len(expert_keys) : len(expert_keys) + len(success_keys)
        ],
        "non_expert_failure": output_keys[
            len(expert_keys) + len(success_keys) :
        ],
    }

    errors: list[str] = []

    def check_text_attr(
        attrs: h5py.AttributeManager,
        key: str,
        expected: str,
        *,
        optional: bool = False,
        location: str = "root",
    ) -> None:
        if key not in attrs:
            if not optional:
                errors.append(f"{location} is missing attribute {key!r}")
            return
        actual = attribute_text(attrs[key])
        if actual != expected:
            errors.append(
                f"{location} attribute {key!r}={actual!r}, expected "
                f"{expected!r}"
            )

    def required_child(
        parent: h5py.Group,
        key: str,
        expected_type: type,
        location: str,
    ):
        child_location = f"{location}/{key}"
        try:
            child = parent[key]
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{child_location} is inaccessible: {exc}")
            return None
        if not isinstance(child, expected_type):
            errors.append(
                f"{child_location} has type {type(child).__name__}, expected "
                f"{expected_type.__name__}"
            )
            return None
        return child

    def check_dataset_length(
        dataset: h5py.Dataset | None,
        expected_count: int,
        location: str,
    ) -> None:
        if dataset is None:
            return
        if dataset.ndim < 1 or int(dataset.shape[0]) != expected_count:
            errors.append(
                f"{location} shape={dataset.shape}, expected first dimension "
                f"{expected_count}"
            )

    def check_observations(
        observations: h5py.Group | None,
        expected_count: int,
        location: str,
    ) -> None:
        if observations is None:
            return
        dataset_count = 0
        for key in observations.keys():
            dataset = required_child(
                observations,
                key,
                h5py.Dataset,
                location,
            )
            if dataset is not None:
                dataset_count += 1
                check_dataset_length(
                    dataset,
                    expected_count,
                    f"{location}/{key}",
                )
        if dataset_count == 0:
            errors.append(f"{location} contains no accessible datasets")

    def check_external_link(
        episode: h5py.Group,
        key: str,
        source_path: Path,
        source_object: str,
        location: str,
    ) -> None:
        try:
            link = episode.get(key, getlink=True)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{location}/{key} link is inaccessible: {exc}")
            return
        if not isinstance(link, h5py.ExternalLink):
            errors.append(f"{location}/{key} is not an external link")
            return
        if not external_link_matches(
            link,
            args.output,
            source_path,
            source_object,
        ):
            errors.append(
                f"{location}/{key} external link=({link.filename!r}, "
                f"{link.path!r}), expected ({str(source_path)!r}, "
                f"{source_object!r})"
            )

    with (
        h5py.File(args.expert_dataset, "r") as expert_file,
        h5py.File(args.rollout_dataset, "r") as rollout_file,
        h5py.File(args.output, "r") as output,
    ):
        check_text_attr(output.attrs, "task", str(args.task))
        check_text_attr(
            output.attrs,
            "reward_definition",
            REWARD_DEFINITIONS[args.reward_mode],
        )
        # Early builder outputs did not record these redundant mode names. The
        # definitions and per-row labels below are authoritative for them.
        check_text_attr(
            output.attrs,
            "reward_mode",
            str(args.reward_mode),
            optional=True,
        )
        check_text_attr(
            output.attrs,
            "actor_condition_definition",
            ACTOR_CONDITION_DEFINITIONS[args.actor_condition_mode],
        )
        check_text_attr(
            output.attrs,
            "actor_condition_mode",
            str(args.actor_condition_mode),
            optional=True,
        )
        if "selection_seed" in output.attrs:
            actual_seed = int(output.attrs["selection_seed"])
            if actual_seed != int(args.seed):
                errors.append(
                    f"root attribute 'selection_seed'={actual_seed}, expected "
                    f"{int(args.seed)}"
                )
        for key, expected_path in (
            ("expert_source", args.expert_dataset),
            ("non_expert_source", args.rollout_dataset),
        ):
            if key not in output.attrs:
                errors.append(f"root is missing attribute {key!r}")
            elif not paths_match(output.attrs[key], expected_path):
                errors.append(
                    f"root attribute {key!r}="
                    f"{attribute_text(output.attrs[key])!r}, expected source "
                    f"{str(expected_path)!r}"
                )

        identity_metadata_present = "source_identity_version" in output.attrs
        if identity_metadata_present:
            try:
                identity_version = int(output.attrs["source_identity_version"])
            except (TypeError, ValueError):
                identity_version = -1
            if identity_version != SOURCE_IDENTITY_VERSION:
                errors.append(
                    "root source_identity_version="
                    f"{identity_version}, expected {SOURCE_IDENTITY_VERSION}"
                )
        for source_kind, expected_identity in (
            ("expert", expected_source_identities["expert"]),
            ("non_expert", expected_source_identities["non_expert"]),
        ):
            identity_attr = SOURCE_IDENTITY_ATTRS[source_kind]
            if identity_attr not in output.attrs:
                if identity_metadata_present:
                    errors.append(
                        f"root is missing attribute {identity_attr!r}"
                    )
                continue
            try:
                actual_identity = json.loads(
                    attribute_text(output.attrs[identity_attr])
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(
                    f"root attribute {identity_attr!r} is invalid JSON: {exc}"
                )
                continue
            if not isinstance(actual_identity, dict):
                errors.append(
                    f"root attribute {identity_attr!r} is not a JSON object"
                )
                continue
            mismatches = source_identity_mismatches(
                actual_identity,
                expected_identity,
            )
            if mismatches:
                errors.append(
                    f"root attribute {identity_attr!r} does not match the "
                    f"current source: mismatched fields={mismatches}"
                )

        if "data" not in output:
            errors.append("missing group 'data'")
            actual_data_keys: set[str] = set()
            output_data = None
        else:
            output_data = output["data"]
            actual_data_keys = set(output_data.keys())
            expected_data_keys = set(output_keys)
            if actual_data_keys != expected_data_keys:
                errors.append(
                    "data episode keys differ: "
                    f"actual={sorted(actual_data_keys, key=demo_sort_key)}, "
                    f"expected={output_keys}"
                )

        transition_counts = {
            "expert": 0,
            "non_expert": 0,
            "non_expert_success": 0,
            "non_expert_failure": 0,
        }
        for index, (source, source_path, source_key) in enumerate(records):
            output_key = f"demo_{index}"
            source_file = expert_file if source == "expert" else rollout_file
            source_location = f"{source_path}:data/{source_key}"
            source_group = required_child(
                source_file["data"],
                source_key,
                h5py.Group,
                f"{source_path}:data",
            )
            source_actions = None
            source_rewards = None
            source_observations = None
            expected_count = 0
            if source_group is not None:
                source_actions = required_child(
                    source_group,
                    "actions",
                    h5py.Dataset,
                    source_location,
                )
                source_rewards = required_child(
                    source_group,
                    "rewards",
                    h5py.Dataset,
                    source_location,
                )
                source_observations = required_child(
                    source_group,
                    "obs",
                    h5py.Group,
                    source_location,
                )
                if "num_samples" in source_group.attrs:
                    try:
                        expected_count = int(source_group.attrs["num_samples"])
                    except (TypeError, ValueError):
                        errors.append(
                            f"{source_location} has invalid num_samples="
                            f"{source_group.attrs['num_samples']!r}"
                        )
                elif source_actions is not None and source_actions.ndim >= 1:
                    expected_count = int(source_actions.shape[0])
            if expected_count < 1:
                errors.append(
                    f"{source_location} has invalid transition count "
                    f"{expected_count}"
                )
            check_dataset_length(
                source_actions,
                expected_count,
                f"{source_location}/actions",
            )
            check_dataset_length(
                source_rewards,
                expected_count,
                f"{source_location}/rewards",
            )
            check_observations(
                source_observations,
                expected_count,
                f"{source_location}/obs",
            )
            if source == "expert":
                transition_counts["expert"] += expected_count
            else:
                transition_counts["non_expert"] += expected_count
                transition_counts[source] += expected_count

            if output_data is None or output_key not in actual_data_keys:
                continue
            episode = output_data[output_key]
            location = f"data/{output_key}"
            check_text_attr(
                episode.attrs,
                "rise_source",
                source,
                location=location,
            )
            check_text_attr(
                episode.attrs,
                "rise_source_demo",
                source_key,
                location=location,
            )
            if "rise_source_file" not in episode.attrs:
                errors.append(
                    f"{location} is missing attribute 'rise_source_file'"
                )
            elif not paths_match(episode.attrs["rise_source_file"], source_path):
                errors.append(
                    f"{location} source file="
                    f"{attribute_text(episode.attrs['rise_source_file'])!r}, "
                    f"expected {str(source_path)!r}"
                )
            actual_count = int(episode.attrs.get("num_samples", -1))
            if actual_count != expected_count:
                errors.append(
                    f"{location} num_samples={actual_count}, expected "
                    f"{expected_count} from data/{source_key}"
                )

            source_base = f"/data/{source_key}"
            target_actions = required_child(
                episode,
                "actions",
                h5py.Dataset,
                location,
            )
            target_observations = required_child(
                episode,
                "obs",
                h5py.Group,
                location,
            )
            target_task_rewards = required_child(
                episode,
                "task_rewards",
                h5py.Dataset,
                location,
            )
            target_rewards = required_child(
                episode,
                "rewards",
                h5py.Dataset,
                location,
            )
            target_dones = required_child(
                episode,
                "dones",
                h5py.Dataset,
                location,
            )
            check_dataset_length(
                target_actions,
                expected_count,
                f"{location}/actions",
            )
            check_observations(
                target_observations,
                expected_count,
                f"{location}/obs",
            )
            check_dataset_length(
                target_task_rewards,
                expected_count,
                f"{location}/task_rewards",
            )
            check_dataset_length(
                target_rewards,
                expected_count,
                f"{location}/rewards",
            )
            check_dataset_length(
                target_dones,
                expected_count,
                f"{location}/dones",
            )
            check_external_link(
                episode,
                "actions",
                source_path,
                f"{source_base}/actions",
                location,
            )
            check_external_link(
                episode,
                "obs",
                source_path,
                f"{source_base}/obs",
                location,
            )
            check_external_link(
                episode,
                "task_rewards",
                source_path,
                f"{source_base}/rewards",
                location,
            )
            if args.reward_mode == "task":
                check_external_link(
                    episode,
                    "rewards",
                    source_path,
                    f"{source_base}/rewards",
                    location,
                )

            expected_expert = source == "expert"
            expected_condition = actor_condition_value(
                source, args.actor_condition_mode
            )
            for label_key, expected_value in (
                ("source_is_expert", expected_expert),
                ("actor_condition", expected_condition),
            ):
                if label_key not in episode:
                    errors.append(f"{location} is missing {label_key}")
                    continue
                labels = np.asarray(episode[label_key][:])
                if labels.shape != (expected_count,):
                    errors.append(
                        f"{location}/{label_key} shape={labels.shape}, expected "
                        f"({expected_count},)"
                    )
                elif not np.all(labels == expected_value):
                    errors.append(
                        f"{location}/{label_key} does not equal "
                        f"{int(expected_value)} for source {source!r}"
                    )

        expected_total = (
            transition_counts["expert"] + transition_counts["non_expert"]
        )
        if output_data is not None:
            actual_total = int(output_data.attrs.get("total", -1))
            if actual_total != expected_total:
                errors.append(
                    f"data total={actual_total}, expected {expected_total}"
                )

        if "mask" not in output:
            errors.append("missing group 'mask'")
        else:
            mask_group = output["mask"]
            actual_mask_keys = set(mask_group.keys())
            expected_mask_keys = set(expected_masks)
            if actual_mask_keys != expected_mask_keys:
                errors.append(
                    "mask keys differ: "
                    f"actual={sorted(actual_mask_keys)}, "
                    f"expected={sorted(expected_mask_keys)}"
                )
            for key, expected in expected_masks.items():
                if key not in mask_group:
                    continue
                actual = decode(np.asarray(mask_group[key][:]))
                if actual != expected:
                    errors.append(
                        f"mask/{key}={actual}, expected {expected}"
                    )

    if errors:
        displayed_errors = errors[:50]
        details = "\n".join(f"- {error}" for error in displayed_errors)
        if len(errors) > len(displayed_errors):
            details += (
                f"\n- ... {len(errors) - len(displayed_errors)} additional "
                "validation errors omitted"
            )
        raise ValueError(
            f"dataset provenance validation failed for {args.output}:\n{details}"
        )
    return {
        "validated": True,
        "output": str(args.output),
        "task": str(args.task),
        "reward_mode": str(args.reward_mode),
        "actor_condition_mode": str(args.actor_condition_mode),
        "selection_seed": int(args.seed),
        "episodes": {
            "expert": len(expert_keys),
            "non_expert_success": len(success_keys),
            "non_expert_failure": len(failure_keys),
            "total": len(records),
        },
        "transitions": {
            **transition_counts,
            "total": expected_total,
        },
    }


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def create_shifted_next_obs(
    target_group: h5py.Group,
    source_path: Path,
    source_obs: h5py.Group,
) -> None:
    next_obs = target_group.create_group("next_obs")
    for key, dataset in source_obs.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if dataset.shape[0] < 1:
            raise ValueError(f"empty observation dataset {dataset.name}")
        source = h5py.VirtualSource(
            str(source_path),
            dataset.name,
            shape=dataset.shape,
        )
        layout = h5py.VirtualLayout(shape=dataset.shape, dtype=dataset.dtype)
        if dataset.shape[0] > 1:
            layout[:-1] = source[1:]
        layout[-1:] = source[-1:]
        next_obs.create_virtual_dataset(key, layout)


def add_episode(
    output_data: h5py.Group,
    output_key: str,
    source_path: Path,
    source_key: str,
    source_label: str,
    reward_mode: str,
    actor_condition_mode: str,
) -> int:
    with h5py.File(source_path, "r") as source_file:
        source_group = source_file[f"data/{source_key}"]
        count = int(source_group.attrs.get("num_samples", len(source_group["actions"])))
        if count < 1:
            raise ValueError(f"empty episode data/{source_key} in {source_path}")
        if "rewards" not in source_group:
            raise KeyError(f"data/{source_key} has no task rewards in {source_path}")
        if len(source_group["rewards"]) != count:
            raise ValueError(
                f"data/{source_key} has {len(source_group['rewards'])} rewards "
                f"for {count} actions in {source_path}"
            )

        target = output_data.create_group(output_key)
        copy_attrs(source_group.attrs, target.attrs)
        target.attrs["num_samples"] = count
        target.attrs["rise_source"] = source_label
        target.attrs["rise_source_demo"] = source_key
        target.attrs["rise_source_file"] = str(source_path)

        for key in source_group.keys():
            if key in {"obs", "next_obs", "rewards", "dones"}:
                continue
            target[key] = h5py.ExternalLink(
                str(source_path),
                f"/data/{source_key}/{key}",
            )
        target["obs"] = h5py.ExternalLink(
            str(source_path),
            f"/data/{source_key}/obs",
        )
        target["task_rewards"] = h5py.ExternalLink(
            str(source_path),
            f"/data/{source_key}/rewards",
        )
        if reward_mode == "task":
            target["rewards"] = h5py.ExternalLink(
                str(source_path),
                f"/data/{source_key}/rewards",
            )
        elif reward_mode == "rise":
            reward = 1.0 if source_label == "expert" else 0.0
            target.create_dataset(
                "rewards",
                data=np.full(count, reward, dtype=np.float32),
            )
        else:
            raise ValueError(f"unsupported reward mode: {reward_mode}")
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
        dones = np.zeros(count, dtype=np.float32)
        dones[-1] = 1.0
        target.create_dataset("dones", data=dones)
        create_shifted_next_obs(target, source_path, source_group["obs"])
    return count


def write_mask(group: h5py.Group, key: str, demos: list[str]) -> None:
    group.create_dataset(key, data=np.asarray(demos, dtype="S"))


def default_output_mode() -> int:
    """Return the mode a regular file would receive under the current umask."""
    current_umask = os.umask(0)
    try:
        return 0o666 & ~current_umask
    finally:
        os.umask(current_umask)


@contextmanager
def atomic_output_path(args: argparse.Namespace):
    """Stage a validated sibling HDF5 and atomically publish it on success."""
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.",
        suffix=".partial",
        dir=str(args.output.parent),
    )
    os.close(descriptor)
    staged_output = Path(staged_name)
    try:
        yield staged_output
        staged_args = argparse.Namespace(**vars(args))
        staged_args.output = staged_output
        validate_existing(staged_args)
        if args.output.exists():
            published_mode = args.output.stat().st_mode & 0o777
        else:
            published_mode = default_output_mode()
        os.chmod(staged_output, published_mode)
        os.replace(staged_output, args.output)
    finally:
        staged_output.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict:
    expert_keys, success_keys, failure_keys, env_args = resolve_selection(args)
    source_identities = {
        "expert": source_file_identity(args.expert_dataset),
        "non_expert": source_file_identity(args.rollout_dataset),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    records = selected_records(
        args,
        expert_keys,
        success_keys,
        failure_keys,
    )
    episode_keys: dict[str, list[str]] = {
        "expert": [],
        "non_expert": [],
        "non_expert_success": [],
        "non_expert_failure": [],
    }
    transition_counts = {key: 0 for key in episode_keys}

    with (
        atomic_output_path(args) as staged_output,
        h5py.File(staged_output, "w", libver="latest") as output,
    ):
        data = output.create_group("data")
        mask = output.create_group("mask")
        for index, (source, source_path, source_key) in enumerate(records):
            output_key = f"demo_{index}"
            count = add_episode(
                data,
                output_key,
                source_path,
                source_key,
                source,
                args.reward_mode,
                args.actor_condition_mode,
            )
            if source == "expert":
                episode_keys["expert"].append(output_key)
                transition_counts["expert"] += count
            else:
                episode_keys["non_expert"].append(output_key)
                transition_counts["non_expert"] += count
                episode_keys[source].append(output_key)
                transition_counts[source] += count

        all_keys = [f"demo_{index}" for index in range(len(records))]
        write_mask(mask, "all", all_keys)
        for key, demos in episode_keys.items():
            write_mask(mask, key, demos)

        total = transition_counts["expert"] + transition_counts["non_expert"]
        data.attrs["total"] = total
        data.attrs["env_args"] = env_args
        output.attrs["reward_mode"] = str(args.reward_mode)
        output.attrs["reward_definition"] = REWARD_DEFINITIONS[args.reward_mode]
        output.attrs["source_label_definition"] = (
            "source_is_expert=1 for human demo; 0 for deployment rollout"
        )
        output.attrs["actor_condition_definition"] = (
            ACTOR_CONDITION_DEFINITIONS[args.actor_condition_mode]
        )
        output.attrs["actor_condition_mode"] = str(args.actor_condition_mode)
        output.attrs["sampling_definition"] = (
            "one concatenated dataset; uniform over SequenceDataset indices"
        )
        output.attrs["expert_source"] = str(args.expert_dataset)
        output.attrs["non_expert_source"] = str(args.rollout_dataset)
        output.attrs["source_identity_version"] = SOURCE_IDENTITY_VERSION
        output.attrs[SOURCE_IDENTITY_ATTRS["expert"]] = json.dumps(
            source_identities["expert"], sort_keys=True
        )
        output.attrs[SOURCE_IDENTITY_ATTRS["non_expert"]] = json.dumps(
            source_identities["non_expert"], sort_keys=True
        )
        output.attrs["selection_seed"] = int(args.seed)
        output.attrs["task"] = str(args.task)
        output.flush()

    total_transitions = transition_counts["expert"] + transition_counts["non_expert"]
    summary = {
        "output": str(args.output),
        "task": str(args.task),
        "reward_mode": str(args.reward_mode),
        "reward_definition": REWARD_DEFINITIONS[args.reward_mode],
        "source_label_definition": (
            "source_is_expert=1 for human demo; 0 for deployment rollout"
        ),
        "actor_condition_definition": (
            ACTOR_CONDITION_DEFINITIONS[args.actor_condition_mode]
        ),
        "actor_condition_mode": str(args.actor_condition_mode),
        "sampling_definition": "one dataset; uniform shuffled transition-window sampling",
        "expert": {
            "source": str(args.expert_dataset),
            "mask": args.expert_mask or None,
            "episodes": len(expert_keys),
            "transitions": transition_counts["expert"],
            "transition_fraction": transition_counts["expert"] / max(total_transitions, 1),
        },
        "non_expert": {
            "source": str(args.rollout_dataset),
            "episodes": len(success_keys) + len(failure_keys),
            "transitions": transition_counts["non_expert"],
            "transition_fraction": transition_counts["non_expert"] / max(total_transitions, 1),
            "success_mask": args.success_mask,
            "success_episodes": len(success_keys),
            "success_transitions": transition_counts["non_expert_success"],
            "failure_mask": args.failure_mask,
            "failure_episodes": len(failure_keys),
            "failure_transitions": transition_counts["non_expert_failure"],
        },
        "total_episodes": len(records),
        "total_transitions": total_transitions,
        "selection_seed": int(args.seed),
        "source_identities": source_identities,
        "storage": "HDF5 external links plus virtual next_obs datasets",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dataset", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument("--rollout-dataset", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        choices=("square", "can", "transport", "tool_hang"),
        default="square",
    )
    parser.add_argument("--expert-mask", default="")
    parser.add_argument("--expert-count", type=int, default=200)
    parser.add_argument("--success-mask", default="success_100")
    parser.add_argument("--success-count", type=int, default=-1)
    parser.add_argument("--failure-mask", default="failure")
    parser.add_argument("--failure-count", type=int, default=-1)
    parser.add_argument(
        "--reward-mode",
        choices=tuple(REWARD_DEFINITIONS),
        default="task",
        help=(
            "task keeps source environment rewards (default); rise assigns "
            "human transition=1 and rollout transition=0"
        ),
    )
    parser.add_argument(
        "--actor-condition-mode",
        choices=tuple(ACTOR_CONDITION_DEFINITIONS),
        default="human_only",
        help=(
            "human_only labels only demonstrations as condition 1; "
            "human_success labels demonstrations and successful deployment "
            "rollouts as 1 and failure rollouts as 0"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate that an existing output exactly matches the requested "
            "source selection and metadata without rewriting any files."
        ),
    )
    args = parser.parse_args(argv)
    for key in ("expert_dataset", "rollout_dataset", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.validate_only:
        if args.overwrite:
            raise ValueError("--validate-only cannot be combined with --overwrite")
        result = validate_existing(args)
        print(json.dumps(result, indent=2), flush=True)
        return result
    return build(args)


if __name__ == "__main__":
    main()
