#!/usr/bin/env python3
"""Validate converted pick-cup HDF5 shards and their temporal provenance."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot.pick_cup_common import (  # noqa: E402
    CONVERSION_MANIFEST_ATTR,
    CONVERSION_VERSION,
    DATASET_COMMIT_FILENAME,
    DEFAULT_DATASET_DIR,
    DEFAULT_SOURCE,
    LOW_DIM_KEYS,
    RGB_KEYS,
    SCHEMA_VERSION,
    atomic_write_json,
    dataset_commit_path,
    decode_hdf5_strings,
    densify_gripper_events,
    eligible_rows,
    round_paths,
    source_identity,
    strictly_increasing,
)


DEMO_PATTERN = re.compile(r"^demo_[0-9]+$")


def _json_attr(attrs: h5py.AttributeManager, key: str, *, location: str) -> Any:
    if key not in attrs:
        raise ValueError(f"{location} is missing attribute {key!r}")
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{location} attribute {key!r} must be JSON text")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} attribute {key!r} is malformed JSON") from exc


def _require_dataset(
    group: h5py.Group,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | type | None = None,
) -> h5py.Dataset:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise ValueError(f"{group.name} is missing dataset {name!r}")
    dataset = group[name]
    if shape is not None and tuple(dataset.shape) != tuple(shape):
        raise ValueError(
            f"{dataset.name} has shape {dataset.shape}, expected {shape}"
        )
    if dtype is not None and np.dtype(dataset.dtype) != np.dtype(dtype):
        raise ValueError(
            f"{dataset.name} has dtype {dataset.dtype}, expected {np.dtype(dtype)}"
        )
    return dataset


def _finite(array: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")


def _validate_masks(
    dataset: h5py.File,
    demo_keys: set[str],
    manifest: dict[str, Any],
) -> dict[str, list[str]]:
    if "mask" not in dataset or not isinstance(dataset["mask"], h5py.Group):
        raise ValueError(f"{dataset.filename} is missing /mask")
    mask = dataset["mask"]
    masks: dict[str, list[str]] = {}
    for key in ("all", "train", "valid"):
        values = decode_hdf5_strings(_require_dataset(mask, key)[:])
        if len(values) != len(set(values)):
            raise ValueError(f"{mask.name}/{key} contains duplicate demo keys")
        unknown = set(values).difference(demo_keys)
        if unknown:
            raise ValueError(f"{mask.name}/{key} references unknown demos: {sorted(unknown)}")
        masks[key] = values
    if set(masks["all"]) != demo_keys:
        raise ValueError("mask/all does not exactly cover /data demos")
    train = set(masks["train"])
    valid = set(masks["valid"])
    if not train or not valid:
        raise ValueError("mask/train and mask/valid must both be non-empty")
    overlap = train.intersection(valid)
    if overlap:
        raise ValueError(f"train and valid masks overlap: {sorted(overlap)}")
    if train.union(valid) != demo_keys:
        raise ValueError("train and valid masks do not exactly partition the demos")

    manifest_split = manifest.get("split")
    if not isinstance(manifest_split, dict):
        raise ValueError("conversion manifest is missing split metadata")
    if set(manifest_split.get("train", ())) != train:
        raise ValueError("manifest train split does not match mask/train")
    if set(manifest_split.get("valid", ())) != valid:
        raise ValueError("manifest valid split does not match mask/valid")
    return masks


def _validate_episode(
    demo: h5py.Group,
    *,
    image_height: int,
    image_width: int,
    max_image_age_sec: float,
    collection_round: int,
) -> dict[str, Any]:
    if "num_samples" not in demo.attrs:
        raise ValueError(f"{demo.name} is missing num_samples")
    count = int(demo.attrs["num_samples"])
    if count < 1:
        raise ValueError(f"{demo.name} has non-positive num_samples={count}")
    if int(demo.attrs.get("collection_round", -1)) != collection_round:
        raise ValueError(f"{demo.name} collection_round does not match its shard")

    actions_ds = _require_dataset(demo, "actions", shape=(count, 7), dtype=np.float32)
    rewards_ds = _require_dataset(demo, "rewards", shape=(count,), dtype=np.float32)
    dones_ds = _require_dataset(demo, "dones", shape=(count,), dtype=np.uint8)
    actions = actions_ds[:]
    rewards = rewards_ds[:]
    dones = dones_ds[:]
    _finite(actions, name=actions_ds.name)
    _finite(rewards, name=rewards_ds.name)
    if float(np.min(actions)) < -1.001 or float(np.max(actions)) > 1.001:
        raise ValueError(f"{actions_ds.name} is outside [-1,1]")
    if not np.all(np.isin(actions[:, 6], (-1.0, 1.0))):
        raise ValueError(f"{actions_ds.name} dense gripper targets must be -1 or 1")
    expected_rewards = np.zeros(count, dtype=np.float32)
    expected_rewards[-1] = 1.0
    if not np.array_equal(rewards, expected_rewards):
        raise ValueError(f"{rewards_ds.name} must be terminal-sparse with final reward 1")
    expected_dones = np.zeros(count, dtype=np.uint8)
    expected_dones[-1] = 1
    if not np.array_equal(dones, expected_dones):
        raise ValueError(f"{dones_ds.name} must contain only a final done flag")

    if "next_obs" in demo:
        raise ValueError(f"{demo.name} should not store next_obs for Diffusion Policy")
    if "obs" not in demo or not isinstance(demo["obs"], h5py.Group):
        raise ValueError(f"{demo.name} is missing /obs")
    obs = demo["obs"]
    if set(obs.keys()) != set((*RGB_KEYS, *LOW_DIM_KEYS)):
        raise ValueError(
            f"{obs.name} keys {sorted(obs.keys())} do not match expected observation schema"
        )
    image_shape = (count, image_height, image_width, 3)
    for key in RGB_KEYS:
        image_ds = _require_dataset(obs, key, shape=image_shape, dtype=np.uint8)
        for index in sorted({0, count // 2, count - 1}):
            image = image_ds[index]
            if image.shape != image_shape[1:]:
                raise ValueError(f"{image_ds.name}[{index}] has an invalid shape")
    positions = _require_dataset(
        obs,
        "robot0_eef_pos",
        shape=(count, 3),
        dtype=np.float32,
    )[:]
    quaternions = _require_dataset(
        obs,
        "robot0_eef_quat",
        shape=(count, 4),
        dtype=np.float32,
    )[:]
    gripper_observations = _require_dataset(
        obs,
        "robot0_gripper_state",
        shape=(count, 1),
        dtype=np.float32,
    )[:, 0]
    _finite(positions, name=f"{obs.name}/robot0_eef_pos")
    _finite(quaternions, name=f"{obs.name}/robot0_eef_quat")
    _finite(gripper_observations, name=f"{obs.name}/robot0_gripper_state")
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=5e-3):
        raise ValueError(f"{obs.name}/robot0_eef_quat contains non-unit quaternions")
    if not np.all(np.isin(gripper_observations, (-1.0, 1.0))):
        raise ValueError(f"{obs.name}/robot0_gripper_state must be -1 or 1")

    if "provenance" not in demo or not isinstance(demo["provenance"], h5py.Group):
        raise ValueError(f"{demo.name} is missing /provenance")
    provenance = demo["provenance"]
    required_provenance = {
        "raw_gripper_event",
        "source_action_array_index",
        "source_action_index",
        "source_action_step",
        "action_target_time",
        "action_source_time",
        "selected_frame_position",
        "selected_frame_index",
        "selected_frame_nominal_time",
        "selected_main_capture_time",
        "selected_wrist_capture_time",
        "image_age_sec",
    }
    if set(provenance.keys()) != required_provenance:
        raise ValueError(
            f"{provenance.name} keys do not match required conversion provenance"
        )
    for key in required_provenance:
        _require_dataset(provenance, key, shape=(count,))

    raw_events = provenance["raw_gripper_event"][:]
    initial_state = float(demo.attrs.get("initial_gripper_state", np.nan))
    expected_obs, expected_targets = densify_gripper_events(
        raw_events,
        initial_state=initial_state,
    )
    if not np.array_equal(gripper_observations, expected_obs):
        raise ValueError(
            f"{demo.name} pre-action gripper observations are inconsistent with raw events"
        )
    if not np.array_equal(actions[:, 6], expected_targets):
        raise ValueError(
            f"{demo.name} dense gripper actions are inconsistent with raw events"
        )

    action_times = np.asarray(provenance["action_target_time"][:], dtype=np.float64)
    main_times = np.asarray(
        provenance["selected_main_capture_time"][:], dtype=np.float64
    )
    wrist_times = np.asarray(
        provenance["selected_wrist_capture_time"][:], dtype=np.float64
    )
    image_ages = np.asarray(provenance["image_age_sec"][:], dtype=np.float64)
    selected_positions = np.asarray(
        provenance["selected_frame_position"][:], dtype=np.int64
    )
    selected_indices = np.asarray(
        provenance["selected_frame_index"][:], dtype=np.int64
    )
    strictly_increasing(action_times, name=f"{demo.name} action target times")
    for values, name in (
        (main_times, "main capture times"),
        (wrist_times, "wrist capture times"),
        (image_ages, "image ages"),
    ):
        _finite(values, name=f"{demo.name} {name}")
    if np.any(main_times > action_times + 1e-6) or np.any(
        wrist_times > action_times + 1e-6
    ):
        raise ValueError(f"{demo.name} uses a future camera capture")
    expected_ages = action_times - np.maximum(main_times, wrist_times)
    if not np.allclose(image_ages, expected_ages, atol=1e-7):
        raise ValueError(f"{demo.name} image_age_sec does not match capture provenance")
    if float(np.min(image_ages)) < -1e-6:
        raise ValueError(f"{demo.name} has negative image age")
    if float(np.max(image_ages)) > max_image_age_sec + 1e-6:
        raise ValueError(f"{demo.name} exceeds the configured image-age limit")
    if np.any(np.diff(selected_positions) < 0) or np.any(np.diff(selected_indices) < 0):
        raise ValueError(f"{demo.name} selected frame indices are not monotonic")

    demo_key = demo.name.rsplit("/", 1)[-1]
    source_episode_number = int(demo.attrs.get("source_episode_number", -1))
    if source_episode_number < 1:
        raise ValueError(f"{demo.name} has an invalid source_episode_number")
    if demo_key != f"demo_{source_episode_number:03d}":
        raise ValueError(
            f"{demo.name} does not match source episode {source_episode_number}"
        )
    expected_round = 1 if source_episode_number <= 50 else 2
    if expected_round != collection_round:
        raise ValueError(
            f"{demo.name} source episode belongs to round {expected_round}, "
            f"not round {collection_round}"
        )

    return {
        "demo_key": demo_key,
        "source_episode_number": source_episode_number,
        "samples": count,
        "dropped_prefix_actions": int(demo.attrs.get("dropped_prefix_actions", -1)),
        "max_image_age_sec": float(np.max(image_ages)),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
    }


def validate_dataset(
    path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset shard does not exist: {path}")
    with h5py.File(path, "r") as dataset:
        if int(dataset.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"{path} has an unsupported schema_version")
        if dataset.attrs.get("conversion_version") != CONVERSION_VERSION:
            raise ValueError(f"{path} has an unsupported conversion_version")
        manifest = _json_attr(
            dataset.attrs,
            CONVERSION_MANIFEST_ATTR,
            location=str(path),
        )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path} manifest schema version mismatch")
        if manifest.get("conversion_version") != CONVERSION_VERSION:
            raise ValueError(f"{path} manifest conversion version mismatch")
        generation_id = manifest.get("generation_id")
        if not isinstance(generation_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", generation_id
        ):
            raise ValueError(f"{path} has an invalid generation_id")
        root_generation_id = dataset.attrs.get("generation_id")
        if isinstance(root_generation_id, bytes):
            root_generation_id = root_generation_id.decode("utf-8")
        if root_generation_id != generation_id:
            raise ValueError(f"{path} generation_id attribute does not match manifest")
        if source_root is not None:
            current_identity = source_identity(source_root.expanduser().resolve())
            if manifest.get("source_identity") != current_identity:
                raise ValueError(f"{path} source identity does not match {source_root}")

        try:
            collection_round = int(manifest["collection_round"])
            image_height = int(manifest["image"]["height"])
            image_width = int(manifest["image"]["width"])
            max_image_age_sec = float(manifest["alignment"]["max_image_age_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} manifest is missing core conversion settings") from exc
        if collection_round not in (1, 2):
            raise ValueError(f"{path} has invalid collection_round={collection_round}")
        if image_height <= 1 or image_width <= 1 or max_image_age_sec <= 0.0:
            raise ValueError(f"{path} has invalid image/alignment settings")

        if "data" not in dataset or not isinstance(dataset["data"], h5py.Group):
            raise ValueError(f"{path} is missing /data")
        data = dataset["data"]
        env_args = _json_attr(data.attrs, "env_args", location=data.name)
        if not isinstance(env_args, dict) or not {
            "env_name",
            "type",
            "env_kwargs",
        }.issubset(env_args):
            raise ValueError(f"{data.name} env_args has an invalid structure")
        if int(env_args["type"]) not in (1, 2, 3):
            raise ValueError(f"{data.name} env_args has an unsupported type")

        demo_keys = set(data.keys())
        if not demo_keys:
            raise ValueError(f"{path} has no demonstrations")
        invalid_names = sorted(key for key in demo_keys if not DEMO_PATTERN.match(key))
        if invalid_names:
            raise ValueError(f"{path} contains invalid /data children: {invalid_names}")
        masks = _validate_masks(dataset, demo_keys, manifest)

        episodes = [
            _validate_episode(
                data[key],
                image_height=image_height,
                image_width=image_width,
                max_image_age_sec=max_image_age_sec,
                collection_round=collection_round,
            )
            for key in sorted(demo_keys)
        ]
        total = sum(episode["samples"] for episode in episodes)
        if int(data.attrs.get("total", -1)) != total:
            raise ValueError(f"{data.name} total does not match demonstration lengths")

        manifest_episode_keys = {
            item.get("demo_key")
            for item in manifest.get("episodes", ())
            if isinstance(item, dict)
        }
        if manifest_episode_keys != demo_keys:
            raise ValueError(f"{path} manifest episodes do not match /data demos")

        signature = {
            "action_shape": [7],
            "image_shape": [image_height, image_width, 3],
            "low_dim_shapes": {
                "robot0_eef_pos": [3],
                "robot0_eef_quat": [4],
                "robot0_gripper_state": [1],
            },
        }
        return {
            "path": str(path),
            "collection_round": collection_round,
            "episodes": len(episodes),
            "train_episodes": len(masks["train"]),
            "valid_episodes": len(masks["valid"]),
            "samples": total,
            "dropped_prefix_actions": sum(
                episode["dropped_prefix_actions"] for episode in episodes
            ),
            "max_image_age_sec": max(
                episode["max_image_age_sec"] for episode in episodes
            ),
            "action_min": min(episode["action_min"] for episode in episodes),
            "action_max": max(episode["action_max"] for episode in episodes),
            "source_episode_numbers": [
                episode["source_episode_number"] for episode in episodes
            ],
            "schema_signature": signature,
            "source_identity": manifest.get("source_identity"),
            "generation_id": generation_id,
        }


def validate_datasets(
    paths: Sequence[Path],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if len(paths) != 2:
        raise ValueError(f"expected exactly two round shards, got {len(paths)}")
    shards = [validate_dataset(path, source_root=source_root) for path in paths]
    rounds = {shard["collection_round"] for shard in shards}
    if rounds != {1, 2}:
        raise ValueError(f"expected collection rounds 1 and 2, got {sorted(rounds)}")
    if shards[0]["schema_signature"] != shards[1]["schema_signature"]:
        raise ValueError("round shards have inconsistent observation/action schemas")
    if shards[0]["source_identity"] != shards[1]["source_identity"]:
        raise ValueError("round shards have inconsistent source identities")
    generation_ids = {shard["generation_id"] for shard in shards}
    if len(generation_ids) != 1:
        raise ValueError("round shards belong to different dataset generations")
    source_sets = [set(shard["source_episode_numbers"]) for shard in shards]
    for shard, source_set in zip(shards, source_sets):
        if len(source_set) != len(shard["source_episode_numbers"]):
            raise ValueError(
                f"round {shard['collection_round']} contains duplicate source episodes"
            )
    overlap = source_sets[0].intersection(source_sets[1])
    if overlap:
        raise ValueError(f"source episodes occur in both round shards: {sorted(overlap)}")
    if source_root is not None:
        expected_by_round = {
            round_id: {
                row.episode_number
                for row in eligible_rows(source_root.expanduser().resolve())
                if row.collection_round == round_id
            }
            for round_id in (1, 2)
        }
        actual_by_round = {
            shard["collection_round"]: set(shard["source_episode_numbers"])
            for shard in shards
        }
        for round_id in (1, 2):
            missing = expected_by_round[round_id].difference(actual_by_round[round_id])
            unexpected = actual_by_round[round_id].difference(expected_by_round[round_id])
            if missing or unexpected:
                raise ValueError(
                    f"round {round_id} does not exactly cover eligible source episodes; "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )

    public_shards = []
    for shard in sorted(shards, key=lambda item: item["collection_round"]):
        public = dict(shard)
        public.pop("schema_signature")
        public.pop("source_identity")
        public.pop("source_episode_numbers")
        public_shards.append(public)
    return {
        "validated": True,
        "shards": public_shards,
        "episodes": sum(shard["episodes"] for shard in shards),
        "samples": sum(shard["samples"] for shard in shards),
        "dropped_prefix_actions": sum(
            shard["dropped_prefix_actions"] for shard in shards
        ),
        "max_image_age_sec": max(
            shard["max_image_age_sec"] for shard in shards
        ),
        "schema_signature": shards[0]["schema_signature"],
        "generation_id": next(iter(generation_ids)),
    }


def validate_published_datasets(
    dataset_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate both shards and the atomic publication commit marker."""

    dataset_dir = dataset_dir.expanduser().resolve()
    paths = round_paths(dataset_dir)
    report = validate_datasets(paths, source_root=source_root)
    commit_path = dataset_commit_path(dataset_dir)
    if not commit_path.is_file():
        raise FileNotFoundError(
            f"published dataset is missing {DATASET_COMMIT_FILENAME}: {commit_path}"
        )
    try:
        commit = json.loads(commit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset commit marker: {commit_path}") from exc
    if not isinstance(commit, dict):
        raise ValueError(f"dataset commit marker must contain a JSON object: {commit_path}")
    if commit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset commit marker schema version mismatch")
    if commit.get("conversion_version") != CONVERSION_VERSION:
        raise ValueError("dataset commit marker conversion version mismatch")
    if commit.get("generation_id") != report["generation_id"]:
        raise ValueError("dataset commit marker does not match shard generation")
    committed_shards = commit.get("shards")
    expected_shards = [
        {"filename": path.name, "size_bytes": int(path.stat().st_size)}
        for path in paths
    ]
    if committed_shards != expected_shards:
        raise ValueError("dataset commit marker does not match published shard files")
    report["commit_path"] = str(commit_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--datasets", type=Path, nargs=2, default=None)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    source_root = None if args.skip_source_check else args.source
    if args.datasets is None:
        report = validate_published_datasets(
            args.dataset_dir,
            source_root=source_root,
        )
    else:
        report = validate_datasets(tuple(args.datasets), source_root=source_root)
    if args.report is not None:
        atomic_write_json(args.report.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
