#!/usr/bin/env python3
"""Build one RGB IDQL dataset from expert and deployment trajectories.

The default ``task`` reward mode keeps each source trajectory's environment
reward. The optional ``rise`` mode reproduces the prior binary imitation reward
(human transition 1, rollout transition 0). Source identity and the actor's
outcome condition are stored separately so actor conditioning never depends on
the critic reward definition: human and successful rollout transitions use
condition 1, while failed rollout transitions use condition 0. Large arrays
stay in their source HDF5 files through external links; shifted ``next_obs``
arrays are HDF5 virtual datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
                source_label in {"expert", "non_expert_success"},
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


def build(args: argparse.Namespace) -> dict:
    for path in (args.expert_dataset, args.rollout_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.expert_dataset, "r") as expert_file:
        if args.expert_mask:
            expert_candidates = read_mask(expert_file, args.expert_mask)
        else:
            expert_candidates = sorted(expert_file["data"].keys(), key=demo_sort_key)
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
    overlap = sorted(set(success_keys).intersection(failure_keys), key=demo_sort_key)
    if overlap:
        raise ValueError(f"success and failure masks overlap: {overlap[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    records = (
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
    episode_keys: dict[str, list[str]] = {
        "expert": [],
        "non_expert": [],
        "non_expert_success": [],
        "non_expert_failure": [],
    }
    transition_counts = {key: 0 for key in episode_keys}

    with h5py.File(args.output, "w", libver="latest") as output:
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
            "human_demo=1; success_rollout=1; failure_rollout=0"
        )
        output.attrs["sampling_definition"] = (
            "one concatenated dataset; uniform over SequenceDataset indices"
        )
        output.attrs["expert_source"] = str(args.expert_dataset)
        output.attrs["non_expert_source"] = str(args.rollout_dataset)
        output.attrs["selection_seed"] = int(args.seed)
        output.attrs["task"] = str(args.task)

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
            "human_demo=1; success_rollout=1; failure_rollout=0"
        ),
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
        "storage": "HDF5 external links plus virtual next_obs datasets",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in ("expert_dataset", "rollout_dataset", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


if __name__ == "__main__":
    build(parse_args())
