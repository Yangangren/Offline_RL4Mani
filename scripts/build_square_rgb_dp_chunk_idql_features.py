#!/usr/bin/env python3
"""Build frozen-DP feature transitions for a Square RGB-DP IDQL baseline.

This is the first, deliberately conservative IDQL-style baseline:

* the RGB DiffusionPolicy is kept frozen;
* its observation encoder is used as a fixed representation;
* the offline RL action is the executed 8-step action chunk;
* rewards are the default robomimic per-step rewards stored in the HDF5 files.

The output is an NPZ file containing chunk transitions

    (phi(s_t), a_{t:t+H-1}, R_t^H, phi(s_{t+H}), done)

where H defaults to the DP action horizon, 8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1"
    / "20260629231002/last.pth"
)
DEFAULT_DEMOS = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_ROLLOUTS = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_chunk_features.npz"
)
OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def selected_demo_keys(path: Path, filter_key: str | None) -> list[str]:
    if filter_key == "":
        filter_key = None
    with h5py.File(path, "r") as f:
        if filter_key is None:
            return sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        return decode(f[f"mask/{filter_key}"][:])


def discount_vector(horizon: int, gamma: float) -> np.ndarray:
    return np.asarray([gamma**i for i in range(horizon)], dtype=np.float32)


def make_obs_batch(group: h5py.Group, steps: np.ndarray, history: int) -> dict[str, np.ndarray]:
    """Return raw HDF5 observations with shape [B, history, ...]."""
    n = int(group.attrs["num_samples"])
    indices = []
    for step in steps:
        step = int(step)
        start = step - history + 1
        indices.append([min(max(j, 0), n - 1) for j in range(start, step + 1)])
    indices = np.asarray(indices, dtype=np.int64)

    obs = {}
    for key in OBS_KEYS:
        obs[key] = np.asarray(group[f"obs/{key}"][:][indices])
    return obs


def process_obs_batch(raw_obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    obs = TensorUtils.to_tensor(raw_obs)
    obs = TensorUtils.to_device(obs, device)
    obs = TensorUtils.to_float(obs)
    return ObsUtils.process_obs_dict(obs)


@torch.no_grad()
def encode_obs_batch(policy, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
    algo = policy.policy
    obs = process_obs_batch(raw_obs, algo.device)
    nets = algo.nets
    if algo.ema is not None:
        nets = algo.ema.averaged_model
    features = algo._encode_obs({"obs": obs, "goal": None}, nets)
    return features.detach().cpu().numpy().astype(np.float32)


def episode_success(group: h5py.Group) -> bool:
    rewards = np.asarray(group["rewards"][:], dtype=np.float32)
    return bool(np.max(rewards) > 0.5)


def build_episode_records(
    *,
    source_name: str,
    path: Path,
    demo_key: str,
    group: h5py.Group,
    chunk_horizon: int,
    stride: int,
    gamma: float,
    include_partial_terminal: bool,
) -> dict[str, np.ndarray | list[str]]:
    n = int(group.attrs["num_samples"])
    actions = np.asarray(group["actions"][:], dtype=np.float32)
    rewards = np.asarray(group["rewards"][:], dtype=np.float32)
    dones = np.asarray(group["dones"][:], dtype=np.float32) if "dones" in group else np.zeros(n, dtype=np.float32)

    if include_partial_terminal:
        steps = np.arange(0, n, stride, dtype=np.int64)
    else:
        steps = np.arange(0, max(n - chunk_horizon + 1, 0), stride, dtype=np.int64)

    if len(steps) == 0:
        return {
            "steps": np.zeros((0,), dtype=np.int64),
            "next_steps": np.zeros((0,), dtype=np.int64),
            "actions": np.zeros((0, chunk_horizon, actions.shape[-1]), dtype=np.float32),
            "returns": np.zeros((0,), dtype=np.float32),
            "dones": np.zeros((0,), dtype=np.float32),
            "success": np.zeros((0,), dtype=np.float32),
            "source": [],
            "demo": [],
        }

    discounts = discount_vector(chunk_horizon, gamma)
    action_chunks = []
    chunk_returns = []
    chunk_dones = []
    next_steps = []
    for step in steps:
        end = min(int(step) + chunk_horizon, n)
        action_chunk = np.zeros((chunk_horizon, actions.shape[-1]), dtype=np.float32)
        real = actions[int(step) : end]
        action_chunk[: len(real)] = real
        action_chunks.append(action_chunk)

        reward_chunk = np.zeros((chunk_horizon,), dtype=np.float32)
        reward_chunk[: end - int(step)] = rewards[int(step) : end]
        chunk_returns.append(float(np.sum(discounts * reward_chunk)))

        done = bool(end >= n or np.any(dones[int(step) : end] > 0.5))
        chunk_dones.append(float(done))
        next_steps.append(min(int(step) + chunk_horizon, n - 1))

    success = float(episode_success(group))
    return {
        "steps": steps.astype(np.int64),
        "next_steps": np.asarray(next_steps, dtype=np.int64),
        "actions": np.asarray(action_chunks, dtype=np.float32),
        "returns": np.asarray(chunk_returns, dtype=np.float32),
        "dones": np.asarray(chunk_dones, dtype=np.float32),
        "success": np.full((len(steps),), success, dtype=np.float32),
        "source": [source_name] * len(steps),
        "demo": [demo_key] * len(steps),
    }


def source_specs(args) -> list[tuple[str, Path, str | None]]:
    specs = []
    if args.include_demos:
        specs.append(("demo", args.demo_dataset, args.demo_filter_key))
    if args.include_rollouts:
        specs.append(("rollout_success", args.rollout_dataset, args.success_filter_key))
        specs.append(("rollout_failure", args.rollout_dataset, args.failure_filter_key))
    return specs


def split_by_episode(source: np.ndarray, demo: np.ndarray, seed: int, val_fraction: float, test_fraction: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episode_ids = np.asarray([f"{s}:{d}" for s, d in zip(source, demo)])
    unique = np.unique(episode_ids)
    rng.shuffle(unique)
    n = len(unique)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = set(unique[:n_test])
    val = set(unique[n_test : n_test + n_val])
    split = np.full(len(episode_ids), "train", dtype=object)
    for i, episode in enumerate(episode_ids):
        if episode in test:
            split[i] = "test"
        elif episode in val:
            split[i] = "val"
    return split.astype("S")


def build(args) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = None
    if args.cache_latent_features:
        device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=str(args.checkpoint),
            device=device,
            verbose=False,
        )
        policy.start_episode()

    all_obs_features = []
    all_next_obs_features = []
    all_actions = []
    all_returns = []
    all_dones = []
    all_success = []
    all_steps = []
    all_next_steps = []
    all_source = []
    all_demo = []
    per_source = {}

    for source_name, path, filter_key in source_specs(args):
        keys = selected_demo_keys(path, filter_key)
        per_source[source_name] = {"path": str(path), "filter_key": filter_key, "episodes": len(keys), "chunks": 0}
        with h5py.File(path, "r") as f:
            for index, demo_key in enumerate(keys):
                group = f[f"data/{demo_key}"]
                records = build_episode_records(
                    source_name=source_name,
                    path=path,
                    demo_key=demo_key,
                    group=group,
                    chunk_horizon=args.chunk_horizon,
                    stride=args.stride,
                    gamma=args.gamma,
                    include_partial_terminal=args.include_partial_terminal,
                )
                if len(records["steps"]) == 0:
                    continue
                if args.cache_latent_features:
                    obs_features = []
                    next_obs_features = []
                    for start in range(0, len(records["steps"]), args.encoder_batch_size):
                        sl = slice(start, start + args.encoder_batch_size)
                        obs = make_obs_batch(group, records["steps"][sl], args.observation_horizon)
                        next_obs = make_obs_batch(group, records["next_steps"][sl], args.observation_horizon)
                        obs_features.append(encode_obs_batch(policy, obs))
                        next_obs_features.append(encode_obs_batch(policy, next_obs))
                    obs_features = np.concatenate(obs_features, axis=0)
                    next_obs_features = np.concatenate(next_obs_features, axis=0)
                    all_obs_features.append(obs_features)
                    all_next_obs_features.append(next_obs_features)
                all_actions.append(records["actions"])
                all_returns.append(records["returns"])
                all_dones.append(records["dones"])
                all_success.append(records["success"])
                all_steps.append(records["steps"])
                all_next_steps.append(records["next_steps"])
                all_source.extend(records["source"])
                all_demo.extend(records["demo"])
                per_source[source_name]["chunks"] += int(len(records["steps"]))
                if args.max_episodes_per_source and index + 1 >= args.max_episodes_per_source:
                    break

    obs_features = np.concatenate(all_obs_features, axis=0) if args.cache_latent_features else None
    next_obs_features = (
        np.concatenate(all_next_obs_features, axis=0) if args.cache_latent_features else None
    )
    actions = np.concatenate(all_actions, axis=0)
    returns = np.concatenate(all_returns, axis=0)
    dones = np.concatenate(all_dones, axis=0)
    success = np.concatenate(all_success, axis=0)
    steps = np.concatenate(all_steps, axis=0)
    next_steps = np.concatenate(all_next_steps, axis=0)
    source = np.asarray(all_source, dtype="S")
    demo = np.asarray(all_demo, dtype="S")
    split = split_by_episode(source.astype(str), demo.astype(str), args.seed, args.val_fraction, args.test_fraction)

    action_mean = actions.reshape(actions.shape[0], -1).mean(axis=0).astype(np.float32)
    action_std = actions.reshape(actions.shape[0], -1).std(axis=0).clip(min=1e-6).astype(np.float32)
    reward_mean = float(returns.mean())
    reward_std = float(max(returns.std(), 1e-6))

    payload = dict(
        action_chunks=actions,
        chunk_returns=returns,
        dones=dones,
        episode_success=success,
        steps=steps,
        next_steps=next_steps,
        source=source,
        demo=demo,
        split=split,
        action_mean=action_mean,
        action_std=action_std,
        reward_mean=np.asarray(reward_mean, dtype=np.float32),
        reward_std=np.asarray(reward_std, dtype=np.float32),
        gamma=np.asarray(args.gamma, dtype=np.float32),
        chunk_horizon=np.asarray(args.chunk_horizon, dtype=np.int64),
        observation_horizon=np.asarray(args.observation_horizon, dtype=np.int64),
        action_dim=np.asarray(actions.shape[-1], dtype=np.int64),
        checkpoint=np.asarray(str(args.checkpoint), dtype="S"),
    )
    if args.cache_latent_features:
        payload["obs_features"] = obs_features
        payload["next_obs_features"] = next_obs_features
    np.savez_compressed(args.output, **payload)
    summary = {
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "cache_latent_features": bool(args.cache_latent_features),
        "index_only": not bool(args.cache_latent_features),
        "num_chunks": int(actions.shape[0]),
        "feature_dim": int(obs_features.shape[-1]) if args.cache_latent_features else None,
        "chunk_horizon": int(args.chunk_horizon),
        "transition_mode": "one_step" if int(args.chunk_horizon) == 1 else "chunk",
        "action_dim": int(actions.shape[-1]),
        "gamma": float(args.gamma),
        "stride": int(args.stride),
        "per_source": per_source,
        "split_counts": {k: int(np.sum(split.astype(str) == k)) for k in ("train", "val", "test")},
        "return_stats": {
            "mean": reward_mean,
            "std": reward_std,
            "min": float(returns.min()),
            "max": float(returns.max()),
            "nonzero_fraction": float(np.mean(returns > 0.0)),
        },
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--demo-dataset", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--rollout-dataset", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-filter-key", type=str, default=None)
    parser.add_argument("--success-filter-key", type=str, default="success")
    parser.add_argument("--failure-filter-key", type=str, default="failure")
    parser.add_argument("--include-demos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-rollouts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--chunk-horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-latent-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--max-episodes-per-source", type=int, default=0)
    parser.add_argument("--include-partial-terminal", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for key in ("checkpoint", "demo_dataset", "rollout_dataset", "output"):
        setattr(args, key, getattr(args, key).resolve())
    build(args)


if __name__ == "__main__":
    main()
