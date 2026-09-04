#!/usr/bin/env python3
"""Extract and plot Transport chunk-critic advantage distributions.

The default analysis evaluates every full horizon-8 deployment-rollout chunk
from the success and failure masks of the dataset used by the selected critic.
Extraction is cached in resumable NPZ shards, so plot-only revisions do not
repeat RGB critic inference.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import SparseChunkSequenceDataset

from eval_rgb_dp_idql import validate_rise_dp_composition
from train_rgb_dp_chunk_idql import (
    RISE_V2_CRITIC_ARCHITECTURE,
    checkpoint_critic_architecture,
    make_rise_v2_system_from_checkpoint,
    match_encoder_normalization_to_checkpoint,
    process_chunk_batch,
)
from train_rgb_dp_idql import action_normalization_stats_match


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/transport_rgb_dp/chunk_idql"
    / "200demo_422success_78failure_h8_rise_v2_obs2_film_dense48_"
    "human_success_condi_terminal_success_reward/models/model_epoch_50.pt"
)
DEFAULT_DP_CHECKPOINT = (
    ROOT
    / "trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1"
    / "models/model_epoch_200.pth"
)
DEFAULT_DATASET = (
    ROOT
    / "datasets/transport/idql"
    / "transport_rgb_dp_idql_200demo_422success_78failure_terminal_success.hdf5"
)
DEFAULT_OUTPUT_DIR = ROOT / "analysis/transport_chunk_advantage/epoch50"
DEFAULT_FIGURE_DIR = ROOT / "figures"

CLASS_SPECS = (
    ("success", "non_expert_success", 1),
    ("failure", "non_expert_failure", 0),
)
ARRAY_KEYS = (
    "dataset_index",
    "episode",
    "timestep",
    "label",
    "valid_length",
    "q1",
    "q2",
    "q_min",
    "value",
    "advantage",
)
FLOAT_KEYS = ("q1", "q2", "q_min", "value", "advantage")
COLORS = {"success": "#4C78A8", "failure": "#C58A3A"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2) + "\n")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, include_hash: bool) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        identity["sha256"] = sha256_file(resolved)
    return identity


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def validate_input_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("critic checkpoint", args.checkpoint),
        ("base DP checkpoint", args.dp_checkpoint),
        ("dataset", args.dataset),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def extraction_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint": file_identity(
            args.checkpoint, include_hash=bool(args.hash_inputs)
        ),
        "dp_checkpoint": file_identity(
            args.dp_checkpoint, include_hash=bool(args.hash_inputs)
        ),
        "dataset": file_identity(
            args.dataset, include_hash=bool(args.hash_inputs)
        ),
        "critic_source": str(args.critic_source),
        "include_partial_chunks": bool(args.include_partial_chunks),
        "max_demos_per_class": int(args.max_demos_per_class),
        "class_masks": {
            class_name: mask_name
            for class_name, mask_name, _ in CLASS_SPECS
        },
    }


def prepare_output_directory(
    args: argparse.Namespace,
    signature: dict[str, Any],
) -> Path:
    manifest_path = args.output_dir / "extraction_manifest.json"
    shard_dir = args.output_dir / "shards"
    if args.force_extract and shard_dir.exists():
        shutil.rmtree(shard_dir)
    if args.force_extract:
        for generated in (
            args.output_dir / "chunk_advantages.npz",
            args.output_dir / "summary.json",
            manifest_path,
        ):
            if generated.exists():
                generated.unlink()

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("signature") != jsonable(signature):
            raise RuntimeError(
                "existing extraction parameters differ; use --force-extract "
                "or choose another --output-dir"
            )
    else:
        atomic_write_json(
            manifest_path,
            {
                "status": "running",
                "signature": signature,
                "command": sys.argv,
            },
        )
    shard_dir.mkdir(parents=True, exist_ok=True)
    return manifest_path


def load_critic_system(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Any,
    torch.nn.ModuleList,
    torch.nn.Module,
    dict[str, Any],
    dict[str, str],
]:
    print(f"Loading critic checkpoint: {args.checkpoint}", flush=True)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("task") != "transport":
        raise ValueError(
            f"checkpoint task must be 'transport', got {checkpoint.get('task')!r}"
        )
    if checkpoint_critic_architecture(checkpoint) != RISE_V2_CRITIC_ARCHITECTURE:
        raise ValueError("this analysis requires a RISE-v2 chunk critic")
    if int(checkpoint.get("num_critics", -1)) != 2:
        raise ValueError("advantage analysis requires exactly two Q critics")

    print(f"Loading base DP checkpoint: {args.dp_checkpoint}", flush=True)
    dp_policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.dp_checkpoint),
        device=device,
        verbose=False,
    )
    validate_rise_dp_composition(
        dp_policy,
        dp_checkpoint,
        checkpoint,
        actor_source="external_dp_chunk_critic",
    )

    critics, target_critics, value_net = make_rise_v2_system_from_checkpoint(
        dp_policy.policy,
        checkpoint,
    )
    if args.critic_source == "online":
        selected_critics = critics
        critic_states = checkpoint["critics"]
        del target_critics
    else:
        selected_critics = target_critics
        critic_states = checkpoint["critic_targets"]
        del critics

    normalization_audit = []
    if len(selected_critics) != 2 or len(critic_states) != 2:
        raise ValueError("checkpoint/model twin-critic counts do not match")
    for critic, state in zip(selected_critics, critic_states):
        normalization_audit.append(
            match_encoder_normalization_to_checkpoint(critic, state)
        )
        critic.load_state_dict(state, strict=True)
    value_normalization_audit = match_encoder_normalization_to_checkpoint(
        value_net,
        checkpoint["vf"],
    )
    value_net.load_state_dict(checkpoint["vf"], strict=True)

    selected_critics = selected_critics.float().to(device)
    value_net = value_net.float().to(device)
    selected_critics.eval().requires_grad_(False)
    value_net.eval().requires_grad_(False)
    dp_policy.policy.set_eval()

    action_stats = copy.deepcopy(checkpoint["action_normalization_stats"])
    if not action_normalization_stats_match(
        action_stats,
        dp_checkpoint["action_normalization_stats"],
    ):
        raise RuntimeError("critic and base-DP action normalization differ")

    metadata = {
        "task": checkpoint["task"],
        "epoch": int(checkpoint["epoch"]),
        "step": int(checkpoint["step"]),
        "critic_architecture": checkpoint_critic_architecture(checkpoint),
        "critic_source": str(args.critic_source),
        "chunk_horizon": int(checkpoint["critic_chunk_horizon"]),
        "critic_observation_horizon": int(
            checkpoint["critic_observation_horizon"]
        ),
        "actor_observation_horizon": int(checkpoint["observation_horizon"]),
        "action_dim": int(checkpoint["action_dim"]),
        "critic_action_space": checkpoint["critic_action_space"],
        "reward_mode": checkpoint["reward_mode"],
        "normalization_audit": normalization_audit,
        "value_normalization_audit": value_normalization_audit,
        "action_normalization_stats": action_stats,
    }
    loader_checkpoint = {
        "algo_name": str(dp_checkpoint["algo_name"]),
        "config": str(dp_checkpoint["config"]),
    }

    # Retain only the actor object needed for batch postprocessing. Optimizer,
    # target, and checkpoint states are no longer needed during inference.
    del checkpoint
    del dp_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dp_policy, selected_critics, value_net, metadata, loader_checkpoint


def dataset_config(
    loader_checkpoint: dict[str, str],
    dataset_path: Path,
    mask_name: str,
    actor_algo,
    *,
    chunk_horizon: int,
    include_partial_chunks: bool,
    max_demos_per_class: int,
    hdf5_cache_mode: str | None,
):
    config, _ = FileUtils.config_from_checkpoint(
        ckpt_dict=loader_checkpoint,
        verbose=False,
    )
    ObsUtils.initialize_obs_utils_with_config(config)

    observation_horizon = int(
        actor_algo.algo_config.horizon.observation_horizon
    )
    with config.values_unlocked():
        dataset_entry: dict[str, Any] = {
            "path": str(dataset_path),
            "filter_key": mask_name,
        }
        if max_demos_per_class > 0:
            dataset_entry["demo_limit"] = int(max_demos_per_class)
        config.train.data = [dataset_entry]
        config.train.normalize_weights_by_ds_size = False
        config.train.hdf5_filter_key = None
        config.train.hdf5_validation_filter_key = None
        config.experiment.validate = False
        config.train.hdf5_cache_mode = hdf5_cache_mode
        config.train.hdf5_load_next_obs = True
        config.train.hdf5_normalize_obs = False
        config.train.seq_length = int(chunk_horizon)
        config.train.frame_stack = observation_horizon
        config.train.pad_frame_stack = True
        config.train.pad_seq_length = bool(include_partial_chunks)
        config.train.batch_size = 1
        config.train.num_data_workers = 0
        config.train.dataset_keys = list(
            dict.fromkeys(
                list(config.train.dataset_keys)
                + ["actions", "rewards", "dones"]
            )
        )
    return config


def build_filtered_dataset(
    args: argparse.Namespace,
    dp_policy,
    metadata: dict[str, Any],
    loader_checkpoint: dict[str, str],
    mask_name: str,
):
    config = dataset_config(
        loader_checkpoint,
        args.dataset,
        mask_name,
        dp_policy.policy,
        chunk_horizon=int(metadata["chunk_horizon"]),
        include_partial_chunks=bool(args.include_partial_chunks),
        max_demos_per_class=int(args.max_demos_per_class),
        hdf5_cache_mode=args.hdf5_cache_mode,
    )
    base = TrainUtils.dataset_factory(
        config,
        obs_keys=list(dp_policy.policy.obs_shapes.keys()),
    )
    if base.__class__.__name__ != "SequenceDataset":
        raise TypeError(f"expected SequenceDataset, got {type(base).__name__}")
    base.set_action_normalization_stats(
        copy.deepcopy(metadata["action_normalization_stats"])
    )
    if not action_normalization_stats_match(
        base.get_action_normalization_stats(),
        metadata["action_normalization_stats"],
    ):
        raise RuntimeError("failed to install checkpoint action normalization")
    return SparseChunkSequenceDataset(
        base,
        chunk_horizon=int(metadata["chunk_horizon"]),
        observation_horizon=int(metadata["actor_observation_horizon"]),
        next_observation_horizon=int(metadata["critic_observation_horizon"]),
        dynamics_prediction_offsets=(),
    )


def shard_paths(shard_dir: Path, class_name: str) -> list[Path]:
    return sorted(shard_dir.glob(f"{class_name}_*.npz"))


def resume_index_and_shard_id(
    shard_dir: Path,
    class_name: str,
) -> tuple[int, int]:
    existing = shard_paths(shard_dir, class_name)
    if not existing:
        return 0, 0
    last = existing[-1]
    with np.load(last, allow_pickle=False) as saved:
        indices = saved["dataset_index"]
        if indices.size == 0:
            raise RuntimeError(f"empty extraction shard: {last}")
        resume_index = int(indices[-1]) + 1
    shard_id = int(last.stem.rsplit("_", 1)[-1]) + 1
    return resume_index, shard_id


def concatenate_records(
    records: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not records:
        return {}
    return {
        key: np.concatenate([record[key] for record in records], axis=0)
        for key in ARRAY_KEYS
    }


def split_record(
    record: dict[str, np.ndarray],
    count: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    head = {key: value[:count] for key, value in record.items()}
    tail = {key: value[count:] for key, value in record.items()}
    return head, tail


def save_shard(
    shard_dir: Path,
    class_name: str,
    shard_id: int,
    record: dict[str, np.ndarray],
) -> Path:
    path = shard_dir / f"{class_name}_{shard_id:06d}.npz"
    atomic_save_npz(path, record)
    print(
        f"Saved {path.name}: {len(record['advantage']):,} chunks "
        f"through dataset index {int(record['dataset_index'][-1]):,}",
        flush=True,
    )
    return path


def evaluate_batch(
    raw_batch: dict[str, Any],
    *,
    dp_policy,
    critics: torch.nn.ModuleList,
    value_net: torch.nn.Module,
    metadata: dict[str, Any],
    dataset,
    class_label: int,
) -> dict[str, np.ndarray]:
    indices = raw_batch["index"].detach().cpu().numpy().astype(np.int64)
    episodes = []
    timesteps = []
    for index in indices:
        episode = dataset._index_to_demo_id[int(index)]
        episodes.append(episode)
        timesteps.append(
            int(index) - int(dataset._demo_id_to_start_indices[episode])
        )

    batch = process_chunk_batch(
        raw_batch,
        dp_policy.policy,
        dp_policy.obs_normalization_stats,
        chunk_horizon=int(metadata["chunk_horizon"]),
        discount=0.99,
        reward_mode=str(metadata["reward_mode"]),
        critic_observation_horizon=int(
            metadata["critic_observation_horizon"]
        ),
        dynamics_prediction_offsets=(),
    )
    with torch.inference_mode():
        q_values = [
            critic(
                obs_dict=batch["obs"],
                acts=batch["actions"],
                action_mask=batch["action_mask"],
                goal_dict=batch["goal_obs"],
            ).reshape(-1)
            for critic in critics
        ]
        value = value_net(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        ).reshape(-1)
        q1, q2 = q_values
        q_min = torch.minimum(q1, q2)
        advantage = q_min - value

    tensors = {
        "q1": q1,
        "q2": q2,
        "q_min": q_min,
        "value": value,
        "advantage": advantage,
    }
    arrays = {
        key: tensor.detach().float().cpu().numpy().astype(np.float32)
        for key, tensor in tensors.items()
    }
    arrays.update(
        {
            "dataset_index": indices,
            "episode": np.asarray(episodes, dtype="U32"),
            "timestep": np.asarray(timesteps, dtype=np.int32),
            "label": np.full(len(indices), class_label, dtype=np.uint8),
            "valid_length": batch["valid_length"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.uint8),
        }
    )
    for key in FLOAT_KEYS:
        if not np.isfinite(arrays[key]).all():
            raise FloatingPointError(f"non-finite values in {key}")
    if not np.allclose(arrays["q_min"], np.minimum(arrays["q1"], arrays["q2"])):
        raise RuntimeError("saved q_min does not equal min(q1, q2)")
    if not np.allclose(
        arrays["advantage"],
        arrays["q_min"] - arrays["value"],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("saved advantage does not equal q_min - value")
    return arrays


def extract_class(
    args: argparse.Namespace,
    *,
    dp_policy,
    critics: torch.nn.ModuleList,
    value_net: torch.nn.Module,
    metadata: dict[str, Any],
    loader_checkpoint: dict[str, str],
    class_name: str,
    mask_name: str,
    class_label: int,
) -> dict[str, Any]:
    dataset = build_filtered_dataset(
        args,
        dp_policy,
        metadata,
        loader_checkpoint,
        mask_name,
    )
    shard_dir = args.output_dir / "shards"
    resume_index, shard_id = resume_index_and_shard_id(
        shard_dir,
        class_name,
    )
    total = len(dataset)
    if resume_index > total:
        raise RuntimeError(
            f"resume index {resume_index} exceeds {class_name} dataset size {total}"
        )
    expected_min_valid = (
        1 if args.include_partial_chunks else int(metadata["chunk_horizon"])
    )
    print(
        f"{class_name}: {dataset.n_demos} episodes, {total:,} chunks; "
        f"resuming at {resume_index:,}",
        flush=True,
    )
    if resume_index == total:
        return {
            "class": class_name,
            "episodes": int(dataset.n_demos),
            "chunks": int(total),
            "resumed_complete": True,
        }

    sampler = range(resume_index, total)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.num_workers > 0),
        drop_last=False,
    )
    buffered: list[dict[str, np.ndarray]] = []
    buffered_count = 0
    for raw_batch in loader:
        record = evaluate_batch(
            raw_batch,
            dp_policy=dp_policy,
            critics=critics,
            value_net=value_net,
            metadata=metadata,
            dataset=dataset,
            class_label=class_label,
        )
        if int(record["valid_length"].min()) < expected_min_valid:
            raise RuntimeError("chunk valid length violates extraction mode")
        buffered.append(record)
        buffered_count += len(record["advantage"])
        while buffered_count >= int(args.shard_size):
            merged = concatenate_records(buffered)
            head, tail = split_record(merged, int(args.shard_size))
            save_shard(shard_dir, class_name, shard_id, head)
            shard_id += 1
            buffered = [tail] if len(tail["advantage"]) else []
            buffered_count = len(tail["advantage"])

    if buffered_count:
        merged = concatenate_records(buffered)
        save_shard(shard_dir, class_name, shard_id, merged)
    return {
        "class": class_name,
        "episodes": int(dataset.n_demos),
        "chunks": int(total),
        "resumed_complete": False,
    }


def consolidate_shards(output_dir: Path) -> dict[str, np.ndarray]:
    shards = []
    for class_name, _, _ in CLASS_SPECS:
        paths = shard_paths(output_dir / "shards", class_name)
        if not paths:
            raise RuntimeError(f"no completed shards found for {class_name}")
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                shards.append({key: saved[key] for key in ARRAY_KEYS})
    merged = concatenate_records(shards)
    order = np.lexsort(
        (
            merged["dataset_index"],
            1 - merged["label"].astype(np.int16),
        )
    )
    merged = {key: value[order] for key, value in merged.items()}
    atomic_save_npz(output_dir / "chunk_advantages.npz", merged)
    return merged


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    quantile_levels = np.asarray([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    quantiles = np.quantile(values, quantile_levels)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def probability_greater(left: np.ndarray, right: np.ndarray) -> float:
    """Return P(left > right) + 0.5 P(left == right) without O(N*M)."""
    ordered = np.sort(right)
    strictly_less = np.searchsorted(ordered, left, side="left")
    less_or_equal = np.searchsorted(ordered, left, side="right")
    ties = less_or_equal - strictly_less
    wins = strictly_less.astype(np.float64) + 0.5 * ties.astype(np.float64)
    return float(wins.mean() / float(len(right)))


def episode_medians(
    episodes: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for episode, value in zip(episodes.tolist(), values.tolist()):
        grouped[str(episode)].append(float(value))
    return np.asarray(
        [np.median(grouped[key]) for key in sorted(grouped)],
        dtype=np.float64,
    )


def episode_bootstrap(
    success_episode_medians: np.ndarray,
    failure_episode_medians: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        success = rng.choice(
            success_episode_medians,
            size=len(success_episode_medians),
            replace=True,
        )
        failure = rng.choice(
            failure_episode_medians,
            size=len(failure_episode_medians),
            replace=True,
        )
        differences[index] = float(success.mean() - failure.mean())
    return {
        "estimand": "difference in mean per-episode median advantage",
        "point_estimate": float(
            success_episode_medians.mean() - failure_episode_medians.mean()
        ),
        "samples": int(samples),
        "seed": int(seed),
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def summarize_results(
    arrays: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any] | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    success_mask = arrays["label"] == 1
    failure_mask = arrays["label"] == 0
    success = arrays["advantage"][success_mask].astype(np.float64)
    failure = arrays["advantage"][failure_mask].astype(np.float64)
    if not success.size or not failure.size:
        raise RuntimeError("both success and failure chunks are required")

    success_ep = episode_medians(arrays["episode"][success_mask], success)
    failure_ep = episode_medians(arrays["episode"][failure_mask], failure)
    auc = probability_greater(success, failure)
    summary = {
        "formula": "min(Q1(z,a), Q2(z,a)) - V(z)",
        "metadata": metadata,
        "success": {
            "episodes": int(len(np.unique(arrays["episode"][success_mask]))),
            "advantage": distribution_summary(success),
            "q_min": distribution_summary(
                arrays["q_min"][success_mask].astype(np.float64)
            ),
            "value": distribution_summary(
                arrays["value"][success_mask].astype(np.float64)
            ),
        },
        "failure": {
            "episodes": int(len(np.unique(arrays["episode"][failure_mask]))),
            "advantage": distribution_summary(failure),
            "q_min": distribution_summary(
                arrays["q_min"][failure_mask].astype(np.float64)
            ),
            "value": distribution_summary(
                arrays["value"][failure_mask].astype(np.float64)
            ),
        },
        "comparison": {
            "mean_difference_success_minus_failure": float(
                success.mean() - failure.mean()
            ),
            "median_difference_success_minus_failure": float(
                np.median(success) - np.median(failure)
            ),
            "probability_success_greater_than_failure": auc,
            "cliffs_delta": float(2.0 * auc - 1.0),
            "episode_bootstrap": episode_bootstrap(
                success_ep,
                failure_ep,
                samples=int(bootstrap_samples),
                seed=int(bootstrap_seed),
            ),
        },
    }
    return summary


def load_cached_arrays(output_dir: Path) -> dict[str, np.ndarray]:
    cache_path = output_dir / "chunk_advantages.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"advantage cache does not exist; run --mode extract or all: {cache_path}"
        )
    with np.load(cache_path, allow_pickle=False) as saved:
        missing = [key for key in ARRAY_KEYS if key not in saved]
        if missing:
            raise KeyError(f"advantage cache is missing arrays: {missing}")
        return {key: saved[key] for key in ARRAY_KEYS}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)


def central_plot_data(
    success: np.ndarray,
    failure: np.ndarray,
    tail_quantile: float,
) -> tuple[list[np.ndarray], tuple[float, float], str | None]:
    distributions = [success, failure]
    pooled = np.concatenate(distributions)
    if tail_quantile <= 0.0:
        return distributions, (float(pooled.min()), float(pooled.max())), None

    lower, upper = np.quantile(
        pooled,
        [tail_quantile, 1.0 - tail_quantile],
    ).tolist()
    central = [
        values[(values >= lower) & (values <= upper)]
        for values in distributions
    ]
    percentile = 100.0 * tail_quantile
    note = f"Pooled {percentile:g}th–{100.0 - percentile:g}th percentile shown"
    return central, (float(lower), float(upper)), note


def add_range_note(ax: plt.Axes, note: str | None) -> None:
    if note is None:
        return
    ax.text(
        0.985,
        0.975,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color="#4a4a4a",
    )


def plot_violin(
    success: np.ndarray,
    failure: np.ndarray,
    figure_dir: Path,
    stem: str,
    *,
    tail_quantile: float,
) -> tuple[list[Path], tuple[float, float]]:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.65), constrained_layout=True)
    distributions, display_range, range_note = central_plot_data(
        success,
        failure,
        tail_quantile,
    )
    positions = [1, 2]
    violin = ax.violinplot(
        distributions,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=160,
        bw_method="scott",
    )
    for body, color in zip(
        violin["bodies"],
        (COLORS["success"], COLORS["failure"]),
    ):
        body.set_facecolor(color)
        body.set_edgecolor("#202020")
        body.set_linewidth(0.8)
        body.set_alpha(0.78)

    ax.boxplot(
        distributions,
        positions=positions,
        widths=0.14,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#202020", "linewidth": 0.9},
        medianprops={"color": "#202020", "linewidth": 1.4},
        whiskerprops={"color": "#202020", "linewidth": 0.9},
        capprops={"color": "#202020", "linewidth": 0.9},
    )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_ylabel(r"Critic advantage  $\min(Q_1,Q_2)-V$")
    ax.set_xticks(
        positions,
        (
            f"Success\n($n$={len(success):,})",
            f"Failure\n($n$={len(failure):,})",
        ),
    )
    ax.set_xlim(0.45, 2.55)
    if range_note is not None:
        span = display_range[1] - display_range[0]
        margin = max(0.05 * span, 1e-6)
        ax.set_ylim(display_range[0] - margin, display_range[1] + margin)
    add_range_note(ax, range_note)
    style_axis(ax)

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths, display_range


def histogram_bin_count(values: np.ndarray) -> int:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    span = float(np.max(values) - np.min(values))
    if iqr <= 0.0 or span <= 0.0:
        return 40
    width = 2.0 * iqr / np.cbrt(float(len(values)))
    if width <= 0.0:
        return 40
    return int(np.clip(math.ceil(span / width), 30, 120))


def plot_histogram(
    success: np.ndarray,
    failure: np.ndarray,
    figure_dir: Path,
    stem: str,
    *,
    tail_quantile: float,
) -> tuple[list[Path], tuple[float, float]]:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.65), constrained_layout=True)
    distributions, display_range, range_note = central_plot_data(
        success,
        failure,
        tail_quantile,
    )
    pooled = np.concatenate(distributions)
    bins = np.linspace(
        display_range[0],
        display_range[1],
        histogram_bin_count(pooled) + 1,
    )
    for values, label, color in (
        (distributions[0], "Success rollouts", COLORS["success"]),
        (distributions[1], "Failure rollouts", COLORS["failure"]),
    ):
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.24,
            color=color,
            linewidth=0.0,
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=1.4,
            label=label,
        )
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Critic advantage  $\min(Q_1,Q_2)-V$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_xlim(display_range)
    add_range_note(ax, range_note)
    style_axis(ax)

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths, display_range


def plot_results(args: argparse.Namespace) -> tuple[dict[str, Any], list[Path]]:
    arrays = load_cached_arrays(args.output_dir)
    success = arrays["advantage"][arrays["label"] == 1].astype(np.float64)
    failure = arrays["advantage"][arrays["label"] == 0].astype(np.float64)
    paths = []
    violin_paths, central_range = plot_violin(
        success,
        failure,
        args.figure_dir,
        "transport_chunk_advantage_violin",
        tail_quantile=float(args.plot_tail_quantile),
    )
    paths.extend(violin_paths)
    histogram_paths, histogram_range = plot_histogram(
        success,
        failure,
        args.figure_dir,
        "transport_chunk_advantage_histogram",
        tail_quantile=float(args.plot_tail_quantile),
    )
    paths.extend(histogram_paths)

    if args.plot_tail_quantile > 0.0:
        full_violin_paths, full_range = plot_violin(
            success,
            failure,
            args.figure_dir,
            "transport_chunk_advantage_violin_full_range",
            tail_quantile=0.0,
        )
        paths.extend(full_violin_paths)
        full_histogram_paths, _ = plot_histogram(
            success,
            failure,
            args.figure_dir,
            "transport_chunk_advantage_histogram_full_range",
            tail_quantile=0.0,
        )
        paths.extend(full_histogram_paths)
    else:
        full_range = central_range

    metadata = None
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        metadata = json.loads(summary_path.read_text()).get("metadata")
    summary = summarize_results(
        arrays,
        metadata=metadata,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    summary["plotting"] = {
        "central_tail_quantile": float(args.plot_tail_quantile),
        "central_violin_range": list(central_range),
        "central_histogram_range": list(histogram_range),
        "full_data_range": list(full_range),
    }
    atomic_write_json(summary_path, summary)
    return summary, paths


def dataset_mask_audit(dataset_path: Path) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as dataset:
        masks = {
            mask_name: int(len(dataset[f"mask/{mask_name}"]))
            for _, mask_name, _ in CLASS_SPECS
        }
        transitions = {}
        for class_name, mask_name, _ in CLASS_SPECS:
            demos = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in dataset[f"mask/{mask_name}"][:]
            ]
            transitions[class_name] = int(
                sum(
                    int(dataset[f"data/{demo}"].attrs["num_samples"])
                    for demo in demos
                )
            )
    return {"episodes": masks, "transitions": transitions}


def extract_results(args: argparse.Namespace) -> dict[str, Any]:
    validate_input_paths(args)
    signature = extraction_signature(args)
    manifest_path = prepare_output_directory(args, signature)
    device = resolve_device(args.device)
    print(f"Inference device: {device}", flush=True)
    mask_audit = dataset_mask_audit(args.dataset)
    print(f"Dataset masks: {mask_audit}", flush=True)

    (
        dp_policy,
        critics,
        value_net,
        metadata,
        loader_checkpoint,
    ) = load_critic_system(
        args,
        device,
    )
    class_audits = []
    for class_name, mask_name, class_label in CLASS_SPECS:
        class_audits.append(
            extract_class(
                args,
                dp_policy=dp_policy,
                critics=critics,
                value_net=value_net,
                metadata=metadata,
                loader_checkpoint=loader_checkpoint,
                class_name=class_name,
                mask_name=mask_name,
                class_label=class_label,
            )
        )

    arrays = consolidate_shards(args.output_dir)
    expected_chunks = sum(int(item["chunks"]) for item in class_audits)
    if len(arrays["advantage"]) != expected_chunks:
        raise RuntimeError(
            f"consolidated {len(arrays['advantage'])} chunks, expected {expected_chunks}"
        )
    summary = summarize_results(
        arrays,
        metadata={
            **metadata,
            "device": str(device),
            "include_partial_chunks": bool(args.include_partial_chunks),
            "dataset_mask_audit": mask_audit,
            "class_extraction_audit": class_audits,
        },
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(
        manifest_path,
        {
            "status": "complete",
            "signature": signature,
            "command": sys.argv,
            "cache": str((args.output_dir / "chunk_advantages.npz").resolve()),
            "summary": str((args.output_dir / "summary.json").resolve()),
            "chunks": int(len(arrays["advantage"])),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract", "plot", "all"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dp-checkpoint", type=Path, default=DEFAULT_DP_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--critic-source", choices=("online", "target"), default="online")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--hdf5-cache-mode", choices=("low_dim", "all", "None"), default="low_dim")
    parser.add_argument("--include-partial-chunks", action="store_true")
    parser.add_argument("--max-demos-per-class", type=int, default=0)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--plot-tail-quantile",
        type=float,
        default=0.01,
        help=(
            "tail fraction omitted from each side of the primary central-view "
            "plots; full-range plots are also saved (default: 0.01)"
        ),
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0 or args.shard_size <= 0:
        parser.error("batch size and shard size must be positive; workers cannot be negative")
    if args.max_demos_per_class < 0:
        parser.error("--max-demos-per-class cannot be negative")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if not 0.0 <= args.plot_tail_quantile < 0.25:
        parser.error("--plot-tail-quantile must be in [0, 0.25)")
    if args.hdf5_cache_mode == "None":
        args.hdf5_cache_mode = None
    return args


def main() -> None:
    args = parse_args()
    summary = None
    if args.mode in ("extract", "all"):
        summary = extract_results(args)
    if args.mode in ("plot", "all"):
        summary, figure_paths = plot_results(args)
        for path in figure_paths:
            print(f"Saved {path}", flush=True)
    if summary is not None:
        comparison = summary["comparison"]
        print(
            "Advantage comparison: "
            f"median difference={comparison['median_difference_success_minus_failure']:.6f}, "
            f"P(success > failure)={comparison['probability_success_greater_than_failure']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
