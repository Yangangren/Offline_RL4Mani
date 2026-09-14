#!/usr/bin/env python3
"""Label chunk-actor rows from a frozen learned Q / V checkpoint.

The emitted sidecar is indexed by the underlying ``SequenceDataset`` index.
Human demonstrations are always condition one. Rollout chunks above and below
two calibrated score thresholds receive active one and zero conditions,
respectively; the uncertain middle band receives the explicit null condition
``(condition=0, mask=0)``.

``relative_q`` instead ranks each rollout chunk against deterministic,
independent alternatives sampled from the frozen pretrained DP at the same
observation. Human demonstrations remain forced positive and are not sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo.diffusion_policy import (
    DETERMINISTIC_DP_PROPOSAL_RNG_SCHEME,
)

from eval_rgb_dp_idql import validate_rise_dp_composition
from rgb_dp_critic_conditions import (
    calibrate_critic_condition_thresholds,
    compute_relative_q_ranks,
    construct_critic_conditions,
    construct_relative_q_conditions,
)
from train_rgb_dp_chunk_idql import (
    CRITIC_ACTOR_CONDITION_SIDECAR_KIND,
    RISE_V2_CRITIC_ARCHITECTURE,
    WCM_CRITIC_ARCHITECTURE,
    atomic_write_json,
    build_single_loader,
    checkpoint_critic_architecture,
    file_stat_identity,
    make_rise_v2_system_from_checkpoint,
    make_wcm_system_from_checkpoint,
    match_encoder_normalization_to_checkpoint,
    mixed_dataset_identity,
    process_chunk_batch,
    validate_mixed_dataset_source_identity,
)
from train_rgb_dp_idql import atomic_torch_save, dataset_audit


LABEL_FORMAT = CRITIC_ACTOR_CONDITION_SIDECAR_KIND
SUPPORTED_ARCHITECTURES = (
    RISE_V2_CRITIC_ARCHITECTURE,
    WCM_CRITIC_ARCHITECTURE,
)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dataset_rng_namespace(identity: dict[str, Any]) -> int:
    """Derive a stable signed-63-bit namespace from immutable dataset identity."""
    encoded = json.dumps(
        _jsonable(identity), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _proposal_scheduler_metadata(actor_algo) -> dict[str, Any]:
    scheduler = actor_algo.noise_scheduler
    if bool(actor_algo.algo_config.ddpm.enabled):
        sampler = "ddpm"
        inference_steps = int(
            actor_algo.algo_config.ddpm.num_inference_timesteps
        )
    elif bool(actor_algo.algo_config.ddim.enabled):
        sampler = "ddim"
        inference_steps = int(
            actor_algo.algo_config.ddim.num_inference_timesteps
        )
    else:
        raise ValueError("pretrained DP has no enabled inference scheduler")
    return {
        "sampler": sampler,
        "class": type(scheduler).__name__,
        "num_inference_timesteps": inference_steps,
        "config": _jsonable(dict(scheduler.config)),
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "device_type": str(actor_algo.device.type),
        "device_name": (
            torch.cuda.get_device_name(actor_algo.device)
            if actor_algo.device.type == "cuda"
            else "cpu"
        ),
    }


def _base_sequence_dataset(dataset):
    base = getattr(dataset, "dataset", dataset)
    if base.__class__.__name__ != "SequenceDataset":
        raise TypeError(
            "critic condition labeling requires an underlying SequenceDataset, "
            f"got {type(base).__name__}"
        )
    return base


def _episode_sources(dataset_path: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    with h5py.File(dataset_path, "r") as handle:
        data = handle.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError(f"dataset {dataset_path} has no data group")
        for episode_key, episode in data.items():
            source = _decode(episode.attrs.get("rise_source", ""))
            if source not in {
                "expert",
                "non_expert_success",
                "non_expert_failure",
            }:
                raise ValueError(
                    f"data/{episode_key} has unsupported rise_source={source!r}; "
                    "rebuild the mixed IDQL dataset"
                )
            sources[str(episode_key)] = source
    return sources


def _actor_has_condition_adapter(actor_algo) -> bool:
    if "condition_adapter" in actor_algo.nets["policy"]:
        return True
    return bool(
        actor_algo.ema is not None
        and "condition_adapter" in actor_algo.ema.averaged_model["policy"]
    )


def _loader_args(
    args: argparse.Namespace,
    dataset: Path,
    metadata: dict[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        hdf5_cache_mode=str(args.hdf5_cache_mode),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
        reward_mode=str(metadata["reward_mode"]),
        conditioned_actor=False,
        sparse_chunk_loader=bool(args.sparse_chunk_loader),
        sparse_one_step_loader=False,
        sparse_dql_loader=False,
        chunk_horizon=int(metadata["chunk_horizon"]),
        critic_observation_horizon=int(metadata["critic_observation_horizon"]),
        dynamics_prediction_offsets=(),
        distributed_world_size=1,
        distributed_rank=0,
        seed=int(args.seed),
    )


def _load_critic(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(
        args.critic_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or not checkpoint.get(
        "rise_style_rgb_chunk_idql", False
    ):
        raise ValueError("critic checkpoint is not a chunk-IDQL checkpoint")
    if bool(checkpoint.get("actor_only", False)) or not bool(
        checkpoint.get("critic_trained", True)
    ):
        raise ValueError("critic condition labeling requires a trained critic")
    if str(checkpoint.get("task", "")) != str(args.task):
        raise ValueError(
            f"critic task={checkpoint.get('task')!r} does not match "
            f"requested task={args.task!r}"
        )
    architecture = checkpoint_critic_architecture(checkpoint)
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "critic condition labeling currently supports "
            f"{SUPPORTED_ARCHITECTURES}, got {architecture!r}"
        )

    policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.dp_checkpoint),
        device=torch.device(args.device),
        verbose=False,
    )
    validate_rise_dp_composition(
        policy,
        dp_checkpoint,
        checkpoint,
        actor_source="external_dp_chunk_critic",
    )
    policy.policy.set_eval()
    policy.policy.nets.requires_grad_(False)
    if policy.policy.ema is not None:
        policy.policy.ema.averaged_model.eval().requires_grad_(False)
    if (
        str(args.condition_mode) == "relative_q"
        and _actor_has_condition_adapter(policy.policy)
    ):
        raise ValueError(
            "relative_q proposals must come from an unconditioned pretrained "
            "DP checkpoint, not an already conditioned actor"
        )

    normalization: dict[str, Any]
    if architecture == RISE_V2_CRITIC_ARCHITECTURE:
        critics, target_critics, value = make_rise_v2_system_from_checkpoint(
            policy.policy, checkpoint
        )
        selected = critics if args.critic_source == "online" else target_critics
        state_key = "critics" if args.critic_source == "online" else "critic_targets"
        critic_states = checkpoint[state_key]
        if len(selected) != len(critic_states) or len(selected) < 2:
            raise ValueError("critic checkpoint has an invalid Q ensemble")
        q_normalization = []
        for critic, state in zip(selected, critic_states):
            q_normalization.append(
                match_encoder_normalization_to_checkpoint(critic, state)
            )
            critic.load_state_dict(state, strict=True)
        value_normalization = match_encoder_normalization_to_checkpoint(
            value, checkpoint["vf"]
        )
        value.load_state_dict(checkpoint["vf"], strict=True)
        selected = selected.float().to(args.device).eval().requires_grad_(False)
        value = value.float().to(args.device).eval().requires_grad_(False)
        model = {
            "architecture": architecture,
            "q": selected,
            "value": value,
        }
        normalization = {
            "q": q_normalization,
            "value": value_normalization,
        }
        del critics, target_critics
    else:
        online, target = make_wcm_system_from_checkpoint(policy.policy, checkpoint)
        online_normalization = match_encoder_normalization_to_checkpoint(
            online, checkpoint["chunk_value_system"]
        )
        online.load_state_dict(checkpoint["chunk_value_system"], strict=True)
        target_normalization = match_encoder_normalization_to_checkpoint(
            target, checkpoint["chunk_value_target"]
        )
        target.load_state_dict(checkpoint["chunk_value_target"], strict=True)
        online = online.float().to(args.device).eval().requires_grad_(False)
        target = target.float().to(args.device).eval().requires_grad_(False)
        model = {
            "architecture": architecture,
            "q": online if args.critic_source == "online" else target,
            "value": online,
        }
        normalization = {
            "online": online_normalization,
            "target": target_normalization,
        }

    metadata = {
        "architecture": architecture,
        "critic_source": str(args.critic_source),
        "chunk_horizon": int(checkpoint["critic_chunk_horizon"]),
        "critic_observation_horizon": int(
            checkpoint["critic_observation_horizon"]
        ),
        "reward_mode": str(checkpoint["reward_mode"]),
        "num_critics": int(checkpoint["num_critics"]),
        "normalization": normalization,
    }
    if str(args.condition_mode) == "relative_q":
        if int(metadata["num_critics"]) != 2:
            raise ValueError(
                "relative_q requires exactly two Q critics for conservative "
                "min(Q1,Q2) scoring"
            )
        actor_algo = policy.policy
        metadata.update(
            {
                "actor_observation_horizon": int(
                    actor_algo.algo_config.horizon.observation_horizon
                ),
                "proposal_action_horizon": int(
                    actor_algo.algo_config.horizon.action_horizon
                ),
                "proposal_actor_source": (
                    "pretrained_dp_checkpoint.ema"
                    if actor_algo.ema is not None
                    else "pretrained_dp_checkpoint.nets"
                ),
                "proposal_scheduler": _proposal_scheduler_metadata(actor_algo),
                "proposal_rng_scheme": DETERMINISTIC_DP_PROPOSAL_RNG_SCHEME,
                "proposal_action_transform": (
                    "raw_normalized_dp_action_chunk_no_extra_clamp"
                ),
                "action_mask_policy": (
                    "logged_terminal_prefix_mask_for_logged;"
                    "all_ones_for_proposals"
                ),
            }
        )
        if int(metadata["proposal_action_horizon"]) < int(
            metadata["chunk_horizon"]
        ):
            raise ValueError(
                "pretrained DP action horizon is shorter than the critic chunk "
                f"horizon: {metadata['proposal_action_horizon']} < "
                f"{metadata['chunk_horizon']}"
            )
    model["dp_checkpoint"] = dp_checkpoint
    return policy, model, metadata


@torch.no_grad()
def _score_batch(
    raw_batch: dict[str, Any],
    *,
    policy,
    model: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = process_chunk_batch(
        raw_batch,
        policy.policy,
        policy.obs_normalization_stats,
        chunk_horizon=int(metadata["chunk_horizon"]),
        discount=0.99,
        reward_mode=str(metadata["reward_mode"]),
        critic_observation_horizon=int(
            metadata["critic_observation_horizon"]
        ),
        dynamics_prediction_offsets=(),
    )
    if model["architecture"] == RISE_V2_CRITIC_ARCHITECTURE:
        q_values = [
            critic(
                obs_dict=batch["obs"],
                acts=batch["actions"],
                action_mask=batch["action_mask"],
                goal_dict=batch.get("goal_obs"),
            ).reshape(-1)
            for critic in model["q"]
        ]
        value = model["value"](
            obs_dict=batch["obs"], goal_dict=batch.get("goal_obs")
        ).reshape(-1)
    else:
        q_state = model["q"].encode_state(
            batch["obs"], batch.get("goal_obs")
        )
        q_values = [
            item.reshape(-1)
            for item in model["q"].q_values_from_state(
                q_state, batch["actions"], batch["action_mask"]
            )
        ]
        value_state = model["value"].encode_state(
            batch["obs"], batch.get("goal_obs")
        )
        value = model["value"].value_from_state(value_state).reshape(-1)
    stacked_q = torch.stack(q_values, dim=1)
    q_min = stacked_q.min(dim=1).values
    twin_gap = stacked_q.max(dim=1).values - q_min
    advantage = q_min - value
    packed = torch.stack((q_min, value, advantage, twin_gap), dim=1)
    if not torch.isfinite(packed).all():
        raise FloatingPointError("critic condition scores contain non-finite values")
    return q_min, value, advantage, twin_gap


def _prepare_actor_observations(raw_batch: dict[str, Any], *, policy):
    """Prepare the actor history ending at the same t used by the critic."""
    actor_algo = policy.policy
    observation_horizon = int(
        actor_algo.algo_config.horizon.observation_horizon
    )
    obs: dict[str, torch.Tensor] = {}
    for key in actor_algo.obs_shapes:
        value = raw_batch["obs"][key]
        if value.ndim < 2 or int(value.shape[1]) < observation_horizon:
            raise ValueError(
                f"actor observation {key!r} does not contain "
                f"history={observation_horizon}: shape={tuple(value.shape)}"
            )
        obs[key] = value[:, :observation_horizon]
    prepared: dict[str, Any] = {"obs": obs}
    if raw_batch.get("goal_obs") is not None:
        prepared["goal_obs"] = raw_batch["goal_obs"]
    prepared = TensorUtils.to_float(
        TensorUtils.to_device(
            prepared,
            actor_algo.device,
            non_blocking=actor_algo.device.type == "cuda",
        )
    )
    prepared = actor_algo.postprocess_batch_for_training(
        prepared,
        obs_normalization_stats=policy.obs_normalization_stats,
    )
    return prepared["obs"], prepared.get("goal_obs")


def _index_tensor_dict(
    values: dict[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        key: value.index_select(0, indices.to(device=value.device))
        for key, value in values.items()
    }


def _repeat_state_rows(
    state: dict[str, torch.Tensor], repeats: int
) -> dict[str, torch.Tensor]:
    return {
        key: value.repeat_interleave(int(repeats), dim=0)
        for key, value in state.items()
    }


def _state_block(
    state: dict[str, torch.Tensor], start: int, end: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in state.items()}


@torch.no_grad()
def _score_relative_q_batch(
    raw_batch: dict[str, Any],
    *,
    policy,
    model: dict[str, Any],
    metadata: dict[str, Any],
    rollout_positions: np.ndarray,
    row_ids: np.ndarray,
    num_alternatives: int,
    candidate_batch_size: int,
    seed: int,
    rng_namespace: int,
) -> dict[str, torch.Tensor | int]:
    """Score logged chunks and state-matched frozen-DP alternatives."""
    batch = process_chunk_batch(
        raw_batch,
        policy.policy,
        policy.obs_normalization_stats,
        chunk_horizon=int(metadata["chunk_horizon"]),
        discount=0.99,
        reward_mode=str(metadata["reward_mode"]),
        critic_observation_horizon=int(
            metadata["critic_observation_horizon"]
        ),
        dynamics_prediction_offsets=(),
    )
    device = batch["actions"].device
    rollout_indices = torch.as_tensor(
        rollout_positions, device=device, dtype=torch.long
    )
    rollout_count = int(rollout_indices.numel())
    proposal_actions = batch["actions"].new_empty(
        (
            rollout_count,
            int(num_alternatives),
            int(metadata["chunk_horizon"]),
            int(batch["actions"].shape[-1]),
        )
    )
    out_of_range_elements = 0
    if rollout_count:
        actor_obs, actor_goal = _prepare_actor_observations(
            raw_batch, policy=policy
        )
        actor_obs = _index_tensor_dict(actor_obs, rollout_indices)
        if actor_goal is not None:
            actor_goal = _index_tensor_dict(actor_goal, rollout_indices)
        sampled = policy.policy.sample_deterministic_action_trajectory_candidates(
            actor_obs,
            row_ids=torch.as_tensor(row_ids, dtype=torch.int64),
            num_candidates=int(num_alternatives),
            candidate_batch_size=int(candidate_batch_size),
            seed=int(seed),
            goal_dict=actor_goal,
            rng_namespace=int(rng_namespace),
        )
        expected_prefix = (
            rollout_count,
            int(num_alternatives),
        )
        if sampled.ndim != 4 or tuple(sampled.shape[:2]) != expected_prefix:
            raise ValueError(
                "pretrained DP proposal sampler returned invalid shape: "
                f"{tuple(sampled.shape)}"
            )
        if int(sampled.shape[2]) < int(metadata["chunk_horizon"]):
            raise ValueError(
                "pretrained DP proposal horizon is shorter than the critic "
                "chunk horizon"
            )
        sampled = sampled[:, :, : int(metadata["chunk_horizon"])]
        if int(sampled.shape[-1]) != int(batch["actions"].shape[-1]):
            raise ValueError(
                "pretrained DP and critic action dimensions do not match"
            )
        if not torch.isfinite(sampled).all():
            raise FloatingPointError("pretrained DP proposals contain non-finite values")
        out_of_range_elements = int(
            ((sampled < -1.0) | (sampled > 1.0)).sum().item()
        )
        proposal_actions.copy_(sampled)

    q_logged_heads: list[torch.Tensor] = []
    q_proposal_heads: list[torch.Tensor] = []
    if model["architecture"] == RISE_V2_CRITIC_ARCHITECTURE:
        for critic in model["q"]:
            state = critic.encode_state(batch["obs"], batch.get("goal_obs"))
            q_logged_heads.append(
                critic.q_from_state(
                    state,
                    batch["actions"],
                    batch["action_mask"],
                ).reshape(-1)
            )
            if rollout_count:
                rollout_state = _index_tensor_dict(state, rollout_indices)
                expanded_state = _repeat_state_rows(
                    rollout_state, int(num_alternatives)
                )
                flat_actions = proposal_actions.reshape(
                    rollout_count * int(num_alternatives),
                    int(metadata["chunk_horizon"]),
                    int(proposal_actions.shape[-1]),
                )
                flat_mask = batch["action_mask"].new_ones(
                    (
                        rollout_count * int(num_alternatives),
                        int(metadata["chunk_horizon"]),
                    )
                )
                blocks = []
                for start in range(
                    0,
                    int(flat_actions.shape[0]),
                    int(candidate_batch_size),
                ):
                    end = min(
                        int(flat_actions.shape[0]),
                        start + int(candidate_batch_size),
                    )
                    blocks.append(
                        critic.q_from_state(
                            _state_block(expanded_state, start, end),
                            flat_actions[start:end],
                            flat_mask[start:end],
                        ).reshape(-1)
                    )
                q_proposal_heads.append(
                    torch.cat(blocks).reshape(
                        rollout_count, int(num_alternatives)
                    )
                )
        value = model["value"](
            obs_dict=batch["obs"], goal_dict=batch.get("goal_obs")
        ).reshape(-1)
    else:
        q_state = model["q"].encode_state(
            batch["obs"], batch.get("goal_obs")
        )
        q_logged_heads = [
            item.reshape(-1)
            for item in model["q"].q_values_from_state(
                q_state, batch["actions"], batch["action_mask"]
            )
        ]
        if rollout_count:
            rollout_state = _index_tensor_dict(q_state, rollout_indices)
            expanded_state = _repeat_state_rows(
                rollout_state, int(num_alternatives)
            )
            flat_actions = proposal_actions.reshape(
                rollout_count * int(num_alternatives),
                int(metadata["chunk_horizon"]),
                int(proposal_actions.shape[-1]),
            )
            flat_mask = batch["action_mask"].new_ones(
                (
                    rollout_count * int(num_alternatives),
                    int(metadata["chunk_horizon"]),
                )
            )
            per_head_blocks: list[list[torch.Tensor]] = [
                [] for _ in q_logged_heads
            ]
            for start in range(
                0,
                int(flat_actions.shape[0]),
                int(candidate_batch_size),
            ):
                end = min(
                    int(flat_actions.shape[0]),
                    start + int(candidate_batch_size),
                )
                predictions = model["q"].q_values_from_state(
                    _state_block(expanded_state, start, end),
                    flat_actions[start:end],
                    flat_mask[start:end],
                )
                if len(predictions) != len(per_head_blocks):
                    raise ValueError("WCM Q head count changed while scoring proposals")
                for target, prediction in zip(per_head_blocks, predictions):
                    target.append(prediction.reshape(-1))
            q_proposal_heads = [
                torch.cat(blocks).reshape(
                    rollout_count, int(num_alternatives)
                )
                for blocks in per_head_blocks
            ]
        value_state = model["value"].encode_state(
            batch["obs"], batch.get("goal_obs")
        )
        value = model["value"].value_from_state(value_state).reshape(-1)

    if len(q_logged_heads) != 2:
        raise ValueError(
            f"relative_q requires exactly two Q predictions, got {len(q_logged_heads)}"
        )
    stacked_logged = torch.stack(q_logged_heads, dim=1)
    q_min = stacked_logged.min(dim=1).values
    twin_gap = stacked_logged.max(dim=1).values - q_min
    advantage = q_min - value
    if rollout_count:
        if len(q_proposal_heads) != 2:
            raise ValueError(
                "relative_q proposal scoring did not produce two Q predictions"
            )
        proposal_q_min = torch.stack(q_proposal_heads, dim=2).min(dim=2).values
        rank_result = compute_relative_q_ranks(
            q_min.index_select(0, rollout_indices).detach().cpu().numpy(),
            proposal_q_min.detach().cpu().numpy(),
        )
        win_count = torch.as_tensor(
            rank_result["relative_q_win_count"], dtype=torch.int64
        )
        tie_count = torch.as_tensor(
            rank_result["relative_q_tie_count"], dtype=torch.int64
        )
        ranks = torch.as_tensor(
            rank_result["relative_q_rank"], dtype=torch.float64
        )
    else:
        proposal_q_min = q_min.new_empty((0, int(num_alternatives)))
        win_count = torch.empty(0, dtype=torch.int64)
        tie_count = torch.empty(0, dtype=torch.int64)
        ranks = torch.empty(0, dtype=torch.float64)
    packed = torch.cat(
        (
            q_min,
            value,
            advantage,
            twin_gap,
            proposal_q_min.reshape(-1),
        )
    )
    if not torch.isfinite(packed).all():
        raise FloatingPointError("relative_q critic scores contain non-finite values")
    return {
        "q_min": q_min,
        "value": value,
        "advantage": advantage,
        "twin_gap": twin_gap,
        "proposal_q_min": proposal_q_min,
        "win_count": win_count,
        "tie_count": tie_count,
        "rank": ranks,
        "out_of_range_elements": out_of_range_elements,
    }


def score_dataset(
    path: Path,
    *,
    args: argparse.Namespace,
    policy,
    model: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    identity = mixed_dataset_identity(path)
    validate_mixed_dataset_source_identity(identity)
    loader_args = _loader_args(args, path, metadata)
    dataset, loader, _generator, _config = build_single_loader(
        loader_args,
        policy,
        # The loader only consumes the pretrained actor metadata and action stats.
        model["dp_checkpoint"],
        sequence_length=int(metadata["chunk_horizon"]),
        shuffle=False,
        drop_last=False,
    )
    dataset_audit(
        path,
        len(dataset),
        expected_task=str(args.task),
        expected_reward_mode=str(metadata["reward_mode"]),
        validity_key=getattr(dataset, "validity_key", None),
    )
    base = _base_sequence_dataset(dataset)
    count = int(len(base))
    fields = {
        "q_min": np.full(count, np.nan, dtype=np.float32),
        "value": np.full(count, np.nan, dtype=np.float32),
        "advantage": np.full(count, np.nan, dtype=np.float32),
        "twin_gap": np.full(count, np.nan, dtype=np.float32),
        "source_is_expert": np.zeros(count, dtype=np.uint8),
        "rollout_success": np.zeros(count, dtype=np.uint8),
        "scored": np.zeros(count, dtype=np.uint8),
    }
    relative_mode = str(args.condition_mode) == "relative_q"
    if relative_mode:
        fields.update(
            {
                "proposal_q_min": np.full(
                    (count, int(args.num_alternatives)),
                    np.nan,
                    dtype=np.float32,
                ),
                "relative_q_win_count": np.full(count, -1, dtype=np.int64),
                "relative_q_tie_count": np.full(count, -1, dtype=np.int64),
                "relative_q_rank": np.full(count, np.nan, dtype=np.float64),
                "relative_q_scored": np.zeros(count, dtype=np.uint8),
            }
        )
    episode_sources = _episode_sources(path)
    rng_namespace = _dataset_rng_namespace(identity)
    proposal_out_of_range_elements = 0
    for raw_batch in loader:
        indices = torch.as_tensor(raw_batch["index"]).long().reshape(-1)
        index_np = indices.cpu().numpy()
        if index_np.size and (
            int(index_np.min()) < 0 or int(index_np.max()) >= count
        ):
            raise IndexError("loader returned a dataset index outside the sidecar")
        if np.any(fields["scored"][index_np] != 0):
            raise RuntimeError("critic condition loader returned duplicate indices")
        batch_is_human = np.empty(index_np.shape, dtype=bool)
        for offset, index in enumerate(index_np):
            demo = str(base._index_to_demo_id[int(index)])
            source = episode_sources[demo]
            batch_is_human[offset] = source == "expert"
            fields["source_is_expert"][index] = int(source == "expert")
            fields["rollout_success"][index] = int(
                source == "non_expert_success"
            )
        rollout_positions = np.flatnonzero(~batch_is_human).astype(
            np.int64, copy=False
        )
        if relative_mode:
            relative_scores = _score_relative_q_batch(
                raw_batch,
                policy=policy,
                model=model,
                metadata=metadata,
                rollout_positions=rollout_positions,
                row_ids=index_np[rollout_positions],
                num_alternatives=int(args.num_alternatives),
                candidate_batch_size=int(args.candidate_batch_size),
                seed=int(args.seed),
                rng_namespace=rng_namespace,
            )
            q_min = relative_scores["q_min"]
            value = relative_scores["value"]
            advantage = relative_scores["advantage"]
            twin_gap = relative_scores["twin_gap"]
            rollout_indices = index_np[rollout_positions]
            fields["proposal_q_min"][rollout_indices] = relative_scores[
                "proposal_q_min"
            ].detach().float().cpu().numpy()
            fields["relative_q_win_count"][rollout_indices] = relative_scores[
                "win_count"
            ].cpu().numpy()
            fields["relative_q_tie_count"][rollout_indices] = relative_scores[
                "tie_count"
            ].cpu().numpy()
            fields["relative_q_rank"][rollout_indices] = relative_scores[
                "rank"
            ].cpu().numpy()
            fields["relative_q_scored"][rollout_indices] = 1
            proposal_out_of_range_elements += int(
                relative_scores["out_of_range_elements"]
            )
        else:
            q_min, value, advantage, twin_gap = _score_batch(
                raw_batch,
                policy=policy,
                model=model,
                metadata=metadata,
            )
        for name, tensor in (
            ("q_min", q_min),
            ("value", value),
            ("advantage", advantage),
            ("twin_gap", twin_gap),
        ):
            fields[name][index_np] = tensor.detach().float().cpu().numpy()
        fields["scored"][index_np] = 1

    scored = fields["scored"].astype(bool)
    if not np.any(scored):
        raise ValueError(f"dataset {path} produced no scoreable chunk rows")
    for name in ("q_min", "value", "advantage", "twin_gap"):
        if not np.isfinite(fields[name][scored]).all():
            raise FloatingPointError(f"scored {name} contains non-finite values")
    proposal_audit = None
    if relative_mode:
        rollout_scored = fields["relative_q_scored"].astype(bool)
        expected_rollout = scored & ~fields["source_is_expert"].astype(bool)
        if not np.array_equal(rollout_scored, expected_rollout):
            raise RuntimeError(
                "relative_q proposal coverage does not match scored rollout rows"
            )
        if not np.isfinite(fields["proposal_q_min"][rollout_scored]).all():
            raise FloatingPointError("relative_q proposal Q contains non-finite values")
        if not np.isfinite(fields["relative_q_rank"][rollout_scored]).all():
            raise FloatingPointError("relative_q ranks contain non-finite values")
        wins, win_frequency = np.unique(
            fields["relative_q_win_count"][rollout_scored],
            return_counts=True,
        )
        proposal_audit = {
            "rng_namespace": int(rng_namespace),
            "sampled_rollout_rows": int(rollout_scored.sum()),
            "skipped_human_rows": int((scored & ~expected_rollout).sum()),
            "num_alternatives": int(args.num_alternatives),
            "sampled_action_chunks": int(
                rollout_scored.sum() * int(args.num_alternatives)
            ),
            "out_of_range_action_elements": int(
                proposal_out_of_range_elements
            ),
            "rows_with_q_ties": int(
                (fields["relative_q_tie_count"][rollout_scored] > 0).sum()
            ),
            "total_q_ties": int(
                fields["relative_q_tie_count"][rollout_scored].sum()
            ),
            "win_count_histogram": {
                str(int(win)): int(frequency)
                for win, frequency in zip(wins, win_frequency)
            },
        }
    return {
        "path": path,
        "identity": identity,
        "num_samples": count,
        "fields": fields,
        "scoreable_samples": int(scored.sum()),
        "filtered_samples": int(count - scored.sum()),
        "proposal_audit": proposal_audit,
    }


def _threshold_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": str(args.condition_mode),
        "threshold_mode": str(args.threshold_mode),
    }
    if args.threshold_mode == "fixed":
        result.update(
            low_threshold=float(args.low_threshold),
            high_threshold=float(args.high_threshold),
        )
    elif args.threshold_mode == "quantile":
        result.update(
            low_quantile=float(args.low_quantile),
            high_quantile=float(args.high_quantile),
        )
    elif args.threshold_mode == "outcome_purity":
        result.update(
            target_purity=float(args.target_purity),
            min_tail_count=int(args.min_tail_count),
        )
    else:
        raise ValueError(f"unsupported threshold mode {args.threshold_mode!r}")
    return result


def _calibrate(
    scored: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    fields = scored["fields"]
    valid = fields["scored"].astype(bool)
    if str(args.condition_mode) == "relative_q":
        conditions = construct_relative_q_conditions(
            fields["relative_q_win_count"][valid],
            fields["source_is_expert"][valid],
            num_alternatives=int(args.num_alternatives),
            low_threshold=float(args.relative_q_low_rank),
            high_threshold=float(args.relative_q_high_rank),
        )
        return {
            "mode": "relative_q",
            "threshold_mode": "fixed_rank",
            "low_threshold": float(args.relative_q_low_rank),
            "high_threshold": float(args.relative_q_high_rank),
            "low_quantile": None,
            "high_quantile": None,
            "audit": {
                "method": "fixed_state_relative_rank",
                **conditions["audit"],
            },
        }
    score_key = (
        "advantage" if args.condition_mode == "critic_advantage" else "q_min"
    )
    kwargs = _threshold_kwargs(args)
    if args.threshold_mode == "outcome_purity":
        kwargs["rollout_success"] = fields["rollout_success"][valid]
    return calibrate_critic_condition_thresholds(
        fields[score_key][valid],
        fields["source_is_expert"][valid],
        **kwargs,
    )


def _build_payload(
    scored: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    args: argparse.Namespace,
    metadata: dict[str, Any],
    calibration_identity: dict[str, Any],
) -> dict[str, Any]:
    source_fields = scored["fields"]
    valid = source_fields["scored"].astype(bool)
    mode = str(args.condition_mode)
    threshold_mode = str(
        thresholds.get(
            "threshold_mode",
            "fixed_rank" if mode == "relative_q" else args.threshold_mode,
        )
    )
    score_key = {
        "critic_advantage": "advantage",
        "critic_q": "q_min",
        "relative_q": "relative_q_rank",
    }[mode]
    if mode == "relative_q":
        conditions = construct_relative_q_conditions(
            source_fields["relative_q_win_count"][valid],
            source_fields["source_is_expert"][valid],
            num_alternatives=int(args.num_alternatives),
            low_threshold=float(thresholds["low_threshold"]),
            high_threshold=float(thresholds["high_threshold"]),
        )
    else:
        conditions = construct_critic_conditions(
            source_fields[score_key][valid],
            source_fields["source_is_expert"][valid],
            mode=mode,
            low_threshold=float(thresholds["low_threshold"]),
            high_threshold=float(thresholds["high_threshold"]),
        )
    actor_condition = np.zeros(scored["num_samples"], dtype=np.uint8)
    actor_condition_mask = np.zeros(scored["num_samples"], dtype=np.uint8)
    actor_condition[valid] = conditions["actor_condition"]
    actor_condition_mask[valid] = conditions["actor_condition_mask"]
    relative_outcome_audit = None
    if mode == "relative_q":
        rollout = valid & ~source_fields["source_is_expert"].astype(bool)
        success = source_fields["rollout_success"].astype(bool)
        positive = rollout & (actor_condition == 1) & (
            actor_condition_mask == 1
        )
        negative = rollout & (actor_condition == 0) & (
            actor_condition_mask == 1
        )
        null = rollout & (actor_condition_mask == 0)

        def outcome_counts(mask: np.ndarray) -> dict[str, Any]:
            count = int(mask.sum())
            successes = int((mask & success).sum())
            failures = count - successes
            return {
                "count": count,
                "success": successes,
                "failure": failures,
                "success_fraction": (
                    float(successes / count) if count else None
                ),
            }

        relative_outcome_audit = {
            "rollout": outcome_counts(rollout),
            "positive": outcome_counts(positive),
            "negative": outcome_counts(negative),
            "null": outcome_counts(null),
            "note": "outcomes_are_audit_only_and_do_not_assign_relative_q_labels",
        }
    fields = {
        key: torch.from_numpy(value)
        for key, value in source_fields.items()
    }
    fields["actor_condition"] = torch.from_numpy(actor_condition)
    fields["actor_condition_mask"] = torch.from_numpy(actor_condition_mask)
    payload = {
        "kind": LABEL_FORMAT,
        "version": 1,
        "mode": str(args.condition_mode),
        "dataset_identity": scored["identity"],
        "num_samples": int(scored["num_samples"]),
        "fields": fields,
        "config": {
            "task": str(args.task),
            "condition_mode": mode,
            "score_key": score_key,
            "threshold_mode": threshold_mode,
            "low_threshold": float(thresholds["low_threshold"]),
            "high_threshold": float(thresholds["high_threshold"]),
            "low_quantile": (
                float(args.low_quantile)
                if threshold_mode == "quantile"
                else None
            ),
            "high_quantile": (
                float(args.high_quantile)
                if threshold_mode == "quantile"
                else None
            ),
            "target_purity": (
                float(args.target_purity)
                if threshold_mode == "outcome_purity"
                else None
            ),
            "min_tail_count": (
                int(args.min_tail_count)
                if threshold_mode == "outcome_purity"
                else None
            ),
            **(
                {
                    "num_alternatives": int(args.num_alternatives),
                    "relative_q_low_rank": float(args.relative_q_low_rank),
                    "relative_q_high_rank": float(args.relative_q_high_rank),
                    "rank_formula": "(1+win_count)/(num_alternatives+1)",
                    "rank_comparison": "logged_q_min>=proposal_q_min",
                    "rank_threshold_comparison": (
                        "rank<=low:0;rank>=high:1;otherwise:null"
                    ),
                    "proposal_seed": int(args.seed),
                    "candidate_batch_size": int(args.candidate_batch_size),
                }
                if mode == "relative_q"
                else {}
            ),
            "critic_source": str(args.critic_source),
            "critic_checkpoint_identity": file_stat_identity(
                args.critic_checkpoint
            ),
            "dp_checkpoint_identity": file_stat_identity(args.dp_checkpoint),
            "calibration_dataset_identity": calibration_identity,
            "sparse_chunk_loader": bool(args.sparse_chunk_loader),
            **metadata,
        },
        "threshold_audit": thresholds.get("audit", thresholds),
        "summary": {
            "scoreable_samples": int(scored["scoreable_samples"]),
            "filtered_samples": int(scored["filtered_samples"]),
            **conditions["audit"],
            **(
                {"proposal_sampling": scored["proposal_audit"]}
                if mode == "relative_q"
                else {}
            ),
            **(
                {"rollout_outcome_audit": relative_outcome_audit}
                if mode == "relative_q"
                else {}
            ),
        },
    }
    return payload


def _publish(payload: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"condition sidecar already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, path)
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    atomic_write_json(
        summary_path,
        _jsonable(
            {
                "kind": payload["kind"],
                "mode": payload["mode"],
                "num_samples": payload["num_samples"],
                "config": payload["config"],
                "threshold_audit": payload["threshold_audit"],
                "summary": payload["summary"],
            }
        ),
    )


def label_conditions(args: argparse.Namespace) -> dict[str, Any]:
    args.dataset = args.dataset.expanduser().resolve()
    args.dp_checkpoint = args.dp_checkpoint.expanduser().resolve()
    args.critic_checkpoint = args.critic_checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.validation_dataset is not None:
        args.validation_dataset = args.validation_dataset.expanduser().resolve()
    if args.validation_output is not None:
        args.validation_output = args.validation_output.expanduser().resolve()
    for path in (args.dataset, args.dp_checkpoint, args.critic_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (args.validation_dataset is None) != (args.validation_output is None):
        raise ValueError(
            "--validation-dataset and --validation-output must be supplied together"
        )
    if args.validation_output is not None and args.validation_output == args.output:
        raise ValueError("training and validation sidecars require different outputs")
    if args.validation_dataset is not None and not args.validation_dataset.is_file():
        raise FileNotFoundError(args.validation_dataset)
    prospective_outputs = [args.output]
    if args.validation_output is not None:
        prospective_outputs.append(args.validation_output)
    prospective_artifacts = prospective_outputs + [
        path.with_suffix(path.suffix + ".summary.json")
        for path in prospective_outputs
    ]
    if len(set(prospective_artifacts)) != len(prospective_artifacts):
        raise ValueError(
            "condition sidecar and summary output paths must not overlap"
        )
    protected_inputs = {
        args.dataset,
        args.dp_checkpoint,
        args.critic_checkpoint,
        *(
            (args.validation_dataset,)
            if args.validation_dataset is not None
            else ()
        ),
    }
    collisions = sorted(
        (path for path in prospective_artifacts if path in protected_inputs),
        key=str,
    )
    if collisions:
        raise ValueError(
            "condition outputs must not overwrite an input artifact: "
            f"{collisions}"
        )
    existing_outputs = [path for path in prospective_artifacts if path.exists()]
    if existing_outputs and not bool(args.overwrite):
        raise FileExistsError(
            "condition sidecar output already exists; pass --overwrite: "
            f"{existing_outputs}"
        )
    if int(args.batch_size) < 1 or int(args.num_workers) < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if str(args.condition_mode) == "relative_q":
        if int(args.num_alternatives) < 1:
            raise ValueError("--num-alternatives must be positive")
        if int(args.candidate_batch_size) < 1:
            raise ValueError("--candidate-batch-size must be positive")
        threshold_check = construct_relative_q_conditions(
            np.asarray([0, int(args.num_alternatives)], dtype=np.int64),
            np.zeros(2, dtype=np.uint8),
            num_alternatives=int(args.num_alternatives),
            low_threshold=float(args.relative_q_low_rank),
            high_threshold=float(args.relative_q_high_rank),
        )
        attainable = threshold_check["audit"]["thresholds"]
        unavailable = [
            name
            for name in (
                "negative_region_attainable",
                "null_region_attainable",
                "positive_region_attainable",
            )
            if not bool(attainable[name])
        ]
        if unavailable:
            raise ValueError(
                "relative_q thresholds leave unattainable condition regions "
                f"for M={args.num_alternatives}: {unavailable}"
            )
        args.threshold_mode = "fixed_rank"
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    policy, model, metadata = _load_critic(args)
    train_scores = score_dataset(
        args.dataset,
        args=args,
        policy=policy,
        model=model,
        metadata=metadata,
    )
    calibration_scores = train_scores
    validation_scores = None
    if args.validation_dataset is not None:
        validation_scores = score_dataset(
            args.validation_dataset,
            args=args,
            policy=policy,
            model=model,
            metadata=metadata,
        )
        calibration_scores = validation_scores
    thresholds = _calibrate(calibration_scores, args)
    train_payload = _build_payload(
        train_scores,
        thresholds=thresholds,
        args=args,
        metadata=metadata,
        calibration_identity=calibration_scores["identity"],
    )
    _publish(train_payload, args.output, overwrite=bool(args.overwrite))
    result = {
        "output": str(args.output),
        "thresholds": {
            "low": float(thresholds["low_threshold"]),
            "high": float(thresholds["high_threshold"]),
        },
        "training": train_payload["summary"],
    }
    if validation_scores is not None:
        validation_payload = _build_payload(
            validation_scores,
            thresholds=thresholds,
            args=args,
            metadata=metadata,
            calibration_identity=calibration_scores["identity"],
        )
        _publish(
            validation_payload,
            args.validation_output,
            overwrite=bool(args.overwrite),
        )
        result["validation_output"] = str(args.validation_output)
        result["validation"] = validation_payload["summary"]
    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, default=None)
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--critic-source",
        choices=("online", "target"),
        default="online",
        help=(
            "Q network stored in the critic checkpoint: online selects critics / "
            "chunk_value_system (default), while target selects critic_targets / "
            "chunk_value_target. V always comes from the learned online value model."
        ),
    )
    parser.add_argument(
        "--condition-mode",
        choices=("critic_advantage", "critic_q", "relative_q"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument(
        "--threshold-mode",
        choices=("fixed", "quantile", "outcome_purity"),
        default="quantile",
    )
    parser.add_argument("--low-threshold", type=float, default=-0.1)
    parser.add_argument("--high-threshold", type=float, default=0.1)
    parser.add_argument("--low-quantile", type=float, default=0.2)
    parser.add_argument("--high-quantile", type=float, default=0.8)
    parser.add_argument("--target-purity", type=float, default=0.9)
    parser.add_argument("--min-tail-count", type=int, default=32)
    parser.add_argument(
        "--num-alternatives",
        type=int,
        default=32,
        help="Number M of frozen-DP alternatives per rollout chunk.",
    )
    parser.add_argument(
        "--relative-q-low-rank",
        type=float,
        default=0.2,
        help="Inclusive relative-rank threshold for active condition zero.",
    )
    parser.add_argument(
        "--relative-q-high-rank",
        type=float,
        default=0.8,
        help="Inclusive relative-rank threshold for active condition one.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=100,
        help="Maximum flattened proposal trajectories per DP/Q microbatch.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--hdf5-cache-mode",
        choices=("low_dim", "none"),
        default="low_dim",
    )
    parser.add_argument(
        "--sparse-chunk-loader",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    label_conditions(make_parser().parse_args())


if __name__ == "__main__":
    main()
