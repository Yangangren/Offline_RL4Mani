#!/usr/bin/env python3
"""Separate MC-value and advantage-conditioned RECAP actor training.

This entrypoint deliberately does not train the chunk-IDQL Q ensemble or its
expectile value objective.  It prepares canonical Monte Carlo targets, trains
one WCM-style state value with optional dynamics / SIGReg auxiliaries, and
then trains a FiLM-conditioned Diffusion Policy from immutable label sidecars.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils

from rgb_dp_chunk_recap import build_canonical_episode_targets
from train_rgb_dp_chunk_idql import (
    WCM_CRITIC_ARCHITECTURE,
    add_actor_condition,
    architecture_q_head_inputs,
    atomic_write_json,
    configure_chunk_actor_optimizer,
    configure_conditioned_actor,
    copy_deployed_dp_encoder_state,
    file_stat_identity,
    make_wcm_chunk_value_system,
    masked_wcm_dynamics_mse,
    match_encoder_normalization_to_checkpoint,
    mixed_dataset_identity,
    process_chunk_batch,
    sigreg_loss,
)
from train_rgb_dp_idql import (
    actor_matches_deployed_ema,
    actor_train_step,
    atomic_torch_save,
    build_single_loader,
    initialize_actor_from_deployed_ema,
    jsonable,
    mean_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET_FORMAT = "rgb_dp_chunk_recap_targets_v1"
VALUE_FORMAT = "rgb_dp_chunk_recap_mc_value_v1"
LABEL_FORMAT = "rgb_dp_chunk_recap_labels_v1"
SOURCE_CODES = {
    "non_expert_failure": 0,
    "non_expert_success": 1,
    "expert": 2,
}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _numeric_demo_key(name: str) -> int:
    try:
        return int(str(name).rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid demonstration key {name!r}") from exc


def _dataset_reward_mode(dataset: Path) -> str:
    with h5py.File(dataset, "r") as handle:
        mode = _decode(handle.attrs.get("reward_mode", ""))
        definition = _decode(handle.attrs.get("reward_definition", ""))
    if not mode and definition == "source_task_reward":
        mode = "task"
    if not mode and definition == "expert_transition=1; non_expert_transition=0":
        mode = "rise"
    if mode not in ("task", "rise"):
        raise ValueError(
            f"mixed dataset has unsupported reward_mode={mode!r}; expected task or rise"
        )
    return mode


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(name: str) -> torch.device:
    if str(name) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(str(name))


def load_sidecar(path: Path, *, expected_kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != expected_kind:
        raise ValueError(
            f"{path} is not a {expected_kind!r} sidecar; "
            f"kind={getattr(payload, 'get', lambda *_: None)('kind')!r}"
        )
    if not isinstance(payload.get("fields"), dict):
        raise ValueError(f"{path} has no sidecar fields")
    return payload


def validate_sidecar_dataset(
    sidecar: dict[str, Any], dataset: Path, *, sidecar_name: str
) -> None:
    current = mixed_dataset_identity(dataset)
    if sidecar.get("dataset_identity") != current:
        raise ValueError(
            f"{sidecar_name} was prepared for a different or modified dataset; "
            "regenerate it instead of reusing stale targets"
        )
    lengths = {
        int(torch.as_tensor(value).shape[0])
        for value in sidecar["fields"].values()
        if torch.as_tensor(value).ndim > 0
    }
    if len(lengths) != 1 or next(iter(lengths)) != int(sidecar["num_samples"]):
        raise ValueError(f"{sidecar_name} fields do not share num_samples")


def sidecar_rows(
    sidecar: dict[str, Any], indices: torch.Tensor, *names: str
) -> tuple[torch.Tensor, ...]:
    indices = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= int(sidecar["num_samples"])
    ):
        raise IndexError("batch sample index is outside the sidecar")
    return tuple(
        torch.as_tensor(sidecar["fields"][name]).index_select(0, indices)
        for name in names
    )


def _stratified_episode_validation_split(
    sources: list[str], fraction: float, seed: int
) -> set[int]:
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    rng = np.random.default_rng(int(seed))
    selected: set[int] = set()
    for source in SOURCE_CODES:
        members = np.asarray(
            [index for index, value in enumerate(sources) if value == source],
            dtype=np.int64,
        )
        if members.size <= 1 or float(fraction) == 0.0:
            count = 0
        else:
            count = min(
                members.size - 1,
                max(1, int(round(float(fraction) * members.size))),
            )
        if count:
            selected.update(int(value) for value in rng.choice(members, count, replace=False))
    return selected


def prepare_targets(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not 0.0 < float(args.gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if int(args.chunk_horizon) < 1:
        raise ValueError("chunk_horizon must be positive")
    if float(args.failure_penalty) <= 0.0 or float(args.return_scale) <= 0.0:
        raise ValueError("failure_penalty and return_scale must be positive")

    episode_records: list[dict[str, Any]] = []
    episode_sources: list[str] = []
    with h5py.File(dataset, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"{dataset} has no data group")
        task = _decode(handle.attrs.get("task", ""))
        source_reward_mode = _decode(handle.attrs.get("reward_mode", ""))
        if not source_reward_mode and _decode(
            handle.attrs.get("reward_definition", "")
        ) == "source_task_reward":
            source_reward_mode = "task"
        if not source_reward_mode and _decode(
            handle.attrs.get("reward_definition", "")
        ) == "expert_transition=1; non_expert_transition=0":
            source_reward_mode = "rise"
        if source_reward_mode not in ("task", "rise"):
            raise ValueError(
                f"mixed dataset has unsupported reward_mode={source_reward_mode!r}"
            )
        demo_keys = sorted(handle["data"], key=_numeric_demo_key)
        if not demo_keys:
            raise ValueError("mixed dataset contains no episodes")
        for demo_key in demo_keys:
            episode = handle["data"][demo_key]
            source = _decode(episode.attrs.get("rise_source", ""))
            if source not in SOURCE_CODES:
                raise ValueError(
                    f"data/{demo_key} has unsupported rise_source={source!r}"
                )
            if source_reward_mode == "rise" and "task_rewards" not in episode:
                raise KeyError(
                    f"data/{demo_key} is from a rise-reward dataset but has no "
                    "preserved task_rewards for success-terminal detection"
                )
            reward_key = "task_rewards" if "task_rewards" in episode else "rewards"
            if reward_key not in episode:
                raise KeyError(f"data/{demo_key} has no rewards")
            rewards = np.asarray(episode[reward_key][:], dtype=np.float64).reshape(-1)
            declared = int(episode.attrs.get("num_samples", rewards.shape[0]))
            if declared != rewards.shape[0]:
                raise ValueError(
                    f"data/{demo_key} num_samples={declared} but rewards has "
                    f"length={rewards.shape[0]}"
                )
            targets = build_canonical_episode_targets(
                rewards,
                source=source,
                gamma=float(args.gamma),
                failure_penalty=float(args.failure_penalty),
                return_scale=float(args.return_scale),
                chunk_horizon=int(args.chunk_horizon),
            )
            episode_records.append(
                {
                    "demo_key": demo_key,
                    "source": source,
                    "num_samples": declared,
                    "targets": targets,
                }
            )
            episode_sources.append(source)

    validation_episodes = _stratified_episode_validation_split(
        episode_sources, float(args.valid_fraction), int(args.seed)
    )
    field_lists: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "canonical_reward",
            "mc_return",
            "value_valid",
            "chunk_return",
            "terminal",
            "valid_length",
            "source_code",
            "source_is_expert",
            "is_validation",
            "episode_index",
        )
    }
    summaries = []
    for episode_index, record in enumerate(episode_records):
        length = int(record["num_samples"])
        targets = record["targets"]
        for name in (
            "canonical_reward",
            "mc_return",
            "value_valid",
            "chunk_return",
            "terminal",
            "valid_length",
        ):
            field_lists[name].append(torch.as_tensor(targets[name]))
        source = str(record["source"])
        field_lists["source_code"].append(
            torch.full((length,), SOURCE_CODES[source], dtype=torch.int8)
        )
        field_lists["source_is_expert"].append(
            torch.full((length,), source == "expert", dtype=torch.bool)
        )
        field_lists["is_validation"].append(
            torch.full((length,), episode_index in validation_episodes, dtype=torch.bool)
        )
        field_lists["episode_index"].append(
            torch.full((length,), episode_index, dtype=torch.int32)
        )
        summaries.append(
            {
                "demo_key": record["demo_key"],
                "source": source,
                "num_samples": length,
                "validation": episode_index in validation_episodes,
                "value_valid_samples": int(
                    torch.as_tensor(targets["value_valid"]).sum().item()
                ),
            }
        )

    fields = {name: torch.cat(parts, dim=0) for name, parts in field_lists.items()}
    num_samples = int(fields["mc_return"].shape[0])
    payload = {
        "kind": TARGET_FORMAT,
        "version": 1,
        "dataset_identity": mixed_dataset_identity(dataset),
        "dataset": str(dataset),
        "num_samples": num_samples,
        "task": task,
        "config": {
            "gamma": float(args.gamma),
            "chunk_horizon": int(args.chunk_horizon),
            "failure_penalty": float(args.failure_penalty),
            "return_scale": float(args.return_scale),
            "terminal_policy": "first_positive_task_reward_for_success",
            "reward_definition": (
                "preterminal=-1; success_terminal=0; "
                "failure_terminal=-failure_penalty; multiplicative_return_scale"
            ),
            "valid_fraction": float(args.valid_fraction),
            "split_seed": int(args.seed),
            "source_reward_mode": source_reward_mode,
        },
        "source_codes": dict(SOURCE_CODES),
        "fields": fields,
        "episodes": summaries,
    }
    if output.exists() and not bool(args.overwrite):
        existing = load_sidecar(output, expected_kind=TARGET_FORMAT)
        if (
            existing.get("dataset_identity") == payload["dataset_identity"]
            and existing.get("config") == payload["config"]
        ):
            print(json.dumps({"reused": str(output), "num_samples": num_samples}, indent=2))
            return existing
        raise FileExistsError(
            f"{output} exists with different provenance; use --overwrite or a new path"
        )
    atomic_torch_save(payload, output)
    result = {
        "targets": str(output),
        "num_samples": num_samples,
        "episodes": len(summaries),
        "value_valid_samples": int(fields["value_valid"].sum().item()),
        "validation_samples": int(fields["is_validation"].sum().item()),
        "source_samples": {
            source: int((fields["source_code"] == code).sum().item())
            for source, code in SOURCE_CODES.items()
        },
    }
    print(json.dumps(result, indent=2))
    return payload


def _loader_namespace(
    args: argparse.Namespace,
    *,
    sparse_chunk_loader: bool,
    dynamics_prediction_offsets: tuple[int, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        hdf5_cache_mode=(
            None if args.hdf5_cache_mode is None else str(args.hdf5_cache_mode)
        ),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        reward_mode=str(args.loader_reward_mode),
        conditioned_actor=False,
        sparse_chunk_loader=bool(sparse_chunk_loader),
        sparse_one_step_loader=False,
        sparse_dql_loader=False,
        chunk_horizon=int(args.chunk_horizon),
        critic_observation_horizon=int(getattr(args, "observation_horizon", 1)),
        dynamics_prediction_offsets=tuple(dynamics_prediction_offsets),
        distributed_world_size=1,
        distributed_rank=0,
        seed=int(args.seed),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
    )


def build_recap_dataset(
    args: argparse.Namespace,
    actor_policy,
    dp_checkpoint: dict[str, Any],
    *,
    sparse_chunk_loader: bool,
    dynamics_prediction_offsets: tuple[int, ...] = (),
    sequence_length: int | None = None,
):
    args.loader_reward_mode = _dataset_reward_mode(args.dataset)
    loader_args = _loader_namespace(
        args,
        sparse_chunk_loader=sparse_chunk_loader,
        dynamics_prediction_offsets=dynamics_prediction_offsets,
    )
    dataset, _unused_loader, generator, config = build_single_loader(
        loader_args,
        actor_policy,
        dp_checkpoint,
        sequence_length=sequence_length,
    )
    return dataset, generator, config


def make_loader(dataset, args: argparse.Namespace, *, shuffle: bool):
    generator = torch.Generator().manual_seed(int(args.seed))
    kwargs: dict[str, Any] = {}
    if int(args.num_workers) > 0:
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and str(args.device) == "cuda"),
        generator=generator,
        **kwargs,
    )


def _architecture_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "chunk_horizon": int(args.chunk_horizon),
        "hidden_dims": tuple(int(value) for value in args.hidden_dims),
        "latent_dim": int(args.latent_dim),
        "action_hidden_dim": int(args.action_hidden_dim),
        "num_attention_heads": int(args.num_attention_heads),
        "num_action_conv_layers": int(args.num_action_conv_layers),
        "dropout": float(args.dropout),
        "num_critics": 2,
        "critic_group_norm": bool(args.group_norm),
        "late_fusion_key": args.late_fusion_key,
        "observation_horizon": int(args.observation_horizon),
        "temporal_num_layers": int(args.temporal_num_layers),
        "temporal_num_heads": int(args.temporal_num_heads),
        "temporal_feedforward_dim": int(args.temporal_feedforward_dim),
        "temporal_dropout": float(args.temporal_dropout),
        "dynamics_prediction_offsets": tuple(
            int(value) for value in args.dynamics_prediction_offsets
        ),
    }


def _freeze_untrained_wcm_modules(system, *, train_dynamics: bool) -> None:
    system.nets["q_action_encoders"].requires_grad_(False)
    system.nets["q_heads"].requires_grad_(False)
    system.nets["dynamics"].requires_grad_(bool(train_dynamics))
    if any(parameter.requires_grad for parameter in system.nets["q_heads"].parameters()):
        raise RuntimeError("RECAP MC training must not enable Q heads")
    if any(
        parameter.requires_grad
        for parameter in system.nets["q_action_encoders"].parameters()
    ):
        raise RuntimeError("RECAP MC training must not enable Q action encoders")


def compute_mc_value_loss(
    system,
    target_system,
    batch: dict[str, Any],
    mc_return: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    dynamics_weight: float,
    sigreg_weight: float,
    sigreg_knots: int,
    sigreg_num_projections: int,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute MC MSE plus auxiliaries, without touching either Q path."""
    state = system.encode_state(batch["obs"], batch.get("goal_obs"))
    value = system.value_from_state(state).reshape(-1)
    train_mask = train_mask.reshape(-1).to(device=value.device, dtype=torch.bool)
    target = mc_return.reshape(-1).to(device=value.device, dtype=value.dtype)
    if not torch.any(train_mask):
        raise ValueError("MC-value batch contains no valid training rows")
    value_loss = F.mse_loss(value[train_mask], target[train_mask])
    total = value_loss
    info: dict[str, torch.Tensor] = {
        "loss/value_mse": value_loss.detach(),
        "value/prediction_mean": value[train_mask].mean().detach(),
        "value/target_mean": target[train_mask].mean().detach(),
        "data/train_rows": train_mask.sum().detach(),
    }

    if float(dynamics_weight) > 0.0:
        if "dynamics_targets" not in batch:
            raise KeyError("nonzero dynamics weight requires dynamics_targets")
        predicted = system.predict_dynamics_from_state(
            state, batch["actions"], batch["action_mask"]
        )
        with torch.no_grad():
            # Use the slow target representation as a stable stop-gradient
            # teacher. This is WCM-style latent prediction, but deliberately
            # differs from the joint-IDQL path's online-encoder target.
            target_latent = target_system.encode_dynamics_targets(
                batch["dynamics_targets"]["next_obs"], batch.get("goal_obs")
            ).detach()
        dynamics_mask = batch["dynamics_targets"]["valid_mask"].clone()
        dynamics_mask = dynamics_mask * train_mask[:, None].to(dynamics_mask.dtype)
        dynamics_loss, dynamics_info = masked_wcm_dynamics_mse(
            predicted,
            target_latent,
            dynamics_mask,
            state["current_frame_latent"],
        )
        total = total + float(dynamics_weight) * dynamics_loss
        info["loss/dynamics"] = dynamics_loss.detach()
        info.update(dynamics_info)
    else:
        info["loss/dynamics"] = value_loss.detach().new_zeros(())

    if float(sigreg_weight) > 0.0:
        regularizer = sigreg_loss(
            state["temporal_state"][train_mask],
            knots=int(sigreg_knots),
            num_projections=int(sigreg_num_projections),
            seed=int(global_step),
        )
        total = total + float(sigreg_weight) * regularizer
        info["loss/sigreg"] = regularizer.detach()
    else:
        info["loss/sigreg"] = value_loss.detach().new_zeros(())
    info["loss/total"] = total.detach()
    return total, info


def _apply_canonical_chunk_fields(
    batch: dict[str, Any],
    target_sidecar: dict[str, Any],
    indices: torch.Tensor,
    *,
    dynamics_offsets: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        mc_return,
        value_valid,
        chunk_return,
        terminal,
        valid_length,
        is_validation,
    ) = sidecar_rows(
        target_sidecar,
        indices,
        "mc_return",
        "value_valid",
        "chunk_return",
        "terminal",
        "valid_length",
        "is_validation",
    )
    device = batch["actions"].device
    dtype = batch["actions"].dtype
    chunk_return = chunk_return.to(device=device, dtype=dtype).reshape(-1, 1)
    terminal = terminal.to(device=device, dtype=dtype).reshape(-1, 1)
    valid_length = valid_length.to(device=device, dtype=dtype).reshape(-1, 1)
    positions = torch.arange(
        batch["actions"].shape[1], device=device, dtype=dtype
    )[None]
    canonical_action_mask = (positions < valid_length).to(dtype=dtype)
    batch["actions"] = batch["actions"] * canonical_action_mask.unsqueeze(-1)
    batch["action_mask"] = canonical_action_mask
    batch["reward"] = chunk_return
    batch["terminal"] = terminal
    batch["valid_length"] = valid_length
    if dynamics_offsets:
        offsets = torch.as_tensor(dynamics_offsets, device=device, dtype=dtype)[None]
        dynamics_valid = offsets < valid_length
        dynamics_valid = dynamics_valid | ((terminal < 0.5) & (offsets <= valid_length))
        availability = batch["dynamics_targets"]["valid_mask"] > 0.5
        batch["dynamics_targets"]["valid_mask"] = (
            dynamics_valid & availability
        ).to(dtype=dtype)
    mc_return = mc_return.to(device=device, dtype=dtype)
    train_mask = (
        value_valid.to(device=device, dtype=torch.bool)
        & ~is_validation.to(device=device, dtype=torch.bool)
    )
    return mc_return, train_mask


def _value_checkpoint_payload(
    args: argparse.Namespace,
    system,
    *,
    dp_checkpoint: dict[str, Any],
    epoch: int,
    global_step: int,
    best_validation_mse: float | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    architecture = _architecture_kwargs(args)
    save_dynamics = float(args.dynamics_weight) > 0.0
    value_state = {
        key: value
        for key, value in system.state_dict().items()
        if not key.startswith("nets.q_action_encoders.")
        and not key.startswith("nets.q_heads.")
        and (save_dynamics or not key.startswith("nets.dynamics."))
    }
    serialized_args = {
        key: jsonable(value)
        for key, value in vars(args).items()
        if key != "handler" and not callable(value)
    }
    return {
        "kind": VALUE_FORMAT,
        "version": 1,
        "rgb_dp_chunk_recap_value": True,
        "q_trained": False,
        "value_objective": "mse_to_canonical_discounted_monte_carlo_return",
        "actor_trained": False,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_mse": best_validation_mse,
        "chunk_value_system": value_state,
        "checkpoint_state_scope": "mc_value_representation_and_optional_dynamics",
        "q_state_saved": False,
        "dynamics_state_saved": save_dynamics,
        "dynamics_teacher": "ema_value_system_encoder_eval_stop_gradient",
        "history": history,
        "args": serialized_args,
        "dataset_identity": mixed_dataset_identity(args.dataset),
        "targets_identity": file_stat_identity(args.targets),
        "pretrained_dp_identity": file_stat_identity(args.checkpoint),
        "pretrained_dp_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "action_normalization_stats": dp_checkpoint.get("action_normalization_stats"),
        "critic_architecture": WCM_CRITIC_ARCHITECTURE,
        "critic_chunk_horizon": architecture["chunk_horizon"],
        "critic_hidden_dims": architecture["hidden_dims"],
        "critic_latent_dim": architecture["latent_dim"],
        "critic_action_hidden_dim": architecture["action_hidden_dim"],
        "critic_num_attention_heads": architecture["num_attention_heads"],
        "critic_num_action_conv_layers": architecture["num_action_conv_layers"],
        "critic_dropout": architecture["dropout"],
        "num_critics": 2,
        "critic_group_norm": architecture["critic_group_norm"],
        "critic_late_fusion_key": architecture["late_fusion_key"],
        "critic_observation_horizon": architecture["observation_horizon"],
        "critic_temporal_num_layers": architecture["temporal_num_layers"],
        "critic_temporal_num_heads": architecture["temporal_num_heads"],
        "critic_temporal_feedforward_dim": architecture["temporal_feedforward_dim"],
        "critic_temporal_dropout": architecture["temporal_dropout"],
        "dynamics_prediction_offsets": architecture["dynamics_prediction_offsets"],
        "critic_q_use_predicted_next_latent": False,
        "critic_q_head_inputs": architecture_q_head_inputs(WCM_CRITIC_ARCHITECTURE),
        "dynamics_prediction_consumed_by_q": False,
    }


@torch.no_grad()
def evaluate_value(
    system,
    loader,
    actor_algo,
    obs_stats,
    targets,
    args: argparse.Namespace,
) -> dict[str, float]:
    system.eval()
    squared_sum = 0.0
    absolute_sum = 0.0
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if (
            args.validation_batches is not None
            and batch_index >= int(args.validation_batches)
        ):
            break
        indices = raw_batch["index"]
        batch = process_chunk_batch(
            raw_batch,
            actor_algo,
            obs_stats,
            chunk_horizon=int(args.chunk_horizon),
            discount=float(targets["config"]["gamma"]),
            reward_mode=str(args.loader_reward_mode),
            critic_observation_horizon=int(args.observation_horizon),
            dynamics_prediction_offsets=(),
        )
        mc_return, value_valid, is_validation = sidecar_rows(
            targets, indices, "mc_return", "value_valid", "is_validation"
        )
        mask = value_valid.bool() & is_validation.bool()
        if not torch.any(mask):
            continue
        prediction = system.value_from_state(
            system.encode_state(batch["obs"], batch.get("goal_obs"))
        ).reshape(-1).detach().cpu()
        error = prediction[mask] - mc_return.float()[mask]
        squared_sum += float(error.square().sum().item())
        absolute_sum += float(error.abs().sum().item())
        count += int(mask.sum().item())
    if count == 0:
        return {"validation_mse": float("nan"), "validation_mae": float("nan"), "validation_rows": 0}
    return {
        "validation_mse": squared_sum / count,
        "validation_mae": absolute_sum / count,
        "validation_rows": count,
    }


def train_value(args: argparse.Namespace) -> dict[str, Any]:
    args.dataset = args.dataset.expanduser().resolve()
    args.targets = args.targets.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.value_architecture != WCM_CRITIC_ARCHITECTURE:
        raise ValueError("RECAP currently supports only wcm_shared_temporal_v1")
    if not args.dataset.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("dataset and pretrained DP checkpoint must exist")
    if int(args.epochs) < 1 or int(args.batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.steps_per_epoch is not None and int(args.steps_per_epoch) < 1:
        raise ValueError("steps_per_epoch must be positive when provided")
    if args.validation_batches is not None and int(args.validation_batches) < 0:
        raise ValueError("validation_batches must be non-negative when provided")
    if float(args.dynamics_weight) < 0.0 or float(args.sigreg_weight) < 0.0:
        raise ValueError("auxiliary weights must be non-negative")
    if int(args.observation_horizon) < 1:
        raise ValueError("observation_horizon must be positive")
    if not 0.0 <= float(args.target_tau) <= 1.0:
        raise ValueError("target_tau must be in [0, 1]")
    if int(args.sigreg_knots) < 2 or int(args.sigreg_num_projections) < 1:
        raise ValueError("SIGReg requires at least two knots and one projection")
    targets = load_sidecar(args.targets, expected_kind=TARGET_FORMAT)
    validate_sidecar_dataset(targets, args.dataset, sidecar_name="target sidecar")
    target_chunk_horizon = int(targets["config"]["chunk_horizon"])
    target_gamma = float(targets["config"]["gamma"])
    if args.chunk_horizon is not None and int(args.chunk_horizon) != target_chunk_horizon:
        raise ValueError(
            f"requested chunk_horizon={args.chunk_horizon} differs from "
            f"prepared targets={target_chunk_horizon}"
        )
    if args.gamma is not None and not np.isclose(
        float(args.gamma), target_gamma, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"requested gamma={args.gamma} differs from prepared targets={target_gamma}"
        )
    args.chunk_horizon = target_chunk_horizon
    args.gamma = target_gamma
    offsets = tuple(int(value) for value in args.dynamics_prediction_offsets)
    if (
        not offsets
        or tuple(sorted(set(offsets))) != offsets
        or offsets[0] < 1
        or offsets[-1] > args.chunk_horizon
    ):
        raise ValueError("dynamics offsets must be sorted, unique, nonempty, and <= chunk horizon")
    args.dynamics_prediction_offsets = offsets

    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    actor_algo = actor_policy.policy
    actor_algo.set_eval()
    actor_algo.nets.requires_grad_(False)
    if actor_algo.ema is not None:
        actor_algo.ema.averaged_model.requires_grad_(False)
    actor_horizon = int(actor_algo.algo_config.horizon.action_horizon)
    if actor_horizon != int(args.chunk_horizon):
        raise ValueError(
            f"target chunk_horizon={args.chunk_horizon} differs from actor action_horizon={actor_horizon}"
        )
    if int(args.observation_horizon) > int(
        actor_algo.algo_config.horizon.observation_horizon
    ):
        raise ValueError("value observation horizon exceeds the DP observation history")

    active_offsets = offsets if float(args.dynamics_weight) > 0.0 else ()
    dataset, _generator, _config = build_recap_dataset(
        args,
        actor_policy,
        dp_checkpoint,
        sparse_chunk_loader=bool(args.sparse_chunk_loader),
        dynamics_prediction_offsets=active_offsets,
        sequence_length=int(args.chunk_horizon),
    )
    if len(dataset) != int(targets["num_samples"]):
        raise ValueError(
            f"dataset length={len(dataset)} differs from target rows={targets['num_samples']}"
        )
    train_loader = make_loader(dataset, args, shuffle=True)
    validation_loader = make_loader(dataset, args, shuffle=False)
    obs_stats = copy.deepcopy(actor_policy.obs_normalization_stats)

    system, target_system = make_wcm_chunk_value_system(
        actor_algo, **_architecture_kwargs(args)
    )
    encoder_audit = copy_deployed_dp_encoder_state(system, actor_algo)
    target_system = copy.deepcopy(system)
    _freeze_untrained_wcm_modules(
        system, train_dynamics=float(args.dynamics_weight) > 0.0
    )
    target_system.requires_grad_(False)
    target_system.eval()
    system = system.float().to(device)
    target_system = target_system.float().to(device)
    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.value_lr), weight_decay=float(args.weight_decay)
    )

    global_step = 0
    history: list[dict[str, Any]] = []
    best_validation_mse: float | None = None
    for epoch in range(1, int(args.epochs) + 1):
        system.train()
        records: list[dict[str, float]] = []
        for batch_index, raw_batch in enumerate(train_loader):
            if (
                args.steps_per_epoch is not None
                and batch_index >= int(args.steps_per_epoch)
            ):
                break
            indices = raw_batch["index"]
            batch = process_chunk_batch(
                raw_batch,
                actor_algo,
                obs_stats,
                chunk_horizon=int(args.chunk_horizon),
                discount=float(args.gamma),
                reward_mode=str(args.loader_reward_mode),
                critic_observation_horizon=int(args.observation_horizon),
                dynamics_prediction_offsets=active_offsets,
            )
            mc_return, train_mask = _apply_canonical_chunk_fields(
                batch,
                targets,
                indices,
                dynamics_offsets=active_offsets,
            )
            if not torch.any(train_mask):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, info = compute_mc_value_loss(
                system,
                target_system,
                batch,
                mc_return,
                train_mask,
                dynamics_weight=float(args.dynamics_weight),
                sigreg_weight=float(args.sigreg_weight),
                sigreg_knots=int(args.sigreg_knots),
                sigreg_num_projections=int(args.sigreg_num_projections),
                global_step=global_step,
            )
            loss.backward()
            if float(args.max_gradient_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_gradient_norm))
            optimizer.step()
            if float(args.dynamics_weight) > 0.0:
                TorchUtils.soft_update(system, target_system, tau=float(args.target_tau))
            if any(parameter.grad is not None for parameter in system.nets["q_heads"].parameters()):
                raise RuntimeError("a Q head received gradients during MC-value training")
            if any(
                parameter.grad is not None
                for parameter in system.nets["q_action_encoders"].parameters()
            ):
                raise RuntimeError("a Q action encoder received gradients during MC-value training")
            records.append(
                {
                    key: float(value.detach().cpu().item())
                    for key, value in info.items()
                    if torch.as_tensor(value).numel() == 1
                }
            )
            global_step += 1

        validation = evaluate_value(
            system, validation_loader, actor_algo, obs_stats, targets, args
        )
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            **mean_metrics(records),
            **validation,
        }
        history.append(epoch_record)
        validation_mse = float(validation["validation_mse"])
        improved = np.isfinite(validation_mse) and (
            best_validation_mse is None or validation_mse < best_validation_mse
        )
        if improved:
            best_validation_mse = validation_mse
        payload = _value_checkpoint_payload(
            args,
            system,
            dp_checkpoint=dp_checkpoint,
            epoch=epoch,
            global_step=global_step,
            best_validation_mse=best_validation_mse,
            history=history,
        )
        atomic_torch_save(payload, args.output_dir / "last.pt")
        # With a deliberately zero validation fraction, retain a deterministic
        # first-epoch fallback so the staged launcher still has a labelable
        # checkpoint. Any later finite validation improvement replaces it.
        if improved or (epoch == 1 and best_validation_mse is None):
            atomic_torch_save(payload, args.output_dir / "best.pt")
        print(json.dumps(jsonable(epoch_record), indent=2), flush=True)

    summary = {
        "kind": VALUE_FORMAT,
        "checkpoint": str(args.output_dir / "last.pt"),
        "best_checkpoint": (
            str(args.output_dir / "best.pt")
            if (args.output_dir / "best.pt").is_file()
            else None
        ),
        "q_trained": False,
        "encoder_initialization": encoder_audit,
        "dynamics_enabled": float(args.dynamics_weight) > 0.0,
        "sigreg_enabled": float(args.sigreg_weight) > 0.0,
        "history": history,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def _save_recap_actor_checkpoint(
    path: Path,
    actor_algo,
    actor_config,
    dp_checkpoint: dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    labels: Path,
    condition_dropout: float,
    obs_normalization_stats,
) -> None:
    variable_state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "posttrain_mode": "advantage_conditioned_recap",
        "rgb_dp_chunk_recap_actor": True,
        "condition_label_sidecar": str(labels),
        "condition_dropout": float(condition_dropout),
        "inference_success_condition": 1.0,
        "inference_success_condition_mask": 1.0,
        "positive_only": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_config = copy.deepcopy(actor_config)
    if obs_normalization_stats is not None:
        with checkpoint_config.values_unlocked():
            checkpoint_config.train.hdf5_normalize_obs = True
    TrainUtils.save_model(
        model=actor_algo,
        config=checkpoint_config,
        env_meta=dp_checkpoint["env_metadata"],
        shape_meta=dp_checkpoint["shape_metadata"],
        ckpt_path=str(path),
        variable_state=variable_state,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=dp_checkpoint.get("action_normalization_stats"),
    )


def train_actor(args: argparse.Namespace) -> dict[str, Any]:
    args.dataset = args.dataset.expanduser().resolve()
    args.labels = args.labels.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.dataset.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("dataset and pretrained DP checkpoint must exist")
    if not 0.0 <= float(args.condition_dropout) < 1.0:
        raise ValueError("condition_dropout must be in [0, 1)")
    if int(args.epochs) < 1 or int(args.batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.steps_per_epoch is not None and int(args.steps_per_epoch) < 1:
        raise ValueError("steps_per_epoch must be positive when provided")
    labels = load_sidecar(args.labels, expected_kind=LABEL_FORMAT)
    validate_sidecar_dataset(labels, args.dataset, sidecar_name="RECAP label sidecar")
    if labels.get("targets_identity") is None or labels.get("value_checkpoint_identity") is None:
        raise ValueError("RECAP labels lack target/value provenance")
    if labels.get("pretrained_dp_identity") != file_stat_identity(args.checkpoint):
        raise ValueError("RECAP labels were produced with a different pretrained DP")
    conditions = torch.as_tensor(labels["fields"]["actor_condition"])
    if not torch.all((conditions == 0) | (conditions == 1)):
        raise ValueError("actor_condition sidecar must be binary")
    source_is_expert = torch.as_tensor(labels["fields"]["source_is_expert"]).bool()
    if not torch.all(conditions[source_is_expert] == 1):
        raise ValueError("all human demonstration conditions must equal one")

    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    actor_algo = actor_policy.policy
    initialized = initialize_actor_from_deployed_ema(actor_algo)
    if actor_algo.ema is not None and not initialized:
        raise RuntimeError("failed to initialize RECAP actor from deployed EMA")
    if not actor_matches_deployed_ema(actor_algo):
        raise RuntimeError("trainable actor differs from the deployed DP EMA")
    if actor_algo.reference_policy_enabled:
        raise RuntimeError("RECAP actor training requires reference/hazard objectives disabled")
    condition_args = SimpleNamespace(
        conditioned_actor=True,
        condition_hidden_dim=int(args.condition_hidden_dim),
        condition_dropout=float(args.condition_dropout),
    )
    configure_conditioned_actor(actor_algo, condition_args)
    prediction_horizon = int(actor_algo.algo_config.horizon.prediction_horizon)
    args.chunk_horizon = int(actor_algo.algo_config.horizon.action_horizon)
    if int(labels.get("config", {}).get("chunk_horizon", -1)) != int(
        args.chunk_horizon
    ):
        raise ValueError(
            "RECAP label chunk horizon differs from the pretrained actor action horizon"
        )
    args.observation_horizon = int(actor_algo.algo_config.horizon.observation_horizon)
    dataset, _generator, actor_config = build_recap_dataset(
        args,
        actor_policy,
        dp_checkpoint,
        sparse_chunk_loader=False,
        sequence_length=prediction_horizon,
    )
    if len(dataset) != int(labels["num_samples"]):
        raise ValueError(
            f"dataset length={len(dataset)} differs from label rows={labels['num_samples']}"
        )
    loader = make_loader(dataset, args, shuffle=True)
    updates_per_epoch = len(loader)
    if args.steps_per_epoch is not None:
        updates_per_epoch = min(updates_per_epoch, int(args.steps_per_epoch))
    total_steps = max(1, int(args.epochs) * updates_per_epoch)
    configure_chunk_actor_optimizer(
        actor_algo,
        adapter_lr=float(args.actor_adapter_lr),
        unet_lr=float(args.actor_unet_lr),
        obs_encoder_lr=float(args.actor_obs_encoder_lr),
        scheduler_type=str(args.lr_scheduler),
        warmup_steps=int(args.lr_warmup_steps),
        total_steps=total_steps,
        num_cycles=float(args.lr_num_cycles),
    )
    obs_stats = copy.deepcopy(actor_policy.obs_normalization_stats)
    global_step = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        actor_algo.set_train()
        records: list[dict[str, float]] = []
        for batch_index, raw_batch in enumerate(loader):
            if (
                args.steps_per_epoch is not None
                and batch_index >= int(args.steps_per_epoch)
            ):
                break
            batch_conditions, batch_expert = sidecar_rows(
                labels,
                raw_batch["index"],
                "actor_condition",
                "source_is_expert",
            )
            if not torch.all(batch_conditions[batch_expert.bool()] == 1):
                raise RuntimeError("human condition invariant failed in actor batch")
            raw_batch = add_actor_condition(raw_batch, batch_conditions.float())
            record = actor_train_step(
                actor_algo,
                raw_batch,
                epoch,
                obs_stats,
                defer_scalar_conversion=False,
            )
            records.append({key: float(value) for key, value in record.items()})
            global_step += 1
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            **mean_metrics(records),
            "condition/positive_fraction": float(conditions.float().mean().item()),
            "condition/dropout": float(args.condition_dropout),
        }
        history.append(epoch_record)
        _save_recap_actor_checkpoint(
            args.output_dir / "last.pth",
            actor_algo,
            actor_config,
            dp_checkpoint,
            epoch=epoch,
            global_step=global_step,
            labels=args.labels,
            condition_dropout=float(args.condition_dropout),
            obs_normalization_stats=obs_stats,
        )
        print(json.dumps(jsonable(epoch_record), indent=2), flush=True)

    summary = {
        "kind": "rgb_dp_chunk_recap_actor_v1",
        "checkpoint": str(args.output_dir / "last.pth"),
        "pretrained_dp_identity": file_stat_identity(args.checkpoint),
        "dataset_identity": mixed_dataset_identity(args.dataset),
        "labels_identity": file_stat_identity(args.labels),
        "positive_only": False,
        "condition_zero_rows_trained": True,
        "condition_dropout": float(args.condition_dropout),
        "inference_condition": {"success_condition": 1.0, "condition_mask": 1.0},
        "history": history,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def _add_loader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--hdf5-cache-mode", choices=("all", "low_dim", "none"), default="low_dim"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-targets")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--gamma", type=float, default=0.99)
    prepare.add_argument("--chunk-horizon", type=int, default=8)
    prepare.add_argument("--failure-penalty", type=float, default=400.0)
    prepare.add_argument("--return-scale", type=float, default=800.0)
    prepare.add_argument("--valid-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=prepare_targets)

    value = subparsers.add_parser("train-value")
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--targets", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--steps-per-epoch", type=int, default=None)
    value.add_argument("--validation-batches", type=int, default=None)
    value.add_argument("--gamma", type=float, default=None)
    value.add_argument("--chunk-horizon", type=int, default=None)
    value.add_argument("--value-lr", type=float, default=1e-4)
    value.add_argument("--weight-decay", type=float, default=1e-6)
    value.add_argument(
        "--value-architecture", choices=(WCM_CRITIC_ARCHITECTURE,), default=WCM_CRITIC_ARCHITECTURE
    )
    value.add_argument("--observation-horizon", type=int, default=2)
    value.add_argument("--hidden-dims", type=int, nargs="+", default=(300, 400, 300))
    value.add_argument("--latent-dim", type=int, default=300)
    value.add_argument("--action-hidden-dim", type=int, default=128)
    value.add_argument("--num-attention-heads", type=int, default=4)
    value.add_argument("--num-action-conv-layers", type=int, default=2)
    value.add_argument("--dropout", type=float, default=0.0)
    value.add_argument("--group-norm", action=argparse.BooleanOptionalAction, default=False)
    value.add_argument("--late-fusion-key", default="robot0_gripper_qpos")
    value.add_argument("--temporal-num-layers", type=int, default=2)
    value.add_argument("--temporal-num-heads", type=int, default=6)
    value.add_argument("--temporal-feedforward-dim", type=int, default=600)
    value.add_argument("--temporal-dropout", type=float, default=0.0)
    value.add_argument("--dynamics-weight", type=float, default=0.0)
    value.add_argument(
        "--dynamics-prediction-offsets", type=int, nargs="+", default=(2, 4, 6, 8)
    )
    value.add_argument("--target-tau", type=float, default=0.01)
    value.add_argument("--sigreg-weight", type=float, default=0.0)
    value.add_argument("--sigreg-knots", type=int, default=17)
    value.add_argument("--sigreg-num-projections", type=int, default=1024)
    value.add_argument("--max-gradient-norm", type=float, default=10.0)
    value.add_argument(
        "--sparse-chunk-loader", action=argparse.BooleanOptionalAction, default=True
    )
    _add_loader_args(value)
    value.set_defaults(handler=train_value)

    actor = subparsers.add_parser("train-actor")
    actor.add_argument("--dataset", type=Path, required=True)
    actor.add_argument("--labels", type=Path, required=True)
    actor.add_argument("--checkpoint", type=Path, required=True)
    actor.add_argument("--output-dir", type=Path, required=True)
    actor.add_argument("--epochs", type=int, default=100)
    actor.add_argument("--steps-per-epoch", type=int, default=None)
    actor.add_argument("--condition-dropout", type=float, default=0.3)
    actor.add_argument("--condition-hidden-dim", type=int, default=256)
    actor.add_argument("--actor-adapter-lr", type=float, default=1e-4)
    actor.add_argument("--actor-unet-lr", type=float, default=1e-4)
    actor.add_argument("--actor-obs-encoder-lr", type=float, default=1e-5)
    actor.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="cosine")
    actor.add_argument("--lr-warmup-steps", type=int, default=500)
    actor.add_argument("--lr-num-cycles", type=float, default=0.5)
    _add_loader_args(actor)
    actor.set_defaults(handler=train_actor)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if getattr(args, "hdf5_cache_mode", None) == "none":
        args.hdf5_cache_mode = None
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
