#!/usr/bin/env python3
"""RISE-style RGB chunk IDQL with a conditional joint DP actor by default."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo.diffusion_policy import replace_bn_with_gn
from robomimic.models.chunk_iql_nets import SequentialActionChunkEncoder, make_mlp
from robomimic.models.obs_core import CropRandomizer

from train_rgb_dp_idql import (
    REWARD_DEFINITIONS,
    RiseLateFusionMLP,
    RiseValueNetwork,
    action_normalization_stats_match,
    actor_matches_deployed_ema,
    actor_train_step,
    actor_trainability,
    align_shared_batch_actions,
    atomic_torch_save,
    build_single_loader,
    configure_actor_optimizer,
    dataset_audit,
    initialize_actor_from_deployed_ema,
    jsonable,
    make_tensorboard_writer,
    make_step_lr_scheduler,
    mean_metrics,
    parameter_count,
    replace_with_hardlink,
    restore_rng_state,
    rng_state,
    scalar_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DP = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/last.pth"
)
DEFAULT_DATASET = (
    ROOT
    / "datasets/square/idql/square_rgb_dp_idql_200demo_100success_94failure_task_reward.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "trained_models/square_rgb_dp_chunk_idql_rise"
    / "200demo_100success_94failure_h8_dynamics_task_reward"
)
ACTOR_CONDITION_DEFINITION = (
    "human_demo=1; success_rollout=0; failure_rollout=0"
)
DYNAMICS_PREDICTION_MODE = "actor_encoder_direct"


class RiseChunkActionValueNetwork(nn.Module):
    """Independent raw-observation Q network over an executable action chunk."""

    def __init__(
        self,
        *,
        obs_shapes: OrderedDict,
        goal_shapes: OrderedDict,
        encoder_kwargs: dict,
        action_dim: int,
        chunk_horizon: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        action_hidden_dim: int,
        num_attention_heads: int,
        num_action_conv_layers: int,
        dropout: float,
        late_fusion_key: str | None,
    ):
        super().__init__()
        observation_group_shapes = OrderedDict(obs=OrderedDict(obs_shapes))
        if goal_shapes is not None and len(goal_shapes) > 0:
            observation_group_shapes["goal"] = OrderedDict(goal_shapes)

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.has_goal = "goal" in observation_group_shapes
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.latent_dim = int(latent_dim)
        self.encoder_output_dim = int(self.nets["encoder"].output_shape()[0])
        self.late_fusion_keys = tuple(
            key.strip()
            for key in str(late_fusion_key or "").split(",")
            if key.strip()
        )
        late_fusion_dim = 0
        for key in self.late_fusion_keys:
            if key not in obs_shapes:
                raise KeyError(f"late_fusion_key={key} is absent from obs_shapes")
            late_fusion_dim += int(np.prod(obs_shapes[key]))

        context_dims = tuple(int(value) for value in hidden_dims[:-1]) + (
            self.latent_dim,
        )
        self.nets["context"] = RiseLateFusionMLP(
            input_dim=int(self.nets["encoder"].output_shape()[0]),
            hidden_dims=context_dims,
            late_fusion_dim=late_fusion_dim,
        )
        self.nets["context_norm"] = nn.LayerNorm(self.latent_dim)
        self.nets["action_encoder"] = SequentialActionChunkEncoder(
            action_dim=self.action_dim,
            chunk_horizon=self.chunk_horizon,
            context_dim=self.latent_dim,
            hidden_dim=int(action_hidden_dim),
            output_dim=self.latent_dim,
            num_heads=int(num_attention_heads),
            num_conv_layers=int(num_action_conv_layers),
            dropout=float(dropout),
        )
        self.nets["state_action_fusion"] = make_mlp(
            2 * self.latent_dim,
            (self.latent_dim,),
            self.latent_dim,
            dropout=float(dropout),
            final_layer_norm=True,
        )
        self.nets["dynamics_predictor"] = make_mlp(
            self.latent_dim,
            hidden_dims,
            self.encoder_output_dim,
            dropout=float(dropout),
        )
        self.nets["q_head"] = make_mlp(
            2 * self.latent_dim,
            hidden_dims,
            1,
            dropout=float(dropout),
        )

    def encode_context(self, obs_dict, goal_dict=None) -> torch.Tensor:
        inputs = {"obs": obs_dict}
        if self.has_goal:
            if goal_dict is None:
                raise ValueError("goal-conditioned chunk critic is missing goal observations")
            inputs["goal"] = goal_dict
        encoded = self.nets["encoder"](**inputs)
        late_parts = [
            obs_dict[key].flatten(start_dim=1) for key in self.late_fusion_keys
        ]
        late_fusion = torch.cat(late_parts, dim=-1) if late_parts else None
        return self.nets["context_norm"](
            self.nets["context"](encoded, late_fusion)
        )

    def action_and_successor(
        self,
        context: torch.Tensor,
        acts: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_repr = self.nets["action_encoder"](context, acts, action_mask)
        fused = self.nets["state_action_fusion"](
            torch.cat((context, action_repr), dim=-1)
        )
        delta = self.nets["transition_delta"](fused)
        gated_delta = self.nets["transition_gate"](fused) * delta
        predicted_next = self.nets["next_context_norm"](context + gated_delta)
        return action_repr, gated_delta, predicted_next

    def forward(
        self,
        obs_dict,
        acts,
        goal_dict=None,
        action_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        expected = (self.chunk_horizon, self.action_dim)
        if acts.ndim != 3 or tuple(acts.shape[1:]) != expected:
            raise ValueError(
                f"chunk critic expected actions [B,{expected[0]},{expected[1]}], "
                f"got {tuple(acts.shape)}"
            )
        context = self.encode_context(obs_dict, goal_dict)
        action_repr, delta, predicted_next = self.action_and_successor(
            context, acts, action_mask
        )
        q_inputs = [context, action_repr]
        if self.q_head_uses_delta:
            q_inputs.append(delta)
        q = self.nets["q_head"](torch.cat(q_inputs, dim=-1))
        if not return_aux:
            return q
        result = {
            "q": q,
            "context": context,
            "action_repr": action_repr,
        }
        if self.dynamics_prediction_mode == DYNAMICS_PREDICTION_MODE:
            result["predicted_next_encoder"] = predicted_next
        else:
            result["predicted_delta"] = delta
            result["predicted_next_context"] = predicted_next
        return result


def make_rise_chunk_value_networks(
    actor_algo,
    *,
    chunk_horizon: int,
    hidden_dims: tuple[int, ...],
    latent_dim: int,
    action_hidden_dim: int,
    num_attention_heads: int,
    num_action_conv_layers: int,
    dropout: float,
    num_critics: int = 2,
    critic_group_norm: bool = False,
    late_fusion_key: str | None = "robot0_gripper_qpos",
    q_head_uses_delta: bool = False,
    dynamics_prediction_mode: str = DYNAMICS_PREDICTION_MODE,
) -> tuple[nn.ModuleList, nn.ModuleList, RiseValueNetwork]:
    encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
        actor_algo.obs_config.encoder
    )
    critics = nn.ModuleList()
    for _ in range(int(num_critics)):
        critic = RiseChunkActionValueNetwork(
            obs_shapes=actor_algo.obs_shapes,
            goal_shapes=actor_algo.goal_shapes,
            encoder_kwargs=copy.deepcopy(encoder_kwargs),
            action_dim=int(actor_algo.ac_dim),
            chunk_horizon=int(chunk_horizon),
            hidden_dims=hidden_dims,
            latent_dim=int(latent_dim),
            action_hidden_dim=int(action_hidden_dim),
            num_attention_heads=int(num_attention_heads),
            num_action_conv_layers=int(num_action_conv_layers),
            dropout=float(dropout),
            late_fusion_key=late_fusion_key,
            q_head_uses_delta=q_head_uses_delta,
            dynamics_prediction_mode=dynamics_prediction_mode,
        )
        if critic_group_norm:
            critic = replace_bn_with_gn(critic)
        critics.append(critic)
    targets = copy.deepcopy(critics)
    vf = RiseValueNetwork(
        obs_shapes=actor_algo.obs_shapes,
        hidden_dims=hidden_dims,
        goal_shapes=actor_algo.goal_shapes,
        encoder_kwargs=copy.deepcopy(encoder_kwargs),
        late_fusion_key=late_fusion_key,
    )
    if critic_group_norm:
        vf = replace_bn_with_gn(vf)
    return critics, targets, vf


def copy_matching_encoder_state(
    critic: RiseChunkActionValueNetwork,
    source_state: dict[str, torch.Tensor],
) -> dict[str, int]:
    destination = critic.state_dict()
    matched = {}
    matched_groups = {"encoder": 0, "context": 0}
    for source_key, value in source_state.items():
        if source_key.startswith("nets.encoder."):
            destination_key = source_key
            group = "encoder"
        elif source_key.startswith("nets.mlp."):
            destination_key = source_key.replace(
                "nets.mlp.", "nets.context.", 1
            )
            group = "context"
        else:
            continue
        if (
            destination_key in destination
            and destination[destination_key].shape == value.shape
        ):
            matched[destination_key] = value
            matched_groups[group] += 1
    if not matched:
        raise RuntimeError(
            "no compatible one-step critic representation weights matched"
        )
    if matched_groups["encoder"] == 0:
        raise RuntimeError("no one-step critic observation-encoder weights matched")
    critic.load_state_dict(matched, strict=False)
    return {
        "tensor_count": int(len(matched)),
        "parameter_count": int(sum(value.numel() for value in matched.values())),
        "encoder_tensor_count": int(matched_groups["encoder"]),
        "context_tensor_count": int(matched_groups["context"]),
    }


def copy_deployed_dp_encoder_state(module: nn.Module, actor_algo) -> dict[str, int]:
    """Copy, but do not share, the deployed DP raw-observation encoder."""
    actor_nets = (
        actor_algo.ema.averaged_model
        if actor_algo.ema is not None
        else actor_algo.nets
    )
    source = actor_nets["policy"]["obs_encoder"]
    # ObservationGroupEncoder is reconstructed from config with BatchNorm, while
    # DiffusionPolicy converts its deployed visual encoder to GroupNorm after
    # construction. Match that deployed architecture before the strict state
    # copy, independently of the optional normalization used by critic heads.
    destination = replace_bn_with_gn(module.nets["encoder"])
    module.nets["encoder"] = destination
    source_state = source.state_dict()
    destination.load_state_dict(source_state, strict=True)
    return {
        "tensor_count": int(len(source_state)),
        "parameter_count": int(
            sum(value.numel() for value in source_state.values())
        ),
    }


@torch.no_grad()
def sync_actor_dynamics_target_encoder(
    target_encoder: nn.Module,
    actor_algo,
) -> dict[str, int]:
    """Hard-sync the frozen dynamics teacher from the deployed actor EMA."""
    source = deployed_actor_obs_encoder(actor_algo)
    source_state = source.state_dict()
    target_encoder.load_state_dict(source_state, strict=True)
    return {
        "tensor_count": int(len(source_state)),
        "parameter_count": int(
            sum(parameter.numel() for parameter in source.parameters())
        ),
    }


def match_encoder_normalization_to_checkpoint(
    module: nn.Module,
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Reconstruct the saved BN/GN encoder architecture before strict load."""
    prefix = "nets.encoder."
    checkpoint_batch_norm = any(
        key.startswith(prefix) and key.endswith(".running_mean")
        for key in state
    )
    constructed_state = module.state_dict()
    constructed_batch_norm = any(
        key.startswith(prefix) and key.endswith(".running_mean")
        for key in constructed_state
    )
    converted_to_group_norm = (
        constructed_batch_norm and not checkpoint_batch_norm
    )
    if converted_to_group_norm:
        module.nets["encoder"] = replace_bn_with_gn(module.nets["encoder"])
        constructed_state = module.state_dict()
        constructed_batch_norm = any(
            key.startswith(prefix) and key.endswith(".running_mean")
            for key in constructed_state
        )
    if constructed_batch_norm != checkpoint_batch_norm:
        raise RuntimeError(
            "could not reconstruct checkpoint observation-encoder "
            "normalization architecture"
        )
    return {
        "checkpoint_batch_norm": bool(checkpoint_batch_norm),
        "converted_to_group_norm": bool(converted_to_group_norm),
    }


def freeze_actor(actor_algo) -> dict[str, Any]:
    actor_algo.set_eval()
    actor_algo.nets.requires_grad_(False)
    if actor_algo.ema is not None:
        actor_algo.ema.averaged_model.requires_grad_(False)
    parameters = list(actor_algo.nets.parameters())
    ema_parameters = (
        list(actor_algo.ema.averaged_model.parameters())
        if actor_algo.ema is not None
        else []
    )
    if any(parameter.requires_grad for parameter in parameters + ema_parameters):
        raise RuntimeError("continuation actor still has trainable parameters")
    return {
        "num_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "num_trainable_parameters": 0,
        "has_ema": actor_algo.ema is not None,
        "ema_num_parameters": int(
            sum(parameter.numel() for parameter in ema_parameters)
        ),
    }


def has_condition_adapter(actor_algo) -> bool:
    policy_has_adapter = "condition_adapter" in actor_algo.nets["policy"]
    ema_has_adapter = (
        actor_algo.ema is not None
        and "condition_adapter" in actor_algo.ema.averaged_model["policy"]
    )
    if actor_algo.ema is not None and policy_has_adapter != ema_has_adapter:
        raise RuntimeError(
            "condition adapter must be present in both actor nets and EMA"
        )
    return bool(policy_has_adapter or ema_has_adapter)


def configure_conditioned_actor(actor_algo, args: argparse.Namespace) -> None:
    """Install and configure the human-demo condition used for train and eval."""
    if not bool(args.conditioned_actor):
        if has_condition_adapter(actor_algo):
            raise ValueError(
                "--no-conditioned-actor was requested, but the initial actor "
                "already contains a condition adapter"
            )
        return
    if not hasattr(actor_algo, "install_success_condition_adapter"):
        raise RuntimeError(
            "loaded DiffusionPolicy does not support a condition adapter"
        )
    actor_algo.install_success_condition_adapter(
        hidden_dim=int(args.condition_hidden_dim)
    )
    adapter = actor_algo.nets["policy"]["condition_adapter"]
    if int(adapter.hidden_dim) != int(args.condition_hidden_dim):
        raise ValueError(
            f"actor condition adapter hidden_dim={adapter.hidden_dim} does not "
            f"match requested condition_hidden_dim={args.condition_hidden_dim}"
        )
    if not has_condition_adapter(actor_algo):
        raise RuntimeError("failed to install actor condition adapter")
    actor_algo.set_inference_success_condition(
        success_condition=1.0,
        condition_mask=1.0,
    )
    actor_algo.success_condition_dropout = float(args.condition_dropout)


def gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, indices]


def source_condition_labels(
    raw_batch: dict,
    *,
    current_index: int,
) -> torch.Tensor:
    """Read outcome conditioning labels independently of critic rewards."""
    batch_size = int(raw_batch["actions"].shape[0])
    labels_by_time = raw_batch.get("actor_condition")
    label_source = "actor_condition"
    if labels_by_time is None:
        raise KeyError(
            "conditioned actor batch is missing actor_condition; rebuild the "
            "mixed dataset with the current build_rgb_dp_idql_dataset.py"
        )
    if labels_by_time.ndim < 2 or labels_by_time.shape[1] <= current_index:
        raise ValueError(
            f"shared batch {label_source} does not contain the current "
            f"transition at index {current_index}: "
            f"shape={tuple(labels_by_time.shape)}"
        )
    labels = labels_by_time[:, current_index].reshape(batch_size, -1)
    if labels.shape[1] != 1:
        raise ValueError(
            "actor condition requires one scalar outcome label per "
            f"transition, got shape={tuple(labels.shape)}"
        )
    labels = labels[:, 0].float()
    zeros = torch.zeros_like(labels)
    ones = torch.ones_like(labels)
    is_negative = torch.isclose(labels, zeros, atol=1e-6, rtol=0.0)
    is_positive = torch.isclose(labels, ones, atol=1e-6, rtol=0.0)
    if not torch.all(is_negative | is_positive):
        invalid = labels[~(is_negative | is_positive)]
        raise ValueError(
            f"actor condition expected {label_source} values 0 or 1, got "
            f"values={invalid[:8].detach().cpu().tolist()}"
        )
    return is_positive.to(dtype=torch.float32)


def source_expert_labels(
    raw_batch: dict,
    *,
    current_index: int,
) -> torch.Tensor:
    """Read the expert-source mask used only by --actor-demo-only."""
    batch_size = int(raw_batch["actions"].shape[0])
    labels_by_time = raw_batch.get("source_is_expert")
    label_source = "source_is_expert"
    if labels_by_time is None:
        labels_by_time = raw_batch["rewards"]
        label_source = "legacy RISE rewards"
    if labels_by_time.ndim < 2 or labels_by_time.shape[1] <= current_index:
        raise ValueError(
            f"shared batch {label_source} does not contain the current "
            f"transition at index {current_index}: "
            f"shape={tuple(labels_by_time.shape)}"
        )
    labels = labels_by_time[:, current_index].reshape(batch_size, -1)
    if labels.shape[1] != 1:
        raise ValueError(
            "expert-source mask requires one scalar label per transition, "
            f"got shape={tuple(labels.shape)}"
        )
    labels = labels[:, 0].float()
    zeros = torch.zeros_like(labels)
    ones = torch.ones_like(labels)
    is_non_expert = torch.isclose(labels, zeros, atol=1e-6, rtol=0.0)
    is_expert = torch.isclose(labels, ones, atol=1e-6, rtol=0.0)
    if not torch.all(is_non_expert | is_expert):
        invalid = labels[~(is_non_expert | is_expert)]
        raise ValueError(
            f"expert-source mask expected {label_source} values 0 or 1, got "
            f"values={invalid[:8].detach().cpu().tolist()}"
        )
    return is_expert


def select_actor_rows(
    raw_batch: dict,
    *,
    current_index: int,
    demo_only: bool,
) -> tuple[dict, torch.Tensor]:
    """Return the actor view of a shared batch and its selected row mask."""
    batch_size = int(raw_batch["actions"].shape[0])
    if not demo_only:
        rows = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=raw_batch["actions"].device,
        )
        return raw_batch, rows

    is_demo = source_expert_labels(
        raw_batch,
        current_index=current_index,
    )

    def take_rows(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0 or int(tensor.shape[0]) != batch_size:
            raise ValueError(
                "all shared-batch tensors must have the same leading batch "
                f"dimension {batch_size}, got shape={tuple(tensor.shape)}"
            )
        return tensor[is_demo]

    return TensorUtils.map_tensor(raw_batch, take_rows), is_demo


def add_actor_condition(
    actor_batch: dict,
    condition_labels: torch.Tensor,
) -> dict:
    """Attach explicit condition and mask tensors consumed by DiffusionPolicy."""
    batch_size = int(actor_batch["actions"].shape[0])
    labels = condition_labels.reshape(-1).to(dtype=torch.float32)
    if int(labels.shape[0]) != batch_size:
        raise ValueError(
            "actor condition batch mismatch: "
            f"labels={tuple(labels.shape)}, batch_size={batch_size}"
        )
    conditioned_batch = dict(actor_batch)
    conditioned_batch["success_condition"] = labels
    conditioned_batch["success_condition_mask"] = torch.ones_like(labels)
    return conditioned_batch


def audit_actor_conditions(
    dataset_path: Path,
    *,
    reward_mode: str,
) -> dict[str, Any]:
    """Validate actor labels independently of the selected critic reward."""
    episode_counts = {
        "human_demo": 0,
        "success_rollout": 0,
        "failure_rollout": 0,
    }
    transition_counts = {key: 0 for key in episode_counts}
    source_names = {
        "expert": "human_demo",
        "non_expert_success": "success_rollout",
        "non_expert_failure": "failure_rollout",
    }
    with h5py.File(dataset_path, "r") as handle:
        for episode_key, episode in handle["data"].items():
            source = episode.attrs.get("rise_source")
            if isinstance(source, bytes):
                source = source.decode("utf-8")
            source = str(source)
            if source not in source_names:
                raise ValueError(
                    f"data/{episode_key} has unsupported rise_source={source!r}"
                )
            source_name = source_names[source]
            expected = 0.0 if source == "non_expert_failure" else 1.0
            if "actor_condition" not in episode:
                raise ValueError(
                    f"data/{episode_key} is missing actor_condition; rebuild "
                    f"the {reward_mode}-reward dataset so human and successful rollout "
                    "rows use condition 1 and failure rows use condition 0"
                )
            label_key = "actor_condition"
            labels = np.asarray(episode[label_key][:], dtype=np.float32)
            if labels.size < 1 or not np.allclose(
                labels,
                expected,
                atol=1e-6,
                rtol=0.0,
            ):
                unique = np.unique(labels).tolist()
                raise ValueError(
                    f"data/{episode_key} source={source!r} must map to "
                    f"condition={expected}, got {label_key}={unique[:8]}"
                )
            episode_counts[source_name] += 1
            transition_counts[source_name] += int(labels.size)
    if episode_counts["human_demo"] == 0:
        raise ValueError("conditioned actor dataset contains no human demos")
    if (
        episode_counts["success_rollout"]
        + episode_counts["failure_rollout"]
        == 0
    ):
        raise ValueError("conditioned actor dataset contains no rollout data")
    return {
        "definition": ACTOR_CONDITION_DEFINITION,
        "positive_sources": ["human_demo", "success_rollout"],
        "negative_sources": ["failure_rollout"],
        "dataset_key": "actor_condition",
        "condition_mask": 1.0,
        "episode_counts": episode_counts,
        "transition_counts": transition_counts,
    }


def process_chunk_batch(
    raw_batch: dict,
    actor_algo,
    obs_normalization_stats,
    *,
    chunk_horizon: int,
    discount: float,
) -> dict[str, Any]:
    """Extract a semi-MDP transition at the first executable DP action."""
    current_index = int(actor_algo.algo_config.horizon.observation_horizon) - 1
    end_index = current_index + int(chunk_horizon)
    if raw_batch["actions"].shape[1] < end_index:
        raise ValueError(
            f"batch sequence length {raw_batch['actions'].shape[1]} is shorter "
            f"than required index {end_index}"
        )

    actions = raw_batch["actions"][:, current_index:end_index]
    rewards = raw_batch["rewards"][:, current_index:end_index].float()
    dones = raw_batch["dones"][:, current_index:end_index].float()
    if rewards.ndim == 3:
        rewards = rewards.squeeze(-1)
    if dones.ndim == 3:
        dones = dones.squeeze(-1)

    continuation = 1.0 - (dones > 0.5).to(rewards.dtype)
    action_mask = torch.cat(
        (
            torch.ones_like(continuation[:, :1]),
            torch.cumprod(continuation[:, :-1], dim=1),
        ),
        dim=1,
    )
    valid_length = action_mask.sum(dim=1)
    terminal = ((dones > 0.5).to(rewards.dtype) * action_mask).amax(dim=1)
    powers = torch.arange(
        int(chunk_horizon), device=rewards.device, dtype=rewards.dtype
    )
    discounts = torch.pow(rewards.new_tensor(float(discount)), powers)
    chunk_return = (rewards * action_mask * discounts[None]).sum(dim=1)
    next_indices = current_index + valid_length.to(torch.long) - 1
    exact_next = (
        (valid_length == float(chunk_horizon)) & (terminal < 0.5)
    ).to(rewards.dtype)

    batch = {
        "obs": {
            key: raw_batch["obs"][key][:, current_index]
            for key in actor_algo.obs_shapes
        },
        "next_obs": {
            key: gather_time(raw_batch["next_obs"][key], next_indices)
            for key in actor_algo.obs_shapes
        },
        "actions": actions * action_mask.unsqueeze(-1),
        "action_mask": action_mask,
        "reward": chunk_return.reshape(-1, 1),
        "terminal": terminal.reshape(-1, 1),
        "valid_length": valid_length.reshape(-1, 1),
        "exact_next": exact_next.reshape(-1, 1),
        "goal_obs": raw_batch.get("goal_obs"),
    }
    batch = TensorUtils.to_device(TensorUtils.to_float(batch), actor_algo.device)
    batch = actor_algo.postprocess_batch_for_training(
        batch, obs_normalization_stats=obs_normalization_stats
    )
    if not torch.isfinite(batch["actions"]).all():
        raise ValueError("chunk critic actions contain non-finite values")
    action_min = float(batch["actions"].min().item())
    action_max = float(batch["actions"].max().item())
    if action_min < -1.001 or action_max > 1.001:
        raise ValueError(
            "chunk actions are outside the pretrained normalized space: "
            f"min={action_min:.6f}, max={action_max:.6f}"
        )
    batch["actions"] = batch["actions"].clamp(-1.0, 1.0)
    return batch


def masked_dynamics_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    exact_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = exact_next.reshape(-1) > 0.5
    if not torch.any(rows):
        zero = predicted.new_tensor(0.0)
        return zero, zero, zero
    predicted = predicted[rows]
    target = target[rows]
    if predicted.shape != target.shape:
        raise ValueError(
            "dynamics prediction and actor-encoder target shapes differ: "
            f"predicted={tuple(predicted.shape)}, target={tuple(target.shape)}"
        )
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    smooth_l1 = F.smooth_l1_loss(predicted, target)
    cosine = (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()
    rmse = torch.sqrt(F.mse_loss(predicted, target).clamp_min(1e-12))
    return smooth_l1, cosine, rmse


@contextmanager
def fork_rng_with_seed(seed: int, device: torch.device):
    """Temporarily seed CPU and the active CUDA generator for paired crops."""
    cuda_devices: list[int] = []
    cuda_index: int | None = None
    if device.type == "cuda":
        cuda_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        cuda_devices = [cuda_index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_index is not None:
            torch.cuda.default_generators[cuda_index].manual_seed(int(seed))
        yield


def configure_target_random_crops(networks: nn.ModuleList) -> None:
    """Keep frozen targets deterministic except for training-style crop draws."""
    networks.eval().requires_grad_(False)
    for network in networks:
        configure_encoder_target_random_crops(network.nets["encoder"])


def configure_encoder_target_random_crops(encoder: nn.Module) -> None:
    """Freeze an encoder while retaining training-style random crop draws."""
    encoder.eval().requires_grad_(False)
    for module in encoder.modules():
        if isinstance(module, CropRandomizer):
            module.train()


def compute_chunk_losses(
    critics: nn.ModuleList,
    targets: nn.ModuleList,
    dynamics_target_encoder: nn.Module,
    vf: RiseValueNetwork,
    batch: dict[str, Any],
    *,
    discount: float,
    expectile: float,
    use_huber: bool,
    dynamics_weight: float,
    dynamics_cosine_weight: float,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    device = batch["actions"].device
    crop_seeds = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (len(critics) + 2,),
        device="cpu",
    ).tolist()
    critic_crop_seeds = crop_seeds[: len(critics)]
    next_v_crop_seed = crop_seeds[-2]
    vf_crop_seed = crop_seeds[-1]

    outputs = []
    for critic, crop_seed in zip(critics, critic_crop_seeds):
        with fork_rng_with_seed(crop_seed, device):
            outputs.append(
                critic(
                    obs_dict=batch["obs"],
                    acts=batch["actions"],
                    action_mask=batch["action_mask"],
                    goal_dict=batch["goal_obs"],
                    return_aux=True,
                )
            )
    with torch.no_grad():
        with fork_rng_with_seed(next_v_crop_seed, device):
            next_v = vf(
                obs_dict=batch["next_obs"],
                goal_dict=batch["goal_obs"],
            )
        bootstrap = torch.pow(
            batch["valid_length"].new_tensor(float(discount)),
            batch["valid_length"],
        )
        q_backup = (
            batch["reward"]
            + (1.0 - batch["terminal"]) * bootstrap * next_v
        )
        target_qs = []
        for target in targets:
            with fork_rng_with_seed(vf_crop_seed, device):
                target_qs.append(
                    target(
                        obs_dict=batch["obs"],
                        acts=batch["actions"],
                        action_mask=batch["action_mask"],
                        goal_dict=batch["goal_obs"],
                    )
                )
        target_q_min = torch.cat(target_qs, dim=1).min(
            dim=1, keepdim=True
        ).values
        target_next_encoder_features = []
        for crop_seed in critic_crop_seeds:
            with fork_rng_with_seed(crop_seed, device):
                target_next_encoder_features.append(
                    dynamics_target_encoder(
                        obs=batch["next_obs"],
                    )
                )

    regression = F.smooth_l1_loss if use_huber else F.mse_loss
    critic_losses = []
    dynamics_l1 = []
    dynamics_cosine = []
    dynamics_rmse = []
    weighted_dynamics_losses = []
    q_losses = []
    for output, target_features in zip(
        outputs,
        target_next_encoder_features,
    ):
        q_loss = regression(output["q"], q_backup)
        dyn_l1, dyn_cos, dyn_rmse = masked_dynamics_losses(
            output["predicted_next_encoder"],
            target_features,
            batch["exact_next"],
        )
        dynamics_loss = (
            float(dynamics_weight) * dyn_l1
            + float(dynamics_cosine_weight) * dyn_cos
        )
        critic_losses.append(q_loss + dynamics_loss)
        q_losses.append(q_loss)
        dynamics_l1.append(dyn_l1)
        dynamics_cosine.append(dyn_cos)
        dynamics_rmse.append(dyn_rmse)
        weighted_dynamics_losses.append(dynamics_loss)

    with fork_rng_with_seed(vf_crop_seed, device):
        vf_pred = vf(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
    vf_error = vf_pred - target_q_min
    vf_weight = torch.where(
        vf_error > 0.0, 1.0 - float(expectile), float(expectile)
    )
    vf_loss = (vf_weight * vf_error.square()).mean()
    q_predictions = torch.cat([output["q"] for output in outputs], dim=1)
    info = {
        **{
            f"critic/q{index + 1}_loss": loss.detach()
            for index, loss in enumerate(q_losses)
        },
        **{
            f"critic/q{index + 1}_total_loss": loss.detach()
            for index, loss in enumerate(critic_losses)
        },
        **{
            f"critic/q{index + 1}_mean": output["q"].mean().detach()
            for index, output in enumerate(outputs)
        },
        "critic/q_target_mean": q_backup.mean().detach(),
        "critic/q_ensemble_std": q_predictions.std(dim=1).mean().detach(),
        "vf/loss": vf_loss.detach(),
        "vf/value_mean": vf_pred.mean().detach(),
        "vf/target_q_min_mean": target_q_min.mean().detach(),
        "vf/error_mean": vf_error.mean().detach(),
        "dynamics/l1": torch.stack(dynamics_l1).mean().detach(),
        "dynamics/cosine": torch.stack(dynamics_cosine).mean().detach(),
        "dynamics/rmse": torch.stack(dynamics_rmse).mean().detach(),
        "dynamics/weighted_loss": torch.stack(
            weighted_dynamics_losses
        ).mean().detach(),
        "dynamics/effective_l1_weight": q_predictions.new_tensor(
            float(dynamics_weight)
        ),
        "dynamics/effective_cosine_weight": q_predictions.new_tensor(
            float(dynamics_cosine_weight)
        ),
        "dynamics/exact_next_fraction": batch["exact_next"].mean().detach(),
        "dynamics/target_feature_std": torch.stack(
            [
                F.normalize(features, dim=-1).std(dim=0).mean()
                for features in target_next_encoder_features
            ]
        ).mean().detach(),
        "dynamics/target_feature_norm": torch.stack(
            [
                features.norm(dim=-1).mean()
                for features in target_next_encoder_features
            ]
        ).mean().detach(),
        "data/chunk_return_mean": batch["reward"].mean().detach(),
        "data/terminal_fraction": batch["terminal"].mean().detach(),
        "data/valid_length_mean": batch["valid_length"].mean().detach(),
        "data/action_abs_mean": batch["actions"].abs().mean().detach(),
        "data/action_min": batch["actions"].min().detach(),
        "data/action_max": batch["actions"].max().detach(),
    }
    return critic_losses, vf_loss, info


def make_critic_optimizer(
    critic: nn.Module,
    critic_lr: float,
    encoder_lr: float,
) -> torch.optim.Optimizer:
    representation_keys = ["encoder"]
    if "context" in critic.nets:
        representation_keys.extend(("context", "context_norm"))
    representation_parameters = [
        parameter
        for key in representation_keys
        for parameter in critic.nets[key].parameters()
    ]
    representation_ids = {
        id(parameter) for parameter in representation_parameters
    }
    head_parameters = [
        parameter
        for parameter in critic.parameters()
        if id(parameter) not in representation_ids
    ]
    return torch.optim.Adam(
        [
            {"params": head_parameters, "lr": float(critic_lr)},
            {
                "params": representation_parameters,
                "lr": float(encoder_lr),
            },
        ]
    )


def set_representation_trainable(
    critics: nn.ModuleList,
    trainable: bool,
) -> None:
    for critic in critics:
        for key in ("encoder", "context", "context_norm"):
            critic.nets[key].requires_grad_(bool(trainable))


def set_vf_encoder_trainable(
    vf: RiseValueNetwork,
    trainable: bool,
) -> None:
    """Freeze only V's raw-observation encoder, never its value head."""
    vf.nets["encoder"].requires_grad_(bool(trainable))


def update_networks(
    critics: nn.ModuleList,
    targets: nn.ModuleList,
    vf: RiseValueNetwork,
    critic_optimizers: list[torch.optim.Optimizer],
    vf_optimizer: torch.optim.Optimizer,
    critic_losses: list[torch.Tensor],
    vf_loss: torch.Tensor,
    *,
    target_tau: float,
    max_gradient_norm: float | None,
) -> None:
    for critic, target, optimizer, loss in zip(
        critics, targets, critic_optimizers, critic_losses
    ):
        TorchUtils.backprop_for_loss(
            net=critic,
            optim=optimizer,
            loss=loss,
            max_grad_norm=max_gradient_norm,
            retain_graph=False,
        )
        with torch.no_grad():
            TorchUtils.soft_update(critic, target, tau=float(target_tau))
    TorchUtils.backprop_for_loss(
        net=vf,
        optim=vf_optimizer,
        loss=vf_loss,
        max_grad_norm=max_gradient_norm,
        retain_graph=False,
    )


def validate_source(source: dict, args: argparse.Namespace) -> None:
    if not source.get("rise_style_rgb_idql", False):
        raise ValueError("source checkpoint is not a RISE-style RGB IDQL checkpoint")
    if source.get("rise_style_rgb_chunk_idql", False):
        raise ValueError("source-idql-checkpoint must be the one-step baseline")
    if str(source.get("task", args.task)) != str(args.task):
        raise ValueError(
            f"source task={source.get('task')} does not match task={args.task}"
        )
    source_num_critics = int(source.get("num_critics", 0))
    if source_num_critics < 2:
        raise ValueError("source checkpoint does not contain twin critics")
    if source_num_critics != int(args.num_critics):
        raise ValueError(
            f"num_critics={args.num_critics} does not match source "
            f"num_critics={source_num_critics}"
        )
    if tuple(source.get("critic_hidden_dims", ())) != tuple(args.critic_hidden_dims):
        raise ValueError("critic hidden dimensions must match the source checkpoint")
    if bool(source.get("critic_group_norm", False)) != bool(args.critic_group_norm):
        raise ValueError("critic-group-norm must match the source checkpoint")
    source_late_fusion = source.get("critic_late_fusion_key")
    if source_late_fusion != args.critic_late_fusion_key:
        raise ValueError(
            f"critic_late_fusion_key={args.critic_late_fusion_key!r} does not "
            f"match source value {source_late_fusion!r}"
        )


def validate_chunk_source(source: dict, args: argparse.Namespace) -> None:
    """Validate a complete chunk checkpoint before round-to-round warm start."""
    if not source.get("rise_style_rgb_chunk_idql", False):
        raise ValueError(
            "source-chunk-idql-checkpoint is not a chunk IDQL checkpoint"
        )
    if str(source.get("task", "")) != str(args.task):
        raise ValueError(
            f"source task={source.get('task')!r} does not match task={args.task!r}"
        )
    source_reward_mode = str(source.get("reward_mode", "rise"))
    if source_reward_mode != str(args.reward_mode):
        raise ValueError(
            f"source reward_mode={source_reward_mode!r} does not match "
            f"requested reward_mode={args.reward_mode!r}"
        )
    if not bool(source.get("conditioned_actor", False)):
        raise ValueError(
            "round-2 joint warm start requires a conditioned source actor"
        )
    source_condition_definition = str(
        source.get("actor_condition_label_definition", "")
    )
    if source_condition_definition != ACTOR_CONDITION_DEFINITION:
        raise ValueError(
            "source actor condition definition="
            f"{source_condition_definition!r} does not match required "
            f"{ACTOR_CONDITION_DEFINITION!r}"
        )
    if not bool(args.conditioned_actor):
        raise ValueError(
            "source_chunk_idql_joint requires --conditioned-actor so human "
            "demonstrations remain condition 1 and every rollout remains 0"
        )

    source_args = source.get("args", {})
    source_condition_hidden_dim = int(
        source_args.get(
            "condition_hidden_dim",
            source.get("actor_condition_hidden_dim", -1),
        )
    )
    if source_condition_hidden_dim != int(args.condition_hidden_dim):
        raise ValueError(
            "condition-hidden-dim must match source chunk checkpoint: "
            f"requested={args.condition_hidden_dim}, "
            f"source={source_condition_hidden_dim}"
        )

    expected_fields = {
        "critic_chunk_horizon": int(args.chunk_horizon),
        "critic_hidden_dims": tuple(int(x) for x in args.critic_hidden_dims),
        "critic_latent_dim": int(args.latent_dim),
        "critic_action_hidden_dim": int(args.action_hidden_dim),
        "critic_num_attention_heads": int(args.num_attention_heads),
        "critic_num_action_conv_layers": int(args.num_action_conv_layers),
        "critic_dropout": float(args.dropout),
        "num_critics": int(args.num_critics),
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
    }
    integer_fields = {
        "critic_chunk_horizon",
        "critic_latent_dim",
        "critic_action_hidden_dim",
        "critic_num_attention_heads",
        "critic_num_action_conv_layers",
        "num_critics",
    }
    for field, expected in expected_fields.items():
        value = source.get(field)
        if field == "critic_hidden_dims":
            value = tuple(value or ())
        elif field in integer_fields:
            value = int(value if value is not None else -1)
        elif field == "critic_dropout":
            value = float(value if value is not None else float("nan"))
        elif field == "critic_group_norm":
            value = bool(value)
        if value != expected:
            raise ValueError(
                f"{field}={expected!r} does not match source value {value!r}"
            )

    if str(source.get("dynamics_prediction_mode", "")) != DYNAMICS_PREDICTION_MODE:
        raise ValueError(
            "source dynamics prediction mode does not match the current "
            f"{DYNAMICS_PREDICTION_MODE!r} architecture"
        )
    if tuple(source.get("critic_q_head_inputs", ())) != (
        "context",
        "action_repr",
    ):
        raise ValueError("source chunk checkpoint uses an incompatible Q head")
    required_keys = (
        "actor_model",
        "critics",
        "critic_targets",
        "dynamics_target_encoder",
        "vf",
        "pretrained_dp_checkpoint",
        "action_normalization_stats",
    )
    missing = [key for key in required_keys if key not in source]
    if missing:
        raise ValueError(
            f"source chunk checkpoint is missing required fields: {missing}"
        )
    if len(source["critics"]) != int(args.num_critics):
        raise ValueError("source checkpoint critic count is inconsistent")
    if len(source["critic_targets"]) != int(args.num_critics):
        raise ValueError("source checkpoint target critic count is inconsistent")


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    actor_model: dict,
    actor_ema_optimization_step: int,
    pretrained_dp_checkpoint: str,
    critics: nn.ModuleList,
    targets: nn.ModuleList,
    dynamics_target_encoder: nn.Module,
    dynamics_target_last_sync_step: int,
    vf: RiseValueNetwork,
    critic_optimizers: list[torch.optim.Optimizer],
    vf_optimizer: torch.optim.Optimizer,
    critic_lr_schedulers: list[Any],
    vf_lr_scheduler: Any,
    action_stats: dict,
    epoch: int,
    global_step: int,
    history: list[dict],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "rise_style_rgb_idql": True,
        "rise_style_rgb_chunk_idql": True,
        "hybrid_dp_chunk_actor_iql": True,
        "visual_critic_idql": True,
        "critic_q_head_inputs": ("context", "action_repr"),
        "critic_representation_modules": (
            "encoder",
            "context",
            "context_norm",
        ),
        "actor_model": actor_model,
        "critics": [critic.state_dict() for critic in critics],
        "critic_targets": [target.state_dict() for target in targets],
        "dynamics_prediction_mode": DYNAMICS_PREDICTION_MODE,
        "dynamics_prediction_target": "normalized_actor_encoder_features",
        "dynamics_target_encoder": dynamics_target_encoder.state_dict(),
        "dynamics_target_last_sync_step": int(
            dynamics_target_last_sync_step
        ),
        "vf": vf.state_dict(),
        "critic_optimizers": [
            optimizer.state_dict() for optimizer in critic_optimizers
        ],
        "vf_optimizer": vf_optimizer.state_dict(),
        "critic_lr_schedulers": [
            scheduler.state_dict() if scheduler is not None else None
            for scheduler in critic_lr_schedulers
        ],
        "vf_lr_scheduler": (
            vf_lr_scheduler.state_dict()
            if vf_lr_scheduler is not None
            else None
        ),
        "args": vars(args),
        "epoch": int(epoch),
        "step": int(global_step),
        "history": history,
        "chunk_initialization": str(args.initialization),
        "source_idql_checkpoint": (
            str(args.source_idql_checkpoint)
            if args.source_idql_checkpoint is not None
            else None
        ),
        "source_chunk_idql_checkpoint": (
            str(args.source_chunk_idql_checkpoint)
            if args.source_chunk_idql_checkpoint is not None
            else None
        ),
        "pretrained_dp_checkpoint": str(pretrained_dp_checkpoint),
        "task": str(args.task),
        "dataset": str(args.dataset),
        "single_dataloader": True,
        "sampling": "uniform_shuffled_SequenceDataset_indices",
        "reward_mode": str(args.reward_mode),
        "reward_definition": REWARD_DEFINITIONS[args.reward_mode],
        "critic_reward_source": (
            "rewards=source_environment_task_reward"
            if args.reward_mode == "task"
            else "rewards=expert_1_non_expert_0"
        ),
        "actor_training_objective": (
            (
                "human_conditioned_diffusion_BC_all_mixed_rows_"
                "human_1_all_rollouts_0_"
                + (
                    "from_source_chunk_IDQL_actor"
                    if args.initialization == "source_chunk_idql_joint"
                    else "from_pretrained_DP_ema"
                )
            )
            if trains_joint_actor(args) and args.conditioned_actor
            else (
                "full_diffusion_BC_all_mixed_rows"
                if trains_joint_actor(args)
                else (
                    "frozen_deployed_dp_actor_from_pretrained_checkpoint"
                    if args.initialization == "pretrained_dp_frozen"
                    else "frozen_one_step_idql_posttrained_ema_actor"
                )
                )
            )
        ),
        "conditioned_actor": bool(args.conditioned_actor),
        "actor_condition_label_definition": (
            ACTOR_CONDITION_DEFINITION
            if args.conditioned_actor
            else None
        ),
        "actor_condition_source": (
            "source_is_expert_at_current_transition"
            if args.conditioned_actor
            else None
        ),
        "actor_condition_mask": (
            "1_for_every_actor_training_row"
            if args.conditioned_actor
            else None
        ),
        "actor_inference_condition": 1.0 if args.conditioned_actor else None,
        "actor_inference_condition_mask": (
            1.0 if args.conditioned_actor else None
        ),
        "actor_condition_dropout": (
            float(args.condition_dropout) if args.conditioned_actor else None
        ),
        "actor_condition_hidden_dim": (
            int(args.condition_hidden_dim) if args.conditioned_actor else None
        ),
        "actor_source_mask": (
            "none_all_shared_batch_rows"
            if trains_joint_actor(args)
            else "none_actor_frozen"
        ),
        "critic_source_mask": "none_all_shared_batch_rows",
        "critic_training_objective": (
            "task_reward_semi_mdp_chunk_iql_with_actor_encoder_dynamics"
            if args.reward_mode == "task"
            else "rise_semi_mdp_chunk_iql_with_actor_encoder_dynamics"
        ),
        "critic_input_mode": "independent_raw_observation_chunk_encoders",
        "critic_action_space": "pretrained_dp_normalized_action_chunk",
        "critic_hidden_dims": tuple(int(x) for x in args.critic_hidden_dims),
        "critic_chunk_horizon": int(args.chunk_horizon),
        "critic_latent_dim": int(args.latent_dim),
        "critic_action_hidden_dim": int(args.action_hidden_dim),
        "critic_num_attention_heads": int(args.num_attention_heads),
        "critic_num_action_conv_layers": int(args.num_action_conv_layers),
        "critic_dropout": float(args.dropout),
        "num_critics": int(args.num_critics),
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
        "action_dim": int(args.action_dim),
        "action_normalization_stats": copy.deepcopy(action_stats),
        "observation_horizon": int(args.observation_horizon),
        "actor_prediction_horizon": int(args.actor_prediction_horizon),
        "actor_action_horizon": int(args.actor_action_horizon),
        "discount": float(args.discount),
        "expectile": float(args.expectile),
        "target_tau": float(args.target_tau),
        "dynamics_target_source": (
            "periodic_deployed_actor_ema_obs_encoder"
            if trains_joint_actor(args)
            else "fixed_deployed_actor_ema_obs_encoder"
        ),
        "dynamics_target_sync_interval": int(
            args.dynamics_target_sync_interval
        ),
        "dynamics_weight": float(args.dynamics_weight),
        "dynamics_cosine_weight": float(args.dynamics_cosine_weight),
        "dynamics_warmup_steps": int(args.dynamics_warmup_steps),
        "augmentation": (
            "paired_training_random_crops_online_dynamics_target_and_vf_target"
        ),
        "q_loss": "huber" if args.use_huber else "mse",
        "max_gradient_norm": (
            float(args.max_gradient_norm)
            if args.max_gradient_norm is not None
            else None
        ),
        "critic_vf_lr_scheduler": str(args.critic_vf_lr_scheduler),
        "critic_vf_lr_warmup_steps": int(
            args.critic_vf_lr_warmup_steps
        ),
        "critic_vf_lr_total_steps": int(args.critic_vf_lr_total_steps),
        "critic_vf_lr_num_cycles": float(args.critic_vf_lr_num_cycles),
        "vf_encoder_freeze_steps": int(args.vf_encoder_freeze_steps),
        "actor_lr": float(args.actor_lr),
        "actor_lr_scheduler": str(args.actor_lr_scheduler),
        "actor_lr_warmup_steps": int(args.actor_lr_warmup_steps),
        "actor_lr_total_steps": int(args.actor_lr_total_steps),
        "actor_lr_num_cycles": float(args.actor_lr_num_cycles),
        "actor_frozen": bool(not trains_joint_actor(args)),
        "actor_encoder_trainable": bool(trains_joint_actor(args)),
        "actor_ema_optimization_step": int(actor_ema_optimization_step),
        "rng_state": rng_state(),
        "loader_generator_state": loader_generator.get_state(),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")

    resume_state = None
    source_for_warm_start = None
    if args.resume_checkpoint is not None:
        resume_state = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not resume_state.get("rise_style_rgb_chunk_idql", False):
            raise ValueError("resume checkpoint is not a chunk IDQL checkpoint")

        saved_dynamics_mode = str(
            resume_state.get(
                "dynamics_prediction_mode",
                "",
            )
        )
        if saved_dynamics_mode != DYNAMICS_PREDICTION_MODE:
            raise ValueError(
                f"resume dynamics_prediction_mode={saved_dynamics_mode!r} "
                f"does not match {DYNAMICS_PREDICTION_MODE!r}; start a "
                "fresh output directory"
            )
        saved_sync_interval = int(
            resume_state.get("args", {}).get(
                "dynamics_target_sync_interval",
                -1,
            )
        )
        if saved_sync_interval != int(args.dynamics_target_sync_interval):
            raise ValueError(
                "resume dynamics_target_sync_interval="
                f"{saved_sync_interval} does not match requested "
                f"{args.dynamics_target_sync_interval}"
            )

        saved_initialization = str(
            resume_state.get("chunk_initialization", "source_idql_frozen")
        )
        if saved_initialization != args.initialization:
            raise ValueError(
                f"resume initialization={saved_initialization} does not match "
                f"requested initialization={args.initialization}"
            )
        saved_reward_mode = str(
            resume_state.get("args", {}).get("reward_mode", "rise")
        )
        if saved_reward_mode != str(args.reward_mode):
            raise ValueError(
                f"resume reward_mode={saved_reward_mode} does not match "
                f"requested reward_mode={args.reward_mode}"
            )
        if saved_initialization in JOINT_ACTOR_INITIALIZATIONS:
            saved_args = resume_state.get("args", {})
            saved_actor_scheduler = str(
                saved_args.get("actor_lr_scheduler", "constant")
            )
            if saved_actor_scheduler != str(args.actor_lr_scheduler):
                raise ValueError(
                    f"resume actor_lr_scheduler={saved_actor_scheduler} does "
                    "not match requested "
                    f"actor_lr_scheduler={args.actor_lr_scheduler}"
                )
            saved_actor_warmup = int(
                saved_args.get("actor_lr_warmup_steps", 0)
            )
            if saved_actor_warmup != int(args.actor_lr_warmup_steps):
                raise ValueError(
                    f"resume actor_lr_warmup_steps={saved_actor_warmup} does "
                    "not match requested "
                    f"actor_lr_warmup_steps={args.actor_lr_warmup_steps}"
                )
            saved_actor_cycles = float(
                saved_args.get("actor_lr_num_cycles", 0.5)
            )
            if saved_actor_cycles != float(args.actor_lr_num_cycles):
                raise ValueError(
                    f"resume actor_lr_num_cycles={saved_actor_cycles} does "
                    "not match requested "
                    f"actor_lr_num_cycles={args.actor_lr_num_cycles}"
                )
        saved_conditioned_actor = bool(
            resume_state.get("args", {}).get("conditioned_actor", False)
        )
        if saved_conditioned_actor != bool(args.conditioned_actor):
            raise ValueError(
                f"resume conditioned_actor={saved_conditioned_actor} does not "
                f"match requested conditioned_actor={args.conditioned_actor}"
            )
        if saved_conditioned_actor:
            saved_condition_definition = str(
                resume_state.get("actor_condition_label_definition", "")
            )
            if saved_condition_definition != ACTOR_CONDITION_DEFINITION:
                raise ValueError(
                    "resume actor condition definition="
                    f"{saved_condition_definition!r} does not match requested "
                    f"{ACTOR_CONDITION_DEFINITION!r}; start a fresh output "
                    "directory for the human-only condition"
                )
            saved_condition_hidden_dim = int(
                resume_state.get("args", {}).get("condition_hidden_dim", 128)
            )
            if saved_condition_hidden_dim != int(args.condition_hidden_dim):
                raise ValueError(
                    "resume condition_hidden_dim="
                    f"{saved_condition_hidden_dim} does not match requested "
                    f"condition_hidden_dim={args.condition_hidden_dim}"
                )
            saved_condition_dropout = float(
                resume_state.get("args", {}).get("condition_dropout", 0.0)
            )
            if saved_condition_dropout != float(args.condition_dropout):
                raise ValueError(
                    f"resume condition_dropout={saved_condition_dropout} does "
                    "not match requested "
                    f"condition_dropout={args.condition_dropout}"
                )
        pretrained_dp_checkpoint = str(
            resume_state["pretrained_dp_checkpoint"]
        )
    elif args.initialization in (
        "source_idql_frozen",
        "source_chunk_idql_joint",
    ):
        source_checkpoint = (
            args.source_idql_checkpoint
            if args.initialization == "source_idql_frozen"
            else args.source_chunk_idql_checkpoint
        )
        source_for_warm_start = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if args.initialization == "source_idql_frozen":
            validate_source(source_for_warm_start, args)
        else:
            validate_chunk_source(source_for_warm_start, args)
        pretrained_dp_checkpoint = str(
            source_for_warm_start["pretrained_dp_checkpoint"]
        )
        if not Path(pretrained_dp_checkpoint).is_file():
            raise FileNotFoundError(
                "pretrained DP checkpoint referenced by source checkpoint "
                f"does not exist: {pretrained_dp_checkpoint}"
            )
    else:
        pretrained_dp_checkpoint = str(args.checkpoint)

    actor_policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=pretrained_dp_checkpoint, device=device, verbose=False
    )
    dp_action_stats = dp_checkpoint.get("action_normalization_stats")
    reference_state = resume_state or source_for_warm_start
    reference_action_stats = (
        reference_state.get("action_normalization_stats")
        if reference_state is not None
        else dp_action_stats
    )
    if reference_action_stats is None or dp_action_stats is None:
        raise ValueError(
            "initial and pretrained DP checkpoints must both contain "
            "action normalization statistics"
        )
    if not action_normalization_stats_match(
        reference_action_stats,
        dp_action_stats,
    ):
        raise RuntimeError(
            "initial action normalization does not match the pretrained DP"
        )
    actor_algo = actor_policy.policy
    if trains_joint_actor(args):
        if resume_state is None:
            if args.initialization == "source_chunk_idql_joint":
                actor_algo.deserialize(
                    source_for_warm_start["actor_model"],
                    load_optimizers=False,
                )
                if actor_algo.ema is not None:
                    actor_algo.ema.optimization_step = int(
                        source_for_warm_start.get(
                            "actor_ema_optimization_step",
                            0,
                        )
                    )
            else:
                initialized_from_ema = initialize_actor_from_deployed_ema(
                    actor_algo
                )
                if actor_algo.ema is not None and not initialized_from_ema:
                    raise RuntimeError(
                        "failed to initialize actor from deployed DP EMA"
                    )
                if not actor_matches_deployed_ema(actor_algo):
                    raise RuntimeError(
                        "trainable actor does not exactly match the pretrained "
                        "deployed DP EMA"
                    )
        configure_conditioned_actor(actor_algo, args)
        actor_audit = None
    elif args.initialization == "source_idql_frozen":
        actor_state = (
            resume_state["actor_model"]
            if resume_state is not None
            else source_for_warm_start["actor_model"]
        )
        actor_algo.deserialize(actor_state, load_optimizers=False)
        configure_conditioned_actor(actor_algo, args)
        actor_audit = freeze_actor(actor_algo)
    else:
        if resume_state is not None:
            actor_algo.deserialize(
                resume_state["actor_model"],
                load_optimizers=False,
            )
        configure_conditioned_actor(actor_algo, args)
        actor_audit = freeze_actor(actor_algo)
    args.checkpoint = Path(pretrained_dp_checkpoint)

    actor_horizon = int(actor_algo.algo_config.horizon.action_horizon)
    if int(args.chunk_horizon) != actor_horizon:
        raise ValueError(
            f"chunk_horizon={args.chunk_horizon} must equal actor "
            f"action_horizon={actor_horizon}"
        )
    args.action_dim = int(actor_algo.ac_dim)
    args.observation_horizon = int(
        actor_algo.algo_config.horizon.observation_horizon
    )
    args.actor_prediction_horizon = int(
        actor_algo.algo_config.horizon.prediction_horizon
    )
    args.actor_action_horizon = actor_horizon
    if (
        args.initialization == "source_chunk_idql_joint"
        and int(source_for_warm_start.get("action_dim", -1))
        != int(args.action_dim)
    ):
        raise ValueError(
            f"source action_dim={source_for_warm_start.get('action_dim')} "
            f"does not match actor action_dim={args.action_dim}"
        )

    sequence_length = (
        int(args.actor_prediction_horizon)
        if trains_joint_actor(args)
        else int(args.chunk_horizon)
    )
    condition_audit = (
        audit_actor_conditions(
            args.dataset,
            reward_mode=str(args.reward_mode),
        )
        if args.conditioned_actor
        else None
    )
    dataset, loader, loader_generator, _ = build_single_loader(
        args,
        actor_policy,
        dp_checkpoint,
        sequence_length=sequence_length,
    )
    if args.steps_per_epoch is None:
        args.steps_per_epoch = int(len(loader))
        args.steps_per_epoch_source = "auto_DataLoader_length"
    else:
        args.steps_per_epoch = int(args.steps_per_epoch)
        args.steps_per_epoch_source = "explicit_command_line"
    args.actor_lr_total_steps = int(args.epochs) * int(args.steps_per_epoch)
    args.critic_vf_lr_total_steps = (
        int(args.epochs) * int(args.steps_per_epoch)
    )
    if (
        args.critic_vf_lr_scheduler == "cosine"
        and int(args.critic_vf_lr_warmup_steps)
        >= int(args.critic_vf_lr_total_steps)
    ):
        raise ValueError(
            "critic_vf_lr_warmup_steps="
            f"{args.critic_vf_lr_warmup_steps} must be smaller than the "
            f"{args.critic_vf_lr_total_steps} critic/VF training steps"
        )
    if resume_state is not None:
        saved_args = resume_state.get("args", {})
        schedule_fields = (
            ("critic_vf_lr_scheduler", str),
            ("critic_vf_lr_warmup_steps", int),
            ("critic_vf_lr_total_steps", int),
            ("critic_vf_lr_num_cycles", float),
        )
        for field, cast in schedule_fields:
            saved_value = cast(saved_args[field])
            requested_value = cast(getattr(args, field))
            if saved_value != requested_value:
                raise ValueError(
                    f"resume {field}={saved_value} does not match requested "
                    f"{field}={requested_value}"
                )
    if trains_joint_actor(args):
        if (
            args.actor_lr_scheduler == "cosine"
            and int(args.actor_lr_warmup_steps)
            >= int(args.actor_lr_total_steps)
        ):
            raise ValueError(
                f"actor_lr_warmup_steps={args.actor_lr_warmup_steps} must be "
                "smaller than the "
                f"{args.actor_lr_total_steps} actor training steps"
            )
        configure_actor_optimizer(
            actor_algo,
            args.actor_lr,
            scheduler_type=args.actor_lr_scheduler,
            warmup_steps=args.actor_lr_warmup_steps,
            total_steps=args.actor_lr_total_steps,
            num_cycles=args.actor_lr_num_cycles,
        )
        if resume_state is not None:
            actor_algo.deserialize(
                resume_state["actor_model"],
                load_optimizers=True,
            )
            if actor_algo.ema is not None:
                actor_algo.ema.optimization_step = int(
                    resume_state.get("actor_ema_optimization_step", 0)
                )
            configure_conditioned_actor(actor_algo, args)
        actor_algo.set_train()
        actor_audit = actor_trainability(actor_algo)
    elif actor_audit is None:
        raise RuntimeError("frozen actor audit was not initialized")

    audit = dataset_audit(
        args.dataset,
        len(dataset),
        expected_task=args.task,
        expected_reward_mode=args.reward_mode,
    )
    action_stats = copy.deepcopy(dp_checkpoint["action_normalization_stats"])
    obs_stats = copy.deepcopy(actor_policy.obs_normalization_stats)
    del dp_checkpoint

    critics, targets, vf = make_rise_chunk_value_networks(
        actor_algo,
        chunk_horizon=args.chunk_horizon,
        hidden_dims=tuple(int(x) for x in args.critic_hidden_dims),
        latent_dim=args.latent_dim,
        action_hidden_dim=args.action_hidden_dim,
        num_attention_heads=args.num_attention_heads,
        num_action_conv_layers=args.num_action_conv_layers,
        dropout=args.dropout,
        num_critics=args.num_critics,
        critic_group_norm=args.critic_group_norm,
        late_fusion_key=args.critic_late_fusion_key,
    )
    warm_start_audit: dict[str, Any] = {"mode": "resume_checkpoint"}
    if resume_state is not None:
        warm_start_audit = {
            "mode": "resume_checkpoint",
            "critics": [
                match_encoder_normalization_to_checkpoint(critic, state)
                for critic, state in zip(
                    critics,
                    resume_state["critics"],
                )
            ],
            "vf": match_encoder_normalization_to_checkpoint(
                vf,
                resume_state["vf"],
            ),
        }
        targets = copy.deepcopy(critics)
    elif (
        source_for_warm_start is not None
        and args.initialization == "source_chunk_idql_joint"
    ):
        warm_start_audit = {
            "mode": "source_chunk_idql_complete_model",
            "checkpoint": str(args.source_chunk_idql_checkpoint),
            "fresh_optimizers_and_schedulers": True,
            "fresh_epoch_and_global_step": True,
            "critics": [
                match_encoder_normalization_to_checkpoint(critic, state)
                for critic, state in zip(
                    critics,
                    source_for_warm_start["critics"],
                )
            ],
            "vf": match_encoder_normalization_to_checkpoint(
                vf,
                source_for_warm_start["vf"],
            ),
        }
        targets = copy.deepcopy(critics)
    elif source_for_warm_start is not None:
        warm_start_audit = {
            "mode": "source_one_step_idql_representations",
            "critics": [
                copy_matching_encoder_state(critic, state)
                for critic, state in zip(
                    critics,
                    source_for_warm_start["critics"],
                )
            ],
        }
        vf.load_state_dict(source_for_warm_start["vf"])
        targets = copy.deepcopy(critics)
    elif resume_state is None:
        warm_start_audit = {
            "mode": "deployed_pretrained_dp_raw_obs_encoder_copy",
            "critics": [
                copy_deployed_dp_encoder_state(critic, actor_algo)
                for critic in critics
            ],
            "vf": copy_deployed_dp_encoder_state(vf, actor_algo),
        }
        targets = copy.deepcopy(critics)

    dynamics_target_encoder = copy.deepcopy(
        deployed_actor_obs_encoder(actor_algo)
    )
    critics = critics.float().to(device)
    targets = targets.float().to(device)
    dynamics_target_encoder = dynamics_target_encoder.float().to(device)
    vf = vf.float().to(device)
    target_encoder_output_dim = int(
        dynamics_target_encoder.output_shape()[0]
    )
    critic_encoder_output_dims = {
        int(critic.encoder_output_dim) for critic in critics
    }
    if critic_encoder_output_dims != {target_encoder_output_dim}:
        raise RuntimeError(
            "actor dynamics target and critic raw encoder output dimensions "
            f"differ: target={target_encoder_output_dim}, "
            f"critics={sorted(critic_encoder_output_dims)}"
        )
    critic_optimizers = [
        make_critic_optimizer(critic, args.critic_lr, args.encoder_lr)
        for critic in critics
    ]
    vf_optimizer = make_critic_optimizer(vf, args.vf_lr, args.encoder_lr)
    critic_lr_schedulers = [
        make_step_lr_scheduler(
            optimizer,
            scheduler_type=args.critic_vf_lr_scheduler,
            warmup_steps=args.critic_vf_lr_warmup_steps,
            total_steps=args.critic_vf_lr_total_steps,
            num_cycles=args.critic_vf_lr_num_cycles,
        )
        for optimizer in critic_optimizers
    ]
    vf_lr_scheduler = make_step_lr_scheduler(
        vf_optimizer,
        scheduler_type=args.critic_vf_lr_scheduler,
        warmup_steps=args.critic_vf_lr_warmup_steps,
        total_steps=args.critic_vf_lr_total_steps,
        num_cycles=args.critic_vf_lr_num_cycles,
    )

    start_epoch = 0
    global_step = 0
    dynamics_target_last_sync_step = 0
    history: list[dict] = []
    if resume_state is not None:
        for critic, state in zip(critics, resume_state["critics"]):
            critic.load_state_dict(state)
        for target, state in zip(targets, resume_state["critic_targets"]):
            target.load_state_dict(state)
        dynamics_target_encoder.load_state_dict(
            resume_state["dynamics_target_encoder"],
            strict=True,
        )
        vf.load_state_dict(resume_state["vf"])
        for optimizer, state in zip(
            critic_optimizers, resume_state["critic_optimizers"]
        ):
            optimizer.load_state_dict(state)
        vf_optimizer.load_state_dict(resume_state["vf_optimizer"])
        scheduler_enabled = args.critic_vf_lr_scheduler != "constant"
        critic_scheduler_states = resume_state["critic_lr_schedulers"]
        if len(critic_scheduler_states) != len(critic_lr_schedulers):
            raise ValueError(
                "resume checkpoint critic LR scheduler count does not match "
                f"num_critics={len(critic_lr_schedulers)}"
            )
        for scheduler, state in zip(
            critic_lr_schedulers,
            critic_scheduler_states,
        ):
            if (state is not None) != scheduler_enabled:
                raise ValueError(
                    "resume critic LR scheduler state does not match "
                    f"critic_vf_lr_scheduler={args.critic_vf_lr_scheduler}"
                )
            if scheduler is not None:
                scheduler.load_state_dict(state)
        vf_scheduler_state = resume_state["vf_lr_scheduler"]
        if (vf_scheduler_state is not None) != scheduler_enabled:
            raise ValueError(
                "resume VF LR scheduler state does not match "
                f"critic_vf_lr_scheduler={args.critic_vf_lr_scheduler}"
            )
        if vf_lr_scheduler is not None:
            vf_lr_scheduler.load_state_dict(vf_scheduler_state)
        start_epoch = int(resume_state["epoch"])
        global_step = int(resume_state["step"])
        dynamics_target_last_sync_step = int(
            resume_state["dynamics_target_last_sync_step"]
        )
        if scheduler_enabled:
            scheduler_steps = [
                *[
                    int(scheduler.last_epoch)
                    for scheduler in critic_lr_schedulers
                    if scheduler is not None
                ],
                int(vf_lr_scheduler.last_epoch),
            ]
            if any(step != global_step for step in scheduler_steps):
                raise ValueError(
                    f"critic/VF LR scheduler steps {scheduler_steps} do not "
                    f"match checkpoint global_step={global_step}"
                )
        history = list(resume_state.get("history", []))
        loader_generator.set_state(
            resume_state["loader_generator_state"].cpu()
        )
        restore_rng_state(resume_state.get("rng_state"))
        print(
            f"Resumed {args.resume_checkpoint} at epoch={start_epoch} "
            f"step={global_step}",
            flush=True,
        )
    elif (
        source_for_warm_start is not None
        and args.initialization == "source_chunk_idql_joint"
    ):
        for critic, state in zip(
            critics,
            source_for_warm_start["critics"],
        ):
            critic.load_state_dict(state, strict=True)
        for target, state in zip(
            targets,
            source_for_warm_start["critic_targets"],
        ):
            target.load_state_dict(state, strict=True)
        dynamics_target_encoder.load_state_dict(
            source_for_warm_start["dynamics_target_encoder"],
            strict=True,
        )
        vf.load_state_dict(source_for_warm_start["vf"], strict=True)
        print(
            "Warm-started actor, actor EMA, twin critics, target critics, VF, "
            "and dynamics target from "
            f"{args.source_chunk_idql_checkpoint}; starting fresh optimizers, "
            "LR schedules, epoch=0, and global_step=0",
            flush=True,
        )
    configure_target_random_crops(targets)
    configure_encoder_target_random_crops(dynamics_target_encoder)
    del resume_state, source_for_warm_start

    if not trains_joint_actor(args):
        actor_algo.nets.cpu()
        if actor_algo.ema is not None:
            actor_algo.ema.averaged_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    architecture = {
        "actor": actor_audit,
        "conditional_diffusion_actor": bool(args.conditioned_actor),
        "actor_condition_adapter": (
            "SuccessConditionResidual"
            if args.conditioned_actor
            else None
        ),
        "critic_parameter_counts": [parameter_count(x) for x in critics],
        "target_critic_parameter_counts": [parameter_count(x) for x in targets],
        "dynamics_target_encoder_parameter_count": parameter_count(
            dynamics_target_encoder
        ),
        "dynamics_target_encoder_output_dim": target_encoder_output_dim,
        "vf_parameter_count": parameter_count(vf),
        "independent_raw_obs_encoders": True,
        "critic_chunk_horizon": int(args.chunk_horizon),
        "critic_q_head_inputs": ["context", "action_repr"],
        "critic_representation_modules": [
            "encoder",
            "context",
            "context_norm",
        ],
        "latent_dynamics": True,
        "actor_encoder_feature_dynamics": True,
        "dynamics_prediction_mode": DYNAMICS_PREDICTION_MODE,
        "dynamics_prediction_output": "raw_actor_encoder_features",
        "dynamics_prediction_output_dim": target_encoder_output_dim,
        "dynamics_prediction_residual": False,
        "dynamics_target_encoder": "shared_deployed_actor_ema_obs_encoder",
        "dynamics_target_update": (
            "periodic_hard_sync"
            if trains_joint_actor(args)
            else "fixed_after_initialization"
        ),
        "dynamics_target_context_mlp": False,
        "training_augmentation": (
            "paired_random_crop_coordinates_for_online_and_target_encoders"
        ),
        "target_encoder_mode": (
            "eval_except_crop_randomizers_in_training_mode"
        ),
        "vf_training": (
            "head_from_step_zero_raw_observation_encoder_delayed"
        ),
        "warm_start": warm_start_audit,
    }
    startup = {
        "task": str(args.task),
        "chunk_initialization": str(args.initialization),
        "source_idql_checkpoint": (
            str(args.source_idql_checkpoint)
            if args.source_idql_checkpoint is not None
            else None
        ),
        "source_chunk_idql_checkpoint": (
            str(args.source_chunk_idql_checkpoint)
            if args.source_chunk_idql_checkpoint is not None
            else None
        ),
        "pretrained_dp_checkpoint": pretrained_dp_checkpoint,
        "actor_initialization_audit": {
            "loaded_with_policy_from_checkpoint": True,
            "trainable_actor_initialized_from_deployed_ema": bool(
                args.initialization == "pretrained_dp_joint"
            ),
            "trainable_actor_initialized_from_source_chunk": bool(
                args.initialization == "source_chunk_idql_joint"
            ),
            "source_actor_ema_optimization_step_preserved": bool(
                args.initialization == "source_chunk_idql_joint"
            ),
        },
        "dataset": {
            **audit,
            "actor_conditioning": condition_audit,
        },
        "loader": {
            "class": dataset.__class__.__name__,
            "num_loaders": 1,
            "sampler": "RandomSampler_without_replacement",
            "balanced_sampling": False,
            "batch_size": int(args.batch_size),
            "num_batches": int(len(loader)),
            "steps_per_epoch": int(args.steps_per_epoch),
            "steps_per_epoch_source": args.steps_per_epoch_source,
            "sequence_length": int(sequence_length),
        },
        "data_routing": {
            "shared_loader": True,
            "critic_rows": "all_human_success_failure",
            "critic_reward_source": (
                "rewards=source_environment_task_reward"
                if args.reward_mode == "task"
                else "rewards=expert_1_non_expert_0"
            ),
            "actor_rows": (
                "all_human_success_failure"
                if trains_joint_actor(args)
                else "none_actor_frozen"
            ),
            "actor_condition_labels": (
                {
                    "human_demo": 1.0,
                    "success_rollout": 0.0,
                    "failure_rollout": 0.0,
                }
                if args.conditioned_actor
                else None
            ),
            "actor_condition_masks": (
                {
                    "human_demo": 1.0,
                    "success_rollout": 1.0,
                    "failure_rollout": 1.0,
                }
                if args.conditioned_actor
                else None
            ),
        },
        "normalization": {
            "action": "pretrained_DP_checkpoint_action_stats",
            "observation": (
                "pretrained_DP_checkpoint_obs_stats"
                if obs_stats is not None
                else "none_as_in_pretrained_DP"
            ),
            "mixed_dataset_statistics_used": False,
        },
        "architecture": architecture,
        "hyperparameters": {
            "epochs": int(args.epochs),
            "discount": float(args.discount),
            "expectile": float(args.expectile),
            "target_tau": float(args.target_tau),
            "dynamics_target_sync_interval": int(
                args.dynamics_target_sync_interval
            ),
            "actor_lr": float(args.actor_lr),
            "actor_lr_scheduler": str(args.actor_lr_scheduler),
            "actor_lr_warmup_steps": int(args.actor_lr_warmup_steps),
            "actor_lr_total_steps": int(args.actor_lr_total_steps),
            "actor_lr_num_cycles": float(args.actor_lr_num_cycles),
            "conditioned_actor": bool(args.conditioned_actor),
            "condition_dropout": float(args.condition_dropout),
            "condition_hidden_dim": int(args.condition_hidden_dim),
            "critic_lr": float(args.critic_lr),
            "encoder_lr": float(args.encoder_lr),
            "vf_lr": float(args.vf_lr),
            "critic_vf_lr_scheduler": str(args.critic_vf_lr_scheduler),
            "critic_vf_lr_warmup_steps": int(
                args.critic_vf_lr_warmup_steps
            ),
            "critic_vf_lr_num_cycles": float(
                args.critic_vf_lr_num_cycles
            ),
            "critic_vf_lr_total_steps": int(
                args.critic_vf_lr_total_steps
            ),
            "critic_vf_lr_scheduler_step_unit": "optimizer_update",
            "dynamics_weight": float(args.dynamics_weight),
            "dynamics_cosine_weight": float(args.dynamics_cosine_weight),
            "dynamics_warmup_steps": int(args.dynamics_warmup_steps),
            "encoder_freeze_steps": int(args.encoder_freeze_steps),
            "vf_encoder_freeze_steps": int(
                args.vf_encoder_freeze_steps
            ),
            "vf_head_freeze_steps": 0,
            "q_loss": "huber" if args.use_huber else "mse",
            "max_gradient_norm": (
                float(args.max_gradient_norm)
                if args.max_gradient_norm is not None
                else None
            ),
        },
    }
    write_json(args.output_dir / "training_config.json", startup)
    print(json.dumps(jsonable(startup), indent=2), flush=True)
    writer = make_tensorboard_writer(args.output_dir)
    max_grad = (
        None
        if args.max_gradient_norm is None or args.max_gradient_norm <= 0.0
        else float(args.max_gradient_norm)
    )

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        iterator = iter(loader)
        records: list[dict[str, float]] = []
        for step_in_epoch in range(1, int(args.steps_per_epoch) + 1):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            raw_batch = align_shared_batch_actions(raw_batch)
            batch = process_chunk_batch(
                raw_batch,
                actor_algo,
                obs_stats,
                chunk_horizon=args.chunk_horizon,
                discount=args.discount,
                reward_mode=args.reward_mode,
            )
            encoder_trainable = global_step >= int(args.encoder_freeze_steps)
            vf_encoder_trainable = (
                global_step >= int(args.vf_encoder_freeze_steps)
            )
            critics.train()
            set_representation_trainable(critics, encoder_trainable)
            vf.train()
            set_vf_encoder_trainable(vf, vf_encoder_trainable)
            ramp = min(
                1.0,
                float(global_step + 1)
                / max(float(args.dynamics_warmup_steps), 1.0),
            )
            effective_dynamics = float(args.dynamics_weight) * ramp
            effective_dynamics_cosine = (
                float(args.dynamics_cosine_weight) * ramp
            )
            critic_losses, vf_loss, info = compute_chunk_losses(
                critics,
                targets,
                dynamics_target_encoder,
                vf,
                batch,
                discount=args.discount,
                expectile=args.expectile,
                use_huber=args.use_huber,
                dynamics_weight=effective_dynamics,
                dynamics_cosine_weight=effective_dynamics_cosine,
            )
            update_networks(
                critics,
                targets,
                vf,
                critic_optimizers,
                vf_optimizer,
                critic_losses,
                vf_loss,
                target_tau=args.target_tau,
                max_gradient_norm=max_grad,
            )

            # update condition diffusion actor
            actor_info: dict[str, float] = {}
            if trains_joint_actor(args):
                current_index = int(args.observation_horizon) - 1
                condition_labels = source_condition_labels(
                    raw_batch,
                    current_index=current_index,
                )
                actor_batch = raw_batch
                if args.conditioned_actor:
                    actor_batch = add_actor_condition(
                        actor_batch,
                        condition_labels,
                    )
                actor_row_count = int(raw_batch["actions"].shape[0])
                actor_info = {
                    "actor/data_rows": float(actor_row_count),
                    "actor/conditioned": float(args.conditioned_actor),
                    "actor/condition_mean": float(
                        condition_labels.mean().item()
                    ),
                    "actor/zero_condition_fraction": float(
                        (condition_labels < 0.5).float().mean().item()
                    ),
                }
                actor_info.update(
                    actor_train_step(
                        actor_algo,
                        actor_batch,
                        epoch,
                        obs_stats,
                    )
                )
                del actor_batch, condition_labels
            for scheduler in critic_lr_schedulers:
                if scheduler is not None:
                    scheduler.step()
            if vf_lr_scheduler is not None:
                vf_lr_scheduler.step()
            actor_info["critic/data_rows"] = float(raw_batch["actions"].shape[0])
            global_step += 1
            dynamics_target_synced = False
            if (
                trains_joint_actor(args)
                and global_step % int(args.dynamics_target_sync_interval) == 0
            ):
                sync_actor_dynamics_target_encoder(
                    dynamics_target_encoder,
                    actor_algo,
                )
                dynamics_target_last_sync_step = global_step
                dynamics_target_synced = True
            metrics = scalar_metrics(info)
            metrics.update(actor_info)
            metrics["dynamics/target_synced_after_update"] = float(
                dynamics_target_synced
            )
            metrics["dynamics/target_last_sync_step"] = float(
                dynamics_target_last_sync_step
            )
            metrics["dynamics/target_sync_age"] = float(
                global_step - dynamics_target_last_sync_step
            )
            metrics["encoder/trainable"] = float(encoder_trainable)
            metrics["representation/trainable"] = float(encoder_trainable)
            metrics["vf/trainable"] = 1.0
            metrics["vf/head_trainable"] = 1.0
            metrics["vf/encoder_trainable"] = float(
                vf_encoder_trainable
            )
            if trains_joint_actor(args):
                metrics["lr/actor"] = float(
                    actor_algo.optimizers["policy"].param_groups[0]["lr"]
                )
            metrics["lr/critic"] = float(
                critic_optimizers[0].param_groups[0]["lr"]
            )
            metrics["lr/encoder"] = float(
                critic_optimizers[0].param_groups[1]["lr"]
            )
            metrics["lr/vf"] = float(vf_optimizer.param_groups[0]["lr"])
            if len(vf_optimizer.param_groups) > 1:
                metrics["lr/vf_encoder"] = float(
                    vf_optimizer.param_groups[1]["lr"]
                )
            records.append(metrics)
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, global_step)
            if global_step % int(args.log_every) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step_in_epoch": step_in_epoch,
                            "global_step": global_step,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
            del batch, critic_losses, vf_loss

        epoch_summary = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": mean_metrics(records),
        }
        history.append(epoch_summary)
        partial = {
            **startup,
            "last_completed_epoch": int(epoch),
            "global_step": int(global_step),
            "last_epoch_metrics": epoch_summary["metrics"],
            "history": history,
            "checkpoints": {
                "latest": str(args.output_dir / "latest.pt"),
                "last": str(args.output_dir / "last.pt"),
            },
        }
        write_json(args.output_dir / "partial_summary.json", partial)

        if (
            epoch % int(args.save_every_epochs) == 0
            or epoch == int(args.epochs)
        ):
            payload = checkpoint_payload(
                args=args,
                actor_model=actor_algo.serialize(),
                actor_ema_optimization_step=int(
                    actor_algo.ema.optimization_step
                    if actor_algo.ema is not None
                    else 0
                ),
                pretrained_dp_checkpoint=pretrained_dp_checkpoint,
                critics=critics,
                targets=targets,
                dynamics_target_encoder=dynamics_target_encoder,
                dynamics_target_last_sync_step=(
                    dynamics_target_last_sync_step
                ),
                vf=vf,
                critic_optimizers=critic_optimizers,
                vf_optimizer=vf_optimizer,
                critic_lr_schedulers=critic_lr_schedulers,
                vf_lr_scheduler=vf_lr_scheduler,
                action_stats=action_stats,
                epoch=epoch,
                global_step=global_step,
                history=history,
                loader_generator=loader_generator,
            )
            latest = args.output_dir / "latest.pt"
            atomic_torch_save(payload, latest)
            if epoch == int(args.epochs):
                replace_with_hardlink(latest, args.output_dir / "last.pt")
            if (
                int(args.snapshot_every_epochs) > 0
                and epoch % int(args.snapshot_every_epochs) == 0
            ):
                replace_with_hardlink(
                    latest,
                    args.output_dir / "models" / f"model_epoch_{epoch}.pt",
                )
            print(
                f"Saved {latest} at epoch={epoch} step={global_step}",
                flush=True,
            )

    if writer is not None:
        writer.flush()
        writer.close()
    final = json.loads((args.output_dir / "partial_summary.json").read_text())
    write_json(args.output_dir / "summary.json", final)
    return final


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("square", "can", "transport", "tool_hang"),
        default="square",
    )
    parser.add_argument(
        "--initialization",
        choices=(
            "pretrained_dp_joint",
            "pretrained_dp_frozen",
            "source_idql_frozen",
            "source_chunk_idql_joint",
        ),
        default="pretrained_dp_joint",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DP)
    parser.add_argument("--source-idql-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--source-chunk-idql-checkpoint",
        type=Path,
        default=None,
        help=(
            "Complete chunk IDQL checkpoint used to warm-start a fresh "
            "joint actor-critic training round."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hdf5-cache-mode",
        choices=("all", "low_dim", "none"),
        default="low_dim",
    )
    parser.add_argument("--chunk-horizon", type=int, default=8)
    parser.add_argument(
        "--reward-mode",
        choices=tuple(REWARD_DEFINITIONS),
        default="task",
        help="Expected dataset reward mode; task is the default.",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument(
        "--dynamics-target-sync-interval",
        type=int,
        default=1000,
        help=(
            "Hard-sync the frozen dynamics target encoder from the deployed "
            "actor EMA after this many joint-training optimizer steps."
        ),
    )
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument(
        "--actor-lr-scheduler",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--actor-lr-warmup-steps", type=int, default=500)
    parser.add_argument("--actor-lr-num-cycles", type=float, default=0.5)
    parser.add_argument(
        "--conditioned-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Condition the jointly trained diffusion actor on data source: "
            "human demonstrations use 1; all deployment rollouts use 0."
        ),
    )
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--vf-lr", type=float, default=1e-4)
    parser.add_argument(
        "--critic-hidden-dims",
        type=int,
        nargs="+",
        default=(300, 400, 300),
    )
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-action-conv-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument(
        "--critic-group-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--critic-late-fusion-key",
        type=str,
        default="robot0_gripper_qpos",
    )
    parser.add_argument("--dynamics-weight", type=float, default=0.05)
    parser.add_argument("--dynamics-cosine-weight", type=float, default=0.05)
    parser.add_argument("--dynamics-warmup-steps", type=int, default=1000)
    parser.add_argument("--encoder-freeze-steps", type=int, default=1000)
    parser.add_argument(
        "--vf-encoder-freeze-steps",
        type=int,
        default=1000,
        help=(
            "Freeze only the VF raw-observation encoder for this many "
            "optimizer steps; the VF head always trains from step zero."
        ),
    )
    parser.add_argument(
        "--use-huber", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-gradient-norm", type=float, default=10.0)
    parser.add_argument(
        "--critic-vf-lr-scheduler",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument(
        "--critic-vf-lr-warmup-steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--critic-vf-lr-num-cycles",
        type=float,
        default=0.5,
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=10)
    parser.add_argument("--snapshot-every-epochs", type=int, default=10)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    for key in (
        "checkpoint",
        "source_idql_checkpoint",
        "source_chunk_idql_checkpoint",
        "dataset",
        "output_dir",
        "resume_checkpoint",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if args.resume_checkpoint is None:
        if args.initialization in ("pretrained_dp_joint", "pretrained_dp_frozen"):
            if args.checkpoint is None or not args.checkpoint.is_file():
                parser.error(
                    f"pretrained DP checkpoint does not exist: {args.checkpoint}"
                )
        elif args.initialization == "source_idql_frozen":
            if (
                args.source_idql_checkpoint is None
                or not args.source_idql_checkpoint.is_file()
            ):
                parser.error(
                    f"source IDQL checkpoint does not exist: "
                    f"{args.source_idql_checkpoint}"
                )
        elif (
            args.source_chunk_idql_checkpoint is None
            or not args.source_chunk_idql_checkpoint.is_file()
        ):
            parser.error(
                f"source chunk IDQL checkpoint does not exist: "
                f"{args.source_chunk_idql_checkpoint}"
            )
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if (
        args.resume_checkpoint is not None
        and not args.resume_checkpoint.is_file()
    ):
        parser.error(
            f"resume checkpoint does not exist: {args.resume_checkpoint}"
        )
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        parser.error("steps-per-epoch must be positive when specified")
    if args.num_critics < 2:
        parser.error("RISE clipped double Q requires at least two critics")
    if args.dynamics_target_sync_interval <= 0:
        parser.error("dynamics-target-sync-interval must be positive")
    if not 0.0 <= args.condition_dropout < 1.0:
        parser.error("condition-dropout must be in [0, 1)")
    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    if not args.critic_late_fusion_key:
        args.critic_late_fusion_key = None
    train(args)


if __name__ == "__main__":
    main()
