#!/usr/bin/env python3
"""RGB Diffusion Q-learning on the same mixed dataset as IDQL.

The actor is the full pretrained RGB Diffusion Policy. Each update combines
its native diffusion behavior-cloning loss with a differentiable Q-maximization
loss, while independent raw-observation twin critics use the conventional
clipped double-Q Bellman target from Diffusion-QL (Wang et al., ICLR 2023).

Human demonstrations, successful rollouts, and failure rollouts are read from
one uniformly shuffled dataset produced by ``build_rgb_dp_idql_dataset.py``.
This deliberately keeps DQL, IDQL, and chunked IDQL on identical offline data.
"""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from train_rgb_dp_idql import (
    REWARD_DEFINITIONS,
    TEMPORAL_CRITIC_ARCHITECTURE,
    TEMPORAL_ONE_STEP_MARKER,
    WeightedScalarMetricAccumulator,
    action_normalization_stats_match,
    actor_validation_step,
    actor_trainability,
    align_shared_batch_actions,
    audit_validation_dataset_split,
    atomic_torch_save,
    batch_scaled_step_count,
    build_single_loader,
    configure_critic_targets,
    configure_actor_optimizer,
    copy_deployed_encoder_state,
    dataset_audit,
    fork_rng_with_seed,
    initialize_actor_from_deployed_ema,
    jsonable,
    make_temporal_one_step_value_networks,
    make_tensorboard_writer,
    make_step_lr_scheduler,
    parameter_count,
    replace_with_hardlink,
    RiseLateFusionMLP,
    restore_rng_state,
    rng_state,
    scalar_metrics,
    set_actor_obs_encoder_trainable,
    set_critic_encoders_trainable,
    write_json,
)
from rgb_dp_distributed import (
    DistributedContext,
    all_reduce_gradients,
    broadcast_module_buffers,
    broadcast_module_state,
    gather_rank_runtime_states,
    initialize_distributed,
    modules_have_mutable_batch_norm,
    restore_process_rng_state,
    seed_process,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "datasets/square/idql/square_rgb_dp_idql_200demo_406success_94failure_terminal_success.hdf5"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth"
)
DEFAULT_OUTPUT = (
    ROOT
    / "trained_models/square_rgb_dp/dql/200demo_406success_94failure_terminal_success_reward"
)

SAMPLED_VALIDATION_SATURATION_THRESHOLD = 0.98
SAMPLED_VALIDATION_RNG_OFFSET = 1


def configure_dql_batch_semantics(
    args: argparse.Namespace,
    world_size: int,
) -> None:
    """Resolve sample-timed schedules while preserving update-timed targets."""
    reference_batch_size = int(args.schedule_reference_batch_size)
    effective_batch_size = int(args.batch_size) * int(world_size)
    if reference_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("reference and effective batch sizes must be positive")
    args.effective_global_batch_size = effective_batch_size
    args.schedule_batch_ratio = (
        float(effective_batch_size) / float(reference_batch_size)
    )
    args.resolved_lr_warmup_steps = batch_scaled_step_count(
        args.lr_warmup_steps,
        reference_batch_size,
        effective_batch_size,
    )
    args.resolved_dql_critic_warmup_steps = batch_scaled_step_count(
        args.dql_critic_warmup_steps,
        reference_batch_size,
        effective_batch_size,
    )
    args.resolved_actor_obs_encoder_freeze_steps = batch_scaled_step_count(
        args.actor_obs_encoder_freeze_steps,
        reference_batch_size,
        effective_batch_size,
    )
    args.resolved_encoder_freeze_steps = batch_scaled_step_count(
        args.encoder_freeze_steps,
        reference_batch_size,
        effective_batch_size,
    )


class DQLStackedActionValueNetwork(nn.Module):
    """One-step Q network over the same observation stack as the DP actor.

    The visual encoder is copied from the deployed pretrained actor, which is
    already GroupNorm based. This avoids randomly initialized BatchNorm target
    encoders and preserves the actor's observation-horizon state definition.
    """

    def __init__(
        self,
        actor_algo,
        hidden_dims: tuple[int, ...],
        observation_horizon: int,
        late_fusion_key: str | None,
    ):
        super().__init__()
        if int(observation_horizon) <= 0:
            raise ValueError("observation_horizon must be positive")
        self.obs_shapes = copy.deepcopy(actor_algo.obs_shapes)
        self.action_dim = int(actor_algo.ac_dim)
        self.observation_horizon = int(observation_horizon)
        self.late_fusion_keys = tuple(
            key.strip()
            for key in str(late_fusion_key or "").split(",")
            if key.strip()
        )
        for key in self.late_fusion_keys:
            if key not in self.obs_shapes:
                raise KeyError(f"late_fusion_key={key} is absent from obs_shapes")

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = copy.deepcopy(
            actor_algo.nets["policy"]["obs_encoder"]
        )
        encoded_dim = int(self.nets["encoder"].output_shape()[0])
        late_fusion_dim = self.observation_horizon * sum(
            int(np.prod(self.obs_shapes[key]))
            for key in self.late_fusion_keys
        )
        self.nets["mlp"] = RiseLateFusionMLP(
            input_dim=self.observation_horizon * encoded_dim + self.action_dim,
            hidden_dims=hidden_dims,
            late_fusion_dim=late_fusion_dim,
        )
        self.nets["decoder"] = nn.Linear(int(hidden_dims[-1]), 1)

    def _stacked_observations(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        stacked = {}
        for key, shape in self.obs_shapes.items():
            value = obs_dict[key]
            no_time_ndim = len(shape) + 1
            if value.ndim == no_time_ndim:
                value = value.unsqueeze(1).expand(
                    -1,
                    self.observation_horizon,
                    *([-1] * len(shape)),
                )
            elif value.ndim == no_time_ndim + 1:
                if value.shape[1] < self.observation_horizon:
                    padding = value[:, :1].expand(
                        -1,
                        self.observation_horizon - value.shape[1],
                        *([-1] * len(shape)),
                    )
                    value = torch.cat((padding, value), dim=1)
                else:
                    value = value[:, -self.observation_horizon :]
            else:
                raise ValueError(
                    f"unexpected DQL observation shape for {key}: "
                    f"{tuple(value.shape)}"
                )
            stacked[key] = value
        return stacked

    def forward(self, obs_dict, acts, goal_dict=None):
        del goal_dict
        if acts.ndim != 2 or acts.shape[-1] != self.action_dim:
            raise ValueError(
                "DQL critic requires single-step actions [B, action_dim], got "
                f"{tuple(acts.shape)}"
            )
        stacked = self._stacked_observations(obs_dict)
        encoded = TensorUtils.time_distributed(
            {"obs": stacked, "goal": None},
            self.nets["encoder"],
            inputs_as_kwargs=True,
        )
        if encoded.ndim != 3:
            raise ValueError(
                f"expected stacked critic features [B,T,D], got {tuple(encoded.shape)}"
            )
        features = torch.cat((encoded.flatten(start_dim=1), acts), dim=-1)
        late_fusion = None
        if self.late_fusion_keys:
            late_fusion = torch.cat(
                [stacked[key].flatten(start_dim=1) for key in self.late_fusion_keys],
                dim=-1,
            )
        return self.nets["decoder"](
            self.nets["mlp"](features, late_fusion)
        )


def make_dql_value_networks(
    actor_algo,
    hidden_dims: tuple[int, ...],
    observation_horizon: int,
    num_critics: int = 2,
    late_fusion_key: str | None = "robot0_gripper_qpos",
) -> tuple[nn.ModuleList, nn.ModuleList, None]:
    critics = nn.ModuleList(
        [
            DQLStackedActionValueNetwork(
                actor_algo=actor_algo,
                hidden_dims=hidden_dims,
                observation_horizon=observation_horizon,
                late_fusion_key=late_fusion_key,
            )
            for _ in range(int(num_critics))
        ]
    )
    targets = copy.deepcopy(critics)
    targets.eval().requires_grad_(False)
    return critics, targets, None


def make_dql_rise_v2_value_networks(
    actor_algo,
    *,
    hidden_dims: tuple[int, ...],
    latent_dim: int,
    action_hidden_dim: int,
    num_attention_heads: int,
    num_action_conv_layers: int,
    dropout: float,
    num_critics: int,
    critic_group_norm: bool,
    late_fusion_key: str | None,
    observation_horizon: int,
    temporal_num_layers: int,
    temporal_num_heads: int,
    temporal_feedforward_dim: int,
    temporal_dropout: float,
    fusion_mode: str,
) -> tuple[nn.ModuleList, nn.ModuleList, list[dict[str, int]]]:
    """Build the same two-frame RISE-v2 Q architecture used by IDQL."""
    critics, unused_targets, unused_vf = make_temporal_one_step_value_networks(
        actor_algo,
        hidden_dims=hidden_dims,
        latent_dim=int(latent_dim),
        action_hidden_dim=int(action_hidden_dim),
        num_attention_heads=int(num_attention_heads),
        num_action_conv_layers=int(num_action_conv_layers),
        dropout=float(dropout),
        num_critics=int(num_critics),
        critic_group_norm=bool(critic_group_norm),
        late_fusion_key=late_fusion_key,
        observation_horizon=int(observation_horizon),
        temporal_num_layers=int(temporal_num_layers),
        temporal_num_heads=int(temporal_num_heads),
        temporal_feedforward_dim=int(temporal_feedforward_dim),
        temporal_dropout=float(temporal_dropout),
        fusion_mode=str(fusion_mode),
        dynamics_enabled=False,
    )
    del unused_targets, unused_vf
    encoder_initialization = [
        copy_deployed_encoder_state(critic, actor_algo) for critic in critics
    ]
    targets = copy.deepcopy(critics)
    configure_critic_targets(targets)
    return critics, targets, encoder_initialization


def make_dql_critic_optimizer(
    critics: nn.ModuleList,
    *,
    learning_rate: float,
    encoder_learning_rate: float,
) -> torch.optim.Optimizer:
    """Use the IDQL learning-rate split for every independent RISE-v2 Q."""
    encoder_parameters = [
        parameter
        for critic in critics
        for parameter in critic.nets["encoder"].parameters()
    ]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter
        for parameter in critics.parameters()
        if id(parameter) not in encoder_ids
    ]
    return torch.optim.Adam(
        [
            {
                "params": other_parameters,
                "lr": float(learning_rate),
                "group_name": "critic",
            },
            {
                "params": encoder_parameters,
                "lr": float(encoder_learning_rate),
                "group_name": "encoder",
            },
        ]
    )


@contextmanager
def evaluating(module: nn.Module):
    """Temporarily use inference-mode normalization without disabling autograd."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def set_requires_grad(module: nn.Module, enabled: bool) -> list[bool]:
    previous = [parameter.requires_grad for parameter in module.parameters()]
    module.requires_grad_(enabled)
    return previous


def restore_requires_grad(module: nn.Module, previous: list[bool]) -> None:
    for parameter, enabled in zip(module.parameters(), previous):
        parameter.requires_grad_(enabled)


def subset_observations(
    observations: dict[str, torch.Tensor],
    count: int,
) -> dict[str, torch.Tensor]:
    return {key: value[:count] for key, value in observations.items()}


def repeat_observations(
    observations: dict[str, torch.Tensor],
    repeats: int,
) -> dict[str, torch.Tensor]:
    return {
        key: value.repeat_interleave(int(repeats), dim=0)
        for key, value in observations.items()
    }


def prepare_actor_batch(actor_algo, raw_batch: dict, obs_normalization_stats) -> dict:
    batch = actor_algo.process_batch_for_training(raw_batch)
    return actor_algo.postprocess_batch_for_training(
        batch,
        obs_normalization_stats=obs_normalization_stats,
    )


def process_dql_critic_batch(
    raw_batch: dict,
    actor_algo,
    obs_normalization_stats,
    *,
    critic_observation_horizon: int = 2,
    validate_actions: bool = True,
) -> dict:
    """Extract an aligned two-frame transition for the RISE-v2 DQL critic."""
    actor_observation_horizon = int(
        actor_algo.algo_config.horizon.observation_horizon
    )
    current_index = actor_observation_horizon - 1
    critic_observation_horizon = int(critic_observation_horizon)
    history_start = current_index - critic_observation_horizon + 1
    if history_start < 0:
        raise ValueError(
            "critic observation horizon exceeds the actor observation history"
        )
    critic_batch = {
        "obs": {
            key: raw_batch["obs"][key][
                :, history_start : current_index + 1
            ]
            for key in actor_algo.obs_shapes
        },
        "next_obs": {
            key: raw_batch["next_obs"][key][
                :, :critic_observation_horizon
            ]
            for key in actor_algo.obs_shapes
        },
        "actions": raw_batch["actions"][:, current_index],
        "rewards": raw_batch["rewards"][:, current_index].reshape(-1, 1),
        "dones": raw_batch["dones"][:, current_index].reshape(-1, 1),
        "goal_obs": raw_batch.get("goal_obs"),
    }
    critic_batch = TensorUtils.to_device(
        TensorUtils.to_float(critic_batch),
        actor_algo.device,
    )
    critic_batch = actor_algo.postprocess_batch_for_training(
        critic_batch,
        obs_normalization_stats=obs_normalization_stats,
    )
    actions = critic_batch["actions"]
    expected_shape = (int(actions.shape[0]), int(actor_algo.ac_dim))
    if actions.ndim != 2 or tuple(actions.shape) != expected_shape:
        raise ValueError(
            "DQL critic requires single-step actions [B, action_dim], got "
            f"{tuple(actions.shape)}"
        )
    if validate_actions:
        if not torch.isfinite(actions).all():
            raise ValueError("DQL critic actions contain non-finite values")
        tolerance = 1e-3
        action_min = float(actions.min())
        action_max = float(actions.max())
        if action_min < -1.0 - tolerance or action_max > 1.0 + tolerance:
            raise ValueError(
                "DQL critic actions are outside the pretrained DP normalized space: "
                f"min={action_min:.6f}, max={action_max:.6f}"
            )
    critic_batch["actions"] = actions.clamp(-1.0, 1.0)
    invalid_history = {
        f"{group}/{key}": tuple(value.shape)
        for group in ("obs", "next_obs")
        for key, value in critic_batch[group].items()
        if int(value.shape[1]) != critic_observation_horizon
    }
    if invalid_history:
        raise ValueError(
            "DQL critic observations have the wrong temporal history: "
            f"{invalid_history}"
        )
    return critic_batch


def prepare_next_actor_observations(
    actor_algo,
    raw_batch: dict,
    obs_normalization_stats,
) -> dict[str, torch.Tensor]:
    next_raw_batch = dict(raw_batch)
    next_raw_batch["obs"] = raw_batch["next_obs"]
    return prepare_actor_batch(
        actor_algo,
        next_raw_batch,
        obs_normalization_stats,
    )["obs"]


def diffusion_bc_loss(
    actor_algo,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Native DDPM noise-prediction loss used by robomimic Diffusion Policy."""
    actions = batch["actions"]
    inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
    obs_condition = actor_algo._encode_obs(inputs, actor_algo.nets)
    obs_condition, condition_stats = actor_algo._apply_success_condition(
        obs_condition,
        nets=actor_algo.nets,
        batch=batch,
        validate=False,
    )

    noise = torch.randn_like(actions)
    timesteps = torch.randint(
        0,
        int(actor_algo.noise_scheduler.config.num_train_timesteps),
        (actions.shape[0],),
        device=actions.device,
    ).long()
    noisy_actions = actor_algo.noise_scheduler.add_noise(actions, noise, timesteps)
    predicted_noise = actor_algo.nets["policy"]["noise_pred_net"](
        noisy_actions,
        timesteps,
        global_cond=obs_condition,
    )
    per_sample = F.mse_loss(
        predicted_noise,
        noise,
        reduction="none",
    ).flatten(start_dim=1).mean(dim=1)
    info = {
        "actor/bc_loss": per_sample.mean().detach(),
        "actor/bc_energy_std": per_sample.std(unbiased=False).detach(),
    }
    if condition_stats is not None:
        for key, value in condition_stats.items():
            info[f"actor/success_condition_{key}"] = value.detach()
    return per_sample.mean(), info


def sample_action_chunks(
    *,
    actor_algo,
    observations: dict[str, torch.Tensor],
    nets: nn.Module,
    num_inference_steps: int,
    clip_actions: bool,
) -> torch.Tensor:
    """Differentiably sample the executable DP action chunk."""
    inputs = {"obs": observations, "goal": None}
    obs_condition = actor_algo._encode_obs(inputs, nets)
    obs_condition, _ = actor_algo._apply_success_condition(
        obs_condition,
        nets=nets,
        success_condition=torch.ones(
            obs_condition.shape[0],
            device=actor_algo.device,
        ),
        condition_mask=torch.ones(
            obs_condition.shape[0],
            device=actor_algo.device,
        ),
        validate=True,
    )

    scheduler = actor_algo.noise_scheduler
    train_steps = int(scheduler.config.num_train_timesteps)
    inference_steps = max(1, min(int(num_inference_steps), train_steps))
    scheduler.set_timesteps(inference_steps, device=actor_algo.device)

    horizon = actor_algo.algo_config.horizon
    trajectory = torch.randn(
        (
            obs_condition.shape[0],
            int(horizon.prediction_horizon),
            int(actor_algo.ac_dim),
        ),
        device=actor_algo.device,
    )
    for timestep in scheduler.timesteps:
        predicted_noise = nets["policy"]["noise_pred_net"](
            sample=trajectory,
            timestep=timestep,
            global_cond=obs_condition,
        )
        trajectory = scheduler.step(
            model_output=predicted_noise,
            timestep=timestep,
            sample=trajectory,
        ).prev_sample

    start = int(horizon.observation_horizon) - 1
    chunks = trajectory[
        :,
        start : start + int(horizon.action_horizon),
    ]
    if clip_actions:
        chunks = chunks.clamp(-1.0, 1.0)
    return chunks


@torch.no_grad()
def sample_target_actions(
    *,
    actor_algo,
    next_actor_observations: dict[str, torch.Tensor],
    num_inference_steps: int,
    num_candidates: int,
    clip_actions: bool,
) -> torch.Tensor:
    target_actor = (
        actor_algo.ema.averaged_model
        if actor_algo.ema is not None
        else actor_algo.nets
    )
    observations = next_actor_observations
    if int(num_candidates) > 1:
        observations = repeat_observations(observations, int(num_candidates))
    with evaluating(target_actor):
        chunks = sample_action_chunks(
            actor_algo=actor_algo,
            observations=observations,
            nets=target_actor,
            num_inference_steps=num_inference_steps,
            clip_actions=clip_actions,
        )
    return chunks[:, 0]


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    use_huber: bool,
) -> torch.Tensor:
    if use_huber:
        return F.smooth_l1_loss(prediction, target)
    return F.mse_loss(prediction, target)


def compute_critic_loss(
    *,
    actor_algo,
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    critic_batch: dict,
    next_actor_observations: dict[str, torch.Tensor],
    discount: float,
    use_huber: bool,
    num_inference_steps: int,
    num_target_candidates: int,
    clip_actions: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observations = critic_batch["obs"]
    next_observations = critic_batch["next_obs"]
    actions = critic_batch["actions"]
    rewards = critic_batch["rewards"]
    dones = critic_batch["dones"]
    goal_observations = critic_batch["goal_obs"]

    with torch.no_grad():
        next_actions = sample_target_actions(
            actor_algo=actor_algo,
            next_actor_observations=next_actor_observations,
            num_inference_steps=num_inference_steps,
            num_candidates=num_target_candidates,
            clip_actions=clip_actions,
        )
        if int(num_target_candidates) > 1:
            target_observations = repeat_observations(
                next_observations,
                int(num_target_candidates),
            )
        else:
            target_observations = next_observations
        target_goal_observations = goal_observations
        if goal_observations is not None and int(num_target_candidates) > 1:
            target_goal_observations = repeat_observations(
                goal_observations,
                int(num_target_candidates),
            )
        target_predictions = torch.cat(
            [
                critic_target(
                    obs_dict=target_observations,
                    acts=next_actions,
                    goal_dict=target_goal_observations,
                )
                for critic_target in critic_targets
            ],
            dim=1,
        )
        if int(num_target_candidates) > 1:
            batch_size = actions.shape[0]
            target_predictions = target_predictions.view(
                batch_size,
                int(num_target_candidates),
                len(critic_targets),
            )
            # Optional max-Q backup from the official Diffusion-QL code: maximize
            # each head over candidates, then apply clipped double Q.
            target_predictions = target_predictions.max(dim=1).values
        target_q = target_predictions.min(dim=1, keepdim=True).values
        backup = rewards + (1.0 - dones) * float(discount) * target_q

    predictions = [
        critic(
            obs_dict=observations,
            acts=actions,
            goal_dict=goal_observations,
        )
        for critic in critics
    ]
    losses = [
        regression_loss(prediction, backup, use_huber)
        for prediction in predictions
    ]
    total = torch.stack(losses).sum()
    with torch.no_grad():
        current_q = torch.cat(predictions, dim=1).min(dim=1).values
    return total, {
        "critic/loss": total.detach(),
        "critic/q_mean": current_q.mean().detach(),
        "critic/target_q_mean": target_q.mean().detach(),
        "critic/backup_mean": backup.mean().detach(),
        "critic/reward_mean": rewards.mean().detach(),
        "critic/done_fraction": dones.mean().detach(),
        **{
            f"critic/q{index + 1}_loss": loss.detach()
            for index, loss in enumerate(losses)
        },
    }


def choose_q_values(
    predictions: list[torch.Tensor],
    q_head: str,
) -> tuple[torch.Tensor, int]:
    if q_head == "min":
        return torch.cat(predictions, dim=1).min(dim=1, keepdim=True).values, -1
    if q_head == "random":
        # Rank-local head sampling is ordinary data-parallel stochasticity. A
        # broadcast here needlessly serialized every actor update.
        index = int(np.random.randint(len(predictions)))
    else:
        index = int(q_head.removeprefix("q")) - 1
        if not 0 <= index < len(predictions):
            raise ValueError(
                f"dql_q_head={q_head} requires critic {index + 1}, "
                f"but num_critics={len(predictions)}"
            )
    return predictions[index], index


def diffusion_q_loss(
    *,
    actor_algo,
    critics: nn.ModuleList,
    actor_observations: dict[str, torch.Tensor],
    critic_observations: dict[str, torch.Tensor],
    goal_observations,
    num_inference_steps: int,
    q_head: str,
    denominator_floor: float,
    clip_actions: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official Diffusion-QL cross-head normalized policy-improvement loss."""
    chunks = sample_action_chunks(
        actor_algo=actor_algo,
        observations=actor_observations,
        nets=actor_algo.nets,
        num_inference_steps=num_inference_steps,
        clip_actions=clip_actions,
    )
    policy_actions = chunks[:, 0]

    previous_grad_state = set_requires_grad(critics, False)
    try:
        policy_predictions = [
            critic(
                obs_dict=critic_observations,
                acts=policy_actions,
                goal_dict=goal_observations,
            )
            for critic in critics
        ]
        policy_q, head_index = choose_q_values(
            policy_predictions,
            q_head,
        )
        with torch.no_grad():
            if head_index < 0:
                normalization_q = policy_q
            else:
                # Match the official implementation: optimize one Q head and
                # normalize it by another head evaluated on generated actions.
                other_index = (head_index + 1) % len(policy_predictions)
                normalization_q = policy_predictions[other_index]
            denominator = normalization_q.abs().mean()
            denominator.clamp_min_(float(denominator_floor))
        q_loss = -policy_q.mean() / denominator
    finally:
        restore_requires_grad(critics, previous_grad_state)

    return q_loss, {
        "actor/q_loss": q_loss.detach(),
        "actor/policy_q_mean": policy_q.mean().detach(),
        "actor/normalization_q_abs_mean": normalization_q.abs().mean().detach(),
        "actor/q_denominator": denominator.detach(),
        "actor/q_head_index": policy_q.new_tensor(float(head_index)),
        "actor/generated_action_abs_mean": policy_actions.abs().mean().detach(),
    }


def actor_train_step(
    *,
    actor_algo,
    critics: nn.ModuleList,
    actor_batch: dict,
    critic_batch: dict,
    args: argparse.Namespace,
    global_step: int,
    gradient_sync_fn=None,
    defer_scalar_conversion: bool = False,
) -> dict[str, Any]:
    bc_loss, bc_info = diffusion_bc_loss(actor_algo, actor_batch)
    available = int(next(iter(actor_batch["obs"].values())).shape[0])
    q_batch_size = min(int(args.dql_q_batch_size), available)
    q_loss = bc_loss.new_zeros(())
    q_info: dict[str, torch.Tensor] = {"actor/q_loss": q_loss.detach()}
    q_guidance_active = (
        float(args.dql_eta) > 0.0
        and q_batch_size > 0
        and int(global_step) >= int(args.resolved_dql_critic_warmup_steps)
    )
    if q_guidance_active:
        goal_observations = critic_batch["goal_obs"]
        if goal_observations is not None:
            goal_observations = {
                key: value[:q_batch_size]
                for key, value in goal_observations.items()
            }
        q_loss, q_info = diffusion_q_loss(
            actor_algo=actor_algo,
            critics=critics,
            actor_observations=subset_observations(
                actor_batch["obs"],
                q_batch_size,
            ),
            critic_observations=subset_observations(
                critic_batch["obs"],
                q_batch_size,
            ),
            goal_observations=goal_observations,
            num_inference_steps=int(args.dql_num_inference_steps),
            q_head=str(args.dql_q_head),
            denominator_floor=float(args.dql_q_denominator_floor),
            clip_actions=bool(args.dql_clip_actions),
        )

    active_eta = float(args.dql_eta) if q_guidance_active else 0.0
    actor_loss = float(args.dql_bc_weight) * bc_loss + active_eta * q_loss
    optimizer = actor_algo.optimizers["policy"]
    optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if gradient_sync_fn is not None:
        gradient_sync_fn(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ),
        float(args.actor_max_gradient_norm),
    )
    optimizer.step()
    ema_update_due = (
        actor_algo.ema is not None
        and int(global_step) + 1
        >= int(args.resolved_dql_critic_warmup_steps)
        and (int(global_step) + 1) % int(args.dql_actor_ema_update_every) == 0
    )
    if ema_update_due:
        actor_algo._update_ema()
    actor_algo.on_gradient_step()

    metrics = {
        "actor/loss": actor_loss.detach(),
        "actor/gradient_norm": gradient_norm,
        "actor/q_batch_size": float(q_batch_size),
        "actor/dql_eta": active_eta,
        "actor/q_guidance_active": float(q_guidance_active),
        "actor/ema_updated": float(ema_update_due),
        "actor/effective_q_scale": (
            float(args.dql_eta) / q_info["actor/q_denominator"]
            if "actor/q_denominator" in q_info
            else 0.0
        ),
        **bc_info,
        **q_info,
    }
    return metrics if defer_scalar_conversion else scalar_metrics(metrics)


def soft_update_targets(
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    tau: float,
) -> None:
    with torch.no_grad():
        for critic, target in zip(critics, critic_targets):
            TorchUtils.soft_update(source=critic, target=target, tau=float(tau))
            source_buffers = dict(critic.named_buffers())
            for name, target_buffer in target.named_buffers():
                source_buffer = source_buffers[name]
                if target_buffer.is_floating_point():
                    target_buffer.lerp_(source_buffer, float(tau))
                else:
                    target_buffer.copy_(source_buffer)


def _local_scalar_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric is not scalar: {tuple(value.shape)}")
        return value.detach().reshape(()).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    scalar = torch.as_tensor(value, device=device, dtype=torch.float32)
    if scalar.numel() != 1:
        raise ValueError(f"metric is not scalar: {tuple(scalar.shape)}")
    return scalar.reshape(())


class LocalScalarMetricAccumulator:
    """Accumulate rank-zero epoch metrics without per-step host or rank sync."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    @torch.no_grad()
    def update(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            scalar = _local_scalar_tensor(value, self.device)
            if key in self.sums:
                self.sums[key].add_(scalar)
                self.counts[key] += 1
            else:
                self.sums[key] = scalar.clone()
                self.counts[key] = 1

    @torch.no_grad()
    def means(self) -> dict[str, float]:
        if not self.sums:
            return {}
        keys = sorted(self.sums)
        values = torch.stack(
            [self.sums[key] / float(self.counts[key]) for key in keys]
        ).cpu().tolist()
        return {key: float(value) for key, value in zip(keys, values)}


@torch.no_grad()
def materialize_local_scalar_metrics(
    metrics: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Transfer packed rank-local log scalars to the host only when logging."""
    if not metrics:
        return {}
    keys = sorted(metrics)
    values = torch.stack(
        [_local_scalar_tensor(metrics[key], device) for key in keys]
    ).cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def deterministic_sampled_validation_indices(
    dataset_size: int,
    row_count: int,
    seed: int,
) -> list[int]:
    """Choose a stable, non-contiguous held-out subset without global RNG use."""
    dataset_size = int(dataset_size)
    row_count = min(int(row_count), dataset_size)
    if dataset_size <= 0:
        raise ValueError("sampled validation requires a non-empty dataset")
    if row_count <= 0:
        raise ValueError("sampled validation row count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (
        torch.randperm(dataset_size, generator=generator)[:row_count]
        .sort()
        .values.tolist()
    )


def sampled_validation_raw_batch(
    dataset,
    *,
    row_count: int,
    seed: int,
) -> tuple[dict, list[int]]:
    """Materialize only actor inputs for a fixed held-out validation subset."""
    indices = deterministic_sampled_validation_indices(
        len(dataset),
        row_count,
        seed,
    )
    samples = []
    for index in indices:
        item = dataset[index]
        sample = {
            "obs": item["obs"],
            "actions": item["actions"],
        }
        goal_obs = item.get("goal_obs")
        if goal_obs is not None:
            sample["goal_obs"] = goal_obs
        samples.append(sample)
    return torch.utils.data.default_collate(samples), indices


def sampled_action_safety_metrics(
    *,
    current_actions: torch.Tensor,
    behavior_actions: torch.Tensor,
    base_actions: torch.Tensor,
    max_error_ratio: float,
    max_saturation_excess: float,
    saturation_threshold: float = SAMPLED_VALIDATION_SATURATION_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Measure sampled-policy support drift against the deployed base actor."""
    if (
        current_actions.ndim != 2
        or behavior_actions.shape != current_actions.shape
        or base_actions.shape != current_actions.shape
    ):
        raise ValueError(
            "sampled validation actions must share shape [rows, action_dim]: "
            f"current={tuple(current_actions.shape)}, "
            f"behavior={tuple(behavior_actions.shape)}, "
            f"base={tuple(base_actions.shape)}"
        )
    if current_actions.shape[0] <= 0 or current_actions.shape[1] <= 0:
        raise ValueError("sampled validation actions must be non-empty")

    current_behavior_l2 = torch.linalg.vector_norm(
        current_actions - behavior_actions,
        dim=1,
    ).mean()
    base_behavior_l2 = torch.linalg.vector_norm(
        base_actions - behavior_actions,
        dim=1,
    ).mean()
    current_base_l2 = torch.linalg.vector_norm(
        current_actions - base_actions,
        dim=1,
    ).mean()
    error_ratio = current_behavior_l2 / base_behavior_l2.clamp_min(1e-8)

    current_saturation = (
        current_actions.abs() >= float(saturation_threshold)
    ).float().mean(dim=0)
    base_saturation = (
        base_actions.abs() >= float(saturation_threshold)
    ).float().mean(dim=0)
    behavior_saturation = (
        behavior_actions.abs() >= float(saturation_threshold)
    ).float().mean(dim=0)

    # The final real-robot action dimension is the absolute gripper state and
    # legitimately spends long periods at +/-1. Gate only motion dimensions.
    motion_dimensions = max(int(current_actions.shape[1]) - 1, 0)
    if motion_dimensions:
        motion_saturation = current_saturation[:motion_dimensions]
        saturation_excess_by_axis = (
            motion_saturation - base_saturation[:motion_dimensions]
        ).clamp_min(0.0)
        max_motion_saturation = motion_saturation.max()
        max_motion_saturation_excess = saturation_excess_by_axis.max()
    else:
        max_motion_saturation = current_behavior_l2.new_zeros(())
        max_motion_saturation_excess = current_behavior_l2.new_zeros(())

    finite = torch.stack(
        [
            torch.isfinite(current_actions).all(),
            torch.isfinite(behavior_actions).all(),
            torch.isfinite(base_actions).all(),
            torch.isfinite(current_behavior_l2),
            torch.isfinite(base_behavior_l2),
            torch.isfinite(error_ratio),
        ]
    ).all()
    error_pass = finite & (error_ratio <= float(max_error_ratio))
    saturation_pass = finite & (
        max_motion_saturation_excess <= float(max_saturation_excess)
    )
    eligible = error_pass & saturation_pass

    metrics = {
        "sampled/rows": current_behavior_l2.new_tensor(
            float(current_actions.shape[0])
        ),
        "sampled/current_behavior_l2_mean": current_behavior_l2,
        "sampled/base_behavior_l2_mean": base_behavior_l2,
        "sampled/current_base_l2_mean": current_base_l2,
        "sampled/behavior_l2_ratio": error_ratio,
        "sampled/max_error_ratio": current_behavior_l2.new_tensor(
            float(max_error_ratio)
        ),
        "sampled/saturation_threshold": current_behavior_l2.new_tensor(
            float(saturation_threshold)
        ),
        "sampled/max_motion_axis_saturation_fraction": max_motion_saturation,
        "sampled/max_motion_axis_saturation_excess": (
            max_motion_saturation_excess
        ),
        "sampled/max_saturation_excess": current_behavior_l2.new_tensor(
            float(max_saturation_excess)
        ),
        "sampled/finite": finite.float(),
        "sampled/error_ratio_pass": error_pass.float(),
        "sampled/saturation_pass": saturation_pass.float(),
        "sampled/eligible": eligible.float(),
    }
    for index in range(int(current_actions.shape[1])):
        prefix = f"sampled/action_{index}"
        metrics.update(
            {
                f"{prefix}/current_mean": current_actions[:, index].mean(),
                f"{prefix}/behavior_mean": behavior_actions[:, index].mean(),
                f"{prefix}/base_mean": base_actions[:, index].mean(),
                f"{prefix}/current_saturation_fraction": current_saturation[index],
                f"{prefix}/behavior_saturation_fraction": behavior_saturation[index],
                f"{prefix}/base_saturation_fraction": base_saturation[index],
            }
        )
    return metrics


@torch.no_grad()
def build_sampled_validation_reference(
    *,
    args: argparse.Namespace,
    dataset,
    actor_algo,
    obs_normalization_stats,
) -> dict[str, Any]:
    """Cache held-out behavior and deployed-base actions before DQL updates."""
    raw_batch, indices = sampled_validation_raw_batch(
        dataset,
        row_count=int(args.dql_sampled_validation_rows),
        seed=int(args.validation_seed),
    )
    actor_batch = prepare_actor_batch(
        actor_algo,
        raw_batch,
        obs_normalization_stats,
    )
    if actor_algo.ema is None:
        raise RuntimeError("sampled validation requires a deployed EMA actor")
    ema_nets = actor_algo.ema.averaged_model
    noise_seed = int(args.validation_seed) + SAMPLED_VALIDATION_RNG_OFFSET
    with fork_rng_with_seed(noise_seed, actor_algo.device), evaluating(ema_nets):
        base_actions = sample_action_chunks(
            actor_algo=actor_algo,
            observations=actor_batch["obs"],
            nets=ema_nets,
            num_inference_steps=int(args.dql_num_inference_steps),
            clip_actions=bool(args.dql_clip_actions),
        )[:, 0]
    current_index = int(actor_algo.algo_config.horizon.observation_horizon) - 1
    behavior_actions = actor_batch["actions"][:, current_index].clamp(-1.0, 1.0)
    return {
        "raw_batch": raw_batch,
        "indices": indices,
        "noise_seed": noise_seed,
        "base_actions": base_actions.detach().cpu(),
        "behavior_actions": behavior_actions.detach().cpu(),
        "action_index": current_index,
    }


def _minimum_q_values(
    critics: nn.ModuleList,
    observations: dict[str, torch.Tensor],
    actions: torch.Tensor,
    goal_observations,
) -> torch.Tensor:
    predictions = [
        critic(
            obs_dict=observations,
            acts=actions,
            goal_dict=goal_observations,
        )
        for critic in critics
    ]
    return torch.cat(predictions, dim=1).min(dim=1).values


@torch.no_grad()
def evaluate_sampled_policy_validation(
    *,
    args: argparse.Namespace,
    reference: dict[str, Any],
    actor_algo,
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    obs_normalization_stats,
) -> dict[str, torch.Tensor]:
    """Evaluate reverse-sampled EMA actions on the fixed held-out subset."""
    actor_batch = prepare_actor_batch(
        actor_algo,
        reference["raw_batch"],
        obs_normalization_stats,
    )
    current_index = int(actor_algo.algo_config.horizon.observation_horizon) - 1
    if current_index != int(reference["action_index"]):
        raise RuntimeError("sampled validation action alignment changed")
    behavior_actions = actor_batch["actions"][:, current_index].clamp(-1.0, 1.0)
    cached_behavior = reference["behavior_actions"].to(
        device=behavior_actions.device,
        dtype=behavior_actions.dtype,
    )
    if not torch.equal(behavior_actions, cached_behavior):
        raise RuntimeError("sampled validation behavior actions changed")
    base_actions = reference["base_actions"].to(
        device=behavior_actions.device,
        dtype=behavior_actions.dtype,
    )
    if actor_algo.ema is None:
        raise RuntimeError("sampled validation requires a deployed EMA actor")
    ema_nets = actor_algo.ema.averaged_model
    with fork_rng_with_seed(
        int(reference["noise_seed"]),
        actor_algo.device,
    ), evaluating(ema_nets):
        current_actions = sample_action_chunks(
            actor_algo=actor_algo,
            observations=actor_batch["obs"],
            nets=ema_nets,
            num_inference_steps=int(args.dql_num_inference_steps),
            clip_actions=bool(args.dql_clip_actions),
        )[:, 0]

    metrics = sampled_action_safety_metrics(
        current_actions=current_actions,
        behavior_actions=behavior_actions,
        base_actions=base_actions,
        max_error_ratio=float(args.dql_sampled_validation_max_error_ratio),
        max_saturation_excess=float(
            args.dql_sampled_validation_max_saturation_excess
        ),
    )
    observations = actor_batch["obs"]
    goal_observations = actor_batch["goal_obs"]
    for label, modules in (
        ("online", critics),
        ("target", critic_targets),
    ):
        current_q = _minimum_q_values(
            modules,
            observations,
            current_actions,
            goal_observations,
        ).mean()
        behavior_q = _minimum_q_values(
            modules,
            observations,
            behavior_actions,
            goal_observations,
        ).mean()
        base_q = _minimum_q_values(
            modules,
            observations,
            base_actions,
            goal_observations,
        ).mean()
        metrics.update(
            {
                f"sampled/{label}_current_q_mean": current_q,
                f"sampled/{label}_behavior_q_mean": behavior_q,
                f"sampled/{label}_base_q_mean": base_q,
                f"sampled/{label}_current_behavior_q_gap": (
                    current_q - behavior_q
                ),
                f"sampled/{label}_current_base_q_gap": current_q - base_q,
            }
        )
    return metrics


def eligible_validation_loss(
    metrics: dict[str, Any],
    *,
    sampled_safety_required: bool,
) -> float | None:
    candidate = metrics.get("validation/actor/Loss")
    if candidate is None or not np.isfinite(float(candidate)):
        return None
    if sampled_safety_required:
        eligible = metrics.get("validation/sampled/eligible")
        if eligible is None or not np.isfinite(float(eligible)):
            return None
        if float(eligible) < 0.5:
            return None
    return float(candidate)


@torch.no_grad()
def evaluate_dql_validation_epoch(
    *,
    args: argparse.Namespace,
    epoch: int,
    loader,
    actor_algo,
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    obs_normalization_stats,
    device: torch.device,
    sampled_validation_reference: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run deterministic, full-coverage held-out DQL validation on rank zero."""
    accumulator = WeightedScalarMetricAccumulator(device)
    batch_count = 0
    window_count = 0
    actor_was_training = bool(actor_algo.nets.training)
    critic_modes = [bool(module.training) for module in critics]
    actor_algo.set_eval()
    critics.eval()
    critic_targets.eval()
    sampled_metrics: dict[str, torch.Tensor] = {}
    try:
        with fork_rng_with_seed(int(args.validation_seed), device):
            for raw_batch in loader:
                raw_batch = align_shared_batch_actions(
                    raw_batch,
                    validate=batch_count == 0,
                )
                rows = int(raw_batch["actions"].shape[0])
                metrics = actor_validation_step(
                    actor_algo,
                    raw_batch,
                    epoch,
                    obs_normalization_stats,
                )
                critic_batch = process_dql_critic_batch(
                    raw_batch,
                    actor_algo,
                    obs_normalization_stats,
                    critic_observation_horizon=int(
                        args.critic_observation_horizon
                    ),
                    validate_actions=batch_count == 0,
                )
                next_actor_observations = prepare_next_actor_observations(
                    actor_algo,
                    raw_batch,
                    obs_normalization_stats,
                )
                _, critic_info = compute_critic_loss(
                    actor_algo=actor_algo,
                    critics=critics,
                    critic_targets=critic_targets,
                    critic_batch=critic_batch,
                    next_actor_observations=next_actor_observations,
                    discount=float(args.discount),
                    use_huber=bool(args.use_huber),
                    num_inference_steps=int(args.dql_num_inference_steps),
                    num_target_candidates=int(
                        args.dql_target_num_candidates
                    ),
                    clip_actions=bool(args.dql_clip_actions),
                )
                metrics.update(critic_info)
                accumulator.update(
                    {
                        f"validation/{key}": value
                        for key, value in metrics.items()
                    },
                    rows,
                )
                batch_count += 1
                window_count += rows
        if sampled_validation_reference is not None:
            sampled_metrics = evaluate_sampled_policy_validation(
                args=args,
                reference=sampled_validation_reference,
                actor_algo=actor_algo,
                critics=critics,
                critic_targets=critic_targets,
                obs_normalization_stats=obs_normalization_stats,
            )
    finally:
        if actor_was_training:
            actor_algo.set_train()
        else:
            actor_algo.set_eval()
        for module, was_training in zip(critics, critic_modes):
            module.train(was_training)
        configure_critic_targets(critic_targets)
    if batch_count == 0 or window_count == 0:
        raise ValueError("validation loader produced no held-out DQL windows")
    result = accumulator.means()
    result.update(
        {
            f"validation/{key}": float(value.detach().cpu())
            for key, value in sampled_metrics.items()
        }
    )
    result["validation/batches"] = float(batch_count)
    result["validation/windows"] = float(window_count)
    return result


def dql_reference_alignment(args: argparse.Namespace) -> dict[str, Any]:
    reward_mode = str(getattr(args, "dataset_reward_mode", "rise"))
    return {
        "paper": "Wang et al., Diffusion Policies as an Expressive Policy Class for Offline RL, ICLR 2023",
        "matched": [
            "diffusion_behavior_cloning_plus_q_maximization_actor_loss",
            "q_gradient_backpropagated_through_full_reverse_diffusion_chain",
            "official_cross_head_generated_action_q_normalization",
            "independent_twin_q_critics_and_target_critics",
            "clipped_double_q_bellman_backup",
            "ema_diffusion_target_actor",
            "uniform_offline_transition_sampling",
        ],
        "robot_policy_adaptations": [
            "pretrained_rgb_diffusion_policy_initialization",
            "pretrained_groupnorm_critic_encoder_initialization",
            "critic_uses_two_frame_RISE_v2_observation_history",
            "actor_predicts_a_chunk_but_q_scores_first_executable_action",
            (
                "source_environment_task_reward"
                if reward_mode == "task"
                else "canonical_first_success_terminal_reward"
                if reward_mode == "terminal_success"
                else "canonical_signed_terminal_outcome_reward"
            ),
            "differentiable_q_sampling_can_use_fewer_reverse_steps_for_memory",
        ],
        "max_q_backup_candidates": int(args.dql_target_num_candidates),
    }


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    actor_algo,
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    critic_optimizer: torch.optim.Optimizer,
    critic_lr_scheduler,
    action_normalization_stats: dict,
    epoch: int,
    global_step: int,
    global_samples_seen: int,
    history: list[dict],
    loader_generator: torch.Generator,
    rank_runtime_states: list[dict[str, Any]] | None = None,
    distributed_context: DistributedContext | None = None,
) -> dict[str, Any]:
    rank_zero_runtime = (
        rank_runtime_states[0] if rank_runtime_states is not None else None
    )
    distributed_world_size = int(
        distributed_context.world_size
        if distributed_context is not None
        else 1
    )
    return {
        "stacked_pretrained_dql_critic": False,
        # Reuse the raw-RGB actor / critic evaluator shared with IDQL.
        "rise_style_rgb_idql": True,
        "rise_style_rgb_dql": True,
        TEMPORAL_ONE_STEP_MARKER: True,
        "hybrid_dp_chunk_actor_iql": True,
        "visual_critic_idql": True,
        "actor_model": actor_algo.serialize(),
        "critics": [critic.state_dict() for critic in critics],
        "critic_targets": [target.state_dict() for target in critic_targets],
        "critic_optimizer": critic_optimizer.state_dict(),
        "critic_lr_scheduler": (
            critic_lr_scheduler.state_dict()
            if critic_lr_scheduler is not None
            else None
        ),
        "args": vars(args),
        "epoch": int(epoch),
        "step": int(global_step),
        "global_samples_seen": int(global_samples_seen),
        "history": history,
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "task": str(args.task),
        "dataset": str(args.dataset),
        "single_dataloader": distributed_world_size == 1,
        "sampling": (
            "distributed_shuffled_SequenceDataset_indices"
            if distributed_world_size > 1
            else "uniform_shuffled_SequenceDataset_indices"
        ),
        "reward_mode": str(getattr(args, "dataset_reward_mode", "rise")),
        "reward_definition": REWARD_DEFINITIONS[
            str(getattr(args, "dataset_reward_mode", "rise"))
        ],
        "actor_training_objective": "diffusion_bc_plus_normalized_q_maximization",
        "actor_data_mode": "all_human_success_failure_rows",
        "critic_training_objective": "diffusion_ql_clipped_double_q_td",
        "critic_input_mode": "independent_causal_two_frame_temporal_encoders",
        "critic_action_space": "pretrained_dp_normalized_action_space",
        "critic_horizon": 1,
        "critic_chunk_horizon": 1,
        "critic_architecture": TEMPORAL_CRITIC_ARCHITECTURE,
        "critic_observation_horizon": int(args.critic_observation_horizon),
        "critic_latent_dim": int(args.latent_dim),
        "critic_action_hidden_dim": int(args.action_hidden_dim),
        "critic_num_attention_heads": int(args.num_attention_heads),
        "critic_num_action_conv_layers": int(args.num_action_conv_layers),
        "critic_dropout": float(args.dropout),
        "critic_temporal_num_layers": int(args.temporal_num_layers),
        "critic_temporal_num_heads": int(args.temporal_num_heads),
        "critic_temporal_feedforward_dim": int(
            args.temporal_feedforward_dim
        ),
        "critic_temporal_dropout": float(args.temporal_dropout),
        "critic_rise_v2_fusion_mode": str(args.rise_v2_fusion_mode),
        "critic_dynamics_target_dim": int(critics[0].dynamics_target_dim),
        "critic_dynamics_prediction_offsets": (),
        "latent_dynamics": False,
        "dynamics_weight": 0.0,
        "critic_encoder_initialized_from_pretrained_dp": True,
        "critic_target_mode": "eval",
        "critic_has_value_net": False,
        "critic_hidden_dims": tuple(int(value) for value in args.critic_hidden_dims),
        "num_critics": int(args.num_critics),
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
        "action_dim": int(actor_algo.ac_dim),
        "action_normalization_stats": copy.deepcopy(action_normalization_stats),
        "observation_horizon": int(actor_algo.algo_config.horizon.observation_horizon),
        "actor_prediction_horizon": int(actor_algo.algo_config.horizon.prediction_horizon),
        "actor_action_horizon": int(actor_algo.algo_config.horizon.action_horizon),
        "discount": float(args.discount),
        "target_tau": float(args.target_tau),
        "dql_eta": float(args.dql_eta),
        "dql_bc_weight": float(args.dql_bc_weight),
        "dql_q_normalization": "official_cross_head_generated_action_q",
        "dql_sampled_validation_safety": {
            "enabled": int(args.dql_sampled_validation_rows) > 0,
            "requested_rows": int(args.dql_sampled_validation_rows),
            "indices": list(
                getattr(args, "dql_sampled_validation_indices", [])
            ),
            "subset_seed": int(args.validation_seed),
            "noise_seed": getattr(
                args,
                "dql_sampled_validation_noise_seed",
                None,
            ),
            "saturation_threshold": float(
                getattr(
                    args,
                    "dql_sampled_validation_saturation_threshold",
                    SAMPLED_VALIDATION_SATURATION_THRESHOLD,
                )
            ),
            "max_behavior_error_ratio": float(
                args.dql_sampled_validation_max_error_ratio
            ),
            "max_motion_axis_saturation_excess": float(
                args.dql_sampled_validation_max_saturation_excess
            ),
            "gripper_excluded_from_saturation_gate": True,
        },
        "dql_reference_alignment": dql_reference_alignment(args),
        "actor_initialized_from_deployed_ema": True,
        "actor_encoder_trainable": int(global_step) >= int(
            args.resolved_actor_obs_encoder_freeze_steps
        ),
        "critic_encoders_trainable": int(global_step) >= int(
            args.resolved_encoder_freeze_steps
        ),
        "actor_ema_optimization_step": int(
            actor_algo.ema.optimization_step if actor_algo.ema is not None else 0
        ),
        "rng_state": (
            rank_zero_runtime["rng_state"]
            if rank_zero_runtime is not None
            else rng_state()
        ),
        "loader_generator_state": (
            rank_zero_runtime["loader_generator_state"]
            if rank_zero_runtime is not None
            else loader_generator.get_state()
        ),
        "distributed_training": {
            "enabled": bool(
                distributed_context is not None
                and distributed_context.enabled
            ),
            "world_size": distributed_world_size,
            "backend": (
                distributed_context.backend
                if distributed_context is not None
                else "none"
            ),
            "gradient_sync": "mean_all_reduce_before_optimizer_step",
            "unused_parameter_detection_collective": False,
            "random_q_head_sync": "none_rank_local_sampling",
            "q_denominator_sync": "none_rank_local_normalization",
            "metric_sync": "none_rank_zero_local_logging",
        },
        "distributed_rank_states": rank_runtime_states,
    }


def validate_resume_args(args: argparse.Namespace, checkpoint: dict) -> None:
    previous = checkpoint.get("args", {})
    exact_keys = (
        "dataset",
        "checkpoint",
        "task",
        "validation_dataset",
        "validation_seed",
        "reward_mode",
        "seed",
        "batch_size",
        "effective_global_batch_size",
        "schedule_reference_batch_size",
        "steps_per_epoch",
        "critic_hidden_dims",
        "critic_observation_horizon",
        "latent_dim",
        "action_hidden_dim",
        "num_attention_heads",
        "num_action_conv_layers",
        "dropout",
        "temporal_num_layers",
        "temporal_num_heads",
        "temporal_feedforward_dim",
        "temporal_dropout",
        "rise_v2_fusion_mode",
        "num_critics",
        "critic_group_norm",
        "critic_late_fusion_key",
        "discount",
        "target_tau",
        "actor_lr",
        "actor_obs_encoder_freeze_steps",
        "resolved_actor_obs_encoder_freeze_steps",
        "critic_lr",
        "encoder_lr",
        "encoder_freeze_steps",
        "resolved_encoder_freeze_steps",
        "lr_scheduler",
        "lr_warmup_steps",
        "resolved_lr_warmup_steps",
        "lr_total_steps",
        "lr_num_cycles",
        "dql_eta",
        "dql_bc_weight",
        "dql_q_batch_size",
        "dql_num_inference_steps",
        "dql_target_num_candidates",
        "dql_q_head",
        "dql_q_denominator_floor",
        "dql_clip_actions",
        "dql_critic_warmup_steps",
        "resolved_dql_critic_warmup_steps",
        "dql_actor_ema_update_every",
        "dql_sampled_validation_rows",
        "dql_sampled_validation_max_error_ratio",
        "dql_sampled_validation_max_saturation_excess",
        "actor_max_gradient_norm",
        "critic_max_gradient_norm",
    )
    for key in exact_keys:
        if key not in previous:
            continue
        old = jsonable(previous[key])
        new = jsonable(getattr(args, key))
        if old != new:
            raise ValueError(
                f"resume argument mismatch for {key}: checkpoint={old}, current={new}"
            )


def train(args: argparse.Namespace) -> dict:
    distributed = initialize_distributed(args)
    args.distributed = bool(distributed.enabled)
    args.distributed_rank = int(distributed.rank)
    args.distributed_local_rank = int(distributed.local_rank)
    args.distributed_world_size = int(distributed.world_size)
    configure_dql_batch_semantics(args, distributed.world_size)
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"{args.dataset} does not exist; build it with run_rgb_dp_idql.sh first"
        )
    if (
        args.validation_dataset is not None
        and not args.validation_dataset.is_file()
    ):
        raise FileNotFoundError(args.validation_dataset)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if int(args.num_critics) < 2:
        raise ValueError("Diffusion-QL requires at least two critics")
    if int(args.critic_observation_horizon) != 2:
        raise ValueError("RISE-v2 DQL requires exactly two critic frames")
    if float(args.dql_eta) < 0.0:
        raise ValueError("dql_eta must be non-negative")
    if int(args.dql_q_batch_size) < 0:
        raise ValueError("dql_q_batch_size must be non-negative")
    if int(args.dql_target_num_candidates) <= 0:
        raise ValueError("dql_target_num_candidates must be positive")
    if float(args.dql_q_denominator_floor) <= 0.0:
        raise ValueError("dql_q_denominator_floor must be positive")
    if int(args.dql_critic_warmup_steps) < 0:
        raise ValueError("dql_critic_warmup_steps must be non-negative")
    if int(args.actor_obs_encoder_freeze_steps) < 0:
        raise ValueError("actor_obs_encoder_freeze_steps must be non-negative")
    if int(args.encoder_freeze_steps) < 0:
        raise ValueError("encoder_freeze_steps must be non-negative")
    if int(args.dql_actor_ema_update_every) <= 0:
        raise ValueError("dql_actor_ema_update_every must be positive")
    if int(args.dql_sampled_validation_rows) < 0:
        raise ValueError("dql_sampled_validation_rows must be non-negative")
    if (
        int(args.dql_sampled_validation_rows) > 0
        and args.validation_dataset is None
    ):
        raise ValueError(
            "sampled validation rows require --validation-dataset"
        )
    if float(args.dql_sampled_validation_max_error_ratio) <= 0.0:
        raise ValueError(
            "dql_sampled_validation_max_error_ratio must be positive"
        )
    if not 0.0 <= float(
        args.dql_sampled_validation_max_saturation_excess
    ) <= 1.0:
        raise ValueError(
            "dql_sampled_validation_max_saturation_excess must be in [0, 1]"
        )
    if distributed.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    device = distributed.device
    # Build identical models before installing independent rank-local streams.
    seed_process(args.seed, device)

    # Checkpoints retain the CUDA device tags they were saved with. Stage on
    # CPU so every torchrun rank does not restore its copy onto cuda:0.
    dp_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    actor_policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_dict=dp_checkpoint,
        device=device,
        verbose=False,
    )
    actor_algo = actor_policy.policy
    initialized_from_ema = initialize_actor_from_deployed_ema(actor_algo)
    if actor_algo.ema is not None and not initialized_from_ema:
        raise RuntimeError("failed to initialize actor from deployed EMA")
    dp_checkpoint.pop("model", None)

    dataset, loader, loader_generator, _ = build_single_loader(
        args,
        actor_policy,
        dp_checkpoint,
    )
    if args.steps_per_epoch is None:
        args.steps_per_epoch = int(len(loader))
        args.steps_per_epoch_source = "auto_DataLoader_length"
    else:
        args.steps_per_epoch = int(args.steps_per_epoch)
        args.steps_per_epoch_source = "explicit_command_line"
    if int(args.steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive")
    args.lr_total_steps = int(args.epochs) * int(args.steps_per_epoch)
    if (
        args.lr_scheduler == "cosine"
        and int(args.resolved_lr_warmup_steps) >= int(args.lr_total_steps)
    ):
        raise ValueError(
            "resolved_lr_warmup_steps="
            f"{args.resolved_lr_warmup_steps} (reference "
            f"{args.lr_warmup_steps}) must be smaller than "
            f"total training steps={args.lr_total_steps}"
        )
    configure_actor_optimizer(
        actor_algo,
        args.actor_lr,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.resolved_lr_warmup_steps,
        total_steps=args.lr_total_steps,
        num_cycles=args.lr_num_cycles,
    )

    audit = dataset_audit(
        args.dataset,
        len(dataset),
        expected_task=args.task,
        expected_reward_mode=args.reward_mode,
        validity_key=getattr(dataset, "validity_key", None),
    )
    args.dataset_validity_key = audit["validity_key"]
    validation_dataset = None
    validation_loader = None
    validation_audit = None
    validation_split_audit = None
    if args.validation_dataset is not None and distributed.is_main_process:
        validation_args = copy.copy(args)
        validation_args.dataset = args.validation_dataset
        validation_args.seed = int(args.validation_seed)
        validation_args.distributed_world_size = 1
        validation_args.distributed_rank = 0
        validation_dataset, validation_loader, _, _ = build_single_loader(
            validation_args,
            actor_policy,
            dp_checkpoint,
            shuffle=False,
            drop_last=False,
        )
        validation_audit = dataset_audit(
            args.validation_dataset,
            len(validation_dataset),
            expected_task=args.task,
            expected_reward_mode=args.reward_mode,
            validity_key=getattr(validation_dataset, "validity_key", None),
        )
        validation_split_audit = audit_validation_dataset_split(
            args.dataset,
            args.validation_dataset,
        )
    args.dataset_reward_mode = str(audit["reward_mode"])
    action_stats = dp_checkpoint.get("action_normalization_stats")
    if action_stats is None:
        raise ValueError("pretrained DP checkpoint has no action normalization stats")
    obs_normalization_stats = copy.deepcopy(actor_policy.obs_normalization_stats)
    sampled_validation_reference = None
    args.dql_sampled_validation_indices = []
    args.dql_sampled_validation_noise_seed = None
    args.dql_sampled_validation_saturation_threshold = (
        SAMPLED_VALIDATION_SATURATION_THRESHOLD
    )
    if (
        int(args.dql_sampled_validation_rows) > 0
        and distributed.is_main_process
    ):
        if validation_dataset is None:
            raise RuntimeError("rank zero has no sampled validation dataset")
        sampled_validation_reference = build_sampled_validation_reference(
            args=args,
            dataset=validation_dataset,
            actor_algo=actor_algo,
            obs_normalization_stats=obs_normalization_stats,
        )
        args.dql_sampled_validation_indices = list(
            sampled_validation_reference["indices"]
        )
        args.dql_sampled_validation_noise_seed = int(
            sampled_validation_reference["noise_seed"]
        )

    critics, critic_targets, encoder_initialization = (
        make_dql_rise_v2_value_networks(
            actor_algo,
            hidden_dims=tuple(int(value) for value in args.critic_hidden_dims),
            latent_dim=int(args.latent_dim),
            action_hidden_dim=int(args.action_hidden_dim),
            num_attention_heads=int(args.num_attention_heads),
            num_action_conv_layers=int(args.num_action_conv_layers),
            dropout=float(args.dropout),
            num_critics=int(args.num_critics),
            critic_group_norm=bool(args.critic_group_norm),
            late_fusion_key=args.critic_late_fusion_key,
            observation_horizon=int(args.critic_observation_horizon),
            temporal_num_layers=int(args.temporal_num_layers),
            temporal_num_heads=int(args.temporal_num_heads),
            temporal_feedforward_dim=int(args.temporal_feedforward_dim),
            temporal_dropout=float(args.temporal_dropout),
            fusion_mode=str(args.rise_v2_fusion_mode),
        )
    )
    critics = critics.float().to(device)
    critic_targets = critic_targets.float().to(device)
    configure_critic_targets(critic_targets)
    critic_optimizer = make_dql_critic_optimizer(
        critics,
        learning_rate=float(args.critic_lr),
        encoder_learning_rate=float(args.encoder_lr),
    )
    critic_parameters = [
        parameter
        for group in critic_optimizer.param_groups
        for parameter in group["params"]
    ]
    critic_lr_scheduler = make_step_lr_scheduler(
        critic_optimizer,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.resolved_lr_warmup_steps,
        total_steps=args.lr_total_steps,
        num_cycles=args.lr_num_cycles,
    )

    start_epoch = 0
    global_step = 0
    global_samples_seen = 0
    history: list[dict] = []
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not checkpoint.get("rise_style_rgb_dql", False):
            raise ValueError("resume checkpoint is not from train_rgb_dp_dql.py")
        saved_distributed = checkpoint.get("distributed_training", {})
        if bool(saved_distributed.get("enabled", False)) and (
            not distributed.enabled
            or int(saved_distributed.get("world_size", 1))
            != distributed.world_size
        ):
            raise ValueError(
                "distributed checkpoints require distributed resume with the "
                "same world size: checkpoint="
                f"{saved_distributed.get('world_size')} requested="
                f"{distributed.world_size}"
            )
        if (
            not checkpoint.get(TEMPORAL_ONE_STEP_MARKER, False)
            or checkpoint.get("critic_architecture")
            != TEMPORAL_CRITIC_ARCHITECTURE
        ):
            incompatibility = {
                "resume_checkpoint": str(args.resume_checkpoint),
                "reason": "critic architecture and stable DQL target semantics changed",
            }
            if distributed.is_main_process:
                write_json(
                    args.output_dir / "resume_incompatible.json",
                    incompatibility,
                )
            raise ValueError(
                "cannot resume a pre-RISE-v2 DQL checkpoint; use a new DQL_OUTPUT_DIR"
            )
        validate_resume_args(args, checkpoint)
        checkpoint_action_stats = checkpoint.get("action_normalization_stats")
        if checkpoint_action_stats is None or not action_normalization_stats_match(
            checkpoint_action_stats,
            action_stats,
        ):
            raise ValueError(
                "resume checkpoint action normalization does not match the DP checkpoint"
            )
        if len(checkpoint.get("critics", [])) != len(critics):
            raise ValueError("resume checkpoint critic count mismatch")
        if len(checkpoint.get("critic_targets", [])) != len(critic_targets):
            raise ValueError("resume checkpoint target critic count mismatch")
        actor_algo.deserialize(checkpoint["actor_model"], load_optimizers=True)
        for critic, state in zip(critics, checkpoint["critics"]):
            critic.load_state_dict(state)
        for target, state in zip(critic_targets, checkpoint["critic_targets"]):
            target.load_state_dict(state)
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        scheduler_state = checkpoint.get("critic_lr_scheduler")
        if (scheduler_state is not None) != (critic_lr_scheduler is not None):
            raise ValueError("resume checkpoint critic scheduler configuration mismatch")
        if critic_lr_scheduler is not None:
            critic_lr_scheduler.load_state_dict(scheduler_state)
        if actor_algo.ema is not None:
            actor_algo.ema.optimization_step = int(
                checkpoint.get("actor_ema_optimization_step", 0)
            )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["step"])
        global_samples_seen = int(
            checkpoint.get(
                "global_samples_seen",
                global_step * int(args.effective_global_batch_size),
            )
        )
        history = list(checkpoint.get("history", []))
        rank_runtime_states = checkpoint.get("distributed_rank_states")
        if distributed.enabled and bool(saved_distributed.get("enabled", False)):
            if not isinstance(rank_runtime_states, (list, tuple)):
                raise ValueError(
                    "distributed checkpoint is missing per-rank runtime states"
                )
            if len(rank_runtime_states) != distributed.world_size:
                raise ValueError(
                    "distributed checkpoint rank-state count does not match "
                    f"world_size={distributed.world_size}"
                )
            rank_runtime = rank_runtime_states[distributed.rank]
            if int(rank_runtime.get("rank", -1)) != distributed.rank:
                raise ValueError("distributed checkpoint rank states are unordered")
            loader_generator.set_state(
                rank_runtime["loader_generator_state"].cpu()
            )
            restore_process_rng_state(rank_runtime.get("rng_state"), device)
        elif distributed.enabled:
            loader_generator.manual_seed(int(args.seed) + distributed.rank)
            seed_process(int(args.seed) + distributed.rank, device)
        else:
            loader_generator.set_state(
                checkpoint["loader_generator_state"].cpu()
            )
            saved_rng_state = checkpoint.get("rng_state")
            if saved_rng_state and "cuda_local" in saved_rng_state:
                restore_process_rng_state(saved_rng_state, device)
            else:
                restore_rng_state(saved_rng_state)
        del checkpoint
        if distributed.is_main_process:
            print(
                f"Resumed {args.resume_checkpoint} at epoch={start_epoch} "
                f"step={global_step}",
                flush=True,
            )

    best_validation_loss = None
    best_validation_epoch = None
    sampled_safety_required = int(args.dql_sampled_validation_rows) > 0
    for completed_epoch in history:
        candidate = eligible_validation_loss(
            completed_epoch.get("metrics", {}),
            sampled_safety_required=sampled_safety_required,
        )
        if candidate is None:
            continue
        if (
            best_validation_loss is None
            or candidate < best_validation_loss
        ):
            best_validation_loss = candidate
            best_validation_epoch = int(completed_epoch.get("epoch", -1))

    actor_algo.set_train()
    critics.train()
    configure_critic_targets(critic_targets)
    synchronized_modules: list[nn.Module] = [
        actor_algo.nets,
        critics,
        critic_targets,
    ]
    if actor_algo.ema is not None:
        synchronized_modules.append(actor_algo.ema.averaged_model)
    broadcast_module_state(synchronized_modules, distributed)
    gradient_sync_fn = (
        (
            lambda parameters: all_reduce_gradients(
                parameters,
                distributed,
                bucket_cap_mb=args.gradient_bucket_cap_mb,
                # RISE-v2 Qs and the unconditioned DP actor use static graphs;
                # every optimizer parameter participates on every update.
                preserve_unused_parameters=False,
            )
        )
        if distributed.enabled
        else None
    )
    synchronize_training_buffers = modules_have_mutable_batch_norm(
        synchronized_modules
    )
    if distributed.enabled and args.resume_checkpoint is None:
        seed_process(int(args.seed) + distributed.rank, device)
    trainability = actor_trainability(actor_algo)
    startup = {
        "stacked_pretrained_dql_critic": False,
        "critic_architecture": TEMPORAL_CRITIC_ARCHITECTURE,
        "task": str(args.task),
        "dataset": audit,
        "validation": {
            "enabled": validation_loader is not None,
            "dataset": validation_audit,
            "split_audit": validation_split_audit,
            "seed": int(args.validation_seed),
            "rank_zero_only": True,
            "full_coverage": True,
            "actor_weights": "ema",
            "selection_metric": "validation/actor/Loss",
            "selection_gate": (
                "validation/sampled/eligible"
                if sampled_safety_required
                else None
            ),
            "sampled_policy": {
                "enabled": sampled_safety_required,
                "requested_rows": int(args.dql_sampled_validation_rows),
                "actual_rows": (
                    len(sampled_validation_reference["indices"])
                    if sampled_validation_reference is not None
                    else 0
                ),
                "indices": (
                    list(sampled_validation_reference["indices"])
                    if sampled_validation_reference is not None
                    else []
                ),
                "subset_seed": int(args.validation_seed),
                "noise_seed": (
                    int(sampled_validation_reference["noise_seed"])
                    if sampled_validation_reference is not None
                    else None
                ),
                "actor_weights": "ema",
                "base_actor": "pretrained_deployed_ema",
                "action": "first_executable_action",
                "saturation_threshold": (
                    SAMPLED_VALIDATION_SATURATION_THRESHOLD
                ),
                "max_behavior_error_ratio": float(
                    args.dql_sampled_validation_max_error_ratio
                ),
                "max_motion_axis_saturation_excess": float(
                    args.dql_sampled_validation_max_saturation_excess
                ),
                "gripper_excluded_from_saturation_gate": True,
            },
        },
        "data_routing": {
            "shared_loader": True,
            "actor_rows": (
                f"{audit['validity_key']}_admitted_shared_rows"
                if audit["validity_key"] is not None
                else "all_human_success_failure_no_mask"
            ),
            "critic_rows": (
                f"{audit['validity_key']}_admitted_shared_rows"
                if audit["validity_key"] is not None
                else "all_human_success_failure"
            ),
            "source_masking": audit["validity_key"] is not None,
            "filtered_rows": int(
                audit["hdf5_total_transitions"]
                - audit["sequence_dataset_size"]
            ),
        },
        "loader": {
            "class": dataset.__class__.__name__,
            "sparse_dql_loader": bool(args.sparse_dql_loader),
            "validity_key": audit["validity_key"],
            "unfiltered_sequence_dataset_size": int(
                audit["hdf5_total_transitions"]
            ),
            "admitted_sequence_dataset_size": int(
                audit["sequence_dataset_size"]
            ),
            "observation_loading": (
                "current_and_next_observation_stacks_only"
                if args.sparse_dql_loader
                else "full_obs_and_next_obs_sequences"
            ),
            "num_loaders": int(distributed.world_size),
            "sampler": loader.sampler.__class__.__name__,
            "batch_size": int(args.batch_size),
            "batch_size_per_rank": int(args.batch_size),
            "effective_global_batch_size": int(
                args.effective_global_batch_size
            ),
            "num_batches": int(len(loader)),
            "steps_per_epoch": int(args.steps_per_epoch),
            "steps_per_epoch_source": str(args.steps_per_epoch_source),
            "seed": int(args.seed),
        },
        "batch_semantics": {
            "batch_size_control": "per_gpu_BATCH_SIZE",
            "batch_size_per_rank": int(args.batch_size),
            "world_size": int(distributed.world_size),
            "effective_global_batch_size": int(
                args.effective_global_batch_size
            ),
            "schedule_reference_batch_size": int(
                args.schedule_reference_batch_size
            ),
            "effective_to_reference_ratio": float(args.schedule_batch_ratio),
            "learning_rates_automatically_scaled": False,
            "dql_q_batch_size_per_rank": int(args.dql_q_batch_size),
            "reference_lr_warmup_steps": int(args.lr_warmup_steps),
            "resolved_lr_warmup_steps": int(args.resolved_lr_warmup_steps),
            "reference_dql_critic_warmup_steps": int(
                args.dql_critic_warmup_steps
            ),
            "resolved_dql_critic_warmup_steps": int(
                args.resolved_dql_critic_warmup_steps
            ),
            "reference_actor_obs_encoder_freeze_steps": int(
                args.actor_obs_encoder_freeze_steps
            ),
            "resolved_actor_obs_encoder_freeze_steps": int(
                args.resolved_actor_obs_encoder_freeze_steps
            ),
            "reference_critic_encoder_freeze_steps": int(
                args.encoder_freeze_steps
            ),
            "resolved_critic_encoder_freeze_steps": int(
                args.resolved_encoder_freeze_steps
            ),
            "target_tau_step_unit": "optimizer_update",
            "actor_ema_update_every_step_unit": "optimizer_update",
        },
        "normalization": {
            "action": "pretrained_dp_checkpoint_action_normalization_stats",
            "observation": (
                "pretrained_dp_checkpoint_obs_normalization_stats"
                if obs_normalization_stats is not None
                else "none_as_in_pretrained_dp_checkpoint"
            ),
            "mixed_dataset_statistics_used": False,
        },
        "architecture": {
            "actor": trainability,
            "critic_parameter_counts": [
                parameter_count(critic) for critic in critics
            ],
            "target_critic_parameter_counts": [
                parameter_count(target) for target in critic_targets
            ],
            "independent_raw_obs_encoders": True,
            "independent_pretrained_dp_obs_encoders": True,
            "critic_encoder_initialized_from_pretrained_dp": True,
            "critic_encoder_initialization": encoder_initialization,
            "critic_architecture": TEMPORAL_CRITIC_ARCHITECTURE,
            "critic_observation_horizon": int(
                args.critic_observation_horizon
            ),
            "causal_temporal_trunk": True,
            "state_action_fusion": str(args.rise_v2_fusion_mode),
            "critic_normalization": "RISE_v2_with_deployed_encoder_GN",
            "critic_group_norm": bool(args.critic_group_norm),
            "target_critic_mode": "eval",
            "critic_late_fusion_key": args.critic_late_fusion_key,
        },
        "dql_reference_alignment": dql_reference_alignment(args),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "discount": float(args.discount),
            "target_tau": float(args.target_tau),
            "actor_lr": float(args.actor_lr),
            "actor_obs_encoder_freeze_steps": int(
                args.actor_obs_encoder_freeze_steps
            ),
            "resolved_actor_obs_encoder_freeze_steps": int(
                args.resolved_actor_obs_encoder_freeze_steps
            ),
            "critic_lr": float(args.critic_lr),
            "encoder_lr": float(args.encoder_lr),
            "encoder_freeze_steps": int(args.encoder_freeze_steps),
            "resolved_encoder_freeze_steps": int(
                args.resolved_encoder_freeze_steps
            ),
            "dql_eta": float(args.dql_eta),
            "dql_bc_weight": float(args.dql_bc_weight),
            "dql_q_batch_size": int(args.dql_q_batch_size),
            "dql_num_inference_steps": int(args.dql_num_inference_steps),
            "dql_target_num_candidates": int(args.dql_target_num_candidates),
            "dql_q_head": str(args.dql_q_head),
            "dql_q_normalization": "official_cross_head_generated_action_q",
            "dql_q_denominator_floor": float(args.dql_q_denominator_floor),
            "dql_critic_warmup_steps": int(args.dql_critic_warmup_steps),
            "resolved_dql_critic_warmup_steps": int(
                args.resolved_dql_critic_warmup_steps
            ),
            "dql_actor_ema_update_every": int(
                args.dql_actor_ema_update_every
            ),
            "dql_sampled_validation_rows": int(
                args.dql_sampled_validation_rows
            ),
            "dql_sampled_validation_max_error_ratio": float(
                args.dql_sampled_validation_max_error_ratio
            ),
            "dql_sampled_validation_max_saturation_excess": float(
                args.dql_sampled_validation_max_saturation_excess
            ),
            "actor_max_gradient_norm": float(args.actor_max_gradient_norm),
            "critic_max_gradient_norm": float(args.critic_max_gradient_norm),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "resolved_lr_warmup_steps": int(args.resolved_lr_warmup_steps),
            "lr_total_steps": int(args.lr_total_steps),
        },
        "distributed": {
            "enabled": bool(distributed.enabled),
            "world_size": int(distributed.world_size),
            "backend": distributed.backend,
            "launcher": "torchrun" if distributed.enabled else "python",
            "gradient_sync": "bounded_async_bucketed_mean_all_reduce",
            "unused_parameter_detection_collective": False,
            "gradient_bucket_cap_mb": float(args.gradient_bucket_cap_mb),
            "random_q_head_sync": "none_rank_local_sampling",
            "q_denominator_sync": "none_rank_local_normalization",
            "per_step_buffer_broadcast": bool(
                synchronize_training_buffers
            ),
            "per_step_metric_collective": False,
            "metric_source": "rank_zero_local",
            "rank_zero_writes_only": True,
        },
    }
    if distributed.is_main_process:
        write_json(args.output_dir / "startup_audit.json", startup)
        print(json.dumps(jsonable(startup), indent=2), flush=True)
        writer = make_tensorboard_writer(args.output_dir)
    else:
        writer = None
    shared_action_range_validated = False
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        if distributed.enabled and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        epoch_iterator = iter(loader)
        metric_accumulator = (
            LocalScalarMetricAccumulator(device)
            if distributed.is_main_process
            else None
        )
        for step_in_epoch in range(1, int(args.steps_per_epoch) + 1):
            try:
                raw_batch = next(epoch_iterator)
            except StopIteration:
                epoch_iterator = iter(loader)
                raw_batch = next(epoch_iterator)
            raw_batch = align_shared_batch_actions(
                raw_batch,
                validate=not shared_action_range_validated,
            )
            shared_action_range_validated = True
            actor_obs_encoder_trainable = global_step >= int(
                args.resolved_actor_obs_encoder_freeze_steps
            )
            critic_encoders_trainable = global_step >= int(
                args.resolved_encoder_freeze_steps
            )
            set_actor_obs_encoder_trainable(
                actor_algo,
                actor_obs_encoder_trainable,
            )
            set_critic_encoders_trainable(
                critics,
                critic_encoders_trainable,
            )
            if synchronize_training_buffers:
                broadcast_module_buffers(
                    synchronized_modules,
                    distributed,
                )
            learning_rates = {
                "lr/actor": float(
                    actor_algo.optimizers["policy"].param_groups[0]["lr"]
                ),
                "lr/critic": float(critic_optimizer.param_groups[0]["lr"]),
                "lr/encoder": float(critic_optimizer.param_groups[1]["lr"]),
            }

            critic_batch = process_dql_critic_batch(
                raw_batch,
                actor_algo,
                obs_normalization_stats,
                critic_observation_horizon=int(
                    args.critic_observation_horizon
                ),
                validate_actions=step_in_epoch == 1 and epoch == start_epoch + 1,
            )
            with torch.no_grad():
                next_actor_observations = prepare_next_actor_observations(
                    actor_algo,
                    raw_batch,
                    obs_normalization_stats,
                )
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss, critic_info = compute_critic_loss(
                actor_algo=actor_algo,
                critics=critics,
                critic_targets=critic_targets,
                critic_batch=critic_batch,
                next_actor_observations=next_actor_observations,
                discount=float(args.discount),
                use_huber=bool(args.use_huber),
                num_inference_steps=int(args.dql_num_inference_steps),
                num_target_candidates=int(args.dql_target_num_candidates),
                clip_actions=bool(args.dql_clip_actions),
            )
            critic_loss.backward()
            if gradient_sync_fn is not None:
                gradient_sync_fn(critic_parameters)
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic_parameters,
                float(args.critic_max_gradient_norm),
            )
            critic_optimizer.step()

            actor_batch = prepare_actor_batch(
                actor_algo,
                raw_batch,
                obs_normalization_stats,
            )
            actor_info = actor_train_step(
                actor_algo=actor_algo,
                critics=critics,
                actor_batch=actor_batch,
                critic_batch=critic_batch,
                args=args,
                global_step=global_step,
                gradient_sync_fn=gradient_sync_fn,
                defer_scalar_conversion=True,
            )
            soft_update_targets(critics, critic_targets, float(args.target_tau))
            if critic_lr_scheduler is not None:
                critic_lr_scheduler.step()

            global_samples_seen += int(
                raw_batch["actions"].shape[0] * distributed.world_size
            )
            global_step += 1
            metrics = {
                **critic_info,
                "critic/gradient_norm": critic_gradient_norm,
            }
            metrics.update(actor_info)
            metrics.update(learning_rates)
            metrics["actor/obs_encoder_trainable"] = float(
                actor_obs_encoder_trainable
            )
            metrics["critic/encoder_trainable"] = float(
                critic_encoders_trainable
            )
            metrics["distributed/world_size"] = float(distributed.world_size)
            metrics["data/effective_global_batch_rows"] = float(
                raw_batch["actions"].shape[0] * distributed.world_size
            )
            should_log = (
                global_step % int(args.log_every) == 0
                or step_in_epoch == int(args.steps_per_epoch)
            )
            if distributed.is_main_process:
                metric_accumulator.update(metrics)
                if should_log:
                    logged_metrics = materialize_local_scalar_metrics(
                        metrics,
                        device,
                    )
                    if writer is not None:
                        for key, value in logged_metrics.items():
                            writer.add_scalar(key, value, global_step)
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "global_step": global_step,
                                **logged_metrics,
                            }
                        ),
                        flush=True,
                    )

        epoch_metrics = (
            metric_accumulator.means()
            if distributed.is_main_process
            else {}
        )
        new_best_validation = False
        if validation_loader is not None:
            validation_metrics = evaluate_dql_validation_epoch(
                args=args,
                epoch=epoch,
                loader=validation_loader,
                actor_algo=actor_algo,
                critics=critics,
                critic_targets=critic_targets,
                obs_normalization_stats=obs_normalization_stats,
                device=device,
                sampled_validation_reference=sampled_validation_reference,
            )
            epoch_metrics.update(validation_metrics)
            if writer is not None:
                for key, value in validation_metrics.items():
                    writer.add_scalar(key, value, global_step)
            raw_selection_loss = validation_metrics.get(
                "validation/actor/Loss"
            )
            if raw_selection_loss is None or not np.isfinite(
                float(raw_selection_loss)
            ):
                raise ValueError(
                    "held-out DQL validation did not produce a finite EMA "
                    "actor loss"
                )
            selection_loss = eligible_validation_loss(
                validation_metrics,
                sampled_safety_required=sampled_safety_required,
            )
            if (
                selection_loss is not None
                and (
                    best_validation_loss is None
                    or selection_loss < best_validation_loss
                )
            ):
                best_validation_loss = selection_loss
                best_validation_epoch = int(epoch)
                new_best_validation = True
        if distributed.enabled and args.validation_dataset is not None:
            best_flag = torch.tensor(
                [int(new_best_validation)],
                device=device,
                dtype=torch.int64,
            )
            dist.broadcast(best_flag, src=0)
            new_best_validation = bool(best_flag.item())

        epoch_summary = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "global_samples_seen": int(global_samples_seen),
            "metrics": epoch_metrics,
        }
        history.append(epoch_summary)
        partial_summary = {
            **startup,
            "last_completed_epoch": int(epoch),
            "global_step": int(global_step),
            "global_samples_seen": int(global_samples_seen),
            "last_epoch_metrics": epoch_summary["metrics"],
            "best_validation": {
                "metric": "validation/actor/Loss",
                "gate": (
                    "validation/sampled/eligible"
                    if sampled_safety_required
                    else None
                ),
                "loss": best_validation_loss,
                "epoch": best_validation_epoch,
                "checkpoint": (
                    str(args.output_dir / "best_validation.pt")
                    if best_validation_epoch is not None
                    else None
                ),
            },
            "history": history,
            "checkpoints": {
                "latest": str(args.output_dir / "latest.pt"),
                "last": str(args.output_dir / "last.pt"),
                "best_validation": (
                    str(args.output_dir / "best_validation.pt")
                    if best_validation_epoch is not None
                    else None
                ),
            },
        }
        if distributed.is_main_process:
            write_json(
                args.output_dir / "partial_summary.json",
                partial_summary,
            )

        should_save = (
            epoch % int(args.save_every_epochs) == 0
            or epoch == int(args.epochs)
            or new_best_validation
        )
        if should_save:
            rank_runtime_states = (
                gather_rank_runtime_states(loader_generator, distributed)
                if distributed.enabled
                else None
            )
            if distributed.is_main_process:
                payload = checkpoint_payload(
                    args=args,
                    actor_algo=actor_algo,
                    critics=critics,
                    critic_targets=critic_targets,
                    critic_optimizer=critic_optimizer,
                    critic_lr_scheduler=critic_lr_scheduler,
                    action_normalization_stats=action_stats,
                    epoch=epoch,
                    global_step=global_step,
                    global_samples_seen=global_samples_seen,
                    history=history,
                    loader_generator=loader_generator,
                    rank_runtime_states=rank_runtime_states,
                    distributed_context=distributed,
                )
                latest_path = args.output_dir / "latest.pt"
                atomic_torch_save(payload, latest_path)
                if new_best_validation:
                    replace_with_hardlink(
                        latest_path,
                        args.output_dir / "best_validation.pt",
                    )
                if (
                    int(args.snapshot_every_epochs) > 0
                    and epoch % int(args.snapshot_every_epochs) == 0
                ):
                    replace_with_hardlink(
                        latest_path,
                        args.output_dir
                        / "models"
                        / f"model_epoch_{epoch}.pt",
                    )
                if writer is not None:
                    writer.flush()
                print(
                    f"Saved {latest_path} at epoch={epoch} "
                    f"step={global_step}",
                    flush=True,
                )

    if distributed.is_main_process:
        last_path = args.output_dir / "last.pt"
        if int(args.epochs) > start_epoch:
            replace_with_hardlink(args.output_dir / "latest.pt", last_path)
        elif not last_path.exists():
            if args.resume_checkpoint is None:
                raise RuntimeError("training is complete but last.pt is missing")
            replace_with_hardlink(args.resume_checkpoint, last_path)
        final_summary = json.loads(
            (args.output_dir / "partial_summary.json").read_text()
        )
        final_summary["complete"] = True
        final_summary["last_checkpoint"] = str(last_path)
        write_json(args.output_dir / "summary.json", final_summary)
        print(
            json.dumps(
                jsonable(
                    {key: value for key, value in final_summary.items() if key != "history"}
                ),
                indent=2,
            )
        )
    else:
        final_summary = {}
    if writer is not None:
        writer.flush()
        writer.close()
    close_dataset = getattr(dataset, "close", None)
    if callable(close_dataset):
        close_dataset()
    close_validation_dataset = getattr(validation_dataset, "close", None)
    if callable(close_validation_dataset):
        close_validation_dataset()
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=(
            "square",
            "can",
            "transport",
            "tool_hang",
            "pick_cup",
            "stack_cup",
            "move_spoon",
        ),
        default="square",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--validation-dataset", type=Path, default=None)
    parser.add_argument("--validation-seed", type=int, default=10000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable torchrun data-parallel training. This is also enabled "
            "automatically when WORLD_SIZE is greater than one."
        ),
    )
    parser.add_argument(
        "--distributed-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-bucket-cap-mb",
        type=float,
        default=100.0,
        help="Maximum size of each flat gradient all-reduce bucket in MiB.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help="Local process rank supplied by torchrun; the environment wins.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--schedule-reference-batch-size",
        type=int,
        default=100,
        help=(
            "Reference global batch used to express LR and DQL critic "
            "warmups in processed-sample units."
        ),
    )
    parser.add_argument(
        "--sparse-dql-loader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load only the current and next observation stacks while "
            "preserving the full action and metadata sequence."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hdf5-cache-mode", choices=("low_dim", "none"), default="low_dim")
    parser.add_argument(
        "--reward-mode",
        choices=tuple(REWARD_DEFINITIONS),
        default="terminal_success",
        help="Expected dataset reward mode; terminal_success is the default.",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--actor-obs-encoder-freeze-steps", type=int, default=0)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--encoder-freeze-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-num-cycles", type=float, default=0.5)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=(300, 400, 300))
    parser.add_argument("--critic-observation-horizon", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-action-conv-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temporal-num-layers", type=int, default=2)
    parser.add_argument("--temporal-num-heads", type=int, default=6)
    parser.add_argument("--temporal-feedforward-dim", type=int, default=600)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument(
        "--rise-v2-fusion-mode",
        choices=("concat", "film"),
        default="film",
    )
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--critic-group-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--critic-late-fusion-key",
        default="robot0_gripper_qpos",
        help="One observation key or a comma-separated key list.",
    )
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--critic-max-gradient-norm", type=float, default=10.0)
    parser.add_argument("--dql-eta", type=float, default=1.0)
    parser.add_argument("--dql-bc-weight", type=float, default=1.0)
    parser.add_argument("--dql-q-batch-size", type=int, default=8)
    parser.add_argument("--dql-num-inference-steps", type=int, default=5)
    parser.add_argument("--dql-target-num-candidates", type=int, default=1)
    parser.add_argument("--dql-q-head", choices=("random", "q1", "q2", "min"), default="random")
    parser.add_argument("--dql-q-denominator-floor", type=float, default=1.0)
    parser.add_argument("--dql-critic-warmup-steps", type=int, default=1000)
    parser.add_argument("--dql-actor-ema-update-every", type=int, default=5)
    parser.add_argument("--dql-clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dql-sampled-validation-rows",
        type=int,
        default=0,
        help=(
            "Number of fixed held-out rows used to reverse-sample the EMA "
            "actor; zero disables the sampled-policy checkpoint gate."
        ),
    )
    parser.add_argument(
        "--dql-sampled-validation-max-error-ratio",
        type=float,
        default=2.0,
        help=(
            "Maximum current-to-behavior action error divided by the "
            "deployed base actor's error."
        ),
    )
    parser.add_argument(
        "--dql-sampled-validation-max-saturation-excess",
        type=float,
        default=0.25,
        help=(
            "Maximum per-motion-axis saturation increase over the deployed "
            "base actor. The final gripper dimension is excluded."
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--snapshot-every-epochs", type=int, default=10)
    args = parser.parse_args()
    for key in (
        "dataset",
        "validation_dataset",
        "checkpoint",
        "output_dir",
        "resume_checkpoint",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    if not args.critic_late_fusion_key:
        args.critic_late_fusion_key = None
    if args.lr_warmup_steps < 0:
        parser.error("lr-warmup-steps must be non-negative")
    if args.actor_obs_encoder_freeze_steps < 0:
        parser.error("actor-obs-encoder-freeze-steps must be non-negative")
    if args.encoder_freeze_steps < 0:
        parser.error("encoder-freeze-steps must be non-negative")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        parser.error("steps-per-epoch must be positive when specified")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.schedule_reference_batch_size <= 0:
        parser.error("schedule-reference-batch-size must be positive")
    if args.gradient_bucket_cap_mb <= 0.0:
        parser.error("gradient-bucket-cap-mb must be positive")
    if args.lr_num_cycles <= 0.0:
        parser.error("lr-num-cycles must be positive")
    if args.actor_max_gradient_norm <= 0.0:
        parser.error("actor-max-gradient-norm must be positive")
    if args.critic_max_gradient_norm <= 0.0:
        parser.error("critic-max-gradient-norm must be positive")
    if args.dql_sampled_validation_rows < 0:
        parser.error("dql-sampled-validation-rows must be non-negative")
    if args.dql_sampled_validation_max_error_ratio <= 0.0:
        parser.error(
            "dql-sampled-validation-max-error-ratio must be positive"
        )
    if not 0.0 <= args.dql_sampled_validation_max_saturation_excess <= 1.0:
        parser.error(
            "dql-sampled-validation-max-saturation-excess must be in [0, 1]"
        )
    if (
        args.dql_sampled_validation_rows > 0
        and args.validation_dataset is None
    ):
        parser.error(
            "dql-sampled-validation-rows requires --validation-dataset"
        )
    if args.log_every <= 0:
        parser.error("log-every must be positive")
    if args.save_every_epochs <= 0:
        parser.error("save-every-epochs must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
