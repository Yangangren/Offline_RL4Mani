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
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from train_rgb_dp_idql import (
    REWARD_DEFINITIONS,
    action_normalization_stats_match,
    actor_trainability,
    align_shared_batch_actions,
    atomic_torch_save,
    build_single_loader,
    configure_actor_optimizer,
    dataset_audit,
    initialize_actor_from_deployed_ema,
    jsonable,
    make_tensorboard_writer,
    make_rise_value_networks,
    make_step_lr_scheduler,
    mean_metrics,
    parameter_count,
    process_critic_batch,
    replace_with_hardlink,
    restore_rng_state,
    rng_state,
    scalar_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "datasets/square/idql/square_rgb_dp_idql_200demo_100success_50failure_task_reward.hdf5"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/models/model_epoch_200.pth"
)
DEFAULT_OUTPUT = (
    ROOT
    / "trained_models/square_rgb_dp/dql/200demo_100success_50failure_task_reward"
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
        index = int(torch.randint(len(predictions), (), device=predictions[0].device))
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
    dataset_actions: torch.Tensor,
    goal_observations,
    num_inference_steps: int,
    q_head: str,
    denominator_floor: float,
    clip_actions: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equation (3), with alpha normalized by dataset-action |Q|."""
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
        with evaluating(critics):
            policy_predictions = [
                critic(
                    obs_dict=critic_observations,
                    acts=policy_actions,
                    goal_dict=goal_observations,
                )
                for critic in critics
            ]
            policy_q, head_index = choose_q_values(policy_predictions, q_head)
            with torch.no_grad():
                dataset_predictions = [
                    critic(
                        obs_dict=critic_observations,
                        acts=dataset_actions,
                        goal_dict=goal_observations,
                    )
                    for critic in critics
                ]
                dataset_q, _ = choose_q_values(
                    dataset_predictions,
                    "min" if head_index < 0 else f"q{head_index + 1}",
                )
                denominator = dataset_q.abs().mean().clamp_min(
                    float(denominator_floor)
                )
            q_loss = -policy_q.mean() / denominator
    finally:
        restore_requires_grad(critics, previous_grad_state)

    return q_loss, {
        "actor/q_loss": q_loss.detach(),
        "actor/policy_q_mean": policy_q.mean().detach(),
        "actor/dataset_abs_q_mean": dataset_q.abs().mean().detach(),
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
) -> dict[str, float]:
    bc_loss, bc_info = diffusion_bc_loss(actor_algo, actor_batch)
    available = int(next(iter(actor_batch["obs"].values())).shape[0])
    q_batch_size = min(int(args.dql_q_batch_size), available)
    q_loss = bc_loss.new_zeros(())
    q_info: dict[str, torch.Tensor] = {"actor/q_loss": q_loss.detach()}
    if float(args.dql_eta) > 0.0 and q_batch_size > 0:
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
            dataset_actions=critic_batch["actions"][:q_batch_size],
            goal_observations=goal_observations,
            num_inference_steps=int(args.dql_num_inference_steps),
            q_head=str(args.dql_q_head),
            denominator_floor=float(args.dql_q_denominator_floor),
            clip_actions=bool(args.dql_clip_actions),
        )

    actor_loss = float(args.dql_bc_weight) * bc_loss + float(args.dql_eta) * q_loss
    optimizer = actor_algo.optimizers["policy"]
    optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        actor_algo.nets.parameters(),
        float(args.max_gradient_norm),
    )
    optimizer.step()
    if actor_algo.ema is not None:
        actor_algo._update_ema()
    actor_algo.on_gradient_step()

    return scalar_metrics(
        {
            "actor/loss": actor_loss.detach(),
            "actor/gradient_norm": gradient_norm,
            "actor/q_batch_size": float(q_batch_size),
            "actor/dql_eta": float(args.dql_eta),
            "actor/effective_q_scale": (
                float(args.dql_eta) / q_info["actor/q_denominator"]
                if "actor/q_denominator" in q_info
                else 0.0
            ),
            **bc_info,
            **q_info,
        }
    )


def soft_update_targets(
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    tau: float,
) -> None:
    with torch.no_grad():
        for critic, target in zip(critics, critic_targets):
            TorchUtils.soft_update(source=critic, target=target, tau=float(tau))


def dql_reference_alignment(args: argparse.Namespace) -> dict[str, Any]:
    reward_mode = str(getattr(args, "dataset_reward_mode", "rise"))
    return {
        "paper": "Wang et al., Diffusion Policies as an Expressive Policy Class for Offline RL, ICLR 2023",
        "matched": [
            "diffusion_behavior_cloning_plus_q_maximization_actor_loss",
            "q_gradient_backpropagated_through_full_reverse_diffusion_chain",
            "alpha_normalized_by_mean_absolute_dataset_action_q",
            "independent_twin_q_critics_and_target_critics",
            "clipped_double_q_bellman_backup",
            "ema_diffusion_target_actor",
            "uniform_offline_transition_sampling",
        ],
        "robot_policy_adaptations": [
            "pretrained_rgb_diffusion_policy_initialization",
            "actor_predicts_a_chunk_but_q_scores_first_executable_action",
            (
                "source_environment_task_reward"
                if reward_mode == "task"
                else "human_demo_reward_1_and_deployment_rollout_reward_0"
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
    history: list[dict],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        # Reuse the raw-RGB actor / critic evaluator shared with IDQL.
        "rise_style_rgb_idql": True,
        "rise_style_rgb_dql": True,
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
        "history": history,
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "task": str(args.task),
        "dataset": str(args.dataset),
        "single_dataloader": True,
        "sampling": "uniform_shuffled_SequenceDataset_indices",
        "reward_mode": str(getattr(args, "dataset_reward_mode", "rise")),
        "reward_definition": REWARD_DEFINITIONS[
            str(getattr(args, "dataset_reward_mode", "rise"))
        ],
        "actor_training_objective": "diffusion_bc_plus_normalized_q_maximization",
        "actor_data_mode": "all_human_success_failure_rows",
        "critic_training_objective": "diffusion_ql_clipped_double_q_td",
        "critic_input_mode": "independent_raw_observation_encoders",
        "critic_action_space": "pretrained_dp_normalized_action_space",
        "critic_horizon": 1,
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
        "dql_q_normalization": "mean_absolute_dataset_action_q",
        "dql_reference_alignment": dql_reference_alignment(args),
        "actor_initialized_from_deployed_ema": True,
        "actor_encoder_trainable": True,
        "actor_ema_optimization_step": int(
            actor_algo.ema.optimization_step if actor_algo.ema is not None else 0
        ),
        "rng_state": rng_state(),
        "loader_generator_state": loader_generator.get_state(),
    }


def validate_resume_args(args: argparse.Namespace, checkpoint: dict) -> None:
    previous = checkpoint.get("args", {})
    exact_keys = (
        "dataset",
        "checkpoint",
        "task",
        "reward_mode",
        "seed",
        "batch_size",
        "steps_per_epoch",
        "critic_hidden_dims",
        "num_critics",
        "critic_group_norm",
        "critic_late_fusion_key",
        "discount",
        "target_tau",
        "actor_lr",
        "critic_lr",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_num_cycles",
        "dql_eta",
        "dql_bc_weight",
        "dql_q_batch_size",
        "dql_num_inference_steps",
        "dql_target_num_candidates",
        "dql_q_head",
        "dql_q_denominator_floor",
        "dql_clip_actions",
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
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"{args.dataset} does not exist; build it with run_rgb_dp_idql.sh first"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if int(args.num_critics) < 2:
        raise ValueError("Diffusion-QL requires at least two critics")
    if float(args.dql_eta) < 0.0:
        raise ValueError("dql_eta must be non-negative")
    if int(args.dql_q_batch_size) < 0:
        raise ValueError("dql_q_batch_size must be non-negative")
    if int(args.dql_target_num_candidates) <= 0:
        raise ValueError("dql_target_num_candidates must be positive")
    if float(args.dql_q_denominator_floor) <= 0.0:
        raise ValueError("dql_q_denominator_floor must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")

    actor_policy, dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
        verbose=False,
    )
    actor_algo = actor_policy.policy
    initialized_from_ema = initialize_actor_from_deployed_ema(actor_algo)
    if actor_algo.ema is not None and not initialized_from_ema:
        raise RuntimeError("failed to initialize actor from deployed EMA")

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
        and int(args.lr_warmup_steps) >= int(args.lr_total_steps)
    ):
        raise ValueError(
            f"lr_warmup_steps={args.lr_warmup_steps} must be smaller than "
            f"total training steps={args.lr_total_steps}"
        )
    configure_actor_optimizer(
        actor_algo,
        args.actor_lr,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.lr_warmup_steps,
        total_steps=args.lr_total_steps,
        num_cycles=args.lr_num_cycles,
    )

    audit = dataset_audit(
        args.dataset,
        len(dataset),
        expected_task=args.task,
        expected_reward_mode=args.reward_mode,
    )
    args.dataset_reward_mode = str(audit["reward_mode"])
    action_stats = dp_checkpoint.get("action_normalization_stats")
    if action_stats is None:
        raise ValueError("pretrained DP checkpoint has no action normalization stats")
    obs_normalization_stats = copy.deepcopy(actor_policy.obs_normalization_stats)

    critics, critic_targets, unused_value_net = make_rise_value_networks(
        actor_algo,
        hidden_dims=tuple(int(value) for value in args.critic_hidden_dims),
        num_critics=int(args.num_critics),
        critic_group_norm=bool(args.critic_group_norm),
        late_fusion_key=args.critic_late_fusion_key,
    )
    del unused_value_net
    critics = critics.float().to(device)
    critic_targets = critic_targets.float().to(device)
    critic_targets.requires_grad_(False)
    critic_optimizer = torch.optim.Adam(
        critics.parameters(),
        lr=float(args.critic_lr),
    )
    critic_lr_scheduler = make_step_lr_scheduler(
        critic_optimizer,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.lr_warmup_steps,
        total_steps=args.lr_total_steps,
        num_cycles=args.lr_num_cycles,
    )

    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not checkpoint.get("rise_style_rgb_dql", False):
            raise ValueError("resume checkpoint is not from train_rgb_dp_dql.py")
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
        history = list(checkpoint.get("history", []))
        loader_generator.set_state(checkpoint["loader_generator_state"].cpu())
        restore_rng_state(checkpoint.get("rng_state"))
        del checkpoint
        print(
            f"Resumed {args.resume_checkpoint} at epoch={start_epoch} step={global_step}",
            flush=True,
        )

    actor_algo.set_train()
    critics.train()
    critic_targets.train().requires_grad_(False)
    trainability = actor_trainability(actor_algo)
    startup = {
        "task": str(args.task),
        "dataset": audit,
        "data_routing": {
            "shared_loader": True,
            "actor_rows": "all_human_success_failure",
            "critic_rows": "all_human_success_failure",
            "source_masking": False,
        },
        "loader": {
            "class": dataset.__class__.__name__,
            "num_loaders": 1,
            "sampler": "RandomSampler_without_replacement",
            "batch_size": int(args.batch_size),
            "num_batches": int(len(loader)),
            "steps_per_epoch": int(args.steps_per_epoch),
            "steps_per_epoch_source": str(args.steps_per_epoch_source),
            "seed": int(args.seed),
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
            "critic_group_norm": bool(args.critic_group_norm),
            "critic_late_fusion_key": args.critic_late_fusion_key,
        },
        "dql_reference_alignment": dql_reference_alignment(args),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "discount": float(args.discount),
            "target_tau": float(args.target_tau),
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
            "dql_eta": float(args.dql_eta),
            "dql_bc_weight": float(args.dql_bc_weight),
            "dql_q_batch_size": int(args.dql_q_batch_size),
            "dql_num_inference_steps": int(args.dql_num_inference_steps),
            "dql_target_num_candidates": int(args.dql_target_num_candidates),
            "dql_q_head": str(args.dql_q_head),
            "dql_q_normalization": "mean_absolute_dataset_action_q",
            "lr_scheduler": str(args.lr_scheduler),
            "lr_total_steps": int(args.lr_total_steps),
        },
    }
    write_json(args.output_dir / "startup_audit.json", startup)
    print(json.dumps(jsonable(startup), indent=2), flush=True)

    writer = make_tensorboard_writer(args.output_dir)
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        epoch_iterator = iter(loader)
        epoch_records: list[dict[str, float]] = []
        for step_in_epoch in range(1, int(args.steps_per_epoch) + 1):
            try:
                raw_batch = next(epoch_iterator)
            except StopIteration:
                epoch_iterator = iter(loader)
                raw_batch = next(epoch_iterator)
            raw_batch = align_shared_batch_actions(raw_batch)
            learning_rates = {
                "lr/actor": float(
                    actor_algo.optimizers["policy"].param_groups[0]["lr"]
                ),
                "lr/critic": float(critic_optimizer.param_groups[0]["lr"]),
            }

            critic_batch = process_critic_batch(
                raw_batch,
                actor_algo,
                obs_normalization_stats,
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
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                critics.parameters(),
                float(args.max_gradient_norm),
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
            )
            soft_update_targets(critics, critic_targets, float(args.target_tau))
            if critic_lr_scheduler is not None:
                critic_lr_scheduler.step()

            global_step += 1
            metrics = scalar_metrics(
                {
                    **critic_info,
                    "critic/gradient_norm": critic_gradient_norm,
                }
            )
            metrics.update(actor_info)
            metrics.update(learning_rates)
            epoch_records.append(metrics)
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

        epoch_summary = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": mean_metrics(epoch_records),
        }
        history.append(epoch_summary)
        partial_summary = {
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
        write_json(args.output_dir / "partial_summary.json", partial_summary)

        should_save = (
            epoch % int(args.save_every_epochs) == 0
            or epoch == int(args.epochs)
        )
        if should_save:
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
                history=history,
                loader_generator=loader_generator,
            )
            latest_path = args.output_dir / "latest.pt"
            atomic_torch_save(payload, latest_path)
            if (
                int(args.snapshot_every_epochs) > 0
                and epoch % int(args.snapshot_every_epochs) == 0
            ):
                replace_with_hardlink(
                    latest_path,
                    args.output_dir / "models" / f"model_epoch_{epoch}.pt",
                )
            if writer is not None:
                writer.flush()
            print(
                f"Saved {latest_path} at epoch={epoch} step={global_step}",
                flush=True,
            )

    last_path = args.output_dir / "last.pt"
    if int(args.epochs) > start_epoch:
        replace_with_hardlink(args.output_dir / "latest.pt", last_path)
    elif not last_path.exists():
        if args.resume_checkpoint is None:
            raise RuntimeError("training is complete but last.pt is missing")
        replace_with_hardlink(args.resume_checkpoint, last_path)
    final_summary = json.loads((args.output_dir / "partial_summary.json").read_text())
    final_summary["complete"] = True
    final_summary["last_checkpoint"] = str(last_path)
    write_json(args.output_dir / "summary.json", final_summary)
    if writer is not None:
        writer.close()
    close_dataset = getattr(dataset, "close", None)
    if callable(close_dataset):
        close_dataset()
    print(json.dumps(jsonable({k: v for k, v in final_summary.items() if k != "history"}), indent=2))
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("square", "can", "transport", "tool_hang"), default="square")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hdf5-cache-mode", choices=("low_dim", "none"), default="low_dim")
    parser.add_argument(
        "--reward-mode",
        choices=tuple(REWARD_DEFINITIONS),
        default="task",
        help="Expected dataset reward mode; task is the default.",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-num-cycles", type=float, default=0.5)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=(300, 400, 300))
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--critic-group-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--critic-late-fusion-key",
        default="robot0_gripper_qpos",
        help="One observation key or a comma-separated key list.",
    )
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-gradient-norm", type=float, default=10.0)
    parser.add_argument("--dql-eta", type=float, default=1.0)
    parser.add_argument("--dql-bc-weight", type=float, default=1.0)
    parser.add_argument("--dql-q-batch-size", type=int, default=8)
    parser.add_argument("--dql-num-inference-steps", type=int, default=5)
    parser.add_argument("--dql-target-num-candidates", type=int, default=1)
    parser.add_argument("--dql-q-head", choices=("random", "q1", "q2", "min"), default="random")
    parser.add_argument("--dql-q-denominator-floor", type=float, default=1e-6)
    parser.add_argument("--dql-clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--snapshot-every-epochs", type=int, default=10)
    args = parser.parse_args()
    for key in ("dataset", "checkpoint", "output_dir", "resume_checkpoint"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    if not args.critic_late_fusion_key:
        args.critic_late_fusion_key = None
    if args.lr_warmup_steps < 0:
        parser.error("lr-warmup-steps must be non-negative")
    if args.lr_num_cycles <= 0.0:
        parser.error("lr-num-cycles must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
