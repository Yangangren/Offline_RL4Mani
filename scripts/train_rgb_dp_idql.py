#!/usr/bin/env python3
"""RISE-style RGB Diffusion Policy + one-step IDQL post-training.

This is intentionally a single-dataset, single-loader implementation. Every
sampled batch updates the full pretrained diffusion actor with chunk BC and
updates independent raw-observation Q1, Q2, and V networks with the one-step
IQL equations used by RISE's ``robomimic/algo/idql.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo.diffusion_policy import replace_bn_with_gn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "datasets/square/idql/square_rgb_dp_idql_200demo_100success_94failure.hdf5"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth"
)
DEFAULT_OUTPUT = (
    ROOT
    / "trained_models/square_rgb_dp_idql_rise/200demo_100success_94failure"
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2))


def make_tensorboard_writer(output_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(output_dir / "tb"))
    except Exception as exc:
        print(f"TensorBoard disabled: {exc}", flush=True)
        return None


def initialize_actor_from_deployed_ema(actor_algo) -> bool:
    """Make the trainable actor exactly equal to the deployed EMA policy."""
    if actor_algo.ema is None:
        return False
    deployed_state = copy.deepcopy(actor_algo.ema.averaged_model.state_dict())
    actor_algo.nets.load_state_dict(deployed_state)
    actor_algo.ema.averaged_model.load_state_dict(actor_algo.nets.state_dict())
    actor_algo.ema.optimization_step = 0
    if hasattr(actor_algo, "_refresh_ema_parameter_views"):
        actor_algo._refresh_ema_parameter_views()
    return True


def make_step_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float,
):
    """Create the per-batch cosine scheduler used by the default RGB DP."""
    if scheduler_type == "constant":
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"unsupported lr scheduler: {scheduler_type}")

    def lr_multiplier(current_step: int) -> float:
        if current_step < int(warmup_steps):
            return float(current_step) / float(max(1, int(warmup_steps)))
        progress = float(current_step - int(warmup_steps)) / float(
            max(1, int(total_steps) - int(warmup_steps))
        )
        return max(
            0.0,
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * float(num_cycles) * 2.0 * progress
                )
            ),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def configure_actor_optimizer(
    actor_algo,
    learning_rate: float,
    *,
    scheduler_type: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 1,
    num_cycles: float = 0.5,
) -> None:
    for parameter in actor_algo.nets["policy"].parameters():
        parameter.requires_grad_(True)
    actor_algo.optimizers["policy"] = torch.optim.Adam(
        actor_algo.nets["policy"].parameters(),
        lr=float(learning_rate),
    )
    actor_algo.lr_schedulers["policy"] = make_step_lr_scheduler(
        actor_algo.optimizers["policy"],
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        num_cycles=num_cycles,
    )
    actor_algo.step_lr_schedulers_every_batch["policy"] = (
        actor_algo.lr_schedulers["policy"] is not None
    )


def actor_trainability(actor_algo) -> dict[str, Any]:
    policy = actor_algo.nets["policy"]
    parameters = list(policy.parameters())
    optimizer_ids = {
        id(parameter)
        for group in actor_algo.optimizers["policy"].param_groups
        for parameter in group["params"]
    }
    result = {
        "num_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "num_trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "all_trainable": all(parameter.requires_grad for parameter in parameters),
        "all_in_optimizer": all(id(parameter) in optimizer_ids for parameter in parameters),
        "obs_encoder_trainable": all(
            parameter.requires_grad
            for parameter in policy["obs_encoder"].parameters()
        ),
    }
    if not result["all_trainable"] or not result["all_in_optimizer"]:
        raise RuntimeError(f"full DP actor is not trainable: {result}")
    return result


class RiseLateFusionMLP(nn.Module):
    """RISE critic MLP with gripper-state fusion at module index 1."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        late_fusion_dim: int = 0,
    ):
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("RISE late-fusion critic requires at least two hidden layers")
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        intermediate_dims = hidden_dims[:-1]
        for index, hidden_dim in enumerate(intermediate_dims):
            if late_fusion_dim > 0 and index == 1:
                current_dim += int(late_fusion_dim)
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, int(hidden_dims[-1])))
        self.layers = nn.ModuleList(layers)
        self.late_fusion_dim = int(late_fusion_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        late_fusion_input: torch.Tensor | None,
    ) -> torch.Tensor:
        output = inputs
        for module_index, layer in enumerate(self.layers):
            # This placement reproduces RISE's modified base_nets.MLP: with
            # late_fusion_layer_index=1, fusion occurs after the first Linear
            # and immediately before its ReLU.
            if self.late_fusion_dim > 0 and module_index == 1:
                if late_fusion_input is None:
                    raise ValueError("late-fusion critic is missing its fusion input")
                output = torch.cat((output, late_fusion_input), dim=-1)
            output = layer(output)
        return output


class RiseValueNetwork(nn.Module):
    """Raw-observation scalar value network used by RISE-style IDQL."""

    def __init__(
        self,
        obs_shapes: OrderedDict,
        hidden_dims: tuple[int, ...],
        goal_shapes: OrderedDict,
        encoder_kwargs: dict,
        late_fusion_key: str | None,
        action_dim: int | None = None,
    ):
        super().__init__()
        encoded_obs_shapes = OrderedDict(obs_shapes)
        if action_dim is not None:
            encoded_obs_shapes["action"] = (int(action_dim),)
        observation_group_shapes = OrderedDict(obs=encoded_obs_shapes)
        if goal_shapes is not None and len(goal_shapes) > 0:
            observation_group_shapes["goal"] = OrderedDict(goal_shapes)

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.late_fusion_keys = tuple(
            key.strip()
            for key in str(late_fusion_key or "").split(",")
            if key.strip()
        )
        late_fusion_dim = 0
        for key in self.late_fusion_keys:
            if key not in obs_shapes:
                raise KeyError(
                    f"late_fusion_key={key} is absent from obs_shapes"
                )
            late_fusion_dim += int(np.prod(obs_shapes[key]))
        self.nets["mlp"] = RiseLateFusionMLP(
            input_dim=int(self.nets["encoder"].output_shape()[0]),
            hidden_dims=hidden_dims,
            late_fusion_dim=late_fusion_dim,
        )
        self.nets["decoder"] = nn.Linear(int(hidden_dims[-1]), 1)
        self.action_dim = action_dim
        self.has_goal = "goal" in observation_group_shapes

    def _forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        goal_dict: dict[str, torch.Tensor] | None,
        acts: torch.Tensor | None,
    ) -> torch.Tensor:
        encoder_obs = dict(obs_dict)
        if self.action_dim is not None:
            if acts is None:
                raise ValueError("action-value network requires actions")
            encoder_obs["action"] = acts
        encoder_inputs = {"obs": encoder_obs}
        if self.has_goal:
            if goal_dict is None:
                raise ValueError("goal-conditioned value network requires goal observations")
            encoder_inputs["goal"] = goal_dict
        encoded = self.nets["encoder"](**encoder_inputs)
        late_fusion_parts = [
            obs_dict[key].flatten(start_dim=1)
            for key in self.late_fusion_keys
        ]
        late_fusion = (
            torch.cat(late_fusion_parts, dim=-1)
            if late_fusion_parts
            else None
        )
        features = self.nets["mlp"](encoded, late_fusion)
        return self.nets["decoder"](features)

    def forward(self, obs_dict, goal_dict=None):
        return self._forward(obs_dict, goal_dict, acts=None)


class RiseActionValueNetwork(RiseValueNetwork):
    def forward(self, obs_dict, acts, goal_dict=None):
        return self._forward(obs_dict, goal_dict, acts=acts)


def action_normalization_stats_match(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    for action_key in left:
        if set(left[action_key]) != set(right[action_key]):
            return False
        for stat_key in left[action_key]:
            if not np.allclose(
                np.asarray(left[action_key][stat_key]),
                np.asarray(right[action_key][stat_key]),
            ):
                return False
    return True


def build_single_loader(
    args: argparse.Namespace,
    actor_policy,
    dp_checkpoint: dict,
    sequence_length: int | None = None,
):
    actor_algo = actor_policy.policy
    config, _ = FileUtils.config_from_checkpoint(
        ckpt_dict=dp_checkpoint,
        verbose=False,
    )
    ObsUtils.initialize_obs_utils_with_config(config)
    observation_horizon = int(actor_algo.algo_config.horizon.observation_horizon)
    prediction_horizon = int(actor_algo.algo_config.horizon.prediction_horizon)

    with config.values_unlocked():
        config.train.data = [{"path": str(args.dataset)}]
        config.train.normalize_weights_by_ds_size = False
        config.train.hdf5_cache_mode = args.hdf5_cache_mode
        config.train.hdf5_load_next_obs = True
        # Never derive normalization from the mixed dataset. Both actor and
        # critic use the pretrained DP checkpoint's observation/action spaces.
        config.train.hdf5_normalize_obs = False
        config.train.seq_length = (
            prediction_horizon
            if sequence_length is None
            else int(sequence_length)
        )
        config.train.frame_stack = observation_horizon
        config.train.pad_seq_length = True
        config.train.pad_frame_stack = True
        config.train.batch_size = int(args.batch_size)
        config.train.num_data_workers = int(args.num_workers)
        config.train.dataset_keys = list(
            dict.fromkeys(list(config.train.dataset_keys) + ["actions", "rewards", "dones"])
        )

    dataset = TrainUtils.dataset_factory(
        config,
        obs_keys=list(actor_algo.obs_shapes.keys()),
    )
    if dataset.__class__.__name__ != "SequenceDataset":
        raise TypeError(
            "RISE-style training requires one SequenceDataset, not a MetaDataset; "
            f"got {dataset.__class__.__name__}"
        )
    action_stats = dp_checkpoint.get("action_normalization_stats")
    if action_stats is None:
        raise ValueError(
            "pretrained DP checkpoint has no action_normalization_stats; refusing "
            "to use statistics computed from the mixed dataset"
        )
    if actor_policy.action_normalization_stats is None or not action_normalization_stats_match(
        actor_policy.action_normalization_stats,
        action_stats,
    ):
        raise RuntimeError(
            "loaded RolloutPolicy action normalization does not match its DP checkpoint"
        )
    dataset.set_action_normalization_stats(copy.deepcopy(action_stats))
    if not action_normalization_stats_match(
        dataset.get_action_normalization_stats(),
        action_stats,
    ):
        raise RuntimeError("failed to install pretrained DP action normalization")

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader_kwargs: dict[str, Any] = {}
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        sampler=None,
        drop_last=len(dataset) >= int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and actor_algo.device.type == "cuda"),
        generator=generator,
        **loader_kwargs,
    )
    return dataset, loader, generator, config


def make_rise_value_networks(
    actor_algo,
    hidden_dims: tuple[int, ...],
    num_critics: int = 2,
    critic_group_norm: bool = False,
    late_fusion_key: str | None = "robot0_gripper_qpos",
) -> tuple[nn.ModuleList, nn.ModuleList, nn.Module]:
    """Construct the independent Q ensemble, target Qs, and V from RISE IDQL."""
    encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
        actor_algo.obs_config.encoder
    )
    critics = nn.ModuleList()
    for _ in range(int(num_critics)):
        critic = RiseActionValueNetwork(
            obs_shapes=actor_algo.obs_shapes,
            action_dim=int(actor_algo.ac_dim),
            hidden_dims=hidden_dims,
            goal_shapes=actor_algo.goal_shapes,
            encoder_kwargs=copy.deepcopy(encoder_kwargs),
            late_fusion_key=late_fusion_key,
        )
        if critic_group_norm:
            critic = replace_bn_with_gn(critic)
        critics.append(critic)
    critic_targets = copy.deepcopy(critics)
    vf = RiseValueNetwork(
        obs_shapes=actor_algo.obs_shapes,
        hidden_dims=hidden_dims,
        goal_shapes=actor_algo.goal_shapes,
        encoder_kwargs=copy.deepcopy(encoder_kwargs),
        late_fusion_key=late_fusion_key,
    )
    if critic_group_norm:
        vf = replace_bn_with_gn(vf)
    return critics, critic_targets, vf


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def rise_reference_alignment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "matched": [
            "one uniformly shuffled mixed SequenceDataset",
            "expert_transition_reward_1_non_expert_transition_reward_0",
            "independent_raw_observation_Q1_Q2_target_Q1_target_Q2_V",
            "one_step_IQL_Q_and_expectile_V_equations",
            "Q_then_target_soft_update_then_V_update_order",
            "pure_diffusion_BC_actor_objective",
            "task_gripper_late_fusion_value_heads",
        ],
        "post_deployment_adaptations": [
            "actor_initialized_from_pretrained_DP_deployed_EMA",
            "actor_keeps_pretrained_DP_native_sequence_alignment_and_scheduler",
            "local_success_and_failure_rollouts_replace_unreleased_RISE_play_data",
            "RISE_nearby_state_and_action_augmentation_not_applied",
        ],
        "critic_group_norm_compatibility_override": bool(args.critic_group_norm),
    }


def process_critic_batch(
    raw_batch: dict,
    actor_algo,
    obs_normalization_stats,
) -> dict:
    """Extract RISE's one-step tuple from the shared sequence batch."""
    current_index = int(actor_algo.algo_config.horizon.observation_horizon) - 1
    critic_batch = {
        "obs": {
            key: raw_batch["obs"][key][:, current_index]
            for key in actor_algo.obs_shapes
        },
        "next_obs": {
            key: raw_batch["next_obs"][key][:, current_index]
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
    if not torch.isfinite(actions).all():
        raise ValueError("critic actions contain non-finite values")
    action_min = actions.min().item()
    action_max = actions.max().item()
    tolerance = 1e-3
    if action_min < -1.0 - tolerance or action_max > 1.0 + tolerance:
        raise ValueError(
            "critic actions are not in the pretrained DP normalized action space: "
            f"min={action_min:.6f}, max={action_max:.6f}"
        )
    if action_min < -1.0 or action_max > 1.0:
        critic_batch["actions"] = actions.clamp(-1.0, 1.0)
    return critic_batch


def align_shared_batch_actions(raw_batch: dict) -> dict:
    """Validate once and give actor and critic the identical clipped tensor."""
    actions = raw_batch["actions"]
    if not torch.isfinite(actions).all():
        raise ValueError("shared actor/critic actions contain non-finite values")
    action_min = actions.min().item()
    action_max = actions.max().item()
    tolerance = 1e-3
    if action_min < -1.0 - tolerance or action_max > 1.0 + tolerance:
        raise ValueError(
            "shared actions are outside the pretrained DP normalized action space: "
            f"min={action_min:.6f}, max={action_max:.6f}"
        )
    if action_min < -1.0 or action_max > 1.0:
        raw_batch = dict(raw_batch)
        raw_batch["actions"] = actions.clamp(-1.0, 1.0)
    return raw_batch


def compute_critic_losses(
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    vf: nn.Module,
    batch: dict,
    *,
    discount: float,
    expectile: float,
    use_huber: bool,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Port of RISE IDQL._compute_critic_loss for the one-step case."""
    obs = batch["obs"]
    actions = batch["actions"]
    next_obs = batch["next_obs"]
    goal_obs = batch["goal_obs"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    pred_qs = [
        critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
        for critic in critics
    ]
    target_vf_pred = vf(obs_dict=next_obs, goal_dict=goal_obs).detach()
    q_target = (
        rewards + (1.0 - dones) * float(discount) * target_vf_pred
    ).detach()

    loss_function: nn.Module
    loss_function = nn.SmoothL1Loss() if use_huber else nn.MSELoss()
    critic_losses = [loss_function(q_pred, q_target) for q_pred in pred_qs]

    target_qs = [
        critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
        for critic in critic_targets
    ]
    target_q_min = torch.cat(target_qs, dim=1).min(dim=1, keepdim=True).values.detach()
    vf_pred = vf(obs_dict=obs, goal_dict=goal_obs)
    vf_error = vf_pred - target_q_min
    vf_weight = torch.where(
        vf_error > 0.0,
        1.0 - float(expectile),
        float(expectile),
    )
    vf_loss = (vf_weight * vf_error.square()).mean()

    info = {
        **{
            f"critic/q{index + 1}_loss": loss
            for index, loss in enumerate(critic_losses)
        },
        **{
            f"critic/q{index + 1}_mean": prediction.mean()
            for index, prediction in enumerate(pred_qs)
        },
        "critic/q_target_mean": q_target.mean(),
        "critic/target_v_mean": target_vf_pred.mean(),
        "vf/loss": vf_loss,
        "vf/value_mean": vf_pred.mean(),
        "vf/target_q_min_mean": target_q_min.mean(),
        "vf/error_mean": vf_error.mean(),
        "data/reward_mean": rewards.mean(),
        "data/done_fraction": dones.mean(),
        "data/action_abs_mean": actions.abs().mean(),
        "data/action_min": actions.min(),
        "data/action_max": actions.max(),
    }
    return critic_losses, vf_loss, info


def update_critics(
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    vf: nn.Module,
    critic_optimizers: list[torch.optim.Optimizer],
    vf_optimizer: torch.optim.Optimizer,
    critic_losses: list[torch.Tensor],
    vf_loss: torch.Tensor,
    *,
    target_tau: float,
    max_gradient_norm: float | None,
) -> None:
    """Port of RISE IDQL._update_critic, including update order."""
    for critic_loss, critic, critic_target, optimizer in zip(
        critic_losses,
        critics,
        critic_targets,
        critic_optimizers,
    ):
        TorchUtils.backprop_for_loss(
            net=critic,
            optim=optimizer,
            loss=critic_loss,
            max_grad_norm=max_gradient_norm,
            retain_graph=False,
        )
        with torch.no_grad():
            TorchUtils.soft_update(
                source=critic,
                target=critic_target,
                tau=float(target_tau),
            )
    TorchUtils.backprop_for_loss(
        net=vf,
        optim=vf_optimizer,
        loss=vf_loss,
        max_grad_norm=max_gradient_norm,
        retain_graph=False,
    )


def actor_train_step(
    actor_algo,
    raw_batch: dict,
    epoch: int,
    obs_normalization_stats,
) -> dict[str, float]:
    actor_batch = actor_algo.process_batch_for_training(raw_batch)
    actor_batch = actor_algo.postprocess_batch_for_training(
        actor_batch,
        obs_normalization_stats=obs_normalization_stats,
    )
    info = actor_algo.train_on_batch(actor_batch, epoch=epoch, validate=False)
    actor_algo.on_gradient_step()
    log = actor_algo.log_info(info)
    return {
        f"actor/{key}" if not str(key).startswith("actor/") else str(key): float(value)
        for key, value in log.items()
        if isinstance(value, (int, float, np.number))
    }


def dataset_audit(
    dataset_path: Path,
    dataset_size: int,
    expected_task: str | None = None,
) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as handle:
        reward_definition = str(handle.attrs.get("reward_definition", ""))
        if reward_definition != "expert_transition=1; non_expert_transition=0":
            raise ValueError(
                "dataset does not use the RISE reward definition: "
                f"{reward_definition!r}"
            )
        masks = {
            key: int(len(value))
            for key, value in handle.get("mask", {}).items()
        }
        hdf5_total = int(handle["data"].attrs.get("total", -1))
        dataset_task = str(handle.attrs.get("task", "")) or None
    if expected_task is not None and dataset_task is not None and dataset_task != expected_task:
        raise ValueError(
            f"dataset task={dataset_task} does not match requested task={expected_task}"
        )
    summary_path = dataset_path.with_suffix(".summary.json")
    builder_summary = (
        json.loads(summary_path.read_text()) if summary_path.is_file() else None
    )
    if hdf5_total >= 0 and int(dataset_size) != hdf5_total:
        raise ValueError(
            f"SequenceDataset has {dataset_size} indices but HDF5 reports {hdf5_total} transitions"
        )
    return {
        "path": str(dataset_path),
        "task": dataset_task,
        "sequence_dataset_size": int(dataset_size),
        "hdf5_total_transitions": hdf5_total,
        "masks": masks,
        "reward_definition": reward_definition,
        "builder_summary": builder_summary,
    }


def scalar_metrics(values: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            output[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float, np.number)):
            output[key] = float(value)
    return output


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record})
    return {
        key: float(np.mean([record[key] for record in records if key in record]))
        for key in keys
    }


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    actor_algo,
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    vf: nn.Module,
    critic_optimizers: list[torch.optim.Optimizer],
    vf_optimizer: torch.optim.Optimizer,
    critic_lr_schedulers: list[Any],
    vf_lr_scheduler: Any,
    action_normalization_stats: dict,
    epoch: int,
    global_step: int,
    history: list[dict],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "rise_style_rgb_idql": True,
        "hybrid_dp_chunk_actor_iql": True,
        "visual_critic_idql": True,
        "actor_model": actor_algo.serialize(),
        "critics": [critic.state_dict() for critic in critics],
        "critic_targets": [critic.state_dict() for critic in critic_targets],
        "vf": vf.state_dict(),
        "critic_optimizers": [optimizer.state_dict() for optimizer in critic_optimizers],
        "vf_optimizer": vf_optimizer.state_dict(),
        "critic_lr_schedulers": [
            scheduler.state_dict() if scheduler is not None else None
            for scheduler in critic_lr_schedulers
        ],
        "vf_lr_scheduler": (
            vf_lr_scheduler.state_dict() if vf_lr_scheduler is not None else None
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
        "reward_definition": "expert_transition=1; non_expert_transition=0",
        "actor_training_objective": "diffusion_bc_full_chunk",
        "actor_source_mask": "none_all_shared_batch_rows",
        "actor_data_mode": "all_human_success_failure_rows",
        "critic_training_objective": "rise_one_step_iql",
        "critic_input_mode": "independent_raw_observation_encoders",
        "critic_action_space": "pretrained_dp_normalized_action_space",
        "critic_horizon": 1,
        "latent_dynamics": False,
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
        "expectile": float(args.expectile),
        "target_tau": float(args.target_tau),
        "actor_initialized_from_deployed_ema": True,
        "actor_encoder_trainable": True,
        "rise_reference_alignment": rise_reference_alignment(args),
        "actor_ema_optimization_step": int(
            actor_algo.ema.optimization_step if actor_algo.ema is not None else 0
        ),
        "rng_state": rng_state(),
        "loader_generator_state": loader_generator.get_state(),
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def replace_with_hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, target)


def validate_resume_args(args: argparse.Namespace, checkpoint: dict) -> None:
    previous = checkpoint.get("args", {})
    exact_keys = (
        "dataset",
        "checkpoint",
        "task",
        "seed",
        "batch_size",
        "steps_per_epoch",
        "critic_hidden_dims",
        "num_critics",
        "critic_group_norm",
        "critic_late_fusion_key",
        "discount",
        "expectile",
        "target_tau",
        "actor_lr",
        "critic_lr",
        "vf_lr",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_num_cycles",
        "lr_total_steps",
        "use_huber",
        "max_gradient_norm",
    )
    for key in exact_keys:
        if key not in previous:
            continue
        old = jsonable(previous[key])
        new = jsonable(getattr(args, key))
        if old != new:
            raise ValueError(f"resume argument mismatch for {key}: checkpoint={old}, current={new}")


def train(args: argparse.Namespace) -> dict:
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"{args.dataset} does not exist; run scripts/build_rgb_dp_idql_dataset.py first"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0.5 <= float(args.expectile) < 1.0:
        raise ValueError("expectile must be in [0.5, 1.0)")
    if int(args.num_critics) < 2:
        raise ValueError("RISE-style clipped double Q requires at least two critics")
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

    dataset, loader, loader_generator, config = build_single_loader(
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
    if args.steps_per_epoch <= 0:
        raise ValueError(
            f"effective steps_per_epoch must be positive, got {args.steps_per_epoch}"
        )
    args.lr_total_steps = int(args.epochs) * int(args.steps_per_epoch)
    if (
        args.lr_scheduler == "cosine"
        and int(args.lr_warmup_steps) >= int(args.lr_total_steps)
    ):
        raise ValueError(
            f"lr_warmup_steps={args.lr_warmup_steps} must be smaller than "
            f"the {args.lr_total_steps} total training steps"
        )
    configure_actor_optimizer(
        actor_algo,
        args.actor_lr,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.lr_warmup_steps,
        total_steps=args.lr_total_steps,
        num_cycles=args.lr_num_cycles,
    )
    audit = dataset_audit(args.dataset, len(dataset), expected_task=args.task)
    action_stats = dp_checkpoint["action_normalization_stats"]
    obs_normalization_stats = copy.deepcopy(actor_policy.obs_normalization_stats)

    critics, critic_targets, vf = make_rise_value_networks(
        actor_algo,
        hidden_dims=tuple(int(value) for value in args.critic_hidden_dims),
        num_critics=args.num_critics,
        critic_group_norm=args.critic_group_norm,
        late_fusion_key=args.critic_late_fusion_key,
    )
    critics = critics.float().to(device)
    critic_targets = critic_targets.float().to(device)
    vf = vf.float().to(device)
    critic_targets.requires_grad_(False)
    critic_optimizers = [
        torch.optim.Adam(critic.parameters(), lr=float(args.critic_lr))
        for critic in critics
    ]
    vf_optimizer = torch.optim.Adam(vf.parameters(), lr=float(args.vf_lr))
    critic_lr_schedulers = [
        make_step_lr_scheduler(
            optimizer,
            scheduler_type=args.lr_scheduler,
            warmup_steps=args.lr_warmup_steps,
            total_steps=args.lr_total_steps,
            num_cycles=args.lr_num_cycles,
        )
        for optimizer in critic_optimizers
    ]
    vf_lr_scheduler = make_step_lr_scheduler(
        vf_optimizer,
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
        if not checkpoint.get("rise_style_rgb_idql", False):
            raise ValueError("resume checkpoint is not from train_rgb_dp_idql.py")
        validate_resume_args(args, checkpoint)
        checkpoint_action_stats = checkpoint.get("action_normalization_stats")
        if checkpoint_action_stats is None or not action_normalization_stats_match(
            checkpoint_action_stats,
            action_stats,
        ):
            raise ValueError(
                "resume checkpoint action normalization does not match the "
                "pretrained DP checkpoint"
            )
        expected_state_counts = {
            "critics": len(critics),
            "critic_targets": len(critic_targets),
            "critic_optimizers": len(critic_optimizers),
            "critic_lr_schedulers": len(critic_lr_schedulers),
        }
        for key, expected_count in expected_state_counts.items():
            states = checkpoint.get(key)
            if not isinstance(states, (list, tuple)) or len(states) != expected_count:
                actual_count = len(states) if isinstance(states, (list, tuple)) else None
                raise ValueError(
                    f"resume checkpoint has {actual_count} {key} states; "
                    f"expected {expected_count}"
                )
        scheduler_enabled = args.lr_scheduler != "constant"
        actor_scheduler_state = (
            checkpoint.get("actor_model", {})
            .get("lr_schedulers", {})
            .get("policy")
        )
        if (actor_scheduler_state is not None) != scheduler_enabled:
            raise ValueError(
                "resume checkpoint actor LR scheduler state does not match "
                f"lr_scheduler={args.lr_scheduler}"
            )
        checkpoint_vf_scheduler = checkpoint.get("vf_lr_scheduler")
        if (checkpoint_vf_scheduler is not None) != scheduler_enabled:
            raise ValueError(
                "resume checkpoint V LR scheduler state does not match "
                f"lr_scheduler={args.lr_scheduler}"
            )
        actor_algo.deserialize(checkpoint["actor_model"], load_optimizers=True)
        actor_algo.step_lr_schedulers_every_batch["policy"] = scheduler_enabled
        for critic, state in zip(critics, checkpoint["critics"]):
            critic.load_state_dict(state)
        for critic_target, state in zip(critic_targets, checkpoint["critic_targets"]):
            critic_target.load_state_dict(state)
        vf.load_state_dict(checkpoint["vf"])
        for optimizer, state in zip(critic_optimizers, checkpoint["critic_optimizers"]):
            optimizer.load_state_dict(state)
        vf_optimizer.load_state_dict(checkpoint["vf_optimizer"])
        for scheduler, state in zip(
            critic_lr_schedulers,
            checkpoint["critic_lr_schedulers"],
        ):
            if (state is not None) != scheduler_enabled:
                raise ValueError(
                    "resume checkpoint critic LR scheduler state does not match "
                    f"lr_scheduler={args.lr_scheduler}"
                )
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if vf_lr_scheduler is not None:
            vf_lr_scheduler.load_state_dict(checkpoint_vf_scheduler)
        if actor_algo.ema is not None:
            actor_algo.ema.optimization_step = int(
                checkpoint.get("actor_ema_optimization_step", 0)
            )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["step"])
        if scheduler_enabled:
            scheduler_steps = [
                int(actor_algo.lr_schedulers["policy"].last_epoch),
                *[
                    int(scheduler.last_epoch)
                    for scheduler in critic_lr_schedulers
                    if scheduler is not None
                ],
                int(vf_lr_scheduler.last_epoch),
            ]
            if any(step != global_step for step in scheduler_steps):
                raise ValueError(
                    f"LR scheduler steps {scheduler_steps} do not match "
                    f"checkpoint global_step={global_step}"
                )
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
    critic_targets.train()
    critic_targets.requires_grad_(False)
    vf.train()
    trainability = actor_trainability(actor_algo)
    architecture = {
        "actor": trainability,
        "critic_parameter_counts": [parameter_count(critic) for critic in critics],
        "target_critic_parameter_counts": [
            parameter_count(critic) for critic in critic_targets
        ],
        "vf_parameter_count": parameter_count(vf),
        "independent_raw_obs_encoders": True,
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
    }
    startup = {
        "task": str(args.task),
        "dataset": audit,
        "data_routing": {
            "shared_loader": True,
            "actor_rows": "all_human_success_failure_no_mask",
            "critic_rows": "all_human_success_failure",
            "actor_filtered_rows": 0,
        },
        "loader": {
            "class": dataset.__class__.__name__,
            "num_loaders": 1,
            "sampler": "RandomSampler_without_replacement",
            "balanced_sampling": False,
            "batch_size": int(args.batch_size),
            "num_batches": int(len(loader)),
            "steps_per_epoch": int(args.steps_per_epoch),
            "steps_per_epoch_source": str(args.steps_per_epoch_source),
            "seed": int(args.seed),
            "shuffle_each_epoch": True,
            "without_replacement_within_each_permutation": True,
            "partial_epoch_behavior": (
                "consume the first steps_per_epoch batches from a fresh "
                "deterministic random permutation"
            ),
            "automatic_steps_formula": (
                "floor(sequence_dataset_size / batch_size) because drop_last=True"
                if len(dataset) >= int(args.batch_size)
                else "1 because drop_last=False for a dataset smaller than one batch"
            ),
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
        "architecture": architecture,
        "rise_reference_alignment": rise_reference_alignment(args),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "discount": float(args.discount),
            "expectile": float(args.expectile),
            "target_tau": float(args.target_tau),
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
            "vf_lr": float(args.vf_lr),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "lr_num_cycles": float(args.lr_num_cycles),
            "lr_total_steps": int(args.lr_total_steps),
            "lr_scheduler_step_unit": "optimizer_update",
        },
        "horizons": {
            "observation": int(actor_algo.algo_config.horizon.observation_horizon),
            "action": int(actor_algo.algo_config.horizon.action_horizon),
            "prediction": int(actor_algo.algo_config.horizon.prediction_horizon),
            "critic": 1,
        },
        "latent_dynamics": {
            "enabled": False,
            "loss_weight": 0.0,
            "target_encoder": False,
        },
    }
    write_json(args.output_dir / "training_config.json", startup)
    print(json.dumps(jsonable(startup), indent=2), flush=True)

    writer = make_tensorboard_writer(args.output_dir)
    max_gradient_norm = (
        None
        if args.max_gradient_norm is None or float(args.max_gradient_norm) <= 0.0
        else float(args.max_gradient_norm)
    )

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
            learning_rates_used = {
                "actor": float(
                    actor_algo.optimizers["policy"].param_groups[0]["lr"]
                ),
                "critic": float(critic_optimizers[0].param_groups[0]["lr"]),
                "vf": float(vf_optimizer.param_groups[0]["lr"]),
            }

            critic_batch = process_critic_batch(
                raw_batch,
                actor_algo,
                obs_normalization_stats,
            )
            critic_losses, vf_loss, critic_info = compute_critic_losses(
                critics,
                critic_targets,
                vf,
                critic_batch,
                discount=args.discount,
                expectile=args.expectile,
                use_huber=args.use_huber,
            )
            update_critics(
                critics,
                critic_targets,
                vf,
                critic_optimizers,
                vf_optimizer,
                critic_losses,
                vf_loss,
                target_tau=args.target_tau,
                max_gradient_norm=max_gradient_norm,
            )
            del critic_batch, critic_losses, vf_loss

            actor_info = actor_train_step(
                actor_algo,
                raw_batch,
                epoch,
                obs_normalization_stats,
            )
            for scheduler in critic_lr_schedulers:
                if scheduler is not None:
                    scheduler.step()
            if vf_lr_scheduler is not None:
                vf_lr_scheduler.step()
            actor_rows = int(raw_batch["actions"].shape[0])
            actor_info.update(
                {
                    "actor/data_rows": float(actor_rows),
                    "actor/filtered_rows": 0.0,
                    "actor/data_fraction": 1.0,
                    "actor/source_mask_applied": 0.0,
                }
            )
            global_step += 1
            metrics = scalar_metrics(critic_info)
            metrics.update(actor_info)
            metrics["lr/actor"] = learning_rates_used["actor"]
            metrics["lr/critic"] = learning_rates_used["critic"]
            metrics["lr/vf"] = learning_rates_used["vf"]
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
                vf=vf,
                critic_optimizers=critic_optimizers,
                vf_optimizer=vf_optimizer,
                critic_lr_schedulers=critic_lr_schedulers,
                vf_lr_scheduler=vf_lr_scheduler,
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
                snapshot = args.output_dir / "models" / f"model_epoch_{epoch}.pt"
                replace_with_hardlink(latest_path, snapshot)
            if writer is not None:
                writer.flush()
            print(f"Saved {latest_path} at epoch={epoch} step={global_step}", flush=True)

    last_path = args.output_dir / "last.pt"
    if int(args.epochs) > start_epoch:
        latest_path = args.output_dir / "latest.pt"
        replace_with_hardlink(latest_path, last_path)
    elif not last_path.exists():
        if args.resume_checkpoint is None:
            raise RuntimeError("training was already complete but last.pt is missing")
        replace_with_hardlink(args.resume_checkpoint, last_path)
    final_summary = json.loads((args.output_dir / "partial_summary.json").read_text())
    final_summary["complete"] = True
    final_summary["checkpoints"]["last"] = str(last_path)
    write_json(args.output_dir / "summary.json", final_summary)
    if writer is not None:
        writer.close()
    print(f"Training complete: {last_path}", flush=True)
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        choices=("square", "can", "transport", "tool_hang"),
        default="square",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Defaults to len(train_loader), i.e. one full uniformly shuffled dataset pass.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hdf5-cache-mode",
        choices=("low_dim", "none"),
        default="low_dim",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--reward-mode",
        choices=tuple(REWARD_DEFINITIONS),
        default="task",
        help="Expected dataset reward mode; task is the default.",
    )
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--vf-lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-num-cycles", type=float, default=0.5)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=(300, 400, 300))
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument(
        "--critic-group-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--critic-late-fusion-key",
        default="robot0_gripper_qpos",
        help="One observation key or a comma-separated key list for critic late fusion.",
    )
    parser.add_argument("--use-huber", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-gradient-norm", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--snapshot-every-epochs", type=int, default=0)
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
    if args.save_every_epochs <= 0:
        parser.error("save-every-epochs must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
