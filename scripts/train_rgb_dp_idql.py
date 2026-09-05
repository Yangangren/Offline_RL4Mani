#!/usr/bin/env python3
"""Temporal RGB Diffusion Policy + one-step IDQL post-training.

Every mixed human / rollout batch continues training the original unconditioned
Diffusion Policy actor with full-horizon diffusion BC and trains independent
two-frame Q1, Q2, and V networks with one-step IQL. An optional one-step visual
latent prediction loss can regularize each online Q encoder; it predicts only
the immediate successor observation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import (
    SparseChunkSequenceDataset,
    SparseDQLSequenceDataset,
    SparseOneStepSequenceDataset,
)
from robomimic.algo.diffusion_policy import replace_bn_with_gn
from robomimic.models.chunk_iql_nets import (
    CausalSequentialActionChunkEncoder,
    CausalTemporalStateTrunk,
    FiLMStateActionFusion,
    MultiHorizonLatentPredictor,
    make_mlp,
)
from robomimic.models.obs_core import CropRandomizer

from rgb_dp_distributed import (
    DistributedContext,
    all_reduce_gradients,
    broadcast_module_buffers,
    broadcast_module_state,
    gather_rank_runtime_states,
    initialize_distributed,
    mean_distributed_scalars,
    modules_have_mutable_batch_norm,
    restore_process_rng_state,
    seed_process,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "datasets/square/idql/square_rgb_dp_idql_200demo_100success_94failure_task_reward.hdf5"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth"
)
DEFAULT_OUTPUT = (
    ROOT
    / "trained_models/square_rgb_dp_idql_rise/200demo_100success_94failure_task_reward"
)
REWARD_DEFINITIONS = {
    "task": "source_task_reward",
    "terminal_success": (
        "successful_episode: truncate_at_first_source_task_reward>0.5, "
        "reward=1_and_done=1_there; failed_episode: reward=0, "
        "done=1_at_source_end"
    ),
    "rise": "expert_transition=1; non_expert_transition=0",
}
TEMPORAL_CRITIC_ARCHITECTURE = "rise_temporal_v2"
TEMPORAL_ONE_STEP_MARKER = "temporal_one_step_idql"


def batch_scaled_step_count(
    reference_steps: int,
    reference_batch_size: int,
    effective_batch_size: int,
) -> int:
    """Translate reference-batch steps to the same processed-sample count."""
    reference_steps = int(reference_steps)
    if reference_steps <= 0:
        return 0
    return max(
        1,
        int(
            round(
                reference_steps
                * float(reference_batch_size)
                / float(effective_batch_size)
            )
        ),
    )


def configure_batch_semantics(args: argparse.Namespace, world_size: int) -> None:
    """Resolve sample-timed LR warmup while preserving update-timed targets."""
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
    args.resolved_vf_encoder_freeze_steps = batch_scaled_step_count(
        args.vf_encoder_freeze_steps,
        reference_batch_size,
        effective_batch_size,
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


def actor_matches_deployed_ema(actor_algo) -> bool:
    """Return whether the trainable policy exactly matches deployed EMA."""
    if actor_algo.ema is None:
        return False
    actor_state = actor_algo.nets.state_dict()
    ema_state = actor_algo.ema.averaged_model.state_dict()
    return actor_state.keys() == ema_state.keys() and all(
        torch.equal(actor_state[key], ema_state[key]) for key in actor_state
    )


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
    obs_encoder_learning_rate: float | None = None,
    scheduler_type: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 1,
    num_cycles: float = 0.5,
) -> None:
    policy = actor_algo.nets["policy"]
    if set(policy.keys()) != {"noise_pred_net", "obs_encoder"}:
        raise RuntimeError(
            "one-step IDQL expected the original plain DP actor modules; got "
            f"{tuple(policy.keys())}"
        )
    if obs_encoder_learning_rate is None:
        for parameter in policy.parameters():
            parameter.requires_grad_(True)
        parameter_groups = [
            {
                "params": list(policy.parameters()),
                "lr": float(learning_rate),
                "group_name": "policy",
            }
        ]
    else:
        parameter_groups = []
        for name, group_learning_rate in (
            ("noise_pred_net", float(learning_rate)),
            ("obs_encoder", float(obs_encoder_learning_rate)),
        ):
            parameters = list(policy[name].parameters())
            for parameter in parameters:
                parameter.requires_grad_(True)
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": group_learning_rate,
                    "group_name": name,
                }
            )
    actor_algo.optimizers["policy"] = torch.optim.Adam(
        parameter_groups,
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
        "optimizer_groups": {
            str(group.get("group_name", "unknown")): {
                "learning_rate": float(group["lr"]),
                "parameter_count": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
            }
            for group in actor_algo.optimizers["policy"].param_groups
        },
    }
    if not result["all_trainable"] or not result["all_in_optimizer"]:
        raise RuntimeError(f"full DP actor is not trainable: {result}")
    return result


def assert_unconditioned_actor(actor_algo) -> dict[str, Any]:
    """Reject condition adapters without changing the original DP architecture."""
    train_policy = actor_algo.nets["policy"]
    ema_policy = (
        actor_algo.ema.averaged_model["policy"]
        if actor_algo.ema is not None
        else None
    )
    if "condition_adapter" in train_policy or (
        ema_policy is not None and "condition_adapter" in ema_policy
    ):
        raise ValueError(
            "one-step IDQL requires the original unconditioned pretrained DP; "
            "condition_adapter is present"
        )
    return {
        "condition_adapter_present": False,
    }


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


def observation_history_frames(
    obs_dict: dict[str, torch.Tensor],
    obs_shapes: OrderedDict,
    observation_horizon: int,
) -> list[dict[str, torch.Tensor]]:
    """Validate and split a critic history in chronological order."""
    horizon = int(observation_horizon)
    frames = [dict() for _ in range(horizon)]
    batch_size = None
    for key, shape in obs_shapes.items():
        if key not in obs_dict:
            raise KeyError(f"critic observation is missing key {key!r}")
        value = obs_dict[key]
        expected_ndim = len(shape) + 2
        if value.ndim != expected_ndim or int(value.shape[1]) != horizon:
            raise ValueError(
                f"critic observation {key!r} must be [B,{horizon},"
                f"{','.join(str(x) for x in shape)}], got {tuple(value.shape)}"
            )
        if tuple(value.shape[2:]) != tuple(shape):
            raise ValueError(
                f"critic observation {key!r} has trailing shape "
                f"{tuple(value.shape[2:])}, expected {tuple(shape)}"
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif int(value.shape[0]) != batch_size:
            raise ValueError("critic observation keys have different batch sizes")
        for frame_index in range(horizon):
            frames[frame_index][key] = value[:, frame_index]
    return frames


def named_crop_randomizers(encoder: nn.Module) -> list[tuple[str, CropRandomizer]]:
    if isinstance(encoder, VisualDynamicsTargetEncoder):
        encoder = encoder.encoder
    return [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, CropRandomizer)
    ]


def make_temporal_crop_plan(
    encoder: nn.Module,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample one crop per trajectory and camera without changing global RNG."""
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    plan = {}
    for name, randomizer in named_crop_randomizers(encoder):
        max_height = int(randomizer.input_shape[1]) - int(randomizer.crop_height)
        max_width = int(randomizer.input_shape[2]) - int(randomizer.crop_width)
        if max_height <= 0 or max_width <= 0:
            raise ValueError(f"invalid crop geometry for {name!r}")
        shape = (int(batch_size), int(randomizer.num_crops))
        height = (
            max_height
            * torch.rand(shape, generator=generator, device=generator_device)
        ).to(dtype=torch.long)
        width = (
            max_width
            * torch.rand(shape, generator=generator, device=generator_device)
        ).to(dtype=torch.long)
        plan[name] = torch.stack((height, width), dim=-1)
    return plan


@contextmanager
def use_temporal_crop_plan(
    encoder: nn.Module,
    crop_plan: dict[str, torch.Tensor] | None,
    group_ids: torch.Tensor,
):
    randomizers = named_crop_randomizers(encoder)
    if crop_plan is None or not randomizers:
        yield
        return
    if set(crop_plan) != {name for name, _ in randomizers}:
        raise ValueError("crop plan does not match encoder randomizers")
    previous = []
    try:
        for name, randomizer in randomizers:
            previous.append(
                (
                    randomizer,
                    randomizer._external_crop_indices,
                    randomizer._external_crop_group_ids,
                )
            )
            randomizer.set_external_crop_plan(crop_plan[name], group_ids)
        yield
    finally:
        for randomizer, indices, group_ids in previous:
            randomizer.set_external_crop_plan(indices, group_ids)


class TemporalObservationNetwork(nn.Module):
    """Independent RISE-v2 encoder with a causal frame-history trunk."""

    def __init__(
        self,
        *,
        obs_shapes: OrderedDict,
        goal_shapes: OrderedDict,
        encoder_kwargs: dict,
        latent_dim: int,
        late_fusion_key: str | None,
        observation_horizon: int,
        temporal_num_layers: int,
        temporal_num_heads: int,
        temporal_feedforward_dim: int,
        temporal_dropout: float,
    ):
        super().__init__()
        self.obs_shapes = OrderedDict(obs_shapes)
        self.goal_shapes = OrderedDict(goal_shapes or {})
        self.observation_horizon = int(observation_horizon)
        self.latent_dim = int(latent_dim)
        if self.observation_horizon != 2:
            raise ValueError(
                "one-step temporal IDQL requires exactly two critic frames"
            )
        self.late_fusion_keys = tuple(
            key.strip()
            for key in str(late_fusion_key or "").split(",")
            if key.strip()
        )
        group_shapes = OrderedDict(obs=self.obs_shapes)
        if self.goal_shapes:
            group_shapes["goal"] = self.goal_shapes
        self.has_goal = "goal" in group_shapes
        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.encoder_output_dim = int(self.nets["encoder"].output_shape()[0])
        late_fusion_dim = 0
        for key in self.late_fusion_keys:
            if key not in self.obs_shapes:
                raise KeyError(f"late_fusion_key={key} is absent from obs_shapes")
            late_fusion_dim += int(np.prod(self.obs_shapes[key]))
        self.nets["frame_projection"] = make_mlp(
            self.encoder_output_dim + late_fusion_dim,
            (),
            self.latent_dim,
            final_layer_norm=True,
        )
        self.nets["temporal_trunk"] = CausalTemporalStateTrunk(
            state_dim=self.latent_dim,
            max_history=self.observation_horizon,
            num_layers=int(temporal_num_layers),
            num_heads=int(temporal_num_heads),
            feedforward_dim=int(temporal_feedforward_dim),
            dropout=float(temporal_dropout),
        )

    def _encode_frame(self, obs_dict, goal_dict=None):
        inputs = {"obs": obs_dict}
        if self.has_goal:
            if goal_dict is None:
                raise ValueError("goal-conditioned critic is missing goal observations")
            inputs["goal"] = goal_dict
        encoded = self.nets["encoder"](**inputs)
        late_parts = [
            obs_dict[key].flatten(start_dim=1) for key in self.late_fusion_keys
        ]
        if late_parts:
            encoded = torch.cat((encoded, *late_parts), dim=-1)
        return self.nets["frame_projection"](encoded)

    def encode_state(self, obs_dict, goal_dict=None, *, crop_plan=None):
        frames = observation_history_frames(
            obs_dict,
            self.obs_shapes,
            self.observation_horizon,
        )
        batch_size = int(next(iter(frames[0].values())).shape[0])
        # Match RISE-v2: one batched encoder call, newest frame first, then
        # restore chronological order before the causal temporal trunk.
        encode_order = (len(frames) - 1, *range(len(frames) - 1))
        flattened_frames = {
            key: torch.cat(
                [frames[index][key] for index in encode_order],
                dim=0,
            )
            for key in self.obs_shapes
        }
        flattened_goal = (
            None
            if goal_dict is None
            else {
                key: torch.cat([value] * len(frames), dim=0)
                for key, value in goal_dict.items()
            }
        )
        crop_groups = torch.arange(
            batch_size,
            device=next(iter(flattened_frames.values())).device,
            dtype=torch.long,
        ).repeat(len(frames))
        with use_temporal_crop_plan(
            self.nets["encoder"], crop_plan, crop_groups
        ):
            encoded = self._encode_frame(flattened_frames, flattened_goal)
        encoded_by_time = encoded.split(batch_size, dim=0)
        frame_latents = [None] * len(frames)
        for frame_index, latent in zip(encode_order, encoded_by_time):
            frame_latents[frame_index] = latent
        stacked = torch.stack(frame_latents, dim=1)
        temporal_tokens = self.nets["temporal_trunk"](stacked)
        return {
            "temporal_state": temporal_tokens[:, -1],
            "current_frame_latent": stacked[:, -1],
            "temporal_tokens": temporal_tokens,
        }

    @staticmethod
    def expand_state(state: dict[str, torch.Tensor], batch_size: int):
        expanded = {}
        for key, value in state.items():
            if int(value.shape[0]) == int(batch_size):
                expanded[key] = value
            elif int(value.shape[0]) == 1:
                expanded[key] = value.expand(
                    int(batch_size), *([-1] * (value.ndim - 1))
                )
            else:
                raise ValueError(
                    f"cannot expand temporal state batch {value.shape[0]} to "
                    f"{batch_size}"
                )
        return expanded


class TemporalOneStepActionValueNetwork(TemporalObservationNetwork):
    """RISE-v2 Q specialized to a single action and optional t+1 prediction."""

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        action_hidden_dim: int,
        num_attention_heads: int,
        num_action_conv_layers: int,
        dropout: float,
        fusion_mode: str,
        dynamics_target_dim: int,
        dynamics_enabled: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_dim = int(action_dim)
        self.chunk_horizon = 1
        self.fusion_mode = str(fusion_mode)
        self.dynamics_prediction_offsets = (1,) if dynamics_enabled else ()
        self.dynamics_target_dim = int(dynamics_target_dim)
        self.q_use_predicted_next_latent = False
        self.nets["action_encoder"] = CausalSequentialActionChunkEncoder(
            action_dim=self.action_dim,
            chunk_horizon=1,
            context_dim=self.latent_dim,
            hidden_dim=int(action_hidden_dim),
            output_dim=self.latent_dim,
            num_heads=int(num_attention_heads),
            num_conv_layers=int(num_action_conv_layers),
            dropout=float(dropout),
        )
        self.nets["state_action_fusion"] = FiLMStateActionFusion(
            latent_dim=self.latent_dim,
            mode=self.fusion_mode,
            dropout=float(dropout),
        )
        self.nets["q_head"] = make_mlp(
            3 * self.latent_dim,
            hidden_dims,
            1,
            dropout=float(dropout),
        )
        if self.dynamics_prediction_offsets:
            self.nets["dynamics_predictor"] = MultiHorizonLatentPredictor(
                latent_dim=self.latent_dim,
                target_dim=self.dynamics_target_dim,
                prediction_offsets=(1,),
                hidden_dims=hidden_dims,
                dropout=float(dropout),
            )

    def q_from_state(self, state, acts, action_mask=None, *, return_aux=False):
        if acts.ndim == 2:
            acts = acts.unsqueeze(1)
        expected = (1, self.action_dim)
        if acts.ndim != 3 or tuple(acts.shape[1:]) != expected:
            raise ValueError(
                f"one-step critic expected actions [B,1,{self.action_dim}], "
                f"got {tuple(acts.shape)}"
            )
        if int(state["temporal_state"].shape[0]) != int(acts.shape[0]):
            state = self.expand_state(state, int(acts.shape[0]))
        temporal_state = state["temporal_state"]
        offsets = self.dynamics_prediction_offsets if return_aux else ()
        action_repr, prefix_repr = self.nets["action_encoder"](
            temporal_state,
            acts,
            action_mask,
            prefix_offsets=offsets,
        )
        fusion = self.nets["state_action_fusion"](temporal_state, action_repr)
        q = self.nets["q_head"](
            torch.cat((temporal_state, action_repr, fusion), dim=-1)
        )
        if not return_aux:
            return q
        predicted = None
        if prefix_repr is not None:
            prefix_state = temporal_state.unsqueeze(1).expand_as(prefix_repr)
            prefix_fusion = self.nets["state_action_fusion"](
                prefix_state, prefix_repr
            )
            predicted = self.nets["dynamics_predictor"](prefix_fusion)
        return {
            "q": q,
            "temporal_state": temporal_state,
            "current_frame_latent": state["current_frame_latent"],
            "action_repr": action_repr,
            "state_action_fusion": fusion,
            "predicted_next_encoder": predicted,
        }

    def forward(
        self,
        obs_dict,
        acts,
        goal_dict=None,
        action_mask=None,
        return_aux=False,
        *,
        crop_plan=None,
    ):
        state = self.encode_state(obs_dict, goal_dict, crop_plan=crop_plan)
        return self.q_from_state(
            state,
            acts,
            action_mask,
            return_aux=return_aux,
        )


class TemporalOneStepValueNetwork(TemporalObservationNetwork):
    def __init__(self, *, hidden_dims, dropout, **kwargs):
        super().__init__(**kwargs)
        self.nets["value_head"] = make_mlp(
            self.latent_dim,
            hidden_dims,
            1,
            dropout=float(dropout),
        )

    def value_from_state(self, state):
        return self.nets["value_head"](state["temporal_state"])

    def forward(self, obs_dict, goal_dict=None, *, crop_plan=None):
        return self.value_from_state(
            self.encode_state(obs_dict, goal_dict, crop_plan=crop_plan)
        )


def deployed_actor_obs_encoder(actor_algo) -> nn.Module:
    actor_nets = (
        actor_algo.ema.averaged_model
        if actor_algo.ema is not None
        else actor_algo.nets
    )
    return actor_nets["policy"]["obs_encoder"]


def actor_visual_feature_indices(encoder: nn.Module) -> tuple[int, ...]:
    obs_encoder = encoder.nets["obs"]
    indices = []
    offset = 0
    for key in obs_encoder.obs_shapes:
        feature_shape = obs_encoder.obs_shapes[key]
        for randomizer in obs_encoder.obs_randomizers[key]:
            if randomizer is not None:
                feature_shape = randomizer.output_shape_in(feature_shape)
        observation_net = obs_encoder.obs_nets[key]
        if observation_net is not None:
            feature_shape = observation_net.output_shape(feature_shape)
        for randomizer in obs_encoder.obs_randomizers[key]:
            if randomizer is not None:
                feature_shape = randomizer.output_shape_out(feature_shape)
        feature_dim = int(np.prod(feature_shape))
        if ObsUtils.key_is_obs_modality(key=key, obs_modality="rgb"):
            indices.extend(range(offset, offset + feature_dim))
        offset += feature_dim
    if offset != int(encoder.output_shape()[0]) or not indices:
        raise RuntimeError("could not resolve actor visual feature coordinates")
    return tuple(indices)


class VisualDynamicsTargetEncoder(nn.Module):
    """Frozen deployed actor encoder exposing only concatenated RGB features."""

    def __init__(self, source_encoder: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(source_encoder)
        self.register_buffer(
            "visual_feature_indices",
            torch.as_tensor(
                actor_visual_feature_indices(self.encoder), dtype=torch.long
            ),
            persistent=False,
        )

    def output_shape(self):
        return [int(self.visual_feature_indices.numel())]

    def forward(self, obs_dict, goal_dict=None, *, crop_plan=None):
        inputs = {"obs": obs_dict}
        if goal_dict:
            inputs["goal"] = goal_dict
        batch_size = int(next(iter(obs_dict.values())).shape[0])
        group_ids = torch.arange(
            batch_size,
            device=next(iter(obs_dict.values())).device,
            dtype=torch.long,
        )
        with use_temporal_crop_plan(self, crop_plan, group_ids):
            encoded = self.encoder(**inputs)
        return encoded.index_select(-1, self.visual_feature_indices)


def configure_target_encoder(target_encoder: VisualDynamicsTargetEncoder) -> None:
    target_encoder.eval().requires_grad_(False)
    for module in target_encoder.modules():
        if isinstance(module, CropRandomizer):
            module.train()


def configure_critic_targets(critic_targets: nn.ModuleList) -> None:
    critic_targets.eval().requires_grad_(False)
    for critic in critic_targets:
        critic.nets["encoder"].eval().requires_grad_(False)
        for module in critic.nets["encoder"].modules():
            if isinstance(module, CropRandomizer):
                module.train()


def copy_deployed_encoder_state(module: nn.Module, actor_algo) -> dict[str, int]:
    source = deployed_actor_obs_encoder(actor_algo)
    module.nets["encoder"] = replace_bn_with_gn(module.nets["encoder"])
    source_state = source.state_dict()
    module.nets["encoder"].load_state_dict(source_state, strict=True)
    return {
        "tensor_count": int(len(source_state)),
        "parameter_count": int(sum(value.numel() for value in source_state.values())),
    }


def make_temporal_one_step_value_networks(
    actor_algo,
    *,
    hidden_dims: tuple[int, ...],
    latent_dim: int = 300,
    action_hidden_dim: int = 128,
    num_attention_heads: int = 4,
    num_action_conv_layers: int = 2,
    dropout: float = 0.0,
    num_critics: int = 2,
    critic_group_norm: bool = False,
    late_fusion_key: str | None = "robot0_gripper_qpos",
    observation_horizon: int = 2,
    temporal_num_layers: int = 2,
    temporal_num_heads: int = 6,
    temporal_feedforward_dim: int = 600,
    temporal_dropout: float = 0.0,
    fusion_mode: str = "film",
    dynamics_enabled: bool = False,
    dynamics_target_dim: int | None = None,
):
    encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
        actor_algo.obs_config.encoder
    )
    if dynamics_target_dim is None:
        dynamics_target_dim = len(
            actor_visual_feature_indices(deployed_actor_obs_encoder(actor_algo))
        )
    common = {
        "obs_shapes": actor_algo.obs_shapes,
        "goal_shapes": actor_algo.goal_shapes,
        "latent_dim": int(latent_dim),
        "late_fusion_key": late_fusion_key,
        "observation_horizon": int(observation_horizon),
        "temporal_num_layers": int(temporal_num_layers),
        "temporal_num_heads": int(temporal_num_heads),
        "temporal_feedforward_dim": int(temporal_feedforward_dim),
        "temporal_dropout": float(temporal_dropout),
    }
    critics = nn.ModuleList(
        [
            TemporalOneStepActionValueNetwork(
                **common,
                encoder_kwargs=copy.deepcopy(encoder_kwargs),
                action_dim=int(actor_algo.ac_dim),
                hidden_dims=hidden_dims,
                action_hidden_dim=int(action_hidden_dim),
                num_attention_heads=int(num_attention_heads),
                num_action_conv_layers=int(num_action_conv_layers),
                dropout=float(dropout),
                fusion_mode=str(fusion_mode),
                dynamics_target_dim=int(dynamics_target_dim),
                dynamics_enabled=bool(dynamics_enabled),
            )
            for _ in range(int(num_critics))
        ]
    )
    vf = TemporalOneStepValueNetwork(
        **common,
        encoder_kwargs=copy.deepcopy(encoder_kwargs),
        hidden_dims=hidden_dims,
        dropout=float(dropout),
    )
    if critic_group_norm:
        critics = replace_bn_with_gn(critics)
        vf = replace_bn_with_gn(vf)
    # The deployed Diffusion Policy always carries a BN->GN converted visual
    # encoder. Keep this conversion architectural (and therefore reproducible
    # during evaluation reconstruction), independently of the optional
    # critic-wide group-normalization override.
    for critic in critics:
        critic.nets["encoder"] = replace_bn_with_gn(critic.nets["encoder"])
    vf.nets["encoder"] = replace_bn_with_gn(vf.nets["encoder"])
    return critics, copy.deepcopy(critics), vf


def make_temporal_one_step_system_from_checkpoint(actor_algo, checkpoint: dict):
    required = (
        "critic_hidden_dims",
        "critic_latent_dim",
        "critic_action_hidden_dim",
        "critic_num_attention_heads",
        "critic_num_action_conv_layers",
        "critic_dropout",
        "num_critics",
        "critic_group_norm",
        "critic_observation_horizon",
        "critic_temporal_num_layers",
        "critic_temporal_num_heads",
        "critic_temporal_feedforward_dim",
        "critic_temporal_dropout",
        "critic_rise_v2_fusion_mode",
        "critic_dynamics_target_dim",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"temporal one-step checkpoint is missing {missing}")
    return make_temporal_one_step_value_networks(
        actor_algo,
        hidden_dims=tuple(int(value) for value in checkpoint["critic_hidden_dims"]),
        latent_dim=int(checkpoint["critic_latent_dim"]),
        action_hidden_dim=int(checkpoint["critic_action_hidden_dim"]),
        num_attention_heads=int(checkpoint["critic_num_attention_heads"]),
        num_action_conv_layers=int(checkpoint["critic_num_action_conv_layers"]),
        dropout=float(checkpoint["critic_dropout"]),
        num_critics=int(checkpoint["num_critics"]),
        critic_group_norm=bool(checkpoint["critic_group_norm"]),
        late_fusion_key=checkpoint.get("critic_late_fusion_key"),
        observation_horizon=int(checkpoint["critic_observation_horizon"]),
        temporal_num_layers=int(checkpoint["critic_temporal_num_layers"]),
        temporal_num_heads=int(checkpoint["critic_temporal_num_heads"]),
        temporal_feedforward_dim=int(
            checkpoint["critic_temporal_feedforward_dim"]
        ),
        temporal_dropout=float(checkpoint["critic_temporal_dropout"]),
        fusion_mode=str(checkpoint["critic_rise_v2_fusion_mode"]),
        dynamics_enabled=bool(checkpoint.get("latent_dynamics", False)),
        dynamics_target_dim=int(checkpoint["critic_dynamics_target_dim"]),
    )


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
    *,
    shuffle: bool = True,
    drop_last: bool | None = None,
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
            dict.fromkeys(
                list(config.train.dataset_keys)
                + ["actions", "rewards", "dones"]
                + (
                    ["task_rewards"]
                    if getattr(args, "reward_mode", None) == "task"
                    else []
                )
                + (
                    ["source_is_expert"]
                    if (
                        getattr(args, "reward_mode", None) == "task"
                        or getattr(args, "conditioned_actor", False)
                    )
                    else []
                )
                + (
                    ["actor_condition"]
                    if getattr(args, "conditioned_actor", False)
                    else []
                )
            )
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

    sparse_chunk_loader = bool(getattr(args, "sparse_chunk_loader", False))
    sparse_one_step_loader = bool(
        getattr(args, "sparse_one_step_loader", False)
    )
    sparse_dql_loader = bool(getattr(args, "sparse_dql_loader", False))
    sparse_loader_count = sum(
        (sparse_chunk_loader, sparse_one_step_loader, sparse_dql_loader)
    )
    if sparse_loader_count > 1:
        raise ValueError(
            "sparse chunk, one-step, and DQL loading are mutually exclusive"
        )
    if sparse_loader_count:
        if dataset.hdf5_cache_mode == "all":
            raise ValueError(
                "sparse image loading is incompatible with "
                "--hdf5-cache-mode all; use low_dim (recommended) or disable "
                "the sparse loader"
            )
        if sparse_chunk_loader:
            dataset = SparseChunkSequenceDataset(
                dataset,
                chunk_horizon=int(args.chunk_horizon),
                observation_horizon=observation_horizon,
                next_observation_horizon=int(
                    getattr(args, "critic_observation_horizon", 1)
                ),
                dynamics_prediction_offsets=tuple(
                    getattr(args, "dynamics_prediction_offsets", ())
                ),
            )
        elif sparse_one_step_loader:
            dataset = SparseOneStepSequenceDataset(
                dataset,
                observation_horizon=observation_horizon,
                critic_observation_horizon=int(
                    getattr(args, "critic_observation_horizon", 1)
                ),
            )
        else:
            dataset = SparseDQLSequenceDataset(
                dataset,
                observation_horizon=observation_horizon,
            )

    generator = torch.Generator()
    distributed_world_size = int(
        getattr(args, "distributed_world_size", 1)
    )
    distributed_rank = int(getattr(args, "distributed_rank", 0))
    distributed_sampler = None
    if distributed_world_size > 1:
        distributed_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=bool(shuffle),
            seed=int(args.seed),
            drop_last=False,
        )
    generator.manual_seed(int(args.seed) + distributed_rank)
    loader_kwargs: dict[str, Any] = {}
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    if drop_last is None:
        available_rows = (
            len(distributed_sampler)
            if distributed_sampler is not None
            else len(dataset)
        )
        drop_last = available_rows >= int(args.batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(shuffle) and distributed_sampler is None,
        sampler=distributed_sampler,
        drop_last=bool(drop_last),
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


def make_temporal_critic_optimizer(
    network: nn.Module,
    *,
    learning_rate: float,
    encoder_learning_rate: float,
) -> torch.optim.Optimizer:
    encoder_parameters = list(network.nets["encoder"].parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter
        for parameter in network.parameters()
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


def set_actor_obs_encoder_trainable(actor_algo, trainable: bool) -> None:
    """Freeze only the online actor encoder; the diffusion U-Net keeps training."""
    actor_algo.nets["policy"]["obs_encoder"].requires_grad_(bool(trainable))


def set_critic_encoders_trainable(
    critics: nn.ModuleList,
    trainable: bool,
) -> None:
    """Freeze only online Q visual encoders, never temporal or value heads."""
    for critic in critics:
        critic.nets["encoder"].requires_grad_(bool(trainable))


def set_vf_encoder_trainable(vf: nn.Module, trainable: bool) -> None:
    """Freeze only V's visual encoder, never its temporal or value heads."""
    vf.nets["encoder"].requires_grad_(bool(trainable))


def rise_reference_alignment(args: argparse.Namespace) -> dict[str, Any]:
    if args.reward_mode == "task":
        reward_alignment = "source_environment_task_reward"
    elif args.reward_mode == "terminal_success":
        reward_alignment = "canonical_first_success_terminal_reward_with_truncation"
    else:
        reward_alignment = "expert_transition_reward_1_non_expert_transition_reward_0"
    return {
        "matched": [
            "one uniformly shuffled mixed SequenceDataset",
            reward_alignment,
            "independent_two_frame_temporal_Q1_Q2_target_Q1_target_Q2_V",
            "one_step_IQL_Q_and_expectile_V_equations",
            "Q_then_target_soft_update_then_V_update_order",
            "unconditioned_pretrained_DP_actor_with_full_chunk_diffusion_BC",
            "causal_temporal_state_and_FiLM_action_fusion",
        ],
        "post_deployment_adaptations": [
            "actor_initialized_from_pretrained_DP_deployed_EMA",
            "local_success_and_failure_rollouts_replace_unreleased_RISE_play_data",
            "RISE_nearby_state_and_action_augmentation_not_applied",
            "optional_immediate_visual_successor_prediction",
        ],
        "critic_group_norm_compatibility_override": bool(args.critic_group_norm),
    }


def process_critic_batch(
    raw_batch: dict,
    actor_algo,
    obs_normalization_stats,
    critic_observation_horizon: int = 2,
) -> dict:
    """Extract a two-frame one-step IQL tuple from the sequence batch."""
    current_index = int(actor_algo.algo_config.horizon.observation_horizon) - 1
    critic_observation_horizon = int(critic_observation_horizon)
    history_start = current_index - critic_observation_horizon + 1
    if history_start < 0:
        raise ValueError(
            "critic observation horizon exceeds the actor observation history"
        )
    sparse_next_obs = "one_step_sparse_next_obs" in raw_batch
    if sparse_next_obs:
        invalid_sparse_shapes = {
            key: tuple(raw_batch["next_obs"][key].shape)
            for key in actor_algo.obs_shapes
            if raw_batch["next_obs"][key].ndim < 2
            or int(raw_batch["next_obs"][key].shape[1])
            != critic_observation_horizon
        }
        if invalid_sparse_shapes:
            raise ValueError(
                "sparse one-step next observations have the wrong history: "
                f"{invalid_sparse_shapes}"
            )
    critic_batch = {
        "obs": {
            key: raw_batch["obs"][key][
                :, history_start : current_index + 1
            ]
            for key in actor_algo.obs_shapes
        },
        "next_obs": {
            key: (
                raw_batch["next_obs"][key]
                if sparse_next_obs
                else raw_batch["next_obs"][key][
                    :, history_start : current_index + 1
                ]
            )
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
            "one-step IDQL critic requires actions [B, action_dim], got "
            f"shape={tuple(actions.shape)}; action chunks are not accepted"
        )
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
    for name in ("obs", "next_obs"):
        invalid_history = {
            key: tuple(value.shape)
            for key, value in critic_batch[name].items()
            if int(value.shape[1]) != critic_observation_horizon
        }
        if invalid_history:
            raise ValueError(
                f"critic {name} must retain {critic_observation_horizon} frames: "
                f"{invalid_history}"
            )
    return critic_batch


def align_shared_batch_actions(raw_batch: dict, validate: bool = True) -> dict:
    """Validate once and give actor and critic the identical clipped tensor."""
    actions = raw_batch["actions"]
    if not validate:
        raw_batch = dict(raw_batch)
        raw_batch["actions"] = actions.clamp(-1.0, 1.0)
        return raw_batch
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


def compute_temporal_one_step_losses(
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    vf: TemporalOneStepValueNetwork,
    dynamics_target_encoder: VisualDynamicsTargetEncoder | None,
    batch: dict,
    *,
    discount: float,
    expectile: float,
    use_huber: bool,
    dynamics_weight: float,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """One-step IQL plus an optional immediate visual successor objective."""
    dynamics_enabled = float(dynamics_weight) > 0.0
    if dynamics_enabled != (dynamics_target_encoder is not None):
        raise ValueError(
            "dynamics target encoder presence must match dynamics_weight > 0"
        )
    batch_size = int(batch["actions"].shape[0])
    device = batch["actions"].device
    crop_seeds = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (len(critics) + 2,),
        device="cpu",
    ).tolist()
    critic_crop_plans = [
        make_temporal_crop_plan(
            critic.nets["encoder"],
            batch_size=batch_size,
            seed=int(seed),
            device=device,
        )
        for critic, seed in zip(critics, crop_seeds[: len(critics)])
    ]
    action_mask = batch["actions"].new_ones((batch_size, 1))
    outputs = [
        critic(
            obs_dict=batch["obs"],
            acts=batch["actions"],
            action_mask=action_mask,
            goal_dict=batch["goal_obs"],
            return_aux=True,
            crop_plan=crop_plan,
        )
        for critic, crop_plan in zip(critics, critic_crop_plans)
    ]
    pred_qs = [output["q"] for output in outputs]

    with torch.no_grad():
        next_v_crop_plan = make_temporal_crop_plan(
            vf.nets["encoder"],
            batch_size=batch_size,
            seed=int(crop_seeds[-2]),
            device=device,
        )
        target_vf_pred = vf(
            obs_dict=batch["next_obs"],
            goal_dict=batch["goal_obs"],
            crop_plan=next_v_crop_plan,
        )
        rewards = batch["rewards"]
        q_target = (
            rewards
            + (1.0 - batch["dones"])
            * float(discount)
            * target_vf_pred
        )
        target_q_crop_plan = make_temporal_crop_plan(
            vf.nets["encoder"],
            batch_size=batch_size,
            seed=int(crop_seeds[-1]),
            device=device,
        )
        target_qs = [
            critic(
                obs_dict=batch["obs"],
                acts=batch["actions"],
                action_mask=action_mask,
                goal_dict=batch["goal_obs"],
                crop_plan=target_q_crop_plan,
            )
            for critic in critic_targets
        ]
        target_q_min = torch.cat(target_qs, dim=1).min(
            dim=1, keepdim=True
        ).values
        target_features = []
        if dynamics_enabled:
            next_frame = {
                key: value[:, -1] for key, value in batch["next_obs"].items()
            }
            target_features = [
                dynamics_target_encoder(
                    next_frame,
                    batch["goal_obs"],
                    crop_plan=crop_plan,
                ).detach()
                for crop_plan in critic_crop_plans
            ]

    regression = F.smooth_l1_loss if use_huber else F.mse_loss
    q_losses = [regression(prediction, q_target) for prediction in pred_qs]
    dynamics_losses = []
    critic_losses = []
    if dynamics_enabled:
        for output, q_loss, target in zip(outputs, q_losses, target_features):
            predicted = output["predicted_next_encoder"]
            if predicted is None or tuple(predicted.shape[1:2]) != (1,):
                raise RuntimeError("one-step dynamics predictor did not produce t+1")
            dynamics_loss = F.smooth_l1_loss(predicted[:, 0], target)
            dynamics_losses.append(dynamics_loss)
            critic_losses.append(q_loss + float(dynamics_weight) * dynamics_loss)
    else:
        critic_losses = q_losses

    vf_crop_plan = make_temporal_crop_plan(
        vf.nets["encoder"],
        batch_size=batch_size,
        seed=int(crop_seeds[-1]),
        device=device,
    )
    vf_state = vf.encode_state(
        batch["obs"], batch["goal_obs"], crop_plan=vf_crop_plan
    )
    vf_pred = vf.value_from_state(vf_state)
    vf_error = vf_pred - target_q_min
    vf_weight = torch.where(
        vf_error > 0.0,
        1.0 - float(expectile),
        float(expectile),
    )
    vf_loss = (vf_weight * vf_error.square()).mean()
    q_tensor = torch.cat(pred_qs, dim=1)
    zero = q_tensor.new_zeros(())
    dynamics_mean = (
        torch.stack(dynamics_losses).mean() if dynamics_losses else zero
    )
    temporal_states = torch.stack(
        [output["temporal_state"] for output in outputs], dim=0
    )
    info = {
        **{
            f"critic/q{index + 1}_loss": loss.detach()
            for index, loss in enumerate(q_losses)
        },
        **{
            f"critic/q{index + 1}_mean": prediction.mean().detach()
            for index, prediction in enumerate(pred_qs)
        },
        "critic/q_target_mean": q_target.mean().detach(),
        "critic/target_v_mean": target_vf_pred.mean().detach(),
        "critic/q_ensemble_std": q_tensor.std(dim=1).mean().detach(),
        "vf/loss": vf_loss.detach(),
        "vf/value_mean": vf_pred.mean().detach(),
        "vf/target_q_min_mean": target_q_min.mean().detach(),
        "vf/error_mean": vf_error.mean().detach(),
        "dynamics/loss": dynamics_mean.detach(),
        "dynamics/weighted_loss": (
            float(dynamics_weight) * dynamics_mean
        ).detach(),
        "dynamics/enabled": q_tensor.new_tensor(float(dynamics_enabled)),
        "representation/temporal_feature_std": temporal_states.std(
            dim=1, unbiased=False
        ).mean().detach(),
        "data/reward_mean": rewards.mean().detach(),
        "data/done_fraction": batch["dones"].mean().detach(),
        "data/action_abs_mean": batch["actions"].abs().mean().detach(),
        "data/action_min": batch["actions"].min().detach(),
        "data/action_max": batch["actions"].max().detach(),
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
    gradient_sync_fn=None,
) -> None:
    """Backpropagate Q1/Q2/V, synchronize once, then update all networks."""
    optimizers = [*critic_optimizers, vf_optimizer]
    parameter_groups = [
        [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        for optimizer in optimizers
    ]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    for loss in critic_losses:
        loss.backward()
    vf_loss.backward()

    if gradient_sync_fn is not None:
        gradient_sync_fn(
            parameter
            for parameters in parameter_groups
            for parameter in parameters
        )
    if max_gradient_norm is not None:
        for parameters in parameter_groups:
            torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    for optimizer in optimizers:
        optimizer.step()
    for critic, critic_target in zip(critics, critic_targets):
        TorchUtils.soft_update(
            source=critic,
            target=critic_target,
            tau=float(target_tau),
        )


def actor_train_step(
    actor_algo,
    raw_batch: dict,
    epoch: int,
    obs_normalization_stats,
    defer_scalar_conversion: bool = False,
) -> dict[str, Any]:
    actor_batch = actor_algo.process_batch_for_training(raw_batch)
    actor_batch = actor_algo.postprocess_batch_for_training(
        actor_batch,
        obs_normalization_stats=obs_normalization_stats,
    )
    info = actor_algo.train_on_batch(actor_batch, epoch=epoch, validate=False)
    actor_algo.on_gradient_step()
    log = (
        actor_algo.log_info(info, materialize=False)
        if defer_scalar_conversion
        else actor_algo.log_info(info)
    )
    allowed_types = (int, float, np.number, torch.Tensor)
    return {
        f"actor/{key}" if not str(key).startswith("actor/") else str(key): (
            value if isinstance(value, torch.Tensor) else float(value)
        )
        for key, value in log.items()
        if isinstance(value, allowed_types)
    }


def dataset_audit(
    dataset_path: Path,
    dataset_size: int,
    expected_task: str | None = None,
    expected_reward_mode: str | None = None,
) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as handle:
        reward_definition = str(handle.attrs.get("reward_definition", ""))
        reward_mode = str(handle.attrs.get("reward_mode", ""))
        if not reward_mode and reward_definition == REWARD_DEFINITIONS["rise"]:
            reward_mode = "rise"
        if reward_mode not in REWARD_DEFINITIONS:
            raise ValueError(
                "dataset has an unsupported or missing reward mode: "
                f"reward_mode={reward_mode!r}, definition={reward_definition!r}"
            )
        expected_definition = REWARD_DEFINITIONS[reward_mode]
        if reward_definition != expected_definition:
            raise ValueError(
                f"dataset reward_mode={reward_mode!r} requires "
                f"reward_definition={expected_definition!r}, got "
                f"{reward_definition!r}"
            )
        if expected_reward_mode is not None and reward_mode != expected_reward_mode:
            raise ValueError(
                f"dataset reward_mode={reward_mode!r} does not match requested "
                f"reward_mode={expected_reward_mode!r}; rebuild the dataset with "
                f"--reward-mode {expected_reward_mode} --overwrite, or explicitly "
                f"train with --reward-mode {reward_mode}"
            )
        task_reward_audit = None
        if reward_mode in ("task", "terminal_success"):
            missing_source_labels = [
                key
                for key, episode in handle["data"].items()
                if "source_is_expert" not in episode
            ]
            if missing_source_labels:
                raise ValueError(
                    f"{reward_mode}-reward dataset is missing source_is_expert labels for "
                    f"episodes={missing_source_labels[:8]}; rebuild it"
                )
            task_reward_audit = {}
            for episode_key, episode in handle["data"].items():
                for key in ("rewards", "task_rewards"):
                    if key not in episode:
                        raise ValueError(
                            f"task-reward dataset data/{episode_key} is missing {key}"
                        )
                rewards = np.asarray(episode["rewards"][:], dtype=np.float32)
                task_rewards = np.asarray(
                    episode["task_rewards"][:],
                    dtype=np.float32,
                )
                source = episode.attrs.get("rise_source", "unknown")
                if isinstance(source, bytes):
                    source = source.decode("utf-8")
                source = str(source)
                if reward_mode == "task" and not np.array_equal(
                    rewards, task_rewards
                ):
                    raise ValueError(
                        f"task-reward dataset data/{episode_key} rewards do not "
                        "match the preserved source task_rewards"
                    )
                if not np.isfinite(rewards).all() or not np.isfinite(task_rewards).all():
                    raise ValueError(
                        f"task-reward dataset data/{episode_key} contains "
                        "non-finite rewards"
                    )
                if reward_mode == "terminal_success":
                    if source not in {"expert", "non_expert_success", "non_expert_failure"}:
                        raise ValueError(f"unsupported terminal-success source={source!r}")
                    expected_rewards = np.zeros_like(rewards)
                    if source in {"expert", "non_expert_success"}:
                        expected_rewards[-1] = 1.0
                        task_positive = np.flatnonzero(task_rewards > 0.5)
                        if not (task_positive.size == 1 and task_positive[0] == rewards.size - 1):
                            raise ValueError(
                                f"terminal-success data/{episode_key} must end "
                                "at its first positive source task reward"
                            )
                    elif np.any(task_rewards > 0.5):
                        raise ValueError(
                            f"terminal-success failure data/{episode_key} has a "
                            "positive source task reward"
                        )
                    if not np.array_equal(rewards, expected_rewards):
                        raise ValueError(
                            f"terminal-success data/{episode_key} has incorrect "
                            "canonical critic rewards"
                        )
                    if "dones" not in episode:
                        raise ValueError(f"data/{episode_key} is missing dones")
                    dones = np.asarray(episode["dones"][:], dtype=np.float32)
                    expected_dones = np.zeros_like(rewards)
                    expected_dones[-1] = 1.0
                    if not np.array_equal(dones, expected_dones):
                        raise ValueError(f"data/{episode_key} has invalid terminal dones")
                    source_count = int(episode.attrs.get("source_num_samples", -1))
                    if source_count < rewards.size:
                        raise ValueError(
                            f"data/{episode_key} source_num_samples={source_count} "
                            f"is shorter than retained length={rewards.size}"
                        )
                    if int(
                        episode.attrs.get("truncated_transition_count", -1)
                    ) != source_count - rewards.size:
                        raise ValueError(
                            f"data/{episode_key} has inconsistent truncation metadata"
                        )
                source_stats = task_reward_audit.setdefault(
                    source,
                    {
                        "episodes": 0,
                        "transitions": 0,
                        "positive_reward_episodes": 0,
                        "positive_reward_transitions": 0,
                        "reward_sum": 0.0,
                        "source_positive_reward_transitions": 0,
                        "source_reward_sum": 0.0,
                    },
                )
                source_stats["episodes"] += 1
                source_stats["transitions"] += int(rewards.size)
                source_stats["positive_reward_episodes"] += int(
                    np.any(rewards > 0.5)
                )
                source_stats["positive_reward_transitions"] += int(
                    np.count_nonzero(rewards > 0.5)
                )
                source_stats["reward_sum"] += float(rewards.sum())
                source_stats["source_positive_reward_transitions"] += int(
                    np.count_nonzero(task_rewards > 0.5)
                )
                source_stats["source_reward_sum"] += float(task_rewards.sum())
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
        "reward_mode": reward_mode,
        "reward_definition": reward_definition,
        "task_reward_audit": task_reward_audit,
        "builder_summary": builder_summary,
    }


def mixed_dataset_source_episodes(path: Path) -> set[tuple[str, str]]:
    """Return immutable source episode identifiers from a mixed HDF5."""
    records: set[tuple[str, str]] = set()
    with h5py.File(path, "r") as handle:
        for episode_key, episode in handle["data"].items():
            source_file = episode.attrs.get("rise_source_file")
            source_demo = episode.attrs.get("rise_source_demo")
            if isinstance(source_file, bytes):
                source_file = source_file.decode("utf-8")
            if isinstance(source_demo, bytes):
                source_demo = source_demo.decode("utf-8")
            if not source_file or not source_demo:
                raise ValueError(
                    f"{path}:data/{episode_key} is missing source provenance"
                )
            identifier = (
                str(Path(str(source_file)).expanduser().resolve()),
                str(source_demo),
            )
            if identifier in records:
                raise ValueError(
                    f"{path} contains source episode more than once: {identifier}"
                )
            records.add(identifier)
    return records


def audit_validation_dataset_split(
    training_dataset: Path,
    validation_dataset: Path,
) -> dict[str, int]:
    """Prove held-out evaluation shares no source episodes with fitting."""
    if training_dataset == validation_dataset:
        raise ValueError("validation dataset must differ from training dataset")
    training_records = mixed_dataset_source_episodes(training_dataset)
    validation_records = mixed_dataset_source_episodes(validation_dataset)
    overlap = sorted(training_records.intersection(validation_records))
    if overlap:
        raise ValueError(
            "training and validation mixed datasets share source episodes: "
            f"{overlap[:10]}"
        )
    if not validation_records:
        raise ValueError("validation dataset contains no source episodes")
    return {
        "training_source_episodes": int(len(training_records)),
        "validation_source_episodes": int(len(validation_records)),
        "overlap_source_episodes": 0,
    }


def _local_scalar_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric is not scalar: {tuple(value.shape)}")
        return value.detach().reshape(()).to(device=device, dtype=torch.float32)
    scalar = torch.as_tensor(value, device=device, dtype=torch.float32)
    if scalar.numel() != 1:
        raise ValueError(f"metric is not scalar: {tuple(scalar.shape)}")
    return scalar.reshape(())


class WeightedScalarMetricAccumulator:
    """Accumulate validation means weighted by held-out window count."""

    def __init__(self, device: torch.device):
        self.device = device
        self.keys: tuple[str, ...] = ()
        self.weighted_sums: torch.Tensor | None = None
        self.total_weight = 0

    @torch.no_grad()
    def update(self, metrics: dict[str, Any], weight: int) -> None:
        if int(weight) <= 0:
            raise ValueError("validation metric weight must be positive")
        keys = tuple(sorted(metrics))
        if self.keys and keys != self.keys:
            raise ValueError("validation metric keys changed between batches")
        values = torch.stack(
            [_local_scalar_tensor(metrics[key], self.device) for key in keys]
        )
        weighted = values * float(weight)
        if self.weighted_sums is None:
            self.keys = keys
            self.weighted_sums = weighted.clone()
        else:
            self.weighted_sums.add_(weighted)
        self.total_weight += int(weight)

    @torch.no_grad()
    def means(self) -> dict[str, float]:
        if self.weighted_sums is None or self.total_weight == 0:
            return {}
        values = (self.weighted_sums / float(self.total_weight)).cpu().tolist()
        return {key: float(value) for key, value in zip(self.keys, values)}


@contextmanager
def fork_rng_with_seed(seed: int, device: torch.device):
    """Temporarily seed validation crops without perturbing training RNG."""
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


def actor_validation_step(
    actor_algo,
    raw_batch: dict,
    epoch: int,
    obs_normalization_stats,
) -> dict[str, Any]:
    """Evaluate the deployed EMA actor without optimizer or EMA updates."""
    if actor_algo.ema is None:
        raise RuntimeError("held-out actor validation requires EMA actor weights")
    online_nets = actor_algo.nets
    ema_nets = actor_algo.ema.averaged_model
    ema_was_training = bool(ema_nets.training)
    try:
        actor_algo.nets = ema_nets
        ema_nets.eval()
        actor_batch = actor_algo.process_batch_for_training(raw_batch)
        actor_batch = actor_algo.postprocess_batch_for_training(
            actor_batch,
            obs_normalization_stats=obs_normalization_stats,
        )
        info = actor_algo.train_on_batch(actor_batch, epoch=epoch, validate=True)
        log = actor_algo.log_info(info, materialize=False)
    finally:
        actor_algo.nets = online_nets
        ema_nets.train(ema_was_training)
    allowed_types = (int, float, np.number, torch.Tensor)
    return {
        f"actor/{key}" if not str(key).startswith("actor/") else str(key): (
            value if isinstance(value, torch.Tensor) else float(value)
        )
        for key, value in log.items()
        if isinstance(value, allowed_types)
        and not str(key).startswith("Optimizer/")
    }


@torch.no_grad()
def evaluate_validation_epoch(
    *,
    args: argparse.Namespace,
    epoch: int,
    loader,
    actor_algo,
    critics: nn.ModuleList,
    critic_targets: nn.ModuleList,
    vf: nn.Module,
    dynamics_target_encoder: VisualDynamicsTargetEncoder | None,
    obs_normalization_stats,
    device: torch.device,
) -> dict[str, float]:
    """Run deterministic, full-coverage held-out actor and critic evaluation."""
    accumulator = WeightedScalarMetricAccumulator(device)
    batch_count = 0
    window_count = 0
    actor_was_training = bool(actor_algo.nets.training)
    critic_modes = [bool(module.training) for module in critics]
    vf_was_training = bool(vf.training)
    actor_algo.set_eval()
    critics.eval()
    critic_targets.eval()
    vf.eval()
    if dynamics_target_encoder is not None:
        dynamics_target_encoder.eval()
    try:
        with fork_rng_with_seed(int(args.validation_seed), device):
            for raw_batch in loader:
                raw_batch = align_shared_batch_actions(raw_batch)
                rows = int(raw_batch["actions"].shape[0])
                metrics = actor_validation_step(
                    actor_algo,
                    raw_batch,
                    epoch,
                    obs_normalization_stats,
                )
                critic_batch = process_critic_batch(
                    raw_batch,
                    actor_algo,
                    obs_normalization_stats,
                    critic_observation_horizon=args.critic_observation_horizon,
                )
                _, _, critic_info = compute_temporal_one_step_losses(
                    critics,
                    critic_targets,
                    vf,
                    dynamics_target_encoder,
                    critic_batch,
                    discount=args.discount,
                    expectile=args.expectile,
                    use_huber=args.use_huber,
                    dynamics_weight=args.dynamics_weight,
                )
                metrics.update(critic_info)
                accumulator.update(
                    {f"validation/{key}": value for key, value in metrics.items()},
                    rows,
                )
                batch_count += 1
                window_count += rows
    finally:
        if actor_was_training:
            actor_algo.set_train()
        else:
            actor_algo.set_eval()
        for module, was_training in zip(critics, critic_modes):
            module.train(was_training)
        vf.train(vf_was_training)
        configure_critic_targets(critic_targets)
        if dynamics_target_encoder is not None:
            configure_target_encoder(dynamics_target_encoder)
    if batch_count == 0 or window_count == 0:
        raise ValueError("validation loader produced no held-out windows")
    result = accumulator.means()
    result["validation/batches"] = float(batch_count)
    result["validation/windows"] = float(window_count)
    return result


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
    dynamics_target_encoder: VisualDynamicsTargetEncoder | None,
    critic_optimizers: list[torch.optim.Optimizer],
    vf_optimizer: torch.optim.Optimizer,
    critic_lr_schedulers: list[Any],
    vf_lr_scheduler: Any,
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
        "rise_style_rgb_idql": True,
        TEMPORAL_ONE_STEP_MARKER: True,
        "hybrid_dp_chunk_actor_iql": True,
        "visual_critic_idql": True,
        "actor_model": actor_algo.serialize(),
        "critics": [critic.state_dict() for critic in critics],
        "critic_targets": [critic.state_dict() for critic in critic_targets],
        "vf": vf.state_dict(),
        "dynamics_target_encoder": (
            dynamics_target_encoder.state_dict()
            if dynamics_target_encoder is not None
            else None
        ),
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
        "global_samples_seen": int(global_samples_seen),
        "history": history,
        "pretrained_dp_checkpoint": str(args.checkpoint),
        "task": str(args.task),
        "dataset": str(args.dataset),
        "validation_dataset": (
            str(args.validation_dataset)
            if args.validation_dataset is not None
            else None
        ),
        "validation_seed": int(args.validation_seed),
        "single_dataloader": distributed_world_size == 1,
        "sampling": (
            "distributed_shuffled_SequenceDataset_indices"
            if distributed_world_size > 1
            else "uniform_shuffled_SequenceDataset_indices"
        ),
        "reward_mode": str(args.reward_mode),
        "reward_definition": REWARD_DEFINITIONS[args.reward_mode],
        "actor_training_objective": "diffusion_bc_full_chunk",
        "actor_grouped_optimizer": bool(args.actor_grouped_optimizer),
        "actor_unet_lr": float(args.actor_unet_lr),
        "actor_obs_encoder_lr": float(args.actor_obs_encoder_lr),
        "actor_obs_encoder_freeze_steps": int(
            args.actor_obs_encoder_freeze_steps
        ),
        "resolved_actor_obs_encoder_freeze_steps": int(
            args.resolved_actor_obs_encoder_freeze_steps
        ),
        "actor_source_mask": "none_all_shared_batch_rows",
        "actor_data_mode": "all_human_success_failure_rows",
        "critic_training_objective": (
            "task_reward_one_step_iql"
            if args.reward_mode == "task"
            else "terminal_success_reward_one_step_iql"
            if args.reward_mode == "terminal_success"
            else "rise_one_step_iql"
        ),
        "critic_reward_source": (
            "rewards=source_environment_task_reward"
            if args.reward_mode == "task"
            else "rewards=canonical_first_success_terminal_reward"
            if args.reward_mode == "terminal_success"
            else "rewards=expert_1_non_expert_0"
        ),
        "critic_input_mode": "independent_causal_two_frame_temporal_encoders",
        "critic_action_space": "pretrained_dp_normalized_action_space",
        "critic_action_input": "single_action_at_current_observation_index",
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
        "critic_dynamics_target_dim": int(
            critics[0].dynamics_target_dim
        ),
        "critic_dynamics_prediction_offsets": (
            (1,) if float(args.dynamics_weight) > 0.0 else ()
        ),
        "latent_dynamics": bool(float(args.dynamics_weight) > 0.0),
        "dynamics_weight": float(args.dynamics_weight),
        "dynamics_prediction_target": (
            "immediate_next_visual_actor_encoder_features"
            if float(args.dynamics_weight) > 0.0
            else None
        ),
        "critic_hidden_dims": tuple(int(value) for value in args.critic_hidden_dims),
        "num_critics": int(args.num_critics),
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
        "critic_encoder_freeze_steps": int(args.encoder_freeze_steps),
        "resolved_critic_encoder_freeze_steps": int(
            args.resolved_encoder_freeze_steps
        ),
        "vf_encoder_freeze_steps": int(args.vf_encoder_freeze_steps),
        "resolved_vf_encoder_freeze_steps": int(
            args.resolved_vf_encoder_freeze_steps
        ),
        "action_dim": int(actor_algo.ac_dim),
        "action_normalization_stats": copy.deepcopy(action_normalization_stats),
        "observation_horizon": int(actor_algo.algo_config.horizon.observation_horizon),
        "actor_prediction_horizon": int(actor_algo.algo_config.horizon.prediction_horizon),
        "actor_action_horizon": int(actor_algo.algo_config.horizon.action_horizon),
        "discount": float(args.discount),
        "expectile": float(args.expectile),
        "target_tau": float(args.target_tau),
        "actor_initialized_from_deployed_ema": True,
        "actor_pretrained_checkpoint_loaded": True,
        "actor_exactly_matched_deployed_ema_at_initialization": True,
        "rise_reference_alignment": rise_reference_alignment(args),
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
        },
        "distributed_rank_states": rank_runtime_states,
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
        "validation_dataset",
        "validation_seed",
        "reward_mode",
        "seed",
        "batch_size",
        "effective_global_batch_size",
        "schedule_reference_batch_size",
        "steps_per_epoch",
        "critic_hidden_dims",
        "num_critics",
        "critic_group_norm",
        "critic_late_fusion_key",
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
        "dynamics_weight",
        "discount",
        "expectile",
        "target_tau",
        "actor_lr",
        "actor_grouped_optimizer",
        "actor_unet_lr",
        "actor_obs_encoder_lr",
        "actor_obs_encoder_freeze_steps",
        "resolved_actor_obs_encoder_freeze_steps",
        "critic_lr",
        "encoder_lr",
        "encoder_freeze_steps",
        "resolved_encoder_freeze_steps",
        "vf_lr",
        "vf_encoder_freeze_steps",
        "resolved_vf_encoder_freeze_steps",
        "lr_scheduler",
        "lr_warmup_steps",
        "resolved_lr_warmup_steps",
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
    distributed = initialize_distributed(args)
    args.distributed = bool(distributed.enabled)
    args.distributed_rank = int(distributed.rank)
    args.distributed_local_rank = int(distributed.local_rank)
    args.distributed_world_size = int(distributed.world_size)
    configure_batch_semantics(args, distributed.world_size)
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"{args.dataset} does not exist; run scripts/build_rgb_dp_idql_dataset.py first"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.validation_dataset is not None and not args.validation_dataset.is_file():
        raise FileNotFoundError(args.validation_dataset)
    if not 0.5 <= float(args.expectile) < 1.0:
        raise ValueError("expectile must be in [0.5, 1.0)")
    if int(args.num_critics) < 2:
        raise ValueError("RISE-style clipped double Q requires at least two critics")
    if distributed.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.enabled:
        dist.barrier()

    device = distributed.device
    # Construct identical initial models on every rank. Rank-specific streams
    # are installed after initialization and any resume state is loaded.
    seed_process(args.seed, device)

    # Stage the CUDA-tagged DP checkpoint on CPU so every torchrun rank does
    # not restore its serialized copy onto cuda:0.
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
    if not actor_matches_deployed_ema(actor_algo):
        raise RuntimeError(
            "trainable actor does not exactly match the pretrained deployed EMA"
        )
    # The serialized weights have been copied into the local-rank actor.
    dp_checkpoint.pop("model", None)

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
        and int(args.resolved_lr_warmup_steps) >= int(args.lr_total_steps)
    ):
        raise ValueError(
            "resolved_lr_warmup_steps="
            f"{args.resolved_lr_warmup_steps} (reference "
            f"{args.lr_warmup_steps}) must be smaller than the "
            f"{args.lr_total_steps} total training steps"
        )
    configure_actor_optimizer(
        actor_algo,
        args.actor_unet_lr,
        obs_encoder_learning_rate=(
            args.actor_obs_encoder_lr
            if args.actor_grouped_optimizer
            else None
        ),
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.resolved_lr_warmup_steps,
        total_steps=args.lr_total_steps,
        num_cycles=args.lr_num_cycles,
    )
    unconditioned_actor_audit = assert_unconditioned_actor(actor_algo)
    audit = dataset_audit(
        args.dataset,
        len(dataset),
        expected_task=args.task,
        expected_reward_mode=args.reward_mode,
    )
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
        )
        validation_split_audit = audit_validation_dataset_split(
            args.dataset,
            args.validation_dataset,
        )
    action_stats = dp_checkpoint["action_normalization_stats"]
    obs_normalization_stats = copy.deepcopy(actor_policy.obs_normalization_stats)

    dynamics_enabled = float(args.dynamics_weight) > 0.0
    dynamics_target_dim = len(
        actor_visual_feature_indices(deployed_actor_obs_encoder(actor_algo))
    )
    dynamics_target_encoder = (
        VisualDynamicsTargetEncoder(deployed_actor_obs_encoder(actor_algo))
        if dynamics_enabled
        else None
    )
    critics, critic_targets, vf = make_temporal_one_step_value_networks(
        actor_algo,
        hidden_dims=tuple(int(value) for value in args.critic_hidden_dims),
        latent_dim=int(args.latent_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        num_attention_heads=int(args.num_attention_heads),
        num_action_conv_layers=int(args.num_action_conv_layers),
        dropout=float(args.dropout),
        num_critics=args.num_critics,
        critic_group_norm=args.critic_group_norm,
        late_fusion_key=args.critic_late_fusion_key,
        observation_horizon=int(args.critic_observation_horizon),
        temporal_num_layers=int(args.temporal_num_layers),
        temporal_num_heads=int(args.temporal_num_heads),
        temporal_feedforward_dim=int(args.temporal_feedforward_dim),
        temporal_dropout=float(args.temporal_dropout),
        fusion_mode=str(args.rise_v2_fusion_mode),
        dynamics_enabled=dynamics_enabled,
        dynamics_target_dim=dynamics_target_dim,
    )
    encoder_initialization = {
        "mode": "deployed_pretrained_dp_raw_obs_encoder_copy",
        "critics": [
            copy_deployed_encoder_state(critic, actor_algo) for critic in critics
        ],
        "vf": copy_deployed_encoder_state(vf, actor_algo),
    }
    critic_targets = copy.deepcopy(critics)
    critics = critics.float().to(device)
    critic_targets = critic_targets.float().to(device)
    vf = vf.float().to(device)
    if dynamics_target_encoder is not None:
        dynamics_target_encoder = dynamics_target_encoder.float().to(device)
        configure_target_encoder(dynamics_target_encoder)
    critic_targets.requires_grad_(False)
    critic_optimizers = [
        make_temporal_critic_optimizer(
            critic,
            learning_rate=args.critic_lr,
            encoder_learning_rate=args.encoder_lr,
        )
        for critic in critics
    ]
    vf_optimizer = make_temporal_critic_optimizer(
        vf,
        learning_rate=args.vf_lr,
        encoder_learning_rate=args.encoder_lr,
    )
    critic_lr_schedulers = [
        make_step_lr_scheduler(
            optimizer,
            scheduler_type=args.lr_scheduler,
            warmup_steps=args.resolved_lr_warmup_steps,
            total_steps=args.lr_total_steps,
            num_cycles=args.lr_num_cycles,
        )
        for optimizer in critic_optimizers
    ]
    vf_lr_scheduler = make_step_lr_scheduler(
        vf_optimizer,
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
        if not checkpoint.get("rise_style_rgb_idql", False):
            raise ValueError("resume checkpoint is not from train_rgb_dp_idql.py")
        if not checkpoint.get(TEMPORAL_ONE_STEP_MARKER, False):
            raise ValueError(
                "legacy one-frame checkpoints cannot resume the temporal "
                "one-step recipe"
            )
        if checkpoint.get("actor_training_objective") != "diffusion_bc_full_chunk":
            raise ValueError(
                "resume checkpoint did not train the DP actor with full-chunk "
                "diffusion BC and cannot resume standard IDQL"
            )
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
        saved_dynamics_target = checkpoint.get("dynamics_target_encoder")
        if dynamics_target_encoder is None:
            if saved_dynamics_target is not None:
                raise ValueError(
                    "resume checkpoint has dynamics state but dynamics is disabled"
                )
        else:
            if saved_dynamics_target is None:
                raise ValueError("resume checkpoint is missing dynamics target state")
            dynamics_target_encoder.load_state_dict(
                saved_dynamics_target, strict=True
            )
            configure_target_encoder(dynamics_target_encoder)
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
        global_samples_seen = int(
            checkpoint.get(
                "global_samples_seen",
                global_step * int(args.effective_global_batch_size),
            )
        )
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
            # A serial checkpoint can initialize a distributed continuation,
            # but every new rank needs an independent deterministic stream.
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

    actor_algo.set_train()
    critics.train()
    configure_critic_targets(critic_targets)
    vf.train()
    synchronized_modules: list[nn.Module] = [
        actor_algo.nets,
        critics,
        critic_targets,
        vf,
    ]
    if dynamics_target_encoder is not None:
        synchronized_modules.append(dynamics_target_encoder)
    if actor_algo.ema is not None:
        synchronized_modules.append(actor_algo.ema.averaged_model)
    broadcast_module_state(synchronized_modules, distributed)
    gradient_sync_fn = (
        (
            lambda parameters: all_reduce_gradients(
                parameters,
                distributed,
                bucket_cap_mb=args.gradient_bucket_cap_mb,
            )
        )
        if distributed.enabled
        else None
    )
    actor_algo.gradient_sync_fn = gradient_sync_fn
    training_buffer_modules = [
        actor_algo.nets,
        critics,
        critic_targets,
        vf,
    ]
    if dynamics_target_encoder is not None:
        training_buffer_modules.append(dynamics_target_encoder)
    synchronize_training_buffers = modules_have_mutable_batch_norm(
        training_buffer_modules
    )
    if distributed.enabled and args.resume_checkpoint is None:
        seed_process(int(args.seed) + distributed.rank, device)
    actor_audit = {
        **actor_trainability(actor_algo),
        **unconditioned_actor_audit,
    }
    architecture = {
        "actor": actor_audit,
        "critic_parameter_counts": [parameter_count(critic) for critic in critics],
        "target_critic_parameter_counts": [
            parameter_count(critic) for critic in critic_targets
        ],
        "vf_parameter_count": parameter_count(vf),
        "independent_raw_obs_encoders": True,
        "encoder_initialization": encoder_initialization,
        "critic_architecture": TEMPORAL_CRITIC_ARCHITECTURE,
        "critic_observation_horizon": int(args.critic_observation_horizon),
        "causal_temporal_trunk": True,
        "state_action_fusion": str(args.rise_v2_fusion_mode),
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
    }
    startup = {
        "task": str(args.task),
        "actor_initialization": {
            "checkpoint": str(args.checkpoint),
            "loaded_with_policy_from_checkpoint": True,
            "trainable_actor_initialized_from_deployed_ema": True,
            "exact_state_match_verified": True,
        },
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
        },
        "data_routing": {
            "shared_loader": True,
            "actor_rows": "all_human_success_failure_no_mask",
            "critic_rows": "all_human_success_failure",
            "actor_filtered_rows": 0,
        },
        "loader": {
            "class": dataset.__class__.__name__,
            "sparse_one_step_loader": bool(args.sparse_one_step_loader),
            "observation_loading": (
                "actor_observation_history_plus_two_frame_successor_critic_history"
                if args.sparse_one_step_loader
                else "full_obs_and_next_obs_sequences"
            ),
            "num_loaders": int(distributed.world_size),
            "sampler": loader.sampler.__class__.__name__,
            "balanced_sampling": False,
            "batch_size": int(args.batch_size),
            "batch_size_per_rank": int(args.batch_size),
            "effective_global_batch_size": int(
                args.effective_global_batch_size
            ),
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
                "DataLoader_length_after_distributed_sharding_and_drop_last"
            ),
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
            "reference_lr_warmup_steps": int(args.lr_warmup_steps),
            "resolved_lr_warmup_steps": int(args.resolved_lr_warmup_steps),
            "target_tau_step_unit": "optimizer_update",
            "actor_ema_step_unit": "optimizer_update",
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
            "actor_unet_lr": float(args.actor_unet_lr),
            "actor_obs_encoder_lr": float(args.actor_obs_encoder_lr),
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
            "vf_lr": float(args.vf_lr),
            "vf_encoder_freeze_steps": int(args.vf_encoder_freeze_steps),
            "resolved_vf_encoder_freeze_steps": int(
                args.resolved_vf_encoder_freeze_steps
            ),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "resolved_lr_warmup_steps": int(args.resolved_lr_warmup_steps),
            "lr_num_cycles": float(args.lr_num_cycles),
            "lr_total_steps": int(args.lr_total_steps),
            "lr_scheduler_step_unit": "optimizer_update",
        },
        "horizons": {
            "observation": int(actor_algo.algo_config.horizon.observation_horizon),
            "action": int(actor_algo.algo_config.horizon.action_horizon),
            "prediction": int(actor_algo.algo_config.horizon.prediction_horizon),
            "critic": 1,
            "critic_observation": int(args.critic_observation_horizon),
        },
        "critic_action_contract": {
            "input": "actions[:, observation_horizon - 1]",
            "shape": ["batch", int(actor_algo.ac_dim)],
            "uses_action_chunk": False,
        },
        "latent_dynamics": {
            "enabled": dynamics_enabled,
            "loss_weight": float(args.dynamics_weight),
            "prediction_offset": 1 if dynamics_enabled else None,
            "target": (
                "immediate_next_visual_actor_encoder_features"
                if dynamics_enabled
                else None
            ),
            "target_encoder": (
                "frozen_deployed_pretrained_dp_visual_encoder"
                if dynamics_enabled
                else False
            ),
        },
        "distributed": {
            "enabled": bool(distributed.enabled),
            "world_size": int(distributed.world_size),
            "backend": distributed.backend,
            "launcher": "torchrun" if distributed.enabled else "python",
            "gradient_sync": "bounded_async_bucketed_mean_all_reduce",
            "gradient_bucket_cap_mb": float(args.gradient_bucket_cap_mb),
            "per_step_buffer_broadcast": bool(
                synchronize_training_buffers
            ),
            "rank_zero_writes_only": True,
        },
    }
    if distributed.is_main_process:
        write_json(args.output_dir / "training_config.json", startup)
        print(json.dumps(jsonable(startup), indent=2), flush=True)
        writer = make_tensorboard_writer(args.output_dir)
    else:
        writer = None
    max_gradient_norm = (
        None
        if args.max_gradient_norm is None or float(args.max_gradient_norm) <= 0.0
        else float(args.max_gradient_norm)
    )

    best_validation_loss: float | None = None
    best_validation_epoch: int | None = None
    for completed_epoch in history:
        candidate = completed_epoch.get("metrics", {}).get(
            "validation/actor/Loss"
        )
        if candidate is None or not np.isfinite(float(candidate)):
            continue
        if (
            best_validation_loss is None
            or float(candidate) < best_validation_loss
        ):
            best_validation_loss = float(candidate)
            best_validation_epoch = int(completed_epoch.get("epoch", -1))

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        if distributed.enabled and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        epoch_iterator = iter(loader)
        epoch_records: list[dict[str, float]] = []
        for step_in_epoch in range(1, int(args.steps_per_epoch) + 1):
            try:
                raw_batch = next(epoch_iterator)
            except StopIteration:
                epoch_iterator = iter(loader)
                raw_batch = next(epoch_iterator)
            raw_batch = align_shared_batch_actions(raw_batch)
            actor_obs_encoder_trainable = global_step >= int(
                args.resolved_actor_obs_encoder_freeze_steps
            )
            critic_encoders_trainable = global_step >= int(
                args.resolved_encoder_freeze_steps
            )
            vf_encoder_trainable = global_step >= int(
                args.resolved_vf_encoder_freeze_steps
            )
            set_actor_obs_encoder_trainable(
                actor_algo,
                actor_obs_encoder_trainable,
            )
            set_critic_encoders_trainable(
                critics,
                critic_encoders_trainable,
            )
            set_vf_encoder_trainable(vf, vf_encoder_trainable)
            if synchronize_training_buffers:
                broadcast_module_buffers(
                    training_buffer_modules,
                    distributed,
                )
            actor_learning_rates = {
                str(group.get("group_name", "policy")): float(group["lr"])
                for group in actor_algo.optimizers["policy"].param_groups
            }
            actor_policy_lr = (
                actor_learning_rates["policy"]
                if "policy" in actor_learning_rates
                else actor_learning_rates["noise_pred_net"]
            )
            learning_rates_used = {
                "actor_unet": actor_learning_rates.get(
                    "noise_pred_net", actor_policy_lr
                ),
                "actor_obs_encoder": actor_learning_rates.get(
                    "obs_encoder", actor_policy_lr
                ),
                "critic": float(critic_optimizers[0].param_groups[0]["lr"]),
                "encoder": float(critic_optimizers[0].param_groups[1]["lr"]),
                "vf": float(vf_optimizer.param_groups[0]["lr"]),
            }

            critic_batch = process_critic_batch(
                raw_batch,
                actor_algo,
                obs_normalization_stats,
                critic_observation_horizon=args.critic_observation_horizon,
            )
            critic_losses, vf_loss, critic_info = compute_temporal_one_step_losses(
                critics,
                critic_targets,
                vf,
                dynamics_target_encoder,
                critic_batch,
                discount=args.discount,
                expectile=args.expectile,
                use_huber=args.use_huber,
                dynamics_weight=args.dynamics_weight,
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
                gradient_sync_fn=gradient_sync_fn,
            )
            del critic_batch, critic_losses, vf_loss

            actor_info = actor_train_step(
                actor_algo,
                raw_batch,
                epoch,
                obs_normalization_stats,
                defer_scalar_conversion=True,
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
                }
            )
            global_samples_seen += int(
                raw_batch["actions"].shape[0] * distributed.world_size
            )
            global_step += 1
            metrics = dict(critic_info)
            metrics.update(actor_info)
            metrics["actor/obs_encoder_trainable"] = float(
                actor_obs_encoder_trainable
            )
            metrics["critic/encoder_trainable"] = float(
                critic_encoders_trainable
            )
            metrics["vf/encoder_trainable"] = float(vf_encoder_trainable)
            metrics["lr/actor_unet"] = learning_rates_used["actor_unet"]
            metrics["lr/actor_obs_encoder"] = learning_rates_used[
                "actor_obs_encoder"
            ]
            metrics["lr/critic"] = learning_rates_used["critic"]
            metrics["lr/encoder"] = learning_rates_used["encoder"]
            metrics["lr/vf"] = learning_rates_used["vf"]
            metrics["distributed/world_size"] = float(distributed.world_size)
            metrics["data/effective_global_batch_rows"] = float(
                raw_batch["actions"].shape[0] * distributed.world_size
            )
            metrics = mean_distributed_scalars(
                metrics,
                distributed,
                reductions={
                    "data/action_min": "min",
                    "data/action_max": "max",
                },
            )
            epoch_records.append(metrics)

            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, global_step)
            if (
                distributed.is_main_process
                and global_step % int(args.log_every) == 0
            ):
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

        epoch_metrics = mean_metrics(epoch_records)
        new_best_validation = False
        if validation_loader is not None:
            validation_metrics = evaluate_validation_epoch(
                args=args,
                epoch=epoch,
                loader=validation_loader,
                actor_algo=actor_algo,
                critics=critics,
                critic_targets=critic_targets,
                vf=vf,
                dynamics_target_encoder=dynamics_target_encoder,
                obs_normalization_stats=obs_normalization_stats,
                device=device,
            )
            epoch_metrics.update(validation_metrics)
            if writer is not None:
                for key, value in validation_metrics.items():
                    writer.add_scalar(key, value, global_step)
            selection_loss = validation_metrics.get("validation/actor/Loss")
            if selection_loss is None or not np.isfinite(float(selection_loss)):
                raise ValueError(
                    "held-out validation did not produce a finite EMA actor loss"
                )
            if (
                best_validation_loss is None
                or float(selection_loss) < best_validation_loss
            ):
                best_validation_loss = float(selection_loss)
                best_validation_epoch = int(epoch)
                new_best_validation = True
        if distributed.enabled:
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
            "history": history,
            "best_validation": {
                "metric": "validation/actor/Loss",
                "value": best_validation_loss,
                "epoch": best_validation_epoch,
            },
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
            write_json(args.output_dir / "partial_summary.json", partial_summary)

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
                    vf=vf,
                    dynamics_target_encoder=dynamics_target_encoder,
                    critic_optimizers=critic_optimizers,
                    vf_optimizer=vf_optimizer,
                    critic_lr_schedulers=critic_lr_schedulers,
                    vf_lr_scheduler=vf_lr_scheduler,
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
                    snapshot = (
                        args.output_dir / "models" / f"model_epoch_{epoch}.pt"
                    )
                    replace_with_hardlink(latest_path, snapshot)
                if writer is not None:
                    writer.flush()
                print(
                    f"Saved {latest_path} at epoch={epoch} step={global_step}",
                    flush=True,
                )
        if distributed.enabled:
            dist.barrier()

    if distributed.is_main_process:
        last_path = args.output_dir / "last.pt"
        if int(args.epochs) > start_epoch:
            latest_path = args.output_dir / "latest.pt"
            replace_with_hardlink(latest_path, last_path)
        elif not last_path.exists():
            if args.resume_checkpoint is None:
                raise RuntimeError(
                    "training was already complete but last.pt is missing"
                )
            replace_with_hardlink(args.resume_checkpoint, last_path)
        final_summary = json.loads(
            (args.output_dir / "partial_summary.json").read_text()
        )
        final_summary["complete"] = True
        final_summary["checkpoints"]["last"] = str(last_path)
        write_json(args.output_dir / "summary.json", final_summary)
        print(f"Training complete: {last_path}", flush=True)
    else:
        final_summary = {}
    if writer is not None:
        writer.close()
    if distributed.enabled:
        dist.barrier()
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help="Optional disjoint mixed HDF5 evaluated in full after every epoch.",
    )
    parser.add_argument("--validation-seed", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        choices=("square", "can", "transport", "tool_hang", "pick_cup", "stack_cup"),
        default="square",
    )
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
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Defaults to len(train_loader), i.e. one full uniformly shuffled dataset pass.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--schedule-reference-batch-size",
        type=int,
        default=64,
        help=(
            "Reference global batch used to express the LR warmup in "
            "processed-sample units."
        ),
    )
    parser.add_argument(
        "--sparse-one-step-loader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load only the critic's current two-frame history, immediate "
            "successor history, and current action."
        ),
    )
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
        default="terminal_success",
        help="Expected dataset reward mode; terminal_success is the default.",
    )
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument(
        "--actor-unet-lr",
        type=float,
        default=None,
        help="Diffusion U-Net LR; defaults to --actor-lr for compatibility.",
    )
    parser.add_argument(
        "--actor-obs-encoder-lr",
        type=float,
        default=None,
        help="Actor observation-encoder LR; defaults to --actor-lr.",
    )
    parser.add_argument(
        "--actor-obs-encoder-freeze-steps",
        type=int,
        default=0,
        help="Reference-batch optimizer steps to freeze the actor encoder.",
    )
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument(
        "--encoder-freeze-steps",
        type=int,
        default=0,
        help="Reference-batch optimizer steps to freeze online Q encoders.",
    )
    parser.add_argument("--vf-lr", type=float, default=1e-4)
    parser.add_argument(
        "--vf-encoder-freeze-steps",
        type=int,
        default=0,
        help="Reference-batch optimizer steps to freeze the V encoder.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default="cosine",
    )
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
    parser.add_argument(
        "--dynamics-weight",
        type=float,
        default=0.0,
        help=(
            "Optional t+1 visual latent Smooth-L1 weight. Zero creates no "
            "dynamics head and performs no target-encoder forward pass."
        ),
    )
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
    args.actor_grouped_optimizer = bool(
        args.actor_unet_lr is not None
        or args.actor_obs_encoder_lr is not None
        or int(args.actor_obs_encoder_freeze_steps) > 0
    )
    if args.actor_unet_lr is None:
        args.actor_unet_lr = float(args.actor_lr)
    if args.actor_obs_encoder_lr is None:
        args.actor_obs_encoder_lr = float(args.actor_lr)
    for key in (
        "actor_obs_encoder_freeze_steps",
        "encoder_freeze_steps",
        "vf_encoder_freeze_steps",
    ):
        if int(getattr(args, key)) < 0:
            parser.error(f"--{key.replace('_', '-')} must be non-negative")
    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    if not args.critic_late_fusion_key:
        args.critic_late_fusion_key = None
    if args.lr_warmup_steps < 0:
        parser.error("lr-warmup-steps must be non-negative")
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
