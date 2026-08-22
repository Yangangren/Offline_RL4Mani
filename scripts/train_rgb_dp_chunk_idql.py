#!/usr/bin/env python3
"""RISE-style RGB chunk IDQL with a conditional joint DP actor by default."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
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
from robomimic.algo.diffusion_policy import replace_bn_with_gn
from robomimic.models.chunk_iql_nets import (
    CausalTemporalStateTrunk,
    ResidualActionLatentRollout,
    SequentialActionChunkEncoder,
    make_mlp,
)
from robomimic.models.obs_core import CropRandomizer

from rgb_dp_distributed import (
    all_reduce_gradients as bounded_all_reduce_gradients,
    mean_distributed_scalars as reduce_distributed_scalars,
)

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
    build_single_loader as _build_single_loader,
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
ACTOR_CONDITION_DEFINITIONS = {
    "human_only": "human_demo=1; success_rollout=0; failure_rollout=0",
    "human_success": "human_demo=1; success_rollout=1; failure_rollout=0",
}
DYNAMICS_PREDICTION_MODE = "actor_encoder_direct"
WCM_DYNAMICS_PREDICTION_MODE = "shared_critic_latent_multi_offset_residual"
WCM_DYNAMICS_TARGET_MODE = "periodic_hard_copy_frame_encoder_v1"
LEGACY_WCM_DYNAMICS_TARGET_MODE = "online_stop_gradient_frame_encoder_v0"
WCM_DYNAMICS_TARGET_STATE_KEY = "wcm_dynamics_frame_target"
PREDICTED_NEXT_Q_NORMALIZATION = "layer_norm"
LEGACY_CRITIC_ARCHITECTURE = "legacy"
WCM_CRITIC_ARCHITECTURE = "wcm_shared_temporal_v1"
CRITIC_ARCHITECTURES = (
    LEGACY_CRITIC_ARCHITECTURE,
    WCM_CRITIC_ARCHITECTURE,
)
DEFAULT_WCM_DYNAMICS_TARGET_SYNC_INTERVAL = 500
DEFAULT_LEGACY_DYNAMICS_TARGET_SYNC_INTERVAL = 1000
ACTOR_OPTIMIZER_TYPE = "adamw"
ACTOR_WEIGHT_DECAY = 1e-6
JOINT_ACTOR_INITIALIZATIONS = frozenset(
    ("pretrained_dp_joint", "source_chunk_idql_joint")
)
LATEST_CHECKPOINT_NAME = "latest.pt"
LATEST_CHECKPOINT_TEMP_PREFIX = f".{LATEST_CHECKPOINT_NAME}.tmp-"


def checkpoint_for_unfiltered_mixed_dataset(dp_checkpoint: dict) -> dict:
    """Return a lightweight checkpoint view with inherited HDF5 splits off.

    Diffusion Policy checkpoints retain the ``train`` / ``valid`` filter keys
    from their original behavior-cloning dataset. Chunk-IDQL receives an
    already selected mixed dataset, whose complete set of trajectories is the
    training population. Reusing the old filters would therefore either drop
    rollout trajectories or fail when the old masks are absent.

    Only the serialized config is copied. Model tensors and normalization
    statistics remain shared with the read-only checkpoint, avoiding a second
    in-memory copy of a large RGB policy.
    """
    if not isinstance(dp_checkpoint, dict):
        raise TypeError("dp_checkpoint must be a dictionary")
    serialized_config = dp_checkpoint.get("config")
    if not isinstance(serialized_config, str):
        raise TypeError("dp_checkpoint['config'] must be serialized JSON")
    try:
        config = json.loads(serialized_config)
    except json.JSONDecodeError as exc:
        raise ValueError("dp_checkpoint contains invalid config JSON") from exc
    if not isinstance(config, dict):
        raise TypeError("dp_checkpoint config JSON must decode to an object")
    train_config = config.get("train")
    experiment_config = config.get("experiment")
    if not isinstance(train_config, dict):
        raise ValueError("dp_checkpoint config is missing the train object")
    if not isinstance(experiment_config, dict):
        raise ValueError("dp_checkpoint config is missing the experiment object")

    train_config["hdf5_filter_key"] = None
    train_config["hdf5_validation_filter_key"] = None
    data_configs = train_config.get("data", [])
    if not isinstance(data_configs, list):
        raise TypeError("dp_checkpoint config train.data must be a list")
    for data_config in data_configs:
        if not isinstance(data_config, dict):
            raise TypeError(
                "dp_checkpoint config train.data entries must be objects"
            )
        data_config.pop("filter_key", None)
    # A preselected mixed dataset has no separate validation loader. Keeping
    # this flag enabled would make robomimic require the filters cleared above.
    experiment_config["validate"] = False

    loader_checkpoint = dict(dp_checkpoint)
    loader_checkpoint["config"] = json.dumps(config)
    return loader_checkpoint


def build_single_loader(
    args: argparse.Namespace,
    actor_policy,
    dp_checkpoint: dict,
    sequence_length: int | None = None,
):
    """Build the mixed-data loader without checkpoint-era split filters."""
    return _build_single_loader(
        args,
        actor_policy,
        checkpoint_for_unfiltered_mixed_dataset(dp_checkpoint),
        sequence_length=sequence_length,
    )


def critic_q_head_inputs(
    use_predicted_next_latent: bool,
) -> tuple[str, ...]:
    inputs = ("context", "action_repr")
    if bool(use_predicted_next_latent):
        inputs += ("predicted_next_encoder",)
    return inputs


def architecture_q_head_inputs(
    architecture: str,
    use_predicted_next_latent: bool = False,
) -> tuple[str, ...]:
    if str(architecture) == WCM_CRITIC_ARCHITECTURE:
        if bool(use_predicted_next_latent):
            raise ValueError("WCM Q cannot consume a predicted-next latent")
        return ("temporal_state", "action_repr")
    return critic_q_head_inputs(use_predicted_next_latent)


def checkpoint_critic_architecture(checkpoint: dict) -> str:
    """Legacy checkpoints predate an explicit architecture marker."""
    return str(
        checkpoint.get(
            "critic_architecture",
            checkpoint.get("args", {}).get(
                "critic_architecture",
                LEGACY_CRITIC_ARCHITECTURE,
            ),
        )
    )


def checkpoint_wcm_dynamics_target_mode(checkpoint: dict) -> str | None:
    """Infer the old online-stop-gradient contract for legacy WCM checkpoints."""
    if checkpoint_critic_architecture(checkpoint) != WCM_CRITIC_ARCHITECTURE:
        return None
    return str(
        checkpoint.get(
            "wcm_dynamics_target_mode",
            LEGACY_WCM_DYNAMICS_TARGET_MODE,
        )
    )


def configure_critic_architecture_args(args: argparse.Namespace) -> None:
    """Install backward-compatible defaults and validate WCM-only contracts."""
    defaults = {
        "critic_architecture": LEGACY_CRITIC_ARCHITECTURE,
        "temporal_num_layers": 2,
        "temporal_num_heads": 6,
        "temporal_feedforward_dim": 600,
        "temporal_dropout": 0.0,
        "dynamics_prediction_offsets": (2, 4, 6, 8),
        "sigreg_weight": 0.0,
        "sigreg_knots": 17,
        "sigreg_num_projections": 1024,
        "sigreg_global_batch": True,
    }
    for field, default in defaults.items():
        if not hasattr(args, field):
            setattr(args, field, default)
    architecture = str(args.critic_architecture)
    if architecture not in CRITIC_ARCHITECTURES:
        raise ValueError(f"unsupported critic architecture: {architecture!r}")
    if getattr(args, "dynamics_target_sync_interval", None) is None:
        args.dynamics_target_sync_interval = int(
            DEFAULT_WCM_DYNAMICS_TARGET_SYNC_INTERVAL
            if architecture == WCM_CRITIC_ARCHITECTURE
            else DEFAULT_LEGACY_DYNAMICS_TARGET_SYNC_INTERVAL
        )
    offsets = tuple(int(value) for value in args.dynamics_prediction_offsets)
    if architecture == LEGACY_CRITIC_ARCHITECTURE:
        # Do not make old training load four unused RGB targets.
        args.dynamics_prediction_offsets = ()
        return
    if bool(getattr(args, "critic_q_use_predicted_next_latent", False)):
        raise ValueError(
            "WCM Q heads cannot consume the predicted-next latent; use "
            "--no-critic-q-use-predicted-next-latent"
        )
    if float(getattr(args, "dynamics_cosine_weight", 0.0)) != 0.0:
        raise ValueError(
            "WCM uses raw latent MSE; set --dynamics-cosine-weight 0"
        )
    if (
        not offsets
        or tuple(sorted(set(offsets))) != offsets
        or offsets[0] < 1
        or offsets[-1] > int(args.chunk_horizon)
    ):
        raise ValueError(
            "WCM dynamics offsets must be sorted, unique, positive, and <= "
            f"chunk_horizon={args.chunk_horizon}; got {offsets}"
        )
    if int(args.temporal_num_layers) < 1:
        raise ValueError("temporal_num_layers must be positive")
    if int(args.temporal_num_heads) < 1:
        raise ValueError("temporal_num_heads must be positive")
    if int(args.latent_dim) % int(args.temporal_num_heads) != 0:
        raise ValueError(
            f"latent_dim={args.latent_dim} must be divisible by "
            f"temporal_num_heads={args.temporal_num_heads}"
        )
    if int(args.temporal_feedforward_dim) < 1:
        raise ValueError("temporal_feedforward_dim must be positive")
    if not 0.0 <= float(args.temporal_dropout) < 1.0:
        raise ValueError("temporal_dropout must be in [0, 1)")
    if float(args.sigreg_weight) < 0.0:
        raise ValueError("sigreg_weight must be non-negative")
    if int(args.sigreg_knots) < 2 or int(args.sigreg_num_projections) < 1:
        raise ValueError("SIGReg requires at least 2 knots and 1 projection")
    if int(args.vf_encoder_freeze_steps) != int(args.encoder_freeze_steps):
        raise ValueError(
            "WCM has one shared raw encoder, so vf_encoder_freeze_steps must "
            "equal encoder_freeze_steps"
        )
    args.dynamics_prediction_offsets = offsets


def checkpoint_critic_observation_horizon(checkpoint: dict) -> int:
    return int(
        checkpoint.get(
            "critic_observation_horizon",
            checkpoint.get("args", {}).get(
                "critic_observation_horizon",
                1,
            ),
        )
    )


def checkpoint_q_uses_predicted_next_latent(checkpoint: dict) -> bool:
    return bool(
        checkpoint.get(
            "critic_q_use_predicted_next_latent",
            checkpoint.get("args", {}).get(
                "critic_q_use_predicted_next_latent",
                False,
            ),
        )
    )


def actor_condition_definition(mode: str) -> str:
    return ACTOR_CONDITION_DEFINITIONS[str(mode)]


def actor_condition_sources(mode: str) -> tuple[list[str], list[str]]:
    if mode == "human_only":
        return ["human_demo"], ["success_rollout", "failure_rollout"]
    if mode == "human_success":
        return ["human_demo", "success_rollout"], ["failure_rollout"]
    raise ValueError(f"unsupported actor condition mode: {mode}")


def actor_condition_labels(mode: str) -> dict[str, float]:
    positive, _ = actor_condition_sources(mode)
    return {
        source: float(source in positive)
        for source in ("human_demo", "success_rollout", "failure_rollout")
    }


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(args: argparse.Namespace) -> DistributedContext:
    """Initialize one torchrun process per GPU, while preserving serial use."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_by_torchrun = all(
        name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    requested = bool(getattr(args, "distributed", False) or env_world_size > 1)
    if not requested:
        device = TorchUtils.get_torch_device(
            try_to_use_cuda=args.device == "cuda"
        )
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend="none",
            device=device,
        )
    if not launched_by_torchrun:
        raise RuntimeError(
            "distributed training must be launched with torchrun (or "
            "python -m torch.distributed.run)"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    cli_local_rank = getattr(args, "local_rank", None)
    if cli_local_rank is not None and int(cli_local_rank) != local_rank:
        raise RuntimeError(
            f"--local-rank={cli_local_rank} disagrees with "
            f"LOCAL_RANK={local_rank}"
        )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA training requested without CUDA")
        device_count = int(torch.cuda.device_count())
        if local_rank < 0 or local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the {device_count} visible "
                "CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", local_rank)
        default_backend = "nccl"
    else:
        device = torch.device("cpu")
        default_backend = "gloo"
    requested_backend = str(getattr(args, "distributed_backend", "auto"))
    backend = default_backend if requested_backend == "auto" else requested_backend
    if backend == "nccl" and device.type != "cuda":
        raise ValueError("the NCCL distributed backend requires --device cuda")
    if backend != "nccl" and device.type == "cuda":
        raise ValueError("distributed CUDA training requires the NCCL backend")
    dist.init_process_group(backend=backend, init_method="env://")
    context = DistributedContext(
        enabled=True,
        rank=int(dist.get_rank()),
        local_rank=local_rank,
        world_size=int(dist.get_world_size()),
        backend=backend,
        device=device,
    )
    if context.world_size != env_world_size:
        raise RuntimeError(
            f"initialized world_size={context.world_size}, expected {env_world_size}"
        )
    return context


SAMPLE_SCALED_STEP_FIELDS = (
    "actor_lr_warmup_steps",
    "critic_vf_lr_warmup_steps",
    "dynamics_warmup_steps",
    "encoder_freeze_steps",
    "vf_encoder_freeze_steps",
)


def batch_scaled_step_count(
    reference_steps,
    reference_batch_size,
    effective_batch_size,
):
    """Translate a reference-batch step count to the same sample count."""
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


def configure_batch_semantics(
    args: argparse.Namespace,
    context: DistributedContext,
) -> None:
    """Resolve sample-timed schedules without rescaling target tracking."""
    reference_batch_size = int(args.schedule_reference_batch_size)
    effective_batch_size = int(args.batch_size) * int(context.world_size)
    if reference_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("reference and effective batch sizes must be positive")
    args.effective_global_batch_size = effective_batch_size
    args.schedule_batch_ratio = (
        float(effective_batch_size) / float(reference_batch_size)
    )
    for field in SAMPLE_SCALED_STEP_FIELDS:
        setattr(
            args,
            f"resolved_{field}",
            batch_scaled_step_count(
                getattr(args, field),
                reference_batch_size,
                effective_batch_size,
            ),
        )
    # These control how learned parameters are tracked, so they remain in
    # optimizer-update units. A large global batch is not equivalent to
    # several sequential optimizer updates.
    args.resolved_target_tau = float(args.target_tau)
    args.resolved_dynamics_target_sync_interval = int(
        args.dynamics_target_sync_interval
    )


def modules_have_mutable_batch_norm(modules) -> bool:
    return any(
        isinstance(layer, nn.modules.batchnorm._BatchNorm)
        and bool(layer.track_running_stats)
        for module in modules
        for layer in module.modules()
    )


def seed_process(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    # Avoid torch.manual_seed here: it seeds every visible CUDA generator and
    # can make each torchrun process touch GPUs owned by other local ranks.
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))


def capture_process_rng_state(device: torch.device) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda_local"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_process_rng_state(
    state: dict[str, Any] | None,
    device: torch.device,
) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda_local" in state:
        torch.cuda.set_rng_state(state["cuda_local"].cpu(), device=device)


@torch.no_grad()
def broadcast_module_state(
    modules: list[nn.Module],
    context: DistributedContext,
) -> None:
    if not context.enabled:
        return
    for module in modules:
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


@torch.no_grad()
def broadcast_module_buffers(
    modules: list[nn.Module],
    context: DistributedContext,
) -> None:
    if not context.enabled:
        return
    for module in modules:
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


@torch.no_grad()
def all_reduce_gradients(
    parameters,
    context: DistributedContext,
    bucket_cap_mb: float = 25.0,
    preserve_unused_parameters: bool = True,
) -> None:
    """Average gradients with a bounded window of flat async buckets."""
    bounded_all_reduce_gradients(
        parameters,
        context,
        bucket_cap_mb=bucket_cap_mb,
        preserve_unused_parameters=preserve_unused_parameters,
    )


def mean_distributed_scalars(
    metrics: dict[str, Any],
    context: DistributedContext,
    reductions: dict[str, str] | None = None,
) -> dict[str, float]:
    """Reduce metrics by mean unless a per-key operation is supplied."""
    return reduce_distributed_scalars(
        metrics,
        context,
        reductions=reductions,
    )


def gather_rank_runtime_states(
    loader_generator: torch.Generator,
    context: DistributedContext,
) -> list[dict[str, Any]]:
    local_state = {
        "rank": int(context.rank),
        "rng_state": capture_process_rng_state(context.device),
        "loader_generator_state": loader_generator.get_state(),
    }
    if not context.enabled:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather all distributed RNG states")
    return [state for state in gathered if state is not None]


def trains_joint_actor(args: argparse.Namespace) -> bool:
    return str(args.initialization) in JOINT_ACTOR_INITIALIZATIONS


def observation_history_frames(
    obs_dict: dict[str, torch.Tensor],
    obs_shapes: OrderedDict,
    observation_horizon: int,
) -> list[dict[str, torch.Tensor]]:
    """Validate and split a critic observation history in chronological order."""
    horizon = int(observation_horizon)
    frames = [dict() for _ in range(horizon)]
    batch_size: int | None = None
    for key, shape in obs_shapes.items():
        if key not in obs_dict:
            raise KeyError(f"critic observation is missing key {key!r}")
        value = obs_dict[key]
        unstacked_ndim = len(shape) + 1
        if horizon == 1 and value.ndim == unstacked_ndim:
            history = value.unsqueeze(1)
        elif (
            value.ndim == unstacked_ndim + 1
            and int(value.shape[1]) == horizon
        ):
            history = value
        else:
            raise ValueError(
                f"critic observation {key!r} expected [B,{horizon},"
                f"{','.join(str(x) for x in shape)}] (or no time axis when "
                f"horizon=1), got {tuple(value.shape)}"
            )
        if tuple(history.shape[2:]) != tuple(shape):
            raise ValueError(
                f"critic observation {key!r} has trailing shape "
                f"{tuple(history.shape[2:])}, expected {tuple(shape)}"
            )
        if batch_size is None:
            batch_size = int(history.shape[0])
        elif int(history.shape[0]) != batch_size:
            raise ValueError("critic observation keys have different batch sizes")
        for frame_index in range(horizon):
            frames[frame_index][key] = history[:, frame_index]
    return frames


def encode_observation_history(
    encoder: nn.Module,
    frames: list[dict[str, torch.Tensor]],
    *,
    has_goal: bool,
    goal_dict: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Encode newest first for paired crops, then concatenate old-to-new."""
    encoded_by_time: list[torch.Tensor | None] = [None] * len(frames)
    encode_order = (len(frames) - 1, *range(len(frames) - 1))
    for frame_index in encode_order:
        inputs = {"obs": frames[frame_index]}
        if has_goal:
            if goal_dict is None:
                raise ValueError(
                    "goal-conditioned chunk critic is missing goal observations"
                )
            inputs["goal"] = goal_dict
        encoded_by_time[frame_index] = encoder(**inputs)
    if any(feature is None for feature in encoded_by_time):
        raise RuntimeError("failed to encode every critic observation frame")
    return torch.cat(
        [feature for feature in encoded_by_time if feature is not None],
        dim=-1,
    )


def history_late_fusion(
    frames: list[dict[str, torch.Tensor]],
    late_fusion_keys: tuple[str, ...],
) -> torch.Tensor | None:
    parts = [
        frames[frame_index][key].flatten(start_dim=1)
        for frame_index in range(len(frames))
        for key in late_fusion_keys
    ]
    return torch.cat(parts, dim=-1) if parts else None


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
        observation_horizon: int = 1,
        q_use_predicted_next_latent: bool = False,
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
        self.obs_shapes = OrderedDict(obs_shapes)
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.observation_horizon = int(observation_horizon)
        self.q_use_predicted_next_latent = bool(
            q_use_predicted_next_latent
        )
        if self.observation_horizon < 1:
            raise ValueError("critic observation horizon must be positive")
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
        late_fusion_dim *= self.observation_horizon

        context_dims = tuple(int(value) for value in hidden_dims[:-1]) + (
            self.latent_dim,
        )
        self.nets["context"] = RiseLateFusionMLP(
            input_dim=self.encoder_output_dim * self.observation_horizon,
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
        q_input_dim = 2 * self.latent_dim
        if self.q_use_predicted_next_latent:
            self.nets["predicted_next_q_norm"] = nn.LayerNorm(
                self.encoder_output_dim
            )
            q_input_dim += self.encoder_output_dim
        self.nets["q_head"] = make_mlp(
            q_input_dim,
            hidden_dims,
            1,
            dropout=float(dropout),
        )

    def encode_context(self, obs_dict, goal_dict=None) -> torch.Tensor:
        frames = observation_history_frames(
            obs_dict,
            self.obs_shapes,
            self.observation_horizon,
        )
        encoded = encode_observation_history(
            self.nets["encoder"],
            frames,
            has_goal=self.has_goal,
            goal_dict=goal_dict,
        )
        late_fusion = history_late_fusion(
            frames,
            self.late_fusion_keys,
        )
        return self.nets["context_norm"](
            self.nets["context"](encoded, late_fusion)
        )

    def predict_successor(
        self,
        context: torch.Tensor,
        action_repr: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.nets["state_action_fusion"](
            torch.cat((context, action_repr), dim=-1)
        )
        return self.nets["dynamics_predictor"](fused)

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
        action_repr = self.nets["action_encoder"](
            context, acts, action_mask
        )
        predicted_next = None
        if self.q_use_predicted_next_latent or return_aux:
            predicted_next = self.predict_successor(context, action_repr)
        q_inputs = [context, action_repr]
        if self.q_use_predicted_next_latent:
            q_inputs.append(
                self.nets["predicted_next_q_norm"](predicted_next)
            )
        q = self.nets["q_head"](torch.cat(q_inputs, dim=-1))
        if not return_aux:
            return q
        return {
            "q": q,
            "context": context,
            "action_repr": action_repr,
            "predicted_next_encoder": predicted_next,
        }


class RiseChunkValueNetwork(RiseValueNetwork):
    """History-aware V network with legacy-compatible one-frame state keys."""

    def __init__(
        self,
        *,
        obs_shapes: OrderedDict,
        hidden_dims: tuple[int, ...],
        goal_shapes: OrderedDict,
        encoder_kwargs: dict,
        late_fusion_key: str | None,
        observation_horizon: int = 1,
    ):
        super().__init__(
            obs_shapes=obs_shapes,
            hidden_dims=hidden_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
            late_fusion_key=late_fusion_key,
        )
        self.obs_shapes = OrderedDict(obs_shapes)
        self.observation_horizon = int(observation_horizon)
        if self.observation_horizon < 1:
            raise ValueError("critic observation horizon must be positive")
        if self.observation_horizon > 1:
            late_fusion_dim = sum(
                int(np.prod(obs_shapes[key]))
                for key in self.late_fusion_keys
            )
            self.nets["mlp"] = RiseLateFusionMLP(
                input_dim=(
                    int(self.nets["encoder"].output_shape()[0])
                    * self.observation_horizon
                ),
                hidden_dims=hidden_dims,
                late_fusion_dim=(
                    late_fusion_dim * self.observation_horizon
                ),
            )

    def forward(self, obs_dict, goal_dict=None):
        frames = observation_history_frames(
            obs_dict,
            self.obs_shapes,
            self.observation_horizon,
        )
        encoded = encode_observation_history(
            self.nets["encoder"],
            frames,
            has_goal=self.has_goal,
            goal_dict=goal_dict,
        )
        late_fusion = history_late_fusion(
            frames,
            self.late_fusion_keys,
        )
        features = self.nets["mlp"](encoded, late_fusion)
        return self.nets["decoder"](features)


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
    observation_horizon: int = 1,
    q_use_predicted_next_latent: bool = False,
) -> tuple[nn.ModuleList, nn.ModuleList, RiseChunkValueNetwork]:
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
            observation_horizon=int(observation_horizon),
            q_use_predicted_next_latent=bool(
                q_use_predicted_next_latent
            ),
        )
        if critic_group_norm:
            critic = replace_bn_with_gn(critic)
        critics.append(critic)
    targets = copy.deepcopy(critics)
    vf = RiseChunkValueNetwork(
        obs_shapes=actor_algo.obs_shapes,
        hidden_dims=hidden_dims,
        goal_shapes=actor_algo.goal_shapes,
        encoder_kwargs=copy.deepcopy(encoder_kwargs),
        late_fusion_key=late_fusion_key,
        observation_horizon=int(observation_horizon),
    )
    if critic_group_norm:
        vf = replace_bn_with_gn(vf)
    return critics, targets, vf


def named_crop_randomizers(encoder: nn.Module) -> list[tuple[str, CropRandomizer]]:
    return [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, CropRandomizer)
    ]


def make_wcm_temporal_crop_plan(
    encoder: nn.Module,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample one crop per trajectory and camera without touching ambient RNG."""
    if int(batch_size) < 1:
        raise ValueError("WCM crop-plan batch size must be positive")
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    plan: dict[str, torch.Tensor] = {}
    for name, randomizer in named_crop_randomizers(encoder):
        image_height = int(randomizer.input_shape[1])
        image_width = int(randomizer.input_shape[2])
        max_height = image_height - int(randomizer.crop_height)
        max_width = image_width - int(randomizer.crop_width)
        if max_height <= 0 or max_width <= 0:
            raise ValueError(
                f"invalid crop geometry for {name!r}: input="
                f"{tuple(randomizer.input_shape)}, crop="
                f"({randomizer.crop_height},{randomizer.crop_width})"
            )
        shape = (int(batch_size), int(randomizer.num_crops))
        height = (
            max_height
            * torch.rand(
                shape,
                generator=generator,
                device=generator_device,
            )
        ).to(dtype=torch.long)
        width = (
            max_width
            * torch.rand(
                shape,
                generator=generator,
                device=generator_device,
            )
        ).to(dtype=torch.long)
        plan[name] = torch.stack((height, width), dim=-1)
    return plan


@contextmanager
def use_wcm_temporal_crop_plan(
    encoder: nn.Module,
    crop_plan: dict[str, torch.Tensor] | None,
    group_ids: torch.Tensor,
):
    """Apply a shared per-trajectory crop plan to one flattened encoder call."""
    randomizers = named_crop_randomizers(encoder)
    if crop_plan is None or not randomizers:
        yield
        return
    expected_names = {name for name, _ in randomizers}
    if set(crop_plan) != expected_names:
        raise ValueError(
            "WCM crop-plan randomizers do not match encoder: "
            f"plan={sorted(crop_plan)}, encoder={sorted(expected_names)}"
        )
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
        for randomizer, crop_indices, prior_group_ids in previous:
            randomizer.set_external_crop_plan(crop_indices, prior_group_ids)


class WCMChunkValueSystem(nn.Module):
    """Shared WCM-style temporal representation for chunk Q, V, and dynamics.

    Q heads never receive a predicted successor. They consume only the final
    causal temporal state and their independently encoded action chunk. The
    auxiliary world model starts from that same temporal state and predicts
    stop-gradient frame latents at fixed action-prefix offsets.
    """

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
        observation_horizon: int,
        num_critics: int,
        temporal_num_layers: int,
        temporal_num_heads: int,
        temporal_feedforward_dim: int,
        temporal_dropout: float,
        dynamics_prediction_offsets: tuple[int, ...],
    ):
        super().__init__()
        if int(observation_horizon) < 1:
            raise ValueError("critic observation horizon must be positive")
        if int(num_critics) < 2:
            raise ValueError("WCM chunk IDQL requires at least two Q heads")
        self.obs_shapes = OrderedDict(obs_shapes)
        self.goal_shapes = OrderedDict(goal_shapes or {})
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.observation_horizon = int(observation_horizon)
        self.latent_dim = int(latent_dim)
        self.num_critics = int(num_critics)
        self.dynamics_prediction_offsets = tuple(
            int(value) for value in dynamics_prediction_offsets
        )
        self.late_fusion_keys = tuple(
            key.strip()
            for key in str(late_fusion_key or "").split(",")
            if key.strip()
        )

        observation_group_shapes = OrderedDict(obs=OrderedDict(obs_shapes))
        if self.goal_shapes:
            observation_group_shapes["goal"] = self.goal_shapes
        self.has_goal = "goal" in observation_group_shapes
        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.encoder_output_dim = int(self.nets["encoder"].output_shape()[0])
        late_fusion_frame_dim = 0
        for key in self.late_fusion_keys:
            if key not in self.obs_shapes:
                raise KeyError(f"late_fusion_key={key} is absent from obs_shapes")
            late_fusion_frame_dim += int(np.prod(self.obs_shapes[key]))
        self.nets["frame_projection"] = make_mlp(
            self.encoder_output_dim + late_fusion_frame_dim,
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
        self.nets["q_action_encoders"] = nn.ModuleList(
            [
                SequentialActionChunkEncoder(
                    action_dim=self.action_dim,
                    chunk_horizon=self.chunk_horizon,
                    context_dim=self.latent_dim,
                    hidden_dim=int(action_hidden_dim),
                    output_dim=self.latent_dim,
                    num_heads=int(num_attention_heads),
                    num_conv_layers=int(num_action_conv_layers),
                    dropout=float(dropout),
                )
                for _ in range(self.num_critics)
            ]
        )
        self.nets["q_heads"] = nn.ModuleList(
            [
                make_mlp(
                    2 * self.latent_dim,
                    hidden_dims,
                    1,
                    dropout=float(dropout),
                )
                for _ in range(self.num_critics)
            ]
        )
        self.nets["value_head"] = make_mlp(
            self.latent_dim,
            hidden_dims,
            1,
            dropout=float(dropout),
        )
        self.nets["dynamics"] = ResidualActionLatentRollout(
            action_dim=self.action_dim,
            chunk_horizon=self.chunk_horizon,
            state_dim=self.latent_dim,
            prediction_offsets=self.dynamics_prediction_offsets,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
        )

    def _encode_frame(
        self,
        obs_dict: dict[str, torch.Tensor],
        goal_dict: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        inputs = {"obs": obs_dict}
        if self.has_goal:
            if goal_dict is None:
                raise ValueError(
                    "goal-conditioned WCM critic is missing goal observations"
                )
            inputs["goal"] = goal_dict
        encoded = self.nets["encoder"](**inputs)
        late_parts = [
            obs_dict[key].flatten(start_dim=1)
            for key in self.late_fusion_keys
        ]
        if late_parts:
            encoded = torch.cat((encoded, *late_parts), dim=-1)
        return self.nets["frame_projection"](encoded)

    def encode_state(
        self,
        obs_dict: dict[str, torch.Tensor],
        goal_dict: dict[str, torch.Tensor] | None = None,
        *,
        crop_plan: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        frames = observation_history_frames(
            obs_dict,
            self.obs_shapes,
            self.observation_horizon,
        )
        batch_size = int(next(iter(frames[0].values())).shape[0])
        # Encode the full history in one larger image batch. Keeping newest
        # first preserves the established frame ordering for paired online /
        # target crop draws while avoiding one encoder launch per timestep.
        encode_order = (len(frames) - 1, *range(len(frames) - 1))
        flattened_frames = {
            key: torch.cat(
                [frames[frame_index][key] for frame_index in encode_order],
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
        group_ids = torch.arange(
            batch_size,
            device=next(iter(flattened_frames.values())).device,
            dtype=torch.long,
        ).repeat(len(frames))
        with use_wcm_temporal_crop_plan(
            self.nets["encoder"],
            crop_plan,
            group_ids,
        ):
            encoded = self._encode_frame(flattened_frames, flattened_goal)
        encoded_in_order = encoded.split(batch_size, dim=0)
        frame_latents: list[torch.Tensor | None] = [None] * len(frames)
        for frame_index, latent in zip(encode_order, encoded_in_order):
            frame_latents[frame_index] = latent
        stacked = torch.stack(frame_latents, dim=1)
        temporal_tokens = self.nets["temporal_trunk"](stacked)
        return {
            "temporal_state": temporal_tokens[:, -1],
            "current_frame_latent": stacked[:, -1],
            "temporal_tokens": temporal_tokens,
        }

    @staticmethod
    def expand_state(
        state: dict[str, torch.Tensor],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key in (
            "temporal_state",
            "current_frame_latent",
            "temporal_tokens",
        ):
            value = state[key]
            if int(value.shape[0]) == int(batch_size):
                result[key] = value
            elif int(value.shape[0]) == 1:
                result[key] = value.expand(
                    int(batch_size), *([-1] * (value.ndim - 1))
                )
            else:
                raise ValueError(
                    f"cannot expand WCM state batch {value.shape[0]} to "
                    f"{batch_size}"
                )
        return result

    def q_values_from_state(
        self,
        state: dict[str, torch.Tensor],
        acts: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        temporal_state = state["temporal_state"]
        if int(temporal_state.shape[0]) != int(acts.shape[0]):
            state = self.expand_state(state, int(acts.shape[0]))
            temporal_state = state["temporal_state"]
        action_representations = [
            encoder(temporal_state, acts, action_mask)
            for encoder in self.nets["q_action_encoders"]
        ]
        return [
            head(torch.cat((temporal_state, action_repr), dim=-1))
            for head, action_repr in zip(
                self.nets["q_heads"], action_representations
            )
        ]

    def value_from_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.nets["value_head"](state["temporal_state"])

    def predict_dynamics_from_state(
        self,
        state: dict[str, torch.Tensor],
        acts: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.nets["dynamics"](
            state["temporal_state"],
            state["current_frame_latent"],
            acts,
            action_mask,
        )

    def encode_dynamics_targets(
        self,
        obs_dict: dict[str, torch.Tensor],
        goal_dict: dict[str, torch.Tensor] | None = None,
        *,
        crop_plan: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        offset_count = len(self.dynamics_prediction_offsets)
        first_key = next(iter(self.obs_shapes))
        first_value = obs_dict[first_key]
        if first_value.ndim != len(self.obs_shapes[first_key]) + 2:
            raise ValueError(
                "dynamics targets must be [B,O,...], got "
                f"{tuple(first_value.shape)} for {first_key!r}"
            )
        batch_size = int(first_value.shape[0])
        if int(first_value.shape[1]) != offset_count:
            raise ValueError(
                f"dynamics target offset count={first_value.shape[1]} does "
                f"not match {offset_count}"
            )
        flattened = {}
        for key, shape in self.obs_shapes.items():
            value = obs_dict[key]
            expected = (batch_size, offset_count, *tuple(shape))
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"dynamics target {key!r} has shape {tuple(value.shape)}, "
                    f"expected {expected}"
                )
            flattened[key] = value.reshape(batch_size * offset_count, *shape)
        flattened_goal = None
        if self.has_goal:
            if goal_dict is None:
                raise ValueError("dynamics target encoding requires goal observations")
            flattened_goal = {
                key: value.repeat_interleave(offset_count, dim=0)
                for key, value in goal_dict.items()
            }
        group_ids = torch.arange(
            batch_size,
            device=first_value.device,
            dtype=torch.long,
        ).repeat_interleave(offset_count)
        with use_wcm_temporal_crop_plan(
            self.nets["encoder"],
            crop_plan,
            group_ids,
        ):
            encoded = self._encode_frame(flattened, flattened_goal)
        return encoded.reshape(batch_size, offset_count, self.latent_dim)


class WCMFrameTargetEncoder(nn.Module):
    """Frozen observation-to-frame-latent teacher for WCM dynamics."""

    def __init__(self, source: WCMChunkValueSystem):
        super().__init__()
        self.obs_shapes = copy.deepcopy(source.obs_shapes)
        self.goal_shapes = copy.deepcopy(source.goal_shapes)
        self.has_goal = bool(source.has_goal)
        self.latent_dim = int(source.latent_dim)
        self.dynamics_prediction_offsets = tuple(
            int(value) for value in source.dynamics_prediction_offsets
        )
        self.late_fusion_keys = tuple(source.late_fusion_keys)
        self.nets = nn.ModuleDict(
            {
                "encoder": copy.deepcopy(source.nets["encoder"]),
                "frame_projection": copy.deepcopy(
                    source.nets["frame_projection"]
                ),
            }
        )

    def _encode_frame(
        self,
        obs_dict: dict[str, torch.Tensor],
        goal_dict: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        return WCMChunkValueSystem._encode_frame(self, obs_dict, goal_dict)

    def encode_dynamics_targets(
        self,
        obs_dict: dict[str, torch.Tensor],
        goal_dict: dict[str, torch.Tensor] | None = None,
        *,
        crop_plan: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return WCMChunkValueSystem.encode_dynamics_targets(
            self,
            obs_dict,
            goal_dict,
            crop_plan=crop_plan,
        )


def make_wcm_chunk_value_system(
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
    observation_horizon: int = 1,
    temporal_num_layers: int = 2,
    temporal_num_heads: int = 3,
    temporal_feedforward_dim: int = 600,
    temporal_dropout: float = 0.0,
    dynamics_prediction_offsets: tuple[int, ...] = (2, 4, 6, 8),
) -> tuple[WCMChunkValueSystem, WCMChunkValueSystem]:
    encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
        actor_algo.obs_config.encoder
    )
    system = WCMChunkValueSystem(
        obs_shapes=actor_algo.obs_shapes,
        goal_shapes=actor_algo.goal_shapes,
        encoder_kwargs=copy.deepcopy(encoder_kwargs),
        action_dim=int(actor_algo.ac_dim),
        chunk_horizon=int(chunk_horizon),
        hidden_dims=tuple(int(value) for value in hidden_dims),
        latent_dim=int(latent_dim),
        action_hidden_dim=int(action_hidden_dim),
        num_attention_heads=int(num_attention_heads),
        num_action_conv_layers=int(num_action_conv_layers),
        dropout=float(dropout),
        late_fusion_key=late_fusion_key,
        observation_horizon=int(observation_horizon),
        num_critics=int(num_critics),
        temporal_num_layers=int(temporal_num_layers),
        temporal_num_heads=int(temporal_num_heads),
        temporal_feedforward_dim=int(temporal_feedforward_dim),
        temporal_dropout=float(temporal_dropout),
        dynamics_prediction_offsets=tuple(
            int(value) for value in dynamics_prediction_offsets
        ),
    )
    if critic_group_norm:
        system = replace_bn_with_gn(system)
    return system, copy.deepcopy(system)


def make_wcm_system_from_checkpoint(
    actor_algo,
    checkpoint: dict[str, Any],
) -> tuple[WCMChunkValueSystem, WCMChunkValueSystem]:
    """Strictly reconstruct every WCM shape-changing choice for evaluation."""
    if checkpoint_critic_architecture(checkpoint) != WCM_CRITIC_ARCHITECTURE:
        raise ValueError("checkpoint is not a WCM shared-temporal critic")
    required = (
        "critic_chunk_horizon",
        "critic_hidden_dims",
        "critic_latent_dim",
        "critic_action_hidden_dim",
        "critic_num_attention_heads",
        "critic_num_action_conv_layers",
        "critic_dropout",
        "num_critics",
        "critic_group_norm",
        "critic_late_fusion_key",
        "critic_observation_horizon",
        "critic_temporal_num_layers",
        "critic_temporal_num_heads",
        "critic_temporal_feedforward_dim",
        "critic_temporal_dropout",
        "dynamics_prediction_offsets",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f"WCM checkpoint is missing architecture fields: {missing}"
        )
    if bool(checkpoint.get("critic_q_use_predicted_next_latent", False)):
        raise ValueError("WCM checkpoint illegally enables predicted-latent Q")
    expected_inputs = architecture_q_head_inputs(WCM_CRITIC_ARCHITECTURE)
    if tuple(checkpoint.get("critic_q_head_inputs", ())) != expected_inputs:
        raise ValueError(
            "WCM checkpoint Q-head metadata must be "
            f"{expected_inputs!r}"
        )
    if bool(checkpoint.get("dynamics_prediction_consumed_by_q", True)):
        raise ValueError("WCM dynamics prediction must not be consumed by Q")
    return make_wcm_chunk_value_system(
        actor_algo,
        chunk_horizon=int(checkpoint["critic_chunk_horizon"]),
        hidden_dims=tuple(int(x) for x in checkpoint["critic_hidden_dims"]),
        latent_dim=int(checkpoint["critic_latent_dim"]),
        action_hidden_dim=int(checkpoint["critic_action_hidden_dim"]),
        num_attention_heads=int(checkpoint["critic_num_attention_heads"]),
        num_action_conv_layers=int(
            checkpoint["critic_num_action_conv_layers"]
        ),
        dropout=float(checkpoint["critic_dropout"]),
        num_critics=int(checkpoint["num_critics"]),
        critic_group_norm=bool(checkpoint["critic_group_norm"]),
        late_fusion_key=checkpoint["critic_late_fusion_key"],
        observation_horizon=int(checkpoint["critic_observation_horizon"]),
        temporal_num_layers=int(checkpoint["critic_temporal_num_layers"]),
        temporal_num_heads=int(checkpoint["critic_temporal_num_heads"]),
        temporal_feedforward_dim=int(
            checkpoint["critic_temporal_feedforward_dim"]
        ),
        temporal_dropout=float(checkpoint["critic_temporal_dropout"]),
        dynamics_prediction_offsets=tuple(
            int(x) for x in checkpoint["dynamics_prediction_offsets"]
        ),
    )


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


def copy_matching_vf_encoder_state(
    vf: RiseChunkValueNetwork,
    source_state: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Warm-start only V's encoder when its history-dependent head changed."""
    destination = vf.state_dict()
    matched = {
        key: value
        for key, value in source_state.items()
        if (
            key.startswith("nets.encoder.")
            and key in destination
            and destination[key].shape == value.shape
        )
    }
    if not matched:
        raise RuntimeError("no one-step VF observation-encoder weights matched")
    vf.load_state_dict(matched, strict=False)
    return {
        "mode": "encoder_only_history_head_fresh",
        "tensor_count": int(len(matched)),
        "parameter_count": int(
            sum(value.numel() for value in matched.values())
        ),
    }



def deployed_actor_obs_encoder(actor_algo) -> nn.Module:
    """Return the EMA observation encoder used by the deployed actor."""
    actor_nets = (
        actor_algo.ema.averaged_model
        if actor_algo.ema is not None
        else actor_algo.nets
    )
    return actor_nets["policy"]["obs_encoder"]


def copy_deployed_dp_encoder_state(module: nn.Module, actor_algo) -> dict[str, int]:
    """Copy, but do not share, the deployed DP raw-observation encoder."""
    source = deployed_actor_obs_encoder(actor_algo)
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
def hard_sync_wcm_dynamics_target_encoder(
    target_encoder: WCMFrameTargetEncoder,
    source_system: WCMChunkValueSystem,
) -> dict[str, float | int]:
    """Hard-copy the complete online frame-latent coordinate system."""
    floating_difference = 0.0
    floating_reference = 0.0
    tensor_count = 0
    parameter_count = 0
    for key in ("encoder", "frame_projection"):
        source_state = source_system.nets[key].state_dict()
        target_state = target_encoder.nets[key].state_dict()
        if set(source_state) != set(target_state):
            raise ValueError(f"WCM dynamics target {key} state keys do not match")
        for name, source_value in source_state.items():
            target_value = target_state[name]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"WCM dynamics target {key}.{name} shape mismatch: "
                    f"{tuple(target_value.shape)} != {tuple(source_value.shape)}"
                )
            tensor_count += 1
            if torch.is_floating_point(source_value):
                difference = (
                    source_value.detach().float() - target_value.detach().float()
                )
                floating_difference += float(difference.square().sum())
                floating_reference += float(
                    target_value.detach().float().square().sum()
                )
        target_encoder.nets[key].load_state_dict(source_state, strict=True)
        parameter_count += sum(
            parameter.numel()
            for parameter in source_system.nets[key].parameters()
        )
    configure_wcm_dynamics_target_random_crops(target_encoder)
    relative_l2 = (
        floating_difference / max(floating_reference, 1e-12)
    ) ** 0.5
    return {
        "tensor_count": int(tensor_count),
        "parameter_count": int(parameter_count),
        "pre_sync_relative_l2": float(relative_l2),
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
    if not has_condition_adapter(actor_algo):
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


def file_stat_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {
            "path": str(resolved),
            "exists": False,
            "size": None,
            "mtime_ns": None,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_external_source(path: str, dataset_path: Path) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = dataset_path.parent / source
    return source.resolve()


def _declared_source_matches_current(
    declared: Any,
    current: dict[str, Any],
    dataset_path: Path,
) -> bool:
    """Compare the stat-bearing portion of a builder source identity."""
    if not isinstance(declared, dict) or not bool(current.get("exists")):
        return False
    try:
        declared_path = _resolve_external_source(
            str(declared["path"]), dataset_path
        )
        declared_size = int(declared["size"])
        declared_mtime_ns = int(declared["mtime_ns"])
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return (
        str(declared_path) == str(current.get("path"))
        and declared_size == int(current.get("size"))
        and declared_mtime_ns == int(current.get("mtime_ns"))
    )


def mixed_dataset_identity(path: Path) -> dict[str, Any]:
    """Identify the mixed HDF5 and the external source files it reads."""
    resolved = path.expanduser().resolve()
    identity: dict[str, Any] = {"dataset": file_stat_identity(resolved)}
    source_paths: dict[str, Any] = {}
    try:
        with h5py.File(resolved, "r") as dataset:
            has_multi_source_contract = any(
                attribute in dataset.attrs
                for attribute in ("human_sources", "rollout_source")
            )
            if has_multi_source_contract:
                if "human_sources" not in dataset.attrs:
                    raise ValueError(
                        "mixed dataset has rollout_source but no human_sources"
                    )
                if "rollout_source" not in dataset.attrs:
                    raise ValueError(
                        "mixed dataset has human_sources but no rollout_source"
                    )
                try:
                    human_sources = json.loads(
                        _hdf5_text(dataset.attrs["human_sources"])
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "mixed dataset human_sources must be a JSON list"
                    ) from exc
                if (
                    not isinstance(human_sources, list)
                    or not human_sources
                    or any(
                        not isinstance(source, str) or not source
                        for source in human_sources
                    )
                ):
                    raise ValueError(
                        "mixed dataset human_sources must be a non-empty "
                        "JSON list of paths"
                    )
                if len(set(human_sources)) != len(human_sources):
                    raise ValueError(
                        "mixed dataset human_sources contains duplicate paths"
                    )
                rollout_source = _hdf5_text(
                    dataset.attrs["rollout_source"]
                )
                if not rollout_source:
                    raise ValueError(
                        "mixed dataset rollout_source must be a path"
                    )

                human_identities = [
                    file_stat_identity(
                        _resolve_external_source(source, resolved)
                    )
                    for source in human_sources
                ]
                rollout_identity = file_stat_identity(
                    _resolve_external_source(rollout_source, resolved)
                )
                source_paths = {
                    "human_sources": human_identities,
                    "rollout_source": rollout_identity,
                }

                declared_identities = None
                declared_checks = None
                if "source_identities" in dataset.attrs:
                    try:
                        declared_identities = json.loads(
                            _hdf5_text(dataset.attrs["source_identities"])
                        )
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise ValueError(
                            "mixed dataset source_identities must be JSON"
                        ) from exc
                    if not isinstance(declared_identities, dict):
                        raise ValueError(
                            "mixed dataset source_identities must be an object"
                        )
                    declared_humans = declared_identities.get("human")
                    declared_rollout = declared_identities.get("rollout")
                    human_checks = (
                        [
                            _declared_source_matches_current(
                                declared,
                                current,
                                resolved,
                            )
                            for declared, current in zip(
                                declared_humans, human_identities
                            )
                        ]
                        if isinstance(declared_humans, list)
                        and len(declared_humans) == len(human_identities)
                        else [False] * len(human_identities)
                    )
                    declared_checks = {
                        "human_sources": human_checks,
                        "rollout_source": _declared_source_matches_current(
                            declared_rollout,
                            rollout_identity,
                            resolved,
                        ),
                    }
                identity["source_identity_manifest"] = {
                    "version": int(
                        dataset.attrs.get("source_identity_version", 0)
                    ),
                    "declared": declared_identities,
                    "matches_current": (
                        None
                        if declared_checks is None
                        else bool(
                            all(declared_checks["human_sources"])
                            and declared_checks["rollout_source"]
                        )
                    ),
                    "checks": declared_checks,
                }
            else:
                for attribute in ("expert_source", "non_expert_source"):
                    stored = dataset.attrs.get(attribute)
                    if isinstance(stored, bytes):
                        stored = stored.decode("utf-8")
                    if stored is None:
                        source_paths[attribute] = None
                        continue
                    source = _resolve_external_source(str(stored), resolved)
                    source_paths[attribute] = file_stat_identity(source)
    except OSError:
        source_paths = {
            "expert_source": None,
            "non_expert_source": None,
        }
    identity["external_sources"] = source_paths
    return identity


def validate_mixed_dataset_source_identity(identity: dict[str, Any]) -> None:
    """Reject a builder manifest whose external sources changed in place."""
    manifest = identity.get("source_identity_manifest")
    if not isinstance(manifest, dict):
        return
    if (
        manifest.get("declared") is not None
        and manifest.get("matches_current") is not True
    ):
        raise ValueError(
            "mixed dataset source identity is stale: one or more human / "
            "rollout source files no longer match the builder manifest; "
            "revalidate and rebuild the mixed dataset"
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_temporary_pid(path: Path) -> int | None:
    """Return the writer PID for a valid atomic ``latest.pt`` temporary."""
    if path.is_symlink() or not path.is_file():
        return None
    if not path.name.startswith(LATEST_CHECKPOINT_TEMP_PREFIX):
        return None
    suffix = path.name[len(LATEST_CHECKPOINT_TEMP_PREFIX) :]
    if not suffix.isdecimal():
        return None
    writer_pid = int(suffix)
    if writer_pid <= 0 or writer_pid > 2_147_483_647:
        return None
    return writer_pid


def process_is_running(pid: int) -> bool:
    """Conservatively report whether a temporary's writer may still exist."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prepare_fresh_output_directory(output_dir: Path) -> list[Path]:
    """Create an empty fresh-run directory, cleaning only stale save temps."""
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(
            f"fresh training output is not a directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(output_dir.iterdir(), key=lambda path: path.name)
    temporary_entries: list[tuple[Path, int]] = []
    unexpected_entries: list[Path] = []
    for entry in entries:
        writer_pid = checkpoint_temporary_pid(entry)
        if writer_pid is None:
            unexpected_entries.append(entry)
        else:
            temporary_entries.append((entry, writer_pid))
    if unexpected_entries:
        names = [entry.name for entry in unexpected_entries]
        raise FileExistsError(
            f"refusing fresh training in non-empty output dir: {output_dir}; "
            f"unexpected entries={names}"
        )
    active_temporaries = [
        entry.name
        for entry, writer_pid in temporary_entries
        if process_is_running(writer_pid)
    ]
    if active_temporaries:
        raise FileExistsError(
            f"refusing to remove active checkpoint temporaries in {output_dir}: "
            f"{active_temporaries}"
        )
    cleaned = [entry for entry, _ in temporary_entries]
    for entry in cleaned:
        entry.unlink()
    return cleaned


def validate_resume_semantics(
    resume_state: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Reject resumes that would silently change the task or objective."""
    saved_args = resume_state.get("args", {})
    requested_critic_observation_horizon = int(
        getattr(args, "critic_observation_horizon", 1)
    )
    requested_q_uses_predicted_next = bool(
        getattr(args, "critic_q_use_predicted_next_latent", False)
    )
    requested_architecture = str(
        getattr(args, "critic_architecture", LEGACY_CRITIC_ARCHITECTURE)
    )
    saved_architecture = checkpoint_critic_architecture(resume_state)
    if saved_architecture != requested_architecture:
        raise ValueError(
            f"resume critic_architecture={saved_architecture!r} does not "
            f"match requested {requested_architecture!r}; use a fresh run"
        )
    if requested_architecture == WCM_CRITIC_ARCHITECTURE:
        saved_target_mode = checkpoint_wcm_dynamics_target_mode(resume_state)
        if saved_target_mode != WCM_DYNAMICS_TARGET_MODE:
            raise ValueError(
                f"resume WCM dynamics target mode={saved_target_mode!r} does "
                f"not match required {WCM_DYNAMICS_TARGET_MODE!r}; use the "
                "checkpoint for evaluation or a fresh source warm start"
            )
        if WCM_DYNAMICS_TARGET_STATE_KEY not in resume_state:
            raise ValueError(
                "hard-copy WCM resume checkpoint is missing "
                f"{WCM_DYNAMICS_TARGET_STATE_KEY!r}"
            )
        saved_interval_value = saved_args.get(
            "dynamics_target_sync_interval"
        )
        if saved_interval_value is None:
            raise ValueError(
                "hard-copy WCM resume checkpoint has no immutable "
                "dynamics_target_sync_interval configuration"
            )
        interval = int(saved_interval_value)
        if interval != int(args.dynamics_target_sync_interval):
            raise ValueError(
                f"resume dynamics_target_sync_interval={interval} does not match "
                f"requested {args.dynamics_target_sync_interval}"
            )
        global_step = int(resume_state.get("step", -1))
        last_sync_step = int(
            resume_state.get("dynamics_target_last_sync_step", -1)
        )
        expected_last_sync = (
            global_step // interval * interval if global_step >= 0 else -1
        )
        if last_sync_step != expected_last_sync:
            raise ValueError(
                "resume WCM dynamics target sync state is inconsistent: "
                f"step={global_step}, last_sync={last_sync_step}, "
                f"expected={expected_last_sync}"
            )
    if str(resume_state.get("task", "")) != str(args.task):
        raise ValueError(
            f"resume task={resume_state.get('task')!r} does not match "
            f"requested task={args.task!r}"
        )
    saved_dataset = resume_state.get("dataset", saved_args.get("dataset"))
    if saved_dataset is None:
        raise ValueError("resume checkpoint has no dataset provenance")
    if Path(saved_dataset).expanduser().resolve() != args.dataset:
        raise ValueError(
            f"resume dataset={saved_dataset!r} does not match requested "
            f"dataset={str(args.dataset)!r}"
        )
    saved_identity = resume_state.get("dataset_identity")
    # if saved_identity is None:
    #     if not bool(getattr(args, "validate_resume_only", False)):
    #         raise ValueError(
    #             "resume checkpoint has no immutable dataset identity; use it as a "
    #             "source warm start for a fresh output instead"
    #         )
    #     print(
    #         "WARNING: legacy checkpoint has no immutable dataset identity; "
    #         "completion validation is limited to its saved dataset path",
    #         flush=True,
    #     )
    # elif saved_identity != args.dataset_identity:
    #     raise ValueError(
    #         "resume dataset identity does not match the current mixed HDF5 "
    #         "and external source files"
    #     )

    exact_fields: dict[str, Any] = {
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "effective_global_batch_size": int(args.effective_global_batch_size),
        "schedule_reference_batch_size": int(args.schedule_reference_batch_size),
        "chunk_horizon": int(args.chunk_horizon),
        "critic_observation_horizon": (
            requested_critic_observation_horizon
        ),
        "critic_q_use_predicted_next_latent": (
            requested_q_uses_predicted_next
        ),
        "discount": float(args.discount),
        "expectile": float(args.expectile),
        "target_tau": float(args.target_tau),
        "critic_hidden_dims": tuple(int(x) for x in args.critic_hidden_dims),
        "latent_dim": int(args.latent_dim),
        "action_hidden_dim": int(args.action_hidden_dim),
        "num_attention_heads": int(args.num_attention_heads),
        "num_action_conv_layers": int(args.num_action_conv_layers),
        "dropout": float(args.dropout),
        "num_critics": int(args.num_critics),
        "critic_group_norm": bool(args.critic_group_norm),
        "critic_late_fusion_key": args.critic_late_fusion_key,
        "dynamics_weight": float(args.dynamics_weight),
        "dynamics_cosine_weight": float(args.dynamics_cosine_weight),
        "dynamics_warmup_steps": int(args.dynamics_warmup_steps),
        "encoder_freeze_steps": int(args.encoder_freeze_steps),
        "vf_encoder_freeze_steps": int(args.vf_encoder_freeze_steps),
        "use_huber": bool(args.use_huber),
        "max_gradient_norm": float(args.max_gradient_norm),
        "critic_vf_lr_scheduler": str(args.critic_vf_lr_scheduler),
        "critic_vf_lr_warmup_steps": int(args.critic_vf_lr_warmup_steps),
        "critic_vf_lr_num_cycles": float(args.critic_vf_lr_num_cycles),
    }
    if requested_architecture == WCM_CRITIC_ARCHITECTURE:
        exact_fields.update(
            {
                "temporal_num_layers": int(args.temporal_num_layers),
                "temporal_num_heads": int(args.temporal_num_heads),
                "temporal_feedforward_dim": int(
                    args.temporal_feedforward_dim
                ),
                "temporal_dropout": float(args.temporal_dropout),
                "dynamics_prediction_offsets": tuple(
                    int(value) for value in args.dynamics_prediction_offsets
                ),
                "sigreg_weight": float(args.sigreg_weight),
                "sigreg_knots": int(args.sigreg_knots),
                "sigreg_num_projections": int(args.sigreg_num_projections),
                "sigreg_global_batch": bool(args.sigreg_global_batch),
            }
        )
    if args.steps_per_epoch is not None:
        exact_fields["steps_per_epoch"] = int(args.steps_per_epoch)
    for field, requested in exact_fields.items():
        if field not in saved_args:
            if bool(args.validate_resume_only) and field == "effective_global_batch_size":
                saved = int(saved_args["batch_size"])
            elif (
                bool(args.validate_resume_only)
                and field == "schedule_reference_batch_size"
            ):
                saved = 100
            elif field == "critic_observation_horizon":
                saved = checkpoint_critic_observation_horizon(resume_state)
            elif field == "critic_q_use_predicted_next_latent":
                saved = checkpoint_q_uses_predicted_next_latent(
                    resume_state
                )
            else:
                raise ValueError(
                    f"resume checkpoint has no immutable {field} configuration; "
                    "use it as a source warm start for a fresh output instead"
                )
        else:
            saved = saved_args[field]
        if field == "critic_hidden_dims":
            saved = tuple(int(x) for x in saved)
        elif field == "dynamics_prediction_offsets":
            saved = tuple(int(x) for x in saved)
        if saved != requested:
            raise ValueError(
                f"resume {field}={saved!r} does not match requested "
                f"{field}={requested!r}; use a source warm start for an "
                "intentional objective change"
            )
    expected_q_inputs = architecture_q_head_inputs(
        requested_architecture,
        requested_q_uses_predicted_next,
    )
    saved_q_inputs = tuple(
        resume_state.get(
            "critic_q_head_inputs",
            critic_q_head_inputs(
                checkpoint_q_uses_predicted_next_latent(resume_state)
            ),
        )
    )
    if saved_q_inputs != expected_q_inputs:
        raise ValueError(
            f"resume critic_q_head_inputs={saved_q_inputs!r} does not match "
            f"requested {expected_q_inputs!r}"
        )
    if (
        requested_q_uses_predicted_next
        and resume_state.get("critic_q_predicted_next_normalization")
        != PREDICTED_NEXT_Q_NORMALIZATION
    ):
        raise ValueError(
            "resume checkpoint has an incompatible predicted-next Q "
            "normalization"
        )


def install_pretrained_actor_reference(
    actor_algo,
    *,
    weight: float,
    batch_fraction: float,
) -> dict[str, Any]:
    """Install an exact frozen copy of the originally deployed DP EMA."""
    if actor_algo.ema is None:
        raise RuntimeError(
            "actor reference distillation requires a pretrained DP EMA"
        )
    if actor_algo.reference_policy_enabled or actor_algo.hazard_constraint_enabled:
        raise RuntimeError(
            "actor reference distillation is incompatible with existing "
            "reference-margin or hazard objectives"
        )
    teacher = copy.deepcopy(actor_algo.ema.averaged_model).float().to(
        actor_algo.device
    )
    if "condition_adapter" in teacher["policy"]:
        raise RuntimeError(
            "the reference teacher must be the original unconditional "
            "pretrained DP EMA"
        )
    teacher.train()
    teacher.requires_grad_(False)
    actor_algo.reference_nets = teacher
    actor_algo.reference_distillation_weight = float(weight)
    actor_algo.reference_distillation_batch_fraction = float(batch_fraction)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("the pretrained actor reference is not frozen")
    return {
        "source": "original_pretrained_DP_deployed_EMA",
        "objective": "same_crop_noisy_action_timestep_noise_prediction_MSE",
        "student_condition": 1.0,
        "student_condition_mask": 1.0,
        "weight": float(weight),
        "batch_fraction": float(batch_fraction),
        "teacher_parameter_count": parameter_count(teacher),
        "teacher_trainable_parameter_count": 0,
    }


def configure_chunk_actor_optimizer(
    actor_algo,
    *,
    conditioned_actor: bool,
    adapter_lr: float,
    unet_lr: float,
    obs_encoder_lr: float,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float,
) -> None:
    """Use the DP AdamW recipe with strict conditioned or plain actor groups."""
    policy = actor_algo.nets["policy"]
    adapter_present = "condition_adapter" in policy
    if adapter_present != bool(conditioned_actor):
        raise RuntimeError(
            "actor conditioning configuration does not match policy modules: "
            f"conditioned_actor={bool(conditioned_actor)}, "
            f"condition_adapter_present={adapter_present}"
        )
    expected_modules: list[tuple[str, float]] = []
    if conditioned_actor:
        expected_modules.append(("condition_adapter", float(adapter_lr)))
    expected_modules.extend(
        (
            ("noise_pred_net", float(unet_lr)),
            ("obs_encoder", float(obs_encoder_lr)),
        )
    )
    if set(policy.keys()) != {name for name, _ in expected_modules}:
        raise RuntimeError(
            "unexpected actor policy modules for grouped optimization: "
            f"{tuple(policy.keys())}"
        )

    parameter_groups = []
    grouped_parameter_ids: set[int] = set()
    for name, learning_rate in expected_modules:
        parameters = list(policy[name].parameters())
        if not parameters:
            raise RuntimeError(f"actor module {name!r} has no parameters")
        duplicates = [
            parameter
            for parameter in parameters
            if id(parameter) in grouped_parameter_ids
        ]
        if duplicates:
            raise RuntimeError(f"actor module {name!r} shares optimizer parameters")
        for parameter in parameters:
            parameter.requires_grad_(True)
            grouped_parameter_ids.add(id(parameter))
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "group_name": name,
            }
        )

    all_parameters = list(policy.parameters())
    if grouped_parameter_ids != {id(parameter) for parameter in all_parameters}:
        raise RuntimeError("the grouped actor optimizer does not cover the policy")
    actor_algo.optimizers["policy"] = torch.optim.AdamW(
        parameter_groups,
        weight_decay=ACTOR_WEIGHT_DECAY,
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


def gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, indices]


def gather_time_history(
    values: torch.Tensor,
    end_indices: torch.Tensor,
    observation_horizon: int,
) -> torch.Tensor:
    """Gather chronological histories ending at one index per batch row."""
    horizon = int(observation_horizon)
    offsets = torch.arange(
        1 - horizon,
        1,
        device=end_indices.device,
        dtype=end_indices.dtype,
    )
    indices = end_indices[:, None] + offsets[None]
    if torch.any(indices < 0) or torch.any(indices >= values.shape[1]):
        raise IndexError(
            "successor observation history is outside the loaded sequence"
        )
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[rows, indices]


def source_condition_labels(
    raw_batch: dict,
    *,
    current_index: int,
) -> torch.Tensor:
    """Read explicit actor labels, independently of source identity and reward."""
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
            "actor condition requires one scalar source label per "
            f"transition, got shape={tuple(labels.shape)}"
        )
    labels = labels[:, 0].float()
    zeros = torch.zeros_like(labels)
    ones = torch.ones_like(labels)
    is_zero = torch.isclose(labels, zeros, atol=1e-6, rtol=0.0)
    is_one = torch.isclose(labels, ones, atol=1e-6, rtol=0.0)
    if not torch.all(is_zero | is_one):
        invalid = labels[~(is_zero | is_one)]
        raise ValueError(
            f"actor condition expected {label_source} values 0 or 1, got "
            f"values={invalid[:8].detach().cpu().tolist()}"
        )
    return is_one.to(dtype=torch.float32)


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
    actor_condition_mode: str,
) -> dict[str, Any]:
    """Validate actor labels independently of the selected critic reward."""
    expected_definition = actor_condition_definition(actor_condition_mode)
    expected_labels = actor_condition_labels(actor_condition_mode)
    positive_sources, negative_sources = actor_condition_sources(
        actor_condition_mode
    )
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
        dataset_mode = str(handle.attrs.get("actor_condition_mode", "human_only"))
        dataset_definition = str(
            handle.attrs.get("actor_condition_definition", "")
        )
        if dataset_mode != actor_condition_mode:
            raise ValueError(
                f"dataset actor_condition_mode={dataset_mode!r} does not "
                f"match requested mode={actor_condition_mode!r}; rebuild it"
            )
        if dataset_definition != expected_definition:
            raise ValueError(
                "dataset actor condition definition="
                f"{dataset_definition!r} does not match requested "
                f"{expected_definition!r}; rebuild it"
            )
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
            if "source_is_expert" not in episode:
                raise ValueError(
                    f"data/{episode_key} is missing source_is_expert; rebuild "
                    f"the {reward_mode}-reward dataset"
                )
            source_labels = np.asarray(
                episode["source_is_expert"][:], dtype=np.float32
            )
            expected_source_label = 1.0 if source == "expert" else 0.0
            if source_labels.size < 1 or not np.allclose(
                source_labels,
                expected_source_label,
                atol=1e-6,
                rtol=0.0,
            ):
                unique = np.unique(source_labels).tolist()
                raise ValueError(
                    f"data/{episode_key} source={source!r} has invalid "
                    f"source_is_expert={unique[:8]}"
                )
            if "actor_condition" not in episode:
                raise ValueError(
                    f"data/{episode_key} is missing actor_condition; rebuild "
                    f"the {reward_mode}-reward dataset"
                )
            condition_labels = np.asarray(
                episode["actor_condition"][:], dtype=np.float32
            )
            expected_condition = expected_labels[source_name]
            if condition_labels.shape != source_labels.shape or not np.allclose(
                condition_labels,
                expected_condition,
                atol=1e-6,
                rtol=0.0,
            ):
                unique = np.unique(condition_labels).tolist()
                raise ValueError(
                    f"data/{episode_key} source={source!r} must map to actor "
                    f"condition={expected_condition}, got {unique[:8]}"
                )
            episode_counts[source_name] += 1
            transition_counts[source_name] += int(condition_labels.size)
    missing_sources = [
        source for source, count in episode_counts.items() if count == 0
    ]
    if missing_sources:
        raise ValueError(
            "conditioned actor dataset is missing required sources: "
            f"{missing_sources}"
        )
    return {
        "mode": str(actor_condition_mode),
        "definition": expected_definition,
        "positive_sources": positive_sources,
        "negative_sources": negative_sources,
        "dataset_key": "actor_condition",
        "source_identity_key": "source_is_expert",
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
    reward_mode: str = "task",
    critic_observation_horizon: int = 1,
    dynamics_prediction_offsets: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Extract a semi-MDP transition at the first executable DP action."""
    current_index = int(actor_algo.algo_config.horizon.observation_horizon) - 1
    critic_observation_horizon = int(critic_observation_horizon)
    if not 1 <= critic_observation_horizon <= current_index + 1:
        raise ValueError(
            "critic_observation_horizon must be in [1, actor observation "
            f"horizon={current_index + 1}], got "
            f"{critic_observation_horizon}"
        )
    end_index = current_index + int(chunk_horizon)
    if raw_batch["actions"].shape[1] < end_index:
        raise ValueError(
            f"batch sequence length {raw_batch['actions'].shape[1]} is shorter "
            f"than required index {end_index}"
        )

    actions = raw_batch["actions"][:, current_index:end_index]
    if "rewards" not in raw_batch:
        raise KeyError("chunk-IDQL batch is missing rewards")
    critic_rewards = raw_batch["rewards"]
    if str(reward_mode) == "task":
        if "task_rewards" not in raw_batch:
            raise KeyError(
                "chunk-IDQL critic requires task_rewards in every batch; "
                "rebuild the dataset with --reward-mode task"
            )
        task_rewards = raw_batch["task_rewards"]
        if task_rewards.shape != critic_rewards.shape or not torch.equal(
            task_rewards,
            critic_rewards,
        ):
            raise ValueError(
                "chunk-IDQL critic rewards must exactly equal preserved source "
                "task_rewards"
            )
        critic_rewards = task_rewards
    elif str(reward_mode) == "terminal_success":
        pass
    elif str(reward_mode) != "rise":
        raise ValueError(f"unsupported chunk critic reward_mode={reward_mode!r}")
    rewards = critic_rewards[:, current_index:end_index].float()
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

    dynamics_prediction_offsets = tuple(
        int(value) for value in dynamics_prediction_offsets
    )
    if dynamics_prediction_offsets and (
        tuple(sorted(set(dynamics_prediction_offsets)))
        != dynamics_prediction_offsets
        or dynamics_prediction_offsets[0] < 1
        or dynamics_prediction_offsets[-1] > int(chunk_horizon)
    ):
        raise ValueError(
            "dynamics_prediction_offsets must be sorted, unique, positive, "
            f"and <= chunk_horizon={chunk_horizon}; got "
            f"{dynamics_prediction_offsets}"
        )
    dynamics_targets = None
    if dynamics_prediction_offsets:
        offset_indices = torch.as_tensor(
            [current_index + value - 1 for value in dynamics_prediction_offsets],
            dtype=torch.long,
            device=dones.device,
        )
        prefix_continuation = torch.cumprod(continuation, dim=1)
        valid_mask = prefix_continuation[
            :,
            torch.as_tensor(
                [value - 1 for value in dynamics_prediction_offsets],
                dtype=torch.long,
                device=dones.device,
            ),
        ]
        if "chunk_dynamics_next_obs" in raw_batch:
            target_obs = {
                key: raw_batch["chunk_dynamics_next_obs"][key]
                for key in actor_algo.obs_shapes
            }
            availability = raw_batch[
                "chunk_dynamics_target_available"
            ].to(dtype=valid_mask.dtype, device=valid_mask.device)
        else:
            target_obs = {
                key: raw_batch["next_obs"][key].index_select(
                    1, offset_indices
                )
                for key in actor_algo.obs_shapes
            }
            if "pad_mask" in raw_batch:
                availability = raw_batch["pad_mask"].index_select(
                    1, offset_indices
                )
                if availability.ndim == 3:
                    availability = availability.squeeze(-1)
                availability = availability.to(dtype=valid_mask.dtype)
            else:
                availability = torch.ones_like(valid_mask)
        dynamics_targets = {
            # This nested key must remain exactly ``next_obs`` so robomimic's
            # postprocessor applies channel conversion and normalization.
            "next_obs": target_obs,
            "valid_mask": valid_mask * availability,
        }

    batch = {
        "obs": {
            key: (
                raw_batch["obs"][key][:, current_index]
                if critic_observation_horizon == 1
                else raw_batch["obs"][key][
                    :,
                    current_index - critic_observation_horizon + 1
                    : current_index + 1,
                ]
            )
            for key in actor_algo.obs_shapes
        },
        "next_obs": (
            {
                key: (
                    raw_batch["next_obs"][key][:, -1]
                    if critic_observation_horizon == 1
                    else raw_batch["next_obs"][key][
                        :, -critic_observation_horizon:
                    ]
                )
                for key in actor_algo.obs_shapes
            }
            if "chunk_sparse_next_obs" in raw_batch
            else {
                key: (
                    gather_time(raw_batch["next_obs"][key], next_indices)
                    if critic_observation_horizon == 1
                    else gather_time_history(
                        raw_batch["next_obs"][key],
                        next_indices,
                        critic_observation_horizon,
                    )
                )
                for key in actor_algo.obs_shapes
            }
        ),
        "actions": actions * action_mask.unsqueeze(-1),
        "action_mask": action_mask,
        "reward": chunk_return.reshape(-1, 1),
        "terminal": terminal.reshape(-1, 1),
        "valid_length": valid_length.reshape(-1, 1),
        "exact_next": exact_next.reshape(-1, 1),
        "goal_obs": raw_batch.get("goal_obs"),
    }
    if dynamics_targets is not None:
        batch["dynamics_targets"] = dynamics_targets
    # Keep DataLoader-pinned uint8 RGB tensors compact during H2D transfer.
    # Casting them on CPU first both expands traffic by 4x and returns an
    # unpinned allocation, defeating the requested non-blocking copy.
    batch = TensorUtils.to_float(
        TensorUtils.to_device(
            batch,
            actor_algo.device,
            non_blocking=actor_algo.device.type == "cuda",
        )
    )
    batch = actor_algo.postprocess_batch_for_training(
        batch, obs_normalization_stats=obs_normalization_stats
    )
    if not bool(getattr(actor_algo, "_chunk_action_range_validated", False)):
        if not torch.isfinite(batch["actions"]).all():
            raise ValueError("chunk critic actions contain non-finite values")
        action_min = float(batch["actions"].min().item())
        action_max = float(batch["actions"].max().item())
        if action_min < -1.001 or action_max > 1.001:
            raise ValueError(
                "chunk actions are outside the pretrained normalized space: "
                f"min={action_min:.6f}, max={action_max:.6f}"
            )
        actor_algo._chunk_action_range_validated = True
    batch["actions"] = batch["actions"].clamp(-1.0, 1.0)
    return batch


def masked_dynamics_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    exact_next: torch.Tensor,
    *,
    distributed_context: DistributedContext | None = None,
    global_valid_row_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = exact_next.reshape(-1) > 0.5
    if predicted.ndim < 2 or target.ndim < 2:
        raise ValueError(
            "dynamics features must include batch and feature dimensions"
        )
    predicted = predicted[rows]
    target = target[rows]
    if predicted.shape != target.shape:
        raise ValueError(
            "dynamics prediction and actor-encoder target shapes differ: "
            f"predicted={tuple(predicted.shape)}, target={tuple(target.shape)}"
        )
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)

    local_valid_row_count = rows.sum().detach()
    if global_valid_row_count is None:
        global_valid_row_count = local_valid_row_count.clone()
        if distributed_context is not None and distributed_context.enabled:
            dist.all_reduce(global_valid_row_count, op=dist.ReduceOp.SUM)
    if global_valid_row_count.numel() != 1:
        raise ValueError("global dynamics valid-row count must be scalar")
    world_size = (
        int(distributed_context.world_size)
        if distributed_context is not None and distributed_context.enabled
        else 1
    )
    denominator = global_valid_row_count.to(dtype=predicted.dtype).clamp_min(1.0)
    # Gradient synchronization averages rank gradients. Scaling each local
    # row-loss sum by world_size / global_count therefore reproduces the
    # gradient of one mean over all valid rows, including when a rank has no
    # valid rows. Empty reductions remain connected to ``predicted`` so every
    # rank participates in the same backward and collective sequence.
    gradient_scale = predicted.new_tensor(float(world_size)) / denominator
    smooth_l1_per_row = (
        F.smooth_l1_loss(
            predicted,
            target,
            reduction="none",
        )
        .flatten(start_dim=1)
        .mean(dim=1)
    )
    smooth_l1 = smooth_l1_per_row.sum() * gradient_scale
    cosine_per_row = 1.0 - F.cosine_similarity(predicted, target, dim=-1)
    cosine = cosine_per_row.sum() * gradient_scale

    squared_error_per_row = (
        (predicted - target).square().flatten(start_dim=1).mean(dim=1)
    )
    global_squared_error_sum = squared_error_per_row.sum().detach().clone()
    if distributed_context is not None and distributed_context.enabled:
        dist.all_reduce(global_squared_error_sum, op=dist.ReduceOp.SUM)
    global_mse = global_squared_error_sum / denominator
    rmse = torch.where(
        global_valid_row_count > 0,
        torch.sqrt(global_mse.clamp_min(1e-12)),
        torch.zeros_like(global_mse),
    )
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
    dynamics_target_encoder: nn.Module | None,
    vf: RiseValueNetwork | None,
    batch: dict[str, Any],
    *,
    discount: float,
    expectile: float,
    use_huber: bool,
    dynamics_weight: float,
    dynamics_cosine_weight: float,
    distributed_context: DistributedContext | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    device = batch["actions"].device
    critic_horizons = {
        int(getattr(critic, "observation_horizon", 1))
        for critic in critics
    }
    if len(critic_horizons) != 1:
        raise ValueError("all chunk critics must use the same observation history")
    critic_observation_horizon = next(iter(critic_horizons))
    if int(getattr(vf, "observation_horizon", 1)) != critic_observation_horizon:
        raise ValueError("Q and V observation histories must match")
    dynamics_next_obs = {
        key: (
            value[:, -1]
            if critic_observation_horizon > 1
            else value
        )
        for key, value in batch["next_obs"].items()
    }
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
                        obs=dynamics_next_obs,
                    )
                )

    regression = F.smooth_l1_loss if use_huber else F.mse_loss
    global_valid_row_count = (
        batch["exact_next"].reshape(-1) > 0.5
    ).sum().detach()
    if distributed_context is not None and distributed_context.enabled:
        dist.all_reduce(global_valid_row_count, op=dist.ReduceOp.SUM)
    critic_losses = []
    dynamics_l1 = []
    dynamics_cosine = []
    dynamics_rmse = []
    weighted_dynamics_losses = []
    q_losses = []
    for output, target_features in zip(outputs, target_next_encoder_features):
        q_loss = regression(output["q"], q_backup)
        dyn_l1, dyn_cos, dyn_rmse = masked_dynamics_losses(
            output["predicted_next_encoder"],
            target_features,
            batch["exact_next"],
            distributed_context=distributed_context,
            global_valid_row_count=global_valid_row_count,
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


def masked_wcm_dynamics_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    current_frame_latent: torch.Tensor,
    *,
    distributed_context: DistributedContext | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Raw latent MSE over one global mean of valid (row, offset) pairs."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError(
            "WCM prediction and target must share [B,O,D] shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if tuple(valid_mask.shape) != tuple(predicted.shape[:2]):
        raise ValueError(
            f"WCM dynamics mask must be [B,O], got {tuple(valid_mask.shape)}"
        )
    rows = valid_mask.reshape(-1) > 0.5
    predicted_rows = predicted.reshape(-1, predicted.shape[-1])[rows]
    target_rows = target.reshape(-1, target.shape[-1])[rows]
    per_row_mse = (predicted_rows - target_rows).square().mean(dim=-1)
    local_count = rows.sum().to(dtype=predicted.dtype).detach()

    copy_prediction = current_frame_latent[:, None, :].expand_as(target)
    copy_rows = copy_prediction.reshape(-1, target.shape[-1])[rows]
    copy_per_row = (copy_rows - target_rows).square().mean(dim=-1)

    # Pack all detached dynamics statistics into one collective. The previous
    # implementation issued count + sum reductions globally and again for
    # every offset (11 blocking NCCL calls for four targets).
    packed_statistics = [
        local_count,
        per_row_mse.detach().sum(),
        copy_per_row.detach().sum(),
    ]
    for offset_index in range(predicted.shape[1]):
        offset_rows = valid_mask[:, offset_index] > 0.5
        errors = (
            predicted[:, offset_index][offset_rows]
            - target[:, offset_index][offset_rows]
        ).square().mean(dim=-1)
        packed_statistics.extend(
            (
                offset_rows.sum().to(dtype=predicted.dtype).detach(),
                errors.detach().sum(),
            )
        )
    global_statistics = torch.stack(packed_statistics)
    if distributed_context is not None and distributed_context.enabled:
        dist.all_reduce(global_statistics, op=dist.ReduceOp.SUM)
    global_count = global_statistics[0]
    world_size = (
        int(distributed_context.world_size)
        if distributed_context is not None and distributed_context.enabled
        else 1
    )
    denominator = global_count.to(dtype=predicted.dtype).clamp_min(1.0)
    # Manual parameter-gradient synchronization later averages ranks. This
    # scaling makes its result equal one mean over the global valid rows.
    loss = per_row_mse.sum() * (
        predicted.new_tensor(float(world_size)) / denominator
    )
    global_squared_sum = global_statistics[1]
    global_mse = torch.where(
        global_count > 0,
        global_squared_sum / denominator,
        torch.zeros_like(global_squared_sum),
    )
    global_copy_sum = global_statistics[2]
    global_copy_mse = torch.where(
        global_count > 0,
        global_copy_sum / denominator,
        torch.zeros_like(global_copy_sum),
    )

    metrics: dict[str, torch.Tensor] = {
        "dynamics/mse": global_mse.detach(),
        "dynamics/rmse": torch.sqrt(global_mse.clamp_min(1e-12)).detach(),
        "dynamics/copy_current_mse": global_copy_mse.detach(),
        "dynamics/valid_pair_count": global_count.to(predicted.dtype),
        "dynamics/valid_fraction": (
            global_count
            / predicted.new_tensor(
                float(valid_mask.numel() * world_size)
            )
        ).detach(),
    }
    for offset_index in range(predicted.shape[1]):
        statistics_start = 3 + 2 * offset_index
        offset_count = global_statistics[statistics_start]
        offset_sum = global_statistics[statistics_start + 1]
        metrics[f"dynamics/offset_index_{offset_index}_mse"] = torch.where(
            offset_count > 0,
            offset_sum / offset_count.clamp_min(1.0),
            torch.zeros_like(offset_sum),
        )
        metrics[f"dynamics/offset_index_{offset_index}_valid_count"] = (
            offset_count.to(predicted.dtype)
        )
    return loss, metrics


def sigreg_loss(
    features: torch.Tensor,
    *,
    knots: int,
    num_projections: int,
    seed: int,
    distributed_context: DistributedContext | None = None,
    projection_chunk_size: int = 128,
) -> torch.Tensor:
    """WCM/LeJEPA Epps-Pulley SIGReg on a differentiable global batch."""
    if features.ndim != 2:
        raise ValueError(f"SIGReg features must be [B,D], got {features.shape}")
    if int(knots) < 2 or int(num_projections) < 1:
        raise ValueError("SIGReg knots and projection count must be positive")
    features = features.float()
    local_batch_size, feature_dim = features.shape
    world_size = (
        int(distributed_context.world_size)
        if distributed_context is not None and distributed_context.enabled
        else 1
    )
    global_batch_size = int(local_batch_size) * world_size
    if global_batch_size < 2:
        return features.sum() * 0.0
    points = torch.linspace(
        0.0,
        3.0,
        int(knots),
        dtype=torch.float32,
        device=features.device,
    )
    delta = 3.0 / float(int(knots) - 1)
    weights = torch.full_like(points, 2.0 * delta)
    weights[[0, -1]] = delta
    gaussian_cf = torch.exp(-0.5 * points.square())
    weights = weights * gaussian_cf
    with fork_rng_with_seed(int(seed), features.device):
        projections = torch.randn(
            int(feature_dim),
            int(num_projections),
            dtype=torch.float32,
            device=features.device,
        )
    projections = F.normalize(projections, dim=0)
    moment_chunks = []
    for start in range(0, int(num_projections), int(projection_chunk_size)):
        directions = projections[
            :, start : start + int(projection_chunk_size)
        ]
        projected = features @ directions
        phases = projected.unsqueeze(-1) * points
        moment_chunks.append(
            torch.stack(
                (phases.cos().sum(dim=0), phases.sin().sum(dim=0)),
                dim=0,
            )
        )
    characteristic_sums = torch.cat(moment_chunks, dim=1)
    if distributed_context is not None and distributed_context.enabled:
        # Reduce only the characteristic-function moments. Gathering the full
        # feature batch made every GPU redundantly evaluate the global
        # B x projections x knots trigonometric objective. One moment
        # collective retains the exact global-batch loss and gradient.
        from torch.distributed.nn.functional import all_reduce

        characteristic_sums = all_reduce(
            characteristic_sums,
            op=dist.ReduceOp.SUM,
        )
    real_error = (
        characteristic_sums[0] / float(global_batch_size) - gaussian_cf
    )
    imaginary = characteristic_sums[1] / float(global_batch_size)
    per_projection = (
        (real_error.square() + imaginary.square()) @ weights
    ) * float(global_batch_size)
    return per_projection.sum() / float(num_projections)


def compute_wcm_chunk_losses(
    system: WCMChunkValueSystem,
    target_system: WCMChunkValueSystem,
    dynamics_target_encoder: WCMFrameTargetEncoder,
    batch: dict[str, Any],
    *,
    discount: float,
    expectile: float,
    use_huber: bool,
    dynamics_weight: float,
    sigreg_weight: float,
    sigreg_knots: int,
    sigreg_num_projections: int,
    sigreg_global_batch: bool,
    global_step: int,
    distributed_context: DistributedContext | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "dynamics_targets" not in batch:
        raise KeyError("WCM batch is missing dense dynamics_targets")
    device = batch["actions"].device
    augmentation_seed = int(
        torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (),
            device="cpu",
        ).item()
    )
    batch_size = int(batch["actions"].shape[0])
    crop_plan = make_wcm_temporal_crop_plan(
        system.nets["encoder"],
        batch_size=batch_size,
        seed=augmentation_seed,
        device=device,
    )
    state = system.encode_state(
        batch["obs"],
        batch["goal_obs"],
        crop_plan=crop_plan,
    )
    q_predictions = system.q_values_from_state(
        state,
        batch["actions"],
        batch["action_mask"],
    )
    vf_pred = system.value_from_state(state)
    predicted_dynamics = system.predict_dynamics_from_state(
        state,
        batch["actions"],
        batch["action_mask"],
    )

    with torch.no_grad():
        next_state = system.encode_state(
            batch["next_obs"],
            batch["goal_obs"],
            crop_plan=crop_plan,
        )
        next_v = system.value_from_state(next_state)
        bootstrap = torch.pow(
            batch["valid_length"].new_tensor(float(discount)),
            batch["valid_length"],
        )
        q_backup = (
            batch["reward"]
            + (1.0 - batch["terminal"]) * bootstrap * next_v
        )
        target_state = target_system.encode_state(
            batch["obs"],
            batch["goal_obs"],
            crop_plan=crop_plan,
        )
        target_qs = target_system.q_values_from_state(
            target_state,
            batch["actions"],
            batch["action_mask"],
        )
        target_q_min = torch.cat(target_qs, dim=1).min(
            dim=1, keepdim=True
        ).values
        target_dynamics = dynamics_target_encoder.encode_dynamics_targets(
            batch["dynamics_targets"]["next_obs"],
            batch["goal_obs"],
            crop_plan=crop_plan,
        ).detach()

    regression = F.smooth_l1_loss if use_huber else F.mse_loss
    q_losses = [regression(prediction, q_backup) for prediction in q_predictions]
    vf_error = vf_pred - target_q_min
    vf_weight = torch.where(
        vf_error > 0.0,
        1.0 - float(expectile),
        float(expectile),
    )
    vf_loss = (vf_weight * vf_error.square()).mean()
    dynamics_mse, dynamics_info = masked_wcm_dynamics_mse(
        predicted_dynamics,
        target_dynamics,
        batch["dynamics_targets"]["valid_mask"],
        state["current_frame_latent"],
        distributed_context=distributed_context,
    )
    if float(sigreg_weight) > 0.0:
        raw_sigreg = sigreg_loss(
            state["temporal_state"],
            knots=int(sigreg_knots),
            num_projections=int(sigreg_num_projections),
            seed=int(global_step) + 1729,
            distributed_context=(
                distributed_context if sigreg_global_batch else None
            ),
        )
    else:
        # Weight zero intentionally performs no distributed gather/projection.
        raw_sigreg = state["temporal_state"].sum() * 0.0
    weighted_dynamics = float(dynamics_weight) * dynamics_mse
    weighted_sigreg = float(sigreg_weight) * raw_sigreg
    total_loss = sum(q_losses) + vf_loss + weighted_dynamics + weighted_sigreg
    q_tensor = torch.cat(q_predictions, dim=1)
    target_valid_rows = target_dynamics[
        batch["dynamics_targets"]["valid_mask"] > 0.5
    ]
    if int(target_valid_rows.shape[0]) > 0:
        target_feature_std = target_valid_rows.std(
            dim=0,
            unbiased=False,
        ).mean()
        target_feature_norm = target_valid_rows.norm(dim=-1).mean()
    else:
        target_feature_std = target_dynamics.new_zeros(())
        target_feature_norm = target_dynamics.new_zeros(())
    info = {
        **{
            f"critic/q{index + 1}_loss": loss.detach()
            for index, loss in enumerate(q_losses)
        },
        **{
            f"critic/q{index + 1}_mean": prediction.mean().detach()
            for index, prediction in enumerate(q_predictions)
        },
        **dynamics_info,
        "critic/total_loss": total_loss.detach(),
        "critic/q_target_mean": q_backup.mean().detach(),
        "critic/q_ensemble_std": q_tensor.std(dim=1).mean().detach(),
        "vf/loss": vf_loss.detach(),
        "vf/value_mean": vf_pred.mean().detach(),
        "vf/target_q_min_mean": target_q_min.mean().detach(),
        "vf/error_mean": vf_error.mean().detach(),
        "dynamics/weighted_loss": weighted_dynamics.detach(),
        "dynamics/effective_mse_weight": q_tensor.new_tensor(
            float(dynamics_weight)
        ),
        "dynamics/target_feature_std": target_feature_std.detach(),
        "dynamics/target_feature_norm": target_feature_norm.detach(),
        "sigreg/raw_loss": raw_sigreg.detach(),
        "sigreg/weighted_loss": weighted_sigreg.detach(),
        "sigreg/effective_weight": q_tensor.new_tensor(float(sigreg_weight)),
        "representation/temporal_feature_std": state[
            "temporal_state"
        ].std(dim=0, unbiased=False).mean().detach(),
        "representation/temporal_mean_norm": state[
            "temporal_state"
        ].mean(dim=0).norm().detach(),
        "representation/frame_feature_std": state[
            "current_frame_latent"
        ].std(dim=0, unbiased=False).mean().detach(),
        "data/chunk_return_mean": batch["reward"].mean().detach(),
        "data/terminal_fraction": batch["terminal"].mean().detach(),
        "data/valid_length_mean": batch["valid_length"].mean().detach(),
        "data/action_abs_mean": batch["actions"].abs().mean().detach(),
        "data/action_min": batch["actions"].min().detach(),
        "data/action_max": batch["actions"].max().detach(),
        "objective/monte_carlo_return_weight": q_tensor.new_zeros(()),
    }
    for offset_index, offset in enumerate(system.dynamics_prediction_offsets):
        for suffix in ("mse", "valid_count"):
            temporary_key = f"dynamics/offset_index_{offset_index}_{suffix}"
            info[f"dynamics/offset_{offset}_{suffix}"] = info.pop(temporary_key)
    return total_loss, info


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
    vf_optimizer: torch.optim.Optimizer | None,
    critic_losses: list[torch.Tensor],
    vf_loss: torch.Tensor,
    *,
    target_tau: float,
    max_gradient_norm: float | None,
    gradient_sync_fn=None,
) -> None:
    """Backpropagate Q1/Q2/V first, then synchronize their gradients once."""
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
    for critic, target in zip(critics, targets):
        TorchUtils.soft_update(critic, target, tau=float(target_tau))


def make_wcm_optimizer(
    system: WCMChunkValueSystem,
    *,
    critic_lr: float,
    encoder_lr: float,
    vf_lr: float,
) -> torch.optim.Optimizer:
    encoder_parameters = list(system.nets["encoder"].parameters())
    value_parameters = list(system.nets["value_head"].parameters())
    excluded = {
        id(parameter)
        for parameter in (*encoder_parameters, *value_parameters)
    }
    critic_parameters = [
        parameter
        for parameter in system.parameters()
        if id(parameter) not in excluded
    ]
    return torch.optim.Adam(
        [
            {
                "params": critic_parameters,
                "lr": float(critic_lr),
                "group_name": "critic",
            },
            {
                "params": encoder_parameters,
                "lr": float(encoder_lr),
                "group_name": "encoder",
            },
            {
                "params": value_parameters,
                "lr": float(vf_lr),
                "group_name": "vf",
            },
        ]
    )


def configure_wcm_target_random_crops(
    target_system: WCMChunkValueSystem,
) -> None:
    target_system.eval().requires_grad_(False)
    configure_encoder_target_random_crops(target_system.nets["encoder"])


def configure_wcm_dynamics_target_random_crops(
    target_encoder: WCMFrameTargetEncoder,
) -> None:
    target_encoder.eval().requires_grad_(False)
    configure_encoder_target_random_crops(target_encoder.nets["encoder"])


def set_wcm_encoder_trainable(
    system: WCMChunkValueSystem,
    trainable: bool,
) -> None:
    """Freeze only the copied visual encoder; fresh WCM modules always train."""
    system.nets["encoder"].requires_grad_(bool(trainable))


def update_wcm_system(
    system: WCMChunkValueSystem,
    target_system: WCMChunkValueSystem,
    optimizer: torch.optim.Optimizer,
    total_loss: torch.Tensor,
    *,
    target_tau: float,
    max_gradient_norm: float | None,
    gradient_sync_fn=None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    parameters = list(system.parameters())
    if gradient_sync_fn is not None:
        gradient_sync_fn(parameters)
    if max_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    optimizer.step()
    TorchUtils.soft_update(system, target_system, tau=float(target_tau))


def validate_source(source: dict, args: argparse.Namespace) -> None:
    if not source.get("rise_style_rgb_idql", False):
        raise ValueError("source checkpoint is not a RISE-style RGB IDQL checkpoint")
    if source.get("rise_style_rgb_chunk_idql", False):
        raise ValueError("source-idql-checkpoint must be the one-step baseline")
    if str(source.get("task", args.task)) != str(args.task):
        raise ValueError(
            f"source task={source.get('task')} does not match task={args.task}"
        )
    if str(args.critic_architecture) == WCM_CRITIC_ARCHITECTURE:
        # A one-step IDQL source contributes only its actor lineage to a fresh
        # WCM system. Its Q/V widths and encoder-head layout are deliberately
        # unrelated to the newly constructed shared-temporal critic.
        required = (
            "actor_model",
            "pretrained_dp_checkpoint",
            "action_normalization_stats",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(
                "WCM actor warm start is missing source fields: "
                f"{missing}"
            )
        return
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
    """Validate a complete chunk checkpoint before a joint warm start."""
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
    source_conditioned = bool(source.get("conditioned_actor", False))
    validating_completed_resume = bool(
        getattr(args, "validate_resume_only", False)
    )
    if validating_completed_resume:
        if source_conditioned != bool(args.conditioned_actor):
            raise ValueError(
                "completed checkpoint conditioned_actor does not match the "
                "requested validation configuration"
            )
    else:
        if not source_conditioned:
            raise ValueError(
                "joint chunk warm start requires a conditioned source actor"
            )
        if not bool(args.conditioned_actor):
            raise ValueError(
                "source_chunk_idql_joint requires --conditioned-actor"
            )
    if source_conditioned:
        source_condition_definition = str(
            source.get("actor_condition_label_definition", "")
        )
        required_condition_definition = actor_condition_definition(
            args.actor_condition_mode
        )
        if source_condition_definition != required_condition_definition:
            raise ValueError(
                "source actor condition definition="
                f"{source_condition_definition!r} does not match required "
                f"{required_condition_definition!r}"
            )

    source_args = source.get("args", {})
    requested_critic_observation_horizon = int(
        getattr(args, "critic_observation_horizon", 1)
    )
    requested_q_uses_predicted_next = bool(
        getattr(args, "critic_q_use_predicted_next_latent", False)
    )
    requested_architecture = str(args.critic_architecture)
    source_architecture = checkpoint_critic_architecture(source)
    if source_architecture != requested_architecture:
        raise ValueError(
            f"source critic_architecture={source_architecture!r} does not "
            f"match requested {requested_architecture!r}"
        )
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
        "critic_observation_horizon": (
            requested_critic_observation_horizon
        ),
        "critic_q_use_predicted_next_latent": (
            requested_q_uses_predicted_next
        ),
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
    if requested_architecture == WCM_CRITIC_ARCHITECTURE:
        expected_fields.update(
            {
                "critic_temporal_num_layers": int(args.temporal_num_layers),
                "critic_temporal_num_heads": int(args.temporal_num_heads),
                "critic_temporal_feedforward_dim": int(
                    args.temporal_feedforward_dim
                ),
                "critic_temporal_dropout": float(args.temporal_dropout),
                "dynamics_prediction_offsets": tuple(
                    args.dynamics_prediction_offsets
                ),
            }
        )
    integer_fields = {
        "critic_chunk_horizon",
        "critic_observation_horizon",
        "critic_latent_dim",
        "critic_action_hidden_dim",
        "critic_num_attention_heads",
        "critic_num_action_conv_layers",
        "num_critics",
        "critic_temporal_num_layers",
        "critic_temporal_num_heads",
        "critic_temporal_feedforward_dim",
    }
    for field, expected in expected_fields.items():
        value = source.get(field)
        if field == "critic_hidden_dims":
            value = tuple(value or ())
        elif field == "dynamics_prediction_offsets":
            value = tuple(value or ())
        elif field == "critic_observation_horizon":
            value = checkpoint_critic_observation_horizon(source)
        elif field == "critic_q_use_predicted_next_latent":
            value = checkpoint_q_uses_predicted_next_latent(source)
        elif field in integer_fields:
            value = int(value if value is not None else -1)
        elif field in ("critic_dropout", "critic_temporal_dropout"):
            value = float(value if value is not None else float("nan"))
        elif field == "critic_group_norm":
            value = bool(value)
        if value != expected:
            raise ValueError(
                f"{field}={expected!r} does not match source value {value!r}"
            )

    expected_dynamics_mode = (
        WCM_DYNAMICS_PREDICTION_MODE
        if requested_architecture == WCM_CRITIC_ARCHITECTURE
        else DYNAMICS_PREDICTION_MODE
    )
    if str(source.get("dynamics_prediction_mode", "")) != expected_dynamics_mode:
        raise ValueError(
            "source dynamics prediction mode does not match the current "
            f"{expected_dynamics_mode!r} architecture"
        )
    expected_q_inputs = architecture_q_head_inputs(
        requested_architecture,
        requested_q_uses_predicted_next,
    )
    source_q_inputs = tuple(
        source.get(
            "critic_q_head_inputs",
            critic_q_head_inputs(
                checkpoint_q_uses_predicted_next_latent(source)
            ),
        )
    )
    if source_q_inputs != expected_q_inputs:
        raise ValueError(
            "source chunk checkpoint uses an incompatible Q head: "
            f"{source_q_inputs!r} != {expected_q_inputs!r}"
        )
    if (
        requested_q_uses_predicted_next
        and source.get("critic_q_predicted_next_normalization")
        != PREDICTED_NEXT_Q_NORMALIZATION
    ):
        raise ValueError(
            "source chunk checkpoint has an incompatible predicted-next Q "
            "normalization"
        )
    required_keys = (
        (
            "actor_model",
            "chunk_value_system",
            "chunk_value_target",
            "pretrained_dp_checkpoint",
            "action_normalization_stats",
        )
        if requested_architecture == WCM_CRITIC_ARCHITECTURE
        else (
            "actor_model",
            "critics",
            "critic_targets",
            "dynamics_target_encoder",
            "vf",
            "pretrained_dp_checkpoint",
            "action_normalization_stats",
        )
    )
    missing = [key for key in required_keys if key not in source]
    if missing:
        raise ValueError(
            f"source chunk checkpoint is missing required fields: {missing}"
        )
    if (
        requested_architecture == WCM_CRITIC_ARCHITECTURE
        and checkpoint_wcm_dynamics_target_mode(source)
        == WCM_DYNAMICS_TARGET_MODE
        and WCM_DYNAMICS_TARGET_STATE_KEY not in source
    ):
        raise ValueError(
            "source declares the hard-copy WCM dynamics target but is missing "
            f"{WCM_DYNAMICS_TARGET_STATE_KEY!r}"
        )
    if requested_architecture != WCM_CRITIC_ARCHITECTURE:
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
    wcm_system: WCMChunkValueSystem | None,
    wcm_target_system: WCMChunkValueSystem | None,
    wcm_dynamics_target_encoder: WCMFrameTargetEncoder | None,
    wcm_optimizer: torch.optim.Optimizer | None,
    wcm_lr_scheduler: Any,
    action_stats: dict,
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
    is_wcm = args.critic_architecture == WCM_CRITIC_ARCHITECTURE
    if is_wcm:
        if any(
            value is None
            for value in (
                wcm_system,
                wcm_target_system,
                wcm_dynamics_target_encoder,
                wcm_optimizer,
            )
        ):
            raise ValueError("WCM checkpoint payload is missing system state")
        model_state = {
            "chunk_value_system": wcm_system.state_dict(),
            "chunk_value_target": wcm_target_system.state_dict(),
            WCM_DYNAMICS_TARGET_STATE_KEY: (
                wcm_dynamics_target_encoder.state_dict()
            ),
            "chunk_value_optimizer": wcm_optimizer.state_dict(),
            "chunk_value_lr_scheduler": (
                wcm_lr_scheduler.state_dict()
                if wcm_lr_scheduler is not None
                else None
            ),
        }
    else:
        if dynamics_target_encoder is None or vf is None or vf_optimizer is None:
            raise ValueError("legacy checkpoint payload is missing critic state")
        model_state = {
            "critics": [critic.state_dict() for critic in critics],
            "critic_targets": [target.state_dict() for target in targets],
            "dynamics_target_encoder": dynamics_target_encoder.state_dict(),
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
        }
    return {
        "rise_style_rgb_idql": True,
        "rise_style_rgb_chunk_idql": True,
        "hybrid_dp_chunk_actor_iql": True,
        "visual_critic_idql": True,
        "critic_architecture": str(args.critic_architecture),
        "critic_q_head_inputs": architecture_q_head_inputs(
            args.critic_architecture,
            args.critic_q_use_predicted_next_latent,
        ),
        "critic_q_use_predicted_next_latent": bool(
            args.critic_q_use_predicted_next_latent
        ),
        "critic_q_predicted_next_normalization": (
            PREDICTED_NEXT_Q_NORMALIZATION
            if args.critic_q_use_predicted_next_latent
            else None
        ),
        "critic_representation_modules": (
            ("encoder", "frame_projection", "temporal_trunk")
            if is_wcm
            else ("encoder", "context", "context_norm")
        ),
        "critic_shared_state_representation": bool(is_wcm),
        "actor_model": actor_model,
        **model_state,
        "dynamics_prediction_mode": (
            WCM_DYNAMICS_PREDICTION_MODE
            if is_wcm
            else DYNAMICS_PREDICTION_MODE
        ),
        "dynamics_prediction_target": (
            "stop_gradient_periodic_hard_copy_critic_frame_latent"
            if is_wcm
            else "normalized_actor_encoder_features"
        ),
        "wcm_dynamics_target_mode": (
            WCM_DYNAMICS_TARGET_MODE if is_wcm else None
        ),
        "dynamics_prediction_offsets": tuple(
            int(value) for value in args.dynamics_prediction_offsets
        ),
        "dynamics_prediction_consumed_by_q": False if is_wcm else bool(
            args.critic_q_use_predicted_next_latent
        ),
        "dynamics_target_last_sync_step": int(
            dynamics_target_last_sync_step
        ),
        "sigreg_weight": float(args.sigreg_weight),
        "sigreg_knots": int(args.sigreg_knots),
        "sigreg_num_projections": int(args.sigreg_num_projections),
        "sigreg_global_batch": bool(args.sigreg_global_batch),
        "monte_carlo_return_weight": 0.0,
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
        "pretrained_dp_identity": copy.deepcopy(args.pretrained_dp_identity),
        "task": str(args.task),
        "dataset": str(args.dataset),
        "dataset_identity": copy.deepcopy(args.dataset_identity),
        "single_dataloader": distributed_world_size == 1,
        "sampling": (
            "distributed_shuffled_SequenceDataset_indices"
            if distributed_world_size > 1
            else "uniform_shuffled_SequenceDataset_indices"
        ),
        "reward_mode": str(args.reward_mode),
        "reward_definition": REWARD_DEFINITIONS[args.reward_mode],
        "critic_reward_source": (
            "rewards=source_environment_task_reward"
            if args.reward_mode == "task"
            else "rewards=canonical_first_success_terminal_reward"
            if args.reward_mode == "terminal_success"
            else "rewards=expert_1_non_expert_0"
        ),
        "actor_training_objective": (
            (
                "conditional_diffusion_BC_all_mixed_rows_"
                f"{args.actor_condition_mode}_"
                + (
                    "from_source_chunk_IDQL_actor"
                    if args.initialization == "source_chunk_idql_joint"
                    else "from_pretrained_DP_ema"
                )
                + (
                    "_plus_pretrained_EMA_noise_prediction_reference_at_condition_1"
                    if float(args.actor_reference_weight) > 0.0
                    else ""
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
        ),
        "conditioned_actor": bool(args.conditioned_actor),
        "actor_condition_mode": (
            str(args.actor_condition_mode) if args.conditioned_actor else None
        ),
        "actor_condition_label_definition": (
            actor_condition_definition(args.actor_condition_mode)
            if args.conditioned_actor
            else None
        ),
        "actor_condition_source": (
            "actor_condition_at_current_transition"
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
        "actor_condition_adapter": (
            str(args.actor_condition_adapter_type)
            if args.conditioned_actor else None
        ),
        "actor_source_mask": (
            "none_all_shared_batch_rows"
            if trains_joint_actor(args)
            else "none_actor_frozen"
        ),
        "critic_source_mask": "none_all_shared_batch_rows",
        "critic_training_objective": (
            "task_reward_semi_mdp_chunk_iql_with_shared_wcm_dynamics"
            if is_wcm and args.reward_mode == "task"
            else "terminal_success_semi_mdp_chunk_iql_with_shared_wcm_dynamics"
            if is_wcm and args.reward_mode == "terminal_success"
            else "rise_semi_mdp_chunk_iql_with_shared_wcm_dynamics"
            if is_wcm
            else "task_reward_semi_mdp_chunk_iql_with_actor_encoder_dynamics"
            if args.reward_mode == "task"
            else "terminal_success_semi_mdp_chunk_iql_with_actor_encoder_dynamics"
            if args.reward_mode == "terminal_success"
            else "rise_semi_mdp_chunk_iql_with_actor_encoder_dynamics"
        ),
        "critic_input_mode": (
            "shared_raw_observation_causal_temporal_state"
            if is_wcm
            else "independent_raw_observation_history_chunk_encoders"
        ),
        "critic_action_space": "pretrained_dp_normalized_action_chunk",
        "critic_hidden_dims": tuple(int(x) for x in args.critic_hidden_dims),
        "critic_chunk_horizon": int(args.chunk_horizon),
        "critic_observation_horizon": int(
            args.critic_observation_horizon
        ),
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
            "periodic_hard_copy_online_critic_frame_encoder_and_projection"
            if is_wcm
            else "periodic_deployed_actor_ema_obs_encoder"
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
            "explicit_per_trajectory_camera_crop_plan_shared_across_all_wcm_"
            "history_bootstrap_q_target_and_periodic_teacher_future_frames"
            if is_wcm
            else "paired_online_and_target_encoder_random_crops_via_rng_fork"
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
        "actor_adapter_lr": float(args.actor_adapter_lr),
        "actor_unet_lr": float(args.actor_unet_lr),
        "actor_obs_encoder_lr": float(args.actor_obs_encoder_lr),
        "actor_reference_weight": float(args.actor_reference_weight),
        "actor_reference_batch_fraction": float(
            args.actor_reference_batch_fraction
        ),
        "actor_lr_scheduler": str(args.actor_lr_scheduler),
        "actor_lr_warmup_steps": int(args.actor_lr_warmup_steps),
        "actor_lr_total_steps": int(args.actor_lr_total_steps),
        "actor_lr_num_cycles": float(args.actor_lr_num_cycles),
        "critic_lr": float(args.critic_lr),
        "encoder_lr": float(args.encoder_lr),
        "vf_lr": float(args.vf_lr),
        "actor_frozen": bool(not trains_joint_actor(args)),
        "actor_encoder_trainable": bool(trains_joint_actor(args)),
        "actor_ema_optimization_step": int(actor_ema_optimization_step),
        "global_samples_seen": int(global_samples_seen),
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
            "world_size": int(
                distributed_world_size
            ),
            "backend": (
                distributed_context.backend
                if distributed_context is not None
                else "none"
            ),
            "gradient_sync": "mean_all_reduce_before_optimizer_step",
        },
        "distributed_rank_states": rank_runtime_states,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    configure_critic_architecture_args(args)
    distributed = initialize_distributed(args)
    args.distributed = bool(distributed.enabled)
    args.distributed_rank = int(distributed.rank)
    args.distributed_local_rank = int(distributed.local_rank)
    args.distributed_world_size = int(distributed.world_size)
    configure_batch_semantics(args, distributed)
    args.dataset_identity = mixed_dataset_identity(args.dataset)
    validate_mixed_dataset_source_identity(args.dataset_identity)
    if distributed.is_main_process:
        if args.resume_checkpoint is None:
            cleaned_temporaries = prepare_fresh_output_directory(
                args.output_dir
            )
            if cleaned_temporaries:
                print(
                    f"Removed stale checkpoint temporaries: "
                    f"{[path.name for path in cleaned_temporaries]}",
                    flush=True,
                )
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.enabled:
        dist.barrier()
    device = distributed.device
    # All ranks construct identical initial parameters. Rank-specific training
    # RNG streams are installed after model initialization and resume loading.
    seed_process(args.seed, device)

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
        validate_resume_semantics(resume_state, args)
        saved_distributed = resume_state.get("distributed_training", {})
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

        saved_dynamics_mode = str(
            resume_state.get(
                "dynamics_prediction_mode",
                "",
            )
        )
        expected_dynamics_mode = (
            WCM_DYNAMICS_PREDICTION_MODE
            if args.critic_architecture == WCM_CRITIC_ARCHITECTURE
            else DYNAMICS_PREDICTION_MODE
        )
        if saved_dynamics_mode != expected_dynamics_mode:
            raise ValueError(
                f"resume dynamics_prediction_mode={saved_dynamics_mode!r} "
                f"does not match {expected_dynamics_mode!r}; start a "
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
        saved_args = resume_state.get("args", {})
        resume_float_fields = {
            "actor_adapter_lr": float(args.actor_adapter_lr),
            "actor_unet_lr": float(args.actor_unet_lr),
            "actor_obs_encoder_lr": float(args.actor_obs_encoder_lr),
            "actor_reference_weight": float(args.actor_reference_weight),
            "actor_reference_batch_fraction": float(args.actor_reference_batch_fraction),
            "critic_lr": float(args.critic_lr),
            "encoder_lr": float(args.encoder_lr),
            "vf_lr": float(args.vf_lr),
        }
        legacy_float_defaults: dict[str, float] = {}
        if bool(args.validate_resume_only):
            legacy_actor_lr = saved_args.get("actor_lr")
            if legacy_actor_lr is not None:
                for field in (
                    "actor_adapter_lr",
                    "actor_unet_lr",
                    "actor_obs_encoder_lr",
                ):
                    legacy_float_defaults[field] = float(legacy_actor_lr)
            legacy_float_defaults["actor_reference_weight"] = 0.0
            legacy_float_defaults["actor_reference_batch_fraction"] = 0.25

        for field, requested_value in resume_float_fields.items():
            saved_value = float(saved_args.get(field, legacy_float_defaults.get(field, 0.0)))
            if saved_value != requested_value:
                raise ValueError(
                    f"resume {field}={saved_value} does not match requested "
                    f"{field}={requested_value}"
                )
        if saved_initialization in JOINT_ACTOR_INITIALIZATIONS:
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
            saved_condition_mode = str(
                resume_state.get("args", {}).get(
                    "actor_condition_mode", "human_only"
                )
            )
            if saved_condition_mode != str(args.actor_condition_mode):
                raise ValueError(
                    "resume actor_condition_mode="
                    f"{saved_condition_mode!r} does not match requested "
                    f"{args.actor_condition_mode!r}; start a fresh output "
                    "directory"
                )
            saved_condition_definition = str(
                resume_state.get("actor_condition_label_definition", "")
            )
            required_condition_definition = actor_condition_definition(
                args.actor_condition_mode
            )
            if saved_condition_definition != required_condition_definition:
                raise ValueError(
                    "resume actor condition definition="
                    f"{saved_condition_definition!r} does not match requested "
                    f"{required_condition_definition!r}; start a fresh output "
                    "directory"
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

    current_dp_identity = file_stat_identity(Path(pretrained_dp_checkpoint))
    if not current_dp_identity["exists"]:
        raise FileNotFoundError(
            f"pretrained DP checkpoint does not exist: {pretrained_dp_checkpoint}"
        )
    if resume_state is not None:
        saved_dp_identity = resume_state.get("pretrained_dp_identity")
        if saved_dp_identity is None:
            if not bool(args.validate_resume_only):
                raise ValueError(
                    "resume checkpoint has no immutable pretrained DP identity"
                )
            print(
                "WARNING: legacy checkpoint has no immutable pretrained DP identity",
                flush=True,
            )
        elif saved_dp_identity != current_dp_identity:
            raise ValueError(
                "resume pretrained DP checkpoint identity has changed"
            )
    elif source_for_warm_start is not None:
        source_dp_identity = source_for_warm_start.get("pretrained_dp_identity")
        if (
            source_dp_identity is not None
            and source_dp_identity != current_dp_identity
        ):
            raise ValueError(
                "source checkpoint pretrained DP identity has changed"
            )
    args.pretrained_dp_identity = current_dp_identity

    if bool(args.validate_resume_only):
        if resume_state is None:
            raise ValueError("--validate-resume-only requires --resume-checkpoint")
        validate_chunk_source(resume_state, args)
        saved_epoch = int(resume_state.get("epoch", -1))
        if saved_epoch != int(args.epochs):
            raise ValueError(
                f"completion checkpoint epoch={saved_epoch} does not match "
                f"requested epochs={args.epochs}"
            )
        result = {
            "validated": True,
            "checkpoint": str(args.resume_checkpoint),
            "epoch": saved_epoch,
            "task": str(args.task),
        }
        print(json.dumps(result, indent=2), flush=True)
        return result

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
    actor_reference_audit = {
        "enabled": False,
        "weight": float(args.actor_reference_weight),
        "batch_fraction": float(args.actor_reference_batch_fraction),
        "reason": "actor_frozen_or_zero_weight",
    }
    if trains_joint_actor(args) and float(args.actor_reference_weight) > 0.0:
        actor_reference_audit = {
            "enabled": True,
            **install_pretrained_actor_reference(
                actor_algo,
                weight=args.actor_reference_weight,
                batch_fraction=args.actor_reference_batch_fraction,
            ),
        }
    if trains_joint_actor(args):
        if resume_state is not None:
            # Install the exact saved adapter architecture before constructing
            # optimizer groups. This retains legacy residual-conditioned actors
            # while new runs continue to use the FiLM adapter.
            actor_algo.deserialize(
                resume_state["actor_model"],
                load_optimizers=False,
            )
            if actor_algo.ema is not None:
                actor_algo.ema.optimization_step = int(
                    resume_state.get("actor_ema_optimization_step", 0)
                )
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
    args.actor_condition_adapter_type = (
        type(actor_algo.nets["policy"]["condition_adapter"]).__name__
        if args.conditioned_actor
        else None
    )

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
    if int(args.critic_observation_horizon) > args.observation_horizon:
        raise ValueError(
            "critic_observation_horizon cannot exceed the pretrained actor "
            f"observation horizon: {args.critic_observation_horizon} > "
            f"{args.observation_horizon}"
        )
    if (
        args.critic_architecture == WCM_CRITIC_ARCHITECTURE
        and int(args.critic_observation_horizon) < 2
    ):
        raise ValueError(
            "WCM causal temporal training requires critic_observation_horizon "
            "of at least 2"
        )
    args.actor_prediction_horizon = int(
        actor_algo.algo_config.horizon.prediction_horizon
    )
    args.actor_action_horizon = actor_horizon
    chunk_reference_state = (
        source_for_warm_start or resume_state
    )
    if (
        args.initialization == "source_chunk_idql_joint"
        and int(chunk_reference_state.get("action_dim", -1))
        != int(args.action_dim)
    ):
        raise ValueError(
            f"source action_dim={chunk_reference_state.get('action_dim')} "
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
            actor_condition_mode=str(args.actor_condition_mode),
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
        and int(args.resolved_critic_vf_lr_warmup_steps)
        >= int(args.critic_vf_lr_total_steps)
    ):
        raise ValueError(
            "resolved_critic_vf_lr_warmup_steps="
            f"{args.resolved_critic_vf_lr_warmup_steps} (reference "
            f"{args.critic_vf_lr_warmup_steps}) must be smaller than the "
            f"{args.critic_vf_lr_total_steps} critic/VF training steps"
        )
    if resume_state is not None:
        saved_args = resume_state.get("args", {})
        schedule_fields = [
            ("batch_size", int),
            ("effective_global_batch_size", int),
            ("schedule_reference_batch_size", int),
            ("steps_per_epoch", int),
            ("critic_vf_lr_scheduler", str),
            ("critic_vf_lr_total_steps", int),
            ("critic_vf_lr_num_cycles", float),
            *[
                (f"resolved_{field}", int)
                for field in SAMPLE_SCALED_STEP_FIELDS
            ],
            ("resolved_dynamics_target_sync_interval", int),
            ("resolved_target_tau", float),
        ]
        if trains_joint_actor(args):
            schedule_fields.append(("actor_lr_total_steps", int))
        for field, cast in schedule_fields:
            if field in saved_args:
                saved_raw_value = saved_args[field]
            elif field.startswith("resolved_"):
                legacy_field = field[len("resolved_") :]
                saved_raw_value = saved_args.get(legacy_field)
                if saved_raw_value is None:
                    raise ValueError(
                        f"resume checkpoint has no {field} configuration"
                    )
            else:
                raise ValueError(
                    f"resume checkpoint has no {field} configuration"
                )
            saved_value = cast(saved_raw_value)
            requested_value = cast(getattr(args, field))
            if saved_value != requested_value:
                raise ValueError(
                    f"resume {field}={saved_value} does not match requested "
                    f"{field}={requested_value}; legacy checkpoints created "
                    "before sample-aware schedules should be used as a "
                    "--source-chunk-idql-checkpoint for a fresh round"
                )
    if trains_joint_actor(args):
        if (
            args.actor_lr_scheduler == "cosine"
            and int(args.resolved_actor_lr_warmup_steps)
            >= int(args.actor_lr_total_steps)
        ):
            raise ValueError(
                "resolved_actor_lr_warmup_steps="
                f"{args.resolved_actor_lr_warmup_steps} (reference "
                f"{args.actor_lr_warmup_steps}) must be "
                "smaller than the "
                f"{args.actor_lr_total_steps} actor training steps"
            )
        configure_chunk_actor_optimizer(
            actor_algo,
            conditioned_actor=bool(args.conditioned_actor),
            adapter_lr=args.actor_adapter_lr,
            unet_lr=args.actor_unet_lr,
            obs_encoder_lr=args.actor_obs_encoder_lr,
            scheduler_type=args.actor_lr_scheduler,
            warmup_steps=args.resolved_actor_lr_warmup_steps,
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
        if actor_algo.reference_nets is not None:
            actor_algo.reference_nets.train()
            actor_algo.reference_nets.requires_grad_(False)
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

    is_wcm = args.critic_architecture == WCM_CRITIC_ARCHITECTURE
    wcm_system: WCMChunkValueSystem | None = None
    wcm_target_system: WCMChunkValueSystem | None = None
    wcm_dynamics_target_encoder: WCMFrameTargetEncoder | None = None
    wcm_optimizer: torch.optim.Optimizer | None = None
    wcm_lr_scheduler = None
    critics = nn.ModuleList()
    targets = nn.ModuleList()
    vf: RiseValueNetwork | None = None
    dynamics_target_encoder: nn.Module | None = None
    critic_optimizers: list[torch.optim.Optimizer] = []
    vf_optimizer: torch.optim.Optimizer | None = None
    critic_lr_schedulers: list[Any] = []
    vf_lr_scheduler = None
    warm_start_audit: dict[str, Any] = {"mode": "resume_checkpoint"}

    if is_wcm:
        wcm_system, wcm_target_system = make_wcm_chunk_value_system(
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
            observation_horizon=args.critic_observation_horizon,
            temporal_num_layers=args.temporal_num_layers,
            temporal_num_heads=args.temporal_num_heads,
            temporal_feedforward_dim=args.temporal_feedforward_dim,
            temporal_dropout=args.temporal_dropout,
            dynamics_prediction_offsets=args.dynamics_prediction_offsets,
        )
        reference_state = None
        if resume_state is not None:
            reference_state = resume_state["chunk_value_system"]
            warm_start_audit = {
                "mode": "resume_checkpoint",
                "system": match_encoder_normalization_to_checkpoint(
                    wcm_system, reference_state
                ),
            }
        elif (
            source_for_warm_start is not None
            and args.initialization == "source_chunk_idql_joint"
        ):
            reference_state = source_for_warm_start["chunk_value_system"]
            warm_start_audit = {
                "mode": "source_chunk_idql_complete_wcm_system",
                "checkpoint": str(args.source_chunk_idql_checkpoint),
                "fresh_optimizers_and_schedulers": True,
                "fresh_epoch_and_global_step": True,
                "system": match_encoder_normalization_to_checkpoint(
                    wcm_system, reference_state
                ),
            }
        else:
            encoder_audit = copy_deployed_dp_encoder_state(
                wcm_system, actor_algo
            )
            warm_start_audit = {
                "mode": "deployed_actor_raw_encoder_fresh_wcm_heads",
                "encoder": encoder_audit,
                "one_step_source_used_for_actor_only": bool(
                    source_for_warm_start is not None
                ),
            }
        wcm_target_system = copy.deepcopy(wcm_system)
        wcm_dynamics_target_encoder = WCMFrameTargetEncoder(wcm_system)
        wcm_system = wcm_system.float().to(device)
        wcm_target_system = wcm_target_system.float().to(device)
        wcm_dynamics_target_encoder = (
            wcm_dynamics_target_encoder.float().to(device)
        )
        target_encoder_output_dim = int(wcm_system.latent_dim)
        wcm_optimizer = make_wcm_optimizer(
            wcm_system,
            critic_lr=args.critic_lr,
            encoder_lr=args.encoder_lr,
            vf_lr=args.vf_lr,
        )
        wcm_lr_scheduler = make_step_lr_scheduler(
            wcm_optimizer,
            scheduler_type=args.critic_vf_lr_scheduler,
            warmup_steps=args.resolved_critic_vf_lr_warmup_steps,
            total_steps=args.critic_vf_lr_total_steps,
            num_cycles=args.critic_vf_lr_num_cycles,
        )
    else:
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
            observation_horizon=args.critic_observation_horizon,
            q_use_predicted_next_latent=(
                args.critic_q_use_predicted_next_latent
            ),
        )
        if resume_state is not None:
            warm_start_audit = {
                "mode": "resume_checkpoint",
                "critics": [
                    match_encoder_normalization_to_checkpoint(critic, state)
                    for critic, state in zip(critics, resume_state["critics"])
                ],
                "vf": match_encoder_normalization_to_checkpoint(
                    vf, resume_state["vf"]
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
                        critics, source_for_warm_start["critics"]
                    )
                ],
                "vf": match_encoder_normalization_to_checkpoint(
                    vf, source_for_warm_start["vf"]
                ),
            }
            targets = copy.deepcopy(critics)
        elif source_for_warm_start is not None:
            if int(args.critic_observation_horizon) == 1:
                vf.load_state_dict(source_for_warm_start["vf"], strict=True)
                vf_warm_start_audit = {
                    "mode": "complete_one_frame_value_network",
                    "tensor_count": int(len(source_for_warm_start["vf"])),
                }
            else:
                vf_warm_start_audit = copy_matching_vf_encoder_state(
                    vf, source_for_warm_start["vf"]
                )
            warm_start_audit = {
                "mode": "source_one_step_idql_representations",
                "critics": [
                    copy_matching_encoder_state(critic, state)
                    for critic, state in zip(
                        critics, source_for_warm_start["critics"]
                    )
                ],
                "vf": vf_warm_start_audit,
            }
            targets = copy.deepcopy(critics)
        else:
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
                warmup_steps=args.resolved_critic_vf_lr_warmup_steps,
                total_steps=args.critic_vf_lr_total_steps,
                num_cycles=args.critic_vf_lr_num_cycles,
            )
            for optimizer in critic_optimizers
        ]
        vf_lr_scheduler = make_step_lr_scheduler(
            vf_optimizer,
            scheduler_type=args.critic_vf_lr_scheduler,
            warmup_steps=args.resolved_critic_vf_lr_warmup_steps,
            total_steps=args.critic_vf_lr_total_steps,
            num_cycles=args.critic_vf_lr_num_cycles,
        )

    start_epoch = 0
    global_step = 0
    global_samples_seen = 0
    dynamics_target_last_sync_step = 0
    history: list[dict] = []
    if resume_state is not None:
        scheduler_enabled = args.critic_vf_lr_scheduler != "constant"
        if is_wcm:
            wcm_system.load_state_dict(
                resume_state["chunk_value_system"], strict=True
            )
            wcm_target_system.load_state_dict(
                resume_state["chunk_value_target"], strict=True
            )
            wcm_dynamics_target_encoder.load_state_dict(
                resume_state[WCM_DYNAMICS_TARGET_STATE_KEY],
                strict=True,
            )
            wcm_optimizer.load_state_dict(
                resume_state["chunk_value_optimizer"]
            )
            scheduler_state = resume_state["chunk_value_lr_scheduler"]
            if (scheduler_state is not None) != scheduler_enabled:
                raise ValueError(
                    "resume WCM LR scheduler state does not match "
                    f"critic_vf_lr_scheduler={args.critic_vf_lr_scheduler}"
                )
            if wcm_lr_scheduler is not None:
                wcm_lr_scheduler.load_state_dict(scheduler_state)
        else:
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
        global_samples_seen = int(
            resume_state.get(
                "global_samples_seen",
                global_step * int(args.effective_global_batch_size),
            )
        )
        dynamics_target_last_sync_step = int(
            resume_state.get("dynamics_target_last_sync_step", 0)
        )
        if scheduler_enabled:
            scheduler_steps = (
                [int(wcm_lr_scheduler.last_epoch)]
                if is_wcm
                else [
                    *[
                        int(scheduler.last_epoch)
                        for scheduler in critic_lr_schedulers
                        if scheduler is not None
                    ],
                    int(vf_lr_scheduler.last_epoch),
                ]
            )
            if any(step != global_step for step in scheduler_steps):
                raise ValueError(
                    f"critic/VF LR scheduler steps {scheduler_steps} do not "
                    f"match checkpoint global_step={global_step}"
                )
        history = list(resume_state.get("history", []))
        rank_runtime_states = resume_state.get("distributed_rank_states")
        if (
            distributed.enabled
            and bool(saved_distributed.get("enabled", False))
            and rank_runtime_states is None
        ):
            raise ValueError(
                "distributed checkpoint is missing per-rank runtime states"
            )
        if distributed.enabled and rank_runtime_states is not None:
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
            restore_process_rng_state(
                rank_runtime.get("rng_state"),
                device,
            )
        elif distributed.enabled:
            # A legacy single-process checkpoint has no per-rank stochastic
            # state. Preserve its model/optimizer state and create independent
            # deterministic streams for the new ranks.
            loader_generator.manual_seed(int(args.seed) + distributed.rank)
            seed_process(int(args.seed) + distributed.rank, device)
        else:
            loader_generator.set_state(
                resume_state["loader_generator_state"].cpu()
            )
            saved_rng_state = resume_state.get("rng_state")
            if saved_rng_state and "cuda_local" in saved_rng_state:
                restore_process_rng_state(saved_rng_state, device)
            else:
                restore_rng_state(saved_rng_state)
        if distributed.is_main_process:
            print(
                f"Resumed {args.resume_checkpoint} at epoch={start_epoch} "
                f"step={global_step}",
                flush=True,
            )
    elif (
        source_for_warm_start is not None
        and args.initialization == "source_chunk_idql_joint"
    ):
        if is_wcm:
            wcm_system.load_state_dict(
                source_for_warm_start["chunk_value_system"], strict=True
            )
            wcm_target_system.load_state_dict(
                source_for_warm_start["chunk_value_target"], strict=True
            )
            dynamics_sync_audit = hard_sync_wcm_dynamics_target_encoder(
                wcm_dynamics_target_encoder,
                wcm_system,
            )
            warm_start_audit["dynamics_target_resync"] = {
                **dynamics_sync_audit,
                "source_target_mode": (
                    checkpoint_wcm_dynamics_target_mode(
                        source_for_warm_start
                    )
                ),
                "fresh_round_sync_step": 0,
            }
        else:
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
            dynamics_sync_audit = sync_actor_dynamics_target_encoder(
                dynamics_target_encoder,
                actor_algo,
            )
            warm_start_audit["dynamics_target_resync"] = {
                **dynamics_sync_audit,
                "source_last_sync_step": int(
                    source_for_warm_start["dynamics_target_last_sync_step"]
                ),
                "fresh_round_sync_step": 0,
            }
            vf.load_state_dict(source_for_warm_start["vf"], strict=True)
        if distributed.is_main_process:
            print(
                "Warm-started actor, actor EMA, complete critic/value system at "
                f"{args.source_chunk_idql_checkpoint}; starting fresh "
                "optimizers, LR schedules, epoch=0, and global_step=0",
                flush=True,
            )
    if resume_state is not None and start_epoch > int(args.epochs):
        raise ValueError(
            f"resume checkpoint passed epoch={start_epoch}; requested "
            f"epochs={args.epochs}. Use that checkpoint for evaluation or "
            "increase epochs with a fresh source warm start and schedule."
        )
    if is_wcm:
        configure_wcm_target_random_crops(wcm_target_system)
        configure_wcm_dynamics_target_random_crops(
            wcm_dynamics_target_encoder
        )
        synchronized_modules: list[nn.Module] = [
            actor_algo.nets,
            wcm_system,
            wcm_target_system,
            wcm_dynamics_target_encoder,
        ]
        training_buffer_modules = (
            [actor_algo.nets, wcm_system]
            if trains_joint_actor(args)
            else [wcm_system]
        )
    else:
        configure_target_random_crops(targets)
        configure_encoder_target_random_crops(dynamics_target_encoder)
        synchronized_modules = [
            actor_algo.nets,
            critics,
            targets,
            dynamics_target_encoder,
            vf,
        ]
        training_buffer_modules = (
            [actor_algo.nets, critics, vf]
            if trains_joint_actor(args)
            else [critics, vf]
        )
    if actor_algo.ema is not None:
        synchronized_modules.append(actor_algo.ema.averaged_model)
    if getattr(actor_algo, "reference_nets", None) is not None:
        synchronized_modules.append(actor_algo.reference_nets)
    synchronize_training_buffers = modules_have_mutable_batch_norm(
        training_buffer_modules
    )
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
    wcm_gradient_sync_fn = (
        (
            lambda parameters: all_reduce_gradients(
                parameters,
                distributed,
                bucket_cap_mb=args.gradient_bucket_cap_mb,
                preserve_unused_parameters=False,
            )
        )
        if distributed.enabled
        else None
    )
    if trains_joint_actor(args):
        actor_algo.gradient_sync_fn = gradient_sync_fn
    if distributed.enabled and resume_state is None:
        seed_process(int(args.seed) + distributed.rank, device)
    repair_completed_resume = (
        resume_state is not None
        and start_epoch == int(args.epochs)
    )
    publish_epoch_zero = resume_state is None
    del resume_state, source_for_warm_start

    if not trains_joint_actor(args):
        actor_algo.nets.cpu()
        if actor_algo.ema is not None:
            actor_algo.ema.averaged_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def current_checkpoint_payload(
        checkpoint_epoch: int,
        rank_runtime_states: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        return checkpoint_payload(
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
            wcm_system=wcm_system,
            wcm_target_system=wcm_target_system,
            wcm_dynamics_target_encoder=wcm_dynamics_target_encoder,
            wcm_optimizer=wcm_optimizer,
            wcm_lr_scheduler=wcm_lr_scheduler,
            action_stats=action_stats,
            epoch=checkpoint_epoch,
            global_step=global_step,
            global_samples_seen=global_samples_seen,
            history=history,
            loader_generator=loader_generator,
            rank_runtime_states=rank_runtime_states,
            distributed_context=distributed,
        )

    if publish_epoch_zero:
        rank_runtime_states = gather_rank_runtime_states(
            loader_generator,
            distributed,
        )
        if distributed.is_main_process:
            latest = args.output_dir / LATEST_CHECKPOINT_NAME
            payload = current_checkpoint_payload(
                0,
                rank_runtime_states,
            )
            atomic_torch_save(payload, latest)
            print(
                f"Saved recovery checkpoint {latest} at epoch=0 step=0",
                flush=True,
            )
        if distributed.enabled:
            dist.barrier()

    architecture = {
        "actor": actor_audit,
        "conditional_diffusion_actor": bool(args.conditioned_actor),
        "actor_condition_adapter": args.actor_condition_adapter_type,
        "critic_architecture": str(args.critic_architecture),
        "critic_parameter_counts": (
            [parameter_count(wcm_system)]
            if is_wcm
            else [parameter_count(x) for x in critics]
        ),
        "target_critic_parameter_counts": (
            [parameter_count(wcm_target_system)]
            if is_wcm
            else [parameter_count(x) for x in targets]
        ),
        "dynamics_target_encoder_parameter_count": (
            parameter_count(wcm_dynamics_target_encoder)
            if is_wcm
            else parameter_count(dynamics_target_encoder)
        ),
        "dynamics_target_encoder_output_dim": target_encoder_output_dim,
        "vf_parameter_count": (
            parameter_count(wcm_system.nets["value_head"])
            if is_wcm
            else parameter_count(vf)
        ),
        "independent_raw_obs_encoders": not is_wcm,
        "shared_state_representation": bool(is_wcm),
        "critic_chunk_horizon": int(args.chunk_horizon),
        "critic_observation_horizon": int(
            args.critic_observation_horizon
        ),
        "critic_q_head_inputs": list(
            architecture_q_head_inputs(
                args.critic_architecture,
                args.critic_q_use_predicted_next_latent,
            )
        ),
        "critic_q_use_predicted_next_latent": bool(
            args.critic_q_use_predicted_next_latent
        ),
        "critic_q_predicted_next_normalization": (
            PREDICTED_NEXT_Q_NORMALIZATION
            if args.critic_q_use_predicted_next_latent
            else None
        ),
        "critic_representation_modules": (
            ["encoder", "frame_projection", "temporal_trunk"]
            if is_wcm
            else ["encoder", "context", "context_norm"]
        ),
        "latent_dynamics": True,
        "actor_encoder_feature_dynamics": not is_wcm,
        "dynamics_prediction_mode": (
            WCM_DYNAMICS_PREDICTION_MODE
            if is_wcm
            else DYNAMICS_PREDICTION_MODE
        ),
        "dynamics_prediction_output": (
            "shared_critic_frame_latents"
            if is_wcm
            else "raw_actor_encoder_features"
        ),
        "dynamics_prediction_output_dim": target_encoder_output_dim,
        "dynamics_prediction_residual": bool(is_wcm),
        "dynamics_prediction_offsets": list(args.dynamics_prediction_offsets),
        "dynamics_prediction_consumed_by_q": (
            False if is_wcm else bool(args.critic_q_use_predicted_next_latent)
        ),
        "dynamics_target_encoder": (
            "frozen_complete_frame_encoder_and_projection_hard_copy"
            if is_wcm
            else "frozen_copy_periodically_hard_synced_from_deployed_actor_ema_"
            "obs_encoder"
            if trains_joint_actor(args)
            else "frozen_copy_of_deployed_actor_ema_obs_encoder"
        ),
        "dynamics_target_update": (
            "periodic_full_state_dict_hard_sync"
            if is_wcm
            else "periodic_hard_sync"
            if trains_joint_actor(args)
            else "fixed_after_initialization"
        ),
        "wcm_dynamics_target_mode": (
            WCM_DYNAMICS_TARGET_MODE if is_wcm else None
        ),
        "dynamics_target_context_mlp": False,
        "training_augmentation": (
            "one_explicit_random_crop_per_trajectory_and_camera_shared_across_"
            "history_bootstrap_q_target_and_periodic_teacher_future_frames"
            if is_wcm
            else "paired_online_and_target_encoder_random_crops_via_rng_fork"
        ),
        "target_encoder_mode": (
            "eval_except_crop_randomizers_in_training_mode"
        ),
        "vf_training": (
            "shared_temporal_state_expectile_head"
            if is_wcm
            else "head_from_step_zero_raw_observation_encoder_delayed"
        ),
        "temporal_num_layers": int(args.temporal_num_layers),
        "temporal_num_heads": int(args.temporal_num_heads),
        "temporal_feedforward_dim": int(args.temporal_feedforward_dim),
        "temporal_dropout": float(args.temporal_dropout),
        "sigreg": {
            "weight": float(args.sigreg_weight),
            "knots": int(args.sigreg_knots),
            "num_projections": int(args.sigreg_num_projections),
            "global_batch": bool(args.sigreg_global_batch),
            "scope": "newest_temporal_context",
            "distributed_reduction": "global_characteristic_moments",
        },
        "monte_carlo_return_weight": 0.0,
        "actor_reference_distillation": actor_reference_audit,
        "warm_start": warm_start_audit,
    }
    startup = {
        "actor_reference_distillation": actor_reference_audit,
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
        "pretrained_dp_identity": copy.deepcopy(args.pretrained_dp_identity),
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
            "provenance_identity": copy.deepcopy(args.dataset_identity),
            "actor_conditioning": condition_audit,
        },
        "loader": {
            "class": dataset.__class__.__name__,
            "num_loaders": int(distributed.world_size),
            "sampler": loader.sampler.__class__.__name__,
            "balanced_sampling": False,
            "batch_size": int(args.batch_size),
            "batch_size_per_rank": int(args.batch_size),
            "effective_global_batch_size": int(
                args.batch_size * distributed.world_size
            ),
            "num_batches": int(len(loader)),
            "steps_per_epoch": int(args.steps_per_epoch),
            "steps_per_epoch_source": args.steps_per_epoch_source,
            "sequence_length": int(sequence_length),
            "sparse_chunk_loader": bool(args.sparse_chunk_loader),
            "num_workers_per_rank": int(args.num_workers),
            "total_worker_processes": int(
                args.num_workers * distributed.world_size
            ),
            "prefetch_factor": (
                int(args.prefetch_factor) if int(args.num_workers) > 0 else None
            ),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "dense_target_read_strategy": (
                "one_coalesced_successor_window_per_observation_key"
                if args.sparse_chunk_loader
                and len(args.dynamics_prediction_offsets) > 0
                else "not_applicable"
            ),
            "observation_frames_per_sample": (
                int(args.observation_horizon)
                + int(args.critic_observation_horizon)
                + len(args.dynamics_prediction_offsets)
                if args.sparse_chunk_loader
                else 2
                * (
                    int(args.observation_horizon) - 1 + int(sequence_length)
                )
            ),
        },
        "data_routing": {
            "shared_loader": True,
            "critic_rows": "all_human_success_failure",
            "critic_reward_source": (
                "rewards=source_environment_task_reward"
                if args.reward_mode == "task"
                else "rewards=canonical_first_success_terminal_reward"
                if args.reward_mode == "terminal_success"
                else "rewards=expert_1_non_expert_0"
            ),
            "actor_rows": (
                "all_human_success_failure"
                if trains_joint_actor(args)
                else "none_actor_frozen"
            ),
            "actor_condition_labels": (
                actor_condition_labels(args.actor_condition_mode)
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
        "batch_semantics": {
            "batch_size_control": "per_gpu_CHUNK_BATCH_SIZE",
            "batch_size_per_rank": int(args.batch_size),
            "world_size": int(distributed.world_size),
            "effective_global_batch_size": int(
                args.effective_global_batch_size
            ),
            "schedule_reference_batch_size": int(
                args.schedule_reference_batch_size
            ),
            "effective_to_reference_ratio": float(
                args.schedule_batch_ratio
            ),
            "learning_rates_automatically_scaled": False,
            "sample_scaled_step_inputs_are_reference_batch_steps": True,
            "resolved_sample_scaled_steps": {
                field: int(getattr(args, f"resolved_{field}"))
                for field in SAMPLE_SCALED_STEP_FIELDS
            },
            "reference_target_tau": float(args.target_tau),
            "resolved_target_tau": float(args.resolved_target_tau),
            "target_tau_step_unit": "optimizer_update",
            "dynamics_target_sync_interval": int(
                args.resolved_dynamics_target_sync_interval
            ),
            "dynamics_target_sync_step_unit": "optimizer_update",
            "actor_ema_step_unit": "optimizer_update",
            "actor_ema_sample_scaled": False,
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
            "actor_adapter_lr": float(args.actor_adapter_lr),
            "actor_unet_lr": float(args.actor_unet_lr),
            "actor_obs_encoder_lr": float(args.actor_obs_encoder_lr),
            "actor_optimizer_type": ACTOR_OPTIMIZER_TYPE,
            "actor_weight_decay": ACTOR_WEIGHT_DECAY,
            "actor_reference_weight": float(args.actor_reference_weight),
            "actor_reference_batch_fraction": float(
                args.actor_reference_batch_fraction
            ),
            "actor_lr_scheduler": str(args.actor_lr_scheduler),
            "actor_lr_warmup_steps": int(args.actor_lr_warmup_steps),
            "actor_lr_total_steps": int(args.actor_lr_total_steps),
            "actor_lr_num_cycles": float(args.actor_lr_num_cycles),
            "conditioned_actor": bool(args.conditioned_actor),
            "actor_condition_mode": str(args.actor_condition_mode),
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
            "critic_observation_horizon": int(
                args.critic_observation_horizon
            ),
            "critic_q_use_predicted_next_latent": bool(
                args.critic_q_use_predicted_next_latent
            ),
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
        "distributed": {
            "enabled": bool(distributed.enabled),
            "world_size": int(distributed.world_size),
            "backend": distributed.backend,
            "launcher": "torchrun" if distributed.enabled else "python",
            "gradient_sync": "bounded_async_bucketed_mean_all_reduce",
            "critic_vf_gradient_sync_phases_per_step": 1,
            "actor_gradient_sync_phases_per_step": int(
                trains_joint_actor(args)
            ),
            "gradient_bucket_cap_mb": float(args.gradient_bucket_cap_mb),
            "per_step_buffer_broadcast": bool(synchronize_training_buffers),
            "rank_zero_writes_only": True,
        },
    }
    if repair_completed_resume:
        last_epoch_metrics = (
            history[-1].get("metrics", {})
            if history
            else {}
        )
        final = {
            **startup,
            "last_completed_epoch": int(start_epoch),
            "global_step": int(global_step),
            "global_samples_seen": int(global_samples_seen),
            "last_epoch_metrics": last_epoch_metrics,
            "history": history,
            "checkpoints": {
                "latest": str(args.output_dir / "latest.pt"),
                "last": str(args.output_dir / "last.pt"),
            },
        }
        if distributed.is_main_process:
            latest = args.output_dir / "latest.pt"
            replace_with_hardlink(args.resume_checkpoint, latest)
            replace_with_hardlink(latest, args.output_dir / "last.pt")
            if (
                int(args.snapshot_every_epochs) > 0
                and start_epoch % int(args.snapshot_every_epochs) == 0
            ):
                replace_with_hardlink(
                    latest,
                    args.output_dir / "models" / f"model_epoch_{start_epoch}.pt",
                )
            atomic_write_json(args.output_dir / "training_config.json", startup)
            atomic_write_json(args.output_dir / "partial_summary.json", final)
            atomic_write_json(args.output_dir / "summary.json", final)
            print(
                f"Repaired completed checkpoint links and summary at epoch={start_epoch}",
                flush=True,
            )
        if distributed.enabled:
            dist.barrier()
        return final if distributed.is_main_process else {}

    if distributed.is_main_process:
        atomic_write_json(args.output_dir / "training_config.json", startup)
        print(json.dumps(jsonable(startup), indent=2), flush=True)
        writer = make_tensorboard_writer(args.output_dir)
    else:
        writer = None
    max_grad = (
        None
        if args.max_gradient_norm is None or args.max_gradient_norm <= 0.0
        else float(args.max_gradient_norm)
    )

    shared_action_range_validated = False
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        if distributed.enabled and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        iterator = iter(loader)
        records: list[dict[str, float]] = []
        for step_in_epoch in range(1, int(args.steps_per_epoch) + 1):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            raw_batch = align_shared_batch_actions(
                raw_batch,
                validate=not shared_action_range_validated,
            )
            shared_action_range_validated = True
            batch = process_chunk_batch(
                raw_batch,
                actor_algo,
                obs_stats,
                chunk_horizon=args.chunk_horizon,
                discount=args.discount,
                reward_mode=args.reward_mode,
                critic_observation_horizon=(
                    args.critic_observation_horizon
                ),
                dynamics_prediction_offsets=(
                    args.dynamics_prediction_offsets
                ),
            )
            encoder_trainable = global_step >= int(
                args.resolved_encoder_freeze_steps
            )
            vf_encoder_trainable = (
                global_step >= int(args.resolved_vf_encoder_freeze_steps)
            )
            if is_wcm:
                wcm_system.train()
                set_wcm_encoder_trainable(wcm_system, encoder_trainable)
            else:
                critics.train()
                set_representation_trainable(critics, encoder_trainable)
                vf.train()
                set_vf_encoder_trainable(vf, vf_encoder_trainable)
            if synchronize_training_buffers:
                broadcast_module_buffers(
                    training_buffer_modules,
                    distributed,
                )
            ramp = min(
                1.0,
                float(global_step + 1)
                / max(float(args.resolved_dynamics_warmup_steps), 1.0),
            )
            effective_dynamics = float(args.dynamics_weight) * ramp
            if is_wcm:
                total_critic_loss, info = compute_wcm_chunk_losses(
                    wcm_system,
                    wcm_target_system,
                    wcm_dynamics_target_encoder,
                    batch,
                    discount=args.discount,
                    expectile=args.expectile,
                    use_huber=args.use_huber,
                    dynamics_weight=effective_dynamics,
                    sigreg_weight=args.sigreg_weight,
                    sigreg_knots=args.sigreg_knots,
                    sigreg_num_projections=args.sigreg_num_projections,
                    sigreg_global_batch=args.sigreg_global_batch,
                    global_step=global_step,
                    distributed_context=distributed,
                )
                update_wcm_system(
                    wcm_system,
                    wcm_target_system,
                    wcm_optimizer,
                    total_critic_loss,
                    target_tau=args.resolved_target_tau,
                    max_gradient_norm=max_grad,
                    gradient_sync_fn=wcm_gradient_sync_fn,
                )
            else:
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
                    distributed_context=distributed,
                )
                update_networks(
                    critics,
                    targets,
                    vf,
                    critic_optimizers,
                    vf_optimizer,
                    critic_losses,
                    vf_loss,
                    target_tau=args.resolved_target_tau,
                    max_gradient_norm=max_grad,
                    gradient_sync_fn=gradient_sync_fn,
                )

            # update condition diffusion actor
            actor_info: dict[str, Any] = {}
            if trains_joint_actor(args):
                actor_batch = raw_batch
                if args.conditioned_actor:
                    current_index = int(args.observation_horizon) - 1
                    condition_labels = source_condition_labels(
                        raw_batch,
                        current_index=current_index,
                    )
                    actor_batch = add_actor_condition(
                        actor_batch,
                        condition_labels,
                    )
                actor_row_count = int(raw_batch["actions"].shape[0])
                actor_info = {
                    "actor/data_rows": float(actor_row_count),
                    "actor/conditioned": float(args.conditioned_actor),
                }
                if args.conditioned_actor:
                    actor_info.update(
                        {
                            "actor/condition_mean": condition_labels.mean(),
                            "actor/zero_condition_fraction": (
                                (condition_labels < 0.5).float().mean()
                            ),
                        }
                    )
                actor_info.update(
                    actor_train_step(
                        actor_algo,
                        actor_batch,
                        epoch,
                        obs_stats,
                        defer_scalar_conversion=True,
                    )
                )
                del actor_batch
                if args.conditioned_actor:
                    del condition_labels
            if is_wcm:
                if wcm_lr_scheduler is not None:
                    wcm_lr_scheduler.step()
            else:
                for scheduler in critic_lr_schedulers:
                    if scheduler is not None:
                        scheduler.step()
                if vf_lr_scheduler is not None:
                    vf_lr_scheduler.step()
            global_samples_seen += int(
                raw_batch["actions"].shape[0] * distributed.world_size
            )
            actor_info["critic/data_rows"] = float(raw_batch["actions"].shape[0])
            dynamics_target_age_used_for_loss = float(
                global_step - dynamics_target_last_sync_step
            )
            global_step += 1
            dynamics_target_synced = False
            dynamics_sync_audit: dict[str, float | int] = {}
            if (
                global_step
                % int(args.resolved_dynamics_target_sync_interval)
                == 0
            ):
                if is_wcm:
                    dynamics_sync_audit = (
                        hard_sync_wcm_dynamics_target_encoder(
                            wcm_dynamics_target_encoder,
                            wcm_system,
                        )
                    )
                    broadcast_module_state(
                        [wcm_dynamics_target_encoder],
                        distributed,
                    )
                    dynamics_target_synced = True
                elif trains_joint_actor(args):
                    dynamics_sync_audit = (
                        sync_actor_dynamics_target_encoder(
                            dynamics_target_encoder,
                            actor_algo,
                        )
                    )
                    dynamics_target_synced = True
                if dynamics_target_synced:
                    dynamics_target_last_sync_step = global_step
            metrics = dict(info)
            metrics.update(actor_info)
            metrics["dynamics/target_synced_after_update"] = float(
                dynamics_target_synced
            )
            metrics["dynamics/target_last_sync_step"] = float(
                dynamics_target_last_sync_step
            )
            metrics["dynamics/target_age_used_for_loss"] = (
                dynamics_target_age_used_for_loss
            )
            metrics["dynamics/target_sync_age"] = float(
                global_step - dynamics_target_last_sync_step
            )
            metrics["dynamics/target_sync_count"] = float(
                dynamics_target_last_sync_step
                // int(args.resolved_dynamics_target_sync_interval)
            )
            metrics["dynamics/target_pre_sync_relative_l2"] = float(
                dynamics_sync_audit.get("pre_sync_relative_l2", 0.0)
            )
            metrics["encoder/trainable"] = float(encoder_trainable)
            metrics["representation/trainable"] = 1.0 if is_wcm else float(
                encoder_trainable
            )
            metrics["vf/trainable"] = 1.0
            metrics["vf/head_trainable"] = 1.0
            metrics["vf/encoder_trainable"] = float(encoder_trainable) if is_wcm else float(
                vf_encoder_trainable
            )
            if trains_joint_actor(args):
                for group in actor_algo.optimizers["policy"].param_groups:
                    group_name = str(group.get("group_name", "unknown"))
                    metrics[f"lr/actor_{group_name}"] = float(group["lr"])
            if is_wcm:
                for group in wcm_optimizer.param_groups:
                    metrics[f"lr/{group['group_name']}"] = float(group["lr"])
            else:
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
            records.append(metrics)
            should_log = (
                global_step % int(args.log_every) == 0
                or step_in_epoch == int(args.steps_per_epoch)
            )
            if writer is not None and should_log:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, global_step)
            if distributed.is_main_process and should_log:
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
            if is_wcm:
                del batch, total_critic_loss
            else:
                del batch, critic_losses, vf_loss

        epoch_summary = {
            "global_samples_seen": int(global_samples_seen),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": mean_metrics(records),
        }
        history.append(epoch_summary)
        partial = {
            **startup,
            "last_completed_epoch": int(epoch),
            "global_step": int(global_step),
            "global_samples_seen": int(global_samples_seen),
            "last_epoch_metrics": epoch_summary["metrics"],
            "history": history,
            "checkpoints": {
                "latest": str(args.output_dir / "latest.pt"),
                "last": str(args.output_dir / "last.pt"),
            },
        }
        if distributed.is_main_process:
            atomic_write_json(args.output_dir / "partial_summary.json", partial)

        if (
            epoch % int(args.save_every_epochs) == 0
            or epoch == int(args.epochs)
        ):
            rank_runtime_states = (
                gather_rank_runtime_states(loader_generator, distributed)
                if distributed.enabled
                else None
            )
            if distributed.is_main_process:
                payload = current_checkpoint_payload(
                    epoch,
                    rank_runtime_states,
                )
                latest = args.output_dir / LATEST_CHECKPOINT_NAME
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
        if distributed.enabled:
            dist.barrier()

    if writer is not None:
        writer.flush()
        writer.close()
    if distributed.is_main_process:
        final = json.loads((args.output_dir / "partial_summary.json").read_text())
        atomic_write_json(args.output_dir / "summary.json", final)
    else:
        final = {}
    if distributed.enabled:
        dist.barrier()
    return final


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("square", "can", "transport", "tool_hang", "pick_cup"),
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
    parser.add_argument("--validate-resume-only", action="store_true")
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
        help="Flat gradient all-reduce bucket size in MiB.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help="Local process rank supplied by torchrun; the environment wins.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--schedule-reference-batch-size",
        type=int,
        default=100,
        help=(
            "Reference global batch used to express sample-timed warmups, "
            "dynamics ramps, and encoder freezes."
        ),
    )
    parser.add_argument(
        "--sparse-chunk-loader",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        "--critic-architecture",
        choices=CRITIC_ARCHITECTURES,
        default=LEGACY_CRITIC_ARCHITECTURE,
        help=(
            "legacy preserves existing checkpoints; wcm_shared_temporal_v1 "
            "uses one causal state representation for twin Q, V, and dynamics"
        ),
    )
    parser.add_argument(
        "--critic-observation-horizon",
        type=int,
        default=1,
        help=(
            "Number of most recent actor observation frames consumed by Q "
            "and V. Use 2 for the full RGB-DP history."
        ),
    )
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
        default=None,
        help=(
            "Optimizer-update interval for hard-copying the WCM frame-latent teacher "
            "(or the legacy joint actor dynamics target encoder); defaults to "
            "500 for WCM and 1000 for the legacy architecture."
        ),
    )
    parser.add_argument("--actor-adapter-lr", type=float, default=1e-4)
    parser.add_argument("--actor-unet-lr", type=float, default=1e-4)
    parser.add_argument("--actor-obs-encoder-lr", type=float, default=1e-4)
    parser.add_argument("--actor-reference-weight", type=float, default=0.0)
    parser.add_argument(
        "--actor-reference-batch-fraction",
        type=float,
        default=0.25,
    )
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
            "Train the joint diffusion actor with explicit binary conditions "
            "from the dataset's actor_condition key."
        ),
    )
    parser.add_argument(
        "--actor-condition-mode",
        choices=tuple(ACTOR_CONDITION_DEFINITIONS),
        default="human_only",
        help=(
            "human_only uses human=1 and all rollouts=0; human_success uses "
            "human and successful rollouts=1 and failure rollouts=0"
        ),
    )
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--condition-hidden-dim", type=int, default=256)
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
    parser.add_argument("--temporal-num-layers", type=int, default=2)
    parser.add_argument(
        "--temporal-num-heads",
        type=int,
        default=6,
        help="Attention heads in the causal state trunk (separate from action heads).",
    )
    parser.add_argument("--temporal-feedforward-dim", type=int, default=600)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
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
    parser.add_argument(
        "--critic-q-use-predicted-next-latent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Concatenate the layer-normalized dynamics-predicted next "
            "actor-encoder feature into the Q head."
        ),
    )
    parser.add_argument("--dynamics-weight", type=float, default=0.05)
    parser.add_argument("--dynamics-cosine-weight", type=float, default=0.05)
    parser.add_argument(
        "--dynamics-prediction-offsets",
        type=int,
        nargs="+",
        default=(2, 4, 6, 8),
        help="WCM future-latent offsets measured in executed chunk actions.",
    )
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-projections", type=int, default=1024)
    parser.add_argument(
        "--sigreg-global-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Differentiably gather temporal contexts across ranks for SIGReg.",
    )
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
    try:
        configure_critic_architecture_args(args)
    except ValueError as error:
        parser.error(str(error))
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
    if args.epochs is None or args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        parser.error("steps-per-epoch must be positive when specified")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if args.num_workers > 0 and args.prefetch_factor <= 0:
        parser.error("prefetch-factor must be positive when workers are enabled")
    if args.chunk_horizon <= 0:
        parser.error("chunk-horizon must be positive")
    if args.critic_observation_horizon <= 0:
        parser.error("critic-observation-horizon must be positive")
    if not 0.0 <= args.discount <= 1.0:
        parser.error("discount must be in [0, 1]")
    if not 0.5 <= args.expectile < 1.0:
        parser.error("expectile must be in [0.5, 1)")
    if not 0.0 < args.target_tau <= 1.0:
        parser.error("target-tau must be in (0, 1]")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    if args.condition_hidden_dim <= 0:
        parser.error("condition-hidden-dim must be positive")
    if args.latent_dim <= 0 or args.action_hidden_dim <= 0:
        parser.error("latent and action hidden dimensions must be positive")
    if args.num_attention_heads <= 0:
        parser.error("num-attention-heads must be positive")
    if args.action_hidden_dim % args.num_attention_heads != 0:
        parser.error("action-hidden-dim must be divisible by num-attention-heads")
    if args.num_action_conv_layers < 0:
        parser.error("num-action-conv-layers must be non-negative")
    if any(int(value) <= 0 for value in args.critic_hidden_dims):
        parser.error("critic-hidden-dims must all be positive")
    for name in (
        "actor_lr_warmup_steps",
        "critic_vf_lr_warmup_steps",
        "dynamics_warmup_steps",
        "encoder_freeze_steps",
        "vf_encoder_freeze_steps",
    ):
        if int(getattr(args, name)) < 0:
            parser.error(f"{name.replace('_', '-')} must be non-negative")
    for name in ("dynamics_weight", "dynamics_cosine_weight"):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"{name.replace('_', '-')} must be non-negative")
    if args.log_every <= 0 or args.save_every_epochs <= 0:
        parser.error("log-every and save-every-epochs must be positive")
    if args.snapshot_every_epochs < 0:
        parser.error("snapshot-every-epochs must be non-negative")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.schedule_reference_batch_size <= 0:
        parser.error("schedule-reference-batch-size must be positive")
    if args.gradient_bucket_cap_mb <= 0.0:
        parser.error("gradient-bucket-cap-mb must be positive")
    if args.num_critics < 2:
        parser.error("RISE clipped double Q requires at least two critics")
    if args.dynamics_target_sync_interval <= 0:
        parser.error("dynamics-target-sync-interval must be positive")
    for name in (
        "actor_adapter_lr",
        "actor_unet_lr",
        "actor_obs_encoder_lr",
        "critic_lr",
        "encoder_lr",
        "vf_lr",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if float(args.actor_reference_weight) < 0.0:
        parser.error("actor-reference-weight must be non-negative")
    if not 0.0 < args.actor_reference_batch_fraction <= 1.0:
        parser.error("actor-reference-batch-fraction must be in (0, 1]")
    if not 0.0 <= args.condition_dropout < 1.0:
        parser.error("condition-dropout must be in [0, 1)")
    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    if not args.critic_late_fusion_key:
        args.critic_late_fusion_key = None
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
