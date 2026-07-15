"""Utilities for self-contained RGB critic encoders.

The critic checkpoint stores the observation-encoder architecture metadata and
the encoder parameters in its own model state. Actor checkpoints are consulted
only once, when initializing a newly trained critic.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import torch


def _resolve_checkpoint_path(path_like, root: Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    if root is None:
        root = Path.cwd()
    return (root / path).resolve()


def _observation_horizon(config) -> int:
    try:
        return int(config.algo.horizon.observation_horizon)
    except (AttributeError, KeyError, TypeError):
        return 1


def _build_encoder_from_metadata(encoder_checkpoint: dict, device):
    import robomimic.models.obs_nets as ObsNets
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.algo.diffusion_policy import replace_bn_with_gn

    config, _ = FileUtils.config_from_checkpoint(
        ckpt_dict=encoder_checkpoint,
        verbose=False,
    )
    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = encoder_checkpoint["shape_metadata"]
    if isinstance(shape_meta, list):
        shape_meta = shape_meta[0]
    policy_obs_keys = {
        key
        for modality in config.observation.modalities.obs.values()
        for key in modality
    }
    observation_shapes = OrderedDict(
        (key, tuple(shape))
        for key, shape in shape_meta["all_shapes"].items()
        if key in policy_obs_keys
    )
    encoder = ObsNets.ObservationGroupEncoder(
        observation_group_shapes=OrderedDict(obs=observation_shapes),
        encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
            config.observation.encoder
        ),
    )
    encoder = replace_bn_with_gn(encoder).float().to(device)
    return encoder, observation_shapes, _observation_horizon(config)


def load_rgb_encoder_from_actor_checkpoint(
    actor_checkpoint_path,
    device,
    *,
    root: Path | None = None,
):
    """Initialize an RGB encoder from a standard or hybrid DP actor.

    Returns the encoder and a self-contained architecture specification. The
    specification is sufficient to reconstruct an empty encoder at inference;
    no actor checkpoint is needed after the critic checkpoint has been saved.
    """
    import robomimic.utils.file_utils as FileUtils

    actor_path = _resolve_checkpoint_path(actor_checkpoint_path, root=root)
    checkpoint = FileUtils.load_dict_from_checkpoint(str(actor_path))
    encoder_checkpoint = checkpoint
    source_kind = "robomimic_policy"
    metadata_path = actor_path
    if bool(checkpoint.get("hybrid_dp_chunk_actor_iql", False)):
        source_kind = "hybrid_dp_chunk_actor_iql"
        base_value = checkpoint.get(
            "pretrained_dp_checkpoint",
            checkpoint.get("args", {}).get("checkpoint"),
        )
        if base_value is None:
            raise RuntimeError(
                "hybrid DP actor checkpoint is missing pretrained_dp_checkpoint"
            )
        metadata_path = _resolve_checkpoint_path(base_value, root=root)
        encoder_checkpoint = FileUtils.load_dict_from_checkpoint(str(metadata_path))
        actor_model = checkpoint.get("actor_model", {})
        policy_state = actor_model.get("ema", None) or actor_model.get("nets", None)
        if policy_state is None:
            raise RuntimeError("hybrid checkpoint contains no actor EMA/nets weights")
    else:
        model_state = checkpoint["model"]
        policy_state = model_state.get("ema", None) or model_state.get("nets", None)
        if policy_state is None:
            raise RuntimeError("DP checkpoint contains no EMA/nets weights")

    encoder, observation_shapes, observation_horizon = _build_encoder_from_metadata(
        encoder_checkpoint,
        device,
    )
    prefix = "policy.obs_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in policy_state.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError("actor checkpoint contains no observation-encoder weights")
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()

    spec = {
        "algo_name": copy.deepcopy(encoder_checkpoint["algo_name"]),
        "config": copy.deepcopy(encoder_checkpoint["config"]),
        "shape_metadata": copy.deepcopy(encoder_checkpoint["shape_metadata"]),
        "observation_shapes": OrderedDict(observation_shapes),
        "observation_horizon": int(observation_horizon),
        "initialization_actor_checkpoint": str(actor_path),
        "metadata_checkpoint": str(metadata_path),
        "source_kind": source_kind,
    }
    return encoder, spec


def build_rgb_encoder_from_critic_spec(spec: dict, device):
    """Reconstruct a critic-owned encoder without loading an actor."""
    if not isinstance(spec, dict):
        raise TypeError("rgb_encoder_spec must be a dictionary")
    mini_checkpoint = {
        "algo_name": copy.deepcopy(spec["algo_name"]),
        "config": copy.deepcopy(spec["config"]),
        "shape_metadata": copy.deepcopy(spec["shape_metadata"]),
    }
    encoder, observation_shapes, observation_horizon = _build_encoder_from_metadata(
        mini_checkpoint,
        device,
    )
    recorded_shapes = OrderedDict(
        (key, tuple(shape))
        for key, shape in spec["observation_shapes"].items()
    )
    if observation_shapes != recorded_shapes:
        raise ValueError(
            "critic encoder metadata reconstructs different observation shapes: "
            f"metadata={observation_shapes}, recorded={recorded_shapes}"
        )
    if int(observation_horizon) != int(spec["observation_horizon"]):
        raise ValueError("critic observation horizon metadata is inconsistent")
    return encoder, recorded_shapes, int(observation_horizon)


def prepare_observation_for_rgb_critic(
    observation: dict,
    observation_shapes: dict,
    device,
) -> dict[str, torch.Tensor]:
    """Apply robomimic observation preprocessing without using a policy."""
    import robomimic.utils.obs_utils as ObsUtils

    prepared = {}
    for key, shape in observation_shapes.items():
        if key not in observation:
            raise KeyError(f"raw observation is missing critic key {key}")
        value = torch.as_tensor(observation[key], device=device).float()
        # Raw RGB is HWC while its configured network shape is CHW, but both
        # have the same rank. Rank is therefore the reliable batch test here.
        if value.ndim == len(shape):
            value = value.unsqueeze(0)
        elif value.ndim != len(shape) + 1:
            raise ValueError(
                f"raw critic observation {key} has shape={tuple(value.shape)}; "
                f"expected rank {len(shape)} (unbatched) or {len(shape) + 1}"
            )
        if ObsUtils.key_is_obs_modality(key=key, obs_modality="rgb") or (
            ObsUtils.key_is_obs_modality(key=key, obs_modality="depth")
        ):
            value = ObsUtils.process_obs(value, obs_key=key)
        if tuple(value.shape[1:]) != tuple(shape):
            raise ValueError(
                f"processed critic observation {key} has shape={tuple(value.shape)}; "
                f"expected=(B, {tuple(shape)})"
            )
        prepared[key] = value
    return prepared
