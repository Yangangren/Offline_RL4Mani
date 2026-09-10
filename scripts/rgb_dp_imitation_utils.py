#!/usr/bin/env python3
"""Actor-only data and optimization helpers for RGB-DP imitation training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import MetaDataset, SparseOneStepSequenceDataset


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2))


def initialize_actor_from_deployed_ema(actor_algo) -> bool:
    if actor_algo.ema is None:
        return False
    ema_state = copy.deepcopy(actor_algo.ema.averaged_model.state_dict())
    actor_algo.nets.load_state_dict(ema_state)
    actor_algo.ema.averaged_model.load_state_dict(actor_algo.nets.state_dict())
    if hasattr(actor_algo, "_refresh_ema_parameter_views"):
        actor_algo._refresh_ema_parameter_views()
    return True


def apply_pretrained_action_normalization(
    actor_dataset,
    checkpoint_dict: dict,
) -> str:
    action_stats = checkpoint_dict.get("action_normalization_stats")
    if action_stats is None:
        raise ValueError(
            "pretrained DP checkpoint is missing action_normalization_stats; "
            "refusing to normalize actor data with mixed-dataset statistics"
        )
    if not hasattr(actor_dataset, "set_action_normalization_stats"):
        raise TypeError("actor dataset does not support set_action_normalization_stats")
    actor_dataset.set_action_normalization_stats(copy.deepcopy(action_stats))
    source = "pretrained_dp_checkpoint_action_normalization_stats"
    setattr(actor_dataset, "actor_action_normalization_source", source)
    return source


def action_normalization_stats_match(left: dict, right: dict) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for action_key in left:
        if set(left[action_key].keys()) != set(right[action_key].keys()):
            return False
        for stat_key in left[action_key]:
            if not np.allclose(
                np.asarray(left[action_key][stat_key]),
                np.asarray(right[action_key][stat_key]),
            ):
                return False
    return True


def configure_actor_optimizer(
    actor_algo,
    lr: float | None,
    disable_lr_scheduler: bool,
    *,
    num_train_batches: int | None = None,
    num_epochs: int | None = None,
    reset_scheduler: bool = False,
    preserve_current_lr: bool = False,
) -> None:
    optim_params = actor_algo.optim_params["policy"]
    if num_train_batches is not None:
        optim_params.num_train_batches = int(num_train_batches)
    if num_epochs is not None:
        optim_params.num_epochs = int(num_epochs)
    configured_lr = float(
        optim_params.learning_rate.initial if lr is None else lr
    )
    if lr is not None:
        optim_params.learning_rate.initial = configured_lr

    optimizer = actor_algo.optimizers["policy"]
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing = [
        parameter
        for parameter in actor_algo.nets["policy"].parameters()
        if parameter.requires_grad and id(parameter) not in optimized_ids
    ]
    if missing:
        optimizer.add_param_group({"params": missing, "lr": configured_lr})
    if not preserve_current_lr:
        for group in optimizer.param_groups:
            group["lr"] = configured_lr
            group["initial_lr"] = configured_lr
    if disable_lr_scheduler:
        actor_algo.lr_schedulers["policy"] = None
        actor_algo.step_lr_schedulers_every_batch["policy"] = False
    elif reset_scheduler:
        actor_algo.lr_schedulers["policy"] = TorchUtils.lr_scheduler_from_optim_params(
            net_optim_params=optim_params,
            net=actor_algo.nets["policy"],
            optimizer=optimizer,
        )
        actor_algo.step_lr_schedulers_every_batch["policy"] = bool(
            optim_params.learning_rate.get("step_every_batch", False)
        )


def set_actor_obs_encoder_trainable(actor_algo, trainable: bool) -> None:
    """Freeze only the online actor encoder; the diffusion U-Net keeps training."""
    actor_algo.nets["policy"]["obs_encoder"].requires_grad_(bool(trainable))


def actor_trainability_summary(actor_algo) -> dict[str, Any]:
    policy = actor_algo.nets["policy"]
    obs_encoder = policy["obs_encoder"]
    optimized_ids = {
        id(parameter)
        for group in actor_algo.optimizers["policy"].param_groups
        for parameter in group["params"]
    }

    def module_summary(module: nn.Module) -> dict[str, Any]:
        parameters = list(module.named_parameters())
        total = sum(parameter.numel() for _, parameter in parameters)
        trainable = sum(
            parameter.numel()
            for _, parameter in parameters
            if parameter.requires_grad
        )
        optimized = sum(
            parameter.numel()
            for _, parameter in parameters
            if id(parameter) in optimized_ids
        )
        frozen_names = [
            name for name, parameter in parameters if not parameter.requires_grad
        ]
        missing_names = [
            name for name, parameter in parameters if id(parameter) not in optimized_ids
        ]
        return {
            "num_parameters": int(total),
            "num_trainable_parameters": int(trainable),
            "num_optimizer_parameters": int(optimized),
            "all_trainable": not frozen_names,
            "all_in_optimizer": not missing_names,
            "num_frozen_tensors": len(frozen_names),
            "num_missing_optimizer_tensors": len(missing_names),
            "first_frozen_tensors": frozen_names[:10],
            "first_missing_optimizer_tensors": missing_names[:10],
        }

    summary = {
        "scope": "full_pretrained_dp_policy",
        "policy": module_summary(policy),
        "obs_encoder": module_summary(obs_encoder),
    }
    if not summary["policy"]["all_trainable"] or not summary["policy"]["all_in_optimizer"]:
        raise RuntimeError(
            "DP actor is not fully trainable. "
            f"trainability={json.dumps(jsonable(summary), indent=2)}"
        )
    if not summary["obs_encoder"]["all_trainable"] or not summary["obs_encoder"]["all_in_optimizer"]:
        raise RuntimeError(
            "DP actor obs encoder is not fully trainable. "
            f"trainability={json.dumps(jsonable(summary), indent=2)}"
        )
    return summary


def actor_source_entries(args: argparse.Namespace) -> list[dict]:
    conditioned = bool(getattr(args, "conditioned_mixed_imitation", False))
    condition_label_mode = str(getattr(args, "condition_label_mode", "outcome"))
    human_only_condition = conditioned and condition_label_mode == "human_only"
    entries = []
    if float(args.actor_demo_weight) > 0.0:
        demo = {
            "path": str(args.demo_dataset),
            "weight": float(args.actor_demo_weight),
            "anti_failure": 0.0,
        }
        if args.demo_filter_key:
            demo["filter_key"] = args.demo_filter_key
        entries.append(demo)
    if float(args.actor_success_weight) > 0.0:
        success = {
            "path": str(args.success_dataset),
            "filter_key": args.success_filter_key,
            "weight": float(args.actor_success_weight),
            "anti_failure": 1.0 if human_only_condition else 0.0,
        }
        if bool(getattr(args, "actor_one_step_rollouts", False)):
            success["one_step_aligned"] = True
        entries.append(success)
    if float(args.actor_failure_weight) > 0.0:
        failure = {
            "path": str(args.failure_dataset),
            "weight": float(args.actor_failure_weight),
            "anti_failure": float(
                getattr(args, "actor_failure_anti_failure_label", 1.0)
            ),
        }
        if args.failure_filter_key:
            failure["filter_key"] = args.failure_filter_key
        if bool(getattr(args, "actor_failure_demo_start_only", False)):
            failure["demo_start_only"] = True
        sample_start_offset = int(
            getattr(args, "actor_failure_sample_start_offset", 0)
        )
        if sample_start_offset != 0:
            failure["sample_start_offset"] = sample_start_offset
        if bool(getattr(args, "actor_one_step_rollouts", False)):
            failure["one_step_aligned"] = True
        entries.append(failure)
    if not entries:
        raise ValueError("at least one actor dataset weight must be positive")
    return entries


class ActorOneStepSourceSequenceDataset(SparseOneStepSequenceDataset):
    """Use one-step IDQL rows from a standalone rollout source HDF5."""

    def _load_valid_indices(self):
        base = self.dataset
        masks = {}
        with base.hdf5_file_opened():
            for demo_id in base.demos:
                episode = base.hdf5_file[f"data/{demo_id}"]
                expected = int(base._demo_id_to_demo_length[demo_id])
                if not bool(episode.attrs.get("one_step_aligned", 0)):
                    raise ValueError(
                        f"data/{demo_id} is not marked one_step_aligned"
                    )
                if "one_step_critic_valid" not in episode:
                    raise ValueError(
                        f"data/{demo_id} is missing one_step_critic_valid"
                    )
                values = np.asarray(
                    episode["one_step_critic_valid"][:], dtype=np.uint8
                ).reshape(-1)
                if values.shape != (expected,) or np.any(
                    ~np.isin(values, (0, 1))
                ):
                    raise ValueError(
                        f"data/{demo_id}/one_step_critic_valid must be a "
                        f"binary vector of length {expected}"
                    )
                aligned_group = episode.get("one_step_obs")
                if aligned_group is None:
                    raise ValueError(
                        f"data/{demo_id} is missing one_step_obs"
                    )
                for obs_key in base.obs_keys:
                    if obs_key not in aligned_group:
                        raise ValueError(
                            f"data/{demo_id}/one_step_obs is missing {obs_key}"
                        )
                    aligned = aligned_group[obs_key]
                    if aligned.shape[:2] != (
                        expected,
                        self.observation_horizon,
                    ):
                        raise ValueError(
                            f"{aligned.name} must have leading shape "
                            f"({expected}, {self.observation_horizon})"
                        )
                masks[demo_id] = values
                self._demo_request_aligned[demo_id] = False
                self._demo_one_step_aligned[demo_id] = True

        admitted = []
        for base_index in range(len(base)):
            demo_id = base._index_to_demo_id[base_index]
            demo_start = base._demo_id_to_start_indices[demo_id]
            demo_offset = 0 if base.pad_frame_stack else base.n_frame_stack - 1
            row = (
                base_index
                - demo_start
                + demo_offset
                + (base.sample_start_offset if base.demo_start_only else 0)
            )
            if int(masks[demo_id][row]) == 1:
                admitted.append(base_index)
        return admitted


class ActorOnlySequenceDataset(torch.utils.data.Dataset):
    """Give every actor source the same two-frame, actor-only batch schema."""

    def __init__(self, dataset, observation_horizon: int):
        super().__init__()
        self.dataset = dataset
        self.observation_horizon = int(observation_horizon)
        self.output_keys = frozenset(
            set(dataset.dataset_keys)
            | {
                "obs",
                "actions",
                "goal_obs",
                "index",
                "anti_failure",
                "hazard_failure",
            }
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        actor_sample = {
            key: value
            for key, value in sample.items()
            if key in self.output_keys
        }
        if "obs" in actor_sample:
            actor_sample["obs"] = {
                key: value[: self.observation_horizon]
                for key, value in actor_sample["obs"].items()
            }
        return actor_sample

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class ActorOneStepRolloutDataset(ActorOnlySequenceDataset):
    """Expose one-step IDQL rollout rows with the actor-only batch schema."""

    def __init__(self, dataset, observation_horizon: int):
        sparse_dataset = ActorOneStepSourceSequenceDataset(
            dataset,
            observation_horizon=int(observation_horizon),
            critic_observation_horizon=1,
        )
        super().__init__(sparse_dataset, observation_horizon)
        self.sparse_dataset = sparse_dataset


def wrap_actor_sources_for_one_step(
    dataset,
    *,
    enabled: bool,
    observation_horizon: int,
    source_entries: list[dict],
    normalize_weights_by_ds_size: bool,
):
    """Apply the one-step IDQL row contract to marked rollout sources."""
    if not enabled:
        return dataset
    base_datasets = (
        list(dataset.datasets)
        if isinstance(dataset, MetaDataset)
        else [dataset]
    )
    if len(base_datasets) != len(source_entries):
        raise RuntimeError(
            "actor source count differs from the constructed dataset: "
            f"sources={len(source_entries)} datasets={len(base_datasets)}"
        )
    wrapped = []
    for source_dataset, entry in zip(base_datasets, source_entries):
        if bool(entry.get("one_step_aligned", False)):
            wrapped.append(
                ActorOneStepRolloutDataset(
                    source_dataset,
                    observation_horizon,
                )
            )
        else:
            # Regular SequenceDataset rows contain the same two actor frames
            # followed by future sequence observations. Trim only that unused
            # tail so they collate with the materialized one-step rollout pair.
            wrapped.append(
                ActorOnlySequenceDataset(
                    source_dataset,
                    observation_horizon,
                )
            )
    if len(wrapped) == 1:
        return wrapped[0]
    return MetaDataset(
        datasets=wrapped,
        ds_weights=[float(entry.get("weight", 1.0)) for entry in source_entries],
        normalize_weights_by_ds_size=bool(normalize_weights_by_ds_size),
    )


def build_actor_loader(
    *,
    args: argparse.Namespace,
    actor_algo,
    checkpoint_dict: dict,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, Any]:
    config, _ = FileUtils.config_from_checkpoint(
        ckpt_dict=checkpoint_dict,
        verbose=False,
    )
    ObsUtils.initialize_obs_utils_with_config(config)
    prediction_horizon = int(actor_algo.algo_config.horizon.prediction_horizon)
    observation_horizon = int(actor_algo.algo_config.horizon.observation_horizon)
    seq_length = int(args.actor_seq_length or prediction_horizon)
    if seq_length < prediction_horizon:
        raise ValueError(
            f"actor_seq_length={seq_length} must be >= "
            f"DP prediction_horizon={prediction_horizon}"
        )

    source_entries = actor_source_entries(args)
    if args.actor_hdf5_cache_mode == "all" and len(source_entries) > 1:
        raise ValueError(
            "actor_hdf5_cache_mode=all is unsupported for multi-source "
            "MetaDataset training; use low_dim or no cache"
        )
    if args.actor_hdf5_cache_mode == "all" and bool(args.actor_one_step_rollouts):
        raise ValueError(
            "actor_hdf5_cache_mode=all is unsupported for one-step rollout "
            "loading; use low_dim or no cache"
        )

    with config.values_unlocked():
        config.train.data = source_entries
        config.train.normalize_weights_by_ds_size = bool(
            args.actor_normalize_weights_by_ds_size
            and not bool(args.actor_uniform_sample_pool)
        )
        config.train.hdf5_cache_mode = args.actor_hdf5_cache_mode
        config.train.hdf5_load_next_obs = False
        config.train.seq_length = seq_length
        config.train.frame_stack = observation_horizon
        config.train.pad_seq_length = True
        config.train.pad_frame_stack = True
        if bool(args.actor_one_step_rollouts):
            config.train.dataset_keys = list(
                dict.fromkeys(list(config.train.dataset_keys) + ["actions", "dones"])
            )
        config.train.batch_size = int(args.actor_batch_size)
        config.train.num_data_workers = int(args.actor_num_workers)

    dataset = TrainUtils.dataset_factory(
        config,
        obs_keys=list(actor_algo.obs_shapes.keys()),
    )
    dataset = wrap_actor_sources_for_one_step(
        dataset,
        enabled=bool(args.actor_one_step_rollouts),
        observation_horizon=observation_horizon,
        source_entries=source_entries,
        normalize_weights_by_ds_size=bool(config.train.normalize_weights_by_ds_size),
    )
    if args.actor_hdf5_cache_mode == "all":
        cached_action_stats = dataset.get_action_normalization_stats()
        checkpoint_action_stats = checkpoint_dict.get("action_normalization_stats")
        if checkpoint_action_stats is None or not action_normalization_stats_match(
            cached_action_stats,
            checkpoint_action_stats,
        ):
            raise ValueError(
                "actor_hdf5_cache_mode=all requires dataset action normalization "
                "to match the pretrained DP checkpoint"
            )
    action_normalization_source = apply_pretrained_action_normalization(
        dataset,
        checkpoint_dict,
    )
    distributed_world_size = int(getattr(args, "distributed_world_size", 1))
    distributed_rank = int(getattr(args, "distributed_rank", 0))
    if distributed_rank == 0:
        print(
            json.dumps(
                {"actor_action_normalization_source": action_normalization_source}
            ),
            flush=True,
        )

    sampler = None
    if not bool(args.actor_uniform_sample_pool):
        sampler = (
            dataset.get_dataset_sampler()
            if hasattr(dataset, "get_dataset_sampler")
            else None
        )
    if distributed_world_size > 1:
        if not bool(args.actor_uniform_sample_pool):
            raise ValueError(
                "distributed actor training requires --actor-uniform-sample-pool; "
                "weighted multi-source sampling cannot be sharded without "
                "changing its source distribution"
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )

    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + distributed_rank)
    loader_kwargs: dict[str, Any] = {}
    if int(args.actor_num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    available_rows = len(sampler) if distributed_world_size > 1 else len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.actor_batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=available_rows >= int(args.actor_batch_size),
        num_workers=int(args.actor_num_workers),
        pin_memory=bool(args.pin_memory and actor_algo.device.type == "cuda"),
        generator=generator,
        **loader_kwargs,
    )
    return dataset, loader, config


def actor_train_step(
    actor_policy,
    raw_batch: dict,
    step: int,
    obs_normalization_stats,
    *,
    materialize_log: bool = True,
) -> dict[str, float]:
    actor_algo = actor_policy.policy
    actor_algo.set_train()
    batch = actor_algo.process_batch_for_training(raw_batch)
    batch = actor_algo.postprocess_batch_for_training(
        batch,
        obs_normalization_stats=obs_normalization_stats,
    )
    info = actor_algo.train_on_batch(batch, epoch=step, validate=False)
    actor_algo.on_gradient_step()
    if not materialize_log:
        return {}
    return {key: float(value) for key, value in actor_algo.log_info(info).items()}


__all__ = [
    "ActorOnlySequenceDataset",
    "ActorOneStepRolloutDataset",
    "ActorOneStepSourceSequenceDataset",
    "actor_source_entries",
    "actor_train_step",
    "actor_trainability_summary",
    "build_actor_loader",
    "configure_actor_optimizer",
    "initialize_actor_from_deployed_ema",
    "set_actor_obs_encoder_trainable",
    "wrap_actor_sources_for_one_step",
    "jsonable",
    "write_json",
]
