#!/usr/bin/env python3
"""Label chunk-actor rows from a frozen learned Q / V checkpoint.

The emitted sidecar is indexed by the underlying ``SequenceDataset`` index.
Human demonstrations are always condition one. Rollout chunks above and below
two calibrated score thresholds receive active one and zero conditions,
respectively; the uncertain middle band receives the explicit null condition
``(condition=0, mask=0)``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils

from eval_rgb_dp_idql import validate_rise_dp_composition
from rgb_dp_critic_conditions import (
    calibrate_critic_condition_thresholds,
    construct_critic_conditions,
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
from train_rgb_dp_idql import atomic_torch_save
from train_rgb_dp_idql import dataset_audit


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
    episode_sources = _episode_sources(path)
    for raw_batch in loader:
        indices = torch.as_tensor(raw_batch["index"]).long().reshape(-1)
        index_np = indices.cpu().numpy()
        if index_np.size and (
            int(index_np.min()) < 0 or int(index_np.max()) >= count
        ):
            raise IndexError("loader returned a dataset index outside the sidecar")
        if np.any(fields["scored"][index_np] != 0):
            raise RuntimeError("critic condition loader returned duplicate indices")
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
        for offset, index in enumerate(index_np):
            demo = str(base._index_to_demo_id[int(index)])
            source = episode_sources[demo]
            fields["source_is_expert"][index] = int(source == "expert")
            fields["rollout_success"][index] = int(
                source == "non_expert_success"
            )
        fields["scored"][index_np] = 1

    scored = fields["scored"].astype(bool)
    if not np.any(scored):
        raise ValueError(f"dataset {path} produced no scoreable chunk rows")
    for name in ("q_min", "value", "advantage", "twin_gap"):
        if not np.isfinite(fields[name][scored]).all():
            raise FloatingPointError(f"scored {name} contains non-finite values")
    return {
        "path": path,
        "identity": identity,
        "num_samples": count,
        "fields": fields,
        "scoreable_samples": int(scored.sum()),
        "filtered_samples": int(count - scored.sum()),
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
    score_key = "advantage" if args.condition_mode == "critic_advantage" else "q_min"
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
    score_key = "advantage" if args.condition_mode == "critic_advantage" else "q_min"
    conditions = construct_critic_conditions(
        source_fields[score_key][valid],
        source_fields["source_is_expert"][valid],
        mode=str(args.condition_mode),
        low_threshold=float(thresholds["low_threshold"]),
        high_threshold=float(thresholds["high_threshold"]),
    )
    actor_condition = np.zeros(scored["num_samples"], dtype=np.uint8)
    actor_condition_mask = np.zeros(scored["num_samples"], dtype=np.uint8)
    actor_condition[valid] = conditions["actor_condition"]
    actor_condition_mask[valid] = conditions["actor_condition_mask"]
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
            "condition_mode": str(args.condition_mode),
            "score_key": score_key,
            "threshold_mode": str(args.threshold_mode),
            "low_threshold": float(thresholds["low_threshold"]),
            "high_threshold": float(thresholds["high_threshold"]),
            "low_quantile": (
                float(args.low_quantile)
                if args.threshold_mode == "quantile"
                else None
            ),
            "high_quantile": (
                float(args.high_quantile)
                if args.threshold_mode == "quantile"
                else None
            ),
            "target_purity": (
                float(args.target_purity)
                if args.threshold_mode == "outcome_purity"
                else None
            ),
            "min_tail_count": (
                int(args.min_tail_count)
                if args.threshold_mode == "outcome_purity"
                else None
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
        "--critic-source", choices=("online", "target"), default="target"
    )
    parser.add_argument(
        "--condition-mode",
        choices=("critic_advantage", "critic_q"),
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
