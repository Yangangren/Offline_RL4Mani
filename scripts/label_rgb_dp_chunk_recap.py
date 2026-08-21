#!/usr/bin/env python3
"""Label mixed RGB-DP chunks with frozen MC-value RECAP advantages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import robomimic.utils.file_utils as FileUtils

from rgb_dp_chunk_recap import masked_chunk_advantage, make_recap_conditions
from train_rgb_dp_chunk_idql import (
    file_stat_identity,
    make_wcm_system_from_checkpoint,
    match_encoder_normalization_to_checkpoint,
    mixed_dataset_identity,
    process_chunk_batch,
)
from train_rgb_dp_chunk_recap import (
    LABEL_FORMAT,
    TARGET_FORMAT,
    VALUE_FORMAT,
    _apply_canonical_chunk_fields,
    _resolve_device,
    _seed_everything,
    build_recap_dataset,
    load_sidecar,
    make_loader,
    sidecar_rows,
    validate_sidecar_dataset,
)
from train_rgb_dp_idql import atomic_torch_save


@torch.no_grad()
def label_chunks(args: argparse.Namespace) -> dict:
    args.dataset = args.dataset.expanduser().resolve()
    args.targets = args.targets.expanduser().resolve()
    args.value_checkpoint = args.value_checkpoint.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for path in (args.dataset, args.targets, args.value_checkpoint, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    targets = load_sidecar(args.targets, expected_kind=TARGET_FORMAT)
    validate_sidecar_dataset(targets, args.dataset, sidecar_name="target sidecar")
    value_checkpoint = torch.load(
        args.value_checkpoint, map_location="cpu", weights_only=False
    )
    if value_checkpoint.get("kind") != VALUE_FORMAT:
        raise ValueError("value checkpoint is not a RECAP MC-value checkpoint")
    if not bool(value_checkpoint.get("rgb_dp_chunk_recap_value")):
        raise ValueError("value checkpoint lacks the RECAP value marker")
    if bool(value_checkpoint.get("q_trained", True)):
        raise ValueError("RECAP labeling requires a value-only checkpoint")
    if value_checkpoint.get("dataset_identity") != mixed_dataset_identity(args.dataset):
        raise ValueError("value checkpoint dataset provenance does not match")
    if value_checkpoint.get("targets_identity") != file_stat_identity(args.targets):
        raise ValueError("value checkpoint target provenance does not match")
    if value_checkpoint.get("pretrained_dp_identity") != file_stat_identity(args.checkpoint):
        raise ValueError("value checkpoint pretrained-DP provenance does not match")

    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    actor_policy, _dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    actor_algo = actor_policy.policy
    actor_algo.set_eval()
    actor_algo.nets.requires_grad_(False)
    args.chunk_horizon = int(targets["config"]["chunk_horizon"])
    args.observation_horizon = int(value_checkpoint["critic_observation_horizon"])
    dataset, _generator, _config = build_recap_dataset(
        args,
        actor_policy,
        _dp_checkpoint,
        sparse_chunk_loader=bool(args.sparse_chunk_loader),
        dynamics_prediction_offsets=(),
        sequence_length=int(args.chunk_horizon),
    )
    if len(dataset) != int(targets["num_samples"]):
        raise ValueError("dataset length differs from prepared target sidecar")
    loader = make_loader(dataset, args, shuffle=False)
    obs_stats = actor_policy.obs_normalization_stats

    system, _target_system = make_wcm_system_from_checkpoint(
        actor_algo, value_checkpoint
    )
    normalization = match_encoder_normalization_to_checkpoint(
        system, value_checkpoint["chunk_value_system"]
    )
    incompatible = system.load_state_dict(
        value_checkpoint["chunk_value_system"], strict=False
    )
    allowed_missing_prefixes = [
        "nets.q_action_encoders.",
        "nets.q_heads.",
    ]
    if not bool(value_checkpoint.get("dynamics_state_saved", False)):
        allowed_missing_prefixes.append("nets.dynamics.")
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "RECAP MC-value checkpoint state is incomplete: "
            f"missing={unexpected_missing[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    system = system.float().to(device).eval()
    system.requires_grad_(False)

    count = int(targets["num_samples"])
    advantages = torch.zeros(count, dtype=torch.float32)
    values = torch.zeros(count, dtype=torch.float32)
    next_values = torch.zeros(count, dtype=torch.float32)
    seen = torch.zeros(count, dtype=torch.bool)
    gamma = float(targets["config"]["gamma"])
    for raw_batch in loader:
        indices = torch.as_tensor(raw_batch["index"], dtype=torch.long).reshape(-1)
        if torch.any(seen.index_select(0, indices)):
            raise RuntimeError("deterministic label loader yielded a duplicate index")
        batch = process_chunk_batch(
            raw_batch,
            actor_algo,
            obs_stats,
            chunk_horizon=int(args.chunk_horizon),
            discount=gamma,
            reward_mode=str(args.loader_reward_mode),
            critic_observation_horizon=int(args.observation_horizon),
            dynamics_prediction_offsets=(),
        )
        _mc_return, _train_mask = _apply_canonical_chunk_fields(
            batch, targets, indices, dynamics_offsets=()
        )
        value = system.value_from_state(
            system.encode_state(batch["obs"], batch.get("goal_obs"))
        )
        next_value = system.value_from_state(
            system.encode_state(batch["next_obs"], batch.get("goal_obs"))
        )
        value_valid, = sidecar_rows(targets, indices, "value_valid")
        eligible_batch = value_valid.to(
            device=value.device, dtype=torch.bool
        ).reshape(-1, 1)
        advantage = masked_chunk_advantage(
            batch["reward"],
            batch["terminal"],
            batch["valid_length"],
            value,
            next_value,
            eligible_batch,
            gamma,
        ).reshape(-1)
        advantages.index_copy_(0, indices, advantage.detach().cpu().float())
        values.index_copy_(0, indices, value.detach().cpu().reshape(-1).float())
        next_values.index_copy_(
            0, indices, next_value.detach().cpu().reshape(-1).float()
        )
        seen.index_fill_(0, indices, True)
    if not torch.all(seen):
        missing = torch.where(~seen)[0][:16].tolist()
        raise RuntimeError(f"label loader missed dataset indices {missing}")
    if not torch.isfinite(advantages).all():
        raise ValueError("computed advantages contain non-finite values")

    source_is_expert = torch.as_tensor(
        targets["fields"]["source_is_expert"]
    ).bool()
    eligible = torch.as_tensor(targets["fields"]["value_valid"]).bool()
    threshold_kwargs: dict[str, float] = {}
    if args.threshold_mode == "fixed":
        threshold_kwargs["fixed_threshold"] = float(args.fixed_threshold)
    elif args.threshold_mode == "rollout_quantile":
        threshold_kwargs["rollout_quantile"] = float(args.rollout_quantile)
    elif args.threshold_mode == "target_positive_fraction":
        threshold_kwargs["target_positive_fraction"] = float(
            args.target_positive_fraction
        )
    else:
        raise ValueError(f"unsupported threshold_mode={args.threshold_mode!r}")
    condition_result = make_recap_conditions(
        advantages,
        source_is_expert,
        eligible=eligible,
        **threshold_kwargs,
    )
    actor_condition = torch.as_tensor(
        condition_result["actor_condition"], dtype=torch.uint8
    )
    if not torch.all(actor_condition[source_is_expert] == 1):
        raise RuntimeError("human-always-positive labeling invariant failed")

    payload = {
        "kind": LABEL_FORMAT,
        "version": 1,
        "dataset": str(args.dataset),
        "dataset_identity": mixed_dataset_identity(args.dataset),
        "num_samples": count,
        "targets_identity": file_stat_identity(args.targets),
        "value_checkpoint_identity": file_stat_identity(args.value_checkpoint),
        "pretrained_dp_identity": file_stat_identity(args.checkpoint),
        "value_epoch": int(value_checkpoint["epoch"]),
        "config": {
            "gamma": gamma,
            "chunk_horizon": int(args.chunk_horizon),
            "advantage_definition": "R_L + gamma**L * (1-terminal) * V(next) - V(current)",
            "threshold_mode": str(args.threshold_mode),
            "threshold": float(condition_result["threshold"]),
            "target_positive_fraction": (
                float(args.target_positive_fraction)
                if args.threshold_mode == "target_positive_fraction"
                else None
            ),
            "rollout_quantile": (
                float(args.rollout_quantile)
                if args.threshold_mode == "rollout_quantile"
                else None
            ),
            "comparison": "strict_greater_than",
            "human_always_positive": True,
            "positive_only": False,
        },
        "fields": {
            "advantage": advantages,
            "actor_condition": actor_condition,
            "source_is_expert": source_is_expert,
            "source_code": torch.as_tensor(targets["fields"]["source_code"]),
            "is_validation": torch.as_tensor(
                targets["fields"]["is_validation"]
            ).bool(),
            "eligible": eligible,
            "value": values,
            "next_value": next_values,
        },
        "summary": {
            "threshold": float(condition_result["threshold"]),
            "rollout_positive_fraction": float(
                condition_result["rollout_positive_fraction"]
            ),
            "overall_positive_fraction": float(
                actor_condition.float().mean().item()
            ),
            "human_rows": int(source_is_expert.sum().item()),
            "eligible_rollout_rows": int((eligible & ~source_is_expert).sum().item()),
            "normalization_reconstruction": normalization,
        },
    }
    if args.output.exists() and not bool(args.overwrite):
        existing = torch.load(args.output, map_location="cpu", weights_only=False)
        if (
            isinstance(existing, dict)
            and existing.get("kind") == LABEL_FORMAT
            and existing.get("dataset_identity") == payload["dataset_identity"]
            and existing.get("targets_identity") == payload["targets_identity"]
            and existing.get("value_checkpoint_identity")
            == payload["value_checkpoint_identity"]
            and existing.get("config") == payload["config"]
        ):
            print(json.dumps({"reused": str(args.output), **existing["summary"]}, indent=2))
            return existing
        raise FileExistsError(
            f"{args.output} exists with different provenance; use --overwrite or a new path"
        )
    atomic_torch_save(payload, args.output)
    print(json.dumps({"labels": str(args.output), **payload["summary"]}, indent=2))
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold-mode",
        choices=("fixed", "rollout_quantile", "target_positive_fraction"),
        default="target_positive_fraction",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.0)
    parser.add_argument("--rollout-quantile", type=float, default=0.6)
    parser.add_argument("--target-positive-fraction", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--hdf5-cache-mode", choices=("all", "low_dim", "none"), default="low_dim"
    )
    parser.add_argument(
        "--sparse-chunk-loader", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    try:
        label_chunks(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
