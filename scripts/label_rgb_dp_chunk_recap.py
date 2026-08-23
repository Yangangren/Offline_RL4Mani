#!/usr/bin/env python3
"""Label mixed RGB-DP chunks with frozen MC-value RECAP advantages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

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
    atomic_write_json,
    build_recap_dataset,
    load_sidecar,
    make_loader,
    sidecar_rows,
    validate_sidecar_dataset,
)
from train_rgb_dp_idql import atomic_torch_save


SUMMARY_QUANTILES = (
    ("p00", 0.00),
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p100", 1.00),
)


def _checkpoint_paths(value: Path | Iterable[Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        paths = [Path(value)]
    else:
        paths = [Path(path) for path in value]
    if not paths:
        raise ValueError("at least one --value-checkpoint is required")
    return [path.expanduser().resolve() for path in paths]


def _validate_value_checkpoint_common(
    checkpoint: dict[str, Any],
    *,
    dataset_identity: dict[str, Any],
    targets_identity: dict[str, Any],
    pretrained_dp_identity: dict[str, Any],
) -> None:
    if checkpoint.get("kind") != VALUE_FORMAT:
        raise ValueError("value checkpoint is not a RECAP MC-value checkpoint")
    if not bool(checkpoint.get("rgb_dp_chunk_recap_value")):
        raise ValueError("value checkpoint lacks the RECAP value marker")
    if bool(checkpoint.get("q_trained", True)):
        raise ValueError("RECAP labeling requires a value-only checkpoint")
    if checkpoint.get("dataset_identity") != dataset_identity:
        raise ValueError("value checkpoint dataset provenance does not match")
    if checkpoint.get("targets_identity") != targets_identity:
        raise ValueError("value checkpoint target provenance does not match")
    if checkpoint.get("pretrained_dp_identity") != pretrained_dp_identity:
        raise ValueError("value checkpoint pretrained-DP provenance does not match")


def load_value_checkpoint_set(
    paths: Path | Iterable[Path],
    targets: dict[str, Any],
    *,
    dataset_identity: dict[str, Any],
    targets_identity: dict[str, Any],
    pretrained_dp_identity: dict[str, Any],
    allow_legacy_single_checkpoint: bool = False,
) -> tuple[list[dict[str, Any]], bool, int]:
    """Validate a complete OOF fold set or one explicitly enabled legacy model."""
    resolved_paths = _checkpoint_paths(paths)
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    fold_field = targets.get("fields", {}).get("fold_index")
    configured_folds = targets.get("config", {}).get("num_folds")
    has_fold_field = fold_field is not None
    has_fold_count = configured_folds is not None
    if has_fold_field != has_fold_count:
        raise ValueError(
            "target sidecar contains partial OOF fold metadata"
        )
    if (
        not has_fold_field
        and targets.get("config", {}).get("fold_seed") is not None
    ):
        raise ValueError(
            "legacy target sidecar contains ambiguous fold_seed metadata"
        )
    strict_oof = has_fold_field and has_fold_count
    if not strict_oof:
        if not allow_legacy_single_checkpoint:
            raise ValueError(
                "target sidecar lacks strict OOF fold metadata; regenerate v2 "
                "targets or pass --allow-legacy-single-checkpoint explicitly"
            )
        if len(resolved_paths) != 1:
            raise ValueError("legacy labeling requires exactly one value checkpoint")
        num_folds = 1
    else:
        num_folds = int(configured_folds)
        if "episode_index" not in targets.get("fields", {}):
            raise ValueError("strict OOF targets require episode_index provenance")
        if num_folds < 2:
            raise ValueError("strict OOF labeling requires num_folds >= 2")
        folds = torch.as_tensor(fold_field, dtype=torch.long).reshape(-1)
        if folds.shape[0] != int(targets["num_samples"]):
            raise ValueError("target fold_index length does not match num_samples")
        if bool(((folds < 0) | (folds >= num_folds)).any()):
            raise ValueError("target fold_index contains an out-of-range fold")
        present = set(int(value) for value in torch.unique(folds).tolist())
        expected = set(range(num_folds))
        if present != expected:
            raise ValueError(
                f"target fold coverage differs from 0..{num_folds - 1}: "
                f"missing={sorted(expected - present)}, "
                f"extra={sorted(present - expected)}"
            )
        if len(resolved_paths) != num_folds:
            raise ValueError(
                f"strict OOF labeling requires exactly {num_folds} value "
                f"checkpoints, got {len(resolved_paths)}"
            )

    records: list[dict[str, Any]] = []
    folds_seen: set[int] = set()
    for path in resolved_paths:
        identity_before = file_stat_identity(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        identity_after = file_stat_identity(path)
        if identity_before != identity_after:
            raise RuntimeError(
                f"value checkpoint changed while it was being loaded: {path}"
            )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"value checkpoint is not a dictionary: {path}")
        _validate_value_checkpoint_common(
            checkpoint,
            dataset_identity=dataset_identity,
            targets_identity=targets_identity,
            pretrained_dp_identity=pretrained_dp_identity,
        )
        if strict_oof:
            if "heldout_fold" not in checkpoint or "num_folds" not in checkpoint:
                raise ValueError(
                    f"OOF value checkpoint lacks heldout_fold/num_folds: {path}"
                )
            fold = int(checkpoint["heldout_fold"])
            if int(checkpoint["num_folds"]) != num_folds:
                raise ValueError(
                    f"value checkpoint num_folds does not match targets: {path}"
                )
            if fold < 0 or fold >= num_folds:
                raise ValueError(f"value checkpoint heldout_fold is invalid: {path}")
            if fold in folds_seen:
                raise ValueError(f"duplicate value checkpoint for heldout fold {fold}")
            folds_seen.add(fold)
            if bool(checkpoint.get("checkpoint_selection_uses_heldout_targets", False)):
                raise ValueError(
                    f"value checkpoint selected epochs using held-out targets: {path}"
                )
            target_fold = torch.as_tensor(
                targets["fields"]["fold_index"], dtype=torch.long
            ).reshape(-1)
            target_episode = torch.as_tensor(
                targets["fields"]["episode_index"], dtype=torch.long
            ).reshape(-1)
            expected_heldout_episodes = {
                int(value)
                for value in torch.unique(target_episode[target_fold == fold]).tolist()
            }
            all_episodes = {
                int(value) for value in torch.unique(target_episode).tolist()
            }
            recorded_heldout = checkpoint.get("heldout_episode_indices")
            if recorded_heldout is not None and {
                int(value) for value in recorded_heldout
            } != expected_heldout_episodes:
                raise ValueError(
                    f"value checkpoint held-out episode provenance is invalid: {path}"
                )
            recorded_training = checkpoint.get("training_episode_indices")
            if recorded_training is not None and {
                int(value) for value in recorded_training
            } != all_episodes - expected_heldout_episodes:
                raise ValueError(
                    f"value checkpoint training episode provenance is invalid: {path}"
                )
        else:
            if checkpoint.get("heldout_fold") is not None or checkpoint.get(
                "num_folds"
            ) not in (None, 1):
                raise ValueError(
                    "legacy single-checkpoint mode refuses partial or ambiguous "
                    "OOF checkpoint metadata"
                )
            fold = 0
        architecture_keys = (
            "critic_architecture",
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
        records.append(
            {
                "fold": fold,
                "path": path,
                "identity": identity_after,
                "epoch": int(checkpoint["epoch"]),
                "critic_observation_horizon": int(
                    checkpoint["critic_observation_horizon"]
                ),
                "architecture_signature": {
                    key: checkpoint.get(key) for key in architecture_keys
                },
            }
        )
        del checkpoint

    if strict_oof and folds_seen != set(range(num_folds)):
        missing = sorted(set(range(num_folds)) - folds_seen)
        raise ValueError(f"missing value checkpoints for heldout folds {missing}")
    reference_architecture = records[0]["architecture_signature"]
    if any(
        record["architecture_signature"] != reference_architecture
        for record in records[1:]
    ):
        raise ValueError("OOF value checkpoints use different architectures")
    records.sort(key=lambda record: int(record["fold"]))
    return records, strict_oof, num_folds


def heldout_prediction_rows(
    fold_index: torch.Tensor,
    fold: int,
) -> torch.Tensor:
    fold_index = torch.as_tensor(fold_index, dtype=torch.long).reshape(-1)
    rows = torch.nonzero(
        fold_index == int(fold), as_tuple=False
    ).reshape(-1)
    if rows.numel() == 0:
        raise ValueError(f"heldout fold {int(fold)} contains no dataset rows")
    return rows


def validate_prediction_coverage(
    prediction_count: torch.Tensor,
    eligible: torch.Tensor,
) -> None:
    prediction_count = torch.as_tensor(
        prediction_count, dtype=torch.int64
    ).reshape(-1)
    eligible = torch.as_tensor(eligible, dtype=torch.bool).reshape(-1)
    if prediction_count.shape != eligible.shape:
        raise ValueError("prediction coverage and eligibility shapes differ")
    invalid = eligible & (prediction_count != 1)
    if bool(invalid.any()):
        examples = torch.where(invalid)[0][:16].tolist()
        raise RuntimeError(
            "eligible OOF rows must be predicted exactly once; "
            f"bad_indices={examples}"
        )


def _finite_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _quantiles(values: torch.Tensor) -> dict[str, float] | None:
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    probabilities = torch.tensor(
        [probability for _name, probability in SUMMARY_QUANTILES],
        dtype=torch.float64,
    )
    result = torch.quantile(values, probabilities)
    return {
        name: float(value)
        for (name, _probability), value in zip(
            SUMMARY_QUANTILES, result.tolist()
        )
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.numel() < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= 0.0:
        return None
    return float((left * right).sum() / denominator)


def _group_diagnostics(
    row_mask: torch.Tensor,
    *,
    eligible: torch.Tensor,
    values: torch.Tensor,
    mc_return: torch.Tensor,
    advantages: torch.Tensor,
    actor_condition: torch.Tensor,
) -> dict[str, Any]:
    row_mask = torch.as_tensor(row_mask, dtype=torch.bool).reshape(-1)
    eligible_mask = row_mask & eligible
    prediction_error = values[eligible_mask] - mc_return[eligible_mask]
    result: dict[str, Any] = {
        "rows": int(row_mask.sum().item()),
        "eligible_rows": int(eligible_mask.sum().item()),
        "positive_rows": int(actor_condition[row_mask].sum().item()),
        "positive_fraction": (
            float(actor_condition[row_mask].float().mean().item())
            if bool(row_mask.any())
            else None
        ),
        "eligible_positive_fraction": (
            float(actor_condition[eligible_mask].float().mean().item())
            if bool(eligible_mask.any())
            else None
        ),
        "advantage_quantiles": _quantiles(advantages[eligible_mask]),
        "mc_return_quantiles": _quantiles(mc_return[eligible_mask]),
        "advantage_mc_return_pearson": _pearson(
            advantages[eligible_mask], mc_return[eligible_mask]
        ),
    }
    if prediction_error.numel() == 0:
        result["value"] = {
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "prediction_target_pearson": None,
        }
        return result

    squared_error = prediction_error.square()
    target = mc_return[eligible_mask]
    centered_target = target - target.mean()
    target_ss = centered_target.square().sum()
    r2 = (
        1.0 - float(squared_error.sum() / target_ss)
        if float(target_ss) > 0.0
        else None
    )
    mse = float(squared_error.mean().item())
    result["value"] = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(prediction_error.abs().mean().item()),
        "r2": _finite_number(r2) if r2 is not None else None,
        "prediction_mean": float(values[eligible_mask].mean().item()),
        "target_mean": float(target.mean().item()),
        "prediction_target_pearson": _pearson(
            values[eligible_mask], target
        ),
    }
    return result


def build_label_summary(
    targets: dict[str, Any],
    *,
    advantages: torch.Tensor,
    values: torch.Tensor,
    actor_condition: torch.Tensor,
    eligible: torch.Tensor,
    prediction_fold: torch.Tensor,
    threshold: float,
    rollout_positive_fraction: float,
    strict_oof: bool,
    num_folds: int,
    normalization_by_fold: dict[str, Any],
) -> dict[str, Any]:
    source_code = torch.as_tensor(targets["fields"]["source_code"]).long()
    source_is_expert = torch.as_tensor(
        targets["fields"]["source_is_expert"]
    ).bool()
    mc_return = torch.as_tensor(targets["fields"]["mc_return"]).float()
    fold_index = (
        torch.as_tensor(targets["fields"]["fold_index"]).long()
        if strict_oof
        else torch.zeros_like(source_code)
    )
    code_to_name = {
        int(code): str(name)
        for name, code in targets.get("source_codes", {}).items()
    }
    all_rows = torch.ones_like(eligible, dtype=torch.bool)
    overall = _group_diagnostics(
        all_rows,
        eligible=eligible,
        values=values,
        mc_return=mc_return,
        advantages=advantages,
        actor_condition=actor_condition,
    )
    source_codes = sorted(
        int(value) for value in torch.unique(source_code).tolist()
    )
    by_source = {
        code_to_name.get(code, f"source_code_{code}"): _group_diagnostics(
            source_code == code,
            eligible=eligible,
            values=values,
            mc_return=mc_return,
            advantages=advantages,
            actor_condition=actor_condition,
        )
        for code in source_codes
    }
    by_fold: dict[str, Any] = {}
    for fold in range(int(num_folds)):
        fold_mask = fold_index == fold
        fold_summary = _group_diagnostics(
            fold_mask,
            eligible=eligible,
            values=values,
            mc_return=mc_return,
            advantages=advantages,
            actor_condition=actor_condition,
        )
        fold_summary["prediction_rows"] = int(
            (prediction_fold == fold).sum().item()
        )
        fold_summary["by_source"] = {
            code_to_name.get(
                code, f"source_code_{code}"
            ): _group_diagnostics(
                fold_mask & (source_code == code),
                eligible=eligible,
                values=values,
                mc_return=mc_return,
                advantages=advantages,
                actor_condition=actor_condition,
            )
            for code in source_codes
        }
        by_fold[str(fold)] = fold_summary

    value_rmse = overall["value"]["rmse"]
    return {
        "labeling_mode": (
            "episode_level_cross_fitted_oof_value"
            if strict_oof
            else "explicit_legacy_single_value"
        ),
        "strict_oof": bool(strict_oof),
        "num_folds": int(num_folds),
        "threshold": float(threshold),
        "rollout_positive_fraction": float(rollout_positive_fraction),
        "overall_positive_fraction": float(
            actor_condition.float().mean().item()
        ),
        "human_rows": int(source_is_expert.sum().item()),
        "eligible_rows": int(eligible.sum().item()),
        "eligible_rollout_rows": int(
            (eligible & ~source_is_expert).sum().item()
        ),
        "threshold_to_value_rmse": (
            float(threshold) / float(value_rmse)
            if value_rmse is not None and value_rmse > 0.0
            else None
        ),
        "absolute_threshold_to_value_rmse": (
            abs(float(threshold)) / float(value_rmse)
            if value_rmse is not None and value_rmse > 0.0
            else None
        ),
        "overall": overall,
        "by_source": by_source,
        "by_fold": by_fold,
        "normalization_reconstruction_by_fold": normalization_by_fold,
    }


@torch.no_grad()
def label_chunks(args: argparse.Namespace) -> dict:
    args.dataset = args.dataset.expanduser().resolve()
    args.targets = args.targets.expanduser().resolve()
    args.value_checkpoint = _checkpoint_paths(args.value_checkpoint)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for path in (args.dataset, args.targets, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    targets = load_sidecar(args.targets, expected_kind=TARGET_FORMAT)
    validate_sidecar_dataset(
        targets, args.dataset, sidecar_name="target sidecar"
    )
    configured_num_folds = targets.get("config", {}).get("num_folds")
    expected_num_folds = getattr(args, "expected_num_folds", None)
    if (
        expected_num_folds is not None
        and configured_num_folds is not None
        and int(expected_num_folds) != int(configured_num_folds)
    ):
        raise ValueError(
            f"requested num_folds={int(expected_num_folds)} differs from "
            f"prepared targets={int(configured_num_folds)}; use the target "
            "fold count and provide one held-out checkpoint per fold"
        )
    dataset_identity = mixed_dataset_identity(args.dataset)
    targets_identity = file_stat_identity(args.targets)
    pretrained_dp_identity = file_stat_identity(args.checkpoint)
    checkpoint_records, strict_oof, num_folds = load_value_checkpoint_set(
        args.value_checkpoint,
        targets,
        dataset_identity=dataset_identity,
        targets_identity=targets_identity,
        pretrained_dp_identity=pretrained_dp_identity,
        allow_legacy_single_checkpoint=bool(
            getattr(args, "allow_legacy_single_checkpoint", False)
        ),
    )

    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    actor_policy, _dp_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    actor_algo = actor_policy.policy
    actor_algo.set_eval()
    actor_algo.nets.requires_grad_(False)
    args.chunk_horizon = int(targets["config"]["chunk_horizon"])
    observation_horizons = {
        int(record["critic_observation_horizon"])
        for record in checkpoint_records
    }
    if len(observation_horizons) != 1:
        raise ValueError(
            "OOF value checkpoints disagree on critic observation horizon"
        )
    args.observation_horizon = next(iter(observation_horizons))
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
    obs_stats = actor_policy.obs_normalization_stats

    count = int(targets["num_samples"])
    advantages = torch.full((count,), float("nan"), dtype=torch.float32)
    values = torch.full((count,), float("nan"), dtype=torch.float32)
    next_values = torch.full((count,), float("nan"), dtype=torch.float32)
    prediction_count = torch.zeros(count, dtype=torch.int16)
    prediction_fold = torch.full((count,), -1, dtype=torch.int16)
    target_fold = (
        torch.as_tensor(targets["fields"]["fold_index"]).long()
        if strict_oof
        else torch.zeros(count, dtype=torch.long)
    )
    eligible = torch.as_tensor(targets["fields"]["value_valid"]).bool()
    normalization_by_fold: dict[str, Any] = {}
    gamma = float(targets["config"]["gamma"])

    for record in checkpoint_records:
        fold = int(record["fold"])
        if file_stat_identity(record["path"]) != record["identity"]:
            raise RuntimeError(
                f"value checkpoint changed during labeling: {record['path']}"
            )
        value_checkpoint = torch.load(
            record["path"], map_location="cpu", weights_only=False
        )
        _validate_value_checkpoint_common(
            value_checkpoint,
            dataset_identity=dataset_identity,
            targets_identity=targets_identity,
            pretrained_dp_identity=pretrained_dp_identity,
        )
        if strict_oof and (
            int(value_checkpoint.get("heldout_fold", -1)) != fold
            or int(value_checkpoint.get("num_folds", -1)) != num_folds
        ):
            raise RuntimeError(
                f"value checkpoint fold metadata changed: {record['path']}"
            )
        fold_rows = heldout_prediction_rows(target_fold, fold)
        loader = make_loader(
            dataset, args, shuffle=False, indices=fold_rows
        )

        system, _target_system = make_wcm_system_from_checkpoint(
            actor_algo, value_checkpoint
        )
        normalization = match_encoder_normalization_to_checkpoint(
            system, value_checkpoint["chunk_value_system"]
        )
        normalization_by_fold[str(fold)] = normalization
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
            if not any(
                key.startswith(prefix)
                for prefix in allowed_missing_prefixes
            )
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "RECAP MC-value checkpoint state is incomplete: "
                f"missing={unexpected_missing[:8]}, "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        system = system.float().to(device).eval()
        system.requires_grad_(False)

        for raw_batch in loader:
            indices = torch.as_tensor(
                raw_batch["index"], dtype=torch.long
            ).reshape(-1)
            if bool((prediction_count.index_select(0, indices) != 0).any()):
                raise RuntimeError(
                    "OOF label loaders yielded a duplicate dataset index"
                )
            if strict_oof and not torch.all(target_fold[indices] == fold):
                raise RuntimeError(
                    "heldout-fold loader crossed an episode-fold boundary"
                )
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
                system.encode_state(
                    batch["next_obs"], batch.get("goal_obs")
                )
            )
            value_valid, = sidecar_rows(
                targets, indices, "value_valid"
            )
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
            advantages.index_copy_(
                0, indices, advantage.detach().cpu().float()
            )
            values.index_copy_(
                0, indices, value.detach().cpu().reshape(-1).float()
            )
            next_values.index_copy_(
                0,
                indices,
                next_value.detach().cpu().reshape(-1).float(),
            )
            prediction_count.index_add_(
                0, indices, torch.ones_like(indices, dtype=torch.int16)
            )
            prediction_fold.index_fill_(0, indices, fold)
        del system, value_checkpoint

    validate_prediction_coverage(prediction_count, eligible)
    if bool((prediction_count != 1).any()):
        missing = torch.where(prediction_count != 1)[0][:16].tolist()
        raise RuntimeError(
            "OOF fold loaders must cover every aligned dataset row exactly "
            f"once; bad_indices={missing}"
        )
    if not torch.isfinite(advantages).all():
        raise ValueError("computed OOF advantages contain non-finite values")
    if not torch.isfinite(values[eligible]).all():
        raise ValueError("computed eligible OOF values contain non-finite values")

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

    summary = build_label_summary(
        targets,
        advantages=advantages,
        values=values,
        actor_condition=actor_condition,
        eligible=eligible,
        prediction_fold=prediction_fold,
        threshold=float(condition_result["threshold"]),
        rollout_positive_fraction=float(
            condition_result["rollout_positive_fraction"]
        ),
        strict_oof=strict_oof,
        num_folds=num_folds,
        normalization_by_fold=normalization_by_fold,
    )
    value_checkpoint_records = [
        {
            "heldout_fold": int(record["fold"]),
            "path": str(record["path"]),
            "identity": record["identity"],
            "epoch": int(record["epoch"]),
        }
        for record in checkpoint_records
    ]
    fields: dict[str, torch.Tensor] = {
        "advantage": advantages,
        "actor_condition": actor_condition,
        "source_is_expert": source_is_expert,
        "source_code": torch.as_tensor(targets["fields"]["source_code"]),
        "eligible": eligible,
        "value": values,
        "next_value": next_values,
        "prediction_fold": prediction_fold,
        "prediction_count": prediction_count,
    }
    if "fold_index" in targets["fields"]:
        fields["fold_index"] = torch.as_tensor(
            targets["fields"]["fold_index"]
        ).to(dtype=torch.int16)
    if "is_validation" in targets["fields"]:
        fields["is_validation"] = torch.as_tensor(
            targets["fields"]["is_validation"]
        ).bool()

    payload = {
        "kind": LABEL_FORMAT,
        "version": 2,
        "dataset": str(args.dataset),
        "dataset_identity": dataset_identity,
        "num_samples": count,
        "targets_identity": targets_identity,
        "value_checkpoints": value_checkpoint_records,
        "value_checkpoint_identities": [
            record["identity"] for record in checkpoint_records
        ],
        "pretrained_dp_identity": pretrained_dp_identity,
        "value_epochs": [
            int(record["epoch"])
            for record in checkpoint_records
        ],
        "strict_oof": bool(strict_oof),
        "num_folds": int(num_folds),
        "config": {
            "gamma": gamma,
            "chunk_horizon": int(args.chunk_horizon),
            "advantage_definition": (
                "R_L + gamma**L * (1-terminal) * V(next) - V(current)"
            ),
            "value_estimator": (
                "episode_level_cross_fitted_oof"
                if strict_oof
                else "explicit_legacy_single_checkpoint"
            ),
            "fold_seed": targets.get("config", {}).get("fold_seed"),
            "num_folds": int(num_folds),
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
        "fields": fields,
        "summary": summary,
    }
    if not strict_oof:
        payload["value_checkpoint_identity"] = checkpoint_records[0][
            "identity"
        ]
        payload["value_epoch"] = int(checkpoint_records[0]["epoch"])

    summary_output = getattr(args, "summary_output", None)
    if summary_output is None:
        summary_output = args.output.with_suffix(".summary.json")
    else:
        summary_output = Path(summary_output).expanduser().resolve()

    if args.output.exists() and not bool(args.overwrite):
        existing = torch.load(args.output, map_location="cpu", weights_only=False)
        if (
            isinstance(existing, dict)
            and existing.get("kind") == LABEL_FORMAT
            and existing.get("dataset_identity") == payload["dataset_identity"]
            and existing.get("targets_identity") == payload["targets_identity"]
            and existing.get("value_checkpoint_identities")
            == payload["value_checkpoint_identities"]
            and existing.get("config") == payload["config"]
        ):
            atomic_write_json(summary_output, existing["summary"])
            print(
                json.dumps(
                    {
                        "reused": str(args.output),
                        "summary": str(summary_output),
                        "strict_oof": bool(existing.get("strict_oof")),
                        "num_folds": int(existing.get("num_folds", 1)),
                        "threshold": existing["summary"]["threshold"],
                        "rollout_positive_fraction": existing["summary"][
                            "rollout_positive_fraction"
                        ],
                    },
                    indent=2,
                )
            )
            return existing
        raise FileExistsError(
            f"{args.output} exists with different provenance; use --overwrite or a new path"
        )
    atomic_torch_save(payload, args.output)
    atomic_write_json(summary_output, summary)
    print(
        json.dumps(
            {
                "labels": str(args.output),
                "summary": str(summary_output),
                "strict_oof": bool(strict_oof),
                "num_folds": int(num_folds),
                "threshold": summary["threshold"],
                "rollout_positive_fraction": summary[
                    "rollout_positive_fraction"
                ],
                "overall_positive_fraction": summary[
                    "overall_positive_fraction"
                ],
                "human_rows": summary["human_rows"],
                "eligible_rollout_rows": summary[
                    "eligible_rollout_rows"
                ],
            },
            indent=2,
        )
    )
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeat once per held-out fold. Strict v2 labeling requires a "
            "complete checkpoint set."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-num-folds", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-single-checkpoint",
        action="store_true",
        help=(
            "Explicitly allow non-OOF labeling only when targets contain no "
            "fold metadata and exactly one unambiguous checkpoint is supplied."
        ),
    )
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
