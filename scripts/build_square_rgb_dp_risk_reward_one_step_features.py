#!/usr/bin/env python3
"""Relabel one-step IDQL features with learned positive-action-risk rewards.

This script keeps the standard one-step IDQL transition format unchanged:

    (phi(o_t), a_t, r_t, phi(o_{t+1}), done_t)

Only the scalar reward is replaced. For each one-step transition we score the
*actual future action chunk from the dataset*, not repeated copies of the
current action:

    A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]

The frozen causal prefix-risk model predicts an incremental action-risk
log-odds delta. Positive delta means the actual future action chunk increases
failure risk relative to the state baseline. Four reward modes are supported:

* risk_only:

    r_t = -lambda * clip(max(delta_t, 0) / threshold, 0, reward_clip)

* hybrid_default_minus_risk:

    r_t = r_t^default - lambda * clip(max(delta_t, 0) / threshold, 0, reward_clip)

* hybrid_default_signed_risk:

    r_t = r_t^default - lambda * clip(delta_t / signed_scale, -reward_clip, reward_clip)

* failure_only_signed_risk:

    r_t = r_t^default - 1[source=rollout_failure] * lambda * clip(delta_t / signed_scale, -reward_clip, reward_clip)

* failure_only_potential_risk_shaping:

    Phi_t = -p_fail(h_t)
    r_t = r_t^default + 1[source=rollout_failure] * lambda * clip(gamma * Phi_{t+1} - Phi_t, -reward_clip, reward_clip)

The signed modes penalize risk-increasing chunks and reward risk-reducing chunks.
The failure-only modes keep human demos and successful rollouts on the clean
sparse default reward, and only shape failed rollouts.
The output NPZ is therefore still compatible with train_square_rgb_dp_one_step_idql.py.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

from robomimic.models.prefix_risk_nets import CausalPrefixRisk
from train_rgb_dp_causal_prefix_risk import predict as predict_prefix_risk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_FEATURES = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/default_reward_one_step_features.npz"
)
DEFAULT_DEMO_DATASET = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_ROLLOUT_DATASET = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_RISK_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp_causal_prefix_risk/epoch190_two_stage_temporal_safe_anchor/best.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/idql/positive_action_risk_reward_one_step_features.npz"
)


def decode_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return array.astype(str)


def encode_string(value: str) -> np.ndarray:
    return np.asarray(value.encode("utf-8"))


def action_chunk_from_episode(
    actions: np.ndarray,
    step: int,
    horizon: int,
    pad_mode: str,
) -> np.ndarray:
    """Return actual future action chunk a[step:step+horizon], padded at end."""
    action_dim = int(actions.shape[-1])
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    step = int(step)
    if step < 0 or step >= len(actions):
        return chunk
    end = min(step + horizon, len(actions))
    length = max(end - step, 0)
    if length > 0:
        chunk[:length] = actions[step:end].astype(np.float32)
    if length < horizon and pad_mode == "repeat_last" and length > 0:
        chunk[length:] = chunk[length - 1]
    return chunk


def build_actual_future_chunks(
    *,
    source: np.ndarray,
    demos: np.ndarray,
    steps: np.ndarray,
    action_dim: int,
    horizon: int,
    demo_dataset: Path,
    rollout_dataset: Path,
    pad_mode: str,
    max_samples: int | None,
) -> np.ndarray:
    num_samples = len(steps) if max_samples is None else min(int(max_samples), len(steps))
    chunks = np.zeros((num_samples, horizon, action_dim), dtype=np.float32)

    source_to_path = {
        "demo": demo_dataset,
        "rollout_success": rollout_dataset,
        "rollout_failure": rollout_dataset,
    }
    for source_name, path in source_to_path.items():
        source_indices = np.flatnonzero(source[:num_samples] == source_name)
        if len(source_indices) == 0:
            continue
        with h5py.File(path, "r") as h5:
            unique_demos = np.unique(demos[source_indices])
            for demo_key in unique_demos:
                episode_indices = source_indices[demos[source_indices] == demo_key]
                action_path = f"data/{demo_key}/actions"
                if action_path not in h5:
                    raise KeyError(f"{action_path} not found in {path}")
                episode_actions = h5[action_path][:].astype(np.float32)
                for index in episode_indices:
                    chunks[index] = action_chunk_from_episode(
                        episode_actions,
                        int(steps[index]),
                        horizon,
                        pad_mode,
                    )
    return chunks


def ordered_episode_offsets(
    *,
    source: np.ndarray,
    demos: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return reorder indices and offsets grouped by source/demo then sorted by step."""
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, (source_name, demo_key) in enumerate(zip(source, demos, strict=True)):
        episode_id = f"{source_name}:{demo_key}"
        groups.setdefault(episode_id, []).append(index)

    reorder = []
    offsets = [0]
    episode_ids = []
    for episode_id, indices in groups.items():
        sorted_indices = sorted(indices, key=lambda idx: int(steps[idx]))
        reorder.extend(sorted_indices)
        offsets.append(len(reorder))
        episode_ids.append(episode_id)
    return (
        np.asarray(reorder, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        episode_ids,
    )


def instantiate_risk_model(checkpoint: dict, device: torch.device) -> CausalPrefixRisk:
    args = checkpoint["args"]
    model = CausalPrefixRisk(
        feature_dim=int(checkpoint["feature_dim"]),
        prediction_horizon=int(checkpoint["prediction_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        action_hidden_dim=int(args["action_hidden_dim"]),
        dropout=float(args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def normalize_for_risk(
    features: np.ndarray,
    actions: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    norm_features = (features - stats["feature_mean"]) / stats["feature_std"]
    norm_actions = (actions - stats["action_mean"]) / stats["action_std"]
    return norm_features.astype(np.float32), norm_actions.astype(np.float32)


def source_stats(
    rewards: np.ndarray,
    risks: np.ndarray,
    source: np.ndarray,
    success: np.ndarray,
    *,
    default_rewards: np.ndarray | None = None,
    risk_cost: np.ndarray | None = None,
    action_deltas: np.ndarray | None = None,
    signed_risk_advantage: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> dict:
    result = {}
    for source_name in sorted(np.unique(source).tolist()):
        mask = source == source_name
        if not np.any(mask):
            continue
        row = {
            "count": int(mask.sum()),
            "reward_mean": float(rewards[mask].mean()),
            "reward_std": float(rewards[mask].std()),
            "positive_action_risk_mean": float(risks[mask].mean()),
            "positive_action_risk_p95": float(np.quantile(risks[mask], 0.95)),
            "episode_success_mean": float(success[mask].mean()),
        }
        if default_rewards is not None:
            row["default_reward_mean"] = float(default_rewards[mask].mean())
            row["default_reward_std"] = float(default_rewards[mask].std())
        if risk_cost is not None:
            row["risk_cost_mean"] = float(risk_cost[mask].mean())
            row["risk_cost_p95"] = float(np.quantile(risk_cost[mask], 0.95))
        if action_deltas is not None:
            row["action_delta_mean"] = float(action_deltas[mask].mean())
            row["action_delta_p05"] = float(np.quantile(action_deltas[mask], 0.05))
            row["action_delta_p95"] = float(np.quantile(action_deltas[mask], 0.95))
        if signed_risk_advantage is not None:
            row["signed_risk_advantage_mean"] = float(signed_risk_advantage[mask].mean())
            row["signed_risk_advantage_p05"] = float(np.quantile(signed_risk_advantage[mask], 0.05))
            row["signed_risk_advantage_p95"] = float(np.quantile(signed_risk_advantage[mask], 0.95))
        if risk_penalty is not None:
            row["risk_penalty_mean"] = float(risk_penalty[mask].mean())
            row["risk_penalty_p05"] = float(np.quantile(risk_penalty[mask], 0.05))
            row["risk_penalty_p95"] = float(np.quantile(risk_penalty[mask], 0.95))
        result[source_name] = row
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--demo-dataset", type=Path, default=DEFAULT_DEMO_DATASET)
    parser.add_argument("--rollout-dataset", type=Path, default=DEFAULT_ROLLOUT_DATASET)
    parser.add_argument("--risk-checkpoint", type=Path, default=DEFAULT_RISK_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--risk-threshold", type=float, default=0.014938089996576302)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        choices=(
            "risk_only",
            "hybrid_default_minus_risk",
            "hybrid_default_signed_risk",
            "failure_only_signed_risk",
            "failure_only_potential_risk_shaping",
        ),
        default="risk_only",
        help=(
            "risk_only preserves the previous behavior; "
            "hybrid_default_minus_risk keeps default reward and subtracts positive risk cost; "
            "hybrid_default_signed_risk keeps default reward, penalizes positive risk deltas, "
            "and rewards negative risk deltas; "
            "failure_only_signed_risk applies the signed risk term only to failed rollout samples; "
            "failure_only_potential_risk_shaping uses state-risk potential progress only on failed rollouts."
        ),
    )
    parser.add_argument(
        "--signed-risk-scale",
        type=float,
        default=None,
        help=(
            "Scale tau for signed risk reward. If omitted, tau is computed as "
            "quantile(abs(action_delta), --signed-risk-quantile)."
        ),
    )
    parser.add_argument(
        "--signed-risk-quantile",
        type=float,
        default=0.95,
        help="Quantile of abs(action_delta) used as tau for signed risk reward when --signed-risk-scale is omitted.",
    )
    parser.add_argument(
        "--potential-type",
        choices=("probability", "logit"),
        default="probability",
        help=(
            "Potential used by failure_only_potential_risk_shaping. "
            "probability uses Phi=-p_fail; logit uses Phi=-logit(p_fail)."
        ),
    )
    parser.add_argument(
        "--potential-gamma",
        type=float,
        default=None,
        help="Discount used inside gamma * Phi(next) - Phi(current). Defaults to base feature gamma.",
    )
    parser.add_argument(
        "--terminal-risk-mode",
        choices=("outcome", "current", "zero"),
        default="outcome",
        help=(
            "How to set p_fail(h_{T+1}) for final transitions when computing potential shaping. "
            "outcome uses 0 for successful episodes and 1 for failed episodes; current copies p_fail(h_T); zero uses 0."
        ),
    )
    parser.add_argument(
        "--pad-mode",
        choices=("zero", "repeat_last"),
        default="zero",
        help="How to pad actual future chunks near the end of an episode.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debugging limit. The saved NPZ will contain only this prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base_npz = np.load(args.base_features, allow_pickle=True)
    base = {key: base_npz[key] for key in base_npz.files}
    num_total = int(base["steps"].shape[0])
    num_samples = num_total if args.max_samples is None else min(int(args.max_samples), num_total)
    if int(base["chunk_horizon"]) != 1:
        raise ValueError(f"expected one-step base features, got chunk_horizon={base['chunk_horizon']}")

    source = decode_array(base["source"])[:num_samples]
    demos = decode_array(base["demo"])[:num_samples]
    steps = base["steps"][:num_samples].astype(np.int64)
    episode_success = base["episode_success"][:num_samples].astype(np.float32)
    action_dim = int(base["action_dim"])

    checkpoint = torch.load(args.risk_checkpoint, map_location="cpu", weights_only=False)
    risk_horizon = int(checkpoint["prediction_horizon"])
    if int(checkpoint["action_dim"]) != action_dim:
        raise ValueError(
            f"risk checkpoint action_dim={checkpoint['action_dim']} but feature action_dim={action_dim}"
        )

    print(f"Loading actual future action chunks: N={num_samples}, H={risk_horizon}")
    future_chunks = build_actual_future_chunks(
        source=source,
        demos=demos,
        steps=steps,
        action_dim=action_dim,
        horizon=risk_horizon,
        demo_dataset=args.demo_dataset,
        rollout_dataset=args.rollout_dataset,
        pad_mode=args.pad_mode,
        max_samples=args.max_samples,
    )

    reorder, offsets, episode_ids = ordered_episode_offsets(
        source=source,
        demos=demos,
        steps=steps,
    )
    inverse = np.empty_like(reorder)
    inverse[reorder] = np.arange(len(reorder), dtype=np.int64)

    seq_features = base["obs_features"][:num_samples][reorder].astype(np.float32)
    seq_chunks = future_chunks[reorder].astype(np.float32)
    norm_features, norm_chunks = normalize_for_risk(
        seq_features,
        seq_chunks,
        checkpoint["stats"],
    )

    model = instantiate_risk_model(checkpoint, device)
    episodes = np.arange(len(offsets) - 1, dtype=np.int64)
    print(f"Scoring {len(episodes)} episodes with frozen prefix-risk model on {device}")
    state_scores_seq, action_scores_seq, action_deltas_seq = predict_prefix_risk(
        model=model,
        features=norm_features,
        actions=norm_chunks,
        offsets=offsets,
        episodes=episodes,
        device=device,
        batch_size=int(args.eval_batch_size),
    )

    next_state_scores_seq = np.empty_like(state_scores_seq, dtype=np.float32)
    seq_success = episode_success[reorder].astype(np.float32)
    for episode_index in range(len(offsets) - 1):
        start = int(offsets[episode_index])
        stop = int(offsets[episode_index + 1])
        if stop <= start:
            continue
        if stop - start > 1:
            next_state_scores_seq[start : stop - 1] = state_scores_seq[start + 1 : stop]
        if args.terminal_risk_mode == "outcome":
            terminal_risk = 0.0 if float(seq_success[stop - 1]) > 0.5 else 1.0
        elif args.terminal_risk_mode == "current":
            terminal_risk = float(state_scores_seq[stop - 1])
        elif args.terminal_risk_mode == "zero":
            terminal_risk = 0.0
        else:
            raise ValueError(f"unknown terminal risk mode: {args.terminal_risk_mode}")
        next_state_scores_seq[stop - 1] = terminal_risk

    state_scores = state_scores_seq[inverse].astype(np.float32)
    next_state_scores = next_state_scores_seq[inverse].astype(np.float32)
    action_scores = action_scores_seq[inverse].astype(np.float32)
    action_deltas = action_deltas_seq[inverse].astype(np.float32)
    clipped_state_scores = np.clip(state_scores, 1e-6, 1.0 - 1e-6).astype(np.float32)
    clipped_next_state_scores = np.clip(next_state_scores, 1e-6, 1.0 - 1e-6).astype(np.float32)
    if args.potential_type == "probability":
        state_potential = (-clipped_state_scores).astype(np.float32)
        next_state_potential = (-clipped_next_state_scores).astype(np.float32)
    elif args.potential_type == "logit":
        state_logit = np.log(clipped_state_scores / (1.0 - clipped_state_scores)).astype(np.float32)
        next_state_logit = np.log(clipped_next_state_scores / (1.0 - clipped_next_state_scores)).astype(np.float32)
        state_potential = (-state_logit).astype(np.float32)
        next_state_potential = (-next_state_logit).astype(np.float32)
    else:
        raise ValueError(f"unknown potential type: {args.potential_type}")
    base_gamma = float(base["gamma"]) if "gamma" in base else 0.99
    potential_gamma = base_gamma if args.potential_gamma is None else float(args.potential_gamma)
    potential_delta_raw = (potential_gamma * next_state_potential - state_potential).astype(np.float32)
    potential_delta = np.clip(
        potential_delta_raw,
        -float(args.reward_clip),
        float(args.reward_clip),
    ).astype(np.float32)
    potential_shaping_reward = (float(args.risk_lambda) * potential_delta).astype(np.float32)
    positive_action_risk = np.maximum(action_deltas, 0.0).astype(np.float32)
    risk_cost = np.clip(
        positive_action_risk / max(float(args.risk_threshold), 1e-8),
        0.0,
        float(args.reward_clip),
    ).astype(np.float32)
    default_rewards = base["chunk_returns"][:num_samples].astype(np.float32)
    signed_risk_quantile = float(args.signed_risk_quantile)
    if not 0.0 < signed_risk_quantile <= 1.0:
        raise ValueError(f"--signed-risk-quantile must be in (0, 1], got {signed_risk_quantile}")
    if args.signed_risk_scale is None:
        signed_risk_scale = float(np.quantile(np.abs(action_deltas), signed_risk_quantile))
    else:
        signed_risk_scale = float(args.signed_risk_scale)
    signed_risk_scale = max(signed_risk_scale, 1e-8)
    signed_risk_advantage = np.clip(
        action_deltas / signed_risk_scale,
        -float(args.reward_clip),
        float(args.reward_clip),
    ).astype(np.float32)

    positive_risk_penalty = (float(args.risk_lambda) * risk_cost).astype(np.float32)
    signed_risk_penalty = (float(args.risk_lambda) * signed_risk_advantage).astype(np.float32)
    signed_risk_bonus = (-signed_risk_penalty).astype(np.float32)
    failure_risk_mask = (source == "rollout_failure").astype(np.float32)

    if args.reward_mode == "risk_only":
        risk_penalty = positive_risk_penalty
        relabeled_reward = (-risk_penalty).astype(np.float32)
        reward_formula = "r_t = -risk_lambda * clip(max(action_delta_t, 0) / risk_threshold, 0, reward_clip)"
    elif args.reward_mode == "hybrid_default_minus_risk":
        risk_penalty = positive_risk_penalty
        relabeled_reward = (default_rewards - risk_penalty).astype(np.float32)
        reward_formula = "r_t = default_reward_t - risk_lambda * clip(max(action_delta_t, 0) / risk_threshold, 0, reward_clip)"
    elif args.reward_mode == "hybrid_default_signed_risk":
        risk_penalty = signed_risk_penalty
        relabeled_reward = (default_rewards - risk_penalty).astype(np.float32)
        reward_formula = "r_t = default_reward_t - risk_lambda * clip(action_delta_t / signed_risk_scale, -reward_clip, reward_clip)"
    elif args.reward_mode == "failure_only_signed_risk":
        risk_penalty = (failure_risk_mask * signed_risk_penalty).astype(np.float32)
        relabeled_reward = (default_rewards - risk_penalty).astype(np.float32)
        reward_formula = "r_t = default_reward_t - 1[source=rollout_failure] * risk_lambda * clip(action_delta_t / signed_risk_scale, -reward_clip, reward_clip)"
    elif args.reward_mode == "failure_only_potential_risk_shaping":
        risk_penalty = (-failure_risk_mask * potential_shaping_reward).astype(np.float32)
        relabeled_reward = (default_rewards - risk_penalty).astype(np.float32)
        reward_formula = "r_t = default_reward_t + 1[source=rollout_failure] * risk_lambda * clip(potential_gamma * Phi(h_{t+1}) - Phi(h_t), -reward_clip, reward_clip)"
    else:
        raise ValueError(f"unknown reward mode: {args.reward_mode}")

    output = {}
    for key, value in base.items():
        if value.shape == ():
            output[key] = value
        elif value.shape[0] == num_total:
            output[key] = value[:num_samples]
        else:
            output[key] = value

    output["default_chunk_returns"] = default_rewards
    output["chunk_returns"] = relabeled_reward
    output["reward_mean"] = np.asarray(relabeled_reward.mean(), dtype=np.float32)
    output["reward_std"] = np.asarray(max(float(relabeled_reward.std()), 1e-6), dtype=np.float32)
    output["risk_future_action_chunks"] = future_chunks.astype(np.float32)
    output["risk_state_scores"] = state_scores
    output["risk_next_state_scores"] = next_state_scores
    output["risk_state_potential"] = state_potential
    output["risk_next_state_potential"] = next_state_potential
    output["risk_potential_delta_raw"] = potential_delta_raw
    output["risk_potential_delta"] = potential_delta
    output["risk_potential_shaping_reward"] = potential_shaping_reward
    output["risk_action_scores"] = action_scores
    output["risk_action_deltas"] = action_deltas
    output["positive_action_risk"] = positive_action_risk
    output["risk_cost"] = risk_cost
    output["risk_penalty"] = risk_penalty
    output["positive_risk_penalty"] = positive_risk_penalty
    output["signed_risk_advantage"] = signed_risk_advantage
    output["signed_risk_penalty"] = signed_risk_penalty
    output["signed_risk_bonus"] = signed_risk_bonus
    output["failure_risk_mask"] = failure_risk_mask.astype(np.float32)
    output["signed_risk_scale"] = np.asarray(float(signed_risk_scale), dtype=np.float32)
    output["signed_risk_quantile"] = np.asarray(float(signed_risk_quantile), dtype=np.float32)
    output["risk_threshold"] = np.asarray(float(args.risk_threshold), dtype=np.float32)
    output["risk_lambda"] = np.asarray(float(args.risk_lambda), dtype=np.float32)
    output["risk_potential_type"] = encode_string(args.potential_type)
    output["risk_potential_gamma"] = np.asarray(float(potential_gamma), dtype=np.float32)
    output["risk_terminal_risk_mode"] = encode_string(args.terminal_risk_mode)
    output["risk_reward_mode"] = encode_string(args.reward_mode)
    output["risk_reward_formula"] = encode_string(reward_formula)
    output["risk_checkpoint"] = encode_string(str(args.risk_checkpoint))
    output["base_features"] = encode_string(str(args.base_features))
    output["risk_action_horizon"] = np.asarray(risk_horizon, dtype=np.int64)
    output["risk_pad_mode"] = encode_string(args.pad_mode)
    output["risk_episode_ids"] = np.asarray([item.encode("utf-8") for item in episode_ids])

    np.savez_compressed(args.output, **output)

    summary = {
        "base_features": str(args.base_features),
        "output": str(args.output),
        "risk_checkpoint": str(args.risk_checkpoint),
        "demo_dataset": str(args.demo_dataset),
        "rollout_dataset": str(args.rollout_dataset),
        "num_samples": int(num_samples),
        "num_total_base_samples": int(num_total),
        "num_episodes": int(len(episode_ids)),
        "risk_action_horizon": int(risk_horizon),
        "reward_mode": args.reward_mode,
        "reward_formula": reward_formula,
        "risk_threshold": float(args.risk_threshold),
        "signed_risk_scale": float(signed_risk_scale),
        "signed_risk_quantile": float(signed_risk_quantile),
        "risk_lambda": float(args.risk_lambda),
        "potential_type": args.potential_type,
        "potential_gamma": float(potential_gamma),
        "terminal_risk_mode": args.terminal_risk_mode,
        "reward_clip": float(args.reward_clip),
        "pad_mode": args.pad_mode,
        "risk_active_count": int(np.count_nonzero(risk_penalty)),
        "risk_active_fraction": float(np.count_nonzero(risk_penalty) / max(len(risk_penalty), 1)),
        "failure_sample_count": int(np.count_nonzero(failure_risk_mask)),
        "failure_sample_fraction": float(np.mean(failure_risk_mask)),
        "reward_stats": {
            "mean": float(relabeled_reward.mean()),
            "std": float(relabeled_reward.std()),
            "min": float(relabeled_reward.min()),
            "p05": float(np.quantile(relabeled_reward, 0.05)),
            "median": float(np.quantile(relabeled_reward, 0.5)),
            "p95": float(np.quantile(relabeled_reward, 0.95)),
            "max": float(relabeled_reward.max()),
        },
        "default_reward_stats": {
            "mean": float(default_rewards.mean()),
            "std": float(default_rewards.std()),
            "min": float(default_rewards.min()),
            "p95": float(np.quantile(default_rewards, 0.95)),
            "max": float(default_rewards.max()),
        },
        "risk_cost_stats": {
            "mean": float(risk_cost.mean()),
            "std": float(risk_cost.std()),
            "min": float(risk_cost.min()),
            "p50": float(np.quantile(risk_cost, 0.5)),
            "p95": float(np.quantile(risk_cost, 0.95)),
            "max": float(risk_cost.max()),
        },
        "action_delta_stats": {
            "mean": float(action_deltas.mean()),
            "std": float(action_deltas.std()),
            "min": float(action_deltas.min()),
            "p05": float(np.quantile(action_deltas, 0.05)),
            "p50": float(np.quantile(action_deltas, 0.5)),
            "p95": float(np.quantile(action_deltas, 0.95)),
            "max": float(action_deltas.max()),
        },
        "signed_risk_advantage_stats": {
            "mean": float(signed_risk_advantage.mean()),
            "std": float(signed_risk_advantage.std()),
            "min": float(signed_risk_advantage.min()),
            "p05": float(np.quantile(signed_risk_advantage, 0.05)),
            "p50": float(np.quantile(signed_risk_advantage, 0.5)),
            "p95": float(np.quantile(signed_risk_advantage, 0.95)),
            "max": float(signed_risk_advantage.max()),
        },
        "risk_penalty_stats": {
            "mean": float(risk_penalty.mean()),
            "std": float(risk_penalty.std()),
            "min": float(risk_penalty.min()),
            "p05": float(np.quantile(risk_penalty, 0.05)),
            "p50": float(np.quantile(risk_penalty, 0.5)),
            "p95": float(np.quantile(risk_penalty, 0.95)),
            "max": float(risk_penalty.max()),
        },
        "state_risk_stats": {
            "mean": float(state_scores.mean()),
            "std": float(state_scores.std()),
            "min": float(state_scores.min()),
            "p05": float(np.quantile(state_scores, 0.05)),
            "p50": float(np.quantile(state_scores, 0.5)),
            "p95": float(np.quantile(state_scores, 0.95)),
            "max": float(state_scores.max()),
        },
        "next_state_risk_stats": {
            "mean": float(next_state_scores.mean()),
            "std": float(next_state_scores.std()),
            "min": float(next_state_scores.min()),
            "p05": float(np.quantile(next_state_scores, 0.05)),
            "p50": float(np.quantile(next_state_scores, 0.5)),
            "p95": float(np.quantile(next_state_scores, 0.95)),
            "max": float(next_state_scores.max()),
        },
        "potential_delta_stats": {
            "mean": float(potential_delta.mean()),
            "std": float(potential_delta.std()),
            "min": float(potential_delta.min()),
            "p05": float(np.quantile(potential_delta, 0.05)),
            "p50": float(np.quantile(potential_delta, 0.5)),
            "p95": float(np.quantile(potential_delta, 0.95)),
            "max": float(potential_delta.max()),
        },
        "potential_shaping_reward_stats": {
            "mean": float(potential_shaping_reward.mean()),
            "std": float(potential_shaping_reward.std()),
            "min": float(potential_shaping_reward.min()),
            "p05": float(np.quantile(potential_shaping_reward, 0.05)),
            "p50": float(np.quantile(potential_shaping_reward, 0.5)),
            "p95": float(np.quantile(potential_shaping_reward, 0.95)),
            "max": float(potential_shaping_reward.max()),
        },
        "positive_action_risk_stats": {
            "mean": float(positive_action_risk.mean()),
            "std": float(positive_action_risk.std()),
            "min": float(positive_action_risk.min()),
            "p50": float(np.quantile(positive_action_risk, 0.5)),
            "p95": float(np.quantile(positive_action_risk, 0.95)),
            "max": float(positive_action_risk.max()),
        },
        "failure_only_stats": {
            "num_failure_samples": int(np.count_nonzero(failure_risk_mask)),
            "failure_reward_mean": float(relabeled_reward[failure_risk_mask > 0].mean()) if np.any(failure_risk_mask > 0) else None,
            "failure_reward_std": float(relabeled_reward[failure_risk_mask > 0].std()) if np.any(failure_risk_mask > 0) else None,
            "failure_risk_penalty_mean": float(risk_penalty[failure_risk_mask > 0].mean()) if np.any(failure_risk_mask > 0) else None,
            "failure_risk_penalty_std": float(risk_penalty[failure_risk_mask > 0].std()) if np.any(failure_risk_mask > 0) else None,
            "failure_positive_reward_fraction": float(np.mean(relabeled_reward[failure_risk_mask > 0] > 0.0)) if np.any(failure_risk_mask > 0) else None,
            "failure_negative_reward_fraction": float(np.mean(relabeled_reward[failure_risk_mask > 0] < 0.0)) if np.any(failure_risk_mask > 0) else None,
            "failure_positive_penalty_fraction": float(np.mean(risk_penalty[failure_risk_mask > 0] > 0.0)) if np.any(failure_risk_mask > 0) else None,
            "failure_negative_penalty_fraction": float(np.mean(risk_penalty[failure_risk_mask > 0] < 0.0)) if np.any(failure_risk_mask > 0) else None,
            "failure_potential_delta_mean": float(potential_delta[failure_risk_mask > 0].mean()) if np.any(failure_risk_mask > 0) else None,
            "failure_potential_delta_std": float(potential_delta[failure_risk_mask > 0].std()) if np.any(failure_risk_mask > 0) else None,
            "failure_positive_shaping_fraction": float(np.mean(potential_shaping_reward[failure_risk_mask > 0] > 0.0)) if np.any(failure_risk_mask > 0) else None,
            "failure_negative_shaping_fraction": float(np.mean(potential_shaping_reward[failure_risk_mask > 0] < 0.0)) if np.any(failure_risk_mask > 0) else None,
        },
        "by_source": source_stats(
            relabeled_reward,
            positive_action_risk,
            source,
            episode_success,
            default_rewards=default_rewards,
            risk_cost=risk_cost,
            action_deltas=action_deltas,
            signed_risk_advantage=signed_risk_advantage,
            risk_penalty=risk_penalty,
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
