#!/usr/bin/env python3
"""Train a chunk-level RGB hazard model from rollout-level outcomes.

Each rollout is a multiple-instance bag. Successful bags supervise every
chunk as non-hazardous. Failed bags only state that at least one chunk is
hazardous; top-k MIL pooling identifies candidate failure regions. The policy
encoder is frozen and only a small observation-action hazard head is trained.

Privileged failed-grasp and safe-reach labels are never used for optimization.
They are loaded only after training to measure whether high MIL scores localize
physical failure events instead of exploiting rollout-level shortcuts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUTS = (
    ROOT / "rollouts/rgb_dp/epoch25_collection/lift_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_DP_CHECKPOINT = (
    ROOT
    / "trained_models/rgb_dp_segment_posttrain"
    / "lift_rgb2_dp_baseline_s1/20260627122714/models/model_epoch_25.pth"
)
DEFAULT_CRITICAL_SUMMARY = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_critical_failure_chunks.summary.json"
)
DEFAULT_SAFE_SUMMARY = (
    ROOT
    / "rollouts/rgb_dp/epoch25_collection"
    / "lift_rgb_dp_good_chunks_fixed_window.summary.json"
)
DEFAULT_OUTPUT = ROOT / "trained_models/rgb_dp_hazard_mil"
DEFAULT_FEATURES = ROOT / "rollouts/rgb_dp/hazard_mil/chunk_features.npz"

OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def sorted_demo_keys(dataset: h5py.File) -> list[str]:
    return sorted(dataset["data"].keys(), key=lambda x: int(x.split("_")[-1]))


def load_boundary_labels(path: Path, field: str = "decision_boundary"):
    labels = defaultdict(set)
    if not path.exists():
        return labels
    summary = json.loads(path.read_text())
    records = summary.get("chunks", summary.get("segments", []))
    for record in records:
        if field in record:
            labels[record["source_demo"]].add(int(record[field]))
    return labels


def resolve_checkpoint_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


@torch.no_grad()
def extract_features(args) -> None:
    import robomimic.models.obs_nets as ObsNets
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.tensor_utils as TensorUtils
    from robomimic.algo.diffusion_policy import replace_bn_with_gn

    device = torch.device(args.device)
    checkpoint = FileUtils.load_dict_from_checkpoint(str(args.dp_checkpoint))
    encoder_checkpoint = checkpoint
    encoder_source_kind = "robomimic_policy"
    encoder_metadata_checkpoint = args.dp_checkpoint
    if bool(checkpoint.get("hybrid_dp_chunk_actor_iql", False)):
        encoder_source_kind = "hybrid_dp_chunk_actor_iql"
        base_checkpoint_value = checkpoint.get(
            "pretrained_dp_checkpoint",
            checkpoint.get("args", {}).get("checkpoint"),
        )
        if base_checkpoint_value is None:
            raise RuntimeError(
                "hybrid DP chunk actor checkpoint is missing pretrained_dp_checkpoint"
            )
        base_checkpoint = resolve_checkpoint_path(base_checkpoint_value)
        encoder_metadata_checkpoint = base_checkpoint
        encoder_checkpoint = FileUtils.load_dict_from_checkpoint(str(base_checkpoint))
        actor_model = checkpoint.get("actor_model", {})
        policy_state = actor_model.get("ema", None) or actor_model.get("nets", None)
        if policy_state is None:
            raise RuntimeError("hybrid checkpoint contains no actor_model EMA/nets weights")
    else:
        model_state = checkpoint["model"]
        policy_state = model_state.get("ema", None) or model_state["nets"]

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
        (key, shape)
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
    prefix = "policy.obs_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in policy_state.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError("checkpoint contains no EMA observation-encoder weights")
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    critical_boundaries = load_boundary_labels(args.critical_summary)
    safe_boundaries = load_boundary_labels(args.safe_summary)
    q_action_horizon = int(
        args.q_action_horizon
        if args.q_action_horizon is not None and args.q_action_horizon > 0
        else args.prediction_horizon
    )
    if q_action_horizon > args.prediction_horizon:
        raise ValueError(
            f"q_action_horizon={q_action_horizon} cannot exceed "
            f"prediction_horizon={args.prediction_horizon}"
        )

    all_features = []
    all_actions = []
    all_steps = []
    all_episode_indices = []
    all_critical = []
    all_safe = []
    episode_keys = []
    episode_labels = []
    episode_offsets = [0]

    with h5py.File(args.rollouts, "r") as dataset:
        failure_keys = set(decode(dataset["mask/failure"][:]))
        demos = sorted_demo_keys(dataset)
        if args.max_episodes is not None:
            demos = demos[: args.max_episodes]

        for episode_index, demo_key in enumerate(demos):
            group = dataset[f"data/{demo_key}"]
            length = int(group.attrs["num_samples"])
            latest = length - q_action_horizon
            boundaries = np.arange(
                0, latest + 1, args.action_horizon, dtype=np.int64
            )
            if len(boundaries) == 0:
                continue
            previous = np.maximum(0, boundaries - 1)

            obs_batch = {}
            for key in OBS_KEYS:
                values = group[f"obs/{key}"]
                obs_batch[key] = np.stack(
                    [values[previous], values[boundaries]], axis=1
                )
            prepared = {}
            for key, value in obs_batch.items():
                tensor = torch.as_tensor(value, device=device).float()
                if key in ("agentview_image", "robot0_eye_in_hand_image"):
                    tensor = ObsUtils.process_obs(tensor, obs_key=key)
                prepared[key] = tensor

            episode_features = []
            for start in range(0, len(boundaries), args.encoder_batch_size):
                end = start + args.encoder_batch_size
                mini_inputs = {
                    "obs": {key: value[start:end] for key, value in prepared.items()},
                    "goal": None,
                }
                encoded = TensorUtils.time_distributed(
                    mini_inputs,
                    encoder,
                    inputs_as_kwargs=True,
                )
                episode_features.append(
                    encoded.flatten(start_dim=1).cpu().numpy().astype(np.float32)
                )
            features = np.concatenate(episode_features, axis=0)
            actions = np.stack(
                [
                    group["actions"][
                        boundary : boundary + q_action_horizon
                    ]
                    for boundary in boundaries
                ]
            ).astype(np.float32)

            all_features.append(features)
            all_actions.append(actions)
            all_steps.append(boundaries.astype(np.int32))
            all_episode_indices.append(
                np.full(len(boundaries), episode_index, dtype=np.int32)
            )
            all_critical.append(
                np.asarray(
                    [
                        int(step) in critical_boundaries.get(demo_key, set())
                        for step in boundaries
                    ],
                    dtype=np.bool_,
                )
            )
            all_safe.append(
                np.asarray(
                    [
                        int(step) in safe_boundaries.get(demo_key, set())
                        for step in boundaries
                    ],
                    dtype=np.bool_,
                )
            )
            episode_keys.append(demo_key)
            episode_labels.append(float(demo_key in failure_keys))
            episode_offsets.append(episode_offsets[-1] + len(boundaries))
            if (episode_index + 1) % 50 == 0:
                print(
                    f"encoded {episode_index + 1}/{len(demos)} episodes; "
                    f"{episode_offsets[-1]} chunks",
                    flush=True,
                )

    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        features=np.concatenate(all_features),
        actions=np.concatenate(all_actions),
        steps=np.concatenate(all_steps),
        episode_indices=np.concatenate(all_episode_indices),
        critical_labels=np.concatenate(all_critical),
        safe_labels=np.concatenate(all_safe),
        episode_keys=np.asarray(episode_keys),
        episode_labels=np.asarray(episode_labels, dtype=np.float32),
        episode_offsets=np.asarray(episode_offsets, dtype=np.int64),
        action_horizon=np.asarray(args.action_horizon),
        prediction_horizon=np.asarray(q_action_horizon),
        q_action_horizon=np.asarray(q_action_horizon),
        dp_prediction_horizon=np.asarray(args.prediction_horizon),
        dp_checkpoint=np.asarray(str(args.dp_checkpoint)),
        encoder_source_kind=np.asarray(encoder_source_kind),
        encoder_metadata_checkpoint=np.asarray(str(encoder_metadata_checkpoint)),
        rollout_path=np.asarray(str(args.rollouts)),
    )
    print(
        f"Wrote {args.features}: episodes={len(episode_keys)}, "
        f"chunks={episode_offsets[-1]}, feature_dim={all_features[0].shape[1]}, "
        f"q_action_horizon={q_action_horizon}, dp_prediction_horizon={args.prediction_horizon}",
        flush=True,
    )


class HazardHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        prediction_horizon: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.obs = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.action = nn.Sequential(
            nn.Linear(prediction_horizon * action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, actions):
        obs = self.obs(features)
        action = self.action(actions.flatten(start_dim=1))
        return self.fusion(torch.cat([obs, action], dim=-1)).squeeze(-1)


def stratified_split(labels: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, int(round(0.15 * len(indices))))
        n_val = max(1, int(round(0.15 * len(indices))))
        splits["test"].extend(indices[:n_test].tolist())
        splits["val"].extend(indices[n_test : n_test + n_val].tolist())
        splits["train"].extend(indices[n_test + n_val :].tolist())
    for key in splits:
        rng.shuffle(splits[key])
        splits[key] = np.asarray(splits[key], dtype=np.int64)
    return splits


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    n_pos = int(np.sum(positive))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels > 0.5
    positives = int(np.sum(labels))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def topk_pool(logits: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(logits, k=min(k, logits.numel())).values.mean()


def episode_chunk_indices(offsets: np.ndarray, episode: int) -> np.ndarray:
    return np.arange(offsets[episode], offsets[episode + 1], dtype=np.int64)


def balanced_episode_batch(
    episode_labels: np.ndarray,
    train_episodes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    failures = train_episodes[episode_labels[train_episodes] > 0.5]
    successes = train_episodes[episode_labels[train_episodes] < 0.5]
    n_failure = batch_size // 2
    n_success = batch_size - n_failure
    selected = np.concatenate(
        [
            rng.choice(failures, size=n_failure, replace=len(failures) < n_failure),
            rng.choice(successes, size=n_success, replace=len(successes) < n_success),
        ]
    )
    rng.shuffle(selected)
    return selected


def compute_batch_loss(
    model,
    features,
    actions,
    episode_labels,
    offsets,
    episodes,
    args,
    device,
):
    index_arrays = [episode_chunk_indices(offsets, int(ep)) for ep in episodes]
    flat_indices = np.concatenate(index_arrays)
    tensor_indices = torch.as_tensor(flat_indices, device=device)
    logits = model(features[tensor_indices], actions[tensor_indices])

    bag_logits = []
    cursor = 0
    success_instance_losses = []
    failure_sparsity = []
    smoothness = []
    for episode, indices in zip(episodes, index_arrays):
        local = logits[cursor : cursor + len(indices)]
        cursor += len(indices)
        bag_logits.append(topk_pool(local, args.top_k))
        probability = torch.sigmoid(local)
        if episode_labels[int(episode)] < 0.5:
            success_instance_losses.append(
                F.binary_cross_entropy_with_logits(
                    local, torch.zeros_like(local)
                )
            )
        else:
            failure_sparsity.append(probability.mean())
        if len(local) > 1:
            smoothness.append(torch.abs(probability[1:] - probability[:-1]).mean())

    bag_logits = torch.stack(bag_logits)
    bag_targets = torch.as_tensor(
        episode_labels[episodes], dtype=torch.float32, device=device
    )
    bag_loss = F.binary_cross_entropy_with_logits(bag_logits, bag_targets)
    zero = bag_loss * 0.0
    success_loss = (
        torch.stack(success_instance_losses).mean()
        if success_instance_losses
        else zero
    )
    sparsity_loss = (
        torch.stack(failure_sparsity).mean() if failure_sparsity else zero
    )
    smoothness_loss = torch.stack(smoothness).mean() if smoothness else zero
    total = (
        bag_loss
        + args.success_instance_weight * success_loss
        + args.failure_sparsity_weight * sparsity_loss
        + args.smoothness_weight * smoothness_loss
    )
    return total, {
        "bag_loss": float(bag_loss.detach()),
        "success_instance_loss": float(success_loss.detach()),
        "failure_sparsity": float(sparsity_loss.detach()),
        "smoothness": float(smoothness_loss.detach()),
    }


@torch.no_grad()
def predict_chunks(model, features, actions, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start : start + batch_size], actions[start : start + batch_size])
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def episode_metrics(scores, labels, offsets, episodes, top_k):
    bag_scores = []
    targets = []
    for episode in episodes:
        local = scores[offsets[episode] : offsets[episode + 1]]
        k = min(top_k, len(local))
        local_logits = np.log(
            np.clip(local, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - local, 1e-6, 1.0)
        )
        pooled_logit = float(np.mean(np.partition(local_logits, -k)[-k:]))
        bag_scores.append(1.0 / (1.0 + math.exp(-pooled_logit)))
        targets.append(float(labels[episode]))
    bag_scores = np.asarray(bag_scores)
    targets = np.asarray(targets)
    predictions = bag_scores >= 0.5
    clipped = np.clip(bag_scores, 1e-6, 1.0 - 1e-6)
    binary_cross_entropy = -np.mean(
        targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)
    )
    return {
        "num_episodes": len(episodes),
        "num_failures": int(np.sum(targets)),
        "roc_auc": rank_auc(targets, bag_scores),
        "average_precision": average_precision(targets, bag_scores),
        "binary_cross_entropy": float(binary_cross_entropy),
        "accuracy_at_0.5": float(np.mean(predictions == (targets > 0.5))),
        "failure_score_mean": float(np.mean(bag_scores[targets > 0.5])),
        "success_score_mean": float(np.mean(bag_scores[targets < 0.5])),
    }


def localization_metrics(
    scores,
    steps,
    critical,
    safe,
    labels,
    offsets,
    episodes,
    action_horizon,
):
    exact_top1 = []
    near_top1 = []
    top3 = []
    reciprocal_ranks = []
    localized_episodes = 0
    for episode in episodes:
        if labels[episode] < 0.5:
            continue
        sl = slice(offsets[episode], offsets[episode + 1])
        local_critical = np.flatnonzero(critical[sl])
        if len(local_critical) == 0:
            continue
        localized_episodes += 1
        local_scores = scores[sl]
        ranking = np.argsort(-local_scores)
        top = int(ranking[0])
        exact_top1.append(top in set(local_critical.tolist()))
        near_top1.append(
            np.min(np.abs(steps[sl][top] - steps[sl][local_critical]))
            <= action_horizon
        )
        top3.append(bool(set(ranking[:3].tolist()) & set(local_critical.tolist())))
        first_rank = min(int(np.where(ranking == target)[0][0]) for target in local_critical)
        reciprocal_ranks.append(1.0 / (first_rank + 1))

    selected_chunks = np.zeros(len(scores), dtype=bool)
    for episode in episodes:
        selected_chunks[offsets[episode] : offsets[episode + 1]] = True
    critical_scores = scores[critical & selected_chunks]
    safe_scores = scores[safe & selected_chunks]
    instance_labels = np.concatenate(
        [np.ones(len(critical_scores)), np.zeros(len(safe_scores))]
    )
    instance_scores = np.concatenate([critical_scores, safe_scores])
    return {
        "episodes_with_privileged_critical_label": localized_episodes,
        "exact_top1_hit_rate": float(np.mean(exact_top1)) if exact_top1 else None,
        "within_one_boundary_top1_hit_rate": (
            float(np.mean(near_top1)) if near_top1 else None
        ),
        "top3_hit_rate": float(np.mean(top3)) if top3 else None,
        "mean_reciprocal_rank": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        ),
        "num_critical_chunks": int(len(critical_scores)),
        "num_safe_chunks": int(len(safe_scores)),
        "critical_score_mean": (
            float(np.mean(critical_scores)) if len(critical_scores) else None
        ),
        "safe_score_mean": float(np.mean(safe_scores)) if len(safe_scores) else None,
        "critical_vs_safe_roc_auc": (
            rank_auc(instance_labels, instance_scores)
            if len(critical_scores) and len(safe_scores)
            else None
        ),
    }


def threshold_localization_metrics(
    scores,
    critical,
    labels,
    offsets,
    episodes,
    threshold,
):
    selected_chunks = np.zeros(len(scores), dtype=bool)
    success_chunks = np.zeros(len(scores), dtype=bool)
    failure_chunks = np.zeros(len(scores), dtype=bool)
    for episode in episodes:
        sl = slice(offsets[episode], offsets[episode + 1])
        selected_chunks[sl] = True
        if labels[episode] < 0.5:
            success_chunks[sl] = True
        else:
            failure_chunks[sl] = True
    critical_selected = critical & selected_chunks
    return {
        "threshold": float(threshold),
        "num_critical_chunks": int(np.sum(critical_selected)),
        "critical_recall": (
            float(np.mean(scores[critical_selected] >= threshold))
            if np.any(critical_selected)
            else None
        ),
        "success_chunk_false_positive_rate": (
            float(np.mean(scores[success_chunks] >= threshold))
            if np.any(success_chunks)
            else None
        ),
        "failure_chunk_flagged_fraction": (
            float(np.mean(scores[failure_chunks] >= threshold))
            if np.any(failure_chunks)
            else None
        ),
    }


def train(args) -> dict:
    raw = np.load(args.features)
    features_np = raw["features"].astype(np.float32)
    actions_np = raw["actions"].astype(np.float32)
    steps = raw["steps"]
    labels = raw["episode_labels"].astype(np.float32)
    offsets = raw["episode_offsets"]
    critical = raw["critical_labels"].astype(bool)
    safe = raw["safe_labels"].astype(bool)
    episode_keys = raw["episode_keys"]
    splits = stratified_split(labels, args.seed)

    train_chunk_indices = np.concatenate(
        [episode_chunk_indices(offsets, int(ep)) for ep in splits["train"]]
    )
    feature_mean = features_np[train_chunk_indices].mean(axis=0)
    feature_std = features_np[train_chunk_indices].std(axis=0)
    feature_std = np.maximum(feature_std, 1e-4)
    features_np = (features_np - feature_mean) / feature_std

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    features = torch.from_numpy(features_np).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    model = HazardHead(
        feature_dim=features.shape[1],
        prediction_horizon=actions.shape[1],
        action_dim=actions.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps
    )
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    best_quality = (-float("inf"), -float("inf"))
    history = []
    for step in range(1, args.total_steps + 1):
        model.train()
        episodes = balanced_episode_batch(
            labels, splits["train"], args.bag_batch_size, rng
        )
        optimizer.zero_grad(set_to_none=True)
        loss, components = compute_batch_loss(
            model,
            features,
            actions,
            labels,
            offsets,
            episodes,
            args,
            device,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.total_steps:
            model.eval()
            scores = predict_chunks(model, features, actions, args.score_batch_size)
            val = episode_metrics(
                scores, labels, offsets, splits["val"], args.top_k
            )
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                **components,
                "val": val,
            }
            history.append(record)
            print(json.dumps(record, indent=2), flush=True)
            val_auc = val["roc_auc"]
            quality = (val_auc, -val["binary_cross_entropy"])
            if math.isfinite(val_auc) and quality > best_quality:
                best_quality = quality
                torch.save(
                    {
                        "model": model.state_dict(),
                        "feature_mean": feature_mean,
                        "feature_std": feature_std,
                        "splits": splits,
                        "args": vars(args),
                        "feature_dim": int(features.shape[1]),
                        "prediction_horizon": int(actions.shape[1]),
                        "action_dim": int(actions.shape[2]),
                        "best_step": step,
                        "best_val_auc": val_auc,
                        "best_val_binary_cross_entropy": val[
                            "binary_cross_entropy"
                        ],
                    },
                    best_path,
                )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scores = predict_chunks(model, features, actions, args.score_batch_size)
    metrics = {
        split: episode_metrics(scores, labels, offsets, episodes, args.top_k)
        for split, episodes in splits.items()
    }
    metrics["privileged_localization_test"] = localization_metrics(
        scores,
        steps,
        critical,
        safe,
        labels,
        offsets,
        splits["test"],
        args.action_horizon,
    )
    metrics["privileged_localization_all"] = localization_metrics(
        scores,
        steps,
        critical,
        safe,
        labels,
        offsets,
        np.arange(len(labels)),
        args.action_horizon,
    )

    success_val_chunks = np.concatenate(
        [
            episode_chunk_indices(offsets, int(ep))
            for ep in splits["val"]
            if labels[int(ep)] < 0.5
        ]
    )
    low_hazard_threshold = float(
        np.quantile(scores[success_val_chunks], args.success_score_quantile)
    )
    metrics["threshold_localization_test"] = threshold_localization_metrics(
        scores,
        critical,
        labels,
        offsets,
        splits["test"],
        low_hazard_threshold,
    )
    metrics["threshold_localization_all"] = threshold_localization_metrics(
        scores,
        critical,
        labels,
        offsets,
        np.arange(len(labels)),
        low_hazard_threshold,
    )
    predictions_path = args.output_dir / "chunk_predictions.npz"
    np.savez_compressed(
        predictions_path,
        scores=scores,
        steps=steps,
        episode_indices=raw["episode_indices"],
        episode_keys=episode_keys,
        episode_labels=labels,
        episode_offsets=offsets,
        critical_labels=critical,
        safe_labels=safe,
        low_hazard_threshold=np.asarray(low_hazard_threshold),
    )
    summary = {
        "features": str(args.features),
        "checkpoint": str(best_path),
        "predictions": str(predictions_path),
        "num_episodes": len(labels),
        "num_failure_episodes": int(np.sum(labels)),
        "num_chunks": len(scores),
        "best_step": checkpoint["best_step"],
        "best_val_auc": checkpoint["best_val_auc"],
        "low_hazard_threshold_from_success_validation_quantile": {
            "quantile": args.success_score_quantile,
            "threshold": low_hazard_threshold,
        },
        "metrics": metrics,
        "history": history,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    print(f"Wrote {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--dp-checkpoint", type=Path, default=DEFAULT_DP_CHECKPOINT)
    parser.add_argument("--critical-summary", type=Path, default=DEFAULT_CRITICAL_SUMMARY)
    parser.add_argument("--safe-summary", type=Path, default=DEFAULT_SAFE_SUMMARY)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument(
        "--q-action-horizon",
        type=int,
        default=None,
        help=(
            "Number of executable actions stored for Q/reward learning. "
            "Defaults to prediction_horizon for backward compatibility. "
            "Use 8 to score only the executed chunk while the DP still predicts 16."
        ),
    )
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--bag-batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--success-instance-weight", type=float, default=0.5)
    parser.add_argument("--failure-sparsity-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--score-batch-size", type=int, default=2048)
    parser.add_argument("--success-score-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    for name in (
        "rollouts",
        "dp_checkpoint",
        "critical_summary",
        "safe_summary",
        "features",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.rebuild_features or not args.features.exists():
        extract_features(args)
    if args.prepare_only:
        return
    train(args)


if __name__ == "__main__":
    main()
