#!/usr/bin/env python3
"""Build Square RGB-DP prefix features from human demos + failed rollouts.

The output is drop-in compatible with ``train_rgb_dp_causal_prefix_risk.py``.
It stores raw failure labels:

* human demo episodes -> episode_labels = 0.0
* deployed failure rollouts -> episode_labels = 1.0

Therefore, for a success/reward model, train with ``--target-outcome success``
so labels are flipped internally to demo/success=1 and failure=0.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

from train_rgb_dp_hazard_mil import (
    OBS_KEYS,
    decode,
    resolve_checkpoint_path,
    sorted_demo_keys,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMOS = ROOT / "datasets/square/ph/image_v15.hdf5"
DEFAULT_FAILURE_ROLLOUTS = (
    ROOT / "rollouts/square_rgb_dp/epoch190_collection/square_rgb_dp_rollouts_rgb2.hdf5"
)
DEFAULT_DP_CHECKPOINT = (
    ROOT
    / "trained_models/square_rgb_dp_idql_visual"
    / "default_reward_dp_chunk_actor_iql/best_success_auc.pt"
)
DEFAULT_FEATURES = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection/risk_model"
    / "chunk_features_200demo_94fail_iql_actor.npz"
)


def maybe_limit(keys: list[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0:
        return keys
    return keys[:limit]


def load_key_list(dataset: h5py.File, mask: str | None, limit: int | None) -> list[str]:
    if mask is None or mask == "all":
        keys = sorted_demo_keys(dataset)
    else:
        if "mask" not in dataset or mask not in dataset["mask"]:
            raise KeyError(f"mask/{mask} not found in {dataset.filename}")
        keys = decode(dataset[f"mask/{mask}"][:])
        key_set = set(dataset["data"].keys())
        missing = [key for key in keys if key not in key_set]
        if missing:
            raise KeyError(f"{len(missing)} mask keys are absent from data, e.g. {missing[:5]}")
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
    return maybe_limit(keys, limit)


def load_encoder(dp_checkpoint: Path, device: torch.device):
    import robomimic.models.obs_nets as ObsNets
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.file_utils as FileUtils
    from robomimic.algo.diffusion_policy import replace_bn_with_gn

    checkpoint = FileUtils.load_dict_from_checkpoint(str(dp_checkpoint))
    encoder_checkpoint = checkpoint
    encoder_source_kind = "robomimic_policy"
    encoder_metadata_checkpoint = dp_checkpoint
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
    return encoder, ObsUtils, encoder_source_kind, encoder_metadata_checkpoint


@torch.no_grad()
def encode_episode(
    *,
    group: h5py.Group,
    encoder,
    ObsUtils,
    device: torch.device,
    action_horizon: int,
    q_action_horizon: int,
    encoder_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = int(group.attrs.get("num_samples", len(group["actions"])))
    latest = length - q_action_horizon
    boundaries = np.arange(0, latest + 1, action_horizon, dtype=np.int64)
    if len(boundaries) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, q_action_horizon, group["actions"].shape[-1]), dtype=np.float32),
            boundaries.astype(np.int32),
        )
    previous = np.maximum(0, boundaries - 1)
    obs_batch = {}
    for key in OBS_KEYS:
        if f"obs/{key}" not in group:
            raise KeyError(f"{group.name} is missing obs/{key}")
        values = group[f"obs/{key}"]
        obs_batch[key] = np.stack([values[previous], values[boundaries]], axis=1)

    prepared = {}
    for key, value in obs_batch.items():
        tensor = torch.as_tensor(value, device=device).float()
        if key in ("agentview_image", "robot0_eye_in_hand_image"):
            tensor = ObsUtils.process_obs(tensor, obs_key=key)
        prepared[key] = tensor

    import robomimic.utils.tensor_utils as TensorUtils

    episode_features = []
    for start in range(0, len(boundaries), encoder_batch_size):
        end = start + encoder_batch_size
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
            group["actions"][boundary : boundary + q_action_horizon]
            for boundary in boundaries
        ]
    ).astype(np.float32)
    return features, actions, boundaries.astype(np.int32)


def append_source(
    *,
    path: Path,
    keys: list[str],
    source_name: str,
    failure_label: float,
    encoder,
    ObsUtils,
    device: torch.device,
    args,
    all_features: list[np.ndarray],
    all_actions: list[np.ndarray],
    all_steps: list[np.ndarray],
    all_episode_indices: list[np.ndarray],
    all_critical: list[np.ndarray],
    all_safe: list[np.ndarray],
    episode_keys: list[str],
    episode_source_keys: list[str],
    episode_sources: list[str],
    episode_labels: list[float],
    episode_offsets: list[int],
) -> dict:
    retained = 0
    skipped_short = 0
    chunks = 0
    with h5py.File(path, "r") as dataset:
        for local_index, demo_key in enumerate(keys):
            group = dataset[f"data/{demo_key}"]
            features, actions, boundaries = encode_episode(
                group=group,
                encoder=encoder,
                ObsUtils=ObsUtils,
                device=device,
                action_horizon=args.action_horizon,
                q_action_horizon=args.q_action_horizon,
                encoder_batch_size=args.encoder_batch_size,
            )
            if len(boundaries) == 0:
                skipped_short += 1
                continue
            episode_index = len(episode_keys)
            all_features.append(features)
            all_actions.append(actions)
            all_steps.append(boundaries)
            all_episode_indices.append(
                np.full(len(boundaries), episode_index, dtype=np.int32)
            )
            all_critical.append(np.zeros(len(boundaries), dtype=np.bool_))
            all_safe.append(np.zeros(len(boundaries), dtype=np.bool_))
            episode_keys.append(f"{source_name}:{demo_key}")
            episode_source_keys.append(demo_key)
            episode_sources.append(source_name)
            episode_labels.append(float(failure_label))
            episode_offsets.append(episode_offsets[-1] + len(boundaries))
            retained += 1
            chunks += len(boundaries)
            if retained % args.log_every == 0:
                print(
                    f"encoded {source_name} {retained}/{len(keys)} episodes; "
                    f"total chunks={episode_offsets[-1]}",
                    flush=True,
                )
    return {
        "path": str(path),
        "source": source_name,
        "requested_keys": len(keys),
        "retained_episodes": retained,
        "skipped_short_episodes": skipped_short,
        "chunks": chunks,
        "failure_label": float(failure_label),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--failure-rollouts", type=Path, default=DEFAULT_FAILURE_ROLLOUTS)
    parser.add_argument("--dp-checkpoint", type=Path, default=DEFAULT_DP_CHECKPOINT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--demo-mask", default="all")
    parser.add_argument("--failure-mask", default="failure")
    parser.add_argument("--max-demos", type=int, default=200, help="<=0 means all selected demos")
    parser.add_argument("--max-failures", type=int, default=94, help="<=0 means all selected failures")
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
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.demos = args.demos.expanduser().resolve()
    args.failure_rollouts = args.failure_rollouts.expanduser().resolve()
    args.dp_checkpoint = args.dp_checkpoint.expanduser().resolve()
    args.features = args.features.expanduser().resolve()
    args.q_action_horizon = int(
        args.q_action_horizon
        if args.q_action_horizon is not None and args.q_action_horizon > 0
        else args.prediction_horizon
    )
    if args.q_action_horizon > args.prediction_horizon:
        raise ValueError(
            f"q_action_horizon={args.q_action_horizon} cannot exceed "
            f"prediction_horizon={args.prediction_horizon}"
        )
    if args.features.exists() and not args.force:
        raise FileExistsError(f"{args.features} exists; pass --force to overwrite")

    device = torch.device(args.device)
    encoder, ObsUtils, encoder_source_kind, encoder_metadata_checkpoint = load_encoder(
        args.dp_checkpoint,
        device,
    )

    with h5py.File(args.demos, "r") as dataset:
        demo_keys = load_key_list(
            dataset,
            None if args.demo_mask == "all" else args.demo_mask,
            args.max_demos,
        )
    with h5py.File(args.failure_rollouts, "r") as dataset:
        failure_keys = load_key_list(
            dataset,
            args.failure_mask,
            args.max_failures,
        )

    all_features: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_steps: list[np.ndarray] = []
    all_episode_indices: list[np.ndarray] = []
    all_critical: list[np.ndarray] = []
    all_safe: list[np.ndarray] = []
    episode_keys: list[str] = []
    episode_source_keys: list[str] = []
    episode_sources: list[str] = []
    episode_labels: list[float] = []
    episode_offsets: list[int] = [0]

    source_summaries = []
    source_summaries.append(
        append_source(
            path=args.demos,
            keys=demo_keys,
            source_name="human_demo",
            failure_label=0.0,
            encoder=encoder,
            ObsUtils=ObsUtils,
            device=device,
            args=args,
            all_features=all_features,
            all_actions=all_actions,
            all_steps=all_steps,
            all_episode_indices=all_episode_indices,
            all_critical=all_critical,
            all_safe=all_safe,
            episode_keys=episode_keys,
            episode_source_keys=episode_source_keys,
            episode_sources=episode_sources,
            episode_labels=episode_labels,
            episode_offsets=episode_offsets,
        )
    )
    source_summaries.append(
        append_source(
            path=args.failure_rollouts,
            keys=failure_keys,
            source_name="policy_failure",
            failure_label=1.0,
            encoder=encoder,
            ObsUtils=ObsUtils,
            device=device,
            args=args,
            all_features=all_features,
            all_actions=all_actions,
            all_steps=all_steps,
            all_episode_indices=all_episode_indices,
            all_critical=all_critical,
            all_safe=all_safe,
            episode_keys=episode_keys,
            episode_source_keys=episode_source_keys,
            episode_sources=episode_sources,
            episode_labels=episode_labels,
            episode_offsets=episode_offsets,
        )
    )

    if not all_features:
        raise RuntimeError("no episodes were encoded")
    labels = np.asarray(episode_labels, dtype=np.float32)
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
        episode_source_keys=np.asarray(episode_source_keys),
        episode_sources=np.asarray(episode_sources),
        episode_labels=labels,
        episode_offsets=np.asarray(episode_offsets, dtype=np.int64),
        action_horizon=np.asarray(args.action_horizon),
        prediction_horizon=np.asarray(args.q_action_horizon),
        q_action_horizon=np.asarray(args.q_action_horizon),
        dp_prediction_horizon=np.asarray(args.prediction_horizon),
        dp_checkpoint=np.asarray(str(args.dp_checkpoint)),
        encoder_source_kind=np.asarray(encoder_source_kind),
        encoder_metadata_checkpoint=np.asarray(str(encoder_metadata_checkpoint)),
        rollout_path=np.asarray(str(args.failure_rollouts)),
        demo_path=np.asarray(str(args.demos)),
        failure_rollout_path=np.asarray(str(args.failure_rollouts)),
        source_summaries_json=np.asarray(json.dumps(source_summaries)),
    )
    summary = {
        "features": str(args.features),
        "dp_checkpoint": str(args.dp_checkpoint),
        "encoder_source_kind": encoder_source_kind,
        "encoder_metadata_checkpoint": str(encoder_metadata_checkpoint),
        "num_episodes": len(episode_keys),
        "num_human_demos": int(np.sum(np.asarray(episode_sources) == "human_demo")),
        "num_policy_failures": int(np.sum(np.asarray(episode_sources) == "policy_failure")),
        "num_failure_label_episodes": int(np.sum(labels > 0.5)),
        "num_success_label_episodes": int(np.sum(labels < 0.5)),
        "num_chunks": int(episode_offsets[-1]),
        "feature_dim": int(all_features[0].shape[1]),
        "action_horizon": int(args.action_horizon),
        "prediction_horizon": int(args.q_action_horizon),
        "q_action_horizon": int(args.q_action_horizon),
        "dp_prediction_horizon": int(args.prediction_horizon),
        "source_summaries": source_summaries,
        "label_semantics": {
            "episode_labels": "raw failure label used by train_rgb_dp_causal_prefix_risk.py",
            "human_demo": 0.0,
            "policy_failure": 1.0,
            "train_success_model_with": "--target-outcome success",
        },
    }
    summary_path = args.features.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.features}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
