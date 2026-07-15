#!/usr/bin/env python3
"""Build transition-complete RGB-DP features for successor-critic training.

Unlike the legacy hazard cache, this cache includes every policy decision
boundary. The final partial chunk is padded but accompanied by an explicit
valid-action mask, so sparse terminal success rewards are not silently dropped.
"""

from __future__ import annotations

import argparse
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


@torch.no_grad()
def build_features(args) -> None:
    import robomimic.models.obs_nets as ObsNets
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.tensor_utils as TensorUtils
    from robomimic.algo.diffusion_policy import replace_bn_with_gn

    if args.q_action_horizon <= 0:
        raise ValueError("--q-action-horizon must be positive")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.q_action_horizon % args.action_horizon != 0:
        raise ValueError(
            "q-action-horizon must be a multiple of action-horizon so the "
            "successor is another policy decision boundary"
        )

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
                "hybrid DP actor checkpoint is missing pretrained_dp_checkpoint"
            )
        base_checkpoint = resolve_checkpoint_path(base_checkpoint_value)
        encoder_metadata_checkpoint = base_checkpoint
        encoder_checkpoint = FileUtils.load_dict_from_checkpoint(str(base_checkpoint))
        actor_model = checkpoint.get("actor_model", {})
        policy_state = actor_model.get("ema", None) or actor_model.get("nets", None)
        if policy_state is None:
            raise RuntimeError("hybrid checkpoint contains no actor EMA/nets weights")
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

    def encode_indices(group: h5py.Group, indices: np.ndarray) -> np.ndarray:
        previous = np.maximum(0, indices - 1)

        def read_ordered(values: h5py.Dataset, requested: np.ndarray) -> np.ndarray:
            # h5py requires increasing, duplicate-free fancy indices. Terminal
            # next states can repeat, so read unique rows and restore order.
            unique, inverse = np.unique(requested, return_inverse=True)
            return values[unique][inverse]

        prepared = {}
        for key in OBS_KEYS:
            values = group[f"obs/{key}"]
            pair = np.stack(
                [read_ordered(values, previous), read_ordered(values, indices)],
                axis=1,
            )
            tensor = torch.as_tensor(pair, device=device).float()
            if key in ("agentview_image", "robot0_eye_in_hand_image"):
                tensor = ObsUtils.process_obs(tensor, obs_key=key)
            prepared[key] = tensor

        encoded_batches = []
        for start in range(0, len(indices), args.encoder_batch_size):
            end = min(start + args.encoder_batch_size, len(indices))
            mini_inputs = {
                "obs": {key: value[start:end] for key, value in prepared.items()},
                "goal": None,
            }
            encoded = TensorUtils.time_distributed(
                mini_inputs,
                encoder,
                inputs_as_kwargs=True,
            )
            encoded_batches.append(
                encoded.flatten(start_dim=1).cpu().numpy().astype(np.float32)
            )
        return np.concatenate(encoded_batches, axis=0)

    all_features = []
    all_next_features = []
    all_actions = []
    all_action_masks = []
    all_rewards = []
    all_terminals = []
    all_steps = []
    all_episode_indices = []
    episode_keys = []
    episode_labels = []
    episode_offsets = [0]

    with h5py.File(args.rollouts, "r") as dataset:
        failure_keys = set(decode(dataset["mask/failure"][:]))
        demos = sorted_demo_keys(dataset)
        if args.max_episodes is not None:
            demos = demos[: args.max_episodes]

        for source_episode_index, demo_key in enumerate(demos):
            group = dataset[f"data/{demo_key}"]
            length = int(group.attrs["num_samples"])
            if length <= 0:
                continue
            boundaries = np.arange(0, length, args.action_horizon, dtype=np.int64)
            endpoints = np.minimum(boundaries + args.q_action_horizon, length)
            next_indices = np.minimum(endpoints, length - 1)

            features = encode_indices(group, boundaries)
            next_features = encode_indices(group, next_indices)
            raw_actions = group["actions"]
            raw_rewards = group["rewards"]
            raw_dones = group["dones"] if "dones" in group else None
            action_dim = int(raw_actions.shape[-1])

            actions = np.zeros(
                (len(boundaries), args.q_action_horizon, action_dim),
                dtype=np.float32,
            )
            action_mask = np.zeros(
                (len(boundaries), args.q_action_horizon),
                dtype=np.bool_,
            )
            rewards = np.zeros(
                (len(boundaries), args.q_action_horizon),
                dtype=np.float32,
            )
            terminals = np.zeros(len(boundaries), dtype=np.bool_)

            for row, (boundary, endpoint) in enumerate(zip(boundaries, endpoints)):
                valid = int(endpoint - boundary)
                if valid <= 0:
                    raise RuntimeError(f"empty action chunk in {demo_key} at {boundary}")
                chunk_actions = raw_actions[boundary:endpoint].astype(np.float32)
                actions[row, :valid] = chunk_actions
                actions[row, valid:] = chunk_actions[-1]
                action_mask[row, :valid] = True
                rewards[row, :valid] = raw_rewards[boundary:endpoint].astype(np.float32)
                terminal = endpoint >= length
                if raw_dones is not None:
                    terminal = terminal or bool(np.any(raw_dones[boundary:endpoint]))
                terminals[row] = terminal

            episode_index = len(episode_keys)
            all_features.append(features)
            all_next_features.append(next_features)
            all_actions.append(actions)
            all_action_masks.append(action_mask)
            all_rewards.append(rewards)
            all_terminals.append(terminals)
            all_steps.append(boundaries.astype(np.int32))
            all_episode_indices.append(
                np.full(len(boundaries), episode_index, dtype=np.int32)
            )
            episode_keys.append(demo_key)
            episode_labels.append(float(demo_key not in failure_keys))
            episode_offsets.append(episode_offsets[-1] + len(boundaries))

            if (source_episode_index + 1) % 50 == 0:
                print(
                    f"encoded {source_episode_index + 1}/{len(demos)} episodes; "
                    f"{episode_offsets[-1]} transitions",
                    flush=True,
                )

    if not all_features:
        raise RuntimeError("no transitions were extracted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(all_features),
        next_features=np.concatenate(all_next_features),
        actions=np.concatenate(all_actions),
        action_valid_mask=np.concatenate(all_action_masks),
        chunk_rewards=np.concatenate(all_rewards),
        terminals=np.concatenate(all_terminals),
        steps=np.concatenate(all_steps),
        episode_indices=np.concatenate(all_episode_indices),
        episode_keys=np.asarray(episode_keys),
        episode_success_labels=np.asarray(episode_labels, dtype=np.float32),
        episode_offsets=np.asarray(episode_offsets, dtype=np.int64),
        action_horizon=np.asarray(args.action_horizon),
        q_action_horizon=np.asarray(args.q_action_horizon),
        prediction_horizon=np.asarray(args.q_action_horizon),
        dp_prediction_horizon=np.asarray(args.dp_prediction_horizon),
        q_boundary_stride=np.asarray(args.q_action_horizon // args.action_horizon),
        transition_complete=np.asarray(True),
        terminal_reward_semantics=np.asarray("robomimic_sparse_success"),
        dp_checkpoint=np.asarray(str(args.dp_checkpoint)),
        encoder_source_kind=np.asarray(encoder_source_kind),
        encoder_metadata_checkpoint=np.asarray(str(encoder_metadata_checkpoint)),
        rollout_path=np.asarray(str(args.rollouts)),
    )

    success_labels = np.asarray(episode_labels, dtype=np.float32)
    reward_rows = np.concatenate(all_rewards).sum(axis=1) > 0
    terminal_rows = np.concatenate(all_terminals)
    print(
        f"Wrote {args.output}: episodes={len(episode_keys)}, "
        f"success={int(success_labels.sum())}, failure={int(len(success_labels)-success_labels.sum())}, "
        f"transitions={episode_offsets[-1]}, terminal_rows={int(terminal_rows.sum())}, "
        f"positive_reward_rows={int(reward_rows.sum())}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--q-action-horizon", type=int, default=8)
    parser.add_argument("--dp-prediction-horizon", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    args.rollouts = args.rollouts.resolve()
    args.dp_checkpoint = args.dp_checkpoint.resolve()
    args.output = args.output.resolve()
    build_features(args)


if __name__ == "__main__":
    main()
