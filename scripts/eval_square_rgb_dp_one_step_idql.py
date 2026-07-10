#!/usr/bin/env python3
"""Closed-loop evaluation for paper-faithful one-step IDQL on Square.

At every environment step this evaluator replans from the current observation.
The trained diffusion actor outputs a single-step action. With N=1, the action
is executed directly and the critic is not queried. With N>1, N single-step
actions are sampled, scored by Q(o_t, a_t), and the highest-scoring action is
executed for exactly one environment step.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

from train_square_rgb_dp_one_step_idql import ChunkIQLCritic, OneStepDiffusionActor, make_scheduler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDQL = (
    ROOT
    / "trained_models/square_rgb_dp_idql/default_reward_one_step_idql_no_rollout/best_success_auc.pt"
)
DEFAULT_OUTPUT = ROOT / "rollouts/square_rgb_dp/one_step_idql_eval"


def clone_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in obs.items()}


def repeat_obs(obs: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in obs.items():
        reps = [batch_size] + [1] * (value.ndim - 1)
        out[key] = value.repeat(*reps)
    return out


def obs_for_encoder(algo, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prepared = clone_obs(obs)
    for key in algo.obs_shapes:
        if prepared[key].ndim - 1 == len(algo.obs_shapes[key]):
            prepared[key] = prepared[key].unsqueeze(1)
    return prepared


@torch.no_grad()
def encode_current_obs(policy, prepared_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    algo = policy.policy
    nets = algo.nets
    if algo.ema is not None:
        nets = algo.ema.averaged_model
    obs = obs_for_encoder(algo, prepared_obs)
    features = algo._encode_obs({"obs": obs, "goal": None}, nets)
    if features.ndim == 3:
        features = features.flatten(start_dim=1)
    return features


def clone_policy_obs_encoder(policy):
    algo = policy.policy
    nets = algo.ema.averaged_model if algo.ema is not None else algo.nets
    return deepcopy(nets["policy"]["obs_encoder"])


@torch.no_grad()
def encode_current_obs_with_encoder(policy, obs_encoder, prepared_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    algo = policy.policy
    obs = obs_for_encoder(algo, prepared_obs)
    features = TensorUtils.time_distributed(
        {"obs": obs, "goal": None},
        obs_encoder,
        inputs_as_kwargs=True,
    )
    if features.ndim == 3:
        features = features.flatten(start_dim=1)
    return features


class OneStepIDQLPolicy:
    def __init__(
        self,
        dp_policy,
        critic: ChunkIQLCritic,
        actor: OneStepDiffusionActor,
        checkpoint: dict,
        *,
        actor_obs_encoder=None,
        critic_obs_encoder=None,
        num_candidates: int,
        candidate_batch_size: int,
        num_inference_steps: int,
        selection: str,
        softmax_temperature: float,
        clip_actions: bool,
        diffusion_clip_sample: bool,
    ):
        self.dp_policy = dp_policy
        self.algo = dp_policy.policy
        self.critic = critic
        self.actor = actor
        self.checkpoint = checkpoint
        self.actor_obs_encoder = actor_obs_encoder
        self.critic_obs_encoder = critic_obs_encoder
        if self.actor_obs_encoder is not None:
            self.actor_obs_encoder.eval().requires_grad_(False)
        if self.critic_obs_encoder is not None:
            self.critic_obs_encoder.eval().requires_grad_(False)
        self.num_candidates = int(num_candidates)
        self.candidate_batch_size = int(candidate_batch_size)
        self.num_inference_steps = int(num_inference_steps)
        self.selection = selection
        self.softmax_temperature = float(softmax_temperature)
        self.clip_actions = bool(clip_actions)
        if self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {self.num_candidates}")
        self.critic_used = self.num_candidates > 1
        self.execution_horizon = 1
        self.action_dim = int(checkpoint["action_dim"])
        self.normalize_actions = bool(checkpoint.get("normalize_actions", True))
        self.action_mean = torch.as_tensor(
            checkpoint["action_mean"], device=self.algo.device, dtype=torch.float32
        )
        self.action_std = torch.as_tensor(
            checkpoint["action_std"], device=self.algo.device, dtype=torch.float32
        )
        scheduler_args = dict(checkpoint["args"])
        scheduler_args["clip_sample"] = bool(diffusion_clip_sample)
        self.diffusion_clip_sample = bool(diffusion_clip_sample)
        self.scheduler = make_scheduler(scheduler_args)
        # Avoid diffusers.DDPMScheduler.set_timesteps() here. In this
        # environment it calls torch.from_numpy() and can occasionally crash
        # because of NumPy / Torch ABI issues. We reproduce the local diffusers
        # 0.11.1 DDPM timestep rule with pure torch once at policy creation.
        train_steps = int(self.scheduler.config.num_train_timesteps)
        self.num_inference_steps = min(train_steps, self.num_inference_steps)
        step_ratio = max(train_steps // self.num_inference_steps, 1)
        self.scheduler.num_inference_steps = self.num_inference_steps
        self.scheduler.timesteps = torch.arange(
            0, train_steps, step_ratio, device=self.algo.device, dtype=torch.long
        ).flip(0).contiguous()
        self.last_q: np.ndarray | None = None
        self.last_v: float | None = None
        self.last_adv: np.ndarray | None = None
        self.last_selected_index: int | None = None

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None

    @torch.no_grad()
    def sample_normalized_actions(self, obs_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        repeated = obs_features.repeat(batch_size, 1)
        sample = torch.randn(batch_size, self.action_dim, device=self.algo.device)
        for timestep in self.scheduler.timesteps:
            t = torch.full((batch_size,), int(timestep), device=self.algo.device, dtype=torch.long)
            noise_pred = self.actor(repeated, sample, t)
            sample = self.scheduler.step(noise_pred, timestep, sample).prev_sample
        return sample

    @torch.no_grad()
    def sample_candidates(self, obs_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = []
        for start in range(0, self.num_candidates, self.candidate_batch_size):
            count = min(self.candidate_batch_size, self.num_candidates - start)
            normalized.append(self.sample_normalized_actions(obs_features, count))
        norm_actions = torch.cat(normalized, dim=0)
        if self.normalize_actions:
            raw_actions = norm_actions * self.action_std[None, :] + self.action_mean[None, :]
        else:
            raw_actions = norm_actions
        if self.clip_actions:
            raw_actions = raw_actions.clamp(-1.0, 1.0)
            if self.normalize_actions:
                norm_actions = (raw_actions - self.action_mean[None, :]) / self.action_std[None, :]
            else:
                norm_actions = raw_actions
        return norm_actions, raw_actions

    def choose_index(self, q: torch.Tensor, v: torch.Tensor) -> int:
        if self.selection in ("argmax", "greedy") or len(q) == 1:
            return int(torch.argmax(q).item())
        if self.selection == "softmax":
            probs = torch.softmax(q / max(self.softmax_temperature, 1e-6), dim=0)
            return int(torch.multinomial(probs, num_samples=1).item())
        if self.selection == "advantage_softmax":
            adv = q - v
            probs = torch.softmax(adv / max(self.softmax_temperature, 1e-6), dim=0)
            return int(torch.multinomial(probs, num_samples=1).item())
        raise ValueError(f"unknown selection={self.selection}")

    def __call__(self, ob) -> np.ndarray:
        # One-step IDQL evaluation: every call corresponds to exactly one
        # environment action and the caller immediately executes env.step(action).
        prepared_obs = self.dp_policy._prepare_observation(ob, batched_ob=False)
        actor_feature = (
            encode_current_obs_with_encoder(self.dp_policy, self.actor_obs_encoder, prepared_obs)
            if self.actor_obs_encoder is not None
            else encode_current_obs(self.dp_policy, prepared_obs)
        )
        norm_actions, raw_actions = self.sample_candidates(actor_feature)

        if not self.critic_used:
            # N=1 is the trained diffusion actor only. No critic is queried and
            # no action selection happens, matching the actor-only ablation.
            self.last_q = None
            self.last_v = None
            self.last_adv = None
            self.last_selected_index = 0
            selected_action = raw_actions[0]
        else:
            critic_feature = (
                encode_current_obs_with_encoder(self.dp_policy, self.critic_obs_encoder, prepared_obs)
                if self.critic_obs_encoder is not None
                else actor_feature
            )
            obs_batch = critic_feature.repeat(norm_actions.shape[0], 1)
            critic_actions = norm_actions[:, None, :]
            q = self.critic.q_min(obs_batch, critic_actions).reshape(-1)
            v = self.critic.value(critic_feature).reshape(())
            selected = self.choose_index(q, v)
            self.last_q = q.detach().cpu().numpy()
            self.last_v = float(v.detach().cpu())
            self.last_adv = (q - v).detach().cpu().numpy()
            self.last_selected_index = int(selected)
            selected_action = raw_actions[selected]

        return selected_action.detach().cpu().numpy().astype(np.float64).copy()


class PretrainedDPFirstActionIDQLPolicy:
    """Ablation mode that uses pretrained DP as the action proposal actor.

    The actor sampler is the original RGB DiffusionPolicy checkpoint. It samples
    full DP trajectories, but this policy only exposes the first action as the
    one-step action a_t. With N=1 the first sampled action is executed directly.
    With N>1 the one-step critic scores Q(o_t, a_t) for the first action from
    each sampled trajectory and executes the argmax action for one env step.
    """

    def __init__(
        self,
        dp_policy,
        critic: ChunkIQLCritic,
        checkpoint: dict,
        *,
        critic_obs_encoder=None,
        num_candidates: int,
        candidate_batch_size: int,
        selection: str,
        softmax_temperature: float,
        clip_actions: bool,
    ):
        self.dp_policy = dp_policy
        self.algo = dp_policy.policy
        self.critic = critic
        self.checkpoint = checkpoint
        self.critic_obs_encoder = critic_obs_encoder
        if self.critic_obs_encoder is not None:
            self.critic_obs_encoder.eval().requires_grad_(False)
        self.num_candidates = int(num_candidates)
        self.candidate_batch_size = int(candidate_batch_size)
        self.selection = selection
        self.softmax_temperature = float(softmax_temperature)
        self.clip_actions = bool(clip_actions)
        if self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {self.num_candidates}")
        self.critic_used = self.num_candidates > 1
        self.execution_horizon = 1
        self.action_dim = int(checkpoint["action_dim"])
        self.normalize_actions = bool(checkpoint.get("normalize_actions", True))
        self.action_mean = torch.as_tensor(
            checkpoint["action_mean"], device=self.algo.device, dtype=torch.float32
        )
        self.action_std = torch.as_tensor(
            checkpoint["action_std"], device=self.algo.device, dtype=torch.float32
        )
        self.last_q: np.ndarray | None = None
        self.last_v: float | None = None
        self.last_adv: np.ndarray | None = None
        self.last_selected_index: int | None = None

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None

    @torch.no_grad()
    def sample_first_actions(self, prepared_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = []
        for start in range(0, self.num_candidates, self.candidate_batch_size):
            batch = min(self.candidate_batch_size, self.num_candidates - start)
            obs_batch = repeat_obs(prepared_obs, batch)
            trajectory = self.algo._get_action_trajectory(obs_dict=obs_batch)
            if trajectory.ndim != 3:
                raise ValueError(f"expected DP trajectory [B,T,A], got {tuple(trajectory.shape)}")
            actions.append(trajectory[:, 0, :])
        first_actions = torch.cat(actions, dim=0)
        if first_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"DP action_dim={first_actions.shape[-1]} does not match critic action_dim={self.action_dim}"
            )
        if self.clip_actions:
            first_actions = first_actions.clamp(-1.0, 1.0)
        return first_actions

    def normalize_for_critic(self, raw_actions: torch.Tensor) -> torch.Tensor:
        if not self.normalize_actions:
            return raw_actions
        return (raw_actions - self.action_mean[None, :]) / self.action_std[None, :]

    def choose_index(self, q: torch.Tensor, v: torch.Tensor) -> int:
        if self.selection in ("argmax", "greedy") or len(q) == 1:
            return int(torch.argmax(q).item())
        if self.selection == "softmax":
            probs = torch.softmax(q / max(self.softmax_temperature, 1e-6), dim=0)
            return int(torch.multinomial(probs, num_samples=1).item())
        if self.selection == "advantage_softmax":
            adv = q - v
            probs = torch.softmax(adv / max(self.softmax_temperature, 1e-6), dim=0)
            return int(torch.multinomial(probs, num_samples=1).item())
        raise ValueError(f"unknown selection={self.selection}")

    def __call__(self, ob) -> np.ndarray:
        prepared_obs = self.dp_policy._prepare_observation(ob, batched_ob=False)
        raw_actions = self.sample_first_actions(prepared_obs)

        if not self.critic_used:
            self.last_q = None
            self.last_v = None
            self.last_adv = None
            self.last_selected_index = 0
            selected_action = raw_actions[0]
        else:
            obs_feature = (
                encode_current_obs_with_encoder(self.dp_policy, self.critic_obs_encoder, prepared_obs)
                if self.critic_obs_encoder is not None
                else encode_current_obs(self.dp_policy, prepared_obs)
            )
            obs_batch = obs_feature.repeat(raw_actions.shape[0], 1)
            norm_actions = self.normalize_for_critic(raw_actions)
            critic_actions = norm_actions[:, None, :]
            q = self.critic.q_min(obs_batch, critic_actions).reshape(-1)
            v = self.critic.value(obs_feature).reshape(())
            selected = self.choose_index(q, v)
            self.last_q = q.detach().cpu().numpy()
            self.last_v = float(v.detach().cpu())
            self.last_adv = (q - v).detach().cpu().numpy()
            self.last_selected_index = int(selected)
            selected_action = raw_actions[selected]

        return selected_action.detach().cpu().numpy().astype(np.float64).copy()


def load_policy(idql_checkpoint: Path, device: torch.device, args):
    checkpoint = torch.load(idql_checkpoint, map_location=device, weights_only=False)
    dp_checkpoint = Path(checkpoint["pretrained_dp_checkpoint"])
    dp_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=str(dp_checkpoint), device=device, verbose=False
    )

    hybrid_dp_chunk_actor_iql = bool(checkpoint.get("hybrid_dp_chunk_actor_iql", False))
    if args.actor_source == "hybrid_dp_chunk_actor":
        if not hybrid_dp_chunk_actor_iql:
            raise ValueError(
                "actor_source=hybrid_dp_chunk_actor requires a checkpoint from "
                "train_square_rgb_dp_chunk_actor_iql.py"
            )
        dp_policy.policy.deserialize(checkpoint["actor_model"], load_optimizers=False)
        dp_policy.policy.set_eval()
        checkpoint["eval_actor_key"] = "actor_model.ema" if dp_policy.policy.ema is not None else "actor_model.nets"
    elif hybrid_dp_chunk_actor_iql and args.actor_source != "pretrained_dp_first_action":
        raise ValueError(
            "hybrid DP-chunk actor checkpoint should be evaluated with "
            "actor_source=hybrid_dp_chunk_actor or pretrained_dp_first_action"
        )
    checkpoint_args = dict(checkpoint.get("args", {}))
    aux_next_pred_enabled = bool(
        checkpoint.get(
            "aux_next_pred_enabled",
            float(checkpoint_args.get("aux_next_pred_weight", 0.0) or 0.0) > 0.0,
        )
    )
    checkpoint["aux_next_pred_enabled"] = aux_next_pred_enabled
    checkpoint["aux_next_pred_weight"] = float(
        checkpoint.get("aux_next_pred_weight", checkpoint_args.get("aux_next_pred_weight", 0.0)) or 0.0
    )
    checkpoint["aux_next_pred_mode"] = str(
        checkpoint.get("aux_next_pred_mode", checkpoint_args.get("aux_next_pred_mode", "delta"))
    )
    visual_critic_idql = bool(checkpoint.get("visual_critic_idql", False) or hybrid_dp_chunk_actor_iql)
    checkpoint["visual_critic_idql"] = visual_critic_idql
    checkpoint["hybrid_dp_chunk_actor_iql"] = hybrid_dp_chunk_actor_iql
    actor_obs_encoder = None
    critic_obs_encoder = None
    if visual_critic_idql:
        if "actor_encoder" in checkpoint:
            actor_obs_encoder = clone_policy_obs_encoder(dp_policy).to(device)
            actor_obs_encoder.load_state_dict(checkpoint["actor_encoder"])
            actor_obs_encoder.eval().requires_grad_(False)
            checkpoint["eval_actor_encoder_key"] = "actor_encoder"
        elif hybrid_dp_chunk_actor_iql:
            checkpoint["eval_actor_encoder_key"] = "actor_model.ema_or_nets.policy.obs_encoder"

        if args.critic_source == "target":
            if "target_critic_encoder" in checkpoint:
                critic_encoder_key = "target_critic_encoder"
            else:
                raise ValueError(
                    "target critic encoder does not exist!"
                )
        else:
            critic_encoder_key = "critic_encoder"

        critic_obs_encoder = clone_policy_obs_encoder(dp_policy).to(device)
        critic_obs_encoder.load_state_dict(checkpoint[critic_encoder_key])
        critic_obs_encoder.eval().requires_grad_(False)
        checkpoint["eval_critic_encoder_key"] = critic_encoder_key
    critic = ChunkIQLCritic(
        feature_dim=int(checkpoint["feature_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        chunk_horizon=1,
        hidden_dims=tuple(int(x) for x in checkpoint_args["critic_hidden_dims"]),
        dropout=float(checkpoint_args.get("critic_dropout", 0.0)),
        aux_next_pred=aux_next_pred_enabled,
    ).to(device)

    if args.critic_source == "target":
        if "target_critic" in checkpoint:
            critic_key = "target_critic"
        else:
            raise ValueError(
                "target critic does not exist!"
            )
    else:
        critic_key = "critic"

    critic.load_state_dict(checkpoint[critic_key])
    critic.eval().requires_grad_(False)
    checkpoint["eval_critic_key"] = critic_key

    if args.actor_source == "pretrained_dp_first_action":
        checkpoint["eval_actor_key"] = "pretrained_dp_checkpoint.ema_or_nets"
    if args.actor_source in ("pretrained_dp_first_action", "hybrid_dp_chunk_actor"):
        return PretrainedDPFirstActionIDQLPolicy(
            dp_policy,
            critic,
            checkpoint,
            critic_obs_encoder=critic_obs_encoder,
            num_candidates=args.num_candidates,
            candidate_batch_size=args.candidate_batch_size,
            selection=args.selection,
            softmax_temperature=args.softmax_temperature,
            clip_actions=args.clip_actions,
        )

    actor = OneStepDiffusionActor(
        feature_dim=int(checkpoint["feature_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=tuple(int(x) for x in checkpoint["args"]["actor_hidden_dims"]),
        time_dim=int(checkpoint["args"].get("time_dim", 64)),
        dropout=float(checkpoint["args"].get("actor_dropout", 0.0)),
    ).to(device)
    actor_key = "target_actor" if args.actor_source == "idql_target_one_step_mlp" and "target_actor" in checkpoint else "actor"
    actor.load_state_dict(checkpoint[actor_key])
    actor.eval().requires_grad_(False)
    checkpoint["eval_actor_key"] = actor_key
    return OneStepIDQLPolicy(
        dp_policy,
        critic,
        actor,
        checkpoint,
        actor_obs_encoder=actor_obs_encoder,
        critic_obs_encoder=critic_obs_encoder,
        num_candidates=args.num_candidates,
        candidate_batch_size=args.candidate_batch_size,
        num_inference_steps=args.num_inference_steps,
        selection=args.selection,
        softmax_temperature=args.softmax_temperature,
        clip_actions=args.clip_actions,
        diffusion_clip_sample=args.diffusion_clip_sample,
    )


def rollout(policy, env, horizon: int, return_obs: bool = False, video_writer=None, video_skip: int = 5, camera_names=None):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)
    total_reward = 0.0
    success = False
    traj = dict(
        actions=[],
        rewards=[],
        dones=[],
        states=[],
        q_selected=[],
        q_mean=[],
        q_max=[],
        q_min=[],
        q_margin=[],
        q_range=[],
        v=[],
        selected_index=[],
        initial_state_dict=state_dict,
    )
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))
    step_i = -1
    video_count = 0
    try:
        for step_i in range(horizon):
            action = np.asarray(policy(obs), dtype=np.float64, order="C").copy()
            next_obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            success = bool(env.is_success()["task"])
            if video_writer is not None and video_count % video_skip == 0:
                frames = [env.render(mode="rgb_array", height=512, width=512, camera_name=name) for name in camera_names]
                video_writer.append_data(np.concatenate(frames, axis=1))
            video_count += 1
            q = policy.last_q
            selected = policy.last_selected_index
            traj["actions"].append(action)
            traj["rewards"].append(float(reward))
            traj["dones"].append(bool(done))
            traj["states"].append(state_dict["states"])
            if q is None or selected is None:
                traj["q_selected"].append(np.nan)
                traj["q_mean"].append(np.nan)
                traj["q_max"].append(np.nan)
                traj["q_min"].append(np.nan)
                traj["q_margin"].append(np.nan)
                traj["q_range"].append(np.nan)
                traj["v"].append(np.nan)
                traj["selected_index"].append(-1)
            else:
                q_selected = float(q[selected])
                q_mean = float(np.mean(q))
                q_max = float(np.max(q))
                q_min = float(np.min(q))
                traj["q_selected"].append(q_selected)
                traj["q_mean"].append(q_mean)
                traj["q_max"].append(q_max)
                traj["q_min"].append(q_min)
                traj["q_margin"].append(q_selected - q_mean)
                traj["q_range"].append(q_max - q_min)
                traj["v"].append(float(policy.last_v))
                traj["selected_index"].append(int(selected))
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)
            if done or success:
                break
            obs = deepcopy(next_obs)
            state_dict = env.get_state()
    except env.rollout_exceptions as exc:
        print(f"WARNING: rollout exception {exc}", flush=True)
    stats = {"Return": total_reward, "Horizon": step_i + 1, "Success_Rate": float(success)}
    for key in traj:
        if key == "initial_state_dict":
            continue
        traj[key] = np.asarray(traj[key])
    selected_index = traj["selected_index"]
    valid_selection = selected_index >= 0
    if np.any(valid_selection):
        for src_key, stat_key in (
            ("q_selected", "Q_Selected_Mean"),
            ("q_mean", "Q_Mean_Mean"),
            ("q_max", "Q_Max_Mean"),
            ("q_min", "Q_Min_Mean"),
            ("q_margin", "Q_Margin_Mean"),
            ("q_range", "Q_Range_Mean"),
            ("v", "V_Mean"),
        ):
            values = np.asarray(traj[src_key], dtype=np.float64)
            finite = np.isfinite(values)
            if np.any(finite):
                stats[stat_key] = float(np.mean(values[finite]))
        valid_indices = selected_index[valid_selection].astype(np.float64)
        stats["Selected_Index_Mean"] = float(np.mean(valid_indices))
        stats["Selected_Index_First_Fraction"] = float(np.mean(valid_indices == 0.0))
    return stats, traj


def aggregate(stats: list[dict]) -> dict:
    successes = float(sum(x["Success_Rate"] for x in stats))
    n = len(stats)
    result = {
        "Num_Rollouts": n,
        "Return": float(np.mean([x["Return"] for x in stats])) if n else float("nan"),
        "Horizon": float(np.mean([x["Horizon"] for x in stats])) if n else float("nan"),
        "Success_Rate": successes / max(n, 1),
        "Num_Success": successes,
    }
    base_keys = set(result.keys())
    extra_keys = sorted({key for item in stats for key in item.keys()} - base_keys)
    for key in extra_keys:
        values = []
        for item in stats:
            value = item.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            result[key] = float(np.mean(values))
    return result


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [center - radius, center + radius]


def build_summary(args, policy, stats: list[dict], complete: bool) -> dict:
    avg = aggregate(stats)
    successes = int(round(avg["Num_Success"]))
    standard_idql_actor = args.actor_source in ("idql_one_step_mlp", "idql_target_one_step_mlp")
    pretrained_dp_actor = args.actor_source == "pretrained_dp_first_action"
    hybrid_dp_chunk_actor = args.actor_source == "hybrid_dp_chunk_actor"
    return {
        "idql_checkpoint": str(args.idql_checkpoint),
        "pretrained_dp_checkpoint": str(policy.checkpoint["pretrained_dp_checkpoint"]),
        "checkpoint_step": int(policy.checkpoint.get("step", -1)),
        "actor_source": args.actor_source,
        "eval_actor_key": policy.checkpoint.get("eval_actor_key", "actor"),
        "critic_source": args.critic_source,
        "eval_critic_key": policy.checkpoint.get("eval_critic_key", "critic"),
        "visual_critic_idql": bool(policy.checkpoint.get("visual_critic_idql", False)),
        "hybrid_dp_chunk_actor_iql": bool(policy.checkpoint.get("hybrid_dp_chunk_actor_iql", False)),
        "eval_actor_encoder_key": policy.checkpoint.get("eval_actor_encoder_key"),
        "eval_critic_encoder_key": policy.checkpoint.get("eval_critic_encoder_key"),
        "aux_next_pred_enabled": bool(policy.checkpoint.get("aux_next_pred_enabled", False)),
        "aux_next_pred_weight": float(policy.checkpoint.get("aux_next_pred_weight", 0.0) or 0.0),
        "aux_next_pred_mode": str(policy.checkpoint.get("aux_next_pred_mode", "delta")),
        "actor_uses_pretrained_dp_weights": bool(pretrained_dp_actor or hybrid_dp_chunk_actor),
        "standard_idql_trained_actor": bool(standard_idql_actor),
        "dp_chunk_actor_bc_trained": bool(hybrid_dp_chunk_actor),
        "pretrained_checkpoint_used_for_encoder": True,
        "num_candidates": args.num_candidates,
        "num_inference_steps": args.num_inference_steps,
        "selection": args.selection,
        "clip_actions": args.clip_actions,
        "diffusion_clip_sample": args.diffusion_clip_sample,
        "paper_faithful_one_step_idql": bool(standard_idql_actor),
        "pretrained_dp_first_action_baseline": bool(pretrained_dp_actor and args.num_candidates == 1),
        "pretrained_dp_proposal_idql_critic_rerank": bool(pretrained_dp_actor and args.num_candidates > 1),
        "hybrid_dp_chunk_actor_actor_only": bool(hybrid_dp_chunk_actor and args.num_candidates == 1),
        "hybrid_dp_chunk_actor_critic_rerank": bool(hybrid_dp_chunk_actor and args.num_candidates > 1),
        "actor_proposal_horizon": int(policy.checkpoint.get("actor_action_horizon", 1)),
        "execution_horizon": 1,
        "replan_every_env_step": True,
        "critic_used_for_action_selection": bool(args.num_candidates > 1),
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "completed_rollouts": len(stats),
        "complete": bool(complete),
        "horizon": args.horizon,
        "average_rollout_stats": avg,
        "wilson_95_interval": wilson(successes, len(stats)),
        "rollouts": stats,
    }


def write_summary(args, policy, stats: list[dict], complete: bool, suffix: str = "") -> Path:
    summary = build_summary(args, policy, stats, complete=complete)
    path = args.output_dir / f"one_step_idql_N{args.num_candidates}_seed{args.seed}{suffix}.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def evaluate(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")
    policy = load_policy(args.idql_checkpoint, device, args)
    _, dp_ckpt = FileUtils.policy_from_checkpoint(
        ckpt_path=str(policy.checkpoint["pretrained_dp_checkpoint"]), device=device, verbose=False
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=dp_ckpt,
        render=False,
        render_offscreen=args.video_dir is not None,
        verbose=False,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    dataset_writer = None
    data_group = None
    total_samples = 0
    if args.dataset_path is not None:
        args.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_writer = h5py.File(args.dataset_path, "w")
        data_group = dataset_writer.create_group("data")
    stats = []
    for i in range(args.n_rollouts):
        writer = None
        if args.video_dir is not None and i < args.num_videos:
            writer = imageio.get_writer(args.video_dir / f"rollout_{i:03d}.mp4", fps=20)
        rollout_stats, traj = rollout(
            policy,
            env,
            horizon=args.horizon,
            return_obs=False,
            video_writer=writer,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
        )
        if writer is not None:
            writer.close()
        stats.append(rollout_stats)
        partial_path = write_summary(args, policy, stats, complete=False, suffix="_partial")
        print(
            f"rollout={i} success={rollout_stats['Success_Rate']:.0f} "
            f"return={rollout_stats['Return']:.3f} horizon={rollout_stats['Horizon']} "
            f"partial_success={aggregate(stats)['Success_Rate']:.3f} partial={partial_path}",
            flush=True,
        )
        if data_group is not None:
            ep = data_group.create_group(f"demo_{i}")
            for key in (
                "actions",
                "rewards",
                "dones",
                "states",
                "q_selected",
                "q_mean",
                "q_max",
                "q_min",
                "q_margin",
                "q_range",
                "v",
                "selected_index",
            ):
                ep.create_dataset(key, data=traj[key])
            if "model" in traj["initial_state_dict"]:
                ep.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep.attrs["num_samples"] = int(traj["actions"].shape[0])
            ep.attrs["success"] = float(rollout_stats["Success_Rate"])
            total_samples += int(traj["actions"].shape[0])
    if dataset_writer is not None:
        data_group.attrs["total"] = total_samples
        dataset_writer.close()
    path = write_summary(args, policy, stats, complete=True, suffix="")
    summary = json.loads(path.read_text())
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idql-checkpoint", type=Path, default=DEFAULT_IDQL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--actor-source",
        choices=(
            "idql_target_one_step_mlp",
            "idql_one_step_mlp",
            "pretrained_dp_first_action",
            "hybrid_dp_chunk_actor",
        ),
        default="idql_target_one_step_mlp",
    )
    parser.add_argument("--critic-source", choices=("target", "online"), default="target")
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--selection", choices=("argmax", "greedy", "softmax", "advantage_softmax"), default="argmax")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-clip-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--camera-names", type=str, nargs="+", default=("agentview", "robot0_eye_in_hand"))
    args = parser.parse_args()
    for key in ("idql_checkpoint", "output_dir", "dataset_path", "video_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    evaluate(args)


if __name__ == "__main__":
    main()
