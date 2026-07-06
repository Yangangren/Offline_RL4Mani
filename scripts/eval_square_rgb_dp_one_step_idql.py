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


class OneStepIDQLPolicy:
    def __init__(
        self,
        dp_policy,
        critic: ChunkIQLCritic,
        actor: OneStepDiffusionActor,
        checkpoint: dict,
        *,
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
        obs_feature = encode_current_obs(self.dp_policy, prepared_obs)
        norm_actions, raw_actions = self.sample_candidates(obs_feature)

        if not self.critic_used:
            # N=1 is the trained diffusion actor only. No critic is queried and
            # no action selection happens, matching the actor-only ablation.
            self.last_q = None
            self.last_v = None
            self.last_adv = None
            self.last_selected_index = 0
            selected_action = raw_actions[0]
        else:
            obs_batch = obs_feature.repeat(norm_actions.shape[0], 1)
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


class PretrainedDPFirstActionIDQLPolicy:
    """Paper-faithful one-step IDQL extraction using pretrained DP as actor.

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
            obs_feature = encode_current_obs(self.dp_policy, prepared_obs)
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
    critic = ChunkIQLCritic(
        feature_dim=int(checkpoint["feature_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        chunk_horizon=1,
        hidden_dims=tuple(int(x) for x in checkpoint["args"]["critic_hidden_dims"]),
        dropout=float(checkpoint["args"].get("critic_dropout", 0.0)),
    ).to(device)
    critic_key = "target_critic" if args.critic_source == "target" and "target_critic" in checkpoint else "critic"
    critic.load_state_dict(checkpoint[critic_key])
    critic.eval().requires_grad_(False)
    checkpoint["eval_critic_key"] = critic_key

    if args.actor_source == "pretrained_dp_first_action":
        return PretrainedDPFirstActionIDQLPolicy(
            dp_policy,
            critic,
            checkpoint,
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
    traj = dict(actions=[], rewards=[], dones=[], states=[], q_selected=[], q_mean=[], q_max=[], q_min=[], v=[], initial_state_dict=state_dict)
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
                traj["v"].append(np.nan)
            else:
                traj["q_selected"].append(float(q[selected]))
                traj["q_mean"].append(float(np.mean(q)))
                traj["q_max"].append(float(np.max(q)))
                traj["q_min"].append(float(np.min(q)))
                traj["v"].append(float(policy.last_v))
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
    return stats, traj


def aggregate(stats: list[dict]) -> dict:
    successes = float(sum(x["Success_Rate"] for x in stats))
    n = len(stats)
    return {
        "Num_Rollouts": n,
        "Return": float(np.mean([x["Return"] for x in stats])) if n else float("nan"),
        "Horizon": float(np.mean([x["Horizon"] for x in stats])) if n else float("nan"),
        "Success_Rate": successes / max(n, 1),
        "Num_Success": successes,
    }


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
    return {
        "idql_checkpoint": str(args.idql_checkpoint),
        "pretrained_dp_checkpoint": str(policy.checkpoint["pretrained_dp_checkpoint"]),
        "checkpoint_step": int(policy.checkpoint.get("step", -1)),
        "actor_source": args.actor_source,
        "eval_actor_key": policy.checkpoint.get("eval_actor_key", "actor"),
        "critic_source": args.critic_source,
        "eval_critic_key": policy.checkpoint.get("eval_critic_key", "critic"),
        "actor_uses_pretrained_dp_weights": bool(args.actor_source == "pretrained_dp_first_action"),
        "standard_idql_trained_actor": bool(args.actor_source in ("idql_one_step_mlp", "idql_target_one_step_mlp")),
        "pretrained_checkpoint_used_for_encoder": True,
        "num_candidates": args.num_candidates,
        "num_inference_steps": args.num_inference_steps,
        "selection": args.selection,
        "clip_actions": args.clip_actions,
        "diffusion_clip_sample": args.diffusion_clip_sample,
        "paper_faithful_one_step_idql": True,
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
            for key in ("actions", "rewards", "dones", "states", "q_selected", "q_mean", "q_max", "q_min", "v"):
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
        choices=("idql_target_one_step_mlp", "idql_one_step_mlp", "pretrained_dp_first_action"),
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
