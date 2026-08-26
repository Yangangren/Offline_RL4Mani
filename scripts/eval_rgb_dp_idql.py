#!/usr/bin/env python3
"""Closed-loop evaluation for chunked actor and one-step critic IDQL.

At every environment step this evaluator replans from the current observation
for the chunked IDQL actor. The DP-proposal modes can optionally queue multiple
actions from the selected DP trajectory via --execution-horizon.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

from train_square_rgb_dp_one_step_idql import ChunkIQLCritic, OneStepDiffusionActor, make_scheduler
from train_rgb_dp_chunk_idql import (
    LEGACY_CRITIC_ARCHITECTURE,
    PREDICTED_NEXT_Q_NORMALIZATION,
    RISE_V2_CRITIC_ARCHITECTURE,
    WCM_CRITIC_ARCHITECTURE,
    architecture_q_head_inputs,
    checkpoint_critic_architecture,
    make_rise_v2_system_from_checkpoint,
    make_wcm_system_from_checkpoint,
    make_rise_chunk_value_networks,
    match_encoder_normalization_to_checkpoint,
)
from train_rgb_dp_idql import action_normalization_stats_match, make_rise_value_networks
from train_rgb_dp_dql import make_dql_value_networks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDQL = (
    ROOT
    / "trained_models/square_rgb_dp_idql/default_reward_one_step_idql_no_rollout/best_success_auc.pt"
)
DEFAULT_DP = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth"
)
DEFAULT_OUTPUT = ROOT / "rollouts/square_rgb_dp/one_step_idql_eval"
SELECTION_CHOICES = (
    "argmax",
    "greedy",
    "actor_first",
    "softmax",
    "advantage_softmax",
    "epsilon_greedy",
)
TASK_ENV_NAMES = {
    "square": "NutAssemblySquare",
    "can": "PickPlaceCan",
    "transport": "TwoArmTransport",
    "tool_hang": "ToolHang",
}
RUNTIME_ONLY_ENV_KWARGS = frozenset(
    (
        "has_renderer",
        "has_offscreen_renderer",
        "ignore_done",
        "render_gpu_device_id",
        "reward_shaping",
    )
)
UINT32_SEED_MODULUS = 1 << 32
SEED_NORMALIZATION_SCHEME = "nonnegative_integer_modulo_2**32"


def normalize_evaluation_seed(seed, *, label: str) -> int:
    """Map a non-negative integer seed into NumPy's legacy uint32 domain."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError(
            f"{label} must be a non-negative integer, got {seed!r}"
        )
    seed = int(seed)
    if seed < 0:
        raise ValueError(
            f"{label} must be a non-negative integer, got {seed}"
        )
    return seed % UINT32_SEED_MODULUS


def evaluation_seed_metadata(args) -> dict[str, Any]:
    """Return requested and effective seeds shared by every RNG backend."""
    raw_seed = args.seed
    raw_env_seed = (
        raw_seed
        if getattr(args, "env_seed", None) is None
        else args.env_seed
    )
    raw_policy_seed = (
        raw_seed
        if getattr(args, "policy_seed", None) is None
        else args.policy_seed
    )
    effective_seed = normalize_evaluation_seed(raw_seed, label="--seed")
    effective_env_seed = normalize_evaluation_seed(
        raw_env_seed,
        label="--env-seed",
    )
    effective_policy_seed = normalize_evaluation_seed(
        raw_policy_seed,
        label="--policy-seed",
    )
    requested_seed = int(raw_seed)
    requested_env_seed = int(raw_env_seed)
    requested_policy_seed = int(raw_policy_seed)
    return {
        "seed": requested_seed,
        "env_seed": requested_env_seed,
        "policy_seed": requested_policy_seed,
        "requested_seed": requested_seed,
        "requested_env_seed": requested_env_seed,
        "requested_policy_seed": requested_policy_seed,
        "effective_seed": effective_seed,
        "effective_env_seed": effective_env_seed,
        "effective_policy_seed": effective_policy_seed,
        "seed_was_normalized": effective_seed != requested_seed,
        "env_seed_was_normalized": effective_env_seed != requested_env_seed,
        "policy_seed_was_normalized": (
            effective_policy_seed != requested_policy_seed
        ),
        "seed_normalization_scheme": SEED_NORMALIZATION_SCHEME,
        "seed_normalization_modulus": UINT32_SEED_MODULUS,
    }


def effective_policy_seed(args) -> int:
    return int(evaluation_seed_metadata(args)["effective_policy_seed"])





def choose_candidate_index(
    q: torch.Tensor,
    v: torch.Tensor,
    *,
    selection: str,
    softmax_temperature: float,
    random_selection_probability: float,
    selection_rng: np.random.Generator,
) -> tuple[int, bool]:
    """Choose one candidate and report uniform epsilon exploration separately."""
    if len(q) == 1:
        return int(torch.argmax(q).item()), False
    if selection == "actor_first":
        return 0, False
    if selection in ("argmax", "greedy"):
        return int(torch.argmax(q).item()), False
    if selection == "epsilon_greedy":
        explore = bool(selection_rng.random() < random_selection_probability)
        if explore:
            return int(selection_rng.integers(0, len(q))), True
        return int(torch.argmax(q).item()), False
    if selection == "softmax":
        probabilities = torch.softmax(
            q / max(softmax_temperature, 1e-6),
            dim=0,
        )
        return int(torch.multinomial(probabilities, num_samples=1).item()), False
    if selection == "advantage_softmax":
        probabilities = torch.softmax(
            (q - v) / max(softmax_temperature, 1e-6),
            dim=0,
        )
        return int(torch.multinomial(probabilities, num_samples=1).item()), False
    raise ValueError(f"unknown selection={selection}")


def atomic_write_json(path: Path, payload: dict) -> None:
    """Keep the previous valid checkpoint if the host dies during a write."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_dataset_target_available(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"trajectory dataset already exists: {path}; "
            "pass --overwrite-dataset to replace it after a successful evaluation"
        )


def make_temporary_dataset_path(path: Path, *, overwrite: bool) -> Path:
    """Allocate a unique sibling file without touching the final dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_dataset_target_available(path, overwrite=overwrite)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    return Path(temporary)


def publish_dataset(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically publish a completed HDF5 file."""
    ensure_dataset_target_available(destination, overwrite=overwrite)
    os.replace(temporary, destination)



def raw_rollout_env(env):
    """Return the simulator-owning environment underneath robomimic wrappers."""
    current = env
    while isinstance(current, EnvWrapper):
        current = current.env
    return getattr(current, "base_env", current)


def configure_env_hard_reset(env, enabled: bool) -> bool | None:
    raw_env = raw_rollout_env(env)
    if not hasattr(raw_env, "hard_reset"):
        return None
    raw_env.hard_reset = bool(enabled)
    return bool(raw_env.hard_reset)


def current_sim_state(env) -> np.ndarray:
    """Read MuJoCo state without serializing the complete model XML."""
    raw_env = raw_rollout_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is not None and hasattr(sim, "get_state"):
        return np.asarray(sim.get_state().flatten()).copy()
    return np.asarray(env.get_state()["states"]).copy()


def close_rollout_env(env) -> None:
    raw_env = raw_rollout_env(env)
    close = getattr(raw_env, "close", None)
    if callable(close):
        close()


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
        random_selection_probability: float,
        selection_seed: int,
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
        self.random_selection_probability = float(random_selection_probability)
        self.selection_seed = int(selection_seed)
        self.selection_rng = np.random.default_rng(self.selection_seed)
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
        self.last_selection_is_random: bool | None = None

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None
        self.last_selection_is_random = None

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
        selected, explored = choose_candidate_index(
            q,
            v,
            selection=self.selection,
            softmax_temperature=self.softmax_temperature,
            random_selection_probability=self.random_selection_probability,
            selection_rng=self.selection_rng,
        )
        self.last_selection_is_random = explored
        return selected

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
            self.last_selection_is_random = None
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
    """DP-trajectory proposal policy with one-step IQL critic scoring.

    The actor sampler is a RGB DiffusionPolicy checkpoint. It samples full DP
    action trajectories, scores only each trajectory's first action with
    Q(o_t, a_t), and then executes the selected trajectory prefix. With
    execution_horizon=1 this is the original one-step replanning behavior; with
    execution_horizon=8 it matches the default DP chunk-execution cadence.
    """

    def __init__(
        self,
        dp_policy,
        dp_ckpt: dict,
        critic: ChunkIQLCritic,
        checkpoint: dict,
        *,
        critic_obs_encoder=None,
        num_candidates: int,
        candidate_batch_size: int,
        selection: str,
        softmax_temperature: float,
        random_selection_probability: float,
        selection_seed: int,
        clip_actions: bool,
        execution_horizon: int,
    ):
        self.dp_policy = dp_policy
        self.dp_ckpt = dp_ckpt
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
        self.random_selection_probability = float(random_selection_probability)
        self.selection_seed = int(selection_seed)
        self.selection_rng = np.random.default_rng(self.selection_seed)
        self.clip_actions = bool(clip_actions)
        if self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {self.num_candidates}")
        self.execution_horizon = int(execution_horizon)
        self.critic_used = self.num_candidates > 1
        self.action_dim = int(checkpoint["action_dim"])
        self.normalize_actions = bool(checkpoint.get("normalize_actions", True))
        self.action_mean = torch.as_tensor(
            checkpoint["action_mean"], device=self.algo.device, dtype=torch.float32
        )
        self.action_std = torch.as_tensor(
            checkpoint["action_std"], device=self.algo.device, dtype=torch.float32
        )
        self.action_queue: deque[torch.Tensor] = deque()
        self.last_q: np.ndarray | None = None
        self.last_v: float | None = None
        self.last_adv: np.ndarray | None = None
        self.last_selected_index: int | None = None
        self.last_selection_is_random: bool | None = None

        horizon = self.algo.algo_config.horizon
        self.checkpoint.setdefault("prediction_horizon", int(horizon.prediction_horizon))
        self.checkpoint.setdefault("action_horizon", int(horizon.action_horizon))
        self.checkpoint["actor_proposal_horizon"] = int(horizon.action_horizon)
        self.checkpoint["execution_horizon"] = int(self.execution_horizon)
        self.checkpoint["queued_action_horizon"] = int(self.execution_horizon)
        self.checkpoint["replan_every_env_step"] = bool(self.execution_horizon == 1)

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        self.action_queue.clear()
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None
        self.last_selection_is_random = None

    def _clear_decision_stats_for_queued_action(self) -> None:
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None
        self.last_selection_is_random = None

    @torch.no_grad()
    def sample_action_trajectories(self, prepared_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        trajectories = []
        for start in range(0, self.num_candidates, self.candidate_batch_size):
            batch = min(self.candidate_batch_size, self.num_candidates - start)
            obs_batch = repeat_obs(prepared_obs, batch)
            trajectory = self.algo._get_action_trajectory(obs_dict=obs_batch)
            if trajectory.ndim != 3:
                raise ValueError(f"expected DP trajectory [B,T,A], got {tuple(trajectory.shape)}")
            trajectories.append(trajectory)
        action_trajectories = torch.cat(trajectories, dim=0)
        if action_trajectories.shape[-1] != self.action_dim:
            raise ValueError(
                f"DP action_dim={action_trajectories.shape[-1]} does not match critic action_dim={self.action_dim}"
            )
        if action_trajectories.shape[1] < self.execution_horizon:
            raise ValueError(
                f"DP trajectory horizon={action_trajectories.shape[1]} is shorter than "
                f"execution_horizon={self.execution_horizon}"
            )
        if self.clip_actions:
            action_trajectories = action_trajectories.clamp(-1.0, 1.0)
        return action_trajectories

    def normalize_for_critic(self, raw_actions: torch.Tensor) -> torch.Tensor:
        if not self.normalize_actions:
            return raw_actions
        return (raw_actions - self.action_mean[None, :]) / self.action_std[None, :]

    def choose_index(self, q: torch.Tensor, v: torch.Tensor) -> int:
        selected, explored = choose_candidate_index(
            q,
            v,
            selection=self.selection,
            softmax_temperature=self.softmax_temperature,
            random_selection_probability=self.random_selection_probability,
            selection_rng=self.selection_rng,
        )
        self.last_selection_is_random = explored
        return selected

    def __call__(self, ob) -> np.ndarray:
        if self.action_queue:
            self._clear_decision_stats_for_queued_action()
            action = self.action_queue.popleft()
            return action.detach().cpu().numpy().astype(np.float64).copy()

        prepared_obs = self.dp_policy._prepare_observation(ob, batched_ob=False)
        trajectories = self.sample_action_trajectories(prepared_obs)
        first_actions = trajectories[:, 0, :]

        if not self.critic_used:
            self.last_q = None
            self.last_v = None
            self.last_adv = None
            self.last_selected_index = 0
            self.last_selection_is_random = None
            selected = 0
        else:
            obs_feature = (
                encode_current_obs_with_encoder(self.dp_policy, self.critic_obs_encoder, prepared_obs)
                if self.critic_obs_encoder is not None
                else encode_current_obs(self.dp_policy, prepared_obs)
            )
            obs_batch = obs_feature.repeat(first_actions.shape[0], 1)
            norm_actions = self.normalize_for_critic(first_actions)
            critic_actions = norm_actions[:, None, :]
            q = self.critic.q_min(obs_batch, critic_actions).reshape(-1)
            v = self.critic.value(obs_feature).reshape(())
            selected = self.choose_index(q, v)
            self.last_q = q.detach().cpu().numpy()
            self.last_v = float(v.detach().cpu())
            self.last_adv = (q - v).detach().cpu().numpy()
            self.last_selected_index = int(selected)

        selected_trajectory = trajectories[selected, : self.execution_horizon, :]
        for action in selected_trajectory[1:]:
            self.action_queue.append(action.detach())
        selected_action = selected_trajectory[0]
        return selected_action.detach().cpu().numpy().astype(np.float64).copy()


def current_obs_for_value_network(algo, prepared_obs: dict[str, torch.Tensor]):
    """Remove an optional rollout frame-stack axis for a one-step value net."""
    current = {}
    for key, shape in algo.obs_shapes.items():
        value = prepared_obs[key]
        expected_batched_ndim = len(shape) + 1
        if value.ndim == expected_batched_ndim + 1:
            value = value[:, -1]
        elif value.ndim != expected_batched_ndim:
            raise ValueError(
                f"unexpected prepared observation shape for {key}: {tuple(value.shape)}; "
                f"expected [B,{','.join(str(x) for x in shape)}] with an optional time axis"
            )
        current[key] = value
    return current


def stacked_obs_for_value_network(
    algo,
    prepared_obs: dict[str, torch.Tensor],
    observation_horizon: int,
):
    """Preserve the DP frame stack for a version-2 DQL critic."""
    stacked = {}
    observation_horizon = int(observation_horizon)
    for key, shape in algo.obs_shapes.items():
        value = prepared_obs[key]
        expected_batched_ndim = len(shape) + 1
        if value.ndim == expected_batched_ndim:
            value = value.unsqueeze(1).expand(
                -1,
                observation_horizon,
                *([-1] * len(shape)),
            )
        elif value.ndim == expected_batched_ndim + 1:
            if value.shape[1] < observation_horizon:
                padding = value[:, :1].expand(
                    -1,
                    observation_horizon - value.shape[1],
                    *([-1] * len(shape)),
                )
                value = torch.cat((padding, value), dim=1)
            else:
                value = value[:, -observation_horizon:]
        else:
            raise ValueError(
                f"unexpected stacked DQL observation shape for {key}: "
                f"{tuple(value.shape)}"
            )
        stacked[key] = value
    return stacked


def unnormalize_dp_action_trajectories(dp_policy, actions: torch.Tensor) -> np.ndarray:
    """Apply the same action conversion as robomimic.algo.RolloutPolicy."""
    action_array = actions.detach().cpu().numpy()
    normalization_stats = dp_policy.action_normalization_stats
    if normalization_stats is None:
        return action_array

    leading_shape = action_array.shape[:-1]
    flat_actions = action_array.reshape(-1, action_array.shape[-1])
    action_keys = dp_policy.policy.global_config.train.action_keys
    action_shapes = {
        key: normalization_stats[key]["offset"].shape[1:]
        for key in normalization_stats
    }
    action_dict = PyUtils.vector_to_action_dict(
        flat_actions,
        action_shapes=action_shapes,
        action_keys=action_keys,
    )
    action_dict = ObsUtils.unnormalize_dict(
        action_dict,
        normalization_stats=normalization_stats,
    )
    action_config = dp_policy.policy.global_config.train.action_config
    for key, value in action_dict.items():
        if action_config[key].get("format") != "rot_6d":
            continue
        rotation_6d = torch.from_numpy(value)
        conversion = action_config[key].get("convert_at_runtime", "rot_axis_angle")
        if conversion == "rot_axis_angle":
            action_dict[key] = TorchUtils.rot_6d_to_axis_angle(rotation_6d).numpy()
        elif conversion == "rot_euler":
            action_dict[key] = TorchUtils.rot_6d_to_euler_angles(
                rotation_6d,
                convention="XYZ",
            ).numpy()
        else:
            raise ValueError(f"unsupported runtime rotation conversion {conversion}")
    flat_env_actions = PyUtils.action_dict_to_vector(
        action_dict,
        action_keys=action_keys,
    )
    return flat_env_actions.reshape(*leading_shape, flat_env_actions.shape[-1])


class RiseStyleRGBIDQLPolicy:
    """Post-trained DP proposals reranked by RISE-style raw RGB critics."""

    def __init__(
        self,
        dp_policy,
        dp_ckpt: dict,
        critics,
        vf,
        checkpoint: dict,
        *,
        num_candidates: int,
        candidate_batch_size: int,
        selection: str,
        softmax_temperature: float,
        random_selection_probability: float,
        selection_seed: int,
        clip_actions: bool,
        execution_horizon: int,
        wcm_q_system=None,
        wcm_value_system=None,
    ):
        self.dp_policy = dp_policy
        self.dp_ckpt = dp_ckpt
        self.algo = dp_policy.policy
        self.critics = critics
        self.vf = vf
        self.wcm_q_system = wcm_q_system
        self.wcm_value_system = wcm_value_system
        self.critic_architecture = checkpoint_critic_architecture(checkpoint)
        if (self.wcm_q_system is None) != (self.wcm_value_system is None):
            raise ValueError("WCM evaluation requires both Q and value systems")
        self.checkpoint = checkpoint
        self.num_candidates = int(num_candidates)
        self.candidate_batch_size = int(candidate_batch_size)
        self.selection = selection
        self.softmax_temperature = float(softmax_temperature)
        self.random_selection_probability = float(random_selection_probability)
        self.selection_seed = int(selection_seed)
        self.selection_rng = np.random.default_rng(self.selection_seed)
        self.clip_actions = bool(clip_actions)
        self.execution_horizon = int(execution_horizon)
        self.critic_chunk_horizon = int(
            checkpoint.get("critic_chunk_horizon", 1)
        )
        self.critic_observation_horizon = int(
            checkpoint.get("critic_observation_horizon", 1)
        )
        if self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {self.num_candidates}")
        if self.execution_horizon <= 0:
            raise ValueError(
                f"execution_horizon must be positive, got {self.execution_horizon}"
            )
        self.critic_used = self.num_candidates > 1
        q_head_count = (
            int(self.wcm_q_system.num_critics)
            if self.wcm_q_system is not None
            else len(self.critics)
        )
        if self.critic_used and q_head_count < 2:
            raise ValueError("RISE-style reranking requires twin critics")
        if (
            self.critic_used
            and self.selection == "advantage_softmax"
            and self.vf is None
            and self.wcm_value_system is None
        ):
            raise ValueError("advantage_softmax selection requires a value net")
        self.action_queue: deque[np.ndarray] = deque()
        self.last_q: np.ndarray | None = None
        self.last_v: float | None = None
        self.last_adv: np.ndarray | None = None
        self.last_selected_index: int | None = None
        self.last_selection_is_random: bool | None = None

        horizon = self.algo.algo_config.horizon
        actor_observation_horizon = int(horizon.observation_horizon)
        if not 1 <= self.critic_observation_horizon <= actor_observation_horizon:
            raise ValueError(
                "critic observation horizon must be in [1, actor observation "
                f"horizon={actor_observation_horizon}], got "
                f"{self.critic_observation_horizon}"
            )
        self.checkpoint["observation_horizon"] = actor_observation_horizon
        self.checkpoint["prediction_horizon"] = int(horizon.prediction_horizon)
        self.checkpoint["action_horizon"] = int(horizon.action_horizon)
        self.checkpoint["actor_proposal_horizon"] = int(horizon.action_horizon)
        self.checkpoint["execution_horizon"] = self.execution_horizon
        self.checkpoint["queued_action_horizon"] = self.execution_horizon
        self.checkpoint["replan_every_env_step"] = self.execution_horizon == 1
        self.checkpoint["critic_chunk_horizon"] = self.critic_chunk_horizon

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        self.action_queue.clear()
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None
        self.last_selection_is_random = None

    def choose_index(self, q: torch.Tensor, v: torch.Tensor) -> int:
        selected, explored = choose_candidate_index(
            q,
            v,
            selection=self.selection,
            softmax_temperature=self.softmax_temperature,
            random_selection_probability=self.random_selection_probability,
            selection_rng=self.selection_rng,
        )
        self.last_selection_is_random = explored
        return selected

    @torch.no_grad()
    def sample_action_trajectories(
        self,
        prepared_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        trajectories = []
        for start in range(0, self.num_candidates, self.candidate_batch_size):
            count = min(self.candidate_batch_size, self.num_candidates - start)
            obs_batch = repeat_obs(prepared_obs, count)
            trajectory = self.algo._get_action_trajectory(obs_dict=obs_batch)
            if trajectory.ndim != 3:
                raise ValueError(
                    f"expected DP trajectory [B,T,A], got {tuple(trajectory.shape)}"
                )
            trajectories.append(trajectory)
        result = torch.cat(trajectories, dim=0)
        if result.shape[1] < self.execution_horizon:
            raise ValueError(
                f"actor returns {result.shape[1]} actions, shorter than "
                f"execution_horizon={self.execution_horizon}"
            )
        if result.shape[-1] != int(self.checkpoint["action_dim"]):
            raise ValueError(
                f"actor action_dim={result.shape[-1]} does not match "
                f"critic action_dim={self.checkpoint['action_dim']}"
            )
        if self.clip_actions:
            result = result.clamp(-1.0, 1.0)
        return result

    def __call__(self, ob) -> np.ndarray:
        if self.action_queue:
            self.last_q = None
            self.last_v = None
            self.last_adv = None
            self.last_selected_index = None
            self.last_selection_is_random = None
            return self.action_queue.popleft().astype(np.float64, copy=True)

        prepared_obs = self.dp_policy._prepare_observation(ob, batched_ob=False)
        normalized_trajectories = self.sample_action_trajectories(prepared_obs)
        first_actions = normalized_trajectories[:, 0]

        if not self.critic_used:
            selected = 0
            self.last_q = None
            self.last_v = None
            self.last_adv = None
            self.last_selected_index = 0
            self.last_selection_is_random = None
        else:
            if self.critic_observation_horizon > 1:
                current_obs = stacked_obs_for_value_network(
                    self.algo,
                    prepared_obs,
                    self.critic_observation_horizon,
                )
            else:
                current_obs = current_obs_for_value_network(
                    self.algo,
                    prepared_obs,
                )
            if self.wcm_q_system is not None:
                if normalized_trajectories.shape[1] < self.critic_chunk_horizon:
                    raise ValueError(
                        "actor proposal is shorter than WCM chunk horizon: "
                        f"proposal={normalized_trajectories.shape[1]}, "
                        f"critic={self.critic_chunk_horizon}"
                    )
                critic_actions = normalized_trajectories[
                    :, : self.critic_chunk_horizon
                ]
                action_mask = torch.ones(
                    critic_actions.shape[:2],
                    device=critic_actions.device,
                    dtype=critic_actions.dtype,
                )
                q_state = self.wcm_q_system.encode_state(current_obs, None)
                q_predictions = self.wcm_q_system.q_values_from_state(
                    q_state,
                    critic_actions,
                    action_mask,
                )
                if len(q_predictions) < 2 or any(
                    tuple(value.shape) != (self.num_candidates, 1)
                    for value in q_predictions
                ):
                    raise ValueError(
                        "WCM Q heads must return twin [N,1] predictions"
                    )
                q = torch.cat(q_predictions, dim=1).min(dim=1).values
                value_state = (
                    q_state
                    if self.wcm_value_system is self.wcm_q_system
                    else self.wcm_value_system.encode_state(current_obs, None)
                )
                v = self.wcm_value_system.value_from_state(
                    value_state
                ).reshape(())
            elif self.critic_architecture == RISE_V2_CRITIC_ARCHITECTURE:
                if normalized_trajectories.shape[1] < self.critic_chunk_horizon:
                    raise ValueError(
                        "actor proposal is shorter than RISE-v2 chunk horizon: "
                        f"proposal={normalized_trajectories.shape[1]}, "
                        f"critic={self.critic_chunk_horizon}"
                    )
                critic_actions = normalized_trajectories[
                    :, : self.critic_chunk_horizon
                ]
                action_mask = torch.ones(
                    critic_actions.shape[:2],
                    device=critic_actions.device,
                    dtype=critic_actions.dtype,
                )
                q_predictions = []
                for critic in self.critics:
                    # Each independent RISE critic encodes the observation once;
                    # only its compact temporal state is expanded across N actor
                    # proposals. This keeps reranking cost independent of N for RGB.
                    state = critic.encode_state(current_obs, None)
                    q_predictions.append(
                        critic.q_from_state(
                            state,
                            critic_actions,
                            action_mask,
                        )
                    )
                q = torch.cat(q_predictions, dim=1).min(dim=1).values
                v = (
                    self.vf(obs_dict=current_obs, goal_dict=None).reshape(())
                    if self.vf is not None
                    else q.new_zeros(())
                )
            else:
                obs_batch = repeat_obs(current_obs, self.num_candidates)
                if self.critic_chunk_horizon == 1:
                    q_predictions = [
                        critic(
                            obs_dict=obs_batch,
                            acts=first_actions,
                            goal_dict=None,
                        )
                        for critic in self.critics
                    ]
                else:
                    if normalized_trajectories.shape[1] < self.critic_chunk_horizon:
                        raise ValueError(
                            "actor proposal is shorter than critic chunk horizon: "
                            f"proposal={normalized_trajectories.shape[1]}, "
                            f"critic={self.critic_chunk_horizon}"
                        )
                    critic_actions = normalized_trajectories[
                        :, : self.critic_chunk_horizon
                    ]
                    action_mask = torch.ones(
                        critic_actions.shape[:2],
                        device=critic_actions.device,
                        dtype=critic_actions.dtype,
                    )
                    q_predictions = [
                        critic(
                            obs_dict=obs_batch,
                            acts=critic_actions,
                            action_mask=action_mask,
                            goal_dict=None,
                        )
                        for critic in self.critics
                    ]
                q = torch.cat(q_predictions, dim=1).min(dim=1).values
                v = (
                    self.vf(obs_dict=current_obs, goal_dict=None).reshape(())
                    if self.vf is not None
                    else q.new_zeros(())
                )
            selected = self.choose_index(q, v)
            self.last_q = q.detach().cpu().numpy()
            self.last_v = float(v.detach().cpu())
            self.last_adv = (q - v).detach().cpu().numpy()
            self.last_selected_index = int(selected)

        env_trajectories = unnormalize_dp_action_trajectories(
            self.dp_policy,
            normalized_trajectories,
        )
        selected_trajectory = env_trajectories[
            selected,
            : self.execution_horizon,
        ]
        for action in selected_trajectory[1:]:
            self.action_queue.append(np.asarray(action).copy())
        return np.asarray(selected_trajectory[0], dtype=np.float64).copy()


def dp_eval_metadata(ckpt_dict: dict) -> dict:
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    horizon = config.algo.horizon
    ddpm_enabled = bool(config.algo.ddpm.enabled)
    ddim_enabled = bool(config.algo.ddim.enabled)
    scheduler_name = "ddpm" if ddpm_enabled else "ddim" if ddim_enabled else "unknown"
    scheduler_config = config.algo.ddpm if ddpm_enabled else config.algo.ddim
    variable_state = ckpt_dict.get("variable_state", {}) or {}
    return {
        "checkpoint_epoch": int(variable_state.get("epoch", -1)),
        "checkpoint_global_step": int(variable_state.get("global_step", -1))
        if "global_step" in variable_state
        else None,
        "observation_horizon": int(horizon.observation_horizon),
        "action_horizon": int(horizon.action_horizon),
        "prediction_horizon": int(horizon.prediction_horizon),
        "execution_horizon": int(horizon.action_horizon),
        "actor_proposal_horizon": int(horizon.prediction_horizon),
        "queued_action_horizon": int(horizon.action_horizon),
        "replan_every_env_step": False,
        "dp_scheduler": scheduler_name,
        "dp_num_train_timesteps": int(scheduler_config.num_train_timesteps),
        "dp_num_inference_timesteps": int(scheduler_config.num_inference_timesteps),
        "dp_beta_schedule": str(scheduler_config.beta_schedule),
        "dp_clip_sample": bool(scheduler_config.clip_sample),
        "dp_prediction_type": str(scheduler_config.prediction_type),
        "dp_ema_enabled": bool(config.algo.ema.enabled),
    }


def configure_success_conditioning(
    actor_algo,
    checkpoint_path: Path,
    *,
    require_adapter: bool,
    forbid_adapter: bool,
    inference_condition: float,
    inference_condition_mask: float,
) -> dict[str, Any]:
    """Validate and configure a diffusion actor's condition adapter."""
    train_policy = actor_algo.nets["policy"]
    ema_policy = (
        actor_algo.ema.averaged_model["policy"]
        if actor_algo.ema is not None
        else None
    )
    adapter_in_nets = "condition_adapter" in train_policy
    adapter_in_ema = (
        ema_policy is not None and "condition_adapter" in ema_policy
    )
    if ema_policy is not None and adapter_in_nets != adapter_in_ema:
        raise RuntimeError(
            "condition adapter must be present in both actor nets and EMA: "
            f"{checkpoint_path}"
        )
    adapter_active = adapter_in_ema if ema_policy is not None else adapter_in_nets
    if forbid_adapter and adapter_active:
        raise ValueError(
            "unconditioned evaluation forbids a success condition adapter, "
            f"but checkpoint {checkpoint_path} contains one"
        )
    if require_adapter and not adapter_active:
        raise ValueError(
            "conditioned evaluation requires a DP checkpoint with a success "
            f"condition adapter: {checkpoint_path}"
        )
    if adapter_active and not hasattr(
        actor_algo, "set_inference_success_condition"
    ):
        if require_adapter:
            raise RuntimeError(
                "loaded DP implementation cannot set inference success "
                "conditions"
            )
    elif adapter_active:
        actor_algo.set_inference_success_condition(
            success_condition=inference_condition,
            condition_mask=inference_condition_mask,
        )
    return {
        "success_condition_adapter_in_nets": bool(adapter_in_nets),
        "success_condition_adapter_in_ema": bool(adapter_in_ema),
        "success_condition_adapter_active": bool(adapter_active),
        "success_condition_adapter_forbidden": bool(forbid_adapter),
        "success_conditioning_applied": bool(adapter_active),
        "inference_success_condition": (
            float(inference_condition) if adapter_active else None
        ),
        "inference_condition_mask": (
            float(inference_condition_mask) if adapter_active else None
        ),
    }


class PlainDPPolicy:
    """Standard robomimic DiffusionPolicy evaluation wrapper.

    This path intentionally calls the RolloutPolicy directly, so the policy uses
    its internal action queue: predict Tp actions, enqueue Ta executable actions,
    execute Ta actions, then replan. It is the standard DP behavior, unlike the
    first-action IDQL ablation that replans every environment step.
    """

    def __init__(
        self,
        dp_policy,
        dp_ckpt: dict,
        checkpoint_path: Path,
        *,
        require_success_condition_adapter: bool = False,
        forbid_success_condition_adapter: bool = False,
        inference_success_condition: float = 1.0,
        inference_condition_mask: float = 1.0,
    ):
        self.dp_policy = dp_policy
        self.dp_ckpt = dp_ckpt
        self.algo = dp_policy.policy
        condition_metadata = configure_success_conditioning(
            self.algo,
            checkpoint_path,
            require_adapter=require_success_condition_adapter,
            forbid_adapter=forbid_success_condition_adapter,
            inference_condition=inference_success_condition,
            inference_condition_mask=inference_condition_mask,
        )
        metadata = dp_eval_metadata(dp_ckpt)
        self.checkpoint = {
            "pretrained_dp_checkpoint": str(checkpoint_path),
            "plain_dp_checkpoint": str(checkpoint_path),
            "step": metadata["checkpoint_global_step"]
            if metadata["checkpoint_global_step"] is not None
            else metadata["checkpoint_epoch"],
            "eval_actor_key": "plain_dp_checkpoint.ema_or_nets",
            "eval_critic_key": None,
            "plain_dp_policy": True,
            **condition_metadata,
            **metadata,
        }
        self.last_q: np.ndarray | None = None
        self.last_v: float | None = None
        self.last_adv: np.ndarray | None = None
        self.last_selected_index: int | None = None
        self.last_selection_is_random: bool | None = None

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None
        self.last_selection_is_random = None

    def __call__(self, ob) -> np.ndarray:
        self.last_q = None
        self.last_v = None
        self.last_adv = None
        self.last_selected_index = None
        self.last_selection_is_random = None
        return np.asarray(self.dp_policy(ob), dtype=np.float64).copy()


def resolve_base_dp_checkpoint(checkpoint: dict, args) -> Path:
    """Choose the DP checkpoint used to construct proposal actors and the env."""
    embedded = Path(checkpoint["pretrained_dp_checkpoint"])
    if args.actor_source == "external_dp_chunk_critic":
        return Path(args.dp_checkpoint)
    if args.actor_source != "hybrid_dp_chunk_actor":
        return embedded

    override_is_explicit = bool(
        getattr(args, "dp_checkpoint_explicit", True)
    )
    if not override_is_explicit:
        return embedded
    override = getattr(args, "dp_checkpoint", None)
    if override is None:
        raise ValueError("explicit hybrid DP checkpoint is missing")
    override = Path(override)
    if not override.is_file():
        raise FileNotFoundError(
            f"explicit hybrid DP checkpoint does not exist: {override}"
        )
    if not checkpoint.get("rise_style_rgb_idql", False):
        if override.expanduser().resolve() != embedded.expanduser().resolve():
            raise ValueError(
                "legacy hybrid checkpoints do not support a DP checkpoint override"
            )
        return embedded
    return override


def observation_shape_contract(
    shape_metadata,
    *,
    label: str,
) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    """Return the ordered observation key/shape contract when metadata has one."""
    if isinstance(shape_metadata, list):
        shape_metadata = shape_metadata[0] if shape_metadata else None
    if not isinstance(shape_metadata, dict):
        return None
    all_shapes = shape_metadata.get("all_shapes")
    if not isinstance(all_shapes, dict):
        return None
    all_obs_keys = shape_metadata.get("all_obs_keys")
    keys = list(all_shapes) if all_obs_keys is None else list(all_obs_keys)
    contract = []
    for key in keys:
        if key not in all_shapes:
            raise ValueError(
                f"{label} shape_metadata lists missing observation key {key!r}"
            )
        flat_shape = np.asarray(all_shapes[key]).reshape(-1)
        contract.append(
            (str(key), tuple(int(value) for value in flat_shape.tolist()))
        )
    return tuple(contract)


def validate_rise_dp_metadata(
    dp_ckpt: dict,
    checkpoint: dict,
    *,
    composition: str,
) -> None:
    """Validate task, environment, and saved observation contracts when present."""
    actor_env = dp_ckpt.get("env_metadata")
    task = checkpoint.get("task")
    expected_env_name = TASK_ENV_NAMES.get(str(task)) if task is not None else None
    if expected_env_name is not None:
        actor_env_name = (
            actor_env.get("env_name")
            if isinstance(actor_env, dict)
            else None
        )
        if actor_env_name != expected_env_name:
            raise ValueError(
                f"{composition} task={task!r} requires DP env_name="
                f"{expected_env_name!r}, found {actor_env_name!r}"
            )

    reference_env = checkpoint.get("env_metadata")
    if isinstance(reference_env, dict):
        if not isinstance(actor_env, dict):
            raise ValueError(
                f"{composition} DP checkpoint is missing env_metadata"
            )
        for key in ("env_name", "type", "env_version"):
            if key not in reference_env:
                continue
            if actor_env.get(key) != reference_env[key]:
                raise ValueError(
                    f"{composition} environment contract differs at "
                    f"env_metadata.{key}: actor={actor_env.get(key)!r}, "
                    f"training={reference_env[key]!r}"
                )
        reference_kwargs = reference_env.get("env_kwargs")
        actor_kwargs = actor_env.get("env_kwargs")
        if isinstance(reference_kwargs, dict):
            if not isinstance(actor_kwargs, dict):
                raise ValueError(
                    f"{composition} DP checkpoint is missing env_metadata.env_kwargs"
                )
            for key, reference_value in reference_kwargs.items():
                if key in RUNTIME_ONLY_ENV_KWARGS:
                    continue
                if key not in actor_kwargs or actor_kwargs[key] != reference_value:
                    raise ValueError(
                        f"{composition} environment contract differs at "
                        f"env_metadata.env_kwargs.{key}: "
                        f"actor={actor_kwargs.get(key)!r}, "
                        f"training={reference_value!r}"
                    )

    reference_shape = observation_shape_contract(
        checkpoint.get("shape_metadata"),
        label="IDQL training DP",
    )
    if reference_shape is not None:
        actor_shape = observation_shape_contract(
            dp_ckpt.get("shape_metadata"),
            label="actor DP",
        )
        if actor_shape is None:
            raise ValueError(
                f"{composition} DP checkpoint is missing shape_metadata"
            )
        reference_keys = tuple(key for key, _ in reference_shape)
        actor_keys = tuple(key for key, _ in actor_shape)
        if actor_keys != reference_keys:
            raise ValueError(
                f"{composition} observation keys differ: "
                f"actor={actor_keys}, training={reference_keys}"
            )
        for (key, actor_dims), (_, reference_dims) in zip(
            actor_shape,
            reference_shape,
        ):
            if actor_dims != reference_dims:
                raise ValueError(
                    f"{composition} observation shape differs for {key!r}: "
                    f"actor={actor_dims}, training={reference_dims}"
                )


def validate_rise_dp_composition(
    dp_policy,
    dp_ckpt: dict,
    checkpoint: dict,
    *,
    actor_source: str,
) -> None:
    """Validate the normalized action space shared by a DP actor and critic."""
    composition = (
        "external DP composition"
        if actor_source == "external_dp_chunk_critic"
        else "hybrid DP composition"
    )
    validate_rise_dp_metadata(
        dp_ckpt,
        checkpoint,
        composition=composition,
    )
    actor_action_dim = int(dp_policy.policy.ac_dim)
    critic_action_dim = int(checkpoint["action_dim"])
    if actor_action_dim != critic_action_dim:
        raise ValueError(
            f"{composition} action dimensions differ: "
            f"actor={actor_action_dim}, critic={critic_action_dim}"
        )
    actor_action_stats = dp_ckpt.get("action_normalization_stats")
    critic_action_stats = checkpoint.get("action_normalization_stats")
    if actor_action_stats is None or critic_action_stats is None:
        raise ValueError(
            f"{composition} requires action normalization statistics "
            "in both checkpoints"
        )
    if not action_normalization_stats_match(
        actor_action_stats,
        critic_action_stats,
    ):
        raise ValueError(
            f"{composition} uses different normalized action spaces"
        )
    if checkpoint.get("rise_style_rgb_chunk_idql", False):
        actor_horizon = dp_policy.policy.algo_config.horizon
        critic_chunk_horizon = int(checkpoint["critic_chunk_horizon"])
        if int(actor_horizon.action_horizon) < critic_chunk_horizon:
            raise ValueError(
                f"{composition} actor action horizon is shorter than the "
                f"chunk critic: actor={int(actor_horizon.action_horizon)}, "
                f"critic={critic_chunk_horizon}"
            )


def configure_dp_inference(
    actor_algo,
    *,
    num_inference_steps: int,
    diffusion_clip_sample: bool,
) -> dict[str, Any]:
    """Apply and verify inference controls used by DP trajectory sampling."""
    num_inference_steps = int(num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}"
        )
    algo_config = actor_algo.algo_config
    ddpm = getattr(algo_config, "ddpm", None)
    ddim = getattr(algo_config, "ddim", None)
    enabled = [
        config
        for config in (ddpm, ddim)
        if config is not None and bool(getattr(config, "enabled", False))
    ]
    if len(enabled) != 1:
        raise RuntimeError(
            "loaded DP actor must have exactly one enabled diffusion scheduler"
        )
    inference_config = enabled[0]
    num_train_timesteps = int(inference_config.num_train_timesteps)
    if num_inference_steps > num_train_timesteps:
        raise ValueError(
            f"num_inference_steps={num_inference_steps} exceeds loaded DP "
            f"num_train_timesteps={num_train_timesteps}"
        )

    unlock = (
        algo_config.values_unlocked()
        if hasattr(algo_config, "values_unlocked")
        else nullcontext()
    )
    try:
        with unlock:
            inference_config.num_inference_timesteps = num_inference_steps
            inference_config.clip_sample = bool(diffusion_clip_sample)
    except Exception as exc:
        raise RuntimeError(
            "loaded DP config does not support inference overrides"
        ) from exc

    scheduler = getattr(actor_algo, "noise_scheduler", None)
    if scheduler is None or not hasattr(scheduler, "config"):
        raise RuntimeError("loaded DP actor has no configurable noise scheduler")
    register_to_config = getattr(scheduler, "register_to_config", None)
    if callable(register_to_config):
        register_to_config(clip_sample=bool(diffusion_clip_sample))
    elif bool(getattr(scheduler.config, "clip_sample", False)) != bool(
        diffusion_clip_sample
    ):
        raise RuntimeError(
            "loaded DP scheduler does not support --diffusion-clip-sample"
        )

    if int(inference_config.num_inference_timesteps) != num_inference_steps:
        raise RuntimeError("loaded DP actor ignored --num-inference-steps")
    if bool(getattr(scheduler.config, "clip_sample", False)) != bool(
        diffusion_clip_sample
    ):
        raise RuntimeError("loaded DP scheduler ignored --diffusion-clip-sample")
    return {
        "dp_num_inference_timesteps": num_inference_steps,
        "dp_clip_sample": bool(diffusion_clip_sample),
        "dp_inference_cli_applied": True,
    }


def load_policy(idql_checkpoint: Path, device: torch.device, args):
    if args.actor_source == "plain_dp":
        if int(args.num_candidates) != 1:
            raise ValueError("actor_source=plain_dp uses the standard DP queue; set --num-candidates 1")
        dp_policy, dp_ckpt = FileUtils.policy_from_checkpoint(
            ckpt_path=str(args.dp_checkpoint), device=device, verbose=False
        )
        dp_policy.policy.set_eval()
        return PlainDPPolicy(
            dp_policy,
            dp_ckpt,
            args.dp_checkpoint,
            require_success_condition_adapter=args.require_success_condition_adapter,
            forbid_success_condition_adapter=args.forbid_success_condition_adapter,
            inference_success_condition=args.inference_success_condition,
            inference_condition_mask=args.inference_condition_mask,
        )

    # Keep optimizer states and unused network states off the GPU. Individual
    # actor / critic modules copy only the selected weights to @device below.
    checkpoint = torch.load(idql_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_task = checkpoint.get("task")
    if args.expected_task is not None and checkpoint_task != args.expected_task:
        raise ValueError(
            f"IDQL checkpoint task mismatch: expected {args.expected_task!r}, "
            f"found {checkpoint_task!r} in {idql_checkpoint}"
        )
    if bool(checkpoint.get("actor_only", False)):
        if args.actor_source != "hybrid_dp_chunk_actor":
            raise ValueError(
                "actor-only checkpoints must evaluate their saved conditioned "
                "actor with actor_source=hybrid_dp_chunk_actor"
            )
        if int(args.num_candidates) != 1:
            raise ValueError(
                "actor-only checkpoints have no trained critic; set "
                "--num-candidates 1"
            )
    external_dp_chunk_critic = args.actor_source == "external_dp_chunk_critic"
    dp_checkpoint = resolve_base_dp_checkpoint(checkpoint, args)
    dp_policy, dp_ckpt = FileUtils.policy_from_checkpoint(
        ckpt_path=str(dp_checkpoint), device=device, verbose=False
    )
    if args.actor_source in (
        "hybrid_dp_chunk_actor",
        "external_dp_chunk_critic",
        "pretrained_dp_first_action",
    ):
        checkpoint["actor_dp_checkpoint"] = str(dp_checkpoint)

    rise_style_rgb_idql = bool(checkpoint.get("rise_style_rgb_idql", False))
    if rise_style_rgb_idql:
        if args.actor_source not in (
            "hybrid_dp_chunk_actor",
            "external_dp_chunk_critic",
        ):
            raise ValueError(
                "RISE-style RGB IDQL checkpoints must use "
                "actor_source=hybrid_dp_chunk_actor or external_dp_chunk_critic"
            )
        validate_rise_dp_composition(
            dp_policy,
            dp_ckpt,
            checkpoint,
            actor_source=args.actor_source,
        )
        checkpoint["critic_training_dp_checkpoint"] = str(
            checkpoint["pretrained_dp_checkpoint"]
        )
        if external_dp_chunk_critic:
            checkpoint["external_dp_chunk_critic"] = True
            checkpoint["actor_loaded_from_idql_checkpoint"] = False
            checkpoint["eval_actor_key"] = (
                "external_dp_checkpoint.ema"
                if dp_policy.policy.ema is not None
                else "external_dp_checkpoint.nets"
            )
        else:
            dp_policy.policy.deserialize(
                checkpoint["actor_model"],
                load_optimizers=False,
            )
            checkpoint["external_dp_chunk_critic"] = False
            checkpoint["actor_loaded_from_idql_checkpoint"] = True
            checkpoint["actor_dp_checkpoint"] = str(dp_checkpoint)
            checkpoint["eval_actor_key"] = (
                "actor_model.ema"
                if dp_policy.policy.ema is not None
                else "actor_model.nets"
            )
        actor_checkpoint_path = (
            dp_checkpoint if external_dp_chunk_critic else idql_checkpoint
        )
        checkpoint.update(
            configure_success_conditioning(
                dp_policy.policy,
                actor_checkpoint_path,
                require_adapter=args.require_success_condition_adapter,
                forbid_adapter=args.forbid_success_condition_adapter,
                inference_condition=args.inference_success_condition,
                inference_condition_mask=args.inference_condition_mask,
            )
        )
        dp_policy.policy.set_eval()
        checkpoint["rise_style_rgb_idql"] = True
        checkpoint["visual_critic_idql"] = True
        checkpoint["hybrid_dp_chunk_actor_iql"] = True
        checkpoint.update(dp_eval_metadata(dp_ckpt))
        checkpoint.update(
            configure_dp_inference(
                dp_policy.policy,
                num_inference_steps=args.num_inference_steps,
                diffusion_clip_sample=args.diffusion_clip_sample,
            )
        )

        selected_critics = torch.nn.ModuleList()
        vf = None
        wcm_q_system = None
        wcm_value_system = None
        critic_architecture = checkpoint_critic_architecture(checkpoint)
        if critic_architecture not in (
            LEGACY_CRITIC_ARCHITECTURE,
            RISE_V2_CRITIC_ARCHITECTURE,
            WCM_CRITIC_ARCHITECTURE,
        ):
            raise ValueError(
                f"unsupported critic architecture: {critic_architecture!r}"
            )
        checkpoint["critic_architecture"] = critic_architecture
        if (
            int(args.num_candidates) > 1
            and critic_architecture != WCM_CRITIC_ARCHITECTURE
        ):
            if checkpoint.get("rise_style_rgb_chunk_idql", False):
                q_uses_predicted_next = bool(
                    checkpoint.get(
                        "critic_q_use_predicted_next_latent",
                        False,
                    )
                )
                expected_q_inputs = architecture_q_head_inputs(
                    critic_architecture,
                    q_uses_predicted_next,
                )
                saved_q_inputs = tuple(
                    checkpoint.get(
                        "critic_q_head_inputs",
                        expected_q_inputs,
                    )
                )
                if saved_q_inputs != expected_q_inputs:
                    raise ValueError(
                        "chunk checkpoint Q-head metadata is inconsistent: "
                        f"{saved_q_inputs!r} != {expected_q_inputs!r}"
                    )
                if (
                    q_uses_predicted_next
                    and checkpoint.get(
                        "critic_q_predicted_next_normalization"
                    )
                    != PREDICTED_NEXT_Q_NORMALIZATION
                ):
                    raise ValueError(
                        "chunk checkpoint uses an unsupported predicted-next "
                        "Q normalization"
                    )
                common_chunk_kwargs = {
                    "chunk_horizon": int(checkpoint["critic_chunk_horizon"]),
                    "hidden_dims": tuple(
                        int(value) for value in checkpoint["critic_hidden_dims"]
                    ),
                    "latent_dim": int(checkpoint.get("critic_latent_dim", 300)),
                    "action_hidden_dim": int(
                        checkpoint.get("critic_action_hidden_dim", 128)
                    ),
                    "num_attention_heads": int(
                        checkpoint.get("critic_num_attention_heads", 4)
                    ),
                    "num_action_conv_layers": int(
                        checkpoint.get("critic_num_action_conv_layers", 2)
                    ),
                    "dropout": float(checkpoint.get("critic_dropout", 0.0)),
                    "num_critics": int(checkpoint.get("num_critics", 2)),
                    "critic_group_norm": bool(
                        checkpoint.get("critic_group_norm", False)
                    ),
                    "late_fusion_key": checkpoint.get(
                        "critic_late_fusion_key", "robot0_gripper_qpos"
                    ),
                    "observation_horizon": int(
                        checkpoint.get("critic_observation_horizon", 1)
                    ),
                }
                if critic_architecture == RISE_V2_CRITIC_ARCHITECTURE:
                    critics, critic_targets, vf = (
                        make_rise_v2_system_from_checkpoint(
                            dp_policy.policy,
                            checkpoint,
                        )
                    )
                else:
                    critics, critic_targets, vf = make_rise_chunk_value_networks(
                        dp_policy.policy,
                        **common_chunk_kwargs,
                        q_use_predicted_next_latent=q_uses_predicted_next,
                    )
            elif bool(checkpoint.get("stacked_pretrained_dql_critic", False)):
                critics, critic_targets, vf = make_dql_value_networks(
                    dp_policy.policy,
                    hidden_dims=tuple(
                        int(value) for value in checkpoint["critic_hidden_dims"]
                    ),
                    observation_horizon=int(
                        checkpoint["critic_observation_horizon"]
                    ),
                    num_critics=int(checkpoint.get("num_critics", 2)),
                    late_fusion_key=checkpoint.get(
                        "critic_late_fusion_key", "robot0_gripper_qpos"
                    ),
                )
            else:
                critics, critic_targets, vf = make_rise_value_networks(
                    dp_policy.policy,
                    hidden_dims=tuple(
                        int(value) for value in checkpoint["critic_hidden_dims"]
                    ),
                    num_critics=int(checkpoint.get("num_critics", 2)),
                    critic_group_norm=bool(
                        checkpoint.get("critic_group_norm", False)
                    ),
                    late_fusion_key=checkpoint.get(
                        "critic_late_fusion_key", "robot0_gripper_qpos"
                    ),
                )
            if args.critic_source == "target":
                selected_critics = critic_targets
                critic_states = checkpoint["critic_targets"]
                checkpoint["eval_critic_key"] = "critic_targets"
                del critics
            else:
                selected_critics = critics
                critic_states = checkpoint["critics"]
                checkpoint["eval_critic_key"] = "critics"
                del critic_targets
            if len(selected_critics) != len(critic_states):
                raise ValueError(
                    f"checkpoint has {len(critic_states)} critic states but model has "
                    f"{len(selected_critics)} critics"
                )
            critic_encoder_normalization = []
            for critic, state in zip(selected_critics, critic_states):
                critic_encoder_normalization.append(
                    match_encoder_normalization_to_checkpoint(critic, state)
                )
                critic.load_state_dict(state)
            checkpoint["eval_critic_encoder_normalization"] = (
                critic_encoder_normalization
            )
            selected_critics = selected_critics.float().to(device)
            selected_critics.eval().requires_grad_(False)
            if checkpoint.get("critic_has_value_net", "vf" in checkpoint):
                checkpoint["eval_vf_encoder_normalization"] = (
                    match_encoder_normalization_to_checkpoint(
                        vf,
                        checkpoint["vf"],
                    )
                )
                vf.load_state_dict(checkpoint["vf"])
                vf = vf.float().to(device)
                vf.eval().requires_grad_(False)
            else:
                vf = None
                checkpoint["eval_vf_encoder_normalization"] = None
        elif int(args.num_candidates) > 1:
            online_system, target_system = make_wcm_system_from_checkpoint(
                dp_policy.policy,
                checkpoint,
            )
            online_state = checkpoint.get("chunk_value_system")
            target_state = checkpoint.get("chunk_value_target")
            if online_state is None or target_state is None:
                raise ValueError(
                    "WCM checkpoint is missing online or target system state"
                )
            online_encoder_normalization = (
                match_encoder_normalization_to_checkpoint(
                    online_system, online_state
                )
            )
            online_system.load_state_dict(online_state, strict=True)
            if args.critic_source == "target":
                target_encoder_normalization = (
                    match_encoder_normalization_to_checkpoint(
                        target_system, target_state
                    )
                )
                target_system.load_state_dict(target_state, strict=True)
                wcm_q_system = target_system
                checkpoint["eval_critic_key"] = "chunk_value_target"
                checkpoint["eval_critic_encoder_normalization"] = [
                    target_encoder_normalization
                ]
            else:
                wcm_q_system = online_system
                checkpoint["eval_critic_key"] = "chunk_value_system"
                checkpoint["eval_critic_encoder_normalization"] = [
                    online_encoder_normalization
                ]
            wcm_value_system = online_system
            checkpoint["eval_vf_key"] = "chunk_value_system.value_head"
            wcm_q_system = wcm_q_system.float().to(device)
            wcm_q_system.eval().requires_grad_(False)
            if wcm_value_system is not wcm_q_system:
                wcm_value_system = wcm_value_system.float().to(device)
                wcm_value_system.eval().requires_grad_(False)
            checkpoint["eval_vf_encoder_normalization"] = (
                online_encoder_normalization
            )
        else:
            checkpoint["eval_critic_key"] = None
            checkpoint["eval_vf_key"] = None

        for heavy_key in (
            "actor_model",
            "critics",
            "critic_targets",
            "dynamics_targets",
            "dynamics_target_encoder",
            "vf",
            "critic_optimizers",
            "vf_optimizer",
            "critic_lr_schedulers",
            "critic_optimizer",
            "critic_lr_scheduler",
            "vf_lr_scheduler",
            "chunk_value_system",
            "chunk_value_target",
            "wcm_dynamics_frame_target",
            "chunk_value_optimizer",
            "chunk_value_lr_scheduler",
            "rng_state",
            "loader_generator_state",
            "history",
        ):
            checkpoint.pop(heavy_key, None)
        return RiseStyleRGBIDQLPolicy(
            dp_policy,
            dp_ckpt,
            selected_critics,
            vf,
            checkpoint,
            num_candidates=args.num_candidates,
            candidate_batch_size=args.candidate_batch_size,
            selection=args.selection,
            softmax_temperature=args.softmax_temperature,
            random_selection_probability=args.random_selection_probability,
            selection_seed=effective_policy_seed(args),
            clip_actions=args.clip_actions,
            execution_horizon=args.execution_horizon,
            wcm_q_system=wcm_q_system,
            wcm_value_system=wcm_value_system,
        )

    hybrid_dp_chunk_actor_iql = bool(checkpoint.get("hybrid_dp_chunk_actor_iql", False))
    if args.actor_source == "hybrid_dp_chunk_actor":
        if not hybrid_dp_chunk_actor_iql:
            raise ValueError(
                "actor_source=hybrid_dp_chunk_actor requires a special checkpoint."
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
            dp_ckpt,
            critic,
            checkpoint,
            critic_obs_encoder=critic_obs_encoder,
            num_candidates=args.num_candidates,
            candidate_batch_size=args.candidate_batch_size,
            selection=args.selection,
            softmax_temperature=args.softmax_temperature,
            random_selection_probability=args.random_selection_probability,
            selection_seed=effective_policy_seed(args),
            clip_actions=args.clip_actions,
            execution_horizon=args.execution_horizon,
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
        random_selection_probability=args.random_selection_probability,
        selection_seed=effective_policy_seed(args),
        clip_actions=args.clip_actions,
        diffusion_clip_sample=args.diffusion_clip_sample,
    )


def rollout(
    policy,
    env,
    horizon: int,
    return_obs: bool = False,
    video_writer=None,
    video_skip: int = 5,
    camera_names=None,
    *,
    reset_to_initial_state: bool = False,
    record_trajectory: bool = False,
):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    policy.start_episode()
    obs = env.reset()
    initial_state_dict = {}
    state = None
    if reset_to_initial_state or record_trajectory:
        state_dict = env.get_state()
        state = np.asarray(state_dict["states"]).copy()
        if record_trajectory:
            initial_state_dict = state_dict
        if reset_to_initial_state:
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
        selection_is_random=[],
        selection_is_greedy=[],
        initial_state_dict=initial_state_dict,
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
            selection_is_random = policy.last_selection_is_random
            if record_trajectory:
                traj["actions"].append(action)
                traj["rewards"].append(float(reward))
                traj["dones"].append(bool(done))
                traj["states"].append(state.copy())
            if q is None or selected is None:
                traj["q_selected"].append(np.nan)
                traj["q_mean"].append(np.nan)
                traj["q_max"].append(np.nan)
                traj["q_min"].append(np.nan)
                traj["q_margin"].append(np.nan)
                traj["q_range"].append(np.nan)
                traj["v"].append(np.nan)
                traj["selected_index"].append(-1)
                traj["selection_is_random"].append(-1)
                traj["selection_is_greedy"].append(-1)
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
                traj["selection_is_random"].append(
                    int(bool(selection_is_random))
                )
                traj["selection_is_greedy"].append(
                    int(int(selected) == int(np.argmax(q)))
                )
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)
            if done or success:
                break
            obs = deepcopy(next_obs)
            if record_trajectory:
                state = current_sim_state(env)
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
        random_decisions = traj["selection_is_random"][valid_selection]
        greedy_decisions = traj["selection_is_greedy"][valid_selection]
        stats["Num_Selection_Decisions"] = int(np.sum(valid_selection))
        stats["Random_Selection_Decision_Fraction"] = float(
            np.mean(random_decisions == 1)
        )
        stats["Non_Greedy_Selection_Decision_Fraction"] = float(
            np.mean(greedy_decisions == 0)
        )
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
    seed_metadata = evaluation_seed_metadata(args)
    standard_idql_actor = args.actor_source in ("idql_one_step_mlp", "idql_target_one_step_mlp")
    pretrained_dp_actor = args.actor_source == "pretrained_dp_first_action"
    hybrid_dp_chunk_actor = args.actor_source == "hybrid_dp_chunk_actor"
    external_dp_chunk_critic = args.actor_source == "external_dp_chunk_critic"
    plain_dp_actor = args.actor_source == "plain_dp"
    rise_style_rgb_idql = bool(policy.checkpoint.get("rise_style_rgb_idql", False))
    return {
        "idql_checkpoint": None if plain_dp_actor else str(args.idql_checkpoint),
        "task": policy.checkpoint.get("task"),
        "dp_checkpoint": (
            str(args.dp_checkpoint)
            if plain_dp_actor
            else policy.checkpoint.get("actor_dp_checkpoint")
        ),
        "pretrained_dp_checkpoint": str(policy.checkpoint["pretrained_dp_checkpoint"]),
        "actor_dp_checkpoint": policy.checkpoint.get("actor_dp_checkpoint"),
        "critic_training_dp_checkpoint": policy.checkpoint.get(
            "critic_training_dp_checkpoint"
        ),
        "checkpoint_epoch": int(policy.checkpoint.get("epoch", -1)),
        "checkpoint_step": int(policy.checkpoint.get("step", -1)),
        "actor_source": args.actor_source,
        "eval_actor_key": policy.checkpoint.get("eval_actor_key", "actor"),
        "critic_source": None if plain_dp_actor else args.critic_source,
        "eval_critic_key": policy.checkpoint.get("eval_critic_key", None if plain_dp_actor else "critic"),
        "visual_critic_idql": bool(policy.checkpoint.get("visual_critic_idql", False)),
        "hybrid_dp_chunk_actor_iql": bool(policy.checkpoint.get("hybrid_dp_chunk_actor_iql", False)),
        "rise_style_rgb_idql": rise_style_rgb_idql,
        "rise_style_rgb_chunk_idql": bool(
            policy.checkpoint.get("rise_style_rgb_chunk_idql", False)
        ),
        "critic_chunk_horizon": int(policy.checkpoint.get("critic_chunk_horizon", 1)),
        "critic_observation_horizon": int(
            policy.checkpoint.get("critic_observation_horizon", 1)
        ),
        "critic_architecture": policy.checkpoint.get(
            "critic_architecture", LEGACY_CRITIC_ARCHITECTURE
        ),
        "critic_shared_state_representation": bool(
            policy.checkpoint.get("critic_shared_state_representation", False)
        ),
        "critic_q_head_inputs": list(
            policy.checkpoint.get(
                "critic_q_head_inputs",
                ("context", "action_repr"),
            )
        ),
        "critic_q_use_predicted_next_latent": bool(
            policy.checkpoint.get(
                "critic_q_use_predicted_next_latent",
                False,
            )
        ),
        "critic_q_predicted_next_normalization": policy.checkpoint.get(
            "critic_q_predicted_next_normalization"
        ),
        "dynamics_prediction_offsets": list(
            policy.checkpoint.get("dynamics_prediction_offsets", ())
        ),
        "dynamics_prediction_consumed_by_q": bool(
            policy.checkpoint.get("dynamics_prediction_consumed_by_q", False)
        ),
        "rise_v2_fusion_mode": policy.checkpoint.get("rise_v2_fusion_mode"),
        "rise_v2_dense_dynamics": policy.checkpoint.get(
            "rise_v2_dense_dynamics"
        ),
        "sigreg_weight": float(policy.checkpoint.get("sigreg_weight", 0.0)),
        "sigreg_global_batch": bool(
            policy.checkpoint.get("sigreg_global_batch", False)
        ),
        "eval_vf_key": policy.checkpoint.get("eval_vf_key"),
        "critic_input_mode": policy.checkpoint.get("critic_input_mode"),
        "critic_action_space": policy.checkpoint.get("critic_action_space"),
        "critic_late_fusion_key": policy.checkpoint.get("critic_late_fusion_key"),
        "critic_group_norm": policy.checkpoint.get("critic_group_norm"),
        "reward_definition": policy.checkpoint.get("reward_definition"),
        "single_dataloader_training": policy.checkpoint.get("single_dataloader"),
        "plain_dp_policy": bool(plain_dp_actor),
        "eval_actor_encoder_key": policy.checkpoint.get("eval_actor_encoder_key"),
        "eval_critic_encoder_key": policy.checkpoint.get("eval_critic_encoder_key"),
        "aux_next_pred_enabled": bool(policy.checkpoint.get("aux_next_pred_enabled", False)),
        "aux_next_pred_weight": float(policy.checkpoint.get("aux_next_pred_weight", 0.0) or 0.0),
        "aux_next_pred_mode": str(policy.checkpoint.get("aux_next_pred_mode", "delta")),
        "actor_uses_pretrained_dp_weights": bool(
            pretrained_dp_actor
            or hybrid_dp_chunk_actor
            or external_dp_chunk_critic
        ),
        "actor_loaded_from_idql_checkpoint": policy.checkpoint.get(
            "actor_loaded_from_idql_checkpoint"
        ),
        "external_dp_chunk_critic": bool(external_dp_chunk_critic),
        "standard_idql_trained_actor": bool(standard_idql_actor),
        "dp_chunk_actor_bc_trained": bool(hybrid_dp_chunk_actor),
        "standard_dp_policy_eval": bool(plain_dp_actor),
        "pretrained_checkpoint_used_for_encoder": True,
        "num_candidates": args.num_candidates,
        "num_inference_steps": int(policy.checkpoint.get("dp_num_inference_timesteps", args.num_inference_steps)),
        "selection": None if plain_dp_actor else args.selection,
        "random_selection_probability": (
            None if plain_dp_actor else args.random_selection_probability
        ),
        "selection_seed": (
            None if plain_dp_actor else seed_metadata["effective_policy_seed"]
        ),
        "requested_selection_seed": (
            None if plain_dp_actor else seed_metadata["requested_policy_seed"]
        ),
        "clip_actions": None if plain_dp_actor else args.clip_actions,
        "diffusion_clip_sample": bool(policy.checkpoint.get("dp_clip_sample", args.diffusion_clip_sample)),
        "observation_horizon": policy.checkpoint.get("observation_horizon"),
        "action_horizon": policy.checkpoint.get("action_horizon"),
        "prediction_horizon": policy.checkpoint.get("prediction_horizon"),
        "queued_action_horizon": policy.checkpoint.get("queued_action_horizon"),
        "dp_scheduler": policy.checkpoint.get("dp_scheduler"),
        "dp_num_train_timesteps": policy.checkpoint.get("dp_num_train_timesteps"),
        "dp_num_inference_timesteps": policy.checkpoint.get("dp_num_inference_timesteps"),
        "dp_beta_schedule": policy.checkpoint.get("dp_beta_schedule"),
        "dp_clip_sample": policy.checkpoint.get("dp_clip_sample"),
        "dp_inference_cli_applied": bool(
            policy.checkpoint.get("dp_inference_cli_applied", False)
        ),
        "dp_prediction_type": policy.checkpoint.get("dp_prediction_type"),
        "dp_ema_enabled": policy.checkpoint.get("dp_ema_enabled"),
        "success_condition_adapter_in_nets": policy.checkpoint.get("success_condition_adapter_in_nets", False),
        "success_condition_adapter_in_ema": policy.checkpoint.get("success_condition_adapter_in_ema", False),
        "success_condition_adapter_active": policy.checkpoint.get("success_condition_adapter_active", False),
        "success_condition_adapter_forbidden": policy.checkpoint.get(
            "success_condition_adapter_forbidden",
            False,
        ),
        "success_conditioning_applied": policy.checkpoint.get("success_conditioning_applied", False),
        "inference_success_condition": policy.checkpoint.get("inference_success_condition"),
        "inference_condition_mask": policy.checkpoint.get("inference_condition_mask"),
        "paper_faithful_one_step_idql": bool(standard_idql_actor),
        "rise_style_reference_equations": rise_style_rgb_idql,
        "pretrained_dp_first_action_baseline": bool(pretrained_dp_actor and args.num_candidates == 1),
        "pretrained_dp_proposal_idql_critic_rerank": bool(pretrained_dp_actor and args.num_candidates > 1),
        "hybrid_dp_chunk_actor_actor_only": bool(hybrid_dp_chunk_actor and args.num_candidates == 1),
        "hybrid_dp_chunk_actor_critic_rerank": bool(hybrid_dp_chunk_actor and args.num_candidates > 1),
        "external_dp_actor_only": bool(
            external_dp_chunk_critic and args.num_candidates == 1
        ),
        "external_dp_chunk_critic_rerank": bool(
            external_dp_chunk_critic and args.num_candidates > 1
        ),
        "actor_proposal_horizon": int(policy.checkpoint.get("actor_proposal_horizon", policy.checkpoint.get("actor_action_horizon", 1))),
        "execution_horizon": int(policy.checkpoint.get("execution_horizon", 1)),
        "replan_every_env_step": bool(policy.checkpoint.get("replan_every_env_step", True)),
        "critic_used_for_action_selection": bool((not plain_dp_actor) and args.num_candidates > 1),
        **seed_metadata,
        "n_rollouts": args.n_rollouts,
        "completed_rollouts": len(stats),
        "complete": bool(complete),
        "horizon": args.horizon,
        "env_hard_reset": bool(args.env_hard_reset),
        "reset_to_initial_state": bool(args.reset_to_initial_state),
        "trajectory_collection_enabled": args.dataset_path is not None,
        "trajectory_dataset_overwrite": bool(
            getattr(args, "overwrite_dataset", False)
        ),
        "average_rollout_stats": avg,
        "wilson_95_interval": wilson(successes, len(stats)),
        "rollouts": stats,
    }


def write_summary(args, policy, stats: list[dict], complete: bool, suffix: str = "") -> Path:
    summary = build_summary(args, policy, stats, complete=complete)
    if args.env_seed is None and args.policy_seed is None:
        seed_label = f"seed{args.seed}"
    else:
        env_seed = args.env_seed if args.env_seed is not None else args.seed
        policy_seed = args.policy_seed if args.policy_seed is not None else args.seed
        seed_label = f"env{env_seed}_policy{policy_seed}"
    path = args.output_dir / f"one_step_idql_N{args.num_candidates}_{seed_label}{suffix}.json"
    atomic_write_json(path, summary)
    return path


def evaluate(args) -> dict:
    seed_metadata = evaluation_seed_metadata(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    dataset_writer = None
    dataset_temporary_path = None
    overwrite_dataset = bool(getattr(args, "overwrite_dataset", False))
    if args.dataset_path is not None:
        ensure_dataset_target_available(
            args.dataset_path,
            overwrite=overwrite_dataset,
        )
    try:
        device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")
        env_seed = int(seed_metadata["effective_env_seed"])
        policy_seed = int(seed_metadata["effective_policy_seed"])
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.manual_seed(policy_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(policy_seed)
        policy = load_policy(args.idql_checkpoint, device, args)
        dp_ckpt = getattr(policy, "dp_ckpt", None)
        if dp_ckpt is None:
            _, dp_ckpt = FileUtils.policy_from_checkpoint(
                ckpt_path=str(policy.checkpoint["pretrained_dp_checkpoint"]),
                device=device,
                verbose=False,
            )
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=dp_ckpt,
            render=False,
            render_offscreen=args.video_dir is not None,
            verbose=False,
        )
        configured_hard_reset = configure_env_hard_reset(env, args.env_hard_reset)
        print(
            "[eval lifecycle] "
            f"env_hard_reset={configured_hard_reset} "
            f"reset_to_initial_state={args.reset_to_initial_state} "
            f"record_trajectory={args.dataset_path is not None}",
            flush=True,
        )
        imageio = None
        if args.video_dir is not None:
            import imageio.v2 as imageio

            args.video_dir.mkdir(parents=True, exist_ok=True)

        data_group = None
        total_samples = 0
        if args.dataset_path is not None:
            import h5py

            dataset_temporary_path = make_temporary_dataset_path(
                args.dataset_path,
                overwrite=overwrite_dataset,
            )
            dataset_writer = h5py.File(dataset_temporary_path, "w")
            data_group = dataset_writer.create_group("data")
            data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
            data_group.attrs["selection"] = args.selection
            data_group.attrs["random_selection_probability"] = float(
                args.random_selection_probability
            )
            data_group.attrs["seed_normalization_scheme"] = (
                seed_metadata["seed_normalization_scheme"]
            )
            data_group.attrs["seed_normalization_modulus"] = int(
                seed_metadata["seed_normalization_modulus"]
            )
            data_group.attrs["requested_env_seed"] = int(
                seed_metadata["requested_env_seed"]
            )
            data_group.attrs["requested_policy_seed"] = int(
                seed_metadata["requested_policy_seed"]
            )
            data_group.attrs["effective_env_seed"] = env_seed
            data_group.attrs["effective_policy_seed"] = policy_seed

        stats = []
        for i in range(args.n_rollouts):
            writer = None
            try:
                if args.video_dir is not None and i < args.num_videos:
                    writer = imageio.get_writer(
                        args.video_dir / f"rollout_{i:03d}.mp4",
                        fps=20,
                    )
                rollout_stats, traj = rollout(
                    policy,
                    env,
                    horizon=args.horizon,
                    return_obs=False,
                    video_writer=writer,
                    video_skip=args.video_skip,
                    camera_names=args.camera_names,
                    reset_to_initial_state=args.reset_to_initial_state,
                    record_trajectory=data_group is not None,
                )
            finally:
                if writer is not None:
                    writer.close()

            stats.append(rollout_stats)
            write_summary(args, policy, stats, complete=False, suffix="_partial")
            print(
                f"rollout={i} success={rollout_stats['Success_Rate']:.0f} "
                f"return={rollout_stats['Return']:.3f} horizon={rollout_stats['Horizon']} "
                f"partial_success={aggregate(stats)['Success_Rate']:.3f}",
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
                    "selection_is_random",
                    "selection_is_greedy",
                ):
                    ep.create_dataset(key, data=traj[key])
                if "model" in traj["initial_state_dict"]:
                    ep.attrs["model_file"] = traj["initial_state_dict"]["model"]
                if "ep_meta" in traj["initial_state_dict"]:
                    ep.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
                ep.attrs["num_samples"] = int(traj["actions"].shape[0])
                ep.attrs["success"] = float(rollout_stats["Success_Rate"])
                ep.attrs["episode_return"] = float(rollout_stats["Return"])
                ep.attrs["env_seed"] = int(seed_metadata["requested_env_seed"])
                ep.attrs["policy_seed"] = int(
                    seed_metadata["requested_policy_seed"]
                )
                ep.attrs["effective_env_seed"] = env_seed
                ep.attrs["effective_policy_seed"] = policy_seed
                ep.attrs["selection"] = args.selection
                ep.attrs["random_selection_probability"] = float(
                    args.random_selection_probability
                )
                total_samples += int(traj["actions"].shape[0])
                data_group.attrs["total"] = total_samples
                dataset_writer.flush()

        if dataset_writer is not None:
            data_group.attrs["total"] = total_samples
            dataset_writer.close()
            dataset_writer = None

        path = write_summary(args, policy, stats, complete=True, suffix="")
        summary = json.loads(path.read_text())
        if dataset_temporary_path is not None:
            publish_dataset(
                dataset_temporary_path,
                args.dataset_path,
                overwrite=overwrite_dataset,
            )
            dataset_temporary_path = None
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Wrote {path}", flush=True)
        return summary
    finally:
        if dataset_writer is not None:
            try:
                dataset_writer.close()
            except Exception as exc:
                print(f"WARNING: failed to close rollout dataset: {exc}", flush=True)
        if dataset_temporary_path is not None:
            try:
                dataset_temporary_path.unlink(missing_ok=True)
            except Exception as exc:
                print(
                    f"WARNING: failed to remove temporary dataset: {exc}",
                    flush=True,
                )
        if env is not None:
            try:
                close_rollout_env(env)
            except Exception as exc:
                print(f"WARNING: failed to close rollout environment: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idql-checkpoint", type=Path, default=DEFAULT_IDQL)
    parser.add_argument("--dp-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--expected-task",
        choices=("square", "can", "transport", "tool_hang"),
        default=None,
        help="Fail before rollout if the IDQL checkpoint belongs to another task.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Seed Python and NumPy for environment randomization; defaults to --seed.",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Seed PyTorch and CUDA for stochastic policy sampling; defaults to --seed.",
    )
    parser.add_argument(
        "--env-hard-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rebuild the MuJoCo simulator and EGL context on every env.reset(). "
            "Disabled by default because task placement is still randomized by soft reset."
        ),
    )
    parser.add_argument(
        "--reset-to-initial-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the legacy robomimic reset_to(full model XML) after env.reset(). "
            "This is only needed for deterministic trajectory playback."
        ),
    )
    parser.add_argument(
        "--actor-source",
        choices=(
            "idql_target_one_step_mlp",
            "idql_one_step_mlp",
            "pretrained_dp_first_action",
            "hybrid_dp_chunk_actor",
            "external_dp_chunk_critic",
            "plain_dp",
        ),
        default="hybrid_dp_chunk_actor",
    )
    parser.add_argument("--critic-source", choices=("target", "online"), default="online")
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help="For DP-proposal actors, execute this many actions from the selected trajectory before replanning.",
    )
    parser.add_argument("--selection", choices=SELECTION_CHOICES, default="argmax")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument(
        "--random-selection-probability",
        type=float,
        default=0.0,
        help=(
            "For epsilon_greedy selection, choose uniformly among all candidates "
            "with this probability and otherwise choose argmax min(Q1,Q2)."
        ),
    )
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-clip-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--require-success-condition-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--forbid-success-condition-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--inference-success-condition", type=float, default=1.0)
    parser.add_argument("--inference-condition-mask", type=float, default=1.0)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="Atomically replace an existing --dataset-path only after success.",
    )

    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--camera-names", type=str, nargs="+", default=("agentview", "robot0_eye_in_hand"))
    args = parser.parse_args()
    args.dp_checkpoint_explicit = args.dp_checkpoint is not None
    if args.dp_checkpoint is None:
        args.dp_checkpoint = DEFAULT_DP
    if args.overwrite_dataset and args.dataset_path is None:
        parser.error("--overwrite-dataset requires --dataset-path")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    try:
        evaluation_seed_metadata(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.require_success_condition_adapter and args.forbid_success_condition_adapter:
        parser.error(
            "--require-success-condition-adapter and "
            "--forbid-success-condition-adapter are mutually exclusive"
        )
    if not 0.0 <= args.random_selection_probability <= 1.0:
        parser.error("--random-selection-probability must be in [0, 1]")
    if (
        args.selection != "epsilon_greedy"
        and args.random_selection_probability != 0.0
    ):
        parser.error(
            "--random-selection-probability is only valid with "
            "--selection epsilon_greedy"
        )
    for key in ("idql_checkpoint", "dp_checkpoint", "output_dir", "dataset_path", "video_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    evaluate(args)


if __name__ == "__main__":
    main()
