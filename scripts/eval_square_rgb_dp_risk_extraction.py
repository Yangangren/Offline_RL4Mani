#!/usr/bin/env python3
"""Evaluate outcome-guided extraction from a frozen Square RGB DiffusionPolicy.

At each DP decision boundary, this policy samples N full 16-slot trajectories
from the frozen pretrained DiffusionPolicy, scores them with a learned causal
prefix outcome model, and executes the selected future action chunk.

This is intentionally not IDQL: the outcome model was trained from rollout-level
success / failure outcomes, not Bellman backups. It is a deployment-time
candidate selector. For the original failure-risk model:

    a* = argmin_a positive_action_risk(o_{<=t}, a)

For a turned-over success model:

    a* = argmax_a positive_action_advantage(o_{<=t}, a)

The DP prior still matters because all candidates come from the frozen DP.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
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
from robomimic.models.prefix_risk_nets import CausalPrefixRisk, make_causal_prefix_model
from robomimic.utils.rgb_critic_utils import (
    build_rgb_encoder_from_critic_spec,
    prepare_observation_for_rgb_critic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1"
    / "20260629231002/last.pth"
)
DEFAULT_RISK = (
    ROOT
    / "trained_models/square_rgb_dp_causal_prefix_risk"
    / "epoch190_two_stage_temporal_safe_anchor/best.pt"
)
DEFAULT_OUTPUT = ROOT / "rollouts/square_rgb_dp/risk_extraction_eval"


def resolve_checkpoint_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def clone_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in obs.items()}


def repeat_obs(obs: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in obs.items():
        reps = [batch_size] + [1] * (value.ndim - 1)
        out[key] = value.repeat(*reps)
    return out


def obs_for_policy_encoder(algo, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prepared = clone_obs(obs)
    for key in algo.obs_shapes:
        if prepared[key].ndim - 1 == len(algo.obs_shapes[key]):
            prepared[key] = prepared[key].unsqueeze(1)
    return prepared


@torch.no_grad()
def encode_pair_feature(
    dp_policy,
    previous_ob: dict | None,
    current_ob: dict,
) -> torch.Tensor:
    """Encode [previous observation, current observation] like the risk cache.

    The Square risk feature cache was produced by encoding two RGB-DP
    observations, [max(0, t-1), t], and flattening the two encoded vectors.
    Online extraction must reconstruct that same 2-frame feature.
    """

    algo = dp_policy.policy
    current = dp_policy._prepare_observation(current_ob, batched_ob=False)

    # Square RGB observations from robomimic already arrive with the DP
    # observation-horizon dimension, e.g. [B, To, C, H, W]. The risk cache was
    # therefore a flattened encoding of this stacked DP observation. Older Lift
    # experiments used unstacked observations; keep the previous/current
    # fallback for that case.
    current_has_time = all(
        current[key].ndim - 2 == len(algo.obs_shapes[key]) for key in algo.obs_shapes
    )
    if current_has_time:
        pair_obs = {key: current[key] for key in algo.obs_shapes}
    else:
        previous_ob = current_ob if previous_ob is None else previous_ob
        previous = dp_policy._prepare_observation(previous_ob, batched_ob=False)
        pair_obs = {}
        for key in algo.obs_shapes:
            pair_obs[key] = torch.cat([previous[key], current[key]], dim=0).unsqueeze(0)

    nets = algo.nets
    if algo.ema is not None:
        nets = algo.ema.averaged_model
    features = TensorUtils.time_distributed(
        {"obs": pair_obs, "goal": None},
        nets["policy"]["obs_encoder"],
        inputs_as_kwargs=True,
    )
    return features.flatten(start_dim=1)


class RiskGuidedDPPolicy:
    def __init__(
        self,
        dp_policy,
        risk_feature_policy,
        risk_feature_policy_path: Path | None,
        risk_checkpoint: dict,
        risk_model: CausalPrefixRisk,
        *,
        num_candidates: int,
        candidate_batch_size: int,
        score_mode: str,
        selection: str,
        softmin_temperature: float,
        risk_threshold: float | None,
        score_gap_threshold: float,
        execute_horizon: int,
        action_start_index: int,
        max_prefix_len: int,
    ):
        self.dp_policy = dp_policy
        self.risk_feature_policy = risk_feature_policy
        self.risk_feature_policy_path = risk_feature_policy_path
        self.algo = dp_policy.policy
        self.risk_checkpoint = risk_checkpoint
        self.risk_model = risk_model
        self.self_contained_rgb_critic = bool(
            risk_checkpoint.get("self_contained_rgb_critic", False)
        )
        self.num_candidates = int(num_candidates)
        self.candidate_batch_size = int(candidate_batch_size)
        self.score_mode = score_mode
        self.selection = selection
        self.softmin_temperature = float(softmin_temperature)
        self.risk_threshold = risk_threshold
        self.score_gap_threshold = max(float(score_gap_threshold), 0.0)
        self.prediction_horizon = int(risk_checkpoint["prediction_horizon"])
        self.action_dim = int(risk_checkpoint["action_dim"])
        self.execute_horizon = (
            int(execute_horizon)
            if int(execute_horizon) > 0
            else int(self.algo.algo_config.horizon.action_horizon)
        )
        self.action_start_index = (
            int(action_start_index)
            if int(action_start_index) >= 0
            else int(self.algo.algo_config.horizon.observation_horizon) - 1
        )
        self.max_prefix_len = int(max_prefix_len)
        self.action_queue: deque[torch.Tensor] = deque()
        self.previous_ob: dict | None = None
        self.prefix_features: list[torch.Tensor] = []
        ckpt_args = risk_checkpoint.get("args", {})
        self.target_outcome = str(ckpt_args.get("target_outcome", "failure"))
        if self.target_outcome not in ("failure", "success"):
            raise ValueError(f"unknown risk target_outcome={self.target_outcome}")

        device = self.algo.device
        stats = risk_checkpoint["stats"]
        self.feature_mean = torch.as_tensor(
            stats["feature_mean"], device=device, dtype=torch.float32
        )
        self.feature_std = torch.as_tensor(
            stats["feature_std"], device=device, dtype=torch.float32
        )
        self.action_mean = torch.as_tensor(
            stats["action_mean"], device=device, dtype=torch.float32
        )
        self.action_std = torch.as_tensor(
            stats["action_std"], device=device, dtype=torch.float32
        )
        if self.feature_mean.numel() != int(risk_checkpoint["feature_dim"]):
            raise ValueError("risk feature normalizer shape does not match checkpoint")
        if self.action_mean.numel() != self.action_dim:
            raise ValueError("risk action normalizer shape does not match action_dim")
        if bool(torch.any(self.feature_std <= 0)) or bool(torch.any(self.action_std <= 0)):
            raise ValueError("risk normalizers contain non-positive std values")

        policy_prediction_horizon = int(self.algo.algo_config.horizon.prediction_horizon)
        if self.prediction_horizon > policy_prediction_horizon:
            raise ValueError(
                "risk model action horizon cannot exceed policy prediction horizon: "
                f"risk={self.prediction_horizon}, policy={policy_prediction_horizon}"
            )
        if self.algo.ac_dim != self.action_dim:
            raise ValueError(
                f"risk action_dim={self.action_dim} != policy action_dim={self.algo.ac_dim}"
            )
        if self.action_start_index < 0 or self.action_start_index >= policy_prediction_horizon:
            raise ValueError(f"invalid action_start_index={self.action_start_index}")
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self.execute_horizon}")
        if self.execute_horizon > self.prediction_horizon:
            raise ValueError(
                "the evaluator would execute actions that were not scored: "
                f"execute_horizon={self.execute_horizon}, "
                f"risk_horizon={self.prediction_horizon}"
            )
        if self.action_start_index + self.prediction_horizon > policy_prediction_horizon:
            raise ValueError(
                "the DP trajectory does not contain enough future slots for the "
                "critic; padding would create a training/inference mismatch: "
                f"start={self.action_start_index}, risk_horizon={self.prediction_horizon}, "
                f"policy_horizon={policy_prediction_horizon}"
            )
        self._validate_score_selection()
        if self.score_mode == "action_probability" and self.score_gap_threshold >= 1e-3:
            print(
                "WARNING: action_probability is often saturated; a probability-space "
                f"score gap of {self.score_gap_threshold:g} may suppress nearly all "
                "critic interventions. Prefer action_advantage_logodds.",
                flush=True,
            )

        self.last_scores: np.ndarray | None = None
        self.last_state_logit: float | None = None
        self.last_state_risk: float | None = None
        self.last_action_logits: np.ndarray | None = None
        self.last_action_probs: np.ndarray | None = None
        self.last_action_deltas: np.ndarray | None = None
        self.last_positive_action_risks: np.ndarray | None = None
        self.last_positive_action_advantages: np.ndarray | None = None
        self.last_score_gap: float | None = None
        self.last_score_std: float | None = None
        self.last_action_delta_std: float | None = None
        self.last_action_prob_std: float | None = None
        self.last_candidate_action_std: float | None = None
        self.last_candidate_action_pairwise_l2: float | None = None
        self.last_selected_index: int | None = None
        self.last_threshold_fallback: bool | None = None

    def _validate_score_selection(self) -> None:
        """Reject score directions that invert the configured outcome target."""

        if self.selection == "threshold_fallback":
            if self.score_mode != "positive_action_risk":
                raise ValueError(
                    "threshold_fallback is defined only for positive_action_risk"
                )
            return

        higher_is_better = self.score_mode in (
            "positive_action_advantage",
            "action_advantage_logodds",
        )
        if self.score_mode in ("action_delta_logodds", "action_logit", "action_probability"):
            higher_is_better = self.target_outcome == "success"
        if self.score_mode == "positive_action_risk":
            higher_is_better = False

        uses_max = self.selection in ("argmax", "softmax")
        uses_min = self.selection in ("argmin", "greedy", "softmin")
        if (higher_is_better and uses_min) or ((not higher_is_better) and uses_max):
            direction = "argmax" if higher_is_better else "argmin"
            raise ValueError(
                f"score_mode={self.score_mode} with target_outcome={self.target_outcome} "
                f"must use {direction}, got selection={self.selection}"
            )

    def start_episode(self) -> None:
        self.dp_policy.start_episode()
        if (
            self.risk_feature_policy is not None
            and self.risk_feature_policy is not self.dp_policy
        ):
            self.risk_feature_policy.start_episode()
        self.action_queue.clear()
        self.previous_ob = None
        self.prefix_features.clear()
        self.last_scores = None
        self.last_state_logit = None
        self.last_state_risk = None
        self.last_action_logits = None
        self.last_action_probs = None
        self.last_action_deltas = None
        self.last_positive_action_risks = None
        self.last_positive_action_advantages = None
        self.last_score_gap = None
        self.last_score_std = None
        self.last_action_delta_std = None
        self.last_action_prob_std = None
        self.last_candidate_action_std = None
        self.last_candidate_action_pairwise_l2 = None
        self.last_selected_index = None
        self.last_threshold_fallback = None

    @torch.no_grad()
    def sample_full_trajectory_batch(self, prepared_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return full denoised DP trajectory [B, Tp, action_dim]."""

        algo = self.algo
        if algo.algo_config.ddpm.enabled is True:
            num_inference_timesteps = algo.algo_config.ddpm.num_inference_timesteps
        elif algo.algo_config.ddim.enabled is True:
            num_inference_timesteps = algo.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError("DP checkpoint has neither DDPM nor DDIM enabled")

        nets = algo.nets
        if algo.ema is not None:
            nets = algo.ema.averaged_model

        obs_dict = obs_for_policy_encoder(algo, prepared_obs)
        obs_features = TensorUtils.time_distributed(
            {"obs": obs_dict, "goal": None},
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        obs_cond = obs_features.flatten(start_dim=1)
        batch_size = obs_cond.shape[0]
        # Match DiffusionPolicyUNet._get_action_trajectory exactly. This is a
        # no-op for ordinary checkpoints, but it is required for actors that
        # contain the optional success-condition adapter.
        obs_cond, _ = algo._apply_success_condition(
            obs_cond,
            nets=nets,
            success_condition=torch.ones(batch_size, device=algo.device),
            condition_mask=torch.ones(batch_size, device=algo.device),
            validate=True,
        )
        trajectory = torch.randn(
            (batch_size, int(algo.algo_config.horizon.prediction_horizon), algo.ac_dim),
            device=algo.device,
        )

        algo.noise_scheduler.set_timesteps(num_inference_timesteps)
        for timestep in algo.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=trajectory,
                timestep=timestep,
                global_cond=obs_cond,
            )
            trajectory = algo.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=trajectory,
            ).prev_sample
        return trajectory

    @torch.no_grad()
    def sample_candidates(self, prepared_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        chunks = []
        for start in range(0, self.num_candidates, self.candidate_batch_size):
            batch = min(self.candidate_batch_size, self.num_candidates - start)
            chunks.append(self.sample_full_trajectory_batch(repeat_obs(prepared_obs, batch)))
        return torch.cat(chunks, dim=0)

    def trajectory_to_future_actions(self, trajectories: torch.Tensor) -> torch.Tensor:
        future = trajectories[:, self.action_start_index :, :]
        if future.shape[1] < self.prediction_horizon:
            raise RuntimeError(
                f"only {future.shape[1]} future DP slots are available, but the "
                f"critic requires {self.prediction_horizon}"
            )
        return future[:, : self.prediction_horizon, :]

    def selected_actions_for_execution(self, selected_trajectory: torch.Tensor) -> torch.Tensor:
        start = self.action_start_index
        end = start + self.execute_horizon
        if selected_trajectory.shape[0] < end:
            raise ValueError(
                f"selected trajectory horizon={selected_trajectory.shape[0]} is shorter "
                f"than start+execute_horizon={end}"
            )
        return selected_trajectory[start:end]

    @torch.no_grad()
    def update_prefix(self, current_ob: dict) -> torch.Tensor:
        if self.self_contained_rgb_critic:
            # This is the important V4 deployment path: RGB preprocessing and
            # encoding are performed entirely by the critic. The candidate DP
            # actor is not consulted here.
            current = prepare_observation_for_rgb_critic(
                current_ob,
                self.risk_model.observation_shapes,
                self.algo.device,
            )
            observation_horizon = int(self.risk_model.observation_horizon)
            leading_sizes = {
                key: int(current[key].shape[0])
                for key in self.risk_model.observation_shapes
            }
            if all(size == observation_horizon for size in leading_sizes.values()):
                # Robomimic's rollout environment is wrapped by FrameStackWrapper.
                # Consequently, the raw current observation already represents
                # [t - To + 1, ..., t]. Treating that axis as a batch and stacking
                # previous/current observations again creates a spurious second
                # temporal axis: [1, To, To, feature_dim]. Training used exactly
                # one To-frame window per decision boundary, so consume this
                # existing window directly.
                observation_window = {
                    key: value.unsqueeze(0) for key, value in current.items()
                }
            elif all(size == 1 for size in leading_sizes.values()):
                # Compatibility path for environments that return a single frame
                # instead of a frame stack. The present critic was trained with a
                # two-frame [previous, current] window.
                if observation_horizon != 2:
                    raise ValueError(
                        "unstacked critic observations are supported only for "
                        f"observation_horizon=2, got {observation_horizon}"
                    )
                previous = (
                    current
                    if self.previous_ob is None
                    else prepare_observation_for_rgb_critic(
                        self.previous_ob,
                        self.risk_model.observation_shapes,
                        self.algo.device,
                    )
                )
                if not all(
                    int(previous[key].shape[0]) == 1
                    for key in self.risk_model.observation_shapes
                ):
                    raise ValueError(
                        "critic observation layout changed within an episode: "
                        f"previous/current leading sizes differ ({leading_sizes})"
                    )
                observation_window = {
                    key: torch.cat([previous[key], current[key]], dim=0).unsqueeze(0)
                    for key in self.risk_model.observation_shapes
                }
            else:
                raise ValueError(
                    "inconsistent critic observation leading dimensions; expected "
                    f"all 1 or all observation_horizon={observation_horizon}, got "
                    f"{leading_sizes}"
                )
            feature = self.risk_model.encode_rgb_boundary(observation_window)
        else:
            # Legacy V1--V3 checkpoints contain only feature-space heads.
            feature = encode_pair_feature(
                self.risk_feature_policy,
                self.previous_ob,
                current_ob,
            )
            if feature.shape[-1] != self.feature_mean.numel():
                raise ValueError(
                    f"encoded risk feature dim={feature.shape[-1]} but checkpoint expects "
                    f"{self.feature_mean.numel()}"
                )
            feature = (
                feature - self.feature_mean[None, :]
            ) / self.feature_std[None, :]
        expected_feature_dim = int(self.feature_mean.numel())
        if tuple(feature.shape) != (1, expected_feature_dim):
            raise RuntimeError(
                "critic boundary encoder must return [1, feature_dim]; "
                f"got {tuple(feature.shape)}, expected (1, {expected_feature_dim})"
            )
        self.prefix_features.append(feature[0])
        if self.max_prefix_len > 0 and len(self.prefix_features) > self.max_prefix_len:
            self.prefix_features = self.prefix_features[-self.max_prefix_len :]
        prefix = torch.stack(self.prefix_features, dim=0).unsqueeze(0)
        if prefix.ndim != 3 or prefix.shape[-1] != expected_feature_dim:
            raise RuntimeError(
                "critic prefix must be [batch, boundary_time, feature_dim]; "
                f"got {tuple(prefix.shape)}"
            )
        return prefix

    @torch.no_grad()
    def score_candidates(self, current_ob: dict, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        prefix = self.update_prefix(current_ob)
        context = self.risk_model.encode_prefix(prefix)
        current_context = context[:, -1:, :]
        state_logit = self.risk_model.state_head(current_context).squeeze(-1).squeeze(-1)

        risk_actions = self.trajectory_to_future_actions(candidates)
        if self.self_contained_rgb_critic:
            normalized_actions = self.risk_model.normalize_actions(risk_actions)
        else:
            normalized_actions = (
                risk_actions - self.action_mean[None, None, :]
            ) / self.action_std[None, None, :]
        repeated_context = current_context.repeat(risk_actions.shape[0], 1, 1)
        action_delta = self.risk_model.action_delta(
            repeated_context,
            normalized_actions[:, None, :, :],
        ).squeeze(1)
        action_logit = state_logit.detach().repeat(action_delta.shape[0]) + action_delta
        action_probability = torch.sigmoid(action_logit)
        if self.target_outcome == "failure":
            positive_action_risk = torch.relu(action_delta)
            positive_action_advantage = torch.relu(-action_delta)
            action_advantage_logodds = -action_delta
        elif self.target_outcome == "success":
            positive_action_risk = torch.relu(-action_delta)
            positive_action_advantage = torch.relu(action_delta)
            action_advantage_logodds = action_delta

        state_probability = torch.sigmoid(state_logit.reshape(()))
        if self.score_mode == "positive_action_risk":
            scores = positive_action_risk
        elif self.score_mode == "positive_action_advantage":
            scores = positive_action_advantage
        elif self.score_mode == "action_delta_logodds":
            scores = action_delta
        elif self.score_mode == "action_advantage_logodds":
            scores = action_advantage_logodds
        elif self.score_mode == "action_logit":
            scores = action_logit
        elif self.score_mode == "action_probability":
            scores = action_probability
        else:
            raise ValueError(f"unknown score_mode={self.score_mode}")
        return {
            "scores": scores,
            "state_logit": state_logit.reshape(()),
            "state_probability": state_probability,
            "action_delta": action_delta,
            "action_logit": action_logit,
            "action_probability": action_probability,
            "positive_action_risk": positive_action_risk,
            "positive_action_advantage": positive_action_advantage,
            "action_advantage_logodds": action_advantage_logodds,
        }

    def choose_index(self, scores: torch.Tensor) -> tuple[int, bool]:
        if len(scores) == 1:
            return 0, False
        if self.selection in ("argmin", "greedy"):
            selected = int(torch.argmin(scores).item())
            if self.score_gap_threshold > 0.0 and selected != 0:
                native = float(scores[0].detach().cpu())
                best = float(scores[selected].detach().cpu())
                if native - best < self.score_gap_threshold:
                    return 0, True
            return selected, False
        if self.selection == "argmax":
            selected = int(torch.argmax(scores).item())
            if self.score_gap_threshold > 0.0 and selected != 0:
                native = float(scores[0].detach().cpu())
                best = float(scores[selected].detach().cpu())
                if best - native < self.score_gap_threshold:
                    return 0, True
            return selected, False
        if self.selection == "softmin":
            probs = torch.softmax(-scores / max(self.softmin_temperature, 1e-6), dim=0)
            return int(torch.multinomial(probs, num_samples=1).item()), False
        if self.selection == "softmax":
            probs = torch.softmax(scores / max(self.softmin_temperature, 1e-6), dim=0)
            return int(torch.multinomial(probs, num_samples=1).item()), False
        if self.selection == "threshold_fallback":
            if self.risk_threshold is None:
                raise ValueError("threshold_fallback requires --risk-threshold")
            # Candidate 0 is the unfiltered DP sample. Keep it when it is already
            # under the calibrated hazard threshold; otherwise intervene.
            if float(scores[0].detach().cpu()) <= float(self.risk_threshold):
                return 0, True
            return int(torch.argmin(scores).item()), False
        raise ValueError(f"unknown selection={self.selection}")

    def __call__(self, ob) -> np.ndarray:
        if len(self.action_queue) == 0:
            prepared_obs = self.dp_policy._prepare_observation(ob, batched_ob=False)
            candidates = self.sample_candidates(prepared_obs)
            risk = self.score_candidates(ob, candidates)
            selected, fallback = self.choose_index(risk["scores"])

            self.last_scores = risk["scores"].detach().cpu().numpy()
            self.last_state_logit = float(risk["state_logit"].detach().cpu())
            self.last_state_risk = float(risk["state_probability"].detach().cpu())
            self.last_action_logits = risk["action_logit"].detach().cpu().numpy()
            self.last_action_probs = risk["action_probability"].detach().cpu().numpy()
            self.last_action_deltas = risk["action_delta"].detach().cpu().numpy()
            self.last_positive_action_risks = (
                risk["positive_action_risk"].detach().cpu().numpy()
            )
            self.last_positive_action_advantages = (
                risk["positive_action_advantage"].detach().cpu().numpy()
            )
            self.last_score_std = float(np.std(self.last_scores))
            self.last_action_delta_std = float(np.std(self.last_action_deltas))
            self.last_action_prob_std = float(np.std(self.last_action_probs))
            native_score = float(self.last_scores[0])
            selected_score = float(self.last_scores[selected])
            if self.selection in ("argmin", "greedy", "threshold_fallback"):
                self.last_score_gap = native_score - selected_score
            else:
                self.last_score_gap = selected_score - native_score
            executed_candidates = candidates[
                :,
                self.action_start_index : self.action_start_index + self.execute_horizon,
                :,
            ]
            self.last_candidate_action_std = float(
                executed_candidates.std(dim=0, unbiased=False).mean().detach().cpu()
            )
            if executed_candidates.shape[0] > 1:
                flat_candidates = executed_candidates.reshape(executed_candidates.shape[0], -1)
                distances = torch.pdist(flat_candidates, p=2)
                self.last_candidate_action_pairwise_l2 = float(
                    distances.mean().detach().cpu()
                )
            else:
                self.last_candidate_action_pairwise_l2 = 0.0
            self.last_selected_index = int(selected)
            self.last_threshold_fallback = bool(fallback)

            for action in self.selected_actions_for_execution(candidates[selected]):
                self.action_queue.append(action.detach())

        action = self.action_queue.popleft().unsqueeze(0)
        self.previous_ob = deepcopy(ob)
        return action.detach().cpu().numpy()[0].copy()


def load_dp_policy_for_rollout(policy_path: Path, device: torch.device):
    """Load either a normal robomimic policy checkpoint or our hybrid actor checkpoint."""

    raw_checkpoint = FileUtils.load_dict_from_checkpoint(str(policy_path))

    # load the actor of IDQL
    if bool(raw_checkpoint.get("hybrid_dp_chunk_actor_iql", False)):
        base_checkpoint_value = raw_checkpoint.get(
            "pretrained_dp_checkpoint",
            raw_checkpoint.get("args", {}).get("checkpoint"),
        )
        if base_checkpoint_value is None:
            raise RuntimeError(
                "hybrid DP chunk actor checkpoint is missing pretrained_dp_checkpoint"
            )
        base_checkpoint = resolve_checkpoint_path(base_checkpoint_value)
        dp_policy, base_ckpt = FileUtils.policy_from_checkpoint(
            ckpt_path=str(base_checkpoint),
            device=device,
            verbose=False,
        )
        dp_policy.policy.deserialize(raw_checkpoint["actor_model"], load_optimizers=False)
        dp_policy.policy.set_eval()
        return dp_policy, base_ckpt, raw_checkpoint

    # load the standard policy trained by BC
    dp_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=raw_checkpoint,
        device=device,
        verbose=False,
    )
    return dp_policy, ckpt_dict, raw_checkpoint


def load_critic_model(policy_path: Path, risk_path: Path, device: torch.device, args):
    dp_policy, env_ckpt, policy_checkpoint = load_dp_policy_for_rollout(
        policy_path,
        device,
    )
    checkpoint = torch.load(risk_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    model_arch = str(ckpt_args.get("model_arch", "v1"))
    rgb_encoder = None
    observation_shapes = None
    observation_horizon = None
    if model_arch == "v4":
        if not bool(checkpoint.get("self_contained_rgb_critic", False)):
            raise ValueError("V4 checkpoint is not marked as a self-contained RGB critic")
        if "rgb_encoder_spec" not in checkpoint:
            raise ValueError("V4 checkpoint is missing rgb_encoder_spec")
        rgb_encoder, observation_shapes, observation_horizon = (
            build_rgb_encoder_from_critic_spec(
                checkpoint["rgb_encoder_spec"],
                device,
            )
        )
        risk_feature_policy = None
        risk_feature_policy_path = None
        print(
            "Using self-contained critic RGB encoder; candidate actor is used "
            "only to generate action chunks.",
            flush=True,
        )
    else:
        expected_policy = ckpt_args.get("expected_dp_checkpoint")
        if expected_policy is not None:
            expected_policy = resolve_checkpoint_path(expected_policy)
            if expected_policy == policy_path.resolve():
                risk_feature_policy = dp_policy
            else:
                risk_feature_policy, _, _ = load_dp_policy_for_rollout(
                    expected_policy,
                    device,
                )
                print(
                    "Using separate policies: candidate actor="
                    f"{policy_path.resolve()}, critic feature encoder={expected_policy}",
                    flush=True,
                )
            risk_feature_policy_path = expected_policy
        else:
            # Legacy checkpoints did not record their feature encoder. Reusing
            # the candidate actor preserves their historical behavior.
            risk_feature_policy = dp_policy
            risk_feature_policy_path = policy_path.resolve()
            print(
                "WARNING: critic checkpoint does not record expected_dp_checkpoint; "
                "using the candidate actor as its feature encoder",
                flush=True,
            )
    stats = checkpoint["stats"]
    risk_model = make_causal_prefix_model(
        model_arch=model_arch,
        feature_dim=int(checkpoint["feature_dim"]),
        prediction_horizon=int(checkpoint["prediction_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(ckpt_args["hidden_dim"]),
        action_hidden_dim=int(ckpt_args["action_hidden_dim"]),
        dropout=float(ckpt_args["dropout"]),
        action_num_heads=int(ckpt_args.get("action_num_heads", 4)),
        action_conv_layers=int(ckpt_args.get("action_conv_layers", 2)),
        prefix_conv_layers=int(ckpt_args.get("prefix_conv_layers", 1)),
        rgb_encoder=rgb_encoder,
        observation_shapes=observation_shapes,
        observation_horizon=observation_horizon,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
    ).to(device)
    if model_arch == "v4":
        risk_model.rgb_encoder_spec = checkpoint["rgb_encoder_spec"]
    risk_model.load_state_dict(checkpoint["model"])
    risk_model.eval()
    risk_model.requires_grad_(False)
    guided_policy = RiskGuidedDPPolicy(
        dp_policy=dp_policy,
        risk_feature_policy=risk_feature_policy,
        risk_feature_policy_path=risk_feature_policy_path,
        risk_checkpoint=checkpoint,
        risk_model=risk_model,
        num_candidates=args.num_candidates,
        candidate_batch_size=args.candidate_batch_size,
        score_mode=args.score_mode,
        selection=args.selection,
        softmin_temperature=args.softmin_temperature,
        risk_threshold=args.risk_threshold,
        score_gap_threshold=args.score_gap_threshold,
        execute_horizon=args.execute_horizon,
        action_start_index=args.action_start_index,
        max_prefix_len=args.max_prefix_len,
    )
    return guided_policy, env_ckpt, policy_checkpoint


def rollout(
    policy,
    env,
    horizon: int,
    return_obs: bool = False,
    video_writer=None,
    video_skip: int = 5,
    camera_names=None,
) -> tuple[dict, dict]:
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
        risk_selected=[],
        risk_mean=[],
        risk_min=[],
        risk_max=[],
        state_risk=[],
        action_delta_selected=[],
        action_prob_selected=[],
        positive_risk_selected=[],
        positive_advantage_selected=[],
        score_gap=[],
        score_std=[],
        action_delta_std=[],
        action_prob_std=[],
        candidate_action_std=[],
        candidate_action_pairwise_l2=[],
        selected_index=[],
        threshold_fallback=[],
        initial_state_dict=state_dict,
    )
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    video_count = 0
    step_i = -1
    try:
        for step_i in range(horizon):
            # Important: robosuite / MuJoCo can crash with NumPy arrays that
            # are views backed by Torch-owned memory. Make a plain C-contiguous
            # NumPy copy before crossing into the simulator.
            action = np.asarray(policy(obs), dtype=np.float64, order="C").copy()
            next_obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            success = bool(env.is_success()["task"])

            if video_writer is not None and video_count % video_skip == 0:
                frames = [
                    env.render(mode="rgb_array", height=512, width=512, camera_name=name)
                    for name in camera_names
                ]
                video_writer.append_data(np.concatenate(frames, axis=1))
            video_count += 1

            scores = policy.last_scores
            selected = policy.last_selected_index
            traj["actions"].append(action)
            traj["rewards"].append(float(reward))
            traj["dones"].append(bool(done))
            traj["states"].append(state_dict["states"])
            if scores is None or selected is None:
                traj["risk_selected"].append(np.nan)
                traj["risk_mean"].append(np.nan)
                traj["risk_min"].append(np.nan)
                traj["risk_max"].append(np.nan)
                traj["state_risk"].append(np.nan)
                traj["action_delta_selected"].append(np.nan)
                traj["action_prob_selected"].append(np.nan)
                traj["positive_risk_selected"].append(np.nan)
                traj["positive_advantage_selected"].append(np.nan)
                traj["score_gap"].append(np.nan)
                traj["score_std"].append(np.nan)
                traj["action_delta_std"].append(np.nan)
                traj["action_prob_std"].append(np.nan)
                traj["candidate_action_std"].append(np.nan)
                traj["candidate_action_pairwise_l2"].append(np.nan)
                traj["selected_index"].append(-1)
                traj["threshold_fallback"].append(False)
            else:
                traj["risk_selected"].append(float(scores[selected]))
                traj["risk_mean"].append(float(np.mean(scores)))
                traj["risk_min"].append(float(np.min(scores)))
                traj["risk_max"].append(float(np.max(scores)))
                traj["state_risk"].append(float(policy.last_state_risk))
                traj["action_delta_selected"].append(
                    float(policy.last_action_deltas[selected])
                )
                traj["action_prob_selected"].append(
                    float(policy.last_action_probs[selected])
                )
                traj["positive_risk_selected"].append(
                    float(policy.last_positive_action_risks[selected])
                )
                traj["positive_advantage_selected"].append(
                    float(policy.last_positive_action_advantages[selected])
                )
                traj["score_gap"].append(float(policy.last_score_gap))
                traj["score_std"].append(float(policy.last_score_std))
                traj["action_delta_std"].append(float(policy.last_action_delta_std))
                traj["action_prob_std"].append(float(policy.last_action_prob_std))
                traj["candidate_action_std"].append(float(policy.last_candidate_action_std))
                traj["candidate_action_pairwise_l2"].append(
                    float(policy.last_candidate_action_pairwise_l2)
                )
                traj["selected_index"].append(int(selected))
                traj["threshold_fallback"].append(bool(policy.last_threshold_fallback))
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            if done or success:
                break
            obs = deepcopy(next_obs)
            state_dict = env.get_state()
    except env.rollout_exceptions as exc:
        print(f"WARNING: rollout exception {exc}", flush=True)

    stats = {
        "Return": total_reward,
        "Horizon": step_i + 1,
        "Success_Rate": float(success),
    }
    for key in traj:
        if key == "initial_state_dict":
            continue
        traj[key] = np.asarray(traj[key])
    return stats, traj


def aggregate(stats: list[dict]) -> dict:
    n = len(stats)
    successes = float(sum(x["Success_Rate"] for x in stats))
    return {
        "Num_Rollouts": n,
        "Return": float(np.mean([x["Return"] for x in stats])) if n else float("nan"),
        "Horizon": float(np.mean([x["Horizon"] for x in stats])) if n else float("nan"),
        "Success_Rate": successes / max(n, 1),
        "Num_Success": successes,
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [center - radius, center + radius]


def score_gap_suffix(score_gap_threshold: float) -> str:
    if float(score_gap_threshold) <= 0.0:
        return ""
    value = f"{float(score_gap_threshold):.6g}".replace("-", "m").replace(".", "p")
    return f"_gap{value}"


def evaluate(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")
    policy, ckpt_dict, policy_checkpoint = load_critic_model(
        args.policy,
        args.risk,
        device,
        args,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=args.video_dir is not None,
        verbose=False,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    dataset_writer = None
    data_group = None
    total_samples = 0
    if args.dataset_path is not None:
        args.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_writer = h5py.File(args.dataset_path, "w")
        data_group = dataset_writer.create_group("data")
    diagnostic_keys = (
        "score_gap",
        "score_std",
        "action_delta_std",
        "action_prob_std",
        "candidate_action_std",
        "candidate_action_pairwise_l2",
        "positive_advantage_selected",
        "action_delta_selected",
        "action_prob_selected",
    )
    diagnostic_values = {key: [] for key in diagnostic_keys}

    for i in range(args.n_rollouts):
        writer = None
        if args.video_dir is not None and i < args.num_videos:
            writer = imageio.get_writer(args.video_dir / f"rollout_{i:03d}.mp4", fps=20)
        rollout_stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=args.horizon,
            return_obs=False,
            video_writer=writer,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
        )
        if writer is not None:
            writer.close()
        stats.append(rollout_stats)
        for key in diagnostic_keys:
            values = np.asarray(traj[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values):
                diagnostic_values[key].append(values)
        print(
            f"rollout={i} success={rollout_stats['Success_Rate']:.0f} "
            f"return={rollout_stats['Return']:.3f} horizon={rollout_stats['Horizon']}",
            flush=True,
        )

        if data_group is not None:
            ep = data_group.create_group(f"demo_{i}")
            for key in (
                "actions",
                "rewards",
                "dones",
                "states",
                "risk_selected",
                "risk_mean",
                "risk_min",
                "risk_max",
                "state_risk",
                "action_delta_selected",
                "action_prob_selected",
                "positive_risk_selected",
                "positive_advantage_selected",
                "score_gap",
                "score_std",
                "action_delta_std",
                "action_prob_std",
                "candidate_action_std",
                "candidate_action_pairwise_l2",
                "selected_index",
                "threshold_fallback",
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

    avg = aggregate(stats)
    successes = int(round(avg["Num_Success"]))
    candidate_diagnostics = {}
    for key, chunks in diagnostic_values.items():
        if not chunks:
            candidate_diagnostics[key] = {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "q10": float("nan"),
                "median": float("nan"),
                "q90": float("nan"),
            }
            continue
        values = np.concatenate(chunks)
        candidate_diagnostics[key] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q10": float(np.quantile(values, 0.1)),
            "median": float(np.quantile(values, 0.5)),
            "q90": float(np.quantile(values, 0.9)),
        }
    summary = {
        "policy": str(args.policy),
        "policy_checkpoint_kind": (
            "hybrid_dp_chunk_actor_iql"
            if bool(policy_checkpoint.get("hybrid_dp_chunk_actor_iql", False))
            else "robomimic_policy"
        ),
        "risk": str(args.risk),
        "risk_feature_policy": (
            str(policy.risk_feature_policy_path)
            if policy.risk_feature_policy_path is not None
            else None
        ),
        "separate_risk_feature_encoder": bool(
            policy.risk_feature_policy is not None
            and policy.risk_feature_policy is not policy.dp_policy
        ),
        "self_contained_rgb_critic": policy.self_contained_rgb_critic,
        "risk_target_outcome": str(policy.risk_checkpoint.get("args", {}).get("target_outcome", "failure")),
        "risk_best_step": int(policy.risk_checkpoint.get("best_step", -1)),
        "risk_best_quality": list(policy.risk_checkpoint.get("best_quality", [])),
        "num_candidates": args.num_candidates,
        "candidate_batch_size": args.candidate_batch_size,
        "score_mode": args.score_mode,
        "selection": args.selection,
        "softmin_temperature": args.softmin_temperature,
        "risk_threshold": args.risk_threshold,
        "score_gap_threshold": args.score_gap_threshold,
        "action_start_index": int(policy.action_start_index),
        "execute_horizon": int(policy.execute_horizon),
        "prediction_horizon": int(policy.prediction_horizon),
        "risk_prediction_horizon": int(policy.prediction_horizon),
        "policy_prediction_horizon": int(policy.algo.algo_config.horizon.prediction_horizon),
        "target_outcome": str(policy.target_outcome),
        "max_prefix_len": int(policy.max_prefix_len),
        "risk_normalization": {
            "feature_dim": int(policy.feature_mean.numel()),
            "action_dim": int(policy.action_mean.numel()),
            "feature_std_min": float(policy.feature_std.min().detach().cpu()),
            "feature_std_max": float(policy.feature_std.max().detach().cpu()),
            "action_std_min": float(policy.action_std.min().detach().cpu()),
            "action_std_max": float(policy.action_std.max().detach().cpu()),
        },
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "average_rollout_stats": avg,
        "candidate_diagnostics": candidate_diagnostics,
        "wilson_95_interval": wilson(successes, args.n_rollouts),
        "rollouts": stats,
    }
    path = args.output_dir / (
        f"risk_eval_{args.score_mode}_{args.selection}_N{args.num_candidates}"
        f"{score_gap_suffix(args.score_gap_threshold)}_seed{args.seed}.json"
    )
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--execute-horizon", type=int, default=8)
    parser.add_argument(
        "--action-start-index",
        type=int,
        default=-1,
        help="DP slot used as future action t. Use -1 for observation_horizon - 1.",
    )
    parser.add_argument(
        "--score-mode",
        choices=(
            "positive_action_risk",
            "positive_action_advantage",
            "action_delta_logodds",
            "action_advantage_logodds",
            "action_logit",
            "action_probability",
        ),
        default="positive_action_risk",
    )
    parser.add_argument(
        "--selection",
        choices=("argmin", "argmax", "greedy", "softmin", "softmax", "threshold_fallback"),
        default="argmin",
    )
    parser.add_argument("--softmin-temperature", type=float, default=1.0)
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument(
        "--score-gap-threshold",
        type=float,
        default=0.0,
        help=(
            "Conservative fallback for argmax/argmin selection. Candidate 0 is "
            "kept unless the selected candidate improves the score over candidate "
            "0 by at least this margin. For argmax improvement is best-native; "
            "for argmin improvement is native-best."
        ),
    )
    parser.add_argument("--max-prefix-len", type=int, default=0)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument(
        "--camera-names",
        type=str,
        nargs="+",
        default=("agentview", "robot0_eye_in_hand"),
    )
    args = parser.parse_args()
    for key in ("policy", "risk", "output_dir", "dataset_path", "video_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    evaluate(args)


if __name__ == "__main__":
    main()
