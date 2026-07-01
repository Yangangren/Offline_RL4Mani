"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
from copy import deepcopy
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1. Import through a local shim so that the old
# diffusers package does not eagerly import pipelines / transformers.
from robomimic.utils.safe_diffusers import load_diffusers_components

DDPMScheduler, DDIMScheduler, EMAModel = load_diffusers_components()

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
from robomimic.models.prefix_risk_nets import CausalPrefixRisk
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self._model_ema_parameters = list(nets.parameters()) if ema is not None else None
        self._averaged_ema_parameters = (
            list(ema.averaged_model.parameters()) if ema is not None else None
        )
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.reference_nets = None
        self.reference_margin_step = 0
        self.hazard_model = None
        self.hazard_action_mean = None
        self.hazard_action_std = None

        if self.reference_margin_enabled and self.hazard_constraint_enabled:
            raise ValueError(
                "reference_margin and hazard_constraint are mutually exclusive"
            )
        if self.hazard_constraint_enabled:
            self._load_hazard_model()

    @property
    def reference_margin_enabled(self):
        return (
            "reference_margin" in self.algo_config
            and self.algo_config.reference_margin.enabled
        )

    @property
    def hazard_constraint_enabled(self):
        return (
            "hazard_constraint" in self.algo_config
            and self.algo_config.hazard_constraint.enabled
        )

    @property
    def reference_policy_enabled(self):
        return self.reference_margin_enabled or self.hazard_constraint_enabled

    def _load_hazard_model(self):
        checkpoint_path = self.algo_config.hazard_constraint.checkpoint
        if checkpoint_path is None:
            raise ValueError("hazard_constraint.checkpoint must be provided")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        args = checkpoint["args"]
        self.hazard_model = CausalPrefixRisk(
            feature_dim=int(checkpoint["feature_dim"]),
            prediction_horizon=int(checkpoint["prediction_horizon"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_dim=int(args["hidden_dim"]),
            action_hidden_dim=int(args["action_hidden_dim"]),
            dropout=float(args["dropout"]),
        ).to(self.device)
        self.hazard_model.load_state_dict(checkpoint["model"])
        self.hazard_model.eval()
        self.hazard_model.requires_grad_(False)
        self.hazard_action_mean = torch.as_tensor(
            checkpoint["stats"]["action_mean"],
            device=self.device,
            dtype=torch.float32,
        )
        self.hazard_action_std = torch.as_tensor(
            checkpoint["stats"]["action_std"],
            device=self.device,
            dtype=torch.float32,
        )
        self.hazard_context_dim = int(args["hidden_dim"])
        if int(checkpoint["prediction_horizon"]) != int(
            self.algo_config.horizon.prediction_horizon
        ):
            raise ValueError(
                "hazard and policy prediction horizons do not match"
            )
        if int(checkpoint["action_dim"]) != self.ac_dim:
            raise ValueError("hazard and policy action dimensions do not match")

    def _create_reference_nets(self):
        """Freeze a function-space reference at the policy initialization."""
        self.reference_nets = deepcopy(self.nets).to(self.device)
        self.reference_nets.train(self.nets.training)
        self.reference_nets.requires_grad_(False)

    def _constraint_multiplier(self, config):
        step = self.reference_margin_step
        if step < config.warmup_steps:
            return 0.0
        if config.ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - config.warmup_steps + 1) / config.ramp_steps)

    def _reference_margin_multiplier(self):
        return self._constraint_multiplier(self.algo_config.reference_margin)

    def _encode_obs(self, inputs, nets):
        features = TensorUtils.time_distributed(
            inputs,
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        assert features.ndim == 3
        return features.flatten(start_dim=1)

    def _encode_current_and_reference(self, inputs):
        """Use exactly the same random crops for current and reference encoders."""
        cpu_before = torch.get_rng_state()
        cuda_before = None
        if self.device.type == "cuda":
            cuda_before = torch.cuda.get_rng_state(self.device)

        current_cond = self._encode_obs(inputs, self.nets)

        cpu_after = torch.get_rng_state()
        cuda_after = None
        if self.device.type == "cuda":
            cuda_after = torch.cuda.get_rng_state(self.device)

        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, self.device)
        with torch.no_grad():
            reference_cond = self._encode_obs(inputs, self.reference_nets)

        # Replaying the augmentation must not rewind randomness for the next batch.
        torch.set_rng_state(cpu_after)
        if cuda_after is not None:
            torch.cuda.set_rng_state(cuda_after, self.device)
        return current_cond, reference_cond
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        input_batch["anti_failure"] = batch.get(
            "anti_failure",
            torch.zeros(batch["actions"].shape[0], device=batch["actions"].device),
        )
        input_batch["hazard_failure"] = batch.get(
            "hazard_failure",
            torch.zeros(batch["actions"].shape[0], device=batch["actions"].device),
        )
        if self.hazard_constraint_enabled:
            if "hazard_context" not in batch:
                raise KeyError(
                    "hazard_constraint requires hazard_context in train.dataset_keys"
                )
            input_batch["hazard_context"] = batch["hazard_context"][:, 0, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            # Some rollout datasets store actions with tiny float32 numerical
            # overshoot, e.g. 1.0000139. Treat that as a serialization /
            # conversion artifact and clamp it, while still catching genuinely
            # unnormalized actions.
            tolerance = 1e-3
            action_min = torch.min(actions).item()
            action_max = torch.max(actions).item()
            if action_min < -1.0 - tolerance or action_max > 1.0 + tolerance:
                raise ValueError(
                    "'actions' must be in range [-1,1] for Diffusion Policy! "
                    "Check if hdf5_normalize_action is enabled. "
                    f"Got min={action_min:.6f}, max={action_max:.6f}."
                )
            if action_min < -1.0 or action_max > 1.0:
                input_batch["actions"] = actions.clamp(-1.0, 1.0)
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _sample_ddim_trajectory(
        self,
        nets,
        obs_cond,
        initial_noise,
        num_inference_steps,
    ):
        """Differentiably sample one deterministic DDIM action trajectory."""
        if not isinstance(self.noise_scheduler, DDIMScheduler):
            raise RuntimeError("hazard_constraint currently requires DDIM")
        self.noise_scheduler.set_timesteps(
            num_inference_steps,
            device=self.device,
        )
        trajectory = initial_noise
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=trajectory,
                timestep=timestep,
                global_cond=obs_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=trajectory,
            ).prev_sample
        return trajectory

    def _trajectory_to_hazard_actions(self, trajectory):
        """Align DP slots with future actions used to train the hazard model."""
        start = int(self.algo_config.hazard_constraint.action_start_index)
        if start < 0 or start >= trajectory.shape[1]:
            raise ValueError("invalid hazard_constraint.action_start_index")
        if start == 0:
            return trajectory
        future = trajectory[:, start:]
        padding = future[:, -1:].expand(-1, start, -1)
        return torch.cat([future, padding], dim=1)

    def _hazard_action_delta(self, context, trajectory):
        actions = self._trajectory_to_hazard_actions(trajectory)
        actions = (
            actions - self.hazard_action_mean[None, None, :]
        ) / self.hazard_action_std[None, None, :]
        return self.hazard_model.action_delta(
            context[:, None, :],
            actions[:, None, :, :],
        ).squeeze(1)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            if self.reference_policy_enabled:
                if self.reference_nets is None:
                    self._create_reference_nets()
                obs_cond, reference_obs_cond = self._encode_current_and_reference(inputs)
            else:
                obs_cond = self._encode_obs(inputs, self.nets)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            per_sample_energy = F.mse_loss(
                noise_pred, noise, reduction="none"
            ).flatten(start_dim=1).mean(dim=1)

            if self.reference_margin_enabled:
                anti_failure = batch["anti_failure"].flatten() > 0.5
                positive = ~anti_failure
                zero = per_sample_energy.sum() * 0.0
                positive_loss = (
                    per_sample_energy[positive].mean() if positive.any() else zero
                )

                with torch.no_grad():
                    reference_noise_pred = self.reference_nets["policy"][
                        "noise_pred_net"
                    ](
                        noisy_actions,
                        timesteps,
                        global_cond=reference_obs_cond,
                    )
                    reference_energy = F.mse_loss(
                        reference_noise_pred,
                        noise,
                        reduction="none",
                    ).flatten(start_dim=1).mean(dim=1)

                energy_gap = per_sample_energy - reference_energy
                if self.algo_config.reference_margin.normalized_margin:
                    energy_gap = energy_gap / reference_energy.clamp_min(1e-6)
                margin_violation = F.relu(
                    self.algo_config.reference_margin.margin - energy_gap
                )
                failure_loss = (
                    margin_violation[anti_failure].mean()
                    if anti_failure.any()
                    else zero
                )

                prediction_drift = F.mse_loss(
                    noise_pred,
                    reference_noise_pred,
                    reduction="none",
                ).flatten(start_dim=1).mean(dim=1)
                reference_loss = (
                    prediction_drift[positive].mean() if positive.any() else zero
                )

                margin_multiplier = self._reference_margin_multiplier()
                loss = (
                    positive_loss
                    + margin_multiplier
                    * self.algo_config.reference_margin.failure_weight
                    * failure_loss
                    + self.algo_config.reference_margin.reference_weight
                    * reference_loss
                )
                negative_gap = (
                    energy_gap[anti_failure].mean() if anti_failure.any() else zero
                )
                active_fraction = (
                    (margin_violation[anti_failure] > 0).float().mean()
                    if anti_failure.any()
                    else zero
                )
            elif self.hazard_constraint_enabled:
                config = self.algo_config.hazard_constraint
                hazard_failure = batch["hazard_failure"].flatten() > 0.5
                positive = ~hazard_failure
                zero = per_sample_energy.sum() * 0.0
                positive_loss = (
                    per_sample_energy[positive].mean() if positive.any() else zero
                )

                with torch.no_grad():
                    reference_noise_pred = self.reference_nets["policy"][
                        "noise_pred_net"
                    ](
                        noisy_actions,
                        timesteps,
                        global_cond=reference_obs_cond,
                    )
                prediction_drift = F.mse_loss(
                    noise_pred,
                    reference_noise_pred,
                    reduction="none",
                ).flatten(start_dim=1).mean(dim=1)
                positive_reference_loss = (
                    prediction_drift[positive].mean() if positive.any() else zero
                )

                hazard_loss = zero
                negative_reference_loss = zero
                current_risk_mean = zero
                reference_risk_mean = zero
                risk_improvement_mean = zero
                active_fraction = zero
                if hazard_failure.any():
                    num_negative = int(hazard_failure.sum().item())
                    action_samples = int(config.action_samples)
                    if action_samples < 1:
                        raise ValueError(
                            "hazard_constraint.action_samples must be positive"
                        )
                    negative_obs_cond = obs_cond[hazard_failure].repeat_interleave(
                        action_samples,
                        dim=0,
                    )
                    negative_reference_obs_cond = reference_obs_cond[
                        hazard_failure
                    ].repeat_interleave(action_samples, dim=0)
                    initial_noise = torch.randn(
                        (
                            num_negative * action_samples,
                            Tp,
                            action_dim,
                        ),
                        device=self.device,
                    )
                    current_trajectory = self._sample_ddim_trajectory(
                        self.nets,
                        negative_obs_cond,
                        initial_noise,
                        config.sampling_steps,
                    )
                    with torch.no_grad():
                        reference_trajectory = self._sample_ddim_trajectory(
                            self.reference_nets,
                            negative_reference_obs_cond,
                            initial_noise,
                            config.sampling_steps,
                        )

                    context = batch["hazard_context"][
                        hazard_failure
                    ].repeat_interleave(action_samples, dim=0)
                    if context.shape[-1] != self.hazard_context_dim:
                        raise ValueError(
                            "hazard_context dimension does not match checkpoint"
                        )
                    current_delta = self._hazard_action_delta(
                        context,
                        current_trajectory,
                    )
                    with torch.no_grad():
                        reference_delta = self._hazard_action_delta(
                            context,
                            reference_trajectory,
                        )
                    current_risk = F.relu(current_delta)
                    reference_risk = F.relu(reference_delta)
                    target_risk = F.relu(reference_risk - config.margin)
                    violation = F.relu(current_risk - target_risk)
                    hazard_loss = violation.mean()
                    negative_reference_loss = F.mse_loss(
                        current_trajectory,
                        reference_trajectory,
                    )
                    current_risk_mean = current_risk.mean()
                    reference_risk_mean = reference_risk.mean()
                    risk_improvement_mean = (
                        reference_risk - current_risk
                    ).mean()
                    active_fraction = (violation > 0).float().mean()

                constraint_multiplier = self._constraint_multiplier(config)
                loss = (
                    positive_loss
                    + config.positive_reference_weight
                    * positive_reference_loss
                    + constraint_multiplier
                    * (
                        config.weight * hazard_loss
                        + config.negative_reference_weight
                        * negative_reference_loss
                    )
                )
            else:
                loss = per_sample_energy.mean()
            
            # logging
            losses = {
                "l2_loss": loss
            }
            if self.reference_margin_enabled:
                losses.update(
                    {
                        "positive_loss": positive_loss,
                        "failure_loss": failure_loss,
                        "reference_loss": reference_loss,
                        "negative_energy_gap": negative_gap,
                        "active_margin_fraction": active_fraction,
                        "negative_batch_fraction": anti_failure.float().mean(),
                        "margin_multiplier": torch.as_tensor(
                            margin_multiplier, device=self.device
                        ),
                    }
                )
            elif self.hazard_constraint_enabled:
                losses.update(
                    {
                        "positive_loss": positive_loss,
                        "positive_reference_loss": positive_reference_loss,
                        "hazard_loss": hazard_loss,
                        "negative_reference_loss": negative_reference_loss,
                        "hazard_current_risk": current_risk_mean,
                        "hazard_reference_risk": reference_risk_mean,
                        "hazard_risk_improvement": risk_improvement_mean,
                        "hazard_active_fraction": active_fraction,
                        "hazard_batch_fraction": hazard_failure.float().mean(),
                        "constraint_multiplier": torch.as_tensor(
                            constraint_multiplier,
                            device=self.device,
                        ),
                    }
                )
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self._update_ema()
                if self.reference_policy_enabled:
                    self.reference_margin_step += 1
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA parameters without rebuilding and loading a state dict.

        The diffusers 0.11 implementation calls ``named_buffers`` and
        ``load_state_dict`` on every gradient step. Apart from being unnecessary
        for this GroupNorm-based network, that path is brittle with modern
        PyTorch. All mutable learned state here is represented by parameters;
        non-learned buffers are constant after network construction.
        """
        decay = self.ema.get_decay(self.ema.optimization_step)
        ema_parameters = self._averaged_ema_parameters
        model_parameters = self._model_ema_parameters
        if len(ema_parameters) != len(model_parameters):
            raise RuntimeError("EMA and policy parameter structures do not match")
        torch._foreach_lerp_(ema_parameters, model_parameters, 1.0 - decay)
        self.ema.decay = decay
        self.ema.optimization_step += 1
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        for key, name in (
            ("positive_loss", "ReferenceMargin/PositiveLoss"),
            ("failure_loss", "ReferenceMargin/FailureLoss"),
            ("reference_loss", "ReferenceMargin/ReferenceLoss"),
            ("negative_energy_gap", "ReferenceMargin/NegativeEnergyGap"),
            ("active_margin_fraction", "ReferenceMargin/ActiveFraction"),
            ("negative_batch_fraction", "ReferenceMargin/NegativeBatchFraction"),
            ("margin_multiplier", "ReferenceMargin/Multiplier"),
        ):
            if key in info["losses"]:
                log[name] = info["losses"][key].item()
        for key, name in (
            ("positive_reference_loss", "HazardConstraint/PositiveReferenceLoss"),
            ("hazard_loss", "HazardConstraint/HazardLoss"),
            ("negative_reference_loss", "HazardConstraint/NegativeReferenceLoss"),
            ("hazard_current_risk", "HazardConstraint/CurrentRisk"),
            ("hazard_reference_risk", "HazardConstraint/ReferenceRisk"),
            ("hazard_risk_improvement", "HazardConstraint/RiskImprovement"),
            ("hazard_active_fraction", "HazardConstraint/ActiveFraction"),
            ("hazard_batch_fraction", "HazardConstraint/BatchFraction"),
            ("constraint_multiplier", "HazardConstraint/Multiplier"),
        ):
            if key in info["losses"]:
                log[name] = info["losses"][key].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        model_dict = {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }
        if self.reference_policy_enabled:
            if self.reference_nets is None:
                self._create_reference_nets()
            model_dict["reference_nets"] = self.reference_nets.state_dict()
            model_dict["reference_margin_step"] = self.reference_margin_step
        return model_dict

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if self.reference_policy_enabled:
            if model_dict.get("reference_nets", None) is not None:
                self._create_reference_nets()
                self.reference_nets.load_state_dict(model_dict["reference_nets"])
                self.reference_margin_step = model_dict.get(
                    "reference_margin_step", 0
                )
            else:
                # The deployed policy uses EMA weights. Start both the trainable
                # and frozen policies from that exact deployed function.
                constraint_config = (
                    self.algo_config.hazard_constraint
                    if self.hazard_constraint_enabled
                    else self.algo_config.reference_margin
                )
                if (
                    constraint_config.initialize_from_ema
                    and model_dict.get("ema", None) is not None
                ):
                    self.nets.load_state_dict(model_dict["ema"])
                self._create_reference_nets()

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
