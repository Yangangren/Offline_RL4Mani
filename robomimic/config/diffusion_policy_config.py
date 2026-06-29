"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"

    def train_config(self):
        """
        Setting up training parameters for Diffusion Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(DiffusionPolicyConfig, self).train_config()
        
        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.horizon.observation_horizon
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16
        
        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        
        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # Optional offline anti-imitation objective. Samples from datasets
        # marked ``anti_failure: true`` are pushed a finite denoising-energy
        # margin above a frozen copy of the initialization policy.
        self.algo.reference_margin.enabled = False
        self.algo.reference_margin.margin = 0.05
        self.algo.reference_margin.failure_weight = 0.02
        self.algo.reference_margin.reference_weight = 0.25
        self.algo.reference_margin.warmup_steps = 250
        self.algo.reference_margin.ramp_steps = 500
        self.algo.reference_margin.normalized_margin = True
        self.algo.reference_margin.initialize_from_ema = True

        # Optional failure-aware policy constraint. High-risk rollout chunks
        # are not imitation targets. Instead, a frozen causal prefix-risk model
        # scores actions sampled from the current and initialization policies.
        # The current policy is penalized only when it does not reduce positive
        # incremental action risk by the requested log-odds margin.
        self.algo.hazard_constraint.enabled = False
        self.algo.hazard_constraint.checkpoint = None
        self.algo.hazard_constraint.margin = 0.1
        self.algo.hazard_constraint.weight = 0.05
        self.algo.hazard_constraint.positive_reference_weight = 0.05
        self.algo.hazard_constraint.negative_reference_weight = 0.1
        self.algo.hazard_constraint.warmup_steps = 250
        self.algo.hazard_constraint.ramp_steps = 500
        self.algo.hazard_constraint.sampling_steps = 10
        self.algo.hazard_constraint.action_samples = 2
        self.algo.hazard_constraint.action_start_index = 1
        self.algo.hazard_constraint.initialize_from_ema = True
        
        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'
