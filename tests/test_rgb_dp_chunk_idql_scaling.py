import argparse
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from robomimic.utils.dataset import (
    SequenceDataset,
    SparseChunkSequenceDataset,
)
from robomimic.algo.diffusion_policy import (
    DiffusionPolicyUNet,
    SuccessConditionFiLM,
    SuccessConditionResidual,
)
from robomimic.models.diffusion_policy_nets import ConditionalUnet1D


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "train_rgb_dp_chunk_idql_test_module",
    SCRIPTS / "train_rgb_dp_chunk_idql.py",
)
CHUNK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CHUNK
SPEC.loader.exec_module(CHUNK)

def _distributed_gradient_worker(rank, init_path):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        context = CHUNK.DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=rank,
            world_size=2,
            backend="gloo",
            device=torch.device("cpu"),
        )
        shared = torch.nn.Parameter(torch.tensor([0.0]))
        shared.grad = torch.tensor([float(rank + 1)])
        globally_unused = torch.nn.Parameter(torch.tensor([0.0]))
        used_on_rank_zero = torch.nn.Parameter(torch.tensor([0.0]))
        if rank == 0:
            used_on_rank_zero.grad = torch.tensor([4.0])
        CHUNK.all_reduce_gradients(
            [shared, globally_unused, used_on_rank_zero],
            context,
            bucket_cap_mb=0.001,
        )
        torch.testing.assert_close(shared.grad, torch.tensor([1.5]))
        if globally_unused.grad is not None:
            raise AssertionError("globally unused gradient was not restored to None")
        torch.testing.assert_close(used_on_rank_zero.grad, torch.tensor([2.0]))
    finally:
        torch.distributed.destroy_process_group()


def _distributed_masked_dynamics_worker(rank, init_path):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        context = CHUNK.DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=rank,
            world_size=2,
            backend="gloo",
            device=torch.device("cpu"),
        )
        all_bases = torch.tensor(
            [
                [
                    [0.2, -0.5, 0.7],
                    [0.8, 0.1, -0.4],
                    [-0.3, 0.6, 0.9],
                ],
                [
                    [0.5, 0.4, -0.8],
                    [-0.6, 0.9, 0.3],
                    [0.7, -0.2, 0.1],
                ],
            ],
            dtype=torch.float64,
        )
        all_targets = torch.tensor(
            [
                [
                    [0.6, -0.1, 0.5],
                    [-0.2, 0.8, 0.4],
                    [0.9, 0.3, -0.7],
                ],
                [
                    [-0.4, 0.7, 0.2],
                    [0.3, -0.9, 0.6],
                    [0.1, 0.5, 0.8],
                ],
            ],
            dtype=torch.float64,
        )
        mask_cases = (
            # Unequal valid counts: one row on rank zero, three on rank one.
            torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float64),
            # One rank has no valid rows and must still backpropagate a zero.
            torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.float64),
            # The globally empty case must also remain graph-connected.
            torch.zeros((2, 3), dtype=torch.float64),
        )
        for all_masks in mask_cases:
            parameter = torch.nn.Parameter(
                torch.tensor([0.15, -0.25, 0.05], dtype=torch.float64)
            )
            l1, cosine, rmse = CHUNK.masked_dynamics_losses(
                all_bases[rank] + parameter,
                all_targets[rank],
                all_masks[rank],
                distributed_context=context,
            )
            loss = l1 + 0.7 * cosine
            loss.backward()
            if parameter.grad is None:
                raise AssertionError("empty masked loss disconnected its gradient")
            CHUNK.all_reduce_gradients([parameter], context)

            reference_parameter = torch.nn.Parameter(
                torch.tensor([0.15, -0.25, 0.05], dtype=torch.float64)
            )
            reference_l1, reference_cosine, reference_rmse = (
                CHUNK.masked_dynamics_losses(
                    all_bases.reshape(-1, 3) + reference_parameter,
                    all_targets.reshape(-1, 3),
                    all_masks.reshape(-1),
                )
            )
            reference_loss = reference_l1 + 0.7 * reference_cosine
            reference_loss.backward()
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                rtol=1e-10,
                atol=1e-12,
            )
            torch.testing.assert_close(
                rmse,
                reference_rmse,
                rtol=1e-10,
                atol=1e-12,
            )

            averaged_metrics = CHUNK.mean_distributed_scalars(
                {"l1": l1, "cosine": cosine, "rmse": rmse},
                context,
            )
            expected_metrics = {
                "l1": reference_l1,
                "cosine": reference_cosine,
                "rmse": reference_rmse,
            }
            for key, expected in expected_metrics.items():
                torch.testing.assert_close(
                    torch.tensor(averaged_metrics[key], dtype=torch.float64),
                    expected.detach(),
                    rtol=1e-6,
                    atol=1e-7,
                )
    finally:
        torch.distributed.destroy_process_group()


class DistributedOptimizationSemanticsTest(unittest.TestCase):
    def test_async_all_reduce_preserves_unused_gradient_semantics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = str(Path(temp_dir) / "distributed_init")
            torch.multiprocessing.spawn(
                _distributed_gradient_worker,
                args=(init_path,),
                nprocs=2,
                join=True,
            )

    def test_masked_dynamics_matches_one_global_valid_row_mean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = str(Path(temp_dir) / "masked_dynamics_init")
            torch.multiprocessing.spawn(
                _distributed_masked_dynamics_worker,
                args=(init_path,),
                nprocs=2,
                join=True,
            )

    def test_actor_ema_advances_once_per_optimizer_update(self):

        class EmaState:
            optimization_step = 10
            decay = 0.0

            @staticmethod
            def get_decay(step):
                self.assertEqual(step, 10)
                return 0.9

        averaged = torch.tensor([0.0])
        model = torch.tensor([1.0])
        actor = argparse.Namespace(
            ema=EmaState(),
            _averaged_ema_parameters=[averaged],
            _model_ema_parameters=[model],
        )
        DiffusionPolicyUNet._update_ema(actor)

        expected_decay = 0.9
        torch.testing.assert_close(
            averaged,
            torch.tensor([1.0 - expected_decay]),
        )
        self.assertAlmostEqual(actor.ema.decay, expected_decay)
        self.assertEqual(actor.ema.optimization_step, 11)

    def test_q_and_v_updates_share_one_gradient_sync_phase(self):
        torch.manual_seed(7)
        critics = torch.nn.ModuleList(
            [torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)]
        )
        targets = copy.deepcopy(critics)
        vf = torch.nn.Linear(3, 1)
        target_parameters_before = [
            [parameter.detach().clone() for parameter in target.parameters()]
            for target in targets
        ]
        critic_optimizers = [
            torch.optim.SGD(critic.parameters(), lr=0.1)
            for critic in critics
        ]
        vf_optimizer = torch.optim.SGD(vf.parameters(), lr=0.1)
        features = torch.randn(5, 3)
        critic_losses = [
            critic(features).square().mean() for critic in critics
        ]
        vf_loss = vf(features).square().mean()
        sync_calls = []

        def sync_once(parameters):
            sync_calls.append(list(parameters))

        CHUNK.update_networks(
            critics,
            targets,
            vf,
            critic_optimizers,
            vf_optimizer,
            critic_losses,
            vf_loss,
            target_tau=0.01,
            max_gradient_norm=None,
            gradient_sync_fn=sync_once,
        )

        self.assertEqual(len(sync_calls), 1)
        expected_count = sum(
            1 for module in [*critics, vf] for _ in module.parameters()
        )
        self.assertEqual(len(sync_calls[0]), expected_count)
        for critic, target, parameters_before in zip(
            critics, targets, target_parameters_before
        ):
            for online_parameter, target_parameter, target_before in zip(
                critic.parameters(), target.parameters(), parameters_before
            ):
                torch.testing.assert_close(
                    target_parameter,
                    0.99 * target_before + 0.01 * online_parameter,
                )

        for module in [*critics, vf]:
            self.assertTrue(
                all(parameter.grad is not None for parameter in module.parameters())
            )


class ConditionalActorRecipeTest(unittest.TestCase):
    @staticmethod
    def _minimal_actor():
        class Encoder(torch.nn.Module):
            def output_shape(self):
                return [3]

        actor = object.__new__(DiffusionPolicyUNet)
        actor.nets = torch.nn.ModuleDict(
            {
                "policy": torch.nn.ModuleDict(
                    {
                        "obs_encoder": Encoder(),
                        "noise_pred_net": ConditionalUnet1D(
                            input_dim=4,
                            global_cond_dim=6,
                            diffusion_step_embed_dim=16,
                            down_dims=(16, 32),
                            kernel_size=3,
                            n_groups=8,
                        ),
                    }
                )
            }
        )
        actor.algo_config = argparse.Namespace(
            horizon=argparse.Namespace(observation_horizon=2)
        )
        actor.device = torch.device("cpu")
        actor.ema = None
        actor.optimizers = {}
        return actor

    def test_condition_film_preserves_pretrained_unet_at_install(self):
        torch.manual_seed(7)
        unet = ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=6,
            diffusion_step_embed_dim=16,
            down_dims=(16, 32),
            kernel_size=3,
            n_groups=8,
        ).eval()
        actions = torch.randn(3, 16, 4)
        timesteps = torch.tensor([1, 2, 3])
        obs_condition = torch.randn(3, 6)
        with torch.no_grad():
            pretrained_output = unet(
                actions,
                timesteps,
                global_cond=obs_condition,
            )

        unet.install_condition_extension(8)
        adapter = SuccessConditionFiLM(global_cond_dim=6, hidden_dim=8)
        for condition, mask in ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0)):
            conditioned = adapter(
                obs_condition,
                torch.full((3,), condition),
                torch.full((3,), mask),
            )
            with torch.no_grad():
                conditioned_output = unet(
                    actions,
                    timesteps,
                    global_cond=conditioned,
                )
            torch.testing.assert_close(
                conditioned_output,
                pretrained_output,
                rtol=0.0,
                atol=0.0,
            )

        masked = adapter(
            obs_condition,
            torch.ones(3),
            torch.zeros(3),
        )
        torch.testing.assert_close(masked[:, 6:], torch.zeros(3, 8))

    def test_chunk_actor_optimizer_matches_default_dp_recipe(self):
        actor = argparse.Namespace(
            nets=torch.nn.ModuleDict(
                {
                    "policy": torch.nn.ModuleDict(
                        {
                            "condition_adapter": torch.nn.Linear(2, 4),
                            "noise_pred_net": torch.nn.Linear(4, 4),
                            "obs_encoder": torch.nn.Linear(4, 4),
                        }
                    )
                }
            ),
            optimizers={},
            lr_schedulers={},
            step_lr_schedulers_every_batch={},
        )
        CHUNK.configure_chunk_actor_optimizer(
            actor,
            adapter_lr=1e-4,
            unet_lr=1e-4,
            obs_encoder_lr=1e-4,
            scheduler_type="cosine",
            warmup_steps=500,
            total_steps=1000,
            num_cycles=0.5,
        )
        optimizer = actor.optimizers["policy"]
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(
            actor.lr_schedulers["policy"].base_lrs,
            [1e-4, 1e-4, 1e-4],
        )
        self.assertTrue(
            all(
                group["weight_decay"] == CHUNK.ACTOR_WEIGHT_DECAY
                for group in optimizer.param_groups
            )
        )

    def test_condition_film_checkpoint_installs_before_state_load(self):
        source = self._minimal_actor()
        source.install_success_condition_adapter(hidden_dim=8)
        state = copy.deepcopy(source.nets.state_dict())

        target = self._minimal_actor()
        target._install_success_condition_adapter_from_state(state)
        target.nets.load_state_dict(state)

        self.assertIsInstance(
            target.nets["policy"]["condition_adapter"],
            SuccessConditionFiLM,
        )
        self.assertEqual(
            target.nets["policy"]["noise_pred_net"].condition_extension_dim,
            8,
        )

    def test_legacy_condition_adapter_checkpoint_remains_loadable(self):
        source = self._minimal_actor()
        source._install_legacy_success_condition_adapter(hidden_dim=8)
        state = copy.deepcopy(source.nets.state_dict())

        target = self._minimal_actor()
        target._install_success_condition_adapter_from_state(state)
        target.nets.load_state_dict(state)

        self.assertIsInstance(
            target.nets["policy"]["condition_adapter"],
            SuccessConditionResidual,
        )
        self.assertEqual(
            target.nets["policy"]["noise_pred_net"].condition_extension_dim,
            0,
        )

    def test_configure_conditioned_actor_retains_legacy_adapter(self):
        actor = self._minimal_actor()
        actor._install_legacy_success_condition_adapter(hidden_dim=8)
        adapter = actor.nets["policy"]["condition_adapter"]

        CHUNK.configure_conditioned_actor(
            actor,
            argparse.Namespace(
                conditioned_actor=True,
                condition_hidden_dim=8,
                condition_dropout=0.0,
            ),
        )

        self.assertIs(actor.nets["policy"]["condition_adapter"], adapter)
        self.assertIsInstance(adapter, SuccessConditionResidual)
        self.assertEqual(actor.inference_success_condition, 1.0)
        self.assertEqual(actor.inference_success_condition_mask, 1.0)

    def test_chunk_defaults_use_human_only_and_default_dp_learning_rates(self):
        defaults = {
            action.dest: action.default
            for action in CHUNK.make_parser()._actions
        }
        self.assertEqual(defaults["actor_condition_mode"], "human_only")
        self.assertEqual(defaults["actor_adapter_lr"], 1e-4)
        self.assertEqual(defaults["actor_unet_lr"], 1e-4)
        self.assertEqual(defaults["actor_obs_encoder_lr"], 1e-4)
        self.assertEqual(defaults["actor_lr_warmup_steps"], 500)
        self.assertEqual(defaults["condition_hidden_dim"], 256)



class FreshOutputRecoveryTest(unittest.TestCase):
    def test_stale_latest_temporary_is_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "training"
            output_dir.mkdir()
            temporary = output_dir / ".latest.pt.tmp-12345"
            temporary.write_bytes(b"interrupted")

            with mock.patch.object(
                CHUNK,
                "process_is_running",
                return_value=False,
            ) as running:
                cleaned = CHUNK.prepare_fresh_output_directory(output_dir)

            self.assertEqual(cleaned, [temporary])
            self.assertFalse(temporary.exists())
            running.assert_called_once_with(12345)

    def test_unexpected_entry_prevents_any_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "training"
            output_dir.mkdir()
            temporary = output_dir / ".latest.pt.tmp-12345"
            temporary.write_bytes(b"interrupted")
            unexpected = output_dir / "training_config.json"
            unexpected.write_text("{}")

            with self.assertRaisesRegex(FileExistsError, "unexpected entries"):
                CHUNK.prepare_fresh_output_directory(output_dir)

            self.assertTrue(temporary.exists())
            self.assertTrue(unexpected.exists())

    def test_live_latest_temporary_is_not_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "training"
            output_dir.mkdir()
            temporary = output_dir / ".latest.pt.tmp-12345"
            temporary.write_bytes(b"in progress")

            with (
                mock.patch.object(CHUNK, "process_is_running", return_value=True),
                self.assertRaisesRegex(FileExistsError, "active checkpoint"),
            ):
                CHUNK.prepare_fresh_output_directory(output_dir)

            self.assertTrue(temporary.exists())


class ResumeSemanticValidationTest(unittest.TestCase):
    @staticmethod
    def _args():
        return argparse.Namespace(
            task="square",
            dataset=Path("/tmp/chunk_idql_mixed.hdf5"),
            dataset_identity={"dataset": {"size": 123}, "external_sources": {}},
            epochs=50,
            seed=0,
            batch_size=100,
            effective_global_batch_size=100,
            schedule_reference_batch_size=100,
            steps_per_epoch=None,
            validate_resume_only=False,
            chunk_horizon=8,
            discount=0.99,
            expectile=0.9,
            target_tau=0.01,
            critic_hidden_dims=(300, 400, 300),
            latent_dim=300,
            action_hidden_dim=128,
            num_attention_heads=4,
            num_action_conv_layers=2,
            dropout=0.0,
            num_critics=2,
            critic_group_norm=False,
            critic_late_fusion_key="robot0_gripper_qpos",
            dynamics_weight=0.5,
            dynamics_cosine_weight=0.5,
            dynamics_warmup_steps=1000,
            encoder_freeze_steps=0,
            vf_encoder_freeze_steps=1000,
            use_huber=True,
            max_gradient_norm=10.0,
            critic_vf_lr_scheduler="cosine",
            critic_vf_lr_warmup_steps=1000,
            critic_vf_lr_num_cycles=0.5,
            sparse_chunk_loader=True,
        )

    @staticmethod
    def _state(args):
        fields = (
            "epochs",
            "seed",
            "batch_size",
            "effective_global_batch_size",
            "schedule_reference_batch_size",
            "chunk_horizon",
            "discount",
            "expectile",
            "target_tau",
            "critic_hidden_dims",
            "latent_dim",
            "action_hidden_dim",
            "num_attention_heads",
            "num_action_conv_layers",
            "dropout",
            "num_critics",
            "critic_group_norm",
            "critic_late_fusion_key",
            "dynamics_weight",
            "dynamics_cosine_weight",
            "dynamics_warmup_steps",
            "encoder_freeze_steps",
            "vf_encoder_freeze_steps",
            "use_huber",
            "max_gradient_norm",
            "critic_vf_lr_scheduler",
            "critic_vf_lr_warmup_steps",
            "critic_vf_lr_num_cycles",
            "sparse_chunk_loader",
        )
        return {
            "task": args.task,
            "dataset": str(args.dataset),
            "dataset_identity": copy.deepcopy(args.dataset_identity),
            "args": {
                field: copy.deepcopy(getattr(args, field))
                for field in fields
            },
        }

    def test_resume_rejects_objective_change(self):
        args = self._args()
        state = self._state(args)
        CHUNK.validate_resume_semantics(state, args)

        state["args"]["expectile"] = 0.8
        with self.assertRaisesRegex(ValueError, "resume expectile"):
            CHUNK.validate_resume_semantics(state, args)


    def test_resume_rejects_dataset_identity_change(self):
        args = self._args()
        state = self._state(args)
        state["dataset_identity"]["dataset"]["size"] += 1

        with self.assertRaisesRegex(ValueError, "dataset identity"):
            CHUNK.validate_resume_semantics(state, args)


    def test_json_summary_publication_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text("old")
            CHUNK.atomic_write_json(path, {"epoch": 3})
            self.assertEqual(json.loads(path.read_text()), {"epoch": 3})
            self.assertFalse(any(path.parent.glob(".summary.json.tmp-*")))


class ChunkObjectiveSemanticsTest(unittest.TestCase):
    class ConstantCritic(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.q = torch.nn.Parameter(torch.tensor(float(value)))

        def forward(
            self,
            *,
            obs_dict,
            acts,
            action_mask,
            goal_dict,
            return_aux=False,
        ):
            del obs_dict, action_mask, goal_dict
            batch_size = acts.shape[0]
            q = self.q.expand(batch_size, 1)
            if return_aux:
                return {
                    "q": q,
                    "predicted_next_encoder": torch.ones(
                        batch_size, 3, device=acts.device
                    ),
                }
            return q

    class MarkerValue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *, obs_dict, goal_dict):
            del goal_dict
            return obs_dict["marker"] * self.scale

    class ConstantDynamicsTarget(torch.nn.Module):
        def forward(self, *, obs):
            return torch.ones(obs["marker"].shape[0], 3)

    def test_terminal_mask_and_gamma_power_define_q_backup(self):
        critics = torch.nn.ModuleList(
            [self.ConstantCritic(0.0), self.ConstantCritic(0.0)]
        )
        targets = torch.nn.ModuleList(
            [self.ConstantCritic(3.0), self.ConstantCritic(4.0)]
        )
        vf = self.MarkerValue()
        batch = {
            "obs": {"marker": torch.tensor([[1.0], [4.0]])},
            "next_obs": {"marker": torch.tensor([[10.0], [20.0]])},
            "actions": torch.zeros(2, 2, 1),
            "action_mask": torch.ones(2, 2),
            "reward": torch.tensor([[1.0], [2.0]]),
            "terminal": torch.tensor([[0.0], [1.0]]),
            "valid_length": torch.tensor([[2.0], [1.0]]),
            "exact_next": torch.tensor([[1.0], [0.0]]),
            "goal_obs": None,
        }

        critic_losses, vf_loss, info = CHUNK.compute_chunk_losses(
            critics,
            targets,
            self.ConstantDynamicsTarget(),
            vf,
            batch,
            discount=0.5,
            expectile=0.8,
            use_huber=False,
            dynamics_weight=0.5,
            dynamics_cosine_weight=0.5,
        )

        # Nonterminal row: 1 + 0.5**2 * 10 = 3.5.
        # Terminal row: reward 2 with no bootstrap.
        self.assertAlmostEqual(info["critic/q_target_mean"].item(), 2.75)
        for loss in critic_losses:
            self.assertAlmostEqual(loss.item(), 8.125)
        # min(target Q)=3; errors are -2 and +1 with expectile weights .8/.2.
        self.assertAlmostEqual(vf_loss.item(), 1.7)
        self.assertAlmostEqual(info["dynamics/weighted_loss"].item(), 0.0, places=6)

        critic_losses[0].backward()
        self.assertIsNone(vf.scale.grad)

    def test_dynamics_teacher_hard_sync_copies_actor_encoder(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with torch.no_grad():
            source.weight.fill_(0.25)
            source.bias.fill_(-0.5)
            target.weight.zero_()
            target.bias.zero_()
        actor = argparse.Namespace(
            nets=torch.nn.ModuleDict(
                {
                    "policy": torch.nn.ModuleDict(
                        {"obs_encoder": source}
                    )
                }
            ),
            ema=None,
        )

        audit = CHUNK.sync_actor_dynamics_target_encoder(target, actor)

        self.assertEqual(audit["tensor_count"], 2)
        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)
        self.assertIsNot(target.weight, source.weight)

class BatchScaledSemanticsTest(unittest.TestCase):
    def test_only_data_exposure_schedules_are_sample_scaled(self):
        args = argparse.Namespace(
            schedule_reference_batch_size=100,
            batch_size=100,
            actor_lr_warmup_steps=1000,
            critic_vf_lr_warmup_steps=1000,
            dynamics_warmup_steps=1000,
            encoder_freeze_steps=1000,
            vf_encoder_freeze_steps=1000,
            dynamics_target_sync_interval=1000,
            target_tau=0.01,
        )
        context = CHUNK.DistributedContext(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=8,
            backend="nccl",
            device=torch.device("cpu"),
        )
        CHUNK.configure_batch_semantics(args, context)

        self.assertEqual(args.effective_global_batch_size, 800)
        self.assertEqual(args.schedule_batch_ratio, 8.0)
        for field in CHUNK.SAMPLE_SCALED_STEP_FIELDS:
            self.assertEqual(getattr(args, f"resolved_{field}"), 125)
        self.assertEqual(
            args.resolved_dynamics_target_sync_interval,
            args.dynamics_target_sync_interval,
        )
        self.assertEqual(args.resolved_target_tau, args.target_tau)

    def test_arbitrary_per_gpu_batch_is_not_constrained(self):
        self.assertEqual(
            CHUNK.batch_scaled_step_count(1000, 100, 8 * 137),
            91,
        )
        self.assertEqual(CHUNK.batch_scaled_step_count(0, 100, 800), 0)


class SparseChunkSequenceDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "fixture.hdf5"
        length = 24
        with h5py.File(self.path, "w") as hdf5_file:
            demo = hdf5_file.create_group("data/demo_0")
            demo.attrs["num_samples"] = length
            obs = demo.create_group("obs")
            next_obs = demo.create_group("next_obs")
            image = np.arange(
                length * 2 * 2 * 3,
                dtype=np.uint8,
            ).reshape(length, 2, 2, 3)
            obs.create_dataset("camera", data=image)
            next_obs.create_dataset("camera", data=(image + 17).astype(np.uint8))
            actions = np.linspace(-0.8, 0.8, length * 2).reshape(length, 2)
            rewards = np.linspace(0.0, 1.0, length, dtype=np.float32)
            dones = np.zeros((length,), dtype=np.float32)
            dones[5] = 1.0
            demo.create_dataset("actions", data=actions.astype(np.float32))
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("task_rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            demo.create_dataset(
                "source_is_expert",
                data=np.ones((length,), dtype=np.float32),
            )
            demo.create_dataset(
                "actor_condition",
                data=np.ones((length,), dtype=np.float32),
            )

        self.base = SequenceDataset(
            hdf5_path=str(self.path),
            obs_keys=("camera",),
            action_keys=("actions",),
            dataset_keys=(
                "actions",
                "rewards",
                "task_rewards",
                "dones",
                "source_is_expert",
                "actor_condition",
            ),
            action_config={},
            frame_stack=2,
            seq_length=16,
            pad_frame_stack=True,
            pad_seq_length=True,
            hdf5_cache_mode=None,
            load_next_obs=True,
        )
        self.base.set_action_normalization_stats(
            {
                "actions": {
                    "offset": np.zeros((1, 2), dtype=np.float32),
                    "scale": np.ones((1, 2), dtype=np.float32),
                }
            }
        )

    def tearDown(self):
        self.base.close_and_delete_hdf5_handle()
        self.temp_dir.cleanup()

    def test_sparse_item_matches_every_consumed_dense_value(self):
        image_frames_read = []
        original_get = self.base.get_dataset_for_ep

        def counted_get(ep, key, seq_begin_index=None, seq_end_index=None):
            value = original_get(ep, key, seq_begin_index, seq_end_index)
            if key in ("obs/camera", "next_obs/camera"):
                image_frames_read.append(int(value.shape[0]))
            return value

        self.base.get_dataset_for_ep = counted_get
        dense = self.base[3]
        dense_frames = sum(image_frames_read)

        image_frames_read.clear()
        sparse_dataset = SparseChunkSequenceDataset(
            self.base,
            chunk_horizon=8,
            observation_horizon=2,
        )
        sparse = sparse_dataset[3]
        sparse_frames = sum(image_frames_read)

        self.assertEqual(dense_frames, 34)
        self.assertEqual(sparse_frames, 3)
        for key in (
            "actions",
            "rewards",
            "task_rewards",
            "dones",
            "source_is_expert",
            "actor_condition",
        ):
            np.testing.assert_array_equal(sparse[key], dense[key])
        np.testing.assert_array_equal(
            sparse["obs"]["camera"],
            dense["obs"]["camera"][:, :][:2],
        )

        current_index = 1
        chunk_dones = dense["dones"][current_index : current_index + 8]
        continuation = 1.0 - (chunk_dones > 0.5).astype(np.float32)
        action_mask = np.concatenate(
            (
                np.ones((1,), dtype=np.float32),
                np.cumprod(continuation[:-1], dtype=np.float32),
            )
        )
        next_index = current_index + int(action_mask.sum()) - 1
        np.testing.assert_array_equal(
            sparse["next_obs"]["camera"][0],
            dense["next_obs"]["camera"][next_index],
        )
        self.assertEqual(float(sparse["chunk_sparse_next_obs"]), 1.0)

    def test_processed_chunk_batch_is_identical(self):
        dense_item = self.base[3]
        sparse_item = SparseChunkSequenceDataset(
            self.base,
            chunk_horizon=8,
            observation_horizon=2,
        )[3]
        dense_batch = torch.utils.data.default_collate([dense_item])
        sparse_batch = torch.utils.data.default_collate([sparse_item])
        actor = argparse.Namespace(
            algo_config=argparse.Namespace(
                horizon=argparse.Namespace(observation_horizon=2)
            ),
            obs_shapes={"camera": (2, 2, 3)},
            device=torch.device("cpu"),
        )
        actor.postprocess_batch_for_training = (
            lambda batch, obs_normalization_stats: batch
        )

        dense_processed = CHUNK.process_chunk_batch(
            dense_batch,
            actor,
            None,
            chunk_horizon=8,
            discount=0.99,
            reward_mode="task",
        )
        sparse_processed = CHUNK.process_chunk_batch(
            sparse_batch,
            actor,
            None,
            chunk_horizon=8,
            discount=0.99,
            reward_mode="task",
        )
        for key in (
            "actions",
            "action_mask",
            "reward",
            "terminal",
            "valid_length",
            "exact_next",
        ):
            torch.testing.assert_close(sparse_processed[key], dense_processed[key])
        for key in ("obs", "next_obs"):
            torch.testing.assert_close(
                sparse_processed[key]["camera"],
                dense_processed[key]["camera"],
            )

if __name__ == "__main__":
    unittest.main()
