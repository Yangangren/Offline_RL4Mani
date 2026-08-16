import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_rgb_dp_idql_eval_grid.py"
LAUNCHER = ROOT / "run_rgb_dp_idql.sh"
SPEC = importlib.util.spec_from_file_location("rgb_dp_idql_eval_grid", SCRIPT)
EVAL_GRID = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL_GRID)


class MultiGpuEvalGridTest(unittest.TestCase):
    def run_launcher(self, stage):
        environment = os.environ.copy()
        environment.update(
            {
                "ROBOMIMIC_PYTHON": "/bin/echo",
                "IDQL_NUM_GPUS": "1",
                "EVAL_NUM_GPUS": "4",
                "EVAL_GPU_IDS": "2 4 6 7",
                "USER": environment.get("USER", "test"),
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER), "tool_hang", stage],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr

    def test_launcher_forwards_gpu_controls_only_to_eval_grid(self):
        grid_output = self.run_launcher("eval_grid_resilient")
        self.assertIn("--num-gpus 4", grid_output)
        self.assertIn("--gpu-ids 2 4 6 7", grid_output)

        direct_output = self.run_launcher("eval")
        self.assertNotIn("--num-gpus", direct_output)
        self.assertNotIn("--gpu-ids", direct_output)

    def test_resolve_visible_gpu_ids_and_child_environment(self):
        args = argparse.Namespace(device="cuda", num_gpus=2, gpu_ids=None)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5,7"}, clear=False):
            self.assertEqual(EVAL_GRID.resolve_gpu_ids(args), [3, 5])

        child_env = EVAL_GRID.process_env("unit_test", gpu_id=6)
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "6")
        self.assertEqual(child_env["MUJOCO_EGL_DEVICE_ID"], "6")

    def test_explicit_gpu_ids_determine_default_worker_count(self):
        args = argparse.Namespace(device="cuda", num_gpus=None, gpu_ids=[2, 4, 6])
        self.assertEqual(EVAL_GRID.resolve_gpu_ids(args), [2, 4, 6])

    def test_scheduler_uses_each_gpu_without_oversubscription(self):
        args = argparse.Namespace(
            num_candidates=[1, 8],
            seeds=[0, 1, 2],
            eval_gpu_ids=[0, 1, 2],
        )
        lock = threading.Lock()
        active = {}
        assignments = []
        summary_calls = []
        frozen_common = {"frozen": True}

        def fake_run_pair(
            args,
            num_candidates,
            seed,
            gpu_id=None,
            *,
            common_inputs=None,
        ):
            self.assertIs(common_inputs, frozen_common)
            with lock:
                self.assertEqual(active.get(gpu_id, 0), 0)
                active[gpu_id] = 1
                assignments.append((num_candidates, seed, gpu_id))
            time.sleep(0.005 * (1 + ((num_candidates + seed) % 3)))
            with lock:
                active[gpu_id] = 0
            return {"num_candidates": num_candidates, "seed": seed}

        def fake_summarize(results, args, *, common_inputs=None):
            self.assertIs(common_inputs, frozen_common)
            summary_calls.append(len(results))
            return {}

        with (
            mock.patch.object(
                EVAL_GRID,
                "common_experiment_inputs",
                return_value=frozen_common,
            ),
            mock.patch.object(EVAL_GRID, "run_pair", side_effect=fake_run_pair),
            mock.patch.object(EVAL_GRID, "validate_grid_results") as validate,
            mock.patch.object(EVAL_GRID, "summarize", side_effect=fake_summarize),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results = EVAL_GRID.run_grid(args)

        expected = {
            (num_candidates, seed)
            for num_candidates in args.num_candidates
            for seed in args.seeds
        }
        self.assertEqual(
            {(result["num_candidates"], result["seed"]) for result in results},
            expected,
        )
        self.assertEqual({gpu_id for _, _, gpu_id in assignments}, {0, 1, 2})
        self.assertEqual(summary_calls, [len(expected)])
        validate.assert_called_once()


class EvalGridCacheProvenanceTest(unittest.TestCase):
    @staticmethod
    def _rollout():
        return {
            "Return": 1.0,
            "Horizon": 7,
            "Success_Rate": 1.0,
        }

    def _make_args(self, root: Path) -> argparse.Namespace:
        idql_checkpoint = root / "idql.pt"
        dp_checkpoint = root / "dp.pth"
        idql_checkpoint.write_bytes(b"idql-checkpoint")
        dp_checkpoint.write_bytes(b"dp-checkpoint")
        output_dir = root / "eval"
        output_dir.mkdir(parents=True)
        return argparse.Namespace(
            idql_checkpoint=idql_checkpoint,
            dp_checkpoint=dp_checkpoint,
            output_dir=output_dir,
            expected_task="square",
            device="cpu",
            actor_source="hybrid_dp_chunk_actor",
            critic_source="online",
            candidate_batch_size=16,
            num_inference_steps=100,
            execution_horizon=8,
            selection="epsilon_greedy",
            softmax_temperature=1.0,
            random_selection_probability=0.25,
            clip_actions=True,
            diffusion_clip_sample=True,
            require_success_condition_adapter=True,
            forbid_success_condition_adapter=False,
            inference_success_condition=1.0,
            inference_condition_mask=1.0,
            env_hard_reset=False,
            reset_to_initial_state=False,
            n_rollouts=1,
            horizon=400,
            rollouts_per_chunk=1,
            accept_partial=True,
            max_retries=1,
            force=False,
            inter_chunk_sleep=0.0,
            num_candidates=[4],
            seeds=[3],
            eval_gpu_ids=[None],
        )

    def _pair_payload(
        self,
        args: argparse.Namespace,
        *,
        num_candidates: int = 4,
        seed: int = 3,
        common_inputs: dict[str, object] | None = None,
    ) -> dict:
        rollout = self._rollout()
        return {
            EVAL_GRID.PROVENANCE_KEY: EVAL_GRID.pair_experiment_provenance(
                args,
                num_candidates,
                seed,
                common_inputs=common_inputs,
            ),
            "num_candidates": num_candidates,
            "seed": seed,
            "average_rollout_stats": {
                "Num_Rollouts": 1,
                "Return": 1.0,
                "Horizon": 7.0,
                "Success_Rate": 1.0,
                "Num_Success": 1.0,
            },
            "rollouts": [rollout],
        }

    def _chunk_payload(
        self,
        args: argparse.Namespace,
        *,
        num_candidates: int = 4,
        seed: int = 3,
        chunk_index: int = 0,
        n_rollouts: int = 1,
    ) -> dict:
        chunk_seed = EVAL_GRID.derive_chunk_seed(seed, chunk_index)
        return {
            EVAL_GRID.PROVENANCE_KEY: EVAL_GRID.chunk_experiment_provenance(
                args,
                num_candidates,
                seed,
                chunk_index,
                chunk_seed,
                n_rollouts,
            ),
            "average_rollout_stats": {"Num_Rollouts": 1},
            "rollouts": [self._rollout()],
        }

    def test_provenance_is_canonical_and_covers_behavior_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            baseline = EVAL_GRID.pair_experiment_provenance(args, 4, 3)
            self.assertEqual(
                baseline,
                EVAL_GRID.pair_experiment_provenance(args, 4, 3),
            )
            json.dumps(baseline, allow_nan=False)

            mutations = (
                ("expected_task", "can"),
                ("device", "cuda"),
                ("actor_source", "external_dp_chunk_critic"),
                ("critic_source", "target"),
                ("candidate_batch_size", 8),
                ("num_inference_steps", 50),
                ("execution_horizon", 4),
                ("selection", "softmax"),
                ("softmax_temperature", 0.5),
                ("random_selection_probability", 0.5),
                ("clip_actions", False),
                ("diffusion_clip_sample", False),
                ("require_success_condition_adapter", False),
                ("forbid_success_condition_adapter", True),
                ("inference_success_condition", 0.0),
                ("inference_condition_mask", 0.0),
                ("env_hard_reset", True),
                ("reset_to_initial_state", True),
                ("n_rollouts", 2),
                ("horizon", 200),
                ("rollouts_per_chunk", 2),
                ("accept_partial", False),
            )
            for attribute, value in mutations:
                with self.subTest(attribute=attribute):
                    changed_args = argparse.Namespace(**vars(args))
                    setattr(changed_args, attribute, value)
                    changed = EVAL_GRID.pair_experiment_provenance(
                        changed_args,
                        4,
                        3,
                    )
                    self.assertNotEqual(
                        baseline["fingerprint"],
                        changed["fingerprint"],
                    )

            self.assertNotEqual(
                baseline["fingerprint"],
                EVAL_GRID.pair_experiment_provenance(args, 8, 3)["fingerprint"],
            )
            self.assertNotEqual(
                baseline["fingerprint"],
                EVAL_GRID.pair_experiment_provenance(args, 4, 9)["fingerprint"],
            )

            idql_identity = baseline["inputs"]["common"]["checkpoints"]["idql"]
            self.assertEqual(
                idql_identity["path"],
                str(args.idql_checkpoint.resolve()),
            )
            self.assertEqual(idql_identity["size"], args.idql_checkpoint.stat().st_size)
            self.assertEqual(
                idql_identity["mtime_ns"],
                args.idql_checkpoint.stat().st_mtime_ns,
            )

            stat = args.idql_checkpoint.stat()
            os.utime(
                args.idql_checkpoint,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
            )
            mtime_changed = EVAL_GRID.pair_experiment_provenance(args, 4, 3)
            self.assertNotEqual(
                baseline["fingerprint"],
                mtime_changed["fingerprint"],
            )
            args.dp_checkpoint.write_bytes(b"changed-dp-checkpoint-size")
            dp_changed = EVAL_GRID.pair_experiment_provenance(args, 4, 3)
            self.assertNotEqual(
                mtime_changed["fingerprint"],
                dp_changed["fingerprint"],
            )

    def test_large_chunk_seed_is_bounded_and_hybrid_reports_dp_checkpoint(self):
        self.assertEqual(EVAL_GRID.derive_chunk_seed(3, 0), 300000)
        self.assertEqual(EVAL_GRID.derive_chunk_seed(0, 7), 7)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            EVAL_GRID.validate_grid_seeds([-1])
        with mock.patch.object(
            EVAL_GRID,
            "derive_chunk_seed",
            return_value=17,
        ):
            with self.assertRaisesRegex(ValueError, "duplicate effective"):
                EVAL_GRID.validate_grid_seeds([1, 2])

        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            large_seed = 8_675_310
            args.seeds = [large_seed]
            common_inputs = EVAL_GRID.common_experiment_inputs(args)
            observed_chunk_seeds = []

            def fake_run_chunk(**kwargs):
                observed_chunk_seeds.append(kwargs["chunk_seed"])
                return {
                    EVAL_GRID.PROVENANCE_KEY: kwargs["expected_provenance"],
                    "average_rollout_stats": {"Num_Rollouts": 1},
                    "rollouts": [self._rollout()],
                }

            with (
                mock.patch.object(
                    EVAL_GRID,
                    "run_chunk",
                    side_effect=fake_run_chunk,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = EVAL_GRID.run_pair(
                    args,
                    4,
                    large_seed,
                    common_inputs=common_inputs,
                )

            expected_seed = EVAL_GRID.derive_chunk_seed(large_seed, 0)
            self.assertEqual(observed_chunk_seeds, [expected_seed])
            self.assertGreaterEqual(expected_seed, 0)
            self.assertLess(expected_seed, EVAL_GRID.UINT32_SEED_MODULUS)
            self.assertEqual(
                expected_seed,
                EVAL_GRID.derive_chunk_seed(large_seed, 0),
            )
            self.assertEqual(result["chunks"][0]["chunk_seed"], expected_seed)
            self.assertEqual(result["dp_checkpoint"], str(args.dp_checkpoint))
            self.assertEqual(
                result[EVAL_GRID.PROVENANCE_KEY]["schema_version"],
                2,
            )
            self.assertEqual(
                result[EVAL_GRID.PROVENANCE_KEY]["inputs"]["chunk_seed_scheme"],
                EVAL_GRID.CHUNK_SEED_SCHEME,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                summary = EVAL_GRID.summarize(
                    [result],
                    args,
                    common_inputs=common_inputs,
                )
            self.assertEqual(summary["dp_checkpoint"], str(args.dp_checkpoint))

    def test_run_grid_rejects_checkpoint_change_before_final_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            args.seeds = [3, 4]
            observed_common = []

            def fake_run_pair(
                args,
                num_candidates,
                seed,
                gpu_id=None,
                *,
                common_inputs=None,
            ):
                observed_common.append(common_inputs)
                result = self._pair_payload(
                    args,
                    num_candidates=num_candidates,
                    seed=seed,
                    common_inputs=common_inputs,
                )
                if seed == 4:
                    args.idql_checkpoint.write_bytes(b"changed-during-grid")
                return result

            with (
                mock.patch.object(
                    EVAL_GRID,
                    "run_pair",
                    side_effect=fake_run_pair,
                ),
                mock.patch.object(EVAL_GRID, "summarize") as summarize,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "changed while running the evaluation grid",
                ):
                    EVAL_GRID.run_grid(args)

            self.assertEqual(len(observed_common), 2)
            self.assertTrue(
                all(common is observed_common[0] for common in observed_common)
            )
            summarize.assert_not_called()

    def test_run_grid_rejects_pair_with_different_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            args.seeds = [3, 4]

            def fake_run_pair(
                args,
                num_candidates,
                seed,
                gpu_id=None,
                *,
                common_inputs=None,
            ):
                result_common = common_inputs
                if seed == 4:
                    result_common = json.loads(json.dumps(common_inputs))
                    result_common["selection"] = "argmax"
                return self._pair_payload(
                    args,
                    num_candidates=num_candidates,
                    seed=seed,
                    common_inputs=result_common,
                )

            with (
                mock.patch.object(
                    EVAL_GRID,
                    "run_pair",
                    side_effect=fake_run_pair,
                ),
                mock.patch.object(EVAL_GRID, "summarize") as summarize,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "provenance does not match pair",
                ):
                    EVAL_GRID.run_grid(args)

            summarize.assert_not_called()

    def test_matching_pair_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            cached = self._pair_payload(args)
            final_json = args.output_dir / "one_step_idql_N4_seed3.json"
            EVAL_GRID.atomic_write_json(final_json, cached)

            with (
                mock.patch.object(EVAL_GRID, "run_chunk") as run_chunk,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = EVAL_GRID.run_pair(args, 4, 3)

            run_chunk.assert_not_called()
            self.assertEqual(result, cached)

    def test_legacy_pair_cache_is_rerun_and_stamped(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            final_json = args.output_dir / "one_step_idql_N4_seed3.json"
            legacy = self._pair_payload(args)
            legacy.pop(EVAL_GRID.PROVENANCE_KEY)
            EVAL_GRID.atomic_write_json(final_json, legacy)
            chunk = self._chunk_payload(args)

            with (
                mock.patch.object(EVAL_GRID, "run_chunk", return_value=chunk) as run_chunk,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = EVAL_GRID.run_pair(args, 4, 3)

            run_chunk.assert_called_once()
            expected = EVAL_GRID.pair_experiment_provenance(args, 4, 3)
            self.assertEqual(result[EVAL_GRID.PROVENANCE_KEY], expected)
            self.assertEqual(
                json.loads(final_json.read_text())[EVAL_GRID.PROVENANCE_KEY],
                expected,
            )

    def test_pair_cache_with_changed_behavior_is_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            final_json = args.output_dir / "one_step_idql_N4_seed3.json"
            EVAL_GRID.atomic_write_json(final_json, self._pair_payload(args))
            args.horizon = 200
            chunk = self._chunk_payload(args)

            with (
                mock.patch.object(EVAL_GRID, "run_chunk", return_value=chunk) as run_chunk,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = EVAL_GRID.run_pair(args, 4, 3)

            run_chunk.assert_called_once()
            self.assertEqual(
                result[EVAL_GRID.PROVENANCE_KEY],
                EVAL_GRID.pair_experiment_provenance(args, 4, 3),
            )

    def test_matching_chunk_and_partial_caches_are_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            chunk_dir = (
                args.output_dir
                / "chunks"
                / "N4_seed3_chunk000"
            )
            chunk_dir.mkdir(parents=True)
            chunk_json = chunk_dir / "one_step_idql_N4_seed300000.json"
            EVAL_GRID.atomic_write_json(chunk_json, self._chunk_payload(args))
            logger = mock.Mock()

            with mock.patch.object(EVAL_GRID.subprocess, "Popen") as popen:
                result = EVAL_GRID.run_chunk(
                    args=args,
                    num_candidates=4,
                    seed=3,
                    chunk_index=0,
                    chunk_seed=300000,
                    n_rollouts=1,
                    logger=logger,
                )
            popen.assert_not_called()
            self.assertEqual(
                result[EVAL_GRID.PROVENANCE_KEY],
                EVAL_GRID.chunk_experiment_provenance(
                    args,
                    4,
                    3,
                    0,
                    300000,
                    1,
                ),
            )

            chunk_json.unlink()
            partial_json = chunk_dir / "one_step_idql_N4_seed300000_partial.json"
            partial = self._chunk_payload(args, n_rollouts=2)
            partial["completed_rollouts"] = 1
            EVAL_GRID.atomic_write_json(partial_json, partial)
            with mock.patch.object(EVAL_GRID.subprocess, "Popen") as popen:
                resumed_partial = EVAL_GRID.run_chunk(
                    args=args,
                    num_candidates=4,
                    seed=3,
                    chunk_index=0,
                    chunk_seed=300000,
                    n_rollouts=2,
                    logger=logger,
                )
            popen.assert_not_called()
            self.assertEqual(len(resumed_partial["rollouts"]), 1)

    def test_legacy_chunk_is_rerun_then_stamped(self):
        class FakeProcess:
            def __init__(self):
                self.returncode = 0
                self.stdout = io.StringIO("generated\n")

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_args(Path(temporary))
            chunk_dir = args.output_dir / "chunks" / "N4_seed3_chunk000"
            chunk_dir.mkdir(parents=True)
            chunk_json = chunk_dir / "one_step_idql_N4_seed300000.json"
            legacy = self._chunk_payload(args)
            legacy.pop(EVAL_GRID.PROVENANCE_KEY)
            EVAL_GRID.atomic_write_json(chunk_json, legacy)
            logger = mock.Mock()

            def fake_popen(*unused_args, **unused_kwargs):
                EVAL_GRID.atomic_write_json(
                    chunk_json,
                    {
                        "average_rollout_stats": {"Num_Rollouts": 1},
                        "rollouts": [self._rollout()],
                    },
                )
                return FakeProcess()

            with (
                mock.patch.object(
                    EVAL_GRID.subprocess,
                    "Popen",
                    side_effect=fake_popen,
                ) as popen,
                mock.patch.object(EVAL_GRID.shutil, "rmtree"),
            ):
                result = EVAL_GRID.run_chunk(
                    args=args,
                    num_candidates=4,
                    seed=3,
                    chunk_index=0,
                    chunk_seed=300000,
                    n_rollouts=1,
                    logger=logger,
                )

            popen.assert_called_once()
            expected = EVAL_GRID.chunk_experiment_provenance(
                args,
                4,
                3,
                0,
                300000,
                1,
            )
            self.assertEqual(result[EVAL_GRID.PROVENANCE_KEY], expected)
            self.assertEqual(
                json.loads(chunk_json.read_text())[EVAL_GRID.PROVENANCE_KEY],
                expected,
            )


if __name__ == "__main__":
    unittest.main()
