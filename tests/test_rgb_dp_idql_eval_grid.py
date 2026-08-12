import argparse
import contextlib
import importlib.util
import io
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_rgb_dp_idql_eval_grid.py"
SPEC = importlib.util.spec_from_file_location("rgb_dp_idql_eval_grid", SCRIPT)
EVAL_GRID = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL_GRID)


class MultiGpuEvalGridTest(unittest.TestCase):
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

        def fake_run_pair(args, num_candidates, seed, gpu_id=None):
            with lock:
                self.assertEqual(active.get(gpu_id, 0), 0)
                active[gpu_id] = 1
                assignments.append((num_candidates, seed, gpu_id))
            time.sleep(0.005 * (1 + ((num_candidates + seed) % 3)))
            with lock:
                active[gpu_id] = 0
            return {"num_candidates": num_candidates, "seed": seed}

        def fake_summarize(results, args):
            summary_calls.append(len(results))
            return {}

        with (
            mock.patch.object(EVAL_GRID, "run_pair", side_effect=fake_run_pair),
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
        self.assertEqual(len(summary_calls), len(expected))


if __name__ == "__main__":
    unittest.main()
