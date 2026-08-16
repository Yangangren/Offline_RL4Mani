#!/usr/bin/env python3
"""Collect policy rollouts in restart-safe shards and merge them.

Each shard is produced in a fresh process, either by
``robomimic.scripts.run_trained_agent`` for a standard policy checkpoint or by
``scripts/eval_rgb_dp_idql.py`` for a chunk-IDQL checkpoint. Simulator /
renderer crashes therefore only invalidate a small shard. The merged output
follows robomimic's normal HDF5 structure and adds masks:

* ``mask/all_rollouts``
* ``mask/all``
* ``mask/success``
* ``mask/failure``

The per-shard logs are streamed in real time, which makes native MuJoCo /
robosuite crashes visible while collection is running.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    os.environ.get(
        "ROBOMIMIC_PYTHON",
        "/home/ryan/miniconda3/envs/robomimic_stable/bin/python",
    )
)
COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}

PROVENANCE_KEY = "experiment_provenance"
PROVENANCE_SCHEMA_VERSION = 2
SHARD_PROVENANCE_ATTR = "collection_provenance_json"
UINT32_SEED_MODULUS = 1 << 32
COLLECTOR_SEED_SCHEME = "identity_if_uint32_else_sha256_v2(raw_integer_seed)"


def to_uint32_seed(value: int) -> int:
    """Map any integer deterministically into NumPy's accepted seed range."""
    raw_seed = int(value)
    if 0 <= raw_seed < UINT32_SEED_MODULUS:
        return raw_seed
    encoded = f"collect_rollout_shards_seed_v2:{raw_seed}".encode("ascii")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def derive_shard_attempt_seed(args, spec: dict, attempt: int) -> int:
    """Derive the actual evaluator seed while retaining the logical seed."""
    if spec.get("policy_seed") is not None:
        return to_uint32_seed(spec["seed"])
    nominal_seed = int(spec.get("nominal_seed", spec["seed"]))
    raw_seed = nominal_seed + (int(attempt) - 1) * int(args.retry_seed_offset)
    return to_uint32_seed(raw_seed)


def validate_collector_seeds(args) -> None:
    """Reject invalid or aliased configured collector seeds."""
    if int(args.seed_base) < 0:
        raise ValueError("--seed-base must be non-negative")
    if int(args.retry_seed_offset) < 0:
        raise ValueError("--retry-seed-offset must be non-negative")
    if args.num_env_seeds is not None and int(args.num_env_seeds) <= 0:
        raise ValueError("--num-env-seeds must be positive")
    if args.policy_seeds is None:
        return
    if len(args.policy_seeds) == 0:
        raise ValueError("--policy-seeds requires at least one seed")
    logical_seeds = [int(seed) for seed in args.policy_seeds]
    if any(seed < 0 for seed in logical_seeds):
        raise ValueError("--policy-seeds must be non-negative")
    effective_seeds = [to_uint32_seed(seed) for seed in logical_seeds]
    if len(set(effective_seeds)) != len(effective_seeds):
        raise ValueError(
            "--policy-seeds produce duplicate effective uint32 seeds under "
            f"{COLLECTOR_SEED_SCHEME}"
        )


class TeeLogger:
    def __init__(self, path: Path, mode: str = "w"):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open(mode, buffering=1)

    def write(self, text: str) -> None:
        print(text, end="", flush=True)
        self.file.write(text)
        self.file.flush()

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def close(self) -> None:
        self.file.close()


def process_env(cache_suffix: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_collect_pycache_{cache_suffix}"
    return env


def sorted_demo_keys(dataset: h5py.File) -> list[str]:
    return sorted(
        dataset["data"].keys(),
        key=lambda key: int(key.split("_")[-1]),
    )


def file_identity(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {
            "path": str(resolved),
            "exists": False,
            "size": None,
            "mtime_ns": None,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def common_collection_inputs(args: argparse.Namespace) -> dict[str, object]:
    idql_backend = args.idql_checkpoint is not None
    return {
        "policy_backend": "chunk_idql" if idql_backend else "robomimic_agent",
        "checkpoints": {
            "agent": file_identity(args.agent),
            "idql": file_identity(args.idql_checkpoint),
            "dp": file_identity(args.dp_checkpoint),
        },
        "dataset_obs": bool(args.dataset_obs),
        "idql_policy": (
            {
                "expected_task": args.expected_task,
                "device": args.device,
                "actor_source": args.actor_source,
                "critic_source": args.critic_source,
                "num_candidates": int(args.num_candidates),
                "candidate_batch_size": int(args.candidate_batch_size),
                "execution_horizon": int(args.execution_horizon),
                "selection": args.selection,
                "random_selection_probability": float(
                    args.random_selection_probability
                ),
                "clip_actions": bool(args.clip_actions),
                "require_success_condition_adapter": bool(
                    args.require_success_condition_adapter
                ),
                "forbid_success_condition_adapter": bool(
                    args.forbid_success_condition_adapter
                ),
                "inference_success_condition": float(
                    args.inference_success_condition
                ),
                "inference_condition_mask": float(args.inference_condition_mask),
                "env_hard_reset": bool(args.env_hard_reset),
                "reset_to_initial_state": bool(args.reset_to_initial_state),
            }
            if idql_backend
            else None
        ),
    }


def make_experiment_provenance(inputs: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "inputs": inputs,
    }


def shard_experiment_provenance(
    args: argparse.Namespace,
    spec: dict,
    attempt: int,
    *,
    common_inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    if common_inputs is None:
        common_inputs = common_collection_inputs(args)
    split_seed_grid = spec.get("policy_seed") is not None
    evaluator_seed = (
        None
        if split_seed_grid
        else derive_shard_attempt_seed(args, spec, attempt)
    )
    return make_experiment_provenance(
        {
            "scope": "rollout_shard",
            "common": common_inputs,
            "shard": {
                "shard_index": int(spec["shard_index"]),
                "rollouts_per_shard": int(args.rollouts_per_shard),
                "horizon": int(args.horizon),
                "attempt": int(attempt),
                "seed_scheme": COLLECTOR_SEED_SCHEME,
                "nominal_seed": int(spec.get("nominal_seed", spec["seed"])),
                "evaluator_seed": evaluator_seed,
                "nominal_env_seed": (
                    None
                    if spec.get("nominal_env_seed") is None
                    else int(spec["nominal_env_seed"])
                ),
                "env_seed": (
                    None
                    if spec.get("env_seed") is None
                    else int(spec["env_seed"])
                ),
                "nominal_policy_seed": (
                    None
                    if spec.get("nominal_policy_seed") is None
                    else int(spec["nominal_policy_seed"])
                ),
                "policy_seed": (
                    None
                    if spec.get("policy_seed") is None
                    else int(spec["policy_seed"])
                ),
                "env_index": (
                    None
                    if spec.get("env_index") is None
                    else int(spec["env_index"])
                ),
                "retry_seed_offset": (
                    None if split_seed_grid else int(args.retry_seed_offset)
                ),
            },
        }
    )


def expected_shard_provenances(
    args: argparse.Namespace,
    spec: dict,
    *,
    common_inputs: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    return [
        shard_experiment_provenance(
            args,
            spec,
            attempt,
            common_inputs=common_inputs,
        )
        for attempt in range(1, int(args.max_retries) + 1)
    ]


def collection_experiment_provenance(
    args: argparse.Namespace,
    shard_provenances: list[dict[str, object]],
    *,
    common_inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    if common_inputs is None:
        common_inputs = common_collection_inputs(args)
    return make_experiment_provenance(
        {
            "scope": "merged_rollout_collection",
            "common": common_inputs,
            "collection": {
                "num_shards": int(args.num_shards),
                "num_env_seeds": (
                    None
                    if args.num_env_seeds is None
                    else int(args.num_env_seeds)
                ),
                "policy_seeds": (
                    None
                    if args.policy_seeds is None
                    else [int(seed) for seed in args.policy_seeds]
                ),
                "rollouts_per_shard": int(args.rollouts_per_shard),
                "horizon": int(args.horizon),
                "seed_base": int(args.seed_base),
                "max_retries": int(args.max_retries),
                "retry_seed_offset": int(args.retry_seed_offset),
                "success_return_threshold": float(
                    args.success_return_threshold
                ),
                "min_success_rollouts": args.min_success_rollouts,
                "min_failure_rollouts": args.min_failure_rollouts,
                "shard_fingerprints": [
                    provenance["fingerprint"]
                    for provenance in shard_provenances
                ],
            },
        }
    )


def provenance_from_hdf5(dataset: h5py.File) -> dict | None:
    if "data" not in dataset:
        return None
    value = dataset["data"].attrs.get(SHARD_PROVENANCE_ATTR)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return None
    try:
        provenance = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return provenance if isinstance(provenance, dict) else None


def read_shard_provenance(path: Path) -> dict | None:
    try:
        with h5py.File(path, "r") as dataset:
            return provenance_from_hdf5(dataset)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def stamp_shard_provenance(
    path: Path,
    provenance: dict[str, object],
) -> None:
    encoded = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    with h5py.File(path, "r+") as dataset:
        dataset["data"].attrs[SHARD_PROVENANCE_ATTR] = encoded
        dataset.flush()


def valid_shard(
    path: Path,
    expected: int,
    require_obs: bool,
    require_idql_diagnostics: bool = False,
    expected_selection: str | None = None,
    expected_random_selection_probability: float | None = None,
    expected_provenances: list[dict[str, object]] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as dataset:
            if "data" not in dataset or len(dataset["data"]) != expected:
                return False
            if "env_args" not in dataset["data"].attrs:
                return False
            if expected_provenances is not None:
                stored_provenance = provenance_from_hdf5(dataset)
                if stored_provenance is None or not any(
                    stored_provenance == expected for expected in expected_provenances
                ):
                    return False
            if expected_selection is not None:
                stored_selection = dataset["data"].attrs.get("selection")
                if isinstance(stored_selection, bytes):
                    stored_selection = stored_selection.decode("utf-8")
                if stored_selection != expected_selection:
                    return False
            if expected_random_selection_probability is not None:
                stored_probability = dataset["data"].attrs.get(
                    "random_selection_probability"
                )
                if stored_probability is None or not np.isclose(
                    float(stored_probability),
                    float(expected_random_selection_probability),
                ):
                    return False
            keys = sorted_demo_keys(dataset)
            if len(keys) == 0:
                return False
            required = ("actions", "states", "rewards", "dones")
            if require_idql_diagnostics:
                required += (
                    "q_selected",
                    "selected_index",
                    "selection_is_random",
                    "selection_is_greedy",
                )
            for key in keys:
                episode = dataset[f"data/{key}"]
                if any(name not in episode for name in required):
                    return False
                if require_obs and "obs" not in episode:
                    return False
                if "model_file" not in episode.attrs:
                    return False
                num_samples = int(episode.attrs.get("num_samples", -1))
                if num_samples < 0:
                    return False
                if any(int(episode[name].shape[0]) != num_samples for name in required):
                    return False
            return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def parse_stats(text: str) -> dict | None:
    marker = "Average Rollout Stats"
    index = text.rfind(marker)
    if index < 0:
        return None
    match = re.search(r"\{.*?\}", text[index + len(marker) :], re.S)
    return json.loads(match.group(0)) if match else None


def shard_outcome_counts(path: Path, success_return_threshold: float) -> tuple[int, int]:
    success = 0
    failure = 0
    with h5py.File(path, "r") as dataset:
        for key in sorted_demo_keys(dataset):
            episode_return = float(np.sum(dataset[f"data/{key}/rewards"][:]))
            if episode_return > success_return_threshold:
                success += 1
            else:
                failure += 1
    return success, failure


def outcome_targets_reached(args, success: int, failure: int) -> bool:
    configured = False
    if args.min_success_rollouts is not None:
        configured = True
        if success < args.min_success_rollouts:
            return False
    if args.min_failure_rollouts is not None:
        configured = True
        if failure < args.min_failure_rollouts:
            return False
    return configured


def seed_group_complete(args, spec: dict) -> bool:
    """Only stop after all policy seeds for the current environment seed."""
    if args.policy_seeds is None:
        return True
    return (int(spec["shard_index"]) + 1) % len(args.policy_seeds) == 0


def build_shard_specs(args) -> list[dict]:
    if args.policy_seeds is None:
        return [
            {
                "shard_index": shard_index,
                "seed": to_uint32_seed(args.seed_base + shard_index),
                "nominal_seed": int(args.seed_base) + shard_index,
                "env_seed": None,
                "policy_seed": None,
            }
            for shard_index in range(args.num_shards)
        ]

    specs = []
    for env_index in range(args.num_env_seeds):
        nominal_env_seed = int(args.seed_base) + env_index
        env_seed = to_uint32_seed(nominal_env_seed)
        for configured_policy_seed in args.policy_seeds:
            nominal_policy_seed = int(configured_policy_seed)
            policy_seed = to_uint32_seed(nominal_policy_seed)
            specs.append(
                {
                    "shard_index": len(specs),
                    "seed": env_seed,
                    "nominal_seed": nominal_env_seed,
                    "env_seed": env_seed,
                    "nominal_env_seed": nominal_env_seed,
                    "policy_seed": policy_seed,
                    "nominal_policy_seed": nominal_policy_seed,
                    "env_index": env_index,
                }
            )
    return specs


def rollout_command(args, shard: Path, spec: dict, attempt: int) -> list[str]:
    attempt_seed = derive_shard_attempt_seed(args, spec, attempt)

    if args.idql_checkpoint is not None:
        eval_output = (
            args.output_dir
            / "eval_shards"
            / f"shard_{int(spec['shard_index']):03d}_attempt_{attempt}"
        )
        command = [
            str(PYTHON),
            "-B",
            "scripts/eval_rgb_dp_idql.py",
            "--idql-checkpoint",
            str(args.idql_checkpoint),
            "--dp-checkpoint",
            str(args.dp_checkpoint),
            "--output-dir",
            str(eval_output),
            "--device",
            args.device,
            "--actor-source",
            args.actor_source,
            "--critic-source",
            args.critic_source,
            "--n-rollouts",
            str(args.rollouts_per_shard),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(attempt_seed),
            "--num-candidates",
            str(args.num_candidates),
            "--candidate-batch-size",
            str(args.candidate_batch_size),
            "--execution-horizon",
            str(args.execution_horizon),
            "--selection",
            args.selection,
            "--random-selection-probability",
            str(args.random_selection_probability),
            "--inference-success-condition",
            str(args.inference_success_condition),
            "--inference-condition-mask",
            str(args.inference_condition_mask),
            "--dataset-path",
            str(shard),
            "--clip-actions" if args.clip_actions else "--no-clip-actions",
            (
                "--require-success-condition-adapter"
                if args.require_success_condition_adapter
                else "--no-require-success-condition-adapter"
            ),
            (
                "--forbid-success-condition-adapter"
                if args.forbid_success_condition_adapter
                else "--no-forbid-success-condition-adapter"
            ),
            "--env-hard-reset" if args.env_hard_reset else "--no-env-hard-reset",
            (
                "--reset-to-initial-state"
                if args.reset_to_initial_state
                else "--no-reset-to-initial-state"
            ),
        ]
        if spec.get("policy_seed") is not None:
            command.extend(["--env-seed", str(spec["env_seed"])])
            command.extend(["--policy-seed", str(spec["policy_seed"])])
        if args.expected_task is not None:
            command.extend(["--expected-task", args.expected_task])
        return command

    command = [
        str(PYTHON),
        "-B",
        "-m",
        "robomimic.scripts.run_trained_agent",
        "--agent",
        str(args.agent),
        "--n_rollouts",
        str(args.rollouts_per_shard),
        "--horizon",
        str(args.horizon),
        "--dataset_path",
        str(shard),
    ]
    if spec.get("policy_seed") is None:
        command.extend(["--seed", str(attempt_seed)])
    else:
        command.extend(["--env_seed", str(spec["env_seed"])])
        command.extend(["--policy_seed", str(spec["policy_seed"])])
    if args.dataset_obs:
        command.append("--dataset_obs")
    return command


def run_shard_attempt(command: list[str], logger: TeeLogger, cache_suffix: str) -> tuple[int, str]:
    shutil.rmtree(f"/tmp/robomimic_collect_pycache_{cache_suffix}", ignore_errors=True)
    logger.line("+ " + " ".join(command))
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=process_env(cache_suffix),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output.append(line)
        logger.write(line)
    proc.wait()
    proc.stdout.close()
    return proc.returncode, "".join(output)


def collect(args) -> list[dict]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_records = []
    total_success = 0
    total_failure = 0
    require_idql_diagnostics = args.idql_checkpoint is not None
    common_inputs = common_collection_inputs(args)
    specs = build_shard_specs(args)
    for spec in specs:
        shard_index = int(spec["shard_index"])
        seed = int(spec["seed"])
        nominal_seed = int(spec.get("nominal_seed", seed))
        shard = args.output_dir / f"shard_{shard_index:03d}.hdf5"
        log = args.output_dir / f"shard_{shard_index:03d}.log"
        base_record = {
            "shard_index": shard_index,
            "seed": seed,
            "nominal_seed": nominal_seed,
            "env_seed": spec.get("env_seed"),
            "nominal_env_seed": spec.get("nominal_env_seed"),
            "policy_seed": spec.get("policy_seed"),
            "nominal_policy_seed": spec.get("nominal_policy_seed"),
            "env_index": spec.get("env_index"),
        }
        expected_provenances = expected_shard_provenances(
            args,
            spec,
            common_inputs=common_inputs,
        )
        record = None
        if (
            not args.force_shards
            and valid_shard(
                shard,
                args.rollouts_per_shard,
                args.dataset_obs,
                require_idql_diagnostics=require_idql_diagnostics,
                expected_selection=(args.selection if require_idql_diagnostics else None),
                expected_random_selection_probability=(
                    args.random_selection_probability
                    if require_idql_diagnostics
                    else None
                ),
                expected_provenances=expected_provenances,
            )
        ):
            stored_provenance = read_shard_provenance(shard)
            assert stored_provenance is not None
            stored_shard_inputs = stored_provenance["inputs"]["shard"]
            source_attempt = int(stored_shard_inputs["attempt"])
            evaluator_seed = stored_shard_inputs["evaluator_seed"]
            record_seed = seed if evaluator_seed is None else int(evaluator_seed)
            print(
                f"[reuse] {shard} "
                f"provenance={stored_provenance['fingerprint']}",
                flush=True,
            )
            stats = parse_stats(log.read_text(errors="replace")) if log.exists() else None
            record = {
                "shard": str(shard),
                "log": str(log),
                **base_record,
                "seed": record_seed,
                "attempts": 0,
                "source_attempt": source_attempt,
                "stats": stats,
                PROVENANCE_KEY: stored_provenance,
            }
        else:
            if shard.exists():
                shard.unlink()
            logger = TeeLogger(log, mode="w")
            try:
                for attempt in range(1, args.max_retries + 1):
                    provenance = shard_experiment_provenance(
                        args,
                        spec,
                        attempt,
                        common_inputs=common_inputs,
                    )
                    command = rollout_command(args, shard, spec, attempt)
                    if spec.get("policy_seed") is None:
                        attempt_seed = derive_shard_attempt_seed(args, spec, attempt)
                        seed_desc = (
                            f"nominal_seed={nominal_seed} seed={seed} "
                            f"attempt_seed={attempt_seed}"
                        )
                        record_seed = attempt_seed
                    else:
                        seed_desc = (
                            f"env_seed={spec['env_seed']} "
                            f"policy_seed={spec['policy_seed']}"
                        )
                        record_seed = seed
                    logger.line(
                        f"\n[collect shard={shard_index} {seed_desc} "
                        f"attempt={attempt}/{args.max_retries} "
                        f"provenance={provenance['fingerprint']}]"
                    )
                    if shard.exists():
                        shard.unlink()
                    returncode, stdout = run_shard_attempt(
                        command,
                        logger,
                        cache_suffix=f"shard_{shard_index}_attempt_{attempt}",
                    )
                    ok = returncode == 0 and valid_shard(
                        shard,
                        args.rollouts_per_shard,
                        args.dataset_obs,
                        require_idql_diagnostics=require_idql_diagnostics,
                        expected_selection=(
                            args.selection if require_idql_diagnostics else None
                        ),
                        expected_random_selection_probability=(
                            args.random_selection_probability
                            if require_idql_diagnostics
                            else None
                        ),
                    )
                    if ok:
                        if common_collection_inputs(args) != common_inputs:
                            shard.unlink(missing_ok=True)
                            raise RuntimeError(
                                "checkpoint identity or another collection input "
                                f"changed while collecting shard {shard_index}"
                            )
                        stamp_shard_provenance(shard, provenance)
                        ok = valid_shard(
                            shard,
                            args.rollouts_per_shard,
                            args.dataset_obs,
                            require_idql_diagnostics=require_idql_diagnostics,
                            expected_selection=(
                                args.selection if require_idql_diagnostics else None
                            ),
                            expected_random_selection_probability=(
                                args.random_selection_probability
                                if require_idql_diagnostics
                                else None
                            ),
                            expected_provenances=[provenance],
                        )
                    if ok:
                        stats = parse_stats(stdout)
                        logger.line(f"[complete] {shard} stats={json.dumps(stats)}")
                        record = {
                            "shard": str(shard),
                            "log": str(log),
                            **base_record,
                            "seed": record_seed,
                            "attempts": attempt,
                            "source_attempt": attempt,
                            "stats": stats,
                            PROVENANCE_KEY: provenance,
                        }
                        break
                    logger.line(
                        f"[retry] shard={shard_index} returncode={returncode}; "
                        "removing incomplete shard"
                    )
                    if shard.exists():
                        shard.unlink()
                else:
                    raise RuntimeError(f"failed to collect shard {shard_index}")
            finally:
                logger.close()

        assert record is not None
        shard_success, shard_failure = shard_outcome_counts(
            shard,
            args.success_return_threshold,
        )
        record["num_success"] = shard_success
        record["num_failure"] = shard_failure
        shard_records.append(record)
        total_success += shard_success
        total_failure += shard_failure
        print(
            f"[outcomes] shards={len(shard_records)} rollouts="
            f"{total_success + total_failure} success={total_success} "
            f"failure={total_failure}",
            flush=True,
        )
        if (
            outcome_targets_reached(args, total_success, total_failure)
            and seed_group_complete(args, spec)
        ):
            print(
                "[quota complete] "
                f"success={total_success}/{args.min_success_rollouts} "
                f"failure={total_failure}/{args.min_failure_rollouts}",
                flush=True,
            )
            break

    if (
        (args.min_success_rollouts is not None or args.min_failure_rollouts is not None)
        and not outcome_targets_reached(args, total_success, total_failure)
    ):
        print(
            "[quota warning] maximum configured shards exhausted before all "
            f"targets were reached: success={total_success}/"
            f"{args.min_success_rollouts}, failure={total_failure}/"
            f"{args.min_failure_rollouts}",
            flush=True,
        )
    return shard_records


def validate_merge_shards(
    args: argparse.Namespace,
    shard_records: list[dict],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not shard_records:
        raise ValueError("cannot merge an empty shard collection")
    common_inputs = common_collection_inputs(args)
    specs_by_index = {
        int(spec["shard_index"]): spec for spec in build_shard_specs(args)
    }
    seen_indices: set[int] = set()
    provenances: list[dict[str, object]] = []
    require_idql_diagnostics = args.idql_checkpoint is not None

    for record in shard_records:
        try:
            shard_index = int(record["shard_index"])
            shard_path = Path(record["shard"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "each shard record must contain shard_index and shard"
            ) from error
        if shard_index in seen_indices:
            raise ValueError(f"duplicate shard_index in merge: {shard_index}")
        seen_indices.add(shard_index)
        spec = specs_by_index.get(shard_index)
        if spec is None:
            raise ValueError(
                f"shard_index {shard_index} is outside the configured seed grid"
            )

        stored_provenance = read_shard_provenance(shard_path)
        if stored_provenance is None:
            raise ValueError(
                f"shard is missing valid experiment provenance: {shard_path}"
            )
        if record.get(PROVENANCE_KEY) != stored_provenance:
            raise ValueError(
                f"shard record provenance differs from HDF5 metadata: {shard_path}"
            )
        expected_provenances = expected_shard_provenances(
            args,
            spec,
            common_inputs=common_inputs,
        )
        if not any(
            stored_provenance == expected
            for expected in expected_provenances
        ):
            raise ValueError(
                f"shard provenance does not match the current collection: {shard_path}"
            )
        stored_attempt = int(
            stored_provenance["inputs"]["shard"]["attempt"]
        )
        if int(record.get("source_attempt", -1)) != stored_attempt:
            raise ValueError(
                f"shard attempt metadata differs from provenance: {shard_path}"
            )
        if not valid_shard(
            shard_path,
            args.rollouts_per_shard,
            args.dataset_obs,
            require_idql_diagnostics=require_idql_diagnostics,
            expected_selection=(
                args.selection if require_idql_diagnostics else None
            ),
            expected_random_selection_probability=(
                args.random_selection_probability
                if require_idql_diagnostics
                else None
            ),
            expected_provenances=[stored_provenance],
        ):
            raise ValueError(
                f"shard failed structural or provenance validation: {shard_path}"
            )
        provenances.append(stored_provenance)

    return common_inputs, provenances


@contextlib.contextmanager
def atomic_hdf5_output(path: Path, validator):
    """Build and validate a sibling HDF5 file before atomically publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as output:
            yield output
        validator(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_hdf5_strings(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def validate_merged_output(
    path: Path,
    *,
    args: argparse.Namespace,
    shard_records: list[dict],
    common_inputs: dict[str, object],
    shard_provenances: list[dict[str, object]],
    collection_provenance: dict[str, object],
    expected_masks: dict[str, list[str]],
    all_keys: list[str],
    total_samples: int,
) -> None:
    """Validate the temporary merge and revalidate every source before publish."""
    final_common, final_provenances = validate_merge_shards(args, shard_records)
    if final_common != common_inputs or final_provenances != shard_provenances:
        raise RuntimeError(
            "checkpoint identity or shard provenance changed while merging"
        )

    require_idql_diagnostics = args.idql_checkpoint is not None
    if not valid_shard(
        path,
        len(all_keys),
        args.dataset_obs,
        require_idql_diagnostics=require_idql_diagnostics,
        expected_selection=(args.selection if require_idql_diagnostics else None),
        expected_random_selection_probability=(
            args.random_selection_probability
            if require_idql_diagnostics
            else None
        ),
        expected_provenances=[collection_provenance],
    ):
        raise RuntimeError(f"temporary merged dataset failed validation: {path}")

    source_fingerprints = {
        Path(record["shard"]).name: record[PROVENANCE_KEY]["fingerprint"]
        for record in shard_records
    }
    if len(source_fingerprints) != len(shard_records):
        raise ValueError("source shard basenames must be unique during merge")

    with h5py.File(path, "r") as dataset:
        if sorted_demo_keys(dataset) != all_keys:
            raise RuntimeError("temporary merged dataset has invalid demo keys")
        if int(dataset["data"].attrs.get("total", -1)) != int(total_samples):
            raise RuntimeError("temporary merged dataset has an invalid sample total")
        for mask_name, expected in expected_masks.items():
            mask_path = f"mask/{mask_name}"
            if mask_path not in dataset:
                raise RuntimeError(
                    f"temporary merged dataset is missing mask {mask_name}"
                )
            actual = _decode_hdf5_strings(dataset[mask_path][:])
            if actual != expected:
                raise RuntimeError(
                    f"temporary merged dataset has invalid mask {mask_name}"
                )
        for demo_key in all_keys:
            episode = dataset[f"data/{demo_key}"]
            source_name = episode.attrs.get("source_shard")
            fingerprint = episode.attrs.get(
                "source_shard_provenance_fingerprint"
            )
            if isinstance(source_name, bytes):
                source_name = source_name.decode("utf-8")
            if isinstance(fingerprint, bytes):
                fingerprint = fingerprint.decode("utf-8")
            if source_fingerprints.get(str(source_name)) != fingerprint:
                raise RuntimeError(
                    f"temporary merged dataset has invalid source provenance for {demo_key}"
                )


def merge(args, shard_records: list[dict]) -> dict:
    common_inputs, shard_provenances = validate_merge_shards(
        args,
        shard_records,
    )
    collection_provenance = collection_experiment_provenance(
        args, shard_provenances, common_inputs=common_inputs
    )

    merged_path = args.output_dir / args.merged_name
    merged_resolved = merged_path.resolve()
    if any(
        Path(record["shard"]).resolve() == merged_resolved
        for record in shard_records
    ):
        raise ValueError(
            f"merged output collides with a source shard: {merged_path}"
        )
    if merged_path.exists():
        if not args.force_merge:
            raise FileExistsError(
                f"{merged_path} exists. Pass --force-merge to overwrite it."
            )

    successes, failures, all_keys = [], [], []
    random_exploration_episodes, no_random_exploration_episodes = [], []
    episode_records = []
    total_samples = 0
    total_selection_decisions = 0
    total_random_selection_decisions = 0
    total_non_greedy_selection_decisions = 0
    env_args = None

    def validate_temporary_merge(path: Path) -> None:
        expected_masks = {
            "all_rollouts": all_keys,
            "all": all_keys,
            "success": successes,
            "failure": failures,
        }
        if args.idql_checkpoint is not None:
            expected_masks.update(
                {
                    "random_exploration": random_exploration_episodes,
                    "no_random_exploration": no_random_exploration_episodes,
                }
            )
        validate_merged_output(
            path,
            args=args,
            shard_records=shard_records,
            common_inputs=common_inputs,
            shard_provenances=shard_provenances,
            collection_provenance=collection_provenance,
            expected_masks=expected_masks,
            all_keys=all_keys,
            total_samples=total_samples,
        )

    with atomic_hdf5_output(merged_path, validate_temporary_merge) as output:
        output_data = output.create_group("data")
        output_index = 0
        for record in shard_records:
            with h5py.File(record["shard"], "r") as shard:
                if env_args is None:
                    env_args = shard["data"].attrs["env_args"]
                for key in sorted_demo_keys(shard):
                    output_key = f"demo_{output_index}"
                    shard.copy(shard[f"data/{key}"], output_data, name=output_key)
                    group = output_data[output_key]
                    episode_return = float(np.sum(group["rewards"][:]))
                    horizon = int(group.attrs["num_samples"])
                    is_success = episode_return > args.success_return_threshold
                    group.attrs["episode_return"] = episode_return
                    group.attrs["policy_success"] = bool(is_success)
                    group.attrs["source_shard"] = Path(record["shard"]).name
                    group.attrs["source_demo"] = key
                    group.attrs["source_shard_provenance_fingerprint"] = record[
                        PROVENANCE_KEY
                    ]["fingerprint"]
                    if record.get("env_seed") is not None:
                        group.attrs["env_seed"] = int(record["env_seed"])
                    if record.get("policy_seed") is not None:
                        group.attrs["policy_seed"] = int(record["policy_seed"])
                    if record.get("env_index") is not None:
                        group.attrs["env_index"] = int(record["env_index"])

                    episode_env_seed = group.attrs.get("env_seed", record.get("env_seed"))
                    episode_policy_seed = group.attrs.get(
                        "policy_seed",
                        record.get("policy_seed"),
                    )
                    if episode_env_seed is not None:
                        episode_env_seed = int(episode_env_seed)
                    if episode_policy_seed is not None:
                        episode_policy_seed = int(episode_policy_seed)

                    selection_decisions = 0
                    random_selection_decisions = 0
                    non_greedy_selection_decisions = 0
                    if "selection_is_random" in group:
                        random_values = np.asarray(
                            group["selection_is_random"][:],
                            dtype=np.int8,
                        )
                        greedy_values = np.asarray(
                            group["selection_is_greedy"][:],
                            dtype=np.int8,
                        )
                        valid_decisions = random_values >= 0
                        selection_decisions = int(np.sum(valid_decisions))
                        random_selection_decisions = int(
                            np.sum(random_values[valid_decisions] == 1)
                        )
                        non_greedy_selection_decisions = int(
                            np.sum(greedy_values[valid_decisions] == 0)
                        )
                        group.attrs["num_selection_decisions"] = selection_decisions
                        group.attrs["num_random_selection_decisions"] = (
                            random_selection_decisions
                        )
                        group.attrs["num_non_greedy_selection_decisions"] = (
                            non_greedy_selection_decisions
                        )
                        group.attrs["random_selection_decision_fraction"] = (
                            random_selection_decisions / max(1, selection_decisions)
                        )
                        if random_selection_decisions > 0:
                            random_exploration_episodes.append(output_key)
                        else:
                            no_random_exploration_episodes.append(output_key)

                    total_selection_decisions += selection_decisions
                    total_random_selection_decisions += random_selection_decisions
                    total_non_greedy_selection_decisions += (
                        non_greedy_selection_decisions
                    )

                    all_keys.append(output_key)
                    if is_success:
                        successes.append(output_key)
                    else:
                        failures.append(output_key)
                    total_samples += horizon
                    episode_records.append(
                        {
                            "demo_key": output_key,
                            "success": bool(is_success),
                            "return": episode_return,
                            "horizon": horizon,
                            "source_shard": str(record["shard"]),
                            "source_demo": key,
                            "env_seed": episode_env_seed,
                            "policy_seed": episode_policy_seed,
                            "env_index": record.get("env_index"),
                            "num_selection_decisions": selection_decisions,
                            "num_random_selection_decisions": (
                                random_selection_decisions
                            ),
                            "num_non_greedy_selection_decisions": (
                                non_greedy_selection_decisions
                            ),
                        }
                    )
                    output_index += 1

        output_data.attrs[SHARD_PROVENANCE_ATTR] = json.dumps(
            collection_provenance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

        output_data.attrs["env_args"] = env_args
        output_data.attrs["total"] = total_samples
        if args.idql_checkpoint is not None:
            output_data.attrs["selection"] = args.selection
            output_data.attrs["random_selection_probability"] = float(
                args.random_selection_probability
            )
        mask = output.create_group("mask")
        encoded_all = np.asarray(all_keys, dtype="S")
        mask["all_rollouts"] = encoded_all
        mask["all"] = encoded_all
        mask["success"] = np.asarray(successes, dtype="S")
        mask["failure"] = np.asarray(failures, dtype="S")
        if args.idql_checkpoint is not None:
            mask["random_exploration"] = np.asarray(
                random_exploration_episodes,
                dtype="S",
            )
            mask["no_random_exploration"] = np.asarray(
                no_random_exploration_episodes,
                dtype="S",
            )

    horizons = np.asarray([ep["horizon"] for ep in episode_records], dtype=np.float64)
    summary = {
        PROVENANCE_KEY: collection_provenance,
        "policy_backend": "chunk_idql" if args.idql_checkpoint is not None else "robomimic_agent",
        "agent": None if args.agent is None else str(args.agent),
        "idql_checkpoint": (
            None if args.idql_checkpoint is None else str(args.idql_checkpoint)
        ),
        "dp_checkpoint": None if args.dp_checkpoint is None else str(args.dp_checkpoint),
        "expected_task": args.expected_task,
        "actor_source": args.actor_source if args.idql_checkpoint is not None else None,
        "critic_source": args.critic_source if args.idql_checkpoint is not None else None,
        "num_candidates": args.num_candidates if args.idql_checkpoint is not None else None,
        "candidate_batch_size": (
            args.candidate_batch_size if args.idql_checkpoint is not None else None
        ),
        "execution_horizon": (
            args.execution_horizon if args.idql_checkpoint is not None else None
        ),
        "selection": args.selection if args.idql_checkpoint is not None else None,
        "random_selection_probability": (
            args.random_selection_probability
            if args.idql_checkpoint is not None
            else None
        ),
        "clip_actions": args.clip_actions if args.idql_checkpoint is not None else None,
        "inference_success_condition": (
            args.inference_success_condition if args.idql_checkpoint is not None else None
        ),
        "inference_condition_mask": (
            args.inference_condition_mask if args.idql_checkpoint is not None else None
        ),
        "merged_dataset": str(merged_path),
        "dataset_obs": bool(args.dataset_obs),
        "success_return_threshold": args.success_return_threshold,
        "min_success_rollouts": args.min_success_rollouts,
        "min_failure_rollouts": args.min_failure_rollouts,
        "outcome_targets_reached": outcome_targets_reached(
            args,
            len(successes),
            len(failures),
        ),
        "max_configured_rollouts": args.num_shards * args.rollouts_per_shard,
        "num_rollouts": len(all_keys),
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": len(successes) / max(1, len(all_keys)),
        "num_selection_decisions": total_selection_decisions,
        "num_random_selection_decisions": total_random_selection_decisions,
        "num_non_greedy_selection_decisions": (
            total_non_greedy_selection_decisions
        ),
        "random_selection_decision_fraction": (
            total_random_selection_decisions / max(1, total_selection_decisions)
        ),
        "non_greedy_selection_decision_fraction": (
            total_non_greedy_selection_decisions
            / max(1, total_selection_decisions)
        ),
        "num_random_exploration_episodes": len(random_exploration_episodes),
        "num_no_random_exploration_episodes": len(
            no_random_exploration_episodes
        ),
        "total_samples": total_samples,
        "mean_horizon": float(np.mean(horizons)) if len(horizons) else 0.0,
        "min_horizon": int(np.min(horizons)) if len(horizons) else 0,
        "max_horizon": int(np.max(horizons)) if len(horizons) else 0,
        "split_seed_grid": args.policy_seeds is not None,
        "num_env_seeds": args.num_env_seeds if args.policy_seeds is not None else None,
        "policy_seeds": args.policy_seeds,
        "shards": shard_records,
        "episodes": episode_records,
    }
    summary_path = args.output_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4))
    print(json.dumps(summary, indent=4), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--agent", type=Path)
    policy_group.add_argument("--idql-checkpoint", type=Path)
    parser.add_argument("--dp-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--expected-task",
        choices=("square", "can", "transport", "tool_hang"),
        default=None,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument(
        "--selection",
        choices=(
            "argmax",
            "greedy",
            "softmax",
            "advantage_softmax",
            "epsilon_greedy",
        ),
        default="argmax",
    )
    parser.add_argument(
        "--random-selection-probability",
        type=float,
        default=0.0,
        help=(
            "For epsilon_greedy collection, uniformly sample one candidate with "
            "this probability and otherwise execute argmax min(Q1,Q2)."
        ),
    )
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--env-hard-reset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reset-to-initial-state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-name", default="rollouts_raw.hdf5")
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--num-env-seeds", type=int, default=None)
    parser.add_argument("--policy-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--rollouts-per-shard", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--retry-seed-offset",
        type=int,
        default=100000,
        help=(
            "After a shard crash, retry with seed + k * retry_seed_offset. "
            "This avoids one pathological simulator seed blocking collection."
        ),
    )
    parser.add_argument("--dataset-obs", action="store_true")
    parser.add_argument("--force-shards", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--success-return-threshold", type=float, default=0.0)
    parser.add_argument("--min-success-rollouts", type=int, default=None)
    parser.add_argument("--min-failure-rollouts", type=int, default=None)
    args = parser.parse_args()
    try:
        validate_collector_seeds(args)
    except ValueError as error:
        parser.error(str(error))
    for key in ("agent", "idql_checkpoint", "dp_checkpoint"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    args.output_dir = args.output_dir.resolve()
    for key in ("agent", "idql_checkpoint", "dp_checkpoint"):
        value = getattr(args, key)
        if value is not None and not value.is_file():
            parser.error(f"--{key.replace('_', '-')} does not exist: {value}")
    if args.idql_checkpoint is not None:
        if args.dp_checkpoint is None:
            parser.error("--idql-checkpoint requires --dp-checkpoint")
        if args.dataset_obs:
            parser.error(
                "chunk-IDQL collection writes simulator states; convert them to RGB "
                "after merging instead of using --dataset-obs"
            )
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
    if (
        args.selection == "epsilon_greedy"
        and args.random_selection_probability > 0.0
        and args.num_candidates < 2
    ):
        parser.error("epsilon_greedy exploration requires --num-candidates >= 2")
    for key in (
        "num_shards",
        "rollouts_per_shard",
        "horizon",
        "max_retries",
        "num_candidates",
        "candidate_batch_size",
        "execution_horizon",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("min_success_rollouts", "min_failure_rollouts"):
        value = getattr(args, key)
        if value is not None and value < 0:
            parser.error(f"--{key.replace('_', '-')} must be non-negative")
    if args.policy_seeds is not None:
        if len(args.policy_seeds) == 0:
            parser.error("--policy-seeds requires at least one seed")
        if args.rollouts_per_shard != 1:
            parser.error("split env/policy seed collection requires --rollouts-per-shard 1")
        if args.num_env_seeds is None:
            args.num_env_seeds = args.num_shards
        args.num_shards = int(args.num_env_seeds) * len(args.policy_seeds)
    records = collect(args)
    merge(args, records)


if __name__ == "__main__":
    main()
