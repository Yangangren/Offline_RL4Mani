#!/usr/bin/env python3
"""Collect policy rollouts in restart-safe shards and merge them.

Each shard is produced by ``robomimic.scripts.run_trained_agent`` in a fresh
process, so simulator / renderer crashes only invalidate a small shard. The
merged output follows robomimic's normal HDF5 structure and adds masks:

* ``mask/all_rollouts``
* ``mask/all``
* ``mask/success``
* ``mask/failure``

The per-shard logs are streamed in real time, which makes native MuJoCo /
robosuite crashes visible while collection is running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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


def valid_shard(path: Path, expected: int, require_obs: bool) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as dataset:
            if "data" not in dataset or len(dataset["data"]) != expected:
                return False
            keys = sorted_demo_keys(dataset)
            if len(keys) == 0:
                return False
            required = ("actions", "states", "rewards", "dones")
            if any(name not in dataset[f"data/{keys[0]}"] for name in required):
                return False
            if require_obs and "obs" not in dataset[f"data/{keys[0]}"]:
                return False
            return True
    except OSError:
        return False


def parse_stats(text: str) -> dict | None:
    marker = "Average Rollout Stats"
    index = text.rfind(marker)
    if index < 0:
        return None
    match = re.search(r"\{.*?\}", text[index + len(marker) :], re.S)
    return json.loads(match.group(0)) if match else None


def build_shard_specs(args) -> list[dict]:
    if args.policy_seeds is None:
        return [
            {
                "shard_index": shard_index,
                "seed": args.seed_base + shard_index,
                "env_seed": None,
                "policy_seed": None,
            }
            for shard_index in range(args.num_shards)
        ]

    specs = []
    for env_index in range(args.num_env_seeds):
        env_seed = args.seed_base + env_index
        for policy_seed in args.policy_seeds:
            specs.append(
                {
                    "shard_index": len(specs),
                    "seed": env_seed,
                    "env_seed": env_seed,
                    "policy_seed": int(policy_seed),
                    "env_index": env_index,
                }
            )
    return specs


def rollout_command(args, shard: Path, spec: dict, attempt: int) -> list[str]:
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
        attempt_seed = int(spec["seed"]) + (attempt - 1) * args.retry_seed_offset
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
    specs = build_shard_specs(args)
    for spec in specs:
        shard_index = int(spec["shard_index"])
        seed = int(spec["seed"])
        shard = args.output_dir / f"shard_{shard_index:03d}.hdf5"
        log = args.output_dir / f"shard_{shard_index:03d}.log"
        base_record = {
            "seed": seed,
            "nominal_seed": seed,
            "env_seed": spec.get("env_seed"),
            "policy_seed": spec.get("policy_seed"),
            "env_index": spec.get("env_index"),
        }
        if (
            not args.force_shards
            and valid_shard(shard, args.rollouts_per_shard, args.dataset_obs)
        ):
            print(f"[reuse] {shard}", flush=True)
            stats = parse_stats(log.read_text(errors="replace")) if log.exists() else None
            shard_records.append(
                {
                    "shard": str(shard),
                    "log": str(log),
                    **base_record,
                    "attempts": 0,
                    "stats": stats,
                }
            )
            continue

        if shard.exists():
            shard.unlink()
        logger = TeeLogger(log, mode="w")
        try:
            for attempt in range(1, args.max_retries + 1):
                command = rollout_command(args, shard, spec, attempt)
                if spec.get("policy_seed") is None:
                    attempt_seed = seed + (attempt - 1) * args.retry_seed_offset
                    seed_desc = f"seed={seed} attempt_seed={attempt_seed}"
                    record_seed = attempt_seed
                else:
                    seed_desc = (
                        f"env_seed={spec['env_seed']} "
                        f"policy_seed={spec['policy_seed']}"
                    )
                    record_seed = seed
                logger.line(
                    f"\n[collect shard={shard_index} {seed_desc} "
                    f"attempt={attempt}/{args.max_retries}]"
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
                )
                if ok:
                    stats = parse_stats(stdout)
                    logger.line(f"[complete] {shard} stats={json.dumps(stats)}")
                    shard_records.append(
                        {
                            "shard": str(shard),
                            "log": str(log),
                            **base_record,
                            "seed": record_seed,
                            "attempts": attempt,
                            "stats": stats,
                        }
                    )
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
    return shard_records


def merge(args, shard_records: list[dict]) -> dict:
    merged_path = args.output_dir / args.merged_name
    if merged_path.exists():
        if not args.force_merge:
            raise FileExistsError(
                f"{merged_path} exists. Pass --force-merge to overwrite it."
            )
        merged_path.unlink()

    successes, failures, all_keys = [], [], []
    episode_records = []
    total_samples = 0
    env_args = None
    with h5py.File(merged_path, "w") as output:
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
                    if record.get("env_seed") is not None:
                        group.attrs["env_seed"] = int(record["env_seed"])
                    if record.get("policy_seed") is not None:
                        group.attrs["policy_seed"] = int(record["policy_seed"])
                    if record.get("env_index") is not None:
                        group.attrs["env_index"] = int(record["env_index"])

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
                            "env_seed": record.get("env_seed"),
                            "policy_seed": record.get("policy_seed"),
                            "env_index": record.get("env_index"),
                        }
                    )
                    output_index += 1

        output_data.attrs["env_args"] = env_args
        output_data.attrs["total"] = total_samples
        mask = output.create_group("mask")
        encoded_all = np.asarray(all_keys, dtype="S")
        mask["all_rollouts"] = encoded_all
        mask["all"] = encoded_all
        mask["success"] = np.asarray(successes, dtype="S")
        mask["failure"] = np.asarray(failures, dtype="S")

    horizons = np.asarray([ep["horizon"] for ep in episode_records], dtype=np.float64)
    summary = {
        "agent": str(args.agent),
        "merged_dataset": str(merged_path),
        "dataset_obs": bool(args.dataset_obs),
        "success_return_threshold": args.success_return_threshold,
        "num_rollouts": len(all_keys),
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": len(successes) / max(1, len(all_keys)),
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
    parser.add_argument("--agent", type=Path, required=True)
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
    args = parser.parse_args()
    args.agent = args.agent.resolve()
    args.output_dir = args.output_dir.resolve()
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
