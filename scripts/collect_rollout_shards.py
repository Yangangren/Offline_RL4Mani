#!/usr/bin/env python3
"""Collect policy rollouts in restart-safe shards and merge them."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ryan/miniconda3/envs/robomimic_clean/bin/python")
COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


def valid_shard(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as dataset:
            return "data" in dataset and len(dataset["data"]) == expected
    except OSError:
        return False


def parse_stats(text: str) -> dict | None:
    marker = "Average Rollout Stats"
    index = text.rfind(marker)
    if index < 0:
        return None
    match = re.search(r"\{.*?\}", text[index + len(marker) :], re.S)
    return json.loads(match.group(0)) if match else None


def collect(args) -> list[dict]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(COMMON_ENV)
    shard_records = []
    for shard_index in range(args.num_shards):
        seed = args.seed_base + shard_index
        shard = args.output_dir / f"shard_{shard_index:03d}.hdf5"
        log = args.output_dir / f"shard_{shard_index:03d}.log"
        if valid_shard(shard, args.rollouts_per_shard):
            print(f"[reuse] {shard}", flush=True)
            stats = parse_stats(log.read_text(errors="replace")) if log.exists() else None
            shard_records.append({"shard": str(shard), "seed": seed, "stats": stats})
            continue
        if shard.exists():
            shard.unlink()

        command = [
            str(PYTHON),
            "-m",
            "robomimic.scripts.run_trained_agent",
            "--agent",
            str(args.agent),
            "--n_rollouts",
            str(args.rollouts_per_shard),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(seed),
            "--dataset_path",
            str(shard),
        ]
        for attempt in range(1, args.max_retries + 1):
            print(
                f"[collect shard={shard_index} seed={seed} attempt={attempt}]",
                flush=True,
            )
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.write_text(process.stdout)
            if process.returncode == 0 and valid_shard(shard, args.rollouts_per_shard):
                stats = parse_stats(process.stdout)
                print(f"[complete] {shard} stats={stats}", flush=True)
                shard_records.append(
                    {"shard": str(shard), "seed": seed, "stats": stats}
                )
                break
            if shard.exists():
                shard.unlink()
            print(
                "\n".join(process.stdout.splitlines()[-12:]),
                flush=True,
            )
        else:
            raise RuntimeError(f"failed to collect shard {shard_index}")
    return shard_records


def merge(args, shard_records: list[dict]) -> dict:
    merged_path = args.output_dir / args.merged_name
    if merged_path.exists():
        merged_path.unlink()
    successes, failures, all_keys = [], [], []
    total_samples = 0
    env_args = None
    with h5py.File(merged_path, "w") as output:
        output_data = output.create_group("data")
        output_index = 0
        for record in shard_records:
            with h5py.File(record["shard"], "r") as shard:
                if env_args is None:
                    env_args = shard["data"].attrs["env_args"]
                keys = sorted(
                    shard["data"].keys(),
                    key=lambda key: int(key.split("_")[-1]),
                )
                for key in keys:
                    output_key = f"demo_{output_index}"
                    shard.copy(shard[f"data/{key}"], output_data, name=output_key)
                    group = output_data[output_key]
                    episode_return = float(np.sum(group["rewards"][:]))
                    all_keys.append(output_key)
                    if episode_return > 0.0:
                        successes.append(output_key)
                    else:
                        failures.append(output_key)
                    total_samples += int(group.attrs["num_samples"])
                    output_index += 1
        output_data.attrs["env_args"] = env_args
        output_data.attrs["total"] = total_samples
        mask = output.create_group("mask")
        mask["all_rollouts"] = np.asarray(all_keys, dtype="S")
        mask["success"] = np.asarray(successes, dtype="S")
        mask["failure"] = np.asarray(failures, dtype="S")

    summary = {
        "agent": str(args.agent),
        "merged_dataset": str(merged_path),
        "num_rollouts": len(all_keys),
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": len(successes) / max(1, len(all_keys)),
        "total_samples": total_samples,
        "shards": shard_records,
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
    parser.add_argument("--rollouts-per-shard", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    args.agent = args.agent.resolve()
    args.output_dir = args.output_dir.resolve()
    records = collect(args)
    merge(args, records)


if __name__ == "__main__":
    main()
