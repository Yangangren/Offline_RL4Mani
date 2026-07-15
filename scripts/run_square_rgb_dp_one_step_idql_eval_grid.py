#!/usr/bin/env python3
"""Resilient evaluation grid for standard one-step IDQL.

Robosuite / MuJoCo image rollouts can occasionally terminate the Python
process through native-code errors. This wrapper keeps the scientific unit
simple: each (N, seed) pair is split into small subprocess chunks, each chunk is
logged and retried, and successful chunks are aggregated into a normal summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    os.environ.get(
        "ROBOMIMIC_PYTHON",
        "/home/ryan/miniconda3/envs/robomimic_stable/bin/python",
    )
)
DEFAULT_IDQL = (
    ROOT
    / "trained_models/square_rgb_dp_idql/default_reward_one_step_idql_paper_faithful"
    / "best_success_auc.pt"
)
DEFAULT_OUTPUT = ROOT / "rollouts/square_rgb_dp/one_step_idql_resilient_eval"

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


def process_env(cache_suffix: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_one_step_idql_eval_pycache_{cache_suffix}"
    return env


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


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - radius, center + radius


def aggregate_rollouts(rollouts: list[dict]) -> dict[str, float]:
    if not rollouts:
        return {
            "Num_Rollouts": 0,
            "Return": float("nan"),
            "Horizon": float("nan"),
            "Success_Rate": float("nan"),
            "Num_Success": 0.0,
        }
    returns = np.asarray([float(x["Return"]) for x in rollouts], dtype=np.float64)
    horizons = np.asarray([float(x["Horizon"]) for x in rollouts], dtype=np.float64)
    successes = np.asarray([float(x["Success_Rate"]) for x in rollouts], dtype=np.float64)
    result = {
        "Num_Rollouts": int(len(rollouts)),
        "Return": float(np.mean(returns)),
        "Horizon": float(np.mean(horizons)),
        "Success_Rate": float(np.mean(successes)),
        "Num_Success": float(np.sum(successes)),
    }
    base_keys = set(result.keys())
    extra_keys = sorted({key for item in rollouts for key in item.keys()} - base_keys)
    for key in extra_keys:
        values = []
        for item in rollouts:
            value = item.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            result[key] = float(np.mean(values))
    return result


def eval_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    n_rollouts: int,
    seed: int,
    num_candidates: int,
) -> list[str]:
    cmd = [
        str(PYTHON),
        "-B",
        "scripts/eval_square_rgb_dp_one_step_idql.py",
        "--idql-checkpoint",
        str(args.idql_checkpoint),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--actor-source",
        args.actor_source,
        "--critic-source",
        args.critic_source,
        "--n-rollouts",
        str(n_rollouts),
        "--horizon",
        str(args.horizon),
        "--seed",
        str(seed),
        "--num-candidates",
        str(num_candidates),
        "--candidate-batch-size",
        str(args.candidate_batch_size),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--selection",
        args.selection,
        "--softmax-temperature",
        str(args.softmax_temperature),
    ]
    cmd.append("--clip-actions" if args.clip_actions else "--no-clip-actions")
    cmd.append("--diffusion-clip-sample" if args.diffusion_clip_sample else "--no-diffusion-clip-sample")
    cmd.append(
        "--require-success-condition-adapter"
        if args.require_success_condition_adapter
        else "--no-require-success-condition-adapter"
    )
    return cmd


def run_chunk(
    *,
    args: argparse.Namespace,
    num_candidates: int,
    seed: int,
    chunk_index: int,
    chunk_seed: int,
    n_rollouts: int,
    logger: TeeLogger,
) -> dict:
    chunk_dir = args.output_dir / "chunks" / f"N{num_candidates}_seed{seed}_chunk{chunk_index:03d}"
    chunk_json = chunk_dir / f"one_step_idql_N{num_candidates}_seed{chunk_seed}.json"
    if chunk_json.exists() and not args.force:
        logger.line(f"[resume chunk] {chunk_json}")
        return json.loads(chunk_json.read_text())

    last_stdout = ""
    for attempt in range(1, args.max_retries + 1):
        cache_suffix = f"N{num_candidates}_s{seed}_c{chunk_index}_a{attempt}_{os.getpid()}"
        shutil.rmtree(f"/tmp/robomimic_one_step_idql_eval_pycache_{cache_suffix}", ignore_errors=True)
        command = eval_command(
            args=args,
            output_dir=chunk_dir,
            n_rollouts=n_rollouts,
            seed=chunk_seed,
            num_candidates=num_candidates,
        )
        logger.line(
            f"\n[chunk start] N={num_candidates} seed={seed} chunk={chunk_index} "
            f"chunk_seed={chunk_seed} rollouts={n_rollouts} attempt={attempt}/{args.max_retries}"
        )
        logger.line(" ".join(command))
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=process_env(cache_suffix),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output_parts: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            output_parts.append(line)
            logger.write(line)
        proc.wait()
        proc.stdout.close()
        proc.stdout = None
        last_stdout = "".join(output_parts)
        if proc.returncode == 0 and chunk_json.exists():
            logger.line(f"[chunk ok] {chunk_json}")
            return json.loads(chunk_json.read_text())
        partial_json = chunk_dir / f"one_step_idql_N{num_candidates}_seed{chunk_seed}_partial.json"
        if args.accept_partial and partial_json.exists():
            partial = json.loads(partial_json.read_text())
            completed = int(partial.get("completed_rollouts", 0))
            if completed > 0:
                logger.line(f"[chunk partial accepted] {partial_json} completed={completed}/{n_rollouts}")
                return partial
        logger.line(
            f"[chunk failed] returncode={proc.returncode}; expected={chunk_json}\n"
            + "\n".join(last_stdout.splitlines()[-80:])
        )
    raise RuntimeError(
        f"failed chunk N={num_candidates} seed={seed} chunk={chunk_index}; "
        f"last output tail:\n" + "\n".join(last_stdout.splitlines()[-80:])
    )


def run_pair(args: argparse.Namespace, num_candidates: int, seed: int) -> dict:
    final_json = args.output_dir / f"one_step_idql_N{num_candidates}_seed{seed}.json"
    if final_json.exists() and not args.force:
        try:
            existing = json.loads(final_json.read_text())
            stats = existing.get("average_rollout_stats", {})
            if int(stats.get("Num_Rollouts", -1)) >= args.n_rollouts:
                print(f"[resume pair] {final_json}", flush=True)
                return existing
        except Exception:
            pass

    log_path = args.output_dir / "logs" / f"one_step_idql_N{num_candidates}_seed{seed}.log"
    logger = TeeLogger(log_path, mode="w")
    logger.line(
        f"[pair] N={num_candidates} seed={seed} n_rollouts={args.n_rollouts} "
        f"rollouts_per_chunk={args.rollouts_per_chunk}"
    )
    all_rollouts: list[dict] = []
    chunk_records: list[dict] = []
    remaining = args.n_rollouts
    chunk_index = 0
    try:
        while remaining > 0:
            count = min(args.rollouts_per_chunk, remaining)
            # Independent chunk seeds make the evaluation resumable after a
            # native crash without replaying earlier rollouts in the same
            # long-lived process.
            chunk_seed = seed * 100000 + chunk_index
            chunk = run_chunk(
                args=args,
                num_candidates=num_candidates,
                seed=seed,
                chunk_index=chunk_index,
                chunk_seed=chunk_seed,
                n_rollouts=count,
                logger=logger,
            )
            rollouts = chunk.get("rollouts", [])
            if len(rollouts) == 0:
                raise RuntimeError(
                    f"chunk returned no rollouts: N={num_candidates}, seed={seed}, chunk={chunk_index}"
                )
            all_rollouts.extend(rollouts)
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_seed": chunk_seed,
                    "requested_rollouts": count,
                    "completed_rollouts": len(rollouts),
                    "json": str(
                        args.output_dir
                        / "chunks"
                        / f"N{num_candidates}_seed{seed}_chunk{chunk_index:03d}"
                        / f"one_step_idql_N{num_candidates}_seed{chunk_seed}.json"
                    ),
                    "average_rollout_stats": chunk.get("average_rollout_stats", {}),
                }
            )
            remaining -= len(rollouts)
            partial = aggregate_rollouts(all_rollouts)
            logger.line("[pair partial] " + json.dumps(partial, sort_keys=True))
            chunk_index += 1
    finally:
        logger.close()

    stats = aggregate_rollouts(all_rollouts)
    successes = int(round(float(stats["Num_Success"])))
    total = int(stats["Num_Rollouts"])
    ci_low, ci_high = wilson_interval(successes, total)
    result = {
        "idql_checkpoint": str(args.idql_checkpoint),
        "actor_source": args.actor_source,
        "critic_source": args.critic_source,
        "num_candidates": num_candidates,
        "num_inference_steps": args.num_inference_steps,
        "diffusion_clip_sample": bool(args.diffusion_clip_sample),
        "selection": args.selection,
        "seed": seed,
        "n_rollouts": args.n_rollouts,
        "completed_rollouts": total,
        "horizon": args.horizon,
        "average_rollout_stats": stats,
        "wilson_95_interval": [ci_low, ci_high],
        "log": str(log_path),
        "chunks": chunk_records,
        "rollouts": all_rollouts,
    }
    final_json.write_text(json.dumps(result, indent=2))
    print(f"[pair wrote] {final_json}", flush=True)
    return result


def summarize(results: list[dict], args: argparse.Namespace) -> dict:
    by_n = []
    for n in args.num_candidates:
        subset = [r for r in results if int(r["num_candidates"]) == int(n)]
        rollouts = []
        seed_runs = []
        for r in subset:
            rollouts.extend(r.get("rollouts", []))
            stats = r["average_rollout_stats"]
            seed_runs.append(
                {
                    "seed": r["seed"],
                    "success_rate": float(stats["Success_Rate"]),
                    "num_success": float(stats["Num_Success"]),
                    "num_rollouts": int(stats["Num_Rollouts"]),
                    "mean_return": float(stats["Return"]),
                    "mean_horizon": float(stats["Horizon"]),
                    "q_selected_mean": stats.get("Q_Selected_Mean"),
                    "q_margin_mean": stats.get("Q_Margin_Mean"),
                    "q_range_mean": stats.get("Q_Range_Mean"),
                    "selected_index_mean": stats.get("Selected_Index_Mean"),
                    "selected_index_first_fraction": stats.get("Selected_Index_First_Fraction"),
                    "json": str(args.output_dir / f"one_step_idql_N{n}_seed{r['seed']}.json"),
                    "log": r.get("log", ""),
                }
            )
        stats = aggregate_rollouts(rollouts)
        successes = int(round(float(stats["Num_Success"])))
        total = int(stats["Num_Rollouts"])
        rates = np.asarray([x["success_rate"] for x in seed_runs], dtype=np.float64)
        if total == 0:
            by_n.append(
                {
                    "num_candidates": n,
                    "status": "pending",
                    "total_rollouts": 0,
                    "total_success": 0,
                    "success_rate": None,
                    "wilson_95_interval": None,
                    "seed_success_rate_mean": None,
                    "seed_success_rate_std": None,
                    "seeds": seed_runs,
                }
            )
            continue
        ci_low, ci_high = wilson_interval(successes, total)
        by_n.append(
            {
                "num_candidates": n,
                "status": "complete" if len(seed_runs) == len(args.seeds) else "partial",
                "total_rollouts": total,
                "total_success": successes,
                "success_rate": float(stats["Success_Rate"]),
                "wilson_95_interval": [ci_low, ci_high],
                "mean_return": float(stats["Return"]),
                "mean_horizon": float(stats["Horizon"]),
                "q_selected_mean": stats.get("Q_Selected_Mean"),
                "q_margin_mean": stats.get("Q_Margin_Mean"),
                "q_range_mean": stats.get("Q_Range_Mean"),
                "selected_index_mean": stats.get("Selected_Index_Mean"),
                "selected_index_first_fraction": stats.get("Selected_Index_First_Fraction"),
                "seed_success_rate_mean": float(np.mean(rates)) if len(rates) else float("nan"),
                "seed_success_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                "seeds": seed_runs,
            }
        )
    summary = {
        "idql_checkpoint": str(args.idql_checkpoint),
        "actor_source": args.actor_source,
        "critic_source": args.critic_source,
        "num_inference_steps": args.num_inference_steps,
        "diffusion_clip_sample": bool(args.diffusion_clip_sample),
        "horizon": args.horizon,
        "n_rollouts_per_seed": args.n_rollouts,
        "rollouts_per_chunk": args.rollouts_per_chunk,
        "candidate_batch_size": args.candidate_batch_size,
        "selection": args.selection,
        "num_candidates": args.num_candidates,
        "seeds": args.seeds,
        "by_num_candidates": by_n,
    }
    path = args.output_dir / "one_step_idql_eval_grid_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idql-checkpoint", type=Path, default=DEFAULT_IDQL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-candidates", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--accept-partial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--actor-source",
        choices=(
            "idql_target_one_step_mlp",
            "idql_one_step_mlp",
            "pretrained_dp_first_action",
            "hybrid_dp_chunk_actor",
        ),
        default="idql_target_one_step_mlp",
    )
    parser.add_argument("--critic-source", choices=("target", "online"), default="target")
    parser.add_argument("--selection", choices=("argmax", "greedy", "softmax", "advantage_softmax"), default="argmax")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-clip-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.idql_checkpoint = args.idql_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rollouts_per_chunk <= 0:
        args.rollouts_per_chunk = args.n_rollouts

    results = []
    for n in args.num_candidates:
        for seed in args.seeds:
            results.append(run_pair(args, int(n), int(seed)))
            summarize(results, args)
    summarize(results, args)


if __name__ == "__main__":
    main()
