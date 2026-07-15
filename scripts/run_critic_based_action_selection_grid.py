#!/usr/bin/env python3
"""Resilient evaluation grid for risk-guided frozen-DP extraction."""

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
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_risk_eval_pycache_{cache_suffix}"
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
    return {
        "Num_Rollouts": int(len(rollouts)),
        "Return": float(np.mean(returns)),
        "Horizon": float(np.mean(horizons)),
        "Success_Rate": float(np.mean(successes)),
        "Num_Success": float(np.sum(successes)),
    }


def eval_command(
    *,
    policy: Path,
    risk: Path,
    output_dir: Path,
    n_rollouts: int,
    horizon: int,
    seed: int,
    num_candidates: int,
    candidate_batch_size: int,
    device: str,
    score_mode: str,
    selection: str,
    softmin_temperature: float,
    risk_threshold: float | None,
    score_gap_threshold: float,
    execute_horizon: int,
    action_start_index: int,
    max_prefix_len: int,
) -> list[str]:
    command = [
        str(PYTHON),
        "-B",
        "scripts/eval_square_rgb_dp_risk_extraction.py",
        "--policy",
        str(policy),
        "--risk",
        str(risk),
        "--output-dir",
        str(output_dir),
        "--n-rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--num-candidates",
        str(num_candidates),
        "--candidate-batch-size",
        str(candidate_batch_size),
        "--device",
        device,
        "--score-mode",
        score_mode,
        "--selection",
        selection,
        "--softmin-temperature",
        str(softmin_temperature),
        "--score-gap-threshold",
        str(score_gap_threshold),
        "--execute-horizon",
        str(execute_horizon),
        "--action-start-index",
        str(action_start_index),
        "--max-prefix-len",
        str(max_prefix_len),
    ]
    if risk_threshold is not None:
        command.extend(["--risk-threshold", str(risk_threshold)])
    return command


def result_json_path(
    output_dir: Path,
    score_mode: str,
    selection: str,
    num_candidates: int,
    seed: int,
    score_gap_threshold: float = 0.0,
) -> Path:
    gap_suffix = ""
    if float(score_gap_threshold) > 0.0:
        gap = f"{float(score_gap_threshold):.6g}".replace("-", "m").replace(".", "p")
        gap_suffix = f"_gap{gap}"
    return output_dir / (
        f"risk_eval_{score_mode}_{selection}_N{num_candidates}{gap_suffix}_seed{seed}.json"
    )


def run_chunk(
    *,
    args: argparse.Namespace,
    score_mode: str,
    selection: str,
    num_candidates: int,
    seed: int,
    chunk_index: int,
    chunk_seed: int,
    n_rollouts: int,
    logger: TeeLogger,
) -> dict:
    chunk_dir = (
        args.output_dir
        / "chunks"
        / (
            f"{score_mode}_{selection}_N{num_candidates}"
            f"{('_gap' + f'{float(args.score_gap_threshold):.6g}'.replace('-', 'm').replace('.', 'p')) if float(args.score_gap_threshold) > 0.0 else ''}"
            f"_seed{seed}_chunk{chunk_index:03d}"
        )
    )
    chunk_json = result_json_path(
        chunk_dir,
        score_mode,
        selection,
        num_candidates,
        chunk_seed,
        args.score_gap_threshold,
    )
    if chunk_json.exists() and not args.force:
        logger.line(f"[resume chunk] {chunk_json}")
        return json.loads(chunk_json.read_text())

    last_stdout = ""
    for attempt in range(1, args.max_retries + 1):
        # A native robosuite / NumPy crash can be trajectory-specific. Repeating
        # the exact same seed merely replays the same failing trajectory, so
        # retries use a deterministic alternate seed while preserving the
        # requested seed in the canonical chunk filename.
        attempt_seed = int(
            (chunk_seed + (attempt - 1) * 1_000_003) % (2**32 - 1)
        )
        attempt_json = result_json_path(
            chunk_dir,
            score_mode,
            selection,
            num_candidates,
            attempt_seed,
            args.score_gap_threshold,
        )
        cache_suffix = (
            f"{score_mode}_{selection}_N{num_candidates}_s{seed}_"
            f"gap{float(args.score_gap_threshold):.6g}_"
            f"c{chunk_index}_a{attempt}_{os.getpid()}"
        )
        shutil.rmtree(f"/tmp/robomimic_risk_eval_pycache_{cache_suffix}", ignore_errors=True)
        command = eval_command(
            policy=args.policy,
            risk=args.risk,
            output_dir=chunk_dir,
            n_rollouts=n_rollouts,
            horizon=args.horizon,
            seed=attempt_seed,
            num_candidates=num_candidates,
            candidate_batch_size=args.candidate_batch_size,
            device=args.device,
            score_mode=score_mode,
            selection=selection,
            softmin_temperature=args.softmin_temperature,
            risk_threshold=args.risk_threshold,
            score_gap_threshold=args.score_gap_threshold,
            execute_horizon=args.execute_horizon,
            action_start_index=args.action_start_index,
            max_prefix_len=args.max_prefix_len,
        )
        logger.line(
            f"\n[chunk start] score={score_mode} selection={selection} "
            f"N={num_candidates} seed={seed} chunk={chunk_index} "
            f"requested_chunk_seed={chunk_seed} attempt_seed={attempt_seed} "
            f"rollouts={n_rollouts} attempt={attempt}/{args.max_retries}"
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
        if proc.returncode == 0 and attempt_json.exists():
            result = json.loads(attempt_json.read_text())
            result["requested_seed"] = int(chunk_seed)
            result["actual_seed"] = int(attempt_seed)
            chunk_json.write_text(json.dumps(result, indent=2))
            logger.line(
                f"[chunk ok] {chunk_json} actual_seed={attempt_seed}"
            )
            return result
        logger.line(
            f"[chunk failed] returncode={proc.returncode}; expected={attempt_json}\n"
            + "\n".join(last_stdout.splitlines()[-40:])
        )
    raise RuntimeError(
        f"failed chunk score={score_mode} selection={selection} "
        f"N={num_candidates} seed={seed} chunk={chunk_index}; "
        f"last output tail:\n" + "\n".join(last_stdout.splitlines()[-80:])
    )


def run_pair(
    args: argparse.Namespace,
    score_mode: str,
    selection: str,
    num_candidates: int,
    seed: int,
) -> dict:
    final_json = result_json_path(
        args.output_dir,
        score_mode,
        selection,
        num_candidates,
        seed,
        args.score_gap_threshold,
    )
    if final_json.exists() and not args.force:
        try:
            existing = json.loads(final_json.read_text())
            stats = existing.get("average_rollout_stats", {})
            if int(stats.get("Num_Rollouts", -1)) == args.n_rollouts:
                print(f"[resume pair] {final_json}", flush=True)
                return existing
        except Exception:
            pass

    log_path = (
        args.output_dir
        / "logs"
        / (
            f"risk_eval_{score_mode}_{selection}_N{num_candidates}"
            f"{('_gap' + f'{float(args.score_gap_threshold):.6g}'.replace('-', 'm').replace('.', 'p')) if float(args.score_gap_threshold) > 0.0 else ''}"
            f"_seed{seed}.log"
        )
    )
    logger = TeeLogger(log_path, mode="w")
    logger.line(
        f"[pair] score={score_mode} selection={selection} N={num_candidates} "
        f"seed={seed} n_rollouts={args.n_rollouts} "
        f"rollouts_per_chunk={args.rollouts_per_chunk}"
    )
    all_rollouts: list[dict] = []
    chunk_records: list[dict] = []
    resolved_action_start_index = None
    risk_feature_policy = None
    separate_risk_feature_encoder = None
    remaining = args.n_rollouts
    chunk_index = 0
    try:
        while remaining > 0:
            count = min(args.rollouts_per_chunk, remaining)
            chunk_seed = seed * 100000 + chunk_index
            chunk = run_chunk(
                args=args,
                score_mode=score_mode,
                selection=selection,
                num_candidates=num_candidates,
                seed=seed,
                chunk_index=chunk_index,
                chunk_seed=chunk_seed,
                n_rollouts=count,
                logger=logger,
            )
            rollouts = chunk.get("rollouts", [])
            resolved_action_start_index = chunk.get(
                "action_start_index", resolved_action_start_index
            )
            risk_feature_policy = chunk.get(
                "risk_feature_policy", risk_feature_policy
            )
            separate_risk_feature_encoder = chunk.get(
                "separate_risk_feature_encoder",
                separate_risk_feature_encoder,
            )
            actual_count = len(rollouts)
            if actual_count <= 0 or actual_count > remaining:
                raise RuntimeError(
                    f"chunk returned {actual_count} rollouts with {remaining} remaining: "
                    f"score={score_mode}, selection={selection}, N={num_candidates}, "
                    f"seed={seed}, chunk={chunk_index}"
                )
            if actual_count != count:
                logger.line(
                    f"[resume variable chunk] requested={count} existing={actual_count}"
                )
            all_rollouts.extend(rollouts)
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_seed": chunk_seed,
                    "num_rollouts": actual_count,
                    "actual_seed": int(
                        chunk.get("actual_seed", chunk.get("seed", chunk_seed))
                    ),
                    "json": str(
                        result_json_path(
                            args.output_dir
                            / "chunks"
                            / (
                                f"{score_mode}_{selection}_N{num_candidates}"
                                f"{('_gap' + f'{float(args.score_gap_threshold):.6g}'.replace('-', 'm').replace('.', 'p')) if float(args.score_gap_threshold) > 0.0 else ''}"
                                f"_seed{seed}_chunk{chunk_index:03d}"
                            ),
                            score_mode,
                            selection,
                            num_candidates,
                            chunk_seed,
                            args.score_gap_threshold,
                        )
                    ),
                    "average_rollout_stats": chunk.get("average_rollout_stats", {}),
                }
            )
            remaining -= actual_count
            partial = aggregate_rollouts(all_rollouts)
            logger.line("[pair partial] " + json.dumps(partial, sort_keys=True))
            chunk_index += 1
    finally:
        logger.close()

    stats = aggregate_rollouts(all_rollouts)
    successes = int(round(float(stats["Num_Success"])))
    ci_low, ci_high = wilson_interval(successes, int(stats["Num_Rollouts"]))
    result = {
        "policy": str(args.policy),
        "risk": str(args.risk),
        "risk_feature_policy": risk_feature_policy,
        "separate_risk_feature_encoder": separate_risk_feature_encoder,
        "score_mode": score_mode,
        "selection": selection,
        "num_candidates": num_candidates,
        "risk_threshold": args.risk_threshold,
        "score_gap_threshold": args.score_gap_threshold,
        "seed": seed,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "execute_horizon": args.execute_horizon,
        "requested_action_start_index": args.action_start_index,
        "action_start_index": resolved_action_start_index,
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
    by_config = []
    for score_mode in args.score_modes:
        for selection in args.selections:
            for n in args.num_candidates:
                subset = [
                    r
                    for r in results
                    if r["score_mode"] == score_mode
                    and r["selection"] == selection
                    and int(r["num_candidates"]) == int(n)
                ]
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
                            "json": str(
                                result_json_path(
                                    args.output_dir,
                                    score_mode,
                                    selection,
                                    int(n),
                                    int(r["seed"]),
                                )
                            ),
                            "log": r.get("log", ""),
                        }
                    )
                stats = aggregate_rollouts(rollouts)
                successes = int(round(float(stats["Num_Success"])))
                total = int(stats["Num_Rollouts"])
                rates = np.asarray([x["success_rate"] for x in seed_runs], dtype=np.float64)
                if total == 0:
                    by_config.append(
                        {
                            "score_mode": score_mode,
                            "selection": selection,
                            "num_candidates": n,
                            "status": "pending",
                            "total_rollouts": 0,
                            "total_success": 0,
                            "success_rate": None,
                            "wilson_95_interval": None,
                            "mean_return": None,
                            "mean_horizon": None,
                            "seed_success_rate_mean": None,
                            "seed_success_rate_std": None,
                            "seeds": seed_runs,
                        }
                    )
                    continue
                ci_low, ci_high = wilson_interval(successes, total)
                by_config.append(
                    {
                        "score_mode": score_mode,
                        "selection": selection,
                        "num_candidates": n,
                        "status": "complete"
                        if len(seed_runs) == len(args.seeds)
                        else "partial",
                        "total_rollouts": total,
                        "total_success": successes,
                        "success_rate": float(stats["Success_Rate"]),
                        "wilson_95_interval": [ci_low, ci_high],
                        "mean_return": float(stats["Return"]),
                        "mean_horizon": float(stats["Horizon"]),
                        "seed_success_rate_mean": float(np.mean(rates)) if len(rates) else float("nan"),
                        "seed_success_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                        "seeds": seed_runs,
                    }
                )

    completed = [x for x in by_config if x["status"] == "complete"]
    partial_or_complete = [
        x for x in by_config if x["success_rate"] is not None and x["total_rollouts"] > 0
    ]

    def ranking_key(item: dict):
        # Primary: success rate. Secondary: lower mean horizon among tied
        # success rates, then more rollouts. This is evaluation-based policy
        # selection, not loss-based checkpoint selection.
        return (
            float(item["success_rate"]),
            -float(item["mean_horizon"]),
            int(item["total_rollouts"]),
        )

    best_complete = max(completed, key=ranking_key) if completed else None
    best_available = (
        max(partial_or_complete, key=ranking_key) if partial_or_complete else None
    )
    resolved_action_start_indices = sorted(
        {
            int(r["action_start_index"])
            for r in results
            if r.get("action_start_index") is not None
            and int(r["action_start_index"]) >= 0
        }
    )
    summary = {
        "policy": str(args.policy),
        "risk": str(args.risk),
        "risk_feature_policies": sorted(
            {
                str(result["risk_feature_policy"])
                for result in results
                if result.get("risk_feature_policy") is not None
            }
        ),
        "uses_separate_risk_feature_encoder": any(
            bool(result.get("separate_risk_feature_encoder", False))
            for result in results
        ),
        "horizon": args.horizon,
        "n_rollouts_per_seed": args.n_rollouts,
        "rollouts_per_chunk": args.rollouts_per_chunk,
        "candidate_batch_size": args.candidate_batch_size,
        "execute_horizon": args.execute_horizon,
        "requested_action_start_index": args.action_start_index,
        "resolved_action_start_indices": resolved_action_start_indices,
        "risk_threshold": args.risk_threshold,
        "score_gap_threshold": args.score_gap_threshold,
        "num_candidates": args.num_candidates,
        "score_modes": args.score_modes,
        "selections": args.selections,
        "seeds": args.seeds,
        "by_config": by_config,
        "best_complete_policy": best_complete,
        "best_available_policy": best_available,
        "selection_note": (
            "Best policy is selected only from closed-loop rollout success, "
            "not from risk-model training loss."
        ),
    }
    path = args.output_dir / "risk_extraction_grid_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-candidates", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument(
        "--score-modes",
        nargs="+",
        choices=(
            "positive_action_risk",
            "positive_action_advantage",
            "action_delta_logodds",
            "action_advantage_logodds",
            "action_logit",
            "action_probability",
        ),
        default=["positive_action_risk"],
    )
    parser.add_argument(
        "--selections",
        nargs="+",
        choices=("argmin", "argmax", "greedy", "softmin", "softmax", "threshold_fallback"),
        default=["argmin"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--rollouts-per-chunk", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--execute-horizon", type=int, default=8)
    parser.add_argument("--action-start-index", type=int, default=-1)
    parser.add_argument("--max-prefix-len", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--softmin-temperature", type=float, default=1.0)
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--score-gap-threshold", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.policy = args.policy.resolve()
    args.risk = args.risk.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rollouts_per_chunk <= 0:
        args.rollouts_per_chunk = args.n_rollouts

    results = []
    for score_mode in args.score_modes:
        for selection in args.selections:
            for n in args.num_candidates:
                for seed in args.seeds:
                    results.append(
                        run_pair(args, score_mode, selection, int(n), int(seed))
                    )
                    summarize(results, args)
    summarize(results, args)


if __name__ == "__main__":
    main()
