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
import queue
import shutil
import subprocess
import threading
import time
import traceback
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
DEFAULT_DP = (
    ROOT
    / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1/20260629231002/last.pth"
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


def process_env(cache_suffix: str, gpu_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_one_step_idql_eval_pycache_{cache_suffix}"
    if gpu_id is not None:
        # The evaluator always uses logical cuda:0. Restricting each child to
        # one physical GPU makes cuda:0 map to that worker's device. MuJoCo EGL
        # uses the physical device index, so bind it explicitly as well.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    return env


def resolve_gpu_ids(args: argparse.Namespace) -> list[int | None]:
    """Resolve the physical GPU assigned to each evaluation worker."""
    requested = args.num_gpus
    explicit = args.gpu_ids

    if args.device != "cuda":
        if explicit or (requested is not None and requested != 1):
            raise ValueError("multi-GPU evaluation requires --device cuda")
        return [None]

    if requested is None:
        requested = len(explicit) if explicit else 1
    if requested <= 0:
        raise ValueError("--num-gpus must be positive")

    if explicit:
        if any(gpu_id < 0 for gpu_id in explicit):
            raise ValueError("--gpu-ids must contain non-negative physical GPU IDs")
        if len(set(explicit)) != len(explicit):
            raise ValueError("--gpu-ids must not contain duplicates")
        if requested > len(explicit):
            raise ValueError(
                f"--num-gpus={requested} exceeds the {len(explicit)} entries in --gpu-ids"
            )
        return [int(gpu_id) for gpu_id in explicit[:requested]]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if not tokens or any(not token.isdigit() for token in tokens):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must contain numeric GPU IDs for MuJoCo EGL; "
                "alternatively pass numeric physical IDs through --gpu-ids"
            )
        available = [int(token) for token in tokens]
        if len(set(available)) != len(available):
            raise ValueError("CUDA_VISIBLE_DEVICES must not contain duplicate GPU IDs")
        if requested > len(available):
            raise ValueError(
                f"--num-gpus={requested} exceeds the {len(available)} devices in "
                "CUDA_VISIBLE_DEVICES"
            )
        return available[:requested]

    return list(range(requested))


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


def load_json_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def completed_partial(path: Path) -> dict | None:
    partial = load_json_object(path)
    if partial is None:
        return None
    rollouts = partial.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        return None
    completed = int(partial.get("completed_rollouts", len(rollouts)))
    if completed != len(rollouts):
        return None
    return partial


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
        "scripts/eval_rgb_dp_idql.py",
        "--idql-checkpoint",
        str(args.idql_checkpoint),
        "--dp-checkpoint",
        str(args.dp_checkpoint),
        "--expected-task",
        args.expected_task,
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
        "--execution-horizon",
        str(args.execution_horizon),
        "--selection",
        args.selection,
        "--softmax-temperature",
        str(args.softmax_temperature),
        "--random-selection-probability",
        str(args.random_selection_probability),
    ]
    if not args.forbid_success_condition_adapter:
        cmd.extend(
            [
                "--inference-success-condition",
                str(args.inference_success_condition),
                "--inference-condition-mask",
                str(args.inference_condition_mask),
            ]
        )
    cmd.append("--clip-actions" if args.clip_actions else "--no-clip-actions")
    cmd.append("--diffusion-clip-sample" if args.diffusion_clip_sample else "--no-diffusion-clip-sample")
    cmd.append("--env-hard-reset" if args.env_hard_reset else "--no-env-hard-reset")
    cmd.append(
        "--reset-to-initial-state"
        if args.reset_to_initial_state
        else "--no-reset-to-initial-state"
    )
    cmd.append(
        "--require-success-condition-adapter"
        if args.require_success_condition_adapter
        else "--no-require-success-condition-adapter"
    )
    cmd.append(
        "--forbid-success-condition-adapter"
        if args.forbid_success_condition_adapter
        else "--no-forbid-success-condition-adapter"
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
    gpu_id: int | None = None,
) -> dict:
    chunk_dir = args.output_dir / "chunks" / f"N{num_candidates}_seed{seed}_chunk{chunk_index:03d}"
    chunk_json = chunk_dir / f"one_step_idql_N{num_candidates}_seed{chunk_seed}.json"
    if chunk_json.exists() and not args.force:
        completed = load_json_object(chunk_json)
        if completed is not None and completed.get("rollouts"):
            logger.line(f"[resume chunk] {chunk_json}")
            return completed
        logger.line(f"[ignore invalid chunk json] {chunk_json}")

    partial_json = chunk_dir / f"one_step_idql_N{num_candidates}_seed{chunk_seed}_partial.json"
    if args.accept_partial and partial_json.exists() and not args.force:
        partial = completed_partial(partial_json)
        if partial is not None:
            completed = len(partial["rollouts"])
            logger.line(
                f"[resume chunk partial] {partial_json} completed={completed}/{n_rollouts}"
            )
            return partial

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
            env=process_env(cache_suffix, gpu_id=gpu_id),
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
            completed = load_json_object(chunk_json)
            if completed is not None and completed.get("rollouts"):
                logger.line(f"[chunk ok] {chunk_json}")
                return completed
            logger.line(f"[ignore invalid chunk json] {chunk_json}")
        if args.accept_partial and partial_json.exists():
            partial = completed_partial(partial_json)
            if partial is not None:
                completed = len(partial["rollouts"])
                logger.line(
                    f"[chunk partial accepted] {partial_json} completed={completed}/{n_rollouts}"
                )
                return partial
        logger.line(
            f"[chunk failed] returncode={proc.returncode}; expected={chunk_json}\n"
            + "\n".join(last_stdout.splitlines()[-80:])
        )
    raise RuntimeError(
        f"failed chunk N={num_candidates} seed={seed} chunk={chunk_index}; "
        f"last output tail:\n" + "\n".join(last_stdout.splitlines()[-80:])
    )


def run_pair(
    args: argparse.Namespace,
    num_candidates: int,
    seed: int,
    gpu_id: int | None = None,
) -> dict:
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
        f"rollouts_per_chunk={args.rollouts_per_chunk} gpu_id={gpu_id}"
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
                gpu_id=gpu_id,
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
            if remaining < 0:
                raise RuntimeError(
                    f"resumed chunks exceed requested rollouts: completed={len(all_rollouts)} "
                    f"requested={args.n_rollouts}"
                )
            partial = aggregate_rollouts(all_rollouts)
            logger.line("[pair partial] " + json.dumps(partial, sort_keys=True))
            chunk_index += 1
            if args.inter_chunk_sleep > 0 and remaining > 0:
                logger.line(f"[inter-chunk sleep] {args.inter_chunk_sleep:.1f}s")
                time.sleep(args.inter_chunk_sleep)
    finally:
        logger.close()

    stats = aggregate_rollouts(all_rollouts)
    successes = int(round(float(stats["Num_Success"])))
    total = int(stats["Num_Rollouts"])
    ci_low, ci_high = wilson_interval(successes, total)
    result = {
        "idql_checkpoint": None if args.actor_source == "plain_dp" else str(args.idql_checkpoint),
        "dp_checkpoint": (
            str(args.dp_checkpoint)
            if args.actor_source in ("plain_dp", "external_dp_chunk_critic")
            else None
        ),
        "actor_source": args.actor_source,
        "expected_task": args.expected_task,
        "critic_source": None if args.actor_source == "plain_dp" else args.critic_source,
        "num_candidates": num_candidates,
        "num_inference_steps": args.num_inference_steps,
        "execution_horizon": args.execution_horizon,
        "diffusion_clip_sample": bool(args.diffusion_clip_sample),
        "success_condition_adapter_required": bool(args.require_success_condition_adapter),
        "success_condition_adapter_forbidden": bool(args.forbid_success_condition_adapter),
        "inference_success_condition": (
            None
            if args.forbid_success_condition_adapter
            else float(args.inference_success_condition)
        ),
        "inference_condition_mask": (
            None
            if args.forbid_success_condition_adapter
            else float(args.inference_condition_mask)
        ),
        "selection": None if args.actor_source == "plain_dp" else args.selection,
        "seed": seed,
        "n_rollouts": args.n_rollouts,
        "completed_rollouts": total,
        "horizon": args.horizon,
        "env_hard_reset": bool(args.env_hard_reset),
        "reset_to_initial_state": bool(args.reset_to_initial_state),
        "average_rollout_stats": stats,
        "wilson_95_interval": [ci_low, ci_high],
        "log": str(log_path),
        "chunks": chunk_records,
        "rollouts": all_rollouts,
    }
    atomic_write_json(final_json, result)
    print(f"[pair wrote] {final_json}", flush=True)
    return result


def run_grid(args: argparse.Namespace) -> list[dict]:
    pairs = [
        (int(num_candidates), int(seed))
        for num_candidates in args.num_candidates
        for seed in args.seeds
    ]
    pair_order = {pair: index for index, pair in enumerate(pairs)}
    worker_gpu_ids = args.eval_gpu_ids[: len(pairs)]

    if len(worker_gpu_ids) == 1:
        results = []
        gpu_id = worker_gpu_ids[0]
        for num_candidates, seed in pairs:
            results.append(run_pair(args, num_candidates, seed, gpu_id=gpu_id))
            summarize(results, args)
        return results

    print(
        f"[multi-gpu eval] pairs={len(pairs)} workers={len(worker_gpu_ids)} "
        f"gpu_ids={worker_gpu_ids}",
        flush=True,
    )
    tasks: queue.Queue[tuple[int, int]] = queue.Queue()
    completed: queue.Queue[tuple[str, int, int, int, object]] = queue.Queue()
    stop_event = threading.Event()
    for pair in pairs:
        tasks.put(pair)

    def gpu_worker(gpu_id: int) -> None:
        print(f"[gpu worker start] gpu_id={gpu_id}", flush=True)
        while not stop_event.is_set():
            try:
                num_candidates, seed = tasks.get_nowait()
            except queue.Empty:
                break
            if stop_event.is_set():
                tasks.task_done()
                break
            print(
                f"[gpu assignment] gpu_id={gpu_id} N={num_candidates} seed={seed}",
                flush=True,
            )
            try:
                result = run_pair(
                    args,
                    num_candidates,
                    seed,
                    gpu_id=gpu_id,
                )
            except Exception:
                stop_event.set()
                completed.put(
                    (
                        "error",
                        gpu_id,
                        num_candidates,
                        seed,
                        traceback.format_exc(),
                    )
                )
            else:
                completed.put(("result", gpu_id, num_candidates, seed, result))
            finally:
                tasks.task_done()
        print(f"[gpu worker stop] gpu_id={gpu_id}", flush=True)

    threads = [
        threading.Thread(
            target=gpu_worker,
            args=(gpu_id,),
            name=f"eval-gpu-{gpu_id}",
            daemon=True,
        )
        for gpu_id in worker_gpu_ids
    ]
    for thread in threads:
        thread.start()

    results: list[dict] = []
    failures: list[str] = []
    while any(thread.is_alive() for thread in threads) or not completed.empty():
        try:
            status, gpu_id, num_candidates, seed, payload = completed.get(timeout=0.2)
        except queue.Empty:
            continue
        if status == "result":
            assert isinstance(payload, dict)
            results.append(payload)
            results.sort(
                key=lambda result: pair_order[
                    (int(result["num_candidates"]), int(result["seed"]))
                ]
            )
            print(
                f"[gpu pair complete] gpu_id={gpu_id} N={num_candidates} seed={seed} "
                f"completed={len(results)}/{len(pairs)}",
                flush=True,
            )
            summarize(results, args)
        else:
            failure = (
                f"gpu_id={gpu_id} N={num_candidates} seed={seed} failed:\n{payload}"
            )
            failures.append(failure)
            print(f"[gpu pair failed] {failure}", flush=True)
        completed.task_done()

    for thread in threads:
        thread.join()

    if failures:
        if results:
            summarize(results, args)
        raise RuntimeError(
            "multi-GPU evaluation stopped after a worker failure:\n"
            + "\n".join(failures)
        )
    if len(results) != len(pairs):
        raise RuntimeError(
            f"multi-GPU evaluation completed {len(results)}/{len(pairs)} pairs"
        )
    return results


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
                    "random_selection_decision_fraction": stats.get(
                        "Random_Selection_Decision_Fraction"
                    ),
                    "non_greedy_selection_decision_fraction": stats.get(
                        "Non_Greedy_Selection_Decision_Fraction"
                    ),
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
                "random_selection_decision_fraction": stats.get(
                    "Random_Selection_Decision_Fraction"
                ),
                "non_greedy_selection_decision_fraction": stats.get(
                    "Non_Greedy_Selection_Decision_Fraction"
                ),
                "seed_success_rate_mean": float(np.mean(rates)) if len(rates) else float("nan"),
                "seed_success_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                "seeds": seed_runs,
            }
        )
    summary = {
        "idql_checkpoint": None if args.actor_source == "plain_dp" else str(args.idql_checkpoint),
        "dp_checkpoint": (
            str(args.dp_checkpoint)
            if args.actor_source in ("plain_dp", "external_dp_chunk_critic")
            else None
        ),
        "actor_source": args.actor_source,
        "expected_task": args.expected_task,
        "critic_source": None if args.actor_source == "plain_dp" else args.critic_source,
        "num_inference_steps": args.num_inference_steps,
        "execution_horizon": args.execution_horizon,
        "diffusion_clip_sample": bool(args.diffusion_clip_sample),
        "success_condition_adapter_required": bool(args.require_success_condition_adapter),
        "success_condition_adapter_forbidden": bool(args.forbid_success_condition_adapter),
        "inference_success_condition": (
            None
            if args.forbid_success_condition_adapter
            else float(args.inference_success_condition)
        ),
        "inference_condition_mask": (
            None
            if args.forbid_success_condition_adapter
            else float(args.inference_condition_mask)
        ),
        "horizon": args.horizon,
        "n_rollouts_per_seed": args.n_rollouts,
        "rollouts_per_chunk": args.rollouts_per_chunk,
        "inter_chunk_sleep": args.inter_chunk_sleep,
        "env_hard_reset": bool(args.env_hard_reset),
        "reset_to_initial_state": bool(args.reset_to_initial_state),
        "candidate_batch_size": None if args.actor_source == "plain_dp" else args.candidate_batch_size,
        "selection": None if args.actor_source == "plain_dp" else args.selection,
        "random_selection_probability": (
            None if args.actor_source == "plain_dp" else args.random_selection_probability
        ),
        "num_gpu_workers": (
            min(len(args.eval_gpu_ids), len(args.num_candidates) * len(args.seeds))
            if args.device == "cuda"
            else 0
        ),
        "gpu_ids": (
            [
                int(gpu_id)
                for gpu_id in args.eval_gpu_ids[
                    : len(args.num_candidates) * len(args.seeds)
                ]
            ]
            if args.device == "cuda"
            else []
        ),
        "num_candidates": args.num_candidates,
        "seeds": args.seeds,
        "by_num_candidates": by_n,
    }
    path = args.output_dir / "one_step_idql_eval_grid_summary.json"
    atomic_write_json(path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idql-checkpoint", type=Path, default=DEFAULT_IDQL)
    parser.add_argument("--dp-checkpoint", type=Path, default=DEFAULT_DP)
    parser.add_argument(
        "--expected-task",
        choices=("square", "can", "transport", "tool_hang"),
        default="square",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-candidates", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--inter-chunk-sleep", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--accept-partial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--env-hard-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reset-to-initial-state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help="For DP-proposal actors, execute this many selected trajectory actions before replanning.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Number of GPU workers. Defaults to one, or to all entries in "
            "--gpu-ids when that option is supplied."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Physical GPU IDs used by CUDA and MuJoCo EGL. By default, take "
            "the first --num-gpus entries from CUDA_VISIBLE_DEVICES, or 0..N-1."
        ),
    )
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
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--random-selection-probability", type=float, default=0.0)
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-clip-sample", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
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

    args.idql_checkpoint = args.idql_checkpoint.resolve()
    args.dp_checkpoint = args.dp_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        args.eval_gpu_ids = resolve_gpu_ids(args)
    except ValueError as error:
        parser.error(str(error))
    if args.actor_source == "plain_dp" and any(int(n) != 1 for n in args.num_candidates):
        parser.error("actor_source=plain_dp evaluates the standard DP queue; use --num-candidates 1")
    if len(set(args.num_candidates)) != len(args.num_candidates):
        parser.error("--num-candidates must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.rollouts_per_chunk <= 0:
        args.rollouts_per_chunk = args.n_rollouts

    results = run_grid(args)
    summarize(results, args)


if __name__ == "__main__":
    main()
