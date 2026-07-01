#!/usr/bin/env python3
"""Validate epoch-50 rollout stability and visualize logged outcomes.

The stability test launches each seed in a fresh process, which avoids sharing
simulator or policy state across repetitions. The visualization replays logged
simulator states, then applies the final logged action so that the terminal
success transition is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    os.environ.get(
        "ROBOMIMIC_PYTHON",
        "/home/ryan/miniconda3/envs/robomimic_stable/bin/python",
    )
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "trained_models/lift_ph_lowdim_bc_full_20260626_session"
    / "20260626214440/models/model_epoch_50.pth"
)
DEFAULT_DATASET = ROOT / "rollouts/lift_bc_epoch50_rollouts_500_lowdim.hdf5"
DEFAULT_OUTPUT = ROOT / "rollouts/epoch50_platform_validation"

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
    # Redirect lookups to an isolated cache tree.
    "PYTHONPYCACHEPREFIX": "/tmp/robomimic_pycache",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


def process_env(cache_suffix: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    if cache_suffix is not None:
        env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_pycache_{cache_suffix}"
    return env


class TeeLogger:
    """Write evaluator progress to both terminal and a per-seed log file."""

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


def parse_rollout_stats(text: str) -> dict[str, float]:
    marker = "Average Rollout Stats"
    idx = text.rfind(marker)
    if idx < 0:
        raise RuntimeError("run_trained_agent output has no Average Rollout Stats")
    match = re.search(r"\{.*?\}", text[idx + len(marker) :], re.S)
    if match is None:
        raise RuntimeError("could not parse Average Rollout Stats JSON")
    return json.loads(match.group(0))


def infer_num_rollouts(stats: dict[str, float]) -> int | None:
    for key in ("Num_Rollouts", "Requested_Rollouts"):
        if key in stats:
            return int(round(float(stats[key])))
    success_rate = float(stats.get("Success_Rate", float("nan")))
    num_success = float(stats.get("Num_Success", float("nan")))
    if np.isfinite(success_rate) and success_rate > 0.0 and np.isfinite(num_success):
        return int(round(num_success / success_rate))
    return None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - radius, center + radius


def evaluate(
    checkpoint: Path,
    output_dir: Path,
    seeds: list[int],
    n_rollouts: int,
    horizon: int,
    max_retries: int,
    force: bool,
    chunk_size: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        log_path = output_dir / f"eval_seed_{seed}.log"
        stats = None
        if log_path.exists() and not force:
            try:
                stats = parse_rollout_stats(log_path.read_text())
                logged_rollouts = infer_num_rollouts(stats)
                if logged_rollouts != n_rollouts:
                    print(
                        f"\n[resume seed={seed}] ignoring {log_path}: "
                        f"logged_rollouts={logged_rollouts}, requested={n_rollouts}",
                        flush=True,
                    )
                    stats = None
                else:
                    print(f"\n[resume seed={seed}] using completed {log_path}", flush=True)
            except RuntimeError:
                pass
        if stats is None:
            stats = evaluate_seed(
                checkpoint=checkpoint,
                seed=seed,
                n_rollouts=n_rollouts,
                horizon=horizon,
                max_retries=max_retries,
                log_path=log_path,
                chunk_size=chunk_size,
            )
            if stats is None:
                raise RuntimeError(f"seed {seed} failed before completion")
        num_success = int(round(float(stats["Num_Success"])))
        run = {
            "seed": seed,
            "num_rollouts": n_rollouts,
            "num_success": num_success,
            "success_rate": num_success / n_rollouts,
            "mean_return": float(stats["Return"]),
            "mean_horizon": float(stats["Horizon"]),
            "log": str(log_path),
        }
        runs.append(run)
        print(
            f"seed={seed}: {num_success}/{n_rollouts} "
            f"({100.0 * run['success_rate']:.1f}%), "
            f"mean horizon={run['mean_horizon']:.2f}",
            flush=True,
        )

    rates = np.asarray([run["success_rate"] for run in runs], dtype=np.float64)
    total_success = sum(run["num_success"] for run in runs)
    total_rollouts = len(runs) * n_rollouts
    pooled_rate = total_success / total_rollouts
    ci_low, ci_high = wilson_interval(total_success, total_rollouts)
    observed_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    expected_std = math.sqrt(pooled_rate * (1.0 - pooled_rate) / n_rollouts)
    dispersion = (
        float(np.var(rates, ddof=1) / (expected_std * expected_std))
        if len(rates) > 1 and expected_std > 0.0
        else 0.0
    )

    summary = {
        "checkpoint": str(checkpoint),
        "horizon": horizon,
        "num_runs": len(runs),
        "rollouts_per_run": n_rollouts,
        "total_rollouts": total_rollouts,
        "total_success": total_success,
        "pooled_success_rate": pooled_rate,
        "wilson_95_interval": [ci_low, ci_high],
        "run_success_rate_mean": float(np.mean(rates)),
        "run_success_rate_std": observed_std,
        "run_success_rate_min": float(np.min(rates)),
        "run_success_rate_max": float(np.max(rates)),
        "expected_binomial_run_std": expected_std,
        "variance_to_binomial_ratio": dispersion,
        "runs": runs,
        "interpretation": (
            "A variance ratio near 1 means the run-to-run spread is consistent "
            "with finite-sample binomial noise; a substantially larger ratio "
            "suggests excess instability."
        ),
    }
    output_path = output_dir / "stability_summary.json"
    output_path.write_text(json.dumps(summary, indent=4))
    print(f"\n{json.dumps(summary, indent=4)}")
    print(f"\nWrote {output_path}")
    return summary


def aggregate_stats(chunk_stats: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total_rollouts = sum(count for count, _ in chunk_stats)
    total_success = sum(float(stats["Num_Success"]) for _, stats in chunk_stats)
    mean_return = sum(count * float(stats["Return"]) for count, stats in chunk_stats) / total_rollouts
    mean_horizon = sum(count * float(stats["Horizon"]) for count, stats in chunk_stats) / total_rollouts
    return {
        "Num_Rollouts": float(total_rollouts),
        "Return": mean_return,
        "Horizon": mean_horizon,
        "Success_Rate": total_success / total_rollouts,
        "Num_Success": total_success,
    }


def rollout_command(checkpoint: Path, n_rollouts: int, horizon: int, seed: int) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        "-m",
        "robomimic.scripts.run_trained_agent",
        "--agent",
        str(checkpoint),
        "--n_rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
    ]


def run_rollout_subprocess(
    *,
    checkpoint: Path,
    n_rollouts: int,
    horizon: int,
    seed: int,
    cache_suffix: str,
    max_retries: int,
    logger: TeeLogger | None = None,
) -> tuple[dict[str, float], str]:
    cmd = rollout_command(checkpoint, n_rollouts, horizon, seed)
    last_proc = None

    def emit(text: str) -> None:
        if logger is None:
            print(text, end="", flush=True)
        else:
            logger.write(text)

    for attempt in range(max_retries + 1):
        attempt_suffix = f"{cache_suffix}_attempt_{attempt + 1}"
        shutil.rmtree(
            f"/tmp/robomimic_pycache_{attempt_suffix}",
            ignore_errors=True,
        )
        emit(
            f"\n[evaluate subprocess seed={seed}, n={n_rollouts}, "
            f"attempt={attempt + 1}] {' '.join(cmd)}\n"
        )
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=process_env(cache_suffix=attempt_suffix),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output_parts = []
        assert proc.stdout is not None
        for line in proc.stdout:
            output_parts.append(line)
            emit(line)
        proc.wait()
        proc.stdout.close()
        proc.stdout = None
        last_proc = proc
        stdout = "".join(output_parts)
        if proc.returncode == 0:
            return parse_rollout_stats(stdout), stdout
        emit(
            f"subprocess seed={seed} n={n_rollouts} attempt={attempt + 1} "
            f"failed with returncode={proc.returncode}:\n"
            + "\n".join(stdout.splitlines()[-30:])
            + "\n"
        )
    raise subprocess.CalledProcessError(
        last_proc.returncode,
        cmd,
        output=stdout,
    )


def evaluate_seed(
    *,
    checkpoint: Path,
    seed: int,
    n_rollouts: int,
    horizon: int,
    max_retries: int,
    log_path: Path,
    chunk_size: int,
) -> dict[str, float]:
    logger = TeeLogger(log_path, mode="w")
    logger.line(
        f"[eval seed={seed}] checkpoint={checkpoint} "
        f"n_rollouts={n_rollouts} chunk_size={chunk_size} horizon={horizon}"
    )
    if chunk_size <= 0 or chunk_size >= n_rollouts:
        try:
            stats, _ = run_rollout_subprocess(
                checkpoint=checkpoint,
                n_rollouts=n_rollouts,
                horizon=horizon,
                seed=seed,
                cache_suffix=f"eval_seed_{seed}",
                max_retries=max_retries,
                logger=logger,
            )
            return stats
        finally:
            logger.close()

    remaining = n_rollouts
    chunk_index = 0
    chunk_stats = []
    try:
        while remaining > 0:
            count = min(chunk_size, remaining)
            # Use independent deterministic seeds per chunk. This preserves the
            # requested total number of rollouts without keeping one MuJoCo /
            # PyTorch process alive for all rollouts.
            chunk_seed = seed * 100000 + chunk_index
            completed_before = n_rollouts - remaining
            logger.line(
                f"\n===== chunk {chunk_index} seed={chunk_seed} "
                f"n_rollouts={count} progress={completed_before}/{n_rollouts} ====="
            )
            stats, _ = run_rollout_subprocess(
                checkpoint=checkpoint,
                n_rollouts=count,
                horizon=horizon,
                seed=chunk_seed,
                cache_suffix=f"eval_seed_{seed}_chunk_{chunk_index}",
                max_retries=max_retries,
                logger=logger,
            )
            chunk_stats.append((count, stats))
            remaining -= count
            chunk_index += 1
            partial = aggregate_stats(chunk_stats)
            logger.line(
                f"\n===== partial aggregated result after "
                f"{n_rollouts - remaining}/{n_rollouts} rollouts ====="
            )
            logger.line("Average Rollout Stats")
            logger.line(json.dumps(partial, indent=4))

        stats = aggregate_stats(chunk_stats)
        logger.line(
            "\n===== aggregated chunked result =====\n"
            "Average Rollout Stats\n"
            f"{json.dumps(stats, indent=4)}"
        )
        return stats
    finally:
        logger.close()


def decode_mask(dataset: h5py.File, key: str) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in dataset[f"mask/{key}"][:]
    ]


def render_samples(
    dataset_path: Path,
    output_dir: Path,
    samples_per_class: int,
    sample_seed: int,
    video_skip: int,
    width: int,
    height: int,
    camera: str,
) -> dict:
    # Imports are delayed so evaluation-only use does not initialize rendering.
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sample_seed)

    dummy_spec = {"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
    )

    manifest = {
        "dataset": str(dataset_path),
        "sample_seed": sample_seed,
        "samples_per_class": samples_per_class,
        "camera": camera,
        "video_skip": video_skip,
        "samples": [],
    }
    with h5py.File(dataset_path, "r") as dataset:
        for outcome in ("success", "failure"):
            candidates = decode_mask(dataset, outcome)
            count = min(samples_per_class, len(candidates))
            selected = rng.choice(candidates, size=count, replace=False).tolist()
            for demo_key in selected:
                group = dataset[f"data/{demo_key}"]
                states = group["states"][:]
                actions = group["actions"][:]
                rewards = group["rewards"][:]
                initial_state = {"states": states[0]}
                if "model_file" in group.attrs:
                    initial_state["model"] = group.attrs["model_file"]
                if "ep_meta" in group.attrs:
                    initial_state["ep_meta"] = group.attrs["ep_meta"]
                env.reset_to(initial_state)

                episode_return = float(np.sum(rewards))
                video_name = (
                    f"{outcome}_{demo_key}_return_{episode_return:.0f}"
                    f"_horizon_{len(actions)}.mp4"
                )
                video_path = video_dir / video_name
                writer = imageio.get_writer(video_path, fps=max(1, 20 // video_skip))
                try:
                    for step, state in enumerate(states):
                        env.reset_to({"states": state})
                        if step % video_skip == 0:
                            frame = env.render(
                                mode="rgb_array",
                                height=height,
                                width=width,
                                camera_name=camera,
                            )
                            writer.append_data(frame)

                    # Logged states are pre-action. Apply the final action once so
                    # the terminal transition, including successful lift, is shown.
                    env.reset_to({"states": states[-1]})
                    env.step(actions[-1])
                    frame = env.render(
                        mode="rgb_array",
                        height=height,
                        width=width,
                        camera_name=camera,
                    )
                    for _ in range(5):
                        writer.append_data(frame)
                    replay_success = bool(env.is_success()["task"])
                finally:
                    writer.close()

                record = {
                    "outcome_label": outcome,
                    "demo_key": demo_key,
                    "return": episode_return,
                    "horizon": int(len(actions)),
                    "terminal_reward": float(rewards[-1]),
                    "terminal_state_replay_success": replay_success,
                    "video": str(video_path),
                }
                manifest["samples"].append(record)
                print(
                    f"[rendered] {outcome} {demo_key}: "
                    f"return={episode_return:.0f}, horizon={len(actions)}, "
                    f"terminal replay success={replay_success} -> {video_path}",
                    flush=True,
                )

    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4))
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "Split each seed evaluation into fresh subprocess chunks. "
            "Use 0 to run all rollouts for a seed in one process."
        ),
    )
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun seeds even when a completed log already exists",
    )
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260627)
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--visualize-only", action="store_true")
    args = parser.parse_args()

    if args.evaluate_only and args.visualize_only:
        parser.error("--evaluate-only and --visualize-only are mutually exclusive")
    if not args.visualize_only:
        evaluate(
            checkpoint=args.checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            seeds=args.seeds,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            max_retries=args.max_retries,
            force=args.force,
            chunk_size=args.chunk_size,
        )
    if not args.evaluate_only:
        render_samples(
            dataset_path=args.dataset.resolve(),
            output_dir=args.output_dir.resolve(),
            samples_per_class=args.samples_per_class,
            sample_seed=args.sample_seed,
            video_skip=args.video_skip,
            width=args.width,
            height=args.height,
            camera=args.camera,
        )


if __name__ == "__main__":
    main()
