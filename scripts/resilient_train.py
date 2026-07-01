#!/usr/bin/env python3
"""Resume robomimic training until a requested checkpoint is safely written.

This wrapper is intentionally process-based: a failed interpreter process is
discarded, while robomimic's per-epoch ``last.pth`` checkpoint allows the next
fresh process to continue. It also stops a longer configured run immediately
after the requested milestone checkpoint appears.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


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
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    # Reduce huge per-epoch checkpoint serialization for large RGB-DP models.
    "ROBOMIMIC_SAVE_LATEST_EVERY_N_EPOCHS": "10",
}


def experiment_root_from_config(config_path: Path) -> Path:
    config = json.loads(config_path.read_text())
    output_dir = Path(config["train"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return output_dir / config["experiment"]["name"]


def latest_run(experiment_root: Path) -> Path | None:
    if not experiment_root.exists():
        return None
    candidates = sorted(path for path in experiment_root.glob("*") if path.is_dir())
    return candidates[-1] if candidates else None


def resumable_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    for name in ("last.pth", "last_bak.pth"):
        checkpoint = run_dir / name
        if checkpoint.exists():
            return checkpoint
    return None


def archive_stale_experiment(experiment_root: Path) -> Path | None:
    if not experiment_root.exists():
        return None
    stamp = time.strftime("%Y%m%d%H%M%S")
    archive = experiment_root.with_name(f"{experiment_root.name}_stale_{stamp}")
    index = 1
    while archive.exists():
        archive = experiment_root.with_name(
            f"{experiment_root.name}_stale_{stamp}_{index:02d}"
        )
        index += 1
    experiment_root.rename(archive)
    print(
        f"[archive stale run] moved incomplete experiment to {archive}",
        flush=True,
    )
    return archive


def target_epoch_from_checkpoint(checkpoint: Path) -> int:
    stem = checkpoint.stem
    prefix = "model_epoch_"
    if not stem.startswith(prefix):
        raise ValueError(f"cannot infer target epoch from checkpoint path: {checkpoint}")
    return int(stem[len(prefix):])


def current_target_checkpoint(experiment_root: Path, target_epoch: int) -> Path | None:
    run_dir = latest_run(experiment_root)
    if run_dir is None:
        return None
    checkpoint = run_dir / f"models/model_epoch_{target_epoch}.pth"
    return checkpoint if checkpoint.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    target_epoch = target_epoch_from_checkpoint(checkpoint)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(COMMON_ENV)

    experiment_root = experiment_root_from_config(config)
    existing_target = current_target_checkpoint(experiment_root, target_epoch)
    if existing_target is not None:
        print(f"target already exists: {existing_target}")
        return

    for attempt in range(1, args.max_attempts + 1):
        run_dir = latest_run(experiment_root)
        resume_checkpoint = resumable_checkpoint(run_dir)
        if run_dir is not None and resume_checkpoint is None:
            archive_stale_experiment(experiment_root)
            run_dir = None

        log_path = log_dir / f"attempt_{attempt:03d}.log"
        attempt_env = env.copy()
        cache_dir = Path(
            f"/tmp/robomimic_resilient_train_pycache_{os.getpid()}_{attempt:03d}"
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        attempt_env["PYTHONPYCACHEPREFIX"] = str(cache_dir)
        command = [
            str(PYTHON),
            "-B",
            "-m",
            "robomimic.scripts.train",
            "--config",
            str(config),
        ]
        if run_dir is not None:
            command.append("--resume")
        print(f"[attempt {attempt}] starting; log={log_path}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=attempt_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                target = current_target_checkpoint(experiment_root, target_epoch)
                if target is not None:
                    print(f"target reached: {target}", flush=True)
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=30)
                    return
                time.sleep(args.poll_seconds)

        target = current_target_checkpoint(experiment_root, target_epoch)
        if target is not None:
            print(f"target reached: {target}", flush=True)
            return
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-8:])
        print(f"[attempt {attempt}] exited before target\n{tail}", flush=True)
        time.sleep(1.0)

    raise RuntimeError(
        f"target checkpoint was not produced after {args.max_attempts} attempts: "
        f"model_epoch_{target_epoch}.pth under {experiment_root}"
    )


if __name__ == "__main__":
    main()
