#!/usr/bin/env python3
"""Resume robomimic training until a requested checkpoint is safely written.

This wrapper is intentionally process-based: a failed interpreter process is
discarded, while robomimic's per-epoch ``last.pth`` checkpoint allows the next
fresh process to continue. It also stops a longer configured run immediately
after the requested milestone checkpoint appears.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


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
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(COMMON_ENV)

    if checkpoint.exists():
        print(f"target already exists: {checkpoint}")
        return

    for attempt in range(1, args.max_attempts + 1):
        log_path = log_dir / f"attempt_{attempt:03d}.log"
        command = [
            str(PYTHON),
            "-m",
            "robomimic.scripts.train",
            "--config",
            str(config),
            "--resume",
        ]
        print(f"[attempt {attempt}] starting; log={log_path}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                if checkpoint.exists():
                    print(f"target reached: {checkpoint}", flush=True)
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=30)
                    return
                time.sleep(args.poll_seconds)

        if checkpoint.exists():
            print(f"target reached: {checkpoint}", flush=True)
            return
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-8:])
        print(f"[attempt {attempt}] exited before target\n{tail}", flush=True)
        time.sleep(1.0)

    raise RuntimeError(
        f"target checkpoint was not produced after {args.max_attempts} attempts: "
        f"{checkpoint}"
    )


if __name__ == "__main__":
    main()
