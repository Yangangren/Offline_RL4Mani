#!/usr/bin/env python3
"""Prepare, train, and evaluate a PH RGB Diffusion Policy baseline.

This started as the Square PH RGB-DP sanity baseline and is now task-aware for
the common robomimic sim PH tasks. By default it
follows the robomimic Diffusion Policy template as closely as possible, while
changing only the dataset and task-specific observation keys:

* RGB cameras from the official image dataset for each task;
* proprioception only, no object-state policy input;
* frame_stack=2, seq_length=16;
* DP horizon: obs=2, action=8, prediction=16;
* process-level resilient training and multi-seed evaluation.

Use ``--recipe fast`` only for quick debugging. The default ``official`` recipe
keeps the default large UNet, DDPM sampler, batch size, epoch length, and long
cosine schedule.

The script does not download datasets during prepare, because that is a network
operation. If the image dataset is missing, it prints the exact command.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    os.environ.get(
        "ROBOMIMIC_PYTHON",
        "/home/ryan/miniconda3/envs/robomimic_stable/bin/python",
    )
)
TEMPLATE = ROOT / "robomimic/exps/templates/diffusion_policy.json"
CONFIG_DIR = ROOT / "robomimic/exps/templates/square_rgb_dp"
MODEL_ROOT = ROOT / "trained_models/square_rgb_dp"
ROLLOUT_ROOT = ROOT / "rollouts/square_rgb_dp"
CACHE_TAG = "square_rgb_dp"

SINGLE_ARM_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
TRANSPORT_PROPRIO_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
]

TASK_ALIASES = {
    "can": "can",
    "square": "square",
    "transport": "transport",
    "tool_hang": "tool_hang",
    "toolhang": "tool_hang",
    "tool-hang": "tool_hang",
}

TASK_DEFAULTS = {
    "can": {
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "rgb_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim_keys": SINGLE_ARM_PROPRIO_KEYS,
        "camera_size": 84,
        "crop_size": 76,
        "horizon": 400,
    },
    "square": {
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "rgb_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim_keys": SINGLE_ARM_PROPRIO_KEYS,
        "camera_size": 84,
        "crop_size": 76,
        "horizon": 400,
    },
    "transport": {
        "camera_names": [
            "shouldercamera0",
            "shouldercamera1",
            "robot0_eye_in_hand",
            "robot1_eye_in_hand",
        ],
        "rgb_keys": [
            "shouldercamera0_image",
            "robot0_eye_in_hand_image",
            "shouldercamera1_image",
            "robot1_eye_in_hand_image",
        ],
        "low_dim_keys": TRANSPORT_PROPRIO_KEYS,
        "camera_size": 84,
        "crop_size": 76,
        "horizon": 700,
    },
    "tool_hang": {
        "camera_names": ["sideview", "robot0_eye_in_hand"],
        "rgb_keys": ["sideview_image", "robot0_eye_in_hand_image"],
        "low_dim_keys": SINGLE_ARM_PROPRIO_KEYS,
        "camera_size": 240,
        "crop_size": 216,
        "horizon": 700,
    },
}

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "NUMBA_DISABLE_JIT": "1",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    # Reduce huge per-epoch checkpoint serialization for large RGB-DP models.
    "ROBOMIMIC_SAVE_LATEST_EVERY_N_EPOCHS": "10",
}


def normalize_task(task: str) -> str:
    key = task.lower().replace("-", "_")
    if key not in TASK_ALIASES:
        supported = ", ".join(sorted(TASK_DEFAULTS))
        raise ValueError(f"unsupported task '{task}'. Supported tasks: {supported}")
    return TASK_ALIASES[key]


def configure_task_paths(args) -> None:
    global CONFIG_DIR, MODEL_ROOT, ROLLOUT_ROOT, CACHE_TAG

    args.task = normalize_task(args.task)
    defaults = TASK_DEFAULTS[args.task]
    tag = f"{args.task}_rgb_dp"
    CACHE_TAG = tag
    CONFIG_DIR = ROOT / f"robomimic/exps/templates/{tag}"
    MODEL_ROOT = ROOT / f"trained_models/{tag}"
    ROLLOUT_ROOT = ROOT / f"rollouts/{tag}"

    if args.raw_dataset is None:
        args.raw_dataset = ROOT / f"datasets/{args.task}/{args.dataset_type}/demo_v15.hdf5"
    if args.dataset is None:
        args.dataset = ROOT / f"datasets/{args.task}/{args.dataset_type}/image_v15.hdf5"
    if args.camera_size is None:
        args.camera_size = int(defaults["camera_size"])
    if args.camera_names is None:
        args.camera_names = list(defaults["camera_names"])
    if args.rgb_keys is None:
        args.rgb_keys = [f"{camera}_image" for camera in args.camera_names]
        if args.camera_names == defaults["camera_names"]:
            args.rgb_keys = list(defaults["rgb_keys"])
    if args.low_dim_keys is None:
        args.low_dim_keys = list(defaults["low_dim_keys"])
    if args.crop_size is None:
        args.crop_size = int(defaults["crop_size"])
    if args.horizon is None:
        args.horizon = int(defaults["horizon"])


def process_env(cache_suffix: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/robomimic_{CACHE_TAG}_{cache_suffix}"
    return env


def image_encoder_config(crop_size: int) -> dict:
    return {
        "core_class": "VisualCore",
        "core_kwargs": {
            "feature_dimension": 64,
            "backbone_class": "ResNet18Conv",
            "backbone_kwargs": {
                "pretrained": False,
                "input_coord_conv": False,
            },
            "pool_class": "SpatialSoftmax",
            "pool_kwargs": {
                "num_kp": 32,
                "learnable_temperature": False,
                "temperature": 1.0,
                "noise_std": 0.0,
            },
        },
        "obs_randomizer_class": "CropRandomizer",
        "obs_randomizer_kwargs": {
            "crop_height": crop_size,
            "crop_width": crop_size,
            "num_crops": 1,
            "pos_enc": False,
        },
    }


def task_title(args) -> str:
    return args.task.replace("_", " ").title()


def launcher_command(args, stages: str) -> str:
    script = Path(sys.argv[0]).as_posix()
    if not script.startswith("/") and not script.startswith("."):
        script = f"./{script}"
    return (
        f"cd {ROOT} && "
        f"{PYTHON} -B {script} --task {args.task} "
        f"--dataset-type {args.dataset_type} --stages {stages}"
    )


def dataset_download_message(path: Path, args) -> str:
    title = task_title(args)
    command = launcher_command(args, "dataset")
    return (
        f"{title} {args.dataset_type.upper()} RGB image dataset was not found:\n"
        f"  {path}\n\n"
        f"Build it with:\n"
        f"  {command}\n\n"
        f"This first downloads the raw {title} {args.dataset_type.upper()} dataset and then runs "
        f"dataset_states_to_obs.py with cameras {args.camera_names} at "
        f"{args.camera_size}x{args.camera_size}. "
        f"If you already built a different image filename, pass it explicitly "
        f"via --dataset."
    )


def require_dataset(path: Path, args) -> None:
    if path.exists():
        return
    raise FileNotFoundError(dataset_download_message(path, args))


def make_config(args) -> dict:
    config = json.loads(TEMPLATE.read_text())
    if args.recipe == "official":
        experiment_name = f"{args.task}_{args.dataset_type}_rgb_dp_official_s{args.seed}"
    else:
        experiment_name = f"{args.task}_{args.dataset_type}_rgb_dp_fast_s{args.seed}"
    config["experiment"]["name"] = experiment_name
    config["experiment"]["render"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["keep_all_videos"] = False
    config["experiment"]["validate"] = False
    # Internal rollouts are disabled by default because our robust workflow
    # evaluates checkpoints in a fresh process. This does not change the
    # training objective, but avoids simulator / h5py / torch state leaking
    # into a long training process.
    config["experiment"]["rollout"]["enabled"] = args.enable_train_rollouts
    config["experiment"]["rollout"]["horizon"] = int(args.horizon)
    config["experiment"]["save"]["every_n_epochs"] = args.save_every_epochs
    config["experiment"]["save"]["epochs"] = sorted(
        set(
            epoch
            for epoch in (*args.extra_save_epochs, args.target_epoch, args.epochs)
            if 0 < epoch <= args.epochs
        )
    )
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["ckpt_path"] = None

    config["train"]["data"] = [{"path": str(args.dataset.resolve())}]
    config["train"]["output_dir"] = str(MODEL_ROOT)
    config["train"]["normalize_weights_by_ds_size"] = False
    config["train"]["num_data_workers"] = 0
    # The PH image datasets are small enough to cache fully. This avoids repeated
    # per-batch HDF5 reads and greatly reduces exposure to intermittent h5py /
    # importlib crashes in this environment.
    config["train"]["hdf5_cache_mode"] = args.hdf5_cache_mode
    config["train"]["hdf5_load_next_obs"] = False
    config["train"]["seq_length"] = 16
    config["train"]["frame_stack"] = 2
    config["train"]["batch_size"] = args.batch_size
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed

    config["algo"]["horizon"] = {
        "observation_horizon": 2,
        "action_horizon": 8,
        "prediction_horizon": 16,
    }
    config["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] = args.lr

    if args.recipe == "fast":
        config["algo"]["unet"] = {
            "enabled": True,
            "diffusion_step_embed_dim": 128,
            "down_dims": [128, 256, 512],
            "kernel_size": 5,
            "n_groups": 8,
        }
        config["algo"]["ddpm"]["enabled"] = False
        config["algo"]["ddim"]["enabled"] = True
        config["algo"]["ddim"]["num_train_timesteps"] = 100
        config["algo"]["ddim"]["num_inference_timesteps"] = args.ddim_steps

    config["observation"]["modalities"]["obs"] = {
        "low_dim": list(args.low_dim_keys),
        "rgb": list(args.rgb_keys),
        "depth": [],
        "scan": [],
    }
    config["observation"]["modalities"]["goal"] = {
        "low_dim": [],
        "rgb": [],
        "depth": [],
        "scan": [],
    }
    config["observation"]["encoder"]["rgb"] = image_encoder_config(int(args.crop_size))
    return config


def write_config(args) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    path = CONFIG_DIR / f"{config['experiment']['name']}.json"
    path.write_text(json.dumps(config, indent=4))
    print(f"Wrote config: {path}", flush=True)
    return path


def build_dataset(args) -> None:
    if args.dataset.exists() and not args.force_dataset:
        print(f"Using existing image dataset: {args.dataset}", flush=True)
        return

    raw_dataset = args.raw_dataset.resolve()
    if not raw_dataset.exists():
        command = [
            str(PYTHON),
            "-B",
            "-m",
            "robomimic.scripts.download_datasets",
            "--tasks",
            args.task,
            "--dataset_types",
            args.dataset_type,
            "--hdf5_types",
            "raw",
        ]
        print("+ " + " ".join(command), flush=True)
        subprocess.run(
            command,
            cwd=ROOT,
            env=process_env(f"download_{args.task}_raw"),
            check=True,
        )

    raw_dataset = args.raw_dataset.resolve()
    if not raw_dataset.exists():
        raise FileNotFoundError(
            f"raw {task_title(args)} {args.dataset_type.upper()} dataset was not created: {raw_dataset}"
        )

    if args.dataset.exists() and args.force_dataset:
        args.dataset.unlink()

    command = [
        str(PYTHON),
        "-B",
        "robomimic/scripts/dataset_states_to_obs.py",
        "--done_mode",
        "2",
        "--dataset",
        str(raw_dataset),
        "--output_name",
        args.dataset.name,
        "--camera_names",
        *args.camera_names,
        "--camera_height",
        str(args.camera_size),
        "--camera_width",
        str(args.camera_size),
        "--compress",
        "--exclude-next-obs",
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=process_env(f"extract_{args.task}_rgb"),
        check=True,
    )
    require_dataset(args.dataset, args)
    print(f"Built image dataset: {args.dataset}", flush=True)


def experiment_root(config_path: Path) -> Path:
    config = json.loads(config_path.read_text())
    output_dir = Path(config["train"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return output_dir / config["experiment"]["name"]


def candidate_run_dirs(experiment: Path) -> list[Path]:
    if not experiment.exists():
        return []
    ignored = {"logs", "models", "videos", "tb"}
    return sorted(
        path
        for path in experiment.glob("*")
        if path.is_dir() and path.name not in ignored and (path / "models").is_dir()
    )


def latest_checkpoint(experiment: Path, epoch: int) -> Path | None:
    if not experiment.exists():
        return None

    direct_checkpoint = experiment / f"models/model_epoch_{epoch}.pth"
    if direct_checkpoint.exists():
        return direct_checkpoint

    for run in reversed(candidate_run_dirs(experiment)):
        checkpoint = run / f"models/model_epoch_{epoch}.pth"
        if checkpoint.exists():
            return checkpoint
    return None


def latest_checkpoint_from_fresh_runs(
    experiment: Path,
    epoch: int,
    started_at: float,
) -> Path | None:
    if not experiment.exists():
        return None

    direct_checkpoint = experiment / f"models/model_epoch_{epoch}.pth"
    if direct_checkpoint.exists() and direct_checkpoint.stat().st_mtime >= started_at - 2.0:
        return direct_checkpoint

    runs = sorted(
        path
        for path in candidate_run_dirs(experiment)
        if path.stat().st_mtime >= started_at - 2.0
    )
    for run in reversed(runs):
        checkpoint = run / f"models/model_epoch_{epoch}.pth"
        if checkpoint.exists():
            return checkpoint
    return None


def train_once(config_path: Path, args) -> Path:
    experiment = experiment_root(config_path)
    started_at = time.time()
    command = [
        str(PYTHON),
        "-B",
        "-m",
        "robomimic.scripts.train",
        "--config",
        str(config_path),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=process_env(f"train_once_{os.getpid()}"),
        check=True,
    )
    target = latest_checkpoint_from_fresh_runs(
        experiment,
        args.target_epoch,
        started_at,
    )
    if target is None:
        raise RuntimeError(
            f"fresh training finished but model_epoch_{args.target_epoch}.pth "
            f"was not found in a new run under {experiment}"
        )
    print(f"Training target: {target}", flush=True)
    return target


def resilient_train(config_path: Path, args) -> Path:
    experiment = experiment_root(config_path)
    target = latest_checkpoint(experiment, args.target_epoch)
    if target is not None and not args.force_train:
        print(f"Using existing checkpoint: {target}", flush=True)
        return target

    log_dir = ROLLOUT_ROOT / "baseline_train_restarts"
    log_dir.mkdir(parents=True, exist_ok=True)
    placeholder = experiment / "placeholder" / "models" / f"model_epoch_{args.target_epoch}.pth"
    command = [
        str(PYTHON),
        "-B",
        "scripts/resilient_train.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(placeholder),
        "--max-attempts",
        str(args.max_train_attempts),
        "--log-dir",
        str(log_dir),
    ]
    cache = Path(f"/tmp/robomimic_{args.task}_rgb_dp_train_launcher")
    shutil.rmtree(cache, ignore_errors=True)
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=process_env("train_launcher"),
        check=True,
    )
    target = latest_checkpoint(experiment, args.target_epoch)
    if target is None:
        raise RuntimeError(f"training finished but model_epoch_{args.target_epoch}.pth was not found")
    print(f"Training target: {target}", flush=True)
    return target


def evaluate(checkpoint: Path, args) -> Path:
    output_dir = ROLLOUT_ROOT / f"{args.recipe}_epoch{args.target_epoch}_eval"
    command = [
        str(PYTHON),
        "-B",
        "scripts/validate_epoch50_platform.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seeds",
        *[str(seed) for seed in args.eval_seeds],
        "--n-rollouts",
        str(args.eval_rollouts),
        "--chunk-size",
        str(args.eval_chunk_size),
        "--horizon",
        str(args.horizon),
        "--max-retries",
        str(args.max_eval_retries),
        "--evaluate-only",
    ]
    if args.force_eval:
        command.append("--force")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=process_env("eval_launcher"),
        check=True,
    )
    summary = output_dir / "stability_summary.json"
    if summary.exists():
        print(summary.read_text(), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="square",
        help="robomimic task name: can, square, transport, or tool_hang",
    )
    parser.add_argument("--dataset-type", default="ph", help="dataset type, e.g. ph or mh")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("dataset", "prepare", "train", "eval"),
        default=["prepare"],
        help="Stages to run. Example: --stages prepare train eval",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--raw-dataset", type=Path, default=None)
    parser.add_argument("--camera-size", type=int, default=None)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help="Override image extraction cameras. Defaults are task-specific.",
    )
    parser.add_argument(
        "--rgb-keys",
        nargs="+",
        default=None,
        help="Override config RGB observation keys. Defaults to <camera>_image.",
    )
    parser.add_argument(
        "--low-dim-keys",
        nargs="+",
        default=None,
        help="Override config proprio observation keys. Defaults are task-specific.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Override square crop size for the RGB randomizer. Defaults are task-specific.",
    )
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument(
        "--recipe",
        choices=("official", "fast"),
        default="official",
        help=(
            "official keeps the robomimic DP template capacity and schedule; "
            "fast uses the old compact UNet + DDIM debug recipe."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--target-epoch", type=int, default=2000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every-epochs", type=int, default=50)
    parser.add_argument(
        "--extra-save-epochs",
        type=int,
        nargs="*",
        default=[50, 100, 200, 500, 1000],
        help="Additional milestone checkpoints to save if they are reached.",
    )
    parser.add_argument("--enable-train-rollouts", action="store_true")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--hdf5-cache-mode", choices=("all", "low_dim"), default="all")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--eval-rollouts", type=int, default=100)
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=10,
        help=(
            "Number of rollouts per fresh evaluator subprocess. "
            "Set to 0 to run each seed in one process."
        ),
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-train-attempts", type=int, default=100)
    parser.add_argument("--max-eval-retries", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument(
        "--resilient-train",
        action="store_true",
        help=(
            "Use the old restart/resume wrapper. Default is a single fresh "
            "training process with no resume."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_task_paths(args)
    args.dataset = args.dataset.resolve()
    args.raw_dataset = args.raw_dataset.resolve()
    if "dataset" in args.stages:
        build_dataset(args)

    config_path = write_config(args)
    if not args.dataset.exists():
        print("\n" + dataset_download_message(args.dataset, args), flush=True)

    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if "train" in args.stages:
        require_dataset(args.dataset, args)
        checkpoint = (
            resilient_train(config_path, args)
            if args.resilient_train
            else train_once(config_path, args)
        )
    if "eval" in args.stages:
        if checkpoint is None:
            checkpoint = latest_checkpoint(experiment_root(config_path), args.target_epoch)
        if checkpoint is None:
            raise FileNotFoundError(
                "No checkpoint provided or found. Run with --stages prepare train eval "
                "or pass --checkpoint."
            )
        evaluate(checkpoint, args)


if __name__ == "__main__":
    main()
