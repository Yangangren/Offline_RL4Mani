#!/usr/bin/env python3
"""Build, validate, configure, and train the round-1 5 Hz pick-cup RGB-DP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot.build_pick_cup_5hz_dataset import (  # noqa: E402
    DEFAULT_5HZ_DATASET_DIR,
    BuildOptions,
    build_dataset,
    dataset_path,
    validate_5hz_dataset,
)
from scripts.real_robot.pick_cup_common import (  # noqa: E402
    CONVERSION_MANIFEST_ATTR,
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MAX_IMAGE_AGE_SEC,
    DEFAULT_SOURCE,
    LOW_DIM_KEYS,
    RGB_KEYS,
    atomic_write_json,
)


PYTHON = Path(
    os.environ.get(
        "ROBOMIMIC_PYTHON",
        "/home/ryan/miniconda3/envs/robomimic_stable/bin/python",
    )
)
TEMPLATE = ROOT / "robomimic/exps/templates/diffusion_policy.json"
DEFAULT_CONFIG_DIR = (
    ROOT / "robomimic/exps/templates/real_robot/pick_cup_5hz_round1"
)
DEFAULT_MODEL_ROOT = (
    ROOT / "trained_models/real_robot/pick_cup_rgb_dp_5hz_round1"
)

COMMON_ENV = {
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


@dataclass(frozen=True)
class TrainingOptions:
    """Configuration inputs for the independent round-1 5 Hz experiment."""

    dataset_dir: Path = DEFAULT_5HZ_DATASET_DIR
    config_dir: Path = DEFAULT_CONFIG_DIR
    model_root: Path = DEFAULT_MODEL_ROOT
    sampler: str = "ddim"
    ddim_steps: int = 10
    seed: int = 1
    epochs: int = 100
    steps_per_epoch: int = 67
    validation_steps: int = 7
    batch_size: int = 64
    num_data_workers: int = 2
    learning_rate: float = 1e-4
    crop_height: int = DEFAULT_CROP_HEIGHT
    crop_width: int = DEFAULT_CROP_WIDTH
    save_every_epochs: int = 10
    cuda: bool = True
    smoke: bool = False
    name_suffix: str = ""


def image_encoder_config(crop_height: int, crop_width: int) -> dict[str, Any]:
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
            "crop_height": crop_height,
            "crop_width": crop_width,
            "num_crops": 1,
            "pos_enc": False,
        },
    }


def _safe_suffix(value: str) -> str:
    if not value:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            "name_suffix may contain only letters, numbers, underscore, dot, and dash"
        )
    return f"_{value}"


def dataset_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint the exact conversion manifest and published file identity."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"5 Hz dataset does not exist: {path}")
    with h5py.File(path, "r") as dataset:
        raw_manifest = dataset.attrs.get(CONVERSION_MANIFEST_ATTR)
    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode("utf-8")
    if not isinstance(raw_manifest, str):
        raise ValueError(f"{path} is missing its conversion manifest")
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} has a malformed conversion manifest") from exc
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _dataset_image_shape(path: Path) -> tuple[int, int]:
    """Read the published image geometry instead of trusting CLI defaults."""

    with h5py.File(path, "r") as dataset:
        demos = sorted(dataset["data"].keys())
        if not demos:
            raise ValueError(f"5 Hz dataset contains no demonstrations: {path}")
        shape = tuple(dataset[f"data/{demos[0]}/obs/{RGB_KEYS[0]}"].shape)
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"unexpected RGB dataset shape in {path}: {shape}")
    return int(shape[1]), int(shape[2])


def make_config(
    options: TrainingOptions,
    *,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
) -> dict[str, Any]:
    """Create a 5 Hz config with To/Ta/Tp = 2/1/4."""

    if options.sampler not in ("ddim", "ddpm"):
        raise ValueError(f"unsupported sampler: {options.sampler}")
    if options.sampler == "ddim" and options.ddim_steps < 1:
        raise ValueError("ddim_steps must be positive")
    if not (0 < options.crop_height < image_height):
        raise ValueError(
            f"crop_height must be in (0, {image_height}), got {options.crop_height}"
        )
    if not (0 < options.crop_width < image_width):
        raise ValueError(
            f"crop_width must be in (0, {image_width}), got {options.crop_width}"
        )
    if min(
        options.epochs,
        options.steps_per_epoch,
        options.validation_steps,
        options.batch_size,
        options.save_every_epochs,
    ) <= 0:
        raise ValueError("epochs, steps, batch size, and save interval must be positive")
    if options.num_data_workers < 0:
        raise ValueError("num_data_workers must be non-negative")

    config = json.loads(TEMPLATE.read_text())
    if options.sampler == "ddim" and options.ddim_steps > int(
        config["algo"]["ddim"]["num_train_timesteps"]
    ):
        raise ValueError(
            "ddim_steps cannot exceed the DDIM training timestep count"
        )

    experiment_name = (
        f"pick_cup_rgb_dp_5hz_round1_{options.sampler}_s{options.seed}"
    )
    if options.smoke:
        experiment_name += "_smoke"
    experiment_name += _safe_suffix(options.name_suffix)

    epochs = 1 if options.smoke else options.epochs
    steps_per_epoch = 2 if options.smoke else options.steps_per_epoch
    validation_steps = 1 if options.smoke else options.validation_steps
    batch_size = 2 if options.smoke else options.batch_size
    workers = 0 if options.smoke else options.num_data_workers
    save_every = 1 if options.smoke else options.save_every_epochs

    config["experiment"]["name"] = experiment_name
    config["experiment"]["validate"] = True
    config["experiment"]["render"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["keep_all_videos"] = False
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["save"]["enabled"] = True
    config["experiment"]["save"]["every_n_epochs"] = save_every
    config["experiment"]["save"]["epochs"] = sorted(
        {
            epoch
            for epoch in (*range(save_every, epochs + 1, save_every), epochs)
            if epoch > 0
        }
    )
    config["experiment"]["save"]["on_best_validation"] = True
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["epoch_every_n_steps"] = steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = validation_steps
    config["experiment"]["ckpt_path"] = None

    shard = dataset_path(options.dataset_dir.expanduser().resolve())
    config["train"]["data"] = [
        {
            "path": str(shard),
            "weight": 1.0,
            "dataset_fingerprint": dataset_fingerprint(shard),
        }
    ]
    config["train"]["output_dir"] = str(options.model_root.expanduser().resolve())
    config["train"]["normalize_weights_by_ds_size"] = False
    config["train"]["num_data_workers"] = workers
    config["train"]["hdf5_cache_mode"] = "low_dim"
    config["train"]["hdf5_use_swmr"] = True
    config["train"]["hdf5_load_next_obs"] = False
    config["train"]["hdf5_normalize_obs"] = False
    config["train"]["hdf5_filter_key"] = "train"
    config["train"]["hdf5_validation_filter_key"] = "valid"
    config["train"]["seq_length"] = 4
    config["train"]["pad_seq_length"] = True
    config["train"]["frame_stack"] = 2
    config["train"]["pad_frame_stack"] = True
    config["train"]["dataset_keys"] = ["actions", "rewards", "dones"]
    config["train"]["action_keys"] = ["actions"]
    config["train"]["action_config"] = {
        "actions": {"normalization": None}
    }
    config["train"]["batch_size"] = batch_size
    config["train"]["num_epochs"] = epochs
    config["train"]["seed"] = options.seed
    config["train"]["cuda"] = options.cuda

    config["algo"]["horizon"] = {
        "observation_horizon": 2,
        "action_horizon": 1,
        "prediction_horizon": 4,
    }
    config["algo"]["optim_params"]["policy"]["learning_rate"][
        "initial"
    ] = options.learning_rate
    config["algo"]["unet"] = {
        "enabled": True,
        "diffusion_step_embed_dim": 256,
        "down_dims": [256, 512, 1024],
        "kernel_size": 5,
        "n_groups": 8,
    }
    config["algo"]["ddpm"]["enabled"] = options.sampler == "ddpm"
    config["algo"]["ddim"]["enabled"] = options.sampler == "ddim"
    config["algo"]["ddim"]["num_inference_timesteps"] = options.ddim_steps

    config["observation"]["modalities"]["obs"] = {
        "low_dim": list(LOW_DIM_KEYS),
        "rgb": list(RGB_KEYS),
        "depth": [],
        "scan": [],
    }
    config["observation"]["modalities"]["goal"] = {
        "low_dim": [],
        "rgb": [],
        "depth": [],
        "scan": [],
    }
    config["observation"]["encoder"]["rgb"] = image_encoder_config(
        options.crop_height,
        options.crop_width,
    )
    return config


def write_config(config: dict[str, Any], config_dir: Path) -> Path:
    path = config_dir.expanduser().resolve() / f"{config['experiment']['name']}.json"
    atomic_write_json(path, config)
    print(f"[pick_cup 5 Hz RGB-DP] wrote config: {path}", flush=True)
    return path


def _close_dataset(dataset: Any) -> None:
    if dataset is None:
        return
    children = getattr(dataset, "datasets", (dataset,))
    for child in children:
        close = getattr(child, "close_and_delete_hdf5_handle", None)
        if close is not None:
            close()


def standard_loader_preflight(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Exercise the stock robomimic train and validation data loaders."""

    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.train_utils as TrainUtils

    config = config_factory(config_dict["algo_name"])
    with config.values_unlocked():
        config.update(config_dict)
    config.lock()
    ObsUtils.initialize_obs_utils_with_config(config)

    if len(config.train.data) != 1:
        raise ValueError("the round-1 5 Hz experiment requires exactly one shard")
    signature = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=config.train.data[0],
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=False,
    )
    schema = {
        "ac_dim": int(signature["ac_dim"]),
        "all_shapes": {
            key: list(value) for key, value in signature["all_shapes"].items()
        },
    }

    trainset, validset = TrainUtils.load_data_for_training(
        config,
        obs_keys=list(config.all_obs_keys),
    )
    try:
        if validset is None:
            raise RuntimeError("standard loader did not create a validation dataset")
        if len(trainset) == 0 or len(validset) == 0:
            raise ValueError("train and validation datasets must both be non-empty")
        train_sample = trainset[0]
        valid_sample = validset[0]
        expected_steps = int(config.train.seq_length + config.train.frame_stack - 1)
        for label, sample in (("train", train_sample), ("valid", valid_sample)):
            actions = np.asarray(sample["actions"])
            if actions.shape != (expected_steps, 7) or not np.all(
                np.isfinite(actions)
            ):
                raise ValueError(
                    f"{label} loader action sample has invalid shape/content: "
                    f"{actions.shape}"
                )
            for key in RGB_KEYS:
                image = np.asarray(sample["obs"][key])
                if image.shape[0] != expected_steps or image.shape[-1] != 3:
                    raise ValueError(
                        f"{label} loader {key} sample has invalid shape: {image.shape}"
                    )
            for key in LOW_DIM_KEYS:
                value = np.asarray(sample["obs"][key])
                if value.shape[0] != expected_steps or not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(
                        f"{label} loader {key} sample has invalid shape/content"
                    )
        return {
            "validated": True,
            "schema": schema,
            "train_sequences": int(len(trainset)),
            "valid_sequences": int(len(validset)),
            "sample_action_shape": list(np.asarray(train_sample["actions"]).shape),
        }
    finally:
        _close_dataset(trainset)
        _close_dataset(validset)


def preflight(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Public shorthand for the stock-loader preflight."""

    return standard_loader_preflight(config_dict)


def experiment_root(config: dict[str, Any]) -> Path:
    return Path(config["train"]["output_dir"]) / config["experiment"]["name"]


def _without_runtime_optimizer_fields(value: Any) -> Any:
    if isinstance(value, dict):
        result = {
            key: _without_runtime_optimizer_fields(item)
            for key, item in value.items()
        }
        if "optimizer_type" in result:
            result.pop("num_epochs", None)
            result.pop("num_train_batches", None)
        return result
    if isinstance(value, list):
        return [_without_runtime_optimizer_fields(item) for item in value]
    return value


def canonical_training_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Materialize defaults while ignoring runtime-only optimizer fields."""

    from robomimic.config import config_factory

    cleaned = _without_runtime_optimizer_fields(payload)
    config = config_factory(cleaned["algo_name"])
    with config.values_unlocked():
        config.update(cleaned)
    return _without_runtime_optimizer_fields(config.to_dict())


def _run_config_matches(run_dir: Path, config: dict[str, Any]) -> bool:
    saved_path = run_dir / "config.json"
    if not saved_path.is_file():
        return False
    try:
        saved = json.loads(saved_path.read_text())
        return canonical_training_config(saved) == canonical_training_config(config)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return False


def _checkpoint_candidates(config: dict[str, Any]) -> list[Path]:
    epoch = int(config["train"]["num_epochs"])
    root = experiment_root(config)
    filename = f"model_epoch_{epoch}.pth"
    suffixed = f"model_epoch_{epoch}_*.pth"
    candidates = [root / "models" / filename]
    candidates.extend((root / "models").glob(suffixed))
    candidates.extend(root.glob(f"*/models/{filename}"))
    candidates.extend(root.glob(f"*/models/{suffixed}"))
    existing = {
        path.resolve()
        for path in candidates
        if path.is_file() and path.stat().st_size > 0
    }
    return sorted(existing, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def find_checkpoint(config: dict[str, Any]) -> Path | None:
    for checkpoint in _checkpoint_candidates(config):
        if _run_config_matches(checkpoint.parent.parent, config):
            return checkpoint
    return None


def _resume_run_dir(config: dict[str, Any]) -> Path:
    """Resolve only an exact config-and-fingerprint-compatible run."""

    root = experiment_root(config)
    if not root.is_dir():
        raise FileNotFoundError(f"cannot resume missing experiment: {root}")
    if (root / "models").is_dir() and any(
        (root / filename).is_file() for filename in ("last.pth", "last_bak.pth")
    ):
        run_dir = root
    else:
        run_dirs = sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and (child / "models").is_dir()
        )
        if not run_dirs:
            raise FileNotFoundError(f"no resumable run exists under {root}")
        run_dir = run_dirs[-1]
        if not any(
            (run_dir / filename).is_file()
            for filename in ("last.pth", "last_bak.pth")
        ):
            raise FileNotFoundError(f"latest run has no resume checkpoint: {run_dir}")
    if not _run_config_matches(run_dir, config):
        raise ValueError(
            f"refusing to resume {run_dir}: its config or dataset fingerprint "
            "does not match this request"
        )
    return run_dir


def process_env(*, smoke: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["PYTHONPYCACHEPREFIX"] = "/tmp/robomimic_pick_cup_5hz_rgb_dp_pycache"
    env["ROBOMIMIC_SAVE_LATEST_EVERY_N_EPOCHS"] = "1" if smoke else "10"
    return env


def train(config_path: Path, config: dict[str, Any], *, resume: bool) -> Path:
    existing_checkpoint = find_checkpoint(config)
    if existing_checkpoint is not None:
        print(
            "[pick_cup 5 Hz RGB-DP] using existing completed checkpoint: "
            f"{existing_checkpoint}",
            flush=True,
        )
        return existing_checkpoint

    output_root = experiment_root(config)
    if resume:
        resume_dir = _resume_run_dir(config)
        print(
            f"[pick_cup 5 Hz RGB-DP] resuming compatible run: {resume_dir}",
            flush=True,
        )
    elif output_root.exists():
        raise FileExistsError(
            f"experiment exists but has no compatible completed checkpoint at "
            f"{output_root}; use --name-suffix for a new run, or --resume only "
            "when the latest run has the same config and dataset fingerprint"
        )

    command = [
        str(PYTHON),
        "-B",
        "-m",
        "robomimic.scripts.train",
        "--config",
        str(config_path),
    ]
    if resume:
        command.append("--resume")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=process_env(smoke="_smoke" in config["experiment"]["name"]),
        check=True,
    )
    checkpoint = find_checkpoint(config)
    if checkpoint is None:
        raise RuntimeError(
            "robomimic training returned without the expected epoch checkpoint; "
            f"inspect logs under {output_root}"
        )
    print(f"[pick_cup 5 Hz RGB-DP] training target: {checkpoint}", flush=True)
    return checkpoint


def _training_options(args: argparse.Namespace) -> TrainingOptions:
    return TrainingOptions(
        dataset_dir=args.dataset_dir,
        config_dir=args.config_dir,
        model_root=args.model_root,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        batch_size=args.batch_size,
        num_data_workers=args.num_data_workers,
        learning_rate=args.learning_rate,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        save_every_epochs=args.save_every_epochs,
        cuda=not args.cpu,
        smoke=args.smoke,
        name_suffix=args.name_suffix,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("dataset", "validate", "prepare", "train"),
        default=["prepare"],
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_5HZ_DATASET_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--crop-height", type=int, default=DEFAULT_CROP_HEIGHT)
    parser.add_argument("--crop-width", type=int, default=DEFAULT_CROP_WIDTH)
    parser.add_argument("--validation-count", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--max-image-age-sec",
        type=float,
        default=DEFAULT_MAX_IMAGE_AGE_SEC,
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
    )
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=67)
    parser.add_argument("--validation-steps", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-data-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every-epochs", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-loader-preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    shard = dataset_path(dataset_dir)
    options = _training_options(args)
    report: dict[str, Any] = {}

    if "dataset" in args.stages:
        report["dataset"] = build_dataset(
            BuildOptions(
                source_root=source,
                output_dir=dataset_dir,
                image_height=args.image_height,
                image_width=args.image_width,
                validation_count=args.validation_count,
                split_seed=args.split_seed,
                max_image_age_sec=args.max_image_age_sec,
                compression=args.compression,
                overwrite=args.force_dataset,
            )
        )

    needs_validation = bool(
        {"validate", "prepare", "train"}.intersection(args.stages)
    )
    if needs_validation:
        report["validation"] = validate_5hz_dataset(
            shard,
            source_root=source,
        )

    config = None
    config_path = None
    if {"prepare", "train"}.intersection(args.stages):
        image_height, image_width = _dataset_image_shape(shard)
        config = make_config(
            options,
            image_height=image_height,
            image_width=image_width,
        )
        config_path = write_config(config, options.config_dir)
        report["config"] = str(config_path)
        if not args.skip_loader_preflight:
            report["loader_preflight"] = standard_loader_preflight(config)

    if "train" in args.stages:
        assert config is not None and config_path is not None
        report["checkpoint"] = str(
            train(config_path, config, resume=args.resume)
        )

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
