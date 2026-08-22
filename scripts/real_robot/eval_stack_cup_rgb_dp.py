#!/usr/bin/env python3
"""Evaluate a 20 Hz stack-cup Diffusion Policy on every held-out window.

The report keeps the checkpoint's two relevant networks separate:

* ``online_validation`` is the native diffusion noise-prediction loss evaluated
  with the online training network.
* ``ema_action_replay`` samples action chunks with the EMA network used by
  rollout and compares all eight executed slots with held-out demonstrations.

Replay errors are open-loop imitation diagnostics, not real-robot task success.
The validation loader is sequential and uses ``drop_last=False`` so, unless
``--max-windows`` is explicitly requested, no held-out window is omitted.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot.run_stack_cup_rgb_dp_baseline import (  # noqa: E402
    dataset_fingerprint,
)
from scripts.real_robot.stack_cup_common import (  # noqa: E402
    ACTION_HZ,
    CONVERSION_MANIFEST_ATTR,
    ROTATION_SCALE_RAD,
    TRANSLATION_SCALE_M,
    atomic_write_json,
    dataset_path,
)


EVALUATOR_VERSION = "stack_cup_rgb_dp_20hz_eval_v1"
ACTION_DIM = 7
MOTION_DIM = 6
CONTROL_HZ = ACTION_HZ
EXPECTED_HORIZONS = (2, 8, 16)


def _decode_hdf5_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


class ActionReplayAccumulator:
    """Accumulate chunk errors without retaining predictions in memory."""

    def __init__(
        self,
        *,
        translation_scale_m: float,
        rotation_scale_rad: float,
    ) -> None:
        self.translation_scale_m = float(translation_scale_m)
        self.rotation_scale_rad = float(rotation_scale_rad)
        self.action_count = 0
        self.motion_abs_sum = 0.0
        self.motion_sq_sum = 0.0
        self.motion_abs_by_dim = np.zeros(MOTION_DIM, dtype=np.float64)
        self.target_motion_abs_sum = 0.0
        self.translation_sq_norm_sum = 0.0
        self.rotation_sq_norm_sum = 0.0
        self.gripper_correct = 0
        self.out_of_bounds = 0
        self.prediction_min = math.inf
        self.prediction_max = -math.inf
        self._horizon: int | None = None
        self._step_counts: np.ndarray | None = None
        self._step_motion_abs: np.ndarray | None = None
        self._step_motion_sq: np.ndarray | None = None
        self._step_translation_sq_norm: np.ndarray | None = None
        self._step_rotation_sq_norm: np.ndarray | None = None
        self._step_gripper_correct: np.ndarray | None = None

    def _initialize_steps(self, horizon: int) -> None:
        self._horizon = int(horizon)
        self._step_counts = np.zeros(horizon, dtype=np.int64)
        self._step_motion_abs = np.zeros(horizon, dtype=np.float64)
        self._step_motion_sq = np.zeros(horizon, dtype=np.float64)
        self._step_translation_sq_norm = np.zeros(horizon, dtype=np.float64)
        self._step_rotation_sq_norm = np.zeros(horizon, dtype=np.float64)
        self._step_gripper_correct = np.zeros(horizon, dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} != "
                f"{target.shape}"
            )
        if prediction.ndim != 3 or prediction.shape[-1] != ACTION_DIM:
            raise ValueError(f"expected [B, T, 7] actions, got {prediction.shape}")
        if prediction.shape[0] < 1 or prediction.shape[1] < 1:
            raise ValueError("action replay batches must be non-empty")
        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
            raise ValueError("action replay contains NaN or Inf")
        if self._horizon is None:
            self._initialize_steps(prediction.shape[1])
        elif prediction.shape[1] != self._horizon:
            raise ValueError(
                f"action horizon changed during replay: {prediction.shape[1]} != "
                f"{self._horizon}"
            )

        target_gripper = target[:, :, 6]
        if not np.all(np.isin(target_gripper, (-1.0, 1.0))):
            raise ValueError("held-out gripper targets must be dense -1 or +1")

        error = prediction[:, :, :MOTION_DIM] - target[:, :, :MOTION_DIM]
        absolute_error = np.abs(error)
        translation_error = error[:, :, :3] * self.translation_scale_m
        rotation_error = error[:, :, 3:6] * self.rotation_scale_rad
        predicted_gripper = np.where(prediction[:, :, 6] >= 0.0, 1.0, -1.0)
        gripper_correct = predicted_gripper == target_gripper

        batch_size, horizon = prediction.shape[:2]
        flat_prediction = prediction.reshape(-1, ACTION_DIM)
        self.action_count += int(batch_size * horizon)
        self.motion_abs_sum += float(absolute_error.sum())
        self.motion_sq_sum += float(np.square(error).sum())
        self.motion_abs_by_dim += absolute_error.sum(axis=(0, 1))
        self.target_motion_abs_sum += float(
            np.abs(target[:, :, :MOTION_DIM]).sum()
        )
        self.translation_sq_norm_sum += float(np.square(translation_error).sum())
        self.rotation_sq_norm_sum += float(np.square(rotation_error).sum())
        self.gripper_correct += int(np.count_nonzero(gripper_correct))
        self.out_of_bounds += int(
            np.count_nonzero(
                (flat_prediction < -1.000001) | (flat_prediction > 1.000001)
            )
        )
        self.prediction_min = min(self.prediction_min, float(flat_prediction.min()))
        self.prediction_max = max(self.prediction_max, float(flat_prediction.max()))

        assert self._step_counts is not None
        assert self._step_motion_abs is not None
        assert self._step_motion_sq is not None
        assert self._step_translation_sq_norm is not None
        assert self._step_rotation_sq_norm is not None
        assert self._step_gripper_correct is not None
        self._step_counts += batch_size
        self._step_motion_abs += absolute_error.sum(axis=(0, 2))
        self._step_motion_sq += np.square(error).sum(axis=(0, 2))
        self._step_translation_sq_norm += np.square(translation_error).sum(
            axis=(0, 2)
        )
        self._step_rotation_sq_norm += np.square(rotation_error).sum(axis=(0, 2))
        self._step_gripper_correct += np.count_nonzero(gripper_correct, axis=0)

    def result(self) -> dict[str, Any]:
        if self.action_count == 0 or self._horizon is None:
            raise ValueError("no action predictions were evaluated")
        assert self._step_counts is not None
        assert self._step_motion_abs is not None
        assert self._step_motion_sq is not None
        assert self._step_translation_sq_norm is not None
        assert self._step_rotation_sq_norm is not None
        assert self._step_gripper_correct is not None

        motion_values = self.action_count * MOTION_DIM
        action_values = self.action_count * ACTION_DIM
        rotation_rmse_rad = math.sqrt(
            self.rotation_sq_norm_sum / self.action_count
        )
        per_step = []
        for step in range(self._horizon):
            count = int(self._step_counts[step])
            if count < 1:
                raise ValueError(f"action step {step} has no predictions")
            per_step.append(
                {
                    "relative_step": step,
                    "action_predictions": count,
                    "normalized_motion_mae": float(
                        self._step_motion_abs[step] / (count * MOTION_DIM)
                    ),
                    "normalized_motion_rmse": float(
                        math.sqrt(
                            self._step_motion_sq[step] / (count * MOTION_DIM)
                        )
                    ),
                    "gripper_sign_accuracy": float(
                        self._step_gripper_correct[step] / count
                    ),
                    "translation_vector_rmse_mm": float(
                        1000.0
                        * math.sqrt(self._step_translation_sq_norm[step] / count)
                    ),
                    "rotation_vector_rmse_deg": float(
                        math.degrees(
                            math.sqrt(self._step_rotation_sq_norm[step] / count)
                        )
                    ),
                }
            )
        return {
            "action_predictions": self.action_count,
            "normalized_motion_mae": self.motion_abs_sum / motion_values,
            "normalized_motion_rmse": math.sqrt(
                self.motion_sq_sum / motion_values
            ),
            "normalized_motion_mae_by_dim": (
                self.motion_abs_by_dim / self.action_count
            ).tolist(),
            "zero_motion_baseline_mae": self.target_motion_abs_sum / motion_values,
            "gripper_sign_accuracy": self.gripper_correct / self.action_count,
            "translation_vector_rmse_m": math.sqrt(
                self.translation_sq_norm_sum / self.action_count
            ),
            "translation_vector_rmse_mm": 1000.0
            * math.sqrt(self.translation_sq_norm_sum / self.action_count),
            "rotation_vector_rmse_rad": rotation_rmse_rad,
            "rotation_vector_rmse_deg": math.degrees(rotation_rmse_rad),
            "per_action_step": per_step,
            "prediction_min": self.prediction_min,
            "prediction_max": self.prediction_max,
            "out_of_bounds_values": self.out_of_bounds,
            "out_of_bounds_fraction": self.out_of_bounds / action_values,
        }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _close_dataset(dataset: Any) -> None:
    if dataset is None:
        return
    children = getattr(dataset, "datasets", (dataset,))
    for child in children:
        close = getattr(child, "close_and_delete_hdf5_handle", None)
        if close is not None:
            close()


def _identity_action_stats(stats: dict[str, Any] | None) -> bool:
    if stats is None or "actions" not in stats:
        return False
    scale = np.asarray(stats["actions"]["scale"], dtype=np.float64)
    offset = np.asarray(stats["actions"]["offset"], dtype=np.float64)
    return bool(
        scale.shape[-1] == ACTION_DIM
        and offset.shape[-1] == ACTION_DIM
        and np.allclose(scale, 1.0, rtol=0.0, atol=1e-8)
        and np.allclose(offset, 0.0, rtol=0.0, atol=1e-8)
    )


def _dataset_contract(
    config_dict: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    data = config_dict["train"]["data"]
    if len(data) != 1:
        raise ValueError("the stack-cup evaluator requires exactly one dataset shard")
    dataset_config = data[0]
    path = Path(dataset_config["path"]).expanduser().resolve()
    expected_path = dataset_path(path.parent).resolve()
    if path != expected_path:
        raise ValueError(
            f"stack-cup checkpoint dataset must be the published shard "
            f"{expected_path}, got {path}"
        )
    from scripts.real_robot.validate_stack_cup_dataset import (
        validate_published_dataset,
    )

    published_report = validate_published_dataset(path.parent, source_root=None)
    if Path(published_report["path"]).resolve() != path:
        raise ValueError("published validation resolved a different dataset shard")
    expected_fingerprint = dataset_config.get("dataset_fingerprint")
    if expected_fingerprint is None:
        raise ValueError("checkpoint config does not contain a dataset fingerprint")
    actual_fingerprint = dataset_fingerprint(path)
    if dict(expected_fingerprint) != actual_fingerprint:
        raise ValueError(
            "the current dataset does not match the checkpoint fingerprint: "
            f"expected={dict(expected_fingerprint)}, actual={actual_fingerprint}"
        )

    validation_key = config_dict["train"]["hdf5_validation_filter_key"]
    with h5py.File(path, "r") as dataset:
        if "mask" not in dataset or validation_key not in dataset["mask"]:
            raise KeyError(f"dataset has no validation mask {validation_key!r}")
        validation_demos = _decode_hdf5_strings(
            dataset[f"mask/{validation_key}"][:]
        )
        validation_samples = sum(
            int(dataset[f"data/{demo}"].attrs["num_samples"])
            for demo in validation_demos
        )
        raw_manifest = dataset.attrs.get(CONVERSION_MANIFEST_ATTR)
    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode("utf-8")
    if not isinstance(raw_manifest, str):
        raise ValueError("dataset is missing its conversion manifest")
    try:
        manifest = json.loads(raw_manifest)
        action_manifest = manifest["action"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("dataset has a malformed action conversion contract") from exc

    expected = {
        "hz": CONTROL_HZ,
        "translation_scale_m": TRANSLATION_SCALE_M,
        "rotation_scale_rad": ROTATION_SCALE_RAD,
    }
    for key, expected_value in expected.items():
        try:
            actual = float(action_manifest[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"dataset action contract is missing {key}") from exc
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"dataset action contract {key}={actual} does not match "
                f"expected {expected_value}"
            )
    return path, {
        "fingerprint": actual_fingerprint,
        "validation_mask": str(validation_key),
        "validation_demos": validation_demos,
        "validation_samples": validation_samples,
        "control_hz": float(action_manifest["hz"]),
        "translation_scale_m": float(action_manifest["translation_scale_m"]),
        "rotation_scale_rad": float(action_manifest["rotation_scale_rad"]),
        "published_generation_id": published_report["generation_id"],
        "commit_path": published_report["commit_path"],
    }


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    max_windows: int | None,
    torch: Any,
) -> tuple[Any, int]:
    selected = dataset
    count = len(dataset)
    if max_windows is not None:
        count = min(count, max_windows)
        selected = torch.utils.data.Subset(dataset, range(count))
    loader = torch.utils.data.DataLoader(
        dataset=selected,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )
    return loader, count


def evaluate_online_validation_loss(
    *,
    model: Any,
    loader: Any,
    seeds: Sequence[int],
    epoch: int,
    obs_normalization_stats: dict[str, Any] | None,
    torch: Any,
) -> dict[str, Any]:
    """Evaluate the checkpoint's online diffusion-noise objective."""

    model.set_eval()
    per_seed = []
    started = time.time()
    for seed in seeds:
        _seed_everything(seed, torch)
        weighted_loss = 0.0
        windows = 0
        with torch.inference_mode():
            for batch in loader:
                batch_windows = int(batch["actions"].shape[0])
                processed = model.process_batch_for_training(batch)
                processed = model.postprocess_batch_for_training(
                    processed,
                    obs_normalization_stats=obs_normalization_stats,
                )
                info = model.train_on_batch(processed, epoch, validate=True)
                loss = float(info["losses"]["total_loss"].item())
                if not math.isfinite(loss):
                    raise ValueError("online validation produced a non-finite loss")
                weighted_loss += loss * batch_windows
                windows += batch_windows
        if windows == 0:
            raise ValueError("online validation loader produced no windows")
        value = weighted_loss / windows
        per_seed.append({"seed": int(seed), "loss": value, "windows": windows})
        print(
            f"[stack-cup eval] online loss seed={seed}: {value:.8f} "
            f"({windows} windows)",
            file=sys.stderr,
            flush=True,
        )
    values = np.asarray([item["loss"] for item in per_seed], dtype=np.float64)
    return {
        "network": "online",
        "description": (
            "online-network diffusion noise-prediction MSE; this is the native "
            "training validation objective"
        ),
        "per_seed": per_seed,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "elapsed_sec": time.time() - started,
    }


def evaluate_ema_action_replay(
    *,
    rollout_policy: Any,
    loader: Any,
    seeds: Sequence[int],
    translation_scale_m: float,
    rotation_scale_rad: float,
    torch: Any,
) -> dict[str, Any]:
    """Generate all eight EMA rollout actions and compare their target slots."""

    model = rollout_policy.policy
    if model.ema is None:
        raise ValueError("the checkpoint has no EMA policy for rollout evaluation")
    model.set_eval()
    observation_horizon = int(model.algo_config.horizon.observation_horizon)
    action_horizon = int(model.algo_config.horizon.action_horizon)
    prediction_horizon = int(model.algo_config.horizon.prediction_horizon)
    horizons = (observation_horizon, action_horizon, prediction_horizon)
    if horizons != EXPECTED_HORIZONS:
        raise ValueError(
            "expected 20 Hz horizons To/Ta/Tp=2/8/16, got "
            f"{observation_horizon}/{action_horizon}/{prediction_horizon}"
        )

    combined = ActionReplayAccumulator(
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
    )
    per_seed = []
    started = time.time()
    target_start = observation_horizon - 1
    target_end = target_start + action_horizon
    obs_keys = tuple(model.global_config.all_obs_keys)

    for seed in seeds:
        _seed_everything(seed, torch)
        accumulator = ActionReplayAccumulator(
            translation_scale_m=translation_scale_m,
            rotation_scale_rad=rotation_scale_rad,
        )
        windows = 0
        with torch.inference_mode():
            for batch in loader:
                raw_obs = {
                    key: batch["obs"][key][:, :observation_horizon]
                    for key in obs_keys
                }
                prepared_obs = rollout_policy._prepare_observation(
                    raw_obs,
                    batched_ob=True,
                )
                prediction = model._get_action_trajectory(
                    obs_dict=prepared_obs,
                    goal_dict=None,
                )
                target = batch["actions"][:, target_start:target_end]
                prediction_np = prediction.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                accumulator.update(prediction_np, target_np)
                combined.update(prediction_np, target_np)
                windows += int(batch["actions"].shape[0])
        seed_result = accumulator.result()
        seed_result.update({"seed": int(seed), "windows": windows})
        per_seed.append(seed_result)
        print(
            f"[stack-cup eval] EMA replay seed={seed}: "
            f"motion_mae={seed_result['normalized_motion_mae']:.8f}, "
            f"motion_rmse={seed_result['normalized_motion_rmse']:.8f}, "
            f"gripper={seed_result['gripper_sign_accuracy']:.4%} "
            f"({windows} windows)",
            file=sys.stderr,
            flush=True,
        )

    aggregate = combined.result()
    aggregate["unique_windows"] = per_seed[0]["windows"] if per_seed else 0
    aggregate["seed_count"] = len(seeds)
    return {
        "network": "ema",
        "description": (
            "EMA rollout slots To-1 through To-1+Ta compared with the eight "
            "demonstrated action targets; open-loop imitation, not task success"
        ),
        "target_slots": list(range(target_start, target_end)),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "elapsed_sec": time.time() - started,
    }


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    seeds: Sequence[int],
    batch_size: int,
    num_workers: int,
    max_windows: int | None,
    device_request: str,
    evaluate_online: bool,
    evaluate_ema: bool,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one random seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive when provided")
    if not evaluate_online and not evaluate_ema:
        raise ValueError("at least one evaluation mode must be enabled")

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    import torch
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.train_utils as TrainUtils

    device = _resolve_device(device_request, torch)
    print(
        f"[stack-cup eval] loading {checkpoint} on {device}",
        file=sys.stderr,
        flush=True,
    )
    rollout_policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_path=str(checkpoint),
        verbose=False,
    )
    model = rollout_policy.policy
    config = model.global_config
    config_dict = config.to_dict()
    if config_dict["algo_name"] != "diffusion_policy":
        raise ValueError("checkpoint is not a Diffusion Policy")
    if config_dict["train"]["action_config"] != {
        "actions": {"normalization": None}
    }:
        raise ValueError("the evaluator requires normalized actions stored directly")
    if not _identity_action_stats(rollout_policy.action_normalization_stats):
        raise ValueError("checkpoint action statistics are not the expected identity map")

    dataset_path, dataset_report = _dataset_contract(config_dict)
    variable_state = checkpoint_dict.get("variable_state") or {}
    checkpoint_epoch = int(variable_state.get("epoch", 0))
    checkpoint_report = {
        "path": str(checkpoint),
        "size_bytes": int(checkpoint.stat().st_size),
        "mtime_ns": int(checkpoint.stat().st_mtime_ns),
        "epoch": checkpoint_epoch,
        "best_valid_loss_at_save": variable_state.get("best_valid_loss"),
        "online_network_for_validation": True,
        "ema_network_for_action_replay": model.ema is not None,
    }

    del checkpoint_dict
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validset = None
    try:
        validset = TrainUtils.dataset_factory(
            config,
            obs_keys=list(config.all_obs_keys),
            filter_by_attribute=config.train.hdf5_validation_filter_key,
        )
        validset.set_action_normalization_stats(
            copy.deepcopy(rollout_policy.action_normalization_stats)
        )
        loader, evaluated_windows = _make_loader(
            validset,
            batch_size=batch_size,
            num_workers=num_workers,
            max_windows=max_windows,
            torch=torch,
        )
        if len(validset) != int(dataset_report["validation_samples"]):
            raise ValueError(
                "validation loader and HDF5 mask counts differ: "
                f"{len(validset)} != {dataset_report['validation_samples']}"
            )

        report: dict[str, Any] = {
            "evaluator_version": EVALUATOR_VERSION,
            "checkpoint": checkpoint_report,
            "dataset": {
                "path": str(dataset_path),
                **dataset_report,
            },
            "evaluation": {
                "device": str(device),
                "seeds": [int(seed) for seed in seeds],
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "available_windows": int(len(validset)),
                "evaluated_windows_per_seed": int(evaluated_windows),
                "full_validation_coverage": evaluated_windows == len(validset),
                "shuffle": False,
                "drop_last": False,
                "boundary_padding_matches_training": True,
            },
        }
        if evaluate_online:
            report["online_validation"] = evaluate_online_validation_loss(
                model=model,
                loader=loader,
                seeds=seeds,
                epoch=checkpoint_epoch,
                obs_normalization_stats=rollout_policy.obs_normalization_stats,
                torch=torch,
            )
        if evaluate_ema:
            report["ema_action_replay"] = evaluate_ema_action_replay(
                rollout_policy=rollout_policy,
                loader=loader,
                seeds=seeds,
                translation_scale_m=float(dataset_report["translation_scale_m"]),
                rotation_scale_rad=float(dataset_report["rotation_scale_rad"]),
                torch=torch,
            )
        return report
    finally:
        _close_dataset(validset)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="evaluate only the first N held-out windows (smoke testing only)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--skip-online-loss", action="store_true")
    parser.add_argument("--skip-ema-replay", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = evaluate_checkpoint(
        args.checkpoint,
        seeds=args.seeds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_windows=args.max_windows,
        device_request=args.device,
        evaluate_online=not args.skip_online_loss,
        evaluate_ema=not args.skip_ema_replay,
    )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        atomic_write_json(output, report)
        print(
            f"[stack-cup eval] wrote report: {output}",
            file=sys.stderr,
            flush=True,
        )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
