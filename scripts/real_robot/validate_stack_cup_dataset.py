#!/usr/bin/env python3
"""Validate the single-shard Real4D stack-cup RGB-DP dataset.

The checks in this module deliberately do more than inspect HDF5 shapes.  When
the raw handoff is available, every normalized motion action and every temporal
selection is reconstructed from ``actions.json`` and ``frames.json``.  This
guards against applying the robot's physical action scale during conversion and
against selecting a future or merely plausible (rather than latest causal) RGB
pair.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import stack_cup_common as StackCupCommon  # noqa: E402


SCHEMA_VERSION = StackCupCommon.SCHEMA_VERSION
CONVERSION_VERSION = StackCupCommon.CONVERSION_VERSION
CONVERSION_MANIFEST_ATTR = StackCupCommon.CONVERSION_MANIFEST_ATTR
DEFAULT_SOURCE = StackCupCommon.DEFAULT_SOURCE
DEFAULT_DATASET_DIR = StackCupCommon.DEFAULT_DATASET_DIR
DATASET_FILENAME = StackCupCommon.DATASET_FILENAME

TASK_NAME = StackCupCommon.TASK_NAME
TASK_LABEL = StackCupCommon.TASK_LABEL
ENV_NAME = StackCupCommon.ENV_NAME
OUTCOME_MANUAL_REVIEW = StackCupCommon.OUTCOME_MANUAL_REVIEW
EXPECTED_SOURCE_EPISODES = (
    StackCupCommon.EXPECTED_EPISODE_NUMBERS
    - frozenset(StackCupCommon.EXCLUDED_EPISODES)
)
EXPECTED_VALID_EPISODES = StackCupCommon.VALIDATION_EPISODE_NUMBERS
EXPECTED_TRAIN_EPISODES = EXPECTED_SOURCE_EPISODES - EXPECTED_VALID_EPISODES
EXPECTED_EXCLUDED_EPISODES = frozenset(StackCupCommon.EXCLUDED_EPISODES)

RGB_KEYS = ("main_image", "wrist_image")
LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
)
EXPECTED_PROVENANCE_KEYS = {
    "raw_gripper_event",
    "source_action_array_index",
    "source_action_index",
    "source_action_step",
    "source_action_offset_ms",
    "action_target_time",
    "action_source_time",
    "selected_frame_position",
    "selected_frame_index",
    "selected_frame_nominal_time",
    "selected_main_capture_time",
    "selected_wrist_capture_time",
    "selected_pair_capture_time",
    "image_age_sec",
}
EXPECTED_MASK_KEYS = {
    "all",
    "train",
    "valid",
    "qa_pass",
    "clean",
    "train_clean",
}
DEMO_PATTERN = re.compile(r"^demo_(?P<number>[0-9]{3})$")
GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")

ACTION_HZ = 20.0
IMAGE_HZ = 5.0
TRANSLATION_SCALE_M = 0.012
ROTATION_SCALE_RAD = 0.036
MAX_IMAGE_AGE_SEC = 0.5
MAX_CAMERA_SKEW_SEC = 0.05


def _decode_text(value: Any, *, name: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _json_attr(attrs: h5py.AttributeManager, key: str, *, location: str) -> Any:
    if key not in attrs:
        raise ValueError(f"{location} is missing attribute {key!r}")
    value = _decode_text(attrs[key], name=f"{location} attribute {key!r}")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} attribute {key!r} is malformed JSON") from exc


def _require_dataset(
    group: h5py.Group,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | type | None = None,
) -> h5py.Dataset:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise ValueError(f"{group.name} is missing dataset {name!r}")
    dataset = group[name]
    if shape is not None and tuple(dataset.shape) != tuple(shape):
        raise ValueError(f"{dataset.name} has shape {dataset.shape}, expected {shape}")
    if dtype is not None and np.dtype(dataset.dtype) != np.dtype(dtype):
        raise ValueError(
            f"{dataset.name} has dtype {dataset.dtype}, expected {np.dtype(dtype)}"
        )
    return dataset


def _finite(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")


def _strictly_increasing(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    _finite(values, name=name)
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _decode_strings(dataset: h5py.Dataset) -> list[str]:
    return [_decode_text(value, name=dataset.name) for value in dataset[:]]


def _demo_key(number: int) -> str:
    return f"demo_{number:03d}"


def _episode_number(demo_key: str) -> int:
    match = DEMO_PATTERN.fullmatch(demo_key)
    if match is None:
        raise ValueError(f"invalid demonstration key {demo_key!r}")
    return int(match.group("number"))


def _densify_gripper(
    raw_events: np.ndarray,
    *,
    initial_state: float,
) -> tuple[np.ndarray, np.ndarray]:
    events = np.asarray(raw_events, dtype=np.float32)
    if events.ndim != 1 or not np.all(np.isin(events, (-1.0, 0.0, 1.0))):
        raise ValueError("raw gripper events must be a 1-D {-1,0,+1} sequence")
    if initial_state not in (-1.0, 1.0):
        raise ValueError(f"initial gripper state must be -1 or +1, got {initial_state}")
    observations = np.empty_like(events)
    targets = np.empty_like(events)
    state = float(initial_state)
    for index, event in enumerate(events):
        observations[index] = state
        if event < 0.0:
            state = -1.0
        elif event > 0.0:
            state = 1.0
        targets[index] = state
    return observations, targets


def _require_manifest_contract(manifest: Mapping[str, Any], *, path: Path) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} manifest schema version mismatch")
    if manifest.get("conversion_version") != CONVERSION_VERSION:
        raise ValueError(f"{path} manifest conversion version mismatch")

    action = manifest.get("action")
    if not isinstance(action, Mapping):
        raise ValueError(f"{path} manifest is missing action metadata")
    try:
        action_hz = float(action["hz"])
        translation_scale = float(action["translation_scale_m"])
        rotation_scale = float(action["rotation_scale_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} manifest action metadata is malformed") from exc
    if action_hz != ACTION_HZ:
        raise ValueError(f"{path} manifest action rate is not 20 Hz")
    if translation_scale != TRANSLATION_SCALE_M or rotation_scale != ROTATION_SCALE_RAD:
        raise ValueError(f"{path} manifest physical action scale is incorrect")
    normalization = str(action.get("normalization", "")).lower()
    if "[-1,1]" not in normalization.replace(" ", ""):
        raise ValueError(f"{path} manifest does not declare normalized actions")

    alignment = manifest.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError(f"{path} manifest is missing alignment metadata")
    try:
        maximum_age = float(alignment["max_image_age_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} manifest alignment metadata is malformed") from exc
    if maximum_age != MAX_IMAGE_AGE_SEC:
        raise ValueError(f"{path} manifest image-age limit must be exactly 0.5 s")
    selection = str(alignment.get("selection", "")).lower()
    if "latest" not in selection or "not later" not in selection:
        raise ValueError(f"{path} manifest does not declare latest-causal RGB selection")

    image = manifest.get("image")
    if not isinstance(image, Mapping):
        raise ValueError(f"{path} manifest is missing image metadata")
    try:
        source_hz = float(image["source_hz"])
        height = int(image["height"])
        width = int(image["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} manifest image metadata is malformed") from exc
    if source_hz != IMAGE_HZ or height < 2 or width < 2:
        raise ValueError(f"{path} manifest image contract is invalid")

    gripper = manifest.get("gripper")
    if not isinstance(gripper, Mapping):
        raise ValueError(f"{path} manifest is missing gripper metadata")
    combined = " ".join(str(value).lower() for value in gripper.values())
    for token in ("-1 close", "0 hold", "+1 open", "before", "after"):
        if token not in combined:
            raise ValueError(f"{path} manifest gripper contract is incomplete")

    exclusions = manifest.get("excluded_episodes")
    if not isinstance(exclusions, list):
        raise ValueError(f"{path} manifest is missing excluded_episodes")
    excluded_numbers = {
        int(item.get("source_episode_number", item.get("episode_number", -1)))
        for item in exclusions
        if isinstance(item, Mapping)
    }
    if excluded_numbers != EXPECTED_EXCLUDED_EPISODES:
        raise ValueError(f"{path} manifest exclusions do not match the task contract")

    outcome = manifest.get("outcome")
    expected_outcome = {
        "source_machine_label": False,
        "manual_review": OUTCOME_MANUAL_REVIEW,
        "robomimic_compatibility_label": (
            "reward=1 and done=1 only on the final retained row; earlier rows are 0"
        ),
        "diffusion_policy_uses_reward": False,
    }
    if outcome != expected_outcome:
        raise ValueError(f"{path} manifest outcome provenance is missing or incorrect")


def _validate_environment(
    data: h5py.Group,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    env_args = _json_attr(data.attrs, "env_args", location=data.name)
    if not isinstance(env_args, dict):
        raise ValueError(f"{data.name} env_args must be an object")
    if env_args.get("env_name") != ENV_NAME:
        raise ValueError(f"{data.name} env_name must be {ENV_NAME}")
    if env_args.get("env_version") != CONVERSION_VERSION:
        raise ValueError(f"{data.name} env_version mismatch")
    if int(env_args.get("type", -1)) != 2:
        raise ValueError(f"{data.name} env type must be 2")
    kwargs = env_args.get("env_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError(f"{data.name} env_kwargs must be an object")
    expected = {
        "real_robot": True,
        "control_freq": int(ACTION_HZ),
        "camera_names": ["main", "wrist"],
        "camera_height": image_height,
        "camera_width": image_width,
        "task": TASK_NAME,
    }
    for key, value in expected.items():
        if kwargs.get(key) != value:
            raise ValueError(f"{data.name} env_kwargs[{key!r}] does not match")
    return env_args


def _validate_masks(
    dataset: h5py.File,
    demo_keys: set[str],
    manifest: Mapping[str, Any],
    demo_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    if "mask" not in dataset or not isinstance(dataset["mask"], h5py.Group):
        raise ValueError(f"{dataset.filename} is missing /mask")
    mask = dataset["mask"]
    if set(mask.keys()) != EXPECTED_MASK_KEYS:
        raise ValueError(
            f"{mask.name} keys {sorted(mask.keys())} do not match the {TASK_LABEL} contract"
        )
    masks: dict[str, list[str]] = {}
    for name in sorted(EXPECTED_MASK_KEYS):
        values = _decode_strings(_require_dataset(mask, name))
        if len(values) != len(set(values)):
            raise ValueError(f"{mask.name}/{name} contains duplicate demos")
        unknown = set(values) - demo_keys
        if unknown:
            raise ValueError(f"{mask.name}/{name} references unknown demos: {sorted(unknown)}")
        masks[name] = values

    expected_all = {_demo_key(number) for number in EXPECTED_SOURCE_EPISODES}
    expected_valid = {_demo_key(number) for number in EXPECTED_VALID_EPISODES}
    expected_train = expected_all - expected_valid
    if demo_keys != expected_all or set(masks["all"]) != expected_all:
        exclusion_text = ""
        if EXPECTED_EXCLUDED_EPISODES:
            exclusion_text = " excluding " + ",".join(
                f"{number:03d}" for number in sorted(EXPECTED_EXCLUDED_EPISODES)
            )
        raise ValueError(
            "mask/all and /data do not match the configured episodes"
            f"{exclusion_text}"
        )
    if set(masks["valid"]) != expected_valid:
        raise ValueError("mask/valid does not match the fixed validation episodes")
    if set(masks["train"]) != expected_train:
        raise ValueError("mask/train does not match all minus the fixed validation set")
    if set(masks["train"]).intersection(masks["valid"]):
        raise ValueError("train and valid masks overlap")
    expected_counts = (len(expected_all), len(expected_train), len(expected_valid))
    actual_counts = (
        len(masks["all"]),
        len(masks["train"]),
        len(masks["valid"]),
    )
    if actual_counts != expected_counts:
        raise ValueError(
            "all/train/valid mask counts do not match the task contract: "
            f"actual={actual_counts}, expected={expected_counts}"
        )

    expected_qa_pass = {
        key for key, metadata in demo_metadata.items() if metadata["qa_status"] == "PASS"
    }
    expected_clean = {
        key
        for key in expected_all
        if key in expected_qa_pass and int(demo_metadata[key]["invalid_windows"]) == 0
    }
    expected_train_clean = {
        key
        for key in expected_train
        if key in expected_clean
    }
    if set(masks["qa_pass"]) != expected_qa_pass:
        raise ValueError("mask/qa_pass does not exactly encode QA PASS episodes")
    if set(masks["clean"]) != expected_clean:
        raise ValueError(
            "mask/clean must be QA PASS intersect invalid_windows==0"
        )
    if set(masks["train_clean"]) != expected_train_clean:
        raise ValueError(
            "mask/train_clean must be train intersect QA PASS intersect invalid_windows==0"
        )

    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("conversion manifest is missing split metadata")
    for name in EXPECTED_MASK_KEYS:
        if set(split.get(name, ())) != set(masks[name]):
            raise ValueError(f"manifest split {name!r} does not match mask/{name}")
    return masks


def _validate_episode_structure(
    demo: h5py.Group,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    key = demo.name.rsplit("/", 1)[-1]
    number = _episode_number(key)
    if number not in EXPECTED_SOURCE_EPISODES:
        raise ValueError(f"{demo.name} is not an allowed {TASK_LABEL} source episode")
    if int(demo.attrs.get("source_episode_number", -1)) != number:
        raise ValueError(f"{demo.name} source_episode_number does not match its key")
    count = int(demo.attrs.get("num_samples", -1))
    if count < 1:
        raise ValueError(f"{demo.name} has invalid num_samples={count}")
    qa_status = _decode_text(demo.attrs.get("qa_status"), name=f"{demo.name} qa_status")
    if qa_status not in {"PASS", "WARN"}:
        raise ValueError(f"{demo.name} has unsupported qa_status={qa_status!r}")
    invalid_windows = int(demo.attrs.get("invalid_windows", -1))
    if invalid_windows < 0:
        raise ValueError(f"{demo.name} has invalid invalid_windows")
    model_window_ready = demo.attrs.get("model_window_ready")
    if not isinstance(model_window_ready, (bool, np.bool_)):
        raise ValueError(f"{demo.name} model_window_ready must be boolean")

    actions_ds = _require_dataset(demo, "actions", shape=(count, 7), dtype=np.float32)
    rewards_ds = _require_dataset(demo, "rewards", shape=(count,), dtype=np.float32)
    dones_ds = _require_dataset(demo, "dones", shape=(count,), dtype=np.uint8)
    actions = actions_ds[:]
    rewards = rewards_ds[:]
    dones = dones_ds[:]
    _finite(actions, name=actions_ds.name)
    if np.any(actions < -1.001) or np.any(actions > 1.001):
        raise ValueError(f"{actions_ds.name} is outside normalized [-1,1]")
    if not np.all(np.isin(actions[:, 6], (-1.0, 1.0))):
        raise ValueError(f"{actions_ds.name} gripper targets are not dense signs")
    expected_rewards = np.zeros(count, dtype=np.float32)
    expected_rewards[-1] = 1.0
    expected_dones = np.zeros(count, dtype=np.uint8)
    expected_dones[-1] = 1
    if not np.array_equal(rewards, expected_rewards):
        raise ValueError(f"{rewards_ds.name} must have only a final reward of one")
    if not np.array_equal(dones, expected_dones):
        raise ValueError(f"{dones_ds.name} must have only a final done flag")
    if "next_obs" in demo:
        raise ValueError(f"{demo.name} must not store next_obs for Diffusion Policy")

    if "obs" not in demo or not isinstance(demo["obs"], h5py.Group):
        raise ValueError(f"{demo.name} is missing /obs")
    obs = demo["obs"]
    if set(obs.keys()) != set((*RGB_KEYS, *LOW_DIM_KEYS)):
        raise ValueError(f"{obs.name} keys do not match the {TASK_LABEL} observation schema")
    image_shape = (count, image_height, image_width, 3)
    for image_key in RGB_KEYS:
        _require_dataset(obs, image_key, shape=image_shape, dtype=np.uint8)
    positions = _require_dataset(
        obs, "robot0_eef_pos", shape=(count, 3), dtype=np.float32
    )[:]
    quaternions = _require_dataset(
        obs, "robot0_eef_quat", shape=(count, 4), dtype=np.float32
    )[:]
    gripper_obs = _require_dataset(
        obs, "robot0_gripper_state", shape=(count, 1), dtype=np.float32
    )[:, 0]
    for values, name in (
        (positions, "positions"),
        (quaternions, "quaternions"),
        (gripper_obs, "gripper observations"),
    ):
        _finite(values, name=f"{demo.name} {name}")
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=5e-3):
        raise ValueError(f"{demo.name} contains non-unit xyzw quaternions")
    if count > 1 and np.any(np.sum(quaternions[:-1] * quaternions[1:], axis=1) < 0.0):
        raise ValueError(f"{demo.name} xyzw quaternion hemisphere is discontinuous")
    if not np.all(np.isin(gripper_obs, (-1.0, 1.0))):
        raise ValueError(f"{demo.name} gripper observations are not logical signs")

    if "provenance" not in demo or not isinstance(demo["provenance"], h5py.Group):
        raise ValueError(f"{demo.name} is missing /provenance")
    provenance = demo["provenance"]
    if set(provenance.keys()) != EXPECTED_PROVENANCE_KEYS:
        raise ValueError(f"{provenance.name} keys do not match the {TASK_LABEL} contract")
    for provenance_key in EXPECTED_PROVENANCE_KEYS:
        _require_dataset(provenance, provenance_key, shape=(count,))

    raw_events = np.asarray(provenance["raw_gripper_event"][:], dtype=np.float32)
    initial_state = float(demo.attrs.get("initial_gripper_state", np.nan))
    expected_obs, expected_targets = _densify_gripper(
        raw_events, initial_state=initial_state
    )
    if not np.array_equal(gripper_obs, expected_obs):
        raise ValueError(f"{demo.name} violates pre-action gripper-state semantics")
    if not np.array_equal(actions[:, 6], expected_targets):
        raise ValueError(f"{demo.name} dense gripper actions are inconsistent")

    action_times = np.asarray(provenance["action_target_time"][:], dtype=np.float64)
    main_times = np.asarray(
        provenance["selected_main_capture_time"][:], dtype=np.float64
    )
    wrist_times = np.asarray(
        provenance["selected_wrist_capture_time"][:], dtype=np.float64
    )
    pair_times = np.asarray(
        provenance["selected_pair_capture_time"][:], dtype=np.float64
    )
    ages = np.asarray(provenance["image_age_sec"][:], dtype=np.float64)
    frame_positions = np.asarray(
        provenance["selected_frame_position"][:], dtype=np.int64
    )
    frame_indices = np.asarray(
        provenance["selected_frame_index"][:], dtype=np.int64
    )
    _strictly_increasing(action_times, name=f"{demo.name} action target times")
    for values, name in (
        (main_times, "main capture times"),
        (wrist_times, "wrist capture times"),
        (pair_times, "paired capture times"),
        (ages, "image ages"),
    ):
        _finite(values, name=f"{demo.name} {name}")
    expected_pair_times = np.maximum(main_times, wrist_times)
    if not np.allclose(pair_times, expected_pair_times, atol=1e-9, rtol=0.0):
        raise ValueError(f"{demo.name} paired capture times are inconsistent")
    if np.any(main_times > action_times + 1e-6) or np.any(
        wrist_times > action_times + 1e-6
    ):
        raise ValueError(f"{demo.name} uses a future camera capture")
    expected_ages = action_times - pair_times
    if not np.allclose(ages, expected_ages, atol=1e-7, rtol=0.0):
        raise ValueError(f"{demo.name} image ages are inconsistent with timestamps")
    if np.any(ages < -1e-6) or float(np.max(ages)) > MAX_IMAGE_AGE_SEC + 1e-6:
        raise ValueError(f"{demo.name} violates the 0.5 s causal image-age limit")
    if np.any(np.abs(main_times - wrist_times) > MAX_CAMERA_SKEW_SEC + 1e-6):
        raise ValueError(f"{demo.name} exceeds the 50 ms paired-camera skew limit")
    if np.any(np.diff(frame_positions) < 0) or np.any(np.diff(frame_indices) < 0):
        raise ValueError(f"{demo.name} frame provenance is not monotonic")
    attr_max_age = float(demo.attrs.get("max_image_age_sec", np.nan))
    if not np.isclose(attr_max_age, float(np.max(ages)), atol=1e-9, rtol=0.0):
        raise ValueError(f"{demo.name} max_image_age_sec attribute is inconsistent")

    return {
        "demo_key": key,
        "source_episode_number": number,
        "samples": count,
        "qa_status": qa_status,
        "invalid_windows": invalid_windows,
        "model_window_ready": bool(model_window_ready),
        "dropped_prefix_actions": int(demo.attrs.get("dropped_prefix_actions", -1)),
        "max_image_age_sec": float(np.max(ages)),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
    }


def _source_episode_dir(source_root: Path, demo: h5py.Group, number: int) -> Path:
    relative = _decode_text(
        demo.attrs.get("source_directory"), name=f"{demo.name} source_directory"
    )
    candidate = (source_root / relative).resolve()
    try:
        candidate.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{demo.name} source_directory escapes the source root") from exc
    if not candidate.is_dir():
        raise FileNotFoundError(f"{demo.name} source episode is missing: {candidate}")
    if not candidate.name.startswith(f"episode_{number:03d}__"):
        raise ValueError(f"{demo.name} source_directory does not match its episode number")
    run_id = _decode_text(demo.attrs.get("source_run_id"), name=f"{demo.name} source_run_id")
    if candidate.name.split("__", 1)[-1] != run_id:
        raise ValueError(f"{demo.name} source_run_id does not match its directory")
    return candidate


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_against_source(
    demo: h5py.Group,
    metadata: Mapping[str, Any],
    *,
    source_root: Path,
) -> None:
    number = int(metadata["source_episode_number"])
    episode_dir = _source_episode_dir(source_root, demo, number)
    actions_doc = _load_json_object(episode_dir / "actions.json")
    frames_doc = _load_json_object(episode_dir / "frames.json")
    qa_doc = _load_json_object(episode_dir / "qa.json")
    windows_doc = _load_json_object(episode_dir / "windows.json")
    samples = actions_doc.get("samples")
    frames = frames_doc.get("frames")
    if not isinstance(samples, list) or not samples or not isinstance(frames, list) or not frames:
        raise ValueError(f"{episode_dir} has empty source action or frame metadata")

    qa_status = str(qa_doc.get("status", ""))
    if qa_status != metadata["qa_status"]:
        raise ValueError(f"{demo.name} qa_status differs from its raw source")
    invalid_windows = int(windows_doc.get("invalid_windows", -1))
    if invalid_windows != metadata["invalid_windows"]:
        raise ValueError(f"{demo.name} invalid_windows differs from its raw source")
    ready = (episode_dir / "MODEL_WINDOW_READY").is_file()
    if ready != metadata["model_window_ready"]:
        raise ValueError(f"{demo.name} model_window_ready differs from its raw source")

    raw_actions = np.asarray([item.get("action") for item in samples], dtype=np.float32)
    raw_poses = np.asarray([item.get("before_pose") for item in samples], dtype=np.float32)
    if raw_actions.shape != (len(samples), 7) or raw_poses.shape != (len(samples), 7):
        raise ValueError(f"{episode_dir} has malformed source action or pose arrays")
    raw_target_times = np.asarray([item["target_time"] for item in samples], dtype=np.float64)
    raw_source_times = np.asarray(
        [item.get("source_time", item["target_time"]) for item in samples],
        dtype=np.float64,
    )
    raw_source_indices = np.asarray(
        [item.get("source_index", index) for index, item in enumerate(samples)],
        dtype=np.int64,
    )
    raw_steps = np.asarray(
        [item.get("step", index) for index, item in enumerate(samples)], dtype=np.int64
    )
    raw_offsets = np.asarray(
        [item.get("offset_ms", 0.0) for item in samples], dtype=np.float64
    )

    frame_indices: list[int] = []
    frame_nominal_times: list[float] = []
    main_times: list[float] = []
    wrist_times: list[float] = []
    for position, frame in enumerate(frames):
        streams = frame.get("streams")
        if not isinstance(streams, dict):
            raise ValueError(f"{episode_dir}/frames.json frame {position} has no streams")
        try:
            main = streams["main_rgb"]
            wrist = streams["wrist_rgb"]
            main_times.append(int(main["header_stamp_ns"]) / 1e9)
            wrist_times.append(int(wrist["header_stamp_ns"]) / 1e9)
            frame_indices.append(int(frame.get("index", position)))
            frame_nominal_times.append(float(frame["target_time"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{episode_dir}/frames.json frame {position} is malformed") from exc
    main_times_array = np.asarray(main_times, dtype=np.float64)
    wrist_times_array = np.asarray(wrist_times, dtype=np.float64)
    paired_times = np.maximum(main_times_array, wrist_times_array)
    _strictly_increasing(paired_times, name=f"{episode_dir} paired capture times")
    mapping = np.searchsorted(paired_times, raw_target_times, side="right") - 1
    causal = mapping >= 0
    if not np.any(causal):
        raise ValueError(f"{episode_dir} has no causal source RGB pair")
    first = int(np.flatnonzero(causal)[0])
    if not np.all(causal[first:]):
        raise ValueError(f"{episode_dir} has an internal non-causal source mapping")
    expected_rows = np.arange(first, len(samples), dtype=np.int64)
    expected_mapping = mapping[first:].astype(np.int64)
    expected_ages = raw_target_times[first:] - paired_times[expected_mapping]
    if float(np.max(expected_ages)) > MAX_IMAGE_AGE_SEC + 1e-6:
        raise ValueError(f"{demo.name} raw source violates the 0.5 s image-age contract")

    provenance = demo["provenance"]
    actual_rows = np.asarray(provenance["source_action_array_index"][:], dtype=np.int64)
    if not np.array_equal(actual_rows, expected_rows):
        raise ValueError(f"{demo.name} source action membership is not the exact causal suffix")
    if int(metadata["dropped_prefix_actions"]) != first:
        raise ValueError(f"{demo.name} dropped_prefix_actions differs from its raw source")

    def require_equal(actual: np.ndarray, expected: np.ndarray, *, label: str) -> None:
        if actual.dtype.kind in "fc" or expected.dtype.kind in "fc":
            equal = np.array_equal(actual, expected)
        else:
            equal = np.array_equal(actual, expected)
        if not equal:
            raise ValueError(f"{demo.name} {label} differs from its raw source")

    require_equal(
        np.asarray(demo["actions"][:, :6]),
        raw_actions[first:, :6],
        label="normalized motion actions (possible physical scaling)",
    )
    require_equal(
        np.asarray(demo["obs/robot0_eef_pos"][:]),
        raw_poses[first:, :3],
        label="EEF positions",
    )
    require_equal(
        np.asarray(demo["obs/robot0_eef_quat"][:]),
        raw_poses[first:, 3:7],
        label="EEF quaternions",
    )
    comparisons = {
        "raw_gripper_event": raw_actions[first:, 6],
        "source_action_index": raw_source_indices[first:],
        "source_action_step": raw_steps[first:],
        "source_action_offset_ms": raw_offsets[first:],
        "action_target_time": raw_target_times[first:],
        "action_source_time": raw_source_times[first:],
        "selected_frame_position": expected_mapping,
        "selected_frame_index": np.asarray(frame_indices, dtype=np.int64)[expected_mapping],
        "selected_frame_nominal_time": np.asarray(
            frame_nominal_times, dtype=np.float64
        )[expected_mapping],
        "selected_main_capture_time": main_times_array[expected_mapping],
        "selected_wrist_capture_time": wrist_times_array[expected_mapping],
        "selected_pair_capture_time": paired_times[expected_mapping],
        "image_age_sec": expected_ages,
    }
    for name, expected in comparisons.items():
        require_equal(np.asarray(provenance[name][:]), expected, label=name)

    # Strongest form of the causal contract: the following source pair, when it
    # exists, must be strictly later than the action.  This rejects selecting an
    # old frame while keeping self-consistent selected timestamps.
    following = expected_mapping + 1
    has_following = following < paired_times.size
    if np.any(paired_times[following[has_following]] <= raw_target_times[first:][has_following]):
        raise ValueError(f"{demo.name} did not select the latest causal source RGB pair")


def validate_dataset(
    path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one task HDF5 shard, optionally against its raw source."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{TASK_LABEL} dataset does not exist: {path}")
    source_root = (
        None if source_root is None else source_root.expanduser().resolve()
    )
    with h5py.File(path, "r") as dataset:
        if int(dataset.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"{path} has an unsupported schema_version")
        if dataset.attrs.get("conversion_version") != CONVERSION_VERSION:
            raise ValueError(f"{path} has an unsupported conversion_version")
        manifest = _json_attr(
            dataset.attrs, CONVERSION_MANIFEST_ATTR, location=str(path)
        )
        if not isinstance(manifest, dict):
            raise ValueError(f"{path} conversion manifest must be an object")
        _require_manifest_contract(manifest, path=path)

        generation_id = manifest.get("generation_id")
        if not isinstance(generation_id, str) or GENERATION_PATTERN.fullmatch(
            generation_id
        ) is None:
            raise ValueError(f"{path} has an invalid generation_id")
        root_generation = _decode_text(
            dataset.attrs.get("generation_id"), name=f"{path} generation_id"
        )
        if root_generation != generation_id:
            raise ValueError(f"{path} generation_id differs between root and manifest")

        identity = manifest.get("source_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{path} manifest source_identity must be an object")
        if source_root is not None:
            actual_identity = StackCupCommon.source_identity(source_root)
            if identity != actual_identity:
                raise ValueError(f"{path} source identity does not match {source_root}")

        image = manifest["image"]
        image_height = int(image["height"])
        image_width = int(image["width"])
        if "data" not in dataset or not isinstance(dataset["data"], h5py.Group):
            raise ValueError(f"{path} is missing /data")
        data = dataset["data"]
        _validate_environment(
            data, image_height=image_height, image_width=image_width
        )
        demo_keys = set(data.keys())
        expected_demo_keys = {_demo_key(number) for number in EXPECTED_SOURCE_EPISODES}
        if demo_keys != expected_demo_keys:
            exclusion_text = ""
            if EXPECTED_EXCLUDED_EPISODES:
                exclusion_text = " excluding " + ",".join(
                    f"{number:03d}"
                    for number in sorted(EXPECTED_EXCLUDED_EPISODES)
                )
            raise ValueError(
                f"{path} demo inventory does not match the {TASK_LABEL} contract"
                f"{exclusion_text}"
            )

        episodes: list[dict[str, Any]] = []
        demo_metadata: dict[str, dict[str, Any]] = {}
        for key in sorted(demo_keys):
            metadata = _validate_episode_structure(
                data[key], image_height=image_height, image_width=image_width
            )
            episodes.append(metadata)
            demo_metadata[key] = metadata
        masks = _validate_masks(dataset, demo_keys, manifest, demo_metadata)

        total = sum(int(item["samples"]) for item in episodes)
        if int(data.attrs.get("total", -1)) != total:
            raise ValueError(f"{data.name} total does not match demonstration lengths")

        manifest_episodes = manifest.get("episodes")
        if not isinstance(manifest_episodes, list):
            raise ValueError(f"{path} manifest episodes must be a list")
        manifest_keys = {
            str(item.get("demo_key"))
            for item in manifest_episodes
            if isinstance(item, Mapping)
        }
        if manifest_keys != demo_keys or len(manifest_episodes) != len(demo_keys):
            raise ValueError(f"{path} manifest episodes do not exactly match /data")

        if source_root is not None:
            for item in episodes:
                _validate_against_source(
                    data[item["demo_key"]], item, source_root=source_root
                )

        return {
            "validated": True,
            "path": str(path),
            "episodes": len(episodes),
            "train_episodes": len(masks["train"]),
            "valid_episodes": len(masks["valid"]),
            "qa_pass_episodes": len(masks["qa_pass"]),
            "clean_episodes": len(masks["clean"]),
            "train_clean_episodes": len(masks["train_clean"]),
            "samples": total,
            "dropped_prefix_actions": sum(
                int(item["dropped_prefix_actions"]) for item in episodes
            ),
            "max_image_age_sec": max(
                float(item["max_image_age_sec"]) for item in episodes
            ),
            "action_min": min(float(item["action_min"]) for item in episodes),
            "action_max": max(float(item["action_max"]) for item in episodes),
            "generation_id": generation_id,
            "source_identity": identity,
            "source_checked": source_root is not None,
            "schema_signature": {
                "ac_dim": 7,
                "action_shape": [7],
                "action_dtype": "float32",
                "image_shape": [image_height, image_width, 3],
                "image_dtype": "uint8",
                "obs_shapes": {
                    "main_image": [image_height, image_width, 3],
                    "wrist_image": [image_height, image_width, 3],
                    "robot0_eef_pos": [3],
                    "robot0_eef_quat": [4],
                    "robot0_gripper_state": [1],
                },
            },
        }


def _dataset_commit_path(dataset_dir: Path) -> Path:
    helper = getattr(StackCupCommon, "dataset_commit_path", None)
    if callable(helper):
        return Path(helper(dataset_dir))
    filename = getattr(StackCupCommon, "DATASET_COMMIT_FILENAME", "dataset_commit.json")
    return dataset_dir / str(filename)


def validate_published_dataset(
    dataset_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the published shard and its generation commit marker."""

    dataset_dir = dataset_dir.expanduser().resolve()
    path = StackCupCommon.dataset_path(dataset_dir)
    report = validate_dataset(path, source_root=source_root)
    commit_path = _dataset_commit_path(dataset_dir)
    if not commit_path.is_file():
        raise FileNotFoundError(
            f"published {TASK_LABEL} dataset is missing commit marker: {commit_path}"
        )
    try:
        commit = json.loads(commit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset commit marker: {commit_path}") from exc
    if not isinstance(commit, dict):
        raise ValueError(f"dataset commit marker must be a JSON object: {commit_path}")
    if commit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset commit marker schema version mismatch")
    if commit.get("conversion_version") != CONVERSION_VERSION:
        raise ValueError("dataset commit marker conversion version mismatch")
    if commit.get("generation_id") != report["generation_id"]:
        raise ValueError("dataset commit marker generation does not match the shard")
    if commit.get("source_identity") != report["source_identity"]:
        raise ValueError("dataset commit marker source identity does not match the shard")
    expected_shards = [{"filename": path.name, "size_bytes": int(path.stat().st_size)}]
    if commit.get("shards") != expected_shards:
        raise ValueError("dataset commit marker does not match the published shard")
    report["commit_path"] = str(commit_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    source_root = None if args.skip_source_check else args.source
    if args.dataset is None:
        report = validate_published_dataset(
            args.dataset_dir, source_root=source_root
        )
    else:
        report = validate_dataset(args.dataset, source_root=source_root)
    if args.report is not None:
        StackCupCommon.atomic_write_json(args.report.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
