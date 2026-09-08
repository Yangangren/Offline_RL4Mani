#!/usr/bin/env python3
"""Shared contract for the canonical PickCup episode-layout dataset."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_NAME = "pick_cup"
TASK_LABEL = "pick-cup"
ENV_NAME = "PickCupReal-v0"
DEFAULT_SOURCE = Path("/home/ryan/datasets_new/pick_cup/human")
DEFAULT_DATASET_DIR = ROOT / "datasets/real_robot/pick_cup"
DATASET_FILENAME = "pick_cup_rgb.hdf5"

SCHEMA_VERSION = 1
CONVERSION_VERSION = "real_robot_episode_layout_idql_sources_v1"
CONVERSION_MANIFEST_ATTR = "real_robot_conversion_manifest"

RGB_KEYS = ("main_image", "wrist_image")
LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_state",
)
OBS_KEYS = (*RGB_KEYS, *LOW_DIM_KEYS)

ACTION_HZ = 20.0
IMAGE_HZ = 5.0
DEFAULT_IMAGE_HEIGHT = 96
DEFAULT_IMAGE_WIDTH = 128
DEFAULT_CROP_HEIGHT = 84
DEFAULT_CROP_WIDTH = 112
DEFAULT_MAX_IMAGE_AGE_SEC = 1.0
VALIDATION_EPISODE_NUMBERS = frozenset({2, 5, 43, 48, 49})


def dataset_path(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Path:
    return dataset_dir.expanduser().resolve() / DATASET_FILENAME


def atomic_write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
