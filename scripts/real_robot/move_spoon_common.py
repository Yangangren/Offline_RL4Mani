#!/usr/bin/env python3
"""Task contract for the real-robot move-spoon demonstrations."""

from __future__ import annotations

from pathlib import Path

from scripts.real_robot import stack_cup_common as core


ROOT = Path(__file__).resolve().parents[2]
TASK_NAME = "move_spoon"
TASK_LABEL = "move-spoon"
ENV_NAME = "MoveSpoonReal-v0"
DEFAULT_SOURCE = Path("/home/ryan/datasets/move_spoon/human_demo")
DEFAULT_DATASET_DIR = ROOT / "datasets/real_robot/move_spoon"
DATASET_FILENAME = "move_spoon_rgb.hdf5"
CONVERSION_VERSION = "move_spoon_rgb_dp_v1"
EXPECTED_EPISODE_NUMBERS = frozenset(range(1, 51))
EXCLUDED_EPISODES: dict[int, str] = {}

# The demonstrations contain three spatial regimes (episodes 1-25, 26-29,
# and 30-50). Keep clean examples from every regime in the held-out split.
VALIDATION_EPISODE_NUMBERS = frozenset({5, 15, 27, 35, 45})
EXPECTED_INCLUDED_EPISODES = 50
OUTCOME_MANUAL_REVIEW = (
    "the source handoff verification marks every retained human demonstration "
    "PASS; Diffusion Policy training does not consume reward or done labels"
)


def configure_core() -> None:
    """Install this task contract in the shared real-robot dataset helpers."""

    core.TASK_NAME = TASK_NAME
    core.TASK_LABEL = TASK_LABEL
    core.ENV_NAME = ENV_NAME
    core.DEFAULT_SOURCE = DEFAULT_SOURCE
    core.DEFAULT_DATASET_DIR = DEFAULT_DATASET_DIR
    core.DATASET_FILENAME = DATASET_FILENAME
    core.CONVERSION_VERSION = CONVERSION_VERSION
    core.EXPECTED_EPISODE_NUMBERS = EXPECTED_EPISODE_NUMBERS
    core.EXCLUDED_EPISODES = EXCLUDED_EPISODES
    core.VALIDATION_EPISODE_NUMBERS = VALIDATION_EPISODE_NUMBERS
    core.EXPECTED_INCLUDED_EPISODES = EXPECTED_INCLUDED_EPISODES
    core.OUTCOME_MANUAL_REVIEW = OUTCOME_MANUAL_REVIEW


configure_core()

# Re-export shared immutable schema and helper functions after configuring the
# task-specific globals used by their implementations.
SCHEMA_VERSION = core.SCHEMA_VERSION
CONVERSION_MANIFEST_ATTR = core.CONVERSION_MANIFEST_ATTR
DATASET_COMMIT_FILENAME = core.DATASET_COMMIT_FILENAME
CONVERSION_SUMMARY_FILENAME = core.CONVERSION_SUMMARY_FILENAME
ACTION_HZ = core.ACTION_HZ
IMAGE_HZ = core.IMAGE_HZ
TRANSLATION_SCALE_M = core.TRANSLATION_SCALE_M
ROTATION_SCALE_RAD = core.ROTATION_SCALE_RAD
DEFAULT_IMAGE_HEIGHT = core.DEFAULT_IMAGE_HEIGHT
DEFAULT_IMAGE_WIDTH = core.DEFAULT_IMAGE_WIDTH
DEFAULT_CROP_HEIGHT = core.DEFAULT_CROP_HEIGHT
DEFAULT_CROP_WIDTH = core.DEFAULT_CROP_WIDTH
DEFAULT_MAX_IMAGE_AGE_SEC = core.DEFAULT_MAX_IMAGE_AGE_SEC
RGB_KEYS = core.RGB_KEYS
LOW_DIM_KEYS = core.LOW_DIM_KEYS
OBS_KEYS = core.OBS_KEYS
dataset_path = core.dataset_path
dataset_commit_path = core.dataset_commit_path
atomic_write_json = core.atomic_write_json
source_identity = core.source_identity

