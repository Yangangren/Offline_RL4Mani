#!/usr/bin/env python3
"""Convert the audited MoveSpoon DDIM-100 rollout handoff to robomimic HDF5.

The handoff is published on a 20 Hz wall-clock grid, but DDIM-100 inference
gaps cause many rows to repeat one immutable controller action. This converter
uses the strict StackCup implementation to verify source provenance, retain one
row per source action, causally align paired RGB observations, and record which
successor transitions are temporally valid for a fixed-rate dynamics loss.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import move_spoon_common  # noqa: E402

move_spoon_common.configure_core()

from scripts.real_robot import build_move_spoon_dataset as shared  # noqa: E402
from scripts.real_robot import (  # noqa: E402
    build_stack_cup_processed_rollout_hdf5 as core,
)


DEFAULT_SOURCE = Path("/home/ryan/datasets/move_spoon/rollout")
DEFAULT_HUMAN_DATASET = (
    ROOT / "datasets/real_robot/move_spoon/move_spoon_rgb.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/move_spoon/idql/"
    "move_spoon_epoch200_ddim100_20hz_rollouts.hdf5"
)
CONVERSION_VERSION = "move_spoon_epoch200_ddim100_executed_actions_v1"
EXPECTED_EPISODES = frozenset(range(1, 41))
EXPECTED_OUTCOMES = {"success": 25, "failure": 15}
EXPECTED_EXECUTED_ACTIONS = 600
SUCCESS_VALID_COUNT = 5
FAILURE_VALID_COUNT = 4
EXPECTED_CHECKPOINT_SHA256 = (
    "c20a6497c82ffedc6dd8849a4bbaad9f29612bff70bb02015b8e2fa59f65ecdf"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "37f11abacf1067a18668e526ac8d5fb152ffe8d6d2887ec74a9c6b66c62dcc75"
)
EXPECTED_COLLECTION_CONTRACT_SHA256 = (
    "3d9b7a33ded0594a97ff1dd54d9f7240d0fe27a3738dd14a7506653c5f724a08"
)
EXPECTED_SERVER_IDENTITY_SHA256 = (
    "9ee2a7f366cdde07ed8732407555a7d5619bad7490ba21443092bbaa1e7d7dfa"
)
EXPECTED_COLLECTION_POLICY = "robomimic_move_spoon_dp_epoch200_ddim100_20hz"
EXPECTED_ENV_NAME = "MoveSpoonReal-v0"
DEFAULT_MAX_DYNAMICS_GAP_SEC = 0.1

# The collection deliberately spans three spoon-placement regimes. The held-out
# split keeps both outcomes from every regime: 3/2 from episodes 1-25, 1/1 from
# 26-29, and 1/1 from 30-40 (success/failure respectively).
VALIDATION_REGIME_QUOTAS = (
    (1, 25, 3, 2),
    (26, 29, 1, 1),
    (30, 40, 1, 1),
)
EPISODE_RE = re.compile(
    r"^episode_(?P<number>\d{3})__"
    r"(?P<run>put_spoon_dp_epoch200_ddim100_20hz_real_"
    r"(?P<day>\d{8})_(?P<clock>\d{6}))$"
)


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output: Path = DEFAULT_OUTPUT
    compression: str = "gzip"
    split_seed: int = 1
    image_height: int = 96
    image_width: int = 128
    max_image_age_sec: float = 0.5
    max_dynamics_gap_sec: float | None = DEFAULT_MAX_DYNAMICS_GAP_SEC
    episode_limit: int | None = None
    overwrite: bool = False
    validate_only: bool = False
    validate_output_only: bool = False


def configure_core() -> None:
    """Install the immutable MoveSpoon collection contract in the shared core."""

    move_spoon_common.configure_core()
    core.__doc__ = __doc__
    core.shared = shared.core
    core.DEFAULT_SOURCE = DEFAULT_SOURCE
    core.DEFAULT_HUMAN_DATASET = DEFAULT_HUMAN_DATASET
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.CONVERSION_VERSION = CONVERSION_VERSION
    core.EXPECTED_EPISODES = EXPECTED_EPISODES
    core.EXPECTED_OUTCOMES = EXPECTED_OUTCOMES
    core.EXPECTED_EXECUTED_ACTIONS = EXPECTED_EXECUTED_ACTIONS
    core.SUCCESS_VALID_COUNT = SUCCESS_VALID_COUNT
    core.FAILURE_VALID_COUNT = FAILURE_VALID_COUNT
    core.EXPECTED_CHECKPOINT_SHA256 = EXPECTED_CHECKPOINT_SHA256
    core.EXPECTED_DATASET_MANIFEST_SHA256 = EXPECTED_DATASET_MANIFEST_SHA256
    core.EXPECTED_COLLECTION_CONTRACT_SHA256 = (
        EXPECTED_COLLECTION_CONTRACT_SHA256
    )
    core.EXPECTED_SERVER_IDENTITY_SHA256 = EXPECTED_SERVER_IDENTITY_SHA256
    core.EXPECTED_COLLECTION_POLICY = EXPECTED_COLLECTION_POLICY
    core.EXPECTED_ENV_NAME = EXPECTED_ENV_NAME
    core.REQUIRE_ZERO_VALID_WINDOWS = False
    core.VALIDATION_REGIME_QUOTAS = VALIDATION_REGIME_QUOTAS
    core.DEFAULT_MAX_DYNAMICS_GAP_SEC = DEFAULT_MAX_DYNAMICS_GAP_SEC
    core.EPISODE_RE = EPISODE_RE


def discover_episodes(source_root: Path):
    configure_core()
    return core.discover_episodes(source_root)


def split_masks(episodes, seed: int):
    configure_core()
    return core.split_masks(episodes, seed)


def build_dataset(options: BuildOptions) -> dict:
    configure_core()
    return core.build_dataset(options)


def parse_args(argv: Sequence[str] | None = None) -> BuildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--max-image-age-sec", type=float, default=0.5)
    parser.add_argument(
        "--max-dynamics-gap-sec",
        type=float,
        default=DEFAULT_MAX_DYNAMICS_GAP_SEC,
        help=(
            "Mark a successor transition valid for fixed-rate dynamics only when "
            "its source time gap does not exceed this value."
        ),
    )
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only and args.validate_output_only:
        parser.error("choose at most one validation mode")
    return BuildOptions(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = build_dataset(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


RolloutConversionError = core.RolloutConversionError


if __name__ == "__main__":
    raise SystemExit(main())
