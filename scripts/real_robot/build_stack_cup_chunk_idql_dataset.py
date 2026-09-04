#!/usr/bin/env python3
"""Build or validate the stack-cup human/rollout mixed IDQL dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import build_pick_cup_chunk_idql_dataset as core


DEFAULT_HUMAN_DATASET = ROOT / "datasets/real_robot/stack_cup/stack_cup_rgb.hdf5"
DEFAULT_ROLLOUT_DATASET = (
    ROOT
    / "datasets/real_robot/stack_cup/idql/stack_cup_epoch200_ddim100_20hz_rollouts.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/stack_cup/idql/"
    "stack_cup_chunk_idql_44demo_20success_10failure_ddim100_terminal_success.hdf5"
)


def configure_core() -> None:
    core.__doc__ = __doc__
    core.DEFAULT_HUMAN_DATASETS = (DEFAULT_HUMAN_DATASET,)
    core.DEFAULT_ROLLOUT_DATASET = DEFAULT_ROLLOUT_DATASET
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.TASK = "stack_cup"
    core.BUILDER_VERSION = "stack_cup_chunk_idql_mixed_v2"
    core.DEFAULT_HUMAN_COUNT = 44
    core.DEFAULT_EXPECTED_HUMAN_TRANSITIONS = 18_062
    core.DEFAULT_SUCCESS_COUNT = 20
    core.DEFAULT_FAILURE_COUNT = 10
    core.DEFAULT_HUMAN_DATASETS_HELP = (
        "Stack-cup human HDF5 (defaults to the Episode-007-excluded train split)."
    )


def main(argv: Sequence[str] | None = None) -> dict:
    configure_core()
    return core.main(argv)


if __name__ == "__main__":
    main()
