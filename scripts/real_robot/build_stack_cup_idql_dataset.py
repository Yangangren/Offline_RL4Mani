#!/usr/bin/env python3
"""Build or validate the action-state StackCup mixed one-step IDQL dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import build_real_robot_mixed_idql_dataset as core


DEFAULT_HUMAN_DATASET = (
    ROOT / "datasets/real_robot/stack_cup/idql/stack_cup_episode_layout_v1_human.hdf5"
)
DEFAULT_ROLLOUT_DATASET = (
    ROOT
    / "datasets/real_robot/stack_cup/idql/"
    "stack_cup_episode_layout_v1_one_step_rollouts.hdf5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/stack_cup/idql/"
    "stack_cup_idql_episode_layout_v1_45demo_20success_10failure_"
    "terminal_success.hdf5"
)


def configure_core() -> None:
    core.__doc__ = __doc__
    core.DEFAULT_HUMAN_DATASETS = (DEFAULT_HUMAN_DATASET,)
    core.DEFAULT_ROLLOUT_DATASET = DEFAULT_ROLLOUT_DATASET
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.TASK = "stack_cup"
    core.BUILDER_VERSION = "stack_cup_idql_mixed_v2_episode_layout_one_step"
    core.REQUIRE_CRITIC_VALIDITY = True
    core.EXPERT_CHUNK_VALIDITY_MODE = "source"
    core.DEFAULT_HUMAN_COUNT = 45
    core.DEFAULT_EXPECTED_HUMAN_TRANSITIONS = 18_841
    core.DEFAULT_SUCCESS_COUNT = 20
    core.DEFAULT_FAILURE_COUNT = 10
    core.DEFAULT_ACTOR_CONDITION_MODE = "human_only"
    core.DEFAULT_HUMAN_DATASETS_HELP = (
        "Stack-cup human HDF5 (defaults to the fixed 45-episode train split)."
    )


def main(argv: Sequence[str] | None = None) -> dict:
    configure_core()
    return core.main(argv)


if __name__ == "__main__":
    main()
