#!/usr/bin/env python3
"""Build the real-robot move-spoon RGB Diffusion Policy dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import move_spoon_common

move_spoon_common.configure_core()

from scripts.real_robot import build_stack_cup_dataset as core  # noqa: E402


core.__doc__ = __doc__
core.VALIDATOR_MODULE = "scripts.real_robot.validate_move_spoon_dataset"

BuildOptions = core.BuildOptions
EpisodePayload = core.EpisodePayload
load_episode_payload = core.load_episode_payload
write_dataset = core.write_dataset
build_dataset = core.build_dataset
parse_args = core.parse_args


def main(argv: Sequence[str] | None = None) -> dict:
    move_spoon_common.configure_core()
    core.__doc__ = __doc__
    core.VALIDATOR_MODULE = "scripts.real_robot.validate_move_spoon_dataset"
    return core.main(argv)


if __name__ == "__main__":
    main()
