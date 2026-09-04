#!/usr/bin/env python3
"""Validate the committed real-robot move-spoon RGB-DP dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import move_spoon_common

move_spoon_common.configure_core()

from scripts.real_robot import validate_stack_cup_dataset as core  # noqa: E402


core.__doc__ = __doc__
validate_dataset = core.validate_dataset
validate_published_dataset = core.validate_published_dataset
parse_args = core.parse_args


def main(argv: Sequence[str] | None = None) -> dict:
    move_spoon_common.configure_core()
    core.__doc__ = __doc__
    return core.main(argv)


if __name__ == "__main__":
    main()
