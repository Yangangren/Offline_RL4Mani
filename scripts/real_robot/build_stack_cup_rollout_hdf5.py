#!/usr/bin/env python3
"""Convert the audited stack-cup epoch-200 20 Hz rollouts to robomimic HDF5.

The binary reconstruction and validation logic is shared with the pick-cup
converter. This module supplies the stack-cup identities, corpus contract, and
the two exact adjacent state-timestamp repeats found in the raw audit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

import h5py

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import build_pick_cup_rollout_hdf5 as core


DEFAULT_SOURCE = Path("/home/ryan/datasets/stack_cup/rollout")
DEFAULT_HUMAN_DATASET = ROOT / "datasets/real_robot/stack_cup/stack_cup_rgb.hdf5"
DEFAULT_OUTPUT = (
    ROOT
    / "datasets/real_robot/stack_cup/idql/stack_cup_epoch200_20hz_rollouts.hdf5"
)


def _stack_cup_env_args() -> dict:
    if not DEFAULT_HUMAN_DATASET.is_file():
        raise FileNotFoundError(
            "stack-cup human dataset is required to inherit exact env_args: "
            f"{DEFAULT_HUMAN_DATASET}"
        )
    with h5py.File(DEFAULT_HUMAN_DATASET, "r") as dataset:
        raw = dataset["data"].attrs["env_args"]
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def configure_core() -> None:
    """Install the immutable stack-cup corpus profile in the shared converter."""

    core.SCHEMA = "real4d.robomimic.stack_cup_rollouts.v1"
    core.CONVERSION_VERSION = "stack_cup_epoch200_20hz_rollouts_v1"
    core.DEFAULT_SOURCE = DEFAULT_SOURCE
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.EXPECTED_CHECKPOINT_SHA256 = (
        "898ddfd8a71dca121df624f16db2359280e5fd02fabe05eaee186ab428ce4a66"
    )
    core.EXPECTED_CONTRACT_SHA256 = (
        "3d9b7a33ded0594a97ff1dd54d9f7240d0fe27a3738dd14a7506653c5f724a08"
    )
    core.EPISODE_RE = re.compile(
        r"^stack_cup_20hz_real_(?P<day>\d{8})_(?P<clock>\d{6})$"
    )
    core.ENV_ARGS = _stack_cup_env_args()
    core.FULL_DEFAULT_EXPECTED_ACTIONS = 600
    core.FULL_DEFAULT_SUCCESS_VALID_COUNT = 6
    core.FULL_DEFAULT_FAILURE_VALID_COUNT = 4
    core.EXPECTED_FULL_OUTCOME_COUNTS = {"success": 32, "failure": 18}
    core.EXPECTED_FULL_SAMPLES = 29_826
    core.EXPECTED_FULL_DROPPED_PREFIX_ACTIONS = 174
    core.ALLOWED_REPEATED_STATE_TIMESTAMP_EDGES = {
        "stack_cup_20hz_real_20260823_180022": (276,),
        "stack_cup_20hz_real_20260823_195540": (6,),
    }
    # One episode required two logger recovery chunks before both cameras were
    # stable. Rows 0..14 are trimmed; row 15 and every later row pass the age
    # bound, and the missing startup frames are digest-verified golden inputs.
    core.STARTUP_PREFIX_TRIM_ACTION_LIMIT = 16


def parse_args(argv: Sequence[str] | None = None) -> core.BuildOptions:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compression", choices=("gzip", "lzf", "none"), default="gzip"
    )
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    return core.BuildOptions(
        source_root=args.source_root,
        output=args.output,
        compression=args.compression,
        split_seed=args.split_seed,
        success_valid_count=6,
        failure_valid_count=4,
        expected_actions=600,
        action_horizon=8,
        episode_limit=args.episode_limit,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import json

    configure_core()
    report = core.build_dataset(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
