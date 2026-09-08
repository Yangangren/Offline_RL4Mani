#!/usr/bin/env python3
"""Validate the canonical PickCup HDF5 against its episode-layout source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import h5py


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import build_episode_layout_idql_sources as core  # noqa: E402
from scripts.real_robot.pick_cup_common import (  # noqa: E402
    CONVERSION_MANIFEST_ATTR,
    DEFAULT_DATASET_DIR,
    DEFAULT_SOURCE,
    LOW_DIM_KEYS,
    RGB_KEYS,
    dataset_path,
)


PROFILE = core.TASK_PROFILES["pick_cup"]


def validate_dataset(path: Path, *, source_root: Path = DEFAULT_SOURCE) -> dict:
    path = path.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    records, identity = core.discover_episodes(
        source_root,
        kind="human",
        profile=PROFILE,
    )
    report = core.validate_output(path, kind="human", profile=PROFILE)
    with h5py.File(path, "r") as handle:
        manifest = json.loads(core._text(handle.attrs[CONVERSION_MANIFEST_ATTR]))
        if manifest.get("source_identity") != identity:
            raise core.ProposalConversionError(f"{path}: source identity changed")
        first = handle["data"][sorted(handle["data"].keys())[0]]
        image_shape = list(first["obs/main_image"].shape[1:])
        action_dim = int(first["actions"].shape[-1])
    if len(records) != PROFILE.expected_humans:
        raise AssertionError("source-backed validation lost human episodes")
    return {
        **report,
        "source_backed": True,
        "schema_signature": {
            "image_shape": image_shape,
            "action_dim": action_dim,
            "rgb_keys": list(RGB_KEYS),
            "low_dim_keys": list(LOW_DIM_KEYS),
        },
    }


def validate_published_dataset(
    output_dir: Path = DEFAULT_DATASET_DIR,
    *,
    source_root: Path = DEFAULT_SOURCE,
) -> dict:
    return validate_dataset(dataset_path(output_dir), source_root=source_root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_published_dataset(
        args.output_dir,
        source_root=args.source_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
