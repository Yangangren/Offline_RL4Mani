#!/usr/bin/env python3
"""Build the canonical PickCup human HDF5 from the episode-layout package."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_robot import build_episode_layout_idql_sources as core  # noqa: E402
from scripts.real_robot.pick_cup_common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MAX_IMAGE_AGE_SEC,
    DEFAULT_SOURCE,
    dataset_path,
)


@dataclass(frozen=True)
class BuildOptions:
    source_root: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_DATASET_DIR
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    max_image_age_sec: float = DEFAULT_MAX_IMAGE_AGE_SEC
    compression: str = "gzip"
    overwrite: bool = False
    validate_only: bool = False


def build_dataset(options: BuildOptions) -> dict:
    if (options.image_height, options.image_width) != (
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
    ):
        raise ValueError("PickCup episode-layout RGB is fixed at 96x128")
    if not np.isclose(
        options.max_image_age_sec,
        DEFAULT_MAX_IMAGE_AGE_SEC,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"PickCup uses the fixed {DEFAULT_MAX_IMAGE_AGE_SEC:.1f}s "
            "human image-age contract"
        )
    return core.build_dataset(
        core.BuildOptions(
            task="pick_cup",
            human_source_root=options.source_root,
            human_output=dataset_path(options.output_dir),
            compression=options.compression,
            overwrite=options.overwrite,
            validate_only=options.validate_only,
            source_kind="human",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> BuildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    return BuildOptions(
        source_root=args.source_root,
        output_dir=args.output_dir,
        compression=args.compression,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build_dataset(parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
