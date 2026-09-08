#!/usr/bin/env python3
"""RISE-style RGB Diffusion Policy + one-step IDQL post-training.

This entry point intentionally excludes DINOv2 preprocessing. It restores the
pretrained deployed Diffusion Policy EMA, adds the released RISE 512-512
post-encoder MLP with a final-layer spectral-norm penalty, and reuses the
one-step temporal IDQL critic implementation.
"""

from __future__ import annotations

import sys

import torch.distributed as dist

from train_rgb_dp_idql import parse_args as parse_idql_args
from train_rgb_dp_idql import train


DEFAULT_SPECTRAL_PENALTY_WEIGHT = 0.1
REWARD_MODE = "rise_source_binary"


def _has_option(argv: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in argv)


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(arguments, "--reward-mode"):
        arguments.extend(("--reward-mode", REWARD_MODE))
    if not _has_option(arguments, "--spectral-penalty-weight"):
        arguments.extend(
            (
                "--spectral-penalty-weight",
                str(DEFAULT_SPECTRAL_PENALTY_WEIGHT),
            )
        )
    args = parse_idql_args(arguments)
    if args.reward_mode != REWARD_MODE:
        raise ValueError(
            "train_rgb_dp_rise.py requires "
            f"--reward-mode {REWARD_MODE}, got {args.reward_mode!r}"
        )
    if float(args.spectral_penalty_weight) <= 0.0:
        raise ValueError(
            "train_rgb_dp_rise.py requires a positive "
            "--spectral-penalty-weight"
        )
    args.baseline = "rise_source_binary_spectral_one_step_idql"
    args.dinov2_augmentation = False
    return args


def main() -> None:
    args = parse_args()
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
