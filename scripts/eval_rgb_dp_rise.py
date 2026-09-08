#!/usr/bin/env python3
"""Evaluate the no-DINO RISE spectral + one-step IDQL baseline."""

from __future__ import annotations

from eval_rgb_dp_idql import evaluate
from eval_rgb_dp_idql import parse_args as parse_idql_eval_args


def parse_args(argv=None):
    args = parse_idql_eval_args(argv)
    if args.actor_source != "hybrid_dp_chunk_actor":
        raise ValueError(
            "eval_rgb_dp_rise.py requires "
            "--actor-source hybrid_dp_chunk_actor"
        )
    args.require_rise_spectral_baseline = True
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
