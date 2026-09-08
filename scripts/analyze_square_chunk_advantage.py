#!/usr/bin/env python3
"""Run the success-versus-failure chunk-advantage analysis for Square.

The selected critic was trained with a 406-success / 94-failure epoch-200
dataset that is not present in this workspace. The local default below is the
available epoch-190 100-success / 50-failure rollout dataset, so this run is a
cross-dataset diagnostic. All paths remain overridable from the command line.
"""

from __future__ import annotations

import sys

from analyze_transport_chunk_advantage import ROOT, main


DEFAULTS = (
    ("--task", "square"),
    (
        "--checkpoint",
        ROOT
        / "trained_models/square_rgb_dp/chunk_idql"
        / "200demo_406success_94failure_h8_rise_v2_obs2_film_dense2468_"
        "human_success_condition_terminal_success/models/model_epoch_50.pt",
    ),
    (
        "--dp-checkpoint",
        ROOT
        / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1"
        / "models/model_epoch_200.pth",
    ),
    (
        "--dataset",
        ROOT
        / "datasets/square/idql"
        / "square_rgb_dp_idql_200demo_100success_50failure.hdf5",
    ),
    (
        "--output-dir",
        ROOT
        / "analysis/square_chunk_advantage"
        / "epoch50_epoch190_100success_50failure",
    ),
    ("--figure-dir", ROOT / "figures"),
)


def inject_square_defaults() -> None:
    arguments = sys.argv[1:]
    injected: list[str] = []
    for option, value in DEFAULTS:
        supplied = any(
            argument == option or argument.startswith(f"{option}=")
            for argument in arguments
        )
        if not supplied:
            injected.extend((option, str(value)))
    sys.argv[1:1] = injected


if __name__ == "__main__":
    inject_square_defaults()
    main()
