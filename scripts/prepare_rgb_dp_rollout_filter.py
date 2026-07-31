#!/usr/bin/env python3
"""Create and record a fixed deterministic rollout subset in an HDF5 mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def decode(items) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def deterministic_subset(demos: list[str], count: int, seed: int) -> list[str]:
    mask = np.zeros(len(demos), dtype=np.int8)
    mask[:count] = 1
    np.random.RandomState(seed).shuffle(mask)
    return [demo for demo, selected in zip(demos, mask) if selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-filter-key", required=True)
    parser.add_argument("--output-filter-key", required=True)
    parser.add_argument("--num-demos", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if args.num_demos <= 0:
        raise ValueError("num-demos must be positive")

    with h5py.File(dataset, "a") as hdf5_file:
        source_path = f"mask/{args.source_filter_key}"
        output_path = f"mask/{args.output_filter_key}"
        if source_path not in hdf5_file:
            raise KeyError(f"{source_path} not found in {dataset}")

        source_demos = sorted(decode(hdf5_file[source_path][:]))
        if args.num_demos > len(source_demos):
            raise ValueError(
                f"requested {args.num_demos} demos from {source_path}, "
                f"but only {len(source_demos)} are available"
            )

        reused = output_path in hdf5_file and not args.force
        expected_demos = deterministic_subset(source_demos, args.num_demos, args.seed)
        if reused:
            selected_demos = decode(hdf5_file[output_path][:])
            if set(selected_demos) != set(expected_demos):
                raise ValueError(
                    f"{output_path} does not match deterministic seed {args.seed}. "
                    "Use --force to replace it."
                )
        else:
            selected_demos = expected_demos
            if output_path in hdf5_file:
                del hdf5_file[output_path]
            hdf5_file[output_path] = np.asarray(selected_demos, dtype="S")
            hdf5_file.flush()

        if len(selected_demos) != args.num_demos:
            raise ValueError(
                f"{output_path} contains {len(selected_demos)} demos; "
                f"expected {args.num_demos}. Use --force to recreate it."
            )
        if len(set(selected_demos)) != len(selected_demos):
            raise ValueError(f"{output_path} contains duplicate demo IDs")
        if not set(selected_demos).issubset(source_demos):
            raise ValueError(f"{output_path} is not a subset of {source_path}")

        selected_lengths = {
            demo: int(hdf5_file[f"data/{demo}"].attrs["num_samples"])
            for demo in selected_demos
        }

    manifest = {
        "dataset": str(dataset),
        "source_filter_key": args.source_filter_key,
        "fixed_filter_key": args.output_filter_key,
        "requested_num_demos": args.num_demos,
        "source_num_demos": len(source_demos),
        "selected_num_demos": len(selected_demos),
        "selected_num_samples": sum(selected_lengths.values()),
        "selection_seed": args.seed,
        "selection_method": "sorted source IDs, shuffled binary mask with numpy RandomState",
        "selection_storage": output_path,
        "reused_existing_filter": reused,
        "physically_copied": False,
        "selected_demos": selected_demos,
        "selected_lengths": selected_lengths,
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    action = "reused" if reused else "created"
    print(
        f"[{action}] {output_path}: {len(selected_demos)}/{len(source_demos)} demos; "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
