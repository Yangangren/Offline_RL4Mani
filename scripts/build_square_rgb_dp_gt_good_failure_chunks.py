#!/usr/bin/env python3
"""Build GT-progress good chunks from failed Square RGB-DP rollouts.

This ablation uses privileged simulator observations only. It does not use a
learned risk model. Each retained output demo contains one context observation
at ``max(0, boundary - 1)`` followed by ``prediction_horizon`` real target
actions from the failed rollout. Train robomimic on this file with
``demo_start_only=True`` and ``sample_start_offset=1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from build_square_rgb_dp_low_risk_failure_chunks import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET_PEG_XY,
    audit_source_match,
    clip_action_dataset,
    copy_fixed_window,
    decode,
    progress_metrics_for_episode,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "rollouts/square_rgb_dp/epoch190_collection"
    / "square_rgb_dp_gt_good_failure_chunks.hdf5"
)


def stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def candidate_steps(num_samples: int, args: argparse.Namespace) -> np.ndarray:
    latest = int(num_samples) - int(args.prediction_horizon)
    if latest < 0:
        return np.zeros(0, dtype=np.int64)
    lo = max(0, int(args.min_start_step))
    hi = latest
    if args.max_start_step is not None:
        hi = min(hi, int(args.max_start_step))
    if args.max_start_fraction is not None:
        hi = min(hi, int(np.floor(float(args.max_start_fraction) * num_samples)))
    if hi < lo:
        return np.zeros(0, dtype=np.int64)
    return np.arange(lo, hi + 1, int(args.stride), dtype=np.int64)


def choose_gt_good_chunks(
    *,
    steps: np.ndarray,
    metrics: dict[str, np.ndarray],
    max_chunks: int,
    minimum_spacing: int,
    prefer: str,
    min_eef_approach_gain: float,
) -> list[int]:
    eligible_mask = metrics["progress_mask"].copy()
    if np.isfinite(min_eef_approach_gain):
        eligible_mask &= metrics["eef_approach_gain"] >= float(min_eef_approach_gain)
    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) == 0:
        return []

    if prefer == "progress":
        order = eligible[np.lexsort((steps[eligible], -metrics["progress_score"][eligible]))]
    elif prefer == "earliest":
        order = eligible[np.argsort(steps[eligible])]
    elif prefer == "latest":
        order = eligible[np.argsort(-steps[eligible])]
    else:
        raise ValueError(f"unknown prefer mode: {prefer}")

    selected: list[int] = []
    for index in order:
        step = int(steps[index])
        if all(abs(step - old_step) >= minimum_spacing for old_step in selected):
            selected.append(step)
        if len(selected) >= max_chunks:
            break
    return sorted(selected)


def build(args: argparse.Namespace) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    source_failure_samples = 0
    retained_source_rollouts: set[str] = set()
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        failure_keys = decode(source["mask/failure"][:])
        data = output.create_group("data")
        output_keys: list[str] = []

        for source_index, source_key in enumerate(failure_keys):
            source_group = source[f"data/{source_key}"]
            num_samples = int(source_group.attrs["num_samples"])
            source_failure_samples += num_samples
            steps = candidate_steps(num_samples, args)
            if len(steps) == 0:
                continue
            metrics = progress_metrics_for_episode(
                source_group=source_group,
                steps=steps,
                prediction_horizon=args.prediction_horizon,
                target_peg_xy=np.asarray(args.target_peg_xy, dtype=np.float64),
                args=args,
            )
            selected = choose_gt_good_chunks(
                steps=steps,
                metrics=metrics,
                max_chunks=int(args.max_chunks_per_failure),
                minimum_spacing=int(args.minimum_spacing),
                prefer=args.prefer,
                min_eef_approach_gain=float(args.min_eef_approach_gain),
            )
            step_to_local = {int(step): i for i, step in enumerate(steps)}
            for boundary in selected:
                local = step_to_local[int(boundary)]
                output_key = f"demo_{len(output_keys)}"
                output_group = data.create_group(output_key)
                copy_fixed_window(
                    source_group,
                    output_group,
                    int(boundary),
                    int(args.prediction_horizon),
                )
                clip_action_dataset(output_group)
                for attr_key, attr_value in source_group.attrs.items():
                    if attr_key != "num_samples":
                        output_group.attrs[attr_key] = attr_value
                output_group.attrs["num_samples"] = int(args.prediction_horizon) + 1
                output_group.attrs["source_demo"] = source_key
                output_group.attrs["source_target_start"] = int(boundary)
                output_group.attrs["sample_start_offset"] = 1
                output_group.attrs["segment_type"] = "square_gt_good_failure_chunk"
                output_group.attrs["privileged_peg_xy_progress"] = metrics["peg_xy_progress"][local]
                output_group.attrs["privileged_nut_displacement"] = metrics["nut_disp_end"][local]
                output_group.attrs["privileged_nut_z_gain"] = metrics["nut_z_gain"][local]
                output_group.attrs["privileged_eef_approach_gain"] = metrics["eef_approach_gain"][local]
                output_group.attrs["privileged_progress_score"] = metrics["progress_score"][local]

                output_keys.append(output_key)
                retained_source_rollouts.add(source_key)
                records.append(
                    {
                        "output_demo": output_key,
                        "source_demo": source_key,
                        "decision_boundary": int(boundary),
                        "target_end_exclusive": int(boundary + args.prediction_horizon),
                        "source_num_samples": int(num_samples),
                        "privileged_peg_xy_progress": float(metrics["peg_xy_progress"][local]),
                        "privileged_nut_displacement": float(metrics["nut_disp_end"][local]),
                        "privileged_nut_z_gain": float(metrics["nut_z_gain"][local]),
                        "privileged_eef_approach_gain": float(metrics["eef_approach_gain"][local]),
                        "privileged_progress_score": float(metrics["progress_score"][local]),
                    }
                )

            if (source_index + 1) % 25 == 0:
                print(
                    f"processed {source_index + 1}/{len(failure_keys)} failures; "
                    f"retained {len(records)} chunks",
                    flush=True,
                )

        if not records:
            raise RuntimeError("GT-good failure filter retained no chunks")
        data.attrs["env_args"] = source["data"].attrs["env_args"]
        data.attrs["total"] = len(output_keys) * (int(args.prediction_horizon) + 1)
        mask = output.create_group("mask")
        encoded_keys = np.asarray(output_keys, dtype="S")
        mask["all"] = encoded_keys
        mask["gt_good_failure"] = encoded_keys
        audit_source_match(source, output, records, int(args.prediction_horizon))

    peg = np.asarray([r["privileged_peg_xy_progress"] for r in records], dtype=np.float64)
    disp = np.asarray([r["privileged_nut_displacement"] for r in records], dtype=np.float64)
    z_gain = np.asarray([r["privileged_nut_z_gain"] for r in records], dtype=np.float64)
    score = np.asarray([r["privileged_progress_score"] for r in records], dtype=np.float64)
    starts = np.asarray([r["decision_boundary"] for r in records], dtype=np.int64)
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "source_failure_rollouts": int(len(failure_keys)),
        "source_failure_samples": int(source_failure_samples),
        "retained_chunks": int(len(records)),
        "retained_source_rollouts": int(len(retained_source_rollouts)),
        "target_actions_per_chunk": int(args.prediction_horizon),
        "stored_frames_per_chunk": int(args.prediction_horizon) + 1,
        "contains_padded_target_actions": False,
        "training_dataset_hints": {
            "filter_key": "gt_good_failure",
            "demo_start_only": True,
            "sample_start_offset": 1,
            "treat_as_positive_imitation": True,
        },
        "selection": {
            "score": "privileged Square task progress only",
            "prefer": args.prefer,
            "stride": int(args.stride),
            "max_chunks_per_failure": int(args.max_chunks_per_failure),
            "minimum_spacing": int(args.minimum_spacing),
            "prediction_horizon": int(args.prediction_horizon),
            "target_peg_xy": list(args.target_peg_xy),
            "min_start_step": int(args.min_start_step),
            "max_start_step": None if args.max_start_step is None else int(args.max_start_step),
            "max_start_fraction": args.max_start_fraction,
            "min_peg_xy_progress": float(args.min_peg_xy_progress),
            "min_nut_displacement": float(args.min_nut_displacement),
            "min_nut_z_gain": float(args.min_nut_z_gain),
            "min_eef_approach_gain": float(args.min_eef_approach_gain),
            "peg_progress_weight": float(args.peg_progress_weight),
            "z_gain_weight": float(args.z_gain_weight),
            "nut_displacement_weight": float(args.nut_displacement_weight),
            "eef_approach_weight": float(args.eef_approach_weight),
        },
        "privileged_progress_stats": {
            "start_step": stats(starts),
            "peg_xy_progress": stats(peg),
            "nut_displacement": stats(disp),
            "nut_z_gain": stats(z_gain),
            "progress_score": stats(score),
        },
        "chunks": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "chunks"}, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-start-step", type=int, default=0)
    parser.add_argument("--max-start-step", type=int, default=300)
    parser.add_argument("--max-start-fraction", type=float, default=None)
    parser.add_argument("--max-chunks-per-failure", type=int, default=9999)
    parser.add_argument("--minimum-spacing", type=int, default=1)
    parser.add_argument("--prefer", choices=("progress", "earliest", "latest"), default="progress")
    parser.add_argument("--target-peg-xy", type=float, nargs=2, default=DEFAULT_TARGET_PEG_XY)
    parser.add_argument("--min-peg-xy-progress", type=float, default=0.02)
    parser.add_argument("--min-nut-displacement", type=float, default=0.02)
    parser.add_argument("--min-nut-z-gain", type=float, default=-0.01)
    parser.add_argument("--min-eef-approach-gain", type=float, default=-np.inf)
    parser.add_argument("--peg-progress-weight", type=float, default=2.0)
    parser.add_argument("--z-gain-weight", type=float, default=1.0)
    parser.add_argument("--nut-displacement-weight", type=float, default=0.25)
    parser.add_argument("--eef-approach-weight", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in ("source", "output"):
        setattr(args, key, getattr(args, key).resolve())
    build(args)


if __name__ == "__main__":
    main()
