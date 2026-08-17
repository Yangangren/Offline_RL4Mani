"""Synthetic Real4D fixture shared by pick-cup pipeline tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


RAW_GRIPPER_EVENTS = np.asarray(
    [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    dtype=np.float32,
)
EXPECTED_DENSE_ACTION = np.asarray(
    [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
EXPECTED_PRE_ACTION_OBS = np.asarray(
    [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
    dtype=np.float32,
)
EXPECTED_FRAME_POSITIONS = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2])


def _write_episode(
    root: Path,
    *,
    number: int,
    base_time: float,
    close_xy: tuple[float, float],
) -> tuple[str, str]:
    run_id = f"fixture_{number:03d}"
    relative = f"episodes/episode_{number:03d}__{run_id}"
    episode = root / relative
    frames_dir = episode / "native/frames"
    frames_dir.mkdir(parents=True)

    samples = []
    for index, gripper_event in enumerate(RAW_GRIPPER_EVENTS):
        position = [
            close_xy[0] + 0.001 * (index - 2),
            close_xy[1] + 0.002 * (index - 2),
            0.22 + 0.001 * index,
        ]
        pose = [*position, 0.0, 0.0, 0.0, 1.0]
        buttons = [0, 0]
        if gripper_event < 0:
            buttons = [1, 0]
        elif gripper_event > 0:
            buttons = [0, 1]
        samples.append(
            {
                "action": [0.1, -0.2, 0.3, 0.0, 0.05, -0.1, float(gripper_event)],
                "raw_action": [0.1, -0.2, 0.3, 0.0, 0.05, -0.1],
                "before_pose": pose,
                "after_pose": pose,
                "buttons": buttons,
                "source_index": index,
                "source_time": base_time + 0.05 * index,
                "target_time": base_time + 0.05 * index,
                "step": index,
                "offset_ms": 0.0,
                "clip": [],
            }
        )
    (episode / "actions.json").write_text(
        json.dumps({"metrics": {"target_hz": 20.0}, "samples": samples})
    )

    frame_rows = []
    for index in range(3):
        nominal_time = base_time + 0.2 * index
        paths = {}
        for camera, color in (
            ("main", (10 + index, 20, 30)),
            ("wrist", (40 + index, 50, 60)),
        ):
            filename = f"{camera}_rgb_{index:03d}.png"
            relative_image = f"native/frames/{filename}"
            Image.fromarray(
                np.full((8, 12, 3), color, dtype=np.uint8),
                mode="RGB",
            ).save(episode / relative_image)
            paths[camera] = relative_image
        frame_rows.append(
            {
                "index": index,
                "target_time": nominal_time,
                "target_stamp_ns": int(round(nominal_time * 1e9)),
                "streams": {
                    "main_rgb": {
                        "path": paths["main"],
                        "encoding": "rgb8",
                        "height": 8,
                        "width": 12,
                        "header_stamp_ns": int(round((nominal_time - 0.02) * 1e9)),
                    },
                    "wrist_rgb": {
                        "path": paths["wrist"],
                        "encoding": "rgb8",
                        "height": 8,
                        "width": 12,
                        "header_stamp_ns": int(round((nominal_time - 0.01) * 1e9)),
                    },
                },
            }
        )
    (episode / "frames.json").write_text(json.dumps({"frames": frame_rows}))
    (episode / "contract.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "actions": {"hz": 20.0, "num_actions": len(samples)},
                "video": {"hz": 5.0, "num_frames": len(frame_rows)},
            }
        )
    )
    (episode / "qa.json").write_text(json.dumps({"status": "PASS"}))
    return run_id, relative


def make_source_fixture(root: Path) -> Path:
    source = root / "raw_pick_cup"
    source.mkdir(parents=True)
    episode_specs = (
        (2, 100.0, (0.52, 0.02), "PASS", True),
        (3, 200.0, (0.58, 0.09), "WARN", True),
        (4, 250.0, (0.55, 0.05), "WARN", False),
        (51, 300.0, (0.31, -0.16), "PASS", True),
        (52, 400.0, (0.64, 0.17), "WARN", True),
    )
    rows = []
    for number, base_time, close_xy, qa_status, eligible in episode_specs:
        if eligible:
            run_id, directory = _write_episode(
                source,
                number=number,
                base_time=base_time,
                close_xy=close_xy,
            )
        else:
            run_id = f"fixture_{number:03d}"
            directory = f"episodes/episode_{number:03d}__not_copied"
        rows.append(
            {
                "episode_number": number,
                "run_id": run_id,
                "qa_status": qa_status,
                "training_eligible_43qa": str(eligible),
                "review_reason": "excluded fixture" if not eligible else "",
                "action_samples_20hz": 9,
                "synchronized_frames_5hz": 3,
                "duration_sec": 0.4,
                "file_count": 0,
                "bytes": 0,
                "directory": directory,
            }
        )

    fields = list(rows[0])
    with (source / "episode_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (source / "manifest.json").write_text(
        json.dumps({"episode_count": 5, "training_eligible_count": 4})
    )
    (source / "metadata_checksums.sha256").write_text("fixture\n")
    (source / "FULL_SEQUENCE_PACKAGE.txt").write_text("synthetic fixture\n")
    return source

