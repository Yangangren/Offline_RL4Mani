#!/usr/bin/env python3
"""Overlay critic-ranked action chunks on high-resolution replayed observations.

This script consumes the candidate cache written by
``visualize_chunk_candidate_selection.py``.  For every requested chunk
boundary, it restores the recorded MuJoCo state, renders a sharp camera image,
and projects all cumulative end-effector displacement proposals into the image.
The conservative-Q argmax is emphasized and annotated.

The projected curves are integrated OSC position commands. They visualize the
actor's proposed action chunks; they are not simulated future trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Set writable caches and headless rendering before importing plotting or
# simulator-adjacent modules.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/robomimic_matplotlib_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robomimic_numba_cache")
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_chunk_candidate_selection import (
    ROOT,
    TASK_SPECS,
    controls_to_position_delta,
    select_episode,
)


SELECTED_COLOR = "#E90A11"
START_COLOR = "#2F2F2F"
VIEW_LABEL_COLOR = "#0000FF"
CANDIDATE_LINESTYLE = (0, (5.0, 3.0))
SELECTED_LINESTYLE = "-"

EXTERNAL_CAMERAS = {
    "square": ("agentview",),
    "transport": ("shouldercamera0", "shouldercamera1"),
}
WRIST_CAMERAS = {
    "square": ("robot0_eye_in_hand",),
    "transport": ("robot0_eye_in_hand", "robot1_eye_in_hand"),
}
CAMERA_LABELS = {
    "agentview": "Agent view",
    "shouldercamera0": "Robot 0 shoulder view",
    "shouldercamera1": "Robot 1 shoulder view",
    "robot0_eye_in_hand": "Robot 0 wrist view",
    "robot1_eye_in_hand": "Robot 1 wrist view",
    "frontview": "Front view",
    "birdview": "Bird's-eye view",
    "sideview": "Side view",
}

CAMERA_ARM_INDEX = {
    "shouldercamera0": 0,
    "shouldercamera1": 1,
    "robot0_eye_in_hand": 0,
    "robot1_eye_in_hand": 1,
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 13.0,
            "axes.titlesize": 13.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_indices(raw_values: list[str] | None, count: int) -> list[int]:
    if not raw_values:
        return list(range(count))
    result: list[int] = []
    for raw in raw_values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                parts = [int(value) if value else None for value in token.split(":")]
                if len(parts) not in (2, 3):
                    raise ValueError(f"invalid boundary slice {token!r}")
                selected = list(range(count))[slice(*parts)]
                result.extend(selected)
            else:
                index = int(token)
                if index < 0:
                    index += count
                if not 0 <= index < count:
                    raise IndexError(
                        f"boundary index {index} is outside [0, {count - 1}]"
                    )
                result.append(index)
    return list(dict.fromkeys(result))


def camera_layout(task: str, view: str, camera: list[str] | None) -> tuple[str, ...]:
    if camera:
        return tuple(camera)
    if view == "external":
        return EXTERNAL_CAMERAS[task]
    if view == "wrist":
        return WRIST_CAMERAS[task]
    return EXTERNAL_CAMERAS[task] + WRIST_CAMERAS[task]


def cache_directory(args: argparse.Namespace, episode: str) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir.resolve()
    return (
        ROOT
        / "analysis/chunk_candidate_selection"
        / args.task
        / f"{episode}_N{args.num_candidates}_seed{args.seed}"
    )


def output_directory(args: argparse.Namespace, episode: str) -> Path:
    if args.figure_dir is not None:
        return args.figure_dir.resolve()
    return (
        ROOT
        / "figures/chunk_candidate_overlays"
        / args.task
        / f"{episode}_N{args.num_candidates}_seed{args.seed}"
        / args.view
    )


def read_cache(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = cache_dir / "candidate_chunks.npz"
    manifest_path = cache_dir / "manifest.json"
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"candidate cache is missing in {cache_dir}. Run "
            f"scripts/visualize_chunk_candidate_selection.py first."
        )
    with np.load(cache_path) as stored:
        arrays = {key: np.asarray(stored[key]) for key in stored.files}
    manifest = json.loads(manifest_path.read_text())
    if len(arrays["timesteps"]) != len(manifest["boundaries"]):
        raise ValueError("candidate cache and manifest have different boundary counts")
    return arrays, manifest


def candidate_world_paths(
    env_actions: np.ndarray,
    eef_positions: np.ndarray,
    manifest: dict[str, Any],
) -> list[np.ndarray]:
    calibration = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in manifest["position_control_calibration"].items()
    }
    paths = []
    for arm_index, (start, end) in enumerate(manifest["action_position_slices"]):
        controls = np.asarray(env_actions[..., start:end], dtype=np.float64)
        deltas = controls_to_position_delta(controls, calibration)
        offsets = np.cumsum(deltas, axis=1)
        origin = np.asarray(eef_positions[arm_index], dtype=np.float64)
        world = origin[None, None, :] + offsets
        origins = np.broadcast_to(origin, (world.shape[0], 1, 3))
        paths.append(np.concatenate((origins, world), axis=1))
    return paths


def project_world_points(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Matplotlib (x, y) pixels and a visibility mask without clipping."""
    homogeneous = np.concatenate(
        (points, np.ones(points.shape[:-1] + (1,), dtype=np.float64)),
        axis=-1,
    )
    projected = np.einsum("ij,...j->...i", world_to_camera, homogeneous)
    depth = projected[..., 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, np.nan)
    xy = projected[..., :2] / safe_depth[..., None]
    valid = (
        np.isfinite(xy).all(axis=-1)
        & (depth > 1e-6)
        & (xy[..., 0] >= 0.0)
        & (xy[..., 0] <= width - 1)
        & (xy[..., 1] >= 0.0)
        & (xy[..., 1] <= height - 1)
    )
    return xy, valid


def q_colors(q_values: np.ndarray) -> list[tuple[float, float, float, float]]:
    q_values = np.asarray(q_values, dtype=np.float64)
    low = float(q_values.min())
    high = float(q_values.max())
    if high - low < 1e-10:
        normalized = np.full_like(q_values, 0.55)
    else:
        normalized = (q_values - low) / (high - low)
    cmap = plt.get_cmap("Greens")
    return [cmap(0.36 + 0.48 * float(value)) for value in normalized]


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(valid, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def draw_projected_path(
    ax: plt.Axes,
    pixels: np.ndarray,
    valid: np.ndarray,
    color: Any,
    linewidth: float,
    alpha: float,
    linestyle: Any,
    arrow_scale: float,
    zorder: float,
) -> np.ndarray | None:
    last_endpoint = None
    for start, end in valid_runs(valid):
        segment = pixels[start:end]
        if len(segment) == 1:
            ax.scatter(
                segment[:, 0],
                segment[:, 1],
                s=linewidth * 4.0,
                color=color,
                alpha=alpha,
                zorder=zorder,
            )
            last_endpoint = segment[-1]
            continue
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
        last_endpoint = segment[-1]
        if end == len(valid):
            arrow = ax.annotate(
                "",
                xy=segment[-1],
                xytext=segment[-2],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": linewidth,
                    "alpha": alpha,
                    "mutation_scale": arrow_scale,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=zorder + 0.1,
            )
            arrow.set_clip_on(True)
    return last_endpoint


def dense_visible_samples(
    pixels: np.ndarray,
    valid: np.ndarray,
    spacing: float = 4.0,
) -> np.ndarray:
    """Sample visible polyline segments densely enough for label collision tests."""
    samples: list[np.ndarray] = []
    for start, end in valid_runs(valid):
        segment = pixels[start:end]
        if len(segment) == 1:
            samples.append(segment)
            continue
        for point_a, point_b in zip(segment[:-1], segment[1:]):
            sample_count = max(
                2,
                int(np.ceil(np.linalg.norm(point_b - point_a) / spacing)) + 1,
            )
            samples.append(np.linspace(point_a, point_b, sample_count))
    if not samples:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(samples, axis=0)


def choose_q_label_position(
    frame_shape: tuple[int, ...],
    projected_paths: list[tuple[np.ndarray, np.ndarray]],
    selected: int,
    endpoint: np.ndarray,
) -> tuple[tuple[float, float], str, str]:
    """Choose a clear callout location for the Q label, away from action curves."""
    height, width = frame_shape[:2]
    endpoint_normalized = endpoint / np.asarray((width, height), dtype=np.float64)

    def text_rectangle(
        anchor: tuple[float, float],
        horizontal_alignment: str,
        vertical_alignment: str,
    ) -> tuple[float, float, float, float]:
        text_width = 0.270
        text_height = 0.105
        x, y = anchor
        if horizontal_alignment == "left":
            left, right = x, x + text_width
        elif horizontal_alignment == "right":
            left, right = x - text_width, x
        else:
            left, right = x - text_width / 2.0, x + text_width / 2.0
        if vertical_alignment == "top":
            top, bottom = y, y + text_height
        elif vertical_alignment == "bottom":
            top, bottom = y - text_height, y
        else:
            top, bottom = y - text_height / 2.0, y + text_height / 2.0
        return left, right, top, bottom

    # Try short, local callouts first. Edge positions are retained as fallbacks
    # for unusually dense projections.
    endpoint_x, endpoint_y = endpoint_normalized
    candidate_specs = (
        ((endpoint_x + 0.055, endpoint_y), "left", "center"),
        ((endpoint_x - 0.055, endpoint_y), "right", "center"),
        ((endpoint_x, endpoint_y - 0.070), "center", "bottom"),
        ((endpoint_x, endpoint_y + 0.070), "center", "top"),
        ((endpoint_x + 0.045, endpoint_y - 0.055), "left", "bottom"),
        ((endpoint_x - 0.045, endpoint_y - 0.055), "right", "bottom"),
        ((endpoint_x + 0.045, endpoint_y + 0.055), "left", "top"),
        ((endpoint_x - 0.045, endpoint_y + 0.055), "right", "top"),
        ((0.965, 0.045), "right", "top"),
        ((0.965, 0.955), "right", "bottom"),
        ((0.035, 0.955), "left", "bottom"),
        ((0.965, 0.520), "right", "center"),
        ((0.035, 0.570), "left", "center"),
        ((0.500, 0.955), "center", "bottom"),
    )
    reserved_view_label = (0.015, 0.520, 0.020, 0.125)

    def rectangles_overlap(
        first: tuple[float, ...], second: tuple[float, ...]
    ) -> bool:
        left_a, right_a, top_a, bottom_a = first
        left_b, right_b, top_b, bottom_b = second
        return not (
            right_a < left_b
            or right_b < left_a
            or bottom_a < top_b
            or bottom_b < top_a
        )

    candidates = []
    for anchor, horizontal_alignment, vertical_alignment in candidate_specs:
        rectangle = text_rectangle(
            anchor, horizontal_alignment, vertical_alignment
        )
        if (
            min(rectangle) < 0.015
            or rectangle[1] > 0.985
            or rectangle[3] > 0.985
            or rectangles_overlap(rectangle, reserved_view_label)
        ):
            continue
        candidates.append(
            (anchor, horizontal_alignment, vertical_alignment, rectangle)
        )

    all_samples: list[np.ndarray] = []
    selected_samples: list[np.ndarray] = []
    for pixels, valid in projected_paths:
        for candidate_index in range(len(pixels)):
            samples = dense_visible_samples(
                pixels[candidate_index], valid[candidate_index]
            )
            if len(samples):
                normalized = samples / np.asarray((width, height), dtype=np.float64)
                all_samples.append(normalized)
                if candidate_index == selected:
                    selected_samples.append(normalized)

    all_points = (
        np.concatenate(all_samples, axis=0)
        if all_samples
        else np.empty((0, 2), dtype=np.float64)
    )
    selected_points = (
        np.concatenate(selected_samples, axis=0)
        if selected_samples
        else np.empty((0, 2), dtype=np.float64)
    )
    def rectangle_hits(points: np.ndarray, rectangle: tuple[float, ...]) -> int:
        if not len(points):
            return 0
        left, right, top, bottom = rectangle
        padding = 0.012
        return int(
            np.count_nonzero(
                (points[:, 0] >= left - padding)
                & (points[:, 0] <= right + padding)
                & (points[:, 1] >= top - padding)
                & (points[:, 1] <= bottom + padding)
            )
        )

    def rectangle_clearance(
        points: np.ndarray, rectangle: tuple[float, ...]
    ) -> float:
        if not len(points):
            return float("inf")
        left, right, top, bottom = rectangle
        zeros = np.zeros(len(points))
        dx = np.maximum.reduce(
            (left - points[:, 0], zeros, points[:, 0] - right)
        )
        dy = np.maximum.reduce(
            (top - points[:, 1], zeros, points[:, 1] - bottom)
        )
        return float(np.min(np.hypot(dx, dy)))

    def candidate_score(candidate: tuple[Any, ...]) -> tuple[float, ...]:
        anchor, _, _, rectangle = candidate
        selected_hits = rectangle_hits(selected_points, rectangle)
        all_hits = rectangle_hits(all_points, rectangle)
        selected_clearance = rectangle_clearance(selected_points, rectangle)
        all_clearance = rectangle_clearance(all_points, rectangle)
        connector_length = float(
            np.linalg.norm(endpoint_normalized - np.asarray(anchor))
        )
        # Lexicographic scoring first avoids the selected red curve, then all
        # other chunks, then favors greater clearance and a shorter leader.
        return (
            float(selected_hits),
            float(all_hits),
            connector_length,
            -selected_clearance,
            -all_clearance,
        )

    anchor, horizontal_alignment, vertical_alignment, _ = min(
        candidates, key=candidate_score
    )
    position = (anchor[0] * width, anchor[1] * height)
    return position, horizontal_alignment, vertical_alignment


def draw_camera_panel(
    ax: plt.Axes,
    frame: np.ndarray,
    projected_paths: list[tuple[np.ndarray, np.ndarray]],
    q_values: np.ndarray,
    selected: int,
    camera_name: str,
    show_q_label: bool,
) -> None:
    ax.imshow(frame, interpolation="none")
    ax.set_xlim(-0.5, frame.shape[1] - 0.5)
    ax.set_ylim(frame.shape[0] - 0.5, -0.5)
    ax.set_axis_off()
    colors = q_colors(q_values)

    for candidate in np.argsort(q_values):
        if int(candidate) == selected:
            continue
        for pixels, valid in projected_paths:
            draw_projected_path(
                ax=ax,
                pixels=pixels[candidate],
                valid=valid[candidate],
                color=colors[candidate],
                linewidth=2.0,
                alpha=0.68,
                linestyle=CANDIDATE_LINESTYLE,
                arrow_scale=12.0,
                zorder=3.0 + float(candidate) * 0.001,
            )

    selected_endpoints = []
    for pixels, valid in projected_paths:
        endpoint = draw_projected_path(
            ax=ax,
            pixels=pixels[selected],
            valid=valid[selected],
            color=SELECTED_COLOR,
            linewidth=4.2,
            alpha=0.98,
            linestyle=SELECTED_LINESTYLE,
            arrow_scale=18.0,
            zorder=6.0,
        )
        if valid[selected, 0]:
            start = pixels[selected, 0]
            ax.scatter(
                start[0],
                start[1],
                marker="o",
                s=52,
                facecolor=START_COLOR,
                edgecolor="none",
                zorder=7.0,
            )
        if endpoint is not None:
            selected_endpoints.append(endpoint)

    if show_q_label and selected_endpoints:
        endpoint = selected_endpoints[0]
        label = f"Q={q_values[selected]:.3f}"
        label_position, horizontal_alignment, vertical_alignment = (
            choose_q_label_position(
                frame_shape=frame.shape,
                projected_paths=projected_paths,
                selected=selected,
                endpoint=endpoint,
            )
        )
        annotation = ax.annotate(
            label,
            xy=endpoint,
            xytext=label_position,
            textcoords="data",
            ha=horizontal_alignment,
            va=vertical_alignment,
            color=SELECTED_COLOR,
            fontsize=19.0,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": "#202020",
                "lw": 1.8,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            annotation_clip=True,
            zorder=8.0,
        )
        annotation.set_clip_on(True)

    ax.text(
        0.025,
        0.965,
        CAMERA_LABELS.get(camera_name, camera_name),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=VIEW_LABEL_COLOR,
        fontsize=19.0,
        fontweight="bold",
        zorder=8.0,
    )


def plot_boundary(
    task: str,
    cameras: tuple[str, ...],
    frames: list[np.ndarray],
    transforms: list[np.ndarray],
    world_paths: list[np.ndarray],
    q_values: np.ndarray,
    selected: int,
    timestep: int,
    boundary_index: int,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    panel_count = len(cameras)
    if panel_count <= 2:
        rows, columns = 1, panel_count
    else:
        columns = 2
        rows = int(np.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 6.15 * rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for camera_index, (camera, frame, transform) in enumerate(
        zip(cameras, frames, transforms)
    ):
        if len(world_paths) > 1 and camera in CAMERA_ARM_INDEX:
            camera_world_paths = [world_paths[CAMERA_ARM_INDEX[camera]]]
        else:
            camera_world_paths = world_paths
        projected = [
            project_world_points(
                points=paths,
                world_to_camera=transform,
                height=frame.shape[0],
                width=frame.shape[1],
            )
            for paths in camera_world_paths
        ]
        draw_camera_panel(
            ax=flat_axes[camera_index],
            frame=frame,
            projected_paths=projected,
            q_values=q_values,
            selected=selected,
            camera_name=camera,
            show_q_label=camera_index == 0,
        )
    for unused in flat_axes[panel_count:]:
        unused.set_visible(False)

    figure.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        wspace=0.012,
        hspace=0.012,
    )
    stem = f"boundary_{boundary_index:03d}_t{timestep:04d}_{task}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def create_replay_environment(dataset_path: Path):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    dummy_spec = {"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASK_SPECS))
    parser.add_argument("--episode", default=None)
    parser.add_argument(
        "--episode-class",
        default="success",
        choices=("success", "failure"),
    )
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--view",
        choices=("external", "wrist", "comparison"),
        default="external",
        help="External is the recommended paper view; comparison shows both.",
    )
    parser.add_argument(
        "--camera",
        nargs="+",
        default=None,
        help="Optional explicit MuJoCo camera name(s), overriding --view.",
    )
    parser.add_argument(
        "--boundary-indices",
        nargs="+",
        default=None,
        help="Indices and/or Python-style slices, e.g. 0 4 10:15 or 0,4,8.",
    )
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_size < 256:
        raise ValueError("--render-size must be at least 256 pixels")
    spec = TASK_SPECS[args.task]
    episode, episode_length, _ = select_episode(
        dataset_path=spec.dataset,
        episode_class=args.episode_class,
        requested_episode=args.episode,
    )
    cache_dir = cache_directory(args, episode)
    arrays, manifest = read_cache(cache_dir)
    cached_episode = manifest.get("signature", {}).get("episode")
    if cached_episode != episode:
        raise ValueError(
            f"cache contains episode {cached_episode!r}, requested {episode!r}"
        )
    if arrays["q"].shape[1] != args.num_candidates:
        raise ValueError(
            f"cache has {arrays['q'].shape[1]} candidates, requested "
            f"{args.num_candidates}"
        )
    cameras = camera_layout(args.task, args.view, args.camera)
    boundary_indices = parse_indices(args.boundary_indices, len(arrays["timesteps"]))
    output_dir = output_directory(args, episode)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    print(
        f"Replaying {spec.title} {episode} ({episode_length} steps) at "
        f"{args.render_size}x{args.render_size}; cameras={list(cameras)}",
        flush=True,
    )
    env = create_replay_environment(spec.dataset)
    records = []
    try:
        with h5py.File(spec.dataset, "r") as dataset:
            demo = dataset[f"data/{episode}"]
            initial_state: dict[str, Any] = {
                "states": np.asarray(demo["states"][0]),
            }
            if "model_file" in demo.attrs:
                initial_state["model"] = demo.attrs["model_file"]
            if "ep_meta" in demo.attrs:
                initial_state["ep_meta"] = demo.attrs["ep_meta"]
            env.reset_to(initial_state)

            for boundary_index in boundary_indices:
                timestep = int(arrays["timesteps"][boundary_index])
                env.reset_to({"states": np.asarray(demo["states"][timestep])})
                frames = [
                    env.render(
                        mode="rgb_array",
                        height=args.render_size,
                        width=args.render_size,
                        camera_name=camera,
                    )
                    for camera in cameras
                ]
                transforms = [
                    env.get_camera_transform_matrix(
                        camera_name=camera,
                        camera_height=args.render_size,
                        camera_width=args.render_size,
                    )
                    for camera in cameras
                ]
                world_paths = candidate_world_paths(
                    env_actions=arrays["env_actions"][boundary_index],
                    eef_positions=arrays["eef_positions"][boundary_index],
                    manifest=manifest,
                )
                q_values = np.asarray(arrays["q"][boundary_index])
                selected = int(arrays["selected"][boundary_index])
                if selected != int(np.argmax(q_values)):
                    raise ValueError("cached selected index is not the Q argmax")
                png_path, pdf_path = plot_boundary(
                    task=args.task,
                    cameras=cameras,
                    frames=frames,
                    transforms=transforms,
                    world_paths=world_paths,
                    q_values=q_values,
                    selected=selected,
                    timestep=timestep,
                    boundary_index=boundary_index,
                    output_dir=output_dir,
                    dpi=args.dpi,
                )
                records.append(
                    {
                        "boundary_index": int(boundary_index),
                        "timestep": timestep,
                        "selected_candidate": selected + 1,
                        "selected_q": float(q_values[selected]),
                        "png": str(png_path.resolve()),
                        "pdf": str(pdf_path.resolve()),
                    }
                )
                print(
                    f"Saved boundary {boundary_index:03d} (t={timestep:04d}, "
                    f"a{selected + 1}, Q={q_values[selected]:.4f})",
                    flush=True,
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    output_manifest = {
        "task": args.task,
        "episode": episode,
        "dataset": str(spec.dataset.resolve()),
        "candidate_cache": str(cache_dir.resolve()),
        "view": args.view,
        "cameras": list(cameras),
        "render_size": args.render_size,
        "interpretation": (
            "Curves are cumulative OSC end-effector displacement commands projected "
            "into a high-resolution replay of the recorded state; they are not "
            "simulated future trajectories. Q_min denotes min(Q1, Q2)."
        ),
        "figures": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(f"Saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
