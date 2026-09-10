#!/usr/bin/env python3
"""Visualize conditioned-DP action chunks and conservative critic selection.

For one recorded rollout episode, the script reproduces the N-candidate
selection performed at every execution-chunk boundary. Expensive actor and
critic inference is cached separately from plotting. Each boundary is exported
as an independent PDF and PNG so representative examples can be selected for a
paper without composing Square and Transport into a single figure.

The plotted paths integrate the OSC position commands using the controller's
input/output scaling. They are action-chunk proposals, not simulated future
end-effector trajectories.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from robomimic.utils import torch_utils as TorchUtils

from eval_rgb_dp_idql import load_policy, parse_args as parse_eval_args


ROOT = Path(__file__).resolve().parents[1]
SELECTED_COLOR = "#D55E00"
OTHER_COLOR = "#8C8C8C"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    idql_checkpoint: Path
    dp_checkpoint: Path
    dataset: Path
    camera_keys: tuple[str, ...]
    camera_labels: tuple[str, ...]
    eef_position_keys: tuple[str, ...]
    action_position_slices: tuple[tuple[int, int], ...]
    arm_titles: tuple[str, ...]


TASK_SPECS = {
    "square": TaskSpec(
        name="square",
        title="Square",
        idql_checkpoint=(
            ROOT
            / "trained_models/square_rgb_dp/chunk_idql"
            / "200demo_406success_94failure_h8_rise_v2_obs2_film_dense2468_"
            "human_success_condition_terminal_success/models/model_epoch_50.pt"
        ),
        dp_checkpoint=(
            ROOT
            / "trained_models/square_rgb_dp/square_ph_rgb_dp_official_s1"
            / "models/model_epoch_200.pth"
        ),
        dataset=(
            ROOT
            / "datasets/square/idql"
            / "square_rgb_dp_idql_200demo_100success_50failure.hdf5"
        ),
        camera_keys=("agentview_image",),
        camera_labels=("Agent view",),
        eef_position_keys=("robot0_eef_pos",),
        action_position_slices=((0, 3),),
        arm_titles=("Robot arm",),
    ),
    "transport": TaskSpec(
        name="transport",
        title="Transport",
        idql_checkpoint=(
            ROOT
            / "trained_models/transport_rgb_dp/chunk_idql"
            / "200demo_422success_78failure_h8_rise_v2_obs2_film_dense48_"
            "human_success_condi_terminal_success_reward/models/model_epoch_50.pt"
        ),
        dp_checkpoint=(
            ROOT
            / "trained_models/transport_rgb_dp/transport_ph_rgb_dp_official_s1"
            / "models/model_epoch_200.pth"
        ),
        dataset=(
            ROOT
            / "datasets/transport/idql"
            / "transport_rgb_dp_idql_200demo_100success_50failure.hdf5"
        ),
        camera_keys=("shouldercamera0_image", "shouldercamera1_image"),
        camera_labels=("Robot 0 view", "Robot 1 view"),
        eef_position_keys=("robot0_eef_pos", "robot1_eef_pos"),
        action_position_slices=((0, 3), (7, 10)),
        arm_titles=("Robot 0", "Robot 1"),
    ),
}


def decode_names(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def select_episode(
    dataset_path: Path,
    episode_class: str,
    requested_episode: str | None,
) -> tuple[str, int, float]:
    mask_name = f"non_expert_{episode_class}"
    with h5py.File(dataset_path, "r") as dataset:
        mask_path = f"mask/{mask_name}"
        if mask_path not in dataset:
            raise KeyError(f"dataset does not contain {mask_path}")
        episodes = decode_names(dataset[mask_path][:])
        if not episodes:
            raise RuntimeError(f"dataset mask {mask_name!r} is empty")
        lengths = np.asarray(
            [int(dataset[f"data/{episode}"].attrs["num_samples"]) for episode in episodes],
            dtype=np.int64,
        )
        median_length = float(np.median(lengths))
        if requested_episode is not None:
            if requested_episode not in episodes:
                raise ValueError(
                    f"episode {requested_episode!r} is not in mask {mask_name!r}"
                )
            selected = requested_episode
            selected_length = int(
                dataset[f"data/{selected}"].attrs["num_samples"]
            )
        else:
            index = min(
                range(len(episodes)),
                key=lambda item: (
                    abs(int(lengths[item]) - median_length),
                    episodes[item],
                ),
            )
            selected = episodes[index]
            selected_length = int(lengths[index])
    return selected, selected_length, median_length


def output_directories(
    args: argparse.Namespace,
    episode: str,
) -> tuple[Path, Path]:
    run_name = f"{episode}_N{args.num_candidates}_seed{args.seed}"
    cache_dir = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "analysis/chunk_candidate_selection" / args.task / run_name
    )
    figure_dir = (
        args.figure_dir
        if args.figure_dir is not None
        else ROOT / "figures/chunk_candidate_selection" / args.task / run_name
    )
    return cache_dir.resolve(), figure_dir.resolve()


def position_control_calibration(dataset: h5py.File) -> dict[str, np.ndarray]:
    env_args_raw = dataset["data"].attrs.get("env_args")
    if env_args_raw is None:
        raise KeyError("dataset is missing data.attrs['env_args']")
    if isinstance(env_args_raw, bytes):
        env_args_raw = env_args_raw.decode("utf-8")
    env_args = json.loads(str(env_args_raw))
    controller = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    input_min = np.broadcast_to(
        np.asarray(controller["input_min"], dtype=np.float64),
        (3,),
    ).copy()
    input_max = np.broadcast_to(
        np.asarray(controller["input_max"], dtype=np.float64),
        (3,),
    ).copy()
    output_min = np.asarray(controller["output_min"][:3], dtype=np.float64)
    output_max = np.asarray(controller["output_max"][:3], dtype=np.float64)
    if np.any(input_max <= input_min) or np.any(output_max <= output_min):
        raise ValueError("invalid OSC position input/output ranges")
    return {
        "input_min": input_min,
        "input_max": input_max,
        "output_min_m": output_min,
        "output_max_m": output_max,
    }


def controls_to_position_delta(
    controls: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> np.ndarray:
    input_min = calibration["input_min"]
    input_max = calibration["input_max"]
    output_min = calibration["output_min_m"]
    output_max = calibration["output_max_m"]
    fraction = (controls - input_min) / (input_max - input_min)
    return output_min + fraction * (output_max - output_min)


def build_eval_args(args: argparse.Namespace, spec: TaskSpec, cache_dir: Path):
    boolean_clip = "--clip-actions" if args.clip_actions else "--no-clip-actions"
    diffusion_clip = (
        "--diffusion-clip-sample"
        if args.diffusion_clip_sample
        else "--no-diffusion-clip-sample"
    )
    return parse_eval_args(
        [
            "--idql-checkpoint",
            str(spec.idql_checkpoint),
            "--dp-checkpoint",
            str(spec.dp_checkpoint),
            "--expected-task",
            spec.name,
            "--output-dir",
            str(cache_dir / "unused_eval_output"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--actor-source",
            "hybrid_dp_chunk_actor",
            "--critic-source",
            args.critic_source,
            "--num-candidates",
            str(args.num_candidates),
            "--candidate-batch-size",
            str(args.candidate_batch_size),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--execution-horizon",
            str(args.execution_horizon),
            "--selection",
            "argmax",
            "--random-selection-probability",
            "0.0",
            "--require-success-condition-adapter",
            "--no-forbid-success-condition-adapter",
            "--inference-success-condition",
            "1.0",
            "--inference-condition-mask",
            "1.0",
            boolean_clip,
            diffusion_clip,
        ]
    )


def manifest_signature(args: argparse.Namespace, spec: TaskSpec, episode: str) -> dict[str, Any]:
    return {
        "task": spec.name,
        "episode": episode,
        "episode_class": args.episode_class,
        "dataset": str(spec.dataset.resolve()),
        "idql_checkpoint": str(spec.idql_checkpoint.resolve()),
        "dp_checkpoint": str(spec.dp_checkpoint.resolve()),
        "num_candidates": int(args.num_candidates),
        "candidate_batch_size": int(args.candidate_batch_size),
        "num_inference_steps": int(args.num_inference_steps),
        "execution_horizon": int(args.execution_horizon),
        "critic_source": args.critic_source,
        "seed": int(args.seed),
        "max_boundaries": int(args.max_boundaries),
        "clip_actions": bool(args.clip_actions),
        "diffusion_clip_sample": bool(args.diffusion_clip_sample),
    }


def assert_cache_compatible(
    args: argparse.Namespace,
    spec: TaskSpec,
    episode: str,
    cache_dir: Path,
) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    expected = manifest_signature(args, spec, episode)
    actual = manifest.get("signature")
    if actual != expected:
        raise RuntimeError(
            f"cached run signature does not match requested inputs in {cache_dir}; "
            "choose another --output-dir or pass --force-extract"
        )


def extract_candidates(
    args: argparse.Namespace,
    spec: TaskSpec,
    episode: str,
    episode_length: int,
    median_episode_length: float,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "candidate_chunks.npz"
    manifest_path = cache_dir / "manifest.json"
    if cache_path.exists() and manifest_path.exists() and not args.force_extract:
        assert_cache_compatible(args, spec, episode, cache_dir)
        print(f"Using cached candidate chunks: {cache_path}", flush=True)
        return json.loads(manifest_path.read_text())

    device = TorchUtils.get_torch_device(try_to_use_cuda=args.device == "cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading {spec.title} conditioned actor and critic on {device}", flush=True)
    eval_args = build_eval_args(args, spec, cache_dir)
    policy = load_policy(spec.idql_checkpoint, device, eval_args)
    required_attributes = (
        "last_q",
        "last_v",
        "last_selected_index",
        "last_env_trajectories",
    )
    if any(not hasattr(policy, attribute) for attribute in required_attributes):
        raise TypeError("loaded policy does not expose chunk-candidate diagnostics")

    timesteps: list[int] = []
    q_values: list[np.ndarray] = []
    values: list[float] = []
    selected_indices: list[int] = []
    env_actions: list[np.ndarray] = []
    eef_positions: list[np.ndarray] = []
    observation_images: list[np.ndarray] = []

    with h5py.File(spec.dataset, "r") as dataset:
        calibration = position_control_calibration(dataset)
        demo = dataset[f"data/{episode}"]
        obs_group = demo["obs"]
        policy_obs_keys = tuple(policy.algo.obs_shapes.keys())
        observation_horizon = int(
            policy.algo.algo_config.horizon.observation_horizon
        )
        missing = [key for key in policy_obs_keys if key not in obs_group]
        if missing:
            raise KeyError(f"episode {episode} is missing policy observations: {missing}")
        for key in (*spec.camera_keys, *spec.eef_position_keys):
            if key not in obs_group:
                raise KeyError(f"episode {episode} is missing visualization key {key!r}")

        policy.start_episode()
        for timestep in range(episode_length):
            history_indices = [
                max(0, timestep - observation_horizon + 1 + offset)
                for offset in range(observation_horizon)
            ]
            observation = {
                key: np.stack(
                    [np.asarray(obs_group[key][index]) for index in history_indices]
                )
                for key in policy_obs_keys
            }
            policy(observation)
            if policy.last_q is None:
                continue
            q = np.asarray(policy.last_q, dtype=np.float32).copy()
            trajectories = np.asarray(
                policy.last_env_trajectories,
                dtype=np.float32,
            ).copy()
            selected = int(policy.last_selected_index)
            if q.shape != (args.num_candidates,):
                raise ValueError(f"unexpected Q shape at t={timestep}: {q.shape}")
            if trajectories.ndim != 3 or trajectories.shape[0] != args.num_candidates:
                raise ValueError(
                    f"unexpected trajectory shape at t={timestep}: {trajectories.shape}"
                )
            if selected != int(np.argmax(q)):
                raise RuntimeError("argmax policy did not select the largest conservative Q")

            timesteps.append(timestep)
            q_values.append(q)
            values.append(float(policy.last_v))
            selected_indices.append(selected)
            env_actions.append(trajectories)
            eef_positions.append(
                np.stack(
                    [np.asarray(obs_group[key][timestep]) for key in spec.eef_position_keys]
                ).astype(np.float32)
            )
            observation_images.append(
                np.stack(
                    [np.asarray(obs_group[key][timestep]) for key in spec.camera_keys]
                ).astype(np.uint8)
            )
            print(
                f"{spec.name} {episode}: boundary {len(timesteps):03d} "
                f"t={timestep:04d}, selected=a{selected + 1}, Q={q[selected]:.4f}",
                flush=True,
            )
            if args.max_boundaries and len(timesteps) >= args.max_boundaries:
                break

    if not timesteps:
        raise RuntimeError("no chunk boundaries were extracted")

    arrays = {
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "q": np.stack(q_values).astype(np.float32),
        "value": np.asarray(values, dtype=np.float32),
        "selected": np.asarray(selected_indices, dtype=np.int16),
        "env_actions": np.stack(env_actions).astype(np.float32),
        "eef_positions": np.stack(eef_positions).astype(np.float32),
        "observation_images": np.stack(observation_images).astype(np.uint8),
    }
    np.savez_compressed(cache_path, **arrays)

    boundaries = []
    for boundary_index, (timestep, q, selected) in enumerate(
        zip(arrays["timesteps"], arrays["q"], arrays["selected"])
    ):
        order = np.argsort(q)[::-1]
        boundaries.append(
            {
                "boundary_index": int(boundary_index),
                "timestep": int(timestep),
                "selected_candidate": int(selected) + 1,
                "selected_q": float(q[selected]),
                "runner_up_candidate": int(order[1]) + 1,
                "runner_up_q": float(q[order[1]]),
                "selection_margin": float(q[order[0]] - q[order[1]]),
                "q_min": float(q.min()),
                "q_max": float(q.max()),
                "q_std": float(q.std()),
            }
        )
    manifest = {
        "signature": manifest_signature(args, spec, episode),
        "episode_length": int(episode_length),
        "median_episode_length_in_mask": float(median_episode_length),
        "boundaries": boundaries,
        "boundary_count": len(boundaries),
        "camera_keys": list(spec.camera_keys),
        "eef_position_keys": list(spec.eef_position_keys),
        "action_position_slices": [list(value) for value in spec.action_position_slices],
        "position_control_calibration": {
            key: value.tolist() for key, value in calibration.items()
        },
        "cache": str(cache_path.resolve()),
        "interpretation": (
            "OSC command deltas are accumulated from the current EEF pose; "
            "the curves are action-chunk proposals, not simulated future trajectories."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {cache_path}", flush=True)
    print(f"Saved {manifest_path}", flush=True)
    return manifest


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compose_observation(images: np.ndarray) -> np.ndarray:
    return images[0] if len(images) == 1 else np.concatenate(list(images), axis=1)


def style_3d_axis(ax: plt.Axes, points: np.ndarray) -> None:
    flattened = points.reshape(-1, 3)
    lower = flattened.min(axis=0)
    upper = flattened.max(axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.0)
    half = 0.58 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r"$\Delta x$ (cm)", labelpad=3)
    ax.set_ylabel(r"$\Delta y$ (cm)", labelpad=3)
    ax.set_zlabel(r"$\Delta z$ (cm)", labelpad=3)
    ax.tick_params(pad=0, length=2.5, width=0.7)
    ax.view_init(elev=24, azim=-58)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#B0B0B0")


def plot_action_proposals(
    ax: plt.Axes,
    trajectories_cm: np.ndarray,
    selected: int,
    title: str,
    selected_color: str,
) -> None:
    candidates = trajectories_cm.shape[0]
    for index in range(candidates):
        if index == selected:
            continue
        path = trajectories_cm[index]
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=OTHER_COLOR,
            alpha=0.30,
            linewidth=0.9,
        )
    selected_path = trajectories_cm[selected]
    ax.plot(
        selected_path[:, 0],
        selected_path[:, 1],
        selected_path[:, 2],
        color=selected_color,
        linewidth=2.8,
        marker="o",
        markersize=3.2,
        markeredgewidth=0,
        zorder=5,
    )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="#202020",
        s=24,
        marker="o",
        depthshade=False,
        zorder=6,
    )
    ax.scatter(
        [selected_path[-1, 0]],
        [selected_path[-1, 1]],
        [selected_path[-1, 2]],
        color=selected_color,
        edgecolor="#202020",
        linewidth=0.6,
        s=78,
        marker="*",
        depthshade=False,
        zorder=7,
    )
    ax.set_title(title, pad=4)
    style_3d_axis(ax, trajectories_cm)


def annotate_score_value(ax: plt.Axes, value: float, y: float, span: float) -> None:
    if value >= 0:
        x = value + 0.025 * span
        ha = "left"
    else:
        x = value - 0.025 * span
        ha = "right"
    ax.text(x, y, f"{value:.4f}", va="center", ha=ha, fontsize=9.2)


def plot_score_ranking(ax: plt.Axes, q: np.ndarray, selected: int) -> None:
    order = np.argsort(q)
    ranked_q = q[order]
    positions = np.arange(len(q))
    colors = [SELECTED_COLOR if index == selected else "#B5B5B5" for index in order]
    q_min = float(q.min())
    q_max = float(q.max())
    span = max(q_max - q_min, max(abs(float(q.mean())) * 0.01, 1e-4))
    baseline = q_min - 0.08 * span
    for y, (value, color, candidate) in enumerate(zip(ranked_q, colors, order)):
        ax.hlines(
            y,
            baseline,
            float(value),
            color=color,
            linewidth=2.2 if int(candidate) == selected else 1.25,
            zorder=2,
        )
        ax.scatter(
            [float(value)],
            [y],
            color=color,
            edgecolor="#303030",
            linewidth=0.55,
            s=42 if int(candidate) == selected else 24,
            zorder=3,
        )
    labels = [rf"$a_{{{index + 1}}}$" for index in order]
    ax.set_yticks(positions, labels)
    for tick_label, candidate in zip(ax.get_yticklabels(), order):
        if int(candidate) == selected:
            tick_label.set_color(SELECTED_COLOR)
            tick_label.set_fontweight("bold")
    ax.set_xlabel(r"$\min(Q_1,Q_2)$")
    ax.set_title("Critic ranking (zoomed)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    ax.set_xlim(baseline, q_max + 0.38 * span)
    top_three = set(np.argsort(q)[-min(3, len(q)):].tolist())
    for y, candidate in enumerate(order):
        if int(candidate) in top_three:
            annotate_score_value(ax, float(q[candidate]), float(y), span)


def proposal_paths_cm(
    env_actions: np.ndarray,
    action_slice: tuple[int, int],
    calibration: dict[str, np.ndarray],
) -> np.ndarray:
    start, end = action_slice
    controls = env_actions[:, :, start:end].astype(np.float64)
    deltas_cm = 100.0 * controls_to_position_delta(controls, calibration)
    cumulative = np.cumsum(deltas_cm, axis=1)
    origin = np.zeros((len(env_actions), 1, 3), dtype=np.float64)
    return np.concatenate((origin, cumulative), axis=1)


def plot_boundary(
    spec: TaskSpec,
    images: np.ndarray,
    env_actions: np.ndarray,
    q: np.ndarray,
    selected: int,
    timestep: int,
    boundary_index: int,
    boundary_count: int,
    calibration: dict[str, np.ndarray],
    figure_dir: Path,
    episode: str,
) -> list[Path]:
    configure_plot_style()
    num_arms = len(spec.action_position_slices)
    if num_arms == 1:
        fig = plt.figure(figsize=(12.2, 4.05), constrained_layout=True)
        grid = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.05, 1.05))
        observation_ax = fig.add_subplot(grid[0, 0])
        action_axes = [fig.add_subplot(grid[0, 1], projection="3d")]
        score_ax = fig.add_subplot(grid[0, 2])
    else:
        fig = plt.figure(figsize=(9.4, 7.0), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.25, 1.0),
            height_ratios=(0.82, 1.18),
        )
        observation_ax = fig.add_subplot(grid[0, 0])
        score_ax = fig.add_subplot(grid[0, 1])
        action_axes = [
            fig.add_subplot(grid[1, arm_index], projection="3d")
            for arm_index in range(num_arms)
        ]

    observation_ax.imshow(compose_observation(images))
    observation_ax.set_title("Current observation", pad=5)
    observation_ax.axis("off")
    if len(images) > 1:
        for index, label in enumerate(spec.camera_labels):
            observation_ax.text(
                (index + 0.5) / len(images),
                0.02,
                label,
                transform=observation_ax.transAxes,
                ha="center",
                va="bottom",
                color="white",
                fontsize=9.5,
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.55,
                    "pad": 1.4,
                },
            )

    for arm_index, (action_slice, arm_title) in enumerate(
        zip(spec.action_position_slices, spec.arm_titles)
    ):
        action_ax = action_axes[arm_index]
        paths_cm = proposal_paths_cm(env_actions, action_slice, calibration)
        plot_action_proposals(
            action_ax,
            paths_cm,
            selected,
            f"{arm_title} proposals",
            SELECTED_COLOR,
        )

    plot_score_ranking(score_ax, q, selected)
    fig.suptitle(
        f"{spec.title}: chunk boundary {boundary_index + 1:02d}/{boundary_count:02d} "
        f"($t={timestep}$), selected $a_{{{selected + 1}}}$ "
        f"with $Q={q[selected]:.4f}$",
        fontsize=13,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{spec.name}_{episode}_boundary_{boundary_index:03d}_t{timestep:04d}"
    )
    paths = [figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[1], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def plot_cached_candidates(
    spec: TaskSpec,
    episode: str,
    cache_dir: Path,
    figure_dir: Path,
) -> list[Path]:
    cache_path = cache_dir / "candidate_chunks.npz"
    manifest_path = cache_dir / "manifest.json"
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"candidate cache is incomplete in {cache_dir}; run --mode extract or all"
        )
    manifest = json.loads(manifest_path.read_text())
    calibration = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in manifest["position_control_calibration"].items()
    }
    with np.load(cache_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    boundary_count = len(arrays["timesteps"])
    paths: list[Path] = []
    figure_records = []
    for boundary_index in range(boundary_count):
        boundary_paths = plot_boundary(
            spec,
            arrays["observation_images"][boundary_index],
            arrays["env_actions"][boundary_index],
            arrays["q"][boundary_index],
            int(arrays["selected"][boundary_index]),
            int(arrays["timesteps"][boundary_index]),
            boundary_index,
            boundary_count,
            calibration,
            figure_dir,
            episode,
        )
        paths.extend(boundary_paths)
        figure_records.append(
            {
                "boundary_index": boundary_index,
                "timestep": int(arrays["timesteps"][boundary_index]),
                "pdf": str(boundary_paths[0].resolve()),
                "png": str(boundary_paths[1].resolve()),
            }
        )
        print(f"Saved {boundary_paths[0]}", flush=True)
        print(f"Saved {boundary_paths[1]}", flush=True)
    (figure_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "task": spec.name,
                "episode": episode,
                "figures": figure_records,
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_SPECS), required=True)
    parser.add_argument("--mode", choices=("extract", "plot", "all"), default="all")
    parser.add_argument("--episode-class", choices=("success", "failure"), default="success")
    parser.add_argument(
        "--episode",
        default=None,
        help="Episode name such as demo_247; default selects the median-duration episode.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--candidate-batch-size", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--critic-source", choices=("online", "target"), default="online")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-boundaries", type=int, default=0)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--diffusion-clip-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.num_candidates < 2:
        parser.error("--num-candidates must be at least 2 for critic selection")
    if args.candidate_batch_size <= 0 or args.num_inference_steps <= 0:
        parser.error("candidate batch size and inference steps must be positive")
    if args.execution_horizon <= 0 or args.max_boundaries < 0:
        parser.error("execution horizon must be positive; max boundaries cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    spec = TASK_SPECS[args.task]
    for path in (spec.idql_checkpoint, spec.dp_checkpoint, spec.dataset):
        if not path.exists():
            raise FileNotFoundError(path)
    episode, episode_length, median_episode_length = select_episode(
        spec.dataset,
        args.episode_class,
        args.episode,
    )
    cache_dir, figure_dir = output_directories(args, episode)
    print(
        f"{spec.title}: selected {args.episode_class} episode {episode} "
        f"(length={episode_length}, class median={median_episode_length:.1f})",
        flush=True,
    )
    if args.mode in ("extract", "all"):
        extract_candidates(
            args,
            spec,
            episode,
            episode_length,
            median_episode_length,
            cache_dir,
        )
    if args.mode in ("plot", "all"):
        plot_cached_candidates(spec, episode, cache_dir, figure_dir)
        print(f"Figure index: {figure_dir / 'figure_index.json'}", flush=True)


if __name__ == "__main__":
    main()
