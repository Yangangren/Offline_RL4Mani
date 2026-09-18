#!/usr/bin/env python3
"""Plot the success-rate ablation for policy conditioning types."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONDITION_TYPES = ("source", "outcome", "Q-V", "Q")
SUCCESS_RATES = np.asarray((85.6, 88.0, 88.8, 91.2), dtype=float)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_success_rates(output_dir: Path) -> list[Path]:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.35), constrained_layout=True)

    x_positions = np.arange(len(CONDITION_TYPES))
    bars = ax.bar(
        x_positions,
        SUCCESS_RATES,
        width=0.62,
        color="#4C78A8",
        edgecolor="#202020",
        linewidth=0.9,
        zorder=2,
    )

    ax.set_xlabel("Condition type")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x_positions, CONDITION_TYPES)
    ax.set_ylim(82.0, 94.0)
    ax.set_yticks(np.arange(82.0, 94.1, 2.0))
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)

    for bar, value in zip(bars, SUCCESS_RATES):
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "condition_type_success_rate.pdf",
        output_dir / "condition_type_success_rate.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.01)
    fig.savefig(paths[1], dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures",
        help="Directory for the PDF and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in plot_success_rates(args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
