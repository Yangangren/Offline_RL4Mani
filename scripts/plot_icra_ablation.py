#!/usr/bin/env python3
"""Plot the ICRA ablation success rates as a grouped bar chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("Can", "Square", "Transport", "ToolHang")
METHODS = ("SUB", "SUB + actor", "SUB + actor + critic")

# Values are reported as mean +/- standard deviation in the result table.
SUCCESS_RATES = np.array(
    [
        [90.0, 78.8, 87.6, 71.2],  # SUB
        [94.4, 82.4, 84.0, 78.4],  # SUB + condition (actor)
        [98.4, 88.0, 90.4, 78.4],  # RAL (actor + critic)
    ]
)
STANDARD_DEVIATIONS = np.array(
    [
        [2.8, 4.4, 2.6, 3.8],
        [4.3, 7.4, 5.1, 8.9],
        [1.7, 3.1, 4.6, 5.0],
    ]
)

# Colorblind-friendly solid fill colors.
COLORS = ("#4C78A8", "#F2A541", "#2A9D8F")


def configure_style() -> None:
    """Apply compact, paper-friendly Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure() -> plt.Figure:
    """Create the grouped bar chart."""
    configure_style()
    fig, ax = plt.subplots(figsize=(5.9, 3.55), constrained_layout=True)

    x = np.arange(len(TASKS))
    width = 0.23
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for index, (method, color) in enumerate(zip(METHODS, COLORS)):
        ax.bar(
            x + offsets[index],
            SUCCESS_RATES[index],
            width=width,
            yerr=STANDARD_DEVIATIONS[index],
            label=method,
            color=color,
            edgecolor="#202020",
            linewidth=0.7,
            error_kw={
                "ecolor": "#202020",
                "elinewidth": 0.9,
                "capsize": 2.5,
                "capthick": 0.9,
            },
            zorder=3,
        )

    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Task", labelpad=5)
    ax.set_xticks(x, TASKS)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.42, len(TASKS) - 0.58)
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="Directory for the PDF and PNG outputs (default: repository/figures).",
    )
    parser.add_argument(
        "--stem",
        default="icra_ablation_success_rate",
        help="Output filename without an extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    png_path = args.output_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
