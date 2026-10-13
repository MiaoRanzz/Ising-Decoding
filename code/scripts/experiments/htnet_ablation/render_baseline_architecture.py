# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the ablation baseline and its three optional improvements."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


BASE = "#334155"
BASE_FILL = "#F8FAFC"
PANEL = "#E2E8F0"
BRANCH = "#2563EB"
AXIS = "#D97706"
SKIP = "#7C3AED"


def box(ax, x, y, w, h, text, *, edge=BASE, fill=BASE_FILL, dashed=False,
        fontsize=10, linewidth=1.8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=fill,
        linestyle=(0, (5, 3)) if dashed else "solid",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color="#0F172A", fontsize=fontsize, zorder=4)
    return patch


def arrow(ax, start, end, *, color=BASE, dashed=False, linewidth=1.8,
          connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        linestyle=(0, (5, 3)) if dashed else "solid",
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def line(ax, xs, ys, *, color=BASE, dashed=False, linewidth=1.8):
    ax.plot(xs, ys, color=color, linewidth=linewidth,
            linestyle=(0, (5, 3)) if dashed else "solid", zorder=1)


def render(output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "axes.linewidth": 0,
    })
    fig, ax = plt.subplots(figsize=(19, 10.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(9.5, 10.12, "Baseline Architecture with Three Optional Improvements",
            ha="center", va="center", fontsize=22, fontweight="bold", color="#0F172A")
    ax.text(9.5, 9.72,
            "Solid path = baseline   ·   Dashed path = optional improvement (OFF in baseline)",
            ha="center", va="center", fontsize=11.5, color="#475569")

    # Main network path.
    main_y, main_h = 8.15, 0.9
    box(ax, 0.35, main_y, 1.75, main_h, "Input syndrome\n[B, 4, T, D, D]", fill="#F1F5F9")
    box(ax, 2.55, main_y, 2.05, main_h, "Stem\nConv 3³: 4→112\nGN + GELU", fill="#E0F2FE", fontsize=9.4)
    box(ax, 5.10, main_y, 2.45, main_h, "STFusion Block ×3\n112 channels", fill="#DBEAFE")
    box(ax, 8.05, main_y, 1.65, main_h, "Head Norm\nGroupNorm", fill="#F1F5F9")
    box(ax, 10.45, main_y, 2.15, main_h, "Head Hidden\nConv 1³ + GELU\n112→112", fill="#F3E8FF")
    box(ax, 13.10, main_y, 1.70, main_h, "Head Out\nConv 1³\n112→4", fill="#F1F5F9")
    box(ax, 15.30, main_y, 2.25, main_h, "Correction logits\n[B, 4, T, D, D]", fill="#ECFDF5")
    for x0, x1 in [(2.10, 2.55), (4.60, 5.10), (7.55, 8.05),
                   (9.70, 10.45), (12.60, 13.10), (14.80, 15.30)]:
        arrow(ax, (x0, main_y + main_h / 2), (x1, main_y + main_h / 2))

    # Optional raw-input skip into the output head.
    box(ax, 8.55, 6.92, 1.95, 0.62, "Concat raw input\n112+4 channels",
        edge=SKIP, fill="#F5F3FF", dashed=True, fontsize=9.2)
    line(ax, [1.22, 1.22, 8.50], [8.15, 7.23, 7.23], color=SKIP, dashed=True, linewidth=2.1)
    arrow(ax, (8.50, 7.23), (8.55, 7.23), color=SKIP, dashed=True, linewidth=2.1)
    arrow(ax, (8.88, 8.15), (8.94, 7.54), color=SKIP, dashed=True, linewidth=2.1)
    line(ax, [9.52, 9.52, 10.42], [7.54, 7.78, 7.78], color=SKIP, dashed=True, linewidth=2.1)
    arrow(ax, (10.42, 7.78), (10.47, 8.14), color=SKIP, dashed=True,
          linewidth=2.1, connectionstyle="arc3,rad=-0.15")
    ax.text(5.1, 7.42, "Raw-input Head Skip  ·  optional", color=SKIP,
            fontsize=10.2, fontweight="bold", ha="center")

    # Expanded block panel.
    panel = FancyBboxPatch(
        (0.28, 0.42), 18.42, 5.92,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.3, edgecolor="#94A3B8", facecolor="#F8FAFC", zorder=0,
    )
    ax.add_patch(panel)
    ax.text(0.62, 6.02, "Expanded STFusion Block (one of ×3)", ha="left", va="center",
            fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(18.30, 6.02, "feature shape: [B, 112, T, D, D]", ha="right", va="center",
            fontsize=10, color="#64748B")

    box(ax, 0.60, 3.10, 1.10, 0.75, "Input\n112 ch", fill="#F1F5F9", fontsize=9.4)
    box(ax, 2.15, 2.95, 1.72, 1.05, "Pre\nGN · Conv 1³\n112→168 · GELU", fill="#E0F2FE", fontsize=9.0)
    arrow(ax, (1.70, 3.48), (2.15, 3.48))

    branches = [
        (4.45, 4.55, "Spatial branch\nDW Conv 1×3×3", "#E0F2FE"),
        (4.45, 3.08, "Temporal branch\nDW Conv 3×1×1", "#E0F2FE"),
        (4.45, 1.61, "Joint branch\nGN · Group Conv 3³", "#E0F2FE"),
    ]
    for x, y, label, fill in branches:
        box(ax, x, y, 2.10, 0.82, label, fill=fill, fontsize=9.2)
        arrow(ax, (3.87, 3.48), (x, y + 0.41), connectionstyle="arc3,rad=0.08")

    box(ax, 7.05, 2.87, 1.82, 1.18, "Adaptive\nBranch Fusion\n168 channels",
        fill="#DBEAFE", fontsize=9.3)
    for _, y, _, _ in branches:
        arrow(ax, (6.55, y + 0.41), (7.05, 3.46), connectionstyle="arc3,rad=-0.08")

    # The baseline uses identity; Branch Mixer is the optional replacement.
    box(ax, 9.35, 2.06, 1.62, 0.74, "Identity\n(Baseline)", fill="#F1F5F9", fontsize=9.2)
    box(ax, 9.35, 4.08, 1.62, 0.92, "Branch Mixer\nGroup Conv 1³\n+ GELU",
        edge=BRANCH, fill="#EFF6FF", dashed=True, fontsize=8.8, linewidth=2.1)
    arrow(ax, (8.87, 3.46), (9.35, 2.43), connectionstyle="arc3,rad=0.18")
    arrow(ax, (8.87, 3.57), (9.35, 4.54), color=BRANCH, dashed=True,
          linewidth=2.1, connectionstyle="arc3,rad=-0.18")

    box(ax, 11.45, 2.92, 1.55, 1.10, "Project\nConv 1³\n168→112\n+ GELU", fill="#E0F2FE", fontsize=8.9)
    arrow(ax, (10.97, 2.43), (11.45, 3.40), connectionstyle="arc3,rad=-0.18")
    arrow(ax, (10.97, 4.54), (11.45, 3.56), color=BRANCH, dashed=True,
          linewidth=2.1, connectionstyle="arc3,rad=0.18")

    # The baseline uses ChannelGate; AxisChannelGate is the optional replacement.
    box(ax, 13.42, 2.06, 1.72, 0.74, "ChannelGate3D\n(Baseline)", fill="#FEF3C7", fontsize=9.0)
    box(ax, 13.42, 4.08, 1.72, 0.92, "AxisChannelGate3D\nChannel + temporal\n+ spatial logits",
        edge=AXIS, fill="#FFF7ED", dashed=True, fontsize=8.5, linewidth=2.1)
    arrow(ax, (13.00, 3.46), (13.42, 2.43), connectionstyle="arc3,rad=0.18")
    arrow(ax, (13.00, 3.57), (13.42, 4.54), color=AXIS, dashed=True,
          linewidth=2.1, connectionstyle="arc3,rad=-0.18")

    box(ax, 15.60, 3.06, 1.10, 0.82, "Dropout", fill="#F1F5F9", fontsize=9.2)
    arrow(ax, (15.14, 2.43), (15.60, 3.40), connectionstyle="arc3,rad=-0.18")
    arrow(ax, (15.14, 4.54), (15.60, 3.56), color=AXIS, dashed=True,
          linewidth=2.1, connectionstyle="arc3,rad=0.18")

    add = Circle((17.38, 3.47), 0.27, edgecolor=BASE, facecolor="white",
                 linewidth=1.8, zorder=3)
    ax.add_patch(add)
    ax.text(17.38, 3.47, "+", ha="center", va="center", fontsize=15,
            color=BASE, fontweight="bold", zorder=4)
    arrow(ax, (16.70, 3.47), (17.11, 3.47))
    box(ax, 17.86, 3.10, 0.58, 0.74, "Out", fill="#ECFDF5", fontsize=9.2)
    arrow(ax, (17.65, 3.47), (17.86, 3.47))

    # Outer residual connection.
    line(ax, [1.15, 1.15, 17.38], [3.10, 0.90, 0.90], color=BASE, linewidth=1.8)
    arrow(ax, (17.38, 0.90), (17.38, 3.18), color=BASE, linewidth=1.8)
    ax.text(7.9, 1.05, "Residual connection", ha="center", va="bottom",
            fontsize=9.5, color="#475569")

    # Legend.
    legend_y = 0.14
    line(ax, [0.65, 1.25], [legend_y, legend_y], color=BASE, linewidth=2.2)
    ax.text(1.36, legend_y, "Baseline path", va="center", fontsize=9.5, color="#334155")
    line(ax, [4.10, 4.70], [legend_y, legend_y], color=BRANCH, dashed=True, linewidth=2.2)
    ax.text(4.81, legend_y, "Branch Mixer", va="center", fontsize=9.5, color=BRANCH)
    line(ax, [7.20, 7.80], [legend_y, legend_y], color=AXIS, dashed=True, linewidth=2.2)
    ax.text(7.91, legend_y, "AxisChannelGate3D", va="center", fontsize=9.5, color=AXIS)
    line(ax, [11.05, 11.65], [legend_y, legend_y], color=SKIP, dashed=True, linewidth=2.2)
    ax.text(11.76, legend_y, "Raw-input Head Skip", va="center", fontsize=9.5, color=SKIP)
    ax.text(18.25, legend_y, "Dashed modules are OFF in the baseline", ha="right", va="center",
            fontsize=9.5, color="#64748B", fontstyle="italic")

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "baseline_architecture.svg"
    png_path = output_dir / "baseline_architecture.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    render(Path("outputs/htnet_ablation/figures"))
