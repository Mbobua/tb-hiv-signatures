"""Shared plotting style and palettes (colour-blind-checked categorical order)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_2 = "#52514e"

# Colour follows the entity, never its rank: the same group keeps its colour everywhere.
PALETTES = {
    "tb_status": {"LTBI": SLOTS[0], "ATB": SLOTS[1], "OD": SLOTS[2], "Control": MUTED},
    "hiv_status": {"HIV-": SLOTS[0], "HIV+": SLOTS[1]},
    "site": {"South Africa": SLOTS[0], "Malawi": SLOTS[1], "London": SLOTS[2]},
    "direction": {"up": SLOTS[1], "down": SLOTS[0], "ns": "#c3c2b7"},
}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=SLOTS),
        "font.family": "DejaVu Sans",
    })


def dodge_labels(ax, xs, ys, labels, fontsize=7, color=INK_2):
    """Place point labels, nudging each one down until it no longer collides with labels
    already placed (a small, dependency-free stand-in for adjustText)."""
    import numpy as np

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = (x1 - x0) * 0.11, (y1 - y0) * 0.035
    placed = []
    order = np.argsort(-np.asarray(ys, dtype=float))
    for i in order:
        x, y, lab = xs[i], ys[i], labels[i]
        ty = y
        for _ in range(40):
            if all(abs(x - px) > w or abs(ty - py) > h for px, py in placed):
                break
            ty -= h
        placed.append((x, ty))
        ax.annotate(lab, (x, y), xytext=(x + (x1 - x0) * 0.01, ty), fontsize=fontsize, color=color,
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.6) if ty != y else None)


def savefig(fig, path=None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    return fig


def close_all():
    plt.close("all")


set_style()
