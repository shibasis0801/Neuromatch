"""Figures for the notebooks: every function takes tidy data and returns a matplotlib Figure."""

from functools import wraps
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

THEME = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
}

CONDITION = {"supervised": "#E07A1F", "unsupervised": "#7B4FA0"}
ROLE = {"leaf": "#2E8B57", "circle": "#2F6DB5", "gray": "#C8C8C8"}
AREA = {"V1": "#1B9E77", "mHV": "#D95F02", "lHV": "#7570B3", "aHV": "#E7298A"}
PHASE = ["before", "after"]


def themed(plot):
    @wraps(plot)
    def draw(*args, **kwargs):
        with mpl.rc_context(THEME):
            return plot(*args, **kwargs)
    return draw


@themed
def corridor(frames: pd.DataFrame, seconds: float = 120) -> Figure:
    shown = frames[frames.time <= seconds]
    fig = Figure(figsize=(9, 4), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    top.axhspan(40, 60, color=ROLE["gray"], alpha=0.3, lw=0)
    for role, part in shown.dropna(subset="role").groupby("role"):
        top.scatter(part.time, part.position, s=4, color=ROLE[role], label=role)
    top.set(ylabel="position in corridor", title="Virtual corridor: texture zone (0-40), then gray zone (40-60)")
    fig.legend(loc="outside right upper", markerscale=3)
    bottom.plot(shown.time, shown.speed, color="black", lw=0.8)
    bottom.set(xlabel="time (s)", ylabel="running speed")
    return fig


@themed
def areas(neurons: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(5, 5), layout="constrained")
    ax = fig.subplots()
    ax.scatter(neurons.x, neurons.y, s=0.2, color="#DDDDDD", rasterized=True)
    for area, color in AREA.items():
        part = neurons[neurons.area == area]
        ax.scatter(part.x, part.y, s=0.2, color=color, rasterized=True, label=f"{area} ({len(part):,})")
    ax.set(aspect="equal", title="Recorded neurons by visual area")
    ax.axis("off")
    ax.legend(markerscale=20, loc="lower left")
    return fig


@themed
def raster(activity: np.ndarray, frames: pd.DataFrame, areas: pd.Series) -> Figure:
    names = areas.fillna("other")
    order = np.argsort(names.map({a: i for i, a in enumerate(AREA)}).fillna(len(AREA)).to_numpy(), kind="stable")
    z = (activity - activity.mean(1, keepdims=True)) / (activity.std(1, keepdims=True) + 1e-8)
    fig = Figure(figsize=(9, 5), layout="constrained")
    strip, ax = fig.subplots(2, 1, sharex=True, height_ratios=[1, 12])
    texture = frames.role.where(frames.position < 40).fillna("gray")
    strip.imshow([[mpl.colors.to_rgb(ROLE[r]) for r in texture]], aspect="auto")
    strip.set(yticks=[], title="Reconstructed activity, neurons sorted by area")
    strip.grid(False)
    ax.imshow(z[order], aspect="auto", cmap="gray_r", vmin=-1, vmax=3, interpolation="nearest")
    ax.grid(False)
    counts = names.iloc[order].value_counts(sort=False)
    ax.hlines(counts.cumsum().iloc[:-1] - 0.5, -0.5, z.shape[1] - 0.5, color="white", lw=2)
    ax.set_yticks(counts.cumsum() - counts / 2, counts.index)
    ax.set_xlabel("frame")
    return fig


@themed
def tuning(profile: pd.DataFrame) -> Figure:
    panels = [a for a in AREA if a in set(profile.area)]
    fig = Figure(figsize=(3 * len(panels), 3), layout="constrained")
    for ax, area in zip(fig.subplots(1, len(panels), sharey=True), panels):
        for role, part in profile[profile.area == area].groupby("role"):
            ax.plot(part.position, part.response, color=ROLE[role], label=role)
        ax.axvspan(40, 60, color=ROLE["gray"], alpha=0.3, lw=0)
        ax.set(title=area, xlabel="position")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
    fig.axes[0].set_ylabel("mean response")
    return fig


@themed
def distributions(histograms: pd.DataFrame, windows: list[int]) -> Figure:
    fig = Figure(figsize=(2.4 * len(windows), 4.6), layout="constrained")
    grid = fig.subplots(len(PHASE), len(windows), sharex=True, sharey=True, squeeze=False)
    for row, phase in zip(grid, PHASE):
        for ax, window in zip(row, windows):
            shown = histograms[(histograms.phase == phase) & (histograms.window == window)]
            for condition, part in shown.groupby("condition"):
                ax.fill_between(part.bin, part.density, step="mid", alpha=0.35, color=CONDITION[condition], lw=0)
                ax.step(part.bin, part.density, where="mid", color=CONDITION[condition], lw=1, label=condition)
            ax.axvline(0, color="gray", ls="--", lw=0.8)
            ax.set_title(f"{phase}, window {window + 1}")
        row[0].set_ylabel("density")
    for ax in grid[-1]:
        ax.set_xlabel("d′")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
    return fig


@themed
def trajectories(stats: pd.DataFrame, metrics: dict[str, str]) -> Figure:
    fig = Figure(figsize=(9, 2.3 * len(metrics)), layout="constrained")
    grid = fig.subplots(len(metrics), len(PHASE), sharex=True, sharey="row", squeeze=False)
    for row, (metric, label) in zip(grid, metrics.items()):
        for ax, phase in zip(row, PHASE):
            shown = stats[stats.phase == phase].groupby(["condition", "window"])[metric].agg(["mean", "sem", "count"])
            for condition, part in shown.reset_index().groupby("condition"):
                part = part[part["count"] >= 2]
                ax.plot(part.window + 1, part["mean"], color=CONDITION[condition], label=condition)
                ax.fill_between(part.window + 1, part["mean"] - part["sem"], part["mean"] + part["sem"], color=CONDITION[condition], alpha=0.2, lw=0)
            ax.set_title(f"{label}, {phase} learning")
        row[0].set_ylabel(label)
    for ax in grid[-1]:
        ax.set_xlabel("window (20 trial pairs, stride 1)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
    return fig


def animate(histograms: pd.DataFrame, path: str | Path, fps: int = 12) -> Path:
    windows = sorted(histograms.window.unique())
    with mpl.rc_context(THEME):
        fig = Figure(figsize=(6, 3.6), layout="constrained")
        ax = fig.subplots()
        top = histograms.density.max() * 1.1

        def show(window):
            ax.clear()
            for condition, part in histograms[histograms.window == window].groupby("condition"):
                ax.fill_between(part.bin, part.density, step="mid", alpha=0.35, color=CONDITION[condition], lw=0)
                ax.step(part.bin, part.density, where="mid", color=CONDITION[condition], label=condition)
            ax.axvline(0, color="gray", ls="--", lw=0.8)
            ax.set(xlim=(histograms.bin.min(), histograms.bin.max()), ylim=(0, top), xlabel="d′", ylabel="density", title=f"Window {window + 1}")
            ax.legend(loc="upper right")

        FuncAnimation(fig, show, frames=windows).save(path, writer=PillowWriter(fps=fps))
    return Path(path)
