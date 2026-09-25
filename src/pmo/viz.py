"""Figures: schedule Gantt charts, GA convergence, cost vs risk trade-off, sensitivity.

Colours come from a validated categorical palette used in fixed slot order. Component
types are also told apart by their lane inside each aircraft row, and every chart
has a legend or direct labels, so colour is never the only cue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from .evaluate import evaluate  # noqa: E402
from .instance import Instance, Schedule  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#f0efec"
NEUTRAL = "#8a8984"  # de-emphasised series (baselines, references)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.size": 9, "text.color": TEXT,
    })


def _money(x: float, _=None) -> str:
    return f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:.0f}k"


def _shade_blackouts(ax, inst: Instance) -> None:
    normal = inst.bays.max()
    for t in range(inst.horizon):
        if inst.bays[t] < normal:
            ax.axvspan(t, t + 1, color=BAND, zorder=0, lw=0)


# ------------------------------------------------------------------------- Gantt

def plot_gantt(inst: Instance, schedules: Mapping[str, Schedule], path: Path) -> None:
    """One panel per schedule: tasks per aircraft (a lane per component type) + hangar occupancy."""
    apply_style()
    K, N, T = len(inst.types), inst.n_aircraft, inst.horizon
    lane = 1.0
    row = K * lane + 0.8
    fig = plt.figure(figsize=(11, 4.2 * len(schedules) + 0.6))
    grid = fig.add_gridspec(2 * len(schedules), 1, height_ratios=[6, 1.2] * len(schedules), hspace=0.35,
                            top=1 - 0.55 / fig.get_figheight())

    for p, (title, schedule) in enumerate(schedules.items()):
        ev = evaluate(inst, schedule)
        ax = fig.add_subplot(grid[2 * p])
        occ = fig.add_subplot(grid[2 * p + 1], sharex=ax)
        _shade_blackouts(ax, inst)
        for comp, events in zip(inst.components, schedule):
            y = (N - 1 - comp.aircraft) * row + (K - 1 - comp.type) * lane
            for t in events:
                ax.add_patch(FancyBboxPatch((t + 0.12, y + 0.12), 0.76, lane - 0.24,
                                            boxstyle="round,pad=0,rounding_size=0.12",
                                            fc=SERIES[comp.type], ec="none", zorder=2))
        ax.set_ylim(-0.4, N * row - 0.4)
        ax.set_yticks([(N - 1 - a) * row + K * lane / 2 for a in range(N)])
        ax.set_yticklabels([f"Aircraft {a + 1}" for a in range(N)])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, T)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{title}  ·  {ev.n_visits} hangar visits  ·  {_money(ev.total_cost)} expected cost  ·  "
                     f"{ev.expected_failures:.1f} expected failures")

        weeks = np.arange(T)
        in_hangar = np.zeros(T, dtype=int)
        for _, t in {(c.aircraft, t) for c, ev_ in zip(inst.components, schedule) for t in ev_}:
            in_hangar[t] += 1
        _shade_blackouts(occ, inst)
        occ.bar(weeks + 0.5, in_hangar, width=0.7, color=SERIES[0], zorder=2)
        occ.step(np.append(weeks, T), np.append(inst.bays, inst.bays[-1]), where="post",
                 color=TEXT_2, lw=1.2, zorder=3)
        occ.set_ylim(0, inst.bays.max() + 0.6)
        occ.set_yticks(range(inst.bays.max() + 1))
        occ.set_ylabel("In hangar", rotation=0, ha="right", va="center")
        occ.grid(axis="x", visible=False)
        if p == len(schedules) - 1:
            occ.set_xlabel("Week")
        else:
            occ.tick_params(labelbottom=False)

    handles = [Patch(color=SERIES[k], label=ct.name) for k, ct in enumerate(inst.types)]
    handles += [Patch(color=BAND, label="Reduced hangar capacity"),
                plt.Line2D([], [], color=TEXT_2, lw=1.2, label="Hangar bays")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gantt_html(inst: Instance, schedules: Mapping[str, Schedule], path: Path) -> None:
    """Interactive Gantt (Plotly), one tab-like dropdown entry per schedule, hover per task."""
    import plotly.graph_objects as go

    fig = go.Figure()
    names = list(schedules)
    for s, (title, schedule) in enumerate(schedules.items()):
        for k, ct in enumerate(inst.types):
            xs, ys, text = [], [], []
            for comp, events in zip(inst.components, schedule):
                if comp.type != k:
                    continue
                for t in events:
                    xs.append(t)
                    ys.append(f"Aircraft {comp.aircraft + 1} · {ct.name}")
                    text.append(f"{title}<br>Aircraft {comp.aircraft + 1} · {ct.name}<br>Week {t}")
            fig.add_trace(go.Bar(
                x=[0.8] * len(xs), base=[x + 0.1 for x in xs], y=ys, orientation="h", name=ct.name,
                marker_color=SERIES[k], hovertext=text, hoverinfo="text", visible=s == 0,
                legendgroup=ct.name, showlegend=True,
            ))
    order = [f"Aircraft {a + 1} · {ct.name}" for a in range(inst.n_aircraft) for ct in inst.types]
    buttons = [dict(label=n, method="update",
                    args=[{"visible": [i // len(inst.types) == s for i in range(len(fig.data))]}])
               for s, n in enumerate(names)]
    fig.update_layout(
        title=f"Maintenance schedule — {inst.name}", barmode="overlay", height=60 + 16 * len(order),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=TEXT, size=11),
        xaxis=dict(title="Week", range=[0, inst.horizon], gridcolor=GRID, dtick=4),
        yaxis=dict(categoryorder="array", categoryarray=order[::-1], gridcolor=GRID),
        updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.06, yanchor="bottom")],
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
    )
    fig.write_html(path, include_plotlyjs="cdn")


# ------------------------------------------------------------------- convergence

def plot_convergence(histories: Sequence[Sequence[dict]], references: Mapping[str, float], path: Path,
                     zoom_from: float = 0.4) -> None:
    """Best objective per generation: median over seeds with the min-max band.

    Left: the whole run against the first reference (the baseline). Right: the last
    ``1 - zoom_from`` of the run, zoomed in so the GA and CP-SAT references separate.
    """
    apply_style()
    best = np.array([[h["best"] for h in hist] for hist in histories])
    gens = np.arange(best.shape[1])
    med = np.median(best, axis=0)
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    first = gens >= int(zoom_from * gens[-1])
    for a, mask in ((ax, slice(None)), (zoom, first)):
        a.fill_between(gens[mask], best.min(axis=0)[mask], best.max(axis=0)[mask], color=SERIES[0], alpha=0.12, lw=0)
        a.plot(gens[mask], med[mask], color=SERIES[0])
        a.yaxis.set_major_formatter(_money)
        a.set_xlabel("Generation")
        a.set_xlim(gens[mask][0], gens[-1])
    ax.set_ylabel("Expected total cost")
    ax.set_title(f"GA convergence (median of {len(histories)} seeds, band = min to max)")
    zoom.set_title("Last generations, zoomed")

    items = list(references.items())
    if items:
        label, value = items[0]
        ax.axhline(value, color=NEUTRAL, lw=1, zorder=1)
        ax.annotate(f"{label}  {_money(value)}", (gens[-1], value), xytext=(-4, -4), textcoords="offset points",
                    ha="right", va="top", color=TEXT_2, fontsize=8)
    ax.annotate(f"GA  {_money(med[-1])}", (gens[-1], med[-1]), xytext=(-4, 6), textcoords="offset points",
                ha="right", va="bottom", color=TEXT, fontsize=8)

    lows = [v for _, v in items[1:]] + [best[:, first].min()]
    highs = [v for _, v in items[1:]] + [best[:, first].max()]
    pad = 0.08 * (max(highs) - min(lows))
    zoom.set_ylim(min(lows) - pad, max(highs) + pad)
    for label, value in items[1:]:
        zoom.axhline(value, color=NEUTRAL, lw=1, zorder=1)
        zoom.annotate(f"{label}  {_money(value)}", (gens[first][0], value), xytext=(4, 3),
                      textcoords="offset points", ha="left", va="bottom", color=TEXT_2, fontsize=8)
    zoom.annotate(f"GA median  {_money(med[-1])}", (gens[-1], med[-1]), xytext=(-4, 5), textcoords="offset points",
                  ha="right", va="bottom", color=TEXT, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- trade-off

def plot_tradeoff(front: Sequence[tuple[float, float]], sweep: pd.DataFrame,
                  points: Mapping[str, tuple[float, float]], path: Path) -> None:
    """Expected cost vs expected failures: NSGA-II front, weighted-sum GA runs, reference schedules."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    fx, fy = zip(*sorted(front))
    ax.plot(fx, fy, color=SERIES[1], lw=2, zorder=2, label="NSGA-II front (one run)")
    ax.scatter(fx, fy, s=18, color=SERIES[1], edgecolor=SURFACE, lw=1, zorder=3)
    med = sweep.groupby("w_risk")[["total_cost", "expected_failures"]].median().reset_index()
    ax.scatter(sweep.total_cost, sweep.expected_failures, s=16, color=SERIES[0], alpha=0.35, lw=0, zorder=3)
    ax.scatter(med.total_cost, med.expected_failures, s=64, color=SERIES[0], edgecolor=SURFACE, lw=2,
               zorder=4, label="Weighted GA (median per risk weight; faint dots: single seeds)")
    for _, r in med.iterrows():
        if r.w_risk in (0, 1e5, 1e6):
            # below-left keeps labels off the front; the λ = 0 point sits next to CP-SAT, so above-right
            offset, ha, va = ((10, 4), "left", "bottom") if r.w_risk == 0 else ((-8, -10), "right", "top")
            ax.annotate(f"λ = {_money(r.w_risk) if r.w_risk else '0'}", (r.total_cost, r.expected_failures),
                        xytext=offset, textcoords="offset points", ha=ha, va=va, color=TEXT_2, fontsize=8)
    markers = ["s", "D", "^", "v"]
    for i, (label, (cost, failures)) in enumerate(points.items()):
        ax.scatter([cost], [failures], s=60, marker=markers[i % len(markers)],
                   color=SERIES[2] if "CP-SAT" in label else NEUTRAL, edgecolor=SURFACE, lw=1.5, zorder=5,
                   label=label)
    ax.xaxis.set_major_formatter(_money)
    ax.set_xlabel("Expected total cost (maintenance + visits + failures), 1 year")
    ax.set_ylabel("Expected failures, fleet, 1 year")
    ax.set_title("Cost vs failure-risk trade-off")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- sensitivity

def _weight_label(w: float) -> str:
    return "0" if w == 0 else _money(w)


def plot_risk_sensitivity(sweep: pd.DataFrame, path: Path) -> None:
    """Small multiples of solution metrics against the risk weight (median, min-max whiskers)."""
    apply_style()
    metrics = [("total_cost", "Expected total cost", _money), ("expected_failures", "Expected failures", None),
               ("n_visits", "Hangar visits", None), ("n_events", "Preventive tasks", None)]
    weights = sorted(sweep.w_risk.unique())
    x = np.arange(len(weights))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, (col, title, fmt) in zip(axes.flat, metrics):
        g = sweep.groupby("w_risk")[col]
        med, lo, hi = g.median().loc[weights], g.min().loc[weights], g.max().loc[weights]
        ax.vlines(x, lo, hi, color=SERIES[0], alpha=0.35, lw=2)
        ax.plot(x, med, color=SERIES[0], marker="o", ms=6, mec=SURFACE, mew=2)
        ax.set_title(title)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        ax.set_xticks(x, [_weight_label(w) for w in weights])
    for ax in axes[1]:
        ax.set_xlabel("Risk weight λ ($ per expected failure)")
    fig.suptitle("Sensitivity to the risk weight (median of seeds; bars span min to max)",
                 x=0.01, ha="left", fontsize=10, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_failure_cost_sensitivity(sweep: pd.DataFrame, inst: Instance, path: Path) -> None:
    """Preventive tasks per component type as the corrective repair cost is scaled."""
    apply_style()
    scales = sorted(sweep.failure_cost_scale.unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, ct in enumerate(inst.types):
        med = sweep.groupby("failure_cost_scale")[f"tasks_{ct.name}"].median().loc[scales]
        ax.plot(scales, med, color=SERIES[k], marker="o", ms=6, mec=SURFACE, mew=2, label=ct.name)
        ax.annotate(ct.name, (scales[-1], med.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TEXT, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(scales, [f"×{s:g}" for s in scales])
    ax.set_xlim(scales[0] / 1.15, scales[-1] * 1.6)
    ax.set_xlabel("Corrective repair cost multiplier")
    ax.set_ylabel("Preventive tasks in the year (fleet)")
    ax.set_title("Preventive effort vs price of failure (median of seeds)")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nsga2_hypervolume(history: Sequence[dict], path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    gens = [h["generation"] for h in history]
    hv = np.array([h["hypervolume"] for h in history])
    ax.plot(gens, hv / hv.max(), color=SERIES[1])
    ax.set_xlim(0, gens[-1])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume (share of final)")
    ax.set_title("NSGA-II convergence")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
