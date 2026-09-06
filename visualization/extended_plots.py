"""
visualization/extended_plots.py
================================
Extended publication-quality charts for the research upgrade.

This module EXTENDS visualization/plots.py — it does NOT modify it.
All 4 original charts continue to work unchanged.

NEW CHARTS (3 additional figures)
-----------------------------------
Chart 5 — boxplots.png
    Side-by-side boxplots of per-trial waiting times for all algorithms.
    Shows distribution shape, median, IQR, and outliers — more informative
    than a simple bar chart with error bars.  Standard in ML papers.

Chart 6 — convergence_curve.png
    RL training convergence with multiple smoothing windows and milestone
    annotations.  Distinguishes exploration phase from exploitation phase.
    Shows *when* the agent learned (the inflection point), not just
    whether it converged.

Chart 7 — scheduler_selection.png
    Horizontal bar chart of how often each underlying scheduler was chosen
    by the contextual bandit across training + evaluation episodes.
    Provides direct interpretability: "the agent learned to prefer SJF_Pred
    in 89% of states."  Critical for viva defence.

USAGE
-----
    from visualization.extended_plots import (
        plot_boxplots,
        plot_convergence_curve,
        plot_scheduler_selection,
        plot_extended_metrics_radar,
    )

    # Called from extended_experiment.py or directly after evaluation
    plot_boxplots(trial_data, output_path="results/boxplots.png")
    plot_convergence_curve(episode_rewards, output_path="results/convergence.png")
    plot_scheduler_selection(selection_counts, output_path="results/selection.png")
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use the same colour palette as the original plots.py for visual consistency
COLORS = {
    "FCFS":          "#4C72B0",
    "SJF_Oracle":    "#B0B0B0",
    "SJF_Predicted": "#55A868",
    "SJF_Pred":      "#55A868",   # alias used in trainer.py
    "RoundRobin":    "#C44E52",
    "Priority":      "#8172B2",
    "RL_Adaptive":   "#FF8C00",
    "MLFQ":          "#1A9850",
    "Lottery":       "#F46D43",
    "EDF":           "#74ADD1",
}
DEFAULT_COLOR = "#95A5A6"

SCHEDULER_DISPLAY = {
    "SJF_Pred":    "SJF_Pred",
    "FCFS":        "FCFS",
    "RoundRobin":  "RoundRobin",
    "Priority":    "Priority",
    "MLFQ":        "MLFQ",
    "EDF":         "EDF",
    "Lottery":     "Lottery",
}


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Boxplots
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(
    trial_data:   Dict[str, List[float]],
    output_path:  str  = "results/boxplots.png",
    metric_label: str  = "Average Waiting Time (ticks)",
    title:        str  = "Scheduler Performance Distribution — {n} Trials",
    show_plot:    bool = False,
) -> None:
    """
    Side-by-side boxplots of per-trial metric values for each algorithm.

    Parameters
    ----------
    trial_data   : Dict mapping algorithm name → list of per-trial values.
                   e.g. {"FCFS": [201, 198, ...], "RL_Adaptive": [148, 151, ...]}
                   Typically the "wait" lists from run_multi_trial_evaluation.
    output_path  : Save path.
    metric_label : Y-axis label.
    title        : Figure title ({n} is replaced with trial count).
    show_plot    : Display interactively (for Jupyter).

    Academic note
    -------------
    Boxplots are standard in systems/ML papers for showing distribution
    shape beyond mean ± std.  The median line, IQR box, and whiskers
    give the reviewer confidence that results are not driven by outliers.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Sort algorithms by median value (ascending = best first)
    ordered = sorted(trial_data.items(), key=lambda kv: float(np.median(kv[1])))
    names   = [k for k, _ in ordered]
    data    = [v for _, v in ordered]
    colors  = [COLORS.get(n, DEFAULT_COLOR) for n in names]
    n_trials = max(len(v) for v in data) if data else 0

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.6), 7),
                           constrained_layout=True)
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FDFEFE")

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        vert=True,
        widths=0.55,
        medianprops=dict(color="#2C3E50", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4, linestyle="--", color="#555"),
        capprops=dict(linewidth=1.8, color="#555"),
        flierprops=dict(marker="o", markersize=4, alpha=0.5,
                        markeredgewidth=0.5),
    )

    # Colour each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Annotate median value above each box
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        med = float(np.median(vals))
        ax.text(i, med + ax.get_ylim()[1] * 0.005 if ax.get_ylim()[1] > 0 else med + 2,
                f"{med:.0f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2C3E50")

    # Mark oracle as unfair
    if "SJF_Oracle" in names:
        idx = names.index("SJF_Oracle") + 1
        ax.text(idx, ax.get_ylim()[0], "oracle\n(unfair)",
                ha="center", va="bottom", fontsize=8,
                color="#888", fontstyle="italic")

    # Highlight RL_Adaptive
    if "RL_Adaptive" in names:
        idx = names.index("RL_Adaptive") + 1
        ax.axvline(idx, color="#FF8C00", linewidth=1.5,
                   linestyle=":", alpha=0.7, zorder=0)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title.format(n=n_trials), fontsize=14,
                 fontweight="bold", color="#2C3E50")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_els = [
        mpatches.Patch(color=COLORS.get(n, DEFAULT_COLOR),
                       alpha=0.75, label=n)
        for n in names
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=min(len(names), 5),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    _save(fig, output_path, show_plot)
    print(f"  Boxplot chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 — Convergence Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_curve(
    episode_rewards:    List[float],
    output_path:        str  = "results/convergence_curve.png",
    smoothing_windows:  List[int] = None,
    curriculum_fracs:   List[float] = None,
    show_plot:          bool = False,
) -> None:
    """
    RL training convergence with multiple smoothing windows and annotations.

    This is a richer version of the basic training_curve.png in plots.py:
      • Multiple smoothing windows (10, 50, 100 episodes)
      • Shaded band showing ±1 rolling std (uncertainty)
      • Phase annotations (Exploration / Exploitation / Convergence)
      • Horizontal reference lines for SJF_Pred and FCFS baselines
      • Milestone markers: first time agent beats FCFS

    Parameters
    ----------
    episode_rewards  : List of per-episode total rewards from trainer.
    output_path      : Save path.
    smoothing_windows: Rolling average window sizes to overlay.
    curriculum_fracs : Fractions at which curriculum stage changes.
    show_plot        : Display interactively.

    Academic note
    -------------
    Showing convergence speed (inflection point) matters as much as
    final performance.  "The agent converged in ~120 episodes" is a
    concrete claim reviewers can evaluate.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if smoothing_windows is None:
        smoothing_windows = [10, 50]
    if curriculum_fracs is None:
        curriculum_fracs = [0.33, 0.66]

    rewards = np.array(episode_rewards, dtype=float)
    n       = len(rewards)
    eps     = np.arange(1, n + 1)

    fig, (ax_main, ax_hist) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#F8F9FA")

    # ── left panel: convergence curve ────────────────────────────────────
    ax_main.set_facecolor("#FDFEFE")
    ax_main.plot(eps, rewards, color="#D0D0D0", linewidth=0.6,
                 alpha=0.5, label="Raw episode reward", zorder=2)

    win_colors = ["#4C72B0", "#FF8C00", "#E74C3C"]
    for wi, (w, wc) in enumerate(zip(smoothing_windows, win_colors)):
        if n >= w:
            ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
            x  = np.arange(w, n + 1)
            ax_main.plot(x, ma, color=wc, linewidth=2.2,
                         label=f"{w}-ep MA", zorder=4)
            # Shaded ±1 std band for widest window only
            if wi == len(smoothing_windows) - 1:
                std = np.array([rewards[max(0, i-w):i].std() for i in range(w, n+1)])
                ax_main.fill_between(x, ma - std, ma + std,
                                     color=wc, alpha=0.12, zorder=1)

    # Reference baselines (SJF_Pred ≈ -147, FCFS ≈ -207)
    _hline(ax_main, -147, "#55A868", "SJF_Pred baseline (~−147)", n)
    _hline(ax_main, -207, "#4C72B0", "FCFS baseline (~−207)",     n)

    # Mark first episode where MA beats FCFS
    w_ref = smoothing_windows[-1] if len(smoothing_windows) else 20
    if n >= w_ref:
        ma_ref = np.convolve(rewards, np.ones(w_ref) / w_ref, mode="valid")
        better = np.where(ma_ref > -207)[0]
        if len(better) > 0:
            ep_beat = better[0] + w_ref
            ax_main.axvline(ep_beat, color="#27AE60", linewidth=1.5,
                            linestyle="--", alpha=0.7)
            ax_main.text(ep_beat + 2, ax_main.get_ylim()[0] * 0.97,
                         f"Beats FCFS\nep {ep_beat}", fontsize=8,
                         color="#27AE60", va="bottom")

    # Curriculum stage boundaries
    phase_names = ["Exploration", "Mixed", "Hard / Exploitation"]
    for i, (frac, pname) in enumerate(zip(curriculum_fracs, phase_names[1:])):
        x_line = int(n * frac)
        ax_main.axvline(x_line, color="#AAAAAA", linewidth=1,
                        linestyle=":", alpha=0.8, zorder=0)
        ax_main.text(x_line + 2, ax_main.get_ylim()[1] * 0.97 if ax_main.get_ylim()[1] < 0
                     else ax_main.get_ylim()[1],
                     pname, fontsize=8, color="#666", va="top")

    ax_main.set_title("Contextual Bandit Training Convergence",
                      fontsize=13, fontweight="bold", color="#2C3E50")
    ax_main.set_xlabel("Training Episode", fontsize=11)
    ax_main.set_ylabel("Episode Reward (−avg_waiting_time)", fontsize=11)
    ax_main.legend(fontsize=9, loc="lower right")
    ax_main.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # ── right panel: reward distribution histogram ───────────────────────
    ax_hist.set_facecolor("#FDFEFE")
    last_third = rewards[int(n * 0.66):]   # exploitation phase rewards
    if len(last_third) > 5:
        ax_hist.hist(last_third, bins=20, orientation="horizontal",
                     color="#FF8C00", alpha=0.75, edgecolor="white")
        ax_hist.axhline(float(np.mean(last_third)), color="#2C3E50",
                        linewidth=1.5, linestyle="--",
                        label=f"Mean={np.mean(last_third):.0f}")
    ax_hist.set_title("Reward\nDistribution\n(last 33%)", fontsize=10,
                      fontweight="bold", color="#2C3E50")
    ax_hist.set_xlabel("Count", fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.suptitle(f"RL Agent Training — {n} Episodes",
                 fontsize=15, fontweight="bold", color="#2C3E50")
    _save(fig, output_path, show_plot)
    print(f"  Convergence curve saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7 — Scheduler Selection Frequency
# ─────────────────────────────────────────────────────────────────────────────

def plot_scheduler_selection(
    selection_counts: Dict[str, int],
    output_path:      str  = "results/scheduler_selection.png",
    title:            str  = "Contextual Bandit — Learned Scheduler Preference",
    show_plot:        bool = False,
) -> None:
    """
    Horizontal bar chart of how often each underlying scheduler was chosen
    by the contextual bandit across all training + evaluation episodes.

    Parameters
    ----------
    selection_counts : Dict mapping scheduler name → count of selections.
                       e.g. {"SJF_Pred": 510, "FCFS": 48, ...}
                       Build from trainer.action_frequency after training.
    output_path      : Save path.
    title            : Figure title.
    show_plot        : Display interactively.

    Academic note
    -------------
    This chart provides direct interpretability evidence:
      "The agent learned to select SJF_Pred in 89% of episodes,
       demonstrating that LSTM burst prediction drives the policy."
    It directly supports the LSTM→Bandit coupling narrative.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not selection_counts:
        print("  [extended_plots] No selection counts — skipping chart 7.")
        return

    total  = max(sum(selection_counts.values()), 1)
    # Sort by count descending
    items  = sorted(selection_counts.items(), key=lambda kv: -kv[1])
    names  = [SCHEDULER_DISPLAY.get(k, k) for k, _ in items]
    counts = [v for _, v in items]
    pcts   = [v / total * 100 for v in counts]
    colors = [COLORS.get(k, DEFAULT_COLOR) for k, _ in items]

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(14, max(4, len(names) * 0.9 + 2)),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#F8F9FA")

    # ── left: horizontal bar ──────────────────────────────────────────────
    ax_bar.set_facecolor("#FDFEFE")
    y_pos = range(len(names))
    bars = ax_bar.barh(list(y_pos), counts,
                       color=colors, alpha=0.85, edgecolor="white",
                       linewidth=1.0, height=0.6)

    for bar, cnt, pct in zip(bars, counts, pcts):
        ax_bar.text(bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,}  ({pct:.1f}%)",
                    va="center", ha="left", fontsize=10, color="#2C3E50",
                    fontweight="bold")

    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(names, fontsize=11)
    ax_bar.set_xlabel("Number of Episodes Selected", fontsize=11)
    ax_bar.set_title("Selection Count", fontsize=12,
                     fontweight="bold", color="#2C3E50")
    ax_bar.set_xlim(0, max(counts) * 1.35)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.invert_yaxis()

    # ── right: pie chart (quick visual impression) ────────────────────────
    ax_pie.set_facecolor("#FDFEFE")
    explode = [0.05 if c == max(counts) else 0.0 for c in counts]
    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=names,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
        startangle=90,
        explode=explode,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax_pie.set_title("Selection Share", fontsize=12,
                     fontweight="bold", color="#2C3E50")

    fig.suptitle(f"{title}\n(Total episodes: {total:,})",
                 fontsize=13, fontweight="bold", color="#2C3E50")
    _save(fig, output_path, show_plot)
    print(f"  Scheduler selection chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Chart — Extended Metrics Radar
# ─────────────────────────────────────────────────────────────────────────────

def plot_extended_metrics_radar(
    results:     List[Dict],
    output_path: str  = "results/metrics_radar.png",
    show_plot:   bool = False,
) -> None:
    """
    Radar / spider chart comparing all algorithms across 5 normalised metrics.

    Metrics plotted (all normalised 0–1, higher = better):
      • 1 - normalised avg_waiting_time
      • 1 - normalised avg_response_time
      • fairness_index
      • cpu_utilization / 100
      • 1 - normalised slowdown

    Parameters
    ----------
    results     : List of dicts from compute_extended_metrics().
    output_path : Save path.
    show_plot   : Display interactively.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    AXES = ["Wait\n(lower=↑)", "Response\n(lower=↑)", "Fairness\n(higher=↑)",
            "CPU util\n(higher=↑)", "Slowdown\n(lower=↑)"]
    N = len(AXES)

    def _get_vals(r):
        return [
            r.get("avg_waiting_time",  0),
            r.get("avg_response_time", 0),
            r.get("fairness_index",    1),
            r.get("cpu_utilization",   100) / 100,
            r.get("slowdown",          1),
        ]

    all_vals = [_get_vals(r) for r in results]
    # Normalise each axis independently
    for col in range(N):
        col_vals = [v[col] for v in all_vals]
        mn, mx = min(col_vals), max(col_vals)
        for row in all_vals:
            if mx > mn:
                row[col] = (row[col] - mn) / (mx - mn)
            else:
                row[col] = 0.5

    # Invert axes where lower is better
    for row in all_vals:
        row[0] = 1 - row[0]   # wait
        row[1] = 1 - row[1]   # response
        row[4] = 1 - row[4]   # slowdown

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"},
                           constrained_layout=True)
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FDFEFE")

    for r, vals in zip(results, all_vals):
        name   = r.get("algorithm", "?")
        color  = COLORS.get(name, DEFAULT_COLOR)
        values = vals + vals[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#888")
    ax.grid(color="#CCCCCC", linewidth=0.8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.9)
    ax.set_title("Multi-Metric Scheduler Comparison\n(normalised, higher = better)",
                 fontsize=13, fontweight="bold", color="#2C3E50", pad=20)

    _save(fig, output_path, show_plot)
    print(f"  Radar chart saved → {output_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _hline(ax, y, color, label, n):
    ax.axhline(y, color=color, linewidth=1.2, linestyle=":",
               alpha=0.7, zorder=1, label=label)


def _save(fig, path, show):
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
