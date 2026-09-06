"""
visualization/plots.py
=======================
Four publication-quality charts for the final year project report.

Chart 1 — comparison_single_run.png
    Bar chart of all 6 algorithms on one workload run.

Chart 2 — training_curve.png
    RL reward per episode with 20-episode moving average.
    Should show improving trend in exploitation phase.

Chart 3 — lstm_accuracy.png
    Per-process absolute prediction error + rolling MAE window.
    Demonstrates the LSTM is genuinely learning, not guessing.

Chart 4 — statistical_comparison.png  ← KEY RESULT for report / viva
    Mean ± std error bars across 30 independent trials.
    This is the figure examiners will scrutinise most carefully.
    No single-run luck — if RL is better here, it's really better.
"""

import os
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {
    "FCFS":          "#4C72B0",
    "SJF_Oracle":    "#B0B0B0",
    "SJF_Predicted": "#55A868",
    "RoundRobin":    "#C44E52",
    "Priority":      "#8172B2",
    "RL_Adaptive":   "#FF8C00",
}
DEFAULT = "#95A5A6"


# ── Chart 1: single-run comparison ───────────────────────────────────────────

def plot_comparison(results: List[Dict],
                    output_path: str = "results/comparison_single_run.png",
                    show_plot: bool = False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    names = [r["algorithm"] for r in results]
    waits = [r.get("avg_waiting_time", 0) for r in results]
    turns = [r.get("avg_turnaround_time", 0) for r in results]
    clrs  = [COLORS.get(n, DEFAULT) for n in names]
    x     = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, vals, title in [
        (ax1, waits, "Average Waiting Time"),
        (ax2, turns, "Average Turnaround Time"),
    ]:
        ax.set_facecolor("#FDFEFE")
        bars = ax.bar(x, vals, color=clrs, edgecolor="white", linewidth=1.2, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+max(vals)*0.01,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#2C3E50")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_","\n") for n in names], fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#2C3E50")
        ax.set_ylabel("Ticks (lower is better)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    _legend(fig, names)
    fig.suptitle("Scheduler Comparison — Single Run", fontsize=15,
                 fontweight="bold", color="#2C3E50")
    _save(fig, output_path, show_plot)
    print(f"  Chart saved → {output_path}")


# ── Chart 2: training curve ───────────────────────────────────────────────────

def plot_training_curve(episode_rewards: List[float],
                        output_path: str = "results/training_curve.png",
                        show_plot: bool = False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FDFEFE")
    eps = list(range(1, len(episode_rewards)+1))
    ax.plot(eps, episode_rewards, color="#BDC3C7", linewidth=0.7,
            alpha=0.5, label="Episode reward")
    w = 20
    if len(episode_rewards) >= w:
        ma = np.convolve(episode_rewards, np.ones(w)/w, mode="valid")
        ax.plot(range(w, len(episode_rewards)+1), ma,
                color="#FF8C00", linewidth=2.5, label=f"{w}-ep moving avg")
    # Mark curriculum boundaries
    n = len(episode_rewards)
    for frac, lbl in [(0.33,"Easy→Mixed"),(0.66,"Mixed→Hard")]:
        ax.axvline(int(n*frac), color="#95A5A6", linestyle="--", linewidth=1,
                   alpha=0.6)
        ax.text(int(n*frac)+2, ax.get_ylim()[0]*0.95, lbl,
                fontsize=8, color="#666666", va="bottom")
    ax.axhline(0, color="#E74C3C", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title("RL Agent Training Progress", fontsize=14,
                 fontweight="bold", color="#2C3E50")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Episode Reward")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _save(fig, output_path, show_plot)
    print(f"  Training curve saved → {output_path}")


# ── Chart 3: LSTM accuracy ────────────────────────────────────────────────────

def plot_prediction_accuracy(prediction_errors: list,
                             output_path: str = "results/lstm_accuracy.png",
                             show_plot: bool = False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not prediction_errors:
        print("  No prediction errors to plot."); return

    errors = np.array(prediction_errors)
    idxs   = np.arange(1, len(errors)+1)
    mae    = float(np.mean(errors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor("#F8F9FA")

    # Left: scatter + trend
    ax1.set_facecolor("#FDFEFE")
    ax1.scatter(idxs, errors, color="#4C72B0", alpha=0.4, s=15, zorder=3,
                label="Per-process |error|")
    if len(errors) > 4:
        z = np.polyfit(idxs, errors, 1)
        ax1.plot(idxs, np.poly1d(z)(idxs), color="#E74C3C", linewidth=2,
                 linestyle="--", label=f"Trend (slope={z[0]:+.3f})")
    ax1.axhline(mae, color="#FF8C00", linewidth=1.5, linestyle=":",
                label=f"MAE={mae:.2f}")
    ax1.set_title("LSTM: Per-Process Prediction Error",
                  fontsize=13, fontweight="bold", color="#2C3E50")
    ax1.set_xlabel("Completion index"); ax1.set_ylabel("|Predicted − Actual| (ticks)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Right: rolling MAE
    ax2.set_facecolor("#FDFEFE")
    w = min(20, max(3, len(errors)//5))
    rolling = np.convolve(errors, np.ones(w)/w, mode="valid")
    ax2.plot(range(w, len(errors)+1), rolling, color="#FF8C00", linewidth=2.5,
             label=f"Rolling MAE (w={w})")
    ax2.axhline(mae, color="#95A5A6", linewidth=1, linestyle=":",
                label=f"Overall MAE={mae:.2f}")
    ax2.set_title("LSTM Learning Curve — Rolling MAE",
                  fontsize=13, fontweight="bold", color="#2C3E50")
    ax2.set_xlabel("Completion index"); ax2.set_ylabel("Rolling MAE (ticks)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("LSTM Burst Time Predictor — Accuracy Analysis",
                 fontsize=15, fontweight="bold", color="#2C3E50")
    _save(fig, output_path, show_plot)
    print(f"  LSTM accuracy chart saved → {output_path}")


# ── Chart 4: statistical comparison (KEY RESULT) ──────────────────────────────

def plot_statistical_comparison(summary: Dict,
                                output_path: str = "results/statistical_comparison.png",
                                show_plot: bool = False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ordered = sorted(summary.items(), key=lambda x: x[1]["mean_wait"])
    algos   = [k for k,_ in ordered]
    mw      = np.array([v["mean_wait"] for _,v in ordered])
    sw      = np.array([v["std_wait"]  for _,v in ordered])
    mt      = np.array([v["mean_turn"] for _,v in ordered])
    st      = np.array([v["std_turn"]  for _,v in ordered])
    clrs    = [COLORS.get(a, DEFAULT) for a in algos]
    x       = np.arange(len(algos))
    n       = list(summary.values())[0].get("n", 30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, means, stds, title in [
        (ax1, mw, sw, "Average Waiting Time"),
        (ax2, mt, st, "Average Turnaround Time"),
    ]:
        ax.set_facecolor("#FDFEFE")
        bars = ax.bar(x, means, yerr=stds, capsize=6,
                      color=clrs, alpha=0.87, zorder=3,
                      error_kw={"elinewidth": 2, "ecolor": "#444444"})
        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x()+b.get_width()/2,
                    m + s + max(means)*0.015,
                    f"{m:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#2C3E50")
        # Mark oracle as unfair
        if "SJF_Oracle" in algos:
            i = algos.index("SJF_Oracle")
            ax.text(i, means[i]/2, "oracle\n(unfair)",
                    ha="center", va="center", fontsize=8,
                    color="#888888", fontstyle="italic")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_","\n") for a in algos], fontsize=9)
        ax.set_title(f"{title}  (mean ± std, n={n})",
                     fontsize=13, fontweight="bold", color="#2C3E50")
        ax.set_ylabel("Ticks (lower is better)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    legend_els = [
        mpatches.Patch(color="#B0B0B0", label="SJF_Oracle  — unfair (true burst known)"),
        mpatches.Patch(color="#55A868", label="SJF_Predicted — fair (LSTM estimates)"),
        mpatches.Patch(color="#FF8C00", label="RL_Adaptive  — our agent"),
        mpatches.Patch(color="#4C72B0", label="Classical schedulers"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        f"Statistical Scheduler Comparison — {n} Independent Trials"
        f" (Mixed Adversarial Workload)",
        fontsize=14, fontweight="bold", color="#2C3E50")
    _save(fig, output_path, show_plot)
    print(f"  Statistical comparison chart saved → {output_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _legend(fig, names):
    patches = [mpatches.Patch(color=COLORS.get(n, DEFAULT), label=n) for n in names]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(names), fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))


def _save(fig, path, show):
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show: plt.show()
    plt.close(fig)
