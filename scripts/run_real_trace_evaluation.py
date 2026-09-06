"""
scripts/run_real_trace_evaluation.py
=====================================
Evaluate every scheduler — including the trained RL_Adaptive policy — on
REAL job data instead of the synthetic workload generator.

What this actually tests
-------------------------
The LSTM and the contextual bandit are trained ONCE on synthetic data
(exactly as in scripts/export_results.py — this script does not retrain
them, it loads the already-trained results/lstm_weights.json and
results/q_table.json). This script then asks a different, and arguably
more interesting, question than "does the policy work on more synthetic
data": **does a policy trained only on synthetic workloads generalize to
a real, unseen production job trace it has never encountered?**

That is an out-of-distribution generalization test, not a "trained on
real data" claim — this project does not train on real data, and does not
claim to. If you want to train on real data too, the natural next step is
feeding load_alibaba_pai_trial_windows() into RLTrainer.train() instead of
WorkloadGenerator — that is NOT what this script does.

Data
----
data/real_traces/alibaba_pai_gpu_trace_100k.csv — 100,000 REAL jobs from
Alibaba's PAI GPU cluster (see data/real_traces/PROVENANCE.md for source,
license, and — importantly — exactly what is real vs. derived in this
data: priority is a derived proxy, not a real field; num_inst is
flattened; gpu_type is informational only).

How the 30 "trials" here differ from the synthetic 30-trial protocol
----------------------------------------------------------------------
scripts/export_results.py draws 30 INDEPENDENT random synthetic workloads.
This script instead draws 30 DIFFERENT, non-overlapping, evenly-spaced
windows of ONE real trace (load_alibaba_pai_trial_windows). They are real
and they do cover different real time-of-day/load conditions across the
trace's ~7.7-day span, but they are not textbook-independent samples the
way separately-seeded synthetic draws are (one non-stationary production
system, not 30 i.i.d. draws). The significance tests below are computed
and reported the same way, but read the two result files as answering
different questions, not as one bigger pooled experiment.

Usage
-----
    python scripts/run_real_trace_evaluation.py                 # 30 windows x 150 real jobs
    python scripts/run_real_trace_evaluation.py --quick          # 5 windows, for smoke-testing
    python scripts/run_real_trace_evaluation.py --window-size 300 --n-windows 20

Output
------
    results/real_trace_summary.json
    results/real_trace_comparison.png
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from workload.trace_loader import load_alibaba_pai_trial_windows
from burst_predictor.lstm_predictor import BurstTimePredictor
from rl_agent.q_table import QTable
from rl_agent.trainer import RLTrainer, SCHEDULER_NAMES

from evaluation.experiment import (
    run_all_static_schedulers, run_predicted_sjf, run_static_scheduler,
)
from evaluation.statistics import run_significance_tests

from schedulers.mlfq import MLFQScheduler
from schedulers.lottery import LotteryScheduler
from schedulers.edf import EDFScheduler

from visualization.plots import plot_statistical_comparison

DEFAULT_TRACE = os.path.join(ROOT, "data", "real_traces",
                              "alibaba_pai_gpu_trace_100k.csv")
LSTM_WEIGHTS_PATH = os.path.join(ROOT, "results", "lstm_weights.json")
Q_TABLE_PATH = os.path.join(ROOT, "results", "q_table.json")

EXTRA_SCHEDULER_FACTORY = {
    "MLFQ":    lambda: MLFQScheduler(),
    "Lottery": lambda: LotteryScheduler(),
    "EDF":     lambda: EDFScheduler(),
}


def _load_trained_predictor(path: str) -> "tuple[BurstTimePredictor, bool]":
    """Load the already-trained LSTM from results/lstm_weights.json.
    Returns (predictor, loaded_from_disk)."""
    predictor = BurstTimePredictor()
    if not os.path.exists(path):
        print(f"  [!] {path} not found — using an UNTRAINED LSTM. "
              f"Run scripts/export_results.py first for a meaningful test.")
        return predictor, False
    with open(path) as f:
        w = json.load(f)
    for k, v in w["cell"].items():
        setattr(predictor.cell, k, np.array(v))
    predictor.Wo = np.array(w["output"]["Wo"])
    predictor.bo = np.array(w["output"]["bo"])
    return predictor, True


def _load_trained_q_table(path: str) -> "tuple[QTable, bool]":
    q_table = QTable(learning_rate=0.25, discount_factor=0.0,
                      epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
    if not os.path.exists(path):
        print(f"  [!] {path} not found — using an UNTRAINED Q-table. "
              f"Run scripts/export_results.py first for a meaningful test.")
        return q_table, False
    q_table.load(path)
    return q_table, True


def main(n_windows: int, window_size: int, trace_path: str, results_dir: str):
    t_start = time.time()
    os.makedirs(results_dir, exist_ok=True)

    print("══ Real-trace evaluation: Alibaba PAI GPU cluster job trace ══")
    print(f"  Trace file : {trace_path}")
    if not os.path.exists(trace_path):
        print(f"  [X] Real trace file not found at {trace_path}.")
        print("      See data/real_traces/PROVENANCE.md to fetch it.")
        sys.exit(1)

    predictor, lstm_loaded = _load_trained_predictor(LSTM_WEIGHTS_PATH)
    q_table, qtable_loaded = _load_trained_q_table(Q_TABLE_PATH)
    trained_from_disk = lstm_loaded and qtable_loaded
    # Deep-copy before wiring into RLTrainer: evaluate() can bleed a few
    # prediction-error samples back into the predictor it's given (for MAE
    # bookkeeping) — copying keeps results/lstm_weights.json's own object
    # untouched by this script's real-trace run.
    trainer = RLTrainer(q_table=copy.deepcopy(q_table),
                         predictor=copy.deepcopy(predictor))

    print(f"  Policy source: {'results/*.json (trained on SYNTHETIC data)' if trained_from_disk else 'freshly initialised, UNTRAINED'}")
    print(f"  Loading {n_windows} real windows of {window_size} jobs each "
          f"(non-overlapping, spread across the real trace)...")

    windows = load_alibaba_pai_trial_windows(
        path=trace_path, n_trials=n_windows, window_size=window_size)

    algo_names = ["FCFS", "SJF_Oracle", "RoundRobin", "Priority",
                  "SJF_Predicted", "MLFQ", "Lottery", "EDF", "RL_Adaptive"]
    raw = {name: {"wait": [], "turn": []} for name in algo_names}
    chosen_scheduler_counts = {}

    for i, wl in enumerate(windows):
        for r in run_all_static_schedulers(wl):
            raw[r["algorithm"]]["wait"].append(r["avg_waiting_time"])
            raw[r["algorithm"]]["turn"].append(r["avg_turnaround_time"])

        pr = run_predicted_sjf(copy.deepcopy(trainer.predictor), wl)
        raw["SJF_Predicted"]["wait"].append(pr["avg_waiting_time"])
        raw["SJF_Predicted"]["turn"].append(pr["avg_turnaround_time"])

        for name, factory in EXTRA_SCHEDULER_FACTORY.items():
            r = run_static_scheduler(factory(), wl)
            raw[name]["wait"].append(r["avg_waiting_time"])
            raw[name]["turn"].append(r["avg_turnaround_time"])

        rl = trainer.evaluate(wl)
        raw["RL_Adaptive"]["wait"].append(rl["avg_waiting_time"])
        raw["RL_Adaptive"]["turn"].append(rl["avg_turnaround_time"])
        chosen_scheduler_counts[rl["chosen_scheduler"]] = (
            chosen_scheduler_counts.get(rl["chosen_scheduler"], 0) + 1)

        print(f"    window {i + 1}/{n_windows} done "
              f"(RL chose {rl['chosen_scheduler']})")

    def stats(xs):
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        return round(mu, 3), round(sd, 3)

    trial_summary = {}
    for name, vals in raw.items():
        mw, sw = stats(vals["wait"])
        mt, st = stats(vals["turn"])
        trial_summary[name] = {"mean_wait": mw, "std_wait": sw,
                                "mean_turn": mt, "std_turn": st,
                                "n": len(vals["wait"])}

    wait_lists = {name: vals["wait"] for name, vals in raw.items()}
    significance = run_significance_tests(wait_lists, target="RL_Adaptive")

    realistic = {k: v for k, v in trial_summary.items()
                 if k not in ("SJF_Oracle", "RL_Adaptive")}
    rl_w = trial_summary["RL_Adaptive"]["mean_wait"]
    beaten = sum(1 for v in realistic.values() if v["mean_wait"] > rl_w)

    summary = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "REAL — Alibaba PAI GPU cluster job trace "
                            "(see data/real_traces/PROVENANCE.md)",
            "policy_trained_on": "synthetic data only (out-of-distribution "
                                  "generalization test, not real-data training)",
            "policy_loaded_from_disk": trained_from_disk,
            "n_windows": n_windows,
            "window_size": window_size,
            "trace_path": os.path.relpath(trace_path, ROOT),
            "runtime_seconds": None,
        },
        "trial_summary": trial_summary,
        "significance_tests": significance,
        "chosen_scheduler_counts": chosen_scheduler_counts,
        "performance_analysis": {
            "beaten_count": beaten,
            "total_realistic": len(realistic),
            "rl_mean_wait": rl_w,
            "ranking_by_mean_wait": [k for k, _ in sorted(
                trial_summary.items(), key=lambda kv: kv[1]["mean_wait"])],
        },
    }
    summary["meta"]["runtime_seconds"] = round(time.time() - t_start, 1)

    out_path = os.path.join(results_dir, "real_trace_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    stat_comp_input = {k: v for k, v in trial_summary.items()
                        if k in ("FCFS", "SJF_Oracle", "SJF_Predicted",
                                  "RoundRobin", "Priority", "RL_Adaptive")}
    plot_statistical_comparison(
        stat_comp_input, os.path.join(results_dir, "real_trace_comparison.png"))

    print(f"\n  ══ REAL TRACE results (RL_Adaptive vs {len(realistic)} realistic baselines) ══")
    for name in algo_names:
        v = trial_summary[name]
        print(f"    {name:15s} mean_wait={v['mean_wait']:>10.2f}  std={v['std_wait']:>8.2f}  n={v['n']}")
    print(f"\n  RL_Adaptive beat {beaten}/{len(realistic)} realistic baselines on REAL data.")
    print(f"  Chose: {chosen_scheduler_counts}")
    print(f"\n  Wrote {out_path}")
    print(f"  Total runtime: {summary['meta']['runtime_seconds']}s")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                     help="5 windows instead of 30, for smoke-testing this script")
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--window-size", type=int, default=150)
    ap.add_argument("--trace-path", type=str, default=DEFAULT_TRACE)
    ap.add_argument("--results-dir", type=str,
                     default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    n_windows = args.n_windows or (5 if args.quick else 30)
    main(n_windows=n_windows, window_size=args.window_size,
         trace_path=args.trace_path, results_dir=args.results_dir)
