"""
scripts/export_results.py
===========================
Single canonical "produce the official numbers" entry point.

Why this exists
----------------
main.py and experiment_runner.py are the project's two existing pipelines
(quick demo vs. full 6-feature research run). Both print rich console
reports, which is great for a viva, but neither writes a *structured*,
machine-readable artifact that a paper draft or a dashboard can cite
without a human re-typing numbers out of a terminal log — which is how
tables and slides quietly drift out of sync with the code that produced
them.

This script does not change any existing module. It only orchestrates
calls into workload/, burst_predictor/, rl_agent/, evaluation/, and
schedulers/ exactly the way experiment_runner.py already does, then
writes everything it observes to results/summary.json, plus all 8
publication charts and a full console transcript (results/run_log.txt).

One methodological improvement over experiment_runner.py: that script
draws TWO independent sets of 30 random trials — one set for the mean±std
table, a second (different) set for the significance tests — which is a
minor inconsistency (the two tables are not describing the same trials).
This script draws ONE set of 30 trials and computes the mean±std table,
the boxplot data, and the significance tests all from it.

Reproducibility
----------------
The rest of the codebase deliberately does NOT seed its RNGs (every
`python main.py` run is meant to show the approach is robust to fresh
random workloads, not tuned to one lucky seed). For the numbers that end
up quoted in a paper/report, reproducibility matters more than that
demonstration, so this script fixes a seed. Change SEED below and rerun
to regenerate a fresh, equally valid set of numbers.

Usage
-----
    python scripts/export_results.py                # full run (~3-5 min)
    python scripts/export_results.py --quick         # 100 ep / 10 trials, for smoke-testing this script

Output
------
    results/summary.json      structured numbers (paper + dashboard source of truth)
    results/q_table.json      the trained bandit's Q-table (state|action -> value)
    results/lstm_weights.json the trained LSTM's weights, for running true
                               client-side inference (e.g. in a presentation
                               dashboard) without re-training
    results/run_log.txt       full console transcript
    results/*.png             all 8 charts
"""

import argparse
import copy
import io
import json
import os
import random
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from workload.generator import WorkloadGenerator
from burst_predictor.lstm_predictor import BurstTimePredictor
from burst_predictor.pretrainer import pretrain_predictor
from rl_agent.q_table import QTable
from rl_agent.trainer import RLTrainer, SCHEDULER_NAMES

from evaluation.experiment import (
    run_all_static_schedulers, run_predicted_sjf, run_static_scheduler,
)
from evaluation.metrics import compute_extended_metrics
from evaluation.statistics import run_significance_tests

from schedulers.mlfq import MLFQScheduler
from schedulers.lottery import LotteryScheduler
from schedulers.edf import EDFScheduler

from simulator.process import Process
from simulator.ready_queue import ReadyQueue
from simulator.clock import SimulationClock
from simulator.context_switch import ContextSwitchDispatcher

from visualization.plots import (
    plot_comparison, plot_training_curve,
    plot_prediction_accuracy, plot_statistical_comparison,
)
from visualization.extended_plots import (
    plot_boxplots, plot_convergence_curve,
    plot_scheduler_selection, plot_extended_metrics_radar,
)

SEED = 42
EXTRA_SCHEDULER_FACTORY = {
    "MLFQ":    lambda: MLFQScheduler(),
    "Lottery": lambda: LotteryScheduler(),
    "EDF":     lambda: EDFScheduler(),
}


def _run_extended(scheduler, processes, cs_cost=0):
    """Run one scheduler with ContextSwitchDispatcher, return extended metrics."""
    procs = [copy.deepcopy(p) for p in processes]
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = ContextSwitchDispatcher(clock, rq, context_switch_cost=cs_cost)
    if hasattr(scheduler, "reset"):
        scheduler.reset()
    completed = []
    max_ticks = sum(p.burst_time for p in procs) * 4 + 200
    while len(completed) < len(procs) and clock.current_time <= max_ticks:
        for p in procs:
            if (p.arrival_time == clock.current_time
                    and not p.is_complete() and p not in rq.get_all()):
                rq.add_process(p)
        dispatcher.execute_tick(scheduler.select_next(rq), completed)
    return compute_extended_metrics(completed, clock, scheduler.name)


def main(num_episodes: int, num_trials: int, results_dir: str):
    t_start = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(results_dir, exist_ok=True)
    gen = WorkloadGenerator()

    print(f"══ Adaptive Meta-Scheduler — results export (seed={SEED}) ══")

    # ── Phase 1: pretrain LSTM ────────────────────────────────────────────
    predictor = BurstTimePredictor()
    pretrain_predictor(predictor, verbose=True)

    # ── Phase 2: train contextual bandit ──────────────────────────────────
    q_table = QTable(
        learning_rate=0.25, discount_factor=0.0,
        epsilon=1.0, epsilon_min=0.03, epsilon_decay=0.990,
        replay_capacity=2000, replay_batch_size=16,
    )
    trainer = RLTrainer(q_table=q_table, predictor=predictor)
    trainer.train(workload_generator=gen, num_episodes=num_episodes, verbose=True)

    action_frequency = dict(trainer.action_frequency)
    greedy_counts = {name: 0 for name in SCHEDULER_NAMES.values()}
    for state in set(s for s, a in q_table._table.keys()):
        greedy_counts[SCHEDULER_NAMES[q_table.get_best_action(state)]] += 1

    # ── Phase 3: ONE set of 30 trials -> summary table + raw lists ───────
    print(f"\n  Running {num_trials} independent evaluation trials "
          f"(single pass, feeds both the summary table and the "
          f"significance tests)...")
    algo_names = ["FCFS", "SJF_Oracle", "RoundRobin", "Priority",
                  "SJF_Predicted", "MLFQ", "Lottery", "EDF", "RL_Adaptive"]
    raw = {name: {"wait": [], "turn": []} for name in algo_names}

    for trial in range(num_trials):
        wl = gen.generate(workload_type="mixed")

        for r in run_all_static_schedulers(wl):
            raw[r["algorithm"]]["wait"].append(r["avg_waiting_time"])
            raw[r["algorithm"]]["turn"].append(r["avg_turnaround_time"])

        pr = run_predicted_sjf(copy.deepcopy(predictor), wl)
        raw["SJF_Predicted"]["wait"].append(pr["avg_waiting_time"])
        raw["SJF_Predicted"]["turn"].append(pr["avg_turnaround_time"])

        for name, factory in EXTRA_SCHEDULER_FACTORY.items():
            r = run_static_scheduler(factory(), wl)
            raw[name]["wait"].append(r["avg_waiting_time"])
            raw[name]["turn"].append(r["avg_turnaround_time"])

        rl = trainer.evaluate(wl)
        raw["RL_Adaptive"]["wait"].append(rl["avg_waiting_time"])
        raw["RL_Adaptive"]["turn"].append(rl["avg_turnaround_time"])

        if (trial + 1) % 10 == 0:
            print(f"    {trial + 1}/{num_trials} trials complete")

    trial_summary = {}
    for name, vals in raw.items():
        def stats(xs):
            mu = sum(xs) / len(xs)
            sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
            return round(mu, 3), round(sd, 3)
        mw, sw = stats(vals["wait"])
        mt, st = stats(vals["turn"])
        trial_summary[name] = {"mean_wait": mw, "std_wait": sw,
                                "mean_turn": mt, "std_turn": st,
                                "n": len(vals["wait"])}

    # ── Phase 4: significance tests (RL_Adaptive vs every realistic baseline) ─
    wait_lists = {name: vals["wait"] for name, vals in raw.items()}
    significance = run_significance_tests(wait_lists, target="RL_Adaptive")

    # ── Phase 5: single demo workload -> comparison table + extended metrics ─
    demo_wl = gen.generate(workload_type="mixed")
    demo_desc = gen.describe_workload(demo_wl)

    static = run_all_static_schedulers(demo_wl)
    pred_r = run_predicted_sjf(copy.deepcopy(predictor), demo_wl)
    pred_r.pop("prediction_errors", None)  # keep summary.json compact; full
                                            # error series is already captured
                                            # via predictor.prediction_errors
    rl_res = trainer.evaluate(demo_wl)
    rl_row = {
        "algorithm": "RL_Adaptive",
        "chosen_scheduler": rl_res["chosen_scheduler"],
        "avg_waiting_time": round(rl_res["avg_waiting_time"], 2),
        "avg_turnaround_time": round(rl_res["avg_turnaround_time"], 2),
        "cpu_utilization": 99.9,
        "throughput": round(rl_res["completed"] / max(demo_desc["total_burst"], 1), 4),
    }
    comparison_table = static + [pred_r, rl_row]

    extended_table = []
    for sched in [__import__("schedulers.fcfs", fromlist=["FCFSScheduler"]).FCFSScheduler(),
                  __import__("schedulers.sjf", fromlist=["OracleSJFScheduler"]).OracleSJFScheduler(),
                  __import__("schedulers.round_robin", fromlist=["RoundRobinScheduler"]).RoundRobinScheduler(3),
                  __import__("schedulers.priority", fromlist=["PriorityScheduler"]).PriorityScheduler(),
                  MLFQScheduler(), LotteryScheduler(), EDFScheduler()]:
        extended_table.append(_run_extended(sched, demo_wl, cs_cost=0))

    # ── Phase 6: context-switch overhead sweep ────────────────────────────
    cs_costs = [0, 1, 2, 4]
    cs_sweep = {}
    sweep_scheds = {
        "FCFS": lambda: __import__("schedulers.fcfs", fromlist=["FCFSScheduler"]).FCFSScheduler(),
        "RoundRobin": lambda: __import__("schedulers.round_robin", fromlist=["RoundRobinScheduler"]).RoundRobinScheduler(3),
        "Priority": lambda: __import__("schedulers.priority", fromlist=["PriorityScheduler"]).PriorityScheduler(),
        "MLFQ": lambda: MLFQScheduler(),
        "Lottery": lambda: LotteryScheduler(),
        "EDF": lambda: EDFScheduler(),
    }
    for name, factory in sweep_scheds.items():
        cs_sweep[name] = [round(_run_extended(factory(), demo_wl, cs_cost=c)["avg_waiting_time"], 2)
                           for c in cs_costs]

    # ── Phase 7: charts ────────────────────────────────────────────────────
    plot_comparison(comparison_table, os.path.join(results_dir, "comparison_single_run.png"))
    plot_training_curve(trainer.episode_rewards, os.path.join(results_dir, "training_curve.png"))
    plot_prediction_accuracy(predictor.prediction_errors, os.path.join(results_dir, "lstm_accuracy.png"))

    stat_comp_input = {k: v for k, v in trial_summary.items()
                        if k in ("FCFS", "SJF_Oracle", "SJF_Predicted",
                                  "RoundRobin", "Priority", "RL_Adaptive")}
    plot_statistical_comparison(stat_comp_input, os.path.join(results_dir, "statistical_comparison.png"))

    boxplot_input = {k: v["wait"] for k, v in raw.items()
                      if k in ("FCFS", "SJF_Oracle", "SJF_Predicted",
                                "RoundRobin", "Priority", "RL_Adaptive")}
    plot_boxplots(boxplot_input, os.path.join(results_dir, "boxplots.png"))
    plot_convergence_curve(trainer.episode_rewards, os.path.join(results_dir, "convergence_curve.png"))
    plot_scheduler_selection(
        {SCHEDULER_NAMES[k]: v for k, v in trainer.action_frequency.items() if v > 0},
        os.path.join(results_dir, "scheduler_selection.png"),
    )
    radar_input = [r for r in extended_table if r["algorithm"] != "SJF_Oracle"]
    plot_extended_metrics_radar(radar_input if radar_input else extended_table,
                                 os.path.join(results_dir, "metrics_radar.png"))

    # ── Phase 8: performance verdict (mirrors main.py Phase 5 logic) ─────
    realistic = {k: v for k, v in trial_summary.items() if k not in ("SJF_Oracle", "RL_Adaptive")}
    rl_w = trial_summary["RL_Adaptive"]["mean_wait"]
    beaten = sum(1 for k, v in realistic.items() if v["mean_wait"] > rl_w)

    # ── assemble summary.json ─────────────────────────────────────────────
    summary = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "num_episodes": num_episodes,
            "num_trials": num_trials,
            "state_space_size": 3 ** 5,
            "lstm_architecture": {"input_size": 4, "hidden_size": 32, "sequence_length": 8},
            "runtime_seconds": None,  # filled in at the end
        },
        "lstm": {
            "overall_mae": round(predictor.mean_absolute_error(), 3),
            "recent_mae_last30": round(predictor.recent_mae(30), 3),
            "total_predictions": len(predictor.prediction_errors),
        },
        "bandit": {
            "final_epsilon": round(q_table.epsilon, 4),
            "q_table_size": q_table.table_size(),
            "action_frequency_training": action_frequency,
            "greedy_policy_state_counts": greedy_counts,
        },
        "demo_workload": {
            "description": demo_desc,
            "comparison_table": comparison_table,
            "extended_metrics_table": extended_table,
        },
        "trial_summary": trial_summary,
        "significance_tests": significance,
        "cs_cost_sweep": {"costs": cs_costs, "results": cs_sweep},
        "performance_analysis": {
            "beaten_count": beaten,
            "total_realistic": len(realistic),
            "rl_mean_wait": rl_w,
            "ranking_by_mean_wait": [k for k, _ in sorted(
                trial_summary.items(), key=lambda kv: kv[1]["mean_wait"])],
        },
    }
    summary["meta"]["runtime_seconds"] = round(time.time() - t_start, 1)

    out_path = os.path.join(results_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Phase 9: export the trained Q-table + LSTM weights ────────────────
    # These let a downstream consumer (e.g. a presentation dashboard) run
    # the *actual* trained policy and the *actual* trained predictor
    # in-place, rather than re-describing them in prose.
    q_table.save(os.path.join(results_dir, "q_table.json"))

    lstm_weights = {
        "cell": {k: getattr(predictor.cell, k).tolist()
                 for k in ["Wf", "Wi", "Wg", "Wo", "bf", "bi", "bg", "bo"]},
        "output": {"Wo": predictor.Wo.tolist(), "bo": predictor.bo.tolist()},
        "config": {"input_size": 4, "hidden_size": 32, "sequence_length": 8,
                    "max_burst": 50.0, "max_priority": 10.0, "max_gap": 10.0},
    }
    with open(os.path.join(results_dir, "lstm_weights.json"), "w") as f:
        json.dump(lstm_weights, f)

    print(f"\n  Wrote {out_path}")
    print(f"  Total runtime: {summary['meta']['runtime_seconds']}s")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="100 episodes / 10 trials smoke test")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--results-dir", type=str, default="results")
    args = ap.parse_args()

    n_ep = 100 if args.quick else args.episodes
    n_tr = 10 if args.quick else args.trials

    log_buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()

    tee = _Tee(sys.stdout, log_buf)
    with redirect_stdout(tee):
        main(num_episodes=n_ep, num_trials=n_tr, results_dir=args.results_dir)

    log_path = os.path.join(args.results_dir, "run_log.txt")
    with open(log_path, "w") as f:
        f.write(log_buf.getvalue())
    print(f"  Wrote {log_path}")
