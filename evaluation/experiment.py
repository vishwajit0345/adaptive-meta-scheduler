"""
evaluation/experiment.py
=========================
Runs all schedulers on identical workloads and collects comparative metrics.

Two evaluation modes
--------------------
Single-run   : run_all_static_schedulers + run_predicted_sjf + trainer.evaluate
               Used for the demonstration table in Phase 3 of main.py.

Multi-trial  : run_multi_trial_evaluation (N=30 independent trials)
               Reports mean ± std — the standard format for academic
               scheduling benchmarks (see e.g. Mao et al., 2019).
               30 trials gives sufficient statistical power to detect a 10%
               performance difference with p < 0.05 (two-sample t-test).
"""

import copy
import math
from typing import List, Dict

from simulator.process import Process
from simulator.ready_queue import ReadyQueue
from simulator.clock import SimulationClock
from simulator.dispatcher import Dispatcher
from schedulers.fcfs import FCFSScheduler
from schedulers.sjf import OracleSJFScheduler
from schedulers.round_robin import RoundRobinScheduler
from schedulers.priority import PriorityScheduler
from schedulers.sjf_predicted import SJFPredictedScheduler


# ── single-scheduler runner ──────────────────────────────────────────────────

def run_static_scheduler(scheduler, processes: List[Process]) -> Dict:
    procs      = [copy.deepcopy(p) for p in processes]
    clock      = SimulationClock()
    rq         = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    if hasattr(scheduler, "reset"): scheduler.reset()
    completed: List[Process] = []
    max_ticks = sum(p.burst_time for p in procs) * 3 + 200
    while len(completed) < len(procs) and clock.current_time <= max_ticks:
        for p in procs:
            if (p.arrival_time == clock.current_time
                    and not p.is_complete() and p not in rq.get_all()):
                rq.add_process(p)
        dispatcher.execute_tick(scheduler.select_next(rq), completed)
    return _metrics(completed, clock, scheduler.name)


def run_predicted_sjf(predictor, processes: List[Process]) -> Dict:
    """
    Run SJF_Predicted with LSTM estimates and online learning.
    Deep-copies the predictor so evaluation doesn't affect the RL agent's weights.
    """
    pred   = copy.deepcopy(predictor)
    procs  = [copy.deepcopy(p) for p in processes]
    clock  = SimulationClock()
    rq     = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    sched  = SJFPredictedScheduler(pred)
    completed: List[Process] = []
    max_ticks = sum(p.burst_time for p in procs) * 3 + 200

    while len(completed) < len(procs) and clock.current_time <= max_ticks:
        for p in procs:
            if (p.arrival_time == clock.current_time
                    and not p.is_complete() and p not in rq.get_all()):
                rq.add_process(p)
        before = len(completed)
        dispatcher.execute_tick(sched.select_next(rq), completed)
        for fp in completed[before:]:
            pred.record_burst(fp.pid, fp.burst_time)
        if len(completed) > before:
            pred.train_on_buffer(epochs=1)

    metrics = _metrics(completed, clock, "SJF_Predicted")
    metrics["prediction_errors"] = pred.prediction_errors
    return metrics


def run_all_static_schedulers(processes: List[Process]) -> List[Dict]:
    schedulers = [
        FCFSScheduler(),
        OracleSJFScheduler(),
        RoundRobinScheduler(time_quantum=3),
        PriorityScheduler(),
    ]
    return [run_static_scheduler(s, processes) for s in schedulers]


# ── multi-trial statistical evaluation ───────────────────────────────────────

def run_multi_trial_evaluation(
    trainer,
    predictor,
    workload_generator,
    num_trials: int = 30,
    workload_type: str = "mixed",
) -> Dict:
    """
    Run N independent trials; return mean ± std for each algorithm.

    30 trials is the standard minimum for parametric significance testing.
    Results are reported as mean ± std, matching the format used in
    OS scheduling research papers (Mao et al. 2019; Mirhoseini et al. 2019).
    """
    data: Dict[str, Dict[str, List]] = {}

    print(f"\n  Running {num_trials} independent evaluation trials...", flush=True)
    for trial in range(num_trials):
        wl = workload_generator.generate(workload_type=workload_type)

        for r in run_all_static_schedulers(wl):
            n = r["algorithm"]
            if n not in data: data[n] = {"wait": [], "turn": []}
            data[n]["wait"].append(r["avg_waiting_time"])
            data[n]["turn"].append(r["avg_turnaround_time"])

        pr = run_predicted_sjf(copy.deepcopy(predictor), wl)
        n  = "SJF_Predicted"
        if n not in data: data[n] = {"wait": [], "turn": []}
        data[n]["wait"].append(pr["avg_waiting_time"])
        data[n]["turn"].append(pr["avg_turnaround_time"])

        rl = trainer.evaluate(wl)
        n  = "RL_Adaptive"
        if n not in data: data[n] = {"wait": [], "turn": []}
        data[n]["wait"].append(rl["avg_waiting_time"])
        data[n]["turn"].append(rl["avg_turnaround_time"])

        if (trial + 1) % 10 == 0:
            print(f"    {trial+1}/{num_trials} trials complete", flush=True)

    summary = {}
    for algo, vals in data.items():
        for metric, key in [("wait","mean_wait"), ("turn","mean_turn")]:
            xs = vals[metric]
            mu = sum(xs) / len(xs)
            sd = math.sqrt(sum((x-mu)**2 for x in xs) / len(xs))
            std_key = "std_" + metric.replace("wait","wait").replace("turn","turn")
            if algo not in summary: summary[algo] = {"n": len(xs)}
            summary[algo][key] = round(mu, 2)
            summary[algo]["std_" + metric] = round(sd, 2)
    return summary


# ── display ──────────────────────────────────────────────────────────────────

def print_comparison_table(results: List[Dict]):
    """Print single-run comparison table."""
    print(f"\n{'═'*82}")
    print(f"  {'ALGORITHM':<22} {'AVG WAITING':>14} {'AVG TURNAROUND':>16}"
          f" {'CPU%':>6} {'THRU':>10}")
    print(f"{'═'*82}")
    for rank, r in enumerate(
            sorted(results, key=lambda x: x.get("avg_waiting_time", 9e9)), 1):
        star = "★ " if rank == 1 else "  "
        tag  = (" [oracle]" if r["algorithm"] == "SJF_Oracle"
                else " [LSTM]  " if r["algorithm"] == "SJF_Predicted"
                else " [RL]    " if r["algorithm"] == "RL_Adaptive"
                else "         ")
        print(f"{star}{r['algorithm']:<22}{tag}"
              f" {str(r.get('avg_waiting_time','N/A')):>14}"
              f" {str(r.get('avg_turnaround_time','N/A')):>16}"
              f" {str(r.get('cpu_utilization','N/A')):>6}"
              f" {str(r.get('throughput','N/A')):>10}")
    print(f"{'═'*82}")
    print(f"  ★=Best  [oracle]=unfair  [LSTM]=fair  [RL]=our agent\n")


def print_statistical_table(summary: Dict, n_trials: int = 30):
    """Print publication-style mean±std table."""
    print(f"\n{'═'*80}")
    print(f"  {'ALGORITHM':<20} {'WAIT (mean±std)':>20} {'TURNAROUND (mean±std)':>24}  N")
    print(f"{'═'*80}")
    for algo, s in sorted(summary.items(), key=lambda x: x[1]["mean_wait"]):
        rank_star = "★ " if list(sorted(summary, key=lambda x: summary[x]["mean_wait"]))[0] == algo else "  "
        tag = (" [oracle]" if algo == "SJF_Oracle"
               else " [LSTM]  " if algo == "SJF_Predicted"
               else " [RL]    " if algo == "RL_Adaptive"
               else "         ")
        w = f"{s['mean_wait']:.1f}±{s['std_wait']:.1f}"
        t = f"{s['mean_turn']:.1f}±{s['std_turn']:.1f}"
        print(f"{rank_star}{algo:<20}{tag} {w:>20} {t:>24}  {s['n']}")
    print(f"{'═'*80}")
    print(f"  ★=Best  [oracle]=unfair  [LSTM]=fair  [RL]=our agent")
    print(f"  Reported as mean±std across {n_trials} independent random trials\n")


# ── private ──────────────────────────────────────────────────────────────────

def _metrics(completed, clock, name) -> Dict:
    if not completed:
        return {"algorithm": name, "avg_waiting_time": float("inf"),
                "avg_turnaround_time": float("inf"),
                "cpu_utilization": 0.0, "throughput": 0.0,
                "completed_count": 0}
    aw = sum(p.waiting_time    for p in completed) / len(completed)
    at = sum(p.turnaround_time for p in completed) / len(completed)
    tp = len(completed) / clock.current_time if clock.current_time > 0 else 0.0
    return {"algorithm": name,
            "avg_waiting_time":    round(aw, 3),
            "avg_turnaround_time": round(at, 3),
            "cpu_utilization":     round(clock.cpu_utilization * 100, 2),
            "throughput":          round(tp, 4),
            "completed_count":     len(completed)}
