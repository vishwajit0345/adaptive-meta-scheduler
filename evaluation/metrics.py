"""
evaluation/metrics.py
======================
Extended evaluation metrics for scheduler comparison.

This module ADDS to the existing metrics in experiment.py — it does NOT
replace or modify them.  The existing _metrics() function in experiment.py
already computes: avg_waiting_time, avg_turnaround_time, cpu_utilization,
throughput, completed_count.

This module adds:
  • avg_response_time      — mean time from arrival to first CPU touch
  • p95_waiting_time       — 95th-percentile waiting time (tail latency)
  • p99_waiting_time       — 99th-percentile waiting time
  • fairness_index         — Jain's fairness index on waiting times
  • slowdown               — mean(turnaround / burst) ratio
  • starvation_count       — number of processes with waiting > 3× burst
  • makespan               — total simulation time (finish - first_arrival)
  • context_switch_count   — from SimulationClock

WHY THESE METRICS MATTER
--------------------------
avg_waiting_time alone is not sufficient for a complete scheduler evaluation:

  1. response_time is critical for interactive systems — a process that
     starts quickly FEELS fast even if it takes a while to complete.

  2. p95/p99 waiting time captures tail latency — a key metric in
     production systems (Google SRE, Amazon).  Two schedulers with the
     same average but different tails have very different user experiences.

  3. Jain's Fairness Index (Jain et al., 1984) measures whether CPU
     time is distributed equitably.  A scheduler that achieves low
     average wait by starving some processes has poor fairness.

  4. Slowdown (bounded/normalised turnaround) is used in cluster
     scheduling papers (Mao et al., 2019) as the primary objective.

USAGE
-----
    from evaluation.metrics import compute_extended_metrics, print_extended_table

    # After running a simulation:
    ext = compute_extended_metrics(completed_processes, clock)
    # ext contains all original metrics + the additions above

    # Or enrich an existing result dict (from experiment.py):
    from evaluation.metrics import enrich_result
    enriched = enrich_result(result_dict, completed_processes, clock)
"""

import math
from typing import List, Dict, Optional
from simulator.process import Process
from simulator.clock   import SimulationClock


# ── core computation ──────────────────────────────────────────────────────────

def compute_extended_metrics(
    completed: List[Process],
    clock:     SimulationClock,
    algo_name: str = "",
) -> Dict:
    """
    Compute the full suite of scheduler performance metrics.

    Parameters
    ----------
    completed  : List of Process objects that finished the simulation.
    clock      : SimulationClock — carries CPU utilisation, context switches.
    algo_name  : Algorithm label for the result dict.

    Returns
    -------
    Dict with keys:
        algorithm
        completed_count
        avg_waiting_time       — mean waiting time (ticks)
        std_waiting_time       — std dev of waiting times
        p95_waiting_time       — 95th-percentile waiting time
        p99_waiting_time       — 99th-percentile waiting time
        avg_turnaround_time    — mean turnaround time (ticks)
        avg_response_time      — mean time from arrival to first CPU tick
        slowdown               — mean(turnaround / burst_time)
        starvation_count       — processes with wait > 3 × burst
        fairness_index         — Jain's fairness index ∈ (0, 1]
        makespan               — total elapsed simulation ticks
        throughput             — completed processes per tick
        cpu_utilization        — fraction of ticks CPU was busy (%)
        context_switch_count   — number of scheduler preemptions
    """
    if not completed:
        return _empty_metrics(algo_name, clock)

    n = len(completed)

    # ── waiting time stats ────────────────────────────────────────────────
    waits  = [p.waiting_time for p in completed]
    wm     = sum(waits) / n
    wvar   = sum((w - wm) ** 2 for w in waits) / n
    wstd   = math.sqrt(wvar)
    wsort  = sorted(waits)
    p95_w  = _percentile(wsort, 95)
    p99_w  = _percentile(wsort, 99)

    # ── turnaround time ────────────────────────────────────────────────────
    turns  = [p.turnaround_time for p in completed]
    tm     = sum(turns) / n

    # ── response time (first CPU touch) ───────────────────────────────────
    # Process.response_time is set by Dispatcher when start_time is recorded.
    # Fall back to waiting_time if not set (e.g. FCFS non-preemptive = equal).
    resp_times = []
    for p in completed:
        rt = p.response_time if (hasattr(p, "response_time") and p.response_time >= 0) \
             else p.waiting_time
        resp_times.append(rt)
    rm = sum(resp_times) / n

    # ── slowdown = normalised turnaround ───────────────────────────────────
    # slowdown = turnaround / burst.  Min = 1.0 (ideal).  Larger = worse.
    slowdowns = [p.turnaround_time / max(p.burst_time, 1) for p in completed]
    slowdown  = sum(slowdowns) / n

    # ── starvation count ───────────────────────────────────────────────────
    starved = sum(1 for p in completed if p.waiting_time > 3 * p.burst_time)

    # ── Jain's fairness index ─────────────────────────────────────────────
    # J = (Σ xᵢ)² / (n · Σ xᵢ²)   where xᵢ = turnaround / burst (slowdown)
    # J = 1.0: perfectly fair.  J → 0: heavily unfair.
    sum_sd  = sum(slowdowns)
    sum_sd2 = sum(s * s for s in slowdowns)
    fairness = (sum_sd ** 2) / (n * sum_sd2) if sum_sd2 > 0 else 1.0

    # ── makespan ───────────────────────────────────────────────────────────
    first_arr = min(p.arrival_time for p in completed)
    last_fin  = max(p.finish_time  for p in completed) if all(
        p.finish_time >= 0 for p in completed) else clock.current_time
    makespan  = last_fin - first_arr

    # ── throughput ────────────────────────────────────────────────────────
    tp = n / clock.current_time if clock.current_time > 0 else 0.0

    return {
        "algorithm":           algo_name,
        "completed_count":     n,
        # ── core ──
        "avg_waiting_time":    round(wm,       3),
        "std_waiting_time":    round(wstd,     3),
        "p95_waiting_time":    round(p95_w,    3),
        "p99_waiting_time":    round(p99_w,    3),
        "avg_turnaround_time": round(tm,       3),
        "avg_response_time":   round(rm,       3),
        # ── quality ──
        "slowdown":            round(slowdown, 4),
        "starvation_count":    starved,
        "fairness_index":      round(fairness, 4),
        # ── system ──
        "makespan":            makespan,
        "throughput":          round(tp,       5),
        "cpu_utilization":     round(clock.cpu_utilization * 100, 2),
        "context_switch_count":clock.context_switches,
    }


def enrich_result(
    result:    Dict,
    completed: List[Process],
    clock:     SimulationClock,
) -> Dict:
    """
    Take an existing result dict (from experiment._metrics) and add the
    extended metrics that weren't in the original dict.

    This is the bridge between the existing experiment.py pipeline and the
    new metrics — no existing code needs to change.

    Parameters
    ----------
    result    : Dict returned by experiment._metrics() — modified in place.
    completed : List of completed Process objects from the same simulation.
    clock     : SimulationClock from the same simulation.

    Returns
    -------
    The same dict with new keys added (existing keys preserved).
    """
    ext = compute_extended_metrics(completed, clock, result.get("algorithm", ""))
    # Merge — new keys only; don't overwrite anything the existing code set
    for key, val in ext.items():
        if key not in result:
            result[key] = val
    return result


# ── printing ──────────────────────────────────────────────────────────────────

def print_extended_table(results: List[Dict]) -> None:
    """
    Print a rich multi-metric comparison table.

    Columns: algorithm | avg_wait | p95_wait | response | slowdown | fairness | CS_count
    """
    hdr = (f"{'ALGORITHM':<22} {'AVG WAIT':>9} {'P95 WAIT':>9} "
           f"{'RESPONSE':>9} {'SLOWDOWN':>9} {'FAIRNESS':>9} {'C.SW':>6}")
    sep = "═" * len(hdr)
    print(f"\n{sep}")
    print(f"  {hdr}")
    print(f"{sep}")

    for r in sorted(results, key=lambda x: x.get("avg_waiting_time", 9e9)):
        name = r.get("algorithm", "?")[:22]
        print(
            f"  {name:<22}"
            f" {r.get('avg_waiting_time',   'N/A'):>9}"
            f" {r.get('p95_waiting_time',   'N/A'):>9}"
            f" {r.get('avg_response_time',  'N/A'):>9}"
            f" {r.get('slowdown',           'N/A'):>9}"
            f" {r.get('fairness_index',     'N/A'):>9}"
            f" {r.get('context_switch_count','N/A'):>6}"
        )
    print(f"{sep}")
    print("  Columns: avg_wait | p95_wait | first_response | slowdown "
          "(lower=better) | fairness (higher=better) | context_switches\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _percentile(sorted_vals: List[float], pct: int) -> float:
    """Linear interpolation percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    n   = len(sorted_vals)
    idx = (pct / 100) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _empty_metrics(algo_name: str, clock: SimulationClock) -> Dict:
    keys = ["avg_waiting_time", "std_waiting_time", "p95_waiting_time",
            "p99_waiting_time", "avg_turnaround_time", "avg_response_time",
            "slowdown", "fairness_index", "makespan", "throughput",
            "cpu_utilization", "starvation_count", "context_switch_count"]
    d = {"algorithm": algo_name, "completed_count": 0}
    for k in keys:
        d[k] = 0.0
    return d
