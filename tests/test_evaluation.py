"""
tests/test_evaluation.py
==========================
Tests for evaluation/metrics.py (extended per-run metrics) and
evaluation/statistics.py (significance testing).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.process import Process
from simulator.clock import SimulationClock
from evaluation.metrics import compute_extended_metrics
from evaluation.statistics import t_test, wilcoxon_test, run_significance_tests


def _completed_process(pid, arrival, burst, wait, finish):
    p = Process(pid=pid, arrival_time=arrival, burst_time=burst, priority=1)
    p.waiting_time = wait
    p.finish_time = finish
    p.turnaround_time = finish - arrival
    p.response_time = wait  # non-preemptive assumption for this fixture
    return p


# ── evaluation/metrics.py ───────────────────────────────────────────────────

def test_compute_extended_metrics_basic_arithmetic():
    # Two processes, hand-computed expected waiting/turnaround means.
    p1 = _completed_process(0, arrival=0, burst=5, wait=0,  finish=5)
    p2 = _completed_process(1, arrival=0, burst=5, wait=10, finish=20)
    clock = SimulationClock()
    for _ in range(20):
        clock.tick(cpu_was_idle=False)

    m = compute_extended_metrics([p1, p2], clock, algo_name="TEST")
    assert m["algorithm"] == "TEST"
    assert m["completed_count"] == 2
    assert abs(m["avg_waiting_time"] - 5.0) < 1e-9      # (0+10)/2
    assert abs(m["avg_turnaround_time"] - 12.5) < 1e-9  # (5+15)/2


def test_compute_extended_metrics_perfect_fairness_when_slowdowns_equal():
    # Both processes have identical slowdown (turnaround/burst = 2.0)
    p1 = _completed_process(0, arrival=0, burst=10, wait=10, finish=20)
    p2 = _completed_process(1, arrival=0, burst=5,  wait=5,  finish=10)
    clock = SimulationClock()
    for _ in range(20):
        clock.tick(cpu_was_idle=False)

    m = compute_extended_metrics([p1, p2], clock, "TEST")
    assert abs(m["fairness_index"] - 1.0) < 1e-6


def test_compute_extended_metrics_starvation_count():
    # waiting > 3*burst counts as starved
    starved     = _completed_process(0, arrival=0, burst=5,  wait=20, finish=25)
    not_starved = _completed_process(1, arrival=0, burst=10, wait=5,  finish=15)
    clock = SimulationClock()
    for _ in range(25):
        clock.tick(cpu_was_idle=False)

    m = compute_extended_metrics([starved, not_starved], clock, "TEST")
    assert m["starvation_count"] == 1


def test_compute_extended_metrics_empty_completed_list():
    clock = SimulationClock()
    m = compute_extended_metrics([], clock, "EMPTY")
    assert m["completed_count"] == 0
    assert m["avg_waiting_time"] == 0.0


def test_percentile_matches_hand_computation_for_uniform_data():
    from evaluation.metrics import _percentile
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    # p50 of 5 evenly spaced points should land exactly on the middle value
    assert abs(_percentile(data, 50) - 30.0) < 1e-9
    assert abs(_percentile(data, 0)  - 10.0) < 1e-9
    assert abs(_percentile(data, 100) - 50.0) < 1e-9


# ── evaluation/statistics.py ─────────────────────────────────────────────────

def test_t_test_identical_distributions_not_significant():
    a = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.5, 98.5] * 3
    res = t_test(a, list(a))  # same distribution compared to itself
    assert res["p_value"] > 0.9
    assert not res["significant"]


def test_t_test_clearly_separated_distributions_is_significant():
    low  = [10.0 + i * 0.1 for i in range(30)]
    high = [50.0 + i * 0.1 for i in range(30)]
    res = t_test(low, high)
    assert res["significant"]
    assert res["p_value"] < 0.05
    assert res["mean_a"] < res["mean_b"]
    assert res["improvement"] > 0  # a (lower wait) is the "better" group


def test_t_test_requires_minimum_observations():
    import pytest
    with pytest.raises(ValueError):
        t_test([1.0], [1.0, 2.0, 3.0])


def test_wilcoxon_requires_equal_length_paired_samples():
    import pytest
    with pytest.raises(ValueError):
        wilcoxon_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_wilcoxon_detects_consistent_paired_improvement():
    baseline = [50.0 + i for i in range(15)]
    improved = [b - 10.0 for b in baseline]  # consistently 10 ticks better
    res = wilcoxon_test(improved, baseline)
    assert res["significant"]
    assert res["mean_diff"] < 0  # improved - baseline is negative


def test_run_significance_tests_excludes_target_and_oracle_by_default():
    trial_data = {
        "RL_Adaptive": [100.0 + i for i in range(10)],
        "FCFS":        [150.0 + i for i in range(10)],
        "SJF_Oracle":  [80.0 + i for i in range(10)],
    }
    report = run_significance_tests(trial_data, target="RL_Adaptive")
    assert "FCFS" in report
    assert "SJF_Oracle" not in report
    assert "RL_Adaptive" not in report
