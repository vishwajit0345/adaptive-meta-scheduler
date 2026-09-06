"""
tests/test_workload.py
========================
Tests for workload/generator.py (synthetic workloads) and
workload/trace_loader.py (CSV trace ingestion).
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload.generator import WorkloadGenerator
from workload.trace_loader import (
    load_custom_trace,
    write_sample_csv,
    load_alibaba_pai_trace,
    load_alibaba_pai_trial_windows,
    _bucket_pai_priority,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_PAI_TRACE = os.path.join(
    _REPO_ROOT, "data", "real_traces", "alibaba_pai_gpu_trace_100k.csv")


# ── WorkloadGenerator ────────────────────────────────────────────────────────

def test_generate_returns_sorted_by_arrival_time():
    gen = WorkloadGenerator()
    procs = gen.generate(workload_type="mixed", seed=1)
    arrivals = [p.arrival_time for p in procs]
    assert arrivals == sorted(arrivals)


def test_generate_process_count_within_profile_range():
    gen = WorkloadGenerator()
    lo, hi = gen.PROFILES["cpu_bound"]["num_processes_range"]
    procs = gen.generate(workload_type="cpu_bound", seed=2)
    assert lo <= len(procs) <= hi


def test_generate_burst_times_within_profile_range():
    gen = WorkloadGenerator()
    lo, hi = gen.PROFILES["interactive"]["burst_time_range"]
    procs = gen.generate(workload_type="interactive", seed=3)
    assert all(lo <= p.burst_time <= hi for p in procs)


def test_generate_is_reproducible_with_same_seed():
    gen = WorkloadGenerator()
    a = gen.generate(workload_type="mixed", seed=42)
    b = gen.generate(workload_type="mixed", seed=42)
    assert [p.burst_time for p in a] == [p.burst_time for p in b]
    assert [p.arrival_time for p in a] == [p.arrival_time for p in b]


def test_generate_unknown_type_falls_back_to_mixed():
    gen = WorkloadGenerator()
    procs = gen.generate(workload_type="not_a_real_type", seed=4)
    lo, hi = gen.PROFILES["mixed"]["num_processes_range"]
    assert lo <= len(procs) <= hi


def test_process_type_classification_matches_burst_thresholds():
    gen = WorkloadGenerator()
    procs = gen.generate(workload_type="mixed", seed=5)
    for p in procs:
        if p.burst_time <= 5:
            assert p.process_type == "interactive"
        elif p.burst_time >= 20:
            assert p.process_type == "cpu_bound"
        else:
            assert p.process_type == "mixed"


def test_describe_workload_matches_generated_data():
    gen = WorkloadGenerator()
    procs = gen.generate(workload_type="mixed", seed=6)
    desc = gen.describe_workload(procs)
    assert desc["count"] == len(procs)
    assert desc["min_burst"] == min(p.burst_time for p in procs)
    assert desc["max_burst"] == max(p.burst_time for p in procs)
    assert desc["total_burst"] == sum(p.burst_time for p in procs)


def test_generate_batch_produces_requested_number_of_workloads():
    gen = WorkloadGenerator()
    batch = gen.generate_batch(workload_types=["mixed", "cpu_bound"], count_per_type=2)
    assert len(batch) == 4


# ── trace_loader ─────────────────────────────────────────────────────────────

def test_load_custom_trace_missing_file_falls_back_with_warning(tmp_path):
    missing = str(tmp_path / "does_not_exist.csv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        procs = load_custom_trace(missing)
    assert len(procs) > 0  # synthetic fallback workload
    assert any("Could not load" in str(w.message) for w in caught)


def test_write_sample_csv_then_load_round_trip(tmp_path):
    path = str(tmp_path / "sample.csv")
    write_sample_csv(path, n=25)
    procs = load_custom_trace(path)
    assert len(procs) == 25
    # arrival times must be non-decreasing (loader sorts by arrival)
    arrivals = [p.arrival_time for p in procs]
    assert arrivals == sorted(arrivals)
    # priorities clamped into [1, 10]
    assert all(1 <= p.priority <= 10 for p in procs)
    # burst times are always >= 1 tick
    assert all(p.burst_time >= 1 for p in procs)


def test_load_custom_trace_respects_custom_column_names(tmp_path):
    path = str(tmp_path / "custom_cols.csv")
    with open(path, "w") as f:
        f.write("submit_ts,runtime_ms,urgency\n")
        f.write("0,10,1\n")
        f.write("2,5,9\n")
        f.write("5,20,3\n")

    procs = load_custom_trace(
        path,
        arrival_col="submit_ts",
        burst_col="runtime_ms",
        priority_col="urgency",
        time_scale=1.0,
    )
    assert len(procs) == 3
    assert {p.burst_time for p in procs} == {10, 5, 20}


# ── real trace: Alibaba PAI GPU cluster (see data/real_traces/PROVENANCE.md) ──

def test_load_alibaba_pai_trace_missing_file_falls_back_with_warning(tmp_path):
    missing = str(tmp_path / "does_not_exist.csv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        procs = load_alibaba_pai_trace(missing, window_size=20)
    assert len(procs) > 0  # synthetic fallback workload
    assert any("Could not load" in str(w.message) for w in caught)


def test_load_alibaba_pai_trace_loads_real_bundled_data():
    if not os.path.exists(_REAL_PAI_TRACE):
        return  # bundled data not present in this checkout — skip quietly
    procs = load_alibaba_pai_trace(_REAL_PAI_TRACE, window_start=0, window_size=150)
    assert len(procs) == 150
    arrivals = [p.arrival_time for p in procs]
    assert arrivals == sorted(arrivals)               # loader sorts by arrival
    assert all(1 <= p.priority <= 10 for p in procs)   # priority still clamped
    assert all(p.burst_time >= 1 for p in procs)       # burst always >= 1 tick
    # gpu_type is real, informational metadata — must survive the load
    assert all(hasattr(p, "_gpu_type") for p in procs)
    assert all(p._gpu_type in {"CPU", "T4", "P100", "V100", "MISC"} for p in procs)


def test_load_alibaba_pai_trace_window_offset_moves_through_real_data():
    if not os.path.exists(_REAL_PAI_TRACE):
        return
    import csv as _csv
    with open(_REAL_PAI_TRACE, newline="", encoding="utf-8") as fh:
        raw_submit_times = [float(r["submit_time"]) for r in _csv.DictReader(fh)]
    # Each window is auto-scaled independently, so compare the REAL
    # (pre-scale) submit_time at each offset, not the derived arrival_time.
    assert raw_submit_times[5000] > raw_submit_times[0]

    first_window = load_alibaba_pai_trace(_REAL_PAI_TRACE, window_start=0, window_size=50)
    later_window = load_alibaba_pai_trace(_REAL_PAI_TRACE, window_start=5000, window_size=50)
    assert len(first_window) == len(later_window) == 50
    # different, non-identical real jobs were actually loaded for each window
    assert [p.burst_time for p in first_window] != [p.burst_time for p in later_window]


def test_load_alibaba_pai_trial_windows_are_nonoverlapping_and_span_the_trace():
    if not os.path.exists(_REAL_PAI_TRACE):
        return
    windows = load_alibaba_pai_trial_windows(
        _REAL_PAI_TRACE, n_trials=30, window_size=150)
    assert len(windows) == 30
    assert all(len(w) == 150 for w in windows)
    # NOTE: each window is auto-scaled independently (time_scale=None), the
    # same way each synthetic trial is its own self-contained episode — so
    # post-scale Process.arrival_time is NOT comparable across windows.
    # What must hold is that the underlying REAL rows are non-overlapping
    # and span later parts of the trace as the window index increases.
    import csv as _csv
    with open(_REAL_PAI_TRACE, newline="", encoding="utf-8") as fh:
        raw_submit_times = [float(r["submit_time"]) for r in _csv.DictReader(fh)]
    total_rows = len(raw_submit_times)
    stride = max(1, max(total_rows - 150, 150) // 30)
    real_window_starts = [min(t * stride, total_rows - 150) for t in range(30)]
    real_first_submits = [raw_submit_times[s] for s in real_window_starts]
    assert real_first_submits == sorted(real_first_submits)
    assert real_first_submits[-1] > real_first_submits[0]  # actually spans the trace


def test_bucket_pai_priority_stays_in_range_and_orders_by_weight():
    lightest = _bucket_pai_priority(num_cpu=0.01, num_gpu=0.0)
    heaviest = _bucket_pai_priority(num_cpu=90.0, num_gpu=8.0)
    assert 1 <= lightest <= 10
    assert 1 <= heaviest <= 10
    # documented convention: heavier requests -> higher number -> less urgent
    assert heaviest > lightest


def test_load_alibaba_pai_trace_rebases_late_windows_to_start_near_zero(tmp_path):
    """
    Regression test for a real bug: a window starting deep into the trace
    (large absolute submit_time) must NOT carry that absolute offset into
    Process.arrival_time. If it did, every process in that window would
    "arrive" long after max_ticks (sized off that window's own small sum
    of burst times), so nothing would ever be admitted to the ready queue
    and every scheduler would report avg_waiting_time == inf with
    completed_count == 0.
    """
    path = str(tmp_path / "late_window.csv")
    with open(path, "w") as f:
        f.write("job_id,num_inst,submit_time,num_cpu,num_gpu,gpu_type,duration,user\n")
        # absolute submit_time deep into a hypothetical trace (~130,000s in),
        # small gaps between jobs, short durations — exactly the shape of a
        # real "later" window.
        base = 130000
        for i in range(20):
            f.write(f"{i},1,{base + i * 4},2,0,CPU,{10 + i},u1\n")

    procs = load_alibaba_pai_trace(path, window_start=0, window_size=20)
    assert len(procs) == 20
    # rebased: first arrival must be near 0, not ~130,000
    assert procs[0].arrival_time < 50
    assert max(p.arrival_time for p in procs) < 500
