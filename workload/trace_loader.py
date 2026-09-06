"""
workload/trace_loader.py
=========================
Real cluster workload trace loader.

Converts CSV trace files from Google, Alibaba, or any custom format into
lists of Process objects that are 100% compatible with the existing simulator,
dispatcher, and evaluation pipeline.  The existing WorkloadGenerator is
untouched — this module is a drop-in alternative source of workloads.

SUPPORTED FORMATS
-----------------
Google Cluster Trace (2011 / 2019)
  Relevant table : task_events (2011) or collection_events (2019)
  Expected columns (subset we use):
    timestamp   → arrival_time  (microseconds, rescaled to ticks)
    priority    → priority      (0-11 in Google trace, remapped 1-10)
    We derive burst_time from "scheduled_runtime" or estimate heuristically.

  Simplified CSV we expect (after pre-processing with the scripts Google
  provides, or any custom export):
    arrival_time, burst_time, priority[, process_type]

Alibaba Container Trace (2018)
  Relevant table : batch_task or container_meta
  Expected columns:
    start_time  → arrival_time  (seconds, rescaled to ticks)
    end_time    (used to compute burst_time = end_time - start_time)
    plan_cpu    → used to derive priority bucket

  Simplified CSV we expect:
    start_time, end_time, plan_cpu[, plan_mem]

Custom trace (any CSV)
  You must supply a CSV with at least these columns:
    arrival_time, burst_time, priority

NORMALISATION STRATEGY
-----------------------
Real traces span hours/days; the simulator works in ticks of ~1 unit.
All loaders apply a time_scale factor so the trace fits the simulator's
expected range.  Default: auto-scale so the median inter-arrival gap ≈ 3
ticks (matching the synthetic workload generator).

USAGE EXAMPLE
-------------
    from workload.trace_loader import load_google_trace, load_custom_trace

    # Load a pre-processed Google trace CSV
    processes = load_google_trace("data/google_tasks.csv")

    # Load a custom CSV
    processes = load_custom_trace("data/my_trace.csv")

    # Then use exactly like a generated workload:
    from evaluation.experiment import run_all_static_schedulers
    results = run_all_static_schedulers(processes)

NOTE ON REAL TRACE AVAILABILITY
---------------------------------
The Google and Alibaba traces are publicly available but large (GB+).
If no file is found at the given path, the loaders emit a clear warning
and return a synthetic fallback workload so the rest of the pipeline
continues uninterrupted.  This makes it easy to run the experiment runner
even without downloading the full traces.
"""

import bisect
import csv
import os
import warnings
from typing import List, Optional

from simulator.process import Process
from workload.generator import WorkloadGenerator


# ── internal helpers ─────────────────────────────────────────────────────────

def _classify_process_type(burst_time: int) -> str:
    """Classify a process by burst duration, mirroring WorkloadGenerator logic."""
    if burst_time <= 5:
        return "interactive"
    elif burst_time >= 20:
        return "cpu_bound"
    return "mixed"


def _auto_scale(raw_times: List[float]) -> float:
    """
    Compute a time_scale so that the median inter-arrival gap ≈ 3 ticks.
    If the trace has only one event, returns 1.0.
    """
    if len(raw_times) < 2:
        return 1.0
    sorted_times = sorted(raw_times)
    gaps = [sorted_times[i+1] - sorted_times[i]
            for i in range(len(sorted_times)-1)
            if sorted_times[i+1] > sorted_times[i]]
    if not gaps:
        return 1.0
    median_gap = sorted(gaps)[len(gaps) // 2]
    target_gap = 3.0
    return target_gap / max(median_gap, 1e-9)


MAX_REAL_BURST_TICKS = 500
# Real cluster job durations are heavy-tailed across orders of magnitude
# (the bundled Alibaba PAI trace alone ranges from 3 real seconds to
# ~599,445 — about a week). A single scale factor calibrated on
# inter-ARRIVAL gaps (as _auto_scale does) is the wrong thing to also
# apply to burst TIMES on data like this: if jobs happen to arrive a few
# seconds apart (common), that calibration is already close to 1:1, so a
# multi-day outlier job becomes a multi-day-in-ticks process and the
# tick-by-tick simulator never finishes. load_google_trace,
# load_alibaba_trace, and load_alibaba_pai_trace therefore scale burst
# times to their OWN median (not the arrival scale) and clip the result
# to this cap. This is a disclosed simulation-tractability clip, not a
# silent one — each loader prints how many jobs it affected.
# NOTE: this only applies on the auto-scale path (time_scale=None). If
# you pass an explicit time_scale, it is applied uniformly to arrival and
# burst with no independent rescaling or cap, exactly as before — you're
# taking responsibility for calibration at that point.


def _auto_scale_to_median(raw_values: List[float], target_median: float) -> float:
    """Compute a scale factor so the median of raw_values maps to target_median."""
    if not raw_values:
        return 1.0
    med = sorted(raw_values)[len(raw_values) // 2]
    return target_median / max(med, 1e-9)


def _scaled_and_capped_bursts(bursts_raw: List[float], label: str) -> List[float]:
    """
    Scale a list of RAW real burst durations to a synthetic-comparable
    median (~15 ticks) and clip to MAX_REAL_BURST_TICKS. Prints how many
    jobs were clipped so the tradeoff is visible, not silent.
    """
    burst_scale = _auto_scale_to_median(bursts_raw, target_median=15.0)
    scaled = [max(1.0, b * burst_scale) for b in bursts_raw]
    n_clipped = sum(1 for b in scaled if b > MAX_REAL_BURST_TICKS)
    if n_clipped:
        print(f"  [trace_loader] {label}: clipped {n_clipped}/{len(scaled)} "
              f"({100*n_clipped/len(scaled):.1f}%) outlier burst times to "
              f"{MAX_REAL_BURST_TICKS} ticks for simulator tractability "
              f"(real durations are heavy-tailed — see MAX_REAL_BURST_TICKS "
              f"docstring in trace_loader.py).")
    return [min(b, MAX_REAL_BURST_TICKS) for b in scaled]


def _synthetic_fallback(label: str, n: int = 30) -> List[Process]:
    """Return a synthetic mixed workload when a trace file cannot be loaded."""
    warnings.warn(
        f"[trace_loader] Could not load {label} trace — "
        f"returning {n}-process synthetic fallback workload.",
        UserWarning,
        stacklevel=3,
    )
    return WorkloadGenerator().generate(workload_type="mixed")


def _clamp_priority(raw: float, lo: float, hi: float,
                    out_lo: int = 1, out_hi: int = 10) -> int:
    """
    Linear remap of a raw priority value from [lo, hi] → [out_lo, out_hi].
    Lower output value = higher urgency (consistent with existing schedulers).
    """
    if hi == lo:
        return out_lo
    frac = (raw - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    return int(round(out_lo + frac * (out_hi - out_lo)))


_PAI_WEIGHT_DECILES = [3.0, 6.0, 8.0, 8.0, 10.0, 14.0, 14.0, 14.0, 14.0]
# 10th-90th percentile breakpoints of `num_cpu + 8*num_gpu`, computed ONCE
# from the full 100,000-row real trace (see
# scripts/run_real_trace_evaluation.py for the one-off computation).
# Hardcoded rather than recomputed at import time so a partial/custom CSV
# passed via `path` still buckets against the trace's real distribution,
# not whatever fraction happens to be loaded.


def _bucket_pai_priority(num_cpu: float, num_gpu: float) -> int:
    """
    Derive a 1-10 priority PROXY from requested resource intensity, for the
    real Alibaba PAI GPU trace (see load_alibaba_pai_trace).

    NOTE: the raw trace has no ground-truth scheduling-priority field —
    everything below is a documented design choice, not a measured fact:
      - weight = num_cpu + 8*num_gpu (GPUs weighted 8x, a scarcity proxy —
        a choice, not a measurement).
      - bucketed against REAL decile breakpoints from the full 100K-row
        trace (_PAI_WEIGHT_DECILES) rather than a guessed fixed range —
        the same "calibrate against actual data" fix already applied to
        rl_agent/state_encoder.py's threshold redesign.
      - direction: heavier/bigger jobs -> LOWER priority (higher output
        number = less urgent), matching how production clusters typically
        treat large batch/training jobs as more delay-tolerant than small
        ones (mirrors the BE-vs-LS QoS split real k8s traces use). A
        defensible convention, not a discovered fact.
    """
    decile_idx = bisect.bisect_right(_PAI_WEIGHT_DECILES, num_cpu + 8.0 * num_gpu)
    return int(1 + decile_idx)  # 1 = lightest/most urgent .. 10 = heaviest/least urgent


def _build_processes(rows: List[dict]) -> List[Process]:
    """
    Convert a list of normalised row-dicts into Process objects.

    Expected keys per row:
        arrival_time : int  (simulation ticks)
        burst_time   : int  (simulation ticks, >= 1)
        priority     : int  (1–10)
        process_type : str  (optional, auto-classified if absent)
    """
    processes: List[Process] = []
    prev_arrival = 0

    for pid, row in enumerate(
            sorted(rows, key=lambda r: r["arrival_time"])):
        burst    = max(1, int(round(row["burst_time"])))
        arrival  = max(0, int(round(row["arrival_time"])))
        priority = max(1, min(10, int(row["priority"])))
        ptype    = row.get("process_type") or _classify_process_type(burst)

        proc = Process(
            pid          = pid,
            arrival_time = arrival,
            burst_time   = burst,
            priority     = priority,
            process_type = ptype,
        )
        proc._arrival_gap = float(arrival - prev_arrival)
        prev_arrival = arrival
        processes.append(proc)

    return processes


# ── public API ────────────────────────────────────────────────────────────────

def load_google_trace(
    path: str,
    time_scale: Optional[float] = None,
    max_processes: int = 500,
) -> List[Process]:
    """
    Load a pre-processed Google Cluster Trace CSV.

    Expected CSV columns (case-insensitive, extra columns ignored):
        arrival_time   — task submission timestamp (any unit)
        burst_time     — scheduled / actual runtime (same unit)
        priority       — Google priority 0–11 (remapped to 1–10)

    The Google 2011 trace uses "priority" field 0–11 where higher = more
    important.  We invert this: priority 11 → output 1 (most urgent).

    Parameters
    ----------
    path          : Path to the CSV file.
    time_scale    : Ticks per raw time unit.  None = auto-scale.
    max_processes : Cap on number of processes loaded (keeps sim tractable).

    Returns
    -------
    List of Process objects, or a synthetic fallback if the file is missing.

    Example
    -------
    >>> procs = load_google_trace("data/google_tasks.csv")
    >>> print(f"Loaded {len(procs)} processes")
    """
    if not os.path.exists(path):
        return _synthetic_fallback(f"Google ({path})")

    raw_rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Normalise header names to lower-case
            for row in reader:
                norm = {k.strip().lower(): v.strip() for k, v in row.items()}
                raw_rows.append(norm)
                if len(raw_rows) >= max_processes:
                    break
    except Exception as exc:
        warnings.warn(f"[trace_loader] Google trace read error: {exc}")
        return _synthetic_fallback(f"Google ({path})")

    if not raw_rows:
        return _synthetic_fallback(f"Google ({path}) — empty file")

    # Extract raw floats
    try:
        arrivals   = [float(r["arrival_time"]) for r in raw_rows]
        bursts_raw = [float(r["burst_time"])   for r in raw_rows]
        # Google priority: higher raw = more important → invert for our convention
        # raw range 0–11 → output 1–10  (raw=11 → priority=1 most urgent)
        prio_raw   = [float(r.get("priority", 5)) for r in raw_rows]
    except (KeyError, ValueError) as exc:
        warnings.warn(f"[trace_loader] Google trace column error: {exc}")
        return _synthetic_fallback(f"Google ({path})")

    # Rebase so this slice starts at ~0, regardless of what absolute
    # arrival_time values the file happens to contain (a middle slice of a
    # larger trace would otherwise carry its absolute offset straight into
    # ticks — see MAX_REAL_BURST_TICKS docstring for the full failure mode
    # this caused with the bundled Alibaba PAI trace).
    t0 = min(arrivals)
    arrivals = [a - t0 for a in arrivals]

    if time_scale is not None:
        # Explicit override: uniform scale, no independent rescaling/cap —
        # caller is taking responsibility for calibration.
        arr_scale, burst_vals = time_scale, [max(1.0, b * time_scale) for b in bursts_raw]
    else:
        arr_scale = _auto_scale(arrivals)
        burst_vals = _scaled_and_capped_bursts(bursts_raw, "Google trace")

    rows = []
    for arr, burst, prio in zip(arrivals, burst_vals, prio_raw):
        rows.append({
            "arrival_time": arr * arr_scale,
            "burst_time":   burst,
            # Invert Google priority (11=most important → 1=most urgent in our scheme)
            "priority":     _clamp_priority(prio, lo=0, hi=11, out_lo=10, out_hi=1),
        })

    processes = _build_processes(rows)
    print(f"  [trace_loader] Loaded {len(processes)} processes from Google trace: {path}")
    return processes


def load_alibaba_trace(
    path: str,
    time_scale: Optional[float] = None,
    max_processes: int = 500,
) -> List[Process]:
    """
    Load a pre-processed Alibaba Cluster Trace CSV.

    Expected CSV columns (case-insensitive, extra columns ignored):
        start_time  — job start timestamp (seconds)
        end_time    — job end timestamp (seconds)
                      burst_time is derived as end_time - start_time
        plan_cpu    — CPU request (normalised 0–100); used to derive priority

    If "end_time" is unavailable or <= start_time, burst_time defaults to
    a random draw matching the mixed workload generator profile.

    Parameters
    ----------
    path          : Path to the CSV file.
    time_scale    : Ticks per raw time unit.  None = auto-scale.
    max_processes : Cap on number of processes loaded.

    Returns
    -------
    List of Process objects, or a synthetic fallback if the file is missing.

    Example
    -------
    >>> procs = load_alibaba_trace("data/alibaba_batch.csv")
    """
    if not os.path.exists(path):
        return _synthetic_fallback(f"Alibaba ({path})")

    raw_rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                norm = {k.strip().lower(): v.strip() for k, v in row.items()}
                raw_rows.append(norm)
                if len(raw_rows) >= max_processes:
                    break
    except Exception as exc:
        warnings.warn(f"[trace_loader] Alibaba trace read error: {exc}")
        return _synthetic_fallback(f"Alibaba ({path})")

    if not raw_rows:
        return _synthetic_fallback(f"Alibaba ({path}) — empty file")

    try:
        starts   = [float(r["start_time"]) for r in raw_rows]
        ends     = [float(r.get("end_time", r["start_time"])) for r in raw_rows]
        plan_cpu = [float(r.get("plan_cpu", 50))               for r in raw_rows]
    except (KeyError, ValueError) as exc:
        warnings.warn(f"[trace_loader] Alibaba trace column error: {exc}")
        return _synthetic_fallback(f"Alibaba ({path})")

    # Rebase so this slice starts at ~0 — see load_google_trace for why.
    t0 = min(starts)
    ends = [e - t0 for e in ends]
    starts = [s - t0 for s in starts]

    import random as _rng
    raw_bursts = []
    for start, end in zip(starts, ends):
        raw_burst = max(0.0, end - start)
        if raw_burst <= 0:
            raw_burst = _rng.uniform(1, 30)   # imputed from mixed profile
        raw_bursts.append(raw_burst)

    if time_scale is not None:
        # Explicit override: uniform scale, no independent rescaling/cap —
        # caller is taking responsibility for calibration.
        arr_scale, burst_vals = time_scale, [max(1.0, b * time_scale) for b in raw_bursts]
    else:
        arr_scale = _auto_scale(starts)
        burst_vals = _scaled_and_capped_bursts(raw_bursts, "Alibaba trace")

    rows = []
    for start, burst, cpu in zip(starts, burst_vals, plan_cpu):
        rows.append({
            "arrival_time": start * arr_scale,
            "burst_time":   burst,
            # plan_cpu 0–100: high CPU request → high priority (low number)
            "priority":     _clamp_priority(cpu, lo=0, hi=100, out_lo=1, out_hi=10),
        })

    processes = _build_processes(rows)
    print(f"  [trace_loader] Loaded {len(processes)} processes from Alibaba trace: {path}")
    return processes


def load_custom_trace(
    path: str,
    time_scale: Optional[float] = None,
    max_processes: int = 500,
    arrival_col:  str = "arrival_time",
    burst_col:    str = "burst_time",
    priority_col: str = "priority",
    type_col:     str = "process_type",
) -> List[Process]:
    """
    Load a workload from any CSV with user-specified column names.

    Minimum required columns: arrival_time, burst_time, priority.
    process_type column is optional (auto-classified if absent).

    Parameters
    ----------
    path          : Path to the CSV file.
    time_scale    : Ticks per raw time unit.  None = auto-scale.
    max_processes : Cap on number of processes loaded.
    arrival_col   : Column name for arrival timestamps.
    burst_col     : Column name for burst durations.
    priority_col  : Column name for scheduling priority.
    type_col      : Column name for process type label (optional).

    Returns
    -------
    List of Process objects, or a synthetic fallback if the file is missing.

    Example
    -------
    >>> # Minimal CSV: arrival_time,burst_time,priority
    >>> procs = load_custom_trace("data/my_trace.csv")

    >>> # Custom column names
    >>> procs = load_custom_trace(
    ...     "data/my_trace.csv",
    ...     arrival_col="submit_ts",
    ...     burst_col="runtime_ms",
    ...     priority_col="urgency",
    ... )
    """
    if not os.path.exists(path):
        return _synthetic_fallback(f"custom ({path})")

    raw_rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                norm = {k.strip().lower(): v.strip() for k, v in row.items()}
                raw_rows.append(norm)
                if len(raw_rows) >= max_processes:
                    break
    except Exception as exc:
        warnings.warn(f"[trace_loader] Custom trace read error: {exc}")
        return _synthetic_fallback(f"custom ({path})")

    if not raw_rows:
        return _synthetic_fallback(f"custom ({path}) — empty file")

    a_col = arrival_col.lower()
    b_col = burst_col.lower()
    p_col = priority_col.lower()
    t_col = type_col.lower()

    try:
        arrivals = [float(r[a_col]) for r in raw_rows]
        bursts   = [float(r[b_col]) for r in raw_rows]
        prios    = [float(r[p_col]) for r in raw_rows]
        types    = [r.get(t_col, "")  for r in raw_rows]
    except (KeyError, ValueError) as exc:
        warnings.warn(f"[trace_loader] Custom trace column error: {exc}. "
                      f"Available columns: {list(raw_rows[0].keys()) if raw_rows else []}")
        return _synthetic_fallback(f"custom ({path})")

    scale = time_scale if time_scale is not None else _auto_scale(arrivals)

    # Detect priority direction: if median > 5, assume higher = more urgent (invert)
    median_prio = sorted(prios)[len(prios)//2]
    invert_prio = (median_prio > 5.5)
    prio_lo, prio_hi = min(prios), max(prios)
    if invert_prio:
        out_lo, out_hi = 10, 1
    else:
        out_lo, out_hi = 1, 10

    rows = []
    for arr, burst, prio, ptype in zip(arrivals, bursts, prios, types):
        rows.append({
            "arrival_time": arr * scale,
            "burst_time":   max(1.0, burst * scale),
            "priority":     _clamp_priority(prio, prio_lo, prio_hi, out_lo, out_hi),
            "process_type": ptype if ptype else None,
        })

    processes = _build_processes(rows)
    print(f"  [trace_loader] Loaded {len(processes)} processes from custom trace: {path}")
    return processes


def load_alibaba_pai_trace(
    path: str = "data/real_traces/alibaba_pai_gpu_trace_100k.csv",
    window_start: int = 0,
    window_size: int = 150,
    time_scale: Optional[float] = None,
) -> List[Process]:
    """
    Load a REAL, contiguous window from Alibaba's PAI GPU-cluster job trace.

    This is a genuine production trace (100,000 real jobs, Alibaba Cluster
    Trace Program, collected on a 6,500+ GPU cluster, July-Aug 2020) — not
    synthetic data. See data/real_traces/PROVENANCE.md for the source URL,
    license, citation, and a full honest breakdown of what's real vs.
    derived in this loader.

    REAL columns used     : submit_time -> arrival_time, duration -> burst_time.
    DERIVED (not measured) : priority — this trace has no ground-truth
                             priority field, so priority is a proxy from
                             requested num_cpu/num_gpu (see
                             _bucket_pai_priority). Treat any "RL beats
                             Priority scheduler on real data" claim with
                             that caveat in mind.
    FLATTENED              : num_inst — multi-instance jobs are loaded as a
                             single task using the job's own duration.
    INFORMATIONAL ONLY     : gpu_type is attached as proc._gpu_type; no
                             scheduler or the LSTM reads it.

    Parameters
    ----------
    path         : Path to the bundled real trace CSV.
    window_start : Row offset into the trace (rows are already submit_time
                   sorted in the source file).
    window_size  : Number of consecutive real jobs to load as one workload.
    time_scale   : Ticks per second of real time. None = auto-scale so the
                   median real inter-arrival gap maps to ~3 ticks, matching
                   the synthetic generator's scale.

    Returns
    -------
    List of Process objects built from real job records, or a synthetic
    fallback (with a warning) if the file is missing or the window is out
    of range.

    Example
    -------
    >>> procs = load_alibaba_pai_trace(window_start=0, window_size=150)
    >>> print(f"Loaded {len(procs)} real jobs")
    """
    if not os.path.exists(path):
        return _synthetic_fallback(f"Alibaba PAI GPU ({path})")

    raw_rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i < window_start:
                    continue
                if i >= window_start + window_size:
                    break
                raw_rows.append({k.strip().lower(): v.strip() for k, v in row.items()})
    except Exception as exc:
        warnings.warn(f"[trace_loader] Alibaba PAI trace read error: {exc}")
        return _synthetic_fallback(f"Alibaba PAI GPU ({path})")

    if not raw_rows:
        return _synthetic_fallback(
            f"Alibaba PAI GPU ({path}) — window [{window_start}:"
            f"{window_start + window_size}) out of range")

    try:
        parsed = [{
            "submit_time": float(r["submit_time"]),
            "duration":    float(r["duration"]),
            "num_cpu":     float(r["num_cpu"]) if r.get("num_cpu") else 1.0,
            "num_gpu":     float(r["num_gpu"]) if r.get("num_gpu") else 0.0,
            "gpu_type":    r.get("gpu_type") or "MISC",
        } for r in raw_rows]
    except (KeyError, ValueError) as exc:
        warnings.warn(f"[trace_loader] Alibaba PAI trace column error: {exc}. "
                      f"Available columns: {list(raw_rows[0].keys())}")
        return _synthetic_fallback(f"Alibaba PAI GPU ({path})")

    # Rows are already submit_time-sorted in the source file; re-sort
    # defensively so pid order == arrival order regardless of input order.
    parsed.sort(key=lambda r: r["submit_time"])

    # Rebase to this WINDOW's own start. Without this, any window other
    # than the very first (whose raw submit_time already starts near 0)
    # carries its absolute position in the 100K-row trace straight into
    # "ticks" — e.g. a window starting at row 20000 begins around raw
    # submit_time ~130,000s, which (at an auto-scale near 1x, since local
    # gaps are only a few seconds) becomes an arrival_time of ~130,000
    # ticks. Every scheduler's max_ticks is sized off this window's own
    # (small) sum-of-bursts, so nothing would ever get admitted before
    # the simulation loop gives up — every process silently "arrives"
    # after the simulation already ended. Rebasing to a window-relative
    # clock is what makes each window a self-contained episode, exactly
    # like each synthetic trial is.
    window_t0 = parsed[0]["submit_time"]
    for r in parsed:
        r["submit_time"] -= window_t0

    if time_scale is not None:
        # Explicit override: uniform scale, no independent rescaling/cap —
        # caller is taking responsibility for calibration.
        arr_scale = time_scale
        burst_vals = [max(1.0, r["duration"] * time_scale) for r in parsed]
    else:
        arr_scale = _auto_scale([r["submit_time"] for r in parsed])
        burst_vals = _scaled_and_capped_bursts(
            [r["duration"] for r in parsed], "Alibaba PAI GPU trace")

    processes: List[Process] = []
    prev_arrival = 0
    for pid, (r, burst_f) in enumerate(zip(parsed, burst_vals)):
        arrival = max(0, int(round(r["submit_time"] * arr_scale)))
        burst   = max(1, int(round(burst_f)))
        proc = Process(
            pid=pid,
            arrival_time=arrival,
            burst_time=burst,
            priority=_bucket_pai_priority(r["num_cpu"], r["num_gpu"]),
            process_type=_classify_process_type(burst),
        )
        proc._arrival_gap = float(arrival - prev_arrival)
        proc._gpu_type = r["gpu_type"]  # informational only — not scheduled on
        prev_arrival = arrival
        processes.append(proc)

    print(f"  [trace_loader] Loaded {len(processes)} REAL jobs from Alibaba "
          f"PAI GPU trace rows [{window_start}:{window_start + len(raw_rows)}) "
          f"of {os.path.basename(path)}")
    return processes


def load_alibaba_pai_trial_windows(
    path: str = "data/real_traces/alibaba_pai_gpu_trace_100k.csv",
    n_trials: int = 30,
    window_size: int = 150,
    time_scale: Optional[float] = None,
) -> List[List["Process"]]:
    """
    Slice the real PAI GPU trace into n_trials evenly-spaced, non-overlapping
    windows spanning its full ~7.7-day range, for the real-trace evaluation
    protocol in scripts/run_real_trace_evaluation.py.

    HOW THIS DIFFERS FROM THE SYNTHETIC 30-TRIAL PROTOCOL (read before citing
    both numbers side by side): the synthetic evaluation draws 30
    INDEPENDENT random workloads — fresh randomness every trial, effectively
    i.i.d. samples. This instead draws 30 DIFFERENT non-overlapping
    contiguous slices of ONE real trace. They genuinely cover different
    real time-of-day / load conditions, which is valuable, but they are not
    "independent" in the textbook sense a significance test assumes —
    adjacent windows of one non-stationary production system can share
    autocorrelated structure. Report these results as "30 windows of one
    real trace," not as if they were 30 independent synthetic-style trials.

    Returns
    -------
    List of n_trials workloads (each a List[Process]).
    """
    if not os.path.exists(path):
        warnings.warn(
            f"[trace_loader] {path} not found — returning {n_trials} "
            f"synthetic fallback windows instead of real trace windows.",
            UserWarning, stacklevel=2,
        )
        return [WorkloadGenerator().generate(workload_type="mixed")
                for _ in range(n_trials)]

    with open(path, newline="", encoding="utf-8") as fh:
        total_rows = sum(1 for _ in fh) - 1  # minus header row

    usable_span = max(total_rows - window_size, window_size)
    stride = max(1, usable_span // n_trials)

    windows = []
    for t in range(n_trials):
        start = min(t * stride, total_rows - window_size)
        windows.append(load_alibaba_pai_trace(
            path=path, window_start=start, window_size=window_size,
            time_scale=time_scale,
        ))
    return windows


# ── convenience: generate a sample CSV for testing ───────────────────────────

def write_sample_csv(path: str = "data/sample_trace.csv", n: int = 40):
    """
    Write a small sample CSV that load_custom_trace() can parse immediately.
    Useful for testing the pipeline without downloading real traces.

    Example
    -------
    >>> from workload.trace_loader import write_sample_csv, load_custom_trace
    >>> write_sample_csv("data/sample_trace.csv")
    >>> procs = load_custom_trace("data/sample_trace.csv")
    """
    import random as _rng
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arrival_time", "burst_time", "priority", "process_type"])
        t = 0
        for _ in range(n):
            burst    = _rng.randint(1, 35)
            priority = _rng.randint(1, 10)
            ptype    = _classify_process_type(burst)
            writer.writerow([t, burst, priority, ptype])
            t += _rng.randint(1, 5)
    print(f"  [trace_loader] Sample CSV written → {path}  ({n} rows)")
