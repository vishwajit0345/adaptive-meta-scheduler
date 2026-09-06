# Real trace provenance

## Source
`alibaba_pai_gpu_trace_100k.csv` is an **unmodified copy** of
`cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_no_estimate_100K.csv`
from the official Alibaba Cluster Trace Program repository:

    https://github.com/alibaba/clusterdata
    (fetched 2026-07-22, commit at time of fetch — see git log in that repo)

Alibaba explicitly permits reuse "however you want as long as it is for
research or study purpose" (top-level repo README). This file is bundled here
under that permission, unmodified, for a final-year academic project.

If you use this data, the trace's own authors ask to be cited:

    Weng, Q., Xiao, W., Yu, Y., Wang, W., Wang, C., He, J., Li, Y., Zhang, L.,
    Lin, W., & Ding, Y. (2022). MLaaS in the Wild: Workload Analysis and
    Scheduling in Large-Scale Heterogeneous GPU Clusters. NSDI '22.

## What this data actually is

A **real job-submission trace from Alibaba's PAI (Platform for AI) production
GPU cluster** — 6,500+ GPUs across ~1,800 machines, collected July–August
2020. 100,000 real jobs, already sorted by `submit_time`, spanning ~7.7 days
of wall-clock activity. Columns used here:

| Column        | Meaning                                          |
|---------------|---------------------------------------------------|
| `job_id`      | Unique job identifier                              |
| `submit_time` | Real submission timestamp, seconds, trace-relative |
| `duration`    | Real measured job runtime, seconds                 |
| `num_cpu`     | Requested CPU cores                                |
| `num_gpu`     | Requested GPUs                                     |
| `gpu_type`    | CPU / T4 / P100 / V100 / MISC                      |

## Honest disclosure of what is NOT in the raw data

This is an **ML/GPU cluster job trace**, not a literal OS-kernel CPU-process
trace — the abstraction (a job submitted at a time, needing a duration of
compute before it can be marked complete) maps directly onto this project's
`Process(arrival_time, burst_time, priority)` model, which is why it's usable
here, but it is not the same thing as a kernel scheduler trace, and this
project does not claim otherwise.

- **There is no ground-truth scheduling-priority field.** `priority` for
  every process loaded from this trace is a **derived proxy** from requested
  resource intensity (`num_cpu + 8×num_gpu`, percentile-bucketed 1–10) —
  exactly the same style of derivation `load_alibaba_trace()` already used
  for the generic `plan_cpu`-based Alibaba format, just applied to this
  schema's actual columns. It is not a real priority label.
- **`num_inst` (multi-instance jobs) is flattened to 1 task per row.** A job
  requesting 12 instances is loaded as a single Process with the job's own
  `duration`, not 12 concurrent tasks. Documented simplification, not a bug.
- **`gpu_type` is kept as informational metadata only** (`proc._gpu_type`) —
  it is not fed into `process_type` classification, which stays
  duration-based for consistency with the synthetic generator.

## How it's used in this project

`load_alibaba_pai_trace()` in `workload/trace_loader.py` loads a contiguous,
unmodified window of real jobs (real `submit_time`, real `duration`, proxy
`priority`) and returns `Process` objects. `load_alibaba_pai_trial_windows()`
slices the 100K-row trace into N evenly-spaced, non-overlapping windows for
the real-trace evaluation in `scripts/run_real_trace_evaluation.py` — see
that script's docstring for why "30 windows of one real trace" is reported
distinctly from "30 independently-sampled synthetic trials," not conflated
with it.
