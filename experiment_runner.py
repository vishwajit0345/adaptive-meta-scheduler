"""
experiment_runner.py
=====================
Research-grade experiment runner — integrates all 6 upgrade features.

Run:
    python experiment_runner.py                     # all features on
    python experiment_runner.py --quick             # quick 100-ep smoke test
    python experiment_runner.py --no-traces         # skip trace loader demo
    python experiment_runner.py --cs-cost 2         # context-switch overhead = 2

INTEGRATION ARCHITECTURE
-------------------------
This file is the ONLY new integration point.  It imports from:

  EXISTING (unchanged):
    main.py pipeline continues to run identically via `python main.py`
    workload/generator.py
    evaluation/experiment.py
    visualization/plots.py
    rl_agent/trainer.py, q_table.py, state_encoder.py
    burst_predictor/lstm_predictor.py, pretrainer.py
    schedulers/fcfs.py, sjf.py, round_robin.py, priority.py

  NEW (added by this upgrade):
    workload/trace_loader.py          Feature 1 — real trace support
    schedulers/mlfq.py                Feature 2 — MLFQ baseline
    schedulers/lottery.py             Feature 2 — Lottery baseline
    schedulers/edf.py                 Feature 2 — EDF baseline
    evaluation/statistics.py          Feature 3 — significance tests
    simulator/context_switch.py       Feature 4 — CS overhead dispatcher
    evaluation/metrics.py             Feature 5 — extended metrics
    visualization/extended_plots.py   Feature 6 — 3 new chart types

FEATURE FLAGS
-------------
All 6 features are individually toggleable:

    cfg = ExperimentConfig(
        use_real_traces       = False,   # Feature 1
        extra_schedulers      = True,    # Feature 2
        significance_testing  = True,    # Feature 3
        context_switch_cost   = 0,       # Feature 4 (0 = disabled)
        extended_metrics      = True,    # Feature 5
        extended_plots        = True,    # Feature 6
        num_episodes          = 600,
        num_trials            = 30,
    )
    run_extended_experiment(cfg)
"""

import argparse
import copy
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── existing pipeline imports (unchanged) ────────────────────────────────────
from workload.generator              import WorkloadGenerator
from burst_predictor.lstm_predictor  import BurstTimePredictor
from burst_predictor.pretrainer      import pretrain_predictor
from rl_agent.q_table                import QTable
from rl_agent.trainer                import RLTrainer
from evaluation.experiment           import (
    run_all_static_schedulers, run_predicted_sjf,
    print_comparison_table, run_multi_trial_evaluation,
    print_statistical_table, run_static_scheduler,
)
from visualization.plots             import (
    plot_comparison, plot_training_curve,
    plot_prediction_accuracy, plot_statistical_comparison,
)

# ── new module imports ────────────────────────────────────────────────────────
# Feature 1
from workload.trace_loader           import (
    load_custom_trace, load_google_trace, load_alibaba_trace,
    write_sample_csv,
)
# Feature 2
from schedulers.mlfq                 import MLFQScheduler
from schedulers.lottery              import LotteryScheduler
from schedulers.edf                  import EDFScheduler
# Feature 3
from evaluation.statistics           import (
    print_significance_report, run_significance_tests,
)
# Feature 4
from simulator.context_switch        import ContextSwitchDispatcher
# Feature 5
from evaluation.metrics              import (
    compute_extended_metrics, print_extended_table,
)
# Feature 6
from visualization.extended_plots    import (
    plot_boxplots, plot_convergence_curve,
    plot_scheduler_selection, plot_extended_metrics_radar,
)

# simulator primitives used for Feature 4/5 custom runner
from simulator.process               import Process
from simulator.ready_queue           import ReadyQueue
from simulator.clock                 import SimulationClock

from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentConfig:
    """
    All tunable parameters for the extended experiment.

    Defaults reproduce the original main.py behaviour plus all new features.
    Set context_switch_cost=0 to keep original timing exactly.
    """
    def __init__(
        self,
        # ── original pipeline settings ────────────────────────────────────
        num_episodes:         int   = 600,
        num_trials:           int   = 30,
        eval_workload_type:   str   = "mixed",
        results_dir:          str   = "results",
        # ── Feature 1 ─────────────────────────────────────────────────────
        use_real_traces:      bool  = True,
        trace_csv_path:       str   = "data/sample_trace.csv",
        # ── Feature 2 ─────────────────────────────────────────────────────
        extra_schedulers:     bool  = True,
        # ── Feature 3 ─────────────────────────────────────────────────────
        significance_testing: bool  = True,
        significance_alpha:   float = 0.05,
        # ── Feature 4 ─────────────────────────────────────────────────────
        context_switch_cost:  int   = 0,
        cs_cost_sweep:        bool  = True,    # run CS cost = [0, 1, 2, 4]
        # ── Feature 5 ─────────────────────────────────────────────────────
        extended_metrics:     bool  = True,
        # ── Feature 6 ─────────────────────────────────────────────────────
        extended_plots:       bool  = True,
    ):
        self.num_episodes         = num_episodes
        self.num_trials           = num_trials
        self.eval_workload_type   = eval_workload_type
        self.results_dir          = results_dir
        self.use_real_traces      = use_real_traces
        self.trace_csv_path       = trace_csv_path
        self.extra_schedulers     = extra_schedulers
        self.significance_testing = significance_testing
        self.significance_alpha   = significance_alpha
        self.context_switch_cost  = context_switch_cost
        self.cs_cost_sweep        = cs_cost_sweep
        self.extended_metrics     = extended_metrics
        self.extended_plots       = extended_plots


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    width = 76
    print(f"\n{'─'*width}\n  {title}\n{'─'*width}")


def _run_scheduler_extended(
    scheduler,
    processes: List[Process],
    context_switch_cost: int = 0,
) -> Dict:
    """
    Run a scheduler with optional CS overhead using ContextSwitchDispatcher.
    Returns extended metrics dict (Feature 4 + 5 combined).
    """
    procs      = [copy.deepcopy(p) for p in processes]
    clock      = SimulationClock()
    rq         = ReadyQueue()
    # Feature 4: use ContextSwitchDispatcher instead of base Dispatcher
    dispatcher = ContextSwitchDispatcher(clock, rq,
                                         context_switch_cost=context_switch_cost)
    if hasattr(scheduler, "reset"):
        scheduler.reset()

    completed: List[Process] = []
    max_ticks = sum(p.burst_time for p in procs) * 4 + 200

    while len(completed) < len(procs) and clock.current_time <= max_ticks:
        for p in procs:
            if (p.arrival_time == clock.current_time
                    and not p.is_complete() and p not in rq.get_all()):
                rq.add_process(p)
        dispatcher.execute_tick(scheduler.select_next(rq), completed)

    # Feature 5: compute full metric suite
    return compute_extended_metrics(completed, clock, scheduler.name)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Real Trace Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_trace_demo(cfg: ExperimentConfig) -> Optional[List[Process]]:
    """
    Feature 1: Load a workload from a real/sample trace CSV and run schedulers.
    Returns the loaded process list so it can be reused elsewhere.
    """
    _section("FEATURE 1 — REAL WORKLOAD TRACE SUPPORT")
    print(
        "  Trace loaders available:\n"
        "    load_google_trace(path)   — Google Cluster Trace 2011/2019 CSV\n"
        "    load_alibaba_trace(path)  — Alibaba Batch Task Trace CSV\n"
        "    load_custom_trace(path)   — Any CSV with arrival_time, burst_time, priority\n"
    )

    # Generate sample CSV if it doesn't exist yet
    if not os.path.exists(cfg.trace_csv_path):
        print(f"  No trace file found at '{cfg.trace_csv_path}'.")
        print("  Generating sample trace CSV for demonstration...")
        os.makedirs(os.path.dirname(cfg.trace_csv_path) or ".", exist_ok=True)
        write_sample_csv(cfg.trace_csv_path, n=40)

    trace_procs = load_custom_trace(cfg.trace_csv_path)

    if not trace_procs:
        print("  [Feature 1] Trace loaded 0 processes — skipping trace demo.")
        return None

    print(f"\n  Trace workload: {len(trace_procs)} processes "
          f"| burst range {min(p.burst_time for p in trace_procs)}–"
          f"{max(p.burst_time for p in trace_procs)}")

    # Run the existing static schedulers on the trace workload
    trace_results = run_all_static_schedulers(trace_procs)
    print("\n  Classical schedulers on trace workload:")
    print_comparison_table(trace_results)

    print(
        "  ─── Usage in your own code ───────────────────────────────────\n"
        "  from workload.trace_loader import load_google_trace\n"
        "  procs = load_google_trace('data/google_tasks.csv')\n"
        "  results = run_all_static_schedulers(procs)   # identical API\n"
    )
    return trace_procs


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Extra Schedulers Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_extra_schedulers(
    processes: List[Process],
    cfg: ExperimentConfig,
) -> List[Dict]:
    """
    Feature 2: Run MLFQ, Lottery, and EDF on the given workload.
    Returns list of result dicts compatible with print_comparison_table.
    """
    _section("FEATURE 2 — ADDITIONAL SCHEDULER BASELINES (MLFQ, Lottery, EDF)")
    print(
        "  Three new schedulers added to schedulers/ (zero modification to existing):\n\n"
        "  MLFQ   (Multi-Level Feedback Queue)\n"
        "         3 priority queues, quantums [4, 8, ∞], boost every 50 ticks.\n"
        "         This is the algorithm used in macOS and early Windows.\n"
        "         Shows how RL compares to the 'real OS' baseline.\n\n"
        "  Lottery (Waldspurger & Weihl, 1994)\n"
        "         Tickets proportional to priority; probabilistic fairness.\n"
        "         Starvation-free by design.\n\n"
        "  EDF    (Earliest Deadline First, Liu & Layland 1973)\n"
        "         Optimal for real-time tasks with known deadlines.\n"
        "         Deadlines derived as arrival + burst × 2.0.\n"
    )

    extra = [MLFQScheduler(), LotteryScheduler(), EDFScheduler()]
    results = []
    for sched in extra:
        r = _run_scheduler_extended(sched, processes,
                                     context_switch_cost=cfg.context_switch_cost)
        results.append(r)
        print(f"  {sched.name:<12}  avg_wait={r['avg_waiting_time']:>8.2f}  "
              f"response={r['avg_response_time']:>8.2f}  "
              f"fairness={r['fairness_index']:>6.4f}  "
              f"CS={r['context_switch_count']:>4}")

    print(
        "\n  ─── Usage in your own code ───────────────────────────────────\n"
        "  from schedulers.mlfq    import MLFQScheduler\n"
        "  from schedulers.lottery import LotteryScheduler\n"
        "  from schedulers.edf     import EDFScheduler\n"
        "  # Plugs directly into run_static_scheduler():\n"
        "  r = run_static_scheduler(MLFQScheduler(), processes)\n"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Statistical Significance Testing
# ─────────────────────────────────────────────────────────────────────────────

def run_significance_analysis(
    trial_data: Dict[str, List[float]],
    cfg: ExperimentConfig,
) -> None:
    """
    Feature 3: Print full significance test report for RL_Adaptive vs baselines.
    trial_data: dict mapping algorithm → list of per-trial waiting times.
    """
    _section("FEATURE 3 — STATISTICAL SIGNIFICANCE TESTING")
    print(
        "  Two complementary tests:\n"
        "    Welch's t-test       (parametric, assumes normality, good for n≥30)\n"
        "    Wilcoxon signed-rank (non-parametric, robust to outliers, paired)\n"
        "  Both must pass (p < α) for a strong publishable claim.\n"
        "  Cohen's d quantifies effect size (>0.8 = large effect).\n"
    )

    if "RL_Adaptive" not in trial_data:
        print("  [Feature 3] RL_Adaptive not found in trial_data — skipping.")
        return

    # Print the full significance table + verdict sentence
    print_significance_report(
        trial_data  = trial_data,
        target      = "RL_Adaptive",
        alpha       = cfg.significance_alpha,
    )

    print(
        "  ─── Usage in your own code ───────────────────────────────────\n"
        "  from evaluation.statistics import t_test, wilcoxon_test\n"
        "  t_res = t_test(rl_waits, fcfs_waits)\n"
        "  w_res = wilcoxon_test(rl_waits, fcfs_waits)\n"
        "  if t_res['significant'] and w_res['significant']:\n"
        "      print('RL scheduler improvement statistically significant (p < 0.05)')\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Context Switch Overhead Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_cs_cost_sweep(
    processes: List[Process],
    cfg: ExperimentConfig,
) -> None:
    """
    Feature 4: Show how scheduler rankings change as CS cost increases.
    Runs all schedulers at CS costs [0, 1, 2, 4] and prints a comparison table.
    """
    _section("FEATURE 4 — CONTEXT SWITCH OVERHEAD SENSITIVITY ANALYSIS")
    print(
        "  Context-switch cost injects N idle ticks whenever the running process\n"
        "  changes.  This penalises preemptive schedulers (RoundRobin, MLFQ)\n"
        "  and benefits non-preemptive ones (FCFS).\n\n"
        "  Real Linux kernel: context-switch overhead ≈ 1–10 μs.\n"
        "  We sweep cost ∈ {0, 1, 2, 4} to show ranking stability.\n"
    )

    schedulers = [
        ("FCFS",       __import__("schedulers.fcfs",        fromlist=["FCFSScheduler"]).FCFSScheduler()),
        ("RoundRobin", __import__("schedulers.round_robin",  fromlist=["RoundRobinScheduler"]).RoundRobinScheduler(3)),
        ("Priority",   __import__("schedulers.priority",     fromlist=["PriorityScheduler"]).PriorityScheduler()),
        ("MLFQ",       MLFQScheduler()),
        ("Lottery",    LotteryScheduler()),
        ("EDF",        EDFScheduler()),
    ]

    sweep_costs = [0, 1, 2, 4] if cfg.cs_cost_sweep else [cfg.context_switch_cost]

    # Header
    header = f"  {'Algorithm':<14}" + "".join(
        f" cs={c:>2}" for c in sweep_costs
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for name, sched in schedulers:
        row = f"  {name:<14}"
        for cost in sweep_costs:
            if hasattr(sched, "reset"):
                sched.reset()
            r = _run_scheduler_extended(sched, processes, context_switch_cost=cost)
            row += f" {r['avg_waiting_time']:>6.1f}"
        print(row)

    print(
        "\n  ─── Usage in your own code ───────────────────────────────────\n"
        "  from simulator.context_switch import ContextSwitchDispatcher\n"
        "  dispatcher = ContextSwitchDispatcher(clock, rq, context_switch_cost=2)\n"
        "  # Drop-in replacement for Dispatcher — same execute_tick() API\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Extended Metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_extended_metrics_demo(
    processes:  List[Process],
    cfg: ExperimentConfig,
) -> List[Dict]:
    """
    Feature 5: Run all schedulers (including extras) with full metric suite.
    Returns list of extended metric dicts for use in charts.
    """
    _section("FEATURE 5 — EXTENDED EVALUATION METRICS")
    print(
        "  New metrics in evaluation/metrics.py (added to existing experiment.py):\n\n"
        "    avg_response_time   — time from arrival to FIRST CPU touch\n"
        "    p95_waiting_time    — 95th-percentile wait (tail latency)\n"
        "    p99_waiting_time    — 99th-percentile wait\n"
        "    fairness_index      — Jain's fairness index ∈ (0,1]\n"
        "    slowdown            — mean(turnaround / burst)  [1.0 = ideal]\n"
        "    starvation_count    — processes waiting > 3× their burst time\n"
        "    makespan            — total simulation duration\n"
        "    context_switch_count— scheduler preemption count\n"
    )

    from schedulers.fcfs       import FCFSScheduler
    from schedulers.round_robin import RoundRobinScheduler
    from schedulers.priority    import PriorityScheduler
    from schedulers.sjf         import OracleSJFScheduler

    all_scheds = [
        FCFSScheduler(), OracleSJFScheduler(),
        RoundRobinScheduler(3), PriorityScheduler(),
        MLFQScheduler(), LotteryScheduler(), EDFScheduler(),
    ]

    results = []
    for sched in all_scheds:
        r = _run_scheduler_extended(sched, processes,
                                     context_switch_cost=cfg.context_switch_cost)
        results.append(r)

    print_extended_table(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Extended Visualization
# ─────────────────────────────────────────────────────────────────────────────

def run_extended_plots(
    trainer:         RLTrainer,
    trial_data:      Dict[str, List[float]],
    extended_results: List[Dict],
    cfg: ExperimentConfig,
) -> None:
    """
    Feature 6: Generate 3 additional publication charts.
    """
    _section("FEATURE 6 — EXTENDED VISUALIZATION (3 NEW CHARTS)")
    rdir = cfg.results_dir
    os.makedirs(rdir, exist_ok=True)

    # Chart 5: Boxplots — distribution shape across 30 trials
    plot_boxplots(
        trial_data   = trial_data,
        output_path  = os.path.join(rdir, "boxplots.png"),
        title        = f"Scheduler Performance Distribution — {{n}} Trials",
    )

    # Chart 6: Rich convergence curve with annotations
    plot_convergence_curve(
        episode_rewards = trainer.episode_rewards,
        output_path     = os.path.join(rdir, "convergence_curve.png"),
    )

    # Chart 7: Scheduler selection frequency (interpretability)
    # trainer.action_frequency maps action_index → count; map to names
    from rl_agent.trainer import SCHEDULER_NAMES
    selection = {
        SCHEDULER_NAMES[k]: v
        for k, v in trainer.action_frequency.items()
        if v > 0
    }
    plot_scheduler_selection(
        selection_counts = selection,
        output_path      = os.path.join(rdir, "scheduler_selection.png"),
    )

    # Bonus: radar chart if extended results are available
    if extended_results:
        # Filter out oracle for cleaner chart
        radar_results = [r for r in extended_results if r.get("algorithm") != "SJF_Oracle"]
        plot_extended_metrics_radar(
            results     = radar_results,
            output_path = os.path.join(rdir, "metrics_radar.png"),
        )

    print(
        "\n  ─── Usage in your own code ───────────────────────────────────\n"
        "  from visualization.extended_plots import (\n"
        "      plot_boxplots, plot_convergence_curve, plot_scheduler_selection\n"
        "  )\n"
        "  plot_boxplots(trial_data, 'results/boxplots.png')\n"
        "  plot_convergence_curve(trainer.episode_rewards, 'results/convergence.png')\n"
        "  plot_scheduler_selection(selection_counts, 'results/selection.png')\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_extended_experiment(cfg: ExperimentConfig = None):
    """
    Full research-grade experiment pipeline.

    Phases:
      0  Pre-train LSTM + train contextual bandit   (same as main.py)
      1  Feature 1  — real trace demo
      2  Feature 2  — extra scheduler baselines
      3  Feature 3  — statistical significance tests
      4  Feature 4  — context-switch overhead sweep
      5  Feature 5  — extended metrics table
      6  Feature 6  — 3 new visualisation charts
      7  Original 4 charts (same as main.py Phase 6)
    """
    if cfg is None:
        cfg = ExperimentConfig()

    gen = WorkloadGenerator()
    os.makedirs(cfg.results_dir, exist_ok=True)

    print("\n" + "╔" + "═" * 74 + "╗")
    print("║" + "  ADAPTIVE META-SCHEDULER — RESEARCH-GRADE EXPERIMENT".center(74) + "║")
    print("║" + "  6-Feature Upgrade  |  Trace + Stats + CS-cost + Extended Metrics".center(74) + "║")
    print("╚" + "═" * 74 + "╝")

    # ── Phase 0: original pipeline (verbatim from main.py) ──────────────────
    _section("PHASE 0 — ORIGINAL PIPELINE (unchanged main.py logic)")

    predictor = BurstTimePredictor()
    pretrain_predictor(predictor, verbose=True)

    q_table = QTable(
        learning_rate=0.25, discount_factor=0.0,
        epsilon=1.0, epsilon_min=0.03, epsilon_decay=0.990,
        replay_capacity=2000, replay_batch_size=16,
    )
    trainer = RLTrainer(q_table=q_table, predictor=predictor)
    trainer.train(workload_generator=gen,
                  num_episodes=cfg.num_episodes, verbose=True)

    # Multi-trial evaluation (same as main.py Phase 4)
    print(f"\n  Running {cfg.num_trials}-trial evaluation...", flush=True)
    summary = run_multi_trial_evaluation(
        trainer=trainer, predictor=predictor,
        workload_generator=gen,
        num_trials=cfg.num_trials,
        workload_type=cfg.eval_workload_type,
    )
    print_statistical_table(summary, n_trials=cfg.num_trials)

    # Collect per-trial waiting times for significance testing (Feature 3)
    # We re-run a smaller trial set to get raw per-trial values (summary only stores mean±std)
    trial_waits: Dict[str, List[float]] = {}
    print(f"\n  Collecting per-trial wait lists for significance tests...", flush=True)
    for _ in range(cfg.num_trials):
        wl = gen.generate(workload_type=cfg.eval_workload_type)
        for r in run_all_static_schedulers(wl):
            n = r["algorithm"]
            trial_waits.setdefault(n, []).append(r["avg_waiting_time"])
        pr = run_predicted_sjf(copy.deepcopy(predictor), wl)
        trial_waits.setdefault("SJF_Predicted", []).append(pr["avg_waiting_time"])
        rl = trainer.evaluate(wl)
        trial_waits.setdefault("RL_Adaptive", []).append(rl["avg_waiting_time"])

    # Demo workload for Features 2, 4, 5
    demo_wl = gen.generate(workload_type=cfg.eval_workload_type)

    # ── Feature 1 ────────────────────────────────────────────────────────────
    if cfg.use_real_traces:
        run_trace_demo(cfg)

    # ── Feature 2 ────────────────────────────────────────────────────────────
    extra_results = []
    if cfg.extra_schedulers:
        extra_results = run_extra_schedulers(demo_wl, cfg)

    # ── Feature 3 ────────────────────────────────────────────────────────────
    if cfg.significance_testing:
        run_significance_analysis(trial_waits, cfg)

    # ── Feature 4 ────────────────────────────────────────────────────────────
    run_cs_cost_sweep(demo_wl, cfg)

    # ── Feature 5 ────────────────────────────────────────────────────────────
    extended_results = []
    if cfg.extended_metrics:
        extended_results = run_extended_metrics_demo(demo_wl, cfg)

    # ── Feature 6 ────────────────────────────────────────────────────────────
    if cfg.extended_plots:
        run_extended_plots(trainer, trial_waits, extended_results, cfg)

    # ── Original 4 charts (identical to main.py Phase 6) ────────────────────
    _section("ORIGINAL CHARTS (same as main.py)")
    static = run_all_static_schedulers(demo_wl)
    pred_r = run_predicted_sjf(copy.deepcopy(predictor), demo_wl)
    rl_res = trainer.evaluate(demo_wl)
    rl_row = {
        "algorithm":           f"RL_Adaptive[{rl_res['chosen_scheduler']}]",
        "avg_waiting_time":    round(rl_res["avg_waiting_time"],    2),
        "avg_turnaround_time": round(rl_res["avg_turnaround_time"], 2),
        "cpu_utilization":     99.9,
        "throughput":          0.0,
    }

    plot_comparison(
        results=static + [pred_r, rl_row],
        output_path=os.path.join(cfg.results_dir, "comparison_single_run.png"),
    )
    plot_training_curve(
        episode_rewards=trainer.episode_rewards,
        output_path=os.path.join(cfg.results_dir, "training_curve.png"),
    )
    plot_prediction_accuracy(
        prediction_errors=predictor.prediction_errors,
        output_path=os.path.join(cfg.results_dir, "lstm_accuracy.png"),
    )
    plot_statistical_comparison(
        summary=summary,
        output_path=os.path.join(cfg.results_dir, "statistical_comparison.png"),
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    rdir = cfg.results_dir
    rl_w  = summary.get("RL_Adaptive", {}).get("mean_wait", 999)
    beaten = sum(
        1 for k, v in summary.items()
        if k not in ("RL_Adaptive", "SJF_Oracle") and v["mean_wait"] > rl_w
    )
    total_r = len([k for k in summary if k not in ("RL_Adaptive", "SJF_Oracle")])

    print(f"""
╔{'═'*74}╗
║  EXPERIMENT COMPLETE{'':53}║
╠{'═'*74}╣
║  Original charts (main.py)                                               ║
║    {rdir}/comparison_single_run.png                                      ║
║    {rdir}/training_curve.png                                             ║
║    {rdir}/lstm_accuracy.png                                              ║
║    {rdir}/statistical_comparison.png  ← KEY RESULT                      ║
╠{'═'*74}╣
║  New charts (Feature 6)                                                  ║
║    {rdir}/boxplots.png                ← distribution shape               ║
║    {rdir}/convergence_curve.png       ← convergence analysis             ║
║    {rdir}/scheduler_selection.png     ← interpretability                 ║
║    {rdir}/metrics_radar.png           ← multi-metric overview            ║
╠{'═'*74}╣
║  RL_Adaptive beats {beaten}/{total_r} realistic algorithms                             ║
║  LSTM MAE: {predictor.mean_absolute_error():.2f} ticks overall                                       ║
╚{'═'*74}╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive Meta-Scheduler — Research-Grade Experiment Runner"
    )
    p.add_argument("--quick",       action="store_true",
                   help="Quick smoke test: 100 episodes, 10 trials")
    p.add_argument("--no-traces",   action="store_true",
                   help="Skip Feature 1 (real trace demo)")
    p.add_argument("--no-extras",   action="store_true",
                   help="Skip Feature 2 (extra schedulers)")
    p.add_argument("--no-stats",    action="store_true",
                   help="Skip Feature 3 (significance tests)")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip Feature 6 (extended plots)")
    p.add_argument("--cs-cost",     type=int, default=0,
                   help="Context-switch cost in ticks (Feature 4, default 0)")
    p.add_argument("--episodes",    type=int, default=600,
                   help="Number of training episodes (default 600)")
    p.add_argument("--trials",      type=int, default=30,
                   help="Number of evaluation trials (default 30)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = ExperimentConfig(
        num_episodes         = 100 if args.quick else args.episodes,
        num_trials           = 10  if args.quick else args.trials,
        use_real_traces      = not args.no_traces,
        extra_schedulers     = not args.no_extras,
        significance_testing = not args.no_stats,
        context_switch_cost  = args.cs_cost,
        cs_cost_sweep        = not args.quick,
        extended_metrics     = True,
        extended_plots       = not args.no_plots,
    )

    run_extended_experiment(cfg)
