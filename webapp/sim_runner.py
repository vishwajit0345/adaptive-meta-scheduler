"""
webapp/sim_runner.py
=====================
Timeline-capturing wrapper around the REAL simulator, for the Flask
frontend's Gantt chart.

This is NOT a second simulation model. docs/dashboard.html is a from-
scratch JS port (necessary there, since it has to run with no server); this
instead calls simulator/dispatcher.py, the actual scheduler classes, and
evaluation.experiment._metrics directly, one tick at a time. If this ever
disagrees with evaluation/experiment.py or rl_agent/trainer.py on the same
input, that is a bug in this file, not a second source of truth.

Why `train_mode` exists (read before changing this file)
----------------------------------------------------------
The codebase has TWO different LSTM-backed schedulers with two different
online-learning cadences, and this wrapper has to reproduce whichever one
is actually in play, or the Flask visualization would show a different
(and better- or worse-looking) run than the real evaluation it claims to
represent:

  - schedulers/sjf_predicted.py's SJFPredictedScheduler self-primes the
    LSTM cache inside its own select_next() ("prime on first encounter"),
    and evaluation.experiment.run_predicted_sjf() trains it with
    record_burst() + train_on_buffer(epochs=1) after EVERY tick that
    completes a process. train_mode="per_tick" replays this.

  - schedulers/sjf.py's SJFScheduler (the one rl_agent/trainer.py's
    _make_schedulers() actually instantiates for the bandit's "SJF_Pred"
    action) does NOT self-prime — it only reads predictor.predict(pid),
    which returns `fallback` for any pid never passed to
    predictor.predict_process() by the CALLER. rl_agent/trainer.py's
    _run_full_episode() calls predict_process() itself at admission time,
    and trains with train_on_buffer(epochs=2) ONCE at the end of the
    episode, not per-tick. train_mode="end_of_episode" replays this.

Get this wrong and the failure mode is silent, not a crash: SJFScheduler
without admission-time priming just quietly falls back to the type-prior
heuristic every time (LSTM contribution always 0), so RL_Adaptive's
Gantt/metrics here would look plausible while not actually reflecting the
model the report describes.
"""

import copy
from typing import List, Optional

from simulator.clock import SimulationClock
from simulator.ready_queue import ReadyQueue
from simulator.dispatcher import Dispatcher
from simulator.process import Process
from evaluation.experiment import _metrics

MAX_TICK_CAP = 20000  # hard ceiling regardless of workload, so a pathological
                       # request from the frontend can't hang the server


def run_with_timeline(scheduler, processes: List[Process], name: Optional[str] = None,
                       predictor=None, train_mode: str = "per_tick") -> dict:
    """
    Run `scheduler` against `processes` tick-by-tick, recording a Gantt-
    ready timeline alongside the same aggregate metrics
    evaluation.experiment.run_static_scheduler() would produce.

    Parameters
    ----------
    predictor  : pass the BurstTimePredictor instance `scheduler` was built
                 with (SJFScheduler or SJFPredictedScheduler) to replay the
                 real online-learning behaviour. None for every classical
                 scheduler (FCFS/RoundRobin/Priority/MLFQ/Lottery/EDF).
    train_mode : "per_tick" - evaluation.experiment.run_predicted_sjf()'s
                 cadence (train_on_buffer(epochs=1) after every completion).
                 Use for the standalone "SJF_Predicted" baseline.
                 "end_of_episode" - rl_agent.trainer._run_full_episode()'s
                 cadence (train_on_buffer(epochs=2) once, at the end). Use
                 when re-running whatever scheduler RL_Adaptive chose.

    Returns
    -------
    {
      "algorithm": str,
      "metrics": {avg_waiting_time, avg_turnaround_time, cpu_utilization,
                  throughput, completed_count},
      "timeline": [{"tick": int, "pid": int|None}, ...],
      "processes": [{pid, arrival_time, burst_time, priority, process_type,
                      waiting_time, turnaround_time, finish_time,
                      response_time}, ...]   # completed processes only
    }
    """
    if train_mode not in ("per_tick", "end_of_episode"):
        raise ValueError(f"train_mode must be 'per_tick' or 'end_of_episode', got {train_mode!r}")

    procs = [copy.deepcopy(p) for p in processes]
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    if hasattr(scheduler, "reset"):
        scheduler.reset()

    completed: List[Process] = []
    max_ticks = min(MAX_TICK_CAP, sum(p.burst_time for p in procs) * 3 + 200)
    timeline = []

    while len(completed) < len(procs) and clock.current_time <= max_ticks:
        for p in procs:
            if (p.arrival_time == clock.current_time
                    and not p.is_complete() and p not in rq.get_all()):
                rq.add_process(p)
                if predictor is not None:
                    # Idempotent/cached (BurstTimePredictor.predict_process
                    # returns early if process.pid is already cached) -
                    # harmless to call even for schedulers that self-prime,
                    # required for ones (SJFScheduler) that don't.
                    predictor.predict_process(p)

        tick = clock.current_time            # capture BEFORE execute_tick advances it
        chosen = scheduler.select_next(rq)
        before = len(completed)
        dispatcher.execute_tick(chosen, completed)
        timeline.append({"tick": tick, "pid": chosen.pid if chosen else None})

        if predictor is not None:
            for fp in completed[before:]:
                predictor.record_burst(fp.pid, fp.burst_time)
            if train_mode == "per_tick" and len(completed) > before:
                predictor.train_on_buffer(epochs=1)

    if predictor is not None and train_mode == "end_of_episode":
        predictor.train_on_buffer(epochs=2)

    algo_name = name or getattr(scheduler, "name", scheduler.__class__.__name__)
    metrics = _metrics(completed, clock, algo_name)

    by_pid = {p.pid: p for p in completed}
    per_process = [{
        "pid": pid,
        "arrival_time": p.arrival_time,
        "burst_time": p.burst_time,
        "priority": p.priority,
        "process_type": p.process_type,
        "waiting_time": p.waiting_time,
        "turnaround_time": p.turnaround_time,
        "finish_time": p.finish_time,
        "response_time": p.response_time,
    } for pid, p in sorted(by_pid.items())]

    return {
        "algorithm": algo_name,
        "metrics": metrics,
        "timeline": timeline,
        "processes": per_process,
        "truncated": len(completed) < len(procs),  # hit the tick cap before finishing
    }
