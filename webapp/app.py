"""
webapp/app.py
=============
Flask frontend for the Adaptive Meta-Scheduler.

How this differs from docs/dashboard.html
------------------------------------------
dashboard.html is a from-scratch JS port of the LSTM + Q-table, built so it
can run with zero server (open the file, done). This app instead imports
and calls the REAL Python simulator, the real scheduler classes, and the
real trained LSTM/bandit directly — there is no second implementation to
drift out of sync with the one described in the report. The tradeoff is
exactly that: this needs `python webapp/app.py` running, dashboard.html
doesn't.

Run
---
    pip install -r requirements.txt   # now includes Flask
    python webapp/app.py
    -> http://127.0.0.1:5000

Routes
------
    GET  /                  the UI
    GET  /api/config        workload types, real-trace window count, algorithm list
    POST /api/workload      generate a workload -> list of process dicts
    POST /api/run           run ONE algorithm against a given workload -> timeline + metrics
    POST /api/compare       run ALL algorithms against a given workload -> metrics table
"""

import copy
import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from workload.generator import WorkloadGenerator
from workload.trace_loader import load_alibaba_pai_trial_windows
from simulator.process import Process
from burst_predictor.lstm_predictor import BurstTimePredictor
from rl_agent.q_table import QTable
from rl_agent.trainer import RLTrainer, SCHEDULER_NAMES, _make_schedulers
from evaluation.experiment import run_static_scheduler, run_predicted_sjf
from schedulers.fcfs import FCFSScheduler
from schedulers.sjf import OracleSJFScheduler
from schedulers.round_robin import RoundRobinScheduler
from schedulers.priority import PriorityScheduler
from schedulers.mlfq import MLFQScheduler
from schedulers.lottery import LotteryScheduler
from schedulers.edf import EDFScheduler
from schedulers.sjf_predicted import SJFPredictedScheduler

from webapp.sim_runner import run_with_timeline

LSTM_WEIGHTS_PATH = os.path.join(ROOT, "results", "lstm_weights.json")
Q_TABLE_PATH = os.path.join(ROOT, "results", "q_table.json")
REAL_TRACE_PATH = os.path.join(ROOT, "data", "real_traces",
                                "alibaba_pai_gpu_trace_100k.csv")
N_REAL_WINDOWS = 30
REAL_WINDOW_SIZE = 150

STATIC_ALGOS = ["FCFS", "SJF_Oracle", "RoundRobin", "Priority",
                "MLFQ", "Lottery", "EDF"]
ALL_ALGOS = STATIC_ALGOS + ["SJF_Predicted", "RL_Adaptive"]

app = Flask(__name__)


# ── model loading (same pattern as scripts/run_real_trace_evaluation.py) ────

def _load_trained_predictor(path: str):
    predictor = BurstTimePredictor()
    if not os.path.exists(path):
        print(f"[webapp] {path} not found — using an UNTRAINED LSTM. "
              f"Run scripts/export_results.py first.")
        return predictor, False
    with open(path) as f:
        w = json.load(f)
    for k, v in w["cell"].items():
        setattr(predictor.cell, k, np.array(v))
    predictor.Wo = np.array(w["output"]["Wo"])
    predictor.bo = np.array(w["output"]["bo"])
    return predictor, True


def _load_trained_q_table(path: str):
    q_table = QTable(learning_rate=0.25, discount_factor=0.0,
                      epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
    if not os.path.exists(path):
        print(f"[webapp] {path} not found — using an UNTRAINED Q-table. "
              f"Run scripts/export_results.py first.")
        return q_table, False
    q_table.load(path)
    return q_table, True


_BASE_PREDICTOR, _LSTM_LOADED = _load_trained_predictor(LSTM_WEIGHTS_PATH)
_BASE_Q_TABLE, _QTABLE_LOADED = _load_trained_q_table(Q_TABLE_PATH)
MODEL_TRAINED = _LSTM_LOADED and _QTABLE_LOADED
if not MODEL_TRAINED:
    print("[webapp] WARNING: serving with an untrained model — "
          "SJF_Predicted and RL_Adaptive results will be meaningless "
          "until you run `python scripts/export_results.py`.")


def _fresh_trainer() -> RLTrainer:
    """A per-request deep copy so concurrent requests never share (or
    silently mutate) the one set of loaded weights."""
    return RLTrainer(q_table=copy.deepcopy(_BASE_Q_TABLE),
                      predictor=copy.deepcopy(_BASE_PREDICTOR))


def _static_scheduler_factory():
    return {
        "FCFS": FCFSScheduler,
        "SJF_Oracle": OracleSJFScheduler,
        "RoundRobin": lambda: RoundRobinScheduler(time_quantum=3),
        "Priority": PriorityScheduler,
        "MLFQ": MLFQScheduler,
        "Lottery": LotteryScheduler,
        "EDF": EDFScheduler,
    }


# ── workload (de)serialization ───────────────────────────────────────────────

def _process_to_dict(p: Process) -> dict:
    return {"pid": p.pid, "arrival_time": p.arrival_time, "burst_time": p.burst_time,
            "priority": p.priority, "process_type": p.process_type}


def _dict_to_process(d: dict) -> Process:
    return Process(pid=d["pid"], arrival_time=d["arrival_time"],
                    burst_time=d["burst_time"], priority=d["priority"],
                    process_type=d.get("process_type", "mixed"))


def _processes_from_request(body: dict):
    raw = body.get("processes")
    if not raw:
        raise ValueError("Request body must include a non-empty 'processes' list "
                          "(call /api/workload first).")
    return [_dict_to_process(d) for d in raw]


# ── routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", model_trained=MODEL_TRAINED)


@app.route("/api/config")
def api_config():
    return jsonify({
        "workload_types": ["interactive", "mixed", "cpu_bound"],
        "n_real_windows": N_REAL_WINDOWS,
        "real_window_size": REAL_WINDOW_SIZE,
        "static_algorithms": STATIC_ALGOS,
        "all_algorithms": ALL_ALGOS,
        "model_trained": MODEL_TRAINED,
    })


@app.route("/api/workload", methods=["POST"])
def api_workload():
    body = request.get_json(force=True) or {}
    source = body.get("source", "synthetic")

    if source == "real":
        window_index = int(body.get("window_index", 0))
        if not (0 <= window_index < N_REAL_WINDOWS):
            return jsonify({"error": f"window_index must be 0-{N_REAL_WINDOWS - 1}"}), 400
        if not os.path.exists(REAL_TRACE_PATH):
            return jsonify({"error": "Real trace file not bundled in this checkout — "
                                      "see data/real_traces/PROVENANCE.md"}), 404
        windows = load_alibaba_pai_trial_windows(
            path=REAL_TRACE_PATH, n_trials=N_REAL_WINDOWS, window_size=REAL_WINDOW_SIZE)
        procs = windows[window_index]
        label = f"Real: Alibaba PAI GPU trace, window {window_index + 1}/{N_REAL_WINDOWS}"
    else:
        workload_type = body.get("workload_type", "mixed")
        gen = WorkloadGenerator()
        procs = gen.generate(workload_type=workload_type)
        label = f"Synthetic: {workload_type}, n={len(procs)}"

    return jsonify({"label": label, "processes": [_process_to_dict(p) for p in procs]})


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True) or {}
    try:
        processes = _processes_from_request(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    algo = body.get("algorithm", "FCFS")

    if algo == "SJF_Predicted":
        trainer = _fresh_trainer()
        sched = SJFPredictedScheduler(trainer.predictor)
        result = run_with_timeline(sched, processes, name="SJF_Predicted",
                                    predictor=trainer.predictor)
        return jsonify(result)

    if algo == "RL_Adaptive":
        trainer = _fresh_trainer()
        decision = trainer.evaluate(processes)
        chosen = decision["chosen_scheduler"]              # e.g. "SJF_Pred"
        ep_pred = copy.deepcopy(trainer.predictor)
        schedulers = _make_schedulers(ep_pred)              # {0: FCFS, 1: SJF_Pred, ...}
        action_idx = {v: k for k, v in SCHEDULER_NAMES.items()}[chosen]
        needs_predictor = chosen == "SJF_Pred"
        result = run_with_timeline(
            schedulers[action_idx], processes, name="RL_Adaptive",
            predictor=ep_pred if needs_predictor else None,
            train_mode="end_of_episode")
        result["chosen_scheduler"] = chosen
        return jsonify(result)

    factory = _static_scheduler_factory().get(algo)
    if factory is None:
        return jsonify({"error": f"Unknown algorithm '{algo}'. Choose from {ALL_ALGOS}"}), 400
    result = run_with_timeline(factory(), processes, name=algo)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    body = request.get_json(force=True) or {}
    try:
        processes = _processes_from_request(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    results = []
    for name, factory in _static_scheduler_factory().items():
        r = run_static_scheduler(factory(), processes)
        results.append(r)

    trainer = _fresh_trainer()
    pr = run_predicted_sjf(trainer.predictor, processes)
    results.append(pr)

    trainer2 = _fresh_trainer()
    rl = trainer2.evaluate(processes)
    results.append({"algorithm": "RL_Adaptive", "avg_waiting_time": rl["avg_waiting_time"],
                     "avg_turnaround_time": rl["avg_turnaround_time"],
                     "cpu_utilization": rl.get("cpu_utilization"),
                     "throughput": rl.get("throughput"),
                     "completed_count": rl.get("completed_count"),
                     "chosen_scheduler": rl["chosen_scheduler"]})

    return jsonify({"results": results, "model_trained": MODEL_TRAINED})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
