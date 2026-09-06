"""
schedulers/sjf.py
==================
Shortest Job First — oracle mode and two-tier predicted mode.

See sjf_predicted.py for full documentation of the estimation strategy.
"""

from typing import Optional
from simulator.ready_queue import ReadyQueue
from simulator.process import Process

TYPE_BURST_PRIOR = {
    "interactive": 4.5,
    "mixed":       18.0,
    "cpu_bound":   27.5,
}


class SJFScheduler:
    """
    SRTF (preemptive SJF).
    predictor=None  → oracle (uses true remaining_time).
    predictor=obj   → two-tier predicted (type prior + LSTM refinement).
    """
    name = "SJF"

    def __init__(self, predictor=None):
        self.predictor = predictor

    def _estimate(self, proc, avg_burst):
        prior     = TYPE_BURST_PRIOR.get(
            getattr(proc, "process_type", "mixed"), avg_burst)
        lstm_pred = self.predictor.predict(proc.pid, fallback=prior)
        return 0.70 * prior + 0.30 * lstm_pred

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        if ready_queue.is_empty():
            return None
        if self.predictor is None:
            return min(ready_queue.get_all(), key=lambda p: p.remaining_time)
        avg = ready_queue.average_burst_time()
        return min(ready_queue.get_all(), key=lambda p: self._estimate(p, avg))


class OracleSJFScheduler:
    """Pure oracle SJF — for benchmark table only."""
    name = "SJF_Oracle"

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        if ready_queue.is_empty():
            return None
        return min(ready_queue.get_all(), key=lambda p: p.remaining_time)
