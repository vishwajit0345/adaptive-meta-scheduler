"""
schedulers/edf.py
==================
Earliest Deadline First (EDF) Scheduler.

EDF is provably optimal for single-processor real-time scheduling: if any
feasible schedule exists that meets all deadlines, EDF will find one.
(Liu & Layland, 1973 — one of the most cited papers in OS research.)

DEADLINE ASSIGNMENT
-------------------
Real-time systems have explicit deadlines.  Our simulator's Process objects
don't carry an explicit deadline field, so we derive one at first-seen time:

    deadline(proc) = proc.arrival_time + proc.burst_time × DEADLINE_FACTOR

DEADLINE_FACTOR = 2.0 means a process must finish within twice its burst
time.  Tighter deadlines (factor → 1.0) make EDF behave more like SJF.
Looser deadlines (factor >> 1) make it behave more like FCFS.

WHY USEFUL AS A BASELINE
--------------------------
EDF is the gold standard for real-time scheduling.  Including it in the
comparison allows the paper to claim:
  "RL_Adaptive is competitive with EDF despite having no deadline knowledge."

EDF also provides a useful academic contrast: it is optimal under its own
model (real-time with known deadlines) but may perform worse than RL on
general-purpose mixed workloads where deadlines are artificial.

INTERFACE
---------
Identical to all other schedulers — implements select_next(ready_queue).
"""

from typing import Optional, Dict
from simulator.ready_queue import ReadyQueue
from simulator.process import Process


DEADLINE_FACTOR = 2.0   # deadline = arrival + burst × factor


class EDFScheduler:
    """
    Earliest Deadline First (preemptive).

    Deadlines are derived at first encounter:
        deadline = arrival_time + burst_time * deadline_factor

    The process with the earliest (smallest) deadline runs next.
    Ties are broken by arrival time (FCFS).

    Attributes
    ----------
    deadline_factor : Multiplier for deriving deadline from burst time.
    _deadlines      : Per-process deadline cache (pid → absolute deadline).
    """

    name = "EDF"

    def __init__(self, deadline_factor: float = DEADLINE_FACTOR):
        self.deadline_factor = deadline_factor
        self._deadlines: Dict[int, float] = {}

    def _get_deadline(self, process: Process) -> float:
        """Return cached deadline, computing it on first encounter."""
        if process.pid not in self._deadlines:
            self._deadlines[process.pid] = (
                process.arrival_time + process.burst_time * self.deadline_factor
            )
        return self._deadlines[process.pid]

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        """
        Select the process with the earliest deadline.

        Args:
            ready_queue: The current ready queue snapshot.

        Returns:
            Process with the minimum deadline, or None if queue is empty.
        """
        if ready_queue.is_empty():
            return None

        return min(
            ready_queue.get_all(),
            key=lambda p: (self._get_deadline(p), p.arrival_time),
        )

    def reset(self):
        """Clear deadline cache for a fresh simulation run."""
        self._deadlines = {}
