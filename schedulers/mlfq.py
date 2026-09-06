"""
schedulers/mlfq.py
==================
Multi-Level Feedback Queue (MLFQ) Scheduler.

MLFQ is the dominant scheduling algorithm in modern operating systems
(macOS, Linux CFS is inspired by it; Windows uses a priority-based variant).
It approximates SJF behaviour WITHOUT needing oracle burst-time knowledge
by demoting long-running jobs to lower-priority queues over time.

ALGORITHM
---------
  • 3 priority queues (Q0 = highest, Q2 = lowest)
  • Q0 : quantum = 4   (short, favours interactive jobs)
  • Q1 : quantum = 8   (medium)
  • Q2 : quantum = ∞   (FCFS — background batch jobs)
  • On arrival   : process enters Q0
  • After using full quantum in Qi : demoted to Q(i+1)
  • If process completes before quantum : stays at same level (it was short)
  • Priority boost : every BOOST_INTERVAL ticks all processes reset to Q0
                     (prevents starvation of long-running jobs)

WHY THIS IS A USEFUL BASELINE
------------------------------
MLFQ is the standard "real OS" comparison point in scheduling research.
Including it allows the paper to make the claim:
  "RL_Adaptive matches or beats MLFQ, which is what real OSes use."

INTERFACE
---------
Identical to FCFS, SJF, RoundRobin — implements select_next(ready_queue).
Plugs into Dispatcher, experiment.py, and the extended experiment runner
with zero modification to any existing file.
"""

from typing import Optional, List, Dict
from simulator.ready_queue import ReadyQueue
from simulator.process import Process


# ── MLFQ configuration ────────────────────────────────────────────────────────

NUM_QUEUES     = 3
QUANTUMS       = [4, 8, None]          # None = run to completion (FCFS)
BOOST_INTERVAL = 50                    # ticks between priority boosts


class MLFQScheduler:
    """
    Multi-Level Feedback Queue scheduler.

    Attributes
    ----------
    num_queues      : Number of priority levels (default 3).
    quantums        : Time quantum per level (None = unlimited).
    boost_interval  : Ticks between full priority resets (anti-starvation).
    """

    name = "MLFQ"

    def __init__(
        self,
        num_queues:     int        = NUM_QUEUES,
        quantums:       List       = None,
        boost_interval: int        = BOOST_INTERVAL,
    ):
        self.num_queues     = num_queues
        self.quantums       = quantums if quantums is not None else QUANTUMS
        self.boost_interval = boost_interval

        # Internal state — reset() clears these
        self._queues:       List[List[Process]] = [[] for _ in range(num_queues)]
        self._proc_level:   Dict[int, int]      = {}   # pid → queue level
        self._ticks_used:   Dict[int, int]      = {}   # pid → ticks at current level
        self._current:      Optional[Process]   = None
        self._global_tick:  int                 = 0

    # ── main scheduling decision ──────────────────────────────────────────────

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        """
        Return the next process to run according to MLFQ policy.

        Called every tick by the dispatcher.
        """
        if ready_queue.is_empty():
            self._current = None
            return None

        # NOTE: iterate a list (ReadyQueue's own insertion/arrival order),
        # not a bare `set(...)`. Process has no custom __hash__/__eq__, so a
        # set of Process objects orders by Python's default identity hash —
        # stable within one run, but not guaranteed stable *across* runs
        # (e.g. under ASLR), which made priority-boost re-admission order,
        # and therefore FCFS tie-breaking within a level, silently
        # irreproducible even with a fixed random.seed()/np.random.seed().
        # `available` (the set) is kept only for O(1) membership tests below.
        available_list = ready_queue.get_all()
        available = set(available_list)

        # Admit any newly-arrived processes into Q0
        for proc in available_list:
            if proc.pid not in self._proc_level:
                self._proc_level[proc.pid] = 0
                self._ticks_used[proc.pid] = 0
                self._queues[0].append(proc)

        # Remove completed processes from internal queues
        for level in range(self.num_queues):
            self._queues[level] = [
                p for p in self._queues[level] if p in available
            ]

        # Priority boost: reset all surviving processes to Q0 periodically
        self._global_tick += 1
        if self._global_tick % self.boost_interval == 0:
            self._priority_boost(available_list)

        # Check if current process has exhausted its quantum → demote
        if self._current is not None and self._current in available:
            pid   = self._current.pid
            level = self._proc_level.get(pid, 0)
            quantum = self.quantums[level]

            if quantum is not None and self._ticks_used.get(pid, 0) >= quantum:
                # Demote to next level
                new_level = min(level + 1, self.num_queues - 1)
                # Move from current level queue to new level queue
                if self._current in self._queues[level]:
                    self._queues[level].remove(self._current)
                if self._current not in self._queues[new_level]:
                    self._queues[new_level].append(self._current)
                self._proc_level[pid] = new_level
                self._ticks_used[pid] = 0
                self._current = None

        # Pick from the highest non-empty queue (FCFS within each queue)
        chosen = None
        for level in range(self.num_queues):
            queue = [p for p in self._queues[level] if p in available]
            if queue:
                chosen = queue[0]   # FCFS within level
                break

        if chosen is None:
            self._current = None
            return None

        # Update tick counter for chosen process
        self._ticks_used[chosen.pid] = self._ticks_used.get(chosen.pid, 0) + 1
        self._current = chosen
        return chosen

    def _priority_boost(self, available_list):
        """
        Move all surviving processes back to Q0 (anti-starvation).

        `available_list` must be an ordered sequence (not a bare set) so
        that FCFS order within the freshly-reset Q0 is reproducible.
        """
        # Clear all queues
        for level in range(self.num_queues):
            self._queues[level] = []

        # Re-admit everything into Q0
        for proc in available_list:
            self._proc_level[proc.pid] = 0
            self._ticks_used[proc.pid] = 0
            self._queues[0].append(proc)

        self._current = None

    def reset(self):
        """Reset all internal state for a fresh simulation run."""
        self._queues      = [[] for _ in range(self.num_queues)]
        self._proc_level  = {}
        self._ticks_used  = {}
        self._current     = None
        self._global_tick = 0
