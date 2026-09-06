"""
priority.py - Preemptive Priority scheduling algorithm.

Processes are assigned integer priorities; lower value = higher urgency.
The scheduler always runs the highest-priority ready process. A new
arrival with a higher priority immediately preempts the running process.

Ageing is applied to prevent starvation: every 10 ticks a waiting
process's effective priority is boosted by 1 level.
"""

from typing import Optional
from simulator.ready_queue import ReadyQueue
from simulator.process import Process


class PriorityScheduler:
    """
    Preemptive Priority scheduler with ageing to prevent starvation.

    Selection policy: run the process with the lowest priority number
                      (i.e. highest urgency).
    Preemption     : Yes — a higher-priority arrival preempts the CPU.
    Anti-starvation: Effective priority improves after prolonged waiting.

    Attributes:
        ageing_interval : Ticks of waiting before a priority boost is awarded.
        _effective_priority: Per-process adjusted priority accounting for ageing.
    """

    name = "Priority"
    ageing_interval: int = 10   # Boost priority every N waiting ticks

    def __init__(self):
        # Maps pid -> effective priority (starts equal to assigned priority)
        self._effective_priority: dict = {}

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        """
        Select the highest effective-priority process, applying ageing.

        Args:
            ready_queue: The current ready queue snapshot.

        Returns:
            The highest-priority process, or None if queue is empty.
        """
        if ready_queue.is_empty():
            return None

        available = ready_queue.get_all()

        # Initialise effective priority for new arrivals
        for process in available:
            if process.pid not in self._effective_priority:
                self._effective_priority[process.pid] = process.priority

        # Apply ageing: boost priority for processes that have waited a long time
        for process in available:
            accumulated_wait = process.waiting_time
            if accumulated_wait > 0 and accumulated_wait % self.ageing_interval == 0:
                # Reduce priority number = increase urgency (floor at 1)
                self._effective_priority[process.pid] = max(
                    1, self._effective_priority[process.pid] - 1
                )

        # Select the process with the smallest effective priority number
        best_process = min(
            available,
            key=lambda proc: (
                self._effective_priority.get(proc.pid, proc.priority),
                proc.arrival_time,   # tie-break by arrival order
            ),
        )
        return best_process

    def reset(self):
        """Reset priority bookkeeping for a fresh simulation run."""
        self._effective_priority = {}
