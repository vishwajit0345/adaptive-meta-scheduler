"""
ready_queue.py - Manages the collection of processes waiting for CPU time.

The ReadyQueue acts as the central waiting area. Processes are admitted
when their arrival_time matches the current simulation clock tick.
"""

from typing import List, Optional
from simulator.process import Process


class ReadyQueue:
    """
    Holds processes that have arrived and are waiting for CPU allocation.

    Provides helper metrics used by the RL state encoder to characterise
    the current workload pressure on the scheduler.
    """

    def __init__(self):
        self._queue: List[Process] = []

    # ------------------------------------------------------------------
    # Core queue operations
    # ------------------------------------------------------------------

    def add_process(self, process: Process):
        """Admit a newly-arrived process into the ready queue."""
        self._queue.append(process)

    def remove_process(self, process: Process):
        """Remove a specific process after it has been dispatched."""
        if process in self._queue:
            self._queue.remove(process)

    def get_all(self) -> List[Process]:
        """Return a shallow copy of all queued processes."""
        return list(self._queue)

    def is_empty(self) -> bool:
        """Return True when no processes are waiting."""
        return len(self._queue) == 0

    def size(self) -> int:
        """Number of processes currently in the ready queue."""
        return len(self._queue)

    def peek_shortest(self) -> Optional[Process]:
        """Return the process with the smallest remaining burst time."""
        if self.is_empty():
            return None
        return min(self._queue, key=lambda proc: proc.remaining_time)

    def peek_highest_priority(self) -> Optional[Process]:
        """Return the process with the highest priority (lowest number)."""
        if self.is_empty():
            return None
        return min(self._queue, key=lambda proc: proc.priority)

    def peek_fcfs(self) -> Optional[Process]:
        """Return the process that arrived earliest."""
        if self.is_empty():
            return None
        return min(self._queue, key=lambda proc: proc.arrival_time)

    # ------------------------------------------------------------------
    # Workload metric helpers used by the RL state encoder
    # ------------------------------------------------------------------

    def average_burst_time(self) -> float:
        """Mean burst time across all queued processes."""
        if self.is_empty():
            return 0.0
        return sum(proc.burst_time for proc in self._queue) / len(self._queue)

    def variance_burst_time(self) -> float:
        """Variance of burst times — indicates workload heterogeneity."""
        if self.size() < 2:
            return 0.0
        mean = self.average_burst_time()
        return sum((proc.burst_time - mean) ** 2 for proc in self._queue) / len(
            self._queue
        )

    def average_waiting_time(self) -> float:
        """Mean accumulated waiting time — measures starvation pressure."""
        if self.is_empty():
            return 0.0
        return sum(proc.waiting_time for proc in self._queue) / len(self._queue)

    def max_waiting_time(self) -> int:
        """Maximum waiting time in the queue — detects starvation risk."""
        if self.is_empty():
            return 0
        return max(proc.waiting_time for proc in self._queue)

    def __repr__(self):
        pids = [proc.pid for proc in self._queue]
        return f"ReadyQueue(size={self.size()}, pids={pids})"
