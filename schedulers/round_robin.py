"""
round_robin.py - Round Robin (RR) scheduling algorithm.

Round Robin gives each process a fixed time quantum. Once a process
exhausts its quantum without completing, it is moved to the back of
the queue, ensuring fairness and bounded response time.
"""

from typing import Optional, List
from simulator.ready_queue import ReadyQueue
from simulator.process import Process


class RoundRobinScheduler:
    """
    Round Robin scheduler with a configurable time quantum.

    Selection policy: Rotate through processes in FIFO arrival order,
                      each getting at most `time_quantum` consecutive ticks.
    Preemption     : Yes — after time_quantum ticks the process is rotated.

    Attributes:
        time_quantum   : Maximum consecutive ticks per process per turn.
        _current_process: The process currently holding the CPU.
        _ticks_used    : How many ticks the current process has consumed
                         in its current quantum.
    """

    name = "RoundRobin"

    def __init__(self, time_quantum: int = 3):
        self.time_quantum = time_quantum
        self._current_process: Optional[Process] = None
        self._ticks_used: int = 0
        # Ordered rotation list — processes join at the back
        self._rotation_queue: List[Process] = []

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        """
        Select the next process according to round-robin rotation.

        Args:
            ready_queue: The current ready queue snapshot.

        Returns:
            The process that should run this tick, or None if queue is empty.
        """
        if ready_queue.is_empty():
            self._current_process = None
            self._ticks_used = 0
            return None

        available = ready_queue.get_all()

        # Add newly arrived processes to the rotation list (avoid duplicates)
        for process in available:
            if process not in self._rotation_queue:
                self._rotation_queue.append(process)

        # Prune processes that are no longer in the ready queue (completed)
        self._rotation_queue = [
            process for process in self._rotation_queue if process in available
        ]

        if not self._rotation_queue:
            return None

        # If current process has used up its quantum, rotate to next
        if (
            self._current_process is None
            or self._current_process not in self._rotation_queue
            or self._ticks_used >= self.time_quantum
        ):
            # Move the current process to the back of the rotation queue
            if (
                self._current_process is not None
                and self._current_process in self._rotation_queue
            ):
                self._rotation_queue.remove(self._current_process)
                self._rotation_queue.append(self._current_process)

            self._current_process = self._rotation_queue[0]
            self._ticks_used = 0

        self._ticks_used += 1
        return self._current_process

    def reset(self):
        """Reset rotation state for a fresh simulation run."""
        self._current_process = None
        self._ticks_used = 0
        self._rotation_queue = []
