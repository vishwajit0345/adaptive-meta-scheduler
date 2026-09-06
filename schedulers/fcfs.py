"""
fcfs.py - First Come First Served (FCFS) scheduling algorithm.

FCFS is the simplest non-preemptive scheduler. Processes are served
in the exact order they arrive. It is fair but can cause the convoy
effect where short jobs wait behind long ones.
"""

from typing import Optional
from simulator.ready_queue import ReadyQueue
from simulator.process import Process


class FCFSScheduler:
    """
    First Come First Served scheduler.

    Selection policy: choose the process with the earliest arrival_time.
    Preemption     : None — once a process starts it runs to completion.
    """

    name = "FCFS"

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        """
        Select the process that arrived earliest.

        Args:
            ready_queue: The current ready queue snapshot.

        Returns:
            The earliest-arriving process, or None if the queue is empty.
        """
        if ready_queue.is_empty():
            return None
        return ready_queue.peek_fcfs()
