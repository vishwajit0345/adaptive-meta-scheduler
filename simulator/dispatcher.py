"""
dispatcher.py - Executes one tick of CPU work for the selected process.

The Dispatcher is the low-level CPU executor. It receives the process
chosen by the active scheduler, runs it for one time unit, updates
process metrics, and reports whether the process completed.
"""

from typing import Optional, List
from simulator.process import Process
from simulator.ready_queue import ReadyQueue
from simulator.clock import SimulationClock


class Dispatcher:
    """
    Executes CPU work, updates process accounting, and manages arrivals.

    Responsibilities:
      - Run the selected process for one tick
      - Increment waiting times for all other queued processes
      - Admit newly-arrived processes into the ready queue
      - Detect and record process completion
    """

    def __init__(self, clock: SimulationClock, ready_queue: ReadyQueue):
        self.clock = clock
        self.ready_queue = ready_queue
        self._previous_process: Optional[Process] = None

    def admit_arrivals(self, all_processes: List[Process]):
        """
        Check the full process list and admit any process whose arrival_time
        matches the current clock tick into the ready queue.
        """
        for process in all_processes:
            if (
                process.arrival_time == self.clock.current_time
                and not process.is_complete()
                and process not in self.ready_queue.get_all()
            ):
                self.ready_queue.add_process(process)

    def execute_tick(
        self,
        selected_process: Optional[Process],
        completed_processes: List[Process],
    ) -> bool:
        """
        Execute one simulation tick.

        Args:
            selected_process  : The process chosen by the active scheduler,
                                or None if the CPU is idle.
            completed_processes: Accumulator list — finished processes are
                                appended here.

        Returns:
            True if selected_process completed during this tick.
        """
        process_completed = False

        # Detect a context switch (process changed since last tick)
        if (
            selected_process is not None
            and self._previous_process is not None
            and selected_process is not self._previous_process
        ):
            self.clock.record_context_switch()

        if selected_process is None:
            # CPU is idle — no runnable process available
            self.clock.tick(cpu_was_idle=True)
            self._previous_process = None
            return False

        # Record the first time this process ever touched the CPU
        if selected_process.start_time == -1:
            selected_process.start_time = self.clock.current_time
            selected_process.response_time = (
                self.clock.current_time - selected_process.arrival_time
            )

        # Run selected process for one tick
        selected_process.remaining_time -= 1

        # Increment waiting time for every OTHER process in the queue
        for queued_process in self.ready_queue.get_all():
            if queued_process is not selected_process:
                queued_process.waiting_time += 1

        self.clock.tick(cpu_was_idle=False)

        # Check if the process just finished
        if selected_process.is_complete():
            selected_process.finish_time = self.clock.current_time
            selected_process.turnaround_time = (
                selected_process.finish_time - selected_process.arrival_time
            )
            self.ready_queue.remove_process(selected_process)
            completed_processes.append(selected_process)
            self.clock.record_completion()
            process_completed = True

        self._previous_process = selected_process
        return process_completed

    def reset(self):
        """Reset dispatcher state for a fresh run."""
        self._previous_process = None
