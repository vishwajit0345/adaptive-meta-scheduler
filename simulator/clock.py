"""
clock.py - Discrete-event simulation clock.

The Clock drives the simulation forward one tick at a time and tracks
key temporal statistics needed for evaluation and RL reward computation.
"""


class SimulationClock:
    """
    Maintains the current simulation time and counts completed processes.

    The clock is the single source of truth for time in the simulator.
    Every component that needs to know 'what time is it?' queries this object.
    """

    def __init__(self):
        self._current_time: int = 0
        self._total_ticks: int = 0
        self._idle_ticks: int = 0          # Ticks where CPU had nothing to run
        self._context_switches: int = 0    # Number of scheduler preemptions
        self._completed_processes: int = 0

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def tick(self, cpu_was_idle: bool = False):
        """Advance the simulation by one time unit."""
        self._current_time += 1
        self._total_ticks += 1
        if cpu_was_idle:
            self._idle_ticks += 1

    def record_context_switch(self):
        """Call this each time the scheduler swaps the running process."""
        self._context_switches += 1

    def record_completion(self):
        """Call this each time a process finishes execution."""
        self._completed_processes += 1

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_time(self) -> int:
        return self._current_time

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    @property
    def idle_ticks(self) -> int:
        return self._idle_ticks

    @property
    def context_switches(self) -> int:
        return self._context_switches

    @property
    def completed_processes(self) -> int:
        return self._completed_processes

    @property
    def cpu_utilization(self) -> float:
        """Fraction of ticks where the CPU was doing useful work."""
        if self._total_ticks == 0:
            return 0.0
        return (self._total_ticks - self._idle_ticks) / self._total_ticks

    def reset(self):
        """Reset all counters for a fresh simulation run."""
        self._current_time = 0
        self._total_ticks = 0
        self._idle_ticks = 0
        self._context_switches = 0
        self._completed_processes = 0

    def __repr__(self):
        return (
            f"SimulationClock(time={self._current_time}, "
            f"idle={self._idle_ticks}, switches={self._context_switches})"
        )
