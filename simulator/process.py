"""
process.py - Defines the Process entity for the CPU scheduling simulator.

Each process represents a task submitted to the OS scheduler with
attributes mirroring real OS process control block (PCB) fields.
"""


class Process:
    """
    Represents a single OS process with scheduling-relevant attributes.

    Attributes:
        pid            : Unique process identifier
        arrival_time   : Simulation tick at which the process arrives
        burst_time     : Total CPU time required to complete the process
        priority       : Scheduling priority (lower number = higher priority)
        remaining_time : CPU time still needed (decremented each tick)
        waiting_time   : Accumulated time spent waiting in the ready queue
        turnaround_time: Total time from arrival to completion
        start_time     : First tick the process received CPU
        finish_time    : Tick at which the process completed execution
        process_type   : Descriptive label ('cpu_bound', 'interactive', 'mixed')
    """

    def __init__(
        self,
        pid: int,
        arrival_time: int,
        burst_time: int,
        priority: int,
        process_type: str = "mixed",
    ):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.priority = priority
        self.process_type = process_type

        # Arrival gap (set by WorkloadGenerator for LSTM feature encoding)
        self._arrival_gap: float = 3.0  # default inter-arrival gap in ticks

        # Runtime tracking fields (set during simulation)
        self.remaining_time = burst_time
        self.waiting_time = 0
        self.turnaround_time = 0
        self.start_time = -1
        self.finish_time = -1
        self.response_time = -1

    def is_complete(self) -> bool:
        """Return True if the process has finished all CPU execution."""
        return self.remaining_time <= 0

    def reset(self):
        """Reset runtime metrics so the same process can be re-simulated."""
        self.remaining_time = self.burst_time
        self.waiting_time = 0
        self.turnaround_time = 0
        self.start_time = -1
        self.finish_time = -1
        self.response_time = -1

    def __repr__(self):
        return (
            f"Process(pid={self.pid}, arrival={self.arrival_time}, "
            f"burst={self.burst_time}, priority={self.priority}, "
            f"type={self.process_type}, remaining={self.remaining_time})"
        )
