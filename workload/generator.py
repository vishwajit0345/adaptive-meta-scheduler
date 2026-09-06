"""
workload/generator.py - Generates heterogeneous CPU process workloads.

Three workload profiles are supported:

  cpu_bound   : Dominated by long-running compute-intensive jobs.
                High burst times, low arrival density, low priority variance.

  interactive : Short, frequent bursts simulating user-facing tasks.
                Low burst times, dense arrivals, mixed priorities.

  mixed       : Adversarial blend of both — designed to stress-test
                schedulers by interleaving long and short jobs unpredictably.

All parameters are randomised per call to ensure each run produces
unique, non-trivial workloads.
"""

import random
from typing import List
from simulator.process import Process


class WorkloadGenerator:
    """
    Produces lists of Process objects representing OS scheduling workloads.

    Randomisation is seeded by the caller or left to system entropy so
    that every program invocation yields different results.
    """

    # ---- Workload configuration ranges -----------------------------------
    PROFILES = {
        "cpu_bound": {
            "num_processes_range": (12, 20),
            "burst_time_range": (15, 40),
            "inter_arrival_range": (2, 6),
            "priority_range": (1, 5),
            "description": "CPU-intensive long jobs",
        },
        "interactive": {
            "num_processes_range": (18, 30),
            "burst_time_range": (1, 8),
            "inter_arrival_range": (1, 3),
            "priority_range": (1, 10),
            "description": "Short interactive user tasks",
        },
        "mixed": {
            "num_processes_range": (20, 35),
            "burst_time_range": (1, 35),
            "inter_arrival_range": (1, 5),
            "priority_range": (1, 10),
            "description": "Adversarial heterogeneous mix",
        },
    }

    def generate(
        self,
        workload_type: str = "mixed",
        seed: int = None,
    ) -> List[Process]:
        """
        Generate a list of Process objects for a given workload profile.

        Args:
            workload_type: One of 'cpu_bound', 'interactive', 'mixed'.
            seed         : Optional RNG seed for reproducibility.

        Returns:
            Sorted list of Process objects ordered by arrival_time.
        """
        if seed is not None:
            random.seed(seed)

        profile = self.PROFILES.get(workload_type, self.PROFILES["mixed"])

        num_processes = random.randint(*profile["num_processes_range"])
        burst_range = profile["burst_time_range"]
        inter_arrival_range = profile["inter_arrival_range"]
        priority_range = profile["priority_range"]

        processes: List[Process] = []
        current_arrival_time = 0

        for pid in range(num_processes):
            burst_time = random.randint(*burst_range)
            priority = random.randint(*priority_range)

            # Classify the individual process type based on burst duration
            if burst_time <= 5:
                process_type = "interactive"
            elif burst_time >= 20:
                process_type = "cpu_bound"
            else:
                process_type = "mixed"

            process = Process(
                pid=pid,
                arrival_time=current_arrival_time,
                burst_time=burst_time,
                priority=priority,
                process_type=process_type,
            )
            processes.append(process)

            # Stagger arrivals using inter-arrival gaps
            gap = random.randint(*inter_arrival_range)
            process._arrival_gap = float(gap)  # stored for LSTM feature encoding
            current_arrival_time += gap

        # Sort by arrival time (should already be sorted, but be explicit)
        processes.sort(key=lambda proc: proc.arrival_time)

        return processes

    def generate_batch(
        self,
        workload_types: List[str] = None,
        count_per_type: int = 3,
    ) -> List[List[Process]]:
        """
        Generate multiple workloads of varying types for batch evaluation.

        Args:
            workload_types : List of type strings to cycle through.
            count_per_type : How many workloads per type.

        Returns:
            List of process lists.
        """
        if workload_types is None:
            workload_types = ["cpu_bound", "interactive", "mixed"]

        batch = []
        for workload_type in workload_types:
            for _ in range(count_per_type):
                batch.append(self.generate(workload_type=workload_type))
        return batch

    def describe_workload(self, processes: List[Process]) -> dict:
        """
        Return summary statistics for a generated workload.

        Useful for debugging and analysis output.
        """
        if not processes:
            return {}

        burst_times = [proc.burst_time for proc in processes]
        return {
            "count": len(processes),
            "total_burst": sum(burst_times),
            "avg_burst": round(sum(burst_times) / len(burst_times), 2),
            "min_burst": min(burst_times),
            "max_burst": max(burst_times),
            "arrival_span": processes[-1].arrival_time - processes[0].arrival_time,
            "types": {
                ptype: sum(1 for p in processes if p.process_type == ptype)
                for ptype in ["cpu_bound", "interactive", "mixed"]
            },
        }
