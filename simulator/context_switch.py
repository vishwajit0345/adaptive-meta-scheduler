"""
simulator/context_switch.py
============================
Context-switch overhead support — pure extension, zero impact on existing code.

HOW IT WORKS
------------
The existing Dispatcher already detects context switches (calls
clock.record_context_switch()) but adds zero time penalty.  This module
provides a ContextSwitchDispatcher that EXTENDS the base Dispatcher with
a configurable cost parameter.

DEFAULT BEHAVIOUR
-----------------
  context_switch_cost = 0  →  identical to original Dispatcher.
                               All existing experiments are unaffected.

  context_switch_cost = N  →  N idle ticks are injected whenever the
                               running process changes. CPU utilisation
                               drops and average wait time rises accordingly.

WHY THIS MATTERS FOR RESEARCH
------------------------------
Context-switch cost is a key parameter in real OS scheduler evaluation.
Linux kernel developers measure context-switch overhead at 1–10 μs.
Adding even a small cost changes the relative ranking of algorithms:

  • RoundRobin preempts frequently → penalised most
  • FCFS never preempts → penalised least
  • MLFQ in its upper queues (short quantum) → penalised moderately
  • SJF runs long jobs to completion → penalised least after FCFS

Showing how RL_Adaptive's ranking holds up under nonzero context-switch
cost is a publishable sensitivity analysis.

USAGE
-----
    # Drop-in replacement for existing Dispatcher
    from simulator.context_switch import ContextSwitchDispatcher

    dispatcher = ContextSwitchDispatcher(clock, ready_queue, context_switch_cost=2)
    # Everything else is identical — same execute_tick() API

    # Or use the helper function in extended_experiment.py
    from evaluation.extended_experiment import run_scheduler_with_cs_cost
    results = run_scheduler_with_cs_cost(FCFSScheduler(), processes, cs_cost=2)
"""

from typing import Optional, List
from simulator.process     import Process
from simulator.ready_queue import ReadyQueue
from simulator.clock       import SimulationClock
from simulator.dispatcher  import Dispatcher   # base class — NOT modified


class ContextSwitchDispatcher(Dispatcher):
    """
    Extends the base Dispatcher with a configurable context-switch penalty.

    When the running process changes, `context_switch_cost` idle ticks are
    injected before the new process starts.  The clock advances, all waiting
    processes accumulate extra waiting time, and CPU utilisation falls.

    Parameters
    ----------
    clock               : SimulationClock — shared with ready queue.
    ready_queue         : ReadyQueue — shared process pool.
    context_switch_cost : Number of idle ticks inserted on each preemption.
                          Default = 0 (identical to base Dispatcher).

    Inherits
    --------
    All methods of Dispatcher.  Only execute_tick() is overridden.
    """

    def __init__(
        self,
        clock:                SimulationClock,
        ready_queue:          ReadyQueue,
        context_switch_cost:  int = 0,
    ):
        super().__init__(clock, ready_queue)
        self.context_switch_cost = context_switch_cost
        self._cs_ticks_injected: int = 0   # diagnostic counter

    # ── overridden method ────────────────────────────────────────────────────

    def execute_tick(
        self,
        selected_process: Optional[Process],
        completed_processes: List[Process],
    ) -> bool:
        """
        Execute one simulation tick, injecting idle ticks on context switches.

        If a context switch is detected AND context_switch_cost > 0,
        this method inserts cost idle ticks BEFORE executing the normal tick.
        Each idle tick:
          • Advances the clock (cpu_was_idle=True)
          • Increments waiting_time for every queued process
          • Does NOT run any process

        Then the normal tick executes exactly as in the base Dispatcher.

        Returns
        -------
        bool — True if selected_process completed during the normal tick.
        """
        # Detect context switch: process changed since last tick
        is_context_switch = (
            selected_process is not None
            and self._previous_process is not None
            and selected_process is not self._previous_process
            and self.context_switch_cost > 0
        )

        if is_context_switch:
            self._inject_idle_ticks(self.context_switch_cost)

        # Delegate to base class for the normal tick
        return super().execute_tick(selected_process, completed_processes)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _inject_idle_ticks(self, n: int):
        """
        Burn n idle ticks to simulate OS overhead during a context switch.
        All processes in the ready queue accumulate waiting time.
        """
        for _ in range(n):
            # Charge waiting time to every queued process
            for proc in self.ready_queue.get_all():
                proc.waiting_time += 1
            # Advance clock (cpu_was_idle=True so utilisation is reduced)
            self.clock.tick(cpu_was_idle=True)
            self._cs_ticks_injected += 1

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def total_cs_ticks_wasted(self) -> int:
        """Total idle ticks injected by context switches this simulation."""
        return self._cs_ticks_injected

    def reset(self):
        """Reset base state plus CS counters."""
        super().reset()
        self._cs_ticks_injected = 0
