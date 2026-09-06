"""
schedulers/lottery.py
=====================
Lottery Scheduling.

Originally proposed by Waldspurger & Weihl (1994), lottery scheduling
gives each process a number of "tickets" proportional to its priority.
Each scheduling decision draws a random winner — processes with more
tickets win more often, but every process has a non-zero chance.

PROPERTIES
----------
  • Probabilistically fair: high-priority processes win more often
  • Starvation-free: even a single-ticket process will eventually run
  • Simple to implement and analyse
  • Naturally handles priority without explicit queue management

TICKET ASSIGNMENT
-----------------
  tickets(proc) = MAX_PRIORITY + 1 - proc.priority

  With priority 1 (highest urgency) → 10 tickets
  With priority 10 (lowest urgency) → 1 ticket

WHY USEFUL AS A BASELINE
--------------------------
Lottery scheduling represents the probabilistic scheduling family.
Comparing RL_Adaptive vs Lottery shows whether deterministic learned
policies outperform randomised proportional-share approaches.

INTERFACE
---------
Identical to all other schedulers — implements select_next(ready_queue).
"""

import random
from typing import Optional, List
from simulator.ready_queue import ReadyQueue
from simulator.process import Process


MAX_PRIORITY = 10   # Matches WorkloadGenerator priority_range upper bound


class LotteryScheduler:
    """
    Lottery scheduler — probabilistically fair, priority-proportional.

    Each process receives tickets = MAX_PRIORITY + 1 - priority.
    A random draw selects the winner each tick.
    """

    name = "Lottery"

    def __init__(self, max_priority: int = MAX_PRIORITY):
        self.max_priority = max_priority

    def _tickets(self, process: Process) -> int:
        """Higher urgency (lower priority number) → more tickets."""
        return max(1, self.max_priority + 1 - process.priority)

    def select_next(self, ready_queue: ReadyQueue) -> Optional[Process]:
        """
        Draw a lottery among all ready processes.

        Each process's probability of winning equals its ticket share.
        Returns None if the queue is empty.
        """
        if ready_queue.is_empty():
            return None

        available  = ready_queue.get_all()
        ticket_map = [(proc, self._tickets(proc)) for proc in available]
        total      = sum(t for _, t in ticket_map)

        if total <= 0:
            return random.choice(available)

        draw = random.randint(1, total)
        cumulative = 0
        for proc, tickets in ticket_map:
            cumulative += tickets
            if draw <= cumulative:
                return proc

        # Fallback (should never reach here)
        return available[-1]

    def reset(self):
        """No persistent state to reset for lottery scheduling."""
        pass
