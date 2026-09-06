"""
tests/test_schedulers.py
=========================
Correctness tests for each scheduling policy's select_next() decision.

Each test builds a small, hand-checkable ReadyQueue and asserts the
scheduler picks the process a textbook definition of the algorithm says
it should. These are pure selection-policy tests (no dispatcher/clock
involved) so a bug in one scheduler can't hide behind a bug in another.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.process import Process
from simulator.ready_queue import ReadyQueue

from schedulers.fcfs import FCFSScheduler
from schedulers.sjf import SJFScheduler, OracleSJFScheduler
from schedulers.sjf_predicted import SJFPredictedScheduler
from schedulers.priority import PriorityScheduler
from schedulers.round_robin import RoundRobinScheduler
from schedulers.mlfq import MLFQScheduler
from schedulers.edf import EDFScheduler
from schedulers.lottery import LotteryScheduler


def _mkqueue(specs):
    """specs: list of (pid, arrival, burst, priority, type) -> ReadyQueue"""
    rq = ReadyQueue()
    procs = []
    for pid, arrival, burst, priority, ptype in specs:
        p = Process(pid=pid, arrival_time=arrival, burst_time=burst,
                    priority=priority, process_type=ptype)
        procs.append(p)
        rq.add_process(p)
    return rq, procs


class _StubPredictor:
    """Deterministic predictor stub: always returns a fixed value."""
    def __init__(self, value=15.0):
        self.value = value

    def predict(self, pid, fallback=15.0):
        return self.value

    def predict_process(self, process):
        return self.value


# ── FCFS ─────────────────────────────────────────────────────────────────────

def test_fcfs_picks_earliest_arrival():
    rq, procs = _mkqueue([
        (0, 5, 10, 1, "mixed"),
        (1, 1, 3,  1, "mixed"),
        (2, 8, 4,  1, "mixed"),
    ])
    chosen = FCFSScheduler().select_next(rq)
    assert chosen.pid == 1


def test_fcfs_empty_queue_returns_none():
    rq = ReadyQueue()
    assert FCFSScheduler().select_next(rq) is None


# ── SJF (oracle) ─────────────────────────────────────────────────────────────

def test_oracle_sjf_picks_smallest_remaining_time():
    rq, procs = _mkqueue([
        (0, 0, 10, 1, "mixed"),
        (1, 0, 2,  1, "mixed"),
        (2, 0, 6,  1, "mixed"),
    ])
    chosen = OracleSJFScheduler().select_next(rq)
    assert chosen.pid == 1


def test_oracle_sjf_uses_remaining_not_original_burst():
    rq, procs = _mkqueue([
        (0, 0, 10, 1, "mixed"),
        (1, 0, 2,  1, "mixed"),
    ])
    # Process 0 has less remaining work now, despite a bigger original burst
    procs[0].remaining_time = 1
    chosen = OracleSJFScheduler().select_next(rq)
    assert chosen.pid == 0


def test_sjf_scheduler_without_predictor_behaves_like_oracle():
    rq, procs = _mkqueue([
        (0, 0, 9, 1, "mixed"),
        (1, 0, 3, 1, "mixed"),
    ])
    chosen = SJFScheduler(predictor=None).select_next(rq)
    assert chosen.pid == 1


# ── SJF_Predicted (two-tier: type prior + LSTM refinement) ────────────────────

def test_sjf_predicted_prefers_interactive_type_over_cpu_bound():
    # Interactive prior (4.5) << cpu_bound prior (27.5), so even with an
    # identical (stub) LSTM correction, interactive must win.
    rq, procs = _mkqueue([
        (0, 0, 30, 1, "cpu_bound"),
        (1, 0, 30, 1, "interactive"),
    ])
    sched = SJFPredictedScheduler(_StubPredictor(value=15.0))
    chosen = sched.select_next(rq)
    assert chosen.pid == 1


def test_sjf_predicted_has_no_access_to_true_remaining_time():
    """
    Regression guard: SJF_Predicted must not silently fall back to oracle
    behaviour. Two same-type processes with a stub predictor returning an
    identical value for both must be indistinguishable by _estimate(),
    i.e. the true (very different) remaining_time must NOT be the
    deciding factor.
    """
    rq, procs = _mkqueue([
        (0, 0, 5,  1, "mixed"),
        (1, 0, 40, 1, "mixed"),
    ])
    sched = SJFPredictedScheduler(_StubPredictor(value=15.0))
    est0 = sched._estimate(procs[0], rq.average_burst_time())
    est1 = sched._estimate(procs[1], rq.average_burst_time())
    assert est0 == est1  # both "mixed" type + identical stub prediction


# ── Priority (with ageing) ──────────────────────────────────────────────────

def test_priority_picks_lowest_priority_number():
    rq, procs = _mkqueue([
        (0, 0, 5, 3, "mixed"),
        (1, 0, 5, 1, "mixed"),
        (2, 0, 5, 7, "mixed"),
    ])
    chosen = PriorityScheduler().select_next(rq)
    assert chosen.pid == 1


def test_priority_ageing_boosts_long_waiting_process():
    sched = PriorityScheduler()
    rq, procs = _mkqueue([
        (0, 0, 5, 5, "mixed"),   # will age
        (1, 0, 5, 2, "mixed"),   # starts higher priority (lower number)
    ])
    # Simulate process 0 having waited exactly one ageing interval (10 ticks)
    procs[0].waiting_time = sched.ageing_interval
    sched.select_next(rq)  # first call establishes effective priorities + applies ageing
    assert sched._effective_priority[procs[0].pid] == 4  # boosted from 5 -> 4
    assert sched._effective_priority[procs[1].pid] == 2  # unchanged


# ── RoundRobin ───────────────────────────────────────────────────────────────

def test_round_robin_rotates_after_quantum_expires():
    sched = RoundRobinScheduler(time_quantum=2)
    rq, procs = _mkqueue([
        (0, 0, 10, 1, "mixed"),
        (1, 0, 10, 1, "mixed"),
    ])
    picks = [sched.select_next(rq).pid for _ in range(5)]
    # quantum=2: pid0, pid0, then rotate to pid1, pid1, then back to pid0
    assert picks == [0, 0, 1, 1, 0]


def test_round_robin_skips_completed_process():
    sched = RoundRobinScheduler(time_quantum=5)
    rq, procs = _mkqueue([
        (0, 0, 10, 1, "mixed"),
        (1, 0, 10, 1, "mixed"),
    ])
    sched.select_next(rq)          # picks pid 0
    rq.remove_process(procs[0])    # pid 0 "completes"
    chosen = sched.select_next(rq)
    assert chosen.pid == 1


# ── MLFQ ─────────────────────────────────────────────────────────────────────

def test_mlfq_new_arrivals_start_in_top_queue():
    sched = MLFQScheduler(quantums=[4, 8, None])
    rq, procs = _mkqueue([(0, 0, 10, 1, "mixed")])
    sched.select_next(rq)
    assert sched._proc_level[procs[0].pid] == 0


def test_mlfq_demotes_after_quantum_exhausted():
    sched = MLFQScheduler(quantums=[2, 8, None], boost_interval=10_000)
    rq, procs = _mkqueue([(0, 0, 10, 1, "mixed")])
    # Consume the level-0 quantum (2 ticks)
    sched.select_next(rq)
    sched.select_next(rq)
    # Third call should detect quantum exhaustion and demote to level 1
    sched.select_next(rq)
    assert sched._proc_level[procs[0].pid] == 1


def test_mlfq_priority_boost_readmits_in_ready_queue_order():
    """
    Regression test: priority-boost re-admission must follow ReadyQueue's
    own (arrival) order, not Python's identity-hash-based set order, or
    FCFS-within-Q0 becomes irreproducible across separate runs even with
    a fixed random seed.
    """
    sched = MLFQScheduler(quantums=[4, 8, None], boost_interval=1)
    # Many processes so a hash-order-dependent bug would very likely show up
    # as a mismatch against insertion order.
    specs = [(pid, 0, 5, 1, "mixed") for pid in range(15)]
    rq, procs = _mkqueue(specs)

    sched.select_next(rq)  # global_tick=1 -> boost_interval=1 triggers a boost immediately

    assert sched._queues[0] == list(rq.get_all())


def test_mlfq_short_job_is_not_demoted_before_completing_quantum():
    sched = MLFQScheduler(quantums=[4, 8, None], boost_interval=10_000)
    rq, procs = _mkqueue([(0, 0, 10, 1, "mixed")])
    sched.select_next(rq)   # 1 tick used, still well under quantum=4
    assert sched._proc_level[procs[0].pid] == 0


# ── EDF ──────────────────────────────────────────────────────────────────────

def test_edf_picks_earliest_deadline():
    sched = EDFScheduler(deadline_factor=2.0)
    rq, procs = _mkqueue([
        (0, 0, 20, 1, "mixed"),   # deadline = 0 + 20*2 = 40
        (1, 0, 5,  1, "mixed"),   # deadline = 0 + 5*2  = 10
    ])
    chosen = sched.select_next(rq)
    assert chosen.pid == 1


def test_edf_deadline_is_cached_at_first_encounter():
    sched = EDFScheduler(deadline_factor=2.0)
    rq, procs = _mkqueue([(0, 0, 10, 1, "mixed")])
    sched.select_next(rq)
    first_deadline = sched._deadlines[procs[0].pid]
    procs[0].burst_time = 999  # mutate after the fact — must NOT change deadline
    assert sched._get_deadline(procs[0]) == first_deadline


# ── Lottery ──────────────────────────────────────────────────────────────────

def test_lottery_only_selects_from_ready_queue():
    sched = LotteryScheduler()
    rq, procs = _mkqueue([
        (0, 0, 5, 1, "mixed"),
        (1, 0, 5, 9, "mixed"),
    ])
    pids = {p.pid for p in procs}
    for _ in range(50):
        chosen = sched.select_next(rq)
        assert chosen.pid in pids


def test_lottery_tickets_favor_high_priority():
    sched = LotteryScheduler(max_priority=10)
    high = Process(pid=0, arrival_time=0, burst_time=5, priority=1)   # 10 tickets
    low  = Process(pid=1, arrival_time=0, burst_time=5, priority=10)  # 1 ticket
    assert sched._tickets(high) > sched._tickets(low)


def test_lottery_empty_queue_returns_none():
    rq = ReadyQueue()
    assert LotteryScheduler().select_next(rq) is None
