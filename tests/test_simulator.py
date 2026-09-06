"""
tests/test_simulator.py
========================
Unit tests for the discrete-event simulation core: Process, SimulationClock,
ReadyQueue, and Dispatcher.

These are the load-bearing primitives every scheduler and every evaluation
number in the project depends on, so they are tested independently of any
scheduling policy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.process import Process
from simulator.clock import SimulationClock
from simulator.ready_queue import ReadyQueue
from simulator.dispatcher import Dispatcher


# ── Process ──────────────────────────────────────────────────────────────────

def test_process_starts_incomplete():
    p = Process(pid=1, arrival_time=0, burst_time=5, priority=1)
    assert not p.is_complete()
    assert p.remaining_time == 5


def test_process_completes_when_remaining_hits_zero():
    p = Process(pid=1, arrival_time=0, burst_time=3, priority=1)
    p.remaining_time = 0
    assert p.is_complete()


def test_process_reset_restores_runtime_fields():
    p = Process(pid=1, arrival_time=0, burst_time=4, priority=1)
    p.remaining_time = 0
    p.waiting_time = 10
    p.turnaround_time = 14
    p.start_time = 2
    p.finish_time = 14
    p.reset()
    assert p.remaining_time == p.burst_time == 4
    assert p.waiting_time == 0
    assert p.turnaround_time == 0
    assert p.start_time == -1
    assert p.finish_time == -1


# ── SimulationClock ──────────────────────────────────────────────────────────

def test_clock_tick_advances_time():
    c = SimulationClock()
    c.tick()
    c.tick()
    assert c.current_time == 2
    assert c.total_ticks == 2


def test_clock_idle_ticks_tracked_separately():
    c = SimulationClock()
    c.tick(cpu_was_idle=True)
    c.tick(cpu_was_idle=False)
    assert c.idle_ticks == 1
    assert c.total_ticks == 2


def test_clock_cpu_utilization():
    c = SimulationClock()
    for _ in range(3):
        c.tick(cpu_was_idle=False)
    for _ in range(1):
        c.tick(cpu_was_idle=True)
    # 3 busy / 4 total
    assert abs(c.cpu_utilization - 0.75) < 1e-9


def test_clock_utilization_with_no_ticks_is_zero():
    c = SimulationClock()
    assert c.cpu_utilization == 0.0


def test_clock_reset_clears_all_counters():
    c = SimulationClock()
    c.tick(cpu_was_idle=True)
    c.record_context_switch()
    c.record_completion()
    c.reset()
    assert c.current_time == 0
    assert c.idle_ticks == 0
    assert c.context_switches == 0
    assert c.completed_processes == 0


# ── ReadyQueue ────────────────────────────────────────────────────────────────

def _procs():
    return [
        Process(pid=0, arrival_time=0, burst_time=10, priority=3),
        Process(pid=1, arrival_time=1, burst_time=2,  priority=1),
        Process(pid=2, arrival_time=2, burst_time=6,  priority=5),
    ]


def test_ready_queue_add_and_size():
    rq = ReadyQueue()
    for p in _procs():
        rq.add_process(p)
    assert rq.size() == 3
    assert not rq.is_empty()


def test_ready_queue_remove():
    rq = ReadyQueue()
    procs = _procs()
    for p in procs:
        rq.add_process(p)
    rq.remove_process(procs[1])
    assert rq.size() == 2
    assert procs[1] not in rq.get_all()


def test_ready_queue_peek_shortest():
    rq = ReadyQueue()
    for p in _procs():
        rq.add_process(p)
    shortest = rq.peek_shortest()
    assert shortest.pid == 1  # burst_time=2 is smallest


def test_ready_queue_peek_highest_priority():
    rq = ReadyQueue()
    for p in _procs():
        rq.add_process(p)
    best = rq.peek_highest_priority()
    assert best.pid == 1  # priority=1 is the most urgent (lowest number)


def test_ready_queue_peek_fcfs():
    rq = ReadyQueue()
    for p in _procs():
        rq.add_process(p)
    first = rq.peek_fcfs()
    assert first.pid == 0  # arrival_time=0 is earliest


def test_ready_queue_average_and_variance_burst_time():
    rq = ReadyQueue()
    for p in _procs():
        rq.add_process(p)
    # bursts: 10, 2, 6 -> mean = 6
    assert abs(rq.average_burst_time() - 6.0) < 1e-9
    # variance = mean((x-mean)^2) = ((16)+(16)+(0))/3 = 32/3
    assert abs(rq.variance_burst_time() - (32 / 3)) < 1e-6


def test_ready_queue_empty_helpers_do_not_crash():
    rq = ReadyQueue()
    assert rq.is_empty()
    assert rq.peek_shortest() is None
    assert rq.peek_highest_priority() is None
    assert rq.peek_fcfs() is None
    assert rq.average_burst_time() == 0.0
    assert rq.variance_burst_time() == 0.0
    assert rq.average_waiting_time() == 0.0
    assert rq.max_waiting_time() == 0


# ── Dispatcher ───────────────────────────────────────────────────────────────

def test_dispatcher_runs_process_for_one_tick():
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    p = Process(pid=0, arrival_time=0, burst_time=3, priority=1)
    rq.add_process(p)

    completed = []
    dispatcher.execute_tick(p, completed)

    assert p.remaining_time == 2
    assert p.start_time == 0
    assert clock.current_time == 1
    assert completed == []


def test_dispatcher_marks_completion_and_computes_turnaround():
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    p = Process(pid=0, arrival_time=0, burst_time=2, priority=1)
    rq.add_process(p)

    completed = []
    dispatcher.execute_tick(p, completed)   # tick 1: remaining 2->1
    dispatcher.execute_tick(p, completed)   # tick 2: remaining 1->0, completes

    assert p.is_complete()
    assert p in completed
    assert p.finish_time == 2
    assert p.turnaround_time == 2 - 0  # finish - arrival
    assert p not in rq.get_all()


def test_dispatcher_idle_tick_advances_clock_without_running_anything():
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    completed = []
    result = dispatcher.execute_tick(None, completed)
    assert result is False
    assert clock.current_time == 1
    assert clock.idle_ticks == 1


def test_dispatcher_increments_waiting_time_for_other_queued_processes():
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    running  = Process(pid=0, arrival_time=0, burst_time=5, priority=1)
    waiting  = Process(pid=1, arrival_time=0, burst_time=5, priority=1)
    rq.add_process(running)
    rq.add_process(waiting)

    completed = []
    dispatcher.execute_tick(running, completed)

    assert running.waiting_time == 0
    assert waiting.waiting_time == 1


def test_dispatcher_records_context_switch_on_process_change():
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    a = Process(pid=0, arrival_time=0, burst_time=5, priority=1)
    b = Process(pid=1, arrival_time=0, burst_time=5, priority=1)
    rq.add_process(a)
    rq.add_process(b)

    completed = []
    dispatcher.execute_tick(a, completed)   # first process: no switch yet
    assert clock.context_switches == 0
    dispatcher.execute_tick(b, completed)   # switched from a -> b
    assert clock.context_switches == 1
    dispatcher.execute_tick(b, completed)   # stayed on b: no new switch
    assert clock.context_switches == 1


def test_dispatcher_admit_arrivals_only_at_matching_tick():
    clock = SimulationClock()
    rq = ReadyQueue()
    dispatcher = Dispatcher(clock, rq)
    p0 = Process(pid=0, arrival_time=0, burst_time=1, priority=1)
    p1 = Process(pid=1, arrival_time=1, burst_time=1, priority=1)
    all_procs = [p0, p1]

    dispatcher.admit_arrivals(all_procs)
    assert rq.get_all() == [p0]

    completed = []
    dispatcher.execute_tick(p0, completed)  # advances clock to 1
    dispatcher.admit_arrivals(all_procs)
    assert p1 in rq.get_all()
