"""
tests/test_rl_agent.py
========================
Unit tests for the contextual-bandit machinery: QTable (Bellman update +
experience replay), RewardCalculator, and StateEncoder.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.q_table import QTable, ExperienceReplayBuffer
from rl_agent.reward import RewardCalculator
from rl_agent.state_encoder import StateEncoder
from simulator.process import Process
from simulator.ready_queue import ReadyQueue


# ── QTable ───────────────────────────────────────────────────────────────────

def test_qtable_unseen_state_action_defaults_to_optimistic_init():
    q = QTable()
    assert q.get_value(state=(0, 0, 0, 0, 0), action=1) == 0.0


def test_qtable_bandit_update_moves_value_toward_reward():
    # discount_factor=0 => pure bandit, next_state term is ignored
    q = QTable(learning_rate=0.5, discount_factor=0.0, replay_batch_size=1000)
    state = (1, 1, 1, 1, 1)
    q.update(state, action=2, reward=10.0, next_state=state)
    # Q <- 0 + 0.5*(10 - 0) = 5.0  (replay buffer has only 1 sample < batch size, no extra update)
    assert abs(q.get_value(state, 2) - 5.0) < 1e-9


def test_qtable_repeated_updates_converge_toward_reward():
    q = QTable(learning_rate=0.3, discount_factor=0.0, replay_capacity=10,
               replay_batch_size=10_000)  # batch size unreachable -> no replay noise
    state = (0, 1, 2, 0, 1)
    for _ in range(200):
        q.update(state, action=0, reward=100.0, next_state=state)
    assert abs(q.get_value(state, 0) - 100.0) < 1.0


def test_qtable_get_best_action_picks_max_value():
    q = QTable()
    state = (0, 0, 0, 0, 0)
    q._table[(state, 0)] = 1.0
    q._table[(state, 1)] = 5.0
    q._table[(state, 2)] = 3.0
    q._table[(state, 3)] = -1.0
    assert q.get_best_action(state) == 1


def test_qtable_epsilon_decay_respects_floor():
    q = QTable(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(20):
        q.decay_epsilon()
    assert q.epsilon == 0.1  # floors out, never goes below epsilon_min


def test_qtable_select_action_always_greedy_when_epsilon_zero():
    q = QTable(epsilon=0.0)
    state = (0, 0, 0, 0, 0)
    q._table[(state, 3)] = 999.0
    for _ in range(20):
        assert q.select_action(state) == 3


def test_qtable_save_and_load_roundtrip(tmp_path):
    q = QTable()
    state = (2, 1, 0, 1, 2)
    q._table[(state, 1)] = 42.5
    path = str(tmp_path / "q.json")
    q.save(path)

    q2 = QTable()
    q2.load(path)
    assert abs(q2.get_value(state, 1) - 42.5) < 1e-9


def test_experience_replay_buffer_respects_capacity():
    buf = ExperienceReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(state=(i,), action=0, reward=float(i), next_state=(i,))
    assert buf.size() == 3  # oldest 2 evicted


def test_experience_replay_buffer_is_ready_threshold():
    buf = ExperienceReplayBuffer(capacity=100)
    assert not buf.is_ready(min_samples=5)
    for i in range(5):
        buf.add((i,), 0, 0.0, (i,))
    assert buf.is_ready(min_samples=5)


# ── RewardCalculator ─────────────────────────────────────────────────────────

def test_reward_zero_wait_ratio_gives_max_speed_bonus():
    rc = RewardCalculator()
    p = Process(pid=0, arrival_time=0, burst_time=10, priority=1)
    p.turnaround_time = 10  # ratio = 1.0 -> full 100-point bonus
    r = rc.compute(
        newly_completed_count=1, avg_waiting_time=0, avg_turnaround_time=10,
        max_waiting_time=0, cpu_was_idle=False, starving_process_count=0,
        all_done=False, newly_completed_processes=[p],
    )
    assert abs(r - (rc.COMPLETION_BASE + rc.SPEED_SCALE)) < 1e-9


def test_reward_worse_ratio_gives_smaller_bonus():
    rc = RewardCalculator()
    fast = Process(pid=0, arrival_time=0, burst_time=10, priority=1)
    fast.turnaround_time = 10   # ratio 1.0
    slow = Process(pid=1, arrival_time=0, burst_time=10, priority=1)
    slow.turnaround_time = 50   # ratio 5.0

    r_fast = rc.compute(1, 0, 10, 0, False, 0, False, newly_completed_processes=[fast])
    rc.reset()
    r_slow = rc.compute(1, 0, 50, 0, False, 0, False, newly_completed_processes=[slow])
    assert r_fast > r_slow


def test_reward_penalises_rising_average_wait():
    rc = RewardCalculator()
    rc._prev_wait = 10.0
    r = rc.compute(0, avg_waiting_time=20.0, avg_turnaround_time=0,
                    max_waiting_time=0, cpu_was_idle=False,
                    starving_process_count=0, all_done=False)
    assert r < 0  # wait got worse -> penalty, no completions to offset it


def test_reward_terminal_bonus_applied_once_all_done():
    rc = RewardCalculator()
    r = rc.compute(0, avg_waiting_time=0, avg_turnaround_time=0,
                    max_waiting_time=0, cpu_was_idle=False,
                    starving_process_count=0, all_done=True)
    assert r >= rc.TERMINAL_BONUS


def test_reward_idle_cpu_is_penalised():
    rc = RewardCalculator()
    r_idle = rc.compute(0, 0, 0, 0, cpu_was_idle=True,
                         starving_process_count=0, all_done=False)
    rc.reset()
    r_busy = rc.compute(0, 0, 0, 0, cpu_was_idle=False,
                         starving_process_count=0, all_done=False)
    assert r_idle < r_busy


def test_count_starving_uses_threshold():
    rc = RewardCalculator()
    rq = ReadyQueue()
    p_ok = Process(pid=0, arrival_time=0, burst_time=5, priority=1)
    p_ok.waiting_time = rc.STARVE_THRESHOLD - 1
    p_starving = Process(pid=1, arrival_time=0, burst_time=5, priority=1)
    p_starving.waiting_time = rc.STARVE_THRESHOLD + 5
    rq.add_process(p_ok)
    rq.add_process(p_starving)
    assert rc.count_starving(rq) == 1


# ── StateEncoder ─────────────────────────────────────────────────────────────

def test_state_encoder_returns_five_tuple():
    rq = ReadyQueue()
    rq.add_process(Process(pid=0, arrival_time=0, burst_time=10, priority=1))
    state = StateEncoder().encode(rq, recent_arrival_rate=0.3)
    assert len(state) == 5
    assert all(v in (0, 1, 2) for v in state)


def test_state_encoder_bin_boundaries():
    enc = StateEncoder()
    lo, hi = 10, 20
    assert enc._bin(5,  (lo, hi)) == 0
    assert enc._bin(15, (lo, hi)) == 1
    assert enc._bin(25, (lo, hi)) == 2
    assert enc._bin(10, (lo, hi)) == 1  # boundary is inclusive on the high side of bin 0


def test_state_encoder_empty_queue_does_not_crash():
    rq = ReadyQueue()
    state = StateEncoder().encode(rq, recent_arrival_rate=0.3)
    assert len(state) == 5


def test_state_space_size_matches_docstring_claim():
    assert StateEncoder().state_space_size() == 3 ** 5 == 243


def test_state_encoder_uses_predictor_when_provided():
    class FixedPredictor:
        def predict(self, pid, fallback=15.0):
            return 1.0  # every process "predicted" identically short

    rq = ReadyQueue()
    rq.add_process(Process(pid=0, arrival_time=0, burst_time=5,  priority=1))
    rq.add_process(Process(pid=1, arrival_time=0, burst_time=50, priority=1))

    enc = StateEncoder()
    state_no_pred   = enc.encode(rq, 0.3, predictor=None)
    state_with_pred = enc.encode(rq, 0.3, predictor=FixedPredictor())
    # True bursts (5 vs 50): ratio = 5/27.5 = 0.182 -> middle bin (1).
    assert state_no_pred[3] == 1
    # Predictor claims both processes are equally short -> ratio 1.0 -> bin 2.
    assert state_with_pred[3] == 2
    # The two encodings disagree on F3 specifically because one is driven
    # by true remaining_time and the other by (fake) LSTM predictions.
    assert state_no_pred[3] != state_with_pred[3]
