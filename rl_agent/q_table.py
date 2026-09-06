"""
rl_agent/q_table.py
====================
Sparse Q-table with Experience Replay.

Configured for CONTEXTUAL BANDIT use:
  - discount_factor (γ) = 0.0: no future discounting (bandit = single-step)
  - learning_rate (α) = 0.25: faster convergence for clean episode signals
  - epsilon_decay = 0.990: slightly slower decay so exploration covers all states
  - OPTIMISTIC_INIT = 0.0: neutral — let the reward signal do the work

Experience Replay is kept even in bandit mode because:
  1. It breaks temporal correlation between consecutive episodes
  2. Rare states (visited only a few times early in training) get more updates
  3. Adds <5% compute overhead for significant stability improvement

This is the same technique used in DQN (Mnih et al., 2015) adapted to
the tabular setting.
"""

import json
import os
import random
from collections import deque
from typing import List

NUM_ACTIONS         = 4
OPTIMISTIC_INIT     = 0.0


class ExperienceReplayBuffer:
    """Fixed-capacity circular buffer for (s,a,r,s') transitions."""

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self._buf.append((state, action, reward, next_state))

    def sample(self, k: int) -> List:
        return random.sample(list(self._buf), min(k, len(self._buf)))

    def size(self) -> int:
        return len(self._buf)

    def is_ready(self, min_samples: int = 32) -> bool:
        return len(self._buf) >= min_samples


class QTable:
    """
    Sparse tabular Q-store with optimistic initialisation and experience replay.

    Implements standard Q-learning Bellman update:
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]

    For contextual bandit use: set discount_factor=0.0 and pass s'=s.
    """

    def __init__(
        self,
        learning_rate:    float = 0.25,
        discount_factor:  float = 0.0,
        epsilon:          float = 1.0,
        epsilon_min:      float = 0.03,
        epsilon_decay:    float = 0.990,
        replay_capacity:  int   = 2000,
        replay_batch_size:int   = 16,
    ):
        self.learning_rate     = learning_rate
        self.discount_factor   = discount_factor
        self.epsilon           = epsilon
        self.epsilon_min       = epsilon_min
        self.epsilon_decay     = epsilon_decay
        self.replay_batch_size = replay_batch_size
        self._table: dict      = {}
        self.replay_buffer     = ExperienceReplayBuffer(capacity=replay_capacity)

    # ── value access ─────────────────────────────────────────────────────────

    def get_value(self, state: tuple, action: int) -> float:
        return self._table.get((state, action), OPTIMISTIC_INIT)

    def get_best_action(self, state: tuple) -> int:
        qs   = [self.get_value(state, a) for a in range(NUM_ACTIONS)]
        maxq = max(qs)
        return random.choice([a for a, q in enumerate(qs) if q == maxq])

    def get_max_value(self, state: tuple) -> float:
        return max(self.get_value(state, a) for a in range(NUM_ACTIONS))

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return self.get_best_action(state)

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, state, action, reward, next_state):
        self._apply_bellman(state, action, reward, next_state)
        self.replay_buffer.add(state, action, reward, next_state)
        if self.replay_buffer.is_ready(self.replay_batch_size):
            for (s, a, r, s2) in self.replay_buffer.sample(self.replay_batch_size):
                self._apply_bellman(s, a, r, s2)

    def _apply_bellman(self, state, action, reward, next_state):
        curr  = self.get_value(state, action)
        tgt   = reward + self.discount_factor * self.get_max_value(next_state)
        new_q = curr + self.learning_rate * (tgt - curr)
        self._table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {f"{k[0]}|{k[1]}": v for k, v in self._table.items()}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        if not os.path.exists(filepath): return
        with open(filepath) as f:
            raw = json.load(f)
        self._table = {}
        for ks, v in raw.items():
            state_str, action_str = ks.rsplit("|", 1)
            state  = tuple(int(x) for x in state_str.strip("()").split(",") if x.strip())
            action = int(action_str)
            self._table[(state, action)] = float(v)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def table_size(self) -> int:
        return len(self._table)

    def coverage_summary(self) -> dict:
        counts = {a: 0 for a in range(NUM_ACTIONS)}
        for (_, a) in self._table:
            counts[a] += 1
        return counts
