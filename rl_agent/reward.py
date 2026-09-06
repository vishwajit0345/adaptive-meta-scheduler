"""
rl_agent/reward.py
==================
Reward signal for the Q-learning meta-scheduler.

DESIGN — ONE DOMINANT SIGNAL + SMALL SECONDARY TERMS
------------------------------------------------------
The uploaded v4 reward had multiple roughly equal-magnitude terms that
cancelled each other, producing a noisy signal the agent couldn't learn from.

Fix: a single dominant term (turnaround-ratio bonus) that strongly separates
good schedulers from bad ones, with small bounded secondary terms.

DOMINANT TERM: TURNAROUND RATIO BONUS
    For each completed process:  reward += 100 / (turnaround / burst)
    
    ratio=1.0 (zero wait)  → bonus = 100   (SJF-like, fast completion)
    ratio=2.0 (wait = burst) → bonus =  50
    ratio=5.0 (wait=4×burst) → bonus =  20   (FCFS/RR with long queue)

    With 20–35 processes, episode bonus difference between best/worst
    scheduler ≈ 30 × (100−20) = 2,400 per episode.
    Noise from secondary terms ≈ ±100 per episode.
    Signal-to-noise ≈ 24:1  →  clearly learnable in <200 episodes.

SECONDARY TERMS (small, bounded)
    Differential wait     : dense tick-level signal, bounded ±5
    Starvation penalty    : teaches Priority in extreme cases, bounded ±15
    Idle penalty / bonus  : keeps CPU busy
    Terminal bonus        : strong episode-end signal
"""


class RewardCalculator:

    SPEED_SCALE        = 100.0   # per-process dominant term
    COMPLETION_BASE    = 5.0     # flat per-completion

    WAIT_WORSEN        = 0.5     # penalty per tick of wait increase
    WAIT_IMPROVE       = 0.25    # reward per tick of wait decrease

    STARVE_THRESHOLD   = 30      # ticks — catch starvation early
    STARVE_BASE        = 3.0
    STARVE_DEPTH       = 0.005   # × depth²

    IDLE_PENALTY       = 0.5
    TERMINAL_BONUS     = 50.0

    def __init__(self):
        self._prev_wait: float = 0.0

    def compute(
        self,
        newly_completed_count: int,
        avg_waiting_time: float,
        avg_turnaround_time: float,
        max_waiting_time: float,
        cpu_was_idle: bool,
        starving_process_count: int,
        all_done: bool,
        newly_completed_processes=None,
    ) -> float:

        r = 0.0

        # ── 1. Dominant: turnaround-ratio bonus ───────────────────────────
        if newly_completed_count > 0:
            if newly_completed_processes:
                for p in newly_completed_processes:
                    if p.burst_time > 0:
                        ratio = p.turnaround_time / p.burst_time
                        r += self.COMPLETION_BASE + self.SPEED_SCALE / max(ratio, 1.0)
            else:
                r += newly_completed_count * self.COMPLETION_BASE

        # ── 2. Differential wait (dense signal) ──────────────────────────
        delta = avg_waiting_time - self._prev_wait
        if delta > 0:
            r -= min(delta * self.WAIT_WORSEN, 5.0)
        elif delta < 0:
            r += min(abs(delta) * self.WAIT_IMPROVE, 2.5)
        self._prev_wait = avg_waiting_time

        # ── 3. Starvation penalty ─────────────────────────────────────────
        if starving_process_count > 0:
            depth = max(0.0, max_waiting_time - self.STARVE_THRESHOLD)
            penalty = starving_process_count * (
                self.STARVE_BASE + self.STARVE_DEPTH * depth ** 2)
            r -= min(penalty, 15.0)

        # ── 4. CPU idle ───────────────────────────────────────────────────
        if cpu_was_idle:
            r -= self.IDLE_PENALTY

        # ── 5. Terminal ───────────────────────────────────────────────────
        if all_done:
            r += self.TERMINAL_BONUS

        return r

    def count_starving(self, ready_queue) -> int:
        return sum(1 for p in ready_queue.get_all()
                   if p.waiting_time >= self.STARVE_THRESHOLD)

    def reset(self):
        self._prev_wait = 0.0
