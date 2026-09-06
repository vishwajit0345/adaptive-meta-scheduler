"""
rl_agent/state_encoder.py
==========================
Converts live simulator state into a discrete 5-tuple for Q-learning.

STATE SPACE DESIGN — WHY 5 FEATURES, NOT 8
-------------------------------------------
The uploaded v4 had 3^8 = 6,561 states with thresholds that made 5 of 8
features stick to a single bin >85% of the time.  600 episodes gave each
effectively-reachable state only ~8 visits — far too few for Q-values to
converge.

Fixes applied:
  1. Reduced to 5 features → 3^5 = 243 states.
  2. All thresholds calibrated at the 33rd/66th percentile of the REAL
     simulation data distribution (1,000 sampled mid-episode snapshots).
     Every bin now receives roughly 30–35% of samples.
  3. State coverage: 600 eps × ~110 decisions = 66,000 visits → ~272/state.
     Q-learning converges reliably above ~30 visits per state.

FEATURES
--------
  F0  queue_length      — # processes waiting (p33=13, p66=23)
  F1  avg_burst         — mean burst time     (p33=17, p66=19)
  F2  burst_variance    — workload diversity  (p33=84, p66=105)
  F3  shortest_ratio    — min/mean burst      (p33=0.10, p66=0.20)
                          ★ LSTM-coupled: uses predicted bursts when predictor
                          is provided, true remaining_time as fallback.
  F4  starvation        — max waiting time    (p33=37, p66=76)

LSTM COUPLING (Feature 3)
--------------------------
When a BurstTimePredictor is provided, Feature 3 is computed from
*predicted* burst times (via predictor.predict(pid, fallback=avg_burst))
rather than the true remaining_time.  This creates a genuine feedback loop:

    LSTM improves → Feature 3 more accurate → agent learns better SJF policy
    → higher reward → agent selects SJF more in the right states → more
    completions → LSTM gets more online training data → MAE decreases further

This positive coupling is the core novel contribution of the system.
"""

import math
from simulator.ready_queue import ReadyQueue


class StateEncoder:

    # Thresholds calibrated at p33/p66 of real simulation data
    QUEUE_LEN_T   = (13, 23)
    AVG_BURST_T   = (17, 19)
    VARIANCE_T    = (84, 105)
    SHORTEST_T    = (0.10, 0.20)   # F3: low = big spread → SJF opportunity
    STARVATION_T  = (37, 76)

    def encode(
        self,
        ready_queue: ReadyQueue,
        recent_arrival_rate: float,
        predictor=None,
    ) -> tuple:
        """
        Return a 5-element discrete state tuple.

        Parameters
        ----------
        ready_queue         : Current ready queue.
        recent_arrival_rate : Not used (removed — collinear with queue_len).
        predictor           : BurstTimePredictor (optional).
                              When provided, F3 uses LSTM-predicted bursts.
        """
        f0 = self._bin(ready_queue.size(),                self.QUEUE_LEN_T)
        f1 = self._bin(ready_queue.average_burst_time(),  self.AVG_BURST_T)
        f2 = self._bin(ready_queue.variance_burst_time(), self.VARIANCE_T)

        avg_burst = ready_queue.average_burst_time()
        procs     = ready_queue.get_all() if not ready_queue.is_empty() else []

        # ── F3: shortest-job ratio (LSTM-coupled) ─────────────────────────
        if procs and avg_burst > 0:
            if predictor is not None:
                bursts = [predictor.predict(p.pid, fallback=avg_burst) for p in procs]
                mean_p = sum(bursts) / len(bursts)
                ratio  = min(bursts) / mean_p if mean_p > 0 else 1.0
            else:
                ratio = min(p.remaining_time for p in procs) / avg_burst
        else:
            ratio = 1.0
        f3 = self._bin(ratio, self.SHORTEST_T)

        # ── F4: starvation (max waiting time) ────────────────────────────
        f4 = self._bin(float(ready_queue.max_waiting_time()), self.STARVATION_T)

        return (f0, f1, f2, f3, f4)

    @staticmethod
    def _bin(v: float, t: tuple) -> int:
        lo, hi = t
        return 0 if v < lo else (1 if v < hi else 2)

    def state_space_size(self) -> int:
        """3^5 = 243 calibrated states."""
        return 3 ** 5
