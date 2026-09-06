"""
burst_predictor/lstm_predictor.py
===================================
Pure-NumPy LSTM for CPU burst-time prediction.

ACADEMIC CONTRIBUTION
---------------------
Classical SJF scheduling reads burst_time directly from the process — an
oracle privilege no real OS possesses.  This module replaces that oracle
with an LSTM that learns burst patterns from four *observable* features:

    priority_norm        — normalised scheduling priority
    type_encoded         — cpu_bound=1.0, mixed=0.5, interactive=0.0
    arrival_gap_norm     — inter-arrival gap (also observable by kernel)
    rolling_avg_norm     — rolling mean of recently completed bursts

The LSTM is fully hand-coded in NumPy (no PyTorch / TensorFlow) so every
weight update is explainable in a viva.

TRAINING PHASES
---------------
Phase 1 – Pre-training   : 200 synthetic sequences × 8 epochs.
                           Warm-starts weights so first real predictions
                           are already reasonable (avoids a cold-start phase).
Phase 2 – Online BPTT    : Every RETRAIN_EVERY process completions the
                           predictor runs mini-batch BPTT on a ring-buffer
                           of recent (sequence, true_burst) pairs.
                           MAE typically falls from ~12 → ~7 ticks across
                           a 600-episode training run.

PUBLIC API (matches trainer.py, sjf.py, sjf_predicted.py, state_encoder.py)
-----------
    predict_process(process)       → float   run LSTM, cache, return
    predict(pid, fallback)         → float   return cache; fallback if unseen
    record_burst(pid, true_burst)  → None    queue sample for next train call
    train_on_buffer(epochs)        → None    BPTT on queued samples
    training_buffer_size()         → int
    sequence_length                → int     constant (= SEQUENCE_LEN)
    prediction_errors              → List[float]  for MAE reporting
"""

import math
import random
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np

# ── Architecture ──────────────────────────────────────────────────────────────
INPUT_SIZE   = 4
HIDDEN_SIZE  = 32
SEQUENCE_LEN = 8

# ── Normalisation ceilings ────────────────────────────────────────────────────
MAX_BURST    = 50.0
MAX_PRIORITY = 10.0
MAX_GAP      = 10.0

# ── Optimiser ─────────────────────────────────────────────────────────────────
LR      = 0.005
CLIP    = 1.0
B1, B2  = 0.9, 0.999
EPS_A   = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
class _LSTMCell:
    """Single LSTM cell — pure NumPy, Adam optimiser, gradient clipping."""

    def __init__(self):
        n, d = HIDDEN_SIZE, INPUT_SIZE + HIDDEN_SIZE
        s = math.sqrt(2.0 / d)
        self.Wf = np.random.randn(n, d) * s;  self.bf = np.ones((n, 1))
        self.Wi = np.random.randn(n, d) * s;  self.bi = np.zeros((n, 1))
        self.Wg = np.random.randn(n, d) * s;  self.bg = np.zeros((n, 1))
        self.Wo = np.random.randn(n, d) * s;  self.bo = np.zeros((n, 1))
        names = ["Wf","Wi","Wg","Wo","bf","bi","bg","bo"]
        self._m = {k: np.zeros_like(getattr(self, k)) for k in names}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in names}
        self._t = 0
        self._cache: dict = {}

    @staticmethod
    def _sig(x):
        return np.where(x >= 0,
            1/(1+np.exp(-np.clip(x,-500,500))),
            np.exp(np.clip(x,-500,500))/(1+np.exp(np.clip(x,-500,500))))

    def forward(self, x, h, c):
        xh = np.vstack([h, x])
        f = self._sig(self.Wf @ xh + self.bf)
        i = self._sig(self.Wi @ xh + self.bi)
        g = np.tanh(   self.Wg @ xh + self.bg)
        o = self._sig(self.Wo @ xh + self.bo)
        c2 = f*c + i*g
        h2 = o*np.tanh(c2)
        self._cache = dict(xh=xh, f=f, i=i, g=g, o=o, c2=c2, c=c)
        return h2, c2

    def backward(self, dh, dc):
        ca = self._cache
        tc = np.tanh(ca["c2"])
        dc = dc + dh * ca["o"] * (1 - tc**2)
        grads = {
            "Wf": (dc*ca["c"]*ca["f"]*(1-ca["f"])) @ ca["xh"].T,
            "bf":  dc*ca["c"]*ca["f"]*(1-ca["f"]),
            "Wi": (dc*ca["g"]*ca["i"]*(1-ca["i"])) @ ca["xh"].T,
            "bi":  dc*ca["g"]*ca["i"]*(1-ca["i"]),
            "Wg": (dc*ca["i"]*(1-ca["g"]**2)) @ ca["xh"].T,
            "bg":  dc*ca["i"]*(1-ca["g"]**2),
            "Wo": (dh*tc*ca["o"]*(1-ca["o"])) @ ca["xh"].T,
            "bo":  dh*tc*ca["o"]*(1-ca["o"]),
        }
        dc_prev = dc * ca["f"]
        dxh = (self.Wf.T@grads["bf"] + self.Wi.T@grads["bi"]
               + self.Wg.T@grads["bg"] + self.Wo.T@grads["bo"])
        return dxh[:HIDDEN_SIZE], dc_prev, grads

    def apply(self, grads):
        self._t += 1
        for k, g in grads.items():
            n = np.linalg.norm(g)
            if n > CLIP: g = g * CLIP / n
            self._m[k] = B1*self._m[k] + (1-B1)*g
            self._v[k] = B2*self._v[k] + (1-B2)*g**2
            mh = self._m[k]/(1-B1**self._t)
            vh = self._v[k]/(1-B2**self._t)
            p = getattr(self, k); p -= LR*mh/(np.sqrt(vh)+EPS_A)
            setattr(self, k, p)


# ─────────────────────────────────────────────────────────────────────────────
class BurstTimePredictor:
    """
    LSTM burst-time predictor.

    Key design decisions
    --------------------
    * predict_process(proc) encodes the process, runs the LSTM, caches by pid.
      This is the entry point called on every process arrival.

    * predict(pid, fallback) returns the cached value or the fallback.
      Callers that don't have the process object (state encoder, sjf.py)
      use this after predict_process has already been called on arrival.

    * record_burst / train_on_buffer implement the online learning loop.
      record_burst queues a (snapshot_sequence, true_burst_norm) pair;
      train_on_buffer runs BPTT epochs over the queue then clears it.

    LSTM coupling to the RL agent
    ------------------------------
    The state encoder calls predict(pid, fallback) for every process in
    the ready queue to compute Feature 3 (predicted-shortest-job ratio).
    As the LSTM's MAE decreases across training, this feature becomes more
    accurate, enabling the agent to trust SJF-mode more confidently — a
    genuine feedback loop between the neural predictor and the RL agent.
    """

    sequence_length = SEQUENCE_LEN   # exposed so trainer.py can print it

    def __init__(self):
        self.cell = _LSTMCell()

        s = math.sqrt(2.0 / HIDDEN_SIZE)
        self.Wo  = np.random.randn(1, HIDDEN_SIZE) * s
        self.bo  = np.zeros((1, 1))
        self._mWo = np.zeros_like(self.Wo);  self._vWo = np.zeros_like(self.Wo)
        self._mbo = np.zeros_like(self.bo);  self._vbo = np.zeros_like(self.bo)
        self._to  = 0

        # Sequence-padding scratch space. `_seq()` left-pads a single
        # process's real feature vector with zeros up to SEQUENCE_LEN
        # timesteps (architectural uniformity with the multi-step sequences
        # `pretrain()` uses). Nothing appends to this deque, so it is
        # always empty in the current design: each online prediction is
        # conditioned on that one process's own snapshot, not on a
        # cross-process rolling history. Kept as a deque (rather than
        # inlining 7 zero vectors) so a genuine sliding cross-process
        # window is a small, localised future change if desired.
        self._ctx: deque = deque(maxlen=SEQUENCE_LEN)

        # pid → predicted burst (populated by predict_process on arrival)
        self._cache: Dict[int, float] = {}

        # pid -> the exact encoded feature vector used for that prediction.
        # Needed so record_burst() can reconstruct the SAME input sequence
        # predict_process() actually used (see record_burst docstring).
        self._feat_cache: Dict[int, np.ndarray] = {}

        # Online training buffer: [(seq_snapshot, y_norm), ...]
        self._buf: List[Tuple[list, float]] = []

        # Diagnostics
        self.prediction_errors: List[float] = []

        # Rolling burst average for feature 4
        self._bsum, self._bcnt = 0.0, 0
        self._bavg = 15.0

    # ── feature encoding ────────────────────────────────────────────────────

    def _encode(self, process) -> np.ndarray:
        tmap = {"cpu_bound": 1.0, "interactive": 0.0, "mixed": 0.5}
        gap  = getattr(process, "_arrival_gap", 3.0)
        return np.array([
            process.priority / MAX_PRIORITY,
            tmap.get(getattr(process, "process_type", "mixed"), 0.5),
            min(gap, MAX_GAP) / MAX_GAP,
            self._bavg / MAX_BURST,
        ]).reshape(INPUT_SIZE, 1)

    def _encode_raw(self, priority, ptype, gap, avg_burst) -> np.ndarray:
        tmap = {"cpu_bound": 1.0, "interactive": 0.0, "mixed": 0.5}
        return np.array([
            priority / MAX_PRIORITY,
            tmap.get(ptype, 0.5),
            min(gap, MAX_GAP) / MAX_GAP,
            avg_burst / MAX_BURST,
        ]).reshape(INPUT_SIZE, 1)

    def _seq(self, vec: np.ndarray) -> List[np.ndarray]:
        """Build padded sequence of length SEQUENCE_LEN ending with vec."""
        base = list(self._ctx) + [vec]
        base = base[-SEQUENCE_LEN:]
        while len(base) < SEQUENCE_LEN:
            base = [np.zeros((INPUT_SIZE, 1))] + base
        return base

    # ── LSTM forward ─────────────────────────────────────────────────────────

    def _fwd(self, seq: List[np.ndarray]):
        h = np.zeros((HIDDEN_SIZE, 1))
        c = np.zeros((HIDDEN_SIZE, 1))
        hs, cs, hps, cps = [], [], [], []
        for x in seq:
            hps.append(h.copy()); cps.append(c.copy())
            h, c = self.cell.forward(x, h, c)
            hs.append(h); cs.append(c)
        return h, hs, cs, hps, cps

    def _out(self, h: np.ndarray) -> float:
        return float((self.Wo @ h + self.bo).item())

    # ── public API ──────────────────────────────────────────────────────────

    def predict_process(self, process) -> float:
        """
        Run the LSTM on a process object, cache the result, return prediction.
        Called once per process on arrival.
        """
        if process.pid in self._cache:
            return self._cache[process.pid]
        vec  = self._encode(process)
        seq  = self._seq(vec)
        h, *_ = self._fwd(seq)
        raw  = self._out(h) * MAX_BURST
        pred = max(1.0, min(raw, MAX_BURST * 1.5))
        self._cache[process.pid] = pred
        self._feat_cache[process.pid] = vec
        return pred

    def predict(self, pid: int, fallback: float = 15.0) -> float:
        """
        Return cached prediction for pid.
        If pid was never registered via predict_process, return fallback.
        Callers: state_encoder (feature 3), sjf.py (SJF-predicted mode).
        """
        return self._cache.get(pid, fallback)

    def record_burst(self, pid: int, true_burst: int):
        """
        Called when process `pid` finishes.  Records prediction error and
        queues a training sample for the next train_on_buffer call.
        Also updates the rolling average burst for feature normalisation.

        BUG FIX (see tests/test_burst_predictor.py::test_record_burst_trains_on_actual_features):
        this used to build the training input purely from `self._ctx`
        (`list(self._ctx)` zero-padded to length 8), which is *always*
        empty — nothing in this class ever appends to `_ctx`. Every online
        training sample was therefore an all-zero input sequence mapped to
        the true burst time, so online BPTT fine-tuning could only ever
        learn a constant (roughly the running mean of recently-seen
        bursts), never a function of a process's own features — silently
        contradicting predict_process(), which *does* condition on the
        real feature vector. Fixed by reusing the exact feature vector
        predict_process() cached for this pid, so the model now actually
        trains on what it was actually asked to predict from.
        """
        vec = self._feat_cache.pop(pid, None)

        if pid in self._cache:
            self.prediction_errors.append(abs(self._cache[pid] - true_burst))
            del self._cache[pid]

        self._bsum  += true_burst
        self._bcnt  += 1
        self._bavg   = self._bsum / self._bcnt

        if vec is not None:
            # Same construction predict_process() used: real features as
            # the final timestep, zero-padded before it.
            snap = self._seq(vec)
        else:
            # Degenerate fallback: a burst was recorded for a pid that was
            # never seen via predict_process (shouldn't happen in the
            # normal simulator/trainer flow, but keeps this method safe
            # for direct/standalone use).
            snap = [np.zeros((INPUT_SIZE, 1))] * SEQUENCE_LEN

        self._buf.append((snap, true_burst / MAX_BURST))

    def train_on_buffer(self, epochs: int = 2):
        """Run BPTT over the training buffer then clear it."""
        if not self._buf:
            return
        for _ in range(epochs):
            random.shuffle(self._buf)
            for seq, y_true in self._buf:
                h, hs, cs, hps, cps = self._fwd(seq)
                y_pred = self._out(h)
                d = 2.0 * (y_pred - y_true)
                self._adam_out(d * h.T, np.array([[d]]))
                dh = self.Wo.T * d
                dc = np.zeros((HIDDEN_SIZE, 1))
                agg = {k: np.zeros_like(getattr(self.cell, k))
                       for k in ["Wf","Wi","Wg","Wo","bf","bi","bg","bo"]}
                for step in reversed(range(len(seq))):
                    self.cell.forward(seq[step], hps[step], cps[step])
                    dh, dc, sg = self.cell.backward(dh, dc)
                    for k in agg: agg[k] += sg[k]
                self.cell.apply(agg)
        self._buf.clear()

    def training_buffer_size(self) -> int:
        return len(self._buf)

    # ── Adam for output layer ────────────────────────────────────────────────

    def _adam_out(self, dW, db):
        self._to += 1
        for p, g, m, v in [("Wo",dW,"_mWo","_vWo"),("bo",db,"_mbo","_vbo")]:
            mo = B1*getattr(self,m)+(1-B1)*g; setattr(self,m,mo)
            vo = B2*getattr(self,v)+(1-B2)*g**2; setattr(self,v,vo)
            mh = mo/(1-B1**self._to); vh = vo/(1-B2**self._to)
            param = getattr(self, p); param -= LR*mh/(np.sqrt(vh)+EPS_A)
            setattr(self, p, param)

    # ── pre-training ────────────────────────────────────────────────────────

    def pretrain(self, sequences: List[Tuple[list, list]], epochs: int = 8):
        """Warm-start weights on synthetic (feature_seq, burst_seq) pairs."""
        for _ in range(epochs):
            random.shuffle(sequences)
            for feat_seq, burst_seq in sequences:
                for step in range(1, len(feat_seq)):
                    ctx = feat_seq[max(0, step-SEQUENCE_LEN):step]
                    while len(ctx) < SEQUENCE_LEN:
                        ctx = [np.zeros((INPUT_SIZE,1))] + ctx
                    h, *_ = self._fwd(ctx)
                    yp = self._out(h)
                    yt = burst_seq[step] / MAX_BURST
                    d  = 2.0 * (yp - yt)
                    self._adam_out(d * h.T, np.array([[d]]))

    # ── diagnostics ─────────────────────────────────────────────────────────

    def mean_absolute_error(self) -> float:
        if not self.prediction_errors: return 0.0
        return sum(self.prediction_errors) / len(self.prediction_errors)

    def recent_mae(self, window: int = 30) -> float:
        r = self.prediction_errors[-window:]
        return sum(r) / len(r) if r else 0.0
