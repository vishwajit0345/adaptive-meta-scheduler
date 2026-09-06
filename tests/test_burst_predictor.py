"""
tests/test_burst_predictor.py
==============================
Tests for the pure-NumPy LSTM burst-time predictor.

We don't assert a specific MAE (that depends on random initialisation and
is properly the job of the reported experiment, not a unit test). Instead
we test the contract every other module relies on:
  - predictions are cached per-pid and bounded to a sane range
  - record_burst / train_on_buffer round-trip without crashing
  - training on a trivial, noise-free task drives the loss down
    (a basic learning sanity check for the hand-rolled BPTT)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.process import Process
from burst_predictor.lstm_predictor import BurstTimePredictor, MAX_BURST


def _make_process(pid, burst=10, priority=5, ptype="mixed", gap=3.0):
    p = Process(pid=pid, arrival_time=0, burst_time=burst,
                priority=priority, process_type=ptype)
    p._arrival_gap = gap
    return p


def test_predict_process_returns_bounded_value_and_caches():
    predictor = BurstTimePredictor()
    p = _make_process(pid=1)
    pred = predictor.predict_process(p)

    assert 1.0 <= pred <= MAX_BURST * 1.5
    assert predictor._cache[1] == pred
    # calling again returns the cached value without recomputation
    assert predictor.predict_process(p) == pred


def test_predict_falls_back_when_pid_unseen():
    predictor = BurstTimePredictor()
    assert predictor.predict(pid=999, fallback=42.0) == 42.0


def test_record_burst_clears_cache_and_tracks_error():
    predictor = BurstTimePredictor()
    p = _make_process(pid=5, burst=20)
    predictor.predict_process(p)
    assert 5 in predictor._cache

    predictor.record_burst(pid=5, true_burst=20)
    assert 5 not in predictor._cache
    assert len(predictor.prediction_errors) == 1
    assert predictor.prediction_errors[0] >= 0.0


def test_record_burst_updates_rolling_average():
    predictor = BurstTimePredictor()
    predictor.record_burst(pid=1, true_burst=30)
    predictor.record_burst(pid=2, true_burst=10)
    assert abs(predictor._bavg - 20.0) < 1e-9


def test_mean_absolute_error_zero_when_no_predictions():
    predictor = BurstTimePredictor()
    assert predictor.mean_absolute_error() == 0.0
    assert predictor.recent_mae() == 0.0


def test_record_burst_trains_on_actual_features():
    """
    Regression test for a real bug: record_burst() used to build its BPTT
    training sample purely from an always-empty context window, so every
    online training sample was an all-zero input sequence regardless of
    which process actually completed. Fixed by caching predict_process()'s
    real feature vector per pid and reusing it here. The last timestep of
    the queued training sequence must therefore be non-zero and must match
    what predict_process() actually used.
    """
    predictor = BurstTimePredictor()
    p = _make_process(pid=7, burst=20, priority=3, ptype="cpu_bound", gap=4.0)
    predictor.predict_process(p)
    expected_vec = predictor._feat_cache[7].copy()

    predictor.record_burst(pid=7, true_burst=20)

    assert predictor.training_buffer_size() == 1
    seq, y_norm = predictor._buf[0]
    assert len(seq) == 8
    # First 7 timesteps are zero-padding; the last is the real feature vector.
    for step in seq[:-1]:
        assert np.allclose(step, 0.0)
    assert np.allclose(seq[-1], expected_vec)
    assert not np.allclose(seq[-1], 0.0)  # i.e. NOT the old all-zero bug
    assert abs(y_norm - 20 / MAX_BURST) < 1e-9


def test_record_burst_without_prior_prediction_falls_back_safely():
    """record_burst() for a pid never seen via predict_process() must not crash."""
    predictor = BurstTimePredictor()
    predictor.record_burst(pid=123, true_burst=15)  # no predict_process() call first
    assert predictor.training_buffer_size() == 1
    seq, _ = predictor._buf[0]
    assert all(np.allclose(step, 0.0) for step in seq)


def test_training_buffer_round_trip_does_not_crash():
    predictor = BurstTimePredictor()
    for pid in range(5):
        p = _make_process(pid=pid, burst=10 + pid)
        predictor.predict_process(p)
        predictor.record_burst(pid=pid, true_burst=10 + pid)
    assert predictor.training_buffer_size() == 5
    predictor.train_on_buffer(epochs=1)
    assert predictor.training_buffer_size() == 0  # buffer cleared after training


def test_lstm_learns_a_trivial_constant_mapping():
    """
    Sanity check for the hand-rolled BPTT: on a noise-free task where every
    input maps to the same target burst, repeated training must reduce the
    output error, not leave it random or diverge.
    """
    np.random.seed(0)
    predictor = BurstTimePredictor()

    TARGET = 20  # ticks

    def make_and_record(pid):
        p = _make_process(pid=pid, burst=TARGET, priority=5,
                           ptype="mixed", gap=3.0)
        pred = predictor.predict_process(p)
        predictor.record_burst(pid=pid, true_burst=TARGET)
        return pred

    # Error before any training (fresh random weights)
    first_errors = [abs(make_and_record(pid) - TARGET) for pid in range(20)]
    predictor.train_on_buffer(epochs=5)

    # Error after training on the same constant task
    later_errors = [abs(make_and_record(pid) - TARGET) for pid in range(100, 120)]
    predictor.train_on_buffer(epochs=5)
    final_errors = [abs(make_and_record(pid) - TARGET) for pid in range(200, 220)]

    assert np.mean(final_errors) < np.mean(first_errors)
