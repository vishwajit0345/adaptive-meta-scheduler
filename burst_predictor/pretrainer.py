"""
burst_predictor/pretrainer.py - Generates synthetic sequences and pre-trains
the LSTM burst predictor before the main simulation begins.

Pre-training strategy:
  1. Generate 200 synthetic workload sequences (various types)
  2. Encode each process into its 4-feature vector
  3. Train the LSTM to map observable features → burst time
  4. After pre-training the predictor makes reasonable estimates immediately
     rather than needing many online completions to warm up

This is important for fairness: SJF_Predicted should be competitive from
the first few processes, not spending the whole simulation in a cold-start
learning phase.
"""

import math
import random
import numpy as np
from typing import List, Tuple


INPUT_SIZE     = 4
MAX_BURST_NORM = 50.0
MAX_PRIORITY   = 10.0
MAX_GAP        = 10.0


def _encode_raw(
    priority: int,
    process_type: str,
    arrival_gap: float,
    recent_avg_burst: float,
) -> "np.ndarray":
    """Encode raw process attributes into a normalised feature vector."""
    type_map = {"cpu_bound": 1.0, "interactive": 0.0, "mixed": 0.5}
    return np.array([
        priority / MAX_PRIORITY,
        type_map.get(process_type, 0.5),
        min(arrival_gap, MAX_GAP) / MAX_GAP,
        recent_avg_burst / MAX_BURST_NORM,
    ]).reshape(INPUT_SIZE, 1)


def generate_pretraining_sequences(
    num_sequences: int = 200,
) -> List[Tuple[list, list]]:
    """
    Generate synthetic (feature_sequence, burst_sequence) pairs for pre-training.

    Each sequence simulates a series of process arrivals across all three
    workload profiles to give the LSTM broad coverage.

    Args:
        num_sequences: How many independent sequences to generate.

    Returns:
        List of (feature_vector_list, burst_int_list) tuples.
    """
    PROFILES = {
        "cpu_bound":   {"burst": (15, 40), "gap": (2, 6),  "priority": (1, 5)},
        "interactive": {"burst": (1,  8),  "gap": (1, 3),  "priority": (1, 10)},
        "mixed":       {"burst": (1,  35), "gap": (1, 5),  "priority": (1, 10)},
    }

    sequences = []

    for _ in range(num_sequences):
        profile_name = random.choice(list(PROFILES.keys()))
        profile      = PROFILES[profile_name]
        seq_length   = random.randint(10, 25)

        feature_seq = []
        burst_seq   = []
        running_burst_sum   = 0.0
        running_burst_count = 0

        for step in range(seq_length):
            burst_time = random.randint(*profile["burst"])
            priority   = random.randint(*profile["priority"])
            gap        = random.randint(*profile["gap"])

            # Determine process type label from burst length
            if burst_time <= 5:
                ptype = "interactive"
            elif burst_time >= 20:
                ptype = "cpu_bound"
            else:
                ptype = "mixed"

            recent_avg = (running_burst_sum / running_burst_count
                          if running_burst_count > 0 else 15.0)

            vec = _encode_raw(priority, ptype, float(gap), recent_avg)
            feature_seq.append(vec)
            burst_seq.append(burst_time)

            running_burst_sum   += burst_time
            running_burst_count += 1

        sequences.append((feature_seq, burst_seq))

    return sequences


def pretrain_predictor(predictor, verbose: bool = True) -> None:
    """
    Run the full pre-training pipeline on the given BurstTimePredictor.

    Args:
        predictor: A BurstTimePredictor instance (weights will be updated in-place).
        verbose  : Print pre-training summary if True.
    """
    sequences = generate_pretraining_sequences(num_sequences=200)

    if verbose:
        print(f"  Pre-training LSTM on {len(sequences)} synthetic sequences "
              f"({sum(len(s[1]) for s in sequences)} data points)...")

    predictor.pretrain(sequences, epochs=8)

    if verbose:
        print(f"  Pre-training complete. LSTM ready for online fine-tuning.\n")
