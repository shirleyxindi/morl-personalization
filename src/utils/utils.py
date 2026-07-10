"""
General helper functions.

Author: Shirley Li
Date: July 2026
"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from itertools import product

def build_state_space(num_vals_per_feature):
    """Build forward and reverse lookup tables for state space.

    Args:
        num_vals_per_feature: Number of values for each feature.

    Returns:
        Tuple of dictionaries mapping state tuples to indices and back.
    """
    states = list(product(*[range(n) for n in num_vals_per_feature]))
    state_to_idx = {s: i for i, s in enumerate(states)}
    idx_to_state = {i: s for i, s in enumerate(states)}
    return state_to_idx, idx_to_state

def calculate_shannon_diversity(counts):
    """Compute normalized Shannon diversity for a vector of category counts.

    Args:
        counts: Category counts or frequencies.

    Returns:
        Normalized Shannon diversity in [0, 1].
    """
    counts = np.array(counts)
    if counts.sum() == 0: return 0.0
    
    ps = counts / counts.sum()
    # Mask zeros to avoid log(0)
    ps = ps[ps > 0]
    h = -np.sum(ps * np.log(ps))
    
    return h / np.log(len(counts))

def action_idx_to_action(idx):
    """Map an action index to its human-readable challenge cluster and diversity label.

    Args:
        idx: Integer action index.

    Returns:
        String label for the action.
    """
    joint_cluster_to_action = {
        0: "(D, 0)",
        1: "(EU, 0)",
        2: "(EUD, 0)",
        3: "(D, 1)",
        4: "(EU, 1)",
        5: "(EUD, 1)",
    }
    return joint_cluster_to_action[idx]
