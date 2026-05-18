import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def user_state_to_idx(u_state, num_vals_per_feature):
    """Converts only the user features to a single index."""
    u_idx = 0
    u_multiplier = 1
    # Process from right to left (mixed radix)
    for val, s_val in zip(reversed(num_vals_per_feature), reversed(u_state)):
        u_idx += s_val * u_multiplier
        u_multiplier *= val
    return u_idx

def idx_to_user_state(u_idx, num_vals_per_feature):
    """Converts a user index back into a feature vector."""
    u_state = []
    rem_u = u_idx
    for val in reversed(num_vals_per_feature):
        u_state.append(rem_u % val)
        rem_u //= val
    u_state.reverse()
    return tuple(u_state)

def count_state_to_idx(c_state, max_count=2):
    """Converts only the count features to a single index."""
    num_counts = len(c_state)
    b = max_count + 1
    c_idx = 0
    for i, s_val in enumerate(c_state):
        c_idx += s_val * (b ** (num_counts - 1 - i))
    return c_idx

def idx_to_count_state(c_idx, num_counts, max_count=2):
    """Converts a count index back into a feature vector."""
    b = max_count + 1
    c_state = []
    rem_c = c_idx
    for _ in range(num_counts):
        c_state.append(rem_c % b)
        rem_c //= b
    c_state.reverse()
    return tuple(c_state)

def full_state_to_factored_idx(state, num_vals_per_feature, max_count=2):
    num_feats = len(num_vals_per_feature)
    u_idx = user_state_to_idx(state[:num_feats], num_vals_per_feature)
    c_idx = count_state_to_idx(state[num_feats:], max_count)
    return (u_idx, c_idx)

def full_state_to_idx(state, num_vals_per_feature, max_count=2):
    num_feats = len(num_vals_per_feature)
    num_counts = len(state) - num_feats
    u_idx = user_state_to_idx(state[:num_feats], num_vals_per_feature)
    c_idx = count_state_to_idx(state[num_feats:], max_count)
    
    count_space_size = (max_count + 1)**num_counts
    return (u_idx * count_space_size) + c_idx

def idx_to_full_state(idx, num_vals_per_feature, num_counts, max_count=2):
    count_space_size = (max_count + 1)**num_counts
    
    u_idx = idx // count_space_size
    c_idx = idx % count_space_size
    
    u_state = idx_to_user_state(u_idx, num_vals_per_feature)
    c_state = idx_to_count_state(c_idx, num_counts, max_count)
    
    return u_state + c_state

def build_next_indices(num_clusters, max_count):
    """
    Pre-compute the next indices for each action category and count state.
    next_indices[k, i] gives the next count state index if we take an action of category k from count state with index i
    This allows us to efficiently look up the next value function values during VI without having to compute the count transitions on the fly.
    """
    idx_to_count = build_idx_to_count(num_clusters, max_count)
    n_c = idx_to_count.shape[0]
    base = max_count + 1
    
    next_indices = np.zeros((num_clusters, n_c), dtype=int)  # 4 categories
    
    for k in range(num_clusters):
        c_next = idx_to_count.copy()
        
        # increment cluster k
        c_next[:, k] = np.minimum(c_next[:, k] + 1, max_count)
        
        # normalize (your logic)
        min_vals = c_next.min(axis=1, keepdims=True)
        c_next = np.minimum(c_next - min_vals, max_count)
        
        next_indices[k] = (
            sum(c_next[:, i] * (base ** (num_clusters - 1 - i)) for i in range(num_clusters))
        )
    
    return next_indices  # shape (num_clusters, n_c)

def build_idx_to_count(num_clusters, max_count):
    dims = [max_count + 1] * num_clusters  # num_clusters categories
    grid = np.indices(dims)     # shape: (num_clusters, ..., ..., ..., ...)
    idx_to_count = grid.reshape(num_clusters, -1).T
    
    return idx_to_count  # shape (n_c, num_clusters)

def calculate_shannon_diversity(counts):
    counts = np.array(counts)
    if counts.sum() == 0: return 0.0
    
    ps = counts / counts.sum()
    # Mask zeros to avoid log(0)
    ps = ps[ps > 0]
    h = -np.sum(ps * np.log(ps))
    
    # Normalize by log(number of categories)
    return h / np.log(len(counts))


if __name__ == "__main__":
    vals = [2, 3]
    counts = 4
    max_c = 3

    state = (1,2,3,3,3,3) # Example state
    u_idx, c_idx = full_state_to_factored_idx(state, vals, max_c)
    print(f"User State Index: {u_idx}")
    print(f"Count State Index: {c_idx}")

    reconstructed = idx_to_user_state(u_idx, vals) + idx_to_count_state(c_idx, counts, max_c)
    print(f"Original: {state}")
    print(f"Reconstructed: {reconstructed}")
    assert state == reconstructed