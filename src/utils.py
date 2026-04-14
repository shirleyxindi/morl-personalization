import numpy as np
from sklearn.cluster import KMeans

def build_strides(num_feats, num_counts, num_vals=3, max_count=3):
    """Precompute strides once to avoid redundant math."""    
    b = max_count + 1
    v = num_vals
    return [v**f * b**num_counts for f in range(num_feats - 1, -1, -1)] + [b**f for f in range(num_counts - 1, -1, -1)]

def state_to_idx(state, num_feats, num_counts, num_vals=3, max_count=3):
    strides = build_strides(num_feats, num_counts, num_vals, max_count)
    return sum(s * w for s, w in zip(state, strides))

def state_to_u_c_idx(state, num_feats, num_counts, num_vals=3, max_count=3):
    """Split state into separate user state index and count state index."""
    b = max_count + 1
    v = num_vals

    # User state: first 3 dims (t, time, m)
    u_strides = [v**f for f in range(num_feats - 1, -1, -1)]
    u_idx = sum(s * w for s, w in zip(state[:num_feats], u_strides))

    # Count state: last 4 dims (ac, ds, ps, ss)
    c_strides = [b**f for f in range(num_counts - 1, -1, -1)]
    c_idx = sum(s * w for s, w in zip(state[num_feats:], c_strides))

    return u_idx, c_idx

def idx_to_state(idx, num_feats, num_counts, num_vals=3, max_count=3):
    strides = build_strides(num_feats, num_counts, num_vals, max_count)
    state = []
    remaining = idx
    for s in strides:
        state.append(remaining // s)
        remaining %= s
    return tuple(state)

def u_c_idx_to_state(u_idx, c_idx, num_feats, num_counts, num_vals=3, max_count=3):
    """Reconstruct full state from separate user and count indices."""
    b = max_count + 1
    v = num_vals

    u_strides = [v**f for f in range(num_feats - 1, -1, -1)]
    u_state = []
    remaining = u_idx
    for s in u_strides:
        u_state.append(remaining // s)
        remaining %= s

    c_strides = [b**f for f in range(num_counts - 1, -1, -1)]
    c_state = []
    remaining = c_idx
    for s in c_strides:
        c_state.append(remaining // s)
        remaining %= s

    return tuple(u_state + c_state)

def get_skill_tiers(skill_vector):
    return np.select(
                [skill_vector >= 0.75, skill_vector >= 0.50, skill_vector >= 0.25],
                [1.0, 0.67, 0.33],
                default=0.0
            )
        
def get_skill_reward(skill_current, action_scores):
    """
    Given a vector of current skill tiers, add action scores, calculate new skill tiers and return summed increase in skill tiers.
    """
    new_skill_tiers = get_skill_tiers(skill_current + action_scores)
    return new_skill_tiers - skill_current

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

def scale_rewards(values, new_min=0, new_max=1, reverse=False):
    old_min = np.min(values)
    old_max = np.max(values)
    
    if old_max == old_min:
        return np.full_like(values, new_min)  # Avoid division by zero, set all to new_min
    
    if not reverse:
        scaled = (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    else:
        scaled = (old_max - values) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled

def cluster_actions(action_data, cluster_vars, num_clusters=5):
    cluster_models = {}

    # Cluster per variable
    for col in cluster_vars:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        action_data[f'{col}_cluster'] = kmeans.fit_predict(action_data[[col]].values)
        cluster_models[col] = kmeans

    # Cluster on all variables together
    kmeans_all = KMeans(n_clusters=num_clusters, random_state=42)
    action_data['cluster_all'] = kmeans_all.fit_predict(action_data[cluster_vars].values)
    cluster_models['all'] = kmeans_all
    cluster_cols = [f'{col}_cluster' for col in cluster_vars] + ['cluster_all']
    return action_data, cluster_models, cluster_cols

# state = (2,0,0,1,0)
# idx = state_to_idx(state, num_feats=5, num_counts=0, num_vals=3, max_count=0)
# print(idx_to_state(idx, num_feats=5, num_counts=0, num_vals=3, max_count=0))

