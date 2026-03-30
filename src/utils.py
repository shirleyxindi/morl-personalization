import numpy as np

def build_strides(num_vals=3, max_count=4):
    """Precompute strides once to avoid redundant math."""
    b = max_count + 1
    v = num_vals
    
    if max_count > 0:
        # 7-dimensional state: (t, time, m, ac, ds, ps, ss)
        return [v**2 * b**4, v**1 * b**4, v**0 * b**4, b**3, b**2, b**1, 1]
    else:
        # 3-dimensional state: (t, time, m)
        return [v**2, v**1, 1]
    
def state_to_idx(state, num_vals=3, max_count=4):
    strides = build_strides(num_vals, max_count)
    return sum(s * w for s, w in zip(state, strides))
    
def idx_to_state(idx, num_vals=3, max_count=4):
    strides = build_strides(num_vals, max_count)
    state = []
    remaining = idx
    for s in strides:
        state.append(remaining // s)
        remaining %= s
    return tuple(state)

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

def build_next_indices(MAX_COUNT):
    """
    Pre-compute the next indices for each action category and count state.
    next_indices[k, i] gives the next count state index if we take an action of category k from count state with index i
    This allows us to efficiently look up the next value function values during VI without having to compute the count transitions on the fly.
    """
    idx_to_count = build_idx_to_count(MAX_COUNT)
    n_c = idx_to_count.shape[0]
    base = MAX_COUNT + 1
    
    next_indices = np.zeros((4, n_c), dtype=int)  # 4 categories
    
    for k in range(4):
        c_next = idx_to_count.copy()
        
        # increment category k
        c_next[:, k] = np.minimum(c_next[:, k] + 1, MAX_COUNT)
        
        # normalize (your logic)
        min_vals = c_next.min(axis=1, keepdims=True)
        c_next = np.minimum(c_next - min_vals, MAX_COUNT)
        
        # map back to indices
        next_indices[k] = (
            c_next[:, 0] * base**3 +
            c_next[:, 1] * base**2 +
            c_next[:, 2] * base +
            c_next[:, 3]
        )
    
    return next_indices  # shape (4, n_c)

def build_idx_to_count(MAX_COUNT):
    dims = [MAX_COUNT + 1] * 4  # 4 categories
    grid = np.indices(dims)     # shape: (4, ..., ..., ..., ...)
    idx_to_count = grid.reshape(4, -1).T
    
    return idx_to_count  # shape (n_c, 4)

