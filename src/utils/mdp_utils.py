"""
Utilities for estimating (MO)MDP components from processed data.

Author: Shirley Li
Date: July 2026
"""

import numpy as np
import pandas as pd

def compute_completion_probabilities(df, num_states, num_actions, action_col='joint_cluster', alpha=None, prior='global'):
    """Estimate completion probabilities P(completion | state, action).

    Args:
        df: Processed DataFrame with `s_idx`, action, and `completed` columns.
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        action_col: Action index column name.
        alpha: Smoothing strength; defaults to 1 when omitted.
        prior: Prior type, one of `global`, `action_only`, or `none`.

    Returns:
        Array of shape (num_states, num_actions) with completion probabilities.
    """
    alpha = 1 if alpha is None else alpha

    group = df.groupby(['s_idx', action_col])
    success_counts = group['completed'].sum()
    total_counts = group['completed'].count()

    new_index = pd.MultiIndex.from_product([range(num_states), range(num_actions)], names=['s_idx', action_col])

    success_counts = success_counts.reindex(new_index, fill_value=0).values.reshape(num_states, num_actions)
    total_counts = total_counts.reindex(new_index, fill_value=0).values.reshape(num_states, num_actions)
    
    if prior == 'global':
        prior = df['completed'].mean()
    elif prior == 'action_only':
        action_p = df.groupby(action_col)['completed'].mean()
        prior = action_p.reindex(range(num_actions), fill_value=df['completed'].mean()).values
        prior = prior[np.newaxis, :]  # reshape for broadcasting over states
    elif prior == 'none':
        prior = 0

    completion_probs = (success_counts + alpha * prior) / (total_counts + alpha)
    return completion_probs

def compute_completion_probabilities_action_only(df, nA, action_col):
    """Estimate completion probabilities per action.

    Args:
        df: Processed DataFrame with `completed` and action columns.
        nA: Number of actions.
        action_col: Action index column name.

    Returns:
        Array of shape (nA,) with mean completion probabilities per action.
    """
    baseline_completion = df['completed'].mean()
    return df.groupby(action_col)['completed'].mean().reindex(range(nA), fill_value=baseline_completion).values

def compute_rewards(df, nU, nA, nO, obj_cols, action_col, mapping=None):
    """Build a dense reward tensor indexed by state, action, and objective.

    Args:
        df: Processed DataFrame with reward columns.
        nU: Number of user-state combinations.
        nA: Number of actions.
        nO: Number of objectives.
        obj_cols: Reward column names.
        action_col: Action index column name.
        mapping: Optional action-to-feature mapping used for diversity reward.

    Returns:
        Array of shape (nU, nA, nO).
    """
    reward_matrix = np.zeros((nU, nA, nO))
    for o_idx, reward_col in enumerate(obj_cols):
        if reward_col == 'r_diversity':
            for a in range(nA):
                if mapping is not None:
                    reward_matrix[:, a, o_idx] = mapping.loc[a]['a_novelty']
                else:
                    reward_matrix[:, a, o_idx] = df[df[action_col] == a][reward_col].mean()
        else:
            cluster_reward = df.groupby(['s_idx', action_col])[reward_col].mean()
            for (s, a), reward in cluster_reward.items():
                reward_matrix[s, a, o_idx] = reward

    return reward_matrix

def compute_avg_rewards(df, nA, action_col, obj_cols):
    """Compute the mean reward per action for each objective.

    Args:
        df: Processed DataFrame with reward columns.
        nA: Number of actions.
        action_col: Action index column name.
        obj_cols: Reward column names.

    Returns:
        Array of shape (nA, len(obj_cols)) with action-level means.
    """
    rewards_per_action = np.zeros((nA, len(obj_cols)))
    for o_idx, reward_col in enumerate(obj_cols):
        action_rewards = df.groupby(action_col)[reward_col].mean()
        for a in range(nA):
            if a in action_rewards.index:
                rewards_per_action[a, o_idx] = action_rewards.loc[a]
    return rewards_per_action

def compute_rewards_global(df, obj_cols):
    """Compute the global mean reward for each objective.

    Args:
        df: Processed DataFrame with reward columns.
        obj_cols: Reward column names.

    Returns:
        Array of shape (len(obj_cols),) with global means.
    """
    global_rewards = np.zeros(len(obj_cols))
    for o_idx, reward_col in enumerate(obj_cols):
        global_rewards[o_idx] = df[reward_col].mean()
    return global_rewards

def compute_transition_probabilities(df, num_states, num_actions, action_col, alpha=None, prior='global'):
    """Estimate transition probabilities P(s' | s, a).

    Args:
        df: Processed DataFrame with state and next-state indices.
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        action_col: Action index column name.
        alpha: Smoothing strength; defaults to 1 when omitted.
        prior: Prior type, typically `global`.

    Returns:
        Array of shape (num_states, num_actions, num_states).
    """
    alpha = 1 if alpha is None else alpha

    counts = df.groupby(['s_idx', action_col, 'sp_idx']).size().unstack(fill_value=0)

    all_states = list(range(num_states))
    all_actions = list(range(num_actions))
    new_index = pd.MultiIndex.from_product([all_states, all_actions], names=['s_idx', action_col])

    # Fill missing counts with 0
    counts = counts.reindex(new_index, fill_value=0)
    counts = counts.reindex(columns=all_states, fill_value=0)
    
    # Apply smoothing and normalize
    if prior == 'global':
        prior = df['sp_idx'].value_counts(normalize=True).sort_index().values 
    else:
        prior = 1  # similar to uniform prior

    counts_smoothed = counts + alpha * prior
    transition_probs = counts_smoothed.div(counts_smoothed.sum(axis=1), axis=0).values
    
    # reshape to (num_states, num_actions, num_states)
    return transition_probs.reshape(num_states, num_actions, num_states)

def compute_transition_probabilities_action_only(df, num_actions, action_col, num_states, alpha=None, prior='global'):
    """Estimate transition probabilities P(s' | a).

    Args:
        df: Processed DataFrame with next-state indices.
        num_actions: Number of discrete actions.
        action_col: Action index column name.
        num_states: Number of discrete states.
        alpha: Smoothing strength; defaults to 1 when omitted.
        prior: Prior type, typically `global`.

    Returns:
        Array of shape (num_actions, num_states).
    """
    alpha = 1 if alpha is None else alpha

    counts = df.groupby([action_col, 'sp_idx']).size().unstack(fill_value=0)

    all_actions = list(range(num_actions))
    new_index = pd.Index(all_actions, name=action_col)

    # Fill missing counts with 0
    counts = counts.reindex(new_index, fill_value=0)
    counts = counts.reindex(columns=range(num_states), fill_value=0)
    
    # Apply Laplace smoothing and normalize
    if prior == 'global':
        prior = df['sp_idx'].value_counts(normalize=True).sort_index().values
    else:
        prior = 1  # similar to uniform prior

    counts_smoothed = counts + alpha * prior
    transition_probs = counts_smoothed.div(counts_smoothed.sum(axis=1), axis=0).values
    
    return transition_probs.reshape(num_actions, num_states)

    
def compute_reward_probabilities(df, reward_col, num_states, num_actions, action_col, lam=1, reward_values=None, prior='action_only'):
    """Estimate categorical reward distributions for each state-action pair.

    Args:
        df: Processed DataFrame with reward observations.
        reward_col: Reward column to model.
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        action_col: Action index column name.
        lam: Smoothing strength.
        reward_values: Optional explicit support values.
        prior: Prior type, either `action_only` or global.

    Returns:
        Array of shape (num_states, num_actions, num_reward_values).
    """
    if reward_col != 'PAY_next':
        df = df[df['completed'] == 1]  # Only consider completed actions for reward probabilities, since rewards are only observed for completed actions
    if reward_values is None:
        reward_values = np.arange(int(df[reward_col].min()), int(df[reward_col].max()) + 1)
    
    reward_values = list(reward_values)
    
    if prior == 'action_only':
        counts_prior = (
            df.groupby(action_col)[reward_col]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=reward_values, fill_value=0)
            .values  # shape: (num_actions, num_rewards)
        )
        action_p = counts_prior / counts_prior.sum(axis=1, keepdims=True)  # normalize to get probabilities

        # reshape for broadcasting over states: (1, num_actions, num_rewards)
        prior = action_p[np.newaxis, :, :]
    else:
        counts_prior = df[reward_col].value_counts().reindex(reward_values, fill_value=0)
        prior = (counts_prior / counts_prior.sum()).values  # shape: (num_rewards,)

    # counts per (s,a) pair for each reward value
    counts = (
        df.groupby(['s_idx', action_col])[reward_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=reward_values, fill_value=0)
        .reindex(pd.MultiIndex.from_product([range(num_states), range(num_actions)], names=['s_idx', action_col]), fill_value=0)
        .values
        .reshape(num_states, num_actions, len(reward_values))
    )
    
    # total counts per (s,a) pair
    N = counts.sum(axis=-1, keepdims=True)
    smoothed = (counts + lam * prior) / (N + lam)

    return smoothed

def compute_reward_probabilities_action_only(df, reward_col, num_actions, action_col, lam=1, reward_values=None):
    """Estimate categorical reward distributions per action.

    Args:
        df: Processed DataFrame with reward observations.
        reward_col: Reward column to model.
        num_actions: Number of discrete actions.
        action_col: Action index column name.
        lam: Smoothing strength.
        reward_values: Optional explicit support values.

    Returns:
        Array of shape (num_actions, num_reward_values).
    """
    if reward_col != 'PAY_next':
        df = df[df['completed'] == 1]

    if reward_values is None:
        reward_values = np.arange(int(df[reward_col].min()), int(df[reward_col].max()) + 1)
    
    reward_values = list(reward_values)
    
    counts_prior = df[reward_col].value_counts().reindex(reward_values, fill_value=0)
    prior = (counts_prior / counts_prior.sum()).values    

    counts = (df.groupby(action_col)[reward_col]
              .value_counts()
              .unstack(fill_value=0)
              .reindex(columns=reward_values, fill_value=0)
              .reindex(range(num_actions), fill_value=0)
              .values
              .reshape(num_actions, len(reward_values)))
    
    N = counts.sum(axis=-1, keepdims=True)
    smoothed = (counts + lam * prior) / (N + lam)

    return smoothed

def compute_reward_probabilities_global(df, reward_col, reward_values=None):
    """Estimate a global categorical reward distribution.

    Args:
        df: Processed DataFrame with reward observations.
        reward_col: Reward column to model.
        reward_values: Optional explicit support values.

    Returns:
        Array of shape (num_reward_values,).
    """
    if reward_col != 'PAY_next':
        df = df[df['completed'] == 1]

    if reward_values is None:
        reward_values = np.arange(int(df[reward_col].min()), int(df[reward_col].max()) + 1)
    reward_values = list(reward_values)
    
    global_probs = df[reward_col].value_counts().reindex(reward_values, fill_value=0)
    global_p = global_probs / global_probs.sum()
    return global_p.values
