import numpy as np
import pandas as pd

def compute_completion_probabilities(df, num_states, num_actions, alpha=None):
    alpha = 1 / num_states if alpha is None else alpha
    total_counts = df.groupby(['s_idx', 'challenge_id'])['completed'].count()
    success_counts = df.groupby(['s_idx', 'challenge_id'])['completed'].sum()
    n_classes = 2

    base_completion = df['completed'].mean()
    completion_probs = (success_counts + alpha) / (total_counts + alpha * n_classes)

    new_index = pd.MultiIndex.from_product([range(num_states), range(num_actions)], names=['s_idx', 'challenge_id'])
    completion_probs = completion_probs.reindex(new_index, fill_value=base_completion)

    return completion_probs.values.reshape(num_states, num_actions)

def compute_completion_probabilities_clustered(df, cluster_col, num_states, num_actions, num_clusters, use_clusters=False, alpha=None):
    alpha = 1 / num_states if alpha is None else alpha
    total_counts = df.groupby(['s_idx', cluster_col])['completed'].count()
    success_counts = df.groupby(['s_idx', cluster_col])['completed'].sum()
    n_classes = 2

    base_completion = df['completed'].mean()
    cluster_probs = (success_counts + alpha) / (total_counts + alpha * n_classes)

    new_index = pd.MultiIndex.from_product([range(num_states), range(num_clusters)], names=['s_idx', cluster_col])
    cluster_probs = cluster_probs.reindex(new_index, fill_value=base_completion)
    cluster_probs = cluster_probs.values.reshape(num_states, num_clusters)

    if use_clusters:
        return cluster_probs
    else:
        action_to_cluster = dict(zip(df['challenge_id'], df[cluster_col]))

        completion_probs_array = np.zeros((num_states, num_actions))
        for a in range(num_actions):
            if a in action_to_cluster:
                cluster_idx = action_to_cluster[a]
                completion_probs_array[:, a] = cluster_probs[:, cluster_idx]

        return completion_probs_array
    
def compute_completion_probabilities_action_only(df, nA, action_col):
    baseline_completion = df['completed'].mean()
    return df.groupby(action_col)['completed'].mean().reindex(range(nA), fill_value=baseline_completion).values

def compute_rewards_clustered(df, completion_probs, clusters, nU, nC, nA, nO, action_categories, num_clusters, count_cluster, use_clusters=False):
    # use_clusters determines whether to use cluster-based rewards or action-based rewards,
    # i.e. whether the reward matrix has shape (nU, nC, num_clusters, nO) or (nU, nC, nA, nO)
    # action_categories maps each action to its count cluster
    # count_cluster is the column name in df that is used to cluster for the count state
    nA = num_clusters if use_clusters else nA
    reward_matrix = np.zeros((nU, nC, nA, nO))

    action_to_clusters = {
        reward_col: dict(zip(df['challenge_id'], df[cluster_col])) 
        for reward_col, cluster_col in clusters.items()
    }

    for o_idx, (reward_col, cluster_col) in enumerate(clusters.items()):
        if reward_col == 'r_diversity':
            # diversity reward is deterministic given cluster and count state
            category_reward = df.groupby(['c_idx', count_cluster])[reward_col].mean()
            if use_clusters:
                for (c, cluster_idx), reward in category_reward.items():
                    reward_matrix[:, c, cluster_idx, o_idx] = reward * completion_probs[:, cluster_idx]
            else:
                # vectorized: build category reward lookup for all (c, a) pairs
                cat_rewards = np.array([
                    [category_reward.get((c, action_categories[a]), 0) for a in range(nA)]
                    for c in range(nC)
                ])  # (nC, nA)
                # completion_probs_array: (nU, nA), cat_rewards: (nC, nA)
                # result: (nU, nC, nA)
                reward_matrix[:, :, :, o_idx] = (
                    completion_probs[:, np.newaxis, :] * cat_rewards[np.newaxis, :, :]
                )
            
        else:
            cluster_reward = df.groupby(['s_idx', cluster_col])[reward_col].mean()
            for (s, cluster_idx), reward in cluster_reward.items():
                if use_clusters:
                    reward_matrix[s, :, cluster_idx, o_idx] = reward
                else:
                    actions = [a for a, cl in action_to_clusters[reward_col].items() if cl == cluster_idx]
                    reward_matrix[s, :, actions, o_idx] = reward

    return reward_matrix

def compute_rewards(df, completion_probs, nU, nC, nA, nO, obj_cols, action_col, mapping=None):
    reward_matrix = np.zeros((nU, nC, nA, nO))
    for o_idx, reward_col in enumerate(obj_cols):
        if reward_col == 'r_diversity':
            for c in range(nC):
                for a in range(nA):
                    if mapping is not None:
                        reward = mapping.loc[a]['a_novelty']
                        reward_matrix[:, c, a, o_idx] = reward * completion_probs[:, a]
                    else:
                        reward_matrix[:, c, a, o_idx] = df[(df[action_col] == a) & (df['c_idx'] == c)][reward_col].mean()
            # cluster_reward = df.groupby(['c_idx', action_col])[reward_col].mean()
            # for (c, a), reward in cluster_reward.items():
            #     reward_matrix[:, c, a, o_idx] = reward 
        else:
            cluster_reward = df.groupby(['s_idx', action_col])[reward_col].mean()
            for (s, a), reward in cluster_reward.items():
                reward_matrix[s, :, a, o_idx] = reward

    return reward_matrix

def compute_avg_rewards(df, nA, action_col, obj_cols):
    rewards_per_action = np.zeros((nA, len(obj_cols)))
    for o_idx, reward_col in enumerate(obj_cols):
        action_rewards = df.groupby(action_col)[reward_col].mean()
        for a in range(nA):
            if a in action_rewards.index:
                rewards_per_action[a, o_idx] = action_rewards.loc[a]
    return rewards_per_action

def compute_transition_probabilities(df, num_states, num_actions, alpha=None):
    alpha = 1 / num_states if alpha is None else alpha

    counts = df.groupby(['s_idx', 'challenge_id', 'sp_idx']).size().unstack(fill_value=0)

    all_states = range(num_states)
    all_actions = range(num_actions)
    new_index = pd.MultiIndex.from_product([all_states, all_actions], names=['s_idx', 'challenge_id'])

    counts = counts.reindex(new_index, fill_value=0)
    counts_smoothed = counts + alpha
    probs = counts_smoothed.div(counts_smoothed.sum(axis=1), axis=0).values
    
    transition_probs_array = probs.reshape(num_states, num_actions, num_states)

    return transition_probs_array

def compute_transition_probabilities_clustered(df, num_states, num_actions, cluster_col, num_clusters, use_clusters=False, alpha=None):
    alpha = 1 / num_states if alpha is None else alpha

    # 1. This creates a DataFrame where columns are 'sp_idx' values present in df
    counts = df.groupby(['s_idx', cluster_col, 'sp_idx']).size().unstack(fill_value=0)

    all_states = list(range(num_states))
    all_clusters = list(range(num_clusters))
    new_index = pd.MultiIndex.from_product([all_states, all_clusters], names=['s_idx', cluster_col])

    # 2. Fill missing ROWS (state-cluster combinations never seen)
    counts = counts.reindex(new_index, fill_value=0)
    
    # 3. CRITICAL: Fill missing COLUMNS (next-states never transitioned into)
    # This ensures you have exactly 'num_states' columns
    counts = counts.reindex(columns=all_states, fill_value=0)
    
    # 4. Apply Laplace smoothing and normalize
    counts_smoothed = counts + alpha
    cluster_probs = counts_smoothed.div(counts_smoothed.sum(axis=1), axis=0).values
    
    # Now the reshape is mathematically guaranteed to work: 
    # (num_states * num_clusters) rows * num_states columns
    P_clustered = cluster_probs.reshape(num_states, len(all_clusters), num_states)
    
    action_to_cluster = df.set_index('challenge_id')[cluster_col].drop_duplicates()

    num_actions = num_clusters if use_clusters else num_actions
    if use_clusters:
        return P_clustered
    else:
        transition_probs_array = np.zeros((num_states, num_actions, num_states))
        for a in range(num_actions):
            if a in action_to_cluster.index:
                cluster = action_to_cluster.loc[a]
                transition_probs_array[:, a, :] = P_clustered[:, cluster, :]

        return transition_probs_array
    
def compute_reward_probabilities(df, reward_col, num_states, num_actions, cluster_col, lam=1, reward_values=None):
    global_probs = df[reward_col].value_counts(normalize=True).sort_index()
    if reward_values is None:
        reward_values = sorted(global_probs.index)
    
    reward_values = list(reward_values)
    
    global_p = global_probs.reindex(reward_values, fill_value=0).values
    
    # Optional: avoid zero-probability prior
    # epsilon = 1e-8
    # global_p = global_p + epsilon
    # global_p = global_p / global_p.sum()
    
    # counts per (s,a) pair for each reward value
    counts = df.groupby(['s_idx', cluster_col])[reward_col].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=reward_values, fill_value=0)
    
    # ensure all (s,a) pairs are represented, even if they have zero counts -> should not be the case though
    all_states = range(num_states)
    all_actions = range(num_actions)
    full_index = pd.MultiIndex.from_product([all_states, all_actions], names=['s_idx', cluster_col])
    counts = counts.reindex(full_index, fill_value=0)
    
    # total counts per (s,a) pair
    N = counts.sum(axis=1)

    # apply smoothing
    smoothed = (counts + lam * global_p).div(N + lam, axis=0).values
    probs = smoothed.reshape(num_states, num_actions, len(reward_values))
    
    return probs