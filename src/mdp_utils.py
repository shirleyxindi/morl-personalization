import numpy as np
import pandas as pd

def compute_return_probabilities(df, rw_col='PAY'):
    # Compute return probabilities for each state.
    # The probabilty they would have returned to the app in the current state
    # Returns array of shape (num_states,) with values in [0,1]
    df['prob_return'] = (df[rw_col] - 1) / 6.0
    return df.groupby('s_idx')['prob_return'].mean().values

def compute_completion_probabilities(df, num_states, num_actions, action_col='joint_cluster', alpha=None):
    # Computes probability of completing action a in state s: P_comp(s,a) = P(completion | s,a) for each state-action pair, with optional Laplace smoothing.
    alpha = 1 / num_states if alpha is None else alpha

    group = df.groupby(['s_idx', action_col])
    success_counts = group['completed'].sum()
    total_counts = group['completed'].count()

    completion_probs = (success_counts + alpha) / (total_counts + alpha * 2)
    base_val = df['completed'].mean()
    new_index = pd.MultiIndex.from_product([range(num_states), range(num_actions)])
    final_matrix = completion_probs.reindex(new_index, fill_value=base_val)

    return final_matrix.values.reshape(num_states, num_actions)
    
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
        if reward_col == 'r_diversity':  # diversity reward is deterministic given count state
            for c in range(nC):
                for a in range(nA):
                    if mapping is not None:
                        reward = mapping.loc[a]['a_novelty']
                        reward_matrix[:, c, a, o_idx] = reward 
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

def compute_rewards_global(df, obj_cols):
    global_rewards = np.zeros(len(obj_cols))
    for o_idx, reward_col in enumerate(obj_cols):
        global_rewards[o_idx] = df[reward_col].mean()
    return global_rewards

def compute_transition_probabilities(df, num_states, num_actions, action_col, alpha=None):
    alpha = 1 / num_states if alpha is None else alpha

    counts = df.groupby(['s_idx', action_col, 'sp_idx']).size().unstack(fill_value=0)

    all_states = list(range(num_states))
    all_actions = list(range(num_actions))
    new_index = pd.MultiIndex.from_product([all_states, all_actions], names=['s_idx', action_col])

    # Fill missing counts with 0
    counts = counts.reindex(new_index, fill_value=0)
    counts = counts.reindex(columns=all_states, fill_value=0)
    
    # Apply Laplace smoothing and normalize
    counts_smoothed = counts + alpha
    transition_probs = counts_smoothed.div(counts_smoothed.sum(axis=1), axis=0).values
    
    # reshape to (num_states, num_actions, num_states)
    return transition_probs.reshape(num_states, num_actions, num_states)

    
def compute_reward_probabilities(df, reward_col, num_states, num_actions, action_col, lam=1, reward_values=None):
    # global_probs = df[reward_col].value_counts(normalize=True).sort_index()
    if reward_values is None:
        reward_values = np.arange(int(df[reward_col].min()), int(df[reward_col].max()) + 1)
    
    reward_values = list(reward_values)
    global_p = compute_reward_probabilities_action_only(df, reward_col, num_actions, action_col, lam=lam, reward_values=reward_values)
    
    # global_p = global_probs.reindex(reward_values, fill_value=0).values
    
    # counts per (s,a) pair for each reward value
    counts = df.groupby(['s_idx', action_col])[reward_col].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=reward_values, fill_value=0)
    
    # ensure all (s,a) pairs are represented, even if they have zero counts
    all_states = range(num_states)
    all_actions = range(num_actions)
    full_index = pd.MultiIndex.from_product([all_states, all_actions], names=['s_idx', action_col])
    counts = counts.reindex(full_index, fill_value=0)
    counts = counts.values.reshape(
        num_states,
        num_actions,
        len(reward_values)
    )
    
    # total counts per (s,a) pair
    N = counts.sum(axis=1)

    # # apply smoothing
    # smoothed = (counts + lam * global_p).div(N + lam, axis=0).values
    # probs = smoothed.reshape(num_states, num_actions, len(reward_values))
    prior_counts = lam * global_p[np.newaxis,:, :]  # broadcast over states
    # BAYESIAN UPDATE
    posterior_counts = counts + prior_counts

    probs = posterior_counts / posterior_counts.sum(axis=2, keepdims=True)
    # probs = probs.values.reshape(num_states, num_actions, len(reward_values))
    return probs

def compute_reward_probabilities_action_only(df, reward_col, num_actions, action_col, lam=1, reward_values=None):
    global_probs = df[reward_col].value_counts(normalize=True).sort_index()
    if reward_values is None:
        reward_values = np.arange(int(df[reward_col].min()), int(df[reward_col].max()) + 1)
    
    reward_values = list(reward_values)
    
    global_p = global_probs.reindex(reward_values, fill_value=0).values
    
    counts = df.groupby(action_col)[reward_col].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=reward_values, fill_value=0)
    
    all_actions = range(num_actions)
    counts = counts.reindex(all_actions, fill_value=0)
    
    N = counts.sum(axis=1)

    smoothed = (counts + lam * global_p).div(N + lam, axis=0).values
    return smoothed.reshape(num_actions, len(reward_values))

def compute_reward_probabilities_global(df, reward_col, reward_values=None):
    global_probs = df[reward_col].value_counts(normalize=True).sort_index()
    if reward_values is None:
        reward_values = np.arange(int(df[reward_col].min()), int(df[reward_col].max()) + 1)
    
    reward_values = list(reward_values)
    global_p = global_probs.reindex(reward_values, fill_value=0).values
    return global_p.reshape(len(reward_values))

def get_time_params(df, num_states, num_actions, action_col, reward_col='time_spent'):
    mean_per_s_a = df.groupby(['s_idx', action_col])[reward_col].mean()
    std_per_s_a = df.groupby(['s_idx', action_col])[reward_col].std()
    all_states = range(num_states)
    all_actions = range(num_actions)
    index = pd.MultiIndex.from_product([all_states, all_actions], names=['s_idx', action_col])
    mean_per_s_a = mean_per_s_a.reindex(index, fill_value=0).values.reshape(num_states, num_actions)
    std_per_s_a = std_per_s_a.reindex(index, fill_value=0).values.reshape(num_states, num_actions)
    return mean_per_s_a, std_per_s_a

def get_lognormal_params(mean, std):
    # Convert mean and std to parameters of a log-normal distribution
    variance = std ** 2
    sigma = np.sqrt(np.log(1 + variance / mean ** 2))
    mu = np.log(mean) - (sigma ** 2) / 2
    return mu, sigma

def sample_time(mean, std):
    mu, sigma = get_lognormal_params(mean, std)
    return np.random.lognormal(mu, sigma)