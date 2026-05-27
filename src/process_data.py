import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import utils

def center_and_scale(df, columns, target_bound=0.5):
    """
    Centers the mean at 0 and scales values to fit within [-target_bound, target_bound].
    """
    df_new = df.copy()
    for col in columns:
        centered = df_new[col] - df_new[col].mean()
        max_deviation = centered.abs().max()
        
        if max_deviation != 0:
            df_new[col] = (centered / max_deviation) * target_bound
        else:
            df_new[col] = 0.0 # Handle case where all values are identical
            
    return df_new

def bin_around_center(df, feature, use_median=True):
    center = df[feature].median() if use_median else df[feature].mean()
    conditions = [
        (df[feature] < center),  # Below -> 0
        (df[feature] == center), # At -> 1
        (df[feature] > center)   # Above -> 2
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices)

def create_state_representations(df, state_features, target_col='s_current', num_vals_per_feature=3, user_col='user_id'):
    df = df.copy()
    
    # Handle different num_bins formats
    if isinstance(num_vals_per_feature, dict):
        num_vals_list = [num_vals_per_feature[feat] for feat in state_features]
    elif isinstance(num_vals_per_feature, int):
        num_vals_list = [num_vals_per_feature] * len(state_features)
    else:
        num_vals_list = num_vals_per_feature
    
    # Create binned versions of each feature
    binned_cols = []
    for feat, bins in zip(state_features, num_vals_list):
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        col_name = f'{feat}_binned'
        if bins == 3 and feat == 'TIR':
            df[col_name] = bin_around_center(df, feat, use_median=True)            
        else:
            df[col_name] = pd.qcut(df[feat], q=int(bins), labels=False, duplicates='drop')
        binned_cols.append(col_name)
    
    df[target_col] = df[binned_cols].values.tolist()
    df['s_next'] = df.groupby(user_col)[binned_cols].shift(-1).values.tolist()
    
    # Clean up temporary binned columns 
    df.drop(columns=binned_cols, inplace=True)
    
    return df

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

def cluster_actions(action_data, cluster_vars, num_clusters=5, cluster_col='cluster_all'):
    action_data = action_data.copy()
    cluster_models = {}

    scaler = StandardScaler()
    action_data[cluster_vars] = scaler.fit_transform(action_data[cluster_vars])

    # Cluster per variable
    for col in cluster_vars + ['time_spent']:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        action_data[f'{col}_cluster'] = kmeans.fit_predict(action_data[[col]].values)
        cluster_models[col] = kmeans

    # Cluster on all variables together
    kmeans_all = KMeans(n_clusters=num_clusters, random_state=42)
    action_data[cluster_col] = kmeans_all.fit_predict(action_data[cluster_vars].values)
    cluster_models['all'] = kmeans_all

    # categories = action_data['category'].unique()
    # for i in range(len(categories)):
    #     category = categories[i]
    #     category_data = action_data[action_data['category'] == category]
    #     kmeans = KMeans(n_clusters=2, random_state=42)
    #     # X_scaled = scaler.fit_transform(category_data[cluster_vars])
    #     category_data[cluster_col] = kmeans.fit_predict(category_data[cluster_vars])
    #     action_data.loc[action_data['category'] == category, cluster_col] = (i*2) + category_data[cluster_col]
    #     print(action_data[action_data['category'] == category][cluster_col].value_counts())
        
    cluster_cols = [f'{col}_cluster' for col in cluster_vars] + [cluster_col]
    return action_data, cluster_models, cluster_cols

def get_joint_cluster(df, cluster_cols, joint_col='joint_cluster'):
    df = df.copy()
    
    grouped = df.groupby(cluster_cols, sort=True)
    df[joint_col] = grouped.ngroup()
    
    mapping = (
        df[[joint_col] + cluster_cols]
        .drop_duplicates()
        .set_index(joint_col)
        .sort_index()
    )
    
    return df, mapping

def create_count_states(df, count_col='cluster_all', max_count=2):
    df = df.copy()
    clusters = sorted(df[count_col].unique())
    
    # cluster_to_idx = {name: i for i, name in enumerate(clusters)}
    cat_cols = []
    for c in clusters:
        col_name = f"cat_{c}"
        df[col_name] = ((df[count_col] == c) & df["completed"]).astype(int)
        cat_cols.append(col_name)

    df[cat_cols] = df.groupby("user_id")[cat_cols].cumsum()
    df["count"] = df[cat_cols].values.tolist()

    def calculate_reward(row):
        idx = row[count_col]
        current_count = row['s_count'][idx]
        return 1 - (1 / (max_count + 1)) * current_count * row['completed']
    
    def get_novelty_feature(row):
        # 1 if the challenge was from least exercised category for that user
        idx = row[count_col]
        counts = row['count']
        if counts[idx] == min(counts) or (len(set(counts)) == 1):
            return 1
        else:
            return 0
    
    df['a_novelty'] = df.apply(get_novelty_feature, axis=1)

    df['s_count_next'] = df['count'].apply(lambda x: [min(count - min(x), max_count) for count in x])
    df['s_count'] = df.groupby('user_id')['s_count_next'].shift(1, fill_value=[0]*len(clusters))
    df['r_diversity'] = df['a_novelty']

    df.drop(cat_cols, axis=1, inplace=True)
    
    return df

def get_state_indices(df, num_vals_per_feature):
    """
    Maps state tuples to a unique integer index (MDP state ID).
    Equivalent to utils.user_state_to_idx.
    """
    df = df.copy()
    df['s_idx'] = df['s_current'].apply(lambda s: utils.user_state_to_idx(tuple(s), num_vals_per_feature))
    df['sp_idx'] = df['s_next'].apply(lambda sp: utils.user_state_to_idx(tuple(sp), num_vals_per_feature))
    return df

def get_time_rewards(df, num_vals_per_feature, time_state_idx=1, time_col='time_spent'):
    df = df.copy()
    completed_times = df[df['completed'] == 1][time_col]
    num_bins = num_vals_per_feature[time_state_idx]
    
    # 1. Create bins and calculate the mean of each bin
    # 'pd.qcut' creates bins with an equal number of samples
    df_temp = pd.DataFrame({time_col: completed_times})
    df_temp['bin'] = pd.qcut(df_temp[time_col], q=num_bins, labels=False)
    
    # Get the mean time for each bin (this is your 'gold standard' for each state)
    bin_means = df_temp.groupby('bin')[time_col].mean().to_dict()
    # round up
    bin_means = {k: np.ceil(v) for k, v in bin_means.items()}
    
    print("Reference Means per Time Bin:")
    for b, m in bin_means.items():
        print(f"Bin {b} Mean: {m:.4f} minutes")
    percentiles = np.linspace(0, 100, num_bins + 1)[1:]
    time_tolerances = [np.percentile(completed_times, p) for p in percentiles]
    print("Time rating percentiles:")
    for p, val in zip(percentiles, time_tolerances):
        print(f"{p}th percentile: {val:.4f}")

    def get_time_reward(obs_time, time_state, time_tolerances):
        overtime = max(0, obs_time - time_tolerances[time_state])
        return max(0, 1 - overtime / time_tolerances[time_state])
    
    def calculate_reward(obs_time, time_state, bin_means):
        # Retrieve the average time for this specific state
        target_time = bin_means.get(time_state, np.mean(list(bin_means.values())))
        
        # Calculate deviation: (Target - Observed) / Target
        # Positive if faster than mean, negative if slower
        deviation = (target_time - obs_time) / target_time
        
        # Scale and clip to [-0.5, 0.5]
        # We clip at -1.0 before multiplying by 0.5 so the worst penalty is -0.5
        return np.clip(deviation, -1.0, 1.0) * 0.5

    df['r_time'] = df.apply(lambda row: get_time_reward(row[time_col], row['s_current'][time_state_idx], bin_means), axis=1)
    return df

def get_rewards(df, scale=(-0.5, 0.5), reward_cols=['likedness', 'usefulness', 'difficulty']):
    df = df.copy()
    df[reward_cols] = df[reward_cols] + 1
    df[reward_cols] = df[reward_cols].fillna(0)
    # df['r_time'] = scale_rewards(df['time_spent'].values, reverse=True) * df['completed']
    # df['r_difficulty'] = scale_rewards(df['difficulty'].values)
    # df['r_likedness'] = scale_rewards(df['likedness'].values)
    # df['r_usefulness'] = scale_rewards(df['usefulness'].values)

    df['r_difficulty'] = df.apply(lambda row: (11 - row['difficulty']) / 11 if row['completed'] == 1 else 0, axis=1)
    df['r_likedness'] = df.apply(lambda row: (row['likedness'] + 1) / 11 if row['completed'] == 1 else 0, axis=1)
    df['r_usefulness'] = df.apply(lambda row: (row['usefulness']) / 7 if row['completed'] == 1 else 0, axis=1)
    df['r_return'] = df.apply(lambda row: row['PAY_next'] / 7, axis=1)
    df['r_expert'] = df['completed'] * df['expert_score']
    
    return df


def process_samples(df, actions_clustered, state_features, num_vals_per_feature, action_col='joint_cluster', cluster_col='cluster_all', max_count=2, verbose=False):
    '''
    Processes dataframe with (s, a, s', r) samples to df with ((u,c), c(a), (u',c'), r) samples
    '''
    df = df.copy()
    df = create_state_representations(df, state_features, num_vals_per_feature=num_vals_per_feature)
    # df.drop(state_features, axis=1, inplace=True)

    num_features = len(state_features)
    num_user_states = np.prod(num_vals_per_feature)

    initial_states = df.groupby('user_id')['s_current'].first().values.tolist()
    initial_states_idx = [utils.user_state_to_idx(state, num_vals_per_feature) for state in initial_states]
    initial_distribution = np.zeros(num_user_states)
    for idx in initial_states_idx:
        initial_distribution[idx] += 1
    initial_distribution /= initial_distribution.sum()

    df = df[~df['s_next'].apply(lambda x: any(pd.isna(i) for i in x))]
    df['s_next'] = df['s_next'].apply(lambda x: [int(i) for i in x])

    df = df.merge(actions_clustered[['challenge_id'] + [cluster_col]], on='challenge_id', how='left')

    df = create_count_states(df, count_col='category_id', max_count=max_count)

    df['c_idx'] = df['s_count'].apply(lambda c: utils.count_state_to_idx(tuple(c), max_count))
    df = get_state_indices(df, num_vals_per_feature)
    df = get_rewards(df)
    df, mapping = get_joint_cluster(df, ['a_novelty', cluster_col], joint_col=action_col)
    # df = get_time_rewards(df, num_vals_per_feature, time_state_idx=state_features.index('TIME_Q'), time_col='time_spent')
    if verbose:
        print("Number of samples after processing:", len(df))
    
    return df, initial_distribution, mapping


