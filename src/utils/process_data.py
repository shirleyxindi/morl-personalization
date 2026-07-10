"""
Helpers for processing raw interaction data into usable (s, a, s', r) samples for (MO)RL.

Author: Shirley Li
Date: July 2026
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from . import utils as utils

def bin_around_center(df, feature, use_median=True):
    """Bin a feature into below-center, at-center, and above-center levels.

    Args:
        df: Input DataFrame.
        feature: Feature column to bin.
        use_median: If True, use the median as the center; otherwise use the mean.

    Returns:
        Numpy array of bin labels in {0, 1, 2}.
    """
    center = df[feature].median() if use_median else df[feature].mean()
    conditions = [
        (df[feature] < center),  # Below -> 0
        (df[feature] == center), # At -> 1
        (df[feature] > center)   # Above -> 2
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices)

def create_state_representations(df, state_features, target_col='s_current', num_vals_per_feature=3, user_col='user_id'):
    """Bin selected features and create current and next-state columns.

    Args:
        df: Input DataFrame.
        state_features: Features to bin.
        target_col: Column name for the current-state representation.
        num_vals_per_feature: Number of bins per feature.
        user_col: Column used to group trajectories.

    Returns:
        A copy of the DataFrame with `target_col` and `s_next` columns.
    """
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
        if bins == 3:
            df[col_name] = bin_around_center(df, feat, use_median=True)            
        else:
            df[col_name] = pd.qcut(df[feat], q=int(bins), labels=False, duplicates='drop')
        binned_cols.append(col_name)
    
    df[target_col] = df[binned_cols].values.tolist()
    df['s_next'] = df.groupby(user_col)[binned_cols].shift(-1).values.tolist()
    
    # Clean up temporary binned columns 
    df.drop(columns=binned_cols, inplace=True)
    
    return df

def cluster_actions(action_data, cluster_vars, num_clusters=5, cluster_col='cluster_all'):
    """Cluster actions on individual variables and jointly across all variables.

    Args:
        action_data: DataFrame with action features.
        cluster_vars: Columns used for clustering.
        num_clusters: Number of clusters to fit.
        cluster_col: Column name for the joint clustering result.

    Returns:
        Tuple of (clustered DataFrame, fitted models, cluster column names).
    """
    action_data = action_data.copy()
    cluster_models = {}

    scaler = StandardScaler()
    action_data[cluster_vars] = scaler.fit_transform(action_data[cluster_vars])

    # Cluster on all variables together
    kmeans_all = KMeans(n_clusters=num_clusters, random_state=42)
    action_data[cluster_col] = kmeans_all.fit_predict(action_data[cluster_vars].values)
    cluster_models['all'] = kmeans_all
        
    cluster_cols = [f'{col}_cluster' for col in cluster_vars] + [cluster_col]
    return action_data, cluster_models, cluster_cols

def get_joint_cluster(df, cluster_cols, joint_col='joint_cluster'):
    """Create a joint categorical action index from multiple cluster columns.

    Args:
        df: Input DataFrame.
        cluster_cols: Columns combined into the joint cluster.
        joint_col: Output column name for the joint cluster index.

    Returns:
        Tuple of (updated DataFrame, mapping table).
    """
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

def create_count_states(df, count_col='cluster_all'):
    """Track cumulative category counts and derive diversity indicators.

    Args:
        df: Input DataFrame with `user_id`, `completed`, and category labels.
        count_col: Column used to identify categories.

    Returns:
        A copy of the DataFrame with count and diversity features added.
    """
    df = df.copy()
    clusters = sorted(df[count_col].unique())
    
    # cluster_to_idx = {name: i for i, name in enumerate(clusters)}
    cat_cols = []
    for c in clusters:
        col_name = f"cat_{c}"
        df[col_name] = ((df[count_col] == c) & df["completed"]).astype(int)
        cat_cols.append(col_name)

    df[cat_cols] = df.groupby("user_id")[cat_cols].cumsum().groupby(df["user_id"]).shift(fill_value=0)
    df["count"] = df[cat_cols].values.tolist()
    
    def get_novelty_feature(row):
        # 1 if the challenge was from least exercised category for that user
        idx = row[count_col]
        counts = row['count']
        if counts[idx] == min(counts) or (len(set(counts)) == 1):
            return 1
        else:
            return 0
    
    df['a_novelty'] = df.apply(lambda row: get_novelty_feature(row), axis=1)
    df['r_diversity'] = df['a_novelty']

    df.drop(cat_cols, axis=1, inplace=True)
    
    return df

def get_rewards(df, reward_cols=['likedness', 'usefulness', 'difficulty']):
    """Create the reward columns used by the MDP pipeline.

    Args:
        df: Input DataFrame.
        reward_cols: Reward columns expected in the input.

    Returns:
        A copy of the DataFrame with reward features added.
    """
    df = df.copy()
    df[['likedness', 'difficulty']] = df[['likedness', 'difficulty']] + 1
    df[reward_cols] = df[reward_cols].fillna(0)
    df['r_likedness'] = df.apply(lambda row: (row['likedness']) / 11 if row['completed'] == 1 else 0, axis=1)
    df['r_usefulness'] = df.apply(lambda row: (row['usefulness']) / 7 if row['completed'] == 1 else 0, axis=1)
    df['r_return'] = df.apply(lambda row: row['PAY_next'] / 7, axis=1)
    df['r_adherence'] = df['completed'] 
    
    return df

def get_deviation_prob(df, agency_conditions=['w3', 'w4']):
    """Estimate deviation probability in agency conditions.

    Args:
        df: Input DataFrame with `within_condition`, `challenge_id`, and `recommended_challenge_id`.
        agency_conditions: Condition labels considered agency settings.

    Returns:
        Mean deviation rate among completed agency samples.
    """
    df_agency = df[df['within_condition'].isin(agency_conditions)].copy()
    agency_deviations = df_agency.copy()
    agency_deviations['deviated'] = agency_deviations['challenge_id'] != agency_deviations['recommended_challenge_id']
    deviation_prob_among_completed = agency_deviations[agency_deviations['completed'] == 1]['deviated'].mean()
    return deviation_prob_among_completed

def process_samples(df, actions_clustered, state_features, num_vals_per_feature, action_col='joint_cluster', cluster_col='cluster_all', verbose=False, agency_conditions=['w3', 'w4']):
    """Convert raw data into usable (s, a, s', r) samples for (MO)RL.

    Args:
        df: Raw interaction samples.
        actions_clustered: Action metadata with cluster assignments.
        state_features: State features.
        num_vals_per_feature: Number of bins per state feature.
        action_col: Column used as the joint action index.
        cluster_col: Column containing the action cluster label.
        verbose: If True, print processing counts.
        agency_conditions: Condition labels used to separate agency samples.

    Returns:
        Tuple of (processed_df, agency_df, initial_distribution, mapping, deviation_prob).
    """
    df = df.copy()
    df = create_state_representations(df, state_features, num_vals_per_feature=num_vals_per_feature)
    # df.drop(state_features, axis=1, inplace=True)
    state_to_idx, idx_to_state = utils.build_state_space(num_vals_per_feature)

    num_features = len(state_features)
    num_user_states = np.prod(num_vals_per_feature)

    initial_states = df.groupby('user_id')['s_current'].first().values.tolist()
    initial_states_idx = [state_to_idx[tuple(state)] for state in initial_states]
    initial_distribution = np.zeros(num_user_states)
    for idx in initial_states_idx:
        initial_distribution[idx] += 1
    initial_distribution /= initial_distribution.sum()

    df = df[~df['s_next'].apply(lambda x: any(pd.isna(i) for i in x))]
    df['s_next'] = df['s_next'].apply(lambda x: [int(i) for i in x])

    df = df.merge(actions_clustered[['challenge_id'] + [cluster_col]], on='challenge_id', how='left')

    df = create_count_states(df, count_col='category_id')

    df['s_idx'] = df['s_current'].apply(lambda x: state_to_idx[tuple(x)])
    df['sp_idx'] = df['s_next'].apply(lambda x: state_to_idx[tuple(x)])
    df = get_rewards(df)
    df, mapping = get_joint_cluster(df, ['a_novelty', cluster_col], joint_col=action_col)
    # df = get_time_rewards(df, num_vals_per_feature, time_state_idx=state_features.index('TIME_Q'), time_col='time_spent')
    
    agency_df = df[df['within_condition'].isin(agency_conditions)]
    df = df[~df['within_condition'].isin(agency_conditions)].copy()
    
    if verbose:
        print("Number of samples after processing:", len(df))
        print("Number of agency samples:", len(agency_df))
        
    return df, agency_df, initial_distribution, mapping


