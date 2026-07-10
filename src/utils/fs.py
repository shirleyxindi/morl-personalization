"""
Feature and binning selection helpers.

Author: Shirley Li
Date: July 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from . import process_data as process_data
from . import mdp_utils as mdp_utils
from . import utils as utils
from morld.mo_pi import MOPolicyIteration

def feature_selection_with_fixed_multiple_weights(df, actions_clustered, state_features, fixed_features, reward_cols, 
                                 cluster_col, action_col, weights_list, num_act, num_vals_per_selected_feature, 
                                 discount_factor=0.7, scalarization='linear'):
    """Select features while forcing a fixed subset and averaging across weights.

    Args:
        df: Raw interaction samples.
        actions_clustered: Action metadata with cluster assignments.
        state_features: Candidate state feature names.
        fixed_features: Feature names that must be included.
        reward_cols: Reward columns used to build the MDP reward tensor.
        cluster_col: Column containing the action cluster label.
        action_col: Column used as the joint action index.
        weights_list: Weight vectors used to repeat the MDP solve.
        num_act: Number of actions in the processed MDP.
        num_vals_per_selected_feature: Number of bins for each selected feature.
        discount_factor: Discount factor for policy iteration.
        scalarization: Scalarization type passed to the solver.

    Returns:
        List of selected feature names, including any fixed features.
    """
    num_objectives = len(reward_cols)
    num_possible_features = len(state_features)
    
    selected_indices = [state_features.index(f) for f in fixed_features] if fixed_features else []
    
    num_already_fixed = len(selected_indices)
    total_to_select = len(num_vals_per_selected_feature)
    remaining_slots = total_to_select - num_already_fixed

    for i in range(remaining_slots):
        candidate_indices = [f for f in range(num_possible_features) if f not in selected_indices]
        avg_f_stats = []
        avg_p_vals = []
        
        for f_idx in candidate_indices:
            # current test set = fixed features + previously selected + this candidate
            current_test_indices = selected_indices + [f_idx]
            current_test_set = [state_features[idx] for idx in current_test_indices]
            
            # find the number of bins for each feature in the current test set
            current_num_vals = num_vals_per_selected_feature[:num_already_fixed + i + 1]

            nS = np.prod(current_num_vals)
            state_to_idx, idx_to_state = utils.build_state_space(current_num_vals)
            
            # process data with the combined feature set
            temp_df, _, _, mapping = process_data.process_samples(df, actions_clustered, current_test_set, 
                                                         current_num_vals, action_col, cluster_col)
            
            # MOMDP components
            R = mdp_utils.compute_rewards(temp_df, nS, num_act, num_objectives, reward_cols, action_col, mapping)
            P = mdp_utils.compute_transition_probabilities(temp_df, nS, num_act, action_col)
        
            f_stats_for_candidate = []
            p_vals_for_candidate = []
            
            # solve MOMDP for each weight vector
            for weights in weights_list:
                solver = MOPolicyIteration(P, R, weights, gamma=discount_factor, scalarization=scalarization)
                Q_vals = solver.get_optimal_Q()
                
                q_groups = []
                candidate_val_range = current_num_vals[-1]  
                
                for val in range(candidate_val_range):
                    indices = [s_idx for s_idx in range(nS) if idx_to_state[s_idx][-1] == val]
                    if len(indices) > 0:
                        q_groups.append(Q_vals[indices, :].flatten())
                
                # ANOVA test on the Q-values for each feature in the current test set
                if len(q_groups) > 1:
                    f_stat, p_value = stats.f_oneway(*q_groups)
                    f_stats_for_candidate.append(f_stat)
                    p_vals_for_candidate.append(p_value)
                else:
                    f_stats_for_candidate.append(0.0)
                    p_vals_for_candidate.append(1.0)
            
            avg_f_stat = np.mean(f_stats_for_candidate)
            avg_p_val = np.mean(p_vals_for_candidate)
            
            avg_f_stats.append(avg_f_stat)
            avg_p_vals.append(avg_p_val)
        
        print(f"Candidate avg p-values: {avg_p_vals}")
        print(f"Candidate avg F-stats: {avg_f_stats}")
        
        if np.all(np.array(avg_p_vals) == avg_p_vals[0]):
            best_candidate_idx = candidate_indices[np.argmax(avg_f_stats)]
        else:
            best_candidate_idx = candidate_indices[np.argmin(avg_p_vals)] 
        selected_indices.append(best_candidate_idx)
        print(f"Added: {state_features[best_candidate_idx]} (Avg p-value: {min(avg_p_vals)})")

    return [state_features[idx] for idx in selected_indices]


def bin_selection_manual_combinations(df, actions_clustered, selected_features, reward_cols,
                                      cluster_col, action_col, num_act, binning_combinations,
                                      weights_list, discount_factor=0.7, scalarization='linear'):
    """Evaluate manual binning configurations across multiple weight vectors.

    Args:
        df: Raw interaction samples.
        actions_clustered: Action metadata with cluster assignments.
        selected_features: Feature names to bin and evaluate.
        reward_cols: Reward columns used to build the MDP reward tensor.
        cluster_col: Column containing the action cluster label.
        action_col: Column used as the joint action index.
        num_act: Number of actions in the processed MDP.
        binning_combinations: Iterable of binning configurations.
        weights_list: Weight vectors used to repeat the MDP solve.
        discount_factor: Discount factor for policy iteration.
        scalarization: Scalarization type passed to the solver.

    Returns:
        DataFrame summarizing the binning combinations.
    """

    rows = []
    cluster_size = num_act / 2
    
    # for each binning combination, compute the mean p-value and returns across all weight vectors
    for binning_config in binning_combinations:
        print(binning_config)
        current_num_vals = list(binning_config.values()) if isinstance(binning_config, dict) else list(binning_config)
        state_to_idx, idx_to_state = utils.build_state_space(current_num_vals)
        binning_dict = dict(zip(selected_features, current_num_vals))
        n_u = np.prod(current_num_vals)

        temp_df, _, temp_dist, mapping = process_data.process_samples(
            df, actions_clustered, selected_features, current_num_vals, action_col, cluster_col
        )

        min_samples_per_pair = temp_df.groupby(['s_idx', action_col]).size().min()

        R = mdp_utils.compute_rewards(temp_df, n_u, num_act,
                                      len(reward_cols), reward_cols, action_col, mapping)
        P = mdp_utils.compute_transition_probabilities(temp_df, n_u, num_act, action_col)

        weight_p_vals = []
        weight_returns = []

        for w in weights_list:

            solver = MOPolicyIteration(P, R, w, gamma=discount_factor,
                                      scalarization=scalarization,
                                      initial_distribution=temp_dist)

            Q_vals = solver.get_optimal_Q()
            weight_returns.append(solver.expected_return)

            feature_p_vals = []
            for f_idx, feat in enumerate(selected_features):
                n_bins = current_num_vals[f_idx]
                groups = []
                for b in range(n_bins):
                    idx = [u for u in range(n_u)
                           if idx_to_state[u][f_idx] == b]
                    if idx:
                        groups.append(Q_vals[idx].flatten())

                if len(groups) > 1:
                    _, p = stats.f_oneway(*groups)
                    feature_p_vals.append(1.0 if np.isnan(p) else p)
                else:
                    feature_p_vals.append(1.0)

            weight_p_vals.append(np.mean(feature_p_vals))

        print(f"Avg p-value: {np.mean(weight_p_vals)}, Avg return: {np.mean(weight_returns)}, Min samples per (s,a): {min_samples_per_pair}")
        rows.append({
            "binning_combination": binning_dict,
            "cluster_size": cluster_size,
            "min_samples_per_s_a": min_samples_per_pair,
            "mean_p_value": np.mean(weight_p_vals),
            "std_p_value": np.std(weight_p_vals),
            "mean_return": np.mean(weight_returns)
        })

    return pd.DataFrame(rows)


