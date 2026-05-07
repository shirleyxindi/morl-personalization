import numpy as np
from scipy import stats
import process_data as process_data
import mdp_utils
import utils
from morld.mo_pi import MOPolicyIteration
from itertools import combinations

def feature_selection(df, actions_clustered, state_features, reward_cols, cluster_col, weights, num_act, 
                             num_vals_per_selected_feature, max_count=2, discount_factor=0.7, scalarization='linear'):
    """
    Feature selection for MORL using Policy Iteration.
    """
    num_objectives = len(reward_cols)
    selected_indices = []
    num_possible_features = len(state_features)
    num_to_select = len(num_vals_per_selected_feature)
    n_c = (max_count + 1) ** num_act 
    
    for i in range(num_to_select):
        candidate_indices = [f for f in range(num_possible_features) if f not in selected_indices]
        f_stats = []
        p_values = []
        
        for f_idx in candidate_indices:
            current_test_set = [state_features[i] for i in selected_indices + [f_idx]]
            current_num_vals = [num_vals_per_selected_feature[j] for j in range(i + 1)]
            # In your logic: n_u is the number of combinations of feature values
            n_u = np.prod(current_num_vals)
            # 1. Process data into indices
            temp_df, _ = process_data.process_samples(df, actions_clustered, current_test_set, current_num_vals, cluster_col, max_count, num_act)
            
            P_comp = mdp_utils.compute_completion_probabilities_clustered(temp_df, cluster_col, n_u, num_act, num_clusters=num_act, use_clusters=True)
            R = mdp_utils.compute_rewards(temp_df, P_comp, n_u, n_c, num_act, num_objectives, reward_cols, cluster_col)
            P = mdp_utils.compute_transition_probabilities_clustered(temp_df, n_u, num_act, cluster_col, num_clusters=num_act, use_clusters=True)
            
            # 3. Solve using Policy Iteration logic
            solver = MOPolicyIteration(P, R, P_comp, weights, gamma=discount_factor, scalarization=scalarization)
            Q_vals = solver.get_optimal_Q()
            
            # 4. ANOVA Test on the scalarized Value Function
            # We use V (state value) because it represents the expected utility of the feature
            q_groups = []
            candidate_val_range = num_vals_per_selected_feature[i]
            
            for val in range(candidate_val_range):
                # Identify indices where the last feature in the tuple has value 'val'
                indices = []
                for u_idx in range(n_u):
                    # Use your provided conversion logic
                    u_state = utils.idx_to_user_state(u_idx, current_num_vals)
                    if u_state[-1] == val: # candidate is always the last added
                        indices.append(u_idx)
                
                # Flatten Q-values for all actions and count-states in these user-states
                q_groups.append(Q_vals[indices, :, :].flatten())
            f_stat, p_val = stats.f_oneway(*q_groups)
            f_stats.append(f_stat)
            p_values.append(p_val if not np.isnan(p_val) else 1.0)
            print_q_stats(q_groups)
            
        best_idx = np.argmax(f_stats)
        print(p_values)
        selected_indices.append(candidate_indices[best_idx])
        print(f"Slot {i+1}: Selected {state_features[selected_indices[-1]]} "
      f"(F-stat: {f_stats[best_idx]:.2f}, p-val: {p_values[best_idx]})")

    return selected_indices

def feature_selection_with_fixed_multiple_weights(df, actions_clustered, state_features, fixed_features, reward_cols, 
                                 cluster_col, weights_list, num_act, num_vals_per_selected_feature, 
                                 max_count=2, discount_factor=0.7, scalarization='linear'):
    """
    fixed_features: List of strings (names of features you definitely want).
    num_vals_per_selected_feature: A list specifying the number of bins for each selected feature.
    weights_list: List of weight vectors for multi-objective optimization. 
                  F-statistics and p-values will be averaged across all weight vectors.
    """
    num_objectives = len(reward_cols)
    num_possible_features = len(state_features)
    
    # 1. Initialize with fixed features
    selected_indices = [state_features.index(f) for f in fixed_features]
    
    # Calculate how many MORE features we need to find
    num_already_fixed = len(selected_indices)
    total_to_select = len(num_vals_per_selected_feature)
    remaining_slots = total_to_select - num_already_fixed
    n_c = (max_count + 1) ** num_act 

    for i in range(remaining_slots):
        candidate_indices = [f for f in range(num_possible_features) if f not in selected_indices]
        avg_f_stats = []
        avg_p_vals = []
        
        for f_idx in candidate_indices:
            # Current test set = fixed features + previously selected + this candidate
            current_test_indices = selected_indices + [f_idx]
            current_test_set = [state_features[idx] for idx in current_test_indices]
            
            # Map the number of bins for each feature in the current test set
            current_num_vals = num_vals_per_selected_feature[:num_already_fixed + i + 1]
            n_u = np.prod(current_num_vals)
            print(current_test_set, current_num_vals)
            
            # 2. Process data with the combined feature set
            temp_df, _ = process_data.process_samples(df, actions_clustered, current_test_set, 
                                                         current_num_vals, cluster_col, max_count, num_act)
            
            # Compute MDP components (same for all weight vectors)
            P_comp = mdp_utils.compute_completion_probabilities_clustered(temp_df, cluster_col, n_u, num_act, num_act, True)
            R = mdp_utils.compute_rewards(temp_df, P_comp, n_u, n_c, num_act, num_objectives, reward_cols, cluster_col)
            P = mdp_utils.compute_transition_probabilities_clustered(temp_df, n_u, num_act, cluster_col, num_act, True)
            
            # 3. Solve MDP for each weight vector and collect statistics
            f_stats_for_candidate = []
            p_vals_for_candidate = []
            
            for weights in weights_list:
                # Solve MDP with current weight vector
                solver = MOPolicyIteration(P, R, P_comp, weights, gamma=discount_factor, scalarization=scalarization, max_count=max_count)
                Q_vals = solver.get_optimal_Q()
                
                # ANOVA Test on the Candidate (always the last index in the tuple)
                q_groups = []
                candidate_val_range = current_num_vals[-1]  # The bin count of the candidate
                
                for val in range(candidate_val_range):
                    indices = [u_idx for u_idx in range(n_u) if utils.idx_to_user_state(u_idx, current_num_vals)[-1] == val]
                    if len(indices) > 0:
                        q_groups.append(Q_vals[indices, :, :].flatten())
                
                if len(q_groups) > 1:
                    f_stat, p_value = stats.f_oneway(*q_groups)
                    f_stats_for_candidate.append(f_stat)
                    p_vals_for_candidate.append(p_value)
                else:
                    f_stats_for_candidate.append(0.0)
                    p_vals_for_candidate.append(1.0)
            
            # 4. Average F-statistics and p-values across all weight vectors
            avg_f_stat = np.mean(f_stats_for_candidate)
            avg_p_val = np.mean(p_vals_for_candidate)
            
            avg_f_stats.append(avg_f_stat)
            avg_p_vals.append(avg_p_val)
        
        print(f"Candidate avg p-values: {avg_p_vals}")
        print(f"Candidate avg F-stats: {avg_f_stats}")
        
        # Select best candidate based on average F-statistic if p-vals are all the same
        if np.all(np.array(avg_p_vals) == avg_p_vals[0]):
            best_candidate_idx = candidate_indices[np.argmax(avg_f_stats)]
        else:
            best_candidate_idx = candidate_indices[np.argmin(avg_p_vals)] 
        selected_indices.append(best_candidate_idx)
        print(f"Added: {state_features[best_candidate_idx]} (Avg p-value: {min(avg_p_vals)})")

    return [state_features[idx] for idx in selected_indices]


def feature_selection_with_fixed(df, actions_clustered, state_features, fixed_features, reward_cols, 
                                 cluster_col, weights, num_act, num_vals_per_selected_feature, 
                                 max_count=2, discount_factor=0.7, scalarization='linear'):
    """
    fixed_features: List of strings (names of features you definitely want).
    num_vals_per_feature_map: A dictionary {feature_name: num_bins} to lookup bin counts.
    total_to_select: Total number of features you want in the final set (fixed + new).
    """
    num_objectives = len(reward_cols)
    num_possible_features = len(state_features)
    
    # 1. Initialize with fixed features
    selected_indices = [state_features.index(f) for f in fixed_features]
    
    # Calculate how many MORE features we need to find
    num_already_fixed = len(selected_indices)
    total_to_select = len(num_vals_per_selected_feature)
    remaining_slots = total_to_select - num_already_fixed
    n_c = (max_count + 1) ** num_act 

    for i in range(remaining_slots):
        candidate_indices = [f for f in range(num_possible_features) if f not in selected_indices]
        f_stats = []
        p_vals = []
        for f_idx in candidate_indices:
            # Current test set = fixed features + previously selected + this candidate
            current_test_indices = selected_indices + [f_idx]
            current_test_set = [state_features[idx] for idx in current_test_indices]
            
            # Map the number of bins for each feature in the current test set
            current_num_vals = [num_vals_per_selected_feature[j] for j in range(num_already_fixed + i + 1)]
            n_u = np.prod(current_num_vals)
            
            # 2. Process data with the combined feature set
            # Pass current_num_vals (the list of bins) into process_samples
            temp_df, _ = process_data.process_samples(df, actions_clustered, current_test_set, 
                                                         current_num_vals, cluster_col, max_count, num_act)
            
            # Compute MDP components
            P_comp = mdp_utils.compute_completion_probabilities_clustered(temp_df, cluster_col, n_u, num_act, num_act, True)
            R = mdp_utils.compute_rewards(temp_df, P_comp, n_u, n_c, num_act, num_objectives, reward_cols, cluster_col)
            P = mdp_utils.compute_transition_probabilities_clustered(temp_df, n_u, num_act, cluster_col, num_act, True)
            
            # Solve MDP
            solver = MOPolicyIteration(P, R, P_comp, weights, gamma=discount_factor, scalarization=scalarization)
            Q_vals = solver.get_optimal_Q()
            
            # 3. ANOVA Test on the Candidate (always the last index in the tuple)
            q_groups = []
            candidate_val_range = current_num_vals[-1] # The bin count of the candidate
            
            for val in range(candidate_val_range):
                indices = [u_idx for u_idx in range(n_u) if utils.idx_to_user_state(u_idx, current_num_vals)[-1] == val]
                if len(indices) > 0:
                    q_groups.append(Q_vals[indices, :, :].flatten())
            
            if len(q_groups) > 1:
                f_stat, p_value = stats.f_oneway(*q_groups)
                # print_q_stats(q_groups)
                p_vals.append(p_value)
                f_stats.append(f_stat)
            else:
                f_stats.append(0.0)
                p_vals.append(1.0)
            
        print(f"Candidate p-values: {p_vals}")
        print(f"Candidate F-stats: {f_stats}")
        # Select best candidate and add to selected_indices
        best_candidate_idx = candidate_indices[np.argmax(f_stats)]
        selected_indices.append(best_candidate_idx)
        print(f"Added: {state_features[best_candidate_idx]} (F-stat: {max(f_stats):.2f})")


    return [state_features[idx] for idx in selected_indices]

def print_q_stats(q_groups):
    for (i, list_a), (j, list_b) in combinations(enumerate(q_groups), 2):
        diff = np.mean(np.abs(np.array(list_a) - np.array(list_b)))
        
        print(f"Mean difference between Q-values for value {i} and {j}: {diff:.4f}")
def bin_selection_manual_combinations(df, actions_clustered, selected_features, reward_cols, 
                                      cluster_col, num_act, binning_combinations,
                                      max_count=2, discount_factor=0.7, scalarization='linear',
                                      n_weight_samples=10, seed=42, min_samples_per_state=10):
    """
    Select optimal binning configuration from manually specified combinations.
    
    Args:
        df: Dataset
        actions_clustered: Clustered actions
        selected_features: List of feature names in order
        reward_cols: Reward column names
        cluster_col: Cluster column name
        num_act: Number of actions
        binning_combinations: List of binning configurations to test
                             Each config is a list of bin counts matching selected_features order
                             e.g., [[2, 3, 2], [3, 3, 2], [2, 4, 3]]
                             Or dict format: [{'age': 2, 'income': 3, 'engagement': 2}, ...]
        n_weight_samples: Number of random weight vectors to test
        min_samples_per_state: Minimum samples per state to accept a binning
        
    Returns:
        optimal_binning: The best binning configuration
        results: Detailed results for all tested combinations
    """
    num_objectives = len(reward_cols)
    n_c = (max_count + 1) ** num_act
    
    # Generate random weight samples
    np.random.seed(seed)
    weight_samples = np.random.dirichlet(np.ones(num_objectives), size=n_weight_samples)
    
    print(f"Testing {len(binning_combinations)} binning combinations")
    print(f"Features (in order): {selected_features}")
    print(f"Using {n_weight_samples} random weight vectors\n")
    
    # Store results for each combination
    results = {
        'weight_vectors': weight_samples,
        'combinations': [],
        'detailed_results': []
    }
    
    best_score = -np.inf
    best_combination = None
    best_combination_idx = None
    
    for combo_idx, binning_config in enumerate(binning_combinations):
        print(f"\n{'='*70}")
        print(f"COMBINATION {combo_idx + 1}/{len(binning_combinations)}")
        print(f"{'='*70}")
        
        # Convert to list format if dict
        if isinstance(binning_config, dict):
            current_num_vals = [binning_config[feat] for feat in selected_features]
            print(f"Configuration: {binning_config}")
        else:
            current_num_vals = list(binning_config)
            binning_dict = dict(zip(selected_features, current_num_vals))
            print(f"Configuration: {binning_dict}")
        
        n_u = np.prod(current_num_vals)
        print(f"State space size: {n_u}")
        
        # Initialize result storage for this combination
        combo_results = {
            'binning': current_num_vals,
            'binning_dict': binning_dict if isinstance(binning_config, dict) else dict(zip(selected_features, current_num_vals)),
            'n_states': n_u,
            'coverage': None,
            'samples_per_state': None,
            'f_stats_by_weight': [],
            'returns_by_weight': [],
            'f_stats_by_feature': {feat: [] for feat in selected_features},
            'aggregate_score': None,
            'mean_f_stat': None,
            'std_f_stat': None,
            'mean_return': None,
            'std_return': None
        }
        
        # Process data with this binning
        try:
            temp_df, temp_dist = process_data.process_samples(
                df, actions_clustered, selected_features, 
                current_num_vals, cluster_col, max_count, num_act
            )
        except Exception as e:
            print(f"ERROR processing data: {e}")
            combo_results['aggregate_score'] = -np.inf
            results['combinations'].append(current_num_vals)
            results['detailed_results'].append(combo_results)
            continue
        
        # Check data quality
        states_with_data = temp_df['s_idx'].nunique()
        coverage = states_with_data / n_u
        samples_per_state = len(temp_df) / states_with_data if states_with_data > 0 else 0
        
        combo_results['coverage'] = coverage
        combo_results['samples_per_state'] = samples_per_state
        
        print(f"Coverage: {coverage:.2%} ({states_with_data}/{n_u} states)")
        print(f"Samples per state: {samples_per_state:.1f}")
        
        # Skip if data quality is too poor
        if samples_per_state < min_samples_per_state:
            print(f"SKIP: Insufficient samples per state (< {min_samples_per_state})")
            combo_results['aggregate_score'] = -np.inf
            results['combinations'].append(current_num_vals)
            results['detailed_results'].append(combo_results)
            continue
        
        # Compute MDP components (same for all weight samples)
        try:
            P_comp = mdp_utils.compute_completion_probabilities_clustered(
                temp_df, cluster_col, n_u, num_act, num_act, True
            )
            R = mdp_utils.compute_rewards(
                temp_df, P_comp, n_u, n_c, num_act, num_objectives, 
                reward_cols, cluster_col
            )
            P = mdp_utils.compute_transition_probabilities_clustered(
                temp_df, n_u, num_act, cluster_col, num_act, True
            )
        except Exception as e:
            print(f"ERROR computing MDP: {e}")
            combo_results['aggregate_score'] = -np.inf
            results['combinations'].append(current_num_vals)
            results['detailed_results'].append(combo_results)
            continue
        
        # Test across multiple weight samples
        print(f"\nEvaluating across {n_weight_samples} weight vectors...")
        
        for weight_idx, weight_vec in enumerate(weight_samples):
            # Solve MDP
            solver = MOPolicyIteration(
                P, R, P_comp, weight_vec, gamma=discount_factor, 
                scalarization=scalarization, initial_distribution=temp_dist, max_count=max_count
            )
            Q_vals = solver.get_optimal_Q()
            expected_return = solver.expected_return
            
            combo_results['returns_by_weight'].append(expected_return)
            
            # Compute F-statistic for each feature
            feature_f_stats = []
            
            for feature_idx, feature_name in enumerate(selected_features):
                n_bins = current_num_vals[feature_idx]
                q_groups = []
                
                for bin_val in range(n_bins):
                    # Find all states where this feature has value bin_val
                    indices = []
                    for u_idx in range(n_u):
                        u_state = utils.idx_to_user_state(u_idx, current_num_vals)
                        if u_state[feature_idx] == bin_val:
                            indices.append(u_idx)
                    
                    if len(indices) > 0:
                        q_groups.append(Q_vals[indices, :, :].flatten())
                
                # Compute F-statistic for this feature
                if len(q_groups) > 1:
                    f_stat, p_val = stats.f_oneway(*q_groups)
                    f_stat_value = f_stat if not np.isnan(f_stat) else 0.0
                else:
                    f_stat_value = 0.0
                
                feature_f_stats.append(f_stat_value)
                combo_results['f_stats_by_feature'][feature_name].append(f_stat_value)
            
            # Overall F-statistic: mean across all features
            overall_f_stat = np.mean(feature_f_stats)
            combo_results['f_stats_by_weight'].append(overall_f_stat)
        
        # Compute summary statistics
        combo_results['mean_f_stat'] = np.mean(combo_results['f_stats_by_weight'])
        combo_results['std_f_stat'] = np.std(combo_results['f_stats_by_weight'])
        combo_results['mean_return'] = np.mean(combo_results['returns_by_weight'])
        combo_results['std_return'] = np.std(combo_results['returns_by_weight'])
        
        # Compute aggregate score
        # Option 1: Primarily F-statistic with data quality penalties
        coverage_penalty = 1.0 if coverage > 0.7 else (0.8 if coverage > 0.5 else 0.5)
        samples_penalty = 1.0 if samples_per_state > 20 else (0.9 if samples_per_state > 10 else 0.7)
        
        aggregate_score = combo_results['mean_f_stat'] * coverage_penalty * samples_penalty
        
        # Option 2: Combine F-stat and return
        # aggregate_score = (0.7 * combo_results['mean_f_stat'] + 0.3 * combo_results['mean_return']) * coverage_penalty * samples_penalty
        
        # Option 3: Penalize variance (prefer consistent across weights)
        # consistency_bonus = 1.0 / (1.0 + combo_results['std_f_stat'])
        # aggregate_score = combo_results['mean_f_stat'] * consistency_bonus * coverage_penalty * samples_penalty
        
        combo_results['aggregate_score'] = aggregate_score
        
        # Print summary
        print(f"\nResults:")
        print(f"  Mean F-stat: {combo_results['mean_f_stat']:.2f} ± {combo_results['std_f_stat']:.2f}")
        print(f"  Mean return: {combo_results['mean_return']:.4f} ± {combo_results['std_return']:.4f}")
        print(f"  Aggregate score: {aggregate_score:.2f}")
        
        print(f"\n  F-stats by feature (mean across weights):")
        for feature_name in selected_features:
            mean_f = np.mean(combo_results['f_stats_by_feature'][feature_name])
            std_f = np.std(combo_results['f_stats_by_feature'][feature_name])
            print(f"    {feature_name}: {mean_f:.2f} ± {std_f:.2f}")
        
        # Track best combination
        if aggregate_score > best_score:
            best_score = aggregate_score
            best_combination = current_num_vals
            best_combination_idx = combo_idx
            print(f"  ✓ NEW BEST!")
        
        # Store results
        results['combinations'].append(current_num_vals)
        results['detailed_results'].append(combo_results)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS:")
    print(f"{'='*70}")
    
    # Sort combinations by score
    sorted_indices = np.argsort([r['aggregate_score'] for r in results['detailed_results']])[::-1]
    
    print("\nTop 5 configurations:")
    for rank, idx in enumerate(sorted_indices[:5]):
        combo_result = results['detailed_results'][idx]
        print(f"\n{rank + 1}. {combo_result['binning_dict']}")
        print(f"   Score: {combo_result['aggregate_score']:.2f}")
        print(f"   F-stat: {combo_result['mean_f_stat']:.2f}, "
              f"Return: {combo_result['mean_return']:.4f}")
        print(f"   Coverage: {combo_result['coverage']:.2%}, "
              f"Samples/state: {combo_result['samples_per_state']:.1f}")
    
    if best_combination is not None:
        print(f"\n{'='*70}")
        print("OPTIMAL CONFIGURATION:")
        print(f"{'='*70}")
        best_dict = dict(zip(selected_features, best_combination))
        for feature, n_bins in best_dict.items():
            print(f"  {feature}: {n_bins} bins")
        print(f"\nBest score: {best_score:.2f}")
    else:
        print("\n⚠️ WARNING: No valid configuration found!")
    
    return best_combination, results




