import numpy as np
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from moocore import hypervolume, hv_contributions

def select_policies_until_action_constraint(policies, policy_evals, weight_samples, max_actions=3):
    """
    Continues adding policies until no more can be added without 
    exceeding the 3-action-per-state limit.
    """
    num_policies = len(policies)
    selected_indices = []
    current_max_utilities = np.zeros(len(weight_samples))
    
    num_states = len(policies[0]['policy'])
    actions_in_subset = [set() for _ in range(num_states)]
    
    all_utilities = np.dot(weight_samples, policy_evals.T) 

    while True:
        best_gain = -1.0
        best_idx = -1
        
        for i in range(num_policies):
            if i in selected_indices: 
                continue

            violation = False
            for s, action in enumerate(policies[i]['policy']):
                if action not in actions_in_subset[s] and len(actions_in_subset[s]) >= max_actions:
                    violation = True
                    break
            
            if violation: 
                continue

            # Calculate how much utility gain this policy would provide
            potential_max = np.maximum(current_max_utilities, all_utilities[:, i])
            gain = np.mean(potential_max - current_max_utilities)
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        # Break if no policies are left that satisfy the constraint 
        # or if adding more policies provides 0 additional utility
        if best_idx == -1 or best_gain <= 0:
            break

        selected_indices.append(best_idx)
        current_max_utilities = np.maximum(current_max_utilities, all_utilities[:, best_idx])
        
        # Update action tracker
        for s, action in enumerate(policies[best_idx]['policy']):
            actions_in_subset[s].add(action)
        
    return selected_indices, np.mean(current_max_utilities)

def select_policies(policies, policy_evals, score_fn, max_actions=3, **score_kwargs):
    num_policies = len(policies)
    num_states = len(policies[0]['policy'])

    selected = []
    actions_in_subset = [set() for _ in range(num_states)]

    while True:
        best_gain = -np.inf
        best_idx = None

        for i in range(num_policies):
            if i in selected:
                continue

            # ---- constraint check ----
            violation = False
            for s, action in enumerate(policies[i]['policy']):
                if action not in actions_in_subset[s] and len(actions_in_subset[s]) >= max_actions:
                    violation = True
                    break
            if violation:
                continue

            # ---- objective plug-in ----
            gain = score_fn(i=i, policies=policies, policy_evals=policy_evals, selected=selected, **score_kwargs)

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx is None or best_gain <= 0:
            break

        selected.append(best_idx)

        # update constraint tracker
        for s, action in enumerate(policies[best_idx]['policy']):
            actions_in_subset[s].add(action)

    return selected

def score_eum(i, policy_evals, selected, weight_samples, **kwargs):
    if len(selected) == 0:
        current = np.zeros(len(weight_samples))
    else:
        current = np.max(
            np.dot(weight_samples, policy_evals[selected].T),
            axis=1
        )

    candidate = np.dot(weight_samples, policy_evals[i])

    new = np.maximum(current, candidate)

    return np.mean(new - current)

def score_weighted_sum(i, policy_evals, weights, **kwargs):
    return np.dot(weights, policy_evals[i])

def retrieve_top_policies(pareto_archive, method, n=5):
    if method == "hv_contribution":
        ref_point_HV = np.min(pareto_archive.evaluations, axis=0) - 0.1  # Set reference point slightly worse than the worst evaluation in the archive
        metric = hv_contributions([eval for eval in pareto_archive.evaluations], ref_point_HV, maximise=True)
    elif method == "crowding":
        cd = get_crowding_function("cd")
        metric = cd.do(pareto_archive.evaluations)
    sorted_inds = np.argsort(metric)[::-1]
    print(sorted_inds[:n])
    top_n_individuals = np.array(pareto_archive.individuals)[sorted_inds][:n]
    top_n_evaluations = np.array(pareto_archive.evaluations)[sorted_inds][:n]
    policies = np.array([ind['policy'] for ind in top_n_individuals])
    weights = [ind['weights'] for ind in top_n_individuals]
    return policies, weights, top_n_evaluations

def get_max_unique_actions(pareto_archive, all_states):
    """
    pareto_archive: List of policies (e.g., each policy is a function or a dict)
    all_states: A list or array of all possible states in your MDP
    """
    state_action_map = {tuple(s) if isinstance(s, np.ndarray) else s: set() for s in range(all_states)}

    for policy in pareto_archive:
        policy = policy['policy']
        for state in range(all_states):
            # Get the action recommended by this specific policy for this state
            # Adjust the call below based on how your policy is structured
            action = policy[state]
            
            # Use tuple if state is an array to make it hashable for the dict
            s_key = tuple(state) if isinstance(state, np.ndarray) else state
            state_action_map[s_key].add(action)

    # Calculate number of unique actions per state
    counts = [len(actions) for actions in state_action_map.values()]
    
    max_unique = max(counts)
    avg_unique = sum(counts) / len(counts)
    
    # Optional: Find which state has the most disagreement
    most_disputed_state = max(state_action_map, key=lambda k: len(state_action_map[k]))

    return {
        "max_unique_actions": max_unique,
        "avg_unique_actions": avg_unique,
        "most_disputed_state": most_disputed_state
    }

def calculate_expected_utility(policy_evals, weight_samples):
    """
    Calculates the EUM for a solution set.
    
    Args:
        policy_evals: Array of shape (num_policies, num_objectives)
        weight_samples: Array of shape (num_samples, num_objectives) 
                        where each row sums to 1.
    """
    utilities = np.dot(weight_samples, policy_evals.T)
    max_utilities = np.max(utilities, axis=1)
    return np.mean(max_utilities)

def get_actions_per_state(policies, all_states):
    state_action_map = {s: set() for s in range(all_states)}

    for policy in policies:
        policy = policy['policy']
        for state in range(all_states):
            action = policy[state]
            state_action_map[state].add(action)

    return state_action_map


    