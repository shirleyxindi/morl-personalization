"""
Utilities to select, score and inspect sets of multi-objective policies.

Author: Shirley Li
Date: July 2026
"""
import numpy as np
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from moocore import hv_contributions

def select_policies(policies, policy_evals, score_fn, max_actions=3, **score_kwargs):
    """Greedily select policies using a custom scoring function.

    Args:
        policies: Iterable of policy objects or dicts with a `policy` entry.
        policy_evals: Array of shape (num_policies, num_objectives).
        score_fn: Callable with signature `score_fn(i, policies, policy_evals, selected, **score_kwargs)`.
        max_actions: Maximum number of distinct actions allowed per state.
        **score_kwargs: Extra keyword arguments forwarded to `score_fn`.

    Returns:
        List of selected policy indices in the order they were added.
    """
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

            # check max actions constraint
            violation = False
            for s, action in enumerate(policies[i]['policy']):
                if action not in actions_in_subset[s] and len(actions_in_subset[s]) >= max_actions:
                    violation = True
                    break
            if violation:
                continue

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
    """Compute the marginal expected utility gain of adding one policy.

    Args:
        i: Candidate policy index.
        policy_evals: Array of shape (num_policies, num_objectives).
        selected: Indices of the policies already selected.
        weight_samples: Array of sampled weight vectors with shape (num_samples, num_objectives).

    Returns:
        Mean per-sample utility gain from adding policy `i`.
    """
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
    """Score a policy with a standard weighted sum.

    Args:
        i: Policy index.
        policy_evals: Array of shape (num_policies, num_objectives).
        weights: Weight vector of length num_objectives.

    Returns:
        Weighted sum of the evaluation vector for policy `i`.
    """
    return np.dot(weights, policy_evals[i])

def retrieve_top_policies(pareto_archive, method, n=5):
    """Retrieve the top policies from a Pareto archive using a ranking metric.

    Args:
        pareto_archive: Archive with `evaluations` and `individuals` attributes.
        method: Ranking method, either `hv_contribution` or `crowding`.
        n: Number of policies to return.

    Returns:
        Tuple of `(policies, weights, evaluations)` for the selected entries.
    """
    if method == "hv_contribution":
        ref_point_HV = np.min(pareto_archive.evaluations, axis=0) - 0.1  # slightly worse than worst
        metric = hv_contributions([eval for eval in pareto_archive.evaluations], ref_point_HV, maximise=True)
    elif method == "crowding":
        cd = get_crowding_function("cd")
        metric = cd.do(pareto_archive.evaluations)
    sorted_inds = np.argsort(metric)[::-1]
    print(sorted_inds[:n])
    top_n_individuals = np.array(pareto_archive.individuals)[sorted_inds][:n]
    top_n_evaluations = np.array(pareto_archive.evaluations)[sorted_inds][:n]
    policies = np.array([ind['policy'] for ind in top_n_individuals])
    weights = [ind.get('weights') for ind in top_n_individuals]
    return policies, weights, top_n_evaluations

def get_max_unique_actions(pareto_archive, all_states):
    """Summarize how many distinct actions appear per state across policies.

    Args:
        pareto_archive: Iterable of policy dicts, each exposing a `policy` entry.
        all_states: Number of states to inspect.

    Returns:
        Dictionary with the maximum, average, and most disputed-state action counts.
    """
    state_action_map = {tuple(s) if isinstance(s, np.ndarray) else s: set() for s in range(all_states)}

    for policy in pareto_archive:
        policy = policy['policy']
        for state in range(all_states):
            action = policy[state]
            
            s_key = tuple(state) if isinstance(state, np.ndarray) else state
            state_action_map[s_key].add(action)

    counts = [len(actions) for actions in state_action_map.values()]
    
    max_unique = max(counts)
    avg_unique = sum(counts) / len(counts)
    
    most_disputed_state = max(state_action_map, key=lambda k: len(state_action_map[k]))

    return {
        "max_unique_actions": max_unique,
        "avg_unique_actions": avg_unique,
        "most_disputed_state": most_disputed_state
    }

def calculate_eum(policy_evals, weight_samples):
    """Compute the expected utility metric (EUM) of a policy set under sampled weights.

    Args:
        policy_evals: Array of shape (num_policies, num_objectives).
        weight_samples: Array of shape (num_samples, num_objectives).

    Returns:
        Expected utility metric of the policy set.
    """
    utilities = np.dot(weight_samples, policy_evals.T)
    max_utilities = np.max(utilities, axis=1)
    return np.mean(max_utilities)

def get_actions_per_state(policies, all_states):
    """Return the set of unique actions observed for each state.

    Args:
        policies: Iterable of policies, either dicts with `policy` or raw sequences.
        all_states: Number of states to inspect.

    Returns:
        Dictionary mapping each state index to a set of distinct actions.
    """
    state_action_map = {s: set() for s in range(all_states)}

    for policy in policies:
        policy = policy['policy'] if isinstance(policy, dict) else policy
        for state in range(all_states):
            action = policy[state]
            state_action_map[state].add(action)

    return state_action_map


