"""
Simulation helpers for running policies in the user environment.

Author: Shirley Li
Date: July 2026
"""

import numpy as np
import pandas as pd

def build_user_row(user, t, action, obs, obs_next, rewards, info):
    """Pack one environment transition into a flat record.

    Args:
        user: User identifier.
        t: Timestep.
        action: Action taken by the policy.
        obs: Current observation.
        obs_next: Next observation.
        rewards: Reward vector returned by the environment.
        info: Info dict returned by the environment.

    Returns:
        Dictionary suitable for appending to a simulation DataFrame.
    """
    return {
            'user': user,
            't': t,
            'action': action,
            'challenge_id': info['challenge_id'],
            'state': obs,
            'next_state': obs_next,
            'completed': info['completed'],
            'num_completed': info['num_completed'],
            'counts_per_category': info['counts_per_category'].copy(),
            'rewards': rewards
        }

def simulate(env, num_users, policy=None, policy_name=None, verbose=False, T=28, seed=66):
    """Simulate a single policy or random behavior for multiple users.

    Args:
        env: Environment with `reset`, `step`, and `get_state_idx` methods.
        num_users: Number of simulated users.
        policy: State-indexed policy array, or None for random actions.
        policy_name: Name of the policy.
        verbose: If True, print step-level transitions.
        T: Maximum number of timesteps per user.
        seed: Base seed; each user uses `seed + user`.

    Returns:
        DataFrame containing one row per simulated step.
    """
    data_list = [] 
    random = policy is None
    for user in range(num_users):
        user_seed = seed + user
        t = 0
        obs, _ = env.reset(seed=user_seed)
        done = False
        while not done and t < T:
            state_idx = env.unwrapped.get_state_idx(obs)
            action = policy[state_idx] if not random else env.action_space.sample()
            obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action)  # use the more stochastic step function for more realistic simulations
            if verbose:
                print(f"Action: {action}, Rewards: {rewards}, Info: {info}")
            user_row = build_user_row(user, t, action, obs, obs_next, rewards, info)
            data_list.append(user_row)

            done = terminated or truncated
            obs = obs_next
            t += 1
    simulation_results = pd.DataFrame(data_list)
    simulation_results['policy'] = policy_name if policy_name is not None else 'Random'
    return simulation_results

def simulate_multiple_policies(env, num_users, policies, policy_evals, selection='random', T=28, seed=66, switch_threshold=12, fall_back_policy=None, policy_name=None):
    """Simulate a menu of policies and let the user model choose among them.

    Args:
        env: Environment with `reset`, `step`, and `step_choice` methods.
        num_users: Number of simulated users.
        policies: Array of shape (n_policies, n_states) with actions per policy.
        policy_evals: Array of policy-level expected returns.
        selection: Selection method to apply. Can be 'most_likely' (user choice), 'expert_priority', or 'combined'.
        T: Maximum number of timesteps per user.
        seed: Base seed; each user uses `seed + user`.
        switch_threshold: Number of completed challenges before switching modes.
        fall_back_policy: Optional policy used in expert-priority mode.
        policy_name: Name of the policy.

    Returns:
        DataFrame containing one row per simulated step.
    """
    data_list = []
    for user in range(num_users):
        user_seed = seed + user
        t = 0
        num_completed = 0
        obs, _ = env.reset(seed=user_seed)
        done = False
        while not done and t < T:
            state_idx = env.unwrapped.get_state_idx(obs)
            if selection == 'expert_priority' or (selection == 'combined' and num_completed >= switch_threshold):
                # pick the action from the policy with the highest value for the expert-driven rewards
                policy_idx = np.argmax(policy_evals[:, 2] + policy_evals[:, 3] + policy_evals[:, 4])  # sum of expert-driven rewards
                action = fall_back_policy[state_idx] if fall_back_policy is not None else policies[policy_idx, state_idx]
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=False)
            else:
                user_type = selection if selection != 'combined' else 'most_likely'
                actions = policies[:, state_idx]
                completion_bias = True
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step_choice(actions, user_type=user_type, completion_bias=completion_bias)
            action = info['action']
            num_completed = info['num_completed']
            user_row = build_user_row(user, t, action, obs, obs_next, rewards, info)
            data_list.append(user_row)
            done = terminated or truncated
            obs = obs_next
            t += 1
    simulation_results = pd.DataFrame(data_list)
    simulation_results['policy'] = policy_name if policy_name is not None else 'Multipolicy ' + selection
    return simulation_results
