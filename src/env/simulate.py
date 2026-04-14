import numpy as np
import matplotlib.pyplot as plt
import utils
from ipywidgets import interact, fixed
import ipywidgets as widgets
import pandas as pd

def simulate(env, num_users, policy=None, verbose=False, T=28, num_vals=3, max_count=3):
    data_list = []    
    for user in range(num_users):
        t = 0
        obs, _ = env.reset()  # later use initial state distribution
        done = False
        while not done and t < T:
            state_idx = utils.state_to_idx(obs, num_feats=3, num_counts=env.unwrapped.num_clusters, num_vals=num_vals, max_count=max_count)
            action = policy[state_idx] if policy is not None else env.action_space.sample()
            obs_next, rewards, terminated, truncated, info = env.step(action)
            if verbose:
                print(f"Action: {action}, Rewards: {rewards}, Info: {info}")
            user_row = {
                'user': user,
                't': t,
                'action': action,
                'state': obs,
                'next_state': obs_next,
                'counts': obs[3:],
                'num_completed': info['num_completed'],
                'counts_per_category': info['counts_per_category'].copy(),
                'rewards': rewards
            }
            data_list.append(user_row)

            done = terminated or truncated
            obs = obs_next
            t += 1
    simulation_results = pd.DataFrame(data_list)
    return simulation_results

def simulate_multiple_policies(env, num_users, policies, policy_evals, selection='random', T=28, num_vals=3, max_count=3):
    data_list = []
    for user in range(num_users):
        t = 0
        obs, _ = env.reset()  # later use initial state distribution
        done = False
        while not done and t < T:
            state_idx = utils.state_to_idx(obs, num_feats=3, num_counts=env.unwrapped.num_clusters, num_vals=num_vals, max_count=max_count)
            actions = policies[:, state_idx]
            if selection == 'random':
                action = np.random.choice(actions)
                obs_next, rewards, terminated, truncated, info = env.step(action)
            elif selection == 'user_choice_most_fun':
                #  policy evals is (num_policies, num_objectives), we pick the action from the policy with highest likability score (objective index 1)
                policy_idx = np.argmax(policy_evals[:, 1])
                action = policies[policy_idx, state_idx]
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=0.1)  # add some bias to increase chance of completion for user choice
            elif selection == 'user_choice_least_time':
                # pick the action from the policy with lowest time required score (objective index 0)
                policy_idx = np.argmax(policy_evals[:, 0])
                action = policies[policy_idx, state_idx]
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=0.1)  # add some bias to increase chance of completion for user choice
            user_row = {
                'user': user,
                't': t,
                'action': action,
                'state': obs,
                'next_state': obs_next,
                'counts': obs[3:],
                'num_completed': info['num_completed'],
                'counts_per_category': info['counts_per_category'].copy(),
                'rewards': rewards
            }
            data_list.append(user_row)

            done = terminated or truncated
            obs = obs_next
            t += 1
    simulation_results = pd.DataFrame(data_list)
    return simulation_results

def plot_simulation_results(simulation_results, policy_names, measure, num_users, num_timesteps, cumulative=False):
    plt.figure(figsize=(10, 6))
    for idx, sim_res in enumerate(simulation_results):
        name = policy_names[idx]
        vals = np.stack(sim_res[measure]).reshape(num_users, num_timesteps)
        if cumulative:
            vals = np.cumsum(vals, axis=1)
        mean_vals = vals.mean(axis=0)
        style = '--' if "Random" in name else '-'
        plt.plot(mean_vals, label=name, linestyle=style)
    plt.xlabel('Timestep')
    plt.ylabel(measure.capitalize())
    plt.title(f'Average {measure.capitalize()} Over Time {"(Cumulative)" if cumulative else ""}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_objective(simulation_results, policy_names, objectives, NUM_USERS, NUM_TIMESTEPS, NUM_OBJECTIVES, objective_idx, is_cumulative):
    plt.figure(figsize=(10, 6))
    obj_name = objectives[objective_idx]
    for idx, sim_res in enumerate(simulation_results):
        name = policy_names[idx]
        raw_rewards = np.stack(sim_res['rewards'])
        rewards = raw_rewards.reshape(NUM_USERS, NUM_TIMESTEPS, NUM_OBJECTIVES)
        data = rewards[:, :, objective_idx]
        if is_cumulative:
            data = np.cumsum(data, axis=1)
        mean_rewards = np.mean(data, axis=0)
        style = '--' if "Random" in name else '-'
        plt.plot(mean_rewards, label=name, linestyle=style)

    plt.title(f"Average {obj_name} Over Time {'(Cumulative)' if is_cumulative else ''}")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def interactive_plot_objective(simulation_results, policy_names, objectives, NUM_USERS, NUM_TIMESTEPS, NUM_OBJECTIVES):
    interact(
        plot_objective,
        simulation_results=fixed(simulation_results),
        policy_names=fixed(policy_names),
        objectives=fixed(objectives),
        NUM_USERS=fixed(NUM_USERS),
        NUM_TIMESTEPS=fixed(NUM_TIMESTEPS),
        NUM_OBJECTIVES=fixed(NUM_OBJECTIVES),
        objective_idx=widgets.Dropdown(
            options=[(name, i) for i, name in enumerate(objectives)],
            value=0,
            description='Objective:',
        ),
        is_cumulative=widgets.Checkbox(
            value=True,
            description='Cumulative Sum',
        )
    )