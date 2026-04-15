import numpy as np
import matplotlib.pyplot as plt
import utils
from ipywidgets import interact, fixed
import ipywidgets as widgets
import pandas as pd
import seaborn as sns

def simulate(env, num_users, policy=None, policy_name=None, verbose=False, T=28, num_vals=3, max_count=3):
    data_list = []    
    for user in range(num_users):
        t = 0
        obs, _ = env.reset()
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
    simulation_results['policy'] = policy_name if policy_name is not None else 'Random'
    return simulation_results

def simulate_multiple_policies(env, num_users, policies, policy_evals, selection='random', T=28, num_vals=3, max_count=3):
    data_list = []
    for user in range(num_users):
        t = 0
        obs, _ = env.reset()
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
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=True)
            elif selection == 'user_choice_least_time':
                # pick the action from the policy with lowest time required score (objective index 0)
                policy_idx = np.argmax(policy_evals[:, 0])
                action = policies[policy_idx, state_idx]
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=True) 
            elif selection == 'user_choice_random':
                action = np.random.choice(actions)
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=True)
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
    simulation_results['policy'] = 'Multipolicy ' + selection
    return simulation_results

def plot_seaborn(plot_df, measure, obj_idx=None, cumulative=False):    
    # If it's the 'rewards' column, we need to extract the specific objective index
    if measure == 'rewards' and obj_idx is not None:
        plot_df['value'] = plot_df['rewards'].apply(lambda r: r[obj_idx])
    else:
        plot_df['value'] = plot_df[measure]
        
    if cumulative:
        # Calculate cumulative sum per policy and per user
        plot_df = plot_df.sort_values(['policy', 'user', 't'])
        plot_df['value'] = plot_df.groupby(['policy', 'user'])['value'].cumsum()

    plt.figure(figsize=(10, 6))
    
    # Seaborn automatically handles the mean and 95% Confidence Interval (shadow)
    sns.lineplot(data=plot_df, x='t', y='value', hue='policy', style='policy')
    
    plt.title(f"Average {measure} {'(Cumulative)' if cumulative else ''}")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_simulation(results_list, measure, obj_idx=None, cumulative=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    for df in results_list:
        name = df['policy'].iloc[0]
        n_users = df['user'].nunique()
        n_steps = df['t'].nunique()

        if measure == 'rewards' and obj_idx is not None:
            vals = np.array(df['rewards'].tolist())[:, obj_idx].reshape(n_users, n_steps)
        else:
            vals = df[measure].values.reshape(n_users, n_steps)

        if cumulative:
            vals = np.cumsum(vals, axis=1)

        mean = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        ci = 1.96 * (std / np.sqrt(n_users))
        
        line, = ax.plot(mean, label=name, lw=2, linestyle='--' if "Random" in name else '-')
        ax.fill_between(range(n_steps), mean - ci, mean + ci, color=line.get_color(), alpha=0.15)
    ax.legend()
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Cumulative Mean {measure}" if cumulative else f"Mean {measure}")
    ax.grid(True, alpha=0.2)
    return ax

def interactive_plot_objective(results_list, objectives):
    @interact
    def ui(obj_idx=widgets.Dropdown(options=[(n, i) for i, n in enumerate(objectives)], description='Objective:'),
           is_cumulative=widgets.Checkbox(value=True, description='Cumulative')):
        
        ax = plot_simulation(
            results_list, 
            measure='rewards', 
            obj_idx=obj_idx, 
            cumulative=is_cumulative
        )
        
        ax.set_title(f"Average {objectives[obj_idx]} Over Time")
        ax.set_ylabel(f"Cumulative Mean {objectives[obj_idx]}" if is_cumulative else f"Mean {objectives[obj_idx]}")
        plt.tight_layout()
        plt.show()