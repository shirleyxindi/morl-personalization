import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import ipywidgets as widgets
import pandas as pd
import seaborn as sns

def build_user_row(user, t, action, obs, obs_next, rewards, info):
    return {
            'user': user,
            't': t,
            'action': action,
            'state': obs,
            'next_state': obs_next,
            'completed': info['completed'],
            'counts': obs[3:],
            'num_completed': info['num_completed'],
            'counts_per_category': info['counts_per_category'].copy(),
            'expert_competencies': info['expert_competencies'].copy(),
            'rewards': rewards
        }

def simulate(env, num_users, policy=None, policy_name=None, verbose=False, T=28, seed=66, random_within_cluster=False):
    data_list = [] 
    random = policy is None
    for user in range(num_users):
        user_seed = seed + user
        t = 0
        obs, _ = env.reset(seed=user_seed)
        done = False
        while not done and t < T:
            state_idx = env.unwrapped.get_full_state_index(obs)
            action = policy[state_idx] if not random else env.action_space.sample()
            obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, random_within_cluster=random_within_cluster)  # use the more stochastic step function for more realistic simulations
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

def simulate_multiple_policies(env, num_users, policies, policy_evals, selection='random', T=28, seed=66, random_within_cluster=False):
    data_list = []
    for user in range(num_users):
        user_seed = seed + user
        rng = np.random.default_rng(user_seed)
        t = 0
        obs, _ = env.reset(seed=user_seed)
        done = False
        while not done and t < T:
            state_idx = env.unwrapped.get_full_state_index(obs)
            actions = policies[:, state_idx]
            completion_bias = True
            if selection == 'expert_priority':
                # pick the action from the policy with the highest expert score
                policy_idx = np.argmax(policy_evals[:, 2])
                actions = [policies[policy_idx, state_idx]]
                completion_bias = False

            obs_next, rewards, terminated, truncated, info = env.unwrapped.step_multi_action(actions, user_type=selection, completion_bias=completion_bias)  # use the more stochastic step function for more realistic simulations
            action = info['action']
            user_row = build_user_row(user, t, action, obs, obs_next, rewards, info)
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