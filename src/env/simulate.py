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
            'challenge_id': info['challenge_id'],
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
        t = 0
        obs, _ = env.reset(seed=user_seed)
        done = False
        while not done and t < T:
            state_idx = env.unwrapped.get_full_state_index(obs)
            if selection == 'expert_priority' or (selection == 'combined' and t >= 15):
                # pick the action from the policy with the highest value for the expert-driven rewards
                policy_idx = np.argmax(policy_evals[:, 2] + policy_evals[:, 3] + policy_evals[:, 4])  # sum of expert-driven rewards
                action = policies[policy_idx, state_idx]
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step(action, completion_bias=False)
            else:
                user_type = selection if selection != 'combined' else 'random'
                actions = policies[:, state_idx]
                completion_bias = True
                obs_next, rewards, terminated, truncated, info = env.unwrapped.step_choice(actions, user_type=user_type, completion_bias=completion_bias)
            action = info['action']
            user_row = build_user_row(user, t, action, obs, obs_next, rewards, info)
            data_list.append(user_row)
            done = terminated or truncated
            obs = obs_next
            t += 1
    simulation_results = pd.DataFrame(data_list)
    simulation_results['policy'] = 'Multipolicy ' + selection
    return simulation_results

def plot_seaborn(df, measure, save_path=None):    
    policy_order = sorted(df["policy"].unique())
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x="t",
        y=measure,
        hue="policy",
        style="policy",
        hue_order=policy_order,
        style_order=policy_order,
        dashes = {p: (3, 2) if "Random" in p else "" for p in policy_order},
        palette="muted",
    )
    plt.xlabel("Timestep")
    plt.ylabel(f"Mean {measure}")
    plt.legend(title="(Meta) Policy")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def plot_simulation(df, measure, obj_idx=None, cumulative=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for policy_name, policy_df in df.groupby("policy"):

        policy_df = policy_df.sort_values(["user", "t"])

        n_users = policy_df["user"].nunique()
        n_steps = policy_df["t"].nunique()

        if measure == "rewards" and obj_idx is not None:
            vals = np.array(policy_df["rewards"].tolist())[:, obj_idx]
        else:
            vals = policy_df[measure].values

        vals = vals.reshape(n_users, n_steps)

        if cumulative:
            vals = np.cumsum(vals, axis=1)

        mean = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        ci = 1.96 * (std / np.sqrt(n_users))

        line, = ax.plot(mean, label=policy_name, lw=2, linestyle="--" if "Rnd" in policy_name or "Random" in policy_name else "-")
        ax.fill_between(range(n_steps), mean - ci, mean + ci, color=line.get_color(), alpha=0.15)

    ax.legend()
    ax.set_xlabel("Timesteps")
    measure = "# Completed Challenges" if measure == "num_completed" else measure
    ax.set_ylabel(f"Cumulative Mean {measure}" if cumulative else f"Mean {measure}")
    ax.grid(True, alpha=0.2)

    return ax


def interactive_plot_objective(df, objectives):

    @interact
    def ui(
        obj_idx=widgets.Dropdown(options=[(n, i) for i, n in enumerate(objectives)], description="Objective:"),
        is_cumulative=widgets.Checkbox(value=True, description="Cumulative")
    ):

        ax = plot_simulation(df=df, measure="rewards", obj_idx=obj_idx, cumulative=is_cumulative)

        ax.set_title(f"Average {objectives[obj_idx]} Over Time")
        ax.set_ylabel(f"Cumulative Mean {objectives[obj_idx]}" if is_cumulative else f"Mean {objectives[obj_idx]}")

        plt.tight_layout()
        plt.show()

def plot_fraction_completed(df, max_count_per_category=4, save_path=None):
    plt.figure(figsize=(10, 6))

    for policy_name, policy_df in df.groupby("policy"):
        n_users = policy_df["user"].nunique()
        n_steps = policy_df["t"].nunique()
        counts_per_category = np.stack(policy_df["counts_per_category"]).reshape(n_users, n_steps, -1)
        counts_per_category = np.clip(counts_per_category, 0, max_count_per_category)
        counts = counts_per_category.sum(axis=2)
        fraction_completed = counts / (max_count_per_category * counts_per_category.shape[2])

        plot_df = pd.DataFrame({
            "timestep": np.tile(np.arange(n_steps), n_users),
            "fraction_completed": fraction_completed.flatten(),
            "user": np.repeat(np.arange(n_users), n_steps)
        })

        style = '--' if 'Random' in policy_name else '-'

        sns.lineplot(data=plot_df, x="timestep", y="fraction_completed", label=policy_name, linestyle=style, errorbar=('ci', 95), err_style='band')

    plt.xlabel("Timestep")
    plt.ylabel(f"Average fraction of completing {max_count_per_category} challenges per category")
    plt.legend(title="(Meta-)Policy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_fraction_completed_over_time(df):
    """
    Plots fraction of completed challenges over time per policy.
    
    Expects df to have columns:
      - 'policy'       : policy name (str)
      - 't'            : timestep (int/float)
      - 'num_completed': number of challenges completed at that timestep (int)
    """
    total_challenges = df.groupby("policy")["num_completed"].transform("max")  # or pass explicitly
    
    plt.figure(figsize=(10, 6))

    all_num_completed = df["num_completed"].values
    total_per_policy = df.groupby("policy")["num_completed"].max()  # use as denominator

    for policy_name, group in df.groupby("policy"):
        total = total_per_policy[policy_name]
        
        plot_df = group[["t", "num_completed"]].copy()
        plot_df["fraction_completed"] = plot_df["num_completed"] / total

        style = '--' if 'Random' in policy_name else '-'
        sns.lineplot(
            data=plot_df,
            x="t",
            y="fraction_completed",
            label=policy_name,
            linestyle=style,
            errorbar=('ci', 95),
            err_style='band'
        )

    plt.xlabel("Timestep")
    plt.ylabel("Fraction of challenges completed")
    plt.title("Fraction of Completed Challenges Over Time")
    plt.legend(title="(Meta-)Policy")
    plt.tight_layout()
    plt.show()

def plot_fraction_of_users_completed(df):
    plt.figure(figsize=(10, 6))

    for policy_name, policy_df in df.groupby("policy"):

        completed = policy_df.groupby(["t", "user"])["completed"].max().reset_index()

        fraction_completed = completed.groupby("t")["completed"].mean().reset_index()

        style = "--" if "Random" in policy_name else "-"

        sns.lineplot(data=fraction_completed, x="t", y="completed", label=policy_name, linestyle=style)

    plt.xlabel("Timestep")
    plt.ylabel("Fraction of users who completed a challenge")
    plt.ylim(0, 1)
    plt.legend(title="Policy")
    plt.tight_layout()
    plt.show()


def get_consecutive_failures(df):
    df = df.sort_values(['user', 't'])
    df['block'] = (df['completed'] != df.groupby('user')['completed'].shift()).cumsum()
    df['consecutive_no'] = df.groupby(['user', 'block']).cumcount() + 1
    df.loc[df['completed'] == 1, 'consecutive_no'] = 0
    return df

def apply_post_hoc_dropout(df, threshold=3):
    if 'consecutive_no' not in df.columns:
        df = get_consecutive_failures(df)
    
    dropouts = df[df['consecutive_no'] >= threshold]
    
    # Get the earliest timestep for each user where they dropped out
    first_dropout_t = dropouts.groupby(['policy', 'user'])['t'].min().reset_index()
    first_dropout_t.rename(columns={'t': 'dropout_time'}, inplace=True)
    
    df = df.merge(first_dropout_t, on=['policy', 'user'], how='left')
    
    # Keep rows where 't' <= 'dropout_time' 
    # (or keep all rows if dropout_time is NaN, meaning they never dropped out)
    df_active = df[(df['t'] <= df['dropout_time']) | (df['dropout_time'].isna())].copy()
    
    return df_active.drop(columns=['dropout_time'])

def plot_dropout(df, threshold=3, save_path=None):
    plt.figure(figsize=(10, 6))

    for policy_name, policy_df in df.groupby("policy"):
        df_active = apply_post_hoc_dropout(policy_df, threshold=threshold)

        survival = (
            df_active.groupby("t")["user"].nunique()
            .div(policy_df["user"].nunique())
            .reset_index(name="survival_rate")
        )

        sns.lineplot(
            data=survival,
            x="t",
            y="survival_rate",
            label=policy_name,
            linestyle="--" if "Random" in policy_name else "-",
        )

    plt.ylabel("Fraction of Active Users")
    plt.xlabel("Timestep")
    plt.legend(title="(Meta-)Policy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_state_trajectory(df, target_feature="Usefulness Belief", target_idx=2, threshold=2, save_path=None):
    trend_df = (
        df.assign(high=(np.vstack(df["state"].values)[:, target_idx] >= threshold))
        .groupby(["policy", "t"])["high"]
        .mean()
        .reset_index()
    )
    policy_order = sorted(trend_df["policy"].unique())
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=trend_df,
        x="t",
        y="high",
        hue="policy",
        style="policy",
        dashes = {p: (3,2) if "Random" in p else "" for p in policy_order},
        errorbar=None
    )

    plt.xlabel("Timestep")
    plt.ylabel("Fraction of Users with High Usefulness Belief")
    plt.legend(title="(Meta-)Policy")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()