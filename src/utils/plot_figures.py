"""
Plotting helpers for simulation outcomes and policy comparisons.

Author: Shirley Li
Date: July 2026
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ipywidgets import interact, fixed
import ipywidgets as widgets

def get_line_style(policy_name):
    """Return a line style for a policy label.

    Args:
        policy_name: Name of the policy.

    Returns:
        Matplotlib line-style string.
    """
    if "Random" in policy_name:
        return '-'
    elif "S" in policy_name:
        return ':'
    else:
        return '-.'

def plot_seaborn(df, measure, save_path=None):    
    """Plot a policy-level time series with seaborn styling.

    Args:
        df: Long-form DataFrame with `t`, `policy`, and the plotted measure.
        measure: Column name to plot on the y-axis.
        save_path: Optional path to save the figure.

    Returns:
        None. Displays the plot and optionally saves it.
    """
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
    plt.legend(title="(Meta)Policy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_simulation(df, measure, obj_idx=None, cumulative=False, ax=None):
    """Plot per-policy mean trajectories with confidence intervals.

    Args:
        df: Simulation DataFrame with `policy`, `user`, and `t` columns.
        measure: Column to aggregate, or `rewards` when selecting an objective.
        obj_idx: Objective index used when `measure == 'rewards'`.
        cumulative: If True, plot cumulative sums over time.
        ax: Optional axis to draw on.

    Returns:
        The matplotlib axis used for plotting.
    """
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

        line, = ax.plot(mean, label=policy_name, lw=2, linestyle=get_line_style(policy_name))
        ax.fill_between(range(n_steps), mean - ci, mean + ci, color=line.get_color(), alpha=0.15)

    ax.legend()
    ax.set_xlabel("Timesteps")
    measure = "# Completed Challenges" if measure == "num_completed" else measure
    ax.set_ylabel(f"Cumulative Mean {measure}" if cumulative else f"Mean {measure}")
    ax.grid(True, alpha=0.2)

    return ax


def interactive_plot_objective(df, objectives):
    """Create a widget UI for exploring objective trajectories that can be used in a notebook.

    Args:
        df: Simulation DataFrame with reward trajectories.
        objectives: Objective names matching reward indices.

    Returns:
        None. Displays the widget-driven plot UI.
    """

    @interact
    def ui(
        obj_idx=widgets.Dropdown(options=[(n, i) for i, n in enumerate(objectives)], description="Objective:"),
        is_cumulative=widgets.Checkbox(value=False, description="Cumulative")
    ):

        ax = plot_simulation(df=df, measure="rewards", obj_idx=obj_idx, cumulative=is_cumulative)

        ax.set_title(f"Average {objectives[obj_idx]} Over Time")
        ax.set_ylabel(f"Cumulative Mean {objectives[obj_idx]}" if is_cumulative else f"Mean {objectives[obj_idx]}")

        plt.tight_layout()
        plt.show()

def plot_fraction_completed(df, max_count_per_category=4, save_path=None):
    """Plot the fraction of completing a target number of challenges per category over time.

    Args:
        df: Simulation DataFrame with `counts_per_category`.
        max_count_per_category: Target number of challenges per category.
        save_path: Optional path to save the figure.

    Returns:
        A dictionary mapping policy names to mean fraction trajectories.
    """
    plt.figure(figsize=(10, 6))

    fraction_at_t = {}

    for policy_name, policy_df in df.groupby("policy"):
        policy_df = policy_df.sort_values(["user", "t"])

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

        sns.lineplot(data=plot_df, x="timestep", y="fraction_completed", label=policy_name, linestyle=get_line_style(policy_name))

        fraction_at_t[policy_name] = fraction_completed[:, :].mean(axis=0)
    print("Average fraction completed at the end of the simulation:")
    for policy_name, fraction in fraction_at_t.items():
        print(f"  {policy_name}: {fraction[-1]:.2f}")

    plt.xlabel("Timestep")
    plt.ylabel(f"Fraction of completing\n{max_count_per_category} challenges per category")
    plt.legend(title="(Meta)Policy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return fraction_at_t    

def get_consecutive_failures(df):
    """Annotate each row with consecutive non-completion counts.

    Args:
        df: Simulation DataFrame with `user`, `t`, and `completed`.

    Returns:
        A copy of the DataFrame with `block` and `consecutive_no` columns.
    """
    df = df.sort_values(['user', 't'])
    df['block'] = (df['completed'] != df.groupby('user')['completed'].shift()).cumsum()
    df['consecutive_no'] = df.groupby(['user', 'block']).cumcount() + 1
    df.loc[df['completed'] == 1, 'consecutive_no'] = 0
    return df

def apply_post_hoc_dropout(df, threshold=3):
    """Remove rows after each user's first dropout event.

    Args:
        df: Simulation DataFrame, optionally already annotated by `get_consecutive_failures`.
        threshold: Consecutive non-completion count used to define dropout.

    Returns:
        A filtered DataFrame containing only active rows.
    """
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
    """Plot fraction of active users after applying a dropout threshold.

    Args:
        df: Simulation DataFrame with `policy`, `user`, and `completed`.
        threshold: Consecutive non-completion count used to define dropout.
        save_path: Optional path to save the figure.

    Returns:
        None. Displays the plot.
    """
    plt.figure(figsize=(10, 6))

    active_at_end = {}
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
            linestyle=get_line_style(policy_name),
        )
        active_at_end[policy_name] = survival["survival_rate"].iloc[-1]

    print("Fraction of users still active at the end of the simulation:")
    for policy_name, survival_rate in active_at_end.items():    
        print(f"  {policy_name}: {survival_rate:.2%}")
    
    plt.ylabel("Fraction of active users")
    plt.xlabel("Timestep")
    plt.legend(title="(Meta)Policy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_state_trajectory(df, target_feature="Motivation", target_idx=2, threshold=2, save_path=None, real_df=None):
    """Plot the fraction of users above a state threshold over time.

    Args:
        df: Simulation DataFrame with a `state` column.
        target_feature: Label used on the y-axis.
        target_idx: Index of the state feature to threshold.
        threshold: Threshold for marking a state as high.
        save_path: Optional path to save the figure.
        real_df: Optional comparison DataFrame plotted in black.

    Returns:
        None. Displays the plot.
    """
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

    if real_df is not None:
        real_trend = (
            real_df.assign(high=(np.vstack(real_df["state"].values)[:, target_idx] >= threshold))
            .groupby("t")["high"]
            .mean()
            .reset_index()
        )

        plt.plot(real_trend["t"], real_trend["high"], color="black", linewidth=2, linestyle="-", label="Real Data", zorder=10)

    plt.xlabel("Timestep")
    plt.ylabel(f"Fraction of Users with High {target_feature}")
    plt.legend(title="(Meta-)Policy")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_batched_dropout(df, threshold=3, batch_size=200, n_batches=5, seed=42, ci=True, save_path=None):
    """Plot fraction of active users after applying a dropout threshold from repeated user batches.

    Args:
        df: Simulation DataFrame with `policy`, `user`, `t`, and `completed`.
        threshold: Consecutive failure count used to define dropout.
        batch_size: Number of users per batch.
        n_batches: Number of batches per policy.
        seed: Random seed for batching.
        ci: If True, show confidence intervals.
        save_path: Optional path to save the figure.

    Returns:
        None. Displays the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    rng = np.random.default_rng(seed)
    active_at_end = {}
    for policy_name, policy_df in df.groupby("policy"):
        users = rng.permutation(policy_df["user"].unique())
        users = users[:batch_size * n_batches]
        batches = np.array_split(users, n_batches)
        end_values = []
        records = []
        for batch in batches:
            batch_df = policy_df[policy_df["user"].isin(batch)]
            active_df = apply_post_hoc_dropout(batch_df, threshold=threshold)

            timesteps = sorted(batch_df["t"].unique())
            survival = (
                active_df.groupby("t")["user"]
                .nunique()
                .reindex(timesteps, fill_value=0)
                .div(len(batch))
            )
            for t, val in survival.items():
                records.append({"t": t, "survival": val})
            end_values.append(survival.iloc[-1])

        melted = pd.DataFrame(records)

        sns.lineplot(
            data=melted,
            x="t",
            y="survival",
            label=policy_name,
            linestyle=get_line_style(policy_name),
            errorbar=("ci", 95) if ci else None,
            ax=ax,
        )
        active_at_end[policy_name] = np.mean(end_values)

    print("Fraction of users still active at the end of the simulation:")
    for policy_name, survival_rate in active_at_end.items():    
        print(f"  {policy_name}: {survival_rate:.2%}")

    ax.set_ylabel("Fraction of active users")
    ax.set_xlabel("Timestep")
    ax.legend(title="(Meta)Policy", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_improvements(improvements_over_random, max_improvements=None, save_path=None, ticks_rotation=0):
    """Plot policy improvements over a random baseline.

    Args:
        improvements_over_random: DataFrame or Series-like table of improvements.
        max_improvements: Optional table used to convert to relative improvement.
        save_path: Optional path to save the figure.
        ticks_rotation: Rotation angle for x-axis labels.

    Returns:
        None. Displays the plot.
    """
    relative = max_improvements is not None
    plot_df = (
        improvements_over_random
        .drop("Random")
        .reset_index()
        .melt(id_vars="policy", var_name="objective", value_name="improvement")
    )

    if relative:
        plot_df['relative_improvement'] = plot_df.apply(lambda row: row['improvement'] / max_improvements.loc[row['objective'], 'Max Improvement'] * 100, axis=1)

    plt.figure(figsize=(11, 5))

    palette = sns.color_palette("Set2")
    palette = [palette[i] for i in [6, 0, 1, 2, 3, 4]]

    sns.barplot(
        data=plot_df,
        x="policy",
        y="improvement" if not relative else "relative_improvement",
        hue="objective",
        palette=palette
    )
    label = "Relative " if relative else ""
    plt.ylabel(f"{label}Improvement over Random Policy (%)")
    plt.xlabel("(Meta)Policy")
    plt.axhline(0, linewidth=0.8, color="black", alpha=0.8)
    plt.xticks(rotation=ticks_rotation)
    plt.legend(title="Outcome Measure", bbox_to_anchor=(1.02, 1), loc="upper left")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()