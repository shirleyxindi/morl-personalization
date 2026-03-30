import numpy as np
import matplotlib.pyplot as plt
import utils

def run_env(env, num_simulations, policy=None, verbose=False):
    rewards_list = []
    counts_list = []
    for _ in range(num_simulations):
        obs, _ = env.reset()
        done = False
        total_rewards = []
        total_counts = []
        while not done:
            state = utils.state_to_idx((obs[0], obs[1], obs[2]), max_count=0)
            action = policy[state] if policy is not None else env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            if verbose:
                print(f"Action: {action}, Rewards: {rewards}, Info: {info}")
            total_rewards.append(rewards)
            total_counts.append(obs[3:])
            done = terminated or truncated
        rewards_list.append(total_rewards)
        counts_list.append(total_counts)
    return rewards_list, counts_list

def visualize_rewards(rewards_list, cumulative=False):
    # visualize rewards over time for each objective, averaged over users
    objectives = ["Time Required", "Likability", "Perceived Usefulness", "Expert Score", "Diversity"] 
    rewards_array = np.array(rewards_list)  # shape: (num_users, episode_length, num_objectives)
    if cumulative:
        rewards_array = np.cumsum(rewards_array, axis=1)
    avg_rewards = rewards_array.mean(axis=0)  # shape: (episode_length, num_objectives)
    plt.figure(figsize=(12, 6))
    for i in range(avg_rewards.shape[1]):
        plt.plot(avg_rewards[:, i], label=objectives[i])
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards Over Time')
    plt.legend()
    plt.show()

def visualize_counts(counts_list):
    # visualize counts of coping strategy categories over time, averaged over users
    categories = ["AC", "DS", "PS", "SS"]
    counts_array = np.array(counts_list)  # shape: (num_users, episode_length, num_categories)
    avg_counts = counts_array.mean(axis=0)  # shape: (episode_length, num_categories)
    plt.figure(figsize=(12, 6))
    for i in range(avg_counts.shape[1]):
        plt.plot(avg_counts[:, i], label=categories[i])
    plt.xlabel('Timestep')
    plt.ylabel('Average Count')
    plt.title('Average Counts of Coping Strategy Categories Over Time')
    plt.legend()
    plt.show()