import pandas as pd
import numpy as np
import os
import mo_gymnasium as mo_gym
import gymnasium as gym
from env.user_sim import UserSimEnv
from morl_baselines.multi_policy.morld import morld
from morl_baselines.common.pareto import ParetoArchive
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
import value_iteration
import utils
from morld.vi_morld_simple import VIMORLD
from morld.mo_vi import MOValueIteration

MAX_COUNT = 3
NUM_VALS = 3
NUM_STOCHASTIC_STATES = NUM_VALS**3
NUM_STATES = NUM_STOCHASTIC_STATES * (MAX_COUNT+1)**4 
NUM_ACTIONS = 104
NUM_OBJECTIVES = 6

data_folder = 'C:\\Users\\shirl\\Documents\\Studie\\2025-2026\\Thesis\\personalized-coping-challenges\\data\\'
results_folder = 'C:\\Users\\shirl\\Documents\\Studie\\2025-2026\\Thesis\\personalized-coping-challenges\\results\\'

action_file = 'normalized_challenges.csv'
action_df = pd.read_csv(data_folder + action_file)
action_df['action_id'] = action_df['action_id'] - 1


expert_score_cols = ['score_acceptance', 'score_distraction', 'score_problem_solving', 'score_social_support']
expert_score_matrix = action_df[expert_score_cols].values

category_mapping = {"acceptance": 0, "distraction": 1, "problem_solving": 2, "social_support": 3}
action_df['category_id'] = action_df['category'].map(category_mapping)  
action_categories = action_df['category_id'].values

transition_probs = np.load(data_folder + 'functions\\2\\transition_probs.npy')
reward_matrix = np.load(data_folder + 'functions\\2\\reward_matrix.npy')

NUM_ACTIONS = len(action_df)

env = mo_gym.make('user_env', num_actions=NUM_ACTIONS, 
                 num_objectives = NUM_OBJECTIVES, 
                 expert_score_matrix=expert_score_matrix, 
                 transition_probs=transition_probs, 
                 reward_matrix=reward_matrix, 
                 action_categories=action_categories,
                 MAX_COUNT=MAX_COUNT)

num_evals = 30
pop_size = 2
num_steps_per_episode=28
total_steps = 1

model_name = f'vimorld_T={total_steps}_P={pop_size}_nO={NUM_OBJECTIVES}_nA={NUM_ACTIONS}_n_eval={num_evals}'
filename = model_name
model_file = os.path.join(results_folder, model_name, 'weights', filename)

# morld_agent = VIMORLD(env, weight_init_method="random", weight_adaptation_method="PSA", pop_size=pop_size)
# if os.path.exists(model_file + '.tar'):
#     print("Loading pretrained MORLD model...")
#     morld_agent.load(path=model_file + '.tar')

# else:
#     print("No pretrained MORLD model found. Training a new model...")
#     eval_env = mo_gym.make('user_env', num_actions=NUM_ACTIONS, 
#                     num_objectives = NUM_OBJECTIVES, 
#                     expert_score_matrix=expert_score_matrix, 
#                     transition_probs=transition_probs, 
#                     reward_matrix=reward_matrix, 
#                     action_categories=action_categories,
#                     MAX_COUNT=MAX_COUNT)
#     ref_point = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])  # reference point for tchebycheff scalarization
#     morld_agent.train(iterations=total_steps)
#     # os.makedirs(os.path.join(results_folder, model_name, 'weights'))
#     # morld_agent.save(filename=model_file)

pareto_archive = ParetoArchive()
    
def adapt_weights_random(agents, num_iterations=5):
    for _ in range(num_iterations):
        print("Adapting weights randomly...")
        for agent in agents:
            # add some random noise to the weights, ensure positivity and normalization
            noise = np.random.normal(1, 0.1, size=agent.weights.shape)
            new_weights = agent.weights * noise
            # normalize the weights
            new_weights /= np.sum(new_weights)
            agent.set_weights(new_weights)

            new_eval = agent.expected_return
            pareto_archive.add(agent.weights, new_eval)

agents = []
for i in range(pop_size):
    weigths = np.random.dirichlet(np.ones(NUM_OBJECTIVES), size=1)[0]
    vi_agent = MOValueIteration(id=i, env=env.unwrapped, weights=weigths, gamma=0.9)
    vi_agent.train()
    agents.append(vi_agent)
    pareto_archive.add(vi_agent.weights, vi_agent.expected_return)


adapt_weights_random(agents)
