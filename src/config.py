import numpy as np

state_features = ['TIR', 'TIME_Q', 'MOT']
feature_names = ['Tiredness', 'Time available', 'Motivation']

action_col = "joint_cluster"

cluster_col = 'cluster_all'
cluster_vars = ['likedness', 'usefulness', 'difficulty']

reward_cols = ["r_likedness", "r_usefulness", "r_return", "r_adherence", "r_diversity"]
reward_names = ["Perceived Enjoyment", "Perceived Usefulness", "Return Willingness", "Adherence", "Diversity"]

NUM_CATEGORIES = 4  # Number of coping categories
NUM_FEATURES = len(state_features)
NUM_VALS_PER_FEATURE = [2, 2, 3]
NUM_STATES = np.prod(NUM_VALS_PER_FEATURE)
NUM_CLUSTERS = 3
NUM_ACTIONS = NUM_CLUSTERS * 2  # Within each cluster we either recommend a 'least completed' coping category or a more familiar one
NUM_OBJECTIVES = len(reward_cols)

# Simulation settings
NUM_TIMESTEPS = 28
NUM_USERS = 1000

# Agency bias parameters (mean, std) for the normal distribution used to sample the agency bias during simulation
agency_bias = (0.031, 0.013)