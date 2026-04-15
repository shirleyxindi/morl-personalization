from typing import Optional
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import utils
category_mapping = {0: "AC", 1: "DS", 2: "PS", 3: "SS"}

class UserSimEnv(gym.Env):

    def __init__(self, 
                 transition_probs, 
                 completion_probs,
                 reward_matrix, 
                 challenge_info,
                 challenges_per_cluster,
                 action_categories,
                 initial_distribution=None,
                 num_states=27, 
                 num_actions=103, 
                 num_objectives=5,  
                 num_clusters=5,
                 max_episode_steps=28, 
                 MAX_COUNT=4,
                 seed=42):
        super(UserSimEnv, self).__init__()
        self.nA = num_actions
        self.nO = num_objectives
        self.nS_user = num_states
        self.nS_count = (MAX_COUNT+1)**num_clusters
        self.nS = num_states * self.nS_count
        self.num_clusters = num_clusters
        self.action_space = gym.spaces.Discrete(self.nA)
        self.MAX_COUNT = MAX_COUNT
        self.initial_distribution = initial_distribution
        self.rng = np.random.default_rng(seed)

        # Observation space is a vector of 7 features: [tiredness, time, motivation] + [count_AC, count_DS, count_PS, count_SS]
        highs = np.array([2, 2, 2] + [MAX_COUNT]*num_clusters)
        self.observation_space = gym.spaces.Box(low=0, high=highs, shape=(len(highs),), dtype=int)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.nO,), dtype=float)
        self.reward_dim = self.nO

        # State variables
        init_user_state = self.rng.choice(self.nS_user, p=initial_distribution) if initial_distribution is not None else 0
        self.tiredness, self.time, self.motivation = utils.idx_to_state(init_user_state, num_feats=3, num_counts=0, num_vals=3, max_count=0)
        self.counts = [0] * num_clusters  # count for each cluster

        # MOMDP components
        self.P_user = transition_probs  # transition probabilities from user state and action to next user state
        self.R = reward_matrix  # reward matrix that gives the reward vector for each full (u, c) state and action 
        self.P_comp = completion_probs  # completion probabilities for each user state and action

        # Additional challenge information
        self.challenge_info = challenge_info  # list of dictionaries: for each challenge, its corresponding cluster, time required, likability, perceived usefulness, expert score
        self.challenges_per_cluster = challenges_per_cluster  # mapping from cluster index to list of challenge indices
        self.action_categories = action_categories  # mapping from action index to cluster index
        self.next_indices = utils.build_next_indices(num_clusters, self.MAX_COUNT)

        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else float('inf') # Default to infinite if not set

        # Simulation statistics
        self.counts_per_category = [0] * 4  # completion counts for each coping strategy category
        self.suggested_challenges = []  # list of suggested challenges
        self.completed_mask = []  # list of whether each suggested challenge was completed
        self.num_completed = 0
        self.t = 0


    def _get_obs(self):
        state_tuple = (self.tiredness, self.time, self.motivation, *self._get_count_state())
        return np.array(state_tuple, dtype=int)
    
    def _get_info(self):
        return {
            'tiredness': self.tiredness,
            'time': self.time,
            'motivation': self.motivation,
            'counts': self.counts,
            'num_completed': self.num_completed,
            'counts_per_category': self.counts_per_category,
            'completed_mask': self.completed_mask
        }
    
    def _get_state_index(self, full=False):
        curr_state = (self.tiredness, self.time, self.motivation)
        if full:
            return utils.state_to_idx(curr_state + self._get_count_state(), num_feats=3, num_counts=self.num_clusters, num_vals=3, max_count=self.MAX_COUNT)
        return utils.state_to_idx(curr_state, num_feats=3, num_counts=0, num_vals=3, max_count=0)
    
    def _get_state_index_factored(self):
        curr_user_state = (self.tiredness, self.time, self.motivation)
        curr_count_state = self._get_count_state()
        u, c = utils.state_to_u_c_idx(curr_user_state + curr_count_state, num_feats=3, num_counts=self.num_clusters, num_vals=3, max_count=self.MAX_COUNT)
        return u, c

    def _get_state_tuple(self, state_index, full=False):
        if full:
            return utils.idx_to_state(state_index, num_feats=3, num_counts=self.num_clusters, num_vals=3, max_count=self.MAX_COUNT)
        return utils.idx_to_state(state_index, num_feats=3, num_counts=0, num_vals=3, max_count=0)

    def _get_count_state(self):
        return tuple(min(self.counts[cat] - min(self.counts), self.MAX_COUNT) for cat in range(self.num_clusters))

    def _get_next_user_state(self, action):
        u, _ = self._get_state_index_factored()
        # transition_probs is a transition matrix of shape (num_user_states, num_actions, num_user_states) where num_user_states = 3^3 = 27
        # we differentiate between user state and count state, since we use different transition logic for user state (stochastic) and count state (semi-deterministic)
        next_state_probs = self.P_user[u, action]
        
        if next_state_probs.sum() == 0:
            # print(f"Warning: No transitions defined for state {curr_state} and action {action}. Defaulting to uniform distribution.")
            next_state_probs = np.ones(self.nS_user) / self.nS_user  # Uniform distribution

        # Sample the next state based on the transition probabilities
        next_state = self.rng.choice(len(next_state_probs), p=next_state_probs)
        return next_state
    
    def get_transition_prob(self, state, action, next_state):
        full_state = self._get_state_tuple(state, full=True)
        next_full_state = self._get_state_tuple(next_state, full=True)

        s_user = utils.state_to_idx((full_state[0], full_state[1], full_state[2]), num_feats=3, num_counts=0, num_vals=3, max_count=0)
        s_user_next = utils.state_to_idx((next_full_state[0], next_full_state[1], next_full_state[2]), num_feats=3, num_counts=0, num_vals=3, max_count=0)
        P_user_next = self.P_user[s_user, action, s_user_next]

        completed = full_state[3:] != next_full_state[3:]
        p_c = self.P_comp[s_user, action]
        P_count = p_c if completed else 1 - p_c
        return P_user_next * P_count
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Randomly initialize the user's state
        # TODO: use initial state distribution
        init_user_state = self.rng.choice(self.nS_user, p=self.initial_distribution) if self.initial_distribution is not None else 0
        self.tiredness, self.time, self.motivation = utils.idx_to_state(init_user_state, num_feats=3, num_counts=0, num_vals=3, max_count=0)
        self.num_completed = 0
        self.t = 0
        self.counts = [0] * self.num_clusters
        self.counts_per_category = [0] * 4
        self.suggested_challenges = []
        self.completed_mask = []
        return self._get_obs(), self._get_info()
    
    def step(self, action, completion_bias=False):
        self.t += 1
        u, c = self._get_state_index_factored()

        # action is a cluster, we have R based on clusters since that reduces amount of required data
        rewards = self.R[u, c, action]

        prob_done = self.P_comp[u, action]
        if completion_bias:
            bias = self.rng.uniform(1.01, 1.3)
            prob_done = np.clip(prob_done * bias, 0, 1)

        done = 1 if self.rng.random() < prob_done else 0
        self.num_completed += done

        # we have a cluster-level policy, so within that cluster we need to pick a specific challenge to present to the user
        # we pick the challenge that has lowest count in its coping strategy category, to encourage diversity
        min_count_category = np.argmin([self.counts_per_category[cat] for cat in range(4)])
        challenges_in_cluster = self.challenges_per_cluster[action]
        challenges_in_min_cat = [c for c in challenges_in_cluster if self.challenge_info[c]['category_id'] == min_count_category]
        if challenges_in_min_cat:
            challenges_not_done = [c for c in challenges_in_min_cat if c not in self.suggested_challenges]
            chosen_challenge = self.rng.choice(challenges_not_done) if challenges_not_done else self.rng.choice(challenges_in_min_cat)
        else:
            chosen_challenge = self.rng.choice(challenges_in_cluster)
        self.suggested_challenges.append(chosen_challenge)
        self.completed_mask.append(done)
        chosen_category = self.challenge_info[chosen_challenge]['category_id']
        
        self.counts_per_category[chosen_category] += done

        # update counts for the cluster of the action
        self.counts[self.action_categories[action]] += done
        
        # get next user state
        next_state_idx = self._get_next_user_state(action)
        self.tiredness, self.time, self.motivation = self._get_state_tuple(next_state_idx)
        
        terminated = False
        truncated = self.t >= self._max_episode_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info
    
    def render(self):
        print(f"Timestep: {self.t}")
        print(f"State: {self._get_state_tuple(self._get_state_index())}")
        print(f"Counts: {self._get_count_state()}")