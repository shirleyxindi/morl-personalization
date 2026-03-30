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
                 expert_score_matrix,
                 action_categories,
                 num_states=27, 
                 num_actions=103, 
                 num_objectives=5,  
                 max_episode_steps=28, 
                 MAX_COUNT=4):
        super(UserSimEnv, self).__init__()
        self.nA = num_actions
        self.nO = num_objectives
        self.nS_user = num_states
        self.nS_count = (MAX_COUNT+1)**4
        self.nS = num_states * self.nS_count
        self.action_space = gym.spaces.Discrete(self.nA)
        self.MAX_COUNT = MAX_COUNT

        # Observation space is a vector of 7 features: [tiredness, time, motivation] + [count_AC, count_DS, count_PS, count_SS]
        highs = np.array([2, 2, 2] + [MAX_COUNT]*4)
        self.observation_space = gym.spaces.Box(low=0, high=highs, shape=(7,), dtype=int)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.nO,), dtype=float)
        self.reward_dim = self.nO

        # State variables
        self.tiredness = -1
        self.time = -1
        self.motivation = -1
        self.counts = {"AC": 0, "DS": 0, "PS": 0, "SS": 0}

        self.P_user = transition_probs  # transition probabilities from partial user state and action to next partial user state
        self.R = reward_matrix  # reward matrix that gives the reward vector for each full state and action 
        self.P_comp = completion_probs  # completion probabilities for each full state and action

        self.expert_score_matrix = expert_score_matrix
        self.action_categories = action_categories
        self.next_indices = utils.build_next_indices(self.MAX_COUNT)

        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else float('inf') # Default to infinite if not set
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
            'counts': self._get_count_state()
        }
    
    def _get_state_index(self, full=False):
        curr_state = (self.tiredness, self.time, self.motivation)
        if full:
            return utils.state_to_idx(curr_state + self._get_count_state(), max_count=self.MAX_COUNT)
        return utils.state_to_idx(curr_state, max_count=0)
    
    def _get_state_tuple(self, state_index, full=False):
        if full:
            return utils.idx_to_state(state_index, max_count=self.MAX_COUNT)
        return utils.idx_to_state(state_index, max_count=0)
    
    def _get_count_state(self):
        return tuple(min(self.counts[cat] - min(self.counts.values()), self.MAX_COUNT) for cat in ["AC", "DS", "PS", "SS"])

    def _get_next_user_state(self, action):
        curr_state = self._get_state_index()
        # transition_probs is a transition matrix of shape (num_user_states, num_actions, num_user_states) where num_user_states = 3^3 = 27
        # we differentiate between user state and count state, since we use different transition logic for user state (stochastic) and count state (semi-deterministic)
        next_state_probs = self.P_user[curr_state, action]
        
        if next_state_probs.sum() == 0:
            # print(f"Warning: No transitions defined for state {curr_state} and action {action}. Defaulting to uniform distribution.")
            next_state_probs = np.ones(self.nS_user) / self.nS_user  # Uniform distribution

        # Sample the next state based on the transition probabilities
        next_state = np.random.choice(len(next_state_probs), p=next_state_probs)
        return next_state
    
    def get_transition_prob(self, state, action, next_state):
        full_state = self._get_state_tuple(state, full=True)
        next_full_state = self._get_state_tuple(next_state, full=True)

        s_user = utils.state_to_idx((full_state[0], full_state[1], full_state[2]), max_count=0)
        s_user_next = utils.state_to_idx((next_full_state[0], next_full_state[1], next_full_state[2]), max_count=0)
        P_user_next = self.P_user[s_user, action, s_user_next]

        completed = full_state[3:] != next_full_state[3:]
        p_c = self.P_comp[state, action]
        P_count = p_c if completed else 1 - p_c
        return P_user_next * P_count
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Randomly initialize the user's state
        # TODO: use initial state distribution
        self.tiredness = self.np_random.integers(0, 3)
        self.time = self.np_random.integers(0, 3)
        self.motivation = self.np_random.integers(0, 3)
        self.num_completed = 0
        self.t = 0
        self.counts = {"AC": 0, "DS": 0, "PS": 0, "SS": 0}
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.t += 1
        curr_state_idx = self._get_state_index(full=True)
        rewards = self.R[curr_state_idx, action]

        prob_done = self.P_comp[curr_state_idx, action]
        done = 1 if np.random.rand() < prob_done else 0
        self.num_completed += done
        # update counts for the category of the action
        self.counts[category_mapping[self.action_categories[action]]] += done
        
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