from typing import Optional
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

class UserSimSkillsEnv(gym.Env):

    def __init__(self, num_states=27, num_actions=126, num_objectives=3, transition_probs=None, reward_matrix=None, expert_score_matrix=None, max_episode_steps=100):
        super(UserSimSkillsEnv, self).__init__()
        self.nA = num_actions
        self.nO = num_objectives
        self.nS = num_states
        self.expert_score_matrix = expert_score_matrix
        self.action_space = gym.spaces.Discrete(self.nA)
        # Observation space is a vector of 3 features: [tiredness, time, motivation]
        self.observation_space = gym.spaces.Dict({
            'tiredness': gym.spaces.Box(low=0, high=2, shape=(1,), dtype=int),
            'time': gym.spaces.Box(low=0, high=2, shape=(1,), dtype=int),
            'motivation': gym.spaces.Box(low=0, high=2, shape=(1,), dtype=int),
            'skill_levels': gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        self.tiredness = -1
        self.time = -1
        self.motivation = -1
        self.transition_probs = transition_probs
        self.reward_matrix = reward_matrix
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else float('inf') # Default to infinite if not set
        self.num_completed = 0
        self.completed_actions = np.zeros(num_actions, dtype=bool)
        self.skill_levels = np.zeros(4) 

        self.all_states = [(t, time, m) for t in range(3) for time in range(3) for m in range(3)]
        

    def _get_obs(self):
        return {
            'tiredness': self.tiredness,
            'time': self.time,
            'motivation': self.motivation,
            'skill_levels': self.skill_levels
        }
    
    def _get_info(self):
        return {
            'tiredness': self.tiredness,
            'time': self.time,
            'motivation': self.motivation,
            'skill_levels': self.skill_levels
        }
    
    def _get_state_index(self):
        curr_state = (self.tiredness, self.time, self.motivation)
        return self.all_states.index(curr_state)
    
    def _get_state_tuple(self, state_index):
        return self.all_states[state_index]
    
    def _get_skill_index(self):
        skill_levels = [0.0, 0.33, 0.67, 1.0]
        all_skill_levels = [[s1, s2, s3, s4] for s1 in skill_levels for s2 in skill_levels for s3 in skill_levels for s4 in skill_levels]
        skill_tiers = self._get_skill_tiers(self.skill_levels)
        return all_skill_levels.index(skill_tiers.tolist())
    
    def _get_next_state(self, action):
        curr_state = self._get_state_index()
        # transition_probs is a transition matrix of shape (num_states, num_actions, num_states) where num_states = 3^3 = 27
        next_state_probs = self.transition_probs[curr_state, action]
        
        if next_state_probs.sum() == 0:
            print(f"Warning: No transitions defined for state {curr_state} and action {action}. Defaulting to uniform distribution.")
            next_state_probs = np.ones(self.nS) / self.nS  # Uniform distribution

        # Sample the next state based on the transition probabilities
        next_state = np.random.choice(len(next_state_probs), p=next_state_probs)
        return next_state
    
    def _get_skill_tiers(self, skill_vector):
        return np.select(
                [skill_vector >= 0.75, skill_vector >= 0.50, skill_vector >= 0.25],
                [1.0, 0.67, 0.33],
                default=0.0
            )
    
    def _get_skill_reward(self, action_idx, tiers=False):
        if self.completed_actions[action_idx]:
            return 0
        
        action_scores = self.expert_score_matrix[action_idx]

        if tiers:
            prev_progress = self.skill_levels.copy()
            self.skill_levels += action_scores
            # should depend on whether they completed the action
            reward_vector = self._get_skill_tiers(self.skill_levels) - self._get_skill_tiers(prev_progress)
        else:
            self.skill_levels += action_scores
            reward_vector = action_scores
        
        self.completed_actions[action_idx] = True
        return sum(reward_vector) / 4
        
    def _get_rewards(self, action):
        curr_state = self._get_state_index()
        # reward_matrix is a matrix of shape (num_states, num_actions, num_objectives) where num_states = 3^3
        # this returns a vector of rewards for the m objectives
        curr_skill_state = self._get_skill_index()
        return self.reward_matrix[curr_state, curr_skill_state, action]
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Randomly initialize the user's state
        self.tiredness = self.np_random.integers(0, 3)
        self.time = self.np_random.integers(0, 3)
        self.motivation = self.np_random.integers(0, 3)
        self.num_completed = 0
        self.completed_actions = np.zeros(self.nA, dtype=bool)
        self.skill_levels = np.zeros(4)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # should depend on whether action is completed
        self.num_completed += 1 

        learned_rewards = self._get_rewards(action)

        prob_done = learned_rewards[-1]  # last element indicates probability of completing the action
        done = np.random.rand() < prob_done
        # reward for skill improvement, also updates internal skill levels and completed actions
        skill_rewards = done * self._get_skill_reward(action)

        rewards = np.append(learned_rewards, skill_rewards)

        next_state_idx = self._get_next_state(action)
        self.tiredness, self.time, self.motivation = self._get_state_tuple(next_state_idx)

        # action_scores = self.expert_score_matrix[action]
        # self.skill_levels += action_scores
        
        terminated = False
        truncated = self.num_completed >= self._max_episode_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info