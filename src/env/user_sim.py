from typing import Optional
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import utils

class UserSimEnv(gym.Env):

    def __init__(self, 
                 transition_probs, 
                 completion_probs,
                 reward_matrix, 
                 challenge_info,
                 challenges_per_cluster,
                 action_categories,
                 expert_ratings,
                 reward_probs=None,
                 initial_distribution=None,
                 num_vals_per_feature=[3, 3, 3],
                 num_actions=103, 
                 num_objectives=5,  
                 num_clusters=6,
                 max_episode_steps=28, 
                 max_count=2,
                 mapping=None):
        super(UserSimEnv, self).__init__()
        self.nA = num_actions
        self.nO = num_objectives
        self.num_vals_per_feature = num_vals_per_feature
        self.num_user_features = len(num_vals_per_feature)
        self.nS_user = np.prod(num_vals_per_feature)
        self.nS_count = (max_count+1)**num_clusters
        self.nS = self.nS_user * self.nS_count
        self.num_clusters = num_clusters
        self.action_space = gym.spaces.Discrete(self.nA)
        self.max_count = max_count
        self.initial_distribution = initial_distribution

        # Observation space is a vector of 7 features: [tiredness, time, motivation] + [count_AC, count_DS, count_PS, count_SS]
        highs_user = np.array(num_vals_per_feature) - 1
        highs = np.append(highs_user, [max_count] * num_clusters)
        self.observation_space = gym.spaces.Box(low=0, high=highs, shape=(len(highs),), dtype=int)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.nO,), dtype=float)
        self.reward_dim = self.nO

        # MOMDP components
        self.P_user = transition_probs  # transition probabilities from user state and action to next user state
        self.R = reward_matrix  # reward matrix that gives the reward vector for each full (u, c) state and action 
        self.P_comp = completion_probs  # completion probabilities for each user state and action
        self.R_probs = reward_probs

        # Additional challenge information
        self.challenge_info = challenge_info  # list of dictionaries: for each challenge, its corresponding clusters, time required, likability, perceived usefulness, expert scores
        self.challenges_per_cluster = challenges_per_cluster  # mapping from cluster index to list of challenge indices
        self.action_categories = action_categories  # mapping from action index to cluster index
        self.next_indices = utils.build_next_indices(num_clusters, self.max_count)
        self.expert_scores = expert_ratings  # expert ratings for each challenge

        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else float('inf') # Default to infinite if not set

        # Initialize variables
        # State variables
        self.user_state = None  # Will be set using reset()
        self.counts = [0] * num_clusters  # count for each cluster

        # Simulation statistics
        self.counts_per_category = [0] * 4  # completion counts for each coping strategy category
        self.count_state = tuple(self.counts_per_category)  # count state is a tuple of counts for each cluster, capped at max_count
        self.suggested_challenges = []  # list of suggested challenges
        self.completed_mask = []  # list of whether each suggested challenge was completed
        self.num_completed = 0
        self.t = 0
        self.expert_competencies = [0] * 4  # cumulative expert scores for each category

        self.mapping = mapping  # mapping from action to cluster and novelty level

    def _get_obs(self):
        state_tuple = self.user_state + self._get_count_state(self.counts_per_category)
        return np.array(state_tuple, dtype=int)
    
    def _get_info(self, done, action_taken=None):
        return {
            'user_state': self.user_state,
            'counts': self.counts,
            'num_completed': self.num_completed,
            'counts_per_category': self.counts_per_category,
            'completed_mask': self.completed_mask,
            'completed': done,
            'expert_competencies': self.expert_competencies,
            'action': action_taken
        }
    
    def get_full_state_index(self, obs):
        return utils.full_state_to_idx(obs, num_vals_per_feature=self.num_vals_per_feature, max_count=self.max_count)
    
    def _get_state_index_factored(self):
        curr_count_state = self._get_count_state(self.counts_per_category)
        u_idx = utils.user_state_to_idx(self.user_state, self.num_vals_per_feature)
        c_idx = utils.count_state_to_idx(curr_count_state, self.max_count)
        return u_idx, c_idx

    def _get_count_state(self, counts):
        return tuple(min(counts[cat] - min(counts), self.max_count) for cat in range(len(counts)))

    def _get_next_user_state(self, action):
        u, _ = self._get_state_index_factored()
        # transition_probs is a transition matrix of shape (num_user_states, num_actions, num_user_states) where num_user_states = 3^3 = 27
        # we differentiate between user state and count state, since we use different transition logic for user state (stochastic) and count state (semi-deterministic)
        next_state_probs = self.P_user[u, action]
        
        if next_state_probs.sum() == 0:
            # print(f"Warning: No transitions defined for state {curr_state} and action {action}. Defaulting to uniform distribution.")
            next_state_probs = np.ones(self.nS_user) / self.nS_user  # Uniform distribution

        # Sample the next state based on the transition probabilities
        next_state = self.np_random.choice(len(next_state_probs), p=next_state_probs)
        return next_state
    
    def _get_observed_rewards(self, u_idx, c_idx, challenge_id, action, done):
        # R_probs is a dictionary of reward probability matrices for each objective, where each matrix has shape (num_user_states, num_actions, num_reward_values)
        # Return vector of reward observations difficulty, fun, perceived usefulness, time, expert rating, diversity
        # TODO: add time as minutes 
        
        if self.R_probs is not None:
            rewards = np.zeros(7)
            for o_idx, reward_col in enumerate(["time_spent", "likedness", "usefulness", "difficulty", "PAY_next"]):
                reward_probs = self.R_probs[reward_col]
                reward_values = np.arange(reward_probs.shape[2])  # Assuming reward values are 0, 1, ..., num_reward_values-1
                # reward_cluster_id = self.challenge_info[challenge_id][reward_col + '_cluster'] if reward_col != 'PAY_next' else action
                reward_cluster_id = action
                probs = reward_probs[u_idx, reward_cluster_id]
                sampled_reward = self.np_random.choice(reward_values, p=probs)
                rewards[o_idx] = sampled_reward
            rewards[-2] = self.expert_scores[challenge_id] * done  # expert rating is deterministic given challenge, but only given if challenge is completed
            rewards[-1] = utils.calculate_shannon_diversity(self.counts_per_category)  # diversity across completed challenges
        else:
            rewards = self.R[u_idx, c_idx, action]

        expert_competencies_build = self.expert_scores[challenge_id] * done
        return rewards, expert_competencies_build

    def get_user_state(self, state):
        return utils.idx_to_user_state(state, self.num_vals_per_feature)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        init_user_state = self.np_random.choice(self.nS_user, p=self.initial_distribution) if self.initial_distribution is not None else 0
        self.user_state = self.get_user_state(init_user_state)
        self.num_completed = 0
        self.t = 0
        self.counts = [0] * self.num_clusters
        self.count_state = tuple(self.counts)
        self.counts_per_category = [0] * 4
        self.suggested_challenges = []
        self.completed_mask = []
        self.expert_competencies = [0] * 4
        return self._get_obs(), self._get_info(0)
    
    def get_specific_challenge(self, challenges_in_cluster, novelty_level):
        min_count, max_count = min(self.counts_per_category), max(self.counts_per_category)

        if novelty_level == 1:
            target_categories = [i for i in range(4) if self.counts_per_category[i] == min_count]  # least done categories
        else:
            target_categories = [i for i in range(4) if self.counts_per_category[i] != min_count]  # other categories

        selected_challenges = [c for c in challenges_in_cluster if self.challenge_info[c]['category_id'] in target_categories]

        if not selected_challenges or novelty_level is None:  # if no challenges in the target categories or no novelty level specified, fall back to any challenge in the cluster
            selected_challenges = challenges_in_cluster

        # Pick a challenge that has not been done before if possible
        completed_challenges = [c for c, done in zip(self.suggested_challenges, self.completed_mask) if done]
        challenges_not_done = [c for c in selected_challenges if c not in completed_challenges]
        return self.np_random.choice(challenges_not_done) if challenges_not_done else self.np_random.choice(selected_challenges)
    

    def step(self, action, completion_bias=False, random_within_cluster=False):
        # action specifies a cluster and the novelty level of the challenge 
        cluster_id = self.mapping.loc[action]['cluster_all'] if self.mapping is not None else action
        novelty_level = self.mapping.loc[action]['a_novelty'] if self.mapping is not None else None
        
        self.t += 1
        u, c = self._get_state_index_factored()

        prob_done = self.P_comp[u, action]
        # completion bias to simulate user choice
        if completion_bias:
            bias = self.np_random.uniform(1.01, 1.3)
            prob_done = np.clip(prob_done * bias, 0, 1)

        done = 1 if self.np_random.random() < prob_done else 0

        # we have a cluster-level policy, so within that cluster we need to pick a specific challenge to present to the user
        # TODO: how to deal with multiple policies giving the same coping strategy category ? 
        challenges_in_cluster = self.challenges_per_cluster[cluster_id]
        if not random_within_cluster:
            chosen_challenge = self.get_specific_challenge(challenges_in_cluster, novelty_level)
        else:
            completed_challenges = [c for c, done in zip(self.suggested_challenges, self.completed_mask) if done]
            challenges_not_done = [c for c in challenges_in_cluster if c not in completed_challenges]
            chosen_challenge = self.np_random.choice(challenges_not_done) if challenges_not_done else self.np_random.choice(challenges_in_cluster)
        selected_challenge = {
            'action_idx': action,
            'challenge_id': chosen_challenge,
            'category_id': self.challenge_info[chosen_challenge]['category_id']
        }
        
        return self._finalize_step(selected_challenge, done)
    
    def step_choice(self, actions, user_type='random', completion_bias=False):
        """
        Simulates a user selecting from a menu of actions based on a specific persona.
        
        Args:
            actions (list): List of action indices from the MORL policy.
            user_type (str): 'max_fun', 'max_usefulness', or 'random'.
        """
        candidates = []
        for action in actions:
            cluster_id = self.mapping.loc[action]['cluster_all'] if self.mapping is not None else action
            novelty = self.mapping.loc[action]['a_novelty'] if self.mapping is not None else 0
            
            challenges_in_cluster = self.challenges_per_cluster[cluster_id]
            chosen_challenge = self.get_specific_challenge(challenges_in_cluster, novelty)
            
            prob_success = self.P_comp[self._get_state_index_factored()[0], action]
            
            candidates.append({
                'action_idx': action,
                'challenge_id': chosen_challenge,
                'category_id': self.challenge_info[chosen_challenge]['category_id'],
                'prob_success': prob_success,
                'novelty': novelty, # Often correlates with 'fun'
                'likedness': self.challenge_info[chosen_challenge]['likedness'], # Or a custom skill-gain metric
                'usefulness': self.challenge_info[chosen_challenge]['usefulness'], # Or a custom utility metric
                'difficulty': self.challenge_info[chosen_challenge]['difficulty'], # Could factor into the decision for balanced users
                'completion_rate': self.challenge_info[chosen_challenge]['completion_rate'] # Could factor into the decision for balanced users
            })

        selected_challenge = self._simulate_user_choice(candidates, user_type)

        prob_done = selected_challenge['prob_success']
        if completion_bias:
            prob_done = np.clip(prob_done + self.np_random.normal(0.04, 0.0132), 0, 1)
            
        done = 1 if self.np_random.random() < prob_done else 0

        return self._finalize_step(selected_challenge, done)

    def _simulate_user_choice(self, candidates, user_type):
        """
        Models the internal decision process of the user.
        """
        if user_type == 'max_fun':
            # Prioritize high novelty/excitement
            return max(candidates, key=lambda x: x['likedness'])
        
        elif user_type == 'max_usefulness':
            return max(candidates, key=lambda x: x['usefulness'])
        
        elif user_type == 'random':
            return self.np_random.choice(candidates)
        
        
    def _finalize_step(self, selected_challenge, done):
        # Unpack for readability
        action_taken = selected_challenge['action_idx']
        challenge_id = selected_challenge['challenge_id']
        category_id = selected_challenge['category_id']
        
        # Internal Tracking
        self.counts_per_category[category_id] += done
        self.suggested_challenges.append(challenge_id)
        self.completed_mask.append(bool(done))
        
        u, c = self._get_state_index_factored()
        rewards, expert_comp_gain = self._get_observed_rewards(
            u, c, challenge_id, action_taken, done
        )
        self.expert_competencies += expert_comp_gain

        next_state_idx = self._get_next_user_state(action_taken)
        self.user_state = self.get_user_state(next_state_idx)

        # Counter Updates (Specific to the category of the challenge and the action type)
        self.counts[self.action_categories[action_taken]] += done
        self.count_state = self._get_count_state(self.counts_per_category)
        self.num_completed += done

        terminated = False
        truncated = self.t >= self._max_episode_steps

        return self._get_obs(), rewards, terminated, truncated, self._get_info(done, action_taken)
    
    