"""
Gymnasium environment for simulating user responses to coping challenges.

Author: Shirley Li
Date: July 2026
"""

from typing import Optional
import gymnasium as gym
import numpy as np
import utils.utils as utils


class UserSimEnv(gym.Env):
    """Simulate a user moving through states and responding to coping challenges."""

    def __init__(self, 
                 transition_probs, 
                 completion_probs,
                 reward_matrix, 
                 challenge_info,
                 challenges_per_cluster,
                 reward_probs=None,
                 initial_distribution=None,
                 num_vals_per_feature=[3, 3, 3],
                 num_actions=6, 
                 num_objectives=5,  
                 num_categories=4,
                 max_episode_steps=28, 
                 mapping=None,
                 P_comp_agency=None,
                 agency_bias_params=(0.038, 0.013),
                 deviation_prob=0.36):
        """Initialize the user simulation environment.

        Args:
            transition_probs: Transition probabilities for user states (nS, nA, nS).
            completion_probs: Completion probabilities for each state-action pair (nS, nA).
            reward_matrix: Rewards matrix for each state-action pair (nS, nA, nO).
            challenge_info: Per-challenge metadata such as category and ratings.
            challenges_per_cluster: Mapping from cluster id to challenge ids.
            reward_probs: Optional reward (outcome) probability tables.
            initial_distribution: Optional initial state distribution.
            num_vals_per_feature: Number of values per state feature.
            num_actions: Number of actions in the action space.
            num_objectives: Number of reward objectives.
            num_categories: Number of coping strategy categories.
            max_episode_steps: Maximum number of timesteps per episode.
            mapping: Optional action-to-cluster/diversity level mapping table.
            P_comp_agency: Optional agency-condition completion probabilities.
            agency_bias_params: Mean and standard deviation of completion bias.
            deviation_prob: Probability of deviating from the recommended option.
        """
        super(UserSimEnv, self).__init__()
        self.nA = num_actions
        self.nO = num_objectives
        self.num_vals_per_feature = num_vals_per_feature
        self.num_user_features = len(num_vals_per_feature)
        self.nS = np.prod(num_vals_per_feature)
        self.num_categories = num_categories
        self.action_space = gym.spaces.Discrete(self.nA)
        self.initial_distribution = initial_distribution

        # Observation space is a vector of 7 features: [tiredness, time, motivation] + [count_AC, count_DS, count_PS, count_SS]
        highs = np.array(num_vals_per_feature) - 1
        self.observation_space = gym.spaces.Box(low=0, high=highs, shape=(len(highs),), dtype=int)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.nO,), dtype=float)
        self.reward_dim = self.nO

        # MOMDP components
        self.P_user = transition_probs  # transition probabilities from user state and action to next user state
        self.R = reward_matrix  # reward matrix that gives the reward vector for each full (u, c) state and action 
        self.P_comp = completion_probs  # completion probabilities for each user state and action
        self.R_probs = reward_probs
        self.P_comp_agency = P_comp_agency if P_comp_agency is not None else completion_probs  # empirical completion probabilities for each (s,a)-pair, used for completion biasing 
        
        self.agency_bias_params = agency_bias_params  # parameters for adding bias to completion probabilities in agency condition (mean, stddev)
        self.deviation_prob = deviation_prob  # probability of deviating from the recommendation in the agency condition, based on observed deviation in our data

        # Additional challenge information
        self.challenge_info = challenge_info  # list of dictionaries: for each challenge, its corresponding clusters, time required, likability, perceived usefulness
        self.challenges_per_cluster = challenges_per_cluster  # mapping from cluster index to list of challenge indices

        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else float('inf') # Default to infinite if not set

        # Initialize variables
        # State variables
        self.user_state = None  # Will be set using reset()

        # Simulation statistics
        self.counts_per_category = [0] * self.num_categories  # completion counts for each coping strategy category
        self.suggested_challenges = []  # list of suggested challenges
        self.completed_mask = []  # list of whether each suggested challenge was completed
        self.num_completed = 0
        self.t = 0

        self.mapping = mapping  # mapping from action to cluster and novelty level

        self.state_to_idx, self.idx_to_state = utils.build_state_space(num_vals_per_feature)

    def _get_obs(self):
        """Return the current observation.

        Returns:
            Current user-state observation.
        """
        return np.array(self.user_state, dtype=int)
    
    def _get_info(self, done, action_taken=None, challenge_id=None):
        """Build the info dictionary returned by environment steps.

        Args:
            done: Completion indicator for the current challenge.
            action_taken: Action index applied at this step.
            challenge_id: Challenge id selected for the step.

        Returns:
            Dictionary with user-state tracking metadata.
        """
        return {
            'user_state': self.user_state,
            'num_completed': self.num_completed,
            'counts_per_category': self.counts_per_category,
            'completed_mask': self.completed_mask,
            'completed': done,
            'action': action_taken,
            'challenge_id': challenge_id
        }
    
    def get_state_idx(self, obs):
        """Convert an observation tuple into its discrete state index.

        Args:
            obs: Observation array or tuple.

        Returns:
            Integer index for the observation in the discrete state space.
        """
        return self.state_to_idx[tuple(obs)]
    
    def _get_current_state_index(self):
        """Return the index of the current user state.

        Returns:
            Integer state index.
        """
        return self.get_state_idx(self.user_state)

    def _get_next_user_state(self, action):
        """Sample the next user state from the transition model.

        Args:
            action: Action index applied in the current state.

        Returns:
            Integer index of the next user state.
        """
        u = self._get_current_state_index()
        # transition_probs is a transition matrix of shape (num_user_states, num_actions, num_user_states) where num_user_states = 3^3 = 27
        # we differentiate between user state and count state, since we use different transition logic for user state (stochastic) and count state (semi-deterministic)
        next_state_probs = self.P_user[u, action] 

        if next_state_probs.sum() == 0:
            # print(f"Warning: No transitions defined for state {curr_state} and action {action}. Defaulting to uniform distribution.")
            next_state_probs = np.ones(self.nS) / self.nS  # Uniform distribution

        # Sample the next state based on the transition probabilities
        next_state = self.np_random.choice(len(next_state_probs), p=next_state_probs)
        return next_state
    
    def _get_observed_rewards(self, u_idx, challenge_id, action, done):
        """Sample or look up the observed reward vector for a step.

        Args:
            u_idx: Current discrete user-state index.
            challenge_id: Selected challenge id.
            action: Action index used for the step.
            done: Completion indicator for the current challenge.

        Returns:
            Reward vector for the step.
        """
        
        if self.R_probs is not None:
            rewards = np.zeros(5)
            for o_idx, reward_col in enumerate(["likedness", "usefulness", "PAY_next"]):
                reward_probs = self.R_probs[reward_col] 
                reward_values = np.arange(reward_probs.shape[2])  # Assuming reward values are 0, 1, ..., num_reward_values-1
                probs = reward_probs[u_idx, action]
                sampled_reward = self.np_random.choice(reward_values, p=probs) + 1
                mask = True if reward_col == "PAY_next" else done  # Only consider the reward if the challenge was completed, except for return willingness. Otherwise set to 0
                rewards[o_idx] = sampled_reward * mask
            rewards[-2] = done  # adherence
            rewards[-1] = utils.calculate_shannon_diversity(self.counts_per_category)  # diversity across completed challenges
        else:
            rewards = self.R[u_idx, action]

        return rewards

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options accepted by Gymnasium.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        init_user_state = self.np_random.choice(self.nS, p=self.initial_distribution) if self.initial_distribution is not None else 0
        self.user_state = self.idx_to_state[init_user_state]
        self.num_completed = 0
        self.t = 0
        self.counts_per_category = [0] * self.num_categories
        self.suggested_challenges = []
        self.completed_mask = []
        return self._get_obs(), self._get_info(0)
    
    def get_specific_challenge(self, challenges_in_cluster, novelty_level):
        """Choose a concrete challenge within a cluster.

        Args:
            challenges_in_cluster: Candidate challenge ids for the cluster.
            novelty_level: Desired diversity level, or None to ignore this.

        Returns:
            Selected challenge id.
        """
        min_count, max_count = min(self.counts_per_category), max(self.counts_per_category)

        if novelty_level == 1:
            target_categories = [i for i in range(self.num_categories) if self.counts_per_category[i] == min_count]  # least done categories
        else:
            target_categories = [i for i in range(self.num_categories) if self.counts_per_category[i] != min_count]  # other categories

        selected_challenges = [c for c in challenges_in_cluster if self.challenge_info[c]['category_id'] in target_categories]

        if not selected_challenges or novelty_level is None:  # if no challenges in the target categories or no novelty level specified, fall back to any challenge in the cluster
            selected_challenges = challenges_in_cluster

        # Pick a challenge that has not been done before if possible
        completed_challenges = [c for c, done in zip(self.suggested_challenges, self.completed_mask) if done]
        challenges_not_done = [c for c in selected_challenges if c not in completed_challenges]
        return self.np_random.choice(challenges_not_done) if challenges_not_done else self.np_random.choice(selected_challenges)
    

    def step(self, action, random_within_cluster=False):
        """Execute one step with a single policy action.

        Args:
            action: Action index specifying the cluster and diversity level.
            random_within_cluster: If True, sample any challenge from the cluster.

        Returns:
            Tuple of (observation, rewards, terminated, truncated, info).
        """
        cluster_id = self.mapping.loc[action]['cluster_all'] if self.mapping is not None else action
        novelty_level = self.mapping.loc[action]['a_novelty'] if self.mapping is not None else None
        
        self.t += 1
        u = self._get_current_state_index()

        prob_done = self.P_comp[u, action]
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
        """Execute one step when the user chooses among multiple actions.

        Args:
            actions: List of action indices proposed by the policy.
            user_type: Choice model used to select among the actions.
            completion_bias: If True, apply the agency completion bias.

        Returns:
            Tuple of (observation, rewards, terminated, truncated, info).
        """
        candidates = []
        for action in actions:
            cluster_id = self.mapping.loc[action]['cluster_all'] if self.mapping is not None else action
            novelty = self.mapping.loc[action]['a_novelty'] if self.mapping is not None else 0
            
            challenges_in_cluster = self.challenges_per_cluster[cluster_id]
            chosen_challenge = self.get_specific_challenge(challenges_in_cluster, novelty)
            
            prob_success = self.P_comp[self._get_current_state_index(), action]
            prob_success_agency = self.P_comp_agency[self._get_current_state_index(), action] if self.P_comp_agency is not None else prob_success
            
            candidates.append({
                'action_idx': action,
                'challenge_id': chosen_challenge,
                'category_id': self.challenge_info[chosen_challenge]['category_id'],
                'prob_success': prob_success,
                'novelty': novelty, 
                'likedness': self.challenge_info[chosen_challenge]['likedness'],
                'usefulness': self.challenge_info[chosen_challenge]['usefulness'],
                'difficulty': self.challenge_info[chosen_challenge]['difficulty'], 
                'prob_success_agency': prob_success_agency
            })

        selected_challenge = self._simulate_user_choice(candidates, user_type)

        prob_done = selected_challenge['prob_success']
        if completion_bias:  
            prob_done = np.clip(prob_done + self.np_random.normal(*self.agency_bias_params), 0, 1)
            
        done = 1 if self.np_random.random() < prob_done else 0

        return self._finalize_step(selected_challenge, done)

    def _simulate_user_choice(self, candidates, user_type):
        """Select one candidate according to the requested user choice model.

        Args:
            candidates: Candidate challenge dictionaries.
            user_type: Choice model name.

        Returns:
            One selected candidate dictionary.
        """
        if user_type == 'max_fun':
            # Prioritize high novelty/excitement
            return max(candidates, key=lambda x: x['likedness'])
        
        elif user_type == 'max_usefulness':
            return max(candidates, key=lambda x: x['usefulness'])
        
        elif user_type == 'random':
            return self.np_random.choice(candidates)
        
        # multinomial logit choice model
        # we use the completion probability as the utility for the choice model
        elif user_type == 'most_likely':
            prob_successes = np.array([c['prob_success_agency'] for c in candidates])
            return self.np_random.choice(candidates, p=np.exp(prob_successes)/sum(np.exp(prob_successes)))
        
        # utility based on usefulness instead of completion probability in agency condition
        elif user_type == 'most_likely_usefulness':
            utilities = np.array([c['usefulness'] for c in candidates])
            return self.np_random.choice(candidates, p=np.exp(utilities)/sum(np.exp(utilities)))
        
        elif user_type == 'informed_most_likely':
            # this user is aware of the best option from expert's perspective
            # we simulate if users are informed of this utility
            deviation_prob = self.deviation_prob 
            # if all candidates are similarly in diversity, don't apply deviation, since there is no 'expert recommendation'
            if np.all([c['novelty'] == 1 for c in candidates]) or np.all([c['novelty'] == 0 for c in candidates]):
                prob_successes = np.array([c['prob_success_agency'] for c in candidates])
                return self.np_random.choice(candidates, p=np.exp(prob_successes)/sum(np.exp(prob_successes)))
            
            elif np.random.rand() < deviation_prob:
                non_novelty_candidates = [c for c in candidates if c['novelty'] != 1]
                prob_successes = np.array([c['prob_success_agency'] for c in non_novelty_candidates])
                return self.np_random.choice(non_novelty_candidates, p=np.exp(prob_successes)/sum(np.exp(prob_successes)))
            else:
                novelty_candidates = [c for c in candidates if c['novelty'] == 1]
                prob_successes = np.array([c['prob_success_agency'] for c in novelty_candidates])
                return self.np_random.choice(novelty_candidates, p=np.exp(prob_successes)/sum(np.exp(prob_successes)))
                
        
    def _finalize_step(self, selected_challenge, done):
        """Update episode state and return the Gymnasium step tuple.

        Args:
            selected_challenge: Dictionary describing the chosen challenge.
            done: Completion indicator for the current challenge.

        Returns:
            Tuple of (observation, rewards, terminated, truncated, info).
        """
        action_taken = selected_challenge['action_idx']
        challenge_id = selected_challenge['challenge_id']
        category_id = selected_challenge['category_id']
        
        # Internal Tracking
        self.counts_per_category[category_id] += done
        self.suggested_challenges.append(challenge_id)
        self.completed_mask.append(bool(done))
        self.num_completed += done

        u = self._get_current_state_index()
        rewards = self._get_observed_rewards(
            u, challenge_id, action_taken, done
        )

        next_state_idx = self._get_next_user_state(action_taken)
        self.user_state = self.idx_to_state[next_state_idx]

        terminated = False
        truncated = self.t >= self._max_episode_steps

        return self._get_obs(), rewards, terminated, truncated, self._get_info(done, action_taken, challenge_id)
    
    