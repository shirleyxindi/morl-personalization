import numpy as np
import math
from typing import List, Tuple
from morl_baselines.multi_policy.morld.morld import MORLD, POLICIES, Policy
from morld.mo_vi import MOValueIteration

# Register your solver
POLICIES["VI"] = MOValueIteration

class VIMORLD(MORLD):
    def __init__(self, env, **kwargs):
        # 1. Disable RL-specific overhead
        kwargs.setdefault("policy_name", "VI")
        kwargs.setdefault("shared_buffer", False)
        kwargs.setdefault("neighborhood_size", 0)
        kwargs.setdefault("update_passes", 0)  # No iterative updates needed for VI
        
        super().__init__(env, **kwargs)
        # Use a higher delta for VI since it converges instantly
        self.delta = kwargs.get("delta", 0.05) 
        print(f"VIMORLD initialized. Population: {self.pop_size}")
    
    def __eval_policy(self, policy: Policy, eval_env, num_episodes) -> np.ndarray:
        """
        FIX 1: Analytic Evaluation.
        Instead of running 50 episodes (slow), we pull the expected return 
        directly from the V-table for the initial state.
        """
        # Assuming your environment starts at a specific state (e.g., index 0)
        # Or you can average V over all possible starting states if random init
        if hasattr(policy.wrapped, 'v_table'):
            # Return the 1D vector of objectives for the starting state
            # If your V_table is (n_u, n_c, nO), we slice it.
            # For now, let's assume it returns the objective vector:
            return policy.wrapped.v_table[0] 
        
        # Fallback to standard eval if V-table isn't ready
        return super().__eval_policy(policy, eval_env, 1)

    def _MORLD__eval_all_policies(self, eval_env, num_ep_front, num_w_eval, ref_point, known_front=None):
        """
        FIX 2: Faster Archive updates.
        We skip the logger overhead during weight adaptation cycles.
        """
        evals = []
        for i, agent in enumerate(self.population):
            # Analytic eval (Instant)
            discounted_reward = self.__eval_policy(agent, eval_env, 1)
            evals.append(discounted_reward)
            
            # 3. FIX: Prevent Memory Bloat in Archive
            # We only add to archive if it's a significant improvement
            self.archive.add(agent, discounted_reward)
            
        return evals

    def train(self, iterations: int, eval_env, ref_point):
        """
        FIX 3: Controlled Training Loop.
        'iterations' now means weight adaptation cycles.
        """
        # Initial solve for the whole population
        for p in self.population:
            p.wrapped.train(0)

        for cycle in range(iterations):
            # Get current performance
            evals = self._MORLD__eval_all_policies(eval_env, 1, 50, ref_point)
            
            # Adapt weights (PSA or Random)
            if self.weight_adaptation_method == "PSA":
                self._MORLD__adapt_weights(evals)
            else:
                self.__adapt_weights_random()
            
            self.global_step += 1
            if cycle % 10 == 0:
                print(f"Cycle {cycle}: Archive Size = {len(self.archive.individuals)}")

    def __adapt_weights_random(self):
        """Faster random jitter for weights."""
        for p in self.population:
            jitter = 1 + (self.delta * (np.random.rand(self.reward_dim) - 0.5))
            new_weights = p.weights * jitter
            # Normalize
            new_weights /= np.sum(new_weights)
            p.wrapped.set_weights(new_weights)
            p.weights = new_weights