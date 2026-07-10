"""
Vectorized multi-objective policy iteration for MOMDPs.

Author: Shirley Li
Date: July 2026
"""

import numpy as np

class MOPolicyIteration():
    """Solve a MOMDP with policy iteration."""

    def __init__(self, P, R, weights, V_init=None, gamma=0.7, scalarization='linear', 
                 ref_point=None, initial_distribution=None):
        """Initialize MOMDP components and policy/value state.

        Args:
            P: Transition probabilities with shape (nS, nA, nS).
            R: Reward tensor with shape (nS, nA, nO).
            weights: Scalarization weights over objectives.
            V_init: Optional initial value function.
            gamma: Discount factor.
            scalarization: Scalarization type, typically `linear` or `chebyshev`.
            ref_point: Optional reference point for Chebyshev scalarization.
            initial_distribution: Optional initial-state distribution.
        """
        self.weights = weights
        self.gamma = gamma
        self.scalarization = scalarization
        self.ref_point = ref_point

        self.nS = P.shape[0]
        self.nA = P.shape[1]
        self.nO = R.shape[-1]

        self.P = P  # shape (nS, nA, nS)
        self.P_all = P.transpose(1, 0, 2)  # (nA, nS, nS)
        self.R = R  # shape (nS, nA, nO)
        self.initial_distribution = initial_distribution if initial_distribution is not None else np.ones(self.nS) / self.nS

        self.policy_table = np.zeros(self.nS, dtype=np.int8)
        self.V_scalarized = np.zeros(self.nS)
        self.expected_return = np.zeros(self.nO)

        if V_init is not None:
            self.V_full = V_init
        else:
            self.V_full = np.zeros((self.nS, self.nO))

    def eval(self, obs):
        """Map an observation or state index to the current greedy action.

        Args:
            obs: Observation array or already-computed state index.

        Returns:
            Integer action index selected by the current policy.
        """
        if isinstance(obs, np.ndarray):
            state_idx = self.env.unwrapped.get_full_state_index(obs)
        else:
            state_idx = obs
        return self.policy_table[state_idx]

    def update(self):
        """Retrain the policy using the current settings."""
        self.train()

    def set_weights(self, weights, max_eval_iters=10):
        """Replace the scalarization weights and retrain the policy.

        Args:
            weights: New scalarization weights.
            max_eval_iters: Maximum policy-evaluation iterations per update.
        """
        self.weights = weights
        self.train(max_eval_iters=max_eval_iters)

    def compute_next_state_values(self, V_full):
        """Compute one-step lookahead values for every action and state.

        Args:
            V_full: State-value tensor with shape (nS, nO).

        Returns:
            Array with shape (nA, nS, nO) containing next-state values.
        """
        # P_all is (nA, nS, nS), V_full is (nS, nO)
        return self.P_all @ V_full  # (nA, nS, nO)

    def policy_evaluation(self, theta=1e-6, max_iterations=1000):
        """Evaluate the current policy until the value function converges.

        Args:
            theta: Convergence tolerance.
            max_iterations: Maximum Bellman update iterations.

        Returns:
            Updated state-value tensor with shape (nS, nO).
        """
        pi = self.policy_table  # (nS,)
        idx = np.arange(self.nS)

        R_pi = self.R[idx, pi]  # (nS, nO) - reward for each state under current policy
        
        V_new = self.V_full.copy()
        for _ in range(max_iterations):
            V_old = V_new.copy()

            # Compute next-state values for all actions
            V_next = self.compute_next_state_values(V_old)  # (nA, nS, nO)

            # Next-state values under current policy
            V_next_pi = V_next[pi, idx]                      # (nS, nO)

            # Bellman update
            V_new = R_pi + self.gamma * V_next_pi  

            if np.max(np.abs(V_new - V_old)) < theta:
                break

        self.V_full = V_new
        return V_new

    def policy_improvement(self, theta=1e-6):
        """Improve the policy by greedily maximizing scalarized Q-values.

        Args:
            theta: Tolerance used to decide whether the policy is stable.

        Returns:
            Tuple of (policy_stable, scalarized_Q_values).
        """
        V_next = self.compute_next_state_values(self.V_full)  # (nA, nS, nO)
        V_next = V_next.transpose(1, 0, 2)                    # (nS, nA, nO)

        Q = self.R + self.gamma * V_next  # (nS, nA, nO)

        if self.scalarization == 'chebyshev' and self.ref_point is not None:
            Q_scalar = np.max(self.weights * np.abs(Q - self.ref_point), axis=-1)  # (nS, nA)
        else:
            Q_scalar = Q @ self.weights  # (nS, nA)

        new_pi = np.argmax(Q_scalar, axis=-1).astype(np.int8)

        idx = np.arange(self.nS)
        old_q = Q_scalar[idx, self.policy_table]
        new_q = Q_scalar[idx, new_pi]
        policy_stable = np.all(np.abs(new_q - old_q) < theta)
        self.policy_table = new_pi
        return policy_stable, Q_scalar

    def policy_iteration(self, max_iterations=100, max_eval_iters=500, theta=1e-6, verbose=False):
        """Alternate evaluation and improvement until the policy stabilizes.

        Args:
            max_iterations: Maximum policy-iteration rounds.
            max_eval_iters: Maximum evaluation iterations per round.
            theta: Convergence tolerance.
            verbose: If True, print convergence information.

        Returns:
            Scalarized Q-values from the final iteration.
        """
        for i in range(max_iterations):
            self.V_full = self.policy_evaluation(max_iterations=max_eval_iters, theta=theta)
            stable, Q_scalar = self.policy_improvement(theta=theta)
            if stable:
                if verbose:
                    print(f"Policy converged after {i + 1} iterations.")
                break

        if self.scalarization == 'chebyshev' and self.ref_point is not None:
            self.V_scalarized = np.max(self.weights * np.abs(self.V_full - self.ref_point), axis=-1)
        else:
            self.V_scalarized = self.V_full @ self.weights

        self.expected_return = self.initial_distribution @ self.V_full
        return Q_scalar

    def train(self, max_iterations=100, max_eval_iters=500, theta=1e-4, verbose=False):
        """Run policy iteration with optional progress logging.

        Args:
            max_iterations: Maximum policy-iteration rounds.
            max_eval_iters: Maximum evaluation iterations per round.
            theta: Convergence tolerance.
            verbose: If True, print progress information.
        """
        if verbose:
            print("Running policy iteration with weights:", self.weights)
        self.policy_iteration(max_iterations=max_iterations, max_eval_iters=max_eval_iters, 
                              theta=theta, verbose=verbose)

    def get_optimal_Q(self, max_iterations=100, max_eval_iters=10, theta=1e-4):
        """Return the scalarized Q-values for the current problem instance.

        Args:
            max_iterations: Maximum policy-iteration rounds.
            max_eval_iters: Maximum evaluation iterations per round.
            theta: Convergence tolerance.

        Returns:
            Scalarized Q-values from policy iteration.
        """
        return self.policy_iteration(max_iterations=max_iterations, 
                                     max_eval_iters=max_eval_iters, theta=theta)

    def evaluate(self, max_eval_iters=1000):
        """Re-evaluate the current policy and refresh expected returns.

        Args:
            max_eval_iters: Maximum evaluation iterations.
        """
        self.V_full = self.policy_evaluation(max_iterations=max_eval_iters)
        if self.scalarization == 'chebyshev' and self.ref_point is not None:
            self.V_scalarized = np.max(self.weights * np.abs(self.V_full - self.ref_point), axis=-1)
        else:
            self.V_scalarized = self.V_full @ self.weights
        self.expected_return = self.initial_distribution @ self.V_full