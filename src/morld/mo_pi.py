import numpy as np
from morl_baselines.common.morl_algorithm import MOPolicy
import utils

class MOPolicyIteration():
    def __init__(self, P, R, P_comp, weights, V_init=None, gamma=0.7, scalarization='linear', ref_point=None, max_count=2, initial_distribution=None, action_to_cluster=None, **kwargs):
        self.weights = weights
        self.gamma = gamma
        self.scalarization = scalarization
        self.ref_point = ref_point

        self.nS_user = P.shape[0]
        self.nS_count = R.shape[1]
        self.nS = self.nS_user * self.nS_count
        self.nA = P.shape[1]
        self.nO = R.shape[-1]

        self.action_to_cluster = action_to_cluster if action_to_cluster is not None else np.arange(self.nA)
        self.num_clusters = len(set(action_to_cluster)) if action_to_cluster is not None else self.nA

        self.P_user = P  # shape (n_u, nA, n_u)
        self.P_user_all = self.P_user.transpose(1, 0, 2)  # reindex to (nA, n_u, n_u) 
        self.P_comp = P_comp  # completion probabilities, shape (nU, nA)
        self.R = R  # reward function, shape (nU, nC, nA, nO)
        self.next_indices = utils.build_next_indices(self.num_clusters, max_count)  # pre-computed next count state indices for each action cluster and count state, shape (num_clusters, nC)
        self.initial_distribution = initial_distribution if initial_distribution is not None else np.ones(self.nS_user) / self.nS_user
        
        self.policy_table = np.zeros(self.nS, dtype=np.int8)  # Policy mapping each state to an action
        self.V_scalarized = np.zeros((self.nS_user, self.nS_count))  # Value function for all states, scalarized with current weights
        self.expected_return = np.zeros(self.nO)  # Vector of expected returns for the start state
        if V_init is not None:
            self.V_full = V_init
        else:
            self.V_full = np.zeros((self.nS_user, self.nS_count, self.nO))  # Full value function for all states and objectives

    def eval(self, obs):
        """
        MORLD calls this with (obs, weights).
        We use w=None as a default to keep it flexible.
        """
        if isinstance(obs, np.ndarray):
            state_idx = self.env.unwrapped.get_full_state_index(obs)
        else:
            state_idx = obs
            
        return self.policy_table[state_idx]

    def update(self):
        self.train()

    def set_weights(self, weights: np.ndarray, max_eval_iters=10):
        self.weights = weights
        self.train(max_eval_iters=max_eval_iters)

    def compute_next_state_values(self, V_full):
        """
        Computes next-state values for all actions given the current value function.
        Returns:
            V_next_stay: (nA, n_u, n_c, nO) - values if count state stays the same
            V_next_increment: (nA, n_u, n_c, nO) - values if count state increments according to next_indices, i.e. if the action is completed
        """
        V_next_stay = np.einsum('auj,jco->auco', self.P_user_all, V_full)
        
        # For the increment case, we need to reindex according to next_indices for each action category
        V_next_increment = np.stack([
            V_next_stay[a][:, self.next_indices[self.action_to_cluster[a]]]
            for a in range(self.nA)
        ])

        return V_next_stay, V_next_increment


    def policy_evaluation(self, theta=1e-4, max_iterations=1000):
        """
        Fully vectorized policy evaluation.
        No loops over states or actions.
        """
        n_u, n_c = self.nS_user, self.nS_count

        pi_2d = self.policy_table.reshape(n_u, n_c)  # (n_u, n_c) - action at each state

        # Gather per-state quantities using the policy — no u,c loops
        idx_u = np.arange(n_u)[:, np.newaxis]             # (n_u, 1)
        idx_c = np.arange(n_c)[np.newaxis, :]             # (1, n_c)

        R_pi = self.R[idx_u, idx_c, pi_2d]           # (n_u, n_c, nO) - rewards for the action chosen by the policy at each state
        P_comp_pi = self.P_comp[idx_u, pi_2d]          # (nU, nC) - completion probabilities for the action chosen by the policy at each state
        p_c = P_comp_pi[:, :, np.newaxis]         # (n_u, n_c, 1) - reshape for broadcasting over objectives

        V_new = self.V_full.copy()  # warm start with current value function

        for _ in range(max_iterations):
            V_old = V_new.copy()

            # Compute next-state values for all actions simultaneously
            V_next_stay, V_next_increment = self.compute_next_state_values(V_old)

            # Select the values for the action chosen by the policy at each state
            V_stay_pi = V_next_stay[pi_2d, idx_u, idx_c]         # (n_u, n_c, nO)
            V_inc_pi = V_next_increment[pi_2d, idx_u, idx_c]    # (n_u, n_c, nO)

            # Bellman update for all states at once
            V_new = R_pi + self.gamma * ((1 - p_c) * V_stay_pi + p_c * V_inc_pi)

            if np.max(np.abs(V_new - V_old)) < theta:
                break

        self.V_full = V_new
        return V_new
    
    def policy_improvement(self):        
        """
        Fully vectorized policy improvement.
        Computes Q for all actions simultaneously, no action loop.
        """
        n_u, n_c = self.nS_user, self.nS_count

        P_comp = self.P_comp[:, np.newaxis, :]  # (nU, 1, nA) for broadcasting

        # Transitions for all actions: (nA, n_u, n_c, nO)
        V_next_stay, V_next_increment = self.compute_next_state_values(self.V_full)

        # Rearrange to (n_u, n_c, nA, nO) for broadcasting with R and P_comp
        V_next_stay = V_next_stay.transpose(1, 2, 0, 3)  # (n_u, n_c, nA, nO)
        V_next_increment = V_next_increment.transpose(1, 2, 0, 3)  # (n_u, n_c, nA, nO)

        p_c = P_comp[:, :, :, np.newaxis]  # (n_u, n_c, nA, 1)

        # Q-values for all actions at once
        Q = self.R + self.gamma * ((1 - p_c) * V_next_stay + p_c * V_next_increment)  # (n_u, n_c, nA, nO)

        # Scalarise and take greedy action
        if self.scalarization == 'chebyshev' and self.ref_point is not None:
            # use chebyshev scalarization with reference point
            Q_scalar = np.max(self.weights * np.abs(Q - self.ref_point), axis=-1)  # (n_u, n_c, nA)
        else:
            Q_scalar = Q @ self.weights  # (n_u, n_c, nA)

        # Greedy action with random tie-breaking
        # max_q = np.max(Q_scalar, axis=-1, keepdims=True)
        # ties = (Q_scalar == max_q)
        # random_values = np.random.random(Q_scalar.shape) * ties * 1e-10  # Add small random noise to break ties randomly
        # new_pi = np.argmax(Q_scalar + random_values, axis=-1).astype(np.int8)
        # new_pi = np.argmax(Q_scalar, axis=-1).astype(np.int8)  # (n_u, n_c)

        # policy_stable = np.array_equal(new_pi, self.policy_table.reshape(n_u, n_c))
        # self.policy_table = new_pi.flatten()
        
        max_q = np.max(Q_scalar, axis=-1, keepdims=True)
        ties = (Q_scalar == max_q)
        random_values = np.random.random(Q_scalar.shape) * ties
        new_pi = np.argmax(Q_scalar + random_values, axis=-1).astype(np.int8)

        u_idx = np.arange(n_u)[:, None]  # (n_u, 1)
        c_idx = np.arange(n_c)[None, :]  # (1, n_c)

        old_q = Q_scalar[u_idx, c_idx, self.policy_table.reshape(n_u, n_c)]
        new_q = Q_scalar[u_idx, c_idx, new_pi]
        policy_stable = np.all(np.abs(new_q - old_q) < 1e-6)
        self.policy_table = new_pi.flatten()
        return policy_stable, Q_scalar

    def policy_iteration(self, max_iterations=100, max_eval_iters=500, theta=1e-4, verbose=False):
        for i in range(max_iterations):
            old_V = self.V_full.copy()
            self.V_full = self.policy_evaluation(max_iterations=max_eval_iters)  # Fewer iterations for faster convergence
            stable, Q_scalar = self.policy_improvement()
            # stable = np.max(np.abs(old_V - self.V_full)) < theta
            if stable:
                if verbose:
                    print(f"Policy converged after {i + 1} iterations.")
                break
        if self.scalarization == 'chebyshev' and self.ref_point is not None:
            self.V_scalarized = np.max(self.weights * np.abs(self.V_full - self.ref_point), axis=-1)
        else:
            self.V_scalarized = self.V_full @ self.weights
        self.expected_return = self.initial_distribution @ self.V_full[:, 0]
        return Q_scalar

    def train(self, max_iterations=100, max_eval_iters=500, theta=1e-4, verbose=False):
        if verbose:
            print("Running policy iteration with weights:", self.weights)
        self.policy_iteration(max_iterations=max_iterations, max_eval_iters=max_eval_iters, theta=theta, verbose=verbose)

    def get_optimal_Q(self, max_iterations=100, max_eval_iters=10, theta=1e-4):
        return self.policy_iteration(max_iterations=max_iterations, max_eval_iters=max_eval_iters, theta=theta)

    def evaluate(self, max_eval_iters=1000):
        self.V_full = self.policy_evaluation(max_iterations=max_eval_iters)
        if self.scalarization == 'chebyshev' and self.ref_point is not None:
            self.V_scalarized = np.max(self.weights * np.abs(self.V_full - self.ref_point), axis=-1)
        else:
            self.V_scalarized = self.V_full @ self.weights

        self.expected_return = self.initial_distribution @ self.V_full[:, 0]