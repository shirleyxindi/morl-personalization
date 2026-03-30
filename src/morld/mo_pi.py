import numpy as np
from morl_baselines.common.morl_algorithm import MOPolicy
import utils

class MOPolicyIteration(MOPolicy):
    def __init__(self, id, env, weights, gamma, max_iters_train, **kwargs):
        super().__init__(id)
        self.weights = weights
        self.gamma = gamma
        self.max_iters_train = max_iters_train

        self.env = env # Access matrices directly
        self.nS = self.env.nS
        self.nS_user = self.env.nS_user
        self.nS_count = env.nS_count
        self.nA = self.env.nA
        self.nO = self.env.nO

        self.P_user = self.env.P_user
        self.P_comp = self.env.P_comp
        self.P_user_all = self.P_user.transpose(1, 0, 2) 
        self.R = self.env.R
        self.next_indices = self.env.next_indices
        
        self.policy_table = np.zeros(self.nS, dtype=np.int8)
        self.v_matrix = np.zeros((self.nS_user, self.nS_count))  
        self.expected_return = np.zeros(self.nO)  # Vector of expected returns for the start state
        self.V_full = np.zeros((self.nS_user, self.nS_count, self.nO))  # Full V-table for all states and objectives

    def eval(self, obs, w=None):
        """
        MORLD calls this with (obs, weights).
        We use w=None as a default to keep it flexible.
        """
        if isinstance(obs, np.ndarray):
            state_idx = utils.state_to_idx(obs, max_count=self.env.MAX_COUNT)  
        else:
            state_idx = obs
            
        return self.policy_table[state_idx]

    def update(self):
        self.train()

    def set_weights(self, weights: np.ndarray):
        self.weights = weights
        # Re-solve the MDP with the new weight profile instantly
        self.train()
        print(f"Updated weights to {weights}, re-computed policy.")

    def compute_transitions(self, V_full):
        """
        Computes next-state values for ALL actions.
        Returns:
            V_next_stay: (nA, n_u, n_c, nO)
            V_next_increment: (nA, n_u, n_c, nO)
        """
        V_next_stay = np.einsum('auj,jco->auco', self.P_user_all, V_full)

        V_next_increment = np.stack([
            V_next_stay[a][:, self.next_indices[self.env.action_categories[a]]]
            for a in range(self.nA)
        ])

        return V_next_stay, V_next_increment


    def policy_evaluation(self, theta=1e-4, max_iterations=1000):
        """
        Fully vectorized policy evaluation.
        No loops over states or actions.
        """
        n_u, n_c = self.nS_user, self.nS_count

        R       = self.R.reshape(n_u, n_c, self.nA, self.nO)
        P_comp  = self.P_comp.reshape(n_u, n_c, self.nA)
        pi_2d   = self.policy_table.reshape(n_u, n_c)

        # Gather per-state quantities using the policy — no u,c loops
        idx_u = np.arange(n_u)[:, None]              # (n_u, 1)
        idx_c = np.arange(n_c)[None, :]              # (1, n_c)

        R_pi      = R[idx_u, idx_c, pi_2d]           # (n_u, n_c, nO)
        P_comp_pi = P_comp[idx_u, idx_c, pi_2d]      # (n_u, n_c)
        p_c       = P_comp_pi[:, :, np.newaxis]       # (n_u, n_c, 1)

        V_new = self.V_full.copy()

        for _ in range(max_iterations):
            V_old = V_new.copy()

            # Precompute transitions for all actions using current V
            # V_next_stay, V_next_increment: (nA, n_u, n_c, nO)
            P_user_all = self.P_user.transpose(1, 0, 2)           # (nA, n_u, n_u)
            V_next_stay, V_next_increment = self.compute_transitions(V_old)

            # Select the transitions for the action chosen by the policy at each state
            # pi_2d[u,c] indexes into the action dimension
            V_stay_pi = V_next_stay[pi_2d, idx_u, idx_c]         # (n_u, n_c, nO)
            V_inc_pi  = V_next_increment[pi_2d, idx_u, idx_c]    # (n_u, n_c, nO)

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

        R      = self.R.reshape(n_u, n_c, self.nA, self.nO)
        P_comp = self.P_comp.reshape(n_u, n_c, self.nA)

        # Transitions for all actions: (nA, n_u, n_c, nO)
        P_user_all = self.P_user.transpose(1, 0, 2)               # (nA, n_u, n_u)
        V_next_stay, V_next_increment = self.compute_transitions(self.V_full)

        # Rearrange to (n_u, n_c, nA, nO) for broadcasting with R and P_comp
        V_next_stay      = V_next_stay.transpose(1, 2, 0, 3)      # (n_u, n_c, nA, nO)
        V_next_increment = V_next_increment.transpose(1, 2, 0, 3) # (n_u, n_c, nA, nO)

        p_c = P_comp[:, :, :, np.newaxis]                         # (n_u, n_c, nA, 1)

        # Q-values for all actions at once
        Q = R + self.gamma * ((1 - p_c) * V_next_stay + p_c * V_next_increment)
                                                                # (n_u, n_c, nA, nO)

        # Scalarise and take greedy action
        Q_scalar = Q @ self.weights                                # (n_u, n_c, nA)
        new_pi   = np.argmax(Q_scalar, axis=-1).astype(np.int8)   # (n_u, n_c)

        policy_stable = np.array_equal(new_pi, self.policy_table.reshape(n_u, n_c))
        self.policy_table = new_pi.flatten()
        return policy_stable


    def policy_iteration(self, max_iterations=100):
        for i in range(max_iterations):
            self.V_full = self.policy_evaluation(max_iterations=self.max_iters_train)  # Fewer iterations for faster convergence
            stable = self.policy_improvement()
            if stable:
                print(f"Policy converged after {i + 1} iterations.")
                break

        self.v_matrix        = self.V_full @ self.weights
        self.expected_return = self.V_full[0, 0]


    def train(self, total_timesteps=0):
        print("Running policy iteration with weights:", self.weights)
        self.policy_iteration()

    