import numpy as np
from morl_baselines.common.morl_algorithm import MOPolicy
import utils

class MOValueIteration(MOPolicy):
    def __init__(self, id, env, weights, gamma, **kwargs):
        super().__init__(id)
        self.weights = weights
        self.gamma = gamma

        self.env = env # Access matrices directly
        self.nS = self.env.nS
        self.nS_user = self.env.nS_user
        self.nS_count = env.nS_count
        self.nA = self.env.nA
        self.nO = self.env.nO

        self.P_user = self.env.P_user  # shape (n_u, n_A, n_u)
        self.P_comp = self.env.P_comp
        self.P_user_all = self.P_user.transpose(1, 0, 2)  # shape (n_A, n_u, n_u) 
        self.R = self.env.R
        self.next_indices = self.env.next_indices
        
        self.policy_table = np.zeros(self.nS, dtype=int)
        self.v_table = np.zeros(self.nS)
        self.v_matrix = np.zeros((self.nS_user, self.nS_count))  
        self.expected_return = np.zeros(self.nO)  # Vector of expected returns for the start state
        self.V_full = np.zeros((self.nS_user, self.nS_count, self.nO))  # Full V-table for all states and objectives

    def train(self, max_eval_iters=10):
        print("Running value iteration with current weights...", self.weights)
        self.v_matrix, self.policy_table = self.value_iteration_factored()
        self.V_full = self.policy_evaluation(max_iterations=max_eval_iters)
        self.expected_return = self.V_full[0, 0]  # Assuming start state is (User 0, Count 0)

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
            V_next_stay: (nA, n_u, n_c, nO) - values if count state stays the same
            V_next_increment: (nA, n_u, n_c, nO) - values if count state increments according to next_indices, i.e. if the action is completed
        """
        V_next_stay = np.einsum('auj,jco->auco', self.P_user_all, V_full)

        V_next_increment = np.stack([
            V_next_stay[a][:, self.next_indices[self.env.action_categories[a]]]
            for a in range(self.nA)
        ])

        return V_next_stay, V_next_increment

    def value_iteration_factored(self, theta=1e-6, max_iterations=1000):
        n_u, n_c = self.nS_user, self.nS_count

        V = self.v_matrix.copy()  # (n_u, n_c)

        # Scalarized rewards: (n_u, n_c, nA)
        R = (self.R * self.weights).sum(axis=-1).reshape(n_u, n_c, self.nA)

        # Completion probabilities: (n_u, n_c, nA, 1) for broadcasting
        P_comp = self.P_comp.reshape(n_u, n_c, self.nA)

        for _ in range(max_iterations):
            V_old = V.copy()

            # All actions at once: (nA, n_u, n_c)
            V_next_stay = np.einsum('auj,jc->auc', self.P_user_all, V_old)

            V_next_increment = np.stack([
                V_next_stay[a][:, self.next_indices[self.env.action_categories[a]]]
                for a in range(self.nA)
            ])  # (nA, n_u, n_c)

            # Rearrange to (n_u, n_c, nA) to match R and P_comp
            V_next_stay = V_next_stay.transpose(1, 2, 0)    # (n_u, n_c, nA)
            V_next_increment = V_next_increment.transpose(1, 2, 0)  # (n_u, n_c, nA)

            # Q: (n_u, n_c, nA)
            Q = R + self.gamma * ((1 - P_comp) * V_next_stay + P_comp * V_next_increment)

            V = np.max(Q, axis=-1)

            if np.max(np.abs(V - V_old)) < theta:
                break

        return V, np.argmax(Q, axis=-1).flatten()
    
    def policy_evaluation(self, theta=1e-4, max_iterations=10):
        """
        Fully vectorized policy evaluation.
        No loops over states or actions.
        """
        n_u, n_c = self.nS_user, self.nS_count

        R = self.R.reshape(n_u, n_c, self.nA, self.nO)
        P_comp = self.P_comp.reshape(n_u, n_c, self.nA)
        pi_2d = self.policy_table.reshape(n_u, n_c)  # (n_u, n_c) - action at each state

        # Gather per-state quantities using the policy — no u,c loops
        idx_u = np.arange(n_u)[:, np.newaxis]              # (n_u, 1)
        idx_c = np.arange(n_c)[np.newaxis, :]              # (1, n_c)

        R_pi = R[idx_u, idx_c, pi_2d]           # (n_u, n_c, nO) - rewards for the action chosen by the policy at each state
        P_comp_pi = P_comp[idx_u, idx_c, pi_2d]      # (n_u, n_c) - completion probabilities for the action chosen by the policy at each state
        p_c = P_comp_pi[:, :, np.newaxis]       # (n_u, n_c, 1) - reshape for broadcasting

        V_new = self.V_full.copy()  # warm start with current value function

        for _ in range(max_iterations):
            V_old = V_new.copy()

            # Compute next-state values for all actions simultaneously
            V_next_stay, V_next_increment = self.compute_transitions(V_old)

            # Select the values for the action chosen by the policy at each state
            V_stay_pi = V_next_stay[pi_2d, idx_u, idx_c]         # (n_u, n_c, nO)
            V_inc_pi  = V_next_increment[pi_2d, idx_u, idx_c]    # (n_u, n_c, nO)

            # Bellman update for all states at once
            V_new = R_pi + self.gamma * ((1 - p_c) * V_stay_pi + p_c * V_inc_pi)

            if np.max(np.abs(V_new - V_old)) < theta:
                break

        return V_new
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the heavy environment and matrices before deepcopy/pickle
        # MORLD will still have access to them via the population refs
        keys_to_del = ['env', 'P_user', 'R', 'P_comp', 'next_indices']
        for key in keys_to_del:
            if key in state:
                del state[key]
        return state