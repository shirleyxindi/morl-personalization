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

        self.P_user = self.env.P_user
        self.P_comp = self.env.P_comp
        self.R = self.env.R
        self.next_indices = self.env.next_indices
        
        self.policy_table = np.zeros(self.nS, dtype=int)
        self.v_table = np.zeros(self.nS)
        self.v_matrix = np.zeros((self.nS_user, self.nS_count))  
        self.expected_return = np.zeros(self.nO)  # Vector of expected returns for the start state
        self.V_full = np.zeros((self.nS_user, self.nS_count, self.nO))  # Full V-table for all states and objectives

    def train(self, total_timesteps=0):
        """In MORL/D, this is called during the improvement step."""
        print("Running value iteration with current weights...", self.weights)
        self.v_matrix, self.policy_table = self.value_iteration_factored()
        self.V_full = self.policy_evaluation()
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

    def value_iteration_factored(self, gamma=0.9, theta=1e-6, max_iterations=1000):
        n_u = self.nS_user
        n_c = self.nS_count
        V = self.v_matrix.copy()  # Shape (n_u, n_c)
        
        # Pre-calculate scalarized rewards R[u, c, a]
        R = (self.R * self.weights).sum(axis=2).reshape(n_u, n_c, self.nA)
        
        # Pre-extract completion probabilities p_c[u, c, a]
        # (Assuming the last objective is completion probability)
        P_comp = self.P_comp.reshape(n_u, n_c, self.nA)

        Q = np.zeros((self.nA, n_u, n_c))
        
        for _ in range(max_iterations):
            V_old = V.copy()
            
            for a in range(self.nA):
                # User state transition 
                # P_user_a is (n_u, n_u)
                P_user_a = self.P_user[:, a, :] 
                
                # Count state transition 
                # Probability of staying vs probability of incrementing
                p_c = P_comp[:, :, a] # Shape (n_u, n_c)

                k = self.env.action_categories[a]  # Category of the current action
                next_indices_k = self.next_indices[k]  # Shape (n_c,)
                
                # We calculate the expected future value for all user and count states at once
                # Values for staying in same count state
                V_next_stay = P_user_a @ V_old 
                # Values for incrementing count state
                V_next_increment = V_next_stay[:, next_indices_k]
                
                # Combine based on completion probability
                Q[a] = R[:, :, a] + gamma * ((1 - p_c) * V_next_stay + p_c * V_next_increment)

            V = np.max(Q, axis=0)
            if np.max(np.abs(V - V_old)) < theta:
                break
                
        return V, np.argmax(Q, axis=0).flatten()
    
    def policy_evaluation(self, theta=1e-4, max_iterations=1000):
        """
        Computes the expected discounted reward vector for the start state.
        Solves: V_pi = R_pi + gamma * P_pi * V_pi
        """
        n_u = self.nS_user
        n_c = self.nS_count
        # Initialize V-table for objectives: (n_u, n_c, n_objectives)
        V_objs = self.V_full.copy()  

        R = self.R.reshape(n_u, n_c, self.nA, self.nO)
        P_comp = self.P_comp.reshape(n_u, n_c, self.nA)
    
        pi_2d = self.policy_table.reshape(n_u, n_c)
        
        # R_pi shape: (n_u, n_c, nO)
        R_pi = np.zeros((n_u, n_c, self.nO))
        P_comp_pi = np.zeros((n_u, n_c))
        
        for u in range(n_u):
            for c in range(n_c):
                a = pi_2d[u, c]
                R_pi[u, c] = R[u, c, a]
                P_comp_pi[u, c] = P_comp[u, c, a] # Completion probability

        # 2. Iterative Policy Evaluation
        for _ in range(max_iterations):
            V_old = V_objs.copy()
            
            for a in range(self.nA):
                # We only update states where this action 'a' is the optimal one
                mask = (pi_2d == a)
                if not np.any(mask):
                    continue
                
                P_user_a = self.P_user[:, a, :] # (n_u, n_u)
                k = self.env.action_categories[a]  # Category of the current action
                next_indices_k = self.next_indices[k]  # Shape (n_c,)
                
                V_next_stay = P_user_a @ V_old 
                # Values for incrementing count state
                V_next_increment = V_next_stay[:, next_indices_k]
                
                p_c = P_comp_pi[:, :, np.newaxis] # (n_u, n_c, 1)
                
                # Bellman Equation for vectors: R + gamma * [ (1-p)V_stay + p*V_inc ]
                V_objs[mask] = R_pi[mask] + self.gamma * (
                    (1 - p_c[mask]) * V_next_stay[mask] + 
                    p_c[mask] * V_next_increment[mask]
                )

            if np.max(np.abs(V_objs - V_old)) < theta:
                break
        
        return V_objs
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the heavy environment and matrices before deepcopy/pickle
        # MORLD will still have access to them via the population refs
        keys_to_del = ['env', 'P_user', 'R', 'P_comp', 'next_indices']
        for key in keys_to_del:
            if key in state:
                del state[key]
        return state