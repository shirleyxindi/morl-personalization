from env.user_sim import UserSimEnv
import numpy as np

def get_action_values(env, state, V, discount_factor=0.9):
    action_values = np.zeros(env.nA)
    for a in range(env.nA):
        for s_next in range(env.nS):
            transition_prob = env.get_transition_prob(state, a, s_next)
            reward = env.reward_matrix[state, a].sum()  # Sum rewards across objectives
            action_values[a] += transition_prob * (reward + discount_factor * V[s_next])
    return action_values

def update_policy(env, policy, V, discount_factor=0.9):
    new_policy = np.zeros_like(policy)
    for s in range(env.nS):
        action_values = get_action_values(env, s, V, discount_factor)
        new_policy[s] = np.argmax(action_values)
    return new_policy

def value_iteration(env, discount_factor=0.9, theta=1e-6):
    num_states = env.nS
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            action_values = get_action_values(env, s, V, discount_factor)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = update_policy(env, np.zeros(num_states, dtype=int), V, discount_factor)
    return V, policy

def value_iteration_mo(env, weights, gamma=0.9, theta=1e-6):
    nS, nA = env.nS, env.nA
    V = np.zeros(nS)
    R = np.tensordot(env.reward_matrix, weights, axes=([2], [0]))
    
    while True:
        delta = 0
        for s in range(nS):
            v_old = V[s]
            action_values = np.zeros(nA)
            for a in range(nA):
                for s_next in range(nS):
                    p = env.get_transition_prob(s, a, s_next)
                    action_values[a] += p * (R[s, a] + gamma * V[s_next])
            
            V[s] = np.max(action_values)
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
            
    # Derive policy
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        # Re-calculate action values one last time to pick the best action
        q_s = [sum(env.get_transition_prob(s, a, sn) * (R[s, a] + gamma * V[sn]) 
                   for sn in range(nS)) for a in range(nA)]
        policy[s] = np.argmax(q_s)
        
    return V, policy