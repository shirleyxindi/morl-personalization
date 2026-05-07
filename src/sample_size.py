import numpy as np
from collections import defaultdict

def create_random_mdp(num_states, num_actions, reward_noise=0.1):
    # Transition probabilities: shape (S, A, S)
    P = np.random.dirichlet(np.ones(num_states), size=(num_states, num_actions))
    
    # Mean rewards: shape (S, A)
    R_mean = np.random.uniform(0, 1, size=(num_states, num_actions))

    def sample_reward(s, a):
        return R_mean[s, a] + np.random.normal(0, reward_noise)

    return P, R_mean, sample_reward

def sample_transition(P, s, a):
    return np.random.choice(len(P), p=P[s, a])

def estimate_model(samples, num_states):
    transition_counts = defaultdict(lambda: np.zeros(num_states))
    reward_sums = defaultdict(float)
    counts = defaultdict(int)

    for s, a, r, s_next in samples:
        transition_counts[(s, a)][s_next] += 1
        reward_sums[(s, a)] += r
        counts[(s, a)] += 1

    est_P = {}
    est_R = {}

    for key in counts:
        est_P[key] = transition_counts[key] / counts[key]
        est_R[key] = reward_sums[key] / counts[key]

    return est_P, est_R

def compute_errors(est_P, est_R, true_P, true_R_mean):
    p_error = 0.0
    r_error = 0.0
    n = 0

    num_states = true_P.shape[0]
    num_actions = true_P.shape[1]

    for s in range(num_states):
        for a in range(num_actions):
            key = (s, a)
            if key in est_P:
                p_error += np.linalg.norm(est_P[key] - true_P[s, a], ord=1)
                r_error += abs(est_R[key] - true_R_mean[s, a])
                n += 1

    return p_error / n, r_error / n

def run_experiment(num_states, num_actions, sample_sizes, trials=10):
    true_P, true_R_mean, sample_reward = create_random_mdp(num_states, num_actions)
    
    results = []

    for N in sample_sizes:
        p_errors = []
        r_errors = []

        for _ in range(trials):
            samples = []

            for _ in range(N):
                s = np.random.randint(num_states)
                a = np.random.randint(num_actions)
                s_next = sample_transition(true_P, s, a)
                r = sample_reward(s, a)
                samples.append((s, a, r, s_next))

            est_P, est_R = estimate_model(samples, num_states)
            p_err, r_err = compute_errors(est_P, est_R, true_P, true_R_mean)

            p_errors.append(p_err)
            r_errors.append(r_err)

        results.append({
            "N": N,
            "P_error_mean": np.mean(p_errors),
            "P_error_std": np.std(p_errors),
            "R_error_mean": np.mean(r_errors),
            "R_error_std": np.std(r_errors),
        })

    return results


# ---- Run example ----
if __name__ == "__main__":
    num_states = 27
    num_actions = 5
    sample_sizes = [100, 500, 1000, 5000, 10000, 50000]

    results = run_experiment(num_states, num_actions, sample_sizes)

    for r in results:
        print(
            f"N={r['N']:6d} | "
            f"P_err={r['P_error_mean']:.4f}±{r['P_error_std']:.4f} | "
            f"R_err={r['R_error_mean']:.4f}±{r['R_error_std']:.4f}"
        )