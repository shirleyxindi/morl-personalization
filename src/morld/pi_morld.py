import pandas as pd
import numpy as np
import os
from env.user_sim import UserSimEnv
import env.simulate as sim_utils
from morl_baselines.common.pareto import ParetoArchive, get_non_pareto_dominated_inds
from morl_baselines.common.weights import random_weights
import pandas as pd
import utils
from morld.mo_pi import MOPolicyIteration
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from moocore import hypervolume, hv_contributions
from morl_baselines.common.utils import nearest_neighbors

class PIMORLD():
    def __init__(self, 
                 env, 
                 pop_size=20, 
                 gamma=0.7, 
                 max_eval_iters=500, 
                 scalarization='linear', 
                 ref_point=None, 
                 weights_init='uniform', 
                 seed=42):
        self.env = env
        self.nO = env.nO
        self.pop_size = pop_size
        self.gamma = gamma
        self.max_eval_iters = max_eval_iters
        self.scalarization = scalarization
        self.ref_point = ref_point
        self.weights_init = weights_init
        self.random_state = np.random.default_rng(seed)
        self.agents = []
        self.pareto_archive = ParetoArchive()  # Initialize Pareto archive with a maximum size of 20

        # Initialize agents and Pareto archive
        self.intialize_agents()

    def intialize_agents(self):
        if self.weights_init == 'random':
            all_weights = random_weights(self.nO, self.pop_size, rng=self.random_state)
        elif self.weights_init == 'uniform':
            all_weights = get_reference_directions("energy", self.nO, self.pop_size, seed=1)
        elif self.weights_init == 'combined':
            uniform_weights = get_reference_directions("energy", self.nO, self.pop_size//2, seed=1)
            random_weights_arr = random_weights(self.nO, self.pop_size//2, rng=self.random_state)
            all_weights = np.vstack((uniform_weights, random_weights_arr))
        print("Initial weights for agents:")
        print(all_weights)

        for i in range(self.pop_size):
            weigths = all_weights[i]
            pi_agent = MOPolicyIteration(id=i, env=self.env, weights=weigths, gamma=self.gamma, ref_point=self.ref_point, scalarization=self.scalarization)
            pi_agent.train(max_eval_iters=self.max_eval_iters, verbose=True)
            pi_agent.evaluate()
            self.agents.append(pi_agent)
            self.add_to_pareto_archive(pi_agent)

    def add_to_pareto_archive(self, agent):
        for ind in self.pareto_archive.individuals:
            if np.array_equal(ind['policy'], agent.policy_table):
                print("Policy already in archive, skipping addition.")
                return
        individual = {
            'weights': agent.weights.copy(),
            'policy': agent.policy_table.copy(),
            'V': agent.V_full.copy()
        }
        self.pareto_archive.add(individual, agent.expected_return)

    def adapt_weights_random(self, agents_to_adapt, num_iterations=2):
        for i in range(num_iterations):
            print("Weight adaptation cycle: ", i+1)
            for agent in agents_to_adapt:
                # add some random noise to the weights, ensure positivity and normalization
                noise = self.random_state.normal(1, 0.1, size=agent.weights.shape)
                new_weights = agent.weights * noise
                # normalize the weights
                new_weights /= np.sum(new_weights)
                agent.set_weights(new_weights)
                self.add_to_pareto_archive(agent)

    def closest_non_dominated(self, eval_policy: np.ndarray):
        """Returns the closest policy to eval_policy currently in the Pareto Archive.

        Args:
            eval_policy: evaluation where we want to find the closest one
            pareto_archive: the Pareto archive to search in
        Return:
            closest individual and evaluation in the pareto archive
        """
        evals = np.array(self.pareto_archive.evaluations)
        distances = np.sum(np.square(evals - eval_policy), axis=1)
        mask = distances > 0.01
        
        if not np.any(mask):
            return None, None

        masked_distances = np.where(mask, distances, np.inf)
        idx = np.argmin(masked_distances)
        return self.pareto_archive.individuals[idx], evals[idx]

    def adapt_weights_psa(self, agents_to_adapt, delta=0.2):   
        # P. Czyzżak and A. Jaszkiewicz,
        # "Pareto simulated annealing—a metaheuristic technique for multiple-objective combinatorial optimization,"
        # Journal of Multi-Criteria Decision Analysis, vol. 7, no. 1, pp. 34–47, 1998,
        # doi: 10.1002/(SICI)1099-1360(199801)7:1<34::AID-MCDA161>3.0.CO;2-6.
        for i, agent in enumerate(agents_to_adapt):
            eval_policy = agent.expected_return
            closest_nd, closest_eval = self.closest_non_dominated(eval_policy)
            new_weights = agent.weights.copy()
            if closest_eval is not None:
                for i in range(len(eval_policy)):
                    # Increases on the weights which are better than closest_eval, decreases on the others
                    if eval_policy[i] >= closest_eval[i]:
                        new_weights[i] = agent.weights[i] * (1 + delta)
                    else:
                        new_weights[i] = agent.weights[i] / (1 + delta)
            # Renormalizes so that the weights sum to 1.
            normalized = np.array(new_weights) / np.linalg.norm(np.array(new_weights), ord=1)
            agent.set_weights(normalized)
            self.add_to_pareto_archive(agent)

    def agent_from_archive(self, individual, ind_eval):
        agent = MOPolicyIteration(id=0, env=self.env, weights=individual['weights'].copy(), gamma=self.gamma, ref_point=self.ref_point)
        agent.policy_table = individual['policy'].copy()
        agent.V_full = individual['V'].copy()
        agent.expected_return = ind_eval.copy()
        return agent

    def generate_new_agents(self, num_new_agents):
        new_population = []
        new_weights = random_weights(self.nO, num_new_agents, rng=self.random_state)

        # for each new agent, we intialize it with new weights but the value function from the closest in weights from the population
        dist_metric = lambda a, b: np.sum(np.square(a - b))
        closest = [nearest_neighbors(1, w, [agent.weights for agent in self.agents], dist_metric) for w in new_weights]

        for i in range(num_new_agents):
            weigths = new_weights[i]
            V_init = self.agents[closest[i][0]].V_full.copy()
            pi_agent = MOPolicyIteration(id=i, env=self.env, weights=weigths, V_init=V_init, gamma=self.gamma, ref_point=self.ref_point, scalarization=self.scalarization)
            pi_agent.train(max_eval_iters=self.max_eval_iters)
            new_population.append(pi_agent)
            self.add_to_pareto_archive(pi_agent)
        return new_population
    
    def get_hv_ref_point(self, delta=0.1):
        evals = np.array(self.pareto_archive.evaluations)
        if len(evals) == 0:
            return np.zeros(self.nO)
        return np.min(evals, axis=0) - delta

    def select_parents(self, num_parents):
        # Select parents based on pareto dominance and hypervolume contribution
        ref_point_HV = self.get_hv_ref_point()
        non_dominated_inds = get_non_pareto_dominated_inds(np.array([agent.expected_return for agent in self.agents]))
        parents = np.array(self.agents)[non_dominated_inds]
        hv_cont = hv_contributions([agent.expected_return for agent in parents], ref_point_HV, maximise=True)
        sorted_inds = np.argsort(hv_cont)[::-1]
        parents = parents[sorted_inds]  # Reorder parents based on hypervolume contribution
        if len(parents) > num_parents:
            parents = parents[:num_parents]  # Select top num_parents based on hypervolume contribution
        return parents
    
    def train(self, num_iterations=5):
        population = self.agents.copy()
        for i in range(num_iterations):
            print(f"Iteration: {i+1}")
            print(f"HV of Pareto Archive: {self.get_hv()}")
            parents = self.select_parents(num_parents=self.pop_size//2)
            offspring = list(parents.copy())
            self.adapt_weights_psa(offspring, delta=0.2)
            population = list(np.concatenate([parents, offspring])) 
        self.agents = population

    def get_hv(self, ref_point=None):
        if ref_point is None:
            ref_point_HV = np.zeros(self.nO)
        else:
            ref_point_HV = ref_point
        return hypervolume(self.pareto_archive.evaluations, ref_point_HV, maximise=True)