"""
Multi-Objective Reinforcement Learning based on Decomposition with Policy Iteration.

Adapted from Felten, Talbi & Danoy (2024) to a tabular, stochastic, model-based setting.

See the original paper for details: https://jair.org/index.php/jair/article/view/15702.

Author: Shirley Li
Date: July 2026
"""

import numpy as np
from morl_baselines.common.pareto import ParetoArchive, get_non_pareto_dominated_inds
from morl_baselines.common.weights import random_weights
from morld.mo_pi import MOPolicyIteration
from pymoo.util.ref_dirs import get_reference_directions
from moocore import hypervolume, hv_contributions
from morl_baselines.common.utils import nearest_neighbors

class PIMORLD():
    """MORL/D with policy iteration."""

    def __init__(self, 
                 env, 
                 pop_size=20, 
                 gamma=0.7, 
                 max_eval_iters=500, 
                 scalarization='linear', 
                 ref_point=None, 
                 weights_init='uniform', 
                 seed=42):
        """Initialize the population, archive, and shared environment state.

        Args:
            env: Environment exposing the MOMDP components.
            pop_size: Number of agents in the initial population.
            gamma: Discount factor used by the solver.
            max_eval_iters: Maximum policy-evaluation iterations per agent.
            scalarization: Scalarization type passed to each agent.
            ref_point: Optional reference point for scalarization (in case of Chebyshev).
            weights_init: Strategy for generating the initial weight vectors.
            seed: Random seed for the internal RNG.
        """
        self.env = env
        self.nO = self.env.nO
        self.P = self.env.P_user
        self.R = self.env.R
        self.P_comp = self.env.P_comp
        self.initial_distribution = self.env.initial_distribution
        self.pop_size = pop_size
        self.gamma = gamma
        self.max_eval_iters = max_eval_iters
        self.scalarization = scalarization
        self.ref_point = ref_point
        self.weights_init = weights_init
        self.rng = np.random.default_rng(seed)
        self.agents = []
        self.pareto_archive = ParetoArchive()  # Initialize Pareto archive with a maximum size of 20

        # Initialize agents and Pareto archive
        self.intialize_agents()

    def intialize_agents(self):
        """Create the initial population and populate the Pareto archive."""

        if self.weights_init == 'random':
            all_weights = random_weights(self.nO, self.pop_size, rng=self.rng)
        elif self.weights_init == 'uniform':
            all_weights = get_reference_directions("energy", self.nO, self.pop_size, seed=1)
        elif self.weights_init == 'combined':
            uniform_weights = get_reference_directions("energy", self.nO, self.pop_size//2, seed=1)
            random_weights_arr = random_weights(self.nO, self.pop_size//2, rng=self.rng)
            all_weights = np.vstack((uniform_weights, random_weights_arr))
        print("Initial weights for agents:")
        print(all_weights)

        for i in range(self.pop_size):
            weights = all_weights[i]
            pi_agent = MOPolicyIteration(self.P, self.R, weights, 
                                         gamma=self.gamma, ref_point=self.ref_point, scalarization=self.scalarization, 
                                         initial_distribution=self.initial_distribution)
            pi_agent.train(max_eval_iters=self.max_eval_iters, verbose=True)
            pi_agent.evaluate()
            self.agents.append(pi_agent)
            self.add_to_pareto_archive(pi_agent)

    def add_to_pareto_archive(self, agent):
        """Add a trained agent to the Pareto archive if its policy is new.

        Args:
            agent: Trained MOPolicyIteration instance.

        """
        if self.is_in_pareto_archive(agent):
            print("Policy already in archive, skipping addition.")
            return
        individual = {
            'weights': agent.weights.copy(),
            'policy': agent.policy_table.copy(),
            'V': agent.V_full.copy()
        }
        self.pareto_archive.add(individual, agent.expected_return)
    
    def is_in_pareto_archive(self, agent):
        """Return whether the agent's policy already exists in the archive.

        Args:
            agent: MOPolicyIteration instance to compare against the archive.

        Returns:
            True if the policy is already present, otherwise False.
        """
        for ind in self.pareto_archive.individuals:
            if np.array_equal(ind['policy'], agent.policy_table):
                return True
        return False

    def adapt_weights_random(self, agents_to_adapt, num_iterations=2):
        """Perturb agent weights with noise and re-evaluate them.

        Args:
            agents_to_adapt: Iterable of agents to update.
            num_iterations: Number of random-adaptation passes.

        Returns:
            None.
        """
        for i in range(num_iterations):
            print("Weight adaptation cycle: ", i+1)
            for agent in agents_to_adapt:
                # add some random noise to the weights, ensure positivity and normalization
                noise = self.rng.normal(1, 0.1, size=agent.weights.shape)
                new_weights = agent.weights * noise
                # normalize the weights
                new_weights /= np.sum(new_weights)
                agent.set_weights(new_weights)
                self.add_to_pareto_archive(agent)

    def closest_non_dominated(self, eval_policy: np.ndarray):
        """Find the nearest non-dominated evaluation to the input policy's evaluation.

        Args:
            eval_policy: Evaluation vector to compare against the archive.

        Returns:
            Tuple of (individual, evaluation) or (None, None) when no match exists.
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
        """Adjust weights using a Pareto simulated annealing style update.
        
        Adapted from the MORL-Baselines implementation:
        https://github.com/LucasAlegre/morl-baselines

        Args:
            agents_to_adapt: Iterable of agents to update.
            delta: Relative increase/decrease applied to each weight.

        Returns:
            None.
        """
        # P. Czyzżak and A. Jaszkiewicz,
        # "Pareto simulated annealing—a metaheuristic technique for multiple-objective combinatorial optimization,"
        # Journal of Multi-Criteria Decision Analysis, vol. 7, no. 1, pp. 34–47, 1998,
        # doi: 10.1002/(SICI)1099-1360(199801)7:1<34::AID-MCDA161>3.0.CO;2-6.
        for agent in agents_to_adapt:
            evaluation = agent.expected_return
            _, reference_eval = self.closest_non_dominated(evaluation)

            updated_weights = agent.weights.copy()
            if reference_eval is not None:
                for objective_idx in range(len(evaluation)):
                    if evaluation[objective_idx] >= reference_eval[objective_idx]:
                        # Increase the weight of objectives with better performance.
                        updated_weights[objective_idx] *= (1 + delta)
                    else:
                        # Decrease the weight of objectives with worse performance.
                        updated_weights[objective_idx] /= (1 + delta)

            # Normalize the weights so they sum to one.
            normalized_weights = updated_weights / np.linalg.norm(updated_weights, ord=1)
            agent.set_weights(normalized_weights)
            self.add_to_pareto_archive(agent)

    def agent_from_archive(self, individual, ind_eval):
        """Reconstruct a policy-iteration agent from archive contents.

        Args:
            individual: Archive entry containing weights, policy, and value tensor.
            ind_eval: Archived expected-return vector.

        Returns:
            Reconstructed MOPolicyIteration instance.
        """
        agent = MOPolicyIteration(self.P, self.R, individual['weights'].copy(), 
                                         gamma=self.gamma, ref_point=self.ref_point, scalarization=self.scalarization, 
                                         initial_distribution=self.initial_distribution)
        agent.policy_table = individual['policy'].copy()
        agent.V_full = individual['V'].copy()
        agent.expected_return = ind_eval.copy()
        return agent

    def generate_new_agents(self, num_new_agents):
        """Create new agents from random weights and warm-start them from neighbors.

        Args:
            num_new_agents: Number of agents to generate.

        Returns:
            List of newly trained agents.
        """
        new_population = []
        new_weights = random_weights(self.nO, num_new_agents, rng=self.rng)

        # for each new agent, we intialize it with new weights but the value function from the closest in weights from the population
        dist_metric = lambda a, b: np.sum(np.square(a - b))
        closest = [nearest_neighbors(1, w, [agent.weights for agent in self.agents], dist_metric) for w in new_weights]

        for i in range(num_new_agents):
            weights = new_weights[i]
            V_init = self.agents[closest[i][0]].V_full.copy()
            pi_agent = MOPolicyIteration(self.P, self.R, weights, V_init=V_init,
                                         gamma=self.gamma, ref_point=self.ref_point, scalarization=self.scalarization, 
                                         initial_distribution=self.initial_distribution)
            pi_agent.train(max_eval_iters=self.max_eval_iters)
            new_population.append(pi_agent)
            self.add_to_pareto_archive(pi_agent)
        return new_population
    
    def get_hv_ref_point(self, evals=None, delta=0.1):
        """Get the reference point for hypervolume-based selection.

        Args:
            evals: Optional evaluation matrix; defaults to the archive evaluations.
            delta: Margin subtracted from the minimum evaluation value.

        Returns:
            Reference point vector.
        """
        if evals is None:
            evals = np.array(self.pareto_archive.evaluations)
        if len(evals) == 0:
            return np.zeros(self.nO)
        return np.min(evals, axis=0) - delta

    def select_parents(self, num_parents):
        """Select the strongest non-dominated agents by hypervolume contribution.

        Args:
            num_parents: Maximum number of parents to return.

        Returns:
            Array-like collection of selected agents.
        """
        ref_point_HV = self.get_hv_ref_point()
        non_dominated_inds = get_non_pareto_dominated_inds(np.array([agent.expected_return for agent in self.agents]))
        parents = np.array(self.agents)[non_dominated_inds]
        hv_cont = hv_contributions([agent.expected_return for agent in parents], ref_point_HV, maximise=True)
        sorted_inds = np.argsort(hv_cont)[::-1]
        parents = parents[sorted_inds]  # Reorder parents based on hypervolume contribution
        if len(parents) > num_parents:
            parents = parents[:num_parents]  # Select top num_parents based on hypervolume contribution
        return parents
    
    def tournament_selection(self, candidates, n, tournament_size=3):
        """Select candidates by repeated tournaments on hypervolume contribution.

        Args:
            candidates: Candidate agents to sample from.
            n: Number of winners to select.
            tournament_size: Number of candidates per tournament.

        Returns:
            List of selected candidate agents.
        """
        candidate_evals = np.array([agent.expected_return for agent in candidates])
        candidate_hv_contributions = hv_contributions(candidate_evals, self.get_hv_ref_point(candidate_evals), maximise=True)
        selected = []
        remaining_indices = list(range(len(candidates)))
        for _ in range(n):
            tournament_inds = self.rng.choice(remaining_indices, size=min(tournament_size, len(remaining_indices)),replace=False)
            tournament_candidates = [candidates[i] for i in tournament_inds]
            tournament_hv_contributions = candidate_hv_contributions[tournament_inds]
            winner_idx = np.argmax(tournament_hv_contributions)
            best_candidate = tournament_candidates[winner_idx]
            selected.append(best_candidate)
            remaining_indices.remove(tournament_inds[winner_idx])
        return selected
    
    def crossover(self, parent1, parent2, crossover_prob=0.5):
        """Combine two parents by averaging weights and value estimates.

        Args:
            parent1: First parent agent.
            parent2: Second parent agent.
            crossover_prob: Probability of performing crossover.

        Returns:
            A child agent, or one of the parents when crossover is skipped.
        """
        if self.rng.random() > crossover_prob:
        # If no crossover, return a clone of the more successful parent 
        # or randomly choose one to proceed
            return parent1 if self.rng.random() > 0.5 else parent2
        # crossover on the weight vectors
        child_weights = (parent1.weights + parent2.weights) / 2
        child_weights /= np.sum(child_weights)  # Normalize to sum to 1
        child_V_init = (parent1.V_full + parent2.V_full) / 2
        child_agent = MOPolicyIteration(self.P, self.R, child_weights, V_init=child_V_init,
                                         gamma=self.gamma, ref_point=self.ref_point, scalarization=self.scalarization, 
                                         initial_distribution=self.initial_distribution)
        child_agent.train(max_eval_iters=self.max_eval_iters)
        self.add_to_pareto_archive(child_agent)
        return child_agent

    
    def train(self, num_iterations=5, hv_threshold=1e-4, hv_patience=3, num_offspring=10, crossover_prob=0.5, max_resets=2):
        """Run the population-level adaptation loop until convergence or reset limits.

        Args:
            num_iterations: Maximum outer-loop iterations.
            hv_threshold: Relative hypervolume improvement threshold.
            hv_patience: Number of low-improvement iterations before stopping or resetting.
            num_offspring: Number of agents generated after a reset.
            crossover_prob: Probability of crossover when generating children.
            max_resets: Maximum number of population resets.

        Returns:
            None. Updates the internal agent population.
        """
        population = self.agents.copy()
        hv_history = []
        archive_sizes = []
        no_improvement_count = 0
        reset_count = 0
        
        for i in range(num_iterations):
            print(f"\n=== Iteration: {i+1} ===")
            
            # Track metrics
            current_hv = self.get_hv()
            archive_size = len(self.pareto_archive.individuals)
            hv_history.append(current_hv)
            archive_sizes.append(archive_size)
            
            print(f"HV: {current_hv:.6f}, Archive size: {archive_size}")
            
            # Check HV convergence
            if len(hv_history) > 1:
                hv_improvement = hv_history[-1] - hv_history[-2]
                relative_improvement = hv_improvement / max(abs(hv_history[-2]), 1e-10)
                print(f"HV improvement: {hv_improvement:.6f} (relative: {relative_improvement:.6f})")
                
                if relative_improvement < hv_threshold:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            
            if no_improvement_count >= hv_patience:
                if reset_count < max_resets:
                    print(f"\nNo significant HV improvement for {hv_patience} iterations.")
                    print(f"Resetting population ({reset_count+1}/{max_resets})")

                    random_pop = self.generate_new_agents(num_offspring)

                    population = self.rng.choice(
                        population, size=self.pop_size // 2, replace=False
                    ).tolist()

                    population.extend(random_pop)

                    no_improvement_count = 0
                    reset_count += 1
                    continue
                else:
                    print(f"\n✓ Converged at iteration {i+1}")
                    print(f"  - HV stable for {no_improvement_count} iterations")
                    print(f"  - Maximum resets ({max_resets}) reached")
                    break
            
            # Continue training
            parents = self.tournament_selection(population, n=self.pop_size//2, tournament_size=3)
            offspring = list(parents.copy())
            self.adapt_weights_psa(offspring, delta=0.2)
            population = list(np.concatenate([parents, offspring]))
        
        self.agents = population

    def get_hv(self, ref_point=None):
        """Return the hypervolume of the current Pareto archive.

        Args:
            ref_point: Optional hypervolume reference point.

        Returns:
            Hypervolume scalar.
        """
        if ref_point is None:
            ref_point_HV = np.zeros(self.nO)
        else:
            ref_point_HV = ref_point
        return hypervolume(self.pareto_archive.evaluations, ref_point_HV, maximise=True)