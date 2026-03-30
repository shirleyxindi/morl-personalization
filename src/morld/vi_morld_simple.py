import numpy as np
import math
from typing import Callable, List, Optional, Tuple, Union
from typing_extensions import override

from morl_baselines.multi_policy.morld.morld import MORLD, POLICIES, ParetoArchive, Policy
from morld.mo_vi import MOValueIteration
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.weights import equally_spaced_weights, random_weights
from morl_baselines.common.scalarization import weighted_sum, tchebicheff
from morl_baselines.common.utils import nearest_neighbors

import gymnasium as gym
import numpy as np
import torch as th
from mo_gymnasium.wrappers import MONormalizeReward
from torch import optim


# Register your solver
POLICIES["VI"] = MOValueIteration

class VIMORLD(MOAgent):
    def __init__(
        self,
        env: gym.Env,
        scalarization_method: str = "ws",  # "ws" or "tch"
        evaluation_mode: str = "ser",  # "esr" or "ser"
        gamma: float = 0.995,
        pop_size: int = 6,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
        neighborhood_size: int = 1,  # n = "n closest neighbors", 0=none
        dist_metric: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.sum(
            np.square(a - b)
        ),  # distance metric between neighbors
        weight_init_method: str = "uniform",
        weight_adaptation_method: Optional[str] = None,  # "PSA" or None
        device: Union[th.device, str] = "auto",
    ):
        self.env = env
        super().__init__(self.env, device, seed=seed)
        self.gamma = gamma
        self.seed = seed

        if rng is not None:
            self.np_random = rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # (!) This is helpful for scalarization (!)
        for i in range(env.unwrapped.reward_space.shape[0]):
            env = MONormalizeReward(env, idx=i)

        self.evaluation_mode = evaluation_mode
        self.pop_size = pop_size

        # Scalarization and weights
        self.weight_init_method = weight_init_method
        self.weight_adaptation_method = weight_adaptation_method
        if self.weight_adaptation_method == "PSA":
            self.delta = 0.05
        else:
            self.delta = None
        if self.weight_init_method == "uniform":
            self.weights = np.array(equally_spaced_weights(self.reward_dim, self.pop_size, self.seed))
        elif self.weight_init_method == "random":
            self.weights = random_weights(self.reward_dim, n=self.pop_size, dist="dirichlet", rng=self.np_random)
        else:
            raise Exception(f"Unsupported weight init method: ${self.weight_init_method}")

        self.scalarization_method = scalarization_method
        if scalarization_method == "ws":
            self.scalarization = weighted_sum
        elif scalarization_method == "tch":
            self.scalarization = tchebicheff(tau=0.5, reward_dim=self.reward_dim)
        else:
            raise Exception(f"Unsupported scalarization method: ${self.scalarization_method}")

        self.neighborhood_size = neighborhood_size
        self.dist_metric = dist_metric
        self.neighborhoods = [
            nearest_neighbors(
                n=self.neighborhood_size,
                current_weight=w,
                all_weights=self.weights,
                dist_metric=self.dist_metric,
            )
            for w in self.weights
        ]
        print("Weights:", self.weights)
        print("Neighborhoods:", self.neighborhoods)

        # Logging
        self.global_step = 0
        self.iteration = 0

        # Policies' population
        self.current_policy = 0  # For turn by turn selection
        self.population = [
            MOValueIteration(id=i, env=self.env.unwrapped, weights=w, gamma=self.gamma)
            for i, w in enumerate(self.weights)
        ]
        self.archive = ParetoArchive()

    @override
    def get_config(self):
        return {
            "env_id": self.env.spec.id,
            "scalarization_method": self.scalarization_method,
            "evaluation_mode": self.evaluation_mode,
            "gamma": self.gamma,
            "pop_size": self.pop_size,
            "seed": self.seed,
            "neighborhood_size": self.neighborhood_size,
            "weight_init_method": self.weight_init_method,
            "weight_adaptation_method": self.weight_adaptation_method,
        }


    def __eval_all_policies(self, *args, **kwargs):
        """
        OVERRIDE: Analytic Evaluation.
        Bypasses the 'policy_eval' loop which runs env.step().
        """
        evals = []
        for p in self.population:
            res = p.expected_return
            evals.append(res)
            self.archive.add(p, res)
        return evals

    def train(self, iterations: int):
        """
        The main loop. Each cycle is a weight adaptation step.
        """
        print(f"Starting VIMORLD with {self.pop_size} agents...")
        
        # Initial Solve
        for p in self.population:
            p.train()

        for c in range(iterations):
            # 1. Get Performance (Instant)
            evals = self.__eval_all_policies()
            
            # 2. Adapt Weights (PSA logic)
            # self._MORLD__adapt_weights(evals)
            
            if c % 5 == 0:
                print(f"Cycle {c} | Archive Size: {len(self.archive.evaluations)}")

    def _MORLD__adapt_weights(self, evals):
        """
        PSA Logic: Nudges weights to favor objectives where 
        the agent is underperforming compared to the Archive.
        """
        # (Standard PSA code goes here, using 'p.wrapped.set_weights(new_w)')
        pass