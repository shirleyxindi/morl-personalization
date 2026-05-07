import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class SyntheticMDPGenerator:
    def __init__(self, 
                 num_states: int = 10,
                 num_actions: int = 5,
                 num_count_states: int = 3,
                 num_objectives: int = 3,
                 num_clusters: int = 3,
                 num_samples: int = 1000,
                 seed: int = 42):
        """
        Generate synthetic MDP data for testing.
        
        Args:
            num_states: Number of user states (nU)
            num_actions: Number of actions (nA)
            num_count_states: Number of count states (nC)
            num_objectives: Number of objectives (nO)
            num_clusters: Number of action clusters
            num_samples: Number of interaction samples to generate
            seed: Random seed
        """
        self.nU = num_states
        self.nA = num_actions
        self.nC = num_count_states
        self.nO = num_objectives
        self.num_clusters = num_clusters
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        
        # Ground truth components
        self.true_P = None
        self.true_completion_probs = None
        self.true_rewards = None
        self.action_to_cluster = None
        
    def generate_ground_truth(self):
        """Generate ground truth MDP components."""
        
        # 1. Transition probabilities P(s'|s,a) - (nU, nA, nU)
        self.true_P = self._generate_transition_probabilities()
        
        # 2. Completion probabilities P(complete|s,a) - (nU, nA)
        self.true_completion_probs = self._generate_completion_probabilities()
        
        # 3. Action clustering
        self.action_to_cluster = self._generate_action_clusters()
        
        # 4. Rewards R(s,c,a,o) - (nU, nC, nA, nO)
        self.true_rewards = self._generate_rewards()
        
        return {
            'P': self.true_P,
            'completion_probs': self.true_completion_probs,
            'rewards': self.true_rewards,
            'action_to_cluster': self.action_to_cluster
        }
    
    def _generate_transition_probabilities(self):
        """Generate valid transition probability matrix."""
        P = self.rng.dirichlet(np.ones(self.nU), size=(self.nU, self.nA))
        return P
    
    def _generate_completion_probabilities(self):
        """Generate completion probabilities with some structure."""
        # Base completion rate varies by state
        base_rates = self.rng.beta(2, 2, size=self.nU)
        
        # Action effects (some actions are better than others)
        action_effects = self.rng.normal(0, 0.2, size=self.nA)
        
        completion_probs = np.zeros((self.nU, self.nA))
        for s in range(self.nU):
            for a in range(self.nA):
                prob = base_rates[s] + action_effects[a]
                completion_probs[s, a] = np.clip(prob, 0.1, 0.9)
        
        return completion_probs
    
    def _generate_action_clusters(self):
        """Assign each action to a cluster."""
        return {a: a % self.num_clusters for a in range(self.nA)}
    
    def _generate_rewards(self):
        """Generate reward matrix with different structures per objective."""
        rewards = np.zeros((self.nU, self.nC, self.nA, self.nO))
        
        for o in range(self.nO):
            if o == 0:  # Diversity reward (depends on count state and action cluster)
                for c in range(self.nC):
                    for cluster in range(self.num_clusters):
                        # Diversity reward increases with count state
                        base_diversity = self.rng.uniform(0.5, 2.0)
                        diversity_bonus = c * 0.3  # More diverse at higher count states
                        
                        for a in range(self.nA):
                            if self.action_to_cluster[a] == cluster:
                                rewards[:, c, a, o] = base_diversity + diversity_bonus
            else:
                # Other rewards depend on state and action
                for s in range(self.nU):
                    for a in range(self.nA):
                        rewards[s, :, a, o] = self.rng.uniform(-1, 5)
        
        return rewards
    
    def generate_interaction_data(self) -> pd.DataFrame:
        """Generate synthetic interaction data based on ground truth."""
        
        if self.true_P is None:
            self.generate_ground_truth()
        
        data = []
        
        for _ in range(self.num_samples):
            # Sample initial state
            s = self.rng.integers(0, self.nU)
            c = self.rng.integers(0, self.nC)
            
            # Sample action (could be uniform or with some policy)
            a = self.rng.integers(0, self.nA)
            
            # Sample completion
            completed = self.rng.random() < self.true_completion_probs[s, a]
            
            # Sample next state
            s_next = self.rng.choice(self.nU, p=self.true_P[s, a])
            
            # Get cluster for this action
            cluster = self.action_to_cluster[a]
            
            # Generate rewards for each objective
            rewards_dict = {}
            for o in range(self.nO):
                if o == 0:  # Diversity
                    reward_name = 'r_diversity'
                else:
                    reward_name = f'r_obj{o}'
                
                # Add some noise to rewards
                true_reward = self.true_rewards[s, c, a, o]
                noisy_reward = true_reward + self.rng.normal(0, 0.1)
                rewards_dict[reward_name] = noisy_reward
            
            data.append({
                's_idx': s,
                'c_idx': c,
                'action_id': a,
                'action_cluster': cluster,
                'count_cluster': cluster,  # Using same clustering for simplicity
                'completed': int(completed),
                'sp_idx': s_next,
                **rewards_dict
            })
        
        return pd.DataFrame(data)
    
    def verify_components(self, 
                         estimated_P, 
                         estimated_completion_probs, 
                         estimated_rewards,
                         threshold: float = 0.1) -> Dict:
        """
        Verify estimated components against ground truth.
        
        Returns:
            Dictionary with verification results and metrics
        """
        results = {}
        
        # 1. Verify transition probabilities
        P_error = np.abs(self.true_P - estimated_P).mean()
        P_max_error = np.abs(self.true_P - estimated_P).max()
        results['P'] = {
            'mean_absolute_error': P_error,
            'max_absolute_error': P_max_error,
            'passed': P_error < threshold,
            'is_valid_distribution': self._verify_probability_distribution(estimated_P)
        }
        
        # 2. Verify completion probabilities
        comp_error = np.abs(self.true_completion_probs - estimated_completion_probs).mean()
        comp_max_error = np.abs(self.true_completion_probs - estimated_completion_probs).max()
        results['completion_probs'] = {
            'mean_absolute_error': comp_error,
            'max_absolute_error': comp_max_error,
            'passed': comp_error < threshold,
            'is_valid_probability': np.all((estimated_completion_probs >= 0) & 
                                          (estimated_completion_probs <= 1))
        }
        
        # 3. Verify rewards
        reward_error = np.abs(self.true_rewards - estimated_rewards).mean()
        reward_max_error = np.abs(self.true_rewards - estimated_rewards).max()
        results['rewards'] = {
            'mean_absolute_error': reward_error,
            'max_absolute_error': reward_max_error,
            'passed': reward_error < threshold
        }
        
        # Overall summary
        all_passed = all(result.get('passed', False) for result in results.values())
        results['overall'] = {
            'all_passed': all_passed,
            'total_mean_error': (P_error + comp_error + reward_error) / 3
        }
        
        return results
    
    def _verify_probability_distribution(self, P):
        """Verify that P forms valid probability distributions."""
        # Check if all values are in [0, 1]
        if not np.all((P >= 0) & (P <= 1)):
            return False
        
        # Check if each P(s,a) sums to 1
        sums = P.sum(axis=2)
        if not np.allclose(sums, 1.0, atol=1e-6):
            return False
        
        return True
    
    def print_verification_report(self, results: Dict):
        """Print a human-readable verification report."""
        print("=" * 60)
        print("MDP COMPONENT VERIFICATION REPORT")
        print("=" * 60)
        
        for component, metrics in results.items():
            if component == 'overall':
                continue
            
            print(f"\n{component.upper()}")
            print("-" * 40)
            for metric, value in metrics.items():
                if isinstance(value, bool):
                    status = "✓" if value else "✗"
                    print(f"  {metric}: {status}")
                elif isinstance(value, float):
                    print(f"  {metric}: {value:.6f}")
        
        print("\n" + "=" * 60)
        if results['overall']['all_passed']:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print(f"Overall Mean Error: {results['overall']['total_mean_error']:.6f}")
        print("=" * 60)


# Usage example and test
def test_synthetic_mdp():
    """Complete test of synthetic MDP generation and verification."""
    
    # Import your actual functions
    from mdp_utils import (
        compute_transition_probabilities,
        compute_completion_probabilities,
        compute_rewards
    )
    
    # 1. Generate synthetic data
    print("Generating synthetic MDP...")
    generator = SyntheticMDPGenerator(
        num_states=10,
        num_actions=8,
        num_count_states=3,
        num_objectives=3,
        num_clusters=4,
        num_samples=5000,  # More samples for better estimation
        seed=42
    )
    
    ground_truth = generator.generate_ground_truth()
    df = generator.generate_interaction_data()
    
    print(f"Generated {len(df)} interaction samples")
    print(f"\nDataFrame columns: {df.columns.tolist()}")
    print(f"\nSample data:\n{df.head()}")
    
    # 2. Estimate components using your functions
    print("\nEstimating MDP components from data...")
    
    obj_cols = ['r_diversity', 'r_obj1', 'r_obj2']
    
    estimated_P = compute_transition_probabilities(
        df, 
        num_states=generator.nU, 
        num_actions=generator.nA,
        alpha=0.1
    )
    
    estimated_completion_probs = compute_completion_probabilities(
        df,
        num_states=generator.nU,
        num_actions=generator.nA,
        alpha=0.1
    )
    
    estimated_rewards = compute_rewards(
        df,
        estimated_completion_probs,
        nU=generator.nU,
        nC=generator.nC,
        nA=generator.nA,
        nO=generator.nO,
        obj_cols=obj_cols,
        action_col='action_id'
    )
    
    # 3. Verify estimates against ground truth
    print("\nVerifying estimated components...")
    results = generator.verify_components(
        estimated_P,
        estimated_completion_probs,
        estimated_rewards,
        threshold=0.15  # Allow 15% error due to estimation from finite samples
    )
    
    generator.print_verification_report(results)
    
    # 4. Additional diagnostics
    print("\n" + "=" * 60)
    print("ADDITIONAL DIAGNOSTICS")
    print("=" * 60)
    
    print(f"\nTransition Matrix P:")
    print(f"  Shape: {estimated_P.shape}")
    print(f"  Min value: {estimated_P.min():.6f}")
    print(f"  Max value: {estimated_P.max():.6f}")
    
    print(f"\nCompletion Probabilities:")
    print(f"  Shape: {estimated_completion_probs.shape}")
    print(f"  Mean: {estimated_completion_probs.mean():.4f}")
    print(f"  Std: {estimated_completion_probs.std():.4f}")
    print(f"  Range: [{estimated_completion_probs.min():.4f}, {estimated_completion_probs.max():.4f}]")
    
    print(f"\nRewards:")
    print(f"  Shape: {estimated_rewards.shape}")
    for o in range(generator.nO):
        print(f"  Objective {o}: mean={estimated_rewards[:,:,:,o].mean():.4f}, "
              f"std={estimated_rewards[:,:,:,o].std():.4f}")
    
    return generator, df, results


if __name__ == "__main__":
    generator, df, results = test_synthetic_mdp()