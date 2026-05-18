import pymc as pm
import arviz as az
import numpy as np

random_seed = 66
effect_size_tests = [0.1, 0.2, 0.3]

def paired_t_test(diffs):
    with pm.Model() as model:
        # Priors
        mu = pm.Normal('improvement', mu=0, sigma=10)
        sigma = pm.Uniform('sigma', lower=0, upper=10)
        nu_minus_one = pm.Exponential('nu_minus_one', 1/29)
        nu = pm.Deterministic('nu', nu_minus_one + 1)
        lam = sigma**-2   # Precision is the inverse of variance

        effect_size = pm.Deterministic('effect_size', mu / sigma)

        # Likelihood
        obs = pm.StudentT('obs', nu=nu, mu=mu, lam=lam, observed=diffs)
        trace = pm.sample(nuts_sampler="numpyro", cores=4, progressbar=False, random_seed=random_seed)
    
    # Analysis
    print(az.summary(trace, var_names=['improvement', 'effect_size'], hdi_prob=0.95, round_to=3))
    
    # Probability State-Action is better (improvement > 0)
    prob_better = (trace.posterior['improvement'] > 0).mean().item()
    print(f"P(improvement > 0): {prob_better:.4f}")

    for test_effect in effect_size_tests:
        prob_test_effect = (np.abs(trace.posterior['effect_size']) > test_effect).mean().item()
        print(f"P(Effect Size > {test_effect}): {prob_test_effect:.4f}")


def two_sample_t_test(group1, group2):
    with pm.Model() as model:
        # Priors
        group1_mean = pm.Normal('group1_mean', mu=4, sigma=3)
        group2_mean = pm.Normal('group2_mean', mu=4, sigma=3)

        group1_std = pm.Uniform('group1_std', lower=0, upper=10)
        group2_std = pm.Uniform('group2_std', lower=0, upper=10)


        nu_minus_one = pm.Exponential('nu_minus_one', 1/29)
        nu = pm.Deterministic('nu', nu_minus_one + 1)
        lam_1 = group1_std**-2   # Precision is the inverse of variance
        lam_2 = group2_std**-2   # Precision is the inverse of variance

        group1_obs = pm.StudentT('group1_obs', nu=nu, mu=group1_mean, lam=lam_1, observed=group1)
        group2_obs = pm.StudentT('group2_obs', nu=nu, mu=group2_mean, lam=lam_2, observed=group2)

        improvement = pm.Deterministic('improvement', group1_mean - group2_mean)

        effect_size = pm.Deterministic('effect_size', improvement / np.sqrt((group1_std**2 + group2_std**2) / 2))

        # Likelihood
        trace = pm.sample(nuts_sampler="numpyro", cores=4, progressbar=False, random_seed=random_seed)
    
    # Analysis
    print(az.summary(trace, var_names=['improvement', 'effect_size', 'group1_mean', 'group2_mean'], kind="stats", hdi_prob=0.95, round_to=3))
    
    # Probability State-Action is better (improvement > 0)
    prob_better = (trace.posterior['improvement'] > 0).mean().item()
    print(f"P(improvement > 0): {prob_better:.4f}")

    for test_effect in effect_size_tests:
        prob_test_effect = (np.abs(trace.posterior['effect_size']) > test_effect).mean().item()
        print(f"P(Effect Size > {test_effect}): {prob_test_effect:.4f}")