import pymc as pm
import arviz as az
import numpy as np
import pandas as pd

def paired_t_test(diffs, name='improvement', mu_prior=None, sigma_prior=None, seed=66, effect_size_labels = ["small", "medium", "large"], effect_size_thresholds = [0.1, 0.2, 0.3]):
    with pm.Model() as model:
        # Priors
        if mu_prior is None:
            mu = pm.Normal('improvement', mu=0, sigma=10)
        else:
            mu = pm.Normal('improvement', mu=mu_prior[0], sigma=mu_prior[1])

        if sigma_prior is None:
            sigma = pm.Uniform('sigma', lower=0, upper=10)
        else:
            sigma = pm.Uniform('sigma', lower=sigma_prior[0], upper=sigma_prior[1])

        nu_minus_one = pm.Exponential('nu_minus_one', 1/29)
        nu = pm.Deterministic('nu', nu_minus_one + 1)

        effect_size = pm.Deterministic('effect_size', mu / sigma)

        # Likelihood
        obs = pm.StudentT('obs', nu=nu, mu=mu, sigma=sigma, observed=diffs)
        trace = pm.sample(nuts_sampler="numpyro", cores=4, progressbar=False, random_seed=seed)
    
    post = trace.posterior

    mu_samples = post["improvement"].values.flatten()
    es_samples = post["effect_size"].values.flatten()

    hdi = az.hdi(mu_samples, hdi_prob=0.95)
    post_prob = (mu_samples > 0).mean()
    print(f"Posterior probability improvement > 0: {post_prob:.4f}")

    result = {
        "objective": name,
        "mean": mu_samples.mean(),
        "hdi_low": hdi[0],
        "hdi_high": hdi[1],
        "post_prob": post_prob,
    }

    # effect size probabilities
    es_probs = {}
    for i, t in enumerate(effect_size_thresholds):
        es_probs[effect_size_labels[i]] = (np.abs(es_samples) > t).mean()

    result.update(es_probs)

    return pd.DataFrame([result])


def two_sample_t_test(group1, group2, name='improvement', seed=66, effect_size_tests = [0.1, 0.2, 0.3]):
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
        trace = pm.sample(nuts_sampler="numpyro", cores=4, progressbar=False, random_seed=seed)
    
    # Analysis
    print(az.summary(trace, var_names=['improvement', 'effect_size', 'group1_mean', 'group2_mean'], kind="stats", hdi_prob=0.95, round_to=3))
    
    # Probability State-Action is better (improvement > 0)
    prob_better = (trace.posterior['improvement'] > 0).mean().item()
    print(f"P(improvement > 0): {prob_better:.4f}")

    for test_effect in effect_size_tests:
        prob_test_effect = (np.abs(trace.posterior['effect_size']) > test_effect).mean().item()
        print(f"P(Effect Size > {test_effect}): {prob_test_effect:.4f}")

def interpret_probability(p: float) -> str:
    """
    Interpret a probability value according to betting strength thresholds.
    
    Args:
        p: Probability value in [0, 1]
    
    Returns:
        Human-readable interpretation string
    """
    thresholds = [
        (0.00005, "Virtually certain against"),
        (0.0005,  "Nearing certainty against"),
        (0.005,   "Very strong bet against"),
        (0.01,    "\\makecell{Strong bet against\\\\irresponsible to avoid}"),
        (0.05,    "\\makecell{Good bet against\\\\too good to disregard}"),
        (0.1,     "\\makecell{A promising but\\\\risky bet against}"),
        (0.25,    "Only a casual bet against"),
        (0.5,     "Not worth betting against"),
        (0.75,    "Not worth betting on"),
        (0.9,     "Only a casual bet"),
        (0.95,    "\\makecell{A promising but\\\\risky bet}"),
        (0.99,    "\\makecell{Good bet\\\\too good to disregard}"),
        (0.995,   "\\makecell{Strong bet\\\\irresponsible to avoid}"),
        (0.9995,  "Very strong bet"),
        (0.99995, "Nearing certainty"),
        (1.0,     "Virtually certain"),
    ]

    for upper, label in thresholds:
        if p < upper:
            return label
    return "Virtually certain"

def to_latex_body(df, effect_cols=("small", "medium", "large")):
    df['mean_hdi'] = df.apply(lambda r: f"\\makecell{{${r['mean']:.3f}$\\\\$[{r['hdi_low']:.3f},\\,{r['hdi_high']:.3f}]$}}", axis=1)
    df['interpretation'] = df['post_prob'].apply(interpret_probability)
    lines = []

    for _, r in df.iterrows():
        lines.append(
            f"{r['objective']} & "
            f"{r['mean_hdi']} & "
            f"${r['post_prob']:.3f}$ & "
            f"{r['interpretation']} & "
            f"${r[effect_cols[0]]:.2f}$ & "
            f"${r[effect_cols[1]]:.2f}$ & "
            f"${r[effect_cols[2]]:.2f}$ \\\\"
        )

    return "\n".join(lines)