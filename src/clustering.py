import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
def get_zscore_categories(df):
    """Calculate coping strategy category z-scores for each user.
    args:
    - df: DataFrame with raw coping strategy scores.
    returns:
    - df: DataFrame with user_id and z-scores for each coping strategy category."""
 
    # get problem solving category score
    df.loc[:, 'active'] = df[['cope_active_coping_1', 'cope_active_coping_2']].mean(axis=1)
    df.loc[:, 'planning'] = df[['cope_planning_1', 'cope_planning_2']].mean(axis=1)
    df.loc[:, 'problem_solving'] = df[['active', 'planning']].mean(axis=1)
 
    # get acceptance category score
    df.loc[:, 'acc'] = df[['cope_acceptance_1', 'cope_acceptance_2']].mean(axis=1)
    df.loc[:, 'positive_reframing'] = df[['cope_positive_reframing_1', 'cope_positive_reframing_2']].mean(axis=1)
    # df['acceptance'] = df[['acc', 'positive_reframing']].mean(axis=1)
    df.loc[:, 'acceptance'] = df['acc']
 
    # get distraction category score
    df.loc[:, 'distraction'] = df[['cope_distraction_1', 'cope_distraction_2']].mean(axis=1)
 
    # get social support category score
    df.loc[:, 'emotional_support'] = df[['cope_emotional_support_1', 'cope_emotional_support_2']].mean(axis=1)
    df.loc[:, 'instrumental_support'] = df[['cope_instrumental_support_1', 'cope_instrumental_support_2']].mean(axis=1)
    df.loc[:, 'social_support'] = df[['emotional_support', 'instrumental_support']].mean(axis=1)
 
    # get z-scores
    df = df[['user_id', 'problem_solving', 'acceptance', 'distraction', 'social_support']]
    df.loc[:, 'mean'] = df[['problem_solving', 'acceptance', 'distraction', 'social_support']].mean(axis=1)
    df.loc[:, 'std'] = df[['problem_solving', 'acceptance', 'distraction', 'social_support']].std(axis=1).replace(0, np.nan) # prevent division by zero
 
    for c in ['problem_solving', 'acceptance', 'distraction', 'social_support']:
        df.loc[:, c + '_z'] = (df[c] - df['mean']) / df['std']
 
    return df
 
def get_cluster_size(df, cluster_n, min_samples=50):
    """Determine optimal cluster size based on silhouette score and minimum cluster size.
    args:
    - df: DataFrame with z-scores for coping strategy categories.
    - cluster_n: List of integers representing different cluster sizes to evaluate.
    - min_samples: Minimum number of samples required in the smallest cluster.
    returns:
    - best_k: Optimal number of clusters."""
 
    X = df[['problem_solving_z', 'acceptance_z', 'distraction_z', 'social_support_z']].fillna(0)
    sil_scores = []
    smallest_cluster_sizes = []
 
    # calculate silhouette scores for different cluster sizes
    for k in cluster_n:
        kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
        labels = kmeans.fit_predict(X)
 
        sil_scores.append(silhouette_score(X, labels))
 
        # get sample size of smallest cluster
        cluster_counts = pd.Series(labels).value_counts()
        smallest_cluster_sizes.append(cluster_counts.min())
 
    # get optimal cluster n based on silhouette score
    best_k = cluster_n[np.argmax(sil_scores)]
 
    # if smallest cluster size is below min_samples, check other cluster n
    if smallest_cluster_sizes[np.argmax(sil_scores)] < min_samples:
        # get cluster n with largest smallest cluster
        best_k = cluster_n[np.argmax(smallest_cluster_sizes)]
 
    return best_k
 
def assign_clusters(df):
    """Assign clusters to users based on coping strategy z-scores.
    args:
    - df: DataFrame with z-scores for coping strategy categories.
    returns:
    - df: DataFrame with assigned cluster labels."""
 
    X = df[['problem_solving_z', 'acceptance_z', 'distraction_z', 'social_support_z']].fillna(0)
 
    kmeans = KMeans(n_clusters=get_cluster_size(df, [2, 3]), n_init=100, random_state=42).fit(X)
    df.loc[:, 'cluster'] = kmeans.labels_
 
    return df