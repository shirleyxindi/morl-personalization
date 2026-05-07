"""
Process action data and normalize expert scores.

This module reads action data from a CSV file, normalizes expert scores across
four coping strategy categories (acceptance, distraction, problem-solving, 
and social support), and saves the results to a new CSV file.
"""

import pandas as pd

difficulty_to_score = {
    "Makkelijk": 0.33,
    "Gemiddeld": 0.66,
    "Moeilijk": 1.0
}



data_folder = 'C:\\Users\\shirl\\Documents\\Studie\\2025-2026\\Thesis\\personalized-coping-challenges\\data\\'

action_file = 'challenges_real.csv'
action_df = pd.read_csv(data_folder + action_file, delimiter=';')

expert_score_cols = ['score_acceptance', 'score_distraction', 'score_problem_solving', 'score_social_support']

def map_score(action_df, col, min_score=1, max_score=3):
    scaled_scores = (action_df[col] - min_score) / (max_score - min_score)
    total = scaled_scores.sum()
    normalized_scores = scaled_scores / total
    return scaled_scores

# for col in expert_score_cols:
    # action_df[col] = map_score(action_df, col, action_df[col].min(), action_df[col].max())

action_df['expert_score'] = action_df['Moeilijkheidsgraad'].map(difficulty_to_score)
action_df.head()
# action_df.to_csv(data_folder + 'normalized_challenges.csv', index=False)