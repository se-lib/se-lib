import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .model import DecisionMatrix, CriterionDirection

class WeightedSumModel:
    def __init__(self, matrix: DecisionMatrix):
        self.matrix = matrix

    def evaluate(self) -> pd.Series:
        df = self.matrix.to_dataframe()
        normalized_df = df.copy()
        
        # Normalize (Min-Max Scaling)
        for crit in self.matrix.criteria:
            col = df[crit.name]
            if crit.direction == CriterionDirection.MAXIMIZE:
                normalized_df[crit.name] = (col - col.min()) / (col.max() - col.min())
            else:
                normalized_df[crit.name] = (col.max() - col) / (col.max() - col.min())
            
            # Fill NaNs with 0 if range is 0
            normalized_df[crit.name] = normalized_df[crit.name].fillna(0.0)

        # Weighted Sum
        scores = {}
        for alt_name, row in normalized_df.iterrows():
            score = 0
            for crit in self.matrix.criteria:
                score += row[crit.name] * crit.weight
            scores[alt_name] = score
            
        return pd.Series(scores).sort_values(ascending=False)

class TOPSIS:
    def __init__(self, matrix: DecisionMatrix):
        self.matrix = matrix

    def evaluate(self) -> pd.Series:
        df = self.matrix.to_dataframe()
        
        # Vector Normalization
        norm_df = df.copy()
        for col_name in df.columns:
            denom = np.sqrt((df[col_name]**2).sum())
            if denom == 0:
                norm_df[col_name] = 0
            else:
                norm_df[col_name] = df[col_name] / denom

        # Weighted Normalized Decision Matrix
        weighted_df = norm_df.copy()
        for crit in self.matrix.criteria:
            weighted_df[crit.name] = norm_df[crit.name] * crit.weight

        # Ideal and Negative-Ideal Solutions
        ideal_best = {}
        ideal_worst = {}
        
        for crit in self.matrix.criteria:
            if crit.direction == CriterionDirection.MAXIMIZE:
                ideal_best[crit.name] = weighted_df[crit.name].max()
                ideal_worst[crit.name] = weighted_df[crit.name].min()
            else:
                ideal_best[crit.name] = weighted_df[crit.name].min()
                ideal_worst[crit.name] = weighted_df[crit.name].max()

        # Euclidean Distances
        separation_best = {}
        separation_worst = {}
        
        for idx, row in weighted_df.iterrows():
            dist_best = np.sqrt(sum((row[c.name] - ideal_best[c.name])**2 for c in self.matrix.criteria))
            dist_worst = np.sqrt(sum((row[c.name] - ideal_worst[c.name])**2 for c in self.matrix.criteria))
            separation_best[idx] = dist_best
            separation_worst[idx] = dist_worst

        # Relative Closeness to Ideal
        closeness = {}
        for idx in weighted_df.index:
            s_plus = separation_best[idx]
            s_minus = separation_worst[idx]
            if (s_plus + s_minus) == 0:
                closeness[idx] = 0.0
            else:
                closeness[idx] = s_minus / (s_plus + s_minus)

        return pd.Series(closeness).sort_values(ascending=False)

class PughMatrix:
    def __init__(self, matrix: DecisionMatrix, baseline_name: str):
        self.matrix = matrix
        # Find baseline
        self.baseline = next((a for a in matrix.alternatives if a.name == baseline_name), None)
        if not self.baseline:
            raise ValueError(f"Baseline '{baseline_name}' not found in alternatives")

    def evaluate(self) -> pd.DataFrame:
        results = {}
        
        for alt in self.matrix.alternatives:
            comparisons = {'S': 0, '+': 0, '-': 0}
            details = {}
            
            for crit in self.matrix.criteria:
                val = alt.scores[crit.name]
                base_val = self.baseline.scores[crit.name]
                
                diff = val - base_val
                
                # Invert direction logic for Minimization? 
                # Pugh usually simple comparison: Better (+), Same (S), Worse (-)
                # If Minimize, Lower is Better.
                
                rating = 'S'
                if diff == 0:
                    rating = 'S'
                    comparisons['S'] += 1
                elif diff > 0:
                    # Value is higher. 
                    if crit.direction == CriterionDirection.MAXIMIZE:
                        rating = '+'
                        comparisons['+'] += 1
                    else:
                        rating = '-' # Higher is worse for Minimize
                        comparisons['-'] += 1
                else: # diff < 0
                     # Value is lower.
                    if crit.direction == CriterionDirection.MAXIMIZE:
                        rating = '-'
                        comparisons['-'] += 1
                    else:
                        rating = '+' # Lower is better for Minimize
                        comparisons['+'] += 1
                        
                details[crit.name] = rating
            
            details['Score'] = comparisons['+'] - comparisons['-']
            details['Sum +'] = comparisons['+']
            details['Sum -'] = comparisons['-']
            details['Sum S'] = comparisons['S']
            results[alt.name] = details
            
        return pd.DataFrame.from_dict(results, orient='index').sort_values(by='Score', ascending=False)
