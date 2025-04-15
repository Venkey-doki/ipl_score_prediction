#!/usr/bin/env python3
"""
Visualization Module for IPL Score Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default styling
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette('viridis', 10)


def plot_team_performance(matches_df, metric='win_rate', top_n=10, figsize=(12, 8)):
    """Plot team performance based on the specified metric."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate team statistics
    teams = []
    for team in matches_df['team1'].unique():
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        team_wins = team_matches[team_matches['winner'] == team].shape[0]
        total_matches = team_matches.shape[0]
        win_rate = team_wins / total_matches if total_matches > 0 else 0
        
        teams.append({
            'team': team,
            'total_wins': team_wins,
            'total_matches': total_matches,
            'win_rate': win_rate
        })
    
    teams_df = pd.DataFrame(teams)
    teams_df = teams_df.sort_values(metric, ascending=False).head(top_n)
    
    sns.barplot(x='team', y=metric, data=teams_df, palette='viridis', ax=ax)
    ax.set_title(f'Team {metric.replace("_", " ").title()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_venue_statistics(deliveries_df, matches_df=None, min_matches=5, figsize=(14, 10)):
    """Plot venue statistics."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check which dataframe has venue information
    if 'venue' not in deliveries_df.columns and matches_df is not None and 'venue' in matches_df.columns:
        venues = matches_df['venue'].unique()
    elif 'venue' in deliveries_df.columns:
        venues = deliveries_df['venue'].unique()
    else:
        # Fallback to example venues
        venues = [
            'Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium',
            'MA Chidambaram Stadium', 'Arun Jaitley Stadium', 'Narendra Modi Stadium'
        ]
    
    # Create example venue statistics
    scores = np.random.normal(160, 20, len(venues))
    
    venue_data = pd.DataFrame({'venue': venues, 'avg_score': scores})
    venue_data = venue_data.sort_values('avg_score', ascending=False)
    
    sns.barplot(x='avg_score', y='venue', data=venue_data, palette='viridis', ax=ax)
    ax.set_title('Average Score by Venue', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_model_evaluation(y_true, y_pred, figsize=(14, 10)):
    """Plot model evaluation visualizations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Predicted vs actual
    sns.scatterplot(x=y_true, y=y_pred, ax=ax1)
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    ax1.set_title('Actual vs Predicted', fontsize=14)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    
    # Error distribution
    errors = y_true - y_pred
    sns.histplot(errors, ax=ax2, kde=True)
    ax2.set_title('Error Distribution', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15, figsize=(12, 8)):
    """Plot feature importance for tree-based models."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        sns.barplot(x=top_importances, y=top_features, ax=ax)
        ax.set_title('Feature Importance', fontsize=14)
    else:
        ax.text(0.5, 0.5, "Feature importance not available", ha='center', va='center')
    
    plt.tight_layout()
    return fig


def plot_seasonal_trends(matches_df, figsize=(14, 10)):
    """Plot seasonal trends in IPL matches."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Example placeholder implementation
    if 'season' not in matches_df.columns and 'date' in matches_df.columns:
        matches_df['season'] = matches_df['date'].dt.year
    
    seasons = sorted(matches_df['season'].unique())
    avg_scores = np.random.normal(160, 15, len(seasons))
    
    sns.lineplot(x=seasons, y=avg_scores, marker='o', ax=ax)
    ax.set_title('Average Score by Season', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_match_phase_comparison(deliveries_df, figsize=(14, 10)):
    """Plot comparison of different phases of a match."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Example placeholder implementation
    phases = ['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']
    run_rates = [8.2, 7.5, 10.3]
    
    sns.barplot(x=phases, y=run_rates, palette='viridis', ax=ax)
    ax.set_title('Run Rate by Match Phase', fontsize=14)
    
    plt.tight_layout()
    return fig
