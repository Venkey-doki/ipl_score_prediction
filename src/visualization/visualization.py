"""
Visualization Module for IPL Score Prediction

This module provides functions for creating various visualizations for the IPL score prediction project,
including team performance comparisons, venue statistics, model evaluation, feature importance,
time-series trends, and match phase statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.ticker as ticker
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_team_performance(
    matches_df: pd.DataFrame,
    metric: str = 'win_rate',
    top_n: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create a bar chart comparing team performance.
    
    Args:
        matches_df: DataFrame containing match data
        metric: Metric to visualize ('win_rate', 'total_wins', 'total_matches')
        top_n: Number of top teams to display
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    logger.info(f"Plotting team performance comparison for metric: {metric}")
    
    # Create a copy of the DataFrame
    df = matches_df.copy()
    
    # Calculate team statistics
    team_stats = {}
    
    # Get all teams
    teams = pd.unique(df[['team1', 'team2']].values.ravel('K'))
    
    for team in teams:
        # Count matches played
        team1_matches = df[df['team1'] == team].shape[0]
        team2_matches = df[df['team2'] == team].shape[0]
        total_matches = team1_matches + team2_matches
        
        # Count wins
        wins = df[df['winner'] == team].shape[0]
        
        # Calculate win rate
        win_rate = wins / total_matches if total_matches > 0 else 0
        
        team_stats[team] = {
            'total_matches': total_matches,
            'total_wins': wins,
            'win_rate': win_rate
        }
    
    # Convert to DataFrame
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    team_stats_df.reset_index(inplace=True)
    team_stats_df.rename(columns={'index': 'team'}, inplace=True)
    
    # Sort by the selected metric
    team_stats_df = team_stats_df.sort_values(metric, ascending=False).reset_index(drop=True)
    
    # Get top N teams
    if top_n > 0 and top_n < len(team_stats_df):
        plot_df = team_stats_df.head(top_n)
    else:
        plot_df = team_stats_df
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(
        plot_df['team'],
        plot_df[metric],
        color=sns.color_palette('viridis', len(plot_df))
    )
    
    # Format y-axis as percentage for win_rate
    if metric == 'win_rate':
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # Add labels and title
    ax.set_xlabel('Team', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'IPL Team Performance - {metric.replace("_", " ").title()}', fontsize=16)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        label_text = f'{height:.0f}' if metric != 'win_rate' else f'{height:.1%}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            label_text,
            ha='center',
            va='bottom',
            fontsize=11
        )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Team performance plot saved to {save_path}")
    
    return fig


def plot_venue_statistics(
    matches_df: pd.DataFrame,
    metric: str = 'avg_score',
    top_n: int = 10,
    min_matches: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create a visualization of venue-specific statistics.
    
    Args:
        matches_df: DataFrame containing match data
        metric: Metric to visualize ('avg_score', 'chase_success', 'matches_played')
        top_n: Number of top venues to display
        min_matches: Minimum number of matches played at a venue to include it
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    logger.info(f"Plotting venue statistics for metric: {metric}")
    
    # Create a copy of the DataFrame
    df = matches_df.copy()
    
    # Calculate venue statistics
    venue_stats = {}
    
    for venue, venue_data in df.groupby('venue'):
        total_matches = len(venue_data)
        
        # Skip venues with too few matches
        if total_matches < min_matches:
            continue
        
        # Calculate first innings scores
        first_innings_scores = []
        chase_success = 0
        chase_attempts = 0
        
        for _, match in venue_data.iterrows():
            if 'target_runs' in match and not pd.isna(match['target_runs']) and match['target_runs'] > 0:
                first_innings_scores.append(match['target_runs'])
            
            # Calculate chase success
            if 'toss_winner' in match and 'toss_decision' in match and 'winner' in match:
                toss_winner = match['toss_winner']
                toss_decision = match['toss_decision']
                winner = match['winner']
                
                if toss_decision == 'field':
                    chase_attempts += 1
                    if toss_winner == winner:
                        chase_success += 1
        
        # Calculate statistics
        avg_score = np.mean(first_innings_scores) if first_innings_scores else 0
        chase_success_rate = chase_success / chase_attempts if chase_attempts > 0 else 0
        
        venue_stats[venue] = {
            'matches_played': total_matches,
            'avg_score': avg_score,
            'chase_success': chase_success_rate
        }
    
    # Convert to DataFrame
    venue_stats_df = pd.DataFrame.from_dict(venue_stats, orient='index')
    venue_stats_df.reset_index(inplace=True)
    venue_stats_df.rename(columns={'index': 'venue'}, inplace=True)
    
    # Sort by the selected metric
    if metric == 'chase_success':
        # For chase success, we want venues with most matches as a tiebreaker
        venue_stats_df = venue_stats_df.sort_values(['chase_success', 'matches_played'], ascending=[False, False])
    else:
        venue_stats_df = venue_stats_df.sort_values(metric, ascending=False)
    
    venue_stats_df.reset_index(drop=True, inplace=True)
    
    # Get top N venues
    if top_n > 0 and top_n < len(venue_stats_df):
        plot_df = venue_stats_df.head(top_n)
    else:
        plot_df = venue_stats_df
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(
        plot_df['venue'],
        plot_df[metric],
        color=sns.color_palette('mako', len(plot_df))
    )
    
    # Format y-axis as percentage for chase_success
    if metric == 'chase_success':
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # Add labels and title
    ax.set_xlabel('Venue', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'IPL Venue Statistics - {metric.replace("_", " ").title()}', fontsize=16)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        label_text = f'{height:.0f}' if metric != 'chase_success' else f'{height:.1%}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            label_text,
            ha='center',
            va='bottom',
            fontsize=11
        )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Venue statistics plot saved to {save_path}")
    
    return fig


def plot_model_evaluation(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
    model_name: str = 'Model',
    plot_type: str = 'all',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> Figure:
    """
    Create visualizations for model evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        plot_type: Type of plot ('all', 'scatter', 'error', 'residual')
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    logger.info(f"Creating model evaluation visualizations for {model_name}")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Create residuals
    residuals = y_true - y_pred
    
    # Set up plot
    if plot_type == 'all':
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axs = [ax]
    
    # Function to add scatter plot
    def add_scatter_plot(ax, title_suffix=''):
        ax.scatter(y_true, y_pred, alpha=0.6, color='#1f77b4')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        ax.set_xlabel('Actual Score', fontsize=14)
        ax.set_ylabel('Predicted Score', fontsize=14)
        ax.set_title(f'{model_name} - Actual vs. Predicted {title_suffix}'.strip(), fontsize=16)
        
        # Add metrics as text
        ax.text(
            0.05, 0.95,
            f'RMSE: {rmse:.2f}\nMSE: {mse:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Function to add error histogram
    def add_error_histogram(ax, title_suffix=''):
        ax.hist(residuals, bins=30, alpha=0.7, color='#ff7f0e')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(f'{model_name} - Error Distribution {title_suffix}'.strip(), fontsize=16)
    
    # Function to add residual plot
    def add_residual_plot(ax, title_suffix=''):
        ax.scatter(y_pred, residuals, alpha=0.6, color='#2ca02c')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Score', fontsize=14)
        ax.set_ylabel('Residual (Actual - Predicted)', fontsize=14)
        ax.set_title(f'{model_name} - Residual Plot {title_suffix}'.strip(), fontsize=16)
    
    # Function to add error vs. actual plot
    def add_error_vs_actual_plot(ax, title_suffix=''):
        ax.scatter(y_true, np.abs(residuals), alpha=0.6, color='#d62728')
        ax.set_xlabel('Actual Score', fontsize=14)
        ax.set_ylabel('Absolute Error', fontsize=14)
        ax.set_title(f'{model_name} - Error vs. Actual {title_suffix}'.strip(), fontsize=16)
    
    # Create plots based on the plot_type
    if plot_type == 'all':
        add_scatter_plot(axs[0])
        add_error_histogram(axs[1])
        add_residual_plot(axs[2])
        add_error_vs_actual_plot(axs[3])
    elif plot_type == 'scatter':
        add_scatter_plot(axs[0])
    elif plot_type == 'error':
        add_error_histogram(axs[0])
    elif plot_type == 'residual':
        add_residual_plot(axs[0])
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model evaluation plot saved to {save_path}")
    
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    importance_type: str = 'built_in',
    top_n: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Create a visualization of feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute or coefficients
        feature_names: List of feature names
        X_test: Test feature matrix (required for permutation importance)
        y_test: Test target vector (required for permutation importance)
        importance_type: Type of importance ('built_in', 'permutation')
        top_n: Number of top features to display
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    logger.info(f"Plotting feature importance using {importance_type} method")
    
    # Get feature importance
    if importance_type == 'built_in':
        # Try to get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1 and importances.shape[0] == 1:
                importances = importances.flatten()
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_ attributes")
    elif importance_type == 'permutation':
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test are required for permutation importance")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
    else:
        raise ValueError(f"Unsupported importance_type: {importance_type}")
    
    # Create DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Get top N features
    if top_n > 0 and top_n < len(importance_df):
        plot_df = importance_df.head(top_n)
    else:
        plot_df = importance_df
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = ax.barh(
        plot_df['feature'],
        plot_df['importance'],
        color=sns.color_palette('viridis', len(plot_df))
    )
    
    # Add labels and title
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title(f'Feature Importance ({importance_type.title()})', fontsize=16)
    
    # Invert y-axis to have most important at the top
    ax.invert_yaxis()
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + width * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}',
            va='center',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_seasonal_trends(
    matches_df: pd.DataFrame,
    metric: str = 'avg_score',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create a visualization of time-series trends over IPL seasons.
    
    Args:
        matches_df: DataFrame containing match data
        metric: Metric to visualize ('avg_score', 'total_matches', 'avg_win_margin')
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    logger.info(f"Plotting seasonal trends for metric: {metric}")
    
    # Create a copy of the DataFrame
    df = matches_df.copy()
    
    # Ensure date is in datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
    elif 'season' in df.columns:
        # Extract year from season if it's in format "2008/09"
        if df['season'].dtype == 'object' and '/' in df['season'].iloc[0]:
            df['year'] = df['season'].str.split('/').str[0].astype(int)
        else:
            df['year'] = df['season']
    else:
        logger.warning("No date or season column found in the data")
        return None
    
    # Calculate seasonal statistics
    seasons = []
    
    for year, year_data in df.groupby('year'):
        season_stats = {'year': year}
        
        # Calculate total matches
        season_stats['total_matches'] = len(year_data)
        
        # Calculate average score
        first_innings_scores = []
        for _, match in year_data.iterrows():
            if 'target_runs' in match and not pd.isna(match['target_runs']) and match['target_runs'] > 0:
                first_innings_scores.append(match['target_runs'])
        
        season_stats['avg_score'] = np.mean(first_innings_scores) if first_innings_scores else 0
        
        # Calculate average win margin
        if 'result_margin' in year_data.columns:
            margins = year_data['result_margin'].dropna()
            season_stats['avg_win_margin'] = margins.mean() if len(margins) > 0 else 0
        else:
            season_stats['avg_win_margin'] = 0
        
        seasons.append(season_stats)
    
    # Convert to DataFrame
    seasons_df = pd.DataFrame(seasons)
    
    # Sort by year
    seasons_df = seasons_df.sort_values('year')
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot line chart
    ax.plot(
        seasons_df['year'],
        seasons_df[metric],
        marker='o',
        linewidth=2,
        markersize=8
    )
    
    # Add labels and title
    ax.set_xlabel('Season (Year)', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'IPL Seasonal Trends - {metric.replace("_", " ").title()}', fontsize=16)
    
    # Add data labels
    for x, y in zip(seasons_df['year'], seasons_df[metric]):
        ax.text(x, y + (y * 0.03), f'{y:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-ticks to show all years
    ax.set_xticks(seasons_df['year'])
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Seasonal trends plot saved to {save_path}")
    
    return fig


def plot_match_phase_comparison(
    deliveries_df: pd.DataFrame,
    stat: str = 'runs',
    by: str = 'team',
    top_n: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Figure:
    """
    Create visualizations of statistics by match phase (powerplay, middle, death).
    
    Args:
        deliveries_df: DataFrame containing ball-by-ball data
        stat: Statistic to visualize ('runs', 'run_rate', 'wickets', 'boundaries')
        by: Group by ('team', 'bowler', 'batter')
        top_n: Number of top entries to display
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    logger.info(f"Plotting match phase comparison for {stat} by {by}")
    
    # Create a copy of the DataFrame
    df = deliveries_df.copy()
    
    # Define match phases based on overs
    powerplay_overs = list(range(0, 6))
    middle_overs = list(range(6, 16))
    death_overs = list(range(16, 20))
    
    # Add match phase column
    def get_phase(over):
        if over in powerplay_overs:
            return 'Powerplay (0-6)'
        elif over in middle_overs:
            return 'Middle (7-15)'
        elif over in death_overs:
            return 'Death (16-20)'
        else:
            return 'Other'
    
    df['match_phase'] = df['over'].apply(get_phase)
    
    # Add boundary indicators
    df['is_four'] = (df['batsman_runs'] == 4).astype(int)
    df['is_six'] = (df['batsman_runs'] == 6).astype(int)
    df['is_boundary'] = ((df['is_four'] + df['is_six']) > 0).astype(int)
    
    # Define the group by column
    if by == 'team':
        group_col = 'batting_team'
    elif by == 'bowler':
        group_col = 'bowler'
    elif by == 'batter':
        group_col = 'batter'
    else:
        raise ValueError(f"Unsupported 'by' value: {by}")
    
    # Group by phase and the selected column
    grouped = df.groupby(['match_phase', group_col])
    
    # Calculate statistics
    if stat == 'runs':
        stats = grouped['total_runs'].sum().reset_index(name='value')
        title = f'Total Runs by Match Phase and {by.title()}'
    elif stat == 'run_rate':
        # Count balls to calculate run rate
        balls = grouped.size().reset_index(name='balls')
        runs = grouped['total_runs'].sum().reset_index(name='runs')
        stats = pd.merge(runs, balls, on=['match_phase', group_col])
        stats['value'] = stats['runs'] / (stats['balls'] / 6)
        title = f'Run Rate by Match Phase and {by.title()}'
    elif stat == 'wickets':
        stats = grouped['is_wicket'].sum().reset_index(name='value')
        title = f'Wickets by Match Phase and {by.title()}'
    elif stat == 'boundaries':
        stats = grouped['is_boundary'].sum().reset_index(name='value')
        title = f'Boundaries by Match Phase and {by.title()}'
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    
    # Get top N entries based on total value across all phases
    
