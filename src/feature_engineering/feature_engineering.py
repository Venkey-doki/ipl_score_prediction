#!/usr/bin/env python3
"""
Feature Engineering Module for IPL Score Prediction

This module provides functions for creating and transforming features
from the raw cricket data to improve prediction accuracy.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_team_performance_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture team performance metrics like win rates,
    average scores, and recent form.
    
    Args:
        matches_df: DataFrame containing match data
        
    Returns:
        DataFrame with team performance features
    """
    logger.info("Creating team performance features...")
    
    # Initialize empty list to store features
    team_features = []
    
    # Get unique matches
    unique_matches = matches_df['id'].unique()
    
    for match_id in unique_matches:
        match = matches_df[matches_df['id'] == match_id].iloc[0]
        
        # Basic match info
        team1 = match['team1']
        team2 = match['team2']
        
        # Calculate features up to this match (to prevent data leakage)
        past_matches = matches_df[matches_df['date'] < match['date']]
        
        # Team 1 performance
        team1_matches = past_matches[(past_matches['team1'] == team1) | (past_matches['team2'] == team1)]
        team1_wins = team1_matches[team1_matches['winner'] == team1].shape[0]
        team1_total = team1_matches.shape[0]
        team1_win_rate = team1_wins / team1_total if team1_total > 0 else 0.5
        
        # Team 2 performance
        team2_matches = past_matches[(past_matches['team1'] == team2) | (past_matches['team2'] == team2)]
        team2_wins = team2_matches[team2_matches['winner'] == team2].shape[0]
        team2_total = team2_matches.shape[0]
        team2_win_rate = team2_wins / team2_total if team2_total > 0 else 0.5
        
        # Recent form (last 5 matches)
        team1_recent = team1_matches.sort_values('date', ascending=False).head(5)
        team1_recent_wins = team1_recent[team1_recent['winner'] == team1].shape[0]
        team1_recent_win_rate = team1_recent_wins / team1_recent.shape[0] if team1_recent.shape[0] > 0 else 0.5
        
        team2_recent = team2_matches.sort_values('date', ascending=False).head(5)
        team2_recent_wins = team2_recent[team2_recent['winner'] == team2].shape[0]
        team2_recent_win_rate = team2_recent_wins / team2_recent.shape[0] if team2_recent.shape[0] > 0 else 0.5
        
        # Store features
        team_features.append({
            'match_id': match_id,
            'team1_win_rate': team1_win_rate,
            'team2_win_rate': team2_win_rate,
            'team1_matches_played': team1_total,
            'team2_matches_played': team2_total,
            'team1_recent_win_rate': team1_recent_win_rate,
            'team2_recent_win_rate': team2_recent_win_rate,
        })
    
    # Create DataFrame
    team_features_df = pd.DataFrame(team_features)
    logger.info(f"Created team performance features with {len(team_features_df)} rows")
    
    return team_features_df

def create_venue_features(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture venue-specific characteristics and performance patterns.
    
    Args:
        matches_df: DataFrame containing match data
        deliveries_df: DataFrame containing ball-by-ball data
        
    Returns:
        DataFrame with venue features
    """
    logger.info("Creating venue features...")
    
    # Group matches by venue
    venue_stats = []
    
    for venue in matches_df['venue'].unique():
        # Get matches at this venue
        venue_matches = matches_df[matches_df['venue'] == venue]
        
        # Calculate various venue statistics
        num_matches = venue_matches.shape[0]
        
        # Average first innings score
        first_innings_scores = []
        for match_id in venue_matches['id']:
            match_deliveries = deliveries_df[deliveries_df['match_id'] == match_id]
            first_innings = match_deliveries[match_deliveries['inning'] == 1]
            if not first_innings.empty:
                score = first_innings['total_runs'].sum()
                first_innings_scores.append(score)
        
        avg_first_innings_score = np.mean(first_innings_scores) if first_innings_scores else 0
        
        # Average run rate
        total_runs = deliveries_df[deliveries_df['match_id'].isin(venue_matches['id'])]['total_runs'].sum()
        total_overs = len(deliveries_df[deliveries_df['match_id'].isin(venue_matches['id'])]) / 6
        avg_run_rate = total_runs / total_overs if total_overs > 0 else 0
        
        # Batting or bowling friendly
        # Higher value means more batting friendly
        batting_friendly_score = avg_first_innings_score / 150 if avg_first_innings_score > 0 else 1
        
        # Store venue features
        venue_stats.append({
            'venue': venue,
            'matches_played': num_matches,
            'avg_first_innings_score': avg_first_innings_score,
            'avg_run_rate': avg_run_rate,
            'batting_friendly_score': batting_friendly_score
        })
    
    venue_stats_df = pd.DataFrame(venue_stats)
    logger.info(f"Created venue features with {len(venue_stats_df)} rows")
    
    return venue_stats_df

def create_head_to_head_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on head-to-head records between teams.
    
    Args:
        matches_df: DataFrame containing match data
        
    Returns:
        DataFrame with head-to-head features
    """
    logger.info("Creating head-to-head features...")
    
    # Initialize features list
    h2h_features = []
    
    # Process each match
    for _, match in matches_df.iterrows():
        match_id = match['id']
        team1 = match['team1']
        team2 = match['team2']
        match_date = match['date']
        
        # Get previous matches between these teams
        prev_h2h = matches_df[
            (((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
            ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))) &
            (matches_df['date'] < match_date)
        ]
        
        # Calculate head-to-head stats
        h2h_matches = len(prev_h2h)
        
        # Team1 as reference
        team1_h2h_wins = len(prev_h2h[prev_h2h['winner'] == team1])
        team1_h2h_win_rate = team1_h2h_wins / h2h_matches if h2h_matches > 0 else 0.5
        
        # Team2 as reference
        team2_h2h_wins = len(prev_h2h[prev_h2h['winner'] == team2])
        team2_h2h_win_rate = team2_h2h_wins / h2h_matches if h2h_matches > 0 else 0.5
        
        # Store features
        h2h_features.append({
            'match_id': match_id,
            'h2h_matches': h2h_matches,
            'team1_h2h_wins': team1_h2h_wins,
            'team1_h2h_win_rate': team1_h2h_win_rate,
            'team2_h2h_wins': team2_h2h_wins,
            'team2_h2h_win_rate': team2_h2h_win_rate
        })
    
    h2h_df = pd.DataFrame(h2h_features)
    logger.info(f"Created head-to-head features with {len(h2h_df)} rows")
    
    return h2h_df

def combine_all_features(
    base_df: pd.DataFrame, 
    team_features_df: Optional[pd.DataFrame] = None,
    venue_features_df: Optional[pd.DataFrame] = None,
    h2h_features_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Combine all feature sets into a single DataFrame.
    
    Args:
        base_df: Base DataFrame with basic features
        team_features_df: DataFrame with team performance features
        venue_features_df: DataFrame with venue features
        h2h_features_df: DataFrame with head-to-head features
        
    Returns:
        Combined DataFrame with all features
    """
    logger.info("Combining all features...")
    
    # Start with the base DataFrame
    combined_df = base_df.copy()
    
    # Merge team features if available
    if team_features_df is not None and not team_features_df.empty:
        combined_df = pd.merge(
            combined_df,
            team_features_df,
            on='match_id',
            how='left'
        )
    
    # Merge venue features if available
    if venue_features_df is not None and not venue_features_df.empty:
        combined_df = pd.merge(
            combined_df,
            venue_features_df,
            on='venue',
            how='left'
        )
    
    # Merge head-to-head features if available
    if h2h_features_df is not None and not h2h_features_df.empty:
        combined_df = pd.merge(
            combined_df,
            h2h_features_df,
            on='match_id',
            how='left'
        )
    
    # Handle missing values
    combined_df = combined_df.fillna(0)
    
    logger.info(f"Combined features DataFrame shape: {combined_df.shape}")
    return combined_df