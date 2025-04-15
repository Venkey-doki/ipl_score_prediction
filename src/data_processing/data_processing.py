#!/usr/bin/env python3
"""
Data Processing Module for IPL Score Prediction

This module provides functions for loading, cleaning, and preparing
the IPL cricket match data for score prediction.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the match and deliveries data from CSV files.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple containing (matches_df, deliveries_df)
    """
    logger.info(f"Loading data from {data_dir}...")
    
    # Define file paths
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    
    # Check if files exist
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Matches file not found at: {matches_path}")
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"Deliveries file not found at: {deliveries_path}")
    
    # Load the data
    matches_df = pd.read_csv(matches_path)
    deliveries_df = pd.read_csv(deliveries_path)
    
    logger.info(f"Data loaded successfully. Matches: {len(matches_df)}, Deliveries: {len(deliveries_df)}")
    return matches_df, deliveries_df


def clean_matches_data(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the matches data.
    
    Args:
        matches_df: DataFrame containing match data
        
    Returns:
        Cleaned matches DataFrame
    """
    logger.info("Cleaning matches data...")
    
    # Create a copy to avoid modifying the original
    df = matches_df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values
    if 'winner' in df.columns:
        df['winner'] = df['winner'].fillna('No Result')
    
    # Handle any other missing values
    df = df.fillna(0)
    
    # Drop any duplicate matches
    df = df.drop_duplicates(subset=['id'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Matches data cleaned. Shape: {df.shape}")
    return df


def clean_deliveries_data(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the deliveries (ball-by-ball) data.
    
    Args:
        deliveries_df: DataFrame containing ball-by-ball data
        
    Returns:
        Cleaned deliveries DataFrame
    """
    logger.info("Cleaning deliveries data...")
    
    # Create a copy to avoid modifying the original
    df = deliveries_df.copy()
    
    # Calculate total runs for each delivery
    if 'batsman_runs' in df.columns and 'extra_runs' in df.columns:
        df['total_runs'] = df['batsman_runs'] + df['extra_runs']
    
    # Handle missing values
    df = df.fillna(0)
    
    # Drop any duplicate deliveries
    if 'match_id' in df.columns and 'over' in df.columns and 'ball' in df.columns:
        df = df.drop_duplicates(subset=['match_id', 'inning', 'over', 'ball'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Deliveries data cleaned. Shape: {df.shape}")
    return df


def prepare_score_prediction_data(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    min_overs: float = 5.0,
    max_overs: float = 19.0
) -> pd.DataFrame:
    """
    Prepare data for score prediction modeling.
    
    Args:
        matches_df: Cleaned matches DataFrame
        deliveries_df: Cleaned deliveries DataFrame
        min_overs: Minimum number of overs to consider
        max_overs: Maximum number of overs to consider
        
    Returns:
        DataFrame prepared for score prediction modeling
    """
    logger.info(f"Preparing score prediction data (overs range: {min_overs}-{max_overs})...")
    
    # Get all match IDs
    match_ids = matches_df['id'].unique()
    
    # Initialize list to store prediction data
    prediction_data = []
    
    # Process each match
    for match_id in match_ids:
        # Get match information
        match_info = matches_df[matches_df['id'] == match_id].iloc[0]
        
        # Get match deliveries for first innings only
        match_deliveries = deliveries_df[
            (deliveries_df['match_id'] == match_id) & 
            (deliveries_df['inning'] == 1)
        ]
        
        if len(match_deliveries) == 0:
            continue
        
        # Get total score for the innings
        final_score = match_deliveries['total_runs'].sum()
        
        # Generate prediction points at different overs
        overs_list = np.arange(min_overs, max_overs + 0.1, 1.0)
        
        for current_over in overs_list:
            over_floor = int(current_over)
            ball_decimal = int((current_over - over_floor) * 10) if current_over > over_floor else 0
            
            # Get deliveries up to current over and ball
            current_deliveries = match_deliveries[
                ((match_deliveries['over'] < over_floor)) |
                ((match_deliveries['over'] == over_floor) & (match_deliveries['ball'] <= ball_decimal))
            ]
            
            if len(current_deliveries) == 0:
                continue
            
            # Calculate current statistics
            current_score = current_deliveries['total_runs'].sum()
            wickets_fallen = current_deliveries['player_dismissed'].notna().sum()
            
            # Calculate run rate
            balls_bowled = len(current_deliveries)
            overs_completed = balls_bowled / 6
            run_rate = current_score / overs_completed if overs_completed > 0 else 0
            
            # Calculate last 5 overs performance
            last_30_balls = current_deliveries.iloc[-min(30, len(current_deliveries)):]
            last_5_overs_runs = last_30_balls['total_runs'].sum()
            last_5_overs_wickets = last_30_balls['player_dismissed'].notna().sum()
            
            # Store prediction data point
            prediction_data.append({
                'match_id': match_id,
                'date': match_info['date'],
                'venue': match_info['venue'],
                'batting_team': match_info['team1'] if match_info['toss_winner'] == match_info['team1'] and match_info['toss_decision'] == 'bat' else match_info['team2'],
                'bowling_team': match_info['team2'] if match_info['toss_winner'] == match_info['team1'] and match_info['toss_decision'] == 'bat' else match_info['team1'],
                'current_over': current_over,
                'current_score': current_score,
                'wickets_fallen': wickets_fallen,
                'run_rate': run_rate,
                'last_5_overs_runs': last_5_overs_runs,
                'last_5_overs_wickets': last_5_overs_wickets,
                'final_score': final_score
            })
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_data)
    
    logger.info(f"Score prediction data prepared. Shape: {prediction_df.shape}")
    return prediction_df