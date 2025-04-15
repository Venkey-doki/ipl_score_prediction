"""
Data Processing Module for IPL Score Prediction

This module handles the loading, cleaning, preprocessing, and joining of the IPL match
and deliveries data. It provides functions to extract relevant features for modeling.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the IPL match and deliveries data from CSV files.
    
    Args:
        data_dir: Directory path containing the CSV files
        
    Returns:
        Tuple containing the matches and deliveries DataFrames
    """
    try:
        matches_path = os.path.join(data_dir, 'matches.csv')
        deliveries_path = os.path.join(data_dir, 'deliveries.csv')
        
        logger.info(f"Loading matches data from {matches_path}")
        matches_df = pd.read_csv(matches_path)
        
        logger.info(f"Loading deliveries data from {deliveries_path}")
        deliveries_df = pd.read_csv(deliveries_path)
        
        logger.info(f"Loaded matches: {matches_df.shape[0]} rows, {matches_df.shape[1]} columns")
        logger.info(f"Loaded deliveries: {deliveries_df.shape[0]} rows, {deliveries_df.shape[1]} columns")
        
        return matches_df, deliveries_df
    
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def clean_matches_data(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the matches DataFrame.
    
    Args:
        matches_df: Raw matches DataFrame
        
    Returns:
        Cleaned matches DataFrame
    """
    logger.info("Cleaning matches data")
    
    # Create a copy to avoid modifying the original
    df = matches_df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year from date
    df['year'] = df['date'].dt.year
    
    # Handle missing values in result_margin
    df['result_margin'] = df['result_margin'].fillna(0)
    
    # Fill missing values in city with 'Unknown'
    df['city'] = df['city'].fillna('Unknown')
    
    # Fill missing values in player_of_match with 'Unknown'
    df['player_of_match'] = df['player_of_match'].fillna('Unknown')
    
    # Fill missing values in venue with 'Unknown'
    df['venue'] = df['venue'].fillna('Unknown')
    
    # Handle missing values in winner column (no result/tie matches)
    df['winner'] = df['winner'].fillna('No Result')
    
    # Convert specific columns to appropriate data types
    if 'target_runs' in df.columns:
        df['target_runs'] = df['target_runs'].fillna(0).astype(int)
    
    if 'target_overs' in df.columns:
        df['target_overs'] = df['target_overs'].fillna(0).astype(float)
    
    logger.info("Matches data cleaning completed")
    return df

def clean_deliveries_data(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the deliveries DataFrame.
    
    Args:
        deliveries_df: Raw deliveries DataFrame
        
    Returns:
        Cleaned deliveries DataFrame
    """
    logger.info("Cleaning deliveries data")
    
    # Create a copy to avoid modifying the original
    df = deliveries_df.copy()
    
    # Convert match_id to integer
    df['match_id'] = df['match_id'].astype(int)
    
    # Convert numeric columns to appropriate types
    numeric_cols = ['batsman_runs', 'extra_runs', 'total_runs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Fill missing values in extras_type with 'none'
    if 'extras_type' in df.columns:
        df['extras_type'] = df['extras_type'].fillna('none')
    
    # Handle is_wicket column
    df['is_wicket'] = df['is_wicket'].fillna(0).astype(int)
    
    # Fill missing player dismissal information
    dismissal_cols = ['player_dismissed', 'dismissal_kind', 'fielder']
    for col in dismissal_cols:
        if col in df.columns:
            df[col] = df[col].fillna('NA')
    
    logger.info("Deliveries data cleaning completed")
    return df

def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame using specified strategies.
    
    Args:
        df: Input DataFrame
        strategy: Dictionary mapping column names to strategies ('mean', 'median', 'mode', 'zero', 'value:X')
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy is None:
        strategy = {}
    
    logger.info(f"Handling missing values with strategy: {strategy}")
    result_df = df.copy()
    
    # Check for missing values
    missing_values = result_df.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0].index
    
    for column in columns_with_missing:
        if column in strategy:
            method = strategy[column]
            
            if method == 'mean':
                if pd.api.types.is_numeric_dtype(result_df[column]):
                    result_df[column] = result_df[column].fillna(result_df[column].mean())
            elif method == 'median':
                if pd.api.types.is_numeric_dtype(result_df[column]):
                    result_df[column] = result_df[column].fillna(result_df[column].median())
            elif method == 'mode':
                result_df[column] = result_df[column].fillna(result_df[column].mode()[0])
            elif method == 'zero':
                result_df[column] = result_df[column].fillna(0)
            elif method.startswith('value:'):
                fill_value = method.split(':', 1)[1]
                result_df[column] = result_df[column].fillna(fill_value)
            else:
                logger.warning(f"Unknown strategy '{method}' for column '{column}', using default")
                result_df[column] = result_df[column].fillna(result_df[column].mode()[0] if result_df[column].mode().size > 0 else 'Unknown')
        else:
            # Default strategy: use mode for categorical, mean for numeric
            if pd.api.types.is_numeric_dtype(result_df[column]):
                result_df[column] = result_df[column].fillna(result_df[column].mean())
            else:
                default_value = result_df[column].mode()[0] if result_df[column].mode().size > 0 else 'Unknown'
                result_df[column] = result_df[column].fillna(default_value)
    
    return result_df

def join_match_and_deliveries(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the matches and deliveries DataFrames on match_id.
    
    Args:
        matches_df: Cleaned matches DataFrame
        deliveries_df: Cleaned deliveries DataFrame
        
    Returns:
        Joined DataFrame
    """
    logger.info("Joining matches and deliveries data")
    
    # Ensure match_id is of the same type in both DataFrames
    matches_df['id'] = matches_df['id'].astype(int)
    deliveries_df['match_id'] = deliveries_df['match_id'].astype(int)
    
    # Merge the DataFrames
    merged_df = pd.merge(
        deliveries_df,
        matches_df,
        left_on='match_id',
        right_on='id',
        how='left'
    )
    
    logger.info(f"Joined data shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    return merged_df

def create_innings_summary(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of each innings with key statistics.
    
    Args:
        deliveries_df: Cleaned deliveries DataFrame
        
    Returns:
        DataFrame with innings summary
    """
    logger.info("Creating innings summary")
    
    # Group by match_id and innings
    innings_summary = deliveries_df.groupby(['match_id', 'inning']).agg({
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'batting_team': 'first',
        'bowling_team': 'first',
    }).reset_index()
    
    innings_summary.rename(columns={
        'total_runs': 'innings_runs',
        'is_wicket': 'wickets_lost'
    }, inplace=True)
    
    logger.info(f"Created innings summary with {innings_summary.shape[0]} rows")
    return innings_summary

def extract_batting_features(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract batting-related features from the deliveries data.
    
    Args:
        deliveries_df: Cleaned deliveries DataFrame
        
    Returns:
        DataFrame with batting features
    """
    logger.info("Extracting batting features")
    
    # Calculate runs by over for each match and innings
    over_runs = deliveries_df.groupby(['match_id', 'inning', 'over'])['total_runs'].sum().reset_index()
    
    # Calculate batting statistics
    batting_stats = deliveries_df.groupby(['match_id', 'inning', 'batting_team']).agg({
        'batsman_runs': 'sum',
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'over': 'max'
    }).reset_index()
    
    # Calculate boundaries
    deliveries_df['is_four'] = (deliveries_df['batsman_runs'] == 4).astype(int)
    deliveries_df['is_six'] = (deliveries_df['batsman_runs'] == 6).astype(int)
    
    boundaries = deliveries_df.groupby(['match_id', 'inning', 'batting_team']).agg({
        'is_four': 'sum',
        'is_six': 'sum'
    }).reset_index()
    
    # Merge the features
    batting_features = pd.merge(batting_stats, boundaries, on=['match_id', 'inning', 'batting_team'])
    
    # Calculate run rate
    batting_features['overs_played'] = batting_features['over'] + 1  # Add 1 since overs start from 0
    batting_features['run_rate'] = batting_features['total_runs'] / batting_features['overs_played']
    
    logger.info(f"Extracted batting features with {batting_features.shape[0]} rows")
    return batting_features

def extract_bowling_features(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract bowling-related features from the deliveries data.
    
    Args:
        deliveries_df: Cleaned deliveries DataFrame
        
    Returns:
        DataFrame with bowling features
    """
    logger.info("Extracting bowling features")
    
    # Calculate bowling statistics per bowler
    bowler_stats = deliveries_df.groupby(['match_id', 'inning', 'bowling_team', 'bowler']).agg({
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'ball': 'count'
    }).reset_index()
    
    # Rename columns for clarity
    bowler_stats.rename(columns={
        'total_runs': 'runs_conceded',
        'is_wicket': 'wickets_taken',
        'ball': 'balls_bowled'
    }, inplace=True)
    
    # Calculate economy rate (runs per over)
    bowler_stats['overs_bowled'] = bowler_stats['balls_bowled'] / 6
    bowler_stats['economy_rate'] = bowler_stats['runs_conceded'] / bowler_stats['overs_bowled']
    
    # Aggregate to team level
    team_bowling = bowler_stats.groupby(['match_id', 'inning', 'bowling_team']).agg({
        'runs_conceded': 'sum',
        'wickets_taken': 'sum',
        'economy_rate': 'mean'
    }).reset_index()
    
    logger.info(f"Extracted bowling features with {team_bowling.shape[0]} rows")
    return team_bowling

def prepare_first_innings_data(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for first innings score prediction.
    
    Args:
        matches_df: Cleaned matches DataFrame
        deliveries_df: Cleaned deliveries DataFrame
        
    Returns:
        DataFrame ready for first innings score prediction modeling
    """
    logger.info("Preparing first innings data")
    
    # Filter first innings data
    first_innings = deliveries_df[deliveries_df['inning'] == 1].copy()
    
    # Merge with matches data
    first_innings = pd.merge(
        first_innings,
        matches_df[['id', 'city', 'venue', 'date']],
        left_on='match_id',
        right_on='id',
        how='left'
    )
    
    # Group by match
    match_summary = first_innings.groupby('match_id').agg({
        'batting_team': 'first',
        'bowling_team': 'first',
        'city': 'first',
        'venue': 'first',
        'date': 'first',
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()
    
    match_summary.rename(columns={
        'total_runs': 'first_innings_score',
        'is_wicket': 'wickets_lost'
    }, inplace=True)
    
    logger.info(f"Prepared first innings data with {match_summary.shape[0]} rows")
    return match_summary

def prepare_score_prediction_data(
    matches_df: pd.DataFrame, 
    deliveries_df: pd.DataFrame,
    target_overs: int = 20
) -> pd.DataFrame:
    """
    Prepare data for score prediction at different stages of a match.
    
    Args:
        matches_df: Cleaned matches DataFrame
        deliveries_df: Cleaned deliveries DataFrame
        target_overs: Number of overs in a full match
        
    Returns:
        DataFrame ready for score prediction modeling
    """
    logger.info(f"Preparing score prediction data for {target_overs} overs")
    
    # Filter valid matches and clean the data
    clean_matches = clean_matches_data(matches_df)
    clean_deliveries = clean_deliveries_data(deliveries_df)
    
    # Join the data
    merged_data = join_match_and_deliveries(clean_matches, clean_deliveries)
    
    # Calculate cumulative statistics for each match and inning
    prediction_data = []
    
    # Group by match_id and inning
    for (match_id, inning), group in merged_data.groupby(['match_id', 'inning']):
        # Sort by over and ball
        group = group.sort_values(['over', 'ball'])
        
        # Calculate cumulative statistics at each over
        cum_runs = 0
        cum_wickets = 0
        cum_boundaries = 0
        cum_sixes = 0
        
        for over in range(target_overs):
            # Filter deliveries up to current over
            current_over_data = group[group['over'] <= over]
            
            if len(current_over_data) == 0:
                continue
                
            # Extract match details
            venue = current_over_data['venue'].iloc[0]
            batting_team = current_over_data['batting_team'].iloc[0]
            bowling_team = current_over_data['bowling_team'].iloc[0]
            season = current_over_data['season'].iloc[0]
            
            # Calculate current statistics
            cum_runs = current_over_data['total_runs'].sum()
            cum_wickets = current_over_data['is_wicket'].sum()
            
            # Calculate boundaries
            current_over_data['is_four'] = (current_over_data['batsman_runs'] == 4).astype(int)
            current_over_data['is_six'] = (current_over_data['batsman_runs'] == 6).astype(int)
            cum_boundaries = current_over_data['is_four'].sum()
            cum_sixes = current_over_data['is_six'].sum()
            
            # Calculate run rate
            current_overs = over + 1
            run_rate = cum_runs / current_overs if current_overs > 0 else 0
            
            # Calculate balls remaining
            balls_in_full_innings = target_overs * 6
            balls_bowled = min(current_over_data['over'].max() * 6 + current_over_data['ball'].max(), balls_in_full_innings)
            balls_remaining = balls_in_full_innings - balls_bowled
            
            # Create a row for the prediction data
            row = {
                'match_id': match_id,
                'inning': inning,
                'venue': venue,
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'season': season,
                'current_over': over,
                'current_score': cum_runs,
                'wickets_fallen': cum_wickets,
                'run_rate': run_rate,
                'boundaries': cum_boundaries,
                'sixes': cum_sixes,
                'balls_remaining': balls_remaining
            }
            
            # For first innings, add the final score as target
            if inning == 1:
                # Get the final score
                final_score = group['total_runs'].sum()
                row['final_score'] = final_score
            
            prediction_data.append(row)
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_data)
    
    logger.info(f"Prepared score prediction data with {prediction_df.shape[0]} rows")
    return prediction_df


def get_team_stats(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate team-level statistics from the deliveries and matches data.
    
    Args:
        deliveries_df: Cleaned deliveries DataFrame
        matches_df: Cleaned matches DataFrame
        
    Returns:
        Dictionary containing team batting and bowling statistics DataFrames
    """
    logger.info("Generating team statistics")
    
    # Calculate team batting statistics
    team_batting = deliveries_df.groupby(['match_id', 'inning', 'batting_team']).agg({
        'total_runs': 'sum',
        'batsman_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()
    
    # Calculate team bowling statistics
    team_bowling = deliveries_df.groupby(['match_id', 'inning', 'bowling_team']).agg({
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()
    
    # Calculate match results
    match_results = matches_df[['id', 'team1', 'team2', 'winner']].copy()
    
    # Create a dictionary of team statistics
    team_stats = {
        'batting': team_batting,
        'bowling': team_bowling,
        'results': match_results
    }
    
    logger.info("Team statistics generated")
    return team_stats


if __name__ == "__main__":
    # Example usage of the data processing functions
    data_dir = "../../data"
    
    try:
        # Load data
        matches_df, deliveries_df = load_data(data_dir)
        
        # Clean data
        clean_matches = clean_matches_data(matches_df)
        clean_deliveries = clean_deliveries_data(deliveries_df)
        
        # Example of joining data
        merged_data = join_match_and_deliveries(clean_matches, clean_deliveries)
        
        # Example of preparing data for score prediction
        prediction_data = prepare_score_prediction_data(clean_matches, clean_deliveries)
        
        print(f"Successfully processed data. Prediction data shape: {prediction_data.shape}")
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
