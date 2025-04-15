#!/usr/bin/env python3
"""
IPL Score Prediction Project - Main Module

This module serves as the entry point for the IPL score prediction project,
implementing the complete workflow from data loading to model deployment.
It provides command-line arguments for different operations like training,
evaluation, prediction, and visualization.

Usage:
    python main.py train             # Train the model using the dataset
    python main.py evaluate          # Evaluate the trained model
    python main.py predict           # Make predictions using sample data
    python main.py visualize         # Generate visualizations
    python main.py workflow          # Run the entire workflow

Example:
    python main.py train --model-type rf
"""

import os
import sys
import argparse
import logging
import time
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt

# Import project modules
from src.data_processing.data_processing import (
    load_data, clean_matches_data, clean_deliveries_data, prepare_score_prediction_data
)
from src.feature_engineering.feature_engineering import (
    create_team_performance_features, create_venue_features, create_head_to_head_features,
    combine_all_features
)
from src.modeling.modeling import (
    split_data, preprocess_data, train_linear_regression, train_random_forest, train_gradient_boosting,
    evaluate_model, tune_hyperparameters, save_model, load_model
)
from src.visualization.visualization import (
    plot_team_performance, plot_venue_statistics, plot_model_evaluation, plot_feature_importance,
    plot_seasonal_trends, plot_match_phase_comparison
)
from src.api.api import IPLScorePredictor, PredictionInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ipl_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Project directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
VISUALIZATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")

# Default model path
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "ipl_score_model.pkl")


def setup_directories() -> None:
    """
    Set up the project directories if they don't exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    logger.info(f"Project directories set up: {DATA_DIR}, {MODELS_DIR}, {VISUALIZATIONS_DIR}")


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the data for modeling.
    
    Returns:
        Tuple containing (cleaned_matches, cleaned_deliveries, prediction_data)
    """
    logger.info("Loading and preparing data...")
    try:
        # Load data
        matches_df, deliveries_df = load_data(DATA_DIR)
        
        # Clean data
        cleaned_matches = clean_matches_data(matches_df)
        cleaned_deliveries = clean_deliveries_data(deliveries_df)
        
        # Prepare prediction data
        prediction_data = prepare_score_prediction_data(cleaned_matches, cleaned_deliveries)
        
        logger.info(f"Data preparation completed. Prediction data shape: {prediction_data.shape}")
        return cleaned_matches, cleaned_deliveries, prediction_data
    
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise


def engineer_features(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    prediction_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Engineer features for the model.
    
    Args:
        matches_df: Cleaned matches DataFrame
        deliveries_df: Cleaned deliveries DataFrame
        prediction_data: Optional prediction data to add features to
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    try:
        # Create features
        team_performance = create_team_performance_features(matches_df)
        venue_features = create_venue_features(matches_df, deliveries_df)
        h2h_features = create_head_to_head_features(matches_df)
        
        # Combine features
        if prediction_data is not None:
            # Use provided prediction data as the base
            model_data = prediction_data.copy()
        else:
            # Create prediction data if not provided
            model_data = prepare_score_prediction_data(matches_df, deliveries_df)
        
        # Merge team performance features
        if len(team_performance) > 0:
            model_data = pd.merge(
                model_data,
                team_performance,
                on='match_id',
                how='left'
            )
        
        # Merge venue features
        if len(venue_features) > 0:
            model_data = pd.merge(
                model_data,
                venue_features,
                on='venue',
                how='left'
            )
        
        # Merge head-to-head features
        if len(h2h_features) > 0:
            model_data = pd.merge(
                model_data,
                h2h_features,
                on='match_id',
                how='left'
            )
        
        # Drop any duplicate columns
        model_data = model_data.loc[:, ~model_data.columns.duplicated()]
        
        logger.info(f"Feature engineering completed. Model data shape: {model_data.shape}")
        return model_data
    
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


def train_model(
    model_data: pd.DataFrame,
    target_col: str = 'final_score',
    model_type: str = 'rf',
    tune: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Any, Any, Dict[str, float], List[str]]:
    """
    Train a model using the prepared data.
    
    Args:
        model_data: DataFrame with features and target
        target_col: Name of the target column
        model_type: Type of model to train ('lr', 'rf', 'gb')
        tune: Whether to tune hyperparameters
        save_path: Path to save the model (if None, uses default)
        
    Returns:
        Tuple containing (model, preprocessor, metrics, feature_names)
    """
    logger.info(f"Training {model_type.upper()} model...")
    try:
        # Ensure the target column exists
        if target_col not in model_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in the data")
        
        # Drop non-feature columns
        drop_cols = ['match_id', 'date', 'id', 'team1_x', 'team2_x', 'team1_y', 'team2_y']
        feature_cols = [col for col in model_data.columns if col != target_col and col not in drop_cols]
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            model_data[feature_cols + [target_col]], 
            target_col=target_col,
            test_size=0.2,
            chronological=True
        )
        
        # Preprocess data
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        X_train_processed, X_test_processed, preprocessor = preprocess_data(
            X_train, X_test, categorical_features, numerical_features
        )
        
        # Train model based on type
        if model_type == 'lr':
            model = train_linear_regression(X_train_processed, y_train)
        elif model_type == 'rf':
            if tune:
                model, _ = tune_hyperparameters(X_train_processed, y_train, model_type='rf')
            else:
                model = train_random_forest(X_train_processed, y_train)
        elif model_type == 'gb':
            if tune:
                model, _ = tune_hyperparameters(X_train_processed, y_train, model_type='gb')
            else:
                model = train_gradient_boosting(X_train_processed, y_train)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Evaluate model
        metrics = evaluate_model(model, X_test_processed, y_test, model_name=model_type.upper())
        
        # Save model
        if save_path is None:
            save_path = os.path.join(MODELS_DIR, f"ipl_score_{model_type}_model.pkl")
        
        save_model(model, preprocessor, MODELS_DIR, model_name=f"ipl_score_{model_type}_model")
        
        logger.info(f"Model training completed. Metrics: {metrics}")
        return model, preprocessor, metrics, feature_cols
    
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def evaluate_trained_model(
    model_path: Optional[str] = None,
    test_data: Optional[pd.DataFrame] = None,
    target_col: str = 'final_score'
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model (if None, uses default)
        test_data: Test data for evaluation (if None, loads and prepares data)
        target_col: Name of the target column
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating trained model...")
    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        # Load model
        model, preprocessor = load_model(model_path)
        
        # Prepare test data if not provided
        if test_data is None:
            matches_df, deliveries_df, prediction_data = load_and_prepare_data()
            test_data = engineer_features(matches_df, deliveries_df, prediction_data)
        
        # Ensure the target column exists
        if target_col not in test_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in the test data")
        
        # Split to get test data only
        _, X_test, _, y_test = split_data(
            test_data, 
            target_col=target_col,
            test_size=0.2,
            chronological=True
        )
        
        # Preprocess test data
        drop_cols = ['match_id', 'date', 'id', 'team1_x', 'team2_x', 'team1_y', 'team2_y']
        feature_cols = [col for col in X_test.columns if col not in drop_cols]
        X_test = X_test[feature_cols]
        
        # Apply preprocessing
        X_test_processed = preprocessor.transform(X_test)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test_processed, y_test, model_name="Trained Model")
        
        # Create evaluation visualizations
        fig = plot_model_evaluation(y_test, model.predict(X_test_processed))
        visualization_path = os.path.join(VISUALIZATIONS_DIR, "model_evaluation.png")
        fig.savefig(visualization_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise


def make_prediction(
    input_data: Dict[str, Any],
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make a prediction using the trained model.
    
    Args:
        input_data: Input data for prediction
        model_path: Path to the trained model (if None, uses default)
        
    Returns:
        Dictionary with prediction results
    """
    logger.info("Making prediction...")
    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        # Create predictor
        predictor = IPLScorePredictor(model_path)
        
        # Make prediction
        result = predictor.predict(input_data)
        
        logger.info(f"Prediction completed: {result.predicted_score} runs")
        return result.dict()
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise


def create_visualizations(
    matches_df: Optional[pd.DataFrame] = None,
    deliveries_df: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    feature_names: Optional[List[str]] = None
) -> List[str]:
    """
    Create visualizations for the project.
    
    Args:
        matches_df: Matches DataFrame (if None, loads data)
        deliveries_df: Deliveries DataFrame (if None, loads data)
        model: Trained model for feature importance (if None, loads model)
        preprocessor: Fitted preprocessor (if None, loads model)
        feature_names: List of feature names (if None, infers from model)
        
    Returns:
        List of paths to the generated visualizations
    """
    logger.info("Creating visualizations...")
    try:
        # Load data if not provided
        if matches_df is None or deliveries_df is None:
            matches_df, deliveries_df = load_data(DATA_DIR)
            matches_df = clean_matches_data(matches_df)
            deliveries_df = clean_deliveries_data(deliveries_df)
        
        # Create visualization directory
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        
        # Create team performance visualization
        team_viz_path = os.path.join(VISUALIZATIONS_DIR, "team_performance.png")
        fig = plot_team_performance(matches_df, metric='win_rate')
        fig.savefig(team_viz_path, dpi=300, bbox_inches='tight')
        
        # Create venue statistics visualization
        venue_viz_path = os.path.join(VISUALIZATIONS_DIR, "venue_statistics.png")
        fig = plot_venue_statistics(deliveries_df, matches_df)
        fig.savefig(venue_viz_path, dpi=300, bbox_inches='tight')

        # Create seasonal trends visualization
        seasonal_viz_path = os.path.join(VISUALIZATIONS_DIR, "seasonal_trends.png")
        fig = plot_seasonal_trends(matches_df)
        fig.savefig(seasonal_viz_path, dpi=300, bbox_inches='tight')

        # Create match phase comparison visualization
        match_phase_viz_path = os.path.join(VISUALIZATIONS_DIR, "match_phase_comparison.png")
        fig = plot_match_phase_comparison(deliveries_df)
        fig.savefig(match_phase_viz_path, dpi=300, bbox_inches='tight')

        # Feature importance (if model and features available)
        if model is not None and feature_names is not None:
            feature_importance_path = os.path.join(VISUALIZATIONS_DIR, "feature_importance.png")
            fig = plot_feature_importance(model, feature_names)
            fig.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        else:
            feature_importance_path = None

        paths = [team_viz_path, venue_viz_path, seasonal_viz_path, match_phase_viz_path]
        if feature_importance_path:
            paths.append(feature_importance_path)

        logger.info("Visualizations created successfully.")
        return paths

    except Exception as e:
        logger.error(f"Error in creating visualizations: {e}")
        raise


def run_workflow(model_type: str = 'rf', tune: bool = True):
    """
    Run the complete workflow from data preparation to model training and evaluation.

    Args:
        model_type: Type of model to use ('lr', 'rf', 'gb')
        tune: Whether to tune hyperparameters
    """
    logger.info("Starting complete workflow...")
    try:
        setup_directories()
        matches_df, deliveries_df, prediction_data = load_and_prepare_data()
        model_data = engineer_features(matches_df, deliveries_df, prediction_data)
        model, preprocessor, metrics, feature_names = train_model(
            model_data=model_data,
            model_type=model_type,
            tune=tune
        )
        evaluate_trained_model(os.path.join(MODELS_DIR, f'ipl_score_{model_type}_model.pkl'), test_data=model_data)
        create_visualizations(matches_df, deliveries_df, model, preprocessor, feature_names)
        logger.info("Workflow completed successfully.")
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="IPL Score Prediction CLI")
    parser.add_argument("action", choices=["train", "evaluate", "predict", "visualize", "workflow"], help="Action to perform")
    parser.add_argument("--model-type", choices=["lr", "rf", "gb"], default="rf", help="Type of model to train")
    parser.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning")
    parser.add_argument("--input", type=str, help="Path to input JSON file for prediction")
    parser.add_argument("--model-path", type=str, help="Path to trained model file")

    args = parser.parse_args()

    if args.action == "train":
        setup_directories()
        matches_df, deliveries_df, prediction_data = load_and_prepare_data()
        model_data = engineer_features(matches_df, deliveries_df, prediction_data)
        train_model(model_data, model_type=args.model_type, tune=not args.no_tune)
    elif args.action == "evaluate":
        evaluate_trained_model(args.model_path)
    elif args.action == "predict":
        if not args.input:
            logger.error("Input file path required for prediction.")
            sys.exit(1)
        with open(args.input, 'r') as f:
            input_data = eval(f.read())  # Replace with json.load(f) for strict JSON input
        result = make_prediction(input_data)
        print("Prediction Result:", result)
    elif args.action == "visualize":
        create_visualizations()
    elif args.action == "workflow":
        run_workflow(model_type=args.model_type, tune=not args.no_tune)


if __name__ == "__main__":
    main()

