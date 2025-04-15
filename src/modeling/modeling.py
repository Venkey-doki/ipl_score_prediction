#!/usr/bin/env python3
"""
Modeling Module for IPL Score Prediction

This module provides functions for training and evaluating machine learning models
to predict IPL cricket match scores.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_data(
    data: pd.DataFrame,
    target_col: str = 'final_score',
    test_size: float = 0.2,
    random_state: int = 42,
    chronological: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    
    Args:
        data: DataFrame containing features and target
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        chronological: Whether to split chronologically (if True) or randomly
        
    Returns:
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}, chronological={chronological}")
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    if chronological:
        # Sort by date if available, otherwise by index
        if 'date' in X.columns:
            X = X.sort_values('date')
            y = y[X.index]
        
        # Use the last test_size proportion of data for testing
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    logger.info(f"Data split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str]
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Preprocess the data by encoding categorical features and scaling numerical features.
    
    Args:
        X_train: Training features
        X_test: Testing features
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        
    Returns:
        Tuple containing (X_train_processed, X_test_processed, preprocessor)
    """
    logger.info("Preprocessing data...")
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit the preprocessor and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Preprocessing complete. X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}")
    return X_train_processed, X_test_processed, preprocessor


def train_linear_regression(
    X_train: np.ndarray,
    y_train: pd.Series
) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target values
        
    Returns:
        Trained LinearRegression model
    """
    logger.info("Training Linear Regression model...")
    
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Linear Regression model trained in {training_time:.2f} seconds")
    
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a Random Forest Regression model.
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target values
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        random_state: Random seed for reproducibility
        
    Returns:
        Trained RandomForestRegressor model
    """
    logger.info(f"Training Random Forest model with n_estimators={n_estimators}...")
    
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Random Forest model trained in {training_time:.2f} seconds")
    
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting Regression model.
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target values
        n_estimators: Number of boosting stages
        learning_rate: Learning rate
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained GradientBoostingRegressor model
    """
    logger.info(f"Training Gradient Boosting model with n_estimators={n_estimators}, learning_rate={learning_rate}...")
    
    start_time = time.time()
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Gradient Boosting model trained in {training_time:.2f} seconds")
    
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on the test data.
    
    Args:
        model: Trained model
        X_test: Preprocessed test features
        y_test: Test target values
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log results
    logger.info(f"{model_name} Evaluation:")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  R² Score: {r2:.3f}")
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_type: str = 'rf'
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune hyperparameters for the specified model type.
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target values
        model_type: Type of model to tune ('lr', 'rf', 'gb')
        
    Returns:
        Tuple containing (best_model, best_params)
    """
    logger.info(f"Tuning hyperparameters for {model_type.upper()} model...")
    
    if model_type == 'lr':
        # Linear Regression has no hyperparameters to tune
        model = train_linear_regression(X_train, y_train)
        return model, {}
    
    elif model_type == 'rf':
        # Define the model and parameter grid
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
    elif model_type == 'gb':
        # Define the model and parameter grid
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    logger.info(f"Hyperparameter tuning complete. Best parameters: {best_params}")
    return best_model, best_params


def save_model(
    model: Any,
    preprocessor: Any,
    model_dir: str,
    model_name: str = "ipl_score_model"
) -> str:
    """
    Save the trained model and preprocessor to disk.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        model_dir: Directory to save the model in
        model_name: Name for the saved model file
        
    Returns:
        Path to the saved model file
    """
    logger.info(f"Saving model to {model_dir}...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a dictionary with model and preprocessor
    model_data = {
        'model': model,
        'preprocessor': preprocessor
    }
    
    # Save the model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model_data, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model_path


def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    Load a trained model and preprocessor from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Tuple containing (model, preprocessor)
    """
    logger.info(f"Loading model from {model_path}...")
    
    # Load the model data
    model_data = joblib.load(model_path)
    
    # Extract model and preprocessor
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    
    logger.info(f"Model loaded successfully from {model_path}")
    return model, preprocessor