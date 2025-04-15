"""
Modeling Module for IPL Score Prediction

This module provides functions for training, evaluating, and using machine learning models
to predict IPL cricket match scores. It includes implementations for multiple algorithms,
model evaluation metrics, hyperparameter tuning, and model serialization.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    chronological: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    
    Args:
        df: DataFrame containing features and target
        target_col: Name of the target column
        test_size: Proportion of the data to include in the test split
        random_state: Random seed for reproducibility
        chronological: If True, splits data chronologically (newer matches for testing)
        
    Returns:
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}, chronological={chronological}")
    
    # Create feature matrix and target vector
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if chronological and 'date' in df.columns:
        # Sort by date
        df = df.sort_values('date')
        
        # Determine split point
        split_idx = int(len(df) * (1 - test_size))
        
        # Split data
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Chronological split: training on data until {X_train['date'].max()}, testing on newer data")
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Random split with {len(X_train)} training samples and {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: List[str] = None,
    numerical_features: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Preprocess the data by scaling numerical features and encoding categorical features.
    
    Args:
        X_train: Training feature matrix
        X_test: Testing feature matrix
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        
    Returns:
        Tuple containing (X_train_processed, X_test_processed, preprocessor)
    """
    logger.info("Preprocessing data")
    
    # If feature lists are not provided, infer them from the data
    if categorical_features is None:
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numerical_features is None:
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove date columns if they exist
        numerical_features = [col for col in numerical_features if 'date' not in col.lower()]
        # Remove ID columns if they exist
        numerical_features = [col for col in numerical_features if not col.endswith('_id') and col != 'match_id']
    
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")
    
    # Create preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit preprocessor on training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Processed data shapes: X_train {X_train_processed.shape}, X_test {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, preprocessor


def train_linear_regression(
    X_train: np.ndarray,
    y_train: pd.Series
) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Args:
        X_train: Processed training feature matrix
        y_train: Training target vector
        
    Returns:
        Trained Linear Regression model
    """
    logger.info("Training Linear Regression model")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a Random Forest Regression model.
    
    Args:
        X_train: Processed training feature matrix
        y_train: Training target vector
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest Regression model
    """
    logger.info(f"Training Random Forest model with {n_estimators} estimators")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
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
        X_train: Processed training feature matrix
        y_train: Training target vector
        n_estimators: Number of boosting stages
        learning_rate: Learning rate shrinks the contribution of each tree
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Gradient Boosting Regression model
    """
    logger.info(f"Training Gradient Boosting model with {n_estimators} estimators, learning_rate={learning_rate}")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate the model performance on the test data.
    
    Args:
        model: Trained model
        X_test: Processed test feature matrix
        y_test: Test target vector
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"{model_name} metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
    
    return metrics


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_type: str = 'rf',
    param_grid: Optional[Dict] = None,
    cv: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune hyperparameters using grid search with cross-validation.
    
    Args:
        X_train: Processed training feature matrix
        y_train: Training target vector
        model_type: Type of model to tune ('lr', 'rf', or 'gb')
        param_grid: Dictionary of parameters to tune
        cv: Number of cross-validation folds
        
    Returns:
        Tuple containing (best_model, best_params)
    """
    logger.info(f"Tuning hyperparameters for {model_type} model with {cv}-fold CV")
    
    # Set default parameter grids if not provided
    if param_grid is None:
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42)
        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'lr':
            # Not much to tune for Linear Regression
            param_grid = {
                'fit_intercept': [True, False],
                'normalize': [True, False] if hasattr(LinearRegression(), 'normalize') else None
            }
            base_model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {-grid_search.best_score_:.2f} MSE")
    
    return best_model, best_params


def save_model(
    model: Any,
    preprocessor: ColumnTransformer,
    model_dir: str,
    model_name: str = "model",
    include_timestamp: bool = True
) -> str:
    """
    Save the trained model and preprocessor to disk.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        model_dir: Directory to save the model
        model_name: Base name for the saved model
        include_timestamp: Whether to include a timestamp in the filename
        
    Returns:
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create filename
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.pkl"
    else:
        filename = f"{model_name}.pkl"
    
    filepath = os.path.join(model_dir, filename)
    
    # Save model and preprocessor
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {filepath}")
    
    return filepath


def load_model(filepath: str) -> Tuple[Any, ColumnTransformer]:
    """
    Load a trained model and preprocessor from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Tuple containing (model, preprocessor)
    """
    logger.info(f"Loading model from {filepath}")
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    
    return model, preprocessor


def predict(
    model: Any,
    preprocessor: ColumnTransformer,
    input_data: Union[pd.DataFrame, Dict[str, Any]],
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None
) -> float:
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        input_data: Input data as a DataFrame or dictionary
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        

