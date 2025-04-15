"""
API Module for IPL Score Prediction

This module provides the core functionality for the IPL score prediction API,
including model loading, input validation, and prediction functions.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from ..modeling.modeling import load_model
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionInput(BaseModel):
    """
    Model for validating prediction input data.
    """
    # Match information
    match_id: Optional[int] = Field(None, description="Unique match identifier")
    venue: str = Field(..., description="Name of the venue")
    batting_team: str = Field(..., description="Name of the batting team")
    bowling_team: str = Field(..., description="Name of the bowling team")
    
    # Current state (when predicting during a match)
    current_score: int = Field(..., ge=0, description="Current score")
    current_over: float = Field(..., ge=0, le=20, description="Current over (e.g., 7.2 for 7 overs and 2 balls)")
    wickets_fallen: int = Field(..., ge=0, le=10, description="Number of wickets fallen")
    
    # Optional features
    season: Optional[str] = Field(None, description="IPL season")
    run_rate: Optional[float] = Field(None, ge=0, description="Current run rate")
    last_five_overs_runs: Optional[int] = Field(None, ge=0, description="Runs in last 5 overs")
    last_five_overs_wickets: Optional[int] = Field(None, ge=0, le=10, description="Wickets in last 5 overs")
    
    # Team form (optional)
    batting_team_win_rate: Optional[float] = Field(None, ge=0, le=1, description="Batting team's win rate")
    bowling_team_win_rate: Optional[float] = Field(None, ge=0, le=1, description="Bowling team's win rate")
    head_to_head_win_rate: Optional[float] = Field(None, ge=0, le=1, description="Batting team's win rate against bowling team")
    
    # Venue stats (optional)
    venue_avg_first_innings_score: Optional[float] = Field(None, ge=0, description="Average first innings score at venue")

    @validator('current_over')
    def check_over_format(cls, v):
        # Ensure the over is in a valid format (e.g., 7.2 means 7 overs and 2 balls)
        full_overs = int(v)
        balls = int((v - full_overs) * 10)
        if balls >= 6:
            raise ValueError("Balls component of over should be between 0 and 5")
        return v

    @validator('bowling_team')
    def check_teams_different(cls, v, values):
        if 'batting_team' in values and v == values['batting_team']:
            raise ValueError("Batting and bowling teams must be different")
        return v


class PredictionResult(BaseModel):
    """
    Model for prediction results.
    """
    predicted_score: int
    confidence_interval_lower: Optional[int] = None
    confidence_interval_upper: Optional[int] = None
    message: str


class PredictionError(BaseModel):
    """
    Model for prediction errors.
    """
    error: str
    details: Optional[Dict[str, Any]] = None


class IPLScorePredictor:
    """
    Class for making IPL score predictions using a trained model.
    """
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        logger.info(f"Initializing IPL Score Predictor with model at {model_path}")
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model from the specified path.
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model, self.preprocessor = load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, input_data: Union[Dict[str, Any], PredictionInput]) -> PredictionResult:
        """
        Make a prediction based on the input data.
        
        Args:
            input_data: Input data for prediction, either as a dictionary or PredictionInput object
            
        Returns:
            PredictionResult object with the prediction details
        """
        try:
            # Validate input if it's a dictionary
            if isinstance(input_data, dict):
                input_data = PredictionInput(**input_data)
            
            # Convert to DataFrame for preprocessing
            input_df = self._prepare_input_data(input_data)
            
            # Make prediction
            predicted_score = self._predict_score(input_df)
            
            # Create prediction result
            result = PredictionResult(
                predicted_score=int(round(predicted_score)),
                confidence_interval_lower=int(round(predicted_score * 0.9)),  # Simplified confidence interval
                confidence_interval_upper=int(round(predicted_score * 1.1)),  # In a real app, this would be more sophisticated
                message="Prediction successful"
            )
            
            logger.info(f"Prediction made: {result.predicted_score} runs")
            return result
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _prepare_input_data(self, input_data: PredictionInput) -> pd.DataFrame:
        """
        Convert the input data to a DataFrame suitable for the model.
        
        Args:
            input_data: Validated input data
            
        Returns:
            DataFrame ready for prediction
        """
        # Convert to dictionary
        data_dict = input_data.dict()
        
        # Create DataFrame
        input_df = pd.DataFrame([data_dict])
        
        # Calculate run rate if not provided
        if input_df['run_rate'].isnull().any():
            # Extract overs and balls
            full_overs = input_df['current_over'].iloc[0].astype(int)
            balls = int((input_df['current_over'].iloc[0] - full_overs) * 10)
            
            # Calculate total balls
            total_balls = full_overs * 6 + balls
            
            # Calculate run rate (runs per over)
            if total_balls > 0:
                input_df['run_rate'] = input_df['current_score'] / (total_balls / 6)
            else:
                input_df['run_rate'] = 0
        
        # Fill missing values with defaults that work with the model
        input_df['season'].fillna('unknown', inplace=True)
        input_df['last_five_overs_runs'].fillna(0, inplace=True)
        input_df['last_five_overs_wickets'].fillna(0, inplace=True)
        input_df['batting_team_win_rate'].fillna(0.5, inplace=True)
        input_df['bowling_team_win_rate'].fillna(0.5, inplace=True)
        input_df['head_to_head_win_rate'].fillna(0.5, inplace=True)
        input_df['venue_avg_first_innings_score'].fillna(160, inplace=True)  # Default to typical T20 score
        
        return input_df
    
    def _predict_score(self, input_df: pd.DataFrame) -> float:
        """
        Make a prediction using the trained model.
        
        Args:
            input_df: Prepared input DataFrame
            
        Returns:
            Predicted score as a float
        """
        if self.model is None or self.preprocessor is None:
            self.load_model()
        
        # Apply preprocessing
        preprocessed_data = self.preprocessor.transform(input_df)
        
        # Make prediction
        prediction = self.model.predict(preprocessed_data)
        
        # Return the first (and only) prediction
        return float(prediction[0])


def get_predictor(model_path: str) -> IPLScorePredictor:
    """
    Get or create an IPLScorePredictor instance.
    This is useful for singleton pattern when used with web frameworks.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        IPLScorePredictor instance
    """
    return IPLScorePredictor(model_path)

