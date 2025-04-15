#!/usr/bin/env python3
"""
API Module for IPL Score Prediction

This module provides the API functionality for the IPL score prediction model.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionInput(BaseModel):
    """
    Input model for score prediction.
    """
    match_id: Optional[int] = Field(None, description="Match ID (optional)")
    venue: str = Field(..., description="Stadium/venue name")
    batting_team: str = Field(..., description="Name of the batting team")
    bowling_team: str = Field(..., description="Name of the bowling team")
    current_score: int = Field(..., description="Current score")
    current_over: float = Field(..., description="Current over (e.g., 10.2)")
    wickets_fallen: int = Field(..., description="Number of wickets fallen")
    season: Optional[str] = Field(None, description="IPL season")
    run_rate: Optional[float] = Field(None, description="Current run rate")
    last_five_overs_runs: Optional[int] = Field(None, description="Runs scored in the last 5 overs")
    last_five_overs_wickets: Optional[int] = Field(None, description="Wickets fallen in the last 5 overs")
    batting_team_win_rate: Optional[float] = Field(None, description="Batting team's win rate")
    bowling_team_win_rate: Optional[float] = Field(None, description="Bowling team's win rate")
    head_to_head_win_rate: Optional[float] = Field(None, description="Batting team's win rate against this opponent")
    venue_avg_first_innings_score: Optional[float] = Field(None, description="Average first innings score at this venue")
    
    @validator('current_over')
    def validate_over(cls, value):
        # Ensure over is in valid format (e.g., 10.2 for 10 overs and 2 balls)
        whole_overs = int(value)
        balls = int((value - whole_overs) * 10)
        if balls >= 6:
            raise ValueError("Balls in an over cannot exceed 5")
        return value
    
    @validator('wickets_fallen')
    def validate_wickets(cls, value):
        if value < 0 or value > 10:
            raise ValueError("Wickets must be between 0 and 10")
        return value


class PredictionResult(BaseModel):
    """
    Result model for score prediction.
    """
    predicted_score: int = Field(..., description="Predicted final score")
    confidence_interval_lower: Optional[int] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_upper: Optional[int] = Field(None, description="Upper bound of confidence interval")
    message: str = Field("Prediction successful", description="Status message")


class PredictionError(BaseModel):
    """
    Error model for prediction failures.
    """
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class IPLScorePredictor:
    """
    IPL score prediction class using machine learning models.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        logger.info(f"Initializing IPL Score Predictor with model: {model_path}")
        
        try:
            # Load model and preprocessor
            model_data = joblib.load(model_path)
            self.model = model_data.get('model')
            self.preprocessor = model_data.get('preprocessor')
            
            if self.model is None or self.preprocessor is None:
                raise ValueError("Model file does not contain valid model and preprocessor")
            
            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """
        Make a prediction based on the input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            PredictionResult with the predicted score
        """
        logger.info(f"Making prediction for match at {input_data.venue}")
        
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data.dict()])
            
            # Fill missing calculated fields
            if input_data.run_rate is None:
                total_balls = int(input_data.current_over) * 6 + int((input_data.current_over - int(input_data.current_over)) * 10)
                input_df['run_rate'] = input_data.current_score / (total_balls / 6) if total_balls > 0 else 0
            
            # Preprocess the input
            input_processed = self.preprocessor.transform(input_df)
            
            # Make prediction
            predicted_score = self.model.predict(input_processed)[0]
            predicted_score = max(int(round(predicted_score)), input_data.current_score)
            
            # Calculate confidence interval (simplified approach)
            confidence_margin = int(round(predicted_score * 0.1))  # 10% margin
            confidence_lower = predicted_score - confidence_margin
            confidence_upper = predicted_score + confidence_margin
            
            # Create result
            result = PredictionResult(
                predicted_score=predicted_score,
                confidence_interval_lower=confidence_lower,
                confidence_interval_upper=confidence_upper,
                message="Prediction successful"
            )
            
            logger.info(f"Prediction: {predicted_score} runs (range: {confidence_lower}-{confidence_upper})")
            return result
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Failed to make prediction: {e}")