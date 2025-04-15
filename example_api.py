"""
Example FastAPI Implementation for IPL Score Prediction

This script demonstrates how to create a FastAPI application using the IPL score prediction API module.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Add the project root to the Python path so we can import from ipl_score_prediction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FastAPI
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

# Import the API module
from src.api.api import IPLScorePredictor, PredictionInput, PredictionResult, PredictionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="IPL Score Prediction API",
    description="API for predicting IPL cricket match scores",
    version="1.0.0"
)

# Model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.pkl")

# Create a dependency for the predictor
def get_predictor():
    """
    FastAPI dependency to get the IPL score predictor.
    """
    try:
        # In a real app, you might want to cache this instance
        return IPLScorePredictor(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")


@app.get("/")
def read_root():
    """
    Root endpoint that returns basic API information.
    """
    return {
        "name": "IPL Score Prediction API",
        "version": "1.0.0",
        "description": "Predict IPL cricket match scores"
    }


@app.post("/predict", response_model=PredictionResult)
def predict_score(
    input_data: PredictionInput,
    predictor: IPLScorePredictor = Depends(get_predictor)
):
    """
    Predict the final score based on the current match state.
    
    Args:
        input_data: Match information and current state
        predictor: IPL score predictor instance (injected by FastAPI)
        
    Returns:
        Prediction result with the predicted score
    """
    try:
        # Make prediction
        result = predictor.predict(input_data)
        return result
    
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Handle other errors
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/teams")
def get_teams():
    """
    Get a list of IPL teams.
    This is a simple example endpoint - in a real app, this would come from a database.
    """
    teams = [
        "Chennai Super Kings",
        "Delhi Capitals",
        "Kings XI Punjab",
        "Kolkata Knight Riders",
        "Mumbai Indians",
        "Rajasthan Royals",
        "Royal Challengers Bangalore",
        "Sunrisers Hyderabad"
    ]
    return {"teams": teams}


@app.get("/venues")
def get_venues():
    """
    Get a list of IPL venues.
    This is a simple example endpoint - in a real app, this would come from a database.
    """
    venues = [
        "M Chinnaswamy Stadium",
        "Eden Gardens",
        "Feroz Shah Kotla",
        "MA Chidambaram Stadium",
        "Wankhede Stadium",
        "Sawai Mansingh Stadium",
        "Punjab Cricket Association Stadium",
        "Rajiv Gandhi International Stadium"
    ]
    return {"venues": venues}


if __name__ == "__main__":
    # This section is for running the app directly with Python
    # For production, you should use a proper ASGI server like uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


