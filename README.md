# IPL Score Prediction

![IPL Score Prediction](https://img.shields.io/badge/IPL-Score%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

A machine learning project to predict IPL cricket match scores using historical match data. The application analyzes ball-by-ball data and match information to provide accurate predictions of final innings scores.

## Features

- **Data Analysis**: Comprehensive data processing and exploratory data analysis of IPL matches
- **Score Prediction**: Machine learning models to predict final innings scores
- **Feature Engineering**: Advanced feature creation from raw cricket data
- **Model Comparison**: Implementation of multiple regression algorithms with performance comparison
- **Visualization**: Interactive visualizations of team performances, venue statistics, and model accuracy
- **REST API**: FastAPI implementation for real-time score predictions
- **Modular Design**: Clean, modular codebase for easy extension and maintenance

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/ipl_score_prediction.git
cd ipl_score_prediction
```

### Step 2: Create a virtual environment (optional but recommended)

#### For Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare data

The project requires two CSV files:
- `deliveries.csv`: Ball-by-ball data for IPL matches
- `matches.csv`: Match metadata

Place these files in the `data` directory.

## Usage

### Training a Model

To train a new model using the dataset:

```bash
python main.py train --model-type rf
```

Available model types:
- `lr`: Linear Regression
- `rf`: Random Forest (default)
- `gb`: Gradient Boosting

### Evaluating a Model

To evaluate a trained model:

```bash
python main.py evaluate --model-path models/ipl_score_rf_model.pkl
```

### Making Predictions

To predict the score for a specific match scenario:

```bash
python main.py predict --batting-team "Mumbai Indians" --bowling-team "Chennai Super Kings" --venue "Wankhede Stadium" --current-score 85 --current-over 10.2 --wickets-fallen 3
```

### Generating Visualizations

To create visualizations from the data:

```bash
python main.py visualize
```

### Running the Complete Workflow

To run the entire pipeline (data processing, feature engineering, model training, evaluation, and visualization):

```bash
python main.py workflow
```

### Starting the API Server

To start the FastAPI server for real-time predictions:

```bash
python example_api.py
```

Then open your browser and navigate to http://localhost:8000/docs to access the Swagger UI documentation.

## Project Structure

```
ipl_score_prediction/
├── data/                   # Data directory
│   ├── deliveries.csv      # Ball-by-ball data
│   ├── matches.csv         # Match metadata
├── src/                    # Source code
│   ├── data_processing/    # Data loading and preprocessing
│   ├── feature_engineering/# Feature creation and transformation
│   ├── modeling/           # Machine learning models
│   ├── visualization/      # Data visualization
│   ├── api/                # API implementation
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── visualizations/         # Generated visualizations
├── main.py                 # Main script
├── example_api.py          # FastAPI implementation
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Dependencies

- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **API**: FastAPI, uvicorn, pydantic
- **Utilities**: joblib, logging

For a complete list of dependencies, see `requirements.txt`.

## API Documentation

The API provides endpoints for predicting IPL scores based on current match conditions.

### Endpoints

#### POST `/predict`

Predicts the final innings score based on current match state.

**Request Body:**

```json
{
  "match_id": 12345,
  "venue": "Wankhede Stadium",
  "batting_team": "Mumbai Indians",
  "bowling_team": "Chennai Super Kings",
  "current_score": 85,
  "current_over": 10.2,
  "wickets_fallen": 3,
  "season": "2022",
  "run_rate": 8.3,
  "last_five_overs_runs": 45,
  "last_five_overs_wickets": 1,
  "batting_team_win_rate": 0.65,
  "bowling_team_win_rate": 0.55,
  "head_to_head_win_rate": 0.6,
  "venue_avg_first_innings_score": 175.5
}
```

**Response:**

```json
{
  "predicted_score": 185,
  "confidence_interval_lower": 167,
  "confidence_interval_upper": 204,
  "message": "Prediction successful"
}
```

#### GET `/teams`

Returns a list of IPL teams.

#### GET `/venues`

Returns a list of IPL venues.

## Model Performance

Our models achieve the following performance metrics on the test dataset:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|---------|
| Linear Regression | 18.75 | 14.28 | 0.741 |
| Random Forest | 15.42 | 11.03 | 0.823 |
| Gradient Boosting | 14.87 | 10.65 | 0.839 |

## Future Improvements

- **Player-level Features**: Include individual player statistics in the model
- **Weather Data Integration**: Add weather conditions to improve prediction accuracy
- **Deployment**: Containerize the application using Docker for easier deployment
- **Real-time Updates**: Implement streaming data integration for live matches
- **Web Interface**: Develop a user-friendly web interface for non-technical users
- **Transfer Learning**: Experiment with transfer learning from other cricket leagues
- **Advanced Models**: Implement deep learning models like LSTM for time-series aspects

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IPL dataset from [Kaggle](https://www.kaggle.com)
- Inspiration from cricket analytics platforms

