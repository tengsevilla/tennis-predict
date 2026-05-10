# Tennis Match Predictor

This repository contains tools to fetch, clean, and use historical ATP tennis data to train a machine learning model capable of predicting match outcomes.

## Scripts

- `data_fetcher.py`: Connects to `JeffSackmann/tennis_atp` to fetch ATP data from 2016 to the present day, clean out invalid records, and save to `cleaned_atp_data.csv`.
- `train_model.py`: Reads the cleaned data, engineers time-based features, processes the data, and trains a RandomForest model using cross-validation over time. Outputs accuracy and generates a feature importance plot.
- `api.py`: A FastAPI application that serves endpoints for making daily predictions, checking results, and triggering model retraining.

## Setup

Install dependencies:
```bash
pip install pandas scikit-learn matplotlib xgboost
```

## Usage
1. Fetch the data:
```bash
python3 data_fetcher.py
```
2. Train the model and see feature importance:
```bash
python3 train_model.py
```
3. Run the API server:
```bash
uvicorn api:app --reload
```

## API Endpoints

Once the FastAPI app is running (default: `http://localhost:8000`), you can interact with the following endpoints:

### POST `/run-daily-predictions`
Fetches the latest ATP matches and odds via the Odds API, runs the machine learning model to predict the winner, and saves the predictions to the database.
```bash
curl -X POST http://localhost:8000/run-daily-predictions
```

### POST `/update-results`
Fetches completed match results from the past 7 days (via ESPN API) and updates pending predictions in the database with the actual winner and final score.
```bash
curl -X POST http://localhost:8000/update-results
```

### GET `/predictions`
Retrieves stored predictions from the database, along with pagination metadata and a dynamically calculated prediction success rate.

**Optional Query Parameters:** `match_date`, `tournament`, `match_completed`, `limit` (default: 100), `offset` (default: 0)
```bash
curl "http://localhost:8000/predictions?match_completed=true&limit=50&offset=0"
```

### POST `/retrain`
Triggers a background task to fetch fresh historical data and completely retrain the machine learning model.
```bash
curl -X POST http://localhost:8000/retrain
```

### GET `/verify-volume`
Checks if the directory containing the model artifacts (`models/`) exists and lists its contents. Useful for verifying Docker volume mounts.
```bash
curl http://localhost:8000/verify-volume
```
```
