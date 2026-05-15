# Tennis Match Predictor

This repository contains tools to fetch, clean, and use historical ATP tennis data to train a machine learning model capable of predicting match outcomes.

## Scripts

- `data_fetcher.py`: Connects to `JeffSackmann/tennis_atp` to fetch ATP data, caches historical years locally, and always re-fetches the current year. Accepts a `years_back` parameter (default 5).
- `train_model.py`: Reads the cleaned data, engineers time-based features, trains an XGBoost model using TimeSeriesSplit cross-validation, and saves all model artifacts to the volume.
- `api.py`: FastAPI application serving endpoints for predictions, result updates, retraining, and data status.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Fetch data and train:
```bash
python3 data_fetcher.py
python3 train_model.py
```
2. Run the API server:
```bash
uvicorn api:app --reload
```

## API Endpoints

Base URL (local): `http://localhost:8000`

---

### POST `/run-daily-predictions`
Fetches active ATP matches and odds from the Odds API, runs the ML model to predict winners, and saves predictions to the database.
```bash
curl -X POST http://localhost:8000/run-daily-predictions
```

---

### POST `/update-results`
Fetches completed match results from the ESPN ATP scoreboard and:
- Updates pending predictions with the actual winner and score
- Inserts new ESPN match records into the `espn_matches` table (deduplicated by ESPN competition ID)

```bash
curl -X POST http://localhost:8000/update-results
```

**Response:**
```json
{ "status": "Success", "updated_matches": 3, "espn_matches_stored": 12 }
```

---

### GET `/predictions`
Retrieves stored predictions with pagination and a prediction success rate calculated across **all** completed matches (not just the current page).

**Query Parameters:** `match_date`, `tournament`, `match_completed`, `limit` (default: 100), `offset` (default: 0)
```bash
curl "http://localhost:8000/predictions?match_completed=true&limit=50"
```

---

### POST `/retrain`
Triggers a background task to fetch fresh data and fully retrain the model.

**Query Parameters:** `years_back` (default: 5) — limits training to the most recent N years.
```bash
curl -X POST "http://localhost:8000/retrain?years_back=5"
```

Data fetching is optimised: historical years are loaded from local cache, only the current year is re-fetched from GitHub.

---

### GET `/retrain-status`
Returns the status of the most recent retrain task. Useful since `/retrain` is fire-and-forget.
```bash
curl http://localhost:8000/retrain-status
```

**Response (running):**
```json
{ "status": "running", "started_at": "2026-05-15T10:00:00", "years_back": 5 }
```

**Response (completed):**
```json
{ "status": "success", "completed_at": "2026-05-15T10:04:32", "years_back": 5, "meta": "..." }
```

**Response (failed):**
```json
{ "status": "failed", "failed_at": "2026-05-15T10:01:11", "error": "..." }
```

---

### GET `/data-status`
Reports training data freshness, player count, and model artifact sizes. Useful for checking when a retrain is needed.
```bash
curl http://localhost:8000/data-status
```

**Response:**
```json
{
  "player_count": 850,
  "stats_file_age_days": 3.2,
  "most_recent_training_match": "2026-05-04",
  "training_data_age_days": 11,
  "training_rows": 42000,
  "artifacts": {
    "tennis_model.joblib": "0.57 MB",
    "scaler.joblib": "0.01 MB",
    "label_encoders.joblib": "0.01 MB",
    "latest_player_stats.joblib": "0.45 MB",
    "latest_h2h.joblib": "1.2 MB"
  }
}
```

---

### GET `/verify-volume`
Lists files and sizes in the model volume directory. Useful for verifying Railway volume mounts.
```bash
curl http://localhost:8000/verify-volume
```

---

### GET `/download-file`
Downloads a single file from the model volume.

**Query Parameters:** `filename` — name of the file to download.
```bash
curl "http://localhost:8000/download-file?filename=tennis_model.joblib" -o tennis_model.joblib
```

---

### GET `/download-volume`
Downloads all model volume files as a single zip archive (`volume_files.zip`).
```bash
curl http://localhost:8000/download-volume -o volume_files.zip
```

---

## Database Tables

| Table | Description |
|---|---|
| `predictions` | ML predictions with odds, probabilities, and results |
| `espn_matches` | Completed ATP match results ingested from ESPN, used as training data |
