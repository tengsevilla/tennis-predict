import os
import subprocess
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- CONFIGURATION ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db") # Fallback for local testing
MODEL_DIR = "/app/models/"

# Ensure volume dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- DATABASE SETUP ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_date = Column(Date, index=True)
    tournament = Column(String)
    player_a = Column(String)
    player_b = Column(String)
    player_a_prob = Column(Float)
    player_b_prob = Column(Float)
    player_a_odds = Column(Float)
    player_b_odds = Column(Float)
    is_value_bet = Column(Boolean)

Base.metadata.create_all(bind=engine)

# --- FASTAPI APP ---
app = FastAPI(title="Tennis Oracle API")

# --- MODEL LOADING ---
def load_artifacts():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "tennis_model.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
        stats = joblib.load(os.path.join(MODEL_DIR, "latest_player_stats.joblib"))
        h2h = joblib.load(os.path.join(MODEL_DIR, "latest_h2h.joblib"))
        return model, scaler, le, stats, h2h
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None, None, None, None, None

def get_prediction_prob(player_a, player_b, model, scaler, le, stats, latest_h2h):
    if player_a not in stats or player_b not in stats:
        return None

    a_stats = stats[player_a]
    b_stats = stats[player_b]

    a_id = a_stats['id']
    b_id = b_stats['id']
    p1 = min(a_id, b_id)
    p2 = max(a_id, b_id)

    h2h = latest_h2h.get((p1, p2), 0.5)
    a_h2h = h2h if a_id == p1 else (1 - h2h)

    # We don't necessarily know surface or tourney level for active odds API fetch natively without extra mapping
    # Default to Hard court ('Hard') and generic level ('A') for generic daily predictions if unknown
    surface = 'Hard'
    tourney_level = 'A'

    a_surf_pct = a_stats['surface_pcts'].get(surface, 0.5)
    b_surf_pct = b_stats['surface_pcts'].get(surface, 0.5)

    row_dict = {
        'player_a_rank': a_stats['rank'],
        'player_a_age': a_stats['age'],
        'player_a_hand': a_stats['hand'],
        'player_a_recent_win_pct': a_stats['recent_win_pct'],
        'player_a_serve_win_pct': a_stats['serve_win_pct'],
        'player_a_return_win_pct': a_stats['return_win_pct'],
        'player_a_sets_dropped_avg': a_stats['sets_dropped_avg'],
        'player_a_surface_win_pct': a_surf_pct,
        'point_difference': a_stats['points'] - b_stats['points'],
        'h2h_win_pct': a_h2h,

        'player_b_rank': b_stats['rank'],
        'player_b_age': b_stats['age'],
        'player_b_hand': b_stats['hand'],
        'player_b_recent_win_pct': b_stats['recent_win_pct'],
        'player_b_serve_win_pct': b_stats['serve_win_pct'],
        'player_b_return_win_pct': b_stats['return_win_pct'],
        'player_b_sets_dropped_avg': b_stats['sets_dropped_avg'],
        'player_b_surface_win_pct': b_surf_pct,

        'surface': surface,
        'tourney_level': tourney_level
    }

    df_pred = pd.DataFrame([row_dict])

    categorical_cols = ['surface', 'tourney_level', 'player_a_hand', 'player_b_hand']
    numerical_cols = [
        'player_a_rank', 'player_a_age',
        'player_a_recent_win_pct', 'player_a_serve_win_pct', 'player_a_return_win_pct', 'player_a_sets_dropped_avg',
        'player_a_surface_win_pct', 'point_difference', 'h2h_win_pct',
        'player_b_rank', 'player_b_age',
        'player_b_recent_win_pct', 'player_b_serve_win_pct', 'player_b_return_win_pct', 'player_b_sets_dropped_avg',
        'player_b_surface_win_pct'
    ]
    features = numerical_cols + categorical_cols

    # Preprocess
    for col in categorical_cols:
        encoder = le[col]
        val = df_pred[col].iloc[0]
        if val in encoder.classes_:
            df_pred[col] = encoder.transform([val])
        else:
            df_pred[col] = encoder.transform([encoder.classes_[0]])

    df_pred[numerical_cols] = scaler.transform(df_pred[numerical_cols])
    df_pred = df_pred[features]

    probs = model.predict_proba(df_pred)[0]
    return probs[1], probs[0] # P(Player A), P(Player B)

# --- ENDPOINTS ---

@app.post("/run-daily-predictions")
def run_daily_predictions():
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY is not set.")

    model, scaler, le, stats, latest_h2h = load_artifacts()
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found. Call /retrain first.")

    # Phase A: Dynamic Discovery
    sports_url = f"https://api.the-odds-api.com/v4/sports/?apiKey={ODDS_API_KEY}"
    sports_resp = requests.get(sports_url)
    if sports_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch active sports from Odds API.")

    sports = sports_resp.json()
    active_keys = [s['key'] for s in sports if s['key'].startswith('tennis_atp_')]

    results = []
    db = SessionLocal()

    # Phase B: Fetch & Predict
    try:
        for key in active_keys:
            odds_url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/?apiKey={ODDS_API_KEY}&regions=us,us2,eu,uk,au&markets=h2h"
            odds_resp = requests.get(odds_url)
            if odds_resp.status_code != 200:
                continue

            matches = odds_resp.json()
            for match in matches:
                home_team = match.get('home_team')
                away_team = match.get('away_team')
                commence_time = pd.to_datetime(match.get('commence_time')).date()

                # Get best odds across bookmakers
                home_odds = 0
                away_odds = 0

                for bookmaker in match.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == home_team and outcome['price'] > home_odds:
                                    home_odds = outcome['price']
                                elif outcome['name'] == away_team and outcome['price'] > away_odds:
                                    away_odds = outcome['price']

                if home_odds == 0 or away_odds == 0:
                    continue

                # Predict
                probs = get_prediction_prob(home_team, away_team, model, scaler, le, stats, latest_h2h)
                if not probs:
                    continue # Players not in our dataset

                p_home_win, p_away_win = probs

                # Value Logic: Flag if Model Prob > (1 / Decimal Odds) + 0.05
                is_value = False
                if p_home_win > (1.0 / home_odds) + 0.05:
                    is_value = True
                elif p_away_win > (1.0 / away_odds) + 0.05:
                    is_value = True

                prediction_record = Prediction(
                    match_date=commence_time,
                    tournament=key,
                    player_a=home_team,
                    player_b=away_team,
                    player_a_prob=p_home_win,
                    player_b_prob=p_away_win,
                    player_a_odds=home_odds,
                    player_b_odds=away_odds,
                    is_value_bet=is_value
                )
                db.add(prediction_record)

                results.append({
                    "match_date": str(commence_time),
                    "tournament": key,
                    "player_a": home_team,
                    "player_b": away_team,
                    "player_a_prob": p_home_win,
                    "player_b_prob": p_away_win,
                    "player_a_odds": home_odds,
                    "player_b_odds": away_odds,
                    "is_value_bet": is_value
                })

        # Phase C: Persistence
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    return results

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):

    def retrain_task():
        # Trigger full retraining
        try:
            import train_model
            # Re-fetch data if needed, or just retrain. The prompt says "using latest historical CSV data"
            # so we'll just run preprocess_and_train which loads cleaned_atp_data.csv and overwrites joblibs
            train_model.preprocess_and_train()

            # Git Sync
            subprocess.run(["git", "add", "cleaned_atp_data.csv", "feature_importances.png", MODEL_DIR], check=True)
            subprocess.run(["git", "commit", "-m", "Auto-update model brain [skip ci]"], check=True)
            subprocess.run(["git", "push"], check=True)

            print("Retraining and sync complete.")
        except Exception as e:
            print(f"Retraining task failed: {e}")

    background_tasks.add_task(retrain_task)
    return {"status": "Retraining task initiated in the background."}
