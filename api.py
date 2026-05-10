import os

# Restrict linear algebra backend thread counts to prevent memory allocation 
# crashes when Uvicorn spins up reloader processes.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
import pandas as pd
import requests
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pymysql
pymysql.install_as_MySQLdb()

# --- CONFIGURATION ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "8889db8b63391328c4478c3ee26cba48")
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:root@localhost:3306/tennis_db") # Fallback for local testing
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/tennis_db"
elif DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
MODEL_DIR = os.getenv("MODEL_DIR", "models/")

# Ensure volume dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- DATABASE SETUP ---
url = make_url(DATABASE_URL)
# Only attempt creation if it's a MySQL URL and specifies a database
if url.database and url.drivername.startswith("mysql"):
    db_name = url.database
    server_url = url.set(database='')
    server_engine = create_engine(server_url, isolation_level="AUTOCOMMIT")
    with server_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_date = Column(Date, index=True)
    tournament = Column(String(255))
    player_a = Column(String(255))
    player_b = Column(String(255))
    player_a_prob = Column(Float)
    player_b_prob = Column(Float)
    player_a_odds = Column(Float)
    player_b_odds = Column(Float)
    value_bet_on_player = Column(String(255), nullable=True)
    match_completed = Column(Boolean, default=False)
    winner = Column(String(255), nullable=True)
    score = Column(String(255), nullable=True)

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
    return float(probs[1]), float(probs[0]) # P(Player A), P(Player B)

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

                # Prediction Logic: Pick the player with the highest model probability regardless of odds
                value_bet_player = None
                if p_home_win > p_away_win:
                    value_bet_player = home_team
                elif p_away_win > p_home_win:
                    value_bet_player = away_team

                prediction_record = Prediction(
                    match_date=commence_time,
                    tournament=key,
                    player_a=home_team,
                    player_b=away_team,
                    player_a_prob=p_home_win,
                    player_b_prob=p_away_win,
                    player_a_odds=home_odds,
                    player_b_odds=away_odds,
                    value_bet_on_player=value_bet_player
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
                "value_bet_on_player": value_bet_player,
                "match_completed": False,
                "winner": None,
                "score": None
                })

        # Phase C: Persistence
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    return results

@app.post("/update-results")
def update_results():
    db = SessionLocal()
    updated_count = 0
    try:
        # Query pending predictions
        pending_preds = db.query(Prediction).filter(Prediction.match_completed == False).all()
        if not pending_preds:
            return {"status": "No pending matches to update.", "updated": 0}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        date_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        scores_url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={date_str}&limit=300"
        
        resp = requests.get(scores_url)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch scores from ESPN.")

        scores_data = resp.json()
        
        scores_map = {}
        for event in scores_data.get('events', []):
            for grouping in event.get('groupings', []):
                for competition in grouping.get('competitions', []):
                    status = competition.get('status', {}).get('type', {})
                    if status.get('completed'):
                        competitors = competition.get('competitors', [])
                        if len(competitors) == 2:
                            c1, c2 = competitors[0], competitors[1]
                            
                            p1_name = c1.get('athlete', {}).get('displayName')
                            p2_name = c2.get('athlete', {}).get('displayName')
                            
                            p1_winner = c1.get('winner', False)
                            p2_winner = c2.get('winner', False)
                            
                            winner_name = p1_name if p1_winner else (p2_name if p2_winner else None)
                            
                            c1_linescores = c1.get('linescores', [])
                            c2_linescores = c2.get('linescores', [])
                            score_str = ""
                            
                            if c1_linescores and c2_linescores and len(c1_linescores) == len(c2_linescores):
                                sets = []
                                for i in range(len(c1_linescores)):
                                    sets.append(f"{int(c1_linescores[i].get('value', 0))}-{int(c2_linescores[i].get('value', 0))}")
                                score_str = f"{p1_name} {', '.join(sets)} {p2_name}"
                            
                            match_info = {'winner': winner_name, 'score': score_str}
                            if p1_name and p2_name:
                                # Use lowercase to make matching more robust
                                scores_map[(p1_name.lower(), p2_name.lower())] = match_info
                                scores_map[(p2_name.lower(), p1_name.lower())] = match_info

        for pred in pending_preds:
            p_a, p_b = pred.player_a.lower(), pred.player_b.lower()
            
            match_info = scores_map.get((p_a, p_b))
            if match_info:
                pred.match_completed = True
                pred.winner = match_info['winner']
                pred.score = match_info['score']
                updated_count += 1
                
        db.commit()
        return {"status": "Success", "updated_matches": updated_count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):

    def retrain_task():
        # Trigger full retraining
        try:
            import data_fetcher
            data_fetcher.fetch_and_process_data()

            import train_model
            train_model.preprocess_and_train()

            print("Retraining complete. Models saved to persistent volume.")
        except Exception as e:
            print(f"Retraining task failed: {e}")

    background_tasks.add_task(retrain_task)
    return {"status": "Retraining task initiated in the background."}

@app.get("/verify-volume")
def verify_volume():
    import os
    try:
        files = os.listdir(MODEL_DIR)
        file_details = []
        for file in files:
            size_mb = os.path.getsize(os.path.join(MODEL_DIR, file)) / (1024 * 1024)
            file_details.append({"filename": file, "size_mb": round(size_mb, 2)})
            
        return {
            "volume_path": MODEL_DIR,
            "status": "Volume is attached and readable" if files else "Volume is empty",
            "files": file_details
        }
    except Exception as e:
        return {"error": str(e), "message": "Volume path not found."}

@app.get("/predictions")
def get_predictions(
    match_date: Optional[str] = None,
    tournament: Optional[str] = None,
    match_completed: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
):
    db = SessionLocal()
    try:
        query = db.query(Prediction)
        
        # Apply optional filters
        if match_date:
            query = query.filter(Prediction.match_date == match_date)
        if tournament:
            query = query.filter(Prediction.tournament == tournament)
        if match_completed is not None:
            query = query.filter(Prediction.match_completed == match_completed)
            
        total_matches = query.count()
        
        predictions = query.offset(offset).limit(limit).all()
        
        total_value_bets = 0
        successful_value_bets = 0
        
        results = []
        for p in predictions:
            results.append({
                "id": p.id,
                "match_date": p.match_date.isoformat() if p.match_date else None,
                "tournament": p.tournament,
                "player_a": p.player_a,
                "player_b": p.player_b,
                "player_a_prob": p.player_a_prob,
                "player_b_prob": p.player_b_prob,
                "player_a_odds": p.player_a_odds,
                "player_b_odds": p.player_b_odds,
                "value_bet_on_player": p.value_bet_on_player,
                "match_completed": p.match_completed,
                "winner": p.winner,
                "score": p.score
            })
            
            # Calculate hit rate only on matches that are completed, had a value bet, and have a confirmed winner
            if p.match_completed and p.value_bet_on_player and p.winner:
                total_value_bets += 1
                if p.value_bet_on_player.lower() == p.winner.lower():
                    successful_value_bets += 1
                    
        prediction_rate = 0.0
        if total_value_bets > 0:
            prediction_rate = round((successful_value_bets / total_value_bets) * 100, 2)
            
        return {
            "pagination": {
                "total_matches": total_matches,
                "limit": limit,
                "offset": offset
            },
            "data": results,
            "stats": {
                "total_value_bets_evaluated": total_value_bets,
                "successful_value_bets": successful_value_bets,
                "prediction_rate_percent": prediction_rate
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()