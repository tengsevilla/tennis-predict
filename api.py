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
from fastapi.responses import FileResponse, StreamingResponse
import zipfile
import io
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, text
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
    match_date = Column(String(255), index=True)
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


class EspnMatch(Base):
    __tablename__ = "espn_matches"

    id = Column(Integer, primary_key=True, index=True)
    espn_id = Column(String(100), unique=True, index=True)
    match_date = Column(String(255), index=True)
    tournament_id = Column(String(255))
    player_1 = Column(String(255))
    player_2 = Column(String(255))
    winner = Column(String(255), nullable=True)
    score = Column(String(255), nullable=True)
    p1_sets_won = Column(Integer, nullable=True)
    p2_sets_won = Column(Integer, nullable=True)


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
        'player_a_elo': a_stats.get('elo', 1500),
        'player_a_recent_win_pct': a_stats['recent_win_pct'],
        'player_a_serve_win_pct': a_stats['serve_win_pct'],
        'player_a_return_win_pct': a_stats['return_win_pct'],
        'player_a_sets_dropped_avg': a_stats['sets_dropped_avg'],
        'player_a_surface_win_pct': a_surf_pct,
        'point_difference': a_stats['points'] - b_stats['points'],
        'elo_difference': a_stats.get('elo', 1500) - b_stats.get('elo', 1500),
        'h2h_win_pct': a_h2h,

        'player_b_rank': b_stats['rank'],
        'player_b_age': b_stats['age'],
        'player_b_hand': b_stats['hand'],
        'player_b_elo': b_stats.get('elo', 1500),
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
        'player_a_elo', 'player_b_elo', 'elo_difference',
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

def _names_match(db_name: str, espn_name: str) -> bool:
    """Return True if db_name and espn_name refer to the same player.

    Handles cases where ESPN uses a longer official name (e.g. ESPN returns
    "Carlos Alcaraz Garfia" while the Odds API returns "Carlos Alcaraz").
    """
    a = db_name.lower().strip()
    b = espn_name.lower().strip()
    if a == b:
        return True
    # All words in the shorter name must appear in the longer name
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return all(word in longer for word in shorter.split())


def _dates_close(db_date_str: str, espn_date_str: str, tolerance_days: int = 4) -> bool:
    """Return True if the two ISO date strings are within tolerance_days of each other.

    Prevents the same player pair from a later tournament overwriting an older
    pending prediction that was never resolved.
    """
    try:
        db_dt = pd.to_datetime(db_date_str, utc=True)
        espn_dt = pd.to_datetime(espn_date_str, utc=True)
        return abs((db_dt - espn_dt).total_seconds()) <= tolerance_days * 86400
    except Exception:
        return True  # If either date is unparseable, allow the match through


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
                
                raw_commence_time = match.get('commence_time')

                # Parse as UTC-aware safely to avoid TypeError
                commence_time_dt = pd.to_datetime(raw_commence_time, utc=True)
                
                # Prevent predicting on matches that have already started
                if commence_time_dt < pd.Timestamp.utcnow():
                    continue
                    
                # Store the exact string from the API (includes timezone)
                commence_time = raw_commence_time

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

                # Check if this match already exists in the database
                existing_prediction = db.query(Prediction).filter(
                    Prediction.player_a == home_team,
                    Prediction.player_b == away_team,
                    Prediction.match_date == commence_time
                ).first()

                if existing_prediction:
                    # Update existing record with latest odds/probs
                    existing_prediction.player_a_prob = p_home_win
                    existing_prediction.player_b_prob = p_away_win
                    existing_prediction.player_a_odds = home_odds
                    existing_prediction.player_b_odds = away_odds
                    existing_prediction.value_bet_on_player = value_bet_player
                else:
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
    # Fetch ESPN scores outside the DB transaction so HTTPException propagates cleanly.
    # The date-range parameter causes ESPN to return 0 events for tennis — omit it to get
    # the current active tournament including all completed matches.
    scores_url = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?limit=300"

    resp = requests.get(scores_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch scores from ESPN.")

    scores_data = resp.json()

    # Parse completed matches from ESPN response.
    # ESPN ATP scoreboard uses events[].competitions[] directly (no groupings level).
    # We also check event.groupings[] as a defensive fallback in case the structure varies.
    completed_matches = []
    for event in scores_data.get('events', []):
        competitions = event.get('competitions', [])
        if not competitions:
            for grouping in event.get('groupings', []):
                competitions.extend(grouping.get('competitions', []))

        for competition in competitions:
            status = competition.get('status', {}).get('type', {})
            if not status.get('completed'):
                continue

            competitors = competition.get('competitors', [])
            if len(competitors) != 2:
                continue

            c1, c2 = competitors[0], competitors[1]
            p1_name = c1.get('athlete', {}).get('displayName')
            p2_name = c2.get('athlete', {}).get('displayName')
            if not p1_name or not p2_name:
                continue

            winner_name = p1_name if c1.get('winner') else (p2_name if c2.get('winner') else None)

            c1_lines = c1.get('linescores', [])
            c2_lines = c2.get('linescores', [])
            score_str = ""
            if c1_lines and c2_lines and len(c1_lines) == len(c2_lines):
                sets = [
                    f"{int(c1_lines[i].get('value', 0))}-{int(c2_lines[i].get('value', 0))}"
                    for i in range(len(c1_lines))
                ]
                score_str = f"{p1_name} {', '.join(sets)} {p2_name}"

            c1_sets_won = sum(
                1 for i in range(len(c1_lines))
                if c1_lines[i].get('value', 0) > c2_lines[i].get('value', 0)
            ) if c1_lines and c2_lines and len(c1_lines) == len(c2_lines) else None
            c2_sets_won = sum(
                1 for i in range(len(c2_lines))
                if c2_lines[i].get('value', 0) > c1_lines[i].get('value', 0)
            ) if c1_lines and c2_lines and len(c1_lines) == len(c2_lines) else None

            completed_matches.append({
                'espn_id': str(competition.get('id', '')),
                'tournament_id': str(competition.get('tournamentId', '')),
                'p1': p1_name,
                'p2': p2_name,
                'winner': winner_name,
                'score': score_str,
                'date': competition.get('date', ''),
                'p1_sets_won': c1_sets_won,
                'p2_sets_won': c2_sets_won,
            })

    db = SessionLocal()
    updated_count = 0
    espn_inserted = 0
    try:
        # Upsert ESPN matches into espn_matches table (dedup by espn_id)
        existing_ids = {
            row[0] for row in db.query(EspnMatch.espn_id).all()
        }
        for m in completed_matches:
            if m['espn_id'] and m['espn_id'] not in existing_ids:
                db.add(EspnMatch(
                    espn_id=m['espn_id'],
                    match_date=m['date'],
                    tournament_id=m['tournament_id'],
                    player_1=m['p1'],
                    player_2=m['p2'],
                    winner=m['winner'],
                    score=m['score'],
                    p1_sets_won=m['p1_sets_won'],
                    p2_sets_won=m['p2_sets_won'],
                ))
                existing_ids.add(m['espn_id'])
                espn_inserted += 1

        # Update pending predictions
        pending_preds = db.query(Prediction).filter(Prediction.match_completed == False).all()
        for pred in pending_preds:
            for m in completed_matches:
                pair_match = (
                    (
                        (_names_match(pred.player_a, m['p1']) and _names_match(pred.player_b, m['p2'])) or
                        (_names_match(pred.player_a, m['p2']) and _names_match(pred.player_b, m['p1']))
                    ) and _dates_close(pred.match_date, m['date'])
                )
                if pair_match:
                    pred.match_completed = True
                    pred.winner = m['winner']
                    pred.score = m['score']
                    updated_count += 1
                    break

        db.commit()
        return {
            "status": "Success",
            "updated_matches": updated_count,
            "espn_matches_stored": espn_inserted,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks, years_back: int = 5):

    def retrain_task():
        try:
            import train_model
            meta = train_model.preprocess_and_train(years_back=years_back)
            print(f"Retraining complete. Data freshness: {meta}")
        except Exception as e:
            print(f"Retraining task failed: {e}")

    background_tasks.add_task(retrain_task)
    return {"status": "Retraining task initiated in the background.", "years_back": years_back}


@app.get("/data-status")
def data_status():
    """Report training data freshness and model artifact state."""
    result = {}

    # Player stats freshness
    stats_path = os.path.join(MODEL_DIR, "latest_player_stats.joblib")
    if os.path.exists(stats_path):
        import joblib as _jl
        stats = _jl.load(stats_path)
        result["player_count"] = len(stats)
        result["stats_file_age_days"] = round(
            (datetime.now() - datetime.fromtimestamp(os.path.getmtime(stats_path))).total_seconds() / 86400, 1
        )
    else:
        result["player_stats"] = "not found"

    # Training data freshness from cleaned CSV
    csv_path = os.path.join(MODEL_DIR, "cleaned_atp_data.csv")
    if os.path.exists(csv_path):
        df_head = pd.read_csv(csv_path, usecols=['tourney_date'])
        dates = pd.to_datetime(df_head['tourney_date'], format='%Y%m%d', errors='coerce')
        most_recent = dates.max()
        result["most_recent_training_match"] = most_recent.strftime('%Y-%m-%d') if pd.notna(most_recent) else "unknown"
        result["training_data_age_days"] = int((datetime.now() - most_recent).days) if pd.notna(most_recent) else None
        result["training_rows"] = len(df_head)
    else:
        result["training_data"] = "not found — run /retrain"

    # Model artifact sizes
    artifacts = ["tennis_model.joblib", "scaler.joblib", "label_encoders.joblib",
                 "latest_player_stats.joblib", "latest_h2h.joblib"]
    result["artifacts"] = {}
    for name in artifacts:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            result["artifacts"][name] = f"{round(os.path.getsize(path) / (1024*1024), 2)} MB"
        else:
            result["artifacts"][name] = "missing"

    return result

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
        query = db.query(Prediction).filter(Prediction.value_bet_on_player.isnot(None))

        if match_date:
            query = query.filter(Prediction.match_date == match_date)
        if tournament:
            query = query.filter(Prediction.tournament == tournament)
        if match_completed is not None:
            query = query.filter(Prediction.match_completed == match_completed)

        total_matches = query.count()

        # Compute hit-rate across ALL matching completed records (not just the current page)
        all_completed = query.filter(
            Prediction.match_completed == True,
            Prediction.winner.isnot(None)
        ).all()

        total_value_bets = 0
        successful_value_bets = 0
        for p in all_completed:
            total_value_bets += 1
            if p.value_bet_on_player and p.value_bet_on_player.lower() == p.winner.lower():
                successful_value_bets += 1

        prediction_rate = 0.0
        if total_value_bets > 0:
            prediction_rate = round((successful_value_bets / total_value_bets) * 100, 2)

        predictions = query.offset(offset).limit(limit).all()

        results = []
        for p in predictions:
            results.append({
                "id": p.id,
                "match_date": p.match_date,
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


@app.get("/download-file")
def download_file(filename: str):
    """Download a single file from the model volume.

    Example: GET /download-file?filename=tennis_model.joblib
    """
    safe_dir = os.path.realpath(MODEL_DIR)
    target = os.path.realpath(os.path.join(MODEL_DIR, filename))
    if not target.startswith(safe_dir + os.sep) and target != safe_dir:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in volume.")
    return FileResponse(path=target, filename=filename, media_type="application/octet-stream")


@app.get("/download-volume")
def download_volume():
    """Download all files in the model volume as a single zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(MODEL_DIR):
            fpath = os.path.join(MODEL_DIR, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, arcname=fname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=volume_files.zip"},
    )