import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import re
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

MODEL_DIR = os.getenv("MODEL_DIR", "models/")

# Global variables to hold model, scaler, encoders, and latest player stats for inference
model = None
scaler = None
label_encoders = {}
latest_player_stats = {}
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

def count_sets_won(score_str):
    if pd.isna(score_str):
        return 0, 0
    # Simply count the number of sets by splitting spaces, ignoring retirements
    sets = score_str.split(' ')
    w_sets, l_sets = 0, 0
    for s in sets:
        if '-' in s:
            try:
                parts = s.split('-')
                w_games_str = parts[0].split('(')[0]
                l_games_str = parts[1].split('(')[0]

                w_games = int(re.sub(r'\D', '', w_games_str)) if w_games_str else 0
                l_games = int(re.sub(r'\D', '', l_games_str)) if l_games_str else 0

                if w_games > l_games:
                    w_sets += 1
                elif l_games > w_games:
                    l_sets += 1
            except:
                pass
    return w_sets, l_sets

def calculate_elo(df, k_factor=32, base_rating=1500):
    """Calculates chronological Elo ratings for all matches in the DataFrame."""
    elo_dict = {}
    winner_elos = []
    loser_elos = []

    for idx, row in df.iterrows():
        w_id = row['winner_id']
        l_id = row['loser_id']

        w_elo = elo_dict.get(w_id, base_rating)
        l_elo = elo_dict.get(l_id, base_rating)

        winner_elos.append(w_elo)
        loser_elos.append(l_elo)

        # Calculate Expected Win Probability
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        expected_l = 1 / (1 + 10 ** ((w_elo - l_elo) / 400))

        # Update Elo Ratings
        elo_dict[w_id] = w_elo + k_factor * (1 - expected_w)
        elo_dict[l_id] = l_elo + k_factor * (0 - expected_l)

    df['winner_elo'] = winner_elos
    df['loser_elo'] = loser_elos
    return df, elo_dict

def load_and_engineer_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(MODEL_DIR, "cleaned_atp_data.csv")
    df = pd.read_csv(filepath)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # Fill NAs
    df['winner_rank_points'] = df['winner_rank_points'].fillna(0)
    df['loser_rank_points'] = df['loser_rank_points'].fillna(0)

    # Calculate chronological Elo before slicing dataframe
    df, final_elos = calculate_elo(df)

    # Also calculate sets won for rolling stats
    w_sets_dropped = []
    l_sets_dropped = []

    for idx, row in df.iterrows():
        w_sets, l_sets = count_sets_won(row['score'])
        w_sets_dropped.append(l_sets)
        l_sets_dropped.append(w_sets)

    df['winner_sets_dropped'] = w_sets_dropped
    df['loser_sets_dropped'] = l_sets_dropped

    # Melt dataframe to player level to calculate rolling stats safely
    winner_cols = ['tourney_date', 'winner_id', 'surface', 'winner_sets_dropped', 'w_1stWon', 'w_2ndWon', 'w_svpt', 'l_svpt', 'l_1stWon', 'l_2ndWon']
    loser_cols = ['tourney_date', 'loser_id', 'surface', 'loser_sets_dropped', 'l_1stWon', 'l_2ndWon', 'l_svpt', 'w_svpt', 'w_1stWon', 'w_2ndWon']

    df_w = df[winner_cols].copy()
    df_w.columns = ['date', 'player_id', 'surface', 'sets_dropped', 'serve_won', 'serve_2_won', 'serve_pt', 'opp_serve_pt', 'opp_serve_won', 'opp_serve_2_won']
    df_w['won_match'] = 1

    df_l = df[loser_cols].copy()
    df_l.columns = ['date', 'player_id', 'surface', 'sets_dropped', 'serve_won', 'serve_2_won', 'serve_pt', 'opp_serve_pt', 'opp_serve_won', 'opp_serve_2_won']
    df_l['won_match'] = 0

    # Keep original index for safe realignment later
    player_history = pd.concat([df_w, df_l])
    # Sort for rolling operations
    player_history = player_history.sort_values(['player_id', 'date'])

    # Compute points
    player_history['serve_pts_won'] = player_history['serve_won'] + player_history['serve_2_won']
    player_history['return_pts_won'] = player_history['opp_serve_pt'] - (player_history['opp_serve_won'] + player_history['opp_serve_2_won'])

    # Surface specific win pct (expanding window to include all historical matches)
    def expanding_surface_avg(group):
        shifted = group.shift(1)
        group['surface_win_pct'] = shifted['won_match'].expanding(min_periods=1).mean()
        group['final_surface_win_pct'] = group['won_match'].expanding(min_periods=1).mean()
        return group

    # Group by both player and surface
    player_history = player_history.groupby(['player_id', 'surface'], group_keys=False)[player_history.columns].apply(expanding_surface_avg)

    # Rolling averages strictly shifted to avoid leakage
    def rolling_avg(group):
        # Shift first so current match doesn't leak into its own calculation
        shifted = group.shift(1)
        # Rolling 10 match stats
        group['recent_win_pct'] = shifted['won_match'].rolling(10, min_periods=1).mean()

        serve_won_10 = shifted['serve_pts_won'].rolling(10, min_periods=1).sum()
        serve_pt_10 = shifted['serve_pt'].rolling(10, min_periods=1).sum()
        group['serve_win_pct'] = serve_won_10 / serve_pt_10

        ret_won_10 = shifted['return_pts_won'].rolling(10, min_periods=1).sum()
        ret_pt_10 = shifted['opp_serve_pt'].rolling(10, min_periods=1).sum()
        group['return_win_pct'] = ret_won_10 / ret_pt_10

        group['sets_dropped_avg'] = shifted['sets_dropped'].rolling(10, min_periods=1).mean()

        # We also need un-shifted rolling stats for the inference
        group['final_recent_win_pct'] = group['won_match'].rolling(10, min_periods=1).mean()
        group['final_serve_win_pct'] = group['serve_pts_won'].rolling(10, min_periods=1).sum() / group['serve_pt'].rolling(10, min_periods=1).sum()
        group['final_return_win_pct'] = group['return_pts_won'].rolling(10, min_periods=1).sum() / group['opp_serve_pt'].rolling(10, min_periods=1).sum()
        group['final_sets_dropped_avg'] = group['sets_dropped'].rolling(10, min_periods=1).mean()

        return group

    # Resort and group by just player for the overall 10 match rolling
    player_history = player_history.sort_values(['player_id', 'date'])
    player_history = player_history.groupby('player_id', group_keys=False)[player_history.columns].apply(rolling_avg)

    # Fill NAs for players with no history yet with 0 or a reasonable default
    player_history['recent_win_pct'] = player_history['recent_win_pct'].fillna(0.5)
    player_history['serve_win_pct'] = player_history['serve_win_pct'].fillna(0.6)
    player_history['return_win_pct'] = player_history['return_win_pct'].fillna(0.4)
    player_history['sets_dropped_avg'] = player_history['sets_dropped_avg'].fillna(1.0)
    player_history['surface_win_pct'] = player_history['surface_win_pct'].fillna(0.5)

    # H2H Calculation
    df_h2h = df[['tourney_date', 'winner_id', 'loser_id']].copy()
    # To treat A vs B same as B vs A for grouping, create sorted pairs
    df_h2h['p1'] = df_h2h[['winner_id', 'loser_id']].min(axis=1)
    df_h2h['p2'] = df_h2h[['winner_id', 'loser_id']].max(axis=1)
    # 1 if p1 won, 0 if p2 won
    df_h2h['p1_won'] = (df_h2h['winner_id'] == df_h2h['p1']).astype(int)

    df_h2h = df_h2h.sort_values('tourney_date')

    def expanding_h2h(group):
        shifted = group.shift(1)
        group['h2h_p1_win_pct'] = shifted['p1_won'].expanding(min_periods=1).mean()
        group['final_h2h_p1_win_pct'] = group['p1_won'].expanding(min_periods=1).mean()
        return group

    df_h2h = df_h2h.groupby(['p1', 'p2'], group_keys=False)[df_h2h.columns].apply(expanding_h2h)
    df_h2h['h2h_p1_win_pct'] = df_h2h['h2h_p1_win_pct'].fillna(0.5)

    # Realign using the original index!
    df_w_features = player_history[player_history['won_match'] == 1].sort_index()
    df_l_features = player_history[player_history['won_match'] == 0].sort_index()

    df['winner_recent_win_pct'] = df_w_features['recent_win_pct']
    df['winner_serve_win_pct'] = df_w_features['serve_win_pct']
    df['winner_return_win_pct'] = df_w_features['return_win_pct']
    df['winner_sets_dropped_avg'] = df_w_features['sets_dropped_avg']
    df['winner_surface_win_pct'] = df_w_features['surface_win_pct']

    df['loser_recent_win_pct'] = df_l_features['recent_win_pct']
    df['loser_serve_win_pct'] = df_l_features['serve_win_pct']
    df['loser_return_win_pct'] = df_l_features['return_win_pct']
    df['loser_sets_dropped_avg'] = df_l_features['sets_dropped_avg']
    df['loser_surface_win_pct'] = df_l_features['surface_win_pct']

    # Map back h2h
    df['p1'] = df_h2h['p1']
    df['p2'] = df_h2h['p2']
    df['h2h_p1_win_pct'] = df_h2h['h2h_p1_win_pct']

    # For inference, track the "final" (un-shifted) rolling stats per player
    global latest_player_stats
    latest_player_stats = {}

    # Also we need to store h2h final states
    global latest_h2h
    latest_h2h = {}
    for (p1, p2), group in df_h2h.groupby(['p1', 'p2']):
        latest_h2h[(p1, p2)] = group.iloc[-1]['final_h2h_p1_win_pct']

    # We need to capture surface stats by surface
    # Iterate players
    for pid, group in player_history.groupby('player_id'):
        last_row = group.iloc[-1]
        name = df.loc[last_row.name, 'winner_name'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_name']
        rank = df.loc[last_row.name, 'winner_rank'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_rank']
        age = df.loc[last_row.name, 'winner_age'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_age']
        points = df.loc[last_row.name, 'winner_rank_points'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_rank_points']
        hand = df.loc[last_row.name, 'winner_hand'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_hand']

        surface_pcts = {}
        # Get surface final stats for this player
        surf_group = group.dropna(subset=['final_surface_win_pct']).drop_duplicates('surface', keep='last')
        for _, r in surf_group.iterrows():
            surface_pcts[r['surface']] = r['final_surface_win_pct']

        latest_player_stats[name] = {
            'id': pid,
            'rank': rank, 'age': age, 'points': points, 'hand': hand,
            'elo': final_elos.get(pid, 1500),
            'recent_win_pct': last_row['final_recent_win_pct'],
            'serve_win_pct': last_row['final_serve_win_pct'],
            'return_win_pct': last_row['final_return_win_pct'],
            'sets_dropped_avg': last_row['final_sets_dropped_avg'],
            'surface_pcts': surface_pcts
        }

    return df

def balance_dataset(df):

    rows = []

    # Fill NA ages
    df['winner_age'] = df['winner_age'].fillna(df['winner_age'].median())
    df['loser_age'] = df['loser_age'].fillna(df['loser_age'].median())

    # Fill NA hands
    df['winner_hand'] = df['winner_hand'].fillna('R')
    df['loser_hand'] = df['loser_hand'].fillna('R')

    for idx, row in df.iterrows():
        surface = row['surface']
        date = row['tourney_date']
        level = row['tourney_level']

        # Winner stats
        w_id = row['winner_id']
        w_name = row['winner_name']
        w_rank = row['winner_rank']
        w_points = row['winner_rank_points']
        w_elo = row['winner_elo']
        w_age = row['winner_age']
        w_hand = row['winner_hand']
        w_recent_win_pct = row['winner_recent_win_pct']
        w_serve_win_pct = row['winner_serve_win_pct']
        w_return_win_pct = row['winner_return_win_pct']
        w_sets_dropped_avg = row['winner_sets_dropped_avg']
        w_surface_win_pct = row['winner_surface_win_pct']

        # Loser stats
        l_id = row['loser_id']
        l_name = row['loser_name']
        l_rank = row['loser_rank']
        l_points = row['loser_rank_points']
        l_elo = row['loser_elo']
        l_age = row['loser_age']
        l_hand = row['loser_hand']
        l_recent_win_pct = row['loser_recent_win_pct']
        l_serve_win_pct = row['loser_serve_win_pct']
        l_return_win_pct = row['loser_return_win_pct']
        l_sets_dropped_avg = row['loser_sets_dropped_avg']
        l_surface_win_pct = row['loser_surface_win_pct']

        h2h = row['h2h_p1_win_pct']
        p1 = row['p1']

        w_h2h = h2h if w_id == p1 else (1 - h2h)
        l_h2h = h2h if l_id == p1 else (1 - h2h)

        # Row 1: Player A is Winner
        rows.append({
            'date': date,
            'player_a_name': w_name,
            'player_b_name': l_name,
            'surface': surface,
            'tourney_level': level,

            'player_a_rank': w_rank,
            'player_a_age': w_age,
            'player_a_hand': w_hand,
            'player_a_elo': w_elo,
            'player_a_recent_win_pct': w_recent_win_pct,
            'player_a_serve_win_pct': w_serve_win_pct,
            'player_a_return_win_pct': w_return_win_pct,
            'player_a_sets_dropped_avg': w_sets_dropped_avg,
            'player_a_surface_win_pct': w_surface_win_pct,
            'point_difference': w_points - l_points,
            'elo_difference': w_elo - l_elo,
            'h2h_win_pct': w_h2h,

            'player_b_rank': l_rank,
            'player_b_age': l_age,
            'player_b_hand': l_hand,
            'player_b_elo': l_elo,
            'player_b_recent_win_pct': l_recent_win_pct,
            'player_b_serve_win_pct': l_serve_win_pct,
            'player_b_return_win_pct': l_return_win_pct,
            'player_b_sets_dropped_avg': l_sets_dropped_avg,
            'player_b_surface_win_pct': l_surface_win_pct,

            'target': 1
        })

        # Row 2: Player A is Loser
        rows.append({
            'date': date,
            'player_a_name': l_name,
            'player_b_name': w_name,
            'surface': surface,
            'tourney_level': level,

            'player_a_rank': l_rank,
            'player_a_age': l_age,
            'player_a_hand': l_hand,
            'player_a_elo': l_elo,
            'player_a_recent_win_pct': l_recent_win_pct,
            'player_a_serve_win_pct': l_serve_win_pct,
            'player_a_return_win_pct': l_return_win_pct,
            'player_a_sets_dropped_avg': l_sets_dropped_avg,
            'player_a_surface_win_pct': l_surface_win_pct,
            'point_difference': l_points - w_points,
            'elo_difference': l_elo - w_elo,
            'h2h_win_pct': l_h2h,

            'player_b_rank': w_rank,
            'player_b_age': w_age,
            'player_b_hand': w_hand,
            'player_b_elo': w_elo,
            'player_b_recent_win_pct': w_recent_win_pct,
            'player_b_serve_win_pct': w_serve_win_pct,
            'player_b_return_win_pct': w_return_win_pct,
            'player_b_sets_dropped_avg': w_sets_dropped_avg,
            'player_b_surface_win_pct': w_surface_win_pct,

            'target': 0
        })

    balanced_df = pd.DataFrame(rows)
    # Sort again just to be safe
    balanced_df = balanced_df.sort_values('date').reset_index(drop=True)
    return balanced_df

def preprocess_and_train():
    global model, scaler, label_encoders

    print("Loading and engineering data...")
    df = load_and_engineer_data()
    print("Balancing dataset...")
    balanced_df = balance_dataset(df)

    # Preprocessing
    print("Preprocessing data...")
    # Drop NAs if any remain
    balanced_df = balanced_df.dropna(subset=features + ['target'])

    X = balanced_df[features].copy()
    y = balanced_df['target'].copy()

    # Label Encoding for categorical
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unseen labels by filling with a string or string conversion
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    print("Training model using TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        tree_method='hist',
        n_jobs=1,
        early_stopping_rounds=50,
        random_state=42,
        eval_metric='logloss'
    )

    # We evaluate on the last split to give an overall accuracy score
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

    y_pred = model.predict(X_test)
    print("\n--- Evaluation on Last Time Split ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nGenerating feature importance plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plot_path = os.path.join(MODEL_DIR, "feature_importances.png")
        plt.savefig(plot_path)
        print(f"Saved {plot_path}")
    else:
        print("matplotlib is not installed. Skipping feature importances plot generation.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Saving model, preprocessors, and inference dictionaries to {MODEL_DIR}...")
    joblib.dump(model, os.path.join(MODEL_DIR, "tennis_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    joblib.dump(latest_player_stats, os.path.join(MODEL_DIR, "latest_player_stats.joblib"))
    joblib.dump(latest_h2h, os.path.join(MODEL_DIR, "latest_h2h.joblib"))
    print(f"Saved all artifacts to {MODEL_DIR}")

def predict_match(player_a_name, player_b_name, surface, tourney_level='G'):
    global model, scaler, label_encoders, latest_player_stats, latest_h2h

    if model is None:
        print("Model not trained yet. Call preprocess_and_train() first.")
        return

    if player_a_name not in latest_player_stats:
        print(f"Error: {player_a_name} not found in historical data.")
        return
    if player_b_name not in latest_player_stats:
        print(f"Error: {player_b_name} not found in historical data.")
        return

    a_stats = latest_player_stats[player_a_name]
    b_stats = latest_player_stats[player_b_name]

    a_id = a_stats['id']
    b_id = b_stats['id']
    p1 = min(a_id, b_id)
    p2 = max(a_id, b_id)

    h2h = latest_h2h.get((p1, p2), 0.5)
    a_h2h = h2h if a_id == p1 else (1 - h2h)

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

    # Preprocess
    for col in categorical_cols:
        le = label_encoders[col]
        # handle unseen
        val = df_pred[col].iloc[0]
        if val in le.classes_:
            df_pred[col] = le.transform([val])
        else:
             # Default to first if unseen
            df_pred[col] = le.transform([le.classes_[0]])

    df_pred[numerical_cols] = scaler.transform(df_pred[numerical_cols])

    # Reorder to match features
    df_pred = df_pred[features]

    prob = model.predict_proba(df_pred)[0]

    print(f"\nPrediction for {player_a_name} vs {player_b_name} on {surface}:")
    print(f"{player_a_name} Win Probability: {prob[1]*100:.2f}%")
    print(f"{player_b_name} Win Probability: {prob[0]*100:.2f}%")

if __name__ == "__main__":
    preprocess_and_train()
    print("\nTesting Inference:")
    predict_match("Novak Djokovic", "Rafael Nadal", "Clay", "G")