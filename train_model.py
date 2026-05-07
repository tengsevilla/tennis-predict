import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import re

# Global variables to hold model, scaler, encoders, and latest player stats for inference
model = None
scaler = None
label_encoders = {}
latest_player_stats = {}
categorical_cols = ['surface']
numerical_cols = [
    'player_a_rank', 'player_a_age', 'player_a_days_since_last_match',
    'player_a_recent_win_pct', 'player_a_serve_win_pct', 'player_a_return_win_pct', 'player_a_sets_dropped_avg',
    'player_b_rank', 'player_b_age', 'player_b_days_since_last_match',
    'player_b_recent_win_pct', 'player_b_serve_win_pct', 'player_b_return_win_pct', 'player_b_sets_dropped_avg'
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

def load_and_engineer_data(filepath="cleaned_atp_data.csv"):
    df = pd.read_csv(filepath)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # Calculate days_since_last_match
    last_match_dates = {}
    winner_days = []
    loser_days = []

    # Also calculate sets won for rolling stats
    w_sets_dropped = []
    l_sets_dropped = []

    for idx, row in df.iterrows():
        w_id = row['winner_id']
        l_id = row['loser_id']
        date = row['tourney_date']

        # Days since last match cap at 30
        w_days = (date - last_match_dates[w_id]).days if w_id in last_match_dates else 30
        l_days = (date - last_match_dates[l_id]).days if l_id in last_match_dates else 30
        winner_days.append(min(w_days, 30))
        loser_days.append(min(l_days, 30))

        last_match_dates[w_id] = date
        last_match_dates[l_id] = date

        w_sets, l_sets = count_sets_won(row['score'])
        # sets dropped by winner = sets won by loser
        w_sets_dropped.append(l_sets)
        # sets dropped by loser = sets won by winner
        l_sets_dropped.append(w_sets)

    df['winner_days_since_last_match'] = winner_days
    df['loser_days_since_last_match'] = loser_days
    df['winner_sets_dropped'] = w_sets_dropped
    df['loser_sets_dropped'] = l_sets_dropped

    # Melt dataframe to player level to calculate rolling stats safely
    winner_cols = ['tourney_date', 'winner_id', 'winner_sets_dropped', 'w_1stWon', 'w_2ndWon', 'w_svpt', 'l_svpt', 'l_1stWon', 'l_2ndWon']
    loser_cols = ['tourney_date', 'loser_id', 'loser_sets_dropped', 'l_1stWon', 'l_2ndWon', 'l_svpt', 'w_svpt', 'w_1stWon', 'w_2ndWon']

    df_w = df[winner_cols].copy()
    df_w.columns = ['date', 'player_id', 'sets_dropped', 'serve_won', 'serve_2_won', 'serve_pt', 'opp_serve_pt', 'opp_serve_won', 'opp_serve_2_won']
    df_w['won_match'] = 1

    df_l = df[loser_cols].copy()
    df_l.columns = ['date', 'player_id', 'sets_dropped', 'serve_won', 'serve_2_won', 'serve_pt', 'opp_serve_pt', 'opp_serve_won', 'opp_serve_2_won']
    df_l['won_match'] = 0

    # Keep original index for safe realignment later
    player_history = pd.concat([df_w, df_l])
    # Sort for rolling operations
    player_history = player_history.sort_values(['player_id', 'date'])

    # Compute points
    player_history['serve_pts_won'] = player_history['serve_won'] + player_history['serve_2_won']
    player_history['return_pts_won'] = player_history['opp_serve_pt'] - (player_history['opp_serve_won'] + player_history['opp_serve_2_won'])

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

    # Do not drop index so it preserves the original DataFrame index mapping
    player_history = player_history.groupby('player_id', group_keys=False)[player_history.columns].apply(rolling_avg)

    # Fill NAs for players with no history yet with 0 or a reasonable default
    player_history['recent_win_pct'] = player_history['recent_win_pct'].fillna(0.5)
    player_history['serve_win_pct'] = player_history['serve_win_pct'].fillna(0.6)
    player_history['return_win_pct'] = player_history['return_win_pct'].fillna(0.4)
    player_history['sets_dropped_avg'] = player_history['sets_dropped_avg'].fillna(1.0)

    # Realign using the original index!
    # player_history was concatenated from df_w and df_l which both had df's original index
    df_w_features = player_history[player_history['won_match'] == 1].sort_index()
    df_l_features = player_history[player_history['won_match'] == 0].sort_index()

    df['winner_recent_win_pct'] = df_w_features['recent_win_pct']
    df['winner_serve_win_pct'] = df_w_features['serve_win_pct']
    df['winner_return_win_pct'] = df_w_features['return_win_pct']
    df['winner_sets_dropped_avg'] = df_w_features['sets_dropped_avg']

    df['loser_recent_win_pct'] = df_l_features['recent_win_pct']
    df['loser_serve_win_pct'] = df_l_features['serve_win_pct']
    df['loser_return_win_pct'] = df_l_features['return_win_pct']
    df['loser_sets_dropped_avg'] = df_l_features['sets_dropped_avg']

    # For inference, track the "final" (un-shifted) rolling stats per player
    global latest_player_stats
    latest_player_stats = {}
    for pid, group in player_history.groupby('player_id'):
        last_row = group.iloc[-1]
        name = df.loc[last_row.name, 'winner_name'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_name']
        rank = df.loc[last_row.name, 'winner_rank'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_rank']
        age = df.loc[last_row.name, 'winner_age'] if last_row['won_match'] == 1 else df.loc[last_row.name, 'loser_age']
        latest_player_stats[name] = {
            'rank': rank, 'age': age,
            'days_since_last_match': 30, # We just default to 30 for future inferences assuming typical tournament breaks
            'recent_win_pct': last_row['final_recent_win_pct'],
            'serve_win_pct': last_row['final_serve_win_pct'],
            'return_win_pct': last_row['final_return_win_pct'],
            'sets_dropped_avg': last_row['final_sets_dropped_avg']
        }

    return df

def balance_dataset(df):
    global latest_player_stats

    rows = []

    # Fill NA ages
    df['winner_age'] = df['winner_age'].fillna(df['winner_age'].median())
    df['loser_age'] = df['loser_age'].fillna(df['loser_age'].median())

    for idx, row in df.iterrows():
        surface = row['surface']
        date = row['tourney_date']

        # Winner stats
        w_name = row['winner_name']
        w_rank = row['winner_rank']
        w_age = row['winner_age']
        w_days = row['winner_days_since_last_match']
        w_recent_win_pct = row['winner_recent_win_pct']
        w_serve_win_pct = row['winner_serve_win_pct']
        w_return_win_pct = row['winner_return_win_pct']
        w_sets_dropped_avg = row['winner_sets_dropped_avg']

        # Loser stats
        l_name = row['loser_name']
        l_rank = row['loser_rank']
        l_age = row['loser_age']
        l_days = row['loser_days_since_last_match']
        l_recent_win_pct = row['loser_recent_win_pct']
        l_serve_win_pct = row['loser_serve_win_pct']
        l_return_win_pct = row['loser_return_win_pct']
        l_sets_dropped_avg = row['loser_sets_dropped_avg']

        # Row 1: Player A is Winner
        rows.append({
            'date': date,
            'player_a_name': w_name,
            'player_b_name': l_name,
            'surface': surface,
            'player_a_rank': w_rank,
            'player_a_age': w_age,
            'player_a_days_since_last_match': w_days,
            'player_a_recent_win_pct': w_recent_win_pct,
            'player_a_serve_win_pct': w_serve_win_pct,
            'player_a_return_win_pct': w_return_win_pct,
            'player_a_sets_dropped_avg': w_sets_dropped_avg,

            'player_b_rank': l_rank,
            'player_b_age': l_age,
            'player_b_days_since_last_match': l_days,
            'player_b_recent_win_pct': l_recent_win_pct,
            'player_b_serve_win_pct': l_serve_win_pct,
            'player_b_return_win_pct': l_return_win_pct,
            'player_b_sets_dropped_avg': l_sets_dropped_avg,
            'target': 1
        })

        # Row 2: Player A is Loser
        rows.append({
            'date': date,
            'player_a_name': l_name,
            'player_b_name': w_name,
            'surface': surface,
            'player_a_rank': l_rank,
            'player_a_age': l_age,
            'player_a_days_since_last_match': l_days,
            'player_a_recent_win_pct': l_recent_win_pct,
            'player_a_serve_win_pct': l_serve_win_pct,
            'player_a_return_win_pct': l_return_win_pct,
            'player_a_sets_dropped_avg': l_sets_dropped_avg,

            'player_b_rank': w_rank,
            'player_b_age': w_age,
            'player_b_days_since_last_match': w_days,
            'player_b_recent_win_pct': w_recent_win_pct,
            'player_b_serve_win_pct': w_serve_win_pct,
            'player_b_return_win_pct': w_return_win_pct,
            'player_b_sets_dropped_avg': w_sets_dropped_avg,
            'target': 0
        })

        # Keep track of latest stats for inference
        latest_player_stats[w_name] = {
            'rank': w_rank, 'age': w_age, 'days_since_last_match': w_days,
            'recent_win_pct': w_recent_win_pct, 'serve_win_pct': w_serve_win_pct,
            'return_win_pct': w_return_win_pct, 'sets_dropped_avg': w_sets_dropped_avg
        }
        latest_player_stats[l_name] = {
            'rank': l_rank, 'age': l_age, 'days_since_last_match': l_days,
            'recent_win_pct': l_recent_win_pct, 'serve_win_pct': l_serve_win_pct,
            'return_win_pct': l_return_win_pct, 'sets_dropped_avg': l_sets_dropped_avg
        }

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

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    print("Saved feature_importances.png")

    print("Saving model and preprocessors using joblib...")
    joblib.dump(model, "tennis_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(label_encoders, "label_encoders.joblib")
    print("Saved tennis_model.joblib, scaler.joblib, and label_encoders.joblib")

def predict_match(player_a_name, player_b_name, surface):
    global model, scaler, label_encoders, latest_player_stats

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

    row_dict = {
        'player_a_rank': a_stats['rank'],
        'player_a_age': a_stats['age'],
        'player_a_days_since_last_match': a_stats['days_since_last_match'],
        'player_a_recent_win_pct': a_stats['recent_win_pct'],
        'player_a_serve_win_pct': a_stats['serve_win_pct'],
        'player_a_return_win_pct': a_stats['return_win_pct'],
        'player_a_sets_dropped_avg': a_stats['sets_dropped_avg'],

        'player_b_rank': b_stats['rank'],
        'player_b_age': b_stats['age'],
        'player_b_days_since_last_match': b_stats['days_since_last_match'],
        'player_b_recent_win_pct': b_stats['recent_win_pct'],
        'player_b_serve_win_pct': b_stats['serve_win_pct'],
        'player_b_return_win_pct': b_stats['return_win_pct'],
        'player_b_sets_dropped_avg': b_stats['sets_dropped_avg'],
        'surface': surface
    }

    df_pred = pd.DataFrame([row_dict])

    # Preprocess
    for col in categorical_cols:
        le = label_encoders[col]
        # handle unseen surface
        if surface in le.classes_:
            df_pred[col] = le.transform([surface])
        else:
             # Default to first surface if unseen
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
    predict_match("Novak Djokovic", "Rafael Nadal", "Clay")
