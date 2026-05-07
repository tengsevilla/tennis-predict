import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Global variables to hold model, scaler, encoders, and latest player stats for inference
model = None
scaler = None
label_encoders = {}
latest_player_stats = {}
categorical_cols = ['surface']
numerical_cols = ['player_a_rank', 'player_a_age', 'player_a_days_since_last_match',
                  'player_b_rank', 'player_b_age', 'player_b_days_since_last_match']
features = numerical_cols + categorical_cols

def load_and_engineer_data(filepath="cleaned_atp_data.csv"):
    df = pd.read_csv(filepath)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # Calculate days_since_last_match
    last_match_dates = {}
    winner_days = []
    loser_days = []

    for idx, row in df.iterrows():
        w_id = row['winner_id']
        l_id = row['loser_id']
        date = row['tourney_date']

        if w_id in last_match_dates:
            winner_days.append((date - last_match_dates[w_id]).days)
        else:
            winner_days.append(90)

        if l_id in last_match_dates:
            loser_days.append((date - last_match_dates[l_id]).days)
        else:
            loser_days.append(90)

        last_match_dates[w_id] = date
        last_match_dates[l_id] = date

    df['winner_days_since_last_match'] = winner_days
    df['loser_days_since_last_match'] = loser_days

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

        w_name = row['winner_name']
        w_rank = row['winner_rank']
        w_age = row['winner_age']
        w_days = row['winner_days_since_last_match']

        l_name = row['loser_name']
        l_rank = row['loser_rank']
        l_age = row['loser_age']
        l_days = row['loser_days_since_last_match']

        # Row 1: Player A is Winner
        rows.append({
            'date': date,
            'player_a_name': w_name,
            'player_b_name': l_name,
            'surface': surface,
            'player_a_rank': w_rank,
            'player_a_age': w_age,
            'player_a_days_since_last_match': w_days,
            'player_b_rank': l_rank,
            'player_b_age': l_age,
            'player_b_days_since_last_match': l_days,
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
            'player_b_rank': w_rank,
            'player_b_age': w_age,
            'player_b_days_since_last_match': w_days,
            'target': 0
        })

        # Keep track of latest stats for inference
        latest_player_stats[w_name] = {'rank': w_rank, 'age': w_age, 'days_since_last_match': w_days, 'last_date': date}
        latest_player_stats[l_name] = {'rank': l_rank, 'age': l_age, 'days_since_last_match': l_days, 'last_date': date}

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

    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)

    # We evaluate on the last split to give an overall accuracy score
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Evaluation on Last Time Split ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
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

    # We don't recalculate days_since_last_match to current real-world date for this example,
    # we just use their most recent days_since_last_match from the dataset to demonstrate the feature.
    # In a real system, you'd calculate: (datetime.now() - a_stats['last_date']).days

    row_dict = {
        'player_a_rank': a_stats['rank'],
        'player_a_age': a_stats['age'],
        'player_a_days_since_last_match': a_stats['days_since_last_match'],
        'player_b_rank': b_stats['rank'],
        'player_b_age': b_stats['age'],
        'player_b_days_since_last_match': b_stats['days_since_last_match'],
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
