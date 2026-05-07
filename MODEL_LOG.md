# Model Training Log
## Run at: Thu May  7 08:41:56 UTC 2026
```
Loading and engineering data...
Balancing dataset...
Preprocessing data...
Training model using TimeSeriesSplit...

--- Evaluation on Last Time Split ---
Overall Accuracy: 0.7573

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.75      0.76      4722
           1       0.76      0.76      0.76      4721

    accuracy                           0.76      9443
   macro avg       0.76      0.76      0.76      9443
weighted avg       0.76      0.76      0.76      9443


Confusion Matrix:
[[3564 1158]
 [1134 3587]]

Generating feature importance plot...
Saved feature_importances.png
Saving model and preprocessors using joblib...
Saved tennis_model.joblib, scaler.joblib, and label_encoders.joblib

Testing Inference:

Prediction for Novak Djokovic vs Rafael Nadal on Clay:
Novak Djokovic Win Probability: 94.99%
Rafael Nadal Win Probability: 5.01%
```

## Run at: Thu May  7 12:50:18 UTC 2026
### Model: XGBoost
```
Loading and engineering data...
Balancing dataset...
Preprocessing data...
Training model using TimeSeriesSplit...

--- Evaluation on Last Time Split ---
Overall Accuracy: 0.7571

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.76      0.76      4722
           1       0.76      0.76      0.76      4721

    accuracy                           0.76      9443
   macro avg       0.76      0.76      0.76      9443
weighted avg       0.76      0.76      0.76      9443


Confusion Matrix:
[[3574 1148]
 [1146 3575]]

Generating feature importance plot...
Saved feature_importances.png
Saving model and preprocessors using joblib...
Saved tennis_model.joblib, scaler.joblib, and label_encoders.joblib

Testing Inference:

Prediction for Novak Djokovic vs Rafael Nadal on Clay:
Novak Djokovic Win Probability: 97.42%
Rafael Nadal Win Probability: 2.58%
```

## Run at: Thu May  7 13:54:29 UTC 2026
### Model: XGBoost (v2 - Overhauled Features)
```
Loading and engineering data...
Balancing dataset...
Preprocessing data...
Training model using TimeSeriesSplit...

--- Evaluation on Last Time Split ---
Overall Accuracy: 0.7595

Confusion Matrix:
[[3601 1121]
 [1150 3571]]

Generating feature importance plot...
Saved feature_importances.png
Saving model and preprocessors using joblib...
Saved tennis_model.joblib, scaler.joblib, and label_encoders.joblib

Testing Inference:

Prediction for Novak Djokovic vs Rafael Nadal on Clay:
Novak Djokovic Win Probability: 98.14%
Rafael Nadal Win Probability: 1.86%
```

## Run at: Thu May  7 14:20:34 UTC 2026
### Model: XGBoost (v3 - Removed days_since_last_match)
```
Loading and engineering data...
Balancing dataset...
Preprocessing data...
Training model using TimeSeriesSplit...

--- Evaluation on Last Time Split ---
Overall Accuracy: 0.6525

Confusion Matrix:
[[3075 1647]
 [1634 3087]]

Generating feature importance plot...
Saved feature_importances.png
Saving model and preprocessors using joblib...
Saved tennis_model.joblib, scaler.joblib, and label_encoders.joblib

Testing Inference:

Prediction for Novak Djokovic vs Rafael Nadal on Clay:
Novak Djokovic Win Probability: 84.71%
Rafael Nadal Win Probability: 15.29%
```

## Run at: Thu May  7 14:56:25 UTC 2026
### Model: XGBoost (v4 - Advanced Historical Context)
```
Loading and engineering data...
Balancing dataset...
Preprocessing data...
Training model using TimeSeriesSplit...

--- Evaluation on Last Time Split ---
Overall Accuracy: 0.6528

Confusion Matrix:
[[3079 1643]
 [1636 3085]]

Generating feature importance plot...
Saved feature_importances.png
Saving model and preprocessors using joblib...
Saved tennis_model.joblib, scaler.joblib, and label_encoders.joblib

Testing Inference:

Prediction for Novak Djokovic vs Rafael Nadal on Clay:
Novak Djokovic Win Probability: 81.52%
Rafael Nadal Win Probability: 18.48%
```
