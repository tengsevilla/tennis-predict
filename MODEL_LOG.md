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
