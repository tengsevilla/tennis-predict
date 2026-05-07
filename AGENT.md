# AI Agent Instructions

- This project is a Python-based machine learning pipeline that relies on `pandas`, `scikit-learn`, and `matplotlib`.
- Raw data files (like `atp_matches_*.csv`) must never be checked into the repository. Only scripts and documentation should be committed.
- When making modifications, ensure chronological order is respected (e.g., use `TimeSeriesSplit` for cross-validation) to prevent data leakage in predictions.
- `cleaned_atp_data.csv` and any generated plots (like `feature_importances.png`) should be added to `.gitignore`.