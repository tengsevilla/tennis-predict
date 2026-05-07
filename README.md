# Tennis Match Predictor

This repository contains tools to fetch, clean, and use historical ATP tennis data to train a machine learning model capable of predicting match outcomes.

## Scripts

- `data_fetcher.py`: Connects to `JeffSackmann/tennis_atp` to fetch ATP data from 2016 to the present day, clean out invalid records, and save to `cleaned_atp_data.csv`.
- `train_model.py`: Reads the cleaned data, engineers time-based features, processes the data, and trains a RandomForest model using cross-validation over time. Outputs accuracy and generates a feature importance plot.

## Setup

Install dependencies:
```bash
pip install pandas scikit-learn matplotlib
```

## Usage
1. Fetch the data:
```bash
python3 data_fetcher.py
```
2. Train the model and see feature importance:
```bash
python3 train_model.py
```
