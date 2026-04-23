# src/model.py
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.features import build_features

MODEL_PATH = "data/model.pkl"


def train_model(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """Train Random Forest model and save to disk."""
    df = build_features(df)
    df = df.dropna(subset=['home_score', 'away_score'])

    feature_cols = ['home_win_rate', 'home_draw_rate', 'away_win_rate', 'away_draw_rate', 'home_advantage']
    X = df[feature_cols]
    y = df['result']

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    parent = os.path.dirname(model_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model


def load_model(model_path: str = MODEL_PATH):
    """Load model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_proba(model, row: dict) -> dict:
    """
    Predict probabilities for a single match.
    row must have: home_win_rate, home_draw_rate, away_win_rate, away_draw_rate, home_advantage
    Returns dict with H, D, A probabilities.
    """
    feature_cols = ['home_win_rate', 'home_draw_rate', 'away_win_rate', 'away_draw_rate', 'home_advantage']
    X = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    return {c: float(p) for c, p in zip(classes, proba)}
