# src/model.py
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "data/model.pkl"
FEATURE_COLS = [
    'home_win_rate', 'home_draw_rate', 'home_goals_scored', 'home_goals_conceded', 'home_form',
    'away_win_rate', 'away_draw_rate', 'away_goals_scored', 'away_goals_conceded', 'away_form',
    'home_advantage', 'form_diff', 'goal_diff',
    'imp_home', 'imp_draw', 'imp_away',  # 배당 내재 확률
]


def train_model(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    Train Random Forest model and save to disk.
    df must already have feature columns (output of build_features).
    """
    X = df[FEATURE_COLS]
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


def predict_proba(model, row) -> dict:
    """
    Predict probabilities for a single match.
    row must have: home_win_rate, home_draw_rate, away_win_rate, away_draw_rate, home_advantage
    Returns dict with home, draw, away probabilities.
    """
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    X = pd.DataFrame([[row.get(col, 0.0) for col in FEATURE_COLS]], columns=FEATURE_COLS)
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    return {c: float(p) for c, p in zip(classes, proba)}
