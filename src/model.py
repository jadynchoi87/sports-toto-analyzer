# src/model.py
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "data/model.pkl"
LABEL_PATH = "data/label_encoder.pkl"

FEATURE_COLS = [
    'home_win_rate', 'home_draw_rate', 'home_goals_scored', 'home_goals_conceded', 'home_form',
    'away_win_rate', 'away_draw_rate', 'away_goals_scored', 'away_goals_conceded', 'away_form',
    'home_advantage', 'form_diff', 'goal_diff', 'goals_conceded_diff',
    'h2h_home_win_rate', 'h2h_draw_rate',
    'imp_home', 'imp_draw', 'imp_away',
]


def train_model(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    XGBoost 모델 학습 및 저장.
    df는 build_features() 결과여야 함.
    """
    X = df[FEATURE_COLS].fillna(0)
    y_raw = df['result']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    parent = os.path.dirname(model_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(LABEL_PATH, 'wb') as f:
        pickle.dump(le, f)

    return model, le


def load_model(model_path: str = MODEL_PATH):
    """모델 및 레이블 인코더 로드."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, le


def predict_proba(model, le, row) -> dict:
    """
    단일 경기 확률 예측.
    Returns: {'home': 0.6, 'draw': 0.2, 'away': 0.2}
    """
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    X = pd.DataFrame([[row.get(col, 0.0) for col in FEATURE_COLS]], columns=FEATURE_COLS)
    proba = model.predict_proba(X)[0]
    classes = le.classes_
    return {c: float(p) for c, p in zip(classes, proba)}
