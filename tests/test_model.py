import pytest
import pandas as pd
import os
from src.model import train_model, load_model, predict_proba


@pytest.fixture
def sample_df():
    rows = []
    scores = [(2, 1), (0, 1), (1, 1), (3, 0), (0, 0), (1, 2)]
    for i in range(60):
        hs, as_ = scores[i % len(scores)]
        rows.append({
            'match_id': str(i),
            'date': f'2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}',
            'home_team': f'Team{i % 5}',
            'away_team': f'Team{(i + 1) % 5}',
            'home_score': hs,
            'away_score': as_,
            'home_odds': 1.8,
            'draw_odds': 3.5,
            'away_odds': 4.0,
            'competition': 'PL',
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_features(sample_df):
    from src.features import build_features
    return build_features(sample_df)


def test_train_model_saves_files(sample_features, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model, le = train_model(sample_features, model_path=model_path)
    assert os.path.exists(model_path)
    assert model is not None
    assert le is not None


def test_predict_proba_returns_dict(sample_features, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model, le = train_model(sample_features, model_path=model_path)
    row = sample_features.iloc[-1]
    result = predict_proba(model, le, row)
    assert isinstance(result, dict)
    assert abs(sum(result.values()) - 1.0) < 0.01


def test_predict_proba_has_all_outcomes(sample_features, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model, le = train_model(sample_features, model_path=model_path)
    row = sample_features.iloc[-1]
    result = predict_proba(model, le, row)
    for outcome in ['home', 'draw', 'away']:
        assert outcome in result
