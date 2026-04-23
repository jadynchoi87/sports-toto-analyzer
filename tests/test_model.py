import pytest
import pandas as pd
import os
from src.model import train_model, load_model, predict_proba


@pytest.fixture
def sample_df():
    rows = []
    scores = [(2, 1), (0, 1), (1, 1), (3, 0), (0, 0), (1, 2)]  # home, away, draw, home, draw, away
    for i in range(60):
        hs, as_ = scores[i % len(scores)]
        rows.append({
            'match_id': str(i),
            'date': f'2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}',
            'home_team': f'Team{i % 5}',
            'away_team': f'Team{(i + 1) % 5}',
            'home_score': hs,
            'away_score': as_,
            'competition': 'PL',
        })
    return pd.DataFrame(rows)


def test_train_model_saves_file(sample_df, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model = train_model(sample_df, model_path=model_path)
    assert os.path.exists(model_path)
    assert model is not None


def test_load_model(sample_df, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    train_model(sample_df, model_path=model_path)
    model = load_model(model_path=model_path)
    assert model is not None


def test_predict_proba_returns_dict(sample_df, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model = train_model(sample_df, model_path=model_path)
    row = {'home_win_rate': 0.6, 'home_draw_rate': 0.2, 'away_win_rate': 0.3, 'away_draw_rate': 0.1, 'home_advantage': 1.0}
    result = predict_proba(model, row)
    assert isinstance(result, dict)
    assert abs(sum(result.values()) - 1.0) < 0.01


def test_predict_proba_has_all_outcomes(sample_df, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model = train_model(sample_df, model_path=model_path)
    row = {'home_win_rate': 0.6, 'home_draw_rate': 0.2, 'away_win_rate': 0.3, 'away_draw_rate': 0.1, 'home_advantage': 1.0}
    result = predict_proba(model, row)
    for outcome in ['home', 'draw', 'away']:
        assert outcome in result
