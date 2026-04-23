import pytest
import pandas as pd
from src.features import build_features


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {'match_id': '1', 'date': '2024-01-01', 'home_team': 'A', 'away_team': 'B', 'home_score': 2, 'away_score': 1, 'competition': 'PL'},
        {'match_id': '2', 'date': '2024-01-08', 'home_team': 'C', 'away_team': 'A', 'home_score': 0, 'away_score': 0, 'competition': 'PL'},
        {'match_id': '3', 'date': '2024-01-15', 'home_team': 'A', 'away_team': 'C', 'home_score': 1, 'away_score': 0, 'competition': 'PL'},
        {'match_id': '4', 'date': '2024-01-22', 'home_team': 'B', 'away_team': 'A', 'home_score': 2, 'away_score': 3, 'competition': 'PL'},
    ])


def test_build_features_returns_dataframe(sample_df):
    result = build_features(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_build_features_adds_columns(sample_df):
    result = build_features(sample_df)
    for col in ['home_win_rate', 'home_draw_rate', 'away_win_rate', 'away_draw_rate', 'home_advantage', 'result']:
        assert col in result.columns, f"Missing column: {col}"


def test_result_column_values(sample_df):
    result = build_features(sample_df)
    assert set(result['result'].unique()).issubset({'home', 'draw', 'away'})


def test_home_win_rate_range(sample_df):
    result = build_features(sample_df)
    assert result['home_win_rate'].between(0.0, 1.0).all()


def test_home_advantage_constant(sample_df):
    result = build_features(sample_df)
    assert (result['home_advantage'] == 1.0).all()


def test_no_data_leakage(sample_df):
    """First match should have 0 past games."""
    result = build_features(sample_df)
    assert result.iloc[0]['home_n'] == 0
    assert result.iloc[0]['away_n'] == 0
