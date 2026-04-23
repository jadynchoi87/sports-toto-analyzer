import pytest
from src.backtest import run_backtest


def make_prediction(outcome, home_score, away_score, odds, kelly=0.1):
    return {
        'recommended_outcome': outcome,
        'kelly_fraction': kelly,
        'home_score': home_score,
        'away_score': away_score,
        'home_odds': odds if outcome == 'home' else 1.5,
        'draw_odds': odds if outcome == 'draw' else 3.0,
        'away_odds': odds if outcome == 'away' else 4.0,
    }


def test_backtest_returns_dict():
    preds = [make_prediction('home', 2, 1, 2.0)]
    result = run_backtest(preds)
    assert 'roi' in result
    assert 'hit_rate' in result
    assert 'total' in result
    assert 'profit' in result


def test_backtest_single_win():
    preds = [make_prediction('home', 2, 1, 2.0, kelly=0.1)]
    result = run_backtest(preds, bankroll=1000.0)
    assert result['total'] == 1
    assert result['hit_rate'] == 100.0
    assert result['profit'] > 0


def test_backtest_single_loss():
    preds = [make_prediction('home', 0, 1, 2.0, kelly=0.1)]
    result = run_backtest(preds, bankroll=1000.0)
    assert result['total'] == 1
    assert result['hit_rate'] == 0.0
    assert result['profit'] < 0


def test_backtest_empty():
    result = run_backtest([])
    assert result['total'] == 0
    assert result['roi'] == 0.0


def test_backtest_hit_rate_calculation():
    preds = [
        make_prediction('home', 2, 1, 2.0),   # win
        make_prediction('home', 0, 1, 2.0),   # loss
        make_prediction('draw', 1, 1, 3.0),   # win
        make_prediction('away', 2, 0, 4.0),   # loss
    ]
    result = run_backtest(preds)
    assert result['total'] == 4
    assert result['hit_rate'] == 50.0
