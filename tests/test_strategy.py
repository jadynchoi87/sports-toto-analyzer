import pytest
from src.strategy import calc_ev, calc_kelly, apply_filter, best_candidate


def test_calc_ev_positive():
    # prob=0.6, odds=2.0 → b=1, EV = 0.6*1 - 0.4 = 0.2
    ev = calc_ev(0.6, 2.0)
    assert abs(ev - 0.2) < 0.001


def test_calc_ev_negative():
    # prob=0.3, odds=2.0 → b=1, EV = 0.3*1 - 0.7 = -0.4
    ev = calc_ev(0.3, 2.0)
    assert ev < 0


def test_calc_kelly_positive():
    kelly = calc_kelly(0.7, 2.0)
    assert kelly > 0


def test_calc_kelly_half():
    full_kelly = calc_kelly(0.7, 2.0, half=False)
    half_kelly = calc_kelly(0.7, 2.0, half=True)
    assert abs(half_kelly - full_kelly / 2) < 0.001


def test_calc_kelly_zero_when_no_edge():
    kelly = calc_kelly(0.4, 2.0)
    assert kelly == 0.0


def test_apply_filter_passes():
    candidates = [{'prob': 0.65, 'ev': 0.10, 'odds_moved_same': True}]
    result = apply_filter(candidates)
    assert len(result) == 1


def test_apply_filter_rejects_low_prob():
    candidates = [{'prob': 0.50, 'ev': 0.10, 'odds_moved_same': True}]
    result = apply_filter(candidates)
    assert len(result) == 0


def test_apply_filter_rejects_low_ev():
    candidates = [{'prob': 0.65, 'ev': 0.03, 'odds_moved_same': True}]
    result = apply_filter(candidates)
    assert len(result) == 0


def test_apply_filter_rejects_wrong_direction():
    candidates = [{'prob': 0.65, 'ev': 0.10, 'odds_moved_same': False}]
    result = apply_filter(candidates)
    assert len(result) == 0


def test_best_candidate_returns_highest_ev():
    # home: prob=0.65, odds=1.9 → ev = 0.65*0.9 - 0.35 = 0.235
    probs = {'home': 0.65, 'draw': 0.20, 'away': 0.15}
    odds = {'home': 1.9, 'draw': 3.5, 'away': 5.0}
    result = best_candidate(probs, odds)
    assert result is not None
    assert result['outcome'] == 'home'


def test_best_candidate_returns_max_ev_candidate():
    """best_candidate returns max EV regardless of filter; apply_filter handles constraints."""
    probs = {'home': 0.40, 'draw': 0.30, 'away': 0.30}
    odds = {'home': 2.0, 'draw': 3.0, 'away': 3.5}
    result = best_candidate(probs, odds)
    # returns the candidate with highest EV, not filtered
    assert result is not None
    assert result['outcome'] == 'away'  # highest EV among the three


def test_best_candidate_with_empty_probs():
    """With zero probs, EV is always negative but still returns a candidate."""
    result = best_candidate({}, {})
    assert result is not None  # returns max EV even if all are negative
