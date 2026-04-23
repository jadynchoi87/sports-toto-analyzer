# src/strategy.py


def calc_ev(prob: float, odds: float) -> float:
    """EV = (p × b) - (1 - p)  where b = decimal_odds - 1"""
    b = odds - 1
    return prob * b - (1 - prob)


def calc_kelly(prob: float, odds: float, half: bool = True) -> float:
    """
    Kelly Criterion: f = (b*p - q) / b  where b = odds - 1, q = 1 - p
    half=True: use half-Kelly for safety
    """
    if odds <= 1:
        return 0.0
    b = odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    f = max(0.0, f)
    return f * 0.5 if half else f


def apply_filter(candidates: list[dict]) -> list[dict]:
    """
    Filter candidates by:
    - prob >= 0.60
    - ev >= 0.05
    - odds_moved_same == True
    """
    return [
        c for c in candidates
        if c.get('prob', 0) >= 0.60
        and c.get('ev', -1) >= 0.05
        and c.get('odds_moved_same', False)
    ]


def best_candidate(probs: dict, odds: dict) -> dict | None:
    """
    Given probs={'home': 0.6, 'draw': 0.2, 'away': 0.2} and odds={'home': 1.8, ...},
    returns the best candidate dict or None if none pass filter.
    """
    candidates = []
    for outcome in ['home', 'draw', 'away']:
        prob = probs.get(outcome, 0.0)
        odd = odds.get(outcome, 1.0)
        if odd <= 0:
            continue
        ev = calc_ev(prob, odd)
        kelly = calc_kelly(prob, odd)
        candidates.append({
            'outcome': outcome,
            'prob': prob,
            'odds': odd,
            'ev': ev,
            'kelly': kelly,
            'odds_moved_same': True,
        })

    if not candidates:
        return None
    return max(candidates, key=lambda x: x['ev'])
