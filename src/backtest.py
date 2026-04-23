# src/backtest.py


def run_backtest(predictions: list[dict], bankroll: float = 1000.0) -> dict:
    """
    Run backtest on predictions.

    Each prediction dict needs:
    - recommended_outcome: 'home', 'draw', or 'away'
    - kelly_fraction: fraction of bankroll to bet
    - home_score, away_score: actual result
    - home_odds/draw_odds/away_odds: the odds at time of prediction

    Returns: roi, hit_rate, total, profit
    """
    outcome_check = {
        'home': lambda p: p['home_score'] > p['away_score'],
        'draw': lambda p: p['home_score'] == p['away_score'],
        'away': lambda p: p['home_score'] < p['away_score'],
    }
    outcome_odds = {
        'home': lambda p: p.get('home_odds', 1.0),
        'draw': lambda p: p.get('draw_odds', 1.0),
        'away': lambda p: p.get('away_odds', 1.0),
    }

    total_wagered = 0.0
    total_profit = 0.0
    hits = 0
    total = 0
    current_bankroll = bankroll

    for pred in predictions:
        outcome = pred.get('recommended_outcome')
        kelly = pred.get('kelly_fraction', 0.0)

        if outcome not in outcome_check or kelly <= 0:
            continue

        bet_size = current_bankroll * kelly
        is_win = outcome_check[outcome](pred)
        odds = outcome_odds[outcome](pred)

        total_wagered += bet_size
        total += 1

        if is_win:
            profit = bet_size * (odds - 1)
            hits += 1
        else:
            profit = -bet_size

        total_profit += profit
        current_bankroll += profit

    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    hit_rate = (hits / total * 100) if total > 0 else 0.0

    return {
        'roi': round(roi, 2),
        'hit_rate': round(hit_rate, 2),
        'total': total,
        'profit': round(total_profit, 2),
    }
