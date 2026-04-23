# src/features.py
import pandas as pd
import numpy as np


def _get_team_stats(past: pd.DataFrame, team: str, n: int = 10) -> dict:
    """팀의 최근 N경기 통계 (홈/원정 구분 없이)."""
    games = past[(past['home_team'] == team) | (past['away_team'] == team)].tail(n)
    if len(games) == 0:
        return {'win_rate': 0.0, 'draw_rate': 0.0, 'goals_scored': 0.0,
                'goals_conceded': 0.0, 'form': 0.0, 'n': 0}

    wins, draws, goals_scored, goals_conceded, points = 0, 0, 0, 0, 0
    for _, g in games.iterrows():
        is_home = g['home_team'] == team
        gs = g['home_score'] if is_home else g['away_score']
        gc = g['away_score'] if is_home else g['home_score']
        goals_scored += gs
        goals_conceded += gc
        if gs > gc:
            wins += 1
            points += 3
        elif gs == gc:
            draws += 1
            points += 1

    n_games = len(games)
    return {
        'win_rate': wins / n_games,
        'draw_rate': draws / n_games,
        'goals_scored': goals_scored / n_games,
        'goals_conceded': goals_conceded / n_games,
        'form': points / (n_games * 3),
        'n': n_games,
    }


def _get_home_win_rate(past: pd.DataFrame, team: str, n: int = 10) -> float:
    """팀의 최근 N 홈경기 승률."""
    games = past[past['home_team'] == team].tail(n)
    if len(games) == 0:
        return 0.0
    return (games['home_score'] > games['away_score']).sum() / len(games)


def _get_h2h_stats(past: pd.DataFrame, home_team: str, away_team: str, n: int = 5) -> dict:
    """두 팀 간 최근 N 맞대결 통계."""
    h2h = past[
        ((past['home_team'] == home_team) & (past['away_team'] == away_team)) |
        ((past['home_team'] == away_team) & (past['away_team'] == home_team))
    ].tail(n)

    if len(h2h) == 0:
        return {'h2h_home_win_rate': 0.33, 'h2h_draw_rate': 0.33, 'h2h_n': 0}

    home_wins, draws = 0, 0
    for _, g in h2h.iterrows():
        if g['home_team'] == home_team:
            if g['home_score'] > g['away_score']:
                home_wins += 1
            elif g['home_score'] == g['away_score']:
                draws += 1
        else:
            if g['away_score'] > g['home_score']:
                home_wins += 1
            elif g['home_score'] == g['away_score']:
                draws += 1

    n_games = len(h2h)
    return {
        'h2h_home_win_rate': home_wins / n_games,
        'h2h_draw_rate': draws / n_games,
        'h2h_n': n_games,
    }


def _implied_probs(row) -> tuple[float, float, float]:
    """Pinnacle 배당 → 마진 제거된 내재 확률."""
    ph = float(row.get('home_odds') or 0)
    pd_ = float(row.get('draw_odds') or 0)
    pa = float(row.get('away_odds') or 0)

    if ph <= 0 or pd_ <= 0 or pa <= 0:
        return 1/3, 1/3, 1/3

    raw_h, raw_d, raw_a = 1/ph, 1/pd_, 1/pa
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from match DataFrame.
    Required columns: home_team, away_team, date, home_score, away_score
    home_odds/draw_odds/away_odds: Pinnacle 배당 (있으면 사용)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['result'] = df.apply(
        lambda r: 'home' if r['home_score'] > r['away_score']
                  else ('draw' if r['home_score'] == r['away_score'] else 'away'),
        axis=1
    )

    features = []
    for idx, row in df.iterrows():
        past = df[df['date'] < row['date']]

        h = _get_team_stats(past, row['home_team'])
        a = _get_team_stats(past, row['away_team'])
        home_wr = _get_home_win_rate(past, row['home_team'])
        h2h = _get_h2h_stats(past, row['home_team'], row['away_team'])
        imp_home, imp_draw, imp_away = _implied_probs(row)

        features.append({
            # 홈팀 폼
            'home_win_rate': h['win_rate'],
            'home_draw_rate': h['draw_rate'],
            'home_goals_scored': h['goals_scored'],
            'home_goals_conceded': h['goals_conceded'],
            'home_form': h['form'],
            # 원정팀 폼
            'away_win_rate': a['win_rate'],
            'away_draw_rate': a['draw_rate'],
            'away_goals_scored': a['goals_scored'],
            'away_goals_conceded': a['goals_conceded'],
            'away_form': a['form'],
            # 홈 어드밴티지
            'home_advantage': home_wr,
            # 상대 비교
            'form_diff': h['form'] - a['form'],
            'goal_diff': h['goals_scored'] - a['goals_scored'],
            'goals_conceded_diff': a['goals_conceded'] - h['goals_conceded'],
            # H2H
            'h2h_home_win_rate': h2h['h2h_home_win_rate'],
            'h2h_draw_rate': h2h['h2h_draw_rate'],
            # 배당 내재 확률 (마진 제거)
            'imp_home': imp_home,
            'imp_draw': imp_draw,
            'imp_away': imp_away,
            # 데이터 수
            'home_n': h['n'],
            'away_n': a['n'],
        })

    feat_df = pd.DataFrame(features, index=df.index)
    df = pd.concat([df, feat_df], axis=1)
    return df
