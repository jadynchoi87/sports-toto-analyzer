# src/features.py
import pandas as pd
import numpy as np


def _get_team_stats(past: pd.DataFrame, team: str, n: int = 10) -> dict:
    """팀의 최근 N경기 통계 (홈/원정 구분 없이)."""
    games = past[(past['home_team'] == team) | (past['away_team'] == team)].tail(n)
    if len(games) == 0:
        return {'win_rate': 0.0, 'draw_rate': 0.0, 'goals_scored': 0.0, 'goals_conceded': 0.0, 'form': 0.0, 'n': 0}

    wins, draws, goals_scored, goals_conceded, points = 0, 0, 0, 0, 0
    for _, g in games.iterrows():
        is_home = g['home_team'] == team
        if is_home:
            gs, gc = g['home_score'], g['away_score']
        else:
            gs, gc = g['away_score'], g['home_score']

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
        'form': points / (n_games * 3),  # 0~1 정규화
        'n': n_games,
    }


def _get_home_win_rate(past: pd.DataFrame, team: str) -> float:
    """팀의 홈 경기 승률."""
    home_games = past[past['home_team'] == team].tail(10)
    if len(home_games) == 0:
        return 0.0
    wins = (home_games['home_score'] > home_games['away_score']).sum()
    return wins / len(home_games)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from match DataFrame.
    Required columns: home_team, away_team, date, home_score, away_score
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

        # 배당 내재 확률 (북메이커 정보)
        ho = float(row['home_odds']) if pd.notna(row.get('home_odds')) and row.get('home_odds') else 0
        do = float(row['draw_odds']) if pd.notna(row.get('draw_odds')) and row.get('draw_odds') else 0
        ao = float(row['away_odds']) if pd.notna(row.get('away_odds')) and row.get('away_odds') else 0
        total_imp = (1/ho if ho > 0 else 0) + (1/do if do > 0 else 0) + (1/ao if ao > 0 else 0)
        imp_home = (1/ho / total_imp) if ho > 0 and total_imp > 0 else 1/3
        imp_draw = (1/do / total_imp) if do > 0 and total_imp > 0 else 1/3
        imp_away = (1/ao / total_imp) if ao > 0 and total_imp > 0 else 1/3

        features.append({
            # 홈팀 피처
            'home_win_rate': h['win_rate'],
            'home_draw_rate': h['draw_rate'],
            'home_goals_scored': h['goals_scored'],
            'home_goals_conceded': h['goals_conceded'],
            'home_form': h['form'],
            # 원정팀 피처
            'away_win_rate': a['win_rate'],
            'away_draw_rate': a['draw_rate'],
            'away_goals_scored': a['goals_scored'],
            'away_goals_conceded': a['goals_conceded'],
            'away_form': a['form'],
            # 홈 어드밴티지 (실제 홈 승률)
            'home_advantage': home_wr,
            # 상대 비교
            'form_diff': h['form'] - a['form'],
            'goal_diff': h['goals_scored'] - a['goals_scored'],
            # 배당 내재 확률 (핵심 피처)
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
