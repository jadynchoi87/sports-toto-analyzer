# src/features.py
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from match DataFrame.

    Required columns: home_team, away_team, date, home_score, away_score
    Returns DataFrame with added feature columns.
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

        home_past = past[past['home_team'] == row['home_team']].tail(10)
        away_past = past[past['away_team'] == row['away_team']].tail(10)

        home_wins = (home_past['result'] == 'home').sum()
        home_draws = (home_past['result'] == 'draw').sum()
        home_n = len(home_past)

        away_wins = (away_past['result'] == 'away').sum()
        away_draws = (away_past['result'] == 'draw').sum()
        away_n = len(away_past)

        features.append({
            'home_win_rate': home_wins / home_n if home_n > 0 else 0.0,
            'home_draw_rate': home_draws / home_n if home_n > 0 else 0.0,
            'away_win_rate': away_wins / away_n if away_n > 0 else 0.0,
            'away_draw_rate': away_draws / away_n if away_n > 0 else 0.0,
            'home_advantage': 1.0,
            'home_n': home_n,
            'away_n': away_n,
        })

    feat_df = pd.DataFrame(features, index=df.index)
    df = pd.concat([df, feat_df], axis=1)
    return df
