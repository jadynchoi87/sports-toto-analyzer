# src/db.py
import sqlite3, os

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER,
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL,
    competition TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    match_id TEXT PRIMARY KEY,
    predicted_at TEXT,
    home_prob REAL,
    draw_prob REAL,
    away_prob REAL,
    recommended_outcome TEXT,
    ev REAL,
    kelly_fraction REAL
);
"""

def init_db(db_path: str = "data/toto.db"):
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)

def insert_match(db_path: str, match: dict):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO matches VALUES
            (:match_id,:date,:home_team,:away_team,:home_score,:away_score,
             :home_odds,:draw_odds,:away_odds,:competition)
        """, match)

def get_matches(db_path: str = "data/toto.db") -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM matches ORDER BY date DESC")]

def save_prediction(db_path: str, pred: dict):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO predictions VALUES
            (:match_id,:predicted_at,:home_prob,:draw_prob,:away_prob,
             :recommended_outcome,:ev,:kelly_fraction)
        """, pred)

def get_predictions(db_path: str = "data/toto.db") -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("""
            SELECT p.*, m.date, m.home_team, m.away_team,
                   m.home_score, m.away_score, m.competition
            FROM predictions p JOIN matches m USING(match_id)
            ORDER BY m.date DESC
        """)]
