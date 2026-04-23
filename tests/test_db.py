# tests/test_db.py
import os, pytest
from src.db import init_db, insert_match, get_matches

def test_insert_and_get(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    insert_match(db_path, {
        "match_id": "1", "date": "2024-01-01",
        "home_team": "팀A", "away_team": "팀B",
        "home_score": 2, "away_score": 1,
        "home_odds": 1.8, "draw_odds": 3.5, "away_odds": 4.0,
        "competition": "PL"
    })
    rows = get_matches(db_path)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "팀A"
