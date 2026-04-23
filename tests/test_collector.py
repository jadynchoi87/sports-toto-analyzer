from src.collector import normalize_match

def test_normalize_match_fields():
    raw = {
        "id": 12345,
        "utcDate": "2024-03-15T15:00:00Z",
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
        "score": {"fullTime": {"home": 2, "away": 1}}
    }
    result = normalize_match(raw, "PL")
    assert result["match_id"] == "12345"
    assert result["date"] == "2024-03-15"
    assert result["home_team"] == "Arsenal FC"
    assert result["home_score"] == 2
    assert result["competition"] == "PL"
    assert result["home_odds"] is None

def test_normalize_match_date_trimmed():
    raw = {
        "id": 1, "utcDate": "2024-06-01T00:00:00Z",
        "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
        "score": {"fullTime": {"home": 0, "away": 0}}
    }
    result = normalize_match(raw, "BL1")
    assert len(result["date"]) == 10  # YYYY-MM-DD 형식

def test_normalize_match_null_score_returns_none():
    raw = {
        "id": 999, "utcDate": "2024-06-01T00:00:00Z",
        "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
        "score": {"fullTime": {"home": None, "away": None}}
    }
    result = normalize_match(raw, "PL")
    assert result is None
