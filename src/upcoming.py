# src/upcoming.py
import os
import requests
from datetime import datetime, timedelta

API_BASE = "https://api.football-data.org/v4"
COMPETITIONS = ["PL", "BL1", "PD", "SA", "FL1"]
COMPETITION_NAMES = {
    "PL": "프리미어리그", "BL1": "분데스리가",
    "PD": "라리가", "SA": "세리에A", "FL1": "리그앙"
}


def fetch_upcoming_matches(days_ahead: int = 7) -> list[dict]:
    """다음 N일 이내 예정 경기 수집."""
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    upcoming = []
    for comp in COMPETITIONS:
        try:
            resp = requests.get(
                f"{API_BASE}/competitions/{comp}/matches",
                params={"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to},
                headers={"X-Auth-Token": api_key},
                timeout=10
            )
            if resp.status_code != 200:
                continue
            matches = resp.json().get("matches", [])
            for m in matches:
                upcoming.append({
                    "match_id": str(m["id"]),
                    "date": m["utcDate"][:10],
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "competition": COMPETITION_NAMES.get(comp, comp),
                    "home_odds": None,
                    "draw_odds": None,
                    "away_odds": None,
                })
        except Exception:
            continue

    return upcoming
