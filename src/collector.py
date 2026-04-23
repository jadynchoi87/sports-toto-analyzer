import requests
import os
from datetime import datetime, timedelta
from src.db import init_db, insert_match

API_BASE = "https://api.football-data.org/v4"
COMPETITIONS = ["PL", "BL1", "PD", "SA", "FL1"]  # 5대 리그


def fetch_finished_matches(competition: str, days_back: int = 30) -> list[dict]:
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")
    resp = requests.get(
        f"{API_BASE}/competitions/{competition}/matches",
        params={"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to},
        headers={"X-Auth-Token": api_key},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def normalize_match(raw: dict, competition: str) -> dict | None:
    score = raw["score"]["fullTime"]
    if score["home"] is None or score["away"] is None:
        return None
    return {
        "match_id": str(raw["id"]),
        "date": raw["utcDate"][:10],
        "home_team": raw["homeTeam"]["name"],
        "away_team": raw["awayTeam"]["name"],
        "home_score": score["home"],
        "away_score": score["away"],
        "home_odds": None,
        "draw_odds": None,
        "away_odds": None,
        "competition": competition
    }


def collect_all(db_path: str = "data/toto.db", days_back: int = 90):
    init_db(db_path)
    total = 0
    for comp in COMPETITIONS:
        try:
            matches = fetch_finished_matches(comp, days_back)
            for m in matches:
                normalized = normalize_match(m, comp)
                if normalized is not None:
                    insert_match(db_path, normalized)
            total += len(matches)
            print(f"{comp}: {len(matches)}경기 수집")
        except Exception as e:
            print(f"{comp} 수집 실패: {e}")
    print(f"총 {total}경기 수집 완료")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    collect_all(days_back=365)
