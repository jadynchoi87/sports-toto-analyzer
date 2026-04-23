import requests
import io
import os
import pandas as pd
from datetime import datetime, timedelta
from src.db import init_db, insert_match

API_BASE = "https://api.football-data.org/v4"
COMPETITIONS = ["PL", "BL1", "PD", "SA", "FL1"]

# football-data.co.uk CSV 리그 코드 매핑
CSV_LEAGUES = {
    "E0": "PL",   # Premier League
    "D1": "BL1",  # Bundesliga
    "SP1": "PD",  # La Liga
    "I1": "SA",   # Serie A
    "F1": "FL1",  # Ligue 1
}
CSV_BASE = "https://www.football-data.co.uk/mmz4281"


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


def _season_codes(num_seasons: int = 3) -> list[str]:
    """최근 N 시즌 코드 생성 (예: '2425', '2324', '2223')."""
    current_year = datetime.now().year
    current_month = datetime.now().month
    # 7월 이후면 새 시즌 시작
    start_year = current_year if current_month >= 7 else current_year - 1
    codes = []
    for i in range(num_seasons):
        y = start_year - i
        codes.append(f"{str(y)[2:]}{str(y+1)[2:]}")
    return codes


def collect_odds_csv(db_path: str = "data/toto.db", num_seasons: int = 3):
    """
    football-data.co.uk에서 CSV로 배당 포함 경기 데이터 수집.
    기존 DB의 home_odds/draw_odds/away_odds를 업데이트하거나 새 레코드 삽입.
    """
    init_db(db_path)
    seasons = _season_codes(num_seasons)
    total = 0

    for league_code, competition in CSV_LEAGUES.items():
        for season in seasons:
            url = f"{CSV_BASE}/{season}/{league_code}.csv"
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                df = pd.read_csv(
                    io.StringIO(resp.content.decode('utf-8', errors='ignore')),
                    on_bad_lines='skip'
                )

                # 필수 컬럼 확인
                required = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                if not all(c in df.columns for c in required):
                    continue

                df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])

                for _, row in df.iterrows():
                    try:
                        # 날짜 파싱 (DD/MM/YYYY 형식)
                        date_str = pd.to_datetime(row['Date'], dayfirst=True).strftime('%Y-%m-%d')
                        match_id = f"csv_{league_code}_{season}_{row['HomeTeam']}_{row['AwayTeam']}_{date_str}"
                        match_id = match_id.replace(' ', '_')

                        match = {
                            "match_id": match_id,
                            "date": date_str,
                            "home_team": str(row['HomeTeam']),
                            "away_team": str(row['AwayTeam']),
                            "home_score": int(row['FTHG']),
                            "away_score": int(row['FTAG']),
                            "home_odds": float(row['B365H']) if 'B365H' in row and pd.notna(row['B365H']) else None,
                            "draw_odds": float(row['B365D']) if 'B365D' in row and pd.notna(row['B365D']) else None,
                            "away_odds": float(row['B365A']) if 'B365A' in row and pd.notna(row['B365A']) else None,
                            "competition": competition,
                        }
                        insert_match(db_path, match)
                        total += 1
                    except Exception:
                        continue

                print(f"{competition} {season}: {len(df)}경기 수집")
            except Exception as e:
                print(f"{competition} {season} 실패: {e}")

    print(f"CSV 총 {total}경기 수집 완료 (배당 포함)")


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
    collect_odds_csv(num_seasons=3)
