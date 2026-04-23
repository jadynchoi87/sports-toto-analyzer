import streamlit as st
import pandas as pd
from datetime import datetime
from src.db import get_matches
from src.model import load_model, predict_proba
from src.features import build_features
from src.strategy import best_candidate, apply_filter


def render(db_path: str):
    st.header("예측")

    matches = get_matches(db_path)
    if not matches:
        st.warning("데이터가 없습니다. 먼저 데이터를 수집하세요.")
        return

    df = pd.DataFrame(matches)
    last_date = df["date"].max()
    days_since = (datetime.now().date() - datetime.strptime(last_date, "%Y-%m-%d").date()).days
    if days_since > 7:
        st.error(f"⚠️ 데이터가 {days_since}일 전 기준입니다. 수집이 필요합니다.")
    else:
        st.success(f"데이터 최신화: {last_date}")

    model = load_model()
    if model is None:
        st.info("모델 없음 — 데이터 수집 후 모델을 학습시켜주세요.")
        st.subheader("최근 경기 데이터")
        display_cols = ['date', 'competition', 'home_team', 'away_team', 'home_score', 'away_score']
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].head(20), use_container_width=True)
        st.caption(f"총 {len(df)}경기 수집됨")
        return

    features = build_features(df)
    recommendations = []
    for _, row in features.iterrows():
        probs = predict_proba(model, row)
        odds = {
            "home": row.get("home_odds") or 2.0,
            "draw": row.get("draw_odds") or 3.5,
            "away": row.get("away_odds") or 3.0,
        }
        best = best_candidate(probs, odds)
        if best:
            best["match_id"] = row.get("match_id", "")
            best["odds_moved_same"] = True
            recommendations.append(best)

    qualified = apply_filter(recommendations)
    st.subheader(f"추천 배팅 슬립 ({len(qualified)}경기)")
    if qualified:
        st.dataframe(pd.DataFrame(qualified)[["match_id", "outcome", "prob", "ev", "odds"]])
    else:
        st.info("조건을 충족하는 경기가 없습니다.")

    st.subheader("최근 경기 데이터")
    display_cols = ['date', 'competition', 'home_team', 'away_team', 'home_score', 'away_score']
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].head(20), use_container_width=True)
    st.caption(f"총 {len(df)}경기 수집됨")
