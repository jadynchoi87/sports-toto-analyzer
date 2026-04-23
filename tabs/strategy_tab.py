import streamlit as st
import pandas as pd
from src.db import get_matches
from src.features import build_features
from src.model import load_model, predict_proba
from src.strategy import best_candidate, apply_filter, calc_kelly, calc_ev


def render(db_path: str):
    st.header("전략")

    bankroll = st.number_input("보유 자금 (원)", min_value=10000, value=100000, step=10000)
    kelly_mode = st.radio("켈리 모드", ["하프 켈리 (안전)", "풀 켈리"])
    half = kelly_mode == "하프 켈리 (안전)"

    model = load_model()
    matches = get_matches(db_path)
    if model and matches:
        df = pd.DataFrame(matches)
        features = build_features(df)
        rows = []
        for _, row in features.tail(20).iterrows():
            probs = predict_proba(model, row)
            odds = {
                "home": row.get("home_odds") or 2.0,
                "draw": row.get("draw_odds") or 3.5,
                "away": row.get("away_odds") or 3.0,
            }
            best = best_candidate(probs, odds)
            if best and best["ev"] > 0:
                best["match_id"] = row.get("match_id", "")
                best["bet_amount"] = bankroll * calc_kelly(best["prob"], best["odds"], half=half)
                best["odds_moved_same"] = True
                rows.append(best)

        qualified = apply_filter(rows)
        if qualified:
            st.subheader(f"경기별 추천 배팅 금액 ({len(qualified)}경기)")
            st.dataframe(
                pd.DataFrame(qualified)[["match_id", "outcome", "prob", "ev", "odds", "bet_amount"]]
                .rename(columns={"bet_amount": "배팅 금액(원)"})
            )
        else:
            st.info("조건 충족 경기 없음")

    st.divider()
    st.subheader("수동 계산기")
    col1, col2 = st.columns(2)
    prob = col1.number_input("예측 확률", 0.0, 1.0, 0.5, 0.01)
    odds_val = col2.number_input("배당률", 1.01, 20.0, 2.0, 0.1)
    ev = calc_ev(prob, odds_val)
    kelly_f = calc_kelly(prob, odds_val, half=half)
    st.metric("기대값 (EV)", f"{ev:.1%}")
    st.metric("추천 배팅 금액", f"{bankroll * kelly_f:,.0f}원")
    if ev < 0:
        st.error("EV 음수 — 배팅 비추천")
