import streamlit as st
import pandas as pd
from src.db import get_predictions
from src.backtest import run_backtest


def render(db_path: str):
    st.header("검증")

    preds = get_predictions(db_path)
    if not preds:
        st.info("아직 예측 기록이 없습니다.")
        return

    df = pd.DataFrame(preds)
    finished = df[df["home_score"].notna()]

    if finished.empty:
        st.info("결과가 확정된 경기가 없습니다.")
        return

    bankroll = st.number_input("기준 자금", value=100000, step=10000)
    metrics = run_backtest(finished.to_dict("records"), bankroll=bankroll)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 예측", metrics["total"])
    col2.metric("적중률", f"{metrics['hit_rate']}%")
    col3.metric("ROI", f"{metrics['roi']}%")
    col4.metric("순이익", f"{metrics['profit']:,.0f}원")

    if "competition" in finished.columns:
        st.subheader("리그별 ROI")
        comp_rows = []
        for comp, grp in finished.groupby("competition"):
            m = run_backtest(grp.to_dict("records"), bankroll=bankroll)
            comp_rows.append({
                "리그": comp,
                "예측수": m["total"],
                "적중률": f"{m['hit_rate']}%",
                "ROI": f"{m['roi']}%",
            })
        st.dataframe(pd.DataFrame(comp_rows))

    st.subheader("예측 내역")
    display_cols = ["date", "home_team", "away_team", "recommended_outcome",
                    "home_score", "away_score", "ev"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].head(50))
