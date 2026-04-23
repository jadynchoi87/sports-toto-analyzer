import streamlit as st
import pandas as pd
from src.db import get_matches
from src.model import load_model, predict_proba
from src.features import build_upcoming_features
from src.strategy import calc_ev, calc_kelly
from src.upcoming import fetch_upcoming_matches


def render(db_path: str):
    st.header("예측")

    # 모델 로드
    try:
        model, le = load_model()
        st.success("모델 로드됨 ✓")
    except Exception:
        st.error("모델 없음 — 모델을 먼저 학습시켜주세요.")
        return

    # 과거 데이터 (피처 계산용)
    matches = get_matches(db_path)
    if not matches:
        st.warning("과거 데이터 없음. 데이터를 먼저 수집하세요.")
        return

    df_hist = pd.DataFrame(matches)

    # 예정 경기 가져오기
    days_ahead = st.slider("예측 기간 (일)", 1, 14, 7)

    with st.spinner("예정 경기 불러오는 중..."):
        upcoming = fetch_upcoming_matches(days_ahead=days_ahead)

    if not upcoming:
        st.info("예정된 경기가 없습니다.")
        # 대신 최근 경기 데이터 표시
        st.subheader("최근 수집된 경기")
        display_cols = ['date', 'competition', 'home_team', 'away_team', 'home_score', 'away_score']
        available = [c for c in display_cols if c in df_hist.columns]
        st.dataframe(df_hist[available].head(20), use_container_width=True)
        return

    st.subheader(f"예정 경기 ({len(upcoming)}경기) 예측")

    # 예정 경기 피처 계산 (과거 전체를 past로 사용 — 빠른 버전)
    df_upcoming = pd.DataFrame(upcoming)
    df_upcoming['home_score'] = 0
    df_upcoming['away_score'] = 0

    with st.spinner("예측 계산 중..."):
        upcoming_features = build_upcoming_features(df_hist, df_upcoming)

        results = []
        for i, (_, row) in enumerate(upcoming_features.iterrows()):
            match_info = upcoming[i]
            probs = predict_proba(model, le, row)

            # 배당이 없으면 확률만 표시
            best_outcome = max(probs, key=probs.get)
            best_prob = probs[best_outcome]

            outcome_kor = {'home': '홈 승', 'draw': '무승부', 'away': '원정 승'}

            results.append({
                '날짜': match_info['date'],
                '리그': match_info['competition'],
                '홈팀': match_info['home_team'],
                '원정팀': match_info['away_team'],
                '예측': outcome_kor.get(best_outcome, best_outcome),
                '홈 승': f"{probs.get('home', 0):.1%}",
                '무승부': f"{probs.get('draw', 0):.1%}",
                '원정 승': f"{probs.get('away', 0):.1%}",
                '신뢰도': f"{best_prob:.1%}",
            })

    result_df = pd.DataFrame(results)
    st.dataframe(result_df, use_container_width=True)
    st.caption("※ 배당 데이터 없이 모델 확률만 표시. EV 계산은 실제 배당 입력 필요.")
