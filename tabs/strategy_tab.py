import streamlit as st
import pandas as pd
from src.db import get_matches
from src.features import build_upcoming_features
from src.model import load_model, predict_proba
from src.strategy import calc_kelly, calc_ev
from src.upcoming import fetch_upcoming_matches


@st.cache_data(ttl=3600)
def _get_hist(db_path):
    return get_matches(db_path)


@st.cache_data(ttl=3600)
def _get_upcoming(days_ahead):
    return fetch_upcoming_matches(days_ahead=days_ahead)


@st.cache_resource
def _get_model():
    return load_model()


def render(db_path: str):
    st.header("전략")

    try:
        model, le = _get_model()
    except Exception:
        st.warning("모델 없음 — 먼저 모델을 학습시켜주세요.")
        return

    matches = _get_hist(db_path)
    if not matches:
        st.warning("과거 데이터 없음")
        return

    df_hist = pd.DataFrame(matches)

    # 설정
    col1, col2, col3 = st.columns(3)
    bankroll = col1.number_input("보유 자금 (원)", min_value=10000, value=100000, step=10000)
    min_prob = col2.slider("최소 신뢰도", 0.50, 0.80, 0.60, 0.01)
    days_ahead = col3.slider("예측 기간 (일)", 1, 14, 7, key="strategy_days")
    kelly_mode = st.radio("켈리 모드", ["하프 켈리 (안전)", "풀 켈리"], horizontal=True)
    half = kelly_mode == "하프 켈리 (안전)"

    with st.spinner("예정 경기 불러오는 중..."):
        upcoming = _get_upcoming(days_ahead)

    if not upcoming:
        st.info("예정된 경기가 없습니다.")
        return

    df_upcoming = pd.DataFrame(upcoming)
    df_upcoming['home_score'] = 0
    df_upcoming['away_score'] = 0

    with st.spinner("확률 계산 중..."):
        upcoming_features = build_upcoming_features(df_hist, df_upcoming)

        outcome_kor = {'home': '홈 승', 'draw': '무승부', 'away': '원정 승'}
        rows = []

        for i, (_, row) in enumerate(upcoming_features.iterrows()):
            match_info = upcoming[i]
            probs = predict_proba(model, le, row)
            best_outcome = max(probs, key=probs.get)
            best_prob = probs[best_outcome]

            if best_prob < min_prob:
                continue

            odds_min_ev0 = round(1 / best_prob, 2)
            odds_min_ev5 = round(1.05 / best_prob, 2)
            kelly_f = calc_kelly(best_prob, odds_min_ev0, half=half)
            bet_amount = bankroll * kelly_f

            rows.append({
                '날짜': match_info['date'],
                '리그': match_info['competition'],
                '홈팀': match_info['home_team'],
                '원정팀': match_info['away_team'],
                '배팅 추천': outcome_kor.get(best_outcome, best_outcome),
                '신뢰도': best_prob,
                '최소 배당 (±EV)': odds_min_ev0,
                '최소 배당 (EV≥5%)': odds_min_ev5,
                '추천 배팅액(원)': int(bet_amount),
            })

    if not rows:
        st.warning(f"신뢰도 {min_prob:.0%} 이상 경기 없음 — 슬라이더를 낮춰보세요.")
        return

    df_result = pd.DataFrame(rows)
    st.subheader(f"배팅 추천 ({len(df_result)}경기)")
    st.caption("'최소 배당' 이상 배당을 찾으면 배팅하세요. 배팅 사이트에서 실제 배당 확인 필수.")
    st.dataframe(
        df_result.style.format({'신뢰도': '{:.1%}', '추천 배팅액(원)': '{:,}'}),
        use_container_width=True
    )

    st.divider()
    st.subheader("EV / Kelly 계산기")
    st.caption("실제 배당을 찾았을 때 검증용")
    c1, c2 = st.columns(2)
    prob_m = c1.number_input("예측 확률", 0.0, 1.0, 0.60, 0.01)
    odds_m = c2.number_input("실제 배당률", 1.01, 20.0, 2.0, 0.1)

    ev = calc_ev(prob_m, odds_m)
    kelly_f2 = calc_kelly(prob_m, odds_m, half=half)

    m1, m2, m3 = st.columns(3)
    m1.metric("기대값 (EV)", f"{ev:.3f}")
    m2.metric("켈리 비율", f"{kelly_f2:.3f}")
    m3.metric("추천 배팅 금액", f"{bankroll * kelly_f2:,.0f}원")

    if ev >= 0.05 and prob_m >= min_prob:
        st.success("✅ 베팅 조건 충족 — 배팅 가능!")
    else:
        st.error("❌ 조건 미충족 — 배팅 비추천")
