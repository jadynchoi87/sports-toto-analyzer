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
    st.header("💰 배팅 추천")

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

    st.markdown("### ⚙️ 내 설정")
    col1, col2 = st.columns(2)
    bankroll = col1.number_input("내 배팅 자금 (원)", min_value=10000, value=100000, step=10000,
                                  help="전체 배팅에 쓸 총 자금")
    days_ahead = col2.slider("며칠 내 경기 볼까요?", 1, 14, 7, key="strategy_days")

    safety = st.radio(
        "배팅 스타일",
        ["🛡️ 안전하게 (추천)", "⚡ 공격적으로"],
        horizontal=True,
        help="안전하게: 적게 걸고 오래 즐기기 / 공격적으로: 많이 걸고 수익 극대화 (리스크 높음)"
    )
    half = safety == "🛡️ 안전하게 (추천)"

    min_prob = st.slider("이 확률 이상만 보기", 0.50, 0.80, 0.60, 0.01,
                          help="숫자가 높을수록 AI가 더 확신하는 경기만 보여줍니다")

    with st.spinner("예정 경기 분석 중..."):
        upcoming = _get_upcoming(days_ahead)

    if not upcoming:
        st.info("예정된 경기가 없습니다.")
        return

    df_upcoming = pd.DataFrame(upcoming)
    df_upcoming['home_score'] = 0
    df_upcoming['away_score'] = 0

    with st.spinner("AI 분석 중..."):
        upcoming_features = build_upcoming_features(df_hist, df_upcoming)

        outcome_kor = {'home': '홈팀 승', 'draw': '무승부', 'away': '원정팀 승'}
        rows = []

        for i, (_, row) in enumerate(upcoming_features.iterrows()):
            match_info = upcoming[i]
            probs = predict_proba(model, le, row)
            best_outcome = max(probs, key=probs.get)
            best_prob = probs[best_outcome]

            if best_prob < min_prob:
                continue

            # 이 배당 이상이면 수익 기대 가능
            min_odds = round(1 / best_prob, 2)
            kelly_f = calc_kelly(best_prob, min_odds, half=half)
            bet_amount = int(bankroll * kelly_f)

            rows.append({
                '날짜': match_info['date'],
                '리그': match_info['competition'],
                '홈팀': match_info['home_team'],
                '원정팀': match_info['away_team'],
                '▶ 배팅 추천': outcome_kor.get(best_outcome, best_outcome),
                'AI 확신도': f"{best_prob:.0%}",
                '이 배당 이상이면 배팅 ✅': min_odds,
                '추천 배팅 금액': f"{bet_amount:,}원",
            })

    if not rows:
        st.warning(f"AI 확신도 {min_prob:.0%} 이상 경기가 없어요. 슬라이더를 낮춰보세요.")
        return

    st.markdown(f"### 📋 배팅 추천 경기 ({len(rows)}경기)")
    st.info("💡 **사용법**: 토토사이트에서 아래 경기의 배당을 확인하고, '이 배당 이상이면 배팅' 숫자보다 높으면 배팅하세요.")

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # 직접 계산기
    with st.expander("🔢 직접 계산해보기 (선택)"):
        st.caption("배팅 전 최종 확인용")
        c1, c2 = st.columns(2)
        prob_m = c1.number_input("AI 확신도 (직접 입력)", 0.01, 1.0, 0.60, 0.01)
        odds_m = c2.number_input("실제 배당률", 1.01, 20.0, 2.0, 0.1)

        ev = calc_ev(prob_m, odds_m)
        kelly_f2 = calc_kelly(prob_m, odds_m, half=half)
        bet2 = int(bankroll * kelly_f2)

        if ev > 0 and prob_m >= min_prob:
            st.success(f"✅ 배팅 추천! → {bet2:,}원 거세요")
        else:
            st.error("❌ 이 조건은 배팅하지 마세요 (기대 손실)")
