import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.db import init_db
from tabs import prediction, strategy_tab, verification

DB_PATH = "data/toto.db"

st.set_page_config(page_title="스포츠토토 분석기", layout="wide")
st.title("⚽ 스포츠토토 분석기")

init_db(DB_PATH)

tab1, tab2, tab3 = st.tabs(["📊 예측", "💰 전략", "✅ 검증"])

with tab1:
    prediction.render(DB_PATH)
with tab2:
    strategy_tab.render(DB_PATH)
with tab3:
    verification.render(DB_PATH)
