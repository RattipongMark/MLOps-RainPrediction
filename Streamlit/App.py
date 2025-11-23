import streamlit as st
from tab import Dashboard
from tab import Prediction

st.set_page_config(
    page_title="Rain Forecast App",
    # layout="wide"
)

st.title("🌦️ Rain Forecast")
tab1, tab2 = st.tabs(["Dashboard", "Prediction"])

with tab1:
    Dashboard.render()  # เรียกฟังก์ชันจาก dashboard.py

with tab2:
    Prediction.render()  # เรียกฟังก์ชันจาก prediction.py