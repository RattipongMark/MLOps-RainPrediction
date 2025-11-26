from tab import Dashboard
from tab import Prediction
import requests
import streamlit as st
import requests

st.set_page_config(
    page_title="Rain Prediction App",
    # layout="wide"
)
mlflow_url = "http://localhost:5000"
airflow_url = "http://localhost:8080"
retrain_url = "http://34.143.237.72:5001/trigger_dag"

        
st.title("🌦️ Rain Prediction")
tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction", "Admin"])



with tab1:
    Dashboard.render()  # เรียกฟังก์ชันจาก dashboard.py

with tab2:
    Prediction.render()  # เรียกฟังก์ชันจาก prediction.py

with tab3:
    left, middle, right = st.columns(3)

    mlflow_url = "https://dagshub.com/RattipongMark/MLOps-RainPrediction"
    airflow_url = "http://34.143.237.72:8080/"
    retrain_url = "http://34.143.237.72:5001/trigger_dag"

    # ----- LEFT: MLFLOW -----
    if left.button("MLFLOW", width="stretch"):
        st.write(f"Open link: {mlflow_url}")

    # ----- MIDDLE: AIRFLOW -----
    if middle.button("AIRFLOW", width="stretch"):
        st.write(f"Open link: {airflow_url}")

    # ----- RIGHT: RETRAIN -----
    if right.button("RETRAIN", width="stretch"):
        try:
            response = requests.post(
                retrain_url,
                headers={"Content-Type": "application/json"},
                json={"conf": {"force_retrain": True}}
            )
            if response.status_code == 200:
                st.success("✅ Retrain triggered successfully!")
            else:
                st.error(f"❌ Failed: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    
