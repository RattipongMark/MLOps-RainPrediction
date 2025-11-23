import streamlit as st
from tab import Dashboard
from tab import Prediction
import requests

st.set_page_config(
    page_title="Rain Forecast App",
    # layout="wide"
)

st.title("🌦️ Rain Forecast")
tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction", "Admin"])

with tab1:
    Dashboard.render()  # เรียกฟังก์ชันจาก dashboard.py

with tab2:
    Prediction.render()  # เรียกฟังก์ชันจาก prediction.py

with tab3:
    st.set_page_config(page_title="ML Ops Dashboard", layout="wide")
    st.title("🔧 ML Ops Dashboard")

    # URLs สำหรับแต่ละ card
    mlflow_url = "https://dagshub.com/RattipongMark/MLOps-RainPrediction"  # ตัวอย่าง MLflow UI
    airflow_url = "http://34.143.237.72:8080/"  # ตัวอย่าง Airflow UI
    retrain_url = "http://34.143.237.72:5001/trigger_dag"  # API retrain

    # สร้าง 3 columns
    col1, col2, col3 = st.columns(3)

    # ---------------- COLUMN 1: MLflow ----------------
    with col1:
        st.markdown("""
        <div style='
            background-color:#1f77b4;
            padding:20px;
            border-radius:10px;
            text-align:center;
            height:150px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
        '>
            <h3 style='color:white;'>MLFLOW</h3>
            <form action='{}' method='post' target='_blank'>
                <button style='
                    padding:10px 20px;
                    background-color:white;
                    color:#1f77b4;
                    border:none;
                    border-radius:5px;
                    font-weight:bold;
                    cursor:pointer;
                '>VISIT</button>
            </form>
        </div>
        """.format(mlflow_url), unsafe_allow_html=True)

    # ---------------- COLUMN 2: Airflow ----------------
    with col2:
                st.markdown("""
        <div style='
            background-color:#ff7f0e;
            padding:20px;
            border-radius:10px;
            text-align:center;
            height:150px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
        '>
            <h3 style='color:white;'>AIRFLOW</h3>
            <form action='{}' method='post' target='_blank'>
                <button style='
                    padding:10px 20px;
                    background-color:white;
                    color:#ff7f0e;
                    border:none;
                    border-radius:5px;
                    font-weight:bold;
                    cursor:pointer;
                '>VISIT</button>
            </form>
        </div>
        """.format(airflow_url), unsafe_allow_html=True)
       
    # ---------------- COLUMN 3: Retrain ----------------
    with col3:
        st.markdown(f"""
        <div style='
            background-color:#2ca02c;
            padding:20px;
            border-radius:10px;
            text-align:center;
            height:150px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
        '>
            <h3 style='color:white;'>Retrain</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Trigger Retrain", key="retrain"):
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