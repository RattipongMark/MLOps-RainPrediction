import streamlit as st
import plotly.express as px
from src.utils import fetch_open_meteo
import random

def render():
    st.set_page_config(layout="wide")
    st.title("📊 Dashboard")

    df = fetch_open_meteo()

    # เลือกเฉพาะ numeric columns สำหรับ plot
    columns_to_plot = df.select_dtypes(include='number').columns.tolist()

    # ใช้ palette สวยๆ ของ Plotly
    color_palette = px.colors.qualitative.Plotly  # หรือ Set2, D3, Pastel1, ฯลฯ

    st.subheader("🌧️ Trend 7 Days")

    # Plot 3 column ต่อแถว
    for i in range(0, len(columns_to_plot), 3):
        cols = st.columns(3)
        for j, col_name in enumerate(columns_to_plot[i:i+3]):
            with cols[j]:
                # สุ่มสีจาก palette
                color = random.choice(color_palette)
                
                fig = px.line(
                    df, x="date", y=col_name, markers=True,
                    title=col_name,
                    labels={col_name: col_name, "date": "Date"},
                    color_discrete_sequence=[color]
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=10, r=10, t=50, b=20),
                    xaxis=dict(showgrid=True, gridcolor="#444"),
                    yaxis=dict(showgrid=True, gridcolor="#444"),
                    title=dict(font=dict(size=16))
                )
                fig.update_traces(line=dict(shape='spline', width=3), marker=dict(size=6))
                st.plotly_chart(fig, use_container_width=True)


    st.subheader("📅 Detail (7 Days)")
    st.dataframe(df, use_container_width=True)
