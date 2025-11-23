import streamlit as st
import plotly.express as px
from src.utils import fetch_open_meteo

# Wide layout ให้กว้างขึ้น
st.set_page_config(layout="wide")

# CSS
st.markdown("""
<style>
.main {
    padding-left: 2rem;
    padding-right: 2rem;
}

/* header */
h3 {
    color: #f0f0f0;
}

/* Chart ให้เต็ม container */
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Rain Forecast Dashboard")

# Fetch API
df = fetch_open_meteo()

# -----------------------
# Trend Charts in 2-column Grid
# -----------------------
st.subheader("🌧️ Trend 7 Days")

label_map = {
    "rain": "Rain (mm)",
    "temperature_2m": "Temperature (°C)",
    "relative_humidity_2m": "Humidity (%)",
    "dew_point_2m": "Dew Point (°C)",
    "cloud_cover_low": "Low Cloud (%)",
    "precipitation_probability": "Precipitation Probability (%)",
    "wind_speed_10m": "Wind Speed (10m, km/h)"
}

columns_to_plot = list(label_map.keys())

for i in range(0, len(columns_to_plot), 2):

    col_left, col_right = st.columns(2)

    # LEFT card
    col = columns_to_plot[i]
    with col_left:
        display_name = label_map[col]
        st.markdown(f"<div class='metric-card'><h3>{display_name}</h3>", unsafe_allow_html=True)

        fig = px.line(
            df,
            x="date",
            y=col,
            markers=True,
            labels={col: display_name, "date": "Date"},
            height=280
        )
        # fig.update_layout(
        #         paper_bgcolor="#1e1e1e",
        #         plot_bgcolor="#1e1e1e",
        #         margin=dict(l=40, r=40, t=40, b=40),
        #         shapes=[
        #             dict(
        #                 type="rect",
        #                 xref="paper",
        #                 yref="paper",
        #                 x0=0, 
        #                 y0=0,
        #                 x1=1,
        #                 y1=1,
        #                 line=dict(color="#555", width=2)
        #             )
        #         ]
        #     )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT card (ถ้ามี)
    if i + 1 < len(columns_to_plot):
        col2 = columns_to_plot[i + 1]
        with col_right:
            display_name = label_map[col2]
            st.markdown(f"<div class='metric-card'><h3>{display_name}</h3>", unsafe_allow_html=True)

            fig2 = px.line(
                df,
                x="date",
                y=col2,
                markers=True,
                labels={col2: display_name, "date": "Date"},
                height=280
            )
            # fig2.update_layout(
            #     paper_bgcolor="#1e1e1e",
            #     plot_bgcolor="#1e1e1e",
            #     margin=dict(l=40, r=40, t=40, b=40),
            #     shapes=[
            #         dict(
            #             type="rect",
            #             xref="paper",
            #             yref="paper",
            #             x0=0, 
            #             y0=0,
            #             x1=1,
            #             y1=1,
            #             line=dict(color="#555", width=2)
            #         )
            #     ]
            # )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# ตารางรายละเอียดรายวัน
# -----------------------
st.subheader("📅 Detail (7 Days)")
st.dataframe(df, use_container_width=True)