import streamlit as st
from src.utils import fetch_open_meteo, fetch_open_meteo_hourly
from src.model_utils import classify_rain_daily, classify_rain_hourly

st.set_page_config(layout="wide")
st.title("🌦️ 7-Day Rain Prediction")

# -----------------------
# FETCH
# -----------------------
daily_df = fetch_open_meteo()
hourly_df = fetch_open_meteo_hourly()

daily_pred = classify_rain_daily(daily_df)

# --- STYLE ---
HOUR_STYLE = "display:inline-block; width:70px; color:#fff; font-size:14px; text-align:center;"
EMOJI_STYLE = "display:inline-block; width:70px; font-size:26px; text-align:center;"
PROBA_STYLE = "display:inline-block; width:70px; color:#F5C542; font-size:14px; font-weight:bold; text-align:center;"

# Scroll container style
st.markdown("""
    <style>
        .scroll-box {
            overflow-x: auto;
            white-space: nowrap;
            padding: 12px 0px;
            border-bottom: 1px solid #333;
        }
        .row-block {
            white-space: nowrap;
            margin-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Helper: build 3-row horizontal scroll
# -----------------------
def render_hourly_matrix(df_h):
    html = "<div class='scroll-box'>"

    # Row 1 — Hours
    html += "<div class='row-block'>"
    for _, r in df_h.iterrows():
        html += f"<div style='{HOUR_STYLE}'>{str(r['hour']).zfill(2)}:00</div>"
    html += "</div>"

    # Row 2 — Emoji
    html += "<div class='row-block'>"
    for _, r in df_h.iterrows():
        icon_hr = "🌧️" if r["rain_label"] == "Rain" else "☀️"
        html += f"<div style='{EMOJI_STYLE}'>{icon_hr}</div>"
    html += "</div>"

    # Row 3 — Probability
    html += "<div class='row-block'>"
    for _, r in df_h.iterrows():
        html += f"<div style='{PROBA_STYLE}'>{r['rain_proba']*100:.0f}%</div>"
    html += "</div>"

    html += "</div>"
    return html


# -----------------------
# LOOP DAYS
# -----------------------
for idx, row in daily_pred.iterrows():

    date = str(row["date"])
    label = row["rain_label"]
    proba = row["rain_proba"]
    icon = "🌧️" if label == "Rain" else "☀️"

    # ================================
    #  CASE 1 — DAY 1 → Show MATRIX directly
    # ================================
    if idx == 0:
        st.markdown(
            f"<h2 style='color:white;'>📅 Today — {date} — {icon}</h2>",
            unsafe_allow_html=True
        )

        df_day = hourly_df[hourly_df["date"] == date].copy()
        df_h = classify_rain_hourly(df_day)

        st.markdown(render_hourly_matrix(df_h), unsafe_allow_html=True)
        st.markdown("---")
        continue

    # ================================
    #  CASE 2 — DAYS 2–7 → Show daily bar + expander with MATRIX
    # ================================
    color_rain = "linear-gradient(90deg, #9DC2F9, #2573EB)"   # ฟ้า ถ้าฝนตก
    color_clear = "linear-gradient(90deg, #FBCA0A, #F05A28)"  # เดิม ถ้าไม่ตก

    bar_color = color_rain if label == "Rain" else color_clear

    st.markdown(f"""
    <div style='display:flex; align-items:center; margin-bottom:12px; margin-top:20px;'>
        <div style='flex:1; text-align:center; color:white; font-size:16px'>{date}</div>
        <div style='flex:1; text-align:center; font-size:28px'>{icon}</div>
        <div style='flex:6; display:flex; align-items:center;'>
            <div style='flex-grow:1; height:20px; border-radius:10px; background-color:#333; margin-right:10px; position:relative;'>
                <div style='width:{proba*100}%; height:100%; border-radius:10px; background: {bar_color};'></div>
            </div>
            <div style='min-width:45px; text-align:center; color:white; font-weight:bold; font-size:16px'>{proba*100:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # EXPANDER — now shows 3-row matrix
    with st.expander(f"🕒 Hourly Forecast for {date}"):

        df_day = hourly_df[hourly_df["date"] == date]
        df_h = classify_rain_hourly(df_day)

        st.markdown(render_hourly_matrix(df_h), unsafe_allow_html=True)
