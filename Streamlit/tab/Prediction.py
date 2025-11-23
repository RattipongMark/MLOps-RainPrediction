# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import date
from src.utils import fetch_open_meteo, fetch_open_meteo_hourly
from src.model_utils import forecast_rain_from_model
import base64
from pathlib import Path
def render():
    # -----------------------
    # PAGE CONFIG
    # -----------------------
    st.set_page_config(page_title="🌦️ Weather Dashboard", layout="wide")

    # -----------------------
    # CACHE FUNCTION
    # -----------------------
    today_str = date.today().isoformat()

    @st.cache_data(show_spinner=False)
    def get_weather_data(cache_date):
        daily_df = fetch_open_meteo()
        hourly_df = fetch_open_meteo_hourly()
        hourly_df = forecast_rain_from_model(hourly_df)
        daily_pred = forecast_rain_from_model(daily_df)
        return daily_df, hourly_df, daily_pred

    daily_df, hourly_df, daily_pred = get_weather_data(today_str)

    # -----------------------
    # STYLE: Font + Background
    # -----------------------
    st.markdown(
        """
        <style>
        body, div, span, h1, h2, h3, p {
            font-family: 'Kanit', sans-serif;
            color:white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------
    # HEADER — LOCATION + TEMP
    # -----------------------
    now = pd.Timestamp.now()
    hourly_df['time'] = pd.to_datetime(hourly_df['time'])
    closest_idx = (hourly_df['time'] - now).abs().idxmin()
    current_temp = hourly_df.loc[closest_idx, 'temperature_2m']
    current_time = now.strftime("%Y-%m-%d %H:%M")
    location_name = "Bangkok, Thailand"

    st.markdown(f"<h1 style='text-align:center;'>{location_name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center;'>{current_time} | {current_temp:.1f} °C</h4>", unsafe_allow_html=True)

    # -----------------------
    # LOAD ICONS
    # -----------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    IMG_DIR = BASE_DIR / "src" / "img"
    cloudy_path = IMG_DIR / "cloudy.png"    
    sun_path = IMG_DIR / "sun.png"
    rain_path = IMG_DIR / "rainy-day.png"
    moon_path = IMG_DIR / "moon.png"

    def img_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    sun_b64 = img_to_base64(sun_path)
    cloudy_b64 = img_to_base64(cloudy_path)
    rain_b64 = img_to_base64(rain_path)
    moon_b64 = img_to_base64(moon_path)

    # -----------------------
    # HOURLY FORECAST — SCROLLABLE
    # -----------------------
    def render_hourly_matrix(df_h, current_time):
        html = """
        <style>
        div.scrollmenu {
            overflow-x: auto;
            white-space: nowrap;
            width: 100%;
            padding: 20px 10px;
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        div.scrollmenu div.item {
            display: inline-block;
            width: 100px;
            text-align: center;
            margin-right: 4px;
            color: white;
        }
        </style>
        """
        html += "<p style='color:white; text-align:left;'>Hourly Forecast</p>"
        html += "<div class='scrollmenu'>"

        sunrise_hour = 6
        sunset_hour = 18

        for _, r in df_h.iterrows():
            hour = r['time'].hour
            style = "font-size:16px; padding:4px;"
            if r['time'].hour == current_time.hour and r['time'].date() == current_time.date():
                style += " background-color: rgba(255,255,255,0.3); border-radius:5px;"
            html += f"<div class='item' style='{style}'>{str(hour).zfill(2)}:00</div>"
        html += "<br>"

        for _, r in df_h.iterrows():
            hour = r['time'].hour
            if hour == sunrise_hour and hour < sunset_hour:
                img = cloudy_b64
            elif hour >= sunset_hour or hour < sunrise_hour:
                img = moon_b64
            else:
                img = sun_b64 if r["predicted_rain"] == 0 else rain_b64
            html += f"<div class='item'><img src='data:image/png;base64,{img}' style='height:50px; width:auto; margin:10px;'></div>"
        html += "<br>"

        for _, r in df_h.iterrows():
            html += f"<div class='item' style='font-size:20px;'>{r['predicted_rain_prob']*100:.0f}%</div>"

        html += "</div>"  
        st.markdown(html, unsafe_allow_html=True)

    hourly_df_filtered = hourly_df[hourly_df['time'] >= now].copy()
    render_hourly_matrix(hourly_df_filtered, now)

    st.markdown("---")

    # -----------------------
    # DAILY FORECAST + MAP
    # -----------------------
    col1, col2 = st.columns([1,2])

    with col1:
        def render_daily_matrix(daily_df):
            html = """
                <style>
                div.daily-container {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    width: 100%;
                    padding: 10px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 10px;
                }
                div.daily-row {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    padding: 10px;
                    border-radius: 8px;
                    min-height: 60px;
                }
                div.daily-row div.day {
                    width: 120px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    text-transform: uppercase;
                }
                div.daily-row div.icon {
                    width: 60px;
                    text-align: center;
                }
                div.daily-row div.prob {
                    width: 100%;
                    height: 25px;
                    background-color: rgba(255,255,255,0.2);
                    border-radius: 6px;
                    overflow: hidden;
                    position: relative;
                }
                div.daily-row div.prob div.fill {
                    height: 100%;
                    width: 100%;
                    background-color: #4caf50;
                }
                div.daily-row div.prob span {
                    position: absolute;
                    right: 4px;
                    top: 3px;
                    color: white;
                    font-size: 12px;
                }
                </style>
                """
            html += "<p style='color:white; font-size:18px;'>📅 Daily Forecast</p>"
            html += "<div class='daily-container'>"

            for _, row in daily_df.iterrows():
                date_obj = pd.to_datetime(row["date"])
                day_text = date_obj.strftime("%a")
                icon_b64 = rain_b64 if row["predicted_rain"] == 1 else sun_b64
                prob = row['predicted_rain_prob'] * 100
                html += f"""
                <div class='daily-row'>
                    <div class='day'>{day_text}</div>
                    <div class='icon'><img src='data:image/png;base64,{icon_b64}' style='height:40px; width:auto;'></div>
                    <div class='prob'>
                        <div class='fill' style='width:{prob}%;'></div>
                        <span>{prob:.0f}%</span>
                    </div>
                        
                </div>
                """
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        render_daily_matrix(daily_df)

    with col2:
        st.markdown("<p style='color:white;'>🌦️ Radar / Satellite</p>", unsafe_allow_html=True)
        rainviewer_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <link href="https://unpkg.com/leaflet/dist/leaflet.css" rel="stylesheet" />
            <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
            <style>
                body, html { margin-top: 3px; height: 100%; }
                #mapid {
                    width: 100%;
                    height: 100%;
                    border-radius: 10px;

                }
            </style>
        </head>
        <body>
            <div id="mapid"></div>
            <script>
                var map = L.map('mapid').setView([13.7, 100.5], 6);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap'
                }).addTo(map);

                fetch('https://api.rainviewer.com/public/weather-maps.json')
                .then(r => r.json())
                .then(data => {
                    const tileHost = data.host;
                    const timestamps = data.radar.past.map(t => t.time);
                    const latest = timestamps[timestamps.length - 1];

                    L.tileLayer(`${tileHost}/v2/radar/${latest}/{z}/{x}/{y}/2/1_1.png`, {
                        tileSize: 256,
                        opacity: 0.7,
                        zIndex: 10
                    }).addTo(map);
                });
            </script>
        </body>
        </html>
        """
        st.components.v1.html(rainviewer_html, height=480)
