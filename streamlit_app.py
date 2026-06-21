import os
import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

# --- CONFIGURATION ---
MODEL_PATH = "models/rf_model_unified.joblib"
MODEL_NAME = "Aurafarm AI"
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"
BASE_LAT = 18.5847
BASE_LON = 99.0256

st.set_page_config(page_title=MODEL_NAME, layout="wide")

# Custom Dark Theme
st.markdown(
    """
    <style>
    .stApp { background-color: #1e1e1e; color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try: return joblib.load(MODEL_PATH)
        except Exception: return None
    return None

my_model = load_model()

# --- 2. DATA LOADING ---
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        column_mapping = {"Unnamed: 0": "Timestamp", "tvoc": "TVOC", "eco2": "eCO2", "temp": "Temp", "humidity": "Humidity", "ch0": "CH0", "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10"}
        df = df.rename(columns=column_mapping)
        if "Timestamp" in df.columns:
            df["Display_Time"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df["Sort_Time"] = df["Display_Time"] + pd.Timedelta(hours=7)
            df = df.sort_values(by="Sort_Time", ascending=False)
        return df
    except Exception: return pd.DataFrame()

df = load_sensor_data()
latest = df.iloc[0].to_frame().T if not df.empty else None

# --- 3. UI & PREDICTIONS ---
st.title(f"{MODEL_NAME} Dashboard")
prediction = None
mapping_dict = {"TVOC": "col_2", "eCO2": "col_3", "Temp": "col_4", "Humidity": "col_5", "PM2.5": "col_6", "CH0": "col_7", "CH3": "col_8", "MQ135": "col_9"}

if my_model and latest is not None:
    features = latest[["TVOC", "eCO2", "Temp", "Humidity", "PM2.5", "CH0", "CH3", "MQ135"]].rename(columns=mapping_dict)
    prediction = my_model.predict(features)[0]
    if prediction == 1: st.error("VAPE DETECTED")
    else: st.success("AIR QUALITY: Clean")

# --- 4. MAP & STATUS ---
st.subheader("Facility Sensor Network")

live_state = 1 if prediction == 1 else 0

mock_sensors = pd.DataFrame({
    'sensor_id': ['SN-01 (Main Lobby)', 'SN-02 (East Restroom)', 'SN-03 (Breakroom)', 'SN-04 (Stairwell B)'],
    'latitude': [BASE_LAT + 0.0004, BASE_LAT + 0.0004, BASE_LAT - 0.0005, BASE_LAT + 0.0002],
    'longitude': [BASE_LON, BASE_LON - 0.0006, BASE_LON - 0.0002, BASE_LON + 0.0005],
    'vape_detected': [live_state, 1, 0, 0],
    'air_quality': ['Clean', 'Vape Detected', 'Clean', 'Clean']
})

# Colors matching standard Streamlit error (red) and success (green)
mock_sensors["color"] = mock_sensors["vape_detected"].map({1: [255, 75, 75, 255], 0: [0, 204, 102, 255]})

col_map, col_text = st.columns([2, 1])

with col_map:
    layer = pdk.Layer("ScatterplotLayer", data=mock_sensors, get_position=["longitude", "latitude"], 
                      get_fill_color="color", get_radius=10, pickable=True)
    # Using 'dark' style which is built-in and doesn't require a Token
    st.pydeck_chart(pdk.Deck(
        layers=[layer], 
        initial_view_state=pdk.ViewState(latitude=BASE_LAT, longitude=BASE_LON, zoom=16), 
        map_style='dark', 
        tooltip={"text": "{sensor_id}: {air_quality}"}
    ))

with col_text:
    st.write("### Live Node Status")
    for _, row in mock_sensors.iterrows():
        color = "#ff4b4b" if row['vape_detected'] == 1 else "#00cc66"
        st.markdown(f"**{row['sensor_id']}** \n<small style='color:{color};'>● {row['air_quality']}</small>", unsafe_allow_html=True)
