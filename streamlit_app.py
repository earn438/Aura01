import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/rf_model_unified.joblib'
MODEL_NAME = "VapeGuard AI v1.0"
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"

st.set_page_config(page_title=MODEL_NAME, page_icon="🚭", layout="wide")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

my_model = load_model()

# --- 2. DATA LOADING & PROCESSING ---
@st.cache_data(ttl=30)
def load_and_process():
    df = pd.read_csv(SHEET_URL)
    column_mapping = {
        "Unnamed: 0": "Timestamp", "tvoc": "TVOC", "eco2": "eCO2",
        "temp": "Temp", "humidity": "Humidity", "ch0": "CH0",
        "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10"
    }
    df = df.rename(columns=column_mapping)
    df['Display_Time'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
    df['Sort_Time'] = df['Display_Time'] + pd.Timedelta(hours=7)
    df = df.sort_values(by='Sort_Time', ascending=False)
    return df

df = load_and_process()

# --- 3. PREDICTION LOGIC ---
st.title(f"🚭 {MODEL_NAME} Dashboard")
mapping_dict = {
    'TVOC': 'col_2', 'eCO2': 'col_3', 'Temp': 'col_4', 
    'Humidity': 'col_5', 'PM2.5': 'col_6', 'CH0': 'col_7', 
    'CH3': 'col_8', 'MQ135': 'col_9'
}

if my_model:
    all_features = df[['TVOC', 'eCO2', 'Temp', 'Humidity', 'PM2.5', 'CH0', 'CH3', 'MQ135']].rename(columns=mapping_dict)
    df['Vape_Prediction'] = my_model.predict(all_features)
    
    if df.iloc[0]['Vape_Prediction'] == 1:
        st.error("🚨 VAPE DETECTED: AI Model indicates vape particles!")
    else:
        st.success("✅ AIR QUALITY: Clean.")
else:
    st.info("ℹ️ Prediction system offline.")

# --- 4. HISTORY LOG ---
st.subheader("📜 Recent Vape Detection Events")
history = df[(df['Vape_Prediction'] == 1) & (df['Sort_Time'] >= (df['Sort_Time'].max() - pd.Timedelta(days=1)))]
history_display = history[['Display_Time', 'TVOC', 'PM2.5']]

if not history_display.empty:
    st.table(history_display.head(10))
else:
    st.write("No vape events detected in the last 24 hours.")

# --- 5. METRICS & GRAPHS ---
latest = df.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temp", f"{latest['Temp']} °C")
col2.metric("Humidity", f"{latest['Humidity']} %")
col3.metric("TVOC", f"{latest['TVOC']} ppb")
col4.metric("PM 2.5", f"{latest['PM2.5']} μg/m³")

st.divider()
st.subheader("📈 Trends")
chart_data = df.set_index('Sort_Time').sort_index()
st.line_chart(chart_data[['PM2.5', 'TVOC']])
