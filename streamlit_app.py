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
latest = df.iloc[0].to_frame().T

# --- 3. PREDICTION LOGIC ---
st.title(f"🚭 {MODEL_NAME} Dashboard")

# Mapping dictionary for model inputs
mapping_dict = {
    'TVOC': 'col_2', 'eCO2': 'col_3', 'Temp': 'col_4', 
    'Humidity': 'col_5', 'PM2.5': 'col_6', 'CH0': 'col_7', 
    'CH3': 'col_8', 'MQ135': 'col_9'
}

if my_model:
    # Run prediction for every row to find history
    all_features = df[['TVOC', 'eCO2', 'Temp', 'Humidity', 'PM2.5', 'CH0', 'CH3', 'MQ135']].rename(columns=mapping_dict)
    df['Vape_Prediction'] = my_model.predict(all_features)
    
    # Check latest
    if df.iloc[0]['Vape_Prediction'] == 1:
        st.error("🚨 VAPE DETECTED: AI Model indicates vape particles!")
    else:
        st.success("✅ AIR QUALITY: Clean.")
else:
    st.info("ℹ️ Prediction system offline.")

# --- 4. HISTORY LOG ---
st.subheader("📜 Recent Vape Detection Events")
# Filter for 24 hours and predictions of 1
history = df[(df['Vape_Prediction'] == 1) & (df['Sort_Time'] >= (df['Sort_Time'].max() - pd.Timedelta(days=1))]
history = history[['Display_Time', 'TVOC', 'PM2.5']]

if not history.empty:
    st.table(history.head(10)) # Show latest 10 events
else:
    st.write("No vape events detected in the last 24 hours.")

# --- 5. GRAPHS ---
st.divider()
st.subheader("📈 Trends")
# (Graphs code remains same as previous version)
