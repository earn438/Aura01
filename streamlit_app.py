import streamlit as st
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VapeRadar Dashboard", page_icon="🚭", layout="wide")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"

# --- SIDEBAR (Settings & Refresh) ---
with st.sidebar:
    st.header("⚙️ Alert Settings")
    st.write("Adjust the thresholds that trigger a vape detection alert.")
    tvoc_limit = st.slider("TVOC Alert Threshold (ppb)", min_value=100, max_value=1000, value=300, step=50)
    pm_limit = st.slider("PM2.5 Alert Threshold (μg/m³)", min_value=10.0, max_value=100.0, value=35.0, step=5.0)
    
    st.divider()
    st.write("Data refreshes automatically, but you can force an update below.")
    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        
        column_mapping = {
            "Unnamed: 0": "Timestamp",
            "tvoc": "TVOC",
            "eco2": "eCO2",
            "temp": "Temp",
            "humidity": "Humidity",
            "ch0": "CH0",
            "ch3": "CH3",
            "mq135": "MQ135",
            "2.5": "PM2.5",
            "10": "PM10"
        }
        df = df.rename(columns=column_mapping)
        
        if "Unnamed: 1" in df.columns:
            df = df.drop(columns=["Unnamed: 1"])
            
        if 'Timestamp' in df.columns:
            # Parse dates (dayfirst=True fixes the DD/MM/YYYY confusion)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
            
            # --- TIMEZONE FIX (+7 UTC) ---
            df['Timestamp'] = df['Timestamp'] + pd.Timedelta(hours=7)
            
            df = df.dropna(subset=['Timestamp'])
            df = df.sort_values(by='Timestamp', ascending=False)
            
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

# --- MAIN DASHBOARD HEADER ---
st.title("🚭 VapeRadar Dashboard")
st.markdown("**Real-Time Vape Detection Monitoring**")

if df.empty:
    st.warning("No data found. Please check your Google Sheet connection.")
    st.stop()

# --- ALERT SYSTEM ---
latest = df.iloc[0]
alerts_triggered = []

if latest.get('TVOC', 0) > tvoc_limit:
    alerts_triggered.append(f"High TVOC ({latest['TVOC']} ppb)")
if latest.get('PM2.5', 0) > pm_limit:
    alerts_triggered.append(f"High PM2.5 ({latest['PM2.5']} μg/m³)")

if alerts_triggered:
    st.error(f"🚨 **VAPE WARNING TRIGGERED:** " + " | ".join(alerts_triggered))
else:
    st.success("✅ **Air Quality Normal:** No significant vapor detected in the last reading.")

st.divider()

# --- LIVE METRICS WITH TRENDS ---
st.subheader("📡 Live Sensor Data")

col1, col2, col3, col4 = st.columns(4)

if len(df) > 1:
    prev = df.iloc[1]
    delta_temp = round(latest.get('Temp', 0) - prev.get('Temp', 0), 2)
    delta_hum = round(latest.get('Humidity', 0) - prev.get('Humidity', 0), 2)
    delta_tvoc = int(latest.get('TVOC', 0) - prev.get('TVOC', 0))
    delta_pm = round(latest.get('PM2.5', 0) - prev.get('PM2.5', 0), 2)
else:
    delta_temp = delta_hum = delta_tvoc = delta_pm = None

with col1:
    st.metric(label="Latest Temp", value=f"{latest.get('Temp', 'N/A')} °C", delta=delta_temp)
with col2:
    st.metric(label="Latest Humidity", value=f"{latest.get('Humidity', 'N/A')} %", delta=delta_hum)
with col3:
    st.metric(label="TVOC Level", value=f"{latest.get('TVOC', 'N/A')} ppb", delta=delta_tvoc, delta_color="inverse")
with col4:
    st.metric(label="PM 2.5", value=f"{latest.get('PM2.5', 'N/A')} μg/m³", delta=delta_pm, delta_color="inverse")
    
st.caption(f"Last updated: {latest.get('Timestamp', 'Unknown time')}")
st.divider()

# --- DATA VISUALIZATION TABS (24H SMOOTHED) ---
st.subheader("📈 Past 24 Hours Trends")

if 'Timestamp' in df.columns:
    # 1. Sort ascending so time flows left to right
    chart_data = df.sort_values(by='Timestamp', ascending=True)
    
    # 2. Filter for only the last 24 hours
    latest_time = chart_data['Timestamp'].max()
    cutoff_time = latest_time - pd.Timedelta(days=1)
    chart_data = chart_data[chart_data['Timestamp'] >= cutoff_time]
    
    # 3. Set index to Timestamp for pandas time-series manipulation
    chart_data = chart_data.set_index('Timestamp')
    
    # 4. Resample (average into 1-minute buckets) and Interpolate (fill gaps)
    # We only apply this to numeric columns so strings don't break the math
    numeric_cols = chart_data.select_dtypes(include='number').columns
    chart_data = chart_data[numeric_cols].resample('1min').mean().interpolate(method='time')
    
    tab1, tab2, tab3 = st.tabs(["🌫️ Particles & Gas (Vape Indicators)", "🌬️ Air Quality (TVOC & eCO2)", "🌡️ Climate (Temp & Humidity)"])
    
    with tab1:
        cols_to_plot = [col for col in ['PM2.5', 'PM10', 'MQ135', 'CH0', 'CH3'] if col in chart_data.columns]
        if cols_to_plot:
            st.line_chart(chart_data[cols_to_plot])
            
    with tab2:
        cols_to_plot = [col for col in ['TVOC', 'eCO2'] if col in chart_data.columns]
        if cols_to_plot:
            st.line_chart(chart_data[cols_to_plot])
            
    with tab3:
        cols_to_plot = [col for col in ['Temp', 'Humidity'] if col in chart_data.columns]
        if cols_to_plot:
            st.line_chart(chart_data[cols_to_plot])

st.divider()

# --- RAW DATA TABLE ---
with st.expander("📊 View Raw Sensor Data"):
    st.dataframe(df, use_container_width=True, hide_index=True)
