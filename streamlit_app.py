import streamlit as st
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VapeRadar Dashboard", page_icon="🚭", layout="wide")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Alert Settings")
    tvoc_limit = st.slider("TVOC Alert Threshold (ppb)", 100, 1000, 300, 50)
    pm_limit = st.slider("PM2.5 Alert Threshold (μg/m³)", 10.0, 100.0, 35.0, 5.0)
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()

# --- DATA LOADING ---
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        column_mapping = {
            "Unnamed: 0": "Timestamp", "tvoc": "TVOC", "eco2": "eCO2",
            "temp": "Temp", "humidity": "Humidity", "ch0": "CH0",
            "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10"
        }
        df = df.rename(columns=column_mapping)
        if "Unnamed: 1" in df.columns: df = df.drop(columns=["Unnamed: 1"])
            
        if 'Timestamp' in df.columns:
            # Create the raw timestamp for display
            df['Display_Time'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
            # Create the offset time for sorting/graphing
            df['Sort_Time'] = df['Display_Time'] + pd.Timedelta(hours=7)
            
            df = df.dropna(subset=['Display_Time'])
            df = df.sort_values(by='Sort_Time', ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

# --- DASHBOARD ---
st.title("🚭 VapeRadar Dashboard")
if df.empty:
    st.warning("No data found.")
    st.stop()

latest = df.iloc[0]
st.success(f"✅ **System Normal:** Air Quality Status Active")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temp", f"{latest.get('Temp', 'N/A')} °C")
col2.metric("Humidity", f"{latest.get('Humidity', 'N/A')} %")
col3.metric("TVOC", f"{latest.get('TVOC', 'N/A')} ppb")
col4.metric("PM 2.5", f"{latest.get('PM2.5', 'N/A')} μg/m³")

# Display the original time from the sheet
st.caption(f"Last updated (Sensor Time): {latest['Display_Time']}")
st.divider()

# --- GRAPHING ---
st.subheader("📈 Past 24 Hours Trends")
# Filter by Sort_Time so the graphs follow your local 24h cycle
chart_data = df.sort_values(by='Sort_Time', ascending=True)
cutoff = chart_data['Sort_Time'].max() - pd.Timedelta(days=1)
chart_data = chart_data[chart_data['Sort_Time'] >= cutoff]
chart_data = chart_data.set_index('Sort_Time')

# Graphs
numeric_cols = chart_data.select_dtypes(include='number').columns
chart_data = chart_data[numeric_cols].resample('1min').mean().interpolate(method='time')

tab1, tab2, tab3 = st.tabs(["🌫️ Particles", "🌬️ Air Quality", "🌡️ Climate"])
with tab1: st.line_chart(chart_data[['PM2.5', 'PM10', 'MQ135']])
with tab2: st.line_chart(chart_data[['TVOC', 'eCO2']])
with tab3: st.line_chart(chart_data[['Temp', 'Humidity']])

with st.expander("📊 View Raw Data"):
    st.dataframe(df, use_container_width=True)
