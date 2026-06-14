import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(page_title="VapeRadar Dashboard", page_icon="🚭", layout="wide")

# Header
st.title("🚭 VapeRadar Dashboard")
st.markdown("**Real-Time Vape Detection Monitoring**")
st.divider()

# --- 1. CONNECT TO GOOGLE SHEETS ---
# Your published CSV link
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"

@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        # Ensure the Timestamp column exists and is readable
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values(by='Timestamp', ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# Load the live data
df = load_sensor_data()
# --- DEBUGGING BLOCK ---
st.warning("🔍 **Debug Mode Active:** Here is the raw data Streamlit is receiving:")
st.dataframe(df)
st.write("🔍 **Exact Column Headers Found:**")
st.write(df.columns.tolist())
st.divider()
# -----------------------
# --- 2. DISPLAY LIVE SUMMARY METRICS ---
if not df.empty:
    latest_reading = df.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Latest Temp", value=f"{latest_reading.get('Temp', 'N/A')} °C")
    with col2:
        st.metric(label="Latest Humidity", value=f"{latest_reading.get('Humidity', 'N/A')} %")
    with col3:
        st.metric(label="TVOC Level", value=f"{latest_reading.get('TVOC', 'N/A')} ppb")
    with col4:
        st.metric(label="PM 2.5", value=f"{latest_reading.get('PM2.5', 'N/A')} μg/m³")
        
    st.caption(f"Last updated: {latest_reading.get('Timestamp', 'Unknown time')}")
else:
    st.warning("No data found. Please check your Google Sheet link or data formatting.")

st.divider()

# --- 3. SHOW THE FULL DATA HISTORY ---
st.subheader("📊 Live Sensor History")
st.markdown("This table updates automatically from your Google Sheet every 30 seconds.")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# --- 4. VISUALIZE THE DATA ---
st.subheader("📈 Environmental Trends")

if not df.empty and 'Timestamp' in df.columns:
    # Set the index to Timestamp for Streamlit charts
    chart_data = df.set_index('Timestamp')
    
    # Create tabs to organize the charts neatly
    tab1, tab2, tab3 = st.tabs(["🌬️ Air Quality (TVOC & eCO2)", "🌡️ Climate (Temp & Humidity)", "🌫️ Particulates & Gas (PM, MQ135)"])
    
    with tab1:
        # Check if columns exist before plotting to prevent errors
        cols_to_plot = [col for col in ['TVOC', 'eCO2'] if col in chart_data.columns]
        if cols_to_plot:
            st.line_chart(chart_data[cols_to_plot])
            
    with tab2:
        cols_to_plot = [col for col in ['Temp', 'Humidity'] if col in chart_data.columns]
        if cols_to_plot:
            st.line_chart(chart_data[cols_to_plot])
            
    with tab3:
        cols_to_plot = [col for col in ['PM2.5', 'PM10', 'MQ135', 'CH0', 'CH3'] if col in chart_data.columns]
        if cols_to_plot:
            st.line_chart(chart_data[cols_to_plot])
