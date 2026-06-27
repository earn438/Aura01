import os
import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH  = "models/rf_model_unified.joblib"
MODEL_NAME  = "Aurafarm AI"
SHEET_URL   = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ"
    "/pub?output=csv"
)
BASE_LAT    = 18.5847
BASE_LON    = 99.0256
REFRESH_MS  = 30_000   # auto-refresh every 30 seconds

FEATURE_COLS   = ["TVOC", "eCO2", "Temp", "Humidity", "PM2.5", "CH0", "CH3", "MQ135"]
MAPPING_DICT   = {
    "TVOC": "col_2", "eCO2": "col_3", "Temp": "col_4", "Humidity": "col_5",
    "PM2.5": "col_6", "CH0": "col_7", "CH3": "col_8", "MQ135": "col_9",
}
COLUMN_RENAME  = {
    "Unnamed: 0": "Timestamp",
    "tvoc": "TVOC", "eco2": "eCO2", "temp": "Temp", "humidity": "Humidity",
    "ch0": "CH0", "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10",
}

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title=MODEL_NAME, page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #21262d; }
    [data-testid="stHeader"]           { background: transparent; }

    /* ── Typography ── */
    h1, h2, h3, .big-title {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        letter-spacing: -0.5px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.5rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    /* ── Alert banners ── */
    .alert-vape {
        background: linear-gradient(90deg, #3d0f0f 0%, #1a0a0a 100%);
        border: 1px solid #ff4b4b;
        border-left: 5px solid #ff4b4b;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }
    .alert-clean {
        background: linear-gradient(90deg, #0a2414 0%, #051209 100%);
        border: 1px solid #00cc66;
        border-left: 5px solid #00cc66;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }
    .alert-offline {
        background: #1c2128;
        border: 1px solid #30363d;
        border-left: 5px solid #8b949e;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    /* ── Detection history cards ── */
    .det-card {
        background: #1a0a0a;
        border-left: 4px solid #ff4b4b;
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .det-time  { color: #ff7b7b; font-weight: 700; }
    .det-label { color: #8b949e; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"]  { background: #161b22; border-radius: 8px; gap: 4px; padding: 4px; }
    .stTabs [data-baseweb="tab"]       { background: transparent; color: #8b949e; border-radius: 6px; }
    .stTabs [aria-selected="true"]     { background: #21262d !important; color: #e6edf3 !important; }

    /* ── Node status pills ── */
    .node-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
    .pill-red   { color: #ff4b4b; font-weight: 700; }
    .pill-green { color: #00cc66; font-weight: 700; }

    /* ── Divider ── */
    hr { border-color: #21262d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# AUTO-REFRESH  (no extra package needed)
# ─────────────────────────────────────────────
st.markdown(
    f'<meta http-equiv="refresh" content="{REFRESH_MS // 1000}">',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

my_model = load_model()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        df = df.rename(columns=COLUMN_RENAME)
        if "Unnamed: 1" in df.columns:
            df = df.drop(columns=["Unnamed: 1"])
        if "Timestamp" in df.columns:
            df["Display_Time"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df["Sort_Time"]    = df["Display_Time"] + pd.Timedelta(hours=7)
            df = df.dropna(subset=["Display_Time"]).sort_values("Sort_Time", ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Aurafarm AI")
    st.markdown("<small style='color:#8b949e'>Environmental Monitor</small>", unsafe_allow_html=True)
    st.divider()

    if not df.empty:
        last_ts = pd.to_datetime(df["Display_Time"].iloc[0])
        st.markdown(f"**Last reading**")
        st.markdown(f"<span style='color:#58a6ff'>{last_ts.strftime('%Y-%m-%d %H:%M:%S')}</span>", unsafe_allow_html=True)
        st.markdown(f"<small style='color:#8b949e'>Auto-refreshes every 30 s</small>", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Model**")
    model_status = "✅ Online" if my_model else "⚠️ Offline"
    st.markdown(model_status)
    if my_model:
        st.markdown(f"<small style='color:#8b949e'>{type(my_model).__name__}</small>", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Data window**")
    hours_back = st.slider("Hours to display", min_value=1, max_value=72, value=24, step=1)

    st.divider()
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
# GUARD: no data
# ─────────────────────────────────────────────
if df.empty:
    st.warning("No data available. Check your Google Sheets connection.")
    st.stop()

latest   = df.iloc[0]
previous = df.iloc[1] if len(df) > 1 else latest

# ─────────────────────────────────────────────
# RUN MODEL PREDICTION
# ─────────────────────────────────────────────
prediction   = None
confidence   = None
all_features = df[FEATURE_COLS].rename(columns=MAPPING_DICT)

if my_model:
    try:
        latest_features = latest[FEATURE_COLS].to_frame().T.rename(columns=MAPPING_DICT)
        prediction = int(my_model.predict(latest_features)[0])

        if hasattr(my_model, "predict_proba"):
            proba      = my_model.predict_proba(latest_features)[0]
            confidence = float(proba[prediction]) * 100

        df["is_vape"] = my_model.predict(all_features)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("## Vapo noWay — Facility Monitor")

# ── Status banner ──────────────────────────────
if prediction is None:
    st.markdown(
        "<div class='alert-offline'>⚠️  <b>Detection system offline</b> — Model not loaded. "
        "Sensor data is still streaming normally.</div>",
        unsafe_allow_html=True,
    )
elif prediction == 1:
    conf_str = f" &nbsp;·&nbsp; Confidence: <b>{confidence:.0f}%</b>" if confidence else ""
    st.markdown(
        f"<div class='alert-vape'>🚨 &nbsp;<b>VAPE DETECTED</b>{conf_str} &nbsp;·&nbsp; "
        f"Timestamp: <b>{latest['Display_Time'].strftime('%H:%M:%S')}</b></div>",
        unsafe_allow_html=True,
    )
else:
    conf_str = f" &nbsp;·&nbsp; Confidence: <b>{confidence:.0f}%</b>" if confidence else ""
    st.markdown(
        f"<div class='alert-clean'>✅ &nbsp;<b>AIR QUALITY CLEAN</b>{conf_str} &nbsp;·&nbsp; "
        f"Last checked: <b>{latest['Display_Time'].strftime('%H:%M:%S')}</b></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# METRIC CARDS  (with deltas vs previous reading)
# ─────────────────────────────────────────────
def safe_delta(col):
    try:
        return round(float(latest[col]) - float(previous[col]), 2)
    except Exception:
        return None

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("🌡 Temp",        f"{latest['Temp']} °C",      delta=safe_delta("Temp"))
c2.metric("💧 Humidity",    f"{latest['Humidity']} %",   delta=safe_delta("Humidity"))
c3.metric("🌫 TVOC",        f"{latest['TVOC']} ppb",     delta=safe_delta("TVOC"),  delta_color="inverse")
c4.metric("💨 PM 2.5",      f"{latest['PM2.5']} μg/m³",  delta=safe_delta("PM2.5"), delta_color="inverse")
c5.metric("🟤 eCO₂",        f"{latest['eCO2']} ppm",     delta=safe_delta("eCO2"),  delta_color="inverse")
c6.metric("🔬 MQ135",       f"{latest['MQ135']}",        delta=safe_delta("MQ135"), delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────
# DETECTION HISTORY  (grouped events, table)
# ─────────────────────────────────────────────
col_hist, col_map = st.columns([1, 1], gap="large")

with col_hist:
    st.markdown("### 🔴 Detection History")

    if my_model and "is_vape" in df.columns:
        vape_rows = df[df["is_vape"] == 1].copy()

        if not vape_rows.empty:
            vape_rows = vape_rows.sort_values("Display_Time")
            vape_rows["block"] = (
                vape_rows["Display_Time"].diff() > pd.Timedelta(minutes=5)
            ).cumsum()

            event_rows = []
            for _, grp in vape_rows.groupby("block"):
                event_rows.append({
                    "Date":       grp["Display_Time"].min().strftime("%Y-%m-%d"),
                    "Start":      grp["Display_Time"].min().strftime("%H:%M"),
                    "End":        grp["Display_Time"].max().strftime("%H:%M"),
                    "Duration":   str(grp["Display_Time"].max() - grp["Display_Time"].min()).split(".")[0],
                    "Peak TVOC":  f"{grp['TVOC'].max():.0f} ppb",
                    "Peak PM2.5": f"{grp['PM2.5'].max():.1f} μg/m³",
                })

            # Most recent events first
            events_df = pd.DataFrame(event_rows[::-1])
            st.dataframe(
                events_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date":        st.column_config.TextColumn("Date"),
                    "Start":       st.column_config.TextColumn("Start"),
                    "End":         st.column_config.TextColumn("End"),
                    "Duration":    st.column_config.TextColumn("Duration"),
                    "Peak TVOC":   st.column_config.TextColumn("Peak TVOC"),
                    "Peak PM2.5":  st.column_config.TextColumn("Peak PM2.5"),
                },
            )
            st.caption(f"{len(events_df)} detection event(s) in available data.")
        else:
            st.markdown(
                "<div style='color:#8b949e;padding:24px 0'>No vape events detected in the available data.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='color:#8b949e;padding:24px 0'>Detection system offline — history unavailable.</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# FACILITY MAP
# ─────────────────────────────────────────────
with col_map:
    st.markdown("### 🗺 Facility Sensor Network")

    live_state = 1 if prediction == 1 else 0

    mock_sensors = pd.DataFrame({
        "sensor_id":    ["SN-01 (Main Lobby)", "SN-02 (East Restroom)", "SN-03 (Breakroom)", "SN-04 (Stairwell B)"],
        "latitude":     [BASE_LAT + 0.0004,    BASE_LAT + 0.0004,       BASE_LAT - 0.0005,   BASE_LAT + 0.0002],
        "longitude":    [BASE_LON,              BASE_LON - 0.0006,       BASE_LON - 0.0002,   BASE_LON + 0.0005],
        "vape_detected":[live_state,            0,                        0,                   0],
        "air_quality":  [
            "⚠ Vape Detected" if live_state else "✅ Clean",
            "✅ Clean",
            "✅ Clean",
            "✅ Clean",
        ],
    })
    mock_sensors["color"] = mock_sensors["vape_detected"].map(
        {1: [255, 75, 75, 220], 0: [0, 204, 102, 220]}
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=mock_sensors,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=6,
        radius_units="meters",
        radius_min_pixels=6,
        radius_max_pixels=18,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=BASE_LAT, longitude=BASE_LON, zoom=16.5, pitch=0)
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="dark",
            tooltip={"text": "Sensor: {sensor_id}\nStatus: {air_quality}"},
        )
    )

    # Node status list
    st.markdown("**Live Node Status**")
    for _, row in mock_sensors.iterrows():
        pill_class = "pill-red" if row["vape_detected"] else "pill-green"
        st.markdown(
            f"<div class='node-card'>"
            f"<span>{row['sensor_id']}</span>"
            f"<span class='{pill_class}'>{row['air_quality']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────
# TREND CHARTS  (respect sidebar hours_back)
# ─────────────────────────────────────────────
st.markdown(f"### 📈 Sensor Trends — Last {hours_back} Hours")

chart_data = df.sort_values("Sort_Time", ascending=True).copy()
cutoff     = chart_data["Sort_Time"].max() - pd.Timedelta(hours=hours_back)
chart_data = chart_data[chart_data["Sort_Time"] >= cutoff].set_index("Sort_Time")
numeric_cols = chart_data.select_dtypes(include="number").columns
chart_data   = chart_data[numeric_cols].resample("1min").mean().interpolate(method="time")

# Overlay vape periods as a column for reference
if my_model and "is_vape" in df.columns:
    vape_overlay = (
        df[df["Sort_Time"] >= cutoff]
        .sort_values("Sort_Time")
        .set_index("Sort_Time")[["is_vape"]]
        .resample("1min")
        .max()
        .rename(columns={"is_vape": "⚠ Vape Event"})
    )
    chart_data = chart_data.join(vape_overlay, how="left")

tab1, tab2, tab3, tab4 = st.tabs(["🟤 Particles", "🌫 Air Quality", "🌡 Climate", "📊 All Sensors"])

with tab1:
    cols_p = [c for c in ["PM2.5", "PM10", "MQ135"] if c in chart_data.columns]
    if "⚠ Vape Event" in chart_data.columns:
        cols_p.append("⚠ Vape Event")
    st.line_chart(chart_data[cols_p])

with tab2:
    cols_a = [c for c in ["TVOC", "eCO2"] if c in chart_data.columns]
    if "⚠ Vape Event" in chart_data.columns:
        cols_a.append("⚠ Vape Event")
    st.line_chart(chart_data[cols_a])

with tab3:
    cols_c = [c for c in ["Temp", "Humidity"] if c in chart_data.columns]
    st.line_chart(chart_data[cols_c])

with tab4:
    display_cols = [c for c in FEATURE_COLS if c in chart_data.columns]
    st.line_chart(chart_data[display_cols])

st.divider()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.78rem;padding:8px 0'>"
    "Aurafarm AI · Auto-refreshes every 30 s · "
    "Sensor data streamed from Google Sheets"
    "</div>",
    unsafe_allow_html=True,
)
