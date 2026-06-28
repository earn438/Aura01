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
HOURS_BACK  = 24        # fixed trend-chart window

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
# PAGE CONFIG & GLOBAL CSS  (White / Light Mode)
# ─────────────────────────────────────────────
st.set_page_config(page_title=MODEL_NAME, layout="wide")

st.markdown(
    """
    <style>
    /* ── Base canvas ── */
    [data-testid="stAppViewContainer"] { background: #fffff7; color: #1a1f29; }
    [data-testid="stHeader"]           { background: transparent; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }

    /* ── Typography ── */
    h1, h2, h3, h4, .big-title {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        letter-spacing: -0.3px;
        color: #1a1f29;
    }
    .eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 10px;
    }
    .eyebrow::before {
        content: "●";
        color: #16a34a;
        margin-right: 6px;
        font-size: 0.6rem;
    }

    /* ── Card containers (every bordered st.container) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.07);
        padding: 24px 26px !important;
        margin-bottom: 22px !important;
    }

    /* ── Metric cards (inside the metrics card, so keep them subtle) ── */
    [data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.75rem !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.45rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    /* ── Status pill (header) ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 16px 48px;
        min-width: 320px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.15rem;
        white-space: nowrap;
    }
    .status-vape    { background: rgba(220,38,38,0.08);  border: 1px solid #dc2626; color: #b91c1c; }
    .status-clean   { background: rgba(22,163,74,0.08);  border: 1px solid #16a34a; color: #15803d; }
    .status-offline { background: rgba(107,114,128,0.08); border: 1px solid #d1d5db; color: #6b7280; }

    /* ── Detection history cards ── */
    .det-card {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .det-time  { color: #b91c1c; font-weight: 700; }
    .det-label { color: #6b7280; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"]  { background: #f1f3f5; border-radius: 10px; gap: 4px; padding: 4px; }
    .stTabs [data-baseweb="tab"]       { background: transparent; color: #6b7280; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] p     { color: inherit; }
    .stTabs [aria-selected="true"]     { background: #ffffff !important; color: #111827 !important; box-shadow: 0 1px 4px rgba(15,23,42,0.08); }

    /* ── Node status pills ── */
    .node-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        color: #1a1f29;
    }
    .pill-red   { color: #dc2626; font-weight: 700; }
    .pill-green { color: #16a34a; font-weight: 700; }

    /* ── Misc ── */
    hr { border-color: #e5e7eb; }
    [data-testid="stCaptionContainer"] { color: #6b7280; }
    p, span, div { }
    </style>
    """,
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
# HERO / HEADER CARD  (title + live status pill)
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='text-align:center;font-size:1.7rem;font-weight:800;line-height:1.2;color:#1a1f29'>Vapo noWay</div>"
        "<div style='text-align:center;color:#6b7280;font-size:0.95rem;margin-top:2px;margin-bottom:18px'>"
        "Facility Air Quality Monitor</div>",
        unsafe_allow_html=True,
    )

    if prediction is None:
        st.markdown(
            "<div style='text-align:center'><span class='status-pill status-offline'>"
            "⚠️ Detection Offline</span></div>",
            unsafe_allow_html=True,
        )
    elif prediction == 1:
        conf_str = f" · {confidence:.0f}%" if confidence else ""
        st.markdown(
            f"<div style='text-align:center'><span class='status-pill status-vape'>"
            f"Vape Detected{conf_str}</span><br>"
            f"<span style='color:#6b7280;font-size:0.78rem'>as of {latest['Display_Time'].strftime('%H:%M:%S')}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        conf_str = f" · {confidence:.0f}%" if confidence else ""
        st.markdown(
            f"<div style='text-align:center'><span class='status-pill status-clean'>"
            f"✅ Air Quality Clean{conf_str}</span><br>"
            f"<span style='color:#6b7280;font-size:0.78rem'>as of {latest['Display_Time'].strftime('%H:%M:%S')}</span></div>",
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

with st.container(border=True):
    st.markdown("<div class='eyebrow'>Live Readings</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric(" TVOC",     f"{latest['TVOC']} ppb",     delta=safe_delta("TVOC"),  delta_color="inverse")
    c2.metric(" PM 2.5",   f"{latest['PM2.5']} μg/m³",  delta=safe_delta("PM2.5"), delta_color="inverse")
    c3.metric(" eCO₂",     f"{latest['eCO2']} ppm",     delta=safe_delta("eCO2"),  delta_color="inverse")
    c4.metric(" MQ7",      f"{latest['CH0']}",          delta=safe_delta("CH0"),   delta_color="inverse")
    c5.metric(" MQ135",    f"{latest['MQ135']}",        delta=safe_delta("MQ135"), delta_color="inverse")
    c6.metric(" Temp",     f"{latest['Temp']} °C",      delta=safe_delta("Temp"))
    c7.metric(" Humidity", f"{latest['Humidity']} %",   delta=safe_delta("Humidity"))

# ─────────────────────────────────────────────
# DETECTION HISTORY  +  FACILITY MAP
# ─────────────────────────────────────────────
col_hist, col_map = st.columns([1, 1], gap="large")

with col_hist:
    with st.container(border=True):
        st.markdown("<div class='eyebrow'>Detection History</div>", unsafe_allow_html=True)

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
                    "<div style='color:#6b7280;padding:24px 0'>No vape events detected in the available data.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#6b7280;padding:24px 0'>Detection system offline — history unavailable.</div>",
                unsafe_allow_html=True,
            )

with col_map:
    with st.container(border=True):
        st.markdown("<div class='eyebrow'>Facility Sensor Network</div>", unsafe_allow_html=True)

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
            {1: [220, 38, 38, 220], 0: [22, 163, 74, 220]}
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
                map_style="light",
                tooltip={"text": "Sensor: {sensor_id}\nStatus: {air_quality}"},
            )
        )

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

# ─────────────────────────────────────────────
# TREND CHARTS  (uses fixed HOURS_BACK window)
# ─────────────────────────────────────────────
chart_data = df.sort_values("Sort_Time", ascending=True).copy()
cutoff     = chart_data["Sort_Time"].max() - pd.Timedelta(hours=HOURS_BACK)
chart_data = chart_data[chart_data["Sort_Time"] >= cutoff].set_index("Sort_Time")
numeric_cols = chart_data.select_dtypes(include="number").columns
chart_data   = chart_data[numeric_cols].resample("1min").mean().interpolate(method="time")

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

with st.container(border=True):
    st.markdown(f"<div class='eyebrow'>Sensor Trends · Last {HOURS_BACK} Hours</div>", unsafe_allow_html=True)

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

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#6b7280;font-size:0.78rem;padding:8px 0'>"
    "Aurafarm AI · Auto-refreshes every 30 s · "
    "Sensor data streamed from Google Sheets"
    "</div>",
    unsafe_allow_html=True,
)
