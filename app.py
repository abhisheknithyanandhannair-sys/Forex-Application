import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import yfinance as yf
from datetime import datetime as dt
import os
import base64

# ==========================================
# 1. PERFORMANCE CACHING (THE SPEED BOOST)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_data(period):
    """Fetches data and caches it so we don't download it every time."""
    from models import fetch_fx_data
    return fetch_fx_data(period=period)

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_models(df_rate, forecast_days):
    """Runs all heavy math models ONCE and saves the results."""
    from models import run_ols_model, run_arima_model, run_garch_model
    # Run all models
    ols_model, ols_forecast = run_ols_model(df_rate, forecast_days)
    arima_model, arima_forecast = run_arima_model(df_rate, forecast_days)
    garch_model, garch_variance = run_garch_model(df_rate, forecast_days)
    return ols_model, ols_forecast, arima_model, arima_forecast, garch_model, garch_variance

# Load other lightweight functions
from models import (
    build_forecast_dates,
    classify_risk_from_variance,
    generate_trading_advice,
    get_rate_for_date,
)
from transactions import (
    append_transaction,
    load_transactions,
    clear_transactions,
    get_savings_stats,
)

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FilterFX - Precision Analytics",
    page_icon="🪙",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================================
# 💎 LUXURY DARK THEME CSS
# ==========================================
st.markdown(
    """
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@400;600&display=swap');

    /* 1. BACKGROUND: Ivory Black / Deep Charcoal */
    .stApp {
        background-color: #121212; /* Deep Ivory Black */
        background-image: linear-gradient(to bottom right, #121212, #1E1E1E);
    }
    
    /* 2. TEXT GLOBALS */
    p, .stMarkdown, .stText, label, .stCaption, li {
        color: #E0E0E0 !important; /* Silver/White */
        font-family: 'Montserrat', sans-serif !important;
    }

    /* 3. SHINY GOLD HEADINGS */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        background: linear-gradient(45deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 1px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
    }
    
    /* 4. METRICS / CARDS */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E; /* Dark Card */
        border: 1px solid #B38728; /* Gold Border */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #C0C0C0 !important; /* Silver Label */
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        /* Shiny Gold Number */
        background: linear-gradient(45deg, #BF953F, #FCF6BA, #B38728);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Playfair Display', serif;
    }
    div[data-testid="stMetricDelta"] {
         color: #E0E0E0 !important; /* Silver Delta */
    }

    /* 5. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0E0E0E;
        border-right: 1px solid #333;
    }
    
    /* 6. BUTTONS */
    .stButton>button {
        background: linear-gradient(to bottom, #1E1E1E, #121212);
        color: #B38728; /* Gold Text */
        border: 1px solid #B38728;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #BF953F, #B38728);
        color: #000000; /* Black Text on Gold Hover */
        border-color: #FCF6BA;
        box-shadow: 0 0 10px rgba(179, 135, 40, 0.5);
    }
    
    /* 7. INPUT FIELDS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stDateInput>div>div>input {
        background-color: #1E1E1E;
        color: #FCF6BA; /* Light Gold Text */
        border: 1px solid #444;
    }

    /* 8. CUSTOM CONTAINERS (For Converter) */
    .gold-card {
        background-color: #1E1E1E;
        border: 1px solid #B38728;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# LOGO HANDLING
# ==========================================
def get_logo_html(width=120):
    """Returns HTML for logo, preferring local file 'logo.jpg'"""
    if os.path.exists("logo.jpg"):
        with open("logo.jpg", "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:image/jpg;base64,{data}" width="{width}" style="border-radius: 10px; border: 2px solid #B38728;">'
    else:
        # Fallback Gold/Black SVG
        fallback = """
        data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMDAgMTAwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImdvbGRHcmFkIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojQkY5NTNGO3N0b3Atb3BhY2l0eToxIiAvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3R5bGU9InN0b3AtY29sb3I6I0ZDRjZCOTtzdG9wLW9wYWNpdHk6MSIgLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48dGV4dCB4PSIxMCIgeT0iNzAiIGZvbnQtZmFtaWx5PSJzZXJpZiIgZm9udC13ZWlnaHQ9ImJvbGQiIGZvbnQtc2l6ZT0iNjAiIGZpbGw9InVybCgjZ29sZEdyYWQpIj5GaWx0ZXJGWDwvdGV4dD48L3N2Zz4=
        """
        return f'<img src="{fallback}" width="{width}">'

# ==========================================
# SPLASH SCREEN (LUXURY VERSION)
# ==========================================
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    splash = st.empty()
    with splash.container():
        logo_html = get_logo_html(width=250)
        st.markdown(
            f"""
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh;'>
                {logo_html}
                <h1 style='font-size: 60px; margin-top: 20px;'>FilterFX</h1>
                <p style='color: #C0C0C0; font-size: 18px; letter-spacing: 2px;'>PRECISION ECONOMETRICS</p>
                <div style='margin-top: 20px; width: 50px; height: 2px; background: linear-gradient(90deg, transparent, #B38728, transparent);'></div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    time.sleep(2.5)
    splash.empty()
    st.session_state.splash_shown = True

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Configuration")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)
data_period = st.sidebar.selectbox("Data Lookback", ["2y", "4y", "5y", "10y", "max"])
st.sidebar.divider()
st.sidebar.markdown("""
<div style='text-align: center; color: #888;'>
    <small>Powered by</small><br>
    <strong style='color: #B38728;'>OLS • ARIMA • GARCH</strong>
</div>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA (CACHED)
# ==========================================
with st.spinner("Accessing Market Data..."):
    df = get_cached_data(data_period)
    if df.empty:
        st.error("Market data unavailable.")
        st.stop()
    
    current_rate = df["Rate"].iloc[-1]
    
    # Run Models
    ols_model, ols_forecast, arima_model, arima_forecast, garch_model, garch_variance = get_cached_models(df["Rate"], forecast_days)

predicted_rate = 0.0
if arima_forecast is not None and len(arima_forecast) > 0:
    predicted_rate = arima_forecast.iloc[-1]

# ==========================================
# HEADER
# ==========================================
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.markdown(get_logo_html(width=100), unsafe_allow_html=True)
with col_title:
    st.title("FilterFX Dashboard")
    st.markdown("<p style='font-style: italic; color: #888 !important; margin-top: -15px;'>Real-time Forex Intelligence</p>", unsafe_allow_html=True)

# Navigation
nav_action = st.radio("", ["🏠 Home", "☕ Savings", "🏆 Rankings"], horizontal=True, label_visibility="collapsed")
if nav_action == "☕ Savings": st.switch_page("pages/01_Savings.py")
elif nav_action == "🏆 Rankings": st.switch_page("pages/02_Rankings.py")

# ==========================================
# VISUALIZATION FUNCTION (DARK MODE)
# ==========================================
def create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate):
    # Dark Theme for Matplotlib
    plt.rcParams.update({
        "axes.facecolor": "#1E1E1E",
        "figure.facecolor": "#121212",
        "text.color": "#E0E0E0",
        "axes.labelcolor": "#C0C0C0",
        "xtick.color": "#C0C0C0",
        "ytick.color": "#C0C0C0",
        "axes.edgecolor": "#444",
        "grid.color": "#444",
        "grid.alpha": 0.3
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # Plot 1: Forecast
    ax1 = axes[0]
    subset = df["Rate"].iloc[-180:]
    ax1.plot(subset.index, subset, label="Historical", color="#C0C0C0", linewidth=1.5)
    
    last_date = df.index[-1]
    if len(ols_forecast) > 0:
        dates_f = build_forecast_dates(last_date, len(ols_forecast))
        ax1.plot(dates_f, ols_forecast, label="OLS Trend", linestyle="--", color="#B38728", linewidth=2) # Gold
    
    if len(arima_forecast) > 0:
        dates_f = build_forecast_dates(last_date, len(arima_forecast))
        ax1.plot(dates_f, arima_forecast.values, label="ARIMA Forecast", color="#00FFCC", linewidth=2.5) # Neon Teal for contrast

    ax1.set_title("EUR/INR Forecast", fontsize=14, fontweight="bold", color="#B38728")
    legend = ax1.legend(loc="upper left", facecolor="#1E1E1E", edgecolor="#B38728")
    for text in legend.get_texts(): text.set_color("#E0E0E0")
    ax1.grid(True)
    
    # Plot 2: MA
    ax2 = axes[1]
    df["MA_30"] = df["Rate"].rolling(window=30).mean()
    subset2 = df.iloc[-250:]
    ax2.plot(subset2.index, subset2["Rate"], color="#555", alpha=0.5)
    ax2.plot(subset2.index, subset2["MA_30"], label="30-Day MA", color="#B38728", linewidth=2)
    
    ax2.set_title("Trend Analysis (Moving Averages)", fontsize=14, fontweight="bold", color="#B38728")
    legend2 = ax2.legend(loc="upper left", facecolor="#1E1E1E", edgecolor="#B38728")
    for text in legend2.get_texts(): text.set_color("#E0E0E0")
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# ==========================================
# SNAPSHOT
# ==========================================
st.subheader("Market Snapshot")
m1, m2 = st.columns(2)
with m1: st.metric("📊 Current Rate", f"₹{current_rate:.4f}")
with m2: 
    chg = ((df["Rate"].iloc[-1] - df["Rate"].iloc[-2])/df["Rate"].iloc[-2])*100
    st.metric("📈 Daily Change", f"{chg:.3f}%", delta=f"{chg:.3f}%")

st.divider()

# ==========================================
# CONVERTER (GOLD CARD STYLE)
# ==========================================
st.subheader("🔄 Smart Converter")

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown(f"""
    <div class="gold-card">
        <div style="color: #C0C0C0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Current Rate</div>
        <div style="font-family: 'Playfair Display'; font-size: 32px; font-weight: bold; background: linear-gradient(45deg, #BF953F, #FCF6BA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ₹{current_rate:.4f}
        </div>
    </div>""", unsafe_allow_html=True)
with rc2:
    diff = predicted_rate - current_rate
    arrow = "↗" if diff > 0 else "↘"
    col = "#00FFCC" if diff > 0 else "#FF4444"
    st.markdown(f"""
    <div class="gold-card">
        <div style="color: #C0C0C0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Forecast (+{forecast_days}d)</div>
        <div style="font-family: 'Playfair Display'; font-size: 32px; font-weight: bold; background: linear-gradient(45deg, #BF953F, #FCF6BA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ₹{predicted_rate:.4f} <span style='color:{col}; font-size:0.6em;'>{arrow}</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
cc1, cc2 = st.columns([1,2])
with cc1: d = st.radio("Direction", ["EUR ➡ INR", "INR ➡ EUR"], label_visibility="collapsed")
with cc2: amt = st.number_input("Amount", value=1000.0, step=100.0, label_visibility="collapsed")

if amt > 0:
    val = amt * current_rate if d == "EUR ➡ INR" else amt / current_rate
    sym_in, sym_out = ("€", "₹") if d == "EUR ➡ INR" else ("₹", "€")
    st.success(f"**{sym_in} {amt:,.2f}** = **{sym_out} {val:,.2f}**")

# ==========================================
# LOGGING
# ==========================================
st.divider()
st.subheader("💾 Log Transaction")
with st.expander("🗑️ Transaction History"):
    if st.button("Clear Log", use_container_width=True):
        clear_transactions()
        st.rerun()

tc1, tc2 = st.columns(2)
with tc1: t_amt = st.number_input("Amount (INR)", min_value=0.0, step=1000.0)
with tc2: t_date = st.date_input("Date")

if st.button("Confirm Log Entry", type="primary", use_container_width=True) and t_amt > 0:
    today = dt.now().date()
    r_use = predicted_rate if t_date > today else (get_rate_for_date(df, t_date) or current_rate)
    append_transaction(t_date, t_amt, r_use)
    st.success(f"Logged successfully @ {r_use:.4f}")

# ==========================================
# RESULTS SECTION
# ==========================================
st.divider()
st.header("📊 Model Analysis")

if hasattr(ols_model, 'params'):
    odir = "UP ↗" if ols_model.params[1] > 0 else "DOWN ↘"
    ostr = f"{ols_model.rsquared*100:.1f}%"
else: odir, ostr = "NEUTRAL ↔", "0%"

ac1, ac2 = st.columns(2)
with ac1: st.markdown(f"**OLS Trend:** {odir} <span style='color:#888'>(Conf: {ostr})</span>", unsafe_allow_html=True)
with ac2: 
    pchg = ((predicted_rate-current_rate)/current_rate)*100
    st.metric(f"Forecast ({forecast_days}D)", f"₹{predicted_rate:.4f}", delta=f"{pchg:.2f}%")

# GARCH Risk
st.subheader("⚠️ Volatility Radar")
rlab, rdesc, avol = classify_risk_from_variance(garch_variance)
st.markdown(f"""
<div style="background-color: #1E1E1E; border-left: 4px solid #B38728; padding: 15px; border-radius: 4px;">
    <strong style="color: #FCF6BA; font-size: 1.1em;">Risk Level: {rlab}</strong> 
    <span style="color: #888;">(Vol Index: {avol:.2f})</span><br>
    <em style="color: #C0C0C0;">{rdesc}</em>
</div>""", unsafe_allow_html=True)

st.divider()
st.subheader("📉 Technical Chart")
st.pyplot(create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate))
st.caption("FilterFX • Precision Econometrics")
