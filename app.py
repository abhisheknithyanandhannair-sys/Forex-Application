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
import base64

# ==========================================
# EMBEDDED LOGO (Base64) - NO FILE NEEDED
# ==========================================
# This is a placeholder SVG logo encoded in base64. 
# If you have your own image, convert it to base64 and replace the string below.
logo_base64 = """
data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMDAgMTAwIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZ29sZEdyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojRDRBRjM3O3N0b3Atb3BhY2l0eToxIiAvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiNCOEM0MjU7c3RvcC1vcGFjaXR5OjEiIC8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGZpbHRlciBpZD0iYmxhY2tPdXRsaW5lIj4KICAgICAgPGZlTW9ycGhvbG9neSBpbj0iU291cmNlQWxwaGEiIHJlc3VWx0PSJkaWxhdGVkIiBvcGVyYXRvcj0iZGlsYXRlIiByYWRpdXM9IjEiLz4KICAgICAgPGZlRmxvb2QgZmxvb2QtY29sb3I9IiMwMDAwMDAiIHJlc3VsdD0iYmxhY2siLz4KICAgICAgPGZlQ29tcG9zaXRlIGluPSJibGFjayIgaW4yPSJkaWxhdGVkIiBvcGVyYXRvcj0iaW4iIHJlc3VsdD0ib3V0bGluZSIvPgogICAgICA8ZmVNZXJnZT4KICAgICAgICA8ZmVNZXJnZU5vZGUgaW49Im91dGxpbmUiLz4KICAgICAgICA8ZmVNZXJnZU5vZGUgaW49IlNvdXJjZUdyYXBoaWMiLz4KICAgICAgPC9mZU1lcmdlPgogICAgPC9maWx0ZXI+CiAgPC9kZWZzPgogIDx0ZXh0IHg9IjEwIiB5PSI3MCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iYm9sZCIgZm9udC1zaXplPSI2MCIgZmlsbD0idXJsKCNnb2xkR3JhZCkiIGZpbHRlcj0idXJsKCNibGFja091dGxpbmUpIj5GaWx0ZXJGWDwvdGV4dD4KICA8cG9seWxpbmUgcG9pbnRzPSIxNzAgODAgMjEwIDQwIDI1MCA3MCAyOTAgMjAiIHN0cm9rZT0idXJsKCNnb2xkR3JhZCkiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgZmlsdGVyPSJ1cmwoI2JsYWNrT3V0bGluZSkiLz4KICA8cG9seWdvbiBwb2ludHM9IjI5MCAyMCAyNzAgMzAgMjkwIDQwIiBmaWxsPSJ1cmwoI2dvbGRHcmFkKSIgZmlsdGVyPSJ1cmwoI2JsYWNrT3V0bGluZSkiLz4KPC9zdmc+
"""

# Assuming these exist in your local models.py
from models import (
    build_forecast_dates,
    classify_risk_from_variance,
    fetch_fx_data,
    generate_trading_advice,
    run_arima_model,
    run_garch_model,
    run_ols_model,
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
    page_title="FilterFX - Forex Prediction",
    page_icon="💱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM STYLING (Gold+Outline, Ivory, Black, White)
# ==========================================
st.markdown(
    """
    <style>
    /* 1. Main Background - White */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* 2. Standard Text Colors - Black */
    p, .stMarkdown, .stText, label, .stCaption {
        color: #000000 !important;
    }

    /* 3. Headings & Bold - Gold with Black Outline for visibility */
    h1, h2, h3, h4, strong {
        color: #D4AF37 !important; /* Metallic Gold */
        font-weight: 800 !important;
        /* This adds the black border around the gold text */
        text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
        letter-spacing: 0.5px;
    }
    
    /* 4. Metrics - Ivory Background, Black Label, Gold+Outline Value */
    div[data-testid="stMetric"] {
        background-color: #FFFFF0; /* Ivory */
        border: 2px solid #D4AF37; /* Gold Border */
        padding: 10px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] {
        color: #000000 !important; /* Black label for readability */
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important; /* Gold Value */
        /* Black outline for the big numbers */
        text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
    }
    div[data-testid="stMetricDelta"] {
         color: #000000 !important; /* Keep delta readable */
         font-weight: bold;
    }

    /* 5. Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 2px solid #D4AF37;
    }
    
    /* 6. Buttons - Ivory/Gold theme */
    .stButton>button {
        color: #000000;
        background-color: #FFFFF0; /* Ivory */
        border: 2px solid #D4AF37; /* Gold */
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #FFFFFF;
        border-color: #000000;
    }
    
    /* 7. Containers/Expanders */
    .stExpander, [data-testid="stForm"] {
         border: 1px solid #D4AF37;
         background-color: #FFFFF0;
    }

    /* Prevent content from hiding behind navbar */
    .block-container {
        padding-bottom: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SPLASH SCREEN LOGIC
# ==========================================
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    # Create an empty container for the splash screen
    splash = st.empty()
    
    with splash.container():
        # Centered Logo and Text for Splash using HTML for embedded image
        st.markdown(
            f"""
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh;'>
                <img src="{logo_base64}" width="300" style="margin-bottom: 20px;">
                <h1 style='color: #D4AF37; font-size: 60px; text-shadow: 2px 2px 0 #000;'>FilterFX</h1>
                <p style='color: #000000; font-size: 24px; font-weight: bold;'>Precision Forex Analytics</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Wait for 3 seconds
    time.sleep(3)
    
    # Clear the splash screen
    splash.empty()
    
    # Set state so it doesn't show again this session
    st.session_state.splash_shown = True

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================

# 1. Logo and Title on Main Page
col_logo, col_title = st.columns([1, 4])
with col_logo:
    # Use HTML to embed base64 image
    st.markdown(f'<img src="{logo_base64}" width="120">', unsafe_allow_html=True)
with col_title:
    st.title("FilterFX Dashboard")

st.caption("Real-time forex analysis with econometric models.")

# Bottom Navigation
nav_action = st.radio(
    "",
    ["🏠 Home", "☕ Savings", "🏆 Rankings"],
    horizontal=True,
    label_visibility="collapsed",
)

if nav_action == "☕ Savings":
    st.switch_page("pages/01_Savings.py")
elif nav_action == "🏆 Rankings":
    st.switch_page("pages/02_Rankings.py")

# ==========================================
# HELPER FUNCTIONS (Visualization)
# ==========================================

def create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate):
    """Create professional visualization matching theme"""
    # Set plot style to match theme
    plt.rcParams.update({
        "axes.facecolor": "#FFFFF0", # Ivory background for plot area
        "figure.facecolor": "#FFFFFF", # White background for figure frame
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "#D4AF37", # Gold borders
        "axes.linewidth": 1.5,
        "grid.color": "#D4AF37",
        "grid.alpha": 0.2
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # Plot 1: Forecasts
    ax1 = axes[0]
    subset = df["Rate"].iloc[-180:]
    ax1.plot(subset.index, subset, label="Historical", color="black", linewidth=2)

    last_date = df.index[-1]
    if ols_forecast is not None and isinstance(ols_forecast, (list, np.ndarray)) and len(ols_forecast) > 0:
        num_f = len(ols_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        # Gold Trend Line
        ax1.plot(dates_f, ols_forecast[: len(dates_f)], label="OLS Trend", linestyle="--", color="#D4AF37", linewidth=2.5, alpha=0.9) 
    
    if arima_forecast is not None and hasattr(arima_forecast, '__len__') and len(arima_forecast) > 0:
        num_f = len(arima_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        # Red Forecast Line
        ax1.plot(
            dates_f,
            arima_forecast.values[: len(dates_f)],
            label="ARIMA Forecast",
            linestyle="-",
            color="#C70039",
            linewidth=3,
        )

    ax1.axhline(y=current_rate, color="green", linestyle=":", linewidth=2, label="Current Rate", alpha=0.7)
    
    # Set title with gold color
    ax1.set_title("EUR/INR Exchange Rate Forecast", fontsize=14, fontweight="bold", color="#D4AF37")
    ax1.set_ylabel("Rate (₹ per EUR)")
    
    # Legend styling
    legend = ax1.legend(loc="best", facecolor="#FFFFF0", edgecolor="#D4AF37")
    for text in legend.get_texts():
        text.set_color("black")

    ax1.grid(True)
    
    # Plot 2: Moving Averages
    ax2 = axes[1]
    df["MA_30"] = df["Rate"].rolling(window=30).mean()
    df["MA_90"] = df["Rate"].rolling(window=90).mean()
    subset2 = df.iloc[-250:]
    
    ax2.plot(subset2.index, subset2["Rate"], label="Daily Rate", color="black", alpha=0.3)
    ax2.plot(subset2.index, subset2["MA_30"], label="30-Day MA", color="#D4AF37", linewidth=2.5) # Gold
    ax2.plot(subset2.index, subset2["MA_90"], label="90-Day MA", color="#800000", linewidth=2.5) # Dark Red
    
    ax2.set_title("Trend Analysis (Moving Averages)", fontsize=14, fontweight="bold", color="#D4AF37")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rate (₹ per EUR)")
    
    legend2 = ax2.legend(loc="best", facecolor="#FFFFF0", edgecolor="#D4AF37")
    for text in legend2.get_texts():
        text.set_color("black")
        
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Settings")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 90, 30)
data_period = st.sidebar.selectbox("Historical Data Period", ["2y", "4y", "5y", "10y", "max"])

st.sidebar.divider()
st.sidebar.header("ℹ️ About FilterFX")
st.sidebar.markdown("Precision econometric tools for personal forex management.")

# ==========================================
# MAIN APP: LOAD DATA & RUN MODELS
# ==========================================
with st.spinner("📥 Loading data & running models..."):
    df = fetch_fx_data(period=data_period)
    if df.empty:
        st.error("No data available.")
        st.stop()
    
    current_rate = df["Rate"].iloc[-1]

    ols_model, ols_forecast = run_ols_model(df["Rate"], forecast_days)
    arima_model, arima_forecast = run_arima_model(df["Rate"], forecast_days)
    garch_model, garch_variance = run_garch_model(df["Rate"], forecast_days)

predicted_rate = 0.0
if arima_forecast is not None and len(arima_forecast) > 0:
    predicted_rate = arima_forecast.iloc[-1]

# ==========================================
# MARKET SNAPSHOT
# ==========================================
st.subheader("Market Snapshot")

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.metric("📊 Current Rate", f"₹{current_rate:.4f}")
with metric_col2:
    daily_change = ((df["Rate"].iloc[-1] - df["Rate"].iloc[-2]) / df["Rate"].iloc[-2]) * 100
    st.metric("📈 Daily Change", f"{daily_change:.3f}%", delta=f"{daily_change:.3f}%")

st.caption(f"Latest Data: {df.index[-1].date()}")

# ==========================================
# SMART CURRENCY CONVERTER
# ==========================================
st.divider()
st.subheader("🔄 Smart Converter")

rate_col1, rate_col2 = st.columns(2)

# Using markdown containers to apply the Gold/Ivory/Black theme
with rate_col1:
    st.markdown(
        f"""
        <div style="background-color: #FFFFF0; border: 2px solid #D4AF37; padding: 15px; border-radius: 10px;">
            <strong style="color: #000; font-weight: 700;">Current Rate</strong>
            <h3 style="margin-top: 5px; color: #D4AF37; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;">
                ₹{current_rate:.4f}
            </h3>
        </div>
        """, unsafe_allow_html=True
    )

with rate_col2:
    diff_val = predicted_rate - current_rate
    direction_arrow = "↗" if diff_val > 0 else "↘"
    color_arrow = "green" if diff_val > 0 else "red"
    st.markdown(
        f"""
        <div style="background-color: #FFFFF0; border: 2px solid #D4AF37; padding: 15px; border-radius: 10px;">
            <strong style="color: #000; font-weight: 700;">Forecast (+{forecast_days}d)</strong>
            <h3 style="margin-top: 5px; color: #D4AF37; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;">
                ₹{predicted_rate:.4f} <span style='color:{color_arrow}; font-size: 0.8em; text-shadow: none;'>{direction_arrow}</span>
            </h3>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

c_col1, c_col2 = st.columns([1, 2])
with c_col1:
    convert_direction = st.radio("Direction", ["EUR ➡ INR", "INR ➡ EUR"], label_visibility="collapsed")
with c_col2:
    amount_input = st.number_input("Amount", value=1000.0, step=100.0, label_visibility="collapsed")

if amount_input > 0:
    if convert_direction == "EUR ➡ INR":
        converted_val = amount_input * current_rate
        st.info(f"💶 **{amount_input:,.2f} EUR** = **₹ {converted_val:,.2f} INR**")
    else:
        converted_val = amount_input / current_rate
        st.info(f"🇮🇳 **₹ {amount_input:,.2f} INR** = **€ {converted_val:,.2f} EUR**")

# ==========================================
# TRANSACTION LOGGING
# ==========================================
st.divider()
st.subheader("💾 Log Transaction")

with st.expander("🗑️ Transaction Management"):
    if st.button("Clear All Transactions", use_container_width=True):
        clear_transactions()
        st.rerun()

tx_col_amount, tx_col_date = st.columns(2)
with tx_col_amount:
    amount_inr = st.number_input("Amount (INR)", min_value=0.0, step=1000.0)
with tx_col_date:
    tx_date = st.date_input("Date")

# (Logic for preview and logging remains same as before, just theme applied via CSS)
today = dt.now().date()
is_future = tx_date > today
is_past = tx_date < today
previous_rate = get_rate_for_date(df, tx_date) if is_past else None

if st.button("Log Transaction", type="primary", use_container_width=True) and amount_inr > 0:
    rate_to_use = predicted_rate if is_future else (previous_rate if is_past and previous_rate else current_rate)
    tx_df, savings_eur = append_transaction(tx_date, amount_inr, rate_to_use)
    st.success(f"Transaction Logged at rate: {rate_to_use:.4f}")

st.divider()

# ==========================================
# ANALYSIS RESULTS
# ==========================================
st.header("📊 Analysis Results")

if ols_model and hasattr(ols_model, 'params'):
    ols_dir = "UP ↗" if ols_model.params[1] > 0 else "DOWN ↘"
    ols_str = f"{ols_model.rsquared * 100:.1f}%"
else:
    ols_dir, ols_str = "NEUTRAL ↔", "0.0%"

ols_col, arima_col = st.columns(2)
with ols_col:
    st.markdown(f"**OLS Trend:** {ols_dir} (Conf: {ols_str})")
with arima_col:
    if arima_forecast is not None:
        chg = ((predicted_rate - current_rate) / current_rate) * 100
        st.metric(f"ARIMA Forecast ({forecast_days}D)", f"₹{predicted_rate:.4f}", delta=f"{chg:.2f}%")

# GARCH
st.subheader("⚠️ Risk (GARCH)")
risk_label, risk_desc, avg_vol = classify_risk_from_variance(garch_variance)
st.markdown(
    f"""
    <div style="background-color: #FFFFF0; border-left: 5px solid #D4AF37; padding: 15px;">
        <strong style="font-size: 1.2em;">Risk Level: {risk_label}</strong><br>
        Volatility Index: {avg_vol:.2f}<br>
        <em>{risk_desc}</em>
    </div>
    """, unsafe_allow_html=True
)

st.divider()

# Visualization
st.subheader("📉 Forecast Visualization")
fig = create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate)
st.pyplot(fig)

st.divider()
st.caption("FilterFX - Precision Econometrics")
