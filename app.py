import warnings
import time  # Added for the splash screen delay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import yfinance as yf
from datetime import datetime as dt

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
# CUSTOM STYLING (Gold, Ivory, Black, White)
# ==========================================
st.markdown(
    """
    <style>
    /* 1. Main Background - White */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* 2. Text Colors - Black */
    p, .stMarkdown, .stText, label {
        color: #000000 !important;
    }

    /* 3. Headings (Bold) - Gold */
    h1, h2, h3, h4, strong {
        color: #D4AF37 !important; /* Metallic Gold */
        font-weight: 700 !important;
    }
    
    /* 4. Semi-bold / Metrics - Ivory Background with Gold/Black Text */
    /* Metric containers */
    div[data-testid="stMetric"] {
        background-color: #FFFFF0; /* Ivory */
        border: 1px solid #D4AF37; /* Gold Border */
        padding: 10px;
        border-radius: 5px;
        color: #000000;
    }
    div[data-testid="stMetricLabel"] {
        color: #000000 !important; /* Black label */
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important; /* Gold Value */
    }

    /* 5. Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA; /* Very light grey/white for contrast */
        border-right: 2px solid #D4AF37;
    }
    
    /* 6. Buttons */
    .stButton>button {
        color: #000000;
        background-color: #FFFFF0; /* Ivory */
        border: 1px solid #D4AF37; /* Gold */
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #FFFFFF;
    }

    /* 7. Bottom Nav Styling */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #FFFFF0; /* Ivory */
        border-top: 2px solid #D4AF37; /* Gold */
        padding: 0.5rem 0.75rem;
        display: flex;
        justify-content: space-around;
        z-index: 9999;
    }
    .bottom-nav button {
        background: none;
        border: none;
        color: #000000;
        font-weight: bold;
        font-size: 0.9rem;
        cursor: pointer;
    }
    .bottom-nav button:hover {
        color: #D4AF37; /* Gold hover */
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
        # Centered Logo and Text for Splash
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            try:
                # Ensure you save your image as 'logo.jpg' in the same folder
                st.image("logo.jpg", use_column_width=True) 
            except:
                st.warning("Logo not found. Please save image as 'logo.jpg'")
            
            st.markdown(
                """
                <div style='text-align: center;'>
                    <h1 style='color: #D4AF37; font-size: 50px;'>FilterFX</h1>
                    <p style='color: #000000; font-size: 20px;'>Precision Forex Analytics</p>
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

# 1. Logo on Main Page (Top)
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        st.image("logo.jpg", width=100)
    except:
        pass
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
    """Create professional visualization"""
    # Set plot style to match theme
    plt.rcParams.update({
        "axes.facecolor": "#FFFFF0", # Ivory background for plot
        "figure.facecolor": "#FFFFFF",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "#D4AF37" # Gold borders
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # Plot 1: Forecasts
    ax1 = axes[0]
    subset = df["Rate"].iloc[-180:]
    ax1.plot(subset.index, subset, label="Historical (≈6 Months)", color="black", linewidth=2)

    last_date = df.index[-1]
    if ols_forecast is not None and isinstance(ols_forecast, (list, np.ndarray)) and len(ols_forecast) > 0:
        num_f = len(ols_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        ax1.plot(dates_f, ols_forecast[: len(dates_f)], label="OLS Trend", linestyle="--", color="#D4AF37", alpha=0.9) # Gold Trend
    
    if arima_forecast is not None and hasattr(arima_forecast, '__len__') and len(arima_forecast) > 0:
        num_f = len(arima_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        ax1.plot(
            dates_f,
            arima_forecast.values[: len(dates_f)],
            label="ARIMA Forecast",
            linestyle="-",
            color="red",
            linewidth=2.5,
        )

    ax1.axhline(y=current_rate, color="green", linestyle=":", linewidth=2, label="Current Rate", alpha=0.7)
    ax1.set_title("EUR/INR Exchange Rate Forecast", fontsize=14, fontweight="bold", color="#D4AF37")
    ax1.set_ylabel("Rate (₹ per EUR)")
    ax1.legend(loc="best", facecolor="#FFFFF0", edgecolor="#D4AF37")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Averages
    ax2 = axes[1]
    df["MA_30"] = df["Rate"].rolling(window=30).mean()
    df["MA_90"] = df["Rate"].rolling(window=90).mean()
    subset2 = df.iloc[-250:]
    
    ax2.plot(subset2.index, subset2["Rate"], label="Daily Rate", color="black", alpha=0.4)
    ax2.plot(subset2.index, subset2["MA_30"], label="30-Day MA", color="#D4AF37", linewidth=2) # Gold
    ax2.plot(subset2.index, subset2["MA_90"], label="90-Day MA", color="#800000", linewidth=2) # Dark Red
    ax2.set_title("Trend Analysis (Moving Averages)", fontsize=14, fontweight="bold", color="#D4AF37")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rate (₹ per EUR)")
    ax2.legend(loc="best", facecolor="#FFFFF0", edgecolor="#D4AF37")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==========================================
# SIDEBAR SETTINGS & ABOUT
# ==========================================
st.sidebar.header("⚙️ Settings")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 90, 30)
data_period = st.sidebar.selectbox("Historical Data Period", ["2y", "4y", "5y", "10y", "max"])

# ==========================================
# SIDEBAR - ABOUT SECTION
# ==========================================
st.sidebar.divider()
st.sidebar.header("ℹ️ About FilterFX")
st.sidebar.markdown("""
This is a **EUR/INR Exchange Rate Prediction** tool powered by advanced econometric models.

**📊 Features:**
- **OLS Regression** - Trend direction
- **ARIMA Model** - Future rates
- **GARCH Model** - Market volatility
""")

# ==========================================
# MAIN APP: LOAD DATA
# ==========================================

with st.spinner("📥 Loading EUR/INR data..."):
    df = fetch_fx_data(period=data_period)

if df.empty:
    st.error("No data available. Please try a different period.")
    st.stop()

current_rate = df["Rate"].iloc[-1]

# ==========================================
# RUN MODELS EARLY 
# ==========================================
with st.spinner("🔄 Running econometric models..."):
    ols_model, ols_forecast = run_ols_model(df["Rate"], forecast_days)
    arima_model, arima_forecast = run_arima_model(df["Rate"], forecast_days)
    garch_model, garch_variance = run_garch_model(df["Rate"], forecast_days)

# Get the final predicted value for display
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

st.caption(f"Data points: **{len(df)}** (from {df.index[0].date()})")

# ==========================================
# SMART CURRENCY CONVERTER
# ==========================================
st.divider()
st.subheader("🔄 Smart Currency Converter")

# 1. Rate Dashboard
rate_col1, rate_col2 = st.columns(2)

with rate_col1:
    st.container(border=True).markdown(
        f"**Current Rate**\n### ₹{current_rate:.4f}"
    )

with rate_col2:
    diff_val = predicted_rate - current_rate
    diff_color = "green" if diff_val > 0 else "red"
    direction_arrow = "↗" if diff_val > 0 else "↘"
    
    st.container(border=True).markdown(
        f"**Predicted Rate (+{forecast_days}d)**\n"
        f"### ₹{predicted_rate:.4f} :{diff_color}[{direction_arrow}]"
    )

# 2. Conversion Inputs
st.markdown("##### Convert Amount")
c_col1, c_col2 = st.columns([1, 2])

with c_col1:
    convert_direction = st.radio(
        "Direction", 
        ["EUR ➡ INR", "INR ➡ EUR"],
        label_visibility="collapsed"
    )

with c_col2:
    amount_input = st.number_input(
        "Amount",
        min_value=0.0,
        step=100.0,
        value=1000.0,
        format="%.2f",
        label_visibility="collapsed"
    )

# 3. Calculation
if amount_input > 0:
    if convert_direction == "EUR ➡ INR":
        converted_val = amount_input * current_rate
        st.success(f"💶 **{amount_input:,.2f} EUR** = **₹ {converted_val:,.2f} INR**")
    else:
        converted_val = amount_input / current_rate
        st.success(f"🇮🇳 **₹ {amount_input:,.2f} INR** = **€ {converted_val:,.2f} EUR**")


# ==========================================
# TRANSACTION LOGGING
# ==========================================
st.divider()
st.subheader("💾 Log Transaction")

with st.expander("🗑️ Transaction Management"):
    col_clear, col_stats = st.columns(2)
    with col_clear:
        if st.button("Clear All Transactions", type="secondary", use_container_width=True):
            clear_transactions()
            st.success("✅ Cleared!")
            st.rerun()
    with col_stats:
        stats = get_savings_stats()
        if stats["total_transactions"] > 0:
            st.metric("Total Transactions", stats["total_transactions"])

tx_col_amount, tx_col_date = st.columns(2)

with tx_col_amount:
    amount_inr = st.number_input("Amount in INR", min_value=0.0, step=1000.0, format="%.2f")
with tx_col_date:
    tx_date = st.date_input("Transaction date")

# Show rate preview
today = dt.now().date()
is_future = tx_date > today
is_past = tx_date < today

previous_rate = None
if is_past:
    previous_rate = get_rate_for_date(df, tx_date)

preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    if is_future:
        st.info(f"📅 **Future** (+{(tx_date - today).days} days)")
        st.metric("Predicted Rate", f"₹{predicted_rate:.4f}")
    elif is_past:
        if previous_rate:
            st.info(f"📅 **Past** ({(today - tx_date).days} days ago)")
            st.metric("Hist. Rate", f"₹{previous_rate:.4f}")
        else:
            st.info(f"📅 **Past** (No Data)")
            st.metric("Current Rate", f"₹{current_rate:.4f}")
    else:
        st.info(f"📅 **Today**")
        st.metric("Current Rate", f"₹{current_rate:.4f}")

with preview_col2:
    if amount_inr > 0:
        if is_future:
            rate_to_show = predicted_rate
        elif is_past and previous_rate:
            rate_to_show = previous_rate
        else:
            rate_to_show = current_rate
        eur_preview = amount_inr / rate_to_show if rate_to_show > 0 else 0
        st.metric("EUR You'll Get", f"€{eur_preview:.4f}")

log_button = st.button("Log Transaction", type="primary", use_container_width=True)

if log_button and amount_inr > 0:
    # Logic for determining rate (Same as previous)
    if is_future:
        rate_to_use = predicted_rate
        rate_type = "PREDICTED"
    elif is_past:
        historical_rate = get_rate_for_date(df, tx_date)
        if historical_rate:
            rate_to_use = historical_rate
            rate_type = "HISTORICAL"
        else:
            rate_to_use = current_rate
            rate_type = "CURRENT"
    else:
        rate_to_use = current_rate
        rate_type = "CURRENT"
    
    tx_df, savings_eur = append_transaction(tx_date, amount_inr, rate_to_use)
    eur_now = amount_inr / rate_to_use if rate_to_use > 0 else 0.0

    st.success(f"Transaction Logged! Rate used: {rate_type}")

    with st.expander("View transaction history"):
        st.dataframe(tx_df.sort_values("date", ascending=False), use_container_width=True)

st.divider()

# ==========================================
# ANALYSIS RESULTS
# ==========================================
st.header("📊 Analysis Results")

if ols_model is not None and hasattr(ols_model, 'params'):
    try:
        ols_direction = "UP ↗" if ols_model.params[1] > 1 else "DOWN ↘"
        ols_strength = round(ols_model.rsquared * 100, 1)
    except:
        ols_direction = "NEUTRAL ↔"
        ols_strength = 0.0
else:
    ols_direction = "NEUTRAL ↔"
    ols_strength = 0.0

st.subheader("📈 OLS Trend & 🎯 ARIMA Forecast")

ols_col, arima_col = st.columns(2)

with ols_col:
    st.markdown(f"**Direction:** {ols_direction}")
    st.markdown(f"**Confidence:** {ols_strength}%")

with arima_col:
    if arima_forecast is not None and len(arima_forecast) > 0:
        change_pct = ((predicted_rate - current_rate) / current_rate) * 100
        st.metric(f"Forecast ({forecast_days}D)", f"₹{predicted_rate:.4f}", delta=f"{change_pct:.3f}%")

st.divider()

# GARCH Volatility
st.subheader("⚠️ Risk Assessment (GARCH)")
risk_label, risk_desc, avg_vol = classify_risk_from_variance(garch_variance)
st.info(f"**Volatility Index:** {avg_vol:.2f} | **Risk Level:** {risk_label}\n\n{risk_desc}")

st.divider()

# Visualization
st.subheader("📉 Interactive Rate Chart")
hist_df = df[["Rate"]].iloc[-180:].reset_index().rename(columns={"index": "Date"})
hist_df.rename(columns={hist_df.columns[0]: "Date"}, inplace=True)
st.line_chart(hist_df, x="Date", y="Rate", use_container_width=True)

st.subheader("📊 Detailed Forecast")
fig = create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate)
st.pyplot(fig)

st.divider()

# Footer
st.markdown("""
---
**FilterFX** - Precision Econometrics for Forex
""")
