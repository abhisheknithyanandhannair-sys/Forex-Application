import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import yfinance as yf

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
    page_title="EUR/INR Forex Prediction",
    page_icon="💱",
    layout="centered",  # more comfortable on mobile
    initial_sidebar_state="expanded",
)

st.title("💱 EUR/INR Exchange Rate Prediction")
st.caption("Real-time forex analysis with econometric models – tuned for mobile screens.")

st.markdown(
    """
    <style>
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #0e1117;
        border-top: 1px solid #262730;
        padding: 0.5rem 0.75rem;
        display: flex;
        justify-content: space-around;
        z-index: 9999;
    }

    .bottom-nav button {
        background: none;
        border: none;
        color: white;
        font-size: 0.9rem;
        cursor: pointer;
    }

    .bottom-nav button:hover {
        color: #00ffcc;
    }

    /* Prevent content from hiding behind navbar */
    .block-container {
        padding-bottom: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
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
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # Plot 1: Forecasts
    ax1 = axes[0]
    subset = df["Rate"].iloc[-180:]
    ax1.plot(subset.index, subset, label="Historical (≈6 Months)", color="black", linewidth=2)

    last_date = df.index[-1]
    if ols_forecast is not None and isinstance(ols_forecast, (list, np.ndarray)) and len(ols_forecast) > 0:
        num_f = len(ols_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        ax1.plot(dates_f, ols_forecast[: len(dates_f)], label="OLS Trend", linestyle="--", color="blue", alpha=0.7)
    
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
    ax1.set_title("EUR/INR Exchange Rate Forecast", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Rate (₹ per EUR)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Averages
    ax2 = axes[1]
    df["MA_30"] = df["Rate"].rolling(window=30).mean()
    df["MA_90"] = df["Rate"].rolling(window=90).mean()
    subset2 = df.iloc[-250:]
    
    ax2.plot(subset2.index, subset2["Rate"], label="Daily Rate", color="black", alpha=0.4)
    ax2.plot(subset2.index, subset2["MA_30"], label="30-Day MA", color="blue", linewidth=2)
    ax2.plot(subset2.index, subset2["MA_90"], label="90-Day MA", color="red", linewidth=2)
    ax2.set_title("Trend Analysis (Moving Averages)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rate (₹ per EUR)")
    ax2.legend(loc="best")
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
st.sidebar.header("ℹ️ About This App")
st.sidebar.markdown("""
This is a **EUR/INR Exchange Rate Prediction** tool powered by advanced econometric models:

**📊 Features:**
- **OLS Regression** - Identifies underlying trend direction
- **ARIMA Model** - Forecasts future exchange rates
- **GARCH Model** - Assesses market volatility & risk
- **Smart Converter** - Real-time currency conversion
- **Transaction Logger** - Track your exchange rates over time

**💡 Use Cases:**
- Monitor EUR/INR trends
- Make informed trading decisions
- Plan international transfers
- Understand market volatility

**📈 Data:**
- Real-time forex data
- EUR/INR rates (₹ per 1 EUR)
- Updated daily

**⚠️ Disclaimer:**
For educational and analysis purposes. Not financial advice.
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
# (Moved up so we can use predictions in the converter)
# ==========================================
with st.spinner("🔄 Running econometric models for prediction..."):
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
    st.metric("📊 Current Rate (₹ / EUR)", f"{current_rate:.4f}")

with metric_col2:
    daily_change = ((df["Rate"].iloc[-1] - df["Rate"].iloc[-2]) / df["Rate"].iloc[-2]) * 100
    st.metric("📈 Daily Change", f"{daily_change:.3f}%", delta=f"{daily_change:.3f}%")

st.caption(f"Data points: **{len(df)}** (from {df.index[0].date()})")

# ==========================================
# 🆕 IMPROVED CURRENCY CONVERTER
# ==========================================
st.divider()
st.subheader("🔄 Smart Currency Converter")

# 1. Rate Dashboard (Current vs Predicted)
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
    # Bi-directional selection
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

# 3. Calculation & Display
if amount_input > 0:
    if convert_direction == "EUR ➡ INR":
        # EUR to INR: Multiply
        converted_val = amount_input * current_rate
        st.success(f"💶 **{amount_input:,.2f} EUR** is approximately **₹ {converted_val:,.2f} INR**")
    else:
        # INR to EUR: Divide
        converted_val = amount_input / current_rate
        st.success(f"🇮🇳 **₹ {amount_input:,.2f} INR** is approximately **€ {converted_val:,.2f} EUR**")


# ==========================================
# TRANSACTION LOGGING
# ==========================================
st.divider()
st.subheader("💾 Log Transaction")
st.caption("Track how many EUR you receive now vs. your last logged rate.")

# Clear log button in expander
with st.expander("🗑️ Transaction Management"):
    col_clear, col_stats = st.columns(2)
    with col_clear:
        if st.button("Clear All Transactions", type="secondary", use_container_width=True):
            clear_transactions()
            st.success("✅ All transactions cleared!")
            st.rerun()
    with col_stats:
        stats = get_savings_stats()
        if stats["total_transactions"] > 0:
            st.metric("Total Transactions", stats["total_transactions"])

tx_col_amount, tx_col_date = st.columns(2)

with tx_col_amount:
    amount_inr = st.number_input(
        "Amount in INR",
        min_value=0.0,
        step=1000.0,
        format="%.2f",
    )
with tx_col_date:
    tx_date = st.date_input("Transaction date")

# Show rate preview based on date
from datetime import datetime as dt
today = dt.now().date()
is_future = tx_date > today
is_past = tx_date < today

# Get rate for past dates from historical data
previous_rate = None
if is_past:
    previous_rate = get_rate_for_date(df, tx_date)

preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    if is_future:
        st.info(f"📅 **Future Date** (in {(tx_date - today).days} days)")
        st.metric("Using Predicted Rate", f"₹{predicted_rate:.4f}", delta=f"{((predicted_rate - current_rate) / current_rate * 100):.2f}%")
    elif is_past:
        if previous_rate:
            st.info(f"📅 **Past Date** ({(today - tx_date).days} days ago)")
            st.metric("Using Historical Rate", f"₹{previous_rate:.4f}", delta=f"{((previous_rate - current_rate) / current_rate * 100):.2f}%")
        else:
            st.info(f"📅 **Past Date** (no data for this date)")
            st.metric("Using Current Rate", f"₹{current_rate:.4f}")
    else:
        st.info(f"📅 **Today's** Transaction")
        st.metric("Using Current Rate", f"₹{current_rate:.4f}")

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
    # Check if transaction date is in the future or past
    from datetime import datetime as dt
    today = dt.now().date()
    is_future = tx_date > today
    is_past = tx_date < today
    
    # Determine which rate to use
    if is_future:
        rate_to_use = predicted_rate
        rate_type = "PREDICTED"
    elif is_past:
        # Get historical rate from forex data
        historical_rate = get_rate_for_date(df, tx_date)
        if historical_rate:
            rate_to_use = historical_rate
            rate_type = "HISTORICAL"
        else:
            rate_to_use = current_rate
            rate_type = "CURRENT"
    else:
        # Today's transaction
        rate_to_use = current_rate
        rate_type = "CURRENT"
    
    tx_df, savings_eur = append_transaction(tx_date, amount_inr, rate_to_use)
    eur_now = amount_inr / rate_to_use if rate_to_use > 0 else 0.0

    if savings_eur is None:
        st.success(
            f"Logged first transaction: **₹{amount_inr:,.2f} ➝ {eur_now:,.4f} EUR** "
            f"at **{rate_type}** rate **{rate_to_use:.4f} ₹/EUR**."
        )
    else:
        sign = "more" if savings_eur > 0 else "less"
        st.success(
            f"Logged transaction: **₹{amount_inr:,.2f} ➝ {eur_now:,.4f} EUR**.\n\n"
            f"Compared to your **previous logged rate**, you receive "
            f"**{abs(savings_eur):.4f} EUR {sign}** for the same INR amount.\n\n"
            f"*Rate used: **{rate_type}** ({rate_to_use:.4f})*"
        )

    with st.expander("View transaction history"):
        st.dataframe(tx_df.sort_values("date", ascending=False), use_container_width=True)

st.divider()

# ==========================================
# ANALYSIS RESULTS (Models already ran above)
# ==========================================
st.header("📊 Analysis Results")

# Safe parameter access for OLS
if ols_model is not None and hasattr(ols_model, 'params'):
    try:
        ols_direction = "UP ↗" if ols_model.params[1] > 1 else "DOWN ↘"
        ols_strength = round(ols_model.rsquared * 100, 1)
    except (KeyError, IndexError, TypeError):
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
    with st.expander("View OLS Model Summary"):
        st.write(ols_model.summary())

with arima_col:
    if arima_forecast is not None and len(arima_forecast) > 0:
        change_pct = ((predicted_rate - current_rate) / current_rate) * 100
        st.metric(f"Predicted Rate ({forecast_days}D)", f"₹{predicted_rate:.4f}", delta=f"{change_pct:.3f}%")
        with st.expander("View ARIMA Model Summary"):
            st.write(arima_model.summary())

st.divider()

# GARCH Volatility
st.subheader("⚠️ Risk Assessment (GARCH Volatility)")

risk_label, risk_desc, avg_vol = classify_risk_from_variance(garch_variance)
st.info(f"**Volatility Index:** {avg_vol:.2f}\n**Risk Level:** {risk_label}\n{risk_desc}")

st.divider()

# Trading Advice
st.subheader("💡 Trading Recommendation")
advice = generate_trading_advice(
    ols_direction,
    risk_label,
    current_rate,
    predicted_rate,
)
st.warning(advice)

st.divider()

# Visualization
st.subheader("📉 Interactive Rate Chart (Last 6 Months)")
hist_df = df[["Rate"]].iloc[-180:].reset_index().rename(columns={"index": "Date"})
hist_df.rename(columns={hist_df.columns[0]: "Date"}, inplace=True)
st.line_chart(hist_df, x="Date", y="Rate", use_container_width=True)

st.subheader("📊 Forecast & Trend (Detailed)")
fig = create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate)
st.pyplot(fig)

st.divider()

# Historical Data Table
st.subheader("📋 Historical Data (Last 20 Days)")
st.dataframe(df[['Rate']].tail(20).style.format({"Rate": "{:.6f}"}))

# Footer
st.divider()
st.markdown("""
---
**About This App:**
- Uses **OLS** for trend analysis
- Uses **ARIMA(5,1,0)** for price forecasting
- Uses **GARCH(1,1)** for volatility assessment
- Data: EUR/INR exchange rates (₹ per 1 EUR)
- Developed for financial analysis and educational purposes
""")
