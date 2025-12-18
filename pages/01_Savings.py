import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transactions import load_transactions, COFFEE_PRICE_INR

st.set_page_config(
    page_title="☕ Savings in Coffees",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Your Savings in Coffees")
st.markdown("**Track your forex savings and see how many coffees you can buy!**")

# ==========================================
# NAVIGATION BUTTONS
# ==========================================
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🏠 Home", use_container_width=True, key="nav_to_home_from_savings"):
        st.switch_page("app.py")

with nav_col2:
    st.button("☕ Savings", use_container_width=True, disabled=True)

with nav_col3:
    if st.button("🏆 Rankings", use_container_width=True, key="nav_to_rankings_from_savings"):
        st.switch_page("pages/02_Rankings.py")

st.divider()

# ==========================================
# LOAD AND CALCULATE SAVINGS
# ==========================================
df = load_transactions()

if df.empty:
    st.warning("📝 No transactions logged yet. Go to the main page and log a transaction first!")
    st.stop()

# Get the base date (first transaction)
base_date = df["date"].min()
base_rate = df[df["date"] == base_date].iloc[0]["rate"]

st.info(f"**Base Date:** {base_date.date()} | **Base Rate:** ₹{base_rate:.4f}")

# Calculate savings for each transaction
df["eur_at_base"] = df["amount_inr"] / base_rate
df["eur_at_actual"] = df["amount_inr"] / df["rate"]
df["eur_saved"] = df["eur_at_base"] - df["eur_at_actual"]
df["coffees_saved"] = (df["eur_saved"] * 100 / COFFEE_PRICE_INR).astype(int)

# Total savings
total_eur_saved = df["eur_saved"].sum()
total_coffees = df["coffees_saved"].sum()
total_amount_inr = df["amount_inr"].sum()

st.divider()

# Overview with animations
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💶 Total EUR Saved",
        f"€{max(0, total_eur_saved):.4f}",
        help="EUR saved from rate fluctuations"
    )

with col2:
    st.metric(
        "☕ Coffee Equivalents",
        f"☕ {total_coffees}",
        help=f"At ₹{COFFEE_PRICE_INR}/cup"
    )

with col3:
    st.metric(
        "💰 Total INR Used",
        f"₹{total_amount_inr:,.2f}",
        help="Total amount exchanged"
    )

with col4:
    avg_rate = df["rate"].mean()
    savings_pct = ((base_rate - avg_rate) / base_rate * 100)
    st.metric(
        "📊 Rate Improvement",
        f"{savings_pct:.2f}%",
        help="Better rate than base date"
    )

st.divider()

# Coffee Animation Section
st.subheader("☕ Your Coffee Rewards!")

# Create animated coffee display
if total_coffees > 0:
    # Coffee emoji animation
    coffee_count = min(total_coffees, 20)  # Cap at 20 for display
    coffee_emojis = " ".join(["☕"] * coffee_count)
    if total_coffees > 20:
        coffee_emojis += f" ... +{total_coffees - 20} more!"
    
    st.success(f"### {coffee_emojis}")
    
    # Progress bar
    st.progress(min(total_coffees / 50, 1.0))
    
    # Fun messages based on coffee count
    if total_coffees < 5:
        msg = "☕ Every cup counts! Keep saving!"
    elif total_coffees < 20:
        msg = "🎉 Nice! You've earned some coffee treats!"
    elif total_coffees < 50:
        msg = "🚀 Impressive! You're a forex saving pro!"
    else:
        msg = "🏆 Wow! You could open a café!"
    
    st.success(msg)
else:
    st.info("📉 Rates aren't in your favor yet. Wait for better rates!")

st.divider()

# Detailed breakdown by transaction
st.subheader("📊 Transaction Breakdown")

display_df = df[["date", "amount_inr", "rate", "eur_at_base", "eur_at_actual", "eur_saved", "coffees_saved"]].copy()
display_df.columns = ["Date", "Amount (₹)", "Rate (₹/EUR)", "EUR (Base Rate)", "EUR (Actual)", "EUR Saved", "Coffees ☕"]

# Format for display
st.dataframe(
    display_df.sort_values("Date", ascending=False).style.format({
        "Amount (₹)": "{:,.2f}",
        "Rate (₹/EUR)": "{:.4f}",
        "EUR (Base Rate)": "{:.4f}",
        "EUR (Actual)": "{:.4f}",
        "EUR Saved": "{:.4f}"
    }).highlight_max(subset=["EUR Saved"], color='lightgreen').highlight_min(subset=["EUR Saved"], color='lightcoral'),
    use_container_width=True
)

st.divider()

# Savings timeline
st.subheader("📈 Savings Growth Timeline")

df_sorted = df.sort_values("date")
df_sorted["cumulative_savings"] = df_sorted["eur_saved"].cumsum()
df_sorted["cumulative_coffees"] = df_sorted["coffees_saved"].cumsum()

# Create chart data
chart_data = df_sorted[["date", "cumulative_savings", "cumulative_coffees"]].copy()
chart_data.columns = ["Date", "EUR Saved", "Coffees ☕"]
chart_data = chart_data.set_index("Date")

st.line_chart(chart_data)

st.divider()

# Period-wise savings
st.subheader("⏰ Savings by Period")

today = datetime.now().date()
today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
week_start = today_start - timedelta(days=today_start.weekday())
month_start = today_start.replace(day=1)

today_df = df[df["date"].dt.date == today]
week_df = df[df["date"] >= week_start]
month_df = df[df["date"] >= month_start]

period_col1, period_col2, period_col3 = st.columns(3)

with period_col1:
    today_eur = today_df["eur_saved"].sum()
    today_coffee = int(today_df["coffees_saved"].sum())
    st.info(f"**Today**\n€{today_eur:.4f}\n☕ {today_coffee}")

with period_col2:
    week_eur = week_df["eur_saved"].sum()
    week_coffee = int(week_df["coffees_saved"].sum())
    st.info(f"**This Week**\n€{week_eur:.4f}\n☕ {week_coffee}")

with period_col3:
    month_eur = month_df["eur_saved"].sum()
    month_coffee = int(month_df["coffees_saved"].sum())
    st.info(f"**This Month**\n€{month_eur:.4f}\n☕ {month_coffee}")

st.divider()

# Statistics
st.subheader("📊 Statistics")

stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    best_rate = df["rate"].min()
    best_date = df[df["rate"] == best_rate]["date"].iloc[0]
    st.metric("🎯 Best Rate", f"₹{best_rate:.4f}", f"on {best_date.date()}")

with stat_col2:
    worst_rate = df["rate"].max()
    worst_date = df[df["rate"] == worst_rate]["date"].iloc[0]
    st.metric("⚠️ Worst Rate", f"₹{worst_rate:.4f}", f"on {worst_date.date()}")

with stat_col3:
    avg_rate = df["rate"].mean()
    st.metric("📈 Average Rate", f"₹{avg_rate:.4f}", help="Mean of all transactions")

st.divider()

st.info("""
**💡 How Savings Are Calculated:**

Your savings are calculated by comparing the EUR you'd get at your **base date rate** 
vs. the EUR you actually got at the **transaction rate**.

**Formula:** EUR Saved = (Amount ₹ / Base Rate) - (Amount ₹ / Actual Rate)

**Coffee Equivalent:** Your EUR savings in INR ÷ Coffee Price (₹250)

The better your actual rates compared to the base rate, the more you save! ☕
""")

