import streamlit as st
import pandas as pd
from transactions import load_transactions, get_savings_stats, COFFEE_PRICE_INR

st.set_page_config(
    page_title="Savings in Coffees",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Your Savings in Coffees")
st.markdown("**See how many coffees you can buy with your forex savings!**")

# Get savings stats
stats = get_savings_stats()

if stats["total_transactions"] == 0:
    st.warning("No transactions logged yet. Go to the main page and log a transaction first!")
    st.stop()

# Load data
df = load_transactions()

# Calculate savings in EUR
first_rate = df["rate"].iloc[0]
last_rate = df["rate"].iloc[-1]
total_inr = df["amount_inr"].sum()
total_eur_saved = (total_inr / first_rate) - (total_inr / last_rate)

# Calculate coffees
coffees_total = int(total_eur_saved * 100 / COFFEE_PRICE_INR)  # Convert EUR to INR (approximate)
coffees_today = stats["coffees_today"]
coffees_week = stats["coffees_week"]
coffees_month = stats["coffees_month"]

# Display savings overview
st.divider()
st.subheader("📊 Your Savings Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Saved (EUR)",
        f"€{max(0, total_eur_saved):.2f}",
        help="EUR saved from rate fluctuations"
    )

with col2:
    st.metric(
        "Total Transactions",
        stats["total_transactions"],
        help="Number of transactions logged"
    )

with col3:
    avg_rate = df["rate"].mean()
    st.metric(
        "Average Rate",
        f"₹{avg_rate:.2f}",
        delta=f"{((last_rate - avg_rate) / avg_rate * 100):.2f}%"
    )

with col4:
    best_rate = df["rate"].min()
    worst_rate = df["rate"].max()
    st.metric(
        "Rate Range",
        f"₹{best_rate:.2f} - ₹{worst_rate:.2f}",
        delta=f"{((worst_rate - best_rate) / best_rate * 100):.2f}%"
    )

st.divider()

# Coffee savings visualization
st.subheader("☕ Coffee Equivalents")
st.markdown(f"*At ₹{COFFEE_PRICE_INR} per coffee*")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if coffees_today > 0:
        st.success(f"**TODAY** ☕ {coffees_today}")
    else:
        st.info("**TODAY** ☕ 0")

with col2:
    if coffees_week > 0:
        st.success(f"**THIS WEEK** ☕ {coffees_week}")
    else:
        st.info("**THIS WEEK** ☕ 0")

with col3:
    if coffees_month > 0:
        st.success(f"**THIS MONTH** ☕ {coffees_month}")
    else:
        st.info("**THIS MONTH** ☕ 0")

with col4:
    if coffees_total > 0:
        st.success(f"**TOTAL** ☕ {coffees_total}")
    else:
        st.info("**TOTAL** ☕ 0")

st.divider()

# Detailed breakdown
st.subheader("📈 Savings Breakdown")

breakdown_col1, breakdown_col2 = st.columns(2)

with breakdown_col1:
    st.markdown("### EUR Savings")
    savings_data = {
        "Period": ["Today", "This Week", "This Month", "Total"],
        "EUR Saved": [
            f"€{max(0, stats['today_savings']):.4f}",
            f"€{max(0, stats['week_savings']):.4f}",
            f"€{max(0, stats['month_savings']):.4f}",
            f"€{max(0, total_eur_saved):.4f}"
        ]
    }
    savings_df = pd.DataFrame(savings_data)
    st.dataframe(savings_df, width='stretch')

with breakdown_col2:
    st.markdown("### Equivalent Coffees")
    coffee_data = {
        "Period": ["Today", "This Week", "This Month", "Total"],
        "Coffees ☕": [
            coffees_today,
            coffees_week,
            coffees_month,
            coffees_total
        ]
    }
    coffee_df = pd.DataFrame(coffee_data)
    st.dataframe(coffee_df, width='stretch')

st.divider()

# Transaction history
st.subheader("📋 Transaction History")

# Add calculations to dataframe
display_df = df.copy()
display_df["EUR Equivalent"] = display_df["amount_inr"] / display_df["rate"]
display_df["Coffees"] = (display_df["amount_inr"] / COFFEE_PRICE_INR).astype(int)
display_df = display_df[["date", "amount_inr", "rate", "EUR Equivalent", "Coffees"]]
display_df.columns = ["Date", "Amount (₹)", "Rate (₹/EUR)", "EUR Equivalent", "Coffees ☕"]

st.dataframe(
    display_df.sort_values("Date", ascending=False).style.format({
        "Amount (₹)": "{:.2f}",
        "Rate (₹/EUR)": "{:.4f}",
        "EUR Equivalent": "{:.4f}"
    }),
    width='stretch'
)

st.divider()

# Info section
st.info("""
**💡 How Savings Are Calculated:**

Your savings come from buying EUR at different rates. If you buy EUR when rates are high (bad for you) 
vs. when rates are low (good for you), you've "saved" money by averaging a better rate.

The "coffee equivalent" shows how much of your savings could be spent on coffee in India at ₹250/cup.
""")
