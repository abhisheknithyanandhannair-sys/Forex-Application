import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transactions import load_transactions, COFFEE_PRICE_INR

st.set_page_config(
    page_title="🏆 User Rankings",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 User Rankings")
st.markdown("**Compare your forex savings with friends!**")

# ==========================================
# NAVIGATION BUTTONS
# ==========================================
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🏠 Home", use_container_width=True, key="nav_to_home_from_rankings"):
        st.switch_page("app.py")

with nav_col2:
    if st.button("☕ Savings", use_container_width=True, key="nav_to_savings_from_rankings"):
        st.switch_page("pages/01_Savings.py")

with nav_col3:
    st.button("🏆 Rankings", use_container_width=True, disabled=True)

st.divider()

# ==========================================
# LOAD USER TRANSACTIONS
# ==========================================
user_tx = load_transactions()

if user_tx.empty:
    st.warning("📝 No transactions logged yet. Log some transactions first to see your ranking!")
    st.stop()

# Calculate your stats
your_base_date = user_tx["date"].min()
your_base_rate = user_tx[user_tx["date"] == your_base_date].iloc[0]["rate"]

your_tx = user_tx.copy()
your_tx["eur_at_base"] = your_tx["amount_inr"] / your_base_rate
your_tx["eur_at_actual"] = your_tx["amount_inr"] / your_tx["rate"]
your_tx["eur_saved"] = your_tx["eur_at_base"] - your_tx["eur_at_actual"]
your_tx["coffees_saved"] = (your_tx["eur_saved"] * 100 / COFFEE_PRICE_INR).astype(int)

your_total_eur = your_tx["eur_saved"].sum()
your_total_inr = your_tx["amount_inr"].sum()
your_total_coffees = your_tx["coffees_saved"].sum()
your_per_currency_savings = your_total_eur / your_total_inr if your_total_inr > 0 else 0

# Simulated friend data
np.random.seed(42)


def generate_friend_data(name, num_transactions_range=(5, 20)):
    """Generate simulated friend transaction data"""
    num_tx = np.random.randint(*num_transactions_range)
    base_rate = np.random.uniform(85, 95)
    
    dates = pd.date_range(end=datetime.now(), periods=num_tx, freq='D')
    rates = base_rate + np.random.normal(0, 1, num_tx)
    amounts = np.random.uniform(5000, 50000, num_tx)
    
    data = {
        "date": dates,
        "rate": rates,
        "amount_inr": amounts
    }
    
    df = pd.DataFrame(data)
    
    # Calculate savings
    first_rate = df["rate"].iloc[0]
    df["eur_at_base"] = df["amount_inr"] / first_rate
    df["eur_at_actual"] = df["amount_inr"] / df["rate"]
    df["eur_saved"] = df["eur_at_base"] - df["eur_at_actual"]
    
    total_eur = df["eur_saved"].sum()
    total_inr = df["amount_inr"].sum()
    total_coffees = int((df["eur_saved"].sum() * 100 / COFFEE_PRICE_INR))
    per_currency = total_eur / total_inr if total_inr > 0 else 0
    
    return {
        "name": name,
        "transactions": len(df),
        "total_inr": total_inr,
        "total_eur": max(0, total_eur),
        "total_coffees": max(0, total_coffees),
        "per_currency_savings": per_currency
    }

# Generate friend rankings
friend_names = ["Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace", "Henry"]
friends_data = [generate_friend_data(name) for name in friend_names]

# Add your data
your_data = {
    "name": "👤 You",
    "transactions": len(your_tx),
    "total_inr": your_total_inr,
    "total_eur": max(0, your_total_eur),
    "total_coffees": max(0, your_total_coffees),
    "per_currency_savings": your_per_currency_savings
}

all_users = friends_data + [your_data]
rankings_df = pd.DataFrame(all_users).sort_values("per_currency_savings", ascending=False)
rankings_df["rank"] = range(1, len(rankings_df) + 1)

st.divider()

# Your Position
st.subheader("📊 Your Position")

your_rank = rankings_df[rankings_df["name"] == "👤 You"]["rank"].values[0]
total_users = len(rankings_df)
your_percentile = ((total_users - your_rank) / total_users) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏅 Your Rank", f"#{your_rank}/{total_users}")

with col2:
    st.metric("📊 Your Percentile", f"Top {your_percentile:.0f}%")

with col3:
    st.metric("💶 Per-Currency Savings", f"€{your_per_currency_savings:.6f}")

with col4:
    st.metric("☕ Your Coffees", f"☕ {your_total_coffees}")

st.divider()

# Main Leaderboard
st.subheader("🏆 Leaderboard (Ranked by Per-Currency Savings)")

# Display leaderboard
display_rankings = rankings_df[["rank", "name", "transactions", "total_inr", "total_eur", "per_currency_savings", "total_coffees"]].copy()
display_rankings.columns = ["🏅 Rank", "👤 User", "📝 Txns", "💰 Total INR", "💶 EUR Saved", "📈 Per ₹1", "☕ Coffees"]

# Format for display
formatted_df = display_rankings.style.format({
    "💰 Total INR": "{:,.2f}",
    "💶 EUR Saved": "{:.4f}",
    "📈 Per ₹1": "{:.8f}"
}).background_gradient(subset=["📈 Per ₹1"], cmap="RdYlGn")

st.dataframe(formatted_df, use_container_width=True)

st.divider()

# Period-wise Rankings
st.subheader("⏰ Period-wise Rankings")

tab1, tab2, tab3 = st.tabs(["📅 This Month", "📆 This Week", "📅 Today"])

def get_period_rankings(days):
    """Get rankings for a specific period"""
    now = datetime.now()
    period_start = now - timedelta(days=days)
    
    period_tx = user_tx[user_tx["date"] >= period_start]
    
    if period_tx.empty:
        return None
    
    period_base_rate = user_tx[user_tx["date"] == user_tx["date"].min()].iloc[0]["rate"]
    
    period_tx_copy = period_tx.copy()
    period_tx_copy["eur_at_base"] = period_tx_copy["amount_inr"] / period_base_rate
    period_tx_copy["eur_at_actual"] = period_tx_copy["amount_inr"] / period_tx_copy["rate"]
    period_tx_copy["eur_saved"] = period_tx_copy["eur_at_base"] - period_tx_copy["eur_at_actual"]
    
    total_eur = period_tx_copy["eur_saved"].sum()
    total_inr = period_tx_copy["amount_inr"].sum()
    per_currency = total_eur / total_inr if total_inr > 0 else 0
    coffees = int(total_eur * 100 / COFFEE_PRICE_INR)
    
    return {
        "total_eur": max(0, total_eur),
        "total_inr": total_inr,
        "per_currency_savings": per_currency,
        "total_coffees": max(0, coffees)
    }

with tab1:
    month_stats = get_period_rankings(30)
    if month_stats:
        st.metric("💶 EUR Saved This Month", f"€{month_stats['total_eur']:.4f}")
        st.metric("📈 Per ₹1 This Month", f"€{month_stats['per_currency_savings']:.8f}")
        st.metric("☕ Coffees This Month", f"☕ {month_stats['total_coffees']}")
    else:
        st.info("No transactions this month")

with tab2:
    week_stats = get_period_rankings(7)
    if week_stats:
        st.metric("💶 EUR Saved This Week", f"€{week_stats['total_eur']:.4f}")
        st.metric("📈 Per ₹1 This Week", f"€{week_stats['per_currency_savings']:.8f}")
        st.metric("☕ Coffees This Week", f"☕ {week_stats['total_coffees']}")
    else:
        st.info("No transactions this week")

with tab3:
    today_stats = get_period_rankings(1)
    if today_stats:
        st.metric("💶 EUR Saved Today", f"€{today_stats['total_eur']:.4f}")
        st.metric("📈 Per ₹1 Today", f"€{today_stats['per_currency_savings']:.8f}")
        st.metric("☕ Coffees Today", f"☕ {today_stats['total_coffees']}")
    else:
        st.info("No transactions today")

st.divider()

# Community Statistics
st.subheader("📈 Community Statistics")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric("👥 Total Users", total_users)

with stat_col2:
    avg_per_currency = rankings_df["per_currency_savings"].mean()
    st.metric("📊 Avg Per ₹1", f"€{avg_per_currency:.8f}")

with stat_col3:
    total_community_eur = rankings_df["total_eur"].sum()
    st.metric("💶 Community EUR", f"€{total_community_eur:.4f}")

with stat_col4:
    total_community_coffees = rankings_df["total_coffees"].sum()
    st.metric("☕ Community Coffees", f"☕ {total_community_coffees}")

st.divider()

# How Ranking Works
st.info("""
**🏆 How Rankings Work:**

Rankings are based on **Per-Currency Savings** (EUR saved per ₹1 used), not total amounts.

**Formula:** Per ₹1 Savings = Total EUR Saved ÷ Total INR Amount Used

This ensures fair comparison regardless of how much each person trades. 
A small trader with great rates can rank higher than a large trader with mediocre rates!

**Your Performance:**
- You've saved **€{:.6f}** for every **₹1** exchanged
- This puts you at **#{}/{}** among {} users
- You've earned **☕ {}** with your savings! 🎉
""".format(your_per_currency_savings, your_rank, total_users, total_users, your_total_coffees))

