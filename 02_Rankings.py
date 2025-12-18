import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transactions import load_transactions, COFFEE_PRICE_INR

st.set_page_config(
    page_title="User Rankings",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 User Rankings")
st.markdown("**Compare your forex savings with other users!**")

# Note about demo data
st.info("""
🎯 **Demo Mode**: This shows simulated user rankings. In a real multi-user app, 
each user's data would be stored on a server. Here we simulate multiple users for demonstration.
""")

def get_simulated_users():
    """
    Generate simulated user data for demonstration
    This would come from a database in a real app
    """
    np.random.seed(42)
    
    users = []
    user_names = ["You", "Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace", "Henry", "Ivy"]
    
    for name in user_names:
        # Simulate transaction data
        num_transactions = np.random.randint(5, 20)
        base_rate = np.random.uniform(85, 95)
        
        total_inr = np.random.uniform(5000, 50000)
        avg_rate = base_rate + np.random.uniform(-2, 2)
        
        # Calculate savings
        eur_at_best = total_inr / (base_rate - 2)
        eur_at_avg = total_inr / avg_rate
        savings_eur = eur_at_best - eur_at_avg
        
        coffees = int(savings_eur * 100 / COFFEE_PRICE_INR)
        
        users.append({
            "User": name,
            "Transactions": num_transactions,
            "Total INR": total_inr,
            "Avg Rate": avg_rate,
            "EUR Saved": max(0, savings_eur),
            "Coffees": max(0, coffees)
        })
    
    return pd.DataFrame(users)

# Load real user data if available
real_df = load_transactions()
has_real_data = not real_df.empty

# Get simulated data
simulated = get_simulated_users()

if has_real_data:
    # Add real user data to rankings
    first_rate = real_df["rate"].iloc[0]
    last_rate = real_df["rate"].iloc[-1]
    total_inr = real_df["amount_inr"].sum()
    eur_saved = (total_inr / first_rate) - (total_inr / last_rate)
    coffees = int(eur_saved * 100 / COFFEE_PRICE_INR)
    
    simulated.loc[simulated["User"] == "You", "Transactions"] = len(real_df)
    simulated.loc[simulated["User"] == "You", "Total INR"] = total_inr
    simulated.loc[simulated["User"] == "You", "Avg Rate"] = real_df["rate"].mean()
    simulated.loc[simulated["User"] == "You", "EUR Saved"] = max(0, eur_saved)
    simulated.loc[simulated["User"] == "You", "Coffees"] = max(0, coffees)

# Sort by EUR Saved
rankings = simulated.sort_values("EUR Saved", ascending=False).reset_index(drop=True)
rankings["Rank"] = range(1, len(rankings) + 1)

st.divider()

# Overall Rankings
st.subheader("📊 Overall Rankings (All Time)")

col1, col2, col3 = st.columns(3)

with col1:
    top_saver = rankings.iloc[0]
    st.metric(
        "🥇 Top Saver",
        top_saver["User"],
        f"€{top_saver['EUR Saved']:.2f}"
    )

with col2:
    avg_saved = rankings["EUR Saved"].mean()
    st.metric(
        "📊 Average Savings",
        f"€{avg_saved:.2f}",
        f"{len(rankings)} users"
    )

with col3:
    total_saved = rankings["EUR Saved"].sum()
    total_coffees = rankings["Coffees"].sum()
    st.metric(
        "☕ Total Coffees Community",
        total_coffees,
        f"€{total_saved:.2f}"
    )

st.divider()

# Rankings table
st.subheader("🏅 Leaderboard")

display_rankings = rankings[["Rank", "User", "Transactions", "EUR Saved", "Coffees"]].copy()
display_rankings.columns = ["🏅 Rank", "👤 User", "📝 Transactions", "💶 EUR Saved", "☕ Coffees"]

# Style the table
styled_table = st.dataframe(
    display_rankings.style.format({
        "💶 EUR Saved": "{:.2f}",
    }).background_gradient(subset=["💶 EUR Saved"], cmap="RdYlGn"),
    use_container_width=True
)

st.divider()

# Time period comparisons
st.subheader("⏰ Period Comparisons")

tab1, tab2, tab3 = st.tabs(["📅 This Month", "📆 This Week", "📅 Today"])

def get_period_rankings(days):
    """Get rankings for a specific period"""
    if not has_real_data:
        # Simulated data - random variations
        period_data = simulated.copy()
        period_data["EUR Saved"] = period_data["EUR Saved"] * np.random.uniform(0.1, 0.5, len(period_data))
        period_data["Coffees"] = (period_data["EUR Saved"] * 100 / COFFEE_PRICE_INR).astype(int)
        return period_data.sort_values("EUR Saved", ascending=False)
    else:
        # Real data filtering
        now = datetime.now()
        period_start = now - timedelta(days=days)
        period_df = real_df[real_df["date"] >= period_start]
        
        if period_df.empty:
            return simulated.copy()
        
        total_inr = period_df["amount_inr"].sum()
        first_rate = period_df["rate"].iloc[0]
        last_rate = period_df["rate"].iloc[-1]
        eur_saved = (total_inr / first_rate) - (total_inr / last_rate)
        
        result = simulated.copy()
        result.loc[result["User"] == "You", "EUR Saved"] = max(0, eur_saved)
        result.loc[result["User"] == "You", "Coffees"] = max(0, int(eur_saved * 100 / COFFEE_PRICE_INR))
        return result.sort_values("EUR Saved", ascending=False)

with tab1:
    month_rankings = get_period_rankings(30)
    month_rankings["Rank"] = range(1, len(month_rankings) + 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🥇 Top This Month",
            month_rankings.iloc[0]["User"],
            f"€{month_rankings.iloc[0]['EUR Saved']:.2f}"
        )
    with col2:
        st.metric(
            "Your Position",
            f"#{month_rankings[month_rankings['User'] == 'You']['Rank'].values[0] if has_real_data else 1}",
            f"€{month_rankings[month_rankings['User'] == 'You']['EUR Saved'].values[0]:.2f}"
        )
    
    display_month = month_rankings[["Rank", "User", "EUR Saved", "Coffees"]].copy()
    display_month.columns = ["🏅 Rank", "👤 User", "💶 EUR Saved", "☕ Coffees"]
    st.dataframe(display_month.style.format({"💶 EUR Saved": "{:.2f}"}), use_container_width=True)

with tab2:
    week_rankings = get_period_rankings(7)
    week_rankings["Rank"] = range(1, len(week_rankings) + 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🥇 Top This Week",
            week_rankings.iloc[0]["User"],
            f"€{week_rankings.iloc[0]['EUR Saved']:.2f}"
        )
    with col2:
        st.metric(
            "Your Position",
            f"#{week_rankings[week_rankings['User'] == 'You']['Rank'].values[0] if has_real_data else 1}",
            f"€{week_rankings[week_rankings['User'] == 'You']['EUR Saved'].values[0]:.2f}"
        )
    
    display_week = week_rankings[["Rank", "User", "EUR Saved", "Coffees"]].copy()
    display_week.columns = ["🏅 Rank", "👤 User", "💶 EUR Saved", "☕ Coffees"]
    st.dataframe(display_week.style.format({"💶 EUR Saved": "{:.2f}"}), use_container_width=True)

with tab3:
    today_rankings = get_period_rankings(1)
    today_rankings["Rank"] = range(1, len(today_rankings) + 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🥇 Top Today",
            today_rankings.iloc[0]["User"],
            f"€{today_rankings.iloc[0]['EUR Saved']:.2f}"
        )
    with col2:
        st.metric(
            "Your Position",
            f"#{today_rankings[today_rankings['User'] == 'You']['Rank'].values[0] if has_real_data else 1}",
            f"€{today_rankings[today_rankings['User'] == 'You']['EUR Saved'].values[0]:.2f}"
        )
    
    display_today = today_rankings[["Rank", "User", "EUR Saved", "Coffees"]].copy()
    display_today.columns = ["🏅 Rank", "👤 User", "💶 EUR Saved", "☕ Coffees"]
    st.dataframe(display_today.style.format({"💶 EUR Saved": "{:.2f}"}), use_container_width=True)

st.divider()

# Statistics
st.subheader("📈 Statistics")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric(
        "Total Users",
        len(rankings),
        help="Number of users in the ranking"
    )

with stat_col2:
    st.metric(
        "Max Savings",
        f"€{rankings['EUR Saved'].max():.2f}",
        help="Highest individual savings"
    )

with stat_col3:
    st.metric(
        "Total Community EUR",
        f"€{rankings['EUR Saved'].sum():.2f}",
        help="Combined savings of all users"
    )

with stat_col4:
    st.metric(
        "Community Coffees",
        f"☕ {rankings['Coffees'].sum()}",
        help="Total coffees from all savings"
    )

st.divider()

# Your stats (if real data)
if has_real_data:
    st.subheader("📊 Your Statistics")
    
    your_rank = rankings[rankings["User"] == "You"]["Rank"].values[0]
    your_savings = rankings[rankings["User"] == "You"]["EUR Saved"].values[0]
    your_coffees = rankings[rankings["User"] == "You"]["Coffees"].values[0]
    
    rank_pct = (your_rank / len(rankings)) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"**Your Rank:** #{your_rank} out of {len(rankings)}")
    
    with col2:
        st.info(f"**Top {100-rank_pct:.0f}%** of savers")
    
    with col3:
        st.warning(f"**Your Savings:** €{your_savings:.2f} (☕ {your_coffees})")
    
    st.info(f"""
    You're doing great! 🎉 
    
    - You're in the top {100-rank_pct:.0f}% of forex savers
    - Your savings could buy **{your_coffees} coffees** in India
    - Average community savings: €{rankings['EUR Saved'].mean():.2f}
    """)
