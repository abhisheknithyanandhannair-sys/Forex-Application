"""
Transaction logging module for EUR/INR Forex application
"""

import pandas as pd
import os
from datetime import datetime


TRANSACTIONS_FILE = "transactions.csv"
COFFEE_PRICE_INR = 250  # Average coffee price in INR


def load_transactions():
    """
    Load transaction history from CSV
    
    Returns:
    --------
    pd.DataFrame
        Transaction history or empty DataFrame if file doesn't exist
    """
    if os.path.exists(TRANSACTIONS_FILE):
        df = pd.read_csv(TRANSACTIONS_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    else:
        return pd.DataFrame(columns=["date", "amount_inr", "rate"])


def clear_transactions():
    """
    Clear all transactions by removing the CSV file
    """
    if os.path.exists(TRANSACTIONS_FILE):
        os.remove(TRANSACTIONS_FILE)
        return True
    return False


def append_transaction(tx_date, amount_inr, current_rate):
    """
    Log a transaction and calculate savings/differences
    
    Parameters:
    -----------
    tx_date : datetime.date
        Transaction date
    amount_inr : float
        Amount in INR
    current_rate : float
        Current EUR/INR rate
    
    Returns:
    --------
    tuple
        (updated_transactions_df, savings_eur or None)
    """
    df = load_transactions()
    
    # Create new transaction record
    new_record = {
        "date": pd.Timestamp(tx_date),
        "amount_inr": amount_inr,
        "rate": current_rate
    }
    
    # Calculate savings compared to previous transaction
    savings_eur = None
    if len(df) > 0:
        last_rate = df.iloc[-1]["rate"]
        eur_last = amount_inr / last_rate
        eur_now = amount_inr / current_rate
        savings_eur = eur_now - eur_last
    
    # Add new record
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(TRANSACTIONS_FILE, index=False)
    
    return df, savings_eur


def get_transaction_statistics():
    """
    Get statistics from transaction history
    
    Returns:
    --------
    dict
        Dictionary with transaction statistics
    """
    df = load_transactions()
    
    if df.empty:
        return {
            "total_transactions": 0,
            "average_rate": 0,
            "best_rate": 0,
            "worst_rate": 0,
            "total_inr": 0,
            "total_eur": 0
        }
    
    total_eur = (df["amount_inr"] / df["rate"]).sum()
    
    return {
        "total_transactions": len(df),
        "average_rate": df["rate"].mean(),
        "best_rate": df["rate"].min(),
        "worst_rate": df["rate"].max(),
        "total_inr": df["amount_inr"].sum(),
        "total_eur": total_eur
    }


def get_savings_stats():
    """
    Get savings statistics for the current user
    
    Returns:
    --------
    dict
        Dictionary with savings information
    """
    from datetime import datetime, timedelta
    
    df = load_transactions()
    
    if df.empty:
        return {
            "total_transactions": 0,
            "total_inr_saved": 0,
            "total_eur_saved": 0,
            "today_savings": 0,
            "week_savings": 0,
            "month_savings": 0,
            "coffees_today": 0,
            "coffees_week": 0,
            "coffees_month": 0,
        }
    
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)
    
    # Total savings
    first_rate = df["rate"].iloc[0]
    last_rate = df["rate"].iloc[-1]
    total_inr = df["amount_inr"].sum()
    total_eur_with_first = (total_inr / first_rate)
    total_eur_with_last = (total_inr / last_rate)
    total_eur_saved = total_eur_with_first - total_eur_with_last
    
    # Today savings
    today_df = df[df["date"].dt.date == today_start.date()]
    today_inr = today_df["amount_inr"].sum() if not today_df.empty else 0
    today_eur_saved = (today_inr / first_rate) - (today_inr / last_rate) if today_inr > 0 else 0
    
    # Week savings
    week_df = df[df["date"] >= week_start]
    week_inr = week_df["amount_inr"].sum() if not week_df.empty else 0
    week_eur_saved = (week_inr / first_rate) - (week_inr / last_rate) if week_inr > 0 else 0
    
    # Month savings
    month_df = df[df["date"] >= month_start]
    month_inr = month_df["amount_inr"].sum() if not month_df.empty else 0
    month_eur_saved = (month_inr / first_rate) - (month_inr / last_rate) if month_inr > 0 else 0
    
    return {
        "total_transactions": len(df),
        "total_inr_saved": max(0, total_inr * (first_rate - last_rate) / first_rate),
        "total_eur_saved": max(0, total_eur_saved),
        "today_savings": max(0, today_eur_saved),
        "week_savings": max(0, week_eur_saved),
        "month_savings": max(0, month_eur_saved),
        "coffees_today": max(0, int(today_inr * 0.1 / COFFEE_PRICE_INR)) if today_inr > 0 else 0,
        "coffees_week": max(0, int(week_inr * 0.1 / COFFEE_PRICE_INR)) if week_inr > 0 else 0,
        "coffees_month": max(0, int(month_inr * 0.1 / COFFEE_PRICE_INR)) if month_inr > 0 else 0,
    }
