"""
Transaction logging module for EUR/INR Forex application
"""

import pandas as pd
import os
from datetime import datetime


TRANSACTIONS_FILE = "transactions.csv"


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
