"""
Models module for EUR/INR Forex prediction
Includes OLS, ARIMA, and GARCH models
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from datetime import datetime, timedelta


def fetch_fx_data(period="2y"):
    """
    Fetch EUR/INR exchange rate data from Yahoo Finance
    
    Parameters:
    -----------
    period : str
        Data period (e.g., "2y", "4y", "5y", "10y", "max")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'Rate' column containing exchange rates
    """
    try:
        ticker = yf.Ticker("EURINR=X")
        df = ticker.history(period=period)
        
        # Use closing price as exchange rate
        df = df[["Close"]].rename(columns={"Close": "Rate"})
        df = df[df["Rate"] > 0]  # Filter invalid data
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def build_forecast_dates(last_date, num_forecasts):
    """
    Build forecast dates starting from the last date
    
    Parameters:
    -----------
    last_date : datetime
        Last date in the dataset
    num_forecasts : int
        Number of forecast periods
    
    Returns:
    --------
    pd.DatetimeIndex
        Array of forecast dates
    """
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_forecasts, freq="D")
    return forecast_dates


def run_ols_model(rates, forecast_days):
    """
    Run OLS regression model to identify trend
    
    Parameters:
    -----------
    rates : pd.Series
        Time series of exchange rates
    forecast_days : int
        Number of days to forecast
    
    Returns:
    --------
    tuple
        (OLS model object, forecast array)
    """
    try:
        # Prepare data
        y = rates.values
        X = np.arange(len(y))
        
        # Create lagged variable for trend
        lag_1 = np.roll(y, 1)
        lag_1[0] = y[0]
        
        # Fit OLS
        X = sm.add_constant(np.column_stack([X, lag_1]))
        model = sm.OLS(y, X).fit()
        
        # Forecast
        last_x = len(y)
        last_rate = y[-1]
        
        forecast = []
        for i in range(1, forecast_days + 1):
            next_x = np.array([1, last_x + i, last_rate])
            next_pred = model.predict(next_x)[0]
            forecast.append(next_pred)
            last_rate = next_pred
        
        return model, np.array(forecast)
    except Exception as e:
        print(f"Error in OLS model: {e}")
        return None, np.array([])


def run_arima_model(rates, forecast_days, order=(5, 1, 0)):
    """
    Run ARIMA model for forecasting
    
    Parameters:
    -----------
    rates : pd.Series
        Time series of exchange rates
    forecast_days : int
        Number of days to forecast
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    tuple
        (ARIMA model object, forecast Series)
    """
    try:
        model = ARIMA(rates, order=order)
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.get_forecast(steps=forecast_days)
        forecast_values = forecast.predicted_mean
        
        return fitted_model, forecast_values
    except Exception as e:
        print(f"Error in ARIMA model: {e}")
        return None, pd.Series([])


def run_garch_model(rates, forecast_days, p=1, q=1):
    """
    Run GARCH model for volatility assessment
    
    Parameters:
    -----------
    rates : pd.Series
        Time series of exchange rates
    forecast_days : int
        Number of days to forecast
    p : int
        GARCH p parameter
    q : int
        GARCH q parameter
    
    Returns:
    --------
    tuple
        (GARCH model object, variance forecast array)
    """
    try:
        # Calculate returns
        returns = rates.pct_change().dropna() * 100
        
        # Fit GARCH
        model = arch_model(returns, vol="Garch", p=p, q=q)
        fitted_model = model.fit(disp="off")
        
        # Forecast variance
        forecast_var = fitted_model.forecast(horizon=forecast_days)
        variance = forecast_var.variance.values[-1, :]
        
        return fitted_model, variance
    except Exception as e:
        print(f"Error in GARCH model: {e}")
        return None, np.array([])


def classify_risk_from_variance(variance):
    """
    Classify risk level based on volatility
    
    Parameters:
    -----------
    variance : np.array
        Array of variance forecasts
    
    Returns:
    --------
    tuple
        (risk_label, risk_description, average_volatility)
    """
    if variance is None or len(variance) == 0:
        return "Unknown", "Unable to assess", 0.0
    
    avg_vol = np.mean(np.sqrt(variance))  # Convert variance to volatility
    
    if avg_vol < 0.5:
        return "🟢 Low", "Market is relatively stable. Good time for planned transfers.", avg_vol
    elif avg_vol < 1.0:
        return "🟡 Moderate", "Market shows normal volatility. Standard conditions.", avg_vol
    elif avg_vol < 1.5:
        return "🟠 High", "Market is volatile. Consider hedging strategies.", avg_vol
    else:
        return "🔴 Very High", "Extreme volatility. Caution advised for large transfers.", avg_vol


def generate_trading_advice(ols_direction, risk_level, current_rate, predicted_rate):
    """
    Generate trading recommendation based on models
    
    Parameters:
    -----------
    ols_direction : str
        Trend direction (UP or DOWN)
    risk_level : str
        Risk classification
    current_rate : float
        Current exchange rate
    predicted_rate : float
        Predicted exchange rate
    
    Returns:
    --------
    str
        Trading recommendation
    """
    trend = "upward" if "UP" in ols_direction else "downward"
    change_pct = ((predicted_rate - current_rate) / current_rate) * 100 if current_rate > 0 else 0
    
    advice = f"**Trend:** {trend} | "
    
    if change_pct > 0:
        advice += f"**EUR expected to strengthen by {change_pct:.2f}%** | "
        advice += "👉 If you're buying EUR, consider waiting a bit for a better rate. "
    else:
        advice += f"**EUR expected to weaken by {abs(change_pct):.2f}%** | "
        advice += "👉 If you're buying EUR, now might be a better time. "
    
    if "Low" in risk_level:
        advice += "✓ Stable environment for transactions."
    elif "Moderate" in risk_level:
        advice += "→ Proceed with normal caution."
    else:
        advice += "⚠️ High volatility—consider breaking large transfers into smaller chunks."
    
    return advice
