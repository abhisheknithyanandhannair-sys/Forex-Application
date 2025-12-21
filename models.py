"""
Models module for EUR/INR Forex prediction
Includes OLS, ARIMA, and GARCH models
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


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
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Fit OLS with constant
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # Forecast
        last_x = len(y)
        
        forecast = []
        for i in range(1, forecast_days + 1):
            next_x = np.array([[1, last_x + i]])
            next_pred = model.predict(next_x)[0]
            forecast.append(next_pred)
        
        return model, np.array(forecast)
    except Exception as e:
        print(f"Error in OLS model: {e}")
        return None, np.array([])


def find_optimal_arima_order(rates, max_p=5, max_d=2, max_q=5):
    """
    Find optimal ARIMA order using grid search and AIC
    
    Parameters:
    -----------
    rates : pd.Series
        Time series of exchange rates
    max_p : int
        Maximum p value to test
    max_d : int
        Maximum d value to test
    max_q : int
        Maximum q value to test
    
    Returns:
    --------
    tuple
        (optimal_order, best_aic, best_model)
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_model = None
    
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(rates, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue
    
    return best_order, best_aic, best_model


def run_arima_model(rates, forecast_days, order=None, auto_tune=True):
    """
    Run ARIMA model for forecasting with optional parameter tuning
    
    Parameters:
    -----------
    rates : pd.Series
        Time series of exchange rates
    forecast_days : int
        Number of days to forecast
    order : tuple, optional
        ARIMA order (p, d, q). If None, will be auto-tuned. Default is (5, 1, 0)
    auto_tune : bool
        Whether to auto-tune parameters (default: True)
    
    Returns:
    --------
    dict
        Dictionary with model, forecast values, order, AIC, BIC, and evaluation metrics
    """
    try:
        if auto_tune and order is None:
            print("\nTuning ARIMA parameters... (testing p, d, q combinations)")
            order, aic, fitted_model = find_optimal_arima_order(rates, max_p=7, max_d=3, max_q=7)
            print(f"Optimal ARIMA order found: {order}")
            print(f"   - AIC: {aic:.2f} | BIC: {fitted_model.bic:.2f}")
        else:
            if order is None:
                order = (5, 1, 0)  # Default fallback (original fixed order)
                print(f"Using default ARIMA order: {order}")
            else:
                print(f"Using specified ARIMA order: {order}")
            model = ARIMA(rates, order=order)
            fitted_model = model.fit()
            aic = fitted_model.aic
        
        # Forecast
        forecast = fitted_model.get_forecast(steps=forecast_days)
        forecast_values = forecast.predicted_mean
        forecast_ci = forecast.conf_int(alpha=0.05)  # 95% confidence interval
        
        return {
            'model': fitted_model,
            'forecast': forecast_values,
            'conf_int': forecast_ci,
            'order': order,
            'aic': aic,
            'bic': fitted_model.bic,
            'rmse_in_sample': np.sqrt(np.mean(fitted_model.resid ** 2)),
            'mae_in_sample': np.mean(np.abs(fitted_model.resid))
        }
    except Exception as e:
        print(f"Error in ARIMA model: {e}")
        return {
            'model': None,
            'forecast': pd.Series([]),
            'conf_int': None,
            'order': None,
            'aic': np.inf,
            'bic': np.inf,
            'rmse_in_sample': np.inf,
            'mae_in_sample': np.inf
        }


def find_optimal_garch_order(returns, max_p=3, max_q=3):
    """
    Find optimal GARCH order using grid search and AIC
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    max_p : int
        Maximum p value to test
    max_q : int
        Maximum q value to test
    
    Returns:
    --------
    tuple
        ((p, q), best_aic, best_model)
    """
    best_aic = np.inf
    best_order = (1, 1)
    best_model = None
    
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                model = arch_model(returns, vol="Garch", p=p, q=q)
                fitted = model.fit(disp="off")
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, q)
                    best_model = fitted
            except:
                continue
    
    return best_order, best_aic, best_model


def run_garch_model(rates, forecast_days, order=None, auto_tune=True):
    """
    Run GARCH model for volatility assessment with optional parameter tuning
    
    Parameters:
    -----------
    rates : pd.Series
        Time series of exchange rates
    forecast_days : int
        Number of days to forecast
    order : tuple, optional
        GARCH order (p, q). If None, will be auto-tuned. Default is (1, 1)
    auto_tune : bool
        Whether to auto-tune parameters (default: True)
    
    Returns:
    --------
    dict
        Dictionary with model, variance forecast, order, AIC, BIC, and evaluation metrics
    """
    try:
        # Calculate returns
        returns = rates.pct_change().dropna() * 100
        
        if auto_tune and order is None:
            print("\nTuning GARCH parameters... (testing p, q combinations)")
            order, aic, fitted_model = find_optimal_garch_order(returns, max_p=5, max_q=5)
            print(f"Optimal GARCH order found: {order}")
            print(f"   - AIC: {aic:.2f} | BIC: {fitted_model.bic:.2f}")
        else:
            if order is None:
                order = (1, 1)  # Default fallback (original fixed order)
                print(f"Using default GARCH order: {order}")
            else:
                print(f"Using specified GARCH order: {order}")
            model = arch_model(returns, vol="Garch", p=order[0], q=order[1])
            fitted_model = model.fit(disp="off")
            aic = fitted_model.aic
        
        # Forecast variance
        forecast_var = fitted_model.forecast(horizon=forecast_days)
        variance = forecast_var.variance.values[-1, :]
        volatility = np.sqrt(variance)
        
        # In-sample metrics
        residuals = fitted_model.std_resid
        mae_in_sample = np.mean(np.abs(residuals))
        rmse_in_sample = np.sqrt(np.mean(residuals ** 2))
        
        return {
            'model': fitted_model,
            'variance': variance,
            'volatility': volatility,
            'order': order,
            'aic': aic,
            'bic': fitted_model.bic,
            'rmse_in_sample': rmse_in_sample,
            'mae_in_sample': mae_in_sample,
            'log_likelihood': fitted_model.loglikelihood
        }
    except Exception as e:
        print(f"Error in GARCH model: {e}")
        return {
            'model': None,
            'variance': np.array([]),
            'volatility': np.array([]),
            'order': None,
            'aic': np.inf,
            'bic': np.inf,
            'rmse_in_sample': np.inf,
            'mae_in_sample': np.inf,
            'log_likelihood': -np.inf
        }


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAE value
    """
    if len(actual) == 0 or len(predicted) == 0:
        return np.inf
    return mean_absolute_error(actual, predicted)


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    float
        RMSE value
    """
    if len(actual) == 0 or len(predicted) == 0:
        return np.inf
    return np.sqrt(mean_squared_error(actual, predicted))


def compare_models(actual_values, ols_forecast, arima_result, garch_result, dates):
    """
    Compare performance of all models using MAE, RMSE, AIC, and BIC metrics
    
    Parameters:
    -----------
    actual_values : pd.Series or array-like
        Actual values (if available for test period)
    ols_forecast : array-like
        OLS model forecast
    arima_result : dict
        ARIMA model results (contains order, AIC, BIC, RMSE, MAE)
    garch_result : dict
        GARCH model results (contains order, AIC, BIC, RMSE, MAE)
    dates : pd.DatetimeIndex
        Forecast dates
    
    Returns:
    --------
    pd.DataFrame
        Comparison table with all evaluation metrics
    """
    comparison_data = []
    
    # OLS Model
    if ols_forecast is not None and len(ols_forecast) > 0:
        mae_ols = calculate_mae(actual_values, ols_forecast) if actual_values is not None else None
        rmse_ols = calculate_rmse(actual_values, ols_forecast) if actual_values is not None else None
        comparison_data.append({
            'Model': 'OLS',
            'Type': 'Trend Analysis',
            'Order/Config': 'Linear',
            'MAE': f"{mae_ols:.6f}" if mae_ols else "N/A",
            'RMSE': f"{rmse_ols:.6f}" if rmse_ols else "N/A",
            'AIC': 'N/A',
            'BIC': 'N/A',
            'Avg Forecast': f"{np.mean(ols_forecast):.4f}"
        })
    
    # ARIMA Model
    if arima_result['model'] is not None and len(arima_result['forecast']) > 0:
        mae_arima = calculate_mae(actual_values, arima_result['forecast'].values) if actual_values is not None else None
        rmse_arima = calculate_rmse(actual_values, arima_result['forecast'].values) if actual_values is not None else None
        comparison_data.append({
            'Model': 'ARIMA',
            'Type': 'AutoRegressive',
            'Order/Config': str(arima_result['order']),
            'MAE': f"{mae_arima:.6f}" if mae_arima else f"{arima_result['mae_in_sample']:.6f} (in-sample)",
            'RMSE': f"{rmse_arima:.6f}" if rmse_arima else f"{arima_result['rmse_in_sample']:.6f} (in-sample)",
            'AIC': f"{arima_result['aic']:.2f}",
            'BIC': f"{arima_result['bic']:.2f}",
            'Avg Forecast': f"{np.mean(arima_result['forecast'].values):.4f}"
        })
    
    # GARCH Model (volatility)
    if garch_result['model'] is not None and len(garch_result['volatility']) > 0:
        avg_vol = np.mean(garch_result['volatility'])
        max_vol = np.max(garch_result['volatility'])
        comparison_data.append({
            'Model': 'GARCH',
            'Type': 'Volatility',
            'Order/Config': str(garch_result['order']),
            'MAE': f"{garch_result['mae_in_sample']:.6f} (volatility)",
            'RMSE': f"{garch_result['rmse_in_sample']:.6f} (volatility)",
            'AIC': f"{garch_result['aic']:.2f}",
            'BIC': f"{garch_result['bic']:.2f}",
            'Avg Volatility': f"{avg_vol:.4f} | Max: {max_vol:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print detailed comparison
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100)
    
    return comparison_df


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


def get_rate_for_date(df, target_date):
    """
    Get the exchange rate for a specific date from historical data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical dataframe with Rate column and DatetimeIndex
    target_date : datetime.date
        Target date to get rate for
    
    Returns:
    --------
    float or None
        Exchange rate for that date, or None if not found
    """
    try:
        # Convert target_date to datetime if needed
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        
        # Find the rate for the exact date
        matching_data = df[df.index.date == target_date]
        
        if not matching_data.empty:
            return matching_data.iloc[0]["Rate"]
        
        # If exact date not found, get the most recent date before target
        before_target = df[df.index.date < target_date]
        if not before_target.empty:
            return before_target.iloc[-1]["Rate"]
        
        return None
    except Exception as e:
        print(f"Error getting rate for date: {e}")
        return None


def generate_trading_advice(ols_direction, risk_level, current_rate, predicted_rate):
    """
    Generate trading recommendation based on models
    
    Parameters:
    -----------
    ols_direction : str
        Trend direction (UP or DOWN or NEUTRAL)
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
    if predicted_rate is None or predicted_rate == 0:
        predicted_rate = current_rate
    
    trend = "upward" if "UP" in ols_direction else ("downward" if "DOWN" in ols_direction else "neutral")
    change_pct = ((predicted_rate - current_rate) / current_rate) * 100 if current_rate > 0 else 0
    
    advice = f"**Trend:** {trend} | "
    
    if change_pct > 0.5:
        advice += f"**EUR expected to strengthen by {change_pct:.2f}%** | "
        advice += "👉 If you're buying EUR, consider waiting a bit for a better rate. "
    elif change_pct < -0.5:
        advice += f"**EUR expected to weaken by {abs(change_pct):.2f}%** | "
        advice += "👉 If you're buying EUR, now might be a better time. "
    else:
        advice += f"**EUR relatively stable** | "
        advice += "👉 Market conditions are stable for transactions. "
    
    if "Low" in risk_level:
        advice += "✓ Stable environment for transactions."
    elif "Moderate" in risk_level:
        advice += "→ Proceed with normal caution."
    else:
        advice += "⚠️ High volatility—consider breaking large transfers into smaller chunks."
    
    return advice


def main():
    """
    Main execution function - Demonstrates all models and generates results
    Run this script standalone to see model outputs and comparison
    """
    print("\n" + "="*100)
    print("EUR/INR FOREX FORECASTING - TIME SERIES ANALYSIS")
    print("="*100)
    
    # Configuration
    PERIOD = "2y"  # Data period
    FORECAST_DAYS = 30  # Forecast horizon
    
    # Fetch data
    print(f"\n1. Fetching EUR/INR data (Period: {PERIOD})...")
    df = fetch_fx_data(period=PERIOD)
    
    if df.empty:
        print("ERROR: Could not fetch data. Please check internet connection.")
        return
    
    print(f"   [OK] Data fetched: {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   [OK] Current rate: {df['Rate'].iloc[-1]:.4f}")
    print(f"   [OK] Rate range: {df['Rate'].min():.4f} - {df['Rate'].max():.4f}")
    
    # Generate forecast dates
    forecast_dates = build_forecast_dates(df.index[-1], FORECAST_DAYS)
    
    # Run OLS Model
    print(f"\n2. Running OLS Regression Model...")
    ols_model, ols_forecast = run_ols_model(df['Rate'], FORECAST_DAYS)
    if ols_model:
        print(f"   [OK] Model R-squared: {ols_model.rsquared:.4f}")
        print(f"   [OK] Average forecast: {np.mean(ols_forecast):.4f}")
    
    # Run ARIMA Model
    print(f"\n3. Running ARIMA Model...")
    arima_result = run_arima_model(df['Rate'], FORECAST_DAYS, auto_tune=True)
    if arima_result['model']:
        print(f"   [OK] Model order: {arima_result['order']}")
        print(f"   [OK] AIC: {arima_result['aic']:.2f} | BIC: {arima_result['bic']:.2f}")
        print(f"   [OK] In-sample RMSE: {arima_result['rmse_in_sample']:.6f}")
        print(f"   [OK] In-sample MAE: {arima_result['mae_in_sample']:.6f}")
        print(f"   [OK] Average forecast: {np.mean(arima_result['forecast'].values):.4f}")
    
    # Run GARCH Model
    print(f"\n4. Running GARCH Model...")
    garch_result = run_garch_model(df['Rate'], FORECAST_DAYS, auto_tune=True)
    if garch_result['model']:
        print(f"   [OK] Model order: {garch_result['order']}")
        print(f"   [OK] AIC: {garch_result['aic']:.2f} | BIC: {garch_result['bic']:.2f}")
        print(f"   [OK] Average volatility: {np.mean(garch_result['volatility']):.4f}%")
        print(f"   [OK] Max volatility: {np.max(garch_result['volatility']):.4f}%")
    
    # Model Comparison
    print(f"\n5. Model Comparison & Evaluation")
    comparison_df = compare_models(
        None, ols_forecast, arima_result, garch_result, forecast_dates
    )
    
    # Risk Assessment
    print(f"\n6. Risk Assessment (Based on GARCH Volatility)")
    risk_label, risk_desc, avg_vol = classify_risk_from_variance(garch_result['volatility'])
    print(f"   {risk_label}")
    print(f"   - {risk_desc}")
    print(f"   - Average volatility: {avg_vol:.4f}%")
    
    # Forecasts Summary
    print(f"\n7. 30-Day Forecast Summary")
    print(f"   {'Model':<15} {'Day 1':<12} {'Day 15':<12} {'Day 30':<12} {'Average':<12}")
    print(f"   {'-'*63}")
    
    # OLS
    if len(ols_forecast) >= FORECAST_DAYS:
        print(f"   {'OLS':<15} {ols_forecast[0]:<12.4f} {ols_forecast[14]:<12.4f} "
              f"{ols_forecast[29]:<12.4f} {np.mean(ols_forecast):<12.4f}")
    
    # ARIMA
    if len(arima_result['forecast']) >= FORECAST_DAYS:
        arima_vals = arima_result['forecast'].values
        print(f"   {'ARIMA':<15} {arima_vals[0]:<12.4f} {arima_vals[14]:<12.4f} "
              f"{arima_vals[29]:<12.4f} {np.mean(arima_vals):<12.4f}")
    
    # GARCH Volatility
    if len(garch_result['volatility']) >= FORECAST_DAYS:
        garch_vol = garch_result['volatility']
        print(f"   {'GARCH (vol%)':<15} {garch_vol[0]:<12.4f} {garch_vol[14]:<12.4f} "
              f"{garch_vol[29]:<12.4f} {np.mean(garch_vol):<12.4f}")
    
    # Final summary
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\nNote: This standalone script demonstrates all model functionalities.")
    print("For interactive forecasting, please use the Streamlit dashboard (app.py)")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
