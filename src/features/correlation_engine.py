import sys
import os
import pandas as pd
import numpy as np
from loguru import logger
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def calculate_portfolio_correlation(symbols_data, returns_col="ret_5", window=60):
    """
    Phase 14: Portfolio Engine (Correlation Matrix)
    Calculates the rolling and static correlation between multiple assets to prevent over-exposure.
    
    symbols_data: Dict of DataFrames {'EURUSD': df1, 'GBPUSD': df2, ...}
    returns_col: The feature column to use for correlation (e.g., daily returns or ret_5)
    """
    logger.info(f"Calculating Portfolio Correlation for {list(symbols_data.keys())}...")
    
    # Extract returns into a single DataFrame
    combined_returns = pd.DataFrame()
    
    for symbol, df in symbols_data.items():
        if returns_col not in df.columns:
            logger.warning(f"Column '{returns_col}' not found in {symbol}. Skipping.")
            continue
            
        # Ensure timestamp is index for proper alignment
        temp_df = df[['timestamp', returns_col]].copy()
        temp_df.set_index('timestamp', inplace=True)
        temp_df.columns = [symbol]
        
        if combined_returns.empty:
            combined_returns = temp_df
        else:
            combined_returns = combined_returns.join(temp_df, how='outer')
            
    # Forward fill small gaps, drop large ones
    combined_returns = combined_returns.ffill().dropna()
    
    if combined_returns.empty:
        logger.error("Combined returns dataframe is empty. Cannot calculate correlation.")
        return None, None
        
    # 1. Static Correlation (Full Period)
    static_corr = combined_returns.corr(method='pearson')
    logger.info("Static Correlation Matrix Generated.")
    
    # 2. Rolling Correlation (Last 'window' days)
    # Important for detecting shifting relationships (e.g., Crisis periods where all assets correlate to 1.0)
    rolling_corr = combined_returns.rolling(window=window).corr()
    logger.info(f"Rolling Correlation ({window}-bar window) Generated.")
    
    return static_corr, rolling_corr

def evaluate_portfolio_exposure(static_corr, proposed_trades, max_correlation=0.75):
    """
    Risk Rule: Do not take new trades that are highly correlated with existing positions in the same direction.
    
    proposed_trades: List of dicts [{'symbol': 'EURUSD', 'signal': 1}, {'symbol': 'GBPUSD', 'signal': 1}]
    Returns approved trades.
    """
    logger.info("Evaluating Portfolio Exposure Limits...")
    
    approved_trades = []
    rejected_trades = []
    
    # Simple Greedy Allocation: Accept trades in order, reject if highly correlated to already approved ones
    for trade in proposed_trades:
        symbol = trade['symbol']
        signal = trade['signal']
        
        is_safe = True
        for approved in approved_trades:
            app_sym = approved['symbol']
            app_sig = approved['signal']
            
            if symbol in static_corr.columns and app_sym in static_corr.columns:
                corr_value = static_corr.loc[symbol, app_sym]
                
                # If they are highly positively correlated AND trading the same direction -> Reject
                if corr_value > max_correlation and signal == app_sig:
                    logger.warning(f"Rejecting {symbol} ({signal}): Highly correlated ({corr_value:.2f}) with approved {app_sym} ({app_sig})")
                    is_safe = False
                    rejected_trades.append(trade)
                    break
                    
                # If they are highly negatively correlated AND trading opposite directions -> Reject
                # Example: Buy EURUSD, Sell USDCHF. Both are essentially "Short USD".
                elif corr_value < -max_correlation and signal != app_sig:
                    logger.warning(f"Rejecting {symbol} ({signal}): Negative correlation risk ({corr_value:.2f}) with approved {app_sym} ({app_sig})")
                    is_safe = False
                    rejected_trades.append(trade)
                    break
                    
        if is_safe:
            approved_trades.append(trade)
            
    logger.info(f"Approved {len(approved_trades)} trades. Rejected {len(rejected_trades)} trades due to correlation risk.")
    return approved_trades

if __name__ == "__main__":
    logger.info("Portfolio Correlation Engine Ready.")
