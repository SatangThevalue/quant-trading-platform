import sys, os, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

def fetch_crypto_data(asset1="BTC-USD", asset2="ETH-USD", period="2y", interval="1d"):
    logger.info(f"Fetching data for {asset1} and {asset2}...")
    df1 = yf.download(asset1, period=period, interval=interval, progress=False)['Close']
    df2 = yf.download(asset2, period=period, interval=interval, progress=False)['Close']
    
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = ['Asset1', 'Asset2']
    return df

def backtest_pairs_trading():
    logger.info("--- Testing Paradigm Shift 2: Statistical Arbitrage on Crypto (BTC vs ETH) ---")
    
    # We use Daily data to find stable macro relationships between BTC and ETH
    df = fetch_crypto_data("BTC-USD", "ETH-USD", period="3y", interval="1d")
    
    if df.empty:
        logger.error("Failed to fetch crypto data.")
        return
        
    # We will use a Rolling Linear Regression to find the dynamic Hedge Ratio
    # This prevents the model from relying on stale relationships.
    rolling_window = 90
    
    logger.info(f"Calculating Rolling Hedge Ratio (Window={rolling_window} days)...")
    
    hedge_ratios = []
    for i in range(len(df)):
        if i < rolling_window:
            hedge_ratios.append(np.nan)
            continue
            
        y = df['Asset1'].iloc[i-rolling_window:i] # BTC
        X = sm.add_constant(df['Asset2'].iloc[i-rolling_window:i]) # ETH
        model = sm.OLS(y, X).fit()
        hedge_ratios.append(model.params.iloc[1]) # The Beta coefficient for Asset2
        
    df['hedge_ratio'] = hedge_ratios
    df = df.dropna()
    
    # Calculate the Spread: Spread = BTC - (Hedge_Ratio * ETH)
    df['spread'] = df['Asset1'] - (df['hedge_ratio'] * df['Asset2'])
    
    # Z-Score of the spread to trigger trades
    df['spread_mean'] = df['spread'].rolling(30).mean()
    df['spread_std'] = df['spread'].rolling(30).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    df = df.dropna()
    
    # Cointegration Check on the whole period
    adf_result = adfuller(df['spread'])
    p_value = adf_result[1]
    logger.info(f"Cointegration P-Value: {p_value:.4f}")
    if p_value < 0.05:
        logger.info("✅ Strong Cointegration found! (Spread is mean-reverting)")
    else:
        logger.warning("❌ Weak Cointegration. Trades might be risky.")
        
    # Trading Logic
    # Z-Score > 2.0: BTC is overpriced relative to ETH -> Sell BTC, Buy ETH
    # Z-Score < -2.0: BTC is underpriced relative to ETH -> Buy BTC, Sell ETH
    entry_z = 2.0
    exit_z = 0.5
    
    df['position'] = 0
    df.loc[df['z_score'] > entry_z, 'position'] = -1
    df.loc[df['z_score'] < -entry_z, 'position'] = 1
    
    # Forward fill positions until z_score reverts
    df['position'] = df['position'].replace(0, np.nan)
    df.loc[abs(df['z_score']) < exit_z, 'position'] = 0
    df['position'] = df['position'].ffill().fillna(0)
    
    # Calculate Returns
    df['ret1'] = df['Asset1'].pct_change()
    df['ret2'] = df['Asset2'].pct_change()
    
    # Portfolio return (assuming equal dollar allocation per leg)
    df['strat_ret'] = df['position'].shift(1) * (-df['ret1'] + df['ret2'])
    
    # Trading Cost (Crypto spot trading is approx 0.1% per trade)
    cost = 0.001 * 2 # Need to pay cost for both legs
    df['strat_ret_net'] = df['strat_ret']
    df.loc[df['position'].diff().fillna(0) != 0, 'strat_ret_net'] -= cost
    
    # Benchmark (Buy and Hold BTC)
    df['bnh_btc'] = df['ret1']
    
    active = df[df['position'] != 0]
    trades = (df['position'].diff().fillna(0) != 0).sum()
    
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(df['strat_ret_net'].cumsum()).iloc[-1] - 1
    bnh_net = np.exp(df['bnh_btc'].cumsum()).iloc[-1] - 1
    
    # Max DD
    cum_ret = np.exp(df['strat_ret_net'].cumsum())
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    btc_cum = np.exp(df['bnh_btc'].cumsum())
    btc_dd = ((btc_cum - btc_cum.cummax()) / btc_cum.cummax()).min()
    
    logger.info("\n=== CRYPTO PAIRS TRADING (BTC vs ETH) RESULTS ===")
    logger.info(f"Total Trade Entries: {trades // 2}") # Divide by 2 (entry and exit)
    logger.info(f"Pairs Strat Net Return : {net_ret*100:.2f}% | Max DD: {max_dd*100:.2f}%")
    logger.info(f"Buy & Hold BTC Return  : {bnh_net*100:.2f}% | Max DD: {btc_dd*100:.2f}%")

if __name__ == "__main__":
    backtest_pairs_trading()
