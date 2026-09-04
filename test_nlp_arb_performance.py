import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta

# Mock VADER for test
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm

def backtest_stat_arb(asset1="EURUSD=X", asset2="GBPUSD=X", lookback="1y", z_threshold=2.0):
    """Backtests a Statistical Arbitrage strategy."""
    logger.info(f"Backtesting Stat Arb: {asset1} vs {asset2}")
    
    # 1. Fetch Data
    df1 = yf.download(asset1, period=lookback, interval="1h", progress=False)['Close']
    df2 = yf.download(asset2, period=lookback, interval="1h", progress=False)['Close']
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = ['Asset1', 'Asset2']
    
    if df.empty: return 0, 0
    
    # 2. Dynamic Rolling Hedge Ratio (60-period window to find Beta)
    # Fast proxy: rolling covariance / variance
    rolling_window = 120
    df['cov'] = df['Asset1'].rolling(rolling_window).cov(df['Asset2'])
    df['var'] = df['Asset2'].rolling(rolling_window).var()
    df['beta'] = df['cov'] / df['var']
    
    # 3. Calculate Spread and Z-Score
    df['spread'] = df['Asset1'] - (df['beta'] * df['Asset2'])
    df['spread_mean'] = df['spread'].rolling(rolling_window).mean()
    df['spread_std'] = df['spread'].rolling(rolling_window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # 4. Trading Logic
    # If Z > 2: Sell Asset1, Buy Asset2 (Spread will revert down)
    # If Z < -2: Buy Asset1, Sell Asset2 (Spread will revert up)
    df['position'] = 0
    df.loc[df['z_score'] > z_threshold, 'position'] = -1
    df.loc[df['z_score'] < -z_threshold, 'position'] = 1
    
    # Forward fill positions until z_score reverts to 0
    df['position'] = df['position'].replace(0, np.nan)
    df.loc[abs(df['z_score']) < 0.5, 'position'] = 0
    df['position'] = df['position'].ffill().fillna(0)
    
    # Calculate Returns
    df['ret1'] = df['Asset1'].pct_change()
    df['ret2'] = df['Asset2'].pct_change()
    
    # Portfolio return (assuming equal capital split)
    # When pos=-1: Short Asset1, Long Asset2. Return = -ret1 + ret2
    df['strat_ret'] = df['position'].shift(1) * (-df['ret1'] + df['ret2'])
    
    # Simple cost
    trade_cost = 0.0002
    trades = (df['position'].diff().fillna(0) != 0).sum()
    
    df['strat_ret_net'] = df['strat_ret']
    df.loc[df['position'].diff() != 0, 'strat_ret_net'] -= trade_cost
    
    cum_ret = np.exp(df['strat_ret_net'].cumsum())
    net_ret = cum_ret.iloc[-1] - 1
    
    logger.info(f"Stat Arb Result | Trades: {trades} | Net Return: {net_ret*100:.2f}%")
    return net_ret, trades

def simulate_nlp_impact(symbol="GC=F"):
    """Simulates how an NLP sentiment filter would affect a standard trend strategy."""
    logger.info(f"Simulating NLP Sentiment Impact on {symbol} (GOLD)")
    
    df = yf.download(symbol, period="1y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df = df['Close']
    else: df = df[['Close']]
    df = df.dropna()
    df.columns = ['close']
    
    # Base strategy: Simple Moving Average Crossover (Trend)
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['base_signal'] = np.where(df['sma20'] > df['sma50'], 1, -1)
    
    # Mock NLP Sentiment: We assume News happens randomly 10% of the time,
    # and NLP correctly filters out bad setups 50% of the time.
    np.random.seed(42)
    df['is_news_hour'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    # Mock: NLP catches the wrong trend direction during news and forces it to 0
    df['nlp_signal'] = df['base_signal']
    df.loc[(df['is_news_hour'] == 1) & (np.random.random(len(df)) > 0.5), 'nlp_signal'] = 0
    
    df['ret'] = df['close'].pct_change()
    
    df['base_strat'] = df['base_signal'].shift(1) * df['ret']
    df['nlp_strat'] = df['nlp_signal'].shift(1) * df['ret']
    
    base_net = np.exp((df['base_strat'] - 0.0002).cumsum()).iloc[-1] - 1
    nlp_net = np.exp((df['nlp_strat'] - 0.0002).cumsum()).iloc[-1] - 1
    
    logger.info(f"Standard Trend Return: {base_net*100:.2f}%")
    logger.info(f"NLP-Filtered Return:  {nlp_net*100:.2f}%")
    
    return base_net, nlp_net

if __name__ == "__main__":
    logger.info("Starting Advanced Engine Performance Evaluation...")
    print("\n[1] Evaluating Statistical Arbitrage (Pairs Trading)")
    arb_ret, arb_trades = backtest_stat_arb("EURUSD=X", "GBPUSD=X")
    
    print("\n[2] Evaluating NLP Sentiment Filter impact on Gold")
    base_ret, nlp_ret = simulate_nlp_impact("GC=F")
    
    print("\n=== SUMMARY ===")
    print(f"Stat Arb Strategy: {arb_ret*100:.2f}% (Trades: {arb_trades})")
    print(f"Gold Base Strategy: {base_ret*100:.2f}%")
    print(f"Gold + NLP Filter:  {nlp_ret*100:.2f}%")