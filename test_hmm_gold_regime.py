import sys, os, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

def build_hmm_features(df):
    """
    HMM requires stationary (mean-reverting) features to define market regimes.
    We use Log Returns and ATR / True Range magnitude.
    """
    df = df.copy()
    # Log Returns (Direction & Momentum)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Range / Volatility proxy (High-Low)
    df['range'] = (df['High'] - df['Low']) / df['Close']
    
    return df.dropna().reset_index()

def backtest_hmm_gold():
    logger.info("--- Testing Paradigm Shift 1: HMM Regime Detection on GOLD (XAUUSD) ---")
    
    # 1. Fetch Gold Data
    df = yf.download("GC=F", period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df_feat = build_hmm_features(df)
    
    # 2. Train Hidden Markov Model (HMM)
    # We define 3 regimes: 0 = Low Vol / Sideway, 1 = High Vol / Downtrend, 2 = High Vol / Uptrend
    # In unsupervised learning, the labels (0,1,2) are assigned randomly, we figure them out later.
    logger.info("Training 3-State Gaussian HMM on Log Returns and Volatility...")
    
    X = df_feat[['log_ret', 'range']].values
    
    # 3-State HMM
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Predict the hidden states (Regimes)
    hidden_states = model.predict(X)
    df_feat['regime'] = hidden_states
    
    # 3. Analyze Regimes to figure out what they mean
    # We calculate the mean return and mean volatility for each regime
    regime_stats = df_feat.groupby('regime').agg({'log_ret': 'mean', 'range': 'mean', 'Close': 'count'}).rename(columns={'Close': 'Days'})
    logger.info(f"\nDiscovered Market Regimes:\n{regime_stats}")
    
    # Identify which regime is the "Trending / High Growth" regime.
    # Usually, it's the one with the highest positive log_ret
    bull_regime = regime_stats['log_ret'].idxmax()
    sideway_regime = regime_stats['range'].idxmin()
    bear_regime = regime_stats['log_ret'].idxmin()
    
    logger.info(f"Identified Bull Regime: {bull_regime}, Bear Regime: {bear_regime}, Quiet Regime: {sideway_regime}")
    
    # 4. Trading Strategy using HMM
    # Strategy: 
    # If in Bull Regime -> Buy and Hold
    # If in Bear Regime -> Sell and Hold (or stay cash)
    # If in Quiet Regime -> Mean Reversion (Simulated as staying out for safety)
    
    df_feat['signal'] = 0
    # Enter long if the previous day was classified as Bull Regime
    df_feat.loc[df_feat['regime'].shift(1) == bull_regime, 'signal'] = 1
    # Enter short if previous day was Bear Regime
    df_feat.loc[df_feat['regime'].shift(1) == bear_regime, 'signal'] = -1
    
    # Calculate Returns
    df_feat['future_ret'] = df_feat['Close'].pct_change().shift(-1)
    df_feat['strat_ret'] = df_feat['signal'] * df_feat['future_ret']
    
    # Apply Trading Cost
    cost = 0.0005 # Gold spread proxy
    df_feat['strat_ret_net'] = np.where(df_feat['signal'].diff() != 0, df_feat['strat_ret'] - cost, df_feat['strat_ret'])
    # Clean up where signal is 0
    df_feat.loc[df_feat['signal'] == 0, 'strat_ret_net'] = 0
    
    # Benchmark (Buy and Hold Gold)
    df_feat['bnh_ret_net'] = df_feat['future_ret']
    
    active = df_feat[df_feat['signal'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(df_feat['strat_ret_net'].cumsum()).iloc[-1] - 1
    bnh_net = np.exp(df_feat['bnh_ret_net'].cumsum()).iloc[-1] - 1
    
    logger.info("\n=== HMM GOLD STRATEGY RESULTS ===")
    logger.info(f"Total Trades / Days in Market: {len(active)}")
    logger.info(f"Win Rate (Daily): {win_rate*100:.1f}%")
    logger.info(f"HMM Strategy Net Return: {net_ret*100:.2f}%")
    logger.info(f"Buy & Hold Gold Return : {bnh_net*100:.2f}%")

if __name__ == "__main__":
    backtest_hmm_gold()