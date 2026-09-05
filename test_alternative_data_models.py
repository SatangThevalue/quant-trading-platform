import sys, os, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from loguru import logger

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.fetch_advanced_alternative_data import AlternativeDataPipeline

def backtest_crypto_liquidation_hunter():
    logger.info("--- Testing BTCUSD with Funding Rate (Liquidation Hunter) ---")
    
    # 1. Get Crypto Microstructure Data
    pipeline = AlternativeDataPipeline()
    micro_df = pipeline.fetch_binance_funding_rates(limit=1000)
    
    # 2. Get Price Data
    df = yf.download("BTC-USD", period="1y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    # 3. Build Basic Features
    import pandas_ta as ta
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['mom10'] = ta.mom(df['close'], length=10)
    df = df.dropna().reset_index()
    
    time_col = 'datetime' if 'datetime' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
    df['timestamp'] = pd.to_datetime(df[time_col], utc=True)
    micro_df['timestamp'] = pd.to_datetime(micro_df['timestamp'], utc=True)
    
    # 4. Merge Data (Mergeof to align 8h funding rate with 1h price bars)
    df = pd.merge_asof(df.sort_values('timestamp'), micro_df.sort_values('timestamp'), on='timestamp', direction='backward')
    df = df.dropna()
    
    # 5. Trading Logic: "The Squeeze"
    # If Funding Rate is extreme Greed (> 0.0001), retail is Long -> We look for a reason to SHORT.
    # We predict the next 12 hours return
    df['future_ret'] = (df['close'].shift(-12) - df['close']) / df['close']
    
    # Binary Classification Target (Will it crash by more than 1%?)
    df['target_crash'] = np.where(df['future_ret'] < -0.01, 1, 0)
    
    features = ['rsi14', 'atr14', 'mom10', 'fundingRate', 'funding_rolling_mean', 'is_extreme_greed']
    df = df.dropna(subset=features + ['target_crash'])
    
    X = df[features]
    y = df['target_crash']
    split = int(len(df) * 0.8)
    
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test = X.iloc[split:]
    
    model = lgb.LGBMClassifier(num_leaves=15, max_depth=4, class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    test_df = df.iloc[split:].copy()
    prob_crash = model.predict_proba(X_test)[:, 1]
    
    test_df['signal'] = 0
    # Snipe logic: Only short when probability of crash is high AND market is in extreme greed
    test_df.loc[(prob_crash > 0.65) & (test_df['is_extreme_greed'] == 1), 'signal'] = -1
    
    cost = 0.002
    test_df['strat_ret_net'] = 0.0
    for i in range(len(test_df)):
        if test_df['signal'].iloc[i] == -1:
            ret = -test_df['future_ret'].iloc[i] # Short profit
            test_df.loc[test_df.index[i], 'strat_ret_net'] = ret - cost
            
    active = test_df[test_df['signal'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(test_df['strat_ret_net'].cumsum()).iloc[-1] - 1 if len(active) > 0 else 0
    
    logger.info(f"[BTC Liquidation Hunter] Trades: {len(active)} | WinRate: {win_rate*100:.1f}% | Net Return: {net_ret*100:.2f}%")

def backtest_gold_real_yields():
    logger.info("--- Testing GOLD with Real Yields Proxy (Macro Alpha) ---")
    pipeline = AlternativeDataPipeline()
    macro_df = pipeline.fetch_real_yields_proxy(period="2y")
    
    df = yf.download("GC=F", period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    import pandas_ta as ta
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df = df.dropna().reset_index()
    
    time_col = 'datetime' if 'datetime' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
    df['timestamp'] = pd.to_datetime(df[time_col], utc=True)
    macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], utc=True)
    
    df = pd.merge_asof(df.sort_values('timestamp'), macro_df.sort_values('timestamp'), on='timestamp', direction='backward')
    df = df.dropna()
    
    # Future 5-day return
    df['future_ret'] = (df['close'].shift(-5) - df['close']) / df['close']
    df['target'] = np.where(df['future_ret'] > 0.005, 1, 0)
    
    features = ['rsi14', 'real_yield_proxy', 'real_yield_momentum']
    df = df.dropna(subset=features + ['target'])
    
    X = df[features]
    y = df['target']
    split = int(len(df) * 0.8)
    
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test = X.iloc[split:]
    
    model = lgb.LGBMClassifier(num_leaves=10, max_depth=3, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    test_df = df.iloc[split:].copy()
    prob = model.predict_proba(X_test)[:, 1]
    
    test_df['signal'] = 0
    # Buy when probability is high and Real Yield Momentum is falling (Negative)
    test_df.loc[(prob > 0.60) & (test_df['real_yield_momentum'] < 0), 'signal'] = 1
    
    cost = 0.0005
    test_df['strat_ret_net'] = np.where(test_df['signal'] == 1, test_df['future_ret'] - cost, 0)
    
    active = test_df[test_df['signal'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(test_df['strat_ret_net'].cumsum()).iloc[-1] - 1 if len(active) > 0 else 0
    
    logger.info(f"[GOLD Macro Yields] Trades: {len(active)} | WinRate: {win_rate*100:.1f}% | Net Return: {net_ret*100:.2f}%")

if __name__ == "__main__":
    logger.info("=== TESTING ALTERNATIVE DATA MODELS (GROUP 3 RESCUE) ===")
    backtest_crypto_liquidation_hunter()
    backtest_gold_real_yields()