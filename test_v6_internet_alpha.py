import sys, os, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from loguru import logger

warnings.filterwarnings('ignore')

def build_simple_features(df):
    import pandas_ta as ta
    df = df.copy()
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['sma20'] = ta.sma(df['close'], length=20) # Proxy for VWAP
    df['dist_sma'] = (df['close'] - df['sma20']) / df['close']
    df['rsi14'] = ta.rsi(df['close'], length=14)
    return df.dropna().reset_index()

def backtest_v6_asian_vwap(symbol="GBPUSD=X"):
    logger.info(f"--- V6: Asian VWAP Mean-Reversion on {symbol} ---")
    df = yf.download(symbol, period="2y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    df = build_simple_features(df)
    
    time_col = 'datetime' if 'datetime' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
    df['hour'] = pd.to_datetime(df[time_col], utc=True).dt.hour

    # Target: Revert to SMA
    df['future_ret'] = (df['close'].shift(-4) - df['close']) / df['close']
    
    # Label 1 if reverting to mean (If below SMA, it should rise. If above SMA, it should fall)
    df['target'] = np.where(df['dist_sma'] < -0.001, np.where(df['future_ret'] > 0, 1, 0),
                   np.where(df['dist_sma'] > 0.001, np.where(df['future_ret'] < 0, 1, 0), 0))

    X = df[['rsi14', 'atr14', 'dist_sma', 'hour']]
    y = df['target']
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test = X.iloc[split:]

    model = lgb.LGBMClassifier(num_leaves=7, max_depth=3, n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    test_df = df.iloc[split:].copy()
    prob = model.predict_proba(X_test)[:, 1]

    # Execution Logic
    test_df['is_asian'] = np.where((test_df['hour'] >= 22) | (test_df['hour'] <= 6), 1, 0)
    test_df['signal'] = 0
    
    # Buy signal (Price is below SMA, predict rise)
    test_df.loc[(prob > 0.60) & (test_df['is_asian'] == 1) & (test_df['dist_sma'] < -0.0005), 'signal'] = 1
    # Sell signal (Price is above SMA, predict fall)
    test_df.loc[(prob > 0.60) & (test_df['is_asian'] == 1) & (test_df['dist_sma'] > 0.0005), 'signal'] = -1

    cost = 0.00015
    test_df['strat_ret'] = test_df['signal'] * np.where(test_df['signal'] == 1, test_df['future_ret'], -test_df['future_ret'])
    test_df['strat_ret_net'] = np.where(test_df['signal'] != 0, test_df['strat_ret'] - cost, 0)

    active = test_df[test_df['signal'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(test_df['strat_ret_net'].cumsum()).iloc[-1] - 1 if len(active) > 0 else 0
    
    gross_prof = active[active['strat_ret_net'] > 0]['strat_ret_net'].sum()
    gross_loss = abs(active[active['strat_ret_net'] <= 0]['strat_ret_net'].sum())
    pf = gross_prof / gross_loss if gross_loss > 0 else float('inf')

    logger.info(f"[{symbol}] Trades: {len(active)} | WinRate: {win_rate*100:.1f}% | PF: {pf:.2f} | Net: {net_ret*100:.2f}%")
    return model

def backtest_v6_gold_micro_breakout():
    logger.info(f"--- V6: Micro-Breakout & Time-Based Exit on GOLD ---")
    df = yf.download("GC=F", period="2y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    import pandas_ta as ta
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['roll_max_4'] = df['high'].rolling(4).max()
    df['roll_min_4'] = df['low'].rolling(4).min()
    df = df.dropna().reset_index()

    # Target: Will it break the 4-bar High + 0.1 ATR and sustain?
    df['target_vol'] = np.where(df['high'].shift(-1) > df['roll_max_4'] + (df['atr14']*0.1), 1, 0)

    X = df[['atr14', 'roll_max_4', 'roll_min_4']]
    y = df['target_vol']
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test = X.iloc[split:]

    model = lgb.LGBMClassifier(num_leaves=15, max_depth=4, n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    test_df = df.iloc[split:].copy()
    prob = model.predict_proba(X_test)[:, 1]

    cost = 0.0003
    test_df['strat_ret_net'] = 0.0
    trades = 0

    for i in range(len(test_df)-6):
        if prob[i] > 0.65:
            # Trap at rolling max + 0.1 ATR (Tighter than V4's 0.5 ATR)
            buy_stop = test_df['roll_max_4'].iloc[i] + (test_df['atr14'].iloc[i] * 0.1)
            next_high = test_df['high'].iloc[i+1]
            
            if next_high > buy_stop:
                trades += 1
                # Time-Based Exit (Exit strictly after 3 bars instead of waiting for TP/SL)
                exit_price = test_df['close'].iloc[i+3] 
                ret = (exit_price - buy_stop) / buy_stop
                test_df.loc[test_df.index[i], 'strat_ret_net'] = ret - cost

    active = test_df[test_df['strat_ret_net'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(test_df['strat_ret_net'].cumsum()).iloc[-1] - 1 if len(active) > 0 else 0
    
    gross_prof = active[active['strat_ret_net'] > 0]['strat_ret_net'].sum()
    gross_loss = abs(active[active['strat_ret_net'] <= 0]['strat_ret_net'].sum())
    pf = gross_prof / gross_loss if gross_loss > 0 else float('inf')

    logger.info(f"[GOLD] Trades: {trades} | WinRate: {win_rate*100:.1f}% | PF: {pf:.2f} | Net: {net_ret*100:.2f}%")

if __name__ == "__main__":
    logger.info("=== V6 INTERNET ALPHA RESEARCH ===")
    backtest_v6_asian_vwap("GBPUSD=X")
    backtest_v6_asian_vwap("JPY=X")
    backtest_v6_gold_micro_breakout()