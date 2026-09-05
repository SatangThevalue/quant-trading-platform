import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def build_simple_features(df):
    import pandas_ta as ta
    df = df.copy()
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    return df.dropna()

def backtest_asian_ranger(symbol="GBPUSD=X"):
    """
    Simulates the 'QuantTrader_Asian_Ranger' MQL5 logic.
    Only trades during Asian Session (22:00 - 06:00 UTC for proxy).
    Uses a Mean-Reversion target (buy when RSI is low, sell when high).
    """
    logger.info(f"--- Backtesting Asian Ranger EA on {symbol} ---")
    
    # Fast fetch to avoid timeout
    df = yf.download(symbol, period="1y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    df = build_simple_features(df)
    
    # Extract Hour for Session Filter
    df = df.reset_index()
    time_col = 'Datetime' if 'Datetime' in df.columns else 'index'
    if time_col not in df.columns and 'datetime' in df.columns: time_col = 'datetime'
    elif time_col not in df.columns and 'date' in df.columns: time_col = 'date'
        
    df['hour'] = pd.to_datetime(df[time_col], utc=True).dt.hour
    
    # Target: Reversion (if RSI < 40, expect rise. If > 60, expect fall)
    df['future_ret'] = (df['close'].shift(-4) - df['close']) / df['close']
    df['target'] = np.where(df['future_ret'] > 0, 1, 0)
    df = df.dropna()
    
    X = df[['rsi14', 'atr14', 'hour']]
    y = df['target']
    
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    model = lgb.LGBMClassifier(num_leaves=7, max_depth=3, n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    test_df = df.iloc[split:].copy()
    prob = model.predict_proba(X_test)[:, 1]
    
    # EA Logic: Only allow trades during Asian Session (approx 22:00 to 06:00 UTC)
    test_df['is_asian'] = np.where((test_df['hour'] >= 22) | (test_df['hour'] <= 6), 1, 0)
    
    test_df['signal'] = 0
    test_df.loc[(prob > 0.55) & (test_df['is_asian'] == 1), 'signal'] = 1
    test_df.loc[(prob < 0.45) & (test_df['is_asian'] == 1), 'signal'] = -1
    
    cost = 0.0002
    test_df['strat_ret'] = test_df['signal'] * test_df['future_ret']
    test_df['strat_ret_net'] = np.where(test_df['signal'] != 0, test_df['strat_ret'] - cost, 0)
    
    active = test_df[test_df['signal'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(test_df['strat_ret_net'].cumsum()).iloc[-1] - 1
    
    logger.info(f"[Asian Ranger] Trades: {len(active)} | Win Rate: {win_rate*100:.1f}% | Net Return: {net_ret*100:.2f}%")
    return net_ret

def backtest_xau_sniper(symbol="GC=F"):
    """
    Simulates the 'QuantTrader_XAU_Sniper' MQL5 logic.
    Instead of entering at market, it places Stop Orders (Breakout).
    """
    logger.info(f"--- Backtesting XAU Sniper EA on {symbol} ---")
    df = yf.download(symbol, period="1y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    df = build_simple_features(df)
    
    # Target: Volatility Expansion (Magnitude of move, regardless of direction)
    df['future_high'] = df['high'].rolling(12).max().shift(-12)
    df['future_low'] = df['low'].rolling(12).min().shift(-12)
    df['max_move'] = (df['future_high'] - df['future_low']) / df['close']
    
    # Train AI to predict if the next 12 hours will have a move > 0.5%
    df['target_vol'] = np.where(df['max_move'] > 0.005, 1, 0)
    df = df.dropna()
    
    X = df[['rsi14', 'atr14', 'ema20', 'ema50']]
    y = df['target_vol']
    
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test = X.iloc[split:]
    
    model = lgb.LGBMClassifier(num_leaves=15, max_depth=4, n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    test_df = df.iloc[split:].copy()
    prob_breakout = model.predict_proba(X_test)[:, 1]
    
    # EA Logic: If breakout probability is high, place pending orders (BuyStop/SellStop)
    # If price moves past the stop order distance (0.5 ATR), we enter the trade.
    test_df['signal'] = 0
    test_df['strat_ret_net'] = 0.0
    cost = 0.0005 # Gold spread
    
    trades = 0
    for i in range(len(test_df)-12):
        if prob_breakout[i] > 0.65:
            # Set traps
            atr = test_df['atr14'].iloc[i]
            close = test_df['close'].iloc[i]
            buy_stop = close + (atr * 0.5)
            sell_stop = close - (atr * 0.5)
            
            # Look ahead 1 bar to see if trap is triggered
            next_high = test_df['high'].iloc[i+1]
            next_low = test_df['low'].iloc[i+1]
            next_close = test_df['close'].iloc[i+1]
            
            # Hit Buy Stop
            if next_high > buy_stop:
                trades += 1
                # Simplified TP/SL logic (Assume we exit 6 bars later for simulation)
                exit_price = test_df['close'].iloc[i+6]
                ret = (exit_price - buy_stop) / buy_stop
                test_df['strat_ret_net'].iloc[i] = ret - cost
                
            # Hit Sell Stop
            elif next_low < sell_stop:
                trades += 1
                exit_price = test_df['close'].iloc[i+6]
                ret = (sell_stop - exit_price) / sell_stop
                test_df['strat_ret_net'].iloc[i] = ret - cost

    active = test_df[test_df['strat_ret_net'] != 0]
    win_rate = len(active[active['strat_ret_net'] > 0]) / len(active) if len(active) > 0 else 0
    net_ret = np.exp(test_df['strat_ret_net'].cumsum()).iloc[-1] - 1
    
    logger.info(f"[XAU Sniper] Trades: {trades} | Win Rate: {win_rate*100:.1f}% | Net Return: {net_ret*100:.2f}%")
    return net_ret

if __name__ == "__main__":
    logger.info("=== EVALUATING SPECIALIZED EA STRATEGIES ===")
    
    # Test Group 2: GBPUSD (Asian Ranger)
    # Recall: Previous standard model lost -12% on GBPUSD 15m.
    backtest_asian_ranger("GBPUSD=X")
    backtest_asian_ranger("JPY=X")
    
    # Test Group 3: GOLD (XAU Sniper)
    # Recall: Previous standard model lost -76% on GOLD.
    backtest_xau_sniper("GC=F")
