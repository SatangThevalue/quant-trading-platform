import sys, os, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# --- Helper Functions ---
def get_data(ticker, period="3y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()

def calc_atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# --- Method 1: Volatility-Targeted CTA (Trend + Risk Parity) ---
def test_vol_target_cta(symbol="GC=F"):
    logger.info(f"Method 1: Volatility-Targeted CTA on {symbol}")
    df = get_data(symbol, "3y", "1d")
    df['atr'] = calc_atr(df, 14)
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df = df.dropna()
    
    # Trend Signal
    df['signal'] = np.where(df['sma50'] > df['sma200'], 1, -1)
    
    # Volatility Sizing: Target daily volatility of 1% (0.01)
    # Position size = Target Vol / Current Vol (ATR%)
    df['atr_pct'] = df['atr'] / df['close']
    df['weight'] = 0.01 / (df['atr_pct'] + 1e-9)
    df['weight'] = df['weight'].clip(0, 2.0) # Max leverage 2x
    
    df['ret'] = df['close'].pct_change().shift(-1)
    df['strat_ret'] = df['signal'] * df['weight'] * df['ret']
    
    df['strat_ret'] = df['strat_ret'].fillna(0)
    net_ret = np.exp(df['strat_ret'].cumsum()).iloc[-1] - 1
    sharpe = np.sqrt(252) * df['strat_ret'].mean() / df['strat_ret'].std() if df['strat_ret'].std() != 0 else 0
    logger.info(f"[M1 Vol-Target] Net Return: {net_ret*100:.2f}% | Sharpe: {sharpe:.2f}")

# --- Method 2: Meta-Labeling (Lopez de Prado) ---
def test_meta_labeling(symbol="GBPUSD=X"):
    logger.info(f"Method 2: Meta-Labeling on {symbol}")
    df = get_data(symbol, "2y", "1h")
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['atr'] = calc_atr(df, 14)
    df = df.dropna()
    
    # Primary Model (Simple trend crossover)
    df['primary_signal'] = np.where(df['sma20'] > df['sma50'], 1, -1)
    df['future_ret'] = (df['close'].shift(-6) - df['close']) / df['close']
    df['primary_profit'] = df['primary_signal'] * df['future_ret'] - 0.0002 # Include spread
    
    # Meta-Label: 1 if Primary Model is profitable, 0 otherwise
    df['meta_target'] = np.where(df['primary_profit'] > 0, 1, 0)
    df = df.dropna()
    
    features = ['sma20', 'sma50', 'atr']
    X = df[features]
    y = df['meta_target']
    
    split = int(len(df)*0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    meta_model = lgb.LGBMClassifier(num_leaves=7, max_depth=3, n_estimators=50, verbose=-1)
    meta_model.fit(X_train, y_train)
    
    test_df = df.iloc[split:].copy()
    test_df['meta_prob'] = meta_model.predict_proba(X_test)[:, 1]
    
    # Trade only if Meta-Model confidence > 0.60
    test_df['final_signal'] = np.where(test_df['meta_prob'] > 0.60, test_df['primary_signal'], 0)
    test_df['final_ret'] = np.where(test_df['final_signal'] != 0, test_df['primary_signal'] * test_df['future_ret'] - 0.0002, 0)
    
    net_ret = np.exp(test_df['final_ret'].cumsum()).iloc[-1] - 1
    logger.info(f"[M2 Meta-Labeling] Net Return: {net_ret*100:.2f}% | Trades: {len(test_df[test_df['final_signal']!=0])}")

# --- Method 3: Macro-Yield Spread Cointegration ---
def test_macro_yield_spread(symbol="JPY=X"):
    logger.info(f"Method 3: Macro-Yield Spread on {symbol} vs US10Y")
    df_jpy = get_data(symbol, "3y", "1d")
    df_yield = get_data("^TNX", "3y", "1d")
    
    df = pd.merge(df_jpy['close'], df_yield['close'], left_index=True, right_index=True, how='inner')
    df.columns = ['jpy', 'us10y']
    
    # Z-Score of the rolling correlation/spread
    df['spread'] = df['jpy'] - (df['us10y'] * 10) # rough scalar
    df['z_score'] = (df['spread'] - df['spread'].rolling(60).mean()) / df['spread'].rolling(60).std()
    df = df.dropna()
    
    # Mean reversion on spread
    df['signal'] = 0
    df.loc[df['z_score'] > 2, 'signal'] = -1
    df.loc[df['z_score'] < -2, 'signal'] = 1
    
    df['ret'] = df['jpy'].pct_change().shift(-1)
    df['strat_ret'] = df['signal'] * df['ret']
    
    df['strat_ret'] = df['strat_ret'].fillna(0)
    net_ret = np.exp(df['strat_ret'].cumsum()).iloc[-1] - 1
    logger.info(f"[M3 Macro-Spread] Net Return: {net_ret*100:.2f}% | Trades: {len(df[df['signal']!=0])}")

# --- Method 4: Deep Learning (MLP / Feedforward Neural Net) ---
def test_deep_learning(symbol="BTC-USD"):
    logger.info(f"Method 4: Deep Learning (MLP Sequence) on {symbol}")
    df = get_data(symbol, "2y", "1h")
    
    # Create Lagged Returns to simulate Sequence Memory
    df['ret'] = df['close'].pct_change()
    for i in range(1, 11):
        df[f'ret_lag_{i}'] = df['ret'].shift(i)
        
    df['future_ret'] = (df['close'].shift(-6) - df['close']) / df['close']
    df['target'] = np.where(df['future_ret'] > 0.005, 1, 0)
    df = df.dropna()
    
    features = [f'ret_lag_{i}' for i in range(1, 11)]
    X = df[features]
    y = df['target']
    
    split = int(len(df)*0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    # MLP acts as a basic Universal Function Approximator
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    
    test_df = df.iloc[split:].copy()
    prob = mlp.predict_proba(X_test)[:, 1]
    
    test_df['signal'] = np.where(prob > 0.65, 1, 0)
    test_df['strat_ret'] = test_df['signal'] * test_df['future_ret'] - (test_df['signal'] * 0.002)
    
    net_ret = np.exp(test_df['strat_ret'].cumsum()).iloc[-1] - 1
    logger.info(f"[M4 Deep Learning] Net Return: {net_ret*100:.2f}% | Trades: {len(test_df[test_df['signal']==1])}")

# --- Method 5: HMM Adaptive Regime Breakout ---
def test_hmm_adaptive(symbol="GC=F"):
    logger.info(f"Method 5: HMM Adaptive Strategy on {symbol}")
    df = get_data(symbol, "3y", "1d")
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close']
    df = df.dropna()
    
    X = df[['ret', 'range']].values
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    hmm.fit(X)
    df['regime'] = hmm.predict(X)
    
    # Strategy: Breakout in Regime 1, Mean Revert in Regime 0
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    
    df['future_ret'] = df['close'].pct_change().shift(-1)
    df = df.dropna()
    
    df['signal'] = 0
    for i in range(len(df)):
        regime = df['regime'].iloc[i]
        close = df['close'].iloc[i]
        upper = df['upper'].iloc[i]
        lower = df['lower'].iloc[i]
        
        if regime == 1: # Assume High Vol -> Breakout
            if close > upper: df['signal'].iloc[i] = 1
            elif close < lower: df['signal'].iloc[i] = -1
        else: # Low Vol -> Mean Reversion
            if close > upper: df['signal'].iloc[i] = -1
            elif close < lower: df['signal'].iloc[i] = 1
            
    df['strat_ret'] = df['signal'] * df['future_ret'] - (abs(df['signal']) * 0.0005)
    df['strat_ret'] = df['strat_ret'].fillna(0)
    net_ret = np.exp(df['strat_ret'].cumsum()).iloc[-1] - 1
    logger.info(f"[M5 HMM Adaptive] Net Return: {net_ret*100:.2f}% | Trades: {len(df[df['signal']!=0])}")

if __name__ == "__main__":
    logger.info("=== INSTITUTIONAL QUANT METHODOLOGY TEST ===")
    test_vol_target_cta("GC=F")        # GOLD
    test_meta_labeling("GBPUSD=X")     # GBP
    test_macro_yield_spread("JPY=X")   # JPY
    test_deep_learning("BTC-USD")      # BTC
    test_hmm_adaptive("GC=F")          # GOLD
