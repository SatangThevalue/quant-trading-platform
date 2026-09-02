import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import sys, os
from loguru import logger
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.feature_pipeline import build_features
from src.models.export_onnx import export_model_to_onnx, validate_onnx_model

# --------------------------------------------------------------------------------
# PM Note: To train an Intraday model (e.g., 1H) that matches D1 stability,
# we MUST adjust the Feature Engineering pipeline internally to handle noise.
# We also use a stricter Target Label (e.g., predicting further out or adding a threshold)
# --------------------------------------------------------------------------------

def build_intraday_features(df):
    """
    V2 Intraday Feature Pipeline (Overriding V1 for 1H/4H specifics)
    In a real repo, this would be a cleanly separated module.
    """
    logger.info("Building Intraday (V2) Features...")
    df = df.copy()
    
    # 1. Price & Volatility (Extended windows to smooth Intraday noise)
    periods = [1, 5, 10, 20, 50]
    for p in periods:
        df[f"ret_{p}"] = df["close"].pct_change(p)
        
    df["body_size"] = abs(df["close"] - df["open"])
    
    import pandas_ta as ta
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    
    # EMA Gaps are critical for filtering sideways 1H noise
    df["ema20_50_gap"] = (df["ema20"] - df["ema50"]) / df["ema50"]
    df["ema50_200_gap"] = (df["ema50"] - df["ema200"]) / df["ema200"]
    
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["rsi_distance_50"] = df["rsi14"] - 50
    
    # Volatility context (Critical for Intraday sizing)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr100"] = ta.atr(df["high"], df["low"], df["close"], length=100)
    df["atr_ratio"] = df["atr14"] / df["atr100"]
    
    # 12. Interaction: Trend Strength * Volatility
    df["trend_volatility_interaction"] = df["ema20_50_gap"] * df["atr_ratio"]
    
    # Intraday Target Label: Predicting the close 12 bars out (e.g., half a day on 1H)
    # Target 1 if the price rises more than the spread/noise buffer (e.g., > 0.05%)
    noise_buffer = 0.0005 
    future_return = (df["close"].shift(-12) - df["close"]) / df["close"]
    df["target"] = np.where(future_return > noise_buffer, 1, 0)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def train_and_export_intraday(symbol, ticker, timeframe="1h"):
    logger.info(f"--- Processing {symbol} for V2 Intraday Export ({timeframe}) ---")
    
    # Yahoo Finance limit for 1h is 730 days max
    period = "730d" if timeframe == "1h" else "1y" if timeframe == "30m" else "2y"
    df = yf.download(ticker, period=period, interval=timeframe, progress=False)
    
    if df.empty: 
        logger.error(f"No data for {symbol} at {timeframe}")
        return
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
    elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns: df[col] = 0
            
    df_feat = build_intraday_features(df)
    
    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[features]
    y = df_feat["target"]
    
    split_idx = int(len(df_feat) * 0.9)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    # -------------------------------------------------------------
    # PM NOTE: Intraday Hyperparameters
    # We restrict max_depth to prevent overfitting to H1/M30 noise.
    # We increase bagging/feature fraction to make the model more robust.
    # -------------------------------------------------------------
    best_params = {
        'num_leaves': 15,          # Smaller to prevent overfitting noise
        'max_depth': 4,            # Shallower trees
        'learning_rate': 0.01,     # Slower learning
        'feature_fraction': 0.8,   # Random subspace
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**best_params, n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Export naming convention indicates V2 and timeframe
    version = f"2.0_{timeframe}"
    export_success = export_model_to_onnx(model, features, symbol=symbol, version=version)
    
    if export_success:
        onnx_path = os.path.join("models", "production", f"{symbol}_model_v{version}.onnx")
        sample_features = X_test.head(10)
        validate_onnx_model(onnx_path, model, sample_features)

if __name__ == "__main__":
    logger.info("Building Intraday V2 Models...")
    # Training H1 models for both
    train_and_export_intraday("EURUSD", "EURUSD=X", timeframe="1h")
    train_and_export_intraday("USDCHF", "CHF=X", timeframe="1h")
    logger.info("V2 Intraday Export Complete.")