import sys
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def create_price_features(df):
    """Layer 1: Return Features"""
    periods = [1, 3, 5, 10, 20]
    for p in periods:
        df[f"ret_{p}"] = df["close"].pct_change(p)
        
    for p in [1, 5, 20]:
        df[f"log_ret_{p}"] = np.log(df["close"] / df["close"].shift(p))
        
    return df

def create_candle_features(df):
    """Layer 2: Candle Features"""
    df["body_size"] = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_shadow"] = df[["open","close"]].min(axis=1) - df["low"]
    
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = df["body_size"] / candle_range
    return df

def create_trend_features(df):
    """Layer 3: Trend Features"""
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)

    df["ema20_50_gap"] = (df["ema20"] - df["ema50"]) / df["ema50"]
    df["ema50_200_gap"] = (df["ema50"] - df["ema200"]) / df["ema200"]
    
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None and not adx.empty:
        df["adx14"] = adx["ADX_14"]
    
    return df

def create_momentum_features(df):
    """Layer 4: Momentum Features"""
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["rsi28"] = ta.rsi(df["close"], length=28)
    
    df["rsi_distance_50"] = df["rsi14"] - 50
    
    macd = ta.macd(df["close"])
    if macd is not None and not macd.empty:
        # Check standard column names for MACD in pandas-ta
        cols = macd.columns.tolist()
        macd_col = [c for c in cols if c.startswith("MACD_")][0]
        sig_col = [c for c in cols if c.startswith("MACDs_")][0]
        hist_col = [c for c in cols if c.startswith("MACDh_")][0]
        
        df["macd"] = macd[macd_col]
        df["macd_signal"] = macd[sig_col]
        df["macd_hist"] = macd[hist_col]
        
    return df

def create_volatility_features(df):
    """Layer 5: Volatility Features"""
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr100"] = ta.atr(df["high"], df["low"], df["close"], length=100)
    df["atr_ratio"] = df["atr14"] / df["atr100"]

    bb = ta.bbands(df["close"], length=20)
    if bb is not None and not bb.empty:
        cols = bb.columns.tolist()
        up_col = [c for c in cols if c.startswith("BBU_")][0]
        dn_col = [c for c in cols if c.startswith("BBL_")][0]
        df["bb_width"] = bb[up_col] - bb[dn_col]
        
    return df

def create_structure_features(df):
    """Layer 7: Market Structure Features"""
    rolling_high = df["high"].rolling(20).max()
    rolling_low = df["low"].rolling(20).min()
    df["donchian_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low)
    return df

def create_interaction_features(df):
    """Layer 12: Feature Interaction"""
    if "rsi14" in df.columns and "atr_ratio" in df.columns:
        df["momentum_volatility"] = df["rsi14"] * df["atr_ratio"]
        
    if "adx14" in df.columns and "ema20_50_gap" in df.columns:
        df["trend_strength"] = df["adx14"] * df["ema20_50_gap"]
        
    return df

def build_features(df):
    """Master Pipeline to generate all features"""
    logger.info("Generating features...")
    df = df.copy()
    
    df = create_price_features(df)
    df = create_candle_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_structure_features(df)
    df = create_interaction_features(df)
    
    df["target"] = np.where(df["close"].shift(-5) > df["close"], 1, 0)
    
    initial_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    logger.info(f"Feature generation complete. Kept {len(df)}/{initial_len} rows after dropping warmup NaNs.")
    return df
