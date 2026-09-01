import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_regime_model(df, n_clusters=4):
    """
    Phase 15: Market Regime Detection
    Trains a K-Means clustering model to identify market regimes based on Volatility and Trend.
    """
    logger.info("Training Market Regime Detection Model...")
    
    # Required features for regime clustering
    required_cols = ['adx14', 'atr_ratio', 'bb_width', 'ema20_50_gap']
    
    # Check if we have the columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns for regime detection: {missing}")
        return None, None
        
    # Drop NaNs just for the clustering train phase
    cluster_df = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(cluster_df) < 100:
        logger.warning("Not enough data to reliably cluster regimes.")
        return None, None
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    
    logger.info(f"Regime model trained with {n_clusters} clusters.")
    
    # Save the models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(models_dir, "regime_scaler.pkl"))
    joblib.dump(kmeans, os.path.join(models_dir, "regime_kmeans.pkl"))
    
    return kmeans, scaler

def apply_regime(df):
    """Applies the trained regime model to label current data."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    scaler_path = os.path.join(models_dir, "regime_scaler.pkl")
    kmeans_path = os.path.join(models_dir, "regime_kmeans.pkl")
    
    if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path):
        logger.warning("Regime models not found. Training new ones...")
        kmeans, scaler = train_regime_model(df)
        if kmeans is None:
            return df
            
    else:
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        
    required_cols = ['adx14', 'atr_ratio', 'bb_width', 'ema20_50_gap']
    
    # For safety, fill missing with forward fill then 0
    clean_data = df[required_cols].ffill().fillna(0)
    scaled_data = scaler.transform(clean_data)
    
    df['market_regime'] = kmeans.predict(scaled_data)
    logger.info("Market Regime labels added to dataset.")
    
    return df

if __name__ == "__main__":
    logger.info("Regime Detection Module Ready.")
