import sys
import os
import pandas as pd
from prefect import task, flow
from loguru import logger
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.database import engine
from src.features.feature_pipeline import build_features

@task
def load_raw_data_from_db(symbol="EURUSD", timeframe="1d"):
    """Load raw OHLCV data from PostgreSQL/SQLite"""
    logger.info(f"Task: Loading raw data for {symbol} ({timeframe})...")
    
    query = f"""
        SELECT m.* 
        FROM market_ohlcv m
        JOIN assets a ON m.asset_id = a.id
        WHERE a.symbol = '{symbol}' AND m.timeframe = '{timeframe}'
        ORDER BY m.timestamp ASC
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.warning(f"No data found for {symbol} in database.")
    else:
        logger.info(f"Loaded {len(df)} rows.")
        
    return df

@task
def apply_feature_engineering(df):
    """Run the feature engineering pipeline"""
    if df.empty:
        return pd.DataFrame()
        
    logger.info("Task: Applying feature engineering pipeline...")
    return build_features(df)

@task
def save_features_to_store(df, symbol):
    """Save processed features to the database and parquet"""
    if df.empty:
        return False
        
    # Save to Gold Lake (Parquet)
    gold_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gold")
    os.makedirs(gold_path, exist_ok=True)
    
    file_path = os.path.join(gold_path, f"{symbol}_features.parquet")
    df.to_parquet(file_path)
    logger.info(f"Task: Saved {len(df)} rows to Gold Lake: {file_path}")
    
    # In a real production system, we would insert these into the 'features_wide_v1' table
    # For this demo, we'll just log success.
    logger.info(f"Task: Features ready for training pipeline.")
    return True

@flow(name="Feature Engineering Flow")
def run_feature_engineering_pipeline():
    """Main Orchestration Flow for Feature Engineering"""
    logger.info("Starting Feature Engineering Pipeline...")
    
    symbols = ["EURUSD", "GBPUSD", "GCF"]
    
    for sym in symbols:
        raw_df = load_raw_data_from_db(symbol=sym, timeframe="1d")
        feature_df = apply_feature_engineering(raw_df)
        save_features_to_store(feature_df, sym)
        
    logger.info("Feature Engineering Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_feature_engineering_pipeline()
