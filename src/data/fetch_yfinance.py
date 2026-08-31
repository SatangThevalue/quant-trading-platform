import sys
import os
from loguru import logger
import yfinance as yf
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.database import engine

def fetch_and_save_data(symbol="EURUSD=X", period="5y", interval="1d"):
    """Fetch market data from yfinance and save to database."""
    logger.info(f"Fetching data for {symbol} ({period} / {interval})...")
    
    # Download data
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        logger.warning(f"No data fetched for {symbol}")
        return
        
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Format to match our schema
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    # Map index column to timestamp if needed
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    elif 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
        
    # Ensure all required columns exist
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return
            
    # Add metadata
    df['symbol'] = symbol
    df['timeframe'] = interval
    df['source'] = 'yfinance'
    df['spread'] = 0.0 # yfinance doesn't provide spread
    
    # Save raw to bronze layer (Parquet) for backup
    bronze_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "bronze")
    os.makedirs(bronze_path, exist_ok=True)
    file_path = os.path.join(bronze_path, f"{symbol.replace('=', '')}_{interval}.parquet")
    df.to_parquet(file_path)
    logger.info(f"Saved {len(df)} rows to Bronze Lake: {file_path}")
    
    # Process for Database
    # In a real system, we'd look up the asset_id first.
    # For this demo, we'll insert it into assets if not exists
    with engine.connect() as conn:
        from sqlalchemy import text
        
        # 1. Insert into Assets
        canonical_symbol = symbol.replace("=X", "")
        conn.execute(text(f"""
            INSERT INTO assets (symbol, asset_class, base_currency, quote_currency)
            VALUES ('{canonical_symbol}', 'FOREX', '{canonical_symbol[:3]}', '{canonical_symbol[3:]}')
            ON CONFLICT (symbol) DO NOTHING;
        """))
        conn.commit()
        
        # 2. Get Asset ID
        result = conn.execute(text(f"SELECT id FROM assets WHERE symbol = '{canonical_symbol}'")).fetchone()
        asset_id = result[0]
        
        # 3. Prepare for market_ohlcv
        db_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'source', 'timeframe']].copy()
        db_df['asset_id'] = asset_id
        
        # 4. Save to Database
        db_df.to_sql("market_ohlcv", engine, if_exists="append", index=False, method="multi", chunksize=1000)
        logger.info(f"Saved {len(db_df)} rows to Database table: market_ohlcv")

if __name__ == "__main__":
    fetch_and_save_data(symbol="EURUSD=X", period="5y", interval="1d")
    fetch_and_save_data(symbol="GC=F", period="5y", interval="1d")  # Gold
