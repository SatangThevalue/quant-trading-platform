import sys
import os
from prefect import task, flow
from loguru import logger
import yfinance as yf
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.database import engine

@task(retries=3, retry_delay_seconds=10)
def fetch_market_data(symbol="EURUSD=X", period="5y", interval="1d"):
    """Fetch market data from yfinance."""
    logger.info(f"Task: Fetching data for {symbol} ({period} / {interval})...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        logger.warning(f"No data fetched for {symbol}")
        return pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    elif 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
        
    df['symbol'] = symbol
    df['timeframe'] = interval
    df['source'] = 'yfinance'
    df['spread'] = 0.0
    
    return df

@task
def save_raw_data_to_lake(df, symbol, interval):
    """Save raw dataframe to Bronze data lake (Parquet)."""
    if df.empty:
        return False
        
    bronze_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "bronze")
    os.makedirs(bronze_path, exist_ok=True)
    
    clean_symbol = symbol.replace('=', '')
    file_path = os.path.join(bronze_path, f"{clean_symbol}_{interval}.parquet")
    
    df.to_parquet(file_path)
    logger.info(f"Task: Saved {len(df)} rows to Bronze Lake: {file_path}")
    return True

@task
def save_data_to_db(df, symbol):
    """Insert or update data in the PostgreSQL/SQLite database."""
    if df.empty:
        return False
        
    canonical_symbol = symbol.replace("=X", "")
    
    with engine.connect() as conn:
        # 1. Insert Asset if not exists
        conn.execute(text(f"""
            INSERT INTO assets (symbol, asset_class, base_currency, quote_currency)
            VALUES ('{canonical_symbol}', 'FOREX', '{canonical_symbol[:3]}', '{canonical_symbol[3:]}')
            ON CONFLICT (symbol) DO NOTHING;
        """))
        conn.commit()
        
        # 2. Get Asset ID
        result = conn.execute(text(f"SELECT id FROM assets WHERE symbol = '{canonical_symbol}'")).fetchone()
        if not result:
            logger.error(f"Failed to find or create asset for {canonical_symbol}")
            return False
            
        asset_id = result[0]
        
        # 3. Save to market_ohlcv (simplified for demo: ignoring duplicates)
        # In production, we'd use UPSERT (ON CONFLICT DO UPDATE)
        db_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'source', 'timeframe']].copy()
        db_df['asset_id'] = asset_id
        
        try:
            db_df.to_sql("market_ohlcv", engine, if_exists="append", index=False, method="multi", chunksize=1000)
            logger.info(f"Task: Saved {len(db_df)} rows to Database table: market_ohlcv")
        except Exception as e:
            # Catching integrity errors for duplicates in this demo script
            logger.warning(f"Database insert warning (might be duplicates): {e}")
            
    return True

@flow(name="Market Data Collection Flow")
def run_data_collection_pipeline():
    """Main Orchestration Flow for Data Collection"""
    logger.info("Starting Market Data Collection Pipeline...")
    
    symbols = ["EURUSD=X", "GBPUSD=X", "GC=F", "JPY=X", "AUDUSD=X"] # EUR, GBP, Gold, JPY, AUD
    
    for sym in symbols:
        # 1. Fetch
        df = fetch_market_data(symbol=sym, period="5y", interval="1d")
        
        # 2. Save to Lake
        save_raw_data_to_lake(df, sym, "1d")
        
        # 3. Save to DB
        save_data_to_db(df, sym)
        
    logger.info("Data Collection Pipeline Completed Successfully.")

if __name__ == "__main__":
    # Test running the flow locally
    run_data_collection_pipeline()
