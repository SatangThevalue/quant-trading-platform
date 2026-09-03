import yfinance as yf
from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
from loguru import logger
import os

app = FastAPI(title="Macro Economics API for MT5")

# Caching mechanism to prevent hitting Yahoo Finance too hard
class MacroCache:
    def __init__(self):
        self.us10y_diff = 0.0
        self.last_update = None

cache = MacroCache()

def update_macro_data():
    try:
        logger.info("Fetching latest US10Y data from Yahoo Finance...")
        # Get last 5 days of hourly data to ensure we have enough for a diff
        df = yf.download("^TNX", period="5d", interval="1h", progress=False)
        
        if df.empty:
            logger.error("Yahoo Finance returned empty data for ^TNX")
            return False
            
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]
            
        df = df.ffill().fillna(0)
        
        if len(df) >= 2:
            # Calculate the absolute difference between the last two closed bars
            latest_val = float(df.iloc[-1].iloc[0] if isinstance(df.iloc[-1], pd.Series) else df.iloc[-1])
            prev_val = float(df.iloc[-2].iloc[0] if isinstance(df.iloc[-2], pd.Series) else df.iloc[-2])
            diff = latest_val - prev_val
            
            cache.us10y_diff = diff
            cache.last_update = pd.Timestamp.utcnow()
            logger.info(f"Updated Cache: US10Y Diff = {diff:.5f}")
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Failed to update macro data: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    # Initial fetch when server starts
    update_macro_data()

@app.get("/macro")
async def get_macro_features():
    """
    Endpoint for MT5 EA to call via WebRequest.
    Forces an update if data is older than 1 hour.
    """
    now = pd.Timestamp.utcnow()
    
    # If cache is older than 1 hour or empty, update it
    if cache.last_update is None or (now - cache.last_update).total_seconds() > 3600:
        success = update_macro_data()
        if not success and cache.last_update is None:
            raise HTTPException(status_code=503, detail="Macro data unavailable")
            
    return {
        "status": "success",
        "timestamp": cache.last_update.isoformat() if cache.last_update else None,
        "features": {
            "us10y_diff": cache.us10y_diff
        }
    }

if __name__ == "__main__":
    logger.info("Starting Macro Economics API Gateway on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)