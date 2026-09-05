import os
import pandas as pd
import numpy as np
import requests
from loguru import logger
import yfinance as yf
from datetime import datetime, timedelta

# Warning: You must set FRED_API_KEY environment variable. 
# For demonstration/paper-trading, we use a placeholder or public proxies if key is missing.
FRED_API_KEY = os.environ.get("FRED_API_KEY", "dummy_key") 

class AlternativeDataPipeline:
    def __init__(self):
        self.fred = None
        # Removed hard dependency on fredapi package for pure public proxy mode
        pass

    # 1. GOLD MACRO: Real Yields (TIPS)
    def fetch_real_yields_proxy(self, period="2y") -> pd.DataFrame:
        """
        Retrieves US 10-Year Real Yields (TIPS). 
        Since we might not have a FRED API key locally, we use a public proxy via yfinance (TIP ETF) 
        as an inverse proxy for real yields, combined with Nominal Yields (^TNX).
        """
        logger.info("Fetching Real Yield Proxies for GOLD...")
        try:
            # ^TNX = Nominal 10Y Yield, TIP = iShares TIPS Bond ETF
            # If TIP price goes up, Real Yields are going DOWN (Bullish for Gold)
            data = yf.download(["^TNX", "TIP"], period=period, interval="1d", progress=False)['Close']
            
            # Forward fill to handle missing days
            data = data.ffill().bfill()
            
            # Feature: Ratio between Nominal Yield and TIPS price
            data['nominal_yield'] = data['^TNX']
            data['tips_price'] = data['TIP']
            data['real_yield_proxy'] = data['nominal_yield'] / data['tips_price']
            
            # Rate of Change
            data['real_yield_momentum'] = data['real_yield_proxy'].pct_change(5)
            
            return data[['real_yield_proxy', 'real_yield_momentum']].reset_index().rename(columns={'Date': 'timestamp', 'index': 'timestamp'})
        except Exception as e:
            logger.error(f"Failed to fetch Real Yields: {e}")
            return pd.DataFrame()

    # 2. CRYPTO MICROSTRUCTURE: Funding Rate & Open Interest (Binance API)
    def fetch_binance_funding_rates(self, symbol="BTCUSDT", limit=1000) -> pd.DataFrame:
        """
        Fetches historical funding rates from Binance Futures.
        High funding rate = Retail is overwhelmingly Long (Danger of Long Squeeze / Crash).
        """
        logger.info(f"Fetching Binance Funding Rates for {symbol}...")
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                
                # Create Features
                df['funding_rolling_mean'] = df['fundingRate'].rolling(window=8).mean() # 8 periods = 24 hours (pays every 8h usually)
                df['is_extreme_greed'] = np.where(df['fundingRate'] > 0.0001, 1, 0) # 0.01% is baseline
                df['is_extreme_fear'] = np.where(df['fundingRate'] < -0.0001, 1, 0)
                
                return df[['timestamp', 'fundingRate', 'funding_rolling_mean', 'is_extreme_greed', 'is_extreme_fear']]
            else:
                logger.error(f"Binance API Error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch Binance data: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    pipeline = AlternativeDataPipeline()
    
    # Test Gold Data
    gold_macro = pipeline.fetch_real_yields_proxy()
    if not gold_macro.empty:
        logger.info(f"Successfully fetched {len(gold_macro)} days of Real Yield Proxy data.")
        
    # Test Crypto Data
    crypto_micro = pipeline.fetch_binance_funding_rates()
    if not crypto_micro.empty:
        logger.info(f"Successfully fetched {len(crypto_micro)} periods of Binance Funding Rates.")