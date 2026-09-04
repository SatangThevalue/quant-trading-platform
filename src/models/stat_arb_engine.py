import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from loguru import logger

class StatisticalArbitrageEngine:
    def __init__(self, asset1="EURUSD=X", asset2="GBPUSD=X", period="60d", interval="15m"):
        self.asset1 = asset1
        self.asset2 = asset2
        self.period = period
        self.interval = interval
        self.hedge_ratio = None
        self.mean = None
        self.std = None

    def fetch_data(self):
        logger.info(f"Fetching Pair Data: {self.asset1} vs {self.asset2}")
        df1 = yf.download(self.asset1, period=self.period, interval=self.interval, progress=False)['Close']
        df2 = yf.download(self.asset2, period=self.period, interval=self.interval, progress=False)['Close']
        
        # Merge and align timestamps
        df = pd.concat([df1, df2], axis=1).dropna()
        df.columns = ['Asset1', 'Asset2']
        return df

    def train_pair(self):
        """Train the stat arb model to find the hedge ratio and spread statistics."""
        df = self.fetch_data()
        if df.empty or len(df) < 100:
            logger.error("Not enough data to train Pair Trading model.")
            return False

        # OLS Regression to find the Hedge Ratio (Asset1 = beta * Asset2 + alpha)
        y = df['Asset1']
        X = sm.add_constant(df['Asset2'])
        model = sm.OLS(y, X).fit()
        self.hedge_ratio = model.params['Asset2']

        # Calculate Spread
        df['Spread'] = df['Asset1'] - (self.hedge_ratio * df['Asset2'])
        
        # Check Cointegration (Is the spread stationary/mean-reverting?)
        adf_result = adfuller(df['Spread'])
        p_value = adf_result[1]
        
        self.mean = df['Spread'].mean()
        self.std = df['Spread'].std()

        logger.info("--- Statistical Arbitrage Training Results ---")
        logger.info(f"Pair: {self.asset1} - ({self.hedge_ratio:.4f} * {self.asset2})")
        logger.info(f"Cointegration P-Value: {p_value:.4f} " + ("(Passed ✅)" if p_value < 0.05 else "(Failed ❌)"))
        logger.info(f"Spread Mean: {self.mean:.5f}, Spread Std: {self.std:.5f}")
        
        # Allow trading only if the pair is historically cointegrated (p-value < 0.05)
        return p_value < 0.05

    def get_current_signal(self):
        """Calculates the current Z-Score of the spread and returns trading signals."""
        if self.hedge_ratio is None:
            logger.error("Model not trained.")
            return 0
            
        # Fetch the latest 1 hour of data to get the current price
        df = yf.download([self.asset1, self.asset2], period="1d", interval="15m", progress=False)['Close']
        df = df.ffill().fillna(0)
        
        if df.empty or len(df.columns) < 2: return 0
            
        current_p1 = float(df[self.asset1].iloc[-1])
        current_p2 = float(df[self.asset2].iloc[-1])
        
        current_spread = current_p1 - (self.hedge_ratio * current_p2)
        z_score = (current_spread - self.mean) / self.std
        
        logger.info(f"[StatArb] Current Z-Score: {z_score:.2f}")
        
        # Trading Logic
        # Z-Score > 2.0 (Spread is too high: Sell Asset1, Buy Asset2)
        if z_score > 2.0:
            logger.warning(f"🚨 ARBITRAGE SIGNAL: SELL {self.asset1} & BUY {self.asset2}")
            return -1
        # Z-Score < -2.0 (Spread is too low: Buy Asset1, Sell Asset2)
        elif z_score < -2.0:
            logger.warning(f"🚨 ARBITRAGE SIGNAL: BUY {self.asset1} & SELL {self.asset2}")
            return 1
        # Close positions if Z-Score returns to 0
        elif abs(z_score) < 0.5:
            logger.info("Spread Normalized. Close all Hedge positions.")
            return 0
            
        return 0

if __name__ == "__main__":
    arb = StatisticalArbitrageEngine()
    is_tradable = arb.train_pair()
    if is_tradable:
        arb.get_current_signal()
