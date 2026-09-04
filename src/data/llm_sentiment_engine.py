import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
import os

# Placeholder for LLM - In real production, we use OpenAI API or FinBERT
class FinLLMSentiment:
    def __init__(self):
        # We simulate a "Financial Context Aware" model.
        # FinBERT/LLM understands that "Weak Jobs" = USD Down = GOLD Up.
        # This is a mock dictionary representing what a FinLLM would map headlines to for GOLD.
        self.financial_context_map = {
            "Weak": 1.0,         # Weak USD -> Gold UP
            "Fall": 1.0,         # Yields/USD Fall -> Gold UP
            "Dovish": 1.0,       # Fed dovish -> Gold UP
            "Geopolitical": 1.0, # Unrest -> Gold UP
            "Strong": -1.0,      # Strong USD -> Gold DOWN
            "Hawkish": -1.0,     # Fed Hawkish -> Gold DOWN
            "Rise": -1.0,        # Yields/USD Rise -> Gold DOWN
        }
        
    def analyze_headline_for_asset(self, headline: str, asset: str = "GOLD") -> float:
        """
        Simulates an LLM parsing a headline and returning a directional probability [-1.0 to 1.0].
        1.0 means Strong Buy for the asset. -1.0 means Strong Sell.
        """
        score = 0.0
        words = headline.split()
        for word in words:
            # Simple matching for simulation
            clean_word = word.replace(",", "").replace(";", "")
            for key, val in self.financial_context_map.items():
                if key.lower() in clean_word.lower():
                    score += val
                    
        # Cap score
        if score > 1.0: score = 1.0
        if score < -1.0: score = -1.0
        
        return score

def backtest_llm_gold_strategy():
    """
    Backtest GOLD 1H Strategy using a simulated FinBERT / LLM Sentiment overlay.
    """
    logger.info("Backtesting GOLD with FinLLM Sentiment Overlay...")
    
    # 1. Fetch Gold Data
    df = yf.download("GC=F", period="1y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df = df['Close']
    else: df = df[['Close']]
    df = df.dropna()
    df.columns = ['close']
    
    # 2. Base Strategy (Mean Reversion / Trend)
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['base_signal'] = np.where(df['sma20'] > df['sma50'], 1, -1)
    
    # 3. Simulate News Data (High Impact News occurs randomly)
    np.random.seed(42)
    df['is_news_hour'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    # We assign random "headlines" to news hours
    sample_headlines = [
        "US Dollar Falls as Yields drop", 
        "Strong NFP Jobs Data boosts DXY",
        "Geopolitical Tensions Rise in Middle East",
        "Fed signals Hawkish stance on inflation"
    ]
    
    llm = FinLLMSentiment()
    df['llm_signal'] = 0.0
    
    for i in range(len(df)):
        if df['is_news_hour'].iloc[i] == 1:
            headline = np.random.choice(sample_headlines)
            df['llm_signal'].iloc[i] = llm.analyze_headline_for_asset(headline, "GOLD")
            
    # 4. Trading Logic (LLM Overrides Base Signal)
    df['final_signal'] = df['base_signal']
    
    for i in range(len(df)):
        if df['is_news_hour'].iloc[i] == 1:
            llm_score = df['llm_signal'].iloc[i]
            
            if llm_score > 0.5:
                df['final_signal'].iloc[i] = 1   # Override to BUY
            elif llm_score < -0.5:
                df['final_signal'].iloc[i] = -1  # Override to SELL
            else:
                df['final_signal'].iloc[i] = 0   # Uncertainty -> Block Trade
                
    # 5. Calculate Returns
    df['ret'] = df['close'].pct_change()
    
    df['base_strat'] = df['base_signal'].shift(1) * df['ret']
    df['llm_strat'] = df['final_signal'].shift(1) * df['ret']
    
    # Apply Cost (High spread for Gold = 0.0005)
    cost = 0.0005
    df.loc[df['base_signal'].diff() != 0, 'base_strat'] -= cost
    df.loc[df['final_signal'].diff() != 0, 'llm_strat'] -= cost
    
    base_net = np.exp(df['base_strat'].cumsum()).iloc[-1] - 1
    llm_net = np.exp(df['llm_strat'].cumsum()).iloc[-1] - 1
    
    logger.info("--- LLM Optimization Results (GOLD 1H) ---")
    logger.info(f"Base Strategy Net Return: {base_net*100:.2f}%")
    logger.info(f"FinLLM Strategy Net Return:  {llm_net*100:.2f}%")
    
    return base_net, llm_net

if __name__ == "__main__":
    backtest_llm_gold_strategy()
