import sys
import os
import pandas as pd
from loguru import logger
from src.features.correlation_engine import calculate_portfolio_correlation, evaluate_portfolio_exposure

def test_portfolio_engine():
    logger.info("=========================================")
    logger.info("TESTING PHASE 14: PORTFOLIO ENGINE (CORRELATION)")
    logger.info("=========================================")
    
    # 1. Load Dummy Data (Using EURUSD and GCF that we fetched earlier)
    gold_path = os.path.join("data", "gold")
    
    eurusd_path = os.path.join(gold_path, "EURUSD_features.parquet")
    # For testing correlation, we'll pretend we have GBPUSD by slightly modifying EURUSD
    # In reality we'd load actual GBPUSD data
    
    if not os.path.exists(eurusd_path):
        logger.error("EURUSD features not found. Please run Data Collection and Feature Eng flows.")
        return
        
    df_eu = pd.read_parquet(eurusd_path)
    
    # Mocking GBPUSD (Highly correlated to EURUSD)
    df_gu = df_eu.copy()
    df_gu['ret_5'] = df_gu['ret_5'] * 0.9 + np.random.normal(0, 0.001, len(df_gu))
    
    # Mocking USDJPY (Negatively correlated to EURUSD)
    df_uj = df_eu.copy()
    df_uj['ret_5'] = df_uj['ret_5'] * -0.8 + np.random.normal(0, 0.002, len(df_uj))
    
    # Mocking Gold (Low correlation)
    df_gold = df_eu.copy()
    df_gold['ret_5'] = np.random.normal(0, 0.005, len(df_gold))
    
    symbols_data = {
        'EURUSD': df_eu,
        'GBPUSD': df_gu,
        'USDJPY': df_uj,
        'GOLD': df_gold
    }
    
    # 2. Test Correlation Matrix Calculation
    static_corr, rolling_corr = calculate_portfolio_correlation(symbols_data)
    
    if static_corr is not None:
        logger.info("\nStatic Correlation Matrix:\n" + str(static_corr))
    
    # 3. Test Exposure Risk Evaluation
    proposed_trades = [
        {'symbol': 'EURUSD', 'signal': 1},   # AI wants to Buy EURUSD
        {'symbol': 'GBPUSD', 'signal': 1},   # AI wants to Buy GBPUSD (Should be rejected, too correlated)
        {'symbol': 'USDJPY', 'signal': -1},  # AI wants to Sell USDJPY (Should be rejected, effectively same as Buying EURUSD)
        {'symbol': 'GOLD', 'signal': 1}      # AI wants to Buy Gold (Should be approved, uncorrelated)
    ]
    
    logger.info("-----------------------------------------")
    logger.info("Testing Trade Risk Filter:")
    for t in proposed_trades:
        logger.info(f"Proposed: {t['symbol']} | Signal: {t['signal']}")
        
    approved_trades = evaluate_portfolio_exposure(static_corr, proposed_trades, max_correlation=0.75)
    
    logger.info("-----------------------------------------")
    logger.info("Final Approved Trades:")
    for t in approved_trades:
        logger.info(f"✅ Executing: {t['symbol']} | Signal: {t['signal']}")
        
    logger.info("=========================================")
    logger.info("✅ Phase 14 Portfolio Engine Test Completed")
    logger.info("=========================================")

if __name__ == "__main__":
    import numpy as np
    test_portfolio_engine()
