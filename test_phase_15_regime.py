import sys
import os
import pandas as pd
from loguru import logger
from src.features.regime_detection import train_regime_model, apply_regime

def test_regimes():
    logger.info("=========================================")
    logger.info("TESTING PHASE 15: REGIME DETECTION")
    logger.info("=========================================")
    
    gold_path = os.path.join("data", "gold", "EURUSD_features.parquet")
    if not os.path.exists(gold_path):
        logger.error("Gold data not found. Please run data collection and features first.")
        return
        
    df = pd.read_parquet(gold_path)
    logger.info(f"Loaded {len(df)} rows for regime testing.")
    
    # Train and Apply
    df_regime = apply_regime(df)
    
    if 'market_regime' in df_regime.columns:
        counts = df_regime['market_regime'].value_counts()
        logger.info("✅ Regime Detection Passed! Distribution:")
        for regime, count in counts.items():
            logger.info(f"  Regime {regime}: {count} rows")
    else:
        logger.error("❌ Failed to create market_regime column.")

if __name__ == "__main__":
    test_regimes()
