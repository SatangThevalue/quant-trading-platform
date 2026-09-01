import numpy as np
import pandas as pd
from loguru import logger

def add_triple_barrier(df, upper_multiplier=2.0, lower_multiplier=2.0, max_holding_period=10):
    """
    Phase 3: Triple Barrier Labeling
    Labels 1 if Upper Barrier (Take Profit) hit first.
    Labels -1 if Lower Barrier (Stop Loss) hit first.
    Labels 0 if Time Limit (Vertical Barrier) hit first.
    """
    logger.info("Applying Triple Barrier Labeling...")
    
    # Needs ATR for dynamic volatility barriers
    if 'atr14' not in df.columns:
        logger.error("ATR14 column missing. Cannot calculate dynamic barriers.")
        return df
        
    targets = []
    
    for i in range(len(df)):
        # If we can't look forward max_holding_period bars, skip
        if i + max_holding_period >= len(df):
            targets.append(np.nan)
            continue
            
        current_close = df['close'].iloc[i]
        current_atr = df['atr14'].iloc[i]
        
        # Dynamic Barriers
        upper_barrier = current_close + (current_atr * upper_multiplier)
        lower_barrier = current_close - (current_atr * lower_multiplier)
        
        hit_target = 0 # Default vertical barrier (time out)
        
        # Look forward window
        for j in range(1, max_holding_period + 1):
            future_high = df['high'].iloc[i+j]
            future_low = df['low'].iloc[i+j]
            
            # Did it hit Take Profit first?
            if future_high >= upper_barrier:
                hit_target = 1
                break
                
            # Did it hit Stop Loss first?
            if future_low <= lower_barrier:
                hit_target = -1
                break
                
        targets.append(hit_target)
        
    df['target_triple_barrier'] = targets
    return df
