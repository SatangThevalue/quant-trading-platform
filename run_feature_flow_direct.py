import sys
import os
import pandas as pd

# Directly call the inner functions to bypass Prefect daemon issue for local testing
from src.flows.feature_engineering_flow import load_raw_data_from_db, apply_feature_engineering, save_features_to_store

if __name__ == "__main__":
    print("Running Feature Engineering directly (bypassing Prefect Daemon)...")
    symbols = ["EURUSD", "GBPUSD", "JPY", "AUDUSD"]
    
    for sym in symbols:
        # 1. Load Data
        print(f"Loading {sym}...")
        # Since these are Prefect tasks, we call .fn() to execute the underlying python function
        raw_df = load_raw_data_from_db.fn(symbol=sym, timeframe="1d")
        
        # 2. Extract Features
        print(f"Applying features to {sym}...")
        feature_df = apply_feature_engineering.fn(raw_df)
        
        # 3. Save to Store
        print(f"Saving {sym}...")
        save_features_to_store.fn(feature_df, sym)
        
    print("Done!")
