import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from loguru import logger

def fetch_macro_data(period="2y", interval="1h"):
    """
    ดึงข้อมูล Macroeconomics (ดัชนีดอลลาร์ และ อัตราผลตอบแทนพันธบัตร 10 ปี)
    เพื่อใช้เป็น Alternative Data สำหรับ GOLD, JPY, GBP
    """
    logger.info("Fetching Macro Data (DXY and US10Y Yield)...")
    try:
        # DXY = US Dollar Index, ^TNX = US 10-Year Treasury Yield
        macro = yf.download(["DX-Y.NYB", "^TNX"], period=period, interval=interval, progress=False)
        if isinstance(macro.columns, pd.MultiIndex):
            macro = macro['Close'] # Keep only close prices
        else:
            macro = macro[['DX-Y.NYB', '^TNX']]
            
        macro.columns = ['dxy', 'us10y']
        macro = macro.ffill().fillna(0)
        
        # Calculate Macro Features
        macro['dxy_ret'] = macro['dxy'].pct_change()
        macro['us10y_diff'] = macro['us10y'].diff() # Yield is already in %, use diff
        return macro.reset_index().rename(columns={'index': 'timestamp', 'Date': 'timestamp', 'Datetime': 'timestamp'})
    except Exception as e:
        logger.error(f"Failed to fetch macro data: {e}")
        return pd.DataFrame()

def apply_asymmetric_barrier(df, tp_multiplier=3.0, sl_multiplier=1.0, look_forward=24):
    """
    เทคนิคสำหรับ BTCUSD และ GBPUSD: Asymmetric Triple Barrier (TP ไกลกว่า SL)
    เพื่อทนความผันผวนและกินกำไรคำใหญ่ชดเชยเวลาโดน Whipsaw
    """
    logger.info(f"Applying Asymmetric Barrier (TP={tp_multiplier}x, SL={sl_multiplier}x)")
    targets = []
    
    if 'atr14' not in df.columns:
        df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
    for i in range(len(df)):
        if i + look_forward >= len(df):
            targets.append(np.nan)
            continue
            
        close = df['close'].iloc[i]
        atr = df['atr14'].iloc[i]
        
        tp_price = close + (atr * tp_multiplier)
        sl_price = close - (atr * sl_multiplier)
        
        hit = 0 # 0 = Time out (Noise)
        
        for j in range(1, look_forward + 1):
            future_high = df['high'].iloc[i+j]
            future_low = df['low'].iloc[i+j]
            
            # Hit TP first -> Label 1 (Buy Signal)
            if future_high >= tp_price:
                hit = 1
                break
            # Hit SL first -> Label -1 (Sell / Invalid)
            elif future_low <= sl_price:
                hit = -1
                break
                
        targets.append(hit)
        
    df['target_asym'] = targets
    return df

def build_advanced_features(df, asset_name, macro_df=pd.DataFrame()):
    """ประกอบร่าง Feature ขั้นสูงตามแต่ละสินทรัพย์"""
    df = df.copy()
    
    # 1. Base Features
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    
    # 2. Macro Integration (For GOLD, JPY, GBP)
    if not macro_df.empty and asset_name in ['GOLD', 'USDJPY', 'GBPUSD']:
        # Ensure timestamp matching
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], utc=True)
        df = pd.merge_asof(df.sort_values('timestamp'), macro_df.sort_values('timestamp'), on='timestamp')
        logger.info(f"Integrated Macro Features into {asset_name}")

    # 3. Asset-Specific Upgrades
    if asset_name == 'BTC':
        # Crypto needs to capture violent bursts
        df['vol_burst'] = df['volume'] / (ta.sma(df['volume'], length=20) + 1e-9)
        # Apply Asymmetric Labeling for Crypto (High Reward/Risk ratio)
        df = apply_asymmetric_barrier(df, tp_multiplier=4.0, sl_multiplier=1.5, look_forward=48)
        df['target'] = np.where(df['target_asym'] == 1, 1, 0)
        
    elif asset_name == 'GOLD':
        # Gold needs Real Yield proxy (US10Y) and DXY inverse correlation
        if 'us10y_diff' in df.columns:
            df['gold_macro_score'] = (df['dxy_ret'] * -1) - df['us10y_diff']
        # Apply Asymmetric Labeling (Gold runs hard when it breaks out)
        df = apply_asymmetric_barrier(df, tp_multiplier=3.0, sl_multiplier=1.0, look_forward=24)
        df['target'] = np.where(df['target_asym'] == 1, 1, 0)

    elif asset_name in ['USDJPY', 'GBPUSD']:
        # JPY/GBP driven by Yield Differentials and DXY
        if 'dxy_ret' in df.columns:
            df['currency_strength'] = df['close'].pct_change() - df['dxy_ret']
        # Predict trend continuation
        future_ret = (df['close'].shift(-12) - df['close']) / df['close']
        df['target'] = np.where(future_ret > 0.001, 1, 0)
        
    return df.dropna()

if __name__ == "__main__":
    logger.info("Advanced Alpha Factory Ready.")
