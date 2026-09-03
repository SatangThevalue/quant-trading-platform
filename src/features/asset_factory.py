import pandas as pd
import pandas_ta as ta
import numpy as np

def build_stable_fiat_features(df):
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema_gap'] = (df['ema20'] - df['ema50']) / (df['ema50'] + 1e-9)
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def build_volatile_fiat_features(df):
    df['rsi14'] = ta.rsi(df['close'], length=14)
    bbands = ta.bbands(df['close'], length=20)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        # Approximate width using standard names if possible, else skip
        cols = df.columns.tolist()
        bbu = [c for c in cols if 'BBU_20' in c]
        bbl = [c for c in cols if 'BBL_20' in c]
        bbm = [c for c in cols if 'BBM_20' in c]
        if bbu and bbl and bbm:
            df['bb_width'] = (df[bbu[0]] - df[bbl[0]]) / (df[bbm[0]] + 1e-9)
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['roc10'] = ta.roc(df['close'], length=10)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def build_commodity_features(df):
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    kc = ta.kc(df['high'], df['low'], df['close'], length=20)
    if kc is not None:
        df = pd.concat([df, kc], axis=1)
    df['vol_sma'] = ta.sma(df['volume'], length=20)
    df['vol_burst'] = df['volume'] / (df['vol_sma'] + 1e-9)
    df['mom10'] = ta.mom(df['close'], length=10)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def build_crypto_features(df):
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ret_1'] = df['close'].pct_change()
    df['norm_ret'] = df['ret_1'] / ((df['atr14'] / df['close']) + 1e-9)
    df['rsi21'] = ta.rsi(df['close'], length=21)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def apply_asset_features(df, asset_name):
    if asset_name in ['EURUSD', 'USDCHF']:
        return build_stable_fiat_features(df)
    elif asset_name in ['GBPUSD', 'USDJPY']:
        return build_volatile_fiat_features(df)
    elif asset_name == 'GOLD':
        return build_commodity_features(df)
    elif asset_name == 'BTC':
        return build_crypto_features(df)
    return df.dropna()
