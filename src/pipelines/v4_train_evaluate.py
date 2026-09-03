import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import sys, os
warnings.filterwarnings('ignore')

def build_v4_features(df, asset_name):
    import pandas_ta as ta
    df = df.copy()
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)

    if asset_name == 'GOLD':
        # Proxies for Long Memory & Volatility
        df['log_ret_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_ret_5'] = np.log(df['close'] / df['close'].shift(5))
        df['log_ret_20'] = np.log(df['close'] / df['close'].shift(20))
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        if kc is not None: 
            df = pd.concat([df, kc], axis=1)
    elif asset_name == 'BTC':
        # Crypto: Volatility Normalization
        df['vol_norm'] = df['close'] / (df['atr14'] + 1e-9)
        df['mom10'] = ta.mom(df['close'], length=10)
    else:
        # Fiat: Standard Mean Reversion/Trend
        df['rsi14'] = ta.rsi(df['close'], length=14)
        df['ema_gap'] = (df['ema20'] - df['ema50']) / (df['ema50'] + 1e-9)

    df = df.dropna()
    
    # V4 Strict Target: Price must move > Noise Buffer
    buffer = 0.001 if asset_name in ['BTC', 'GOLD'] else 0.0005
    # Predicting 6 bars out (e.g., 6 hours)
    df['future_ret'] = (df['close'].shift(-6) - df['close']) / df['close']
    df['target'] = np.where(df['future_ret'] > buffer, 1, 0)
    return df.dropna()

symbols = {"EURUSD": "EURUSD=X", "GOLD": "GC=F", "BTC": "BTC-USD"}
results = []

print("Training V4 Advanced Models (Asset-Specific Features & Constraints)...")

for name, ticker in symbols.items():
    try:
        df = yf.download(ticker, period="2y", interval="1h", progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
        elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})

        df_feat = build_v4_features(df, name)
        
        drop_cols = ["timestamp", "target", "future_ret", "symbol", "date", "datetime"]
        features = [c for c in df_feat.columns if c not in drop_cols]
        X = df_feat[features]
        y = df_feat["target"]

        split_idx = int(len(df_feat) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        # V4 Constraints: Shallow trees to prevent noise memorization in 1H
        model = lgb.LGBMClassifier(num_leaves=7, max_depth=3, learning_rate=0.01,
                                   n_estimators=150, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X_test)[:, 1]
        test_df = df_feat.iloc[split_idx:].copy()
        test_df['signal'] = 0
        
        # Asymmetric threshold based on asset
        buy_t = 0.60
        sell_t = 0.40
        test_df.loc[prob_up > buy_t, 'signal'] = 1
        test_df.loc[prob_up < sell_t, 'signal'] = -1

        test_df['strat_ret'] = test_df['signal'] * test_df['future_ret']
        
        # Spread/Cost penalty
        cost = 0.0002 if name == 'EURUSD' else 0.0005 if name == 'GOLD' else 0.002
        test_df['strat_ret_net'] = np.where(test_df['signal'] != 0, test_df['strat_ret'] - cost, 0)

        active = test_df[test_df['signal'] != 0]
        if len(active) > 0:
            win_rate = len(active[active['strat_ret_net'] > 0]) / len(active)
            gross_prof = active.loc[active['strat_ret_net'] > 0, 'strat_ret_net'].sum()
            gross_loss = abs(active.loc[active['strat_ret_net'] <= 0, 'strat_ret_net'].sum())
            pf = gross_prof / gross_loss if gross_loss > 0 else float('inf')
            sharpe = np.sqrt(24*252) * (active['strat_ret_net'].mean() / active['strat_ret_net'].std()) if active['strat_ret_net'].std() > 0 else 0
            
            test_df['strategy_equity_net'] = test_df['strat_ret_net'].cumsum()
            cum_net = np.exp(test_df['strategy_equity_net'])
            max_dd = ((cum_net - cum_net.cummax()) / cum_net.cummax()).min()
        else:
            win_rate, pf, sharpe, max_dd = 0, 0, 0, 0

        results.append({
            "Asset": name, 
            "Trades": len(active), 
            "WinRate": f"{win_rate*100:.1f}%",
            "ProfitFactor": f"{pf:.2f}", 
            "Sharpe": f"{sharpe:.2f}", 
            "MaxDD": f"{max_dd*100:.1f}%"
        })
        
        # Mock Export to ONNX directly using existing tools if needed
        # We will skip raw physical export here to save VPS memory and focus on performance report.
        
    except Exception as e:
        print(f"Error on {name}: {e}")

df_res = pd.DataFrame(results)
print("\n=== V4 ADVANCED PIPELINE PERFORMANCE ===")
print(df_res.to_markdown(index=False))

# Save to docs
os.makedirs("docs", exist_ok=True)
with open("docs/V4_ADVANCED_REPORT.md", "w") as f:
    f.write("# V4 Advanced Pipeline Performance (1H)\n\n")
    f.write("Trained with Asset-Specific Features & Bounded Hyperparameters (max_depth=3).\n\n")
    f.write(df_res.to_markdown(index=False))