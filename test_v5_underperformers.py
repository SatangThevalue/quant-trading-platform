import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import sys, os
from loguru import logger

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.advanced_alpha_factory import build_advanced_features, fetch_macro_data

symbols = {
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "GOLD": "GC=F",
    "BTC": "BTC-USD"
}

results = []

logger.info("Fetching Macro Data...")
macro_df = fetch_macro_data(period="2y", interval="1h")

logger.info("Training V5 Models for Underperforming Assets...")

for name, ticker in symbols.items():
    try:
        logger.info(f"--- Processing {name} ---")
        df = yf.download(ticker, period="2y", interval="1h", progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
        elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})

        # Build Advanced Features
        df_feat = build_advanced_features(df, name, macro_df)
        
        drop_cols = ["timestamp", "target", "target_asym", "source", "symbol", "date", "datetime"]
        features = [c for c in df_feat.columns if c not in drop_cols]
        X = df_feat[features]
        y = df_feat["target"]

        split_idx = int(len(df_feat) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        # Asymmetric models need more trees to learn rare breakout events
        model = lgb.LGBMClassifier(num_leaves=15, max_depth=4, learning_rate=0.01,
                                   n_estimators=300, class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X_test)[:, 1]
        test_df = df_feat.iloc[split_idx:].copy()
        test_df['signal'] = 0
        
        # High confidence threshold for breakout
        test_df.loc[prob_up > 0.65, 'signal'] = 1
        
        # Calculate returns based on Asymmetric TP/SL assumption or standard if not applied
        if name in ['BTC', 'GOLD']:
            # For Asymmetric models, if target_asym == 1 (Hit TP), we gain TP multiplier * ATR
            # If target_asym == -1 (Hit SL), we lose SL multiplier * ATR
            tp_mult = 4.0 if name == 'BTC' else 3.0
            sl_mult = 1.5 if name == 'BTC' else 1.0
            
            test_df['strat_ret_net'] = 0.0
            for i in range(len(test_df)):
                if test_df['signal'].iloc[i] == 1:
                    outcome = test_df['target_asym'].iloc[i]
                    atr = test_df['atr14'].iloc[i] / test_df['close'].iloc[i] # ATR in %
                    
                    cost = 0.002 if name == 'BTC' else 0.0005
                    if outcome == 1:
                        test_df['strat_ret_net'].iloc[i] = (atr * tp_mult) - cost
                    elif outcome == -1:
                        test_df['strat_ret_net'].iloc[i] = -(atr * sl_mult) - cost
        else:
            # Standard return for GBP/JPY
            future_ret = (test_df['close'].shift(-6) - test_df['close']) / test_df['close']
            test_df['strat_ret'] = test_df['signal'] * future_ret
            cost = 0.0002
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
        
    except Exception as e:
        logger.error(f"Error on {name}: {e}")

df_res = pd.DataFrame(results)
print("\n=== V5 ADVANCED ALPHA PERFORMANCE (Underperformers Rescue) ===")
print(df_res.to_markdown(index=False))

# Save to docs
with open("docs/V5_ADVANCED_ALPHA_REPORT.md", "w") as f:
    f.write("# V5 Advanced Alpha Performance (Rescuing Underperformers)\n\n")
    f.write("Trained with Asymmetric Barrier Labeling and Macro-Economic Data Integration (DXY & US10Y).\n\n")
    f.write(df_res.to_markdown(index=False))
