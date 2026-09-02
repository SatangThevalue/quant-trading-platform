import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import sys, os
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.feature_pipeline import build_features

symbols = {
    "EURUSD": "EURUSD=X",
    "USDCHF": "CHF=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "GOLD": "GC=F",
    "BTC": "BTC-USD"
}

results = []

for name, ticker in symbols.items():
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
        elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0
                
        df_feat = build_features(df)
        
        drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
        features = [c for c in df_feat.columns if c not in drop_cols]
        X = df_feat[features]
        y = df_feat["target"]
        
        split_idx = int(len(df_feat) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
        best_params = {'num_leaves': 31, 'max_depth': 4, 'learning_rate': 0.03, 'verbose': -1}
        model = lgb.LGBMClassifier(**best_params, n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        prob_up = model.predict_proba(X_test)[:, 1]
        test_df = df_feat.iloc[split_idx:].copy()
        test_df['signal'] = 0
        test_df.loc[prob_up > 0.55, 'signal'] = 1
        test_df.loc[prob_up < 0.45, 'signal'] = -1
        
        test_df['future_return'] = np.log(test_df['close'].shift(-5) / test_df['close'])
        test_df['strategy_return'] = test_df['signal'].shift(1) * test_df['future_return']
        
        cost = 0.00015 if name != 'BTC' else 0.002 
        trade_occurred = test_df['signal'].shift(1) != 0
        test_df.loc[trade_occurred, 'strategy_return_net'] = test_df['strategy_return'] - cost
        test_df.loc[~trade_occurred, 'strategy_return_net'] = test_df['strategy_return']
        
        active_trades = test_df.dropna(subset=['strategy_return_net'])[test_df.dropna(subset=['strategy_return_net'])['signal'].shift(1) != 0]
        
        if len(active_trades) == 0:
            continue
            
        winning_trades = len(active_trades[active_trades['strategy_return_net'] > 0])
        win_rate = winning_trades / len(active_trades)
        
        gross_profit = active_trades.loc[active_trades['strategy_return_net'] > 0, 'strategy_return_net'].sum()
        gross_loss = abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        daily_returns = active_trades['strategy_return_net'] / 5
        sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0
        
        test_df['strategy_equity_net'] = test_df['strategy_return_net'].cumsum()
        cum_net = np.exp(test_df['strategy_equity_net'])
        max_dd = ((cum_net - cum_net.cummax()) / cum_net.cummax()).min()
        
        net_ret = cum_net.iloc[-6] - 1
        
        results.append({
            "Asset": name,
            "Trades": len(active_trades),
            "WinRate": f"{win_rate*100:.1f}%",
            "ProfitFactor": f"{pf:.2f}",
            "Sharpe": f"{sharpe:.2f}",
            "MaxDD": f"{max_dd*100:.1f}%",
            "NetReturn": f"{net_ret*100:.1f}%"
        })
    except Exception as e:
        pass

df_res = pd.DataFrame(results)
print("\n=== PORTFOLIO PERFORMANCE REPORT (Out-of-Sample) ===")
print(df_res.to_markdown(index=False))
