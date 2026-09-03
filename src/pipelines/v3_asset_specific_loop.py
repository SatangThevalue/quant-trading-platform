import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import sys, os
from loguru import logger
import mlflow

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.asset_factory import apply_asset_features
from src.models.meta_labeling import MetaLabelingPipeline

# MLflow Config
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{mlruns_path}")

ASSETS = {
    "EURUSD": "EURUSD=X",
    "USDCHF": "CHF=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "GOLD": "GC=F",
    "BTC": "BTC-USD"
}

results = []

def run_v3_pipeline():
    logger.info("=========================================")
    logger.info("🚀 STARTING V3 ASSET-SPECIFIC PIPELINE")
    logger.info("=========================================")
    
    for asset_name, ticker in ASSETS.items():
        try:
            logger.info(f"--- Processing {asset_name} (V3 Architecture) ---")
            
            # Fetch 15m data for Meta-Labeling challenge
            df = yf.download(ticker, period="60d", interval="15m", progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
            elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns: df[col] = 0
                
            # 1. Asset Specific Features
            df_feat = apply_asset_features(df, asset_name)
            
            # Target Label (Future 12 bars)
            df_feat["future_return"] = (df_feat["close"].shift(-12) - df_feat["close"]) / df_feat["close"]
            noise_buffer = 0.0005 if asset_name not in ['BTC', 'GOLD'] else 0.002
            df_feat["target"] = np.where(df_feat["future_return"] > noise_buffer, 1, 0)
            df_feat = df_feat.dropna()
            
            drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "future_return", "source", "symbol", "date", "datetime", "created_at"]
            features = [c for c in df_feat.columns if c not in drop_cols]
            X = df_feat[features]
            y = df_feat["target"]
            
            if len(X) < 100: continue
            
            split_idx = int(len(df_feat) * 0.8)
            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
            
            # Asset Specific Base Params
            if asset_name in ['BTC', 'GOLD']:
                base_params = {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.01, 'verbose': -1, 'random_state': 42}
            else:
                base_params = {'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.03, 'verbose': -1, 'random_state': 42}
                
            meta_params = {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.01, 'verbose': -1, 'random_state': 42}
            
            mlflow.set_experiment(f"V3_{asset_name}")
            with mlflow.start_run(run_name="Meta_Labeling_15m"):
                pipeline = MetaLabelingPipeline(base_params, meta_params)
                
                base_prob, meta_prob = pipeline.fit_predict(X_train, y_train, X_test, df_feat["future_return"].iloc[split_idx:], 0.00015)
                
                test_df = df_feat.iloc[split_idx:].copy()
                test_df['signal'] = 0
                
                # Base Signal
                test_df.loc[base_prob >= 0.55, 'signal'] = 1
                test_df.loc[base_prob <= 0.45, 'signal'] = -1
                
                base_trades = len(test_df[test_df['signal'] != 0])
                
                # Meta Filter (Only trade if meta-model prob > 0.5)
                test_df.loc[meta_prob < 0.50, 'signal'] = 0
                filtered_trades = len(test_df[test_df['signal'] != 0])
                
                test_df['strategy_return'] = test_df['signal'] * test_df['future_return']
                
                cost = 0.00015 if asset_name not in ['BTC', 'GOLD'] else 0.002
                trade_occurred = test_df['signal'] != 0
                test_df['strategy_return_net'] = 0.0
                test_df.loc[trade_occurred, 'strategy_return_net'] = test_df['strategy_return'] - cost
                
                active_trades = test_df[test_df['signal'] != 0].copy()
                
                if len(active_trades) > 0:
                    winning_trades = len(active_trades[active_trades['strategy_return_net'] > 0])
                    win_rate = winning_trades / len(active_trades)
                    
                    gross_profit = active_trades.loc[active_trades['strategy_return_net'] > 0, 'strategy_return_net'].sum()
                    gross_loss = abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum())
                    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    
                    ann_factor = 96 * 252 # 15m
                    returns = active_trades['strategy_return_net']
                    sharpe = np.sqrt(ann_factor) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
                    
                    test_df['strategy_equity_net'] = test_df['strategy_return_net'].cumsum()
                    cum_net = np.exp(test_df['strategy_equity_net'])
                    max_dd = ((cum_net - cum_net.cummax()) / cum_net.cummax()).min()
                    net_ret = cum_net.iloc[-1] - 1 if len(cum_net) > 0 else 0
                else:
                    win_rate, pf, sharpe, max_dd, net_ret = 0, 0, 0, 0, 0
                    
                results.append({
                    "Asset": asset_name,
                    "BaseTrades": base_trades,
                    "FilteredTrades": filtered_trades,
                    "WinRate": f"{win_rate*100:.1f}%",
                    "ProfitFactor": f"{pf:.2f}",
                    "Sharpe": f"{sharpe:.2f}",
                    "MaxDD": f"{max_dd*100:.1f}%",
                    "NetReturn": f"{net_ret*100:.1f}%"
                })
                
                mlflow.log_metrics({
                    "base_trades": base_trades,
                    "filtered_trades": filtered_trades,
                    "win_rate": win_rate,
                    "profit_factor": pf,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd
                })
                
        except Exception as e:
            logger.error(f"Error processing {asset_name}: {e}")

    df_res = pd.DataFrame(results)
    
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "V3_PERFORMANCE_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# V3 Asset-Specific & Meta-Labeling Performance Report\n\n")
        f.write("Generated using Factory Pattern and Two-Stage LightGBM Models on 15m timeframes.\n\n")
        f.write(df_res.to_markdown(index=False))
        
    print("\n=== V3 ASSET-SPECIFIC PORTFOLIO REPORT ===")
    print(df_res.to_markdown(index=False))
    logger.info("✅ V3 Pipeline Complete. Report saved to docs/V3_PERFORMANCE_REPORT.md")

if __name__ == "__main__":
    run_v3_pipeline()
