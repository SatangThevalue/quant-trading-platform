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
from src.features.feature_pipeline import build_features
from export_production_v2_intraday import build_intraday_features
from src.models.export_onnx import export_model_to_onnx

mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Configuration
ASSETS = {
    "EURUSD": "EURUSD=X",
    "USDCHF": "CHF=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "GOLD": "GC=F",
    "BTC": "BTC-USD"
}

TIMEFRAMES = {
    "1d": {"period": "5y", "features": "v1"},
    "4h": {"period": "730d", "features": "v2", "yf_proxy": "1h", "resample": "4h"}, # yfinance doesn't natively support 4h well
    "1h": {"period": "730d", "features": "v2"},
    "30m": {"period": "60d", "features": "v2"},
    "15m": {"period": "60d", "features": "v2"}
}

results = []

def run_full_pipeline():
    logger.info("=========================================")
    logger.info("🚀 STARTING FULL LOOP QUANT PIPELINE")
    logger.info("=========================================")
    
    for asset_name, ticker in ASSETS.items():
        for tf_name, tf_config in TIMEFRAMES.items():
            try:
                logger.info(f"--- Processing {asset_name} [{tf_name}] ---")
                
                # 1. Fetch Data
                fetch_tf = tf_config.get("yf_proxy", tf_name)
                df = yf.download(ticker, period=tf_config["period"], interval=fetch_tf, progress=False)
                if df.empty: 
                    logger.warning(f"No data for {asset_name} at {fetch_tf}")
                    continue
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
                elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
                
                # Resample 4H if needed
                if tf_config.get("resample"):
                    df = df.set_index('timestamp')
                    df = df.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                    df = df.reset_index()
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in df.columns: df[col] = 0
                
                # 2. Build Features
                if tf_config["features"] == "v1":
                    df_feat = build_features(df)
                    buy_thresh, sell_thresh = 0.55, 0.45
                    target_shift = 5 # 5 bars
                else:
                    df_feat = build_intraday_features(df)
                    buy_thresh, sell_thresh = 0.60, 0.40
                    target_shift = 12 # 12 bars
                    
                drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
                features = [c for c in df_feat.columns if c not in drop_cols]
                X = df_feat[features]
                y = df_feat["target"]
                
                # 3. Train/Test Split (80/20)
                split_idx = int(len(df_feat) * 0.8)
                X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
                X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
                
                if len(X_train) < 50:
                    logger.warning(f"Not enough training data for {asset_name} {tf_name}")
                    continue
                
                # 4. Train Model
                mlflow.set_experiment(f"FULL_LOOP_{asset_name}")
                with mlflow.start_run(run_name=f"Model_{tf_name}"):
                    if tf_config["features"] == "v1":
                        best_params = {'num_leaves': 31, 'max_depth': 4, 'learning_rate': 0.03, 'verbose': -1}
                    else:
                        best_params = {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1}
                        
                    model = lgb.LGBMClassifier(**best_params, n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # 5. Export ONNX
                    version_name = f"{tf_config['features']}_{tf_name}"
                    export_model_to_onnx(model, features, symbol=asset_name, version=version_name)
                    
                    # 6. Backtest (Out of sample)
                    prob_up = model.predict_proba(X_test)[:, 1]
                    test_df = df_feat.iloc[split_idx:].copy()
                    test_df['signal'] = 0
                    test_df.loc[prob_up >= buy_thresh, 'signal'] = 1
                    test_df.loc[prob_up <= sell_thresh, 'signal'] = -1
                    
                    # Calculate returns
                    test_df['future_return'] = (test_df['close'].shift(-target_shift) - test_df['close']) / test_df['close']
                    test_df['strategy_return'] = test_df['signal'] * test_df['future_return']
                    
                    # Adjust spread cost based on asset and timeframe
                    base_cost = 0.00015
                    if asset_name == 'BTC': base_cost = 0.002
                    elif asset_name == 'GOLD': base_cost = 0.0005
                    elif asset_name == 'USDJPY': base_cost = 0.0002
                    
                    # Scalping timeframes hit spread harder
                    if tf_name in ['15m', '30m']: base_cost *= 1.5
                    
                    trade_occurred = test_df['signal'] != 0
                    test_df['strategy_return_net'] = 0.0
                    test_df.loc[trade_occurred, 'strategy_return_net'] = test_df['strategy_return'] - base_cost
                    
                    active_trades = test_df[test_df['signal'] != 0].copy()
                    
                    if len(active_trades) > 0:
                        winning_trades = len(active_trades[active_trades['strategy_return_net'] > 0])
                        win_rate = winning_trades / len(active_trades)
                        
                        gross_profit = active_trades.loc[active_trades['strategy_return_net'] > 0, 'strategy_return_net'].sum()
                        gross_loss = abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum())
                        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                        
                        # Annualization factor
                        if tf_name == '1d': ann_factor = 252
                        elif tf_name == '4h': ann_factor = 6 * 252
                        elif tf_name == '1h': ann_factor = 24 * 252
                        elif tf_name == '30m': ann_factor = 48 * 252
                        else: ann_factor = 96 * 252
                        
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
                        "TF": tf_name,
                        "Trades": len(active_trades),
                        "WinRate": f"{win_rate*100:.1f}%",
                        "ProfitFactor": f"{pf:.2f}",
                        "Sharpe": f"{sharpe:.2f}",
                        "MaxDD": f"{max_dd*100:.1f}%",
                        "NetReturn": f"{net_ret*100:.1f}%"
                    })
                    
                    mlflow.log_metrics({
                        "trades": len(active_trades),
                        "win_rate": win_rate,
                        "profit_factor": pf,
                        "sharpe_ratio": sharpe,
                        "max_drawdown": max_dd
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {asset_name} {tf_name}: {e}")

    df_res = pd.DataFrame(results)
    
    # Save markdown report
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "FULL_LOOP_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Full Loop Quantitative Model Performance Report\n\n")
        f.write("Generated across all assets and timeframes (1D, 4H, 1H, 30M, 15M).\n\n")
        f.write("## Performance Summary\n")
        f.write(df_res.to_markdown(index=False))
        
    print("\n=== FULL LOOP PORTFOLIO REPORT ===")
    print(df_res.to_markdown(index=False))
    logger.info("✅ Full Loop Pipeline Complete. Report saved to docs/FULL_LOOP_REPORT.md")

if __name__ == "__main__":
    run_full_pipeline()
