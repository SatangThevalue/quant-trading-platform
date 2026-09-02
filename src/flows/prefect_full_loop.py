import sys
import os

# HARD DISABLE PREFECT DAEMON FOR LOCAL VPS HARDWARE
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_LOCAL_STORAGE_PATH"] = os.path.join(os.getcwd(), ".prefect")

import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from loguru import logger
import mlflow
from prefect import task, flow
from prefect.artifacts import create_markdown_artifact
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.concurrency.sync import rate_limit

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.feature_pipeline import build_features
from export_production_v2_intraday import build_intraday_features
from src.models.export_onnx import export_model_to_onnx

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

TIMEFRAMES = {
    "1d": {"period": "5y", "features": "v1"},
    "4h": {"period": "730d", "features": "v2", "yf_proxy": "1h", "resample": "4h"},
    "1h": {"period": "730d", "features": "v2"},
    "30m": {"period": "60d", "features": "v2"},
    "15m": {"period": "60d", "features": "v2"}
}

@task(retries=3, retry_delay_seconds=10, name="Fetch Market Data")
def fetch_market_data(ticker: str, tf_config: dict, tf_name: str):
    """Fetch data with auto-retry in case Yahoo Finance API limits/fails."""
    fetch_tf = tf_config.get("yf_proxy", tf_name)
    df = yf.download(ticker, period=tf_config["period"], interval=fetch_tf, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} at {fetch_tf}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
    elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
    
    if tf_config.get("resample"):
        df = df.set_index('timestamp')
        df = df.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        df = df.reset_index()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns: df[col] = 0
    return df

@task(name="Build Quant Features")
def generate_features(df: pd.DataFrame, tf_config: dict):
    """Generates features depending on the timeframe (V1 for D1, V2 for Intraday)"""
    if tf_config["features"] == "v1":
        return build_features(df), 0.55, 0.45, 5
    else:
        return build_intraday_features(df), 0.60, 0.40, 12

@task(name="Train & Export ONNX")
def train_and_export(df_feat: pd.DataFrame, asset_name: str, tf_name: str, tf_config: dict, buy_thresh: float, sell_thresh: float, target_shift: int):
    """Trains LightGBM, evaluates Out-Of-Sample, logs to MLflow, and exports ONNX."""
    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[features]
    y = df_feat["target"]
    
    split_idx = int(len(df_feat) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    if len(X_train) < 50:
        return None
        
    mlflow.set_experiment(f"PREFECT_{asset_name}")
    with mlflow.start_run(run_name=f"Model_{tf_name}"):
        if tf_config["features"] == "v1":
            best_params = {'num_leaves': 31, 'max_depth': 4, 'learning_rate': 0.03, 'verbose': -1}
        else:
            best_params = {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1}
            
        model = lgb.LGBMClassifier(**best_params, n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Export
        version_name = f"{tf_config['features']}_{tf_name}"
        export_model_to_onnx(model, features, symbol=asset_name, version=version_name)
        
        # Backtest
        prob_up = model.predict_proba(X_test)[:, 1]
        test_df = df_feat.iloc[split_idx:].copy()
        test_df['signal'] = 0
        test_df.loc[prob_up >= buy_thresh, 'signal'] = 1
        test_df.loc[prob_up <= sell_thresh, 'signal'] = -1
        
        test_df['future_return'] = (test_df['close'].shift(-target_shift) - test_df['close']) / test_df['close']
        test_df['strategy_return'] = test_df['signal'] * test_df['future_return']
        
        base_cost = 0.00015
        if asset_name == 'BTC': base_cost = 0.002
        elif asset_name == 'GOLD': base_cost = 0.0005
        elif asset_name == 'USDJPY': base_cost = 0.0002
        if tf_name in ['15m', '30m']: base_cost *= 1.5
        
        trade_occurred = test_df['signal'] != 0
        test_df['strategy_return_net'] = 0.0
        test_df.loc[trade_occurred, 'strategy_return_net'] = test_df['strategy_return'] - base_cost
        
        active_trades = test_df[test_df['signal'] != 0].copy()
        if len(active_trades) > 0:
            winning_trades = len(active_trades[active_trades['strategy_return_net'] > 0])
            win_rate = winning_trades / len(active_trades)
            pf = active_trades.loc[active_trades['strategy_return_net'] > 0, 'strategy_return_net'].sum() / abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum()) if abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum()) > 0 else float('inf')
            
            ann_factor = 252 if tf_name == '1d' else 6*252 if tf_name == '4h' else 24*252 if tf_name == '1h' else 48*252 if tf_name == '30m' else 96*252
            returns = active_trades['strategy_return_net']
            sharpe = np.sqrt(ann_factor) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
            
            test_df['strategy_equity_net'] = test_df['strategy_return_net'].cumsum()
            cum_net = np.exp(test_df['strategy_equity_net'])
            max_dd = ((cum_net - cum_net.cummax()) / cum_net.cummax()).min()
            net_ret = cum_net.iloc[-1] - 1 if len(cum_net) > 0 else 0
        else:
            win_rate, pf, sharpe, max_dd, net_ret = 0, 0, 0, 0, 0
            
        mlflow.log_metrics({
            "trades": len(active_trades),
            "win_rate": win_rate,
            "profit_factor": pf,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd
        })
        
        return {
            "Asset": asset_name, "TF": tf_name, "Trades": len(active_trades),
            "WinRate": f"{win_rate*100:.1f}%", "ProfitFactor": f"{pf:.2f}",
            "Sharpe": f"{sharpe:.2f}", "MaxDD": f"{max_dd*100:.1f}%", "NetReturn": f"{net_ret*100:.1f}%"
        }

@flow(name="Asset Timeframe Subflow")
def process_single_asset_tf(asset_name, ticker, tf_name, tf_config):
    """Subflow to handle a single Asset and Timeframe combination"""
    try:
        df = fetch_market_data(ticker, tf_config, tf_name)
        df_feat, buy_thresh, sell_thresh, target_shift = generate_features(df, tf_config)
        result = train_and_export(df_feat, asset_name, tf_name, tf_config, buy_thresh, sell_thresh, target_shift)
        return result
    except Exception as e:
        logger.error(f"Failed {asset_name} [{tf_name}]: {e}")
        return None

@flow(name="Master Quant Platform Full Loop", task_runner=ThreadPoolTaskRunner(max_workers=1))
def prefect_full_loop_orchestrator():
    """Main Orchestrator Flow: Loops through all assets and timeframes sequentially to prevent SQLite locks on VPS."""
    all_results = []
    
    for asset_name, ticker in ASSETS.items():
        for tf_name, tf_config in TIMEFRAMES.items():
            # Rate limit to prevent overloading local SQLite or hitting Yahoo Finance API limits too fast
            rate_limit("yahoo_finance_api", occupy=1)
            
            res = process_single_asset_tf(asset_name, ticker, tf_name, tf_config)
            if res:
                all_results.append(res)
                
    if all_results:
        df_res = pd.DataFrame(all_results)
        markdown_report = f"# Prefect Automated Quant Report\n\n{df_res.to_markdown(index=False)}"
        
        # Save locally
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "PREFECT_FULL_LOOP_REPORT.md")
        with open(report_path, "w") as f:
            f.write(markdown_report)
            
        # Create Prefect UI Artifact
        try:
            create_markdown_artifact(
                key="quant-portfolio-report",
                markdown=markdown_report,
                description="Out-of-Sample Performance across multiple Assets and Timeframes"
            )
        except Exception as e:
            logger.warning("Could not create Prefect artifact (might need Prefect Server connected).")
            
        logger.info("✅ Master Flow Complete. Report generated.")
    return all_results

if __name__ == "__main__":
    # Disable local ephemeral sqlite API issue for script execution
    os.environ["PREFECT_API_URL"] = ""
    prefect_full_loop_orchestrator()
