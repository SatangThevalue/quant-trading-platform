import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from loguru import logger
import mlflow
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from export_production_v2_intraday import build_intraday_features

mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{mlruns_path}")

def run_intraday_backtest(symbol="EURUSD", ticker="EURUSD=X", timeframe="1h"):
    logger.info(f"--- Running V2 Intraday Backtest for {symbol} ({timeframe}) ---")
    
    # 1. Fetch Data
    period = "730d" if timeframe == "1h" else "60d"
    df = yf.download(ticker, period=period, interval=timeframe, progress=False)
    if df.empty: return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
    elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns: df[col] = 0
            
    # 2. Features
    df_feat = build_intraday_features(df)
    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[features]
    y = df_feat["target"]
    
    split_idx = int(len(df_feat) * 0.8) # 80/20 split
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    best_params = {
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    mlflow.set_experiment(f"{symbol}_V2_INTRADAY")
    with mlflow.start_run(run_name=f"Backtest_V2_{timeframe}"):
        model = lgb.LGBMClassifier(**best_params, n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        prob_up = model.predict_proba(X_test)[:, 1]
        test_df = df_feat.iloc[split_idx:].copy()
        test_df['signal'] = 0
        
        # Intraday Thresholds (Stricter to beat the spread)
        buy_threshold = 0.65
        sell_threshold = 0.35
        test_df.loc[prob_up >= buy_threshold, 'signal'] = 1
        test_df.loc[prob_up <= sell_threshold, 'signal'] = -1
        
        # Future return over next 12 bars
        test_df['future_return'] = (test_df['close'].shift(-12) - test_df['close']) / test_df['close']
        test_df['strategy_return'] = test_df['signal'] * test_df['future_return']
        
        # Spread cost approx 1.5 pips
        cost = 0.00015
        trade_occurred = test_df['signal'] != 0
        test_df['strategy_return_net'] = 0.0
        test_df.loc[trade_occurred, 'strategy_return_net'] = test_df['strategy_return'] - cost
        
        active_trades = test_df[test_df['signal'] != 0].copy()
        if len(active_trades) == 0:
            logger.warning("No trades executed. Thresholds might be too strict.")
            return
            
        winning_trades = len(active_trades[active_trades['strategy_return_net'] > 0])
        win_rate = winning_trades / len(active_trades)
        
        gross_profit = active_trades.loc[active_trades['strategy_return_net'] > 0, 'strategy_return_net'].sum()
        gross_loss = abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Ann Sharpe (Assuming 24 * 252 hours per year)
        hourly_returns = test_df['strategy_return_net']
        sharpe = np.sqrt(24 * 252) * (hourly_returns.mean() / hourly_returns.std()) if hourly_returns.std() > 0 else 0
        
        test_df['strategy_equity_net'] = test_df['strategy_return_net'].cumsum()
        cum_net = np.exp(test_df['strategy_equity_net'])
        max_dd = ((cum_net - cum_net.cummax()) / cum_net.cummax()).min()
        
        net_ret = cum_net.iloc[-13] - 1 if len(cum_net) > 13 else 0
        
        logger.info("--------------------------------------------------")
        logger.info(f"Results for {symbol} ({timeframe}):")
        logger.info(f"Trades: {len(active_trades)}")
        logger.info(f"Win Rate: {win_rate*100:.2f}%")
        logger.info(f"Profit Factor: {pf:.2f}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_dd*100:.2f}%")
        logger.info(f"Net Return: {net_ret*100:.2f}%")
        logger.info("--------------------------------------------------")
        
        mlflow.log_params(best_params)
        mlflow.log_param("timeframe", timeframe)
        mlflow.log_param("buy_threshold", buy_threshold)
        mlflow.log_param("sell_threshold", sell_threshold)
        mlflow.log_param("cost_spread", cost)
        
        mlflow.log_metrics({
            "trades": len(active_trades),
            "win_rate": win_rate,
            "profit_factor": pf,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "net_return": net_ret
        })
        
        logger.info(f"✅ Logged to MLflow Experiment: {symbol}_V2_INTRADAY")

if __name__ == "__main__":
    run_intraday_backtest("EURUSD", "EURUSD=X", "1h")
    run_intraday_backtest("USDCHF", "CHF=X", "1h")
