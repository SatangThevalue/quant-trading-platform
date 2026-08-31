import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from loguru import logger
import mlflow

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure MLflow Local Storage
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{mlruns_path}")

def load_data(symbol="EURUSD"):
    gold_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gold")
    file_path = os.path.join(gold_path, f"{symbol}_features.parquet")
    if not os.path.exists(file_path):
        logger.error(f"Feature file not found: {file_path}")
        return None
    df = pd.read_parquet(file_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def run_simple_backtest(df, symbol="EURUSD"):
    logger.info(f"Starting Vectorised Backtest for {symbol}...")
    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df["target"]
    split_idx = int(len(df) * 0.8)
    
    test_df = df.iloc[split_idx:].copy()
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    logger.info(f"Backtesting on {len(test_df)} out-of-sample days (approx {len(test_df)/252:.1f} years)")
    
    best_params = {'num_leaves': 37, 'max_depth': 3, 'learning_rate': 0.01348, 'feature_fraction': 0.732, 'bagging_fraction': 0.961}
    model = lgb.LGBMClassifier(**best_params, n_estimators=100, random_state=42, verbose=-1)
    model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    
    test_df['prob_up'] = model.predict_proba(X_test)[:, 1]
    buy_threshold = 0.55
    sell_threshold = 0.45
    
    test_df['signal'] = 0
    test_df.loc[test_df['prob_up'] > buy_threshold, 'signal'] = 1
    test_df.loc[test_df['prob_up'] < sell_threshold, 'signal'] = -1
    
    test_df['future_return'] = np.log(test_df['close'].shift(-5) / test_df['close'])
    test_df['strategy_return'] = test_df['signal'].shift(1) * test_df['future_return']
    
    transaction_cost = 0.00015 
    trade_occurred = test_df['signal'].shift(1) != 0
    test_df.loc[trade_occurred, 'strategy_return_net'] = test_df['strategy_return'] - transaction_cost
    test_df.loc[~trade_occurred, 'strategy_return_net'] = test_df['strategy_return']
    
    test_df['buy_hold_equity'] = test_df['future_return'].cumsum()
    test_df['strategy_equity_net'] = test_df['strategy_return_net'].cumsum()
    
    active_trades = test_df.dropna(subset=['strategy_return_net'])[test_df.dropna(subset=['strategy_return_net'])['signal'].shift(1) != 0]
    total_trades = len(active_trades)
    winning_trades = len(active_trades[active_trades['strategy_return_net'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    gross_profit = active_trades.loc[active_trades['strategy_return_net'] > 0, 'strategy_return_net'].sum()
    gross_loss = abs(active_trades.loc[active_trades['strategy_return_net'] <= 0, 'strategy_return_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    daily_returns = active_trades['strategy_return_net'] / 5
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0
    
    cumulative_net = np.exp(test_df['strategy_equity_net'])
    running_max = cumulative_net.cummax()
    max_dd = ((cumulative_net - running_max) / running_max).min()
    
    buy_hold_return = np.exp(test_df['buy_hold_equity'].iloc[-6]) - 1
    strategy_net_return = np.exp(test_df['strategy_equity_net'].iloc[-6]) - 1

    # Print log
    logger.info("========================================")
    logger.info("📊 BACKTEST RESULTS (Out-of-Sample)")
    logger.info("========================================")
    logger.info(f"Total Trades Taken : {total_trades}")
    logger.info(f"Win Rate           : {win_rate*100:.2f}%")
    logger.info(f"Profit Factor      : {profit_factor:.2f}")
    logger.info(f"Sharpe Ratio (Ann.): {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown       : {max_dd*100:.2f}%")
    logger.info("----------------------------------------")
    logger.info(f"Buy & Hold Return  : {buy_hold_return:.2%}")
    logger.info(f"Strategy Net Return: {strategy_net_return:.2%}")
    logger.info("========================================")

    # MLFLOW INTEGRATION
    experiment_name = f"{symbol}_LGBM_DIRECTION_V1"
    mlflow.set_experiment(experiment_name)
    
    from datetime import datetime
    run_name = f"{datetime.now().strftime('%Y%m%d')}_{symbol}_1D_BACKTEST_V1"
    
    with mlflow.start_run(run_name=run_name):
        logger.info("Logging Backtest results to MLflow...")
        mlflow.set_tags({"stage": "Backtesting", "symbol": symbol, "type": "Out-of-Sample"})
        mlflow.log_params(best_params)
        mlflow.log_param("buy_threshold", buy_threshold)
        mlflow.log_param("sell_threshold", sell_threshold)
        mlflow.log_param("transaction_cost", transaction_cost)
        
        mlflow.log_metrics({
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "buy_hold_return": buy_hold_return,
            "strategy_net_return": strategy_net_return
        })
        logger.info("✅ Backtest logged successfully to MLflow")

if __name__ == "__main__":
    df = load_data("EURUSD")
    if df is not None:
        run_simple_backtest(df)
