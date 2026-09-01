import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from loguru import logger
import mlflow

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{mlruns_path}")

def calc_trading_metrics(y_true, y_pred, returns):
    """Calculate basic trading metrics from predictions."""
    signals = np.where(y_pred > 0.5, 1, -1)
    strategy_returns = signals * returns
    
    winning_trades = strategy_returns[strategy_returns > 0]
    losing_trades = strategy_returns[strategy_returns <= 0]
    
    gross_profit = winning_trades.sum()
    gross_loss = abs(losing_trades.sum())
    
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Approx Sharpe
    sharpe = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std()) if strategy_returns.std() > 0 else 0
    
    return pf, sharpe

def run_walk_forward(symbol="EURUSD", n_splits=5):
    """Phase 7: Rolling Window Walk Forward Validation"""
    logger.info(f"Starting Walk Forward Validation for {symbol} with {n_splits} splits...")
    
    gold_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gold")
    df = pd.read_parquet(os.path.join(gold_path, f"{symbol}_features.parquet"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df["target"]
    future_returns = np.log(df['close'].shift(-5) / df['close']).fillna(0)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    pf_list = []
    sharpe_list = []
    
    best_params = {'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.05, 'verbose': -1}
    
    mlflow.set_experiment(f"{symbol}_WALK_FORWARD_V1")
    with mlflow.start_run(run_name=f"WFA_{n_splits}_FOLDS"):
        for i, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            ret_test = future_returns.iloc[test_index]
            
            model = lgb.LGBMClassifier(**best_params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            
            pf, sharpe = calc_trading_metrics(y_test, preds, ret_test)
            pf_list.append(pf)
            sharpe_list.append(sharpe)
            
            logger.info(f"Window {i+1} | Train: {len(X_train)} Test: {len(X_test)} | PF: {pf:.2f} | Sharpe: {sharpe:.2f}")
            mlflow.log_metric(f"fold_{i+1}_pf", pf)
            mlflow.log_metric(f"fold_{i+1}_sharpe", sharpe)
            
        mean_pf = np.mean(pf_list)
        std_pf = np.std(pf_list)
        mean_sharpe = np.mean(sharpe_list)
        
        logger.info("=========================================")
        logger.info(f"WALK FORWARD RESULTS ({n_splits} Windows)")
        logger.info(f"Mean Profit Factor : {mean_pf:.2f}")
        logger.info(f"Std Profit Factor  : {std_pf:.2f} (Stability)")
        logger.info(f"Mean Sharpe Ratio  : {mean_sharpe:.2f}")
        logger.info("=========================================")
        
        mlflow.log_metrics({
            "mean_pf": mean_pf,
            "std_pf": std_pf,
            "mean_sharpe": mean_sharpe
        })

if __name__ == "__main__":
    logger.info("Walk Forward Engine ready for local execution.")
