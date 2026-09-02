import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure MLflow Local Storage
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{mlruns_path}")

def load_feature_data(symbol="EURUSD"):
    gold_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gold")
    file_path = os.path.join(gold_path, f"{symbol}_features.parquet")
    
    if not os.path.exists(file_path):
        logger.error(f"Feature file not found: {file_path}")
        return None
        
    df = pd.read_parquet(file_path)
    # Ensure chronological order
    df = df.sort_values("timestamp")
    return df

def train_and_optimize(df, symbol="EURUSD"):
    # Drop columns not used for training
    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df["target"]
    
    # Phase 6 Baseline: Simple Time Series Split (80% Train, 20% Test)
    # (Walk Forward will be implemented in Phase 7)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows")
    logger.info(f"Using {len(features)} features")

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": 42
        }
        
        model = lgb.LGBMClassifier(**params, n_estimators=100)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc

    # 1. Run Optuna
    logger.info("Starting Optuna Hyperparameter Optimization...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) # Keep small for MVP demo
    
    best_params = study.best_params
    best_acc = study.best_value
    logger.info(f"Best Optuna Accuracy: {best_acc:.4f}")
    logger.info(f"Best Params: {best_params}")
    
    # 2. MLflow Tracking
    experiment_name = f"{symbol}_LGBM_DIRECTION_V1"
    mlflow.set_experiment(experiment_name)
    
    run_name = f"{datetime.now().strftime('%Y%m%d')}_{symbol}_1D_FS_V1"
    
    with mlflow.start_run(run_name=run_name):
        logger.info(f"Logging to MLflow (Run: {run_name})...")
        
        # Log Tags
        mlflow.set_tags({
            "symbol": symbol,
            "timeframe": "1D",
            "market": "Forex",
            "model": "LightGBM",
            "feature_version": "FS_V1",
            "label_version": "LB_V1",
            "stage": "Research"
        })
        
        # Log Hyperparameters
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", 20)
        mlflow.log_param("initial_features", len(features))
        mlflow.log_param("dataset_rows", len(df))
        
        # 3. Train final model with best params
        final_model = lgb.LGBMClassifier(**best_params, n_estimators=150, random_state=42, verbose=-1)
        final_model.fit(X_train, y_train)
        
        # Evaluate
        preds = final_model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision
        })
        
        # Log model (Artifact)
        mlflow.lightgbm.log_model(final_model, "model")
        
        logger.info(f"Successfully logged model and metrics to MLflow. Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    for sym in ["EURUSD", "GBPUSD", "JPY", "AUDUSD"]:
        logger.info(f"--- Processing {sym} ---")
        df = load_feature_data(sym)
        if df is not None:
            train_and_optimize(df, sym)
