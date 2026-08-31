# AI Quant Trading Platform

A modern, MLOps-based Quantitative Trading Platform designed for scale and institutional standards. Built on top of MT5, LightGBM, ONNX, MLflow, and Prefect Orchestration.

## 🏗️ Architecture

This platform shifts the paradigm from "Building AI Bots" to "Building a Trading Platform".

1. **Prediction Engine:** Powered by LightGBM
2. **Risk Engine:** Guardrails protecting against DD, Exposure, Spread, and News
3. **Execution Engine:** Smart Entry/Exit and Position Sizing (Volatility Adjusted)
4. **Portfolio Engine:** Correlation Control and Risk Parity

### Stack
- **Data:** MetaTrader 5, Yahoo Finance, FRED, TradingEconomics, CCXT
- **Data Processing:** Polars, DuckDB, Pandas-TA
- **Database:** PostgreSQL (Master, Features, Predictions, Metrics)
- **Machine Learning:** LightGBM, Optuna, scikit-learn
- **MLOps & Orchestration:** Prefect, MLflow, Evidently (Drift Detection)
- **Deployment:** ONNX, MQL5 (MT5 EA)

## 📁 Directory Structure
```text
project/
├── src/
│   ├── data/             # Data collection scripts
│   ├── features/         # Feature engineering logic (Price, Trend, Volatility, etc.)
│   ├── selection/        # Feature selection (Correlation, SHAP, RFE)
│   ├── models/           # Model training and ONNX export
│   ├── pipelines/        # Scikit-learn pipelines
│   ├── flows/            # Prefect flows (Orchestration)
│   ├── tasks/            # Prefect tasks
│   ├── config/           # Pydantic Settings
│   └── utils/            # Helper functions
├── data/
│   ├── bronze/           # Raw data
│   ├── silver/           # Cleaned data
│   └── gold/             # Feature Store / Ready for modeling
├── notebooks/            # Research and Exploratory Data Analysis (EDA)
├── mlruns/               # MLflow tracking local storage
├── docs/                 # Platform documentation
├── tests/                # Pytest unit testing
└── requirements.txt      # Dependency list
```

## 🚀 The Gold Standard Quant Workflow
`Market Data` → `Data Validation` → `Feature Eng.` → `Feature Selection` → `Label Eng.` → `LightGBM` → `Optuna` → `Walk Forward` → `Regime Analysis` → `Portfolio Construction` → `Risk Engine` → `MLflow` → `ONNX` → `Paper Trading` → `Production` → `Monitoring` → `Drift Detection` → `Retraining`
