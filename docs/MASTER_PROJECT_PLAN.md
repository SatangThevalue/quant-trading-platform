# Master Project Plan: AI Quant Trading Platform

แผนแม่บท (Master Project Plan) ในการสร้าง AI Quant Trading Platform ระยะเวลา 6-8 เดือน

## Phase 0: Foundation & Architecture (2 Weeks)
- [ ] Project Architecture & Directory Setup
- [ ] Python Environment (uv) & Git Repository
- [ ] Standardization (UTC, Asset/Feature Naming, Versioning)
- [ ] Database Setup (PostgreSQL)

## Phase 1: Data Platform (3-4 Weeks)
- [ ] Integrate MT5, Yahoo Finance, CCXT, FRED, TradingEconomics
- [ ] Build Schema: `market_ohlcv`, `economic_events`, `news_events`
- [ ] Data validation and missing value checks

## Phase 2: Feature Store (4 Weeks)
- [ ] Implement Feature categories: Price, Trend, Momentum, Volatility, Regime, Macro
- [ ] Setup `feature_definitions` & `feature_values` tables
- [ ] Create Feature Contracts

## Phase 3: Label Engineering (1 Week)
- [ ] Target labeling: Direction, Future Return, Triple Barrier

## Phase 4: Baseline Research (2 Weeks)
- [ ] Baseline models: Logistic Regression, Random Forest

## Phase 5: Feature Selection (2 Weeks)
- [ ] Filter pipeline: Correlation -> Mutual Information -> RFE -> SHAP

## Phase 6: Model Training (3 Weeks)
- [ ] LightGBM implementation
- [ ] Optuna integration
- [ ] MLflow logging (Parameters, Metrics, Artifacts)

## Phase 7: Walk Forward Validation (3 Weeks)
- [ ] Rolling Windows engine
- [ ] Stability analysis (PF, Sharpe, Max Drawdown)

## Phase 8: ONNX Deployment Pipeline (2 Weeks)
- [ ] Export ONNX model, feature_order.json, model_config.json
- [ ] Validation between Python and ONNX runtime

## Phase 9: MT5 Integration (4 Weeks)
- [ ] MQL5 Feature Engine
- [ ] ONNX Runtime Layer
- [ ] Signal Layer

## Phase 10: Position Sizing Engine (2 Weeks)
- [ ] Logic: `Base Risk % × Confidence × Regime ÷ ATR`

## Phase 11: Risk Engine (3 Weeks)
- [ ] Daily Loss Limit, Drawdown Limit
- [ ] Spread, Slippage, News Protection

## Phase 12: Execution Engine (3 Weeks)
- [ ] Smart Entry / Smart Exit
- [ ] Trailing Stop, Break Even, Partial Close

## Phase 13: Monitoring Platform (3 Weeks)
- [ ] Data Drift (PSI), Concept Drift detection
- [ ] Performance monitoring dashboards

## Phase 14: Portfolio Engine (4 Weeks)
- [ ] Capital Allocation
- [ ] Correlation Matrix & Exposure control

## Phase 15: Regime Detection & Strategy Router (4 Weeks)
- [ ] Detect Trend, Range, Volatility Regimes
- [ ] Route execution based on regime

## Phase 16: Auto Retraining (2 Weeks)
- [ ] Retrain Trigger (e.g., PSI > 0.25 AND Performance drop)
- [ ] Prefect retraining flow automation
