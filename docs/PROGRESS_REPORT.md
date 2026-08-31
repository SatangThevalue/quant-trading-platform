# Progress Report & Implementation Notes

## Current Status: Phase 8 (ONNX Deployment Pipeline)
This document outlines the current state of the AI Quant Trading Platform development, the challenges encountered on the VPS, and the handover instructions for executing on a local PC/Mac.

### ✅ Completed Work (Phases 0 - 8)
1. **Foundation & Architecture:**
   - Designed a 16-phase Master Project Plan (`MASTER_PROJECT_PLAN.md`).
   - Established the directory structure for MLOps (`src/data`, `src/features`, `src/models`, `src/flows`, `src/config`).
2. **Data Platform (Phase 1):**
   - Implemented `src/data/fetch_yfinance.py` to download OHLCV data.
   - Built a 9-layer PostgreSQL/SQLite database schema (`src/data/init_db.py`).
3. **Feature Engineering (Phase 2):**
   - Created `src/features/feature_pipeline.py` implementing institutional-grade feature logic (EMA Gaps, RSI Distances, ATR Ratios) using `pandas-ta`.
4. **Orchestration (Phase 3 & 4):**
   - Wrapped Data and Feature pipelines in Prefect (`src/flows/data_collection_flow.py`, `src/flows/feature_engineering_flow.py`).
5. **Model Training & MLflow (Phase 6):**
   - Developed `src/models/train_lightgbm.py` (LightGBM, Optuna, MLflow tracking).
6. **ONNX Deployment Pipeline (Phase 8):**
   - Created `src/models/export_onnx.py`.
   - Exported `model.onnx`, `feature_order.json`, and `model_config.json`.
   - Included Python vs ONNX prediction validation logic.

---

## ⚠️ Known Issues & VPS Limitations (Why testing was moved to PC)

During remote execution on the VPS, we encountered limits:

1. **Storage Capacity (No space left on device):**
   - The Root Partition reached 100% capacity (49GB). Unpacking large data science libraries caused I/O errors.
   
2. **CPU Instruction Set Compatibility (Illegal instruction):**
   - The VPS CPU lacks modern instruction sets (`avx2, fma`). Running `Polars` and `LightGBM` resulted in a `core dumped` error.
   
3. **Prefect Daemon Timeout:**
   - The Prefect ephemeral API server timed out due to system resource constraints.

---

## 🚀 Next Steps: Instructions for Local PC Execution

Since the code architecture is complete, clone this repository to your local PC/Mac and run the pipelines there.

**Execution Order:**
1. Git clone the repository.
2. Initialize environment (`uv venv --python 3.12`) and install packages.
3. Run Data Collection: `python src/flows/data_collection_flow.py`
4. Run Feature Engineering: `python src/flows/feature_engineering_flow.py`
5. Run Training: `python src/models/train_lightgbm.py`