import sys
import os
from loguru import logger
import subprocess

logger.info("=========================================")
logger.info("E2E PIPELINE TEST: DATA -> FEATURES")
logger.info("=========================================")

# Step 1: Clear old data (to avoid Unique Constraint duplicate error on re-run)
try:
    from sqlalchemy import text
    from src.config.database import engine
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM market_ohlcv;"))
        conn.commit()
    logger.info("✅ Cleared old OHLCV data to prevent duplicates during test.")
except Exception as e:
    logger.warning(f"Could not clear db: {e}")

# Step 2: Run Data Collection
logger.info("Running Data Collection...")
result = subprocess.run(["python", "src/data/fetch_yfinance.py"], capture_output=True, text=True)
if result.returncode != 0:
    logger.error(f"❌ Data Collection Failed:\n{result.stderr}")
    sys.exit(1)
logger.info("✅ Data Collection Passed.")

# Step 3: Run Feature Engineering
logger.info("Running Feature Engineering...")
result = subprocess.run(["python", "run_feature_flow_direct.py"], capture_output=True, text=True)
if result.returncode != 0:
    logger.error(f"❌ Feature Engineering Failed:\n{result.stderr}")
    sys.exit(1)
logger.info("✅ Feature Engineering Passed.")

logger.info("=========================================")
logger.info("🎉 E2E TEST COMPLETED SUCCESSFULLY")
logger.info("=========================================")
