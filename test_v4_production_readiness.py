import sys
import os
import requests
import time
import pandas as pd
from loguru import logger
import subprocess

logger.info("=========================================")
logger.info("PRODUCTION READINESS TEST (PHASE 1-15)")
logger.info("=========================================")

# 1. Test News API Gateway
logger.info("\n[1] Testing News API Gateway (FastAPI on Port 8001)...")
try:
    resp = requests.get("http://localhost:8001/news?symbol=EURUSD", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        logger.info(f"✅ News API Responded Successfully!")
        logger.info(f"   Symbol: {data['symbol']}")
        logger.info(f"   Is Safe to Trade?: {data['safe']}")
        logger.info(f"   Minutes to next High-Impact News: {data['minutes_to_next_high_impact']} mins")
    else:
        logger.error(f"❌ News API returned status code {resp.status_code}")
except requests.exceptions.ConnectionError:
    logger.error("❌ News API is NOT RUNNING on Port 8001. Please start it using 'python src/data/news_api_server.py'")

# 2. Test ONNX Models existence
logger.info("\n[2] Checking ONNX Production Models...")
prod_dir = "models/production"
if not os.path.exists(prod_dir):
    logger.error(f"❌ Production directory '{prod_dir}' not found.")
else:
    files = os.listdir(prod_dir)
    onnx_files = [f for f in files if f.endswith('.onnx')]
    json_files = [f for f in files if f.endswith('.json')]
    logger.info(f"✅ Found {len(onnx_files)} ONNX models and {len(json_files)} JSON config files.")
    if len(onnx_files) >= 2:
        logger.info(f"   Examples: {onnx_files[:2]}")
    else:
        logger.warning("   ⚠️ Not enough models generated. Did you run the export script?")

# 3. Test MQL5 Logic Check
logger.info("\n[3] Checking MQL5 Smart EA Architecture...")
mql5_file = "src/mql5/QuantTrader_ONNX_V2.mq5"
if os.path.exists(mql5_file):
    with open(mql5_file, 'r') as f:
        content = f.read()
        has_dynamic_features = "ArrayResize(features, num_features)" in content
        has_onnx_run = "OnnxRun" in content
        logger.info(f"✅ MQL5 EA File Found: {mql5_file}")
        logger.info(f"   Dynamic Feature Allocation: {'✅ Passed' if has_dynamic_features else '❌ Failed'}")
        logger.info(f"   ONNX Inference Layer: {'✅ Passed' if has_onnx_run else '❌ Failed'}")
else:
    logger.error(f"❌ MQL5 EA File not found at {mql5_file}")

logger.info("\n=========================================")
logger.info("SYSTEM DIAGNOSTIC COMPLETE")
logger.info("=========================================")
