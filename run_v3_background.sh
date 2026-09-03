#!/bin/bash
echo "Starting V3 Asset-Specific Pipeline in Background..."
echo "Logs will be written to background_v3.log"
source .venv/bin/activate
export PREFECT_API_URL=""
export PREFECT_LOGGING_LEVEL="ERROR"
nohup python src/pipelines/v3_asset_specific_loop.py > background_v3.log 2>&1 &
echo "Process started with PID $!"
