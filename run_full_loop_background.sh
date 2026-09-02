#!/bin/bash
# ---------------------------------------------------------
# Background Runner for Quant Trading Platform Full Loop
# ---------------------------------------------------------

echo "Starting Full Loop Pipeline in Background..."
echo "Logs will be written to background_loop.log"

# Activate environment
source .venv/bin/activate

# Hard disable Prefect to prevent OOM / SQLite Lock overhead on constrained VPS
export PREFECT_API_URL=""
export PREFECT_LOGGING_LEVEL="ERROR"

# Run the native full loop python script (Not the Prefect one) using nohup
nohup python src/pipelines/full_loop_pipeline.py > background_loop.log 2>&1 &

echo "Process started with PID $!"
echo "You can view progress by typing: tail -f background_loop.log"