#!/bin/bash
echo "Starting V5 Rescue Pipeline in Background..."
echo "Logs will be written to background_v5.log"
source .venv/bin/activate
nohup python test_v5_underperformers.py > background_v5.log 2>&1 &
echo "Process started with PID $!"
