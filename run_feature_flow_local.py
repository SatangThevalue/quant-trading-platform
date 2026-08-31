import sys
import os

# Set Prefect to use the local ephemeral SQLite backend specifically and disable temporary server boot issue
os.environ["PREFECT_API_URL"] = ""

from src.flows.feature_engineering_flow import run_feature_engineering_pipeline

if __name__ == "__main__":
    run_feature_engineering_pipeline()
