import os
import sys
from loguru import logger
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.database import engine

def init_db():
    """Initialize the PostgreSQL/SQLite schema for the Quant Platform"""
    logger.info("Starting database schema initialization...")
    
    with engine.connect() as conn:
        # SQLite uses INTEGER PRIMARY KEY AUTOINCREMENT instead of SERIAL
        # Let's write universal definitions where possible or fallback to standard types.
        # For simplicity in this demo, SQLite handles INTEGER PRIMARY KEY as AUTOINCREMENT implicitly.
        
        # Layer 1: Master Data
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR(50) UNIQUE NOT NULL,
            asset_class VARCHAR(50) NOT NULL,
            base_currency VARCHAR(10),
            quote_currency VARCHAR(10),
            exchange VARCHAR(50),
            status VARCHAR(20) DEFAULT 'ACTIVE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))
        
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS asset_sources (
            id INTEGER PRIMARY KEY,
            asset_id INTEGER REFERENCES assets(id),
            provider VARCHAR(50) NOT NULL,
            provider_symbol VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider, provider_symbol)
        );
        """))
        
        # Layer 2: Raw Market Data
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS market_ohlcv (
            id INTEGER PRIMARY KEY,
            asset_id INTEGER REFERENCES assets(id),
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC,
            spread NUMERIC,
            source VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asset_id, timeframe, timestamp)
        );
        """))
        
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS economic_events (
            event_id INTEGER PRIMARY KEY,
            country VARCHAR(10),
            currency VARCHAR(10),
            event_name VARCHAR(255) NOT NULL,
            importance VARCHAR(20),
            actual VARCHAR(50),
            forecast VARCHAR(50),
            previous VARCHAR(50),
            event_time TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))

        # Layer 3: Feature Store
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feature_definitions (
            feature_id INTEGER PRIMARY KEY,
            feature_name VARCHAR(100) NOT NULL,
            feature_version VARCHAR(50) NOT NULL,
            description TEXT,
            formula TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(feature_name, feature_version)
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS features_wide_v1 (
            id INTEGER PRIMARY KEY,
            asset_id INTEGER REFERENCES assets(id),
            timeframe VARCHAR(10),
            timestamp TIMESTAMP NOT NULL,
            ret_5 NUMERIC,
            ret_20 NUMERIC,
            rsi14 NUMERIC,
            atr_ratio NUMERIC,
            ema_gap NUMERIC,
            trend_score NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asset_id, timeframe, timestamp)
        );
        """))

        # Layer 4: Label Store
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS label_definitions (
            label_id INTEGER PRIMARY KEY,
            label_name VARCHAR(100) NOT NULL,
            version VARCHAR(50) NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(label_name, version)
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY,
            asset_id INTEGER REFERENCES assets(id),
            timestamp TIMESTAMP NOT NULL,
            label_id INTEGER REFERENCES label_definitions(label_id),
            target NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asset_id, timestamp, label_id)
        );
        """))

        # Layer 6: Model Registry Metadata
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY,
            mlflow_run_id VARCHAR(100),
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, model_version)
        );
        """))

        # Layer 7: Prediction Store
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY,
            asset_id INTEGER REFERENCES assets(id),
            timestamp TIMESTAMP NOT NULL,
            model_id INTEGER REFERENCES models(model_id),
            prediction VARCHAR(50) NOT NULL,
            probability NUMERIC,
            regime VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))
        
        # Layer 9: Monitoring
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS drift_reports (
            report_id INTEGER PRIMARY KEY,
            feature_name VARCHAR(100) NOT NULL,
            psi_score NUMERIC NOT NULL,
            drift_status VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))

        conn.commit()
        logger.info("Database schema initialized successfully! All 9 layers created.")

if __name__ == "__main__":
    init_db()
