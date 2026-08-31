import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

# Use SQLite for local development phase
# Data will be stored in data/quant_platform.db
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "quant_platform.db")
DATABASE_URL = f"sqlite:///{db_path}"

try:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database engine configured for local SQLite: {db_path}")
except Exception as e:
    logger.error(f"Failed to configure database engine: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
