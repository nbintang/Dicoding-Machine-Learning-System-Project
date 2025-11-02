from fastapi import FastAPI
from contextlib import asynccontextmanager
from config import (
    MLFLOW_MODEL_URL,
)
from logger import logger
from helper import check_model_availability

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # STARTUP
    logger.info("🚀 Starting IEA Global EV Prediction API...")
    logger.info(f"📊 MLflow Model URL: {MLFLOW_MODEL_URL}")
    try:
        if check_model_availability():
            logger.info("✅ MLflow model server is available")
        else:
            logger.warning("⚠️ MLflow model server is not available")
            logger.warning("   Please start the model server with:")
            logger.warning("   mlflow models serve -m runs:/<RUN_ID>/model -p 5000")
    except Exception as e:
        logger.exception(f"Error while checking model availability on startup: {e}")
    yield  
    # SHUTDOWN (opsional)
    logger.info("🛑 Shutting down IEA Global EV Prediction API...")