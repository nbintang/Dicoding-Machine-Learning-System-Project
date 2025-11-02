from fastapi import HTTPException
import requests
import psutil
from logger import logger
from config import (
    MLFLOW_MODEL_URL, 
    CPU_USAGE,
    RAM_USAGE,
    DISK_USAGE,
)

def update_system_metrics():
    """Update system metrics (CPU, RAM, Disk)"""
    try:
        CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
        RAM_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage("/").percent)
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")


def check_model_availability():
    """Check if MLflow model server is available"""
    try:
        response = requests.get(
            MLFLOW_MODEL_URL.replace("/invocations", "/ping"), timeout=2
        )
        return response.status_code == 200
    except:
        return False


def call_mlflow_model(data: dict):
    try:
        response = requests.post(
            MLFLOW_MODEL_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Model server timeout")
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to model server. Make sure MLflow serving is running.",
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))