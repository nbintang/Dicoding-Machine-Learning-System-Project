# -*- coding: utf-8 -*-
"""
prometheus_exporter.py
FastAPI server dengan Prometheus monitoring untuk IEA Global EV Data prediction
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging
from config import MLFLOW_MODEL_URL, FEATURE_COLS, TARGET_COL
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IEA Global EV Prediction API with Prometheus",
    description="API untuk prediksi IEA Global EV Data dengan monitoring Prometheus",
    version="1.0.0"
)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Metrik untuk API requests
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP Requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP Request Latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_duration_seconds',
    'Model prediction latency'
)

# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage_percent', 'RAM Usage Percentage')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk Usage Percentage')

# Metrik untuk model performance
MODEL_ERRORS = Counter('model_error_total', 'Total model errors', ['error_type'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionInput(BaseModel):
    """Model untuk input prediksi"""
    region: str = Field(..., example="China")
    category: str = Field(..., example="EV_sales")
    parameter: str = Field(..., example="BEV")
    mode: str = Field(..., example="Publicly_available_fast")
    powertrain: str = Field(..., example="BEV")
    year: int = Field(..., ge=2000, le=2100, example=2023)
    
    @validator('year')
    def validate_year(cls, v):
        if not (2000 <= v <= 2100):
            raise ValueError('Year must be between 2000 and 2100')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "region": "China",
                "category": "EV_sales",
                "parameter": "BEV",
                "mode": "Publicly_available_fast",
                "powertrain": "BEV",
                "year": 2023
            }
        }


class BatchPredictionInput(BaseModel):
    """Model untuk batch prediction"""
    instances: List[PredictionInput] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "region": "China",
                        "category": "EV_sales",
                        "parameter": "BEV",
                        "mode": "Publicly_available_fast",
                        "powertrain": "BEV",
                        "year": 2023
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Model untuk response prediksi"""
    predictions: List[float]
    latency_seconds: float
    timestamp: float


class HealthResponse(BaseModel):
    """Model untuk health check response"""
    status: str
    model_available: bool
    system_info: dict



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_system_metrics():
    """Update system metrics (CPU, RAM, Disk)"""
    try:
        CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
        RAM_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage('/').percent)
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")


def check_model_availability():
    """Check if MLflow model server is available"""
    try:
        response = requests.get(MLFLOW_MODEL_URL.replace('/invocations', '/ping'), timeout=2)
        return response.status_code == 200
    except:
        return False


def call_mlflow_model(data: dict):
    """
    Call MLflow model serving endpoint
    
    Args:
        data: Payload in dataframe_split format
        
    Returns:
        dict: Model prediction response
    """
    try:
        response = requests.post(
            MLFLOW_MODEL_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Model server timeout")
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="Cannot connect to model server. Make sure MLflow serving is running."
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware untuk monitoring semua requests"""
    start_time = time.time()
    
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    
    finally:
        ACTIVE_REQUESTS.dec()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "IEA Global EV Prediction API with Prometheus Monitoring",
        "endpoints": {
            "predict": "/predict - Single prediction",
            "batch_predict": "/batch_predict - Batch predictions",
            "metrics": "/metrics - Prometheus metrics",
            "health": "/health - Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    update_system_metrics()
    
    model_available = check_model_availability()
    
    return {
        "status": "healthy" if model_available else "degraded",
        "model_available": model_available,
        "system_info": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    # Update system metrics before exposing
    update_system_metrics()
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Single prediction endpoint
    
    Args:
        input_data: PredictionInput model dengan 6 fitur
        
    Returns:
        PredictionResponse dengan hasil prediksi
    """
    start_time = time.time()
    
    try:
        # Convert input to list format
        instance = [
            input_data.region,
            input_data.category,
            input_data.parameter,
            input_data.mode,
            input_data.powertrain,
            input_data.year
        ]
        
        # Prepare payload for MLflow
        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLS,
                "data": [instance]
            }
        }
        
        logger.info(f"Sending prediction request: {payload}")
        
        # Call MLflow model
        result = call_mlflow_model(payload)
        
        duration = time.time() - start_time
        
        # Record metrics
        PREDICTION_COUNT.labels(status='success').inc()
        PREDICTION_LATENCY.observe(duration)
        
        return {
            "predictions": result.get("predictions", []),
            "latency_seconds": duration,
            "timestamp": time.time()
        }
    
    except HTTPException:
        PREDICTION_COUNT.labels(status='error').inc()
        MODEL_ERRORS.labels(error_type='http_error').inc()
        raise
    
    except Exception as e:
        PREDICTION_COUNT.labels(status='error').inc()
        MODEL_ERRORS.labels(error_type='internal_error').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(input_data: BatchPredictionInput):
    """
    Batch prediction endpoint
    
    Args:
        input_data: BatchPredictionInput dengan list of instances
        
    Returns:
        PredictionResponse dengan hasil prediksi batch
    """
    start_time = time.time()
    
    try:
        # Convert all instances to list format
        instances = [
            [
                inst.region,
                inst.category,
                inst.parameter,
                inst.mode,
                inst.powertrain,
                inst.year
            ]
            for inst in input_data.instances
        ]
        
        # Prepare payload for MLflow
        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLS,
                "data": instances
            }
        }
        
        logger.info(f"Sending batch prediction request with {len(instances)} instances")
        
        # Call MLflow model
        result = call_mlflow_model(payload)
        
        duration = time.time() - start_time
        
        # Record metrics
        PREDICTION_COUNT.labels(status='success').inc()
        PREDICTION_LATENCY.observe(duration)
        
        return {
            "predictions": result.get("predictions", []),
            "latency_seconds": duration,
            "timestamp": time.time()
        }
    
    except HTTPException:
        PREDICTION_COUNT.labels(status='error').inc()
        MODEL_ERRORS.labels(error_type='http_error').inc()
        raise
    
    except Exception as e:
        PREDICTION_COUNT.labels(status='error').inc()
        MODEL_ERRORS.labels(error_type='internal_error').inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Event yang dijalankan saat aplikasi startup"""
    logger.info("🚀 Starting IEA Global EV Prediction API...")
    logger.info(f"📊 MLflow Model URL: {MLFLOW_MODEL_URL}")
    
    # Check model availability
    if check_model_availability():
        logger.info("✅ MLflow model server is available")
    else:
        logger.warning("⚠️ MLflow model server is not available")
        logger.warning("   Please start the model server with:")
        logger.warning("   mlflow models serve -m runs:/<RUN_ID>/model -p 5000")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 Starting FastAPI Server with Prometheus Monitoring")
    print("="*70)
    print("\n📌 Endpoints:")
    print("   - API Docs (Swagger): http://127.0.0.1:8000/docs")
    print("   - API Docs (ReDoc):   http://127.0.0.1:8000/redoc")
    print("   - Metrics:            http://127.0.0.1:8000/metrics")
    print("   - Health Check:       http://127.0.0.1:8000/health")
    print("   - Predict:            http://127.0.0.1:8000/predict")
    print("   - Batch Predict:      http://127.0.0.1:8000/batch_predict")
    print("\n💡 MLflow Model Server should be running at:")
    print("   http://127.0.0.1:5000/invocations")
    print("="*70 + "\n")
    
    uvicorn.run(
        "prometheus_exporter:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )