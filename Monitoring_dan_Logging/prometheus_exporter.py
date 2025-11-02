from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import time
import psutil
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from logger import logger
from dto import (
    PredictionInput,
    PredictionResponse,
    BatchPredictionInput,
    HealthResponse,
)
from ctx_manager import lifespan
from config import (
    FEATURE_COLS,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    MODEL_ERRORS,
)
from helper import (
    update_system_metrics,
    check_model_availability,
    call_mlflow_model,
)

app = FastAPI(
    title="IEA Global EV Prediction API with Prometheus",
    description="API untuk prediksi IEA Global EV Data dengan monitoring Prometheus",
    version="1.0.0",
    lifespan=lifespan,
)





@app.get("/input-information", tags=["Information"])
async def get_input_information():
    return {
        "region": {
            "China": 1,
            "Europe": 2,
            "United_States": 3,
            "India": 4,
            "Other": 5,
        },
        "category": {
            "EV_sales": 1,
            "Stock": 2,
            "EV_share": 3,
            "Energy_demand": 4,
        },
        "parameter": {
            "BEV": 1,
            "PHEV": 2,
            "FCEV": 3,
        },
        "mode": {
            "Publicly_available_fast": 1,
            "Publicly_available_slow": 2,
            "Private_slow": 3,
        },
        "powertrain": {
            "BEV": 1,
            "PHEV": 2,
            "FCEV": 3,
        },
        "year": "Integer between 2000 and 2100 (e.g., 2023)",
    }

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "IEA Global EV Prediction API with Prometheus Monitoring",
        "endpoints": {
            "predict": "/predict - Single prediction",
            "batch_predict": "/batch_predict - Batch predictions",
            "metrics": "/metrics - Prometheus metrics",
            "health": "/health - Health check",
        },
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
            "disk_percent": psutil.disk_usage("/").percent,
        },
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    # Update system metrics before exposing
    update_system_metrics()

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    start_time = time.time()

    try:
        # Convert input to list format
        instance = [
            input_data.region,
            input_data.category,
            input_data.parameter,
            input_data.mode,
            input_data.powertrain,
            input_data.year,
        ]

        # Prepare payload for MLflow
        payload = {"dataframe_split": {"columns": FEATURE_COLS, "data": [instance]}}

        logger.info(f"Sending prediction request: {payload}")

        # Call MLflow model
        result = call_mlflow_model(payload)

        duration = time.time() - start_time

        # Record metrics
        PREDICTION_COUNT.labels(status="success").inc()
        PREDICTION_LATENCY.observe(duration)

        return {
            "predictions": result.get("predictions", []),
            "latency_seconds": duration,
            "timestamp": time.time(),
        }

    except HTTPException:
        PREDICTION_COUNT.labels(status="error").inc()
        MODEL_ERRORS.labels(error_type="http_error").inc()
        raise

    except Exception as e:
        PREDICTION_COUNT.labels(status="error").inc()
        MODEL_ERRORS.labels(error_type="internal_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(input_data: BatchPredictionInput):
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
                inst.year,
            ]
            for inst in input_data.instances
        ]

        # Prepare payload for MLflow
        payload = {"dataframe_split": {"columns": FEATURE_COLS, "data": instances}}

        logger.info(f"Sending batch prediction request with {len(instances)} instances")

        # Call MLflow model
        result = call_mlflow_model(payload)

        duration = time.time() - start_time

        # Record metrics
        PREDICTION_COUNT.labels(status="success").inc()
        PREDICTION_LATENCY.observe(duration)

        return {
            "predictions": result.get("predictions", []),
            "latency_seconds": duration,
            "timestamp": time.time(),
        }

    except HTTPException:
        PREDICTION_COUNT.labels(status="error").inc()
        MODEL_ERRORS.labels(error_type="http_error").inc()
        raise

    except Exception as e:
        PREDICTION_COUNT.labels(status="error").inc()
        MODEL_ERRORS.labels(error_type="internal_error").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
