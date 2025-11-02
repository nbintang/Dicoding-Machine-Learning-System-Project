from fastapi import Request
import time
from config import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ACTIVE_REQUESTS,
)
from prometheus_exporter import app


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
            method=request.method, endpoint=request.url.path
        ).observe(duration)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        return response

    finally:
        ACTIVE_REQUESTS.dec()
