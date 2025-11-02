from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
)

MLFLOW_MODEL_URL = "http://127.0.0.1:5000/invocations"
FEATURE_COLS = ["region", "category", "parameter", "mode", "powertrain", "year"]
TARGET_COL = "value_scaled"

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint"]
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total", "Total prediction requests", ["status"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds", "Model prediction latency"
)

# Metrik untuk sistem
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage_percent", "RAM Usage Percentage")
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk Usage Percentage")

# Metrik untuk model performance
MODEL_ERRORS = Counter("model_error_total", "Total model errors", ["error_type"])
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")
