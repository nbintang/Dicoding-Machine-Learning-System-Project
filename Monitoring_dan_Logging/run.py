import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Starting FastAPI Server with Prometheus Monitoring")
    print("=" * 70)
    print("\n📌 Endpoints:")
    print("   - API Docs (Swagger): http://127.0.0.1:8000/docs")
    print("   - API Docs (ReDoc):   http://127.0.0.1:8000/redoc")
    print("   - Metrics:            http://127.0.0.1:8000/metrics")
    print("   - Health Check:       http://127.0.0.1:8000/health")
    print("   - Input Information:  http://127.0.0.1:8000/input-information")
    print("   - Predict:            http://127.0.0.1:8000/predict")
    print("   - Batch Predict:      http://127.0.0.1:8000/batch_predict")
    print("\n💡 MLflow Model Server should be running at:")
    print("   http://127.0.0.1:5000/invocations")
    print("=" * 70 + "\n")
    uvicorn.run(
        "prometheus_exporter:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
