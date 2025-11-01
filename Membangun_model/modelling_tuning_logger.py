import mlflow # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # type: ignore
import numpy as np

def manual_log_metrics(y_true, y_pred, model_name):
    """
    Manual logging dengan metrik tambahan untuk Advanced
    """
    # Metrik standar
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Metrik tambahan (Advanced requirements)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    median_error = np.median(np.abs(y_true - y_pred))
    std_error = np.std(y_true - y_pred)
    
    # Log semua metrik ke DagsHub
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape": mape,  
        "max_error": max_error,
        "median_error": median_error,
        "std_error": std_error
    }
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    print(f"\n📊 Hasil Evaluasi {model_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    return metrics