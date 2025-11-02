# -*- coding: utf-8 -*-
"""
modelling_tuning.py - Advanced Level
Model dengan DagsHub Integration
"""

import pandas as pd  # type: ignore
import mlflow  # type: ignore
import mlflow.sklearn  # type: ignore
import dagshub  # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from modelling_tuning_logger import manual_log_metrics
from dotenv import load_dotenv

load_dotenv()

from model_config import (
    dagshub_repo_owner,
    dagshub_repo_name,
    dagshub_token,
    model_tuning_type,
    model_tuning_storage,
    feature_cols,
    target_col,
    dagshub_repo_url,
    experiment_tuning_name,
    model_tuning_name,
    dataset_path,
)

if dagshub_token is not None:
    dagshub.auth.add_app_token(dagshub_token)
    dagshub.init(
        repo_owner=dagshub_repo_owner, repo_name=dagshub_repo_name, mlflow=True
    )
else:
    raise ValueError("DAGSHUB_TOKEN environment variable is not set.")


def train_with_dagshub():
    print("🚀 Memulai training dengan DagsHub integration...\n")
    mlflow.set_tracking_uri(dagshub_repo_url)
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {df.shape}")
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mlflow.set_experiment(experiment_tuning_name)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    
    with mlflow.start_run(run_name=model_tuning_name):
        print("\n🔧 Training model dengan hyperparameter tuning...")

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        mlflow.log_param("model_type", model_tuning_type)
        mlflow.log_param("storage", model_tuning_storage)

        y_pred_test = best_model.predict(X_test)

        manual_log_metrics(y_test, y_pred_test, "Test")

        mlflow.sklearn.log_model(best_model, "model")

        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id

        print(f"\n✅ Model berhasil disimpan ke DagsHub!")
        print(f"🔗 Run ID: {run_id}")
        print(f"🔗 Experiment ID: {experiment_id}")
        print(
            f"🌐 URL: https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}/experiments"
        )


if __name__ == "__main__":
    train_with_dagshub()
