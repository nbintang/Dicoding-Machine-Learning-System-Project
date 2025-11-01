# -*- coding: utf-8 -*-
"""
modelling.py - Basic Level
Model Machine Learning dengan MLflow Autolog
"""

import pandas as pd # type: ignore
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # type: ignore
from config import (
    feature_cols,
    target_col,
    experiment_name,
    model_autolog_name,
    dataset_path
)

def main():
    print("🚀 Memulai training model dengan MLflow Autolog...\n")
    
    # 1. Load dataset hasil preprocessing
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # 2. Persiapan data
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")
    
    
    # 3. Setup MLflow
    mlflow.set_experiment(experiment_name)
    
    # 4. Enable autolog untuk Scikit-learn
    mlflow.sklearn.autolog()
    
    # 5. Training model
    with mlflow.start_run(run_name=model_autolog_name):
        print("\n🔧 Training Random Forest model...")
        
        # Inisialisasi dan training model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Evaluasi (autolog akan mencatat ini otomatis)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n📊 Hasil Evaluasi:")
        print(f"   MSE: {mse:.4f}")
        print(f"   R2 Score: {r2:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        print("\n✅ Model berhasil dilatih dan disimpan di MLflow!")
        print("💡 Jalankan 'mlflow ui' untuk melihat dashboard")

if __name__ == "__main__":
    main()