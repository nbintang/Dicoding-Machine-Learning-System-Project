# -*- coding: utf-8 -*-
"""
inference.py - Basic Level
Script untuk melakukan prediksi IEA Global EV Data
"""

import requests
import json
import pandas as pd
from config import (
    FEATURE_COLS,
    TARGET_COL
)
def infer(instances, url="http://127.0.0.1:5000/invocations", format="dataframe_split"):
    headers = {"Content-Type": "application/json"}
    payload = {}

    if format == "dataframe_split":
        if len(FEATURE_COLS) != len(instances[0]):
            print(f"❌ Error: Jumlah fitur tidak cocok. Expected {len(FEATURE_COLS)}, got {len(instances[0])}")
            return None
        
        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLS,
                "data": instances
            }
        }
    else:
        print(f"❌ Error: Format '{format}' tidak didukung")
        return None

    data_json = json.dumps(payload)
    print("\n--- Payload JSON yang Dikirim ---")
    print(data_json)
    print("---------------------------------\n")

    try:
        response = requests.post(url, data=data_json, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Request gagal: {e}")
        return None

def display_sample_info(sample_df, idx):
    """
    Menampilkan informasi detail tentang sampel
    
    Args:
        sample_df (DataFrame): DataFrame sampel
        idx (int): Index sampel dalam DataFrame
    """
    row = sample_df.iloc[idx]
    print(f"\n📋 Detail Sampel:")
    for col in FEATURE_COLS:
        print(f"   {col}: {row[col]}")


def infer_from_manual_input():
    print("\n" + "="*60)
    print("🔮 Manual Input Mode - IEA Global EV Data")
    print("="*60)
    
    print(f"\nMasukkan nilai untuk {len(FEATURE_COLS)} fitur:")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"   {i}. {col}")
    
    print("\n💡 Contoh input:")
    print("   China EV_sales BEV Publicly_available_fast BEV 2023")

    input_str = input("\nMasukkan nilai: ").strip()
    if not input_str:
        print("❌ Input kosong!")
        return
    
    values = input_str.split()
    if len(values) != len(FEATURE_COLS):
        print(f"❌ Error: Jumlah nilai tidak sesuai! ({len(values)}/{len(FEATURE_COLS)})")
        return
    
    # Konversi tahun ke integer
    try:
        values[-1] = int(values[-1])
    except ValueError:
        print("⚠️ Warning: Tahun harus berupa angka")
        return

    manual_data = [values]
    print(f"\n📤 Mengirim data ke model server...")
    print(f"   Data: {manual_data}")

    result = infer(manual_data)
    if not result:
        print("\n❌ Prediksi Gagal")
        return

    # ✅ Format hasil prediksi agar lebih mudah dibaca
    try:
        predictions = result.get("predictions", None)
        if predictions is None:
            print(f"⚠️ Format hasil tak dikenal: {result}")
            return

        pred_value = predictions[0]
        print("\n" + "="*60)
        print("🎯 HASIL PREDIKSI")
        print("="*60)
        print("📊 Input:")
        for col, val in zip(FEATURE_COLS, values):
            print(f"   {col:<15}: {val}")

        print("\n📈 Output Model:")
        print(f"   Nilai prediksi terstandarisasi (value_scaled): {pred_value:.4f}")

        print("="*60)

    except Exception as e:
        print(f"⚠️ Gagal memproses hasil prediksi: {e}")


def main():
    print("\n" + "="*60)
    print("🚀 MLflow Model Inference - IEA Global EV Data")
    print("="*60)
    
    print("\n📌 Dataset: IEA Global EV Data 2024")
    print(f"📌 Fitur: {', '.join(FEATURE_COLS)}")
    print(f"📌 Target: {TARGET_COL}")
    
    print("\n" + "-"*60)
    print("Pilih mode inference:")
    print("  1. Input manual")
    print("  2. Exit")
    print("-"*60)
    
    try:
        choice = input("\nPilihan (1/2): ").strip()

        if choice == "1":
            infer_from_manual_input()
        elif choice == "2":
            print("\n👋 Terima kasih!")
            return
        else:
            print("❌ Pilihan tidak valid")
    
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan")
    except Exception as e:
        print(f"\n❌ Error: {e}")



if __name__ == "__main__":
    print("\n💡 PETUNJUK PENGGUNAAN:")
    print("="*60)
    print("1. Pastikan MLflow model serving sudah berjalan")
    print("2. Command untuk serving:")
    print("   mlflow models serve -m runs:/<RUN_ID>/model -p 5005")
    print("3. Ganti <RUN_ID> dengan ID dari MLflow UI")
    print("4. Buka MLflow UI: mlflow ui")
    print("="*60)
    
    main()